"""Index management and unified background job status routes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from ...core.scope import ImageScope
from ...image_io import canonical_path
from ...indexing.jobs import JobDependencyError
from ...services.common import attach_provenance, gallery_result, resolve_folder_keys_from_db
from ..deps import AppContainer, get_container
from ..schemas import (
    AutoFillQueueRequest,
    CcipResetRequest,
    CcipTaskRequest,
    IndexAddRequest,
    IndexRemoveRequest,
    JobMoveRequest,
    ProvenanceEnrichRequest,
    ProvenanceScanRequest,
    RetagJobRequest,
    SingleRetagRequest,
    VisionTaskRequest,
)
from .provenance import drain_provenance_refresh

router = APIRouter()


def _job_status(c: AppContainer) -> dict[str, Any]:
    job = c.index_job
    return {
        "running": job.running,
        "path": job.path,
        "done": job.done,
        "total": job.total,
        "message": job.message,
        "error": job.error,
        "summary": job.summary,
        # Monotonic stamp of the newest committed row, so the waterfall can tell
        # whether anything actually landed without re-querying the whole page.
        "revision": c.images.max_id(),
    }


def _resolve_index_directories(request: IndexAddRequest) -> list[Path]:
    raw_paths = [*request.paths]
    if request.path.strip():
        raw_paths.insert(0, request.path)
    directories: list[Path] = []
    seen: set[str] = set()
    for raw in raw_paths:
        text = raw.strip().strip('"').strip("'")
        if not text:
            continue
        directory = Path(text).expanduser()
        try:
            directory = directory.resolve()
        except OSError as exc:
            raise HTTPException(400, f"无法解析路径：{exc}") from exc
        if not directory.exists() or not directory.is_dir():
            raise HTTPException(400, f"目录不存在：{directory}")
        key = canonical_path(directory)
        if key in seen:
            continue
        seen.add(key)
        directories.append(directory)
    if not directories:
        raise HTTPException(400, "请提供要索引的本地目录")
    return directories


@router.get("/api/index-status")
def index_status(c: AppContainer = Depends(get_container)) -> dict[str, Any]:
    status = _job_status(c)
    if not status["running"]:
        drain_provenance_refresh(c)
    return status


@router.post("/api/index")
def add_index(
    request: IndexAddRequest,
    c: AppContainer = Depends(get_container),
) -> dict[str, Any]:
    directories = _resolve_index_directories(request)
    try:
        c.job_manager.submit_index_job(
            directories,
            recursive=request.recursive,
            force=False,
            auto_enrich=request.auto_enrich,
        )
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    return _job_status(c)


@router.post("/api/index-remove")
def remove_index(
    request: IndexRemoveRequest,
    c: AppContainer = Depends(get_container),
) -> dict[str, Any]:
    if c.index_job.running:
        raise HTTPException(409, "正在索引，请稍后再移出目录")
    key = canonical_path(request.folder_key)
    row_ids, point_ids = c.images.soft_delete_folder_subtree(key)
    if not row_ids and not c.db.read_conn().execute(
        "SELECT 1 FROM folders WHERE folder_key = ?", (key,)
    ).fetchone():
        raise HTTPException(404, "该目录不在索引中")
    if row_ids:
        c.vectors.zero_rows(row_ids)
        c.vectors.flush()
        for pid in point_ids:
            thumb = c.config.thumbnail_dir / f"{pid}.webp"
            thumb.unlink(missing_ok=True)
    c.invalidate_caches()
    return c.gallery.library_tree()


@router.get("/api/jobs")
def list_jobs(c: AppContainer = Depends(get_container)) -> dict[str, Any]:
    return {"jobs": c.jobs_repo.list_recent(limit=20)}


@router.get("/api/jobs/task-stats")
def task_stats(c: AppContainer = Depends(get_container)) -> dict[str, Any]:
    return {"counts": c.job_manager.task_summary()}


@router.get("/api/jobs/queue")
def job_queue(c: AppContainer = Depends(get_container)) -> dict[str, Any]:
    return c.job_manager.queue_snapshot()


@router.post("/api/jobs/auto-fill")
def auto_fill_queue(
    request: AutoFillQueueRequest,
    c: AppContainer = Depends(get_container),
) -> dict[str, Any]:
    if c.index_job.running:
        raise HTTPException(409, "正在索引图片，请稍后再安排队列")
    scope_rows = _scope_rows_for_folders(c, request.folder_keys)
    try:
        added = c.job_manager.auto_fill_queue(
            backend=request.backend,
            scope_row_ids=scope_rows,
            include_provenance=request.include_provenance,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"added": added, "queue": c.job_manager.queue_snapshot()}


@router.post("/api/jobs/provenance-scan")
def enqueue_provenance_scan(
    request: ProvenanceScanRequest,
    c: AppContainer = Depends(get_container),
) -> dict[str, Any]:
    if c.index_job.running:
        raise HTTPException(409, "正在索引图片，请稍后再安排扫描任务")
    auto_enrich = (
        c.config.provenance.auto_enrich
        if request.auto_enrich is None
        else request.auto_enrich
    )
    job_id = c.job_manager.submit_provenance_scan_job(
        deep=request.deep,
        only_new=request.only_new,
        auto_enrich=auto_enrich,
    )
    return {"job_id": job_id, "job": c.jobs_repo.get(job_id)}


@router.post("/api/jobs/provenance-enrich")
def enqueue_provenance_enrich(
    request: ProvenanceEnrichRequest,
    c: AppContainer = Depends(get_container),
) -> dict[str, Any]:
    if c.index_job.running:
        raise HTTPException(409, "正在索引图片，请稍后再安排补全任务")
    job_id = c.job_manager.submit_provenance_enrich_job(
        platforms=list(request.platforms),
        limit=request.limit,
    )
    return {"job_id": job_id, "job": c.jobs_repo.get(job_id)}


@router.post("/api/jobs/clear-finished")
def clear_finished_jobs(c: AppContainer = Depends(get_container)) -> dict[str, Any]:
    removed = c.job_manager.clear_finished_jobs()
    return {"removed": removed, "queue": c.job_manager.queue_snapshot()}


@router.post("/api/jobs/purge-legacy-tags")
def purge_legacy_tags(c: AppContainer = Depends(get_container)) -> dict[str, Any]:
    """Queue the legacy auto-tag cleanup; it runs on the shared worker instead of blocking."""
    if c.index_job.running:
        raise HTTPException(409, "正在索引图片，请稍后再安排清理任务")
    if c.job_manager.is_busy(("purge_legacy",)):
        raise HTTPException(409, "清理旧标签任务已在队列中")
    job_id = c.job_manager.submit_purge_legacy_job(keep_new_models=True)
    return {"job_id": job_id, "job": c.jobs_repo.get(job_id)}


@router.get("/api/jobs/{job_id}")
def get_job(job_id: str, c: AppContainer = Depends(get_container)) -> dict[str, Any]:
    row = c.jobs_repo.get(job_id)
    if row is None:
        raise HTTPException(404, "任务不存在")
    return row


@router.post("/api/jobs/{job_id}/move")
def move_queued_job(
    job_id: str,
    request: JobMoveRequest,
    c: AppContainer = Depends(get_container),
) -> dict[str, Any]:
    moved = c.job_manager.move_queued_job(job_id, request.direction)
    return {"moved": moved, "queue": c.job_manager.queue_snapshot()}


@router.delete("/api/jobs/{job_id}")
def remove_job(
    job_id: str,
    c: AppContainer = Depends(get_container),
) -> dict[str, Any]:
    try:
        removed = c.job_manager.remove_job(job_id)
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    if not removed:
        raise HTTPException(404, "任务不存在或已不在队列中")
    return {"removed": True, "queue": c.job_manager.queue_snapshot()}


@router.post("/api/jobs/pause")
def pause_jobs(c: AppContainer = Depends(get_container)) -> dict[str, Any]:
    return {"paused": c.job_manager.pause()}


@router.post("/api/jobs/resume")
def resume_jobs(c: AppContainer = Depends(get_container)) -> dict[str, Any]:
    return {"resumed": c.job_manager.resume()}


@router.post("/api/jobs/stop")
def stop_jobs(c: AppContainer = Depends(get_container)) -> dict[str, Any]:
    return {"stopped": c.job_manager.stop()}


def _scope_rows_for_folders(
    c: AppContainer,
    folder_keys: list[str] | None,
) -> list[int] | None:
    if folder_keys is None:
        return None
    resolved = resolve_folder_keys_from_db(c.images, folder_keys)
    arr = c.resolver.rows(ImageScope(folders=tuple(resolved), folder_recursive=False))
    return arr.tolist() if arr is not None else None


@router.post("/api/jobs/retag")
def submit_retag_job(
    request: RetagJobRequest,
    c: AppContainer = Depends(get_container),
) -> dict[str, Any]:
    if c.index_job.running:
        raise HTTPException(409, "正在索引图片，请稍后再安排标签任务")
    scope_rows = _scope_rows_for_folders(c, request.folder_keys)
    try:
        job_id = c.job_manager.submit_retag_job(
            backend=request.backend,
            scope_row_ids=scope_rows,
            skip_completed=request.skip_completed,
            limit=request.limit,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"job_id": job_id, "job": c.jobs_repo.get(job_id)}


@router.post("/api/jobs/vision-enrich")
def submit_vision_enrich_job(
    request: VisionTaskRequest,
    c: AppContainer = Depends(get_container),
) -> dict[str, Any]:
    if c.index_job.running:
        raise HTTPException(409, "正在索引图片，请稍后再安排视觉属性任务")
    scope_rows = _scope_rows_for_folders(c, request.folder_keys)
    job_id = c.job_manager.submit_vision_enrich_job(
        scope_row_ids=scope_rows,
        skip_completed=request.skip_completed,
        limit=request.limit,
    )
    return {"job_id": job_id, "job": c.jobs_repo.get(job_id)}


@router.post("/api/jobs/ccip")
def submit_ccip_job(
    request: CcipTaskRequest,
    c: AppContainer = Depends(get_container),
) -> dict[str, Any]:
    if c.index_job.running:
        raise HTTPException(409, "正在索引图片，请稍后再安排角色特征任务")
    scope_rows = _scope_rows_for_folders(c, request.folder_keys)
    try:
        job_id = c.job_manager.submit_ccip_job(
            scope_row_ids=scope_rows,
            skip_completed=request.skip_completed,
            limit=request.limit,
            require_vision=not request.allow_incomplete_vision,
        )
    except JobDependencyError as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"job_id": job_id, "job": c.jobs_repo.get(job_id)}


@router.post("/api/jobs/ccip/reset")
def reset_ccip_job(
    request: CcipResetRequest,
    c: AppContainer = Depends(get_container),
) -> dict[str, Any]:
    """Invalidate stored CCIP vectors so they are recomputed from freshly detected crops."""
    if c.index_job.running:
        raise HTTPException(409, "正在索引图片，请稍后再安排向量重置任务")
    if c.job_manager.is_busy(("ccip", "ccip_reset")):
        raise HTTPException(409, "CCIP 任务已在队列中，请先等待或停止")
    scope_rows = _scope_rows_for_folders(c, request.folder_keys)
    job_id = c.job_manager.submit_ccip_reset_job(scope_row_ids=scope_rows)
    return {"job_id": job_id, "job": c.jobs_repo.get(job_id)}


@router.post("/api/images/{point_id}/retag")
def retag_single_image(
    point_id: str,
    request: SingleRetagRequest,
    c: AppContainer = Depends(get_container),
) -> dict[str, Any]:
    if c.index_job.running:
        raise HTTPException(409, "正在索引图片，请稍后再重标标签")
    try:
        payload = c.job_manager.retag_single_image(point_id, backend=request.backend)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    items = attach_provenance(c.prov_repo, [gallery_result(point_id, payload)])
    return {"image": items[0]}
