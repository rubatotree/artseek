"""Index management and unified background job status routes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from ...image_io import canonical_path
from ..deps import AppContainer, get_container
from ..schemas import IndexAddRequest, IndexRemoveRequest
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


@router.get("/api/jobs/{job_id}")
def get_job(job_id: str, c: AppContainer = Depends(get_container)) -> dict[str, Any]:
    row = c.jobs_repo.get(job_id)
    if row is None:
        raise HTTPException(404, "任务不存在")
    return row


@router.post("/api/jobs/pause")
def pause_jobs(c: AppContainer = Depends(get_container)) -> dict[str, Any]:
    return {"paused": c.job_manager.pause()}


@router.post("/api/jobs/resume")
def resume_jobs(c: AppContainer = Depends(get_container)) -> dict[str, Any]:
    return {"resumed": c.job_manager.resume()}


@router.post("/api/jobs/stop")
def stop_jobs(c: AppContainer = Depends(get_container)) -> dict[str, Any]:
    return {"stopped": c.job_manager.stop()}
