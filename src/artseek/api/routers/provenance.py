"""Provenance statistics, scan/enrich/auto-tag background jobs, and manual edits."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from ..deps import AppContainer, get_container
from ..schemas import (
    AutoTagRequest,
    ProvenanceEnrichRequest,
    ProvenanceManualRequest,
    ProvenanceScanRequest,
    ProvenanceUrlRequest,
    UrlIgnoredRequest,
)

router = APIRouter()


def _job_payload(c: AppContainer) -> dict[str, Any]:
    job = c.provenance_job
    return {
        "running": job.running,
        "paused": job.paused,
        "kind": job.kind,
        "done": job.done,
        "total": job.total,
        "message": job.message,
        "error": job.error,
        "summary": job.summary,
        "budgets": c.provenance.budgets(),
    }


def start_auto_tag_job(c: AppContainer, *, limit: int | None = 500) -> bool:
    job = c.provenance_job
    pending = c.prov_repo.point_ids_without_auto_tags(limit)
    with c.provenance_lock:
        if job.running:
            return False
        job.running = True
        job.paused = False
        job.kind = "auto_tag"
        job.done = 0
        job.total = len(pending)
        job.message = (
            f"待补 Tag {len(pending)} 张" if pending else "没有待自动 Tag 的图片"
        )
        job.error = None
        job.summary = None
        job.stop_event.clear()
        job.pause_event.clear()

    if not pending:
        job.running = False
        drain_provenance_refresh(c)
        return True

    def progress(done: int, total: int, _label: str) -> None:
        job.done = done
        job.total = total
        job.message = f"正在生成自动 Tag {done}/{total}"

    def run() -> None:
        try:
            summary = c.provenance.run_auto_tag(
                limit=limit,
                progress=progress,
                stop_event=job.stop_event,
                pause_event=job.pause_event,
            )
            job.summary = summary
            job.message = (
                f"自动 Tag 完成：成功 {summary['ok']}，"
                f"跳过 {summary['skipped']}，失败 {summary['failed']}"
            )
        except Exception as exc:
            job.error = str(exc)
            job.message = "自动 Tag 失败"
        finally:
            job.running = False
            job.paused = False
            drain_provenance_refresh(c)

    threading.Thread(target=run, daemon=True, name="artseek-auto-tag").start()
    return True


def start_enrich_job(
    c: AppContainer,
    *,
    platforms: list[str],
    limit: int | None = None,
    point_ids: list[str] | None = None,
    reset_stop: bool = True,
    allow_running: bool = False,
    then_auto_tag: bool = False,
) -> bool:
    job = c.provenance_job
    clean_platforms = [p for p in platforms if p in {"pixiv", "twitter"}]
    if not clean_platforms:
        return False
    records = c.provenance.select_enrich_records(
        platforms=clean_platforms,
        limit=limit,
        point_ids=point_ids,
    )
    with c.provenance_lock:
        if job.running and not allow_running:
            return False
        if not reset_stop and job.stop_event.is_set():
            return False
        job.running = True
        job.paused = False
        job.kind = "enrich"
        job.done = 0
        job.total = len(records)
        job.message = f"待补全 {len(records)} 条来源" if records else "没有待补全的来源"
        job.error = None
        job.summary = None
        if reset_stop:
            job.stop_event.clear()
        job.pause_event.clear()

    if not records:
        job.running = False
        if then_auto_tag:
            return start_auto_tag_job(c, limit=None)
        drain_provenance_refresh(c)
        return True

    def progress(done: int, total: int, label: str) -> None:
        job.done = done
        job.total = total
        job.message = f"正在补全来源 {done}/{total}" + (f"：{label}" if label else "")

    def run() -> None:
        from ...provenance.scheduler import BudgetExhausted

        try:
            summary = c.provenance.run_enrich(
                records,
                progress=progress,
                stop_event=job.stop_event,
                pause_event=job.pause_event,
            )
            job.summary = summary
            job.message = (
                f"来源补全完成：成功 {summary['ok']}，失败 {summary['failed']}"
                if not summary.get("stopped")
                else f"来源补全已暂停：成功 {summary['ok']}，失败 {summary['failed']}"
            )
        except BudgetExhausted as exc:
            job.message = str(exc)
        except Exception as exc:
            job.error = str(exc)
            job.message = "来源补全失败"
        finally:
            job.running = False
            job.paused = False
            if then_auto_tag and not job.stop_event.is_set():
                start_auto_tag_job(c, limit=None)
            else:
                drain_provenance_refresh(c)

    threading.Thread(target=run, daemon=True, name="artseek-provenance").start()
    return True


def start_scan_job(
    c: AppContainer,
    *,
    deep: bool = False,
    only_new: bool = False,
    auto_enrich: bool | None = None,
) -> bool:
    job = c.provenance_job
    with c.provenance_lock:
        if job.running:
            return False
        job.running = True
        job.paused = False
        c.provenance_refresh_pending = False
        job.kind = "scan"
        job.done = 0
        job.total = 0
        job.message = "正在扫描文件名与嵌入元信息…"
        job.error = None
        job.summary = None
        job.stop_event.clear()
        job.pause_event.clear()

    def progress(done: int, total: int, label: str) -> None:
        job.done = done
        job.total = total
        if label:
            job.message = f"正在扫描来源 {done}/{total}：{Path(label).name}"

    should_enrich = c.config.provenance.auto_enrich if auto_enrich is None else auto_enrich

    def run() -> None:
        try:
            summary = c.provenance.run_scan(
                deep=deep,
                only_new=only_new,
                progress=progress,
                stop_event=job.stop_event,
                pause_event=job.pause_event,
            )
            job.summary = summary
            job.message = (
                f"本地扫描完成：识别 {summary['identified']}，"
                f"未识别 {summary['unresolved']}"
            )
        except Exception as exc:
            job.error = str(exc)
            job.message = "来源扫描失败"
        finally:
            started_next = False
            if not job.stop_event.is_set() and job.error is None:
                if should_enrich:
                    started_next = start_enrich_job(
                        c,
                        platforms=["pixiv", "twitter"],
                        reset_stop=False,
                        allow_running=True,
                        then_auto_tag=True,
                    )
                else:
                    job.running = False
                    job.paused = False
                    started_next = start_auto_tag_job(c, limit=None)
            if not started_next:
                job.running = False
                job.paused = False
                drain_provenance_refresh(c)

    threading.Thread(target=run, daemon=True, name="artseek-provenance-scan").start()
    return True


def drain_provenance_refresh(c: AppContainer) -> None:
    if not c.provenance_refresh_pending or c.provenance_job.running:
        return
    try:
        start_scan_job(c, deep=False, only_new=True)
    except Exception:
        pass


@router.get("/api/provenance/stats")
def provenance_stats(c: AppContainer = Depends(get_container)) -> dict[str, Any]:
    stats = c.provenance.stats()
    stats["job"] = _job_payload(c)
    return stats


@router.get("/api/provenance/job")
def provenance_job_status(c: AppContainer = Depends(get_container)) -> dict[str, Any]:
    return _job_payload(c)


@router.post("/api/provenance/scan")
def provenance_scan(
    request: ProvenanceScanRequest,
    c: AppContainer = Depends(get_container),
) -> dict[str, Any]:
    if c.index_job.running:
        raise HTTPException(409, "正在索引图片，请稍后再扫描来源")
    if not start_scan_job(
        c,
        deep=request.deep,
        only_new=request.only_new,
        auto_enrich=request.auto_enrich,
    ):
        raise HTTPException(409, "已有来源任务在运行")
    return _job_payload(c)


@router.post("/api/provenance/enrich")
def provenance_enrich(
    request: ProvenanceEnrichRequest,
    c: AppContainer = Depends(get_container),
) -> dict[str, Any]:
    if c.index_job.running:
        raise HTTPException(409, "正在索引图片，请稍后再补全来源")
    if not request.point_ids and c.provenance.stats()["total"] == 0:
        if not start_scan_job(c, deep=False, only_new=True, auto_enrich=True):
            raise HTTPException(409, "已有来源任务在运行")
        return _job_payload(c)
    if not start_enrich_job(
        c,
        platforms=list(request.platforms),
        limit=request.limit,
        point_ids=list(request.point_ids),
    ):
        raise HTTPException(409, "已有来源任务在运行")
    return _job_payload(c)


@router.post("/api/provenance/auto-tag")
def provenance_auto_tag(
    request: AutoTagRequest,
    c: AppContainer = Depends(get_container),
) -> dict[str, Any]:
    if not start_auto_tag_job(c, limit=request.limit):
        raise HTTPException(409, "已有来源任务在运行")
    return _job_payload(c)


@router.post("/api/provenance/job/pause")
def provenance_pause(c: AppContainer = Depends(get_container)) -> dict[str, Any]:
    job = c.provenance_job
    if job.running:
        job.pause_event.set()
        job.paused = True
        job.message = "来源任务已暂停"
    return _job_payload(c)


@router.post("/api/provenance/job/resume")
def provenance_resume(c: AppContainer = Depends(get_container)) -> dict[str, Any]:
    job = c.provenance_job
    if job.running:
        job.pause_event.clear()
        job.paused = False
        job.message = "来源任务已继续"
    return _job_payload(c)


@router.post("/api/provenance/job/stop")
def provenance_stop(c: AppContainer = Depends(get_container)) -> dict[str, Any]:
    job = c.provenance_job
    if job.running:
        job.stop_event.set()
        job.pause_event.clear()
        job.paused = False
        job.message = "正在停止来源任务…"
    return _job_payload(c)


@router.get("/api/images/{point_id}/provenance")
def image_provenance(
    point_id: str,
    c: AppContainer = Depends(get_container),
) -> dict[str, Any]:
    return c.provenance.get_image_provenance(point_id)


@router.put("/api/images/{point_id}/provenance")
def update_image_provenance(
    point_id: str,
    request: ProvenanceManualRequest,
    c: AppContainer = Depends(get_container),
) -> dict[str, Any]:
    return c.provenance.update_manual(
        point_id,
        clear_manual=request.clear_manual,
        url=request.url,
        title=request.title,
        description=request.description,
        tags=request.tags,
        manual_tags=request.manual_tags,
        author_key=request.author_key,
        author_query=request.author_query,
        author_name=request.author_name,
        author_handle=request.author_handle,
        pixiv_id=request.pixiv_id,
        x_handle=request.x_handle,
    )


@router.post("/api/images/{point_id}/provenance/resolve")
def resolve_image_provenance(
    point_id: str,
    request: ProvenanceUrlRequest,
    c: AppContainer = Depends(get_container),
) -> dict[str, Any]:
    return c.provenance.resolve_url(point_id, request.url)


@router.post("/api/images/{point_id}/provenance/ignore-url")
def ignore_image_provenance_url(
    point_id: str,
    request: UrlIgnoredRequest,
    c: AppContainer = Depends(get_container),
) -> dict[str, Any]:
    return c.provenance.set_url_ignored(point_id, ignored=request.ignored)
