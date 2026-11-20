"""Index management and unified background job status routes."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from ...db.repos.images import ImageRepo
from ...image_io import (
    canonical_path,
    file_digest,
    load_rgb,
    perceptual_hash,
    point_id,
    scan_images,
)
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


def _run_index_job_m3(c: AppContainer, directories: list[Path], recursive: bool) -> None:
    """SQLite + VectorIndex indexer bridge for M3 before M4 replaces it with GPU pipeline."""
    import numpy as np

    job = c.index_job
    job.path = "、".join(str(item) for item in directories)
    job.done = 0
    job.total = 0
    job.error = None
    job.summary = None
    job.message = "正在准备索引…"
    tagger = None
    try:
        discovered = list(scan_images(directories, recursive=recursive))
        job.total = len(discovered)
        existing_rows: dict[str, tuple[int, int]] = {}
        conn = c.db.read_conn()
        for r in conn.execute(
            "SELECT path_key, mtime_ns, size_bytes FROM images WHERE deleted = 0"
        ).fetchall():
            existing_rows[str(r["path_key"])] = (int(r["mtime_ns"]), int(r["size_bytes"]))

        todo: list[tuple[Path, Path]] = []
        for root_dir, p in discovered:
            pkey = canonical_path(p)
            stat = p.stat()
            prev = existing_rows.get(pkey)
            if prev is not None and prev == (stat.st_mtime_ns, stat.st_size):
                continue
            todo.append((root_dir, p))

        job.message = "正在载入图文向量模型…"
        indexed_count = 0
        failed_count = 0
        if todo:
            with c.lifecycle.hold_model() as embedder:
                if c.config.tagger.enabled:
                    from ...models.tagger import WDTagger

                    tagger = WDTagger(c.config.tagger, c.config.embedding)
                job.message = "正在扫描并编码图片…"
                batch_size = max(1, int(c.config.embedding.batch_size))
                for start in range(0, len(todo), batch_size):
                    chunk = todo[start : start + batch_size]
                    records = []
                    rgb_imgs = []
                    desc_texts = []
                    for root_dir, img_path in chunk:
                        try:
                            rgb = load_rgb(img_path)
                            w, h = rgb.size
                            ph = perceptual_hash(rgb)
                            digest = file_digest(img_path)
                            stat = img_path.stat()
                            tags = tagger.predict(rgb) if tagger is not None else {}
                            tag_list = list(tags.keys())
                            desc = ", ".join(t.replace("_", " ") for t in tag_list) or img_path.stem
                            records.append(
                                ImageRepo.build_upsert_record(
                                    img_path,
                                    root=root_dir,
                                    point_id=point_id(img_path),
                                    width=w,
                                    height=h,
                                    mtime_ns=stat.st_mtime_ns,
                                    size_bytes=stat.st_size,
                                    content_hash=digest,
                                    phash=ph,
                                    tags=tag_list,
                                    tag_scores=tags,
                                    descriptor_text=desc,
                                )
                            )
                            rgb_imgs.append(rgb)
                            desc_texts.append(desc)
                        except Exception:
                            failed_count += 1
                    if records:
                        vis_vecs = embedder.encode_images(rgb_imgs)
                        desc_vecs = embedder.encode_texts(desc_texts)
                        row_ids = c.images.upsert_images_batch(records)
                        c.vectors.set_rows(
                            row_ids,
                            np.asarray(vis_vecs, dtype=np.float32),
                            np.asarray(desc_vecs, dtype=np.float32),
                        )
                        indexed_count += len(records)
                    job.done = min(start + len(chunk), len(todo))
                    job.message = f"正在索引 {job.done}/{len(todo)}"
                c.vectors.flush()
        job.summary = {
            "discovered": len(discovered),
            "indexed": indexed_count,
            "unchanged": len(discovered) - len(todo),
            "deleted": 0,
            "failed": failed_count,
        }
        job.message = "索引完成"
    except Exception as exc:
        job.error = str(exc)
        job.message = "索引失败"
    finally:
        closer = getattr(tagger, "close", None)
        if callable(closer):
            closer()
        c.invalidate_caches()
        job.running = False
        if job.error is None:
            c.provenance_refresh_pending = True
            drain_provenance_refresh(c)


@router.get("/api/index-status")
def index_status(c: AppContainer = Depends(get_container)) -> dict[str, Any]:
    return _job_status(c)


@router.post("/api/index")
def add_index(
    request: IndexAddRequest,
    c: AppContainer = Depends(get_container),
) -> dict[str, Any]:
    directories = _resolve_index_directories(request)
    with c.index_lock:
        if c.index_job.running:
            raise HTTPException(409, "已有索引任务正在进行")
        c.index_job.running = True
        c.index_job.message = "正在排队…"
        c.index_job.error = None
        c.index_job.summary = None
    threading.Thread(
        target=_run_index_job_m3,
        args=(c, directories, request.recursive),
        daemon=True,
        name="artseek-index",
    ).start()
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
