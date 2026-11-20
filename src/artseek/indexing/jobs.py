"""Unified background JobManager for indexing, provenance, caption backfill, and compact."""

from __future__ import annotations

import itertools
import threading
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from ..compute import release_accelerator
from ..config import AppConfig
from ..db.connection import Database
from ..db.repos.images import ImageRepo
from ..db.repos.jobs import JobRepo
from ..db.repos.provenance import ProvenanceRepo
from ..db.repos.task_state import TaskStateRepo
from ..image_io import composite_on_white, load_rgb
from ..ml.captioner import Captioner
from ..ml.ccip import CCIPExtractor
from ..ml.llm import OllamaUnavailable
from ..ml.tag_postprocess import (
    CAT_ATTRIBUTE,
    CAT_MEDIUM,
    CAT_RATING,
    build_high_density_descriptor,
)
from ..ml.tagger import (
    build_tagger,
    model_versions_for_backend,
    normalize_tagger_backend,
)
from ..ml.vision_attributes import VISION_META_VERSION, VisionAttributeAnalyzer
from ..services.model_lifecycle import ModelLifecycle
from ..services.provenance import ProvenanceAppService
from ..vectors.index import VectorIndex
from .scan import IncrementalScanner

JOB_PRIORITIES: dict[str, int] = {
    "index": 10,
    "compact": 20,
    "provenance_scan": 30,
    "provenance_enrich": 40,
    "retag": 45,
    "auto_tag": 50,
    "vision_enrich": 55,
    "ccip": 60,
    "caption": 90,
}


@dataclass(order=True)
class _QueuedTask:
    priority: int
    seq: int
    job_id: str = field(compare=False)
    kind: str = field(compare=False)
    runner: Callable[[], None] = field(compare=False)


class JobManager:
    """Persists background jobs in SQLite `jobs` table and executes them on a priority worker."""

    def __init__(
        self,
        config: AppConfig,
        db: Database,
        images: ImageRepo,
        prov_repo: ProvenanceRepo,
        jobs_repo: JobRepo,
        vectors: VectorIndex,
        lifecycle: ModelLifecycle,
        provenance_service: ProvenanceAppService,
        *,
        on_caches_invalidated: Callable[[], None] | None = None,
        on_index_completed: Callable[[], None] | None = None,
        index_state: Any = None,
        provenance_state: Any = None,
    ) -> None:
        self.config = config
        self.db = db
        self.images = images
        self.prov_repo = prov_repo
        self.jobs_repo = jobs_repo
        self.task_state = TaskStateRepo(db)
        self.vectors = vectors
        self.lifecycle = lifecycle
        self.provenance_service = provenance_service
        self._on_caches_invalidated = on_caches_invalidated
        self._on_index_completed = on_index_completed
        self.index_state = index_state
        self.provenance_state = provenance_state

        self._lock = threading.RLock()
        self._cond = threading.Condition(self._lock)
        self._queue: list[_QueuedTask] = []
        self._seq = itertools.count(1)
        self._active_job_id: str | None = None
        self._active_kind: str | None = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._shutdown = False

        try:
            self.jobs_repo.mark_interrupted_running()
        except Exception:
            pass

        self._worker = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="artseek-job-worker",
        )
        self._worker.start()

    def close(self) -> None:
        with self._cond:
            self._shutdown = True
            self._stop_event.set()
            self._pause_event.clear()
            self._queue.clear()
            self._cond.notify_all()

    def is_busy(self, kinds: Sequence[str] | None = None) -> bool:
        with self._lock:
            if kinds is None:
                return self._active_job_id is not None or bool(self._queue)
            kset = set(kinds)
            if self._active_kind in kset:
                return True
            return any(t.kind in kset for t in self._queue)

    def list_jobs(self, limit: int = 20) -> list[dict[str, Any]]:
        return self.jobs_repo.list_recent(limit=limit)

    def pause(self) -> bool:
        with self._lock:
            if self._active_job_id is None:
                return False
            self._pause_event.set()
            if self.provenance_state is not None:
                self.provenance_state.paused = True
                self.provenance_state.pause_event.set()
            self.jobs_repo.update(self._active_job_id, status="paused")
            return True

    def resume(self) -> bool:
        with self._lock:
            if self._active_job_id is None:
                return False
            self._pause_event.clear()
            if self.provenance_state is not None:
                self.provenance_state.paused = False
                self.provenance_state.pause_event.clear()
            self.jobs_repo.update(self._active_job_id, status="running")
            return True

    def stop(self) -> bool:
        with self._cond:
            had_work = self._active_job_id is not None or bool(self._queue)
            self._stop_event.set()
            self._pause_event.clear()
            if self.provenance_state is not None:
                self.provenance_state.stop_event.set()
                self.provenance_state.pause_event.clear()
            for task in self._queue:
                self.jobs_repo.update(task.job_id, status="stopped", message="已取消")
            self._queue.clear()
            self._cond.notify_all()
            return had_work

    # ---------------------------------------------------------------- enqueue helpers
    def submit_index_job(
        self,
        directories: Sequence[Path],
        *,
        recursive: bool = True,
        force: bool = False,
        captions: bool | None = None,
    ) -> str:
        with self._cond:
            if self.is_busy(("index",)):
                raise RuntimeError("已有索引任务正在进行")
            job_id = f"index-{uuid.uuid4().hex[:12]}"
            path_str = "、".join(str(d) for d in directories)
            self.jobs_repo.create(
                job_id,
                "index",
                {
                    "paths": [str(d) for d in directories],
                    "recursive": bool(recursive),
                    "force": bool(force),
                },
                status="queued",
                message="正在排队…",
            )
            if self.index_state is not None:
                self.index_state.running = True
                self.index_state.path = path_str
                self.index_state.done = 0
                self.index_state.total = 0
                self.index_state.message = "正在排队…"
                self.index_state.error = None
                self.index_state.summary = None

            use_captions = (
                self.config.llm.caption_on_index if captions is None else bool(captions)
            )

            def _run() -> None:
                self._execute_index(
                    job_id,
                    list(directories),
                    recursive=recursive,
                    force=force,
                    enqueue_captions=use_captions,
                )

            self._insert_task_locked(
                _QueuedTask(
                    JOB_PRIORITIES["index"],
                    next(self._seq),
                    job_id,
                    "index",
                    _run,
                ),
                use_priority=True,
            )
            self._cond.notify_all()
            return job_id

    def submit_caption_backfill_job(self, *, limit: int | None = None) -> str:
        with self._cond:
            job_id = f"caption-{uuid.uuid4().hex[:12]}"
            self.jobs_repo.create(
                job_id,
                "caption",
                {"limit": limit},
                status="queued",
                message="等待生成 VLM 图片长描述…",
            )

            def _run() -> None:
                self._execute_caption_backfill(job_id, limit=limit)

            self._insert_task_locked(
                _QueuedTask(
                    JOB_PRIORITIES["caption"],
                    next(self._seq),
                    job_id,
                    "caption",
                    _run,
                ),
                use_priority=True,
            )
            self._cond.notify_all()
            return job_id

    def _insert_task_locked(self, task: _QueuedTask, *, use_priority: bool = True) -> None:
        if not use_priority:
            self._queue.append(task)
            return
        idx = len(self._queue)
        for pos, existing in enumerate(self._queue):
            if task.priority < existing.priority:
                idx = pos
                break
        self._queue.insert(idx, task)

    def _enqueue_job(
        self,
        kind: str,
        prefix: str,
        params: dict[str, Any],
        message: str,
        runner: Callable[[str], None],
        *,
        use_priority: bool = False,
    ) -> str:
        with self._cond:
            job_id = f"{prefix}-{uuid.uuid4().hex[:12]}"
            self.jobs_repo.create(
                job_id,
                kind,
                params,
                status="queued",
                message=message,
            )

            def _run() -> None:
                runner(job_id)

            self._insert_task_locked(
                _QueuedTask(
                    JOB_PRIORITIES.get(kind, 50),
                    next(self._seq),
                    job_id,
                    kind,
                    _run,
                ),
                use_priority=use_priority,
            )
            self._cond.notify_all()
            return job_id

    def queue_snapshot(self, *, history_limit: int = 12) -> dict[str, Any]:
        """Return active job, ordered queued jobs, and recent finished jobs."""
        with self._cond:
            active_id = self._active_job_id
            queued_ids = [t.job_id for t in self._queue]
            paused = self._pause_event.is_set()
        active = self.jobs_repo.get(active_id) if active_id else None
        queued: list[dict[str, Any]] = []
        for jid in queued_ids:
            row = self.jobs_repo.get(jid)
            if row is not None:
                queued.append(row)
        skip_ids = {jid for jid in ([active_id] if active_id else [])} | set(queued_ids)
        history = [
            row
            for row in self.jobs_repo.list_recent(limit=max(1, int(history_limit)) + len(skip_ids))
            if row["id"] not in skip_ids
        ][: max(1, int(history_limit))]
        return {
            "active": active,
            "queued": queued,
            "history": history,
            "paused": paused,
        }

    def move_queued_job(
        self,
        job_id: str,
        direction: Literal["up", "down", "top", "bottom"],
    ) -> bool:
        with self._cond:
            idx = next((i for i, t in enumerate(self._queue) if t.job_id == job_id), None)
            if idx is None:
                return False
            task = self._queue.pop(idx)
            if direction == "up":
                new_idx = max(0, idx - 1)
            elif direction == "down":
                new_idx = min(len(self._queue), idx + 1)
            elif direction == "top":
                new_idx = 0
            else:
                new_idx = len(self._queue)
            self._queue.insert(new_idx, task)
            return True

    def remove_job(self, job_id: str) -> bool:
        with self._cond:
            if self._active_job_id == job_id:
                raise RuntimeError("运行中的任务无法直接删除，请先停止当前任务")
            before = len(self._queue)
            self._queue = [t for t in self._queue if t.job_id != job_id]
            in_queue = len(self._queue) < before
            self._cond.notify_all()
        deleted = self.jobs_repo.delete(job_id)
        return in_queue or deleted

    def clear_finished_jobs(self) -> int:
        return self.jobs_repo.clear_finished()

    def task_summary(self, scope_row_ids: Sequence[int] | None = None) -> dict[str, int]:
        return self.task_state.summary_counts(scope_row_ids=scope_row_ids)

    def submit_retag_job(
        self,
        *,
        backend: str | None = None,
        scope_row_ids: Sequence[int] | None = None,
        skip_completed: bool = True,
        limit: int | None = None,
    ) -> str:
        canon = normalize_tagger_backend(backend or self.config.tagger.backend)
        row_list = [int(r) for r in scope_row_ids] if scope_row_ids is not None else None
        return self._enqueue_job(
            "retag",
            "retag",
            {
                "backend": canon,
                "scope_count": len(row_list) if row_list is not None else None,
                "skip_completed": bool(skip_completed),
                "limit": limit,
            },
            f"等待执行自动标签 ({canon})…",
            lambda jid: self._execute_retag(
                jid,
                backend=canon,
                scope_row_ids=row_list,
                skip_completed=skip_completed,
                limit=limit,
            ),
        )

    def submit_vision_enrich_job(
        self,
        *,
        scope_row_ids: Sequence[int] | None = None,
        skip_completed: bool = True,
        limit: int | None = None,
    ) -> str:
        row_list = [int(r) for r in scope_row_ids] if scope_row_ids is not None else None
        return self._enqueue_job(
            "vision_enrich",
            "vision",
            {
                "scope_count": len(row_list) if row_list is not None else None,
                "skip_completed": bool(skip_completed),
                "limit": limit,
            },
            "等待提取视觉属性（媒介/分级/黑白/AI/人脸焦点）…",
            lambda jid: self._execute_vision_enrich(
                jid,
                scope_row_ids=row_list,
                skip_completed=skip_completed,
                limit=limit,
            ),
        )

    def submit_ccip_job(
        self,
        *,
        scope_row_ids: Sequence[int] | None = None,
        skip_completed: bool = True,
        limit: int | None = None,
    ) -> str:
        row_list = [int(r) for r in scope_row_ids] if scope_row_ids is not None else None
        return self._enqueue_job(
            "ccip",
            "ccip",
            {
                "scope_count": len(row_list) if row_list is not None else None,
                "skip_completed": bool(skip_completed),
                "limit": limit,
            },
            "等待提取 CCIP 角色特征向量…",
            lambda jid: self._execute_ccip(
                jid,
                scope_row_ids=row_list,
                skip_completed=skip_completed,
                limit=limit,
            ),
        )

    def submit_provenance_scan_job(
        self,
        *,
        deep: bool = False,
        only_new: bool = True,
        auto_enrich: bool = True,
    ) -> str:
        label = "深度来源扫描" if deep else "本地增量扫源"
        return self._enqueue_job(
            "provenance_scan",
            "pscan",
            {"deep": bool(deep), "only_new": bool(only_new), "auto_enrich": bool(auto_enrich)},
            f"等待执行{label}…",
            lambda jid: self._execute_provenance_scan(
                jid,
                deep=deep,
                only_new=only_new,
                auto_enrich=auto_enrich,
            ),
        )

    def submit_provenance_enrich_job(
        self,
        *,
        platforms: Sequence[str] = ("pixiv", "twitter"),
        limit: int | None = None,
    ) -> str:
        clean_platforms = [
            p for p in platforms if p in {"pixiv", "twitter"}
        ] or ["pixiv", "twitter"]
        return self._enqueue_job(
            "provenance_enrich",
            "penrich",
            {"platforms": clean_platforms, "limit": limit},
            f"等待补全 {'/'.join(clean_platforms)} 来源元数据…",
            lambda jid: self._execute_provenance_enrich(
                jid,
                platforms=clean_platforms,
                limit=limit,
            ),
        )

    def auto_fill_queue(
        self,
        *,
        backend: str | None = None,
        scope_row_ids: Sequence[int] | None = None,
        include_provenance: bool = False,
    ) -> list[str]:
        """Automatically enqueue incomplete pipeline steps not already active or queued."""
        canon = normalize_tagger_backend(backend or self.config.tagger.backend)
        with self._cond:
            busy_kinds = {t.kind for t in self._queue}
            if self._active_kind:
                busy_kinds.add(self._active_kind)

        added: list[str] = []
        if (
            include_provenance
            and "provenance_scan" not in busy_kinds
            and self.images.count() > int(self.prov_repo.stats().get("total") or 0)
        ):
            added.append(
                self.submit_provenance_scan_job(deep=False, only_new=True, auto_enrich=True)
            )

        req_versions = model_versions_for_backend(canon)
        if "retag" not in busy_kinds and self.task_state.pending_for_tagger(
            req_versions, scope_row_ids=scope_row_ids, skip_completed=True, limit=1
        ):
            added.append(
                self.submit_retag_job(
                    backend=canon,
                    scope_row_ids=scope_row_ids,
                    skip_completed=True,
                )
            )

        if "vision_enrich" not in busy_kinds and self.task_state.pending_for_vision(
            min_version=VISION_META_VERSION,
            scope_row_ids=scope_row_ids,
            skip_completed=True,
            limit=1,
        ):
            added.append(
                self.submit_vision_enrich_job(
                    scope_row_ids=scope_row_ids,
                    skip_completed=True,
                )
            )

        if "ccip" not in busy_kinds and self.task_state.pending_for_ccip(
            scope_row_ids=scope_row_ids,
            skip_completed=True,
            limit=1,
        ):
            added.append(
                self.submit_ccip_job(
                    scope_row_ids=scope_row_ids,
                    skip_completed=True,
                )
            )

        if include_provenance and "provenance_enrich" not in busy_kinds:
            pending_recs = self.provenance_service.select_enrich_records(
                platforms=("pixiv", "twitter"), limit=1
            )
            if pending_recs:
                added.append(self.submit_provenance_enrich_job())

        return added

    # ---------------------------------------------------------------- worker loop
    def _worker_loop(self) -> None:
        while True:
            with self._cond:
                while not self._queue and not self._shutdown:
                    self._cond.wait()
                if self._shutdown:
                    return
                task = self._queue.pop(0)
                self._active_job_id = task.job_id
                self._active_kind = task.kind
                self._stop_event.clear()
                self._pause_event.clear()

            try:
                task.runner()
            finally:
                with self._cond:
                    self._active_job_id = None
                    self._active_kind = None
                    self._cond.notify_all()

    # ---------------------------------------------------------------- task implementations
    def _execute_index(
        self,
        job_id: str,
        directories: list[Path],
        *,
        recursive: bool,
        force: bool,
        enqueue_captions: bool,
    ) -> None:
        idx_state = self.index_state
        self.jobs_repo.update(job_id, status="running", message="正在准备索引…")
        if idx_state is not None:
            idx_state.running = True
            idx_state.message = "正在准备索引…"

        scanner = IncrementalScanner(
            self.config,
            self.db,
            self.images,
            self.vectors,
            cpu_workers=8,
        )

        def _on_progress(done: int, total: int, root_label: str) -> None:
            msg = f"正在索引 {done}/{total} ({Path(root_label).name or root_label})"
            if idx_state is not None:
                idx_state.done = done
                idx_state.total = total
                idx_state.message = msg
            self.jobs_repo.update(job_id, done=done, total=total, message=msg)

        try:
            with self.lifecycle.hold_model() as embedder:
                summary = scanner.index_roots(
                    directories,
                    recursive=recursive,
                    force=force,
                    embedder=embedder,
                    progress=_on_progress,
                    stop_event=self._stop_event,
                    pause_event=self._pause_event,
                )
            sum_dict = summary.to_dict()
            status = "stopped" if summary.stopped else "completed"
            msg = "索引已停止" if summary.stopped else "索引完成"
            if idx_state is not None:
                idx_state.done = summary.indexed
                idx_state.total = summary.discovered
                idx_state.summary = sum_dict
                idx_state.message = msg
            self.jobs_repo.update(
                job_id,
                status=status,
                done=summary.indexed,
                total=summary.discovered,
                message=msg,
                summary=sum_dict,
            )
            if self._on_caches_invalidated is not None:
                self._on_caches_invalidated()
            if not summary.stopped and self._on_index_completed is not None:
                self._on_index_completed()
            if enqueue_captions and summary.indexed > 0 and not summary.stopped:
                self.submit_caption_backfill_job()
        except Exception as exc:
            err = str(exc)
            if idx_state is not None:
                idx_state.error = err
                idx_state.message = "索引失败"
            self.jobs_repo.update(
                job_id,
                status="failed",
                message="索引失败",
                error=err,
            )
        finally:
            if idx_state is not None:
                idx_state.running = False

    def _execute_caption_backfill(
        self,
        job_id: str,
        *,
        limit: int | None = None,
    ) -> None:
        conn = self.db.read_conn()
        sql = """
            SELECT id, path
            FROM images
            WHERE deleted = 0
              AND (caption_json IS NULL OR caption_json = '' OR caption_json = 'null')
            ORDER BY id
        """
        params: tuple[Any, ...] = ()
        if limit is not None and limit > 0:
            sql += " LIMIT ?"
            params = (int(limit),)
        rows = conn.execute(sql, params).fetchall()
        total = len(rows)
        if total == 0:
            self.jobs_repo.update(
                job_id,
                status="completed",
                done=0,
                total=0,
                message="所有图片均已有长描述",
                summary={"updated": 0, "failed": 0},
            )
            return

        self.jobs_repo.update(
            job_id,
            status="running",
            done=0,
            total=total,
            message=f"正在后台生成长描述 0/{total}…",
        )
        captioner = Captioner(self.config.llm, client=self.lifecycle.llm)
        updated = 0
        failed = 0
        warning: str | None = None
        try:
            with self.lifecycle.hold_model() as embedder:
                for idx, row in enumerate(rows, start=1):
                    while self._pause_event.is_set() and not self._stop_event.is_set():
                        time.sleep(0.05)
                    if self._stop_event.is_set():
                        break
                    img_id = int(row["id"])
                    img_path = Path(str(row["path"]))
                    if not img_path.is_file():
                        failed += 1
                        continue
                    try:
                        cap = captioner.caption_image(img_path)
                    except OllamaUnavailable as exc:
                        warning = str(exc)
                        break
                    except Exception:
                        failed += 1
                        continue
                    payload = self.images.get_payloads_by_ids([img_id]).get(img_id)
                    tags = list(payload.get("tags", [])) if payload else []
                    desc_text = captioner.build_descriptor_text(tags, cap)
                    self.images.update_caption_and_descriptor(img_id, cap, desc_text)
                    self.task_state.mark_caption_done(img_id)
                    if desc_text.strip():
                        desc_vec = embedder.encode_texts([desc_text])[0]
                        vis_vec = self.vectors.visual_vectors([img_id]).get(img_id)
                        if vis_vec is not None:
                            self.vectors.set_rows(
                                [img_id],
                                vis_vec.reshape(1, -1),
                                np.asarray(desc_vec, dtype=np.float32).reshape(1, -1),
                            )
                    updated += 1
                    if idx % 10 == 0 or idx == total:
                        self.jobs_repo.update(
                            job_id,
                            done=idx,
                            total=total,
                            message=f"正在生成长描述 {idx}/{total}",
                        )
                if updated > 0:
                    self.vectors.flush()
            status = "stopped" if self._stop_event.is_set() else "completed"
            msg = (
                f"长描述已暂停：{updated}/{total}"
                if warning
                else f"长描述生成完成：更新 {updated} 张"
            )
            self.jobs_repo.update(
                job_id,
                status=status,
                done=updated,
                total=total,
                message=msg,
                error=warning,
                summary={"updated": updated, "failed": failed, "warning": warning},
            )
        except Exception as exc:
            self.jobs_repo.update(
                job_id,
                status="failed",
                message="长描述任务失败",
                error=str(exc),
            )

    def _free_vram_for_worker(self) -> None:
        if self.lifecycle.embedder is not None:
            self.lifecycle.unload_model(force=False)
        release_accelerator()

    def _wait_if_paused(self) -> bool:
        while self._pause_event.is_set() and not self._stop_event.is_set():
            time.sleep(0.05)
        return self._stop_event.is_set()

    @staticmethod
    def _load_image_preserve_info(path: Path) -> Any:
        from PIL import Image, ImageOps

        with Image.open(path) as img:
            img.seek(0)
            info = dict(getattr(img, "info", {}) or {})
            rgb = composite_on_white(ImageOps.exif_transpose(img))
            rgb.info = info
            return rgb

    @staticmethod
    def _build_descriptor_for_payload(
        pred: Any,
        payload: dict[str, Any],
        *,
        drop_overlap: bool,
    ) -> str:
        tags = list(pred.tags)
        cats = dict(pred.categories)
        if payload.get("media_type"):
            mtag = f"medium:{payload['media_type']}"
            tags.append(mtag)
            cats[mtag] = CAT_MEDIUM
        if payload.get("rating"):
            rtag = f"rating:{payload['rating']}"
            tags.append(rtag)
            cats[rtag] = CAT_RATING
        if payload.get("is_monochrome"):
            tags.append("monochrome")
            cats["monochrome"] = CAT_ATTRIBUTE
        if payload.get("is_ai") is not None:
            atag = "ai_generated" if payload["is_ai"] else "hand_drawn"
            tags.append(atag)
            cats[atag] = CAT_ATTRIBUTE
        return build_high_density_descriptor(
            tags,
            cats,
            caption=payload.get("caption"),
            drop_overlap=drop_overlap,
        )

    def _encode_and_flush_descriptors(
        self,
        updated_descriptors: list[tuple[int, str]],
    ) -> None:
        if not updated_descriptors:
            return
        with self.lifecycle.hold_model() as embedder:
            batch_size = max(1, int(self.config.embedding.batch_size))
            for start in range(0, len(updated_descriptors), batch_size):
                chunk = updated_descriptors[start : start + batch_size]
                row_ids = [rid for rid, _ in chunk]
                texts = [txt for _, txt in chunk]
                vecs = embedder.encode_texts(texts)
                self.vectors.set_descriptor_rows(
                    row_ids,
                    np.asarray(vecs, dtype=np.float32),
                )
        self.vectors.flush()

    def _execute_retag(
        self,
        job_id: str,
        *,
        backend: str,
        scope_row_ids: Sequence[int] | None = None,
        skip_completed: bool = True,
        limit: int | None = None,
    ) -> None:
        req_versions = model_versions_for_backend(backend)
        pending = self.task_state.pending_for_tagger(
            req_versions,
            scope_row_ids=scope_row_ids,
            skip_completed=skip_completed,
            limit=limit,
        )
        total = len(pending)
        if total == 0:
            self.jobs_repo.update(
                job_id,
                status="completed",
                done=0,
                total=0,
                message=f"所有目标图片均已完成 {backend} 标签",
                summary={"updated": 0, "failed": 0, "backend": backend},
            )
            return

        self.jobs_repo.update(
            job_id,
            status="running",
            done=0,
            total=total,
            message=f"正在加载标签器 {backend}（共 {total} 张待处理）…",
        )
        self._free_vram_for_worker()

        tagger = None
        updated = 0
        failed = 0
        updated_descriptors: list[tuple[int, str]] = []
        chunk_size = max(4, min(16, int(self.config.tagger.batch_size)))
        try:
            tagger = build_tagger(self.config.tagger, self.config.embedding, backend=backend)
            if tagger is None:
                raise RuntimeError(f"无法初始化标签器 {backend}")
            versions = dict(getattr(tagger, "model_versions", req_versions))

            for start in range(0, total, chunk_size):
                if self._wait_if_paused():
                    break
                chunk = pending[start : start + chunk_size]
                loaded: list[tuple[int, str, str, Any]] = []
                for img_id, pid, path_str, chash in chunk:
                    p = Path(path_str)
                    if not p.is_file():
                        failed += 1
                        continue
                    try:
                        loaded.append((img_id, pid, chash, load_rgb(p)))
                    except Exception:
                        failed += 1
                if not loaded:
                    continue

                preds = tagger.predict([item[3] for item in loaded])
                ids = [item[0] for item in loaded]
                existing_payloads = self.images.get_payloads_by_ids(ids, include_tags=False)

                db_updates: list[tuple[int, list[str], dict[str, float], dict[str, int], str]] = []
                done_entries: list[tuple[str, int, str]] = []
                for idx, (img_id, pid, chash, _) in enumerate(loaded):
                    pred = preds[idx]
                    payload = existing_payloads.get(img_id) or {}
                    desc_text = self._build_descriptor_for_payload(
                        pred, payload, drop_overlap=self.config.tagger.drop_overlap
                    )
                    db_updates.append((img_id, pred.tags, pred.scores, pred.categories, desc_text))
                    done_entries.append((pid, img_id, chash))
                    if desc_text.strip():
                        updated_descriptors.append((img_id, desc_text))

                self.images.update_tags_and_descriptor_batch(db_updates)
                self.task_state.mark_tag_models_done(done_entries, versions, descriptor_done=True)
                updated += len(loaded)
                processed = min(total, start + len(chunk))
                self.jobs_repo.update(
                    job_id,
                    done=processed,
                    total=total,
                    message=f"正在打标签 ({backend}) {processed}/{total}",
                )
        except Exception as exc:
            if tagger is not None:
                tagger.close()
            release_accelerator()
            self.jobs_repo.update(
                job_id,
                status="failed",
                done=updated,
                total=total,
                message=f"自动标签失败 ({backend})",
                error=str(exc),
            )
            return

        if tagger is not None:
            tagger.close()
        release_accelerator()

        try:
            if updated_descriptors:
                self.jobs_repo.update(
                    job_id,
                    done=updated,
                    total=total,
                    message=f"正在重建 {len(updated_descriptors)} 条文本语义向量…",
                )
                self._encode_and_flush_descriptors(updated_descriptors)
            if callable(self._on_caches_invalidated):
                self._on_caches_invalidated()
            status = "stopped" if self._stop_event.is_set() else "completed"
            self.jobs_repo.update(
                job_id,
                status=status,
                done=updated,
                total=total,
                message=f"自动标签完成 ({backend})：更新 {updated} 张，失败 {failed} 张",
                summary={"updated": updated, "failed": failed, "backend": backend},
            )
        except Exception as exc:
            self.jobs_repo.update(
                job_id,
                status="failed",
                done=updated,
                total=total,
                message="重建文本语义向量失败",
                error=str(exc),
            )

    def _execute_vision_enrich(
        self,
        job_id: str,
        *,
        scope_row_ids: Sequence[int] | None = None,
        skip_completed: bool = True,
        limit: int | None = None,
    ) -> None:
        pending = self.task_state.pending_for_vision(
            min_version=VISION_META_VERSION,
            scope_row_ids=scope_row_ids,
            skip_completed=skip_completed,
            limit=limit,
        )
        total = len(pending)
        if total == 0:
            self.jobs_repo.update(
                job_id,
                status="completed",
                done=0,
                total=0,
                message="所有目标图片均已提取视觉属性",
                summary={"updated": 0, "failed": 0},
            )
            return

        self.jobs_repo.update(
            job_id,
            status="running",
            done=0,
            total=total,
            message=f"正在加载视觉属性模型（共 {total} 张待处理）…",
        )
        self._free_vram_for_worker()

        analyzer: VisionAttributeAnalyzer | None = None
        updated = 0
        failed = 0
        chunk_size = 16
        try:
            analyzer = VisionAttributeAnalyzer(self.config.embedding)
            for start in range(0, total, chunk_size):
                if self._wait_if_paused():
                    break
                chunk = pending[start : start + chunk_size]
                loaded: list[tuple[int, str, str, Any]] = []
                for img_id, pid, path_str, chash in chunk:
                    p = Path(path_str)
                    if not p.is_file():
                        failed += 1
                        continue
                    try:
                        loaded.append((img_id, pid, chash, self._load_image_preserve_info(p)))
                    except Exception:
                        failed += 1
                if not loaded:
                    continue

                updates = analyzer.analyze_batch([(item[0], item[3]) for item in loaded])
                self.images.update_vision_attributes_batch(updates)
                self.task_state.mark_vision_done(
                    [(item[1], item[0], item[2]) for item in loaded],
                    version=VISION_META_VERSION,
                )
                updated += len(loaded)
                processed = min(total, start + len(chunk))
                self.jobs_repo.update(
                    job_id,
                    done=processed,
                    total=total,
                    message=f"正在提取视觉属性 {processed}/{total}",
                )
            if callable(self._on_caches_invalidated):
                self._on_caches_invalidated()
            status = "stopped" if self._stop_event.is_set() else "completed"
            self.jobs_repo.update(
                job_id,
                status=status,
                done=updated,
                total=total,
                message=f"视觉属性提取完成：更新 {updated} 张，失败 {failed} 张",
                summary={"updated": updated, "failed": failed},
            )
        except Exception as exc:
            self.jobs_repo.update(
                job_id,
                status="failed",
                done=updated,
                total=total,
                message="视觉属性提取失败",
                error=str(exc),
            )
        finally:
            if analyzer is not None:
                analyzer.close()
            release_accelerator()

    def _execute_ccip(
        self,
        job_id: str,
        *,
        scope_row_ids: Sequence[int] | None = None,
        skip_completed: bool = True,
        limit: int | None = None,
    ) -> None:
        pending = self.task_state.pending_for_ccip(
            scope_row_ids=scope_row_ids,
            skip_completed=skip_completed,
            limit=limit,
        )
        total = len(pending)
        if total == 0:
            self.jobs_repo.update(
                job_id,
                status="completed",
                done=0,
                total=0,
                message="所有目标图片均已提取 CCIP 角色向量",
                summary={"updated": 0, "failed": 0},
            )
            return

        self.jobs_repo.update(
            job_id,
            status="running",
            done=0,
            total=total,
            message=f"正在加载 CCIP 角色编码器（共 {total} 张待处理）…",
        )
        self._free_vram_for_worker()

        extractor: CCIPExtractor | None = None
        updated = 0
        failed = 0
        chunk_size = 16
        try:
            extractor = CCIPExtractor(self.config.embedding)
            for start in range(0, total, chunk_size):
                if self._wait_if_paused():
                    break
                chunk = pending[start : start + chunk_size]
                loaded: list[tuple[int, str, str, Any, str | None]] = []
                for img_id, pid, path_str, chash, crop_json in chunk:
                    p = Path(path_str)
                    if not p.is_file():
                        failed += 1
                        continue
                    try:
                        loaded.append((img_id, pid, chash, load_rgb(p), crop_json))
                    except Exception:
                        failed += 1
                if not loaded:
                    continue

                feats = extractor.extract_batch([(item[3], item[4]) for item in loaded])
                self.vectors.set_ccip_rows([item[0] for item in loaded], feats)
                self.task_state.mark_ccip_done([(item[1], item[0], item[2]) for item in loaded])
                updated += len(loaded)
                if updated % 128 == 0:
                    self.vectors.flush()
                processed = min(total, start + len(chunk))
                self.jobs_repo.update(
                    job_id,
                    done=processed,
                    total=total,
                    message=f"正在提取 CCIP 角色特征 {processed}/{total}",
                )
            if updated > 0:
                self.vectors.flush()
            status = "stopped" if self._stop_event.is_set() else "completed"
            self.jobs_repo.update(
                job_id,
                status=status,
                done=updated,
                total=total,
                message=f"CCIP 角色特征提取完成：更新 {updated} 张，失败 {failed} 张",
                summary={"updated": updated, "failed": failed},
            )
        except Exception as exc:
            self.jobs_repo.update(
                job_id,
                status="failed",
                done=updated,
                total=total,
                message="CCIP 角色特征提取失败",
                error=str(exc),
            )
        finally:
            if extractor is not None:
                extractor.close()
            release_accelerator()

    def _execute_provenance_scan(
        self,
        job_id: str,
        *,
        deep: bool,
        only_new: bool,
        auto_enrich: bool,
    ) -> None:
        self.jobs_repo.update(job_id, status="running", message="正在扫描本地图库来源…")

        def _progress(done: int, total: int, label: str) -> None:
            self.jobs_repo.update(
                job_id,
                done=done,
                total=total,
                message=f"正在扫描来源 {done}/{total} · {label}",
            )

        try:
            summary = self.provenance_service.run_scan(
                deep=deep,
                only_new=only_new,
                progress=_progress,
                stop_event=self._stop_event,
                pause_event=self._pause_event,
            )
            if auto_enrich and not self._stop_event.is_set():
                pending_recs = self.provenance_service.select_enrich_records(
                    platforms=("pixiv", "twitter"), limit=None
                )
                if pending_recs:
                    self.jobs_repo.update(
                        job_id,
                        message=f"本地扫描完成，正在补全 {len(pending_recs)} 条远端元数据…",
                    )
                    self.provenance_service.run_enrich(
                        pending_recs,
                        progress=_progress,
                        stop_event=self._stop_event,
                        pause_event=self._pause_event,
                    )
            status = "stopped" if self._stop_event.is_set() else "completed"
            self.jobs_repo.update(
                job_id,
                status=status,
                message=f"本地来源扫描完成：扫描 {summary.get('scanned', 0)} 张",
                summary=summary,
            )
        except Exception as exc:
            self.jobs_repo.update(
                job_id,
                status="failed",
                message="本地来源扫描失败",
                error=str(exc),
            )

    def _execute_provenance_enrich(
        self,
        job_id: str,
        *,
        platforms: Sequence[str],
        limit: int | None = None,
    ) -> None:
        records = self.provenance_service.select_enrich_records(
            platforms=platforms, limit=limit
        )
        total = len(records)
        if total == 0:
            self.jobs_repo.update(
                job_id,
                status="completed",
                done=0,
                total=0,
                message="没有待补全的 Pixiv / X 来源记录",
                summary={"total": 0, "ok": 0, "failed": 0},
            )
            return
        self.jobs_repo.update(
            job_id,
            status="running",
            done=0,
            total=total,
            message=f"正在补全 Pixiv / X 元数据 0/{total}…",
        )

        def _progress(done: int, total_cnt: int, label: str) -> None:
            self.jobs_repo.update(
                job_id,
                done=done,
                total=total_cnt,
                message=f"正在补全元数据 {done}/{total_cnt} · {label}",
            )

        try:
            summary = self.provenance_service.run_enrich(
                records,
                progress=_progress,
                stop_event=self._stop_event,
                pause_event=self._pause_event,
            )
            status = "stopped" if self._stop_event.is_set() else "completed"
            ok_cnt = int(summary.get("ok", 0))
            fail_cnt = int(summary.get("failed", 0))
            self.jobs_repo.update(
                job_id,
                status=status,
                done=ok_cnt,
                total=total,
                message=f"远端元数据补全完成：成功 {ok_cnt}，失败 {fail_cnt}",
                summary=summary,
            )
        except Exception as exc:
            self.jobs_repo.update(
                job_id,
                status="failed",
                message="远端元数据补全失败",
                error=str(exc),
            )

    def retag_single_image(
        self,
        point_id: str,
        *,
        backend: str | None = None,
    ) -> dict[str, Any]:
        """Synchronously re-tag a single indexed image and refresh its descriptor vector."""
        canon = normalize_tagger_backend(backend or self.config.tagger.backend)
        payloads = self.images.get_payloads_by_point_ids([point_id], include_tags=False)
        payload = payloads.get(point_id)
        if payload is None:
            raise KeyError(f"未找到图片：{point_id}")
        img_id = int(payload["row_id"])
        img_path = Path(str(payload["path"]))
        if not img_path.is_file():
            raise FileNotFoundError(f"图片文件不存在：{img_path}")

        self._free_vram_for_worker()
        tagger = build_tagger(self.config.tagger, self.config.embedding, backend=canon)
        if tagger is None:
            raise RuntimeError(f"无法初始化标签器 {canon}")
        try:
            pil = load_rgb(img_path)
            pred = tagger.predict([pil])[0]
            versions = dict(getattr(tagger, "model_versions", model_versions_for_backend(canon)))
        finally:
            tagger.close()
            release_accelerator()

        desc_text = self._build_descriptor_for_payload(
            pred, payload, drop_overlap=self.config.tagger.drop_overlap
        )
        self.images.update_tags_and_descriptor_batch(
            [(img_id, pred.tags, pred.scores, pred.categories, desc_text)]
        )
        self.task_state.mark_tag_models_done(
            [(point_id, img_id, str(payload.get("content_hash") or ""))],
            versions,
            descriptor_done=True,
        )
        if desc_text.strip():
            self._encode_and_flush_descriptors([(img_id, desc_text)])
        if callable(self._on_caches_invalidated):
            self._on_caches_invalidated()
        refreshed = self.images.get_payloads_by_point_ids([point_id]).get(point_id)
        return refreshed or payload
