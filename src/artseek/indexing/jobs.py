"""Unified background JobManager for indexing, provenance, caption backfill, and compact."""

from __future__ import annotations

import heapq
import itertools
import threading
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..config import AppConfig
from ..db.connection import Database
from ..db.repos.images import ImageRepo
from ..db.repos.jobs import JobRepo
from ..db.repos.provenance import ProvenanceRepo
from ..db.repos.task_state import TaskStateRepo
from ..ml.captioner import Captioner
from ..ml.llm import OllamaUnavailable
from ..services.model_lifecycle import ModelLifecycle
from ..services.provenance import ProvenanceAppService
from ..vectors.index import VectorIndex
from .scan import IncrementalScanner

JOB_PRIORITIES: dict[str, int] = {
    "index": 10,
    "compact": 20,
    "provenance_scan": 30,
    "provenance_enrich": 40,
    "auto_tag": 50,
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

            heapq.heappush(
                self._queue,
                _QueuedTask(
                    JOB_PRIORITIES["index"],
                    next(self._seq),
                    job_id,
                    "index",
                    _run,
                ),
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

            heapq.heappush(
                self._queue,
                _QueuedTask(
                    JOB_PRIORITIES["caption"],
                    next(self._seq),
                    job_id,
                    "caption",
                    _run,
                ),
            )
            self._cond.notify_all()
            return job_id

    # ---------------------------------------------------------------- worker loop
    def _worker_loop(self) -> None:
        while True:
            with self._cond:
                while not self._queue and not self._shutdown:
                    self._cond.wait()
                if self._shutdown:
                    return
                task = heapq.heappop(self._queue)
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
