"""Storage path configuration and atomic directory migration for index, notes, and cache."""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..config import (
    AppConfig,
    migrate_index_sidecars,
    migrate_storage_tree,
    paths_equal,
    save_storage_paths,
)
from ..db.connection import Database
from ..db.repos.images import ImageRepo
from ..search_cache import SearchSnapshotStore
from ..vectors.index import VectorIndex
from .common import ServiceError
from .notes import NoteService


class StorageService:
    """Updates and migrates persistent ArtSeek storage locations safely."""

    def __init__(
        self,
        config: AppConfig,
        db: Database,
        images: ImageRepo,
        vectors: VectorIndex,
        notes: NoteService,
        snapshots: SearchSnapshotStore,
        *,
        is_busy: Callable[[], bool] | None = None,
        on_index_reopened: Callable[[], None] | None = None,
    ) -> None:
        self.config = config
        self.db = db
        self.images = images
        self.vectors = vectors
        self.notes = notes
        self.snapshots = snapshots
        self._is_busy = is_busy or (lambda: False)
        self._on_index_reopened = on_index_reopened
        self._storage_lock = threading.Lock()
        self.active_qdrant_path = Path(config.qdrant_path)

    def get_paths(self, *, qdrant_restart_required: bool | None = None) -> dict[str, Any]:
        restart = (
            qdrant_restart_required
            if qdrant_restart_required is not None
            else not paths_equal(self.config.qdrant_path, self.active_qdrant_path)
        )
        return {
            "qdrant_path": str(self.config.qdrant_path),
            "notes_dir": str(self.config.notes_dir),
            "cache_dir": str(self.config.cache_dir),
            "index_dir": str(self.config.index_dir),
            "active_qdrant_path": str(self.active_qdrant_path),
            "qdrant_restart_required": restart,
        }

    def update_paths(
        self,
        *,
        qdrant_path: str | Path | None = None,
        notes_dir: str | Path | None = None,
        cache_dir: str | Path | None = None,
        index_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        update_qdrant = qdrant_path is not None
        update_notes = notes_dir is not None
        update_cache = cache_dir is not None
        update_index = index_dir is not None
        if not update_qdrant and not update_notes and not update_cache and not update_index:
            raise ServiceError(400, "请提供要保存的存储路径")
        prev_index_dir = Path(self.config.index_dir)
        try:
            save_storage_paths(
                self.config,
                qdrant_path=qdrant_path,
                notes_dir=notes_dir,
                cache_dir=cache_dir,
                index_dir=index_dir,
                update_qdrant=update_qdrant,
                update_notes=update_notes,
                update_cache=update_cache,
                update_index=update_index,
            )
        except ValueError as exc:
            raise ServiceError(400, str(exc)) from exc

        if update_notes or update_cache or update_index:
            self.config.ensure_dirs()
        if update_notes:
            self.notes.set_root(self.config.notes_dir)
        if update_cache:
            self.snapshots.set_root(self.config.search_snapshot_dir)
        if update_index and not paths_equal(prev_index_dir, self.config.index_dir):
            self._reopen_index_at_config()
        return self.get_paths()

    def _reopen_index_at_config(self) -> None:
        self.db.reopen(self.config.db_path)
        self.vectors.reopen(
            self.config.visual_vectors_path,
            self.config.descriptor_vectors_path,
            active_rows=self.images.active_row_ids(),
        )
        if callable(self._on_index_reopened):
            self._on_index_reopened()

    def migrate_path(self, kind: str, destination: str) -> dict[str, Any]:
        if self._is_busy():
            raise ServiceError(409, "正在索引或执行任务，请稍后再迁移")
        if not self._storage_lock.acquire(blocking=False):
            raise ServiceError(409, "正在迁移存储目录")
        try:
            if kind == "notes":
                source = Path(self.notes.store.root)
                try:
                    dest, copied = migrate_storage_tree(source, destination)
                    save_storage_paths(self.config, notes_dir=dest, update_notes=True)
                except ValueError as exc:
                    raise ServiceError(400, str(exc)) from exc
                self.notes.set_root(self.config.notes_dir)
                return {
                    **self.get_paths(),
                    "kind": "notes",
                    "copied": copied,
                    "source": str(source),
                    "destination": str(self.config.notes_dir),
                }

            if kind == "cache":
                source = Path(self.config.cache_dir)
                try:
                    dest, copied = migrate_storage_tree(source, destination)
                    save_storage_paths(self.config, cache_dir=dest, update_cache=True)
                except ValueError as exc:
                    raise ServiceError(400, str(exc)) from exc
                self.config.ensure_dirs()
                self.snapshots.set_root(self.config.search_snapshot_dir)
                return {
                    **self.get_paths(),
                    "kind": "cache",
                    "copied": copied,
                    "source": str(source),
                    "destination": str(self.config.cache_dir),
                }

            if kind == "index":
                source = Path(self.config.index_dir)
                self.vectors.release_cuda()
                try:
                    self.db.checkpoint()
                except Exception:
                    pass
                self.db.close()
                switched = False
                try:
                    dest, copied = migrate_index_sidecars(source, destination)
                    save_storage_paths(self.config, index_dir=dest, update_index=True)
                    self.config.ensure_dirs()
                    self._reopen_index_at_config()
                    switched = True
                except ValueError as exc:
                    if not switched:
                        save_storage_paths(self.config, index_dir=source, update_index=True)
                        self._reopen_index_at_config()
                    raise ServiceError(400, str(exc)) from exc
                except Exception as exc:
                    if not switched:
                        save_storage_paths(self.config, index_dir=source, update_index=True)
                        self._reopen_index_at_config()
                    raise ServiceError(500, f"迁移失败：{exc}") from exc
                return {
                    **self.get_paths(),
                    "kind": "index",
                    "copied": copied,
                    "source": str(source),
                    "destination": str(self.config.index_dir),
                }

            if kind == "qdrant":
                source = Path(self.active_qdrant_path)
                try:
                    dest, copied = migrate_storage_tree(source, destination)
                    save_storage_paths(self.config, qdrant_path=dest, update_qdrant=True)
                except ValueError as exc:
                    raise ServiceError(400, str(exc)) from exc
                self.active_qdrant_path = Path(self.config.qdrant_path)
                return {
                    **self.get_paths(),
                    "kind": "qdrant",
                    "copied": copied,
                    "source": str(source),
                    "destination": str(self.config.qdrant_path),
                }

            raise ServiceError(400, f"未知存储类型：{kind}")
        finally:
            self._storage_lock.release()
