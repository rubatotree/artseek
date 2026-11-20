"""Shared FastAPI dependency container holding v2 DB, vectors, and domain services."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from fastapi import Request

from ..config import AppConfig
from ..db.connection import Database
from ..db.repos.collections import CollectionRepo
from ..db.repos.images import ImageRepo
from ..db.repos.jobs import JobRepo
from ..db.repos.notes import NotesIndexRepo
from ..db.repos.provenance import ProvenanceRepo
from ..db.repos.query_cache import QueryPlanCacheRepo
from ..db.scope_resolver import ScopeResolver
from ..ml.llm import OllamaClient
from ..notes import NoteStore
from ..search_cache import SearchSnapshotStore
from ..services import (
    ArtistService,
    CollectionService,
    GalleryService,
    ModelLifecycle,
    NoteService,
    ProvenanceAppService,
    SearchService,
    StorageService,
)
from ..vectors.index import VectorIndex


@dataclass
class IndexJobState:
    running: bool = False
    path: str = ""
    done: int = 0
    total: int = 0
    message: str = ""
    error: str | None = None
    summary: dict[str, Any] | None = None


@dataclass(slots=True)
class ProvenanceJobState:
    running: bool = False
    paused: bool = False
    kind: str = ""
    done: int = 0
    total: int = 0
    message: str = ""
    error: str | None = None
    summary: dict[str, Any] | None = None
    stop_event: threading.Event = field(default_factory=threading.Event)
    pause_event: threading.Event = field(default_factory=threading.Event)


class AppContainer:
    """Lifecycle-managed bundle of v2 repositories, vector matrices, ML lifecycle, and services."""

    def __init__(
        self,
        config: AppConfig,
        *,
        embedder_factory: Callable[[], Any] | None = None,
        llm: OllamaClient | None = None,
    ) -> None:
        self.config = config
        config.ensure_dirs()

        self.db = Database(config.db_path)
        self.images = ImageRepo(self.db)
        self.collections_repo = CollectionRepo(self.db)
        self.prov_repo = ProvenanceRepo(self.db)
        self.notes_index = NotesIndexRepo(self.db)
        self.query_cache = QueryPlanCacheRepo(self.db)
        self.jobs_repo = JobRepo(self.db)
        self.resolver = ScopeResolver(self.db)

        self.vectors = VectorIndex(
            config.visual_vectors_path,
            config.descriptor_vectors_path,
            active_rows=self.images.active_row_ids(),
        )
        self.note_store = NoteStore(config.notes_dir)
        self.snapshots = SearchSnapshotStore(config.search_snapshot_dir)

        self.index_job = IndexJobState()
        self.index_lock = threading.Lock()
        self.provenance_job = ProvenanceJobState()
        self.provenance_lock = threading.RLock()
        self.provenance_refresh_pending = False

        self.lifecycle = ModelLifecycle(
            config,
            embedder_factory=embedder_factory,
            llm=llm,
            vectors=self.vectors,
            on_dimension_known=self._on_dimension_known,
        )

        self.gallery = GalleryService(self.images, self.prov_repo, self.resolver)
        self.artists = ArtistService(
            self.images,
            self.prov_repo,
            self.resolver,
            self.collections_repo,
            self.snapshots,
        )
        self.collections = CollectionService(
            self.collections_repo,
            self.images,
            self.prov_repo,
        )
        self.notes = NoteService(self.note_store, self.notes_index, self.images)
        self.provenance = ProvenanceAppService(
            config,
            self.images,
            self.prov_repo,
            self.vectors,
            on_author_change=self.artists.invalidate_cache,
        )
        self.search = SearchService(
            config,
            self.images,
            self.prov_repo,
            self.vectors,
            self.resolver,
            self.lifecycle,
            self.query_cache,
            self.snapshots,
        )
        self.storage = StorageService(
            config,
            self.db,
            self.images,
            self.vectors,
            self.notes,
            self.snapshots,
            is_busy=lambda: self.index_job.running or self.provenance_job.running,
            on_index_reopened=self.invalidate_caches,
        )

    def _on_dimension_known(self, dim: int) -> None:
        known = self.images.known_dimension()
        if known is None and dim > 0:
            self.images.meta_set("dimension", str(int(dim)))

    def invalidate_caches(self) -> None:
        self.gallery.invalidate_tag_cache()
        self.artists.invalidate_cache()

    def close(self) -> None:
        self.provenance_job.stop_event.set()
        self.provenance_job.pause_event.clear()
        self.lifecycle.close()
        self.vectors.release_cuda()
        self.db.close()


def get_container(request: Request) -> AppContainer:
    return request.app.state.container
