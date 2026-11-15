from __future__ import annotations

import dataclasses
import json
import mimetypes
import random
import tempfile
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..collections import (
    COLLECTION_SORT_KEYS,
    DEFAULT_COLLECTION_SORT,
    PAYLOAD_SORT_KEYS,
    SORT_ADDED_DESC,
    SORT_MANUAL,
    Collection,
    CollectionStore,
    HistoryAction,
)
from ..config import (
    DEFAULT_WEB_SEARCH_LIMIT,
    MAX_WEB_SEARCH_LIMIT,
    MIN_WEB_SEARCH_LIMIT,
    WEB_SEARCH_LIMIT_STEP,
    AppConfig,
    WebSearchSettings,
    load_web_search_settings,
    migrate_index_sidecars,
    migrate_storage_tree,
    paths_equal,
    save_storage_paths,
    save_web_search_settings,
)
from ..image_io import PHASH_LENGTH, canonical_path, make_thumbnail, perceptual_hash
from ..incremental_state import IncrementalIndexState
from ..indexer import ImageIndexer
from ..library import (
    DEFAULT_SORT,
    SORT_KEYS,
    apply_scope,
    day_end_ns,
    day_start_ns,
    library_payload,
    load_scope,
    rebuild_catalog,
    save_scope,
)
from ..notes import NoteError, NoteStore, resolve_content_hash
from ..provenance import (
    PIXIV_DOMAIN,
    TWITTER_DOMAIN,
    BudgetExhausted,
    FetchError,
    FxTwitterClient,
    PixivClient,
    ProvenanceService,
    ProvenanceStore,
    RateLimiter,
    manual_author_key,
    parse_payload_path,
    parse_source_text,
    scan_payloads,
)
from ..search import SearchEngine
from ..search_cache import SearchSnapshotStore, is_snapshot_id
from ..store import VectorStore


def _parse_id_list(raw: str) -> list[str]:
    if not raw.strip():
        return []
    try:
        loaded = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(400, "exclude_ids 不是合法 JSON") from exc
    if not isinstance(loaded, list) or not all(isinstance(item, str) for item in loaded):
        raise HTTPException(400, "exclude_ids 必须是字符串数组")
    return loaded


def _parse_weight_list(raw: str, expected: int) -> list[float]:
    if not raw.strip():
        return [1.0] * expected
    try:
        loaded = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(400, "file_weights 不是合法 JSON") from exc
    if not isinstance(loaded, list) or not all(isinstance(item, (int, float)) for item in loaded):
        raise HTTPException(400, "file_weights 必须是数字数组")
    weights = [max(0.0, min(float(item), 8.0)) for item in loaded]
    if len(weights) < expected:
        weights.extend([1.0] * (expected - len(weights)))
    return weights[:expected]


class ReferenceSpec(BaseModel):
    id: str
    weight: float = Field(default=1.0, ge=0, le=8)


class TextSearchRequest(BaseModel):
    query: str = ""
    text_weight: float = Field(default=1.0, ge=0, le=8)
    reference_id: str | None = None
    references: list[ReferenceSpec] = Field(default_factory=list)
    limit: int = Field(default=40, ge=1, le=200)
    offset: int = Field(default=0, ge=0, le=4000)
    expand: bool = True
    strict_tags: bool = False
    folder_keys: list[str] | None = None
    mtime_after: str | None = None
    mtime_before: str | None = None
    exclude_ids: list[str] = Field(default_factory=list)
    dedupe: bool = False
    snapshot_id: str | None = None
    author_key: str | None = None


class RandomBrowseRequest(BaseModel):
    """A stable random ordering for one paginated gallery session."""

    folder_keys: list[str] | None = None
    mtime_after: str | None = None
    mtime_before: str | None = None
    seed: int | None = None
    limit: int = Field(default=40, ge=1, le=200)
    offset: int = Field(default=0, ge=0, le=50_000)
    author_key: str | None = None


class FolderScopeRequest(BaseModel):
    sort: str = DEFAULT_SORT
    selected: list[str] = Field(default_factory=list)
    mtime_after: str | None = None
    mtime_before: str | None = None

    @field_validator("sort", mode="before")
    @classmethod
    def coerce_sort(cls, value: Any) -> str:
        text = str(value or "").strip()
        return text if text in SORT_KEYS else DEFAULT_SORT

    @field_validator("selected", mode="before")
    @classmethod
    def coerce_selected(cls, value: Any) -> list[str]:
        if not value:
            return []
        if isinstance(value, str):
            return [value] if value.strip() else []
        if not isinstance(value, list):
            return []
        keys: list[str] = []
        for item in value:
            if item is None:
                continue
            if isinstance(item, dict):
                item = item.get("key") or item.get("path") or ""
            text = str(item).strip()
            if text:
                keys.append(text)
        return keys

    @field_validator("mtime_after", "mtime_before", mode="before")
    @classmethod
    def coerce_day(cls, value: Any) -> str | None:
        if value is None or value == "":
            return None
        return str(value)


class IndexAddRequest(BaseModel):
    path: str = ""
    paths: list[str] = Field(default_factory=list)
    recursive: bool = True


class IndexRemoveRequest(BaseModel):
    folder_key: str


class ModelControlRequest(BaseModel):
    loaded: bool | None = None
    auto_unload_minutes: int | None = Field(default=None, ge=0, le=1440)


class WebSearchSettingsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    use_llm: bool = True
    strict_tags: bool = False
    dedupe: bool = True
    limit: int = Field(
        default=DEFAULT_WEB_SEARCH_LIMIT,
        ge=MIN_WEB_SEARCH_LIMIT,
        le=MAX_WEB_SEARCH_LIMIT,
    )

    @field_validator("limit")
    @classmethod
    def validate_limit_step(cls, value: int) -> int:
        if (value - MIN_WEB_SEARCH_LIMIT) % WEB_SEARCH_LIMIT_STEP:
            raise ValueError(f"每批数量必须以 {WEB_SEARCH_LIMIT_STEP} 为步长")
        return value


class CollectionNameRequest(BaseModel):
    name: str

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        name = value.strip()
        if len(name) > 80:
            raise ValueError("收藏夹名称不能超过 80 个字符")
        return name


class CollectionUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    sort: str | None = None
    seed_sort: str | None = None

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        name = value.strip()
        if len(name) > 80:
            raise ValueError("收藏夹名称不能超过 80 个字符")
        return name

    @field_validator("sort", "seed_sort")
    @classmethod
    def validate_sort(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if value not in COLLECTION_SORT_KEYS:
            raise ValueError("未知的收藏夹排序方式")
        return value

    @model_validator(mode="after")
    def validate_update(self) -> CollectionUpdateRequest:
        if self.name is None and self.sort is None:
            raise ValueError("请提供新的收藏夹名称或排序方式")
        return self


class CollectionItemOrderRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image_id: str
    before_id: str | None = None
    after_id: str | None = None

    @field_validator("image_id")
    @classmethod
    def validate_image_id(cls, value: str) -> str:
        image_id = value.strip()
        if not image_id:
            raise ValueError("图片 ID 不能为空")
        return image_id

    @model_validator(mode="after")
    def validate_anchor(self) -> CollectionItemOrderRequest:
        if self.before_id is not None and self.after_id is not None:
            raise ValueError("before_id 与 after_id 只能提供一个")
        return self


class CollectionOrderRequest(BaseModel):
    collection_ids: list[str]

    @field_validator("collection_ids")
    @classmethod
    def validate_collection_ids(cls, value: list[str]) -> list[str]:
        if any(not collection_id.strip() for collection_id in value):
            raise ValueError("收藏夹 ID 不能为空")
        if len(set(value)) != len(value):
            raise ValueError("收藏夹 ID 不能重复")
        return value


class StoragePathsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    qdrant_path: str | None = None
    notes_dir: str | None = None
    cache_dir: str | None = None
    index_dir: str | None = None


class StorageMigrateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["qdrant", "notes", "cache", "index"]
    destination: str


class NoteCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str | None = None
    content: str = ""


class NoteWriteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str | None = None
    content: str | None = None

    @model_validator(mode="after")
    def validate_write(self) -> NoteWriteRequest:
        if self.title is None and self.content is None:
            raise ValueError("请提供笔记标题或内容")
        return self


class NoteOrderRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    names: list[str]

    @field_validator("names")
    @classmethod
    def validate_names(cls, value: list[str]) -> list[str]:
        if any(not str(name).strip() for name in value):
            raise ValueError("笔记名称不能为空")
        return value


class ProvenanceScanRequest(BaseModel):
    deep: bool = False
    only_new: bool = False
    auto_enrich: bool | None = None


class ProvenanceEnrichRequest(BaseModel):
    platforms: list[Literal["pixiv", "twitter"]] = Field(
        default_factory=lambda: ["pixiv", "twitter"]
    )
    limit: int | None = Field(default=None, ge=1, le=100_000)
    point_ids: list[str] = Field(default_factory=list)


class ProvenanceUrlRequest(BaseModel):
    url: str = Field(default="", max_length=2000)


class ProvenanceManualRequest(BaseModel):
    url: str | None = Field(default=None, max_length=2000)
    title: str | None = Field(default=None, max_length=500)
    description: str | None = Field(default=None, max_length=20_000)
    author_name: str | None = Field(default=None, max_length=200)
    author_handle: str | None = Field(default=None, max_length=100)
    tags: list[str] | None = None
    clear_manual: bool = False


class CollectionItemRequest(BaseModel):
    image_id: str

    @field_validator("image_id")
    @classmethod
    def validate_image_id(cls, value: str) -> str:
        point_id = value.strip()
        if not point_id:
            raise ValueError("请选择要收藏的图片")
        return point_id


@dataclass
class IndexJob:
    running: bool = False
    path: str = ""
    done: int = 0
    total: int = 0
    message: str = ""
    error: str | None = None
    summary: dict[str, Any] | None = None


@dataclass(slots=True)
class ProvenanceJob:
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


COLLECTION_SCAN_BATCH_SIZE = 256
MAX_CACHED_COLLECTIONS = 4
# The public offset cap (10,000) plus the largest page (200) and one lookahead
# all fit here. Cache only IDs, never the comparatively large Qdrant payloads.
MAX_CACHED_COLLECTION_IDS = 10_240


@dataclass(slots=True)
class CollectionIdCursor:
    """Bounded, payload-free materialization state for sequential page reads."""

    membership_token: int
    index_generation: int
    sort_mode: str = DEFAULT_COLLECTION_SORT
    manual_token: int = 0
    live_item_ids: list[str] = field(default_factory=list)
    source_cursor: int = 0
    scan_complete: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


@dataclass(slots=True)
class CollectionSortCursor:
    """Bounded ID materialization for sorts which need indexed payload fields."""

    membership_token: int
    index_generation: int
    sort_mode: str
    ordered_ids: list[str] = field(default_factory=list)
    live_count: int = 0
    scan_complete: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


def _is_missing_qdrant_collection(error: ValueError) -> bool:
    message = str(error).casefold()
    return "collection" in message and ("not found" in message or "does not exist" in message)


def create_app(config: AppConfig, *, engine: SearchEngine | None = None) -> FastAPI:
    app = FastAPI(title="ArtSeek", docs_url="/api/docs", redoc_url=None)
    app.state.config = config
    app.state.engine = engine or SearchEngine(config)
    app.state.tagger = None
    app.state.index_job = IndexJob()
    app.state.index_lock = threading.Lock()
    app.state.collections = CollectionStore(config.collections_path)
    app.state.notes = NoteStore(config.notes_dir)
    app.state.provenance = ProvenanceStore(config.provenance_path)
    app.state.provenance_job = ProvenanceJob()
    app.state.provenance_lock = threading.RLock()
    app.state.search_snapshots = SearchSnapshotStore(config.search_snapshot_dir)
    app.state.active_qdrant_path = config.qdrant_path
    app.state.storage_lock = threading.Lock()
    app.state.collection_index_empty = False
    app.state.collection_index_generation = 0
    app.state.collection_id_cursors: OrderedDict[str, CollectionIdCursor] = OrderedDict()
    app.state.collection_sort_cursors: OrderedDict[str, CollectionSortCursor] = OrderedDict()
    app.state.collection_id_cursors_lock = threading.RLock()
    static_path = Path(__file__).with_name("static") / "index.html"

    @app.on_event("shutdown")
    def shutdown() -> None:
        app.state.provenance_job.stop_event.set()
        app.state.provenance_job.pause_event.clear()
        app.state.engine.close()

    def spa_page() -> HTMLResponse:
        return HTMLResponse(
            static_path.read_text(encoding="utf-8"),
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/", response_class=HTMLResponse)
    def home() -> HTMLResponse:
        return spa_page()

    @app.get("/search", response_class=HTMLResponse)
    @app.get("/folder", response_class=HTMLResponse)
    @app.get("/random", response_class=HTMLResponse)
    def spa_gallery() -> HTMLResponse:
        return spa_page()

    @app.get("/c/{collection_id}", response_class=HTMLResponse)
    def spa_collection(collection_id: str) -> HTMLResponse:
        del collection_id
        return spa_page()

    @app.get("/author/{author_key}", response_class=HTMLResponse)
    def spa_author(author_key: str) -> HTMLResponse:
        del author_key
        return spa_page()

    @app.get("/i/{point_id}", response_class=HTMLResponse)
    @app.get("/i/{point_id}/notes", response_class=HTMLResponse)
    def spa_image(point_id: str) -> HTMLResponse:
        del point_id
        return spa_page()

    @app.get("/i/{point_id}/notes/{name:path}", response_class=HTMLResponse)
    def spa_note(point_id: str, name: str) -> HTMLResponse:
        del point_id, name
        return spa_page()

    def resolve_folder_keys(explicit: list[str] | None) -> list[str]:
        if explicit is not None:
            return explicit
        return load_scope(config).selected

    def gallery_result(
        point_id: str,
        payload: dict[str, Any],
        fallback_name: str = "",
    ) -> dict[str, Any]:
        """Project an indexed payload into the one gallery-card response shape."""
        return {
            "id": point_id,
            "score": None,
            "path": str(payload.get("path") or ""),
            "name": str(payload.get("name") or fallback_name),
            "width": int(payload.get("width") or 0),
            "height": int(payload.get("height") or 0),
            "tags": list(payload.get("tags") or []),
            "tag_hits": [],
            "caption": payload.get("caption"),
        }

    def collection_summary(collection: Collection) -> dict[str, Any]:
        return collection.summary()

    def require_collection(collection_id: str) -> Collection:
        collection = app.state.collections.get(collection_id)
        if collection is None:
            raise HTTPException(404, "收藏夹不存在")
        return collection

    def reset_collection_index_state() -> None:
        """Allow collections to retry materialization after an index lifecycle change."""
        with app.state.collection_id_cursors_lock:
            app.state.collection_index_empty = False
            app.state.collection_index_generation += 1
            app.state.collection_id_cursors.clear()
            app.state.collection_sort_cursors.clear()

    def invalidate_collection_id_cursor(collection_id: str) -> None:
        with app.state.collection_id_cursors_lock:
            app.state.collection_id_cursors.pop(collection_id, None)
            app.state.collection_sort_cursors.pop(collection_id, None)

    def collection_id_cursor(collection: Collection, sort: str) -> CollectionIdCursor:
        """Return one LRU cursor whose data fits the public pagination window."""
        membership_token = hash(collection.item_ids)
        manual_token = hash(collection.manual_order)
        with app.state.collection_id_cursors_lock:
            cursor = app.state.collection_id_cursors.get(collection.id)
            generation = app.state.collection_index_generation
            if (
                cursor is None
                or cursor.membership_token != membership_token
                or cursor.manual_token != manual_token
                or cursor.sort_mode != sort
                or cursor.index_generation != generation
            ):
                cursor = CollectionIdCursor(
                    membership_token=membership_token,
                    index_generation=generation,
                    sort_mode=sort,
                    manual_token=manual_token,
                )
                app.state.collection_id_cursors[collection.id] = cursor
            app.state.collection_id_cursors.move_to_end(collection.id)
            while len(app.state.collection_id_cursors) > MAX_CACHED_COLLECTIONS:
                app.state.collection_id_cursors.popitem(last=False)
            return cursor

    def collection_source_ids(collection: Collection, sort: str) -> list[str]:
        """Cheap source ordering for sorts which never inspect payloads."""
        if sort == SORT_ADDED_DESC:
            return list(reversed(collection.item_ids))
        if sort == SORT_MANUAL:
            return list(collection.ordered_item_ids())
        return list(collection.item_ids)

    def collection_sort_key(payload: dict[str, Any], sort: str) -> int | str:
        if sort.startswith("mtime"):
            return int(payload.get("mtime_ns") or 0)
        return str(payload.get("name") or "").casefold()

    def collection_sort_cursor(collection: Collection, sort: str) -> CollectionSortCursor:
        """Return the LRU cursor holding a full payload-sorted ID ordering."""
        membership_token = hash(collection.item_ids)
        generation = app.state.collection_index_generation
        with app.state.collection_id_cursors_lock:
            cursor = app.state.collection_sort_cursors.get(collection.id)
            if (
                cursor is None
                or cursor.membership_token != membership_token
                or cursor.sort_mode != sort
                or cursor.index_generation != generation
            ):
                cursor = CollectionSortCursor(
                    membership_token=membership_token,
                    index_generation=generation,
                    sort_mode=sort,
                )
                app.state.collection_sort_cursors[collection.id] = cursor
            app.state.collection_sort_cursors.move_to_end(collection.id)
            while len(app.state.collection_sort_cursors) > MAX_CACHED_COLLECTIONS:
                app.state.collection_sort_cursors.popitem(last=False)
            return cursor

    def build_collection_sort_cursor(
        cursor: CollectionSortCursor,
        collection: Collection,
    ) -> bool:
        """Materialize the full sorted ID order once; return False if index is gone."""
        if cursor.scan_complete:
            return True
        entries: list[tuple[Any, str]] = []
        live_count = 0
        item_ids = collection.item_ids
        for batch_start in range(0, len(item_ids), COLLECTION_SCAN_BATCH_SIZE):
            batch = list(item_ids[batch_start : batch_start + COLLECTION_SCAN_BATCH_SIZE])
            payloads = retrieve_collection_payloads(batch)
            if app.state.collection_index_empty:
                return False
            for point_id in batch:
                payload = payloads.get(point_id)
                if not isinstance(payload, dict):
                    continue
                live_count += 1
                entries.append((collection_sort_key(payload, cursor.sort_mode), point_id))
        # Stable sort: equal keys retain insertion order for ascending modes;
        # reverse also reverses ties, matching "newest first" expectations.
        entries.sort(
            key=lambda item: item[0],
            reverse=cursor.sort_mode.endswith("_desc"),
        )
        cursor.ordered_ids = [point_id for _key, point_id in entries[:MAX_CACHED_COLLECTION_IDS]]
        cursor.live_count = live_count
        cursor.scan_complete = True
        return True

    def collection_ordered_live_ids(collection: Collection, sort: str) -> list[str]:
        """Return every live item ID in the order a view would display."""
        if sort not in PAYLOAD_SORT_KEYS:
            return collection_source_ids(collection, sort)
        cursor = collection_sort_cursor(collection, sort)
        with cursor.lock:
            if not build_collection_sort_cursor(cursor, collection):
                return []
            return list(cursor.ordered_ids)

    def materialize_sorted_collection_page(
        collection: Collection,
        sort: str,
        start: int,
        page_size: int,
    ) -> tuple[list[tuple[str, dict[str, Any]]], int | None, bool]:
        """Read one page from a payload-sorted cursor."""
        cursor = collection_sort_cursor(collection, sort)
        with cursor.lock:
            if not build_collection_sort_cursor(cursor, collection):
                return [], 0, False
            page_ids = cursor.ordered_ids[start : start + page_size]
            page_payloads = retrieve_collection_payloads(page_ids)
            if app.state.collection_index_empty:
                return [], 0, False
            page = [
                (point_id, page_payloads[point_id])
                for point_id in page_ids
                if isinstance(page_payloads.get(point_id), dict)
            ]
            truncated = cursor.live_count > len(cursor.ordered_ids)
            total = None if truncated else cursor.live_count
            has_more = start + len(page_ids) < len(cursor.ordered_ids)
            return page, total, has_more

    def retrieve_collection_payloads(point_ids: list[str]) -> dict[str, dict[str, Any]]:
        if app.state.collection_index_empty:
            return {}
        try:
            return app.state.engine.store.retrieve_payloads(point_ids)
        except ValueError as exc:
            if _is_missing_qdrant_collection(exc):
                # A removed/cleared Qdrant collection has no materialized images;
                # leave the user-owned ID list intact for a later re-index.
                with app.state.collection_id_cursors_lock:
                    app.state.collection_index_empty = True
                    app.state.collection_id_cursors.clear()
                    app.state.collection_sort_cursors.clear()
                return {}
            raise

    def materialize_unfiltered_collection_page(
        collection: Collection,
        sort: str,
        start: int,
        page_size: int,
    ) -> tuple[list[tuple[str, dict[str, Any]]], int | None, bool]:
        """Read an ordered page with a bounded ID cursor, never cached payloads."""
        cursor = collection_id_cursor(collection, sort)
        item_ids = collection_source_ids(collection, sort)
        target_count = start + page_size + 1
        page_payloads: dict[str, dict[str, Any]] = {}

        # This is per collection, so a deep scan cannot block other collection
        # requests or the global LRU lock. It also serializes cursor advancement.
        with cursor.lock:
            if app.state.collection_index_empty:
                return [], 0, False
            while len(cursor.live_item_ids) < target_count and not cursor.scan_complete:
                end = min(cursor.source_cursor + COLLECTION_SCAN_BATCH_SIZE, len(item_ids))
                batch = list(item_ids[cursor.source_cursor:end])
                payloads = retrieve_collection_payloads(batch)
                cursor.source_cursor = end
                if app.state.collection_index_empty:
                    return [], 0, False
                for point_id in batch:
                    payload = payloads.get(point_id)
                    if not isinstance(payload, dict):
                        continue
                    if len(cursor.live_item_ids) >= MAX_CACHED_COLLECTION_IDS:
                        break
                    live_index = len(cursor.live_item_ids)
                    cursor.live_item_ids.append(point_id)
                    if start <= live_index < start + page_size:
                        page_payloads[point_id] = payload
                cursor.scan_complete = cursor.source_cursor >= len(item_ids)

            page_ids = cursor.live_item_ids[start : start + page_size]
            missing_payload_ids = [
                point_id for point_id in page_ids if point_id not in page_payloads
            ]
            if missing_payload_ids:
                page_payloads.update(retrieve_collection_payloads(missing_payload_ids))
            if app.state.collection_index_empty:
                return [], 0, False

            page = [
                (point_id, page_payloads[point_id])
                for point_id in page_ids
                if isinstance(page_payloads.get(point_id), dict)
            ]
            has_more = len(cursor.live_item_ids) > start + len(page_ids)
            total = len(cursor.live_item_ids) if cursor.scan_complete else None
            return page, total, has_more

    def current_library() -> dict:
        catalog = app.state.engine.folder_catalog()
        previous = load_scope(config)
        merged, new_keys = apply_scope(catalog, previous)
        if merged != previous:
            save_scope(config, merged)
        return library_payload(catalog, merged, new_keys)

    @app.get("/api/library-tree")
    def library_tree() -> dict:
        try:
            return current_library()
        except Exception as exc:
            raise HTTPException(500, f"读取目录树失败：{exc}") from exc

    @app.get("/api/folder-images")
    def folder_images(
        folder_key: str,
        sort: str = DEFAULT_SORT,
        limit: int = 40,
        offset: int = 0,
        author_key: str | None = None,
    ) -> dict:
        key = canonical_path(folder_key)
        catalog = app.state.engine.folder_catalog()
        acc = catalog.folders.get(key)
        if acc is None:
            raise HTTPException(404, "该目录不在索引中")
        method = sort if sort in SORT_KEYS else DEFAULT_SORT
        page_size = max(1, min(limit, 200))
        start = max(0, offset)
        refs = catalog.list_folder_images(key, method)
        if author_key:
            allowed = set(app.state.provenance.point_ids_for_author(author_key))
            refs = [item for item in refs if item[0] in allowed]
        total = len(refs)
        page = refs[start : start + page_size]
        payloads = app.state.engine.store.retrieve_payloads([point_id for point_id, _, _ in page])
        results: list[dict[str, Any]] = []
        for point_id, name, _mtime in page:
            payload = payloads.get(point_id) or {}
            results.append(gallery_result(point_id, payload, name))
        return {
            "folder_key": key,
            "folder_name": acc.name,
            "folder_path": acc.path,
            "sort": method,
            "offset": start,
            "limit": page_size,
            "total": total,
            "has_more": start + len(results) < total,
            "results": results,
        }

    def collections_payload(
        collections: list[Collection],
        history: dict[str, Any],
        image_id: str | None = None,
    ) -> dict[str, Any]:
        summaries: list[dict[str, Any]] = []
        for collection in collections:
            summary = collection_summary(collection)
            if image_id:
                summary["contains"] = image_id in collection.item_ids
            summaries.append(summary)
        return {"collections": summaries, "history": history}

    def history_action_payload(result: HistoryAction) -> dict[str, Any]:
        payload = result.payload()
        invalidate_collection_id_cursor(result.collection_id)
        return payload

    def current_history() -> dict[str, Any]:
        _, history = app.state.collections.snapshot()
        return history

    def image_display_name(point_id: str) -> str:
        payload = app.state.engine.store.get(point_id)
        if not isinstance(payload, dict):
            return ""
        return str(payload.get("name") or "")

    @app.get("/api/collections")
    def list_collections(image_id: str | None = None) -> dict[str, Any]:
        collections, history = app.state.collections.snapshot()
        return collections_payload(collections, history, image_id)

    @app.post("/api/collections", status_code=201)
    def create_collection(request: CollectionNameRequest) -> dict[str, Any]:
        collection = app.state.collections.create(request.name)
        return {"collection": collection_summary(collection)}

    @app.put("/api/collections/order")
    def reorder_collections(request: CollectionOrderRequest) -> dict[str, Any]:
        collections = app.state.collections.reorder(request.collection_ids)
        if collections is None:
            raise HTTPException(409, "收藏夹已更新，请刷新后重试")
        _, history = app.state.collections.snapshot()
        return collections_payload(collections, history)

    @app.post("/api/collections/undo")
    def undo_collection_history() -> dict[str, Any]:
        result = app.state.collections.undo()
        if result is None:
            raise HTTPException(409, "没有可撤销的操作")
        return history_action_payload(result)

    @app.post("/api/collections/redo")
    def redo_collection_history() -> dict[str, Any]:
        result = app.state.collections.redo()
        if result is None:
            raise HTTPException(409, "没有可重做的操作")
        return history_action_payload(result)

    @app.patch("/api/collections/{collection_id}")
    def update_collection(collection_id: str, request: CollectionUpdateRequest) -> dict[str, Any]:
        current = require_collection(collection_id)
        manual_order: tuple[str, ...] | None = None
        if (
            request.sort == SORT_MANUAL
            and current.sort != SORT_MANUAL
            and not current.manual_order
            and current.item_ids
        ):
            # Entering manual mode keeps the order the user was looking at
            # instead of jumping back to insertion order.
            seed_sort = request.seed_sort or current.sort
            if seed_sort not in COLLECTION_SORT_KEYS:
                seed_sort = DEFAULT_COLLECTION_SORT
            live_ids = collection_ordered_live_ids(current, seed_sort)
            seen = set(live_ids)
            manual_order = tuple(live_ids) + tuple(
                point_id for point_id in current.item_ids if point_id not in seen
            )
        collection = app.state.collections.update(
            collection_id,
            name=request.name,
            sort=request.sort,
            manual_order=manual_order,
        )
        if collection is None:
            raise HTTPException(404, "收藏夹不存在")
        return {
            "collection": collection_summary(collection),
            "history": current_history(),
        }

    @app.delete("/api/collections/{collection_id}")
    def delete_collection(collection_id: str, discard: bool = False) -> dict[str, Any]:
        collection = app.state.collections.delete(collection_id, record_history=not discard)
        if collection is None:
            raise HTTPException(404, "收藏夹不存在")
        invalidate_collection_id_cursor(collection_id)
        return {
            "deleted": True,
            "collection": collection_summary(collection),
            "history": current_history(),
        }

    @app.post("/api/collections/{collection_id}/items")
    def add_collection_item(collection_id: str, request: CollectionItemRequest) -> dict[str, Any]:
        require_collection(collection_id)
        if not app.state.engine.store.get(request.image_id):
            raise HTTPException(404, "图片不在索引中，无法收藏")
        collection, added = app.state.collections.add_item(
            collection_id,
            request.image_id,
            item_name=image_display_name(request.image_id),
        )
        if collection is None:
            raise HTTPException(404, "收藏夹不存在")
        if added:
            invalidate_collection_id_cursor(collection_id)
        return {
            "collection": collection_summary(collection),
            "added": added,
            "history": current_history(),
        }

    @app.delete("/api/collections/{collection_id}/items/{point_id}")
    def remove_collection_item(collection_id: str, point_id: str) -> dict[str, Any]:
        collection, removed = app.state.collections.remove_item(
            collection_id,
            point_id,
            item_name=image_display_name(point_id),
        )
        if collection is None:
            raise HTTPException(404, "收藏夹不存在")
        if removed:
            invalidate_collection_id_cursor(collection_id)
        return {
            "collection": collection_summary(collection),
            "removed": removed,
            "history": current_history(),
        }

    @app.put("/api/collections/{collection_id}/items/order")
    def reorder_collection_item(
        collection_id: str,
        request: CollectionItemOrderRequest,
    ) -> dict[str, Any]:
        require_collection(collection_id)
        collection, moved = app.state.collections.reorder_item(
            collection_id,
            request.image_id,
            before_id=request.before_id,
            after_id=request.after_id,
        )
        if collection is None:
            raise HTTPException(409, "收藏夹内容已更新，请刷新后重试")
        if moved:
            invalidate_collection_id_cursor(collection_id)
        return {
            "collection": collection_summary(collection),
            "moved": moved,
            "history": current_history(),
        }

    @app.get("/api/collections/{collection_id}/images")
    def collection_images(
        collection_id: str,
        query: str = "",
        sort: str = "",
        limit: int = 40,
        offset: int = Query(default=0, ge=0, le=10_000),
        author_key: str | None = None,
    ) -> dict[str, Any]:
        collection = require_collection(collection_id)
        # A query-level sort is a temporary view override. It never persists.
        method = sort if sort in COLLECTION_SORT_KEYS else collection.sort
        if method not in COLLECTION_SORT_KEYS:
            method = DEFAULT_COLLECTION_SORT
        needle = query.strip().casefold()
        page_size = max(1, min(limit, 200))
        start = offset
        allowed = (
            set(app.state.provenance.point_ids_for_author(author_key))
            if author_key
            else None
        )

        def matches(point_id: str, payload: dict[str, Any]) -> bool:
            if allowed is not None and point_id not in allowed:
                return False
            if not needle:
                return True
            tags = payload.get("tags") or []
            searchable = [
                str(payload.get("name") or ""),
                str(payload.get("path") or ""),
                str(payload.get("caption") or ""),
                *(str(tag) for tag in tags),
            ]
            return needle in "\n".join(searchable).casefold()

        if needle or allowed is not None:
            filtered: list[tuple[str, dict[str, Any]]] = []
            for batch_start in range(0, len(collection.item_ids), 256):
                batch = list(collection.item_ids[batch_start : batch_start + 256])
                payloads = retrieve_collection_payloads(batch)
                filtered.extend(
                    (point_id, payloads[point_id])
                    for point_id in batch
                    if point_id in payloads
                    and isinstance(payloads[point_id], dict)
                    and matches(point_id, payloads[point_id])
                )
            if method in PAYLOAD_SORT_KEYS:
                filtered.sort(
                    key=lambda item: collection_sort_key(item[1], method),
                    reverse=method.endswith("_desc"),
                )
            elif method == SORT_ADDED_DESC:
                filtered.reverse()
            elif method == SORT_MANUAL:
                rank = {
                    point_id: index
                    for index, point_id in enumerate(collection.ordered_item_ids())
                }
                filtered.sort(key=lambda item: rank.get(item[0], len(rank)))
            page = filtered[start : start + page_size]
            total = len(filtered)
            has_more = start + len(page) < total
            total_exact = True
        elif method in PAYLOAD_SORT_KEYS:
            page, total, has_more = materialize_sorted_collection_page(
                collection,
                method,
                start,
                page_size,
            )
            total_exact = total is not None
        else:
            page, total, has_more = materialize_unfiltered_collection_page(
                collection,
                method,
                start,
                page_size,
            )
            total_exact = total is not None
        return {
            "collection": collection_summary(collection),
            "query": query.strip(),
            "sort": method,
            "saved_sort": collection.sort,
            "offset": start,
            "limit": page_size,
            "total": total,
            "total_exact": total_exact,
            "has_more": has_more,
            "results": [gallery_result(point_id, payload) for point_id, payload in page],
        }

    @app.post("/api/random-browse")
    def random_browse(request: RandomBrowseRequest) -> dict:
        """Return one deterministic page from a random ordering in the selected scope."""
        if app.state.index_job.running:
            raise HTTPException(409, "正在索引图片，请稍后再浏览")
        folder_keys = resolve_folder_keys(request.folder_keys)
        if not folder_keys:
            raise HTTPException(400, "请先勾选要随机浏览的文件夹")

        catalog = app.state.engine.folder_catalog()
        point_ids = catalog.point_ids(
            folder_keys,
            after_ns=day_start_ns(request.mtime_after),
            before_ns=day_end_ns(request.mtime_before),
        )
        if point_ids is None:
            # `None` is an optimization for vector search meaning every point.
            # Random browse needs concrete IDs but only retains the requested page.
            point_ids = [point_id for acc in catalog.folders.values() for point_id in acc.ids]
        if request.author_key:
            allowed = set(app.state.provenance.point_ids_for_author(request.author_key))
            point_ids = [point_id for point_id in point_ids if point_id in allowed]

        total = len(point_ids)
        start = min(request.offset, total)
        page_size = min(request.limit, max(0, total - start))
        seed = (
            request.seed
            if request.seed is not None
            else random.SystemRandom().randrange(1 << 53)
        )
        sampled = random.Random(seed).sample(point_ids, start + page_size)
        page_ids = sampled[start:]
        payloads = app.state.engine.store.retrieve_payloads(page_ids)
        results: list[dict[str, Any]] = []
        for point_id in page_ids:
            payload = payloads.get(point_id) or {}
            results.append(gallery_result(point_id, payload))
        return {
            "seed": seed,
            "offset": start,
            "limit": request.limit,
            "total": total,
            "has_more": start + len(results) < total,
            "results": results,
        }

    @app.put("/api/library-tree")
    def update_library_tree(request: FolderScopeRequest) -> dict:
        catalog = app.state.engine.folder_catalog()
        current, _ = apply_scope(catalog, load_scope(config))
        current.sort = request.sort
        current.selected = request.selected
        current.mtime_after = request.mtime_after
        current.mtime_before = request.mtime_before
        merged, new_keys = apply_scope(catalog, current)
        save_scope(config, merged)
        return library_payload(catalog, merged, new_keys)

    def resolve_library_references(specs: list[ReferenceSpec]) -> tuple[list[Path], list[float]]:
        paths: list[Path] = []
        weights: list[float] = []
        seen: set[str] = set()
        for spec in specs:
            if spec.id in seen:
                continue
            seen.add(spec.id)
            payload = app.state.engine.store.get(spec.id)
            if not payload:
                raise HTTPException(404, "参考图片不在索引中")
            paths.append(Path(payload["path"]))
            weights.append(spec.weight)
        return paths, weights

    def parse_folder_keys(raw: str) -> list[str] | None:
        if not raw.strip():
            return None
        try:
            loaded = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise HTTPException(400, "folder_keys 不是合法 JSON") from exc
        if not isinstance(loaded, list) or not all(isinstance(item, str) for item in loaded):
            raise HTTPException(400, "folder_keys 必须是字符串数组")
        return loaded

    def parse_reference_specs(raw: str) -> list[ReferenceSpec]:
        if not raw.strip():
            return []
        try:
            loaded = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise HTTPException(400, "references 不是合法 JSON") from exc
        if not isinstance(loaded, list):
            raise HTTPException(400, "references 必须是数组")
        try:
            return [ReferenceSpec.model_validate(item) for item in loaded]
        except Exception as exc:
            raise HTTPException(400, "references 格式无效") from exc

    def persist_search_snapshot(
        *,
        snapshot_id: str | None,
        query: str,
        text_weight: float,
        references: list[ReferenceSpec],
        expand: bool,
        strict_tags: bool,
        dedupe: bool,
        limit: int,
        folder_keys: list[str] | None,
        mtime_after: str | None,
        mtime_before: str | None,
        offset: int,
        payload: dict[str, Any],
        had_file_refs: bool = False,
    ) -> dict[str, Any]:
        store: SearchSnapshotStore = app.state.search_snapshots
        refs = [{"id": spec.id, "weight": spec.weight} for spec in references]
        results = list(payload.get("results") or [])
        if offset <= 0:
            payload["snapshot_id"] = store.create(
                query=query,
                text_weight=text_weight,
                references=refs,
                expand=expand,
                strict_tags=strict_tags,
                dedupe=dedupe,
                limit=limit,
                folder_keys=folder_keys,
                mtime_after=mtime_after,
                mtime_before=mtime_before,
                query_plan=payload.get("query_plan")
                if isinstance(payload.get("query_plan"), dict)
                else {},
                llm_warning=payload.get("llm_warning"),
                results=results,
                has_more=bool(payload.get("has_more")),
                had_file_refs=had_file_refs,
            )
            return payload
        if is_snapshot_id(snapshot_id) and store.append(
            snapshot_id or "",
            results,
            has_more=bool(payload.get("has_more")),
        ):
            payload["snapshot_id"] = snapshot_id
        elif snapshot_id:
            payload["snapshot_id"] = snapshot_id
        return payload

    def materialize_snapshot_results(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        getter = getattr(app.state.engine.store, "get", None)
        live: list[dict[str, Any]] = []
        for row in rows:
            point_id = str(row.get("id") or "")
            if not point_id:
                continue
            payload = getter(point_id) if callable(getter) else None
            if payload:
                item = gallery_result(point_id, payload, fallback_name=str(row.get("name") or ""))
                item["score"] = row.get("score")
                item["tag_hits"] = list(row.get("tag_hits") or [])
                item["visual_score"] = row.get("visual_score")
                item["descriptor_score"] = row.get("descriptor_score")
                live.append(item)
            else:
                live.append(row)
        return live

    @app.get("/api/search-snapshots/{snapshot_id}")
    def read_search_snapshot(snapshot_id: str) -> dict:
        if not is_snapshot_id(snapshot_id):
            raise HTTPException(400, "检索快照编号不合法")
        data = app.state.search_snapshots.get(snapshot_id)
        if data is None:
            raise HTTPException(404, "检索快照不存在或已过期")
        data["results"] = materialize_snapshot_results(list(data.get("results") or []))
        return data

    @app.post("/api/search")
    def search(request: TextSearchRequest) -> dict:
        if app.state.index_job.running:
            raise HTTPException(409, "正在索引图片，请稍后再检索")
        specs = list(request.references)
        if request.reference_id:
            specs.append(ReferenceSpec(id=request.reference_id, weight=1.0))
        paths, weights = resolve_library_references(specs)
        restrict_ids = (
            app.state.provenance.point_ids_for_author(request.author_key)
            if request.author_key
            else None
        )
        try:
            response = app.state.engine.search(
                request.query,
                images=paths,
                text_weight=request.text_weight,
                image_weights=weights,
                limit=request.limit,
                expand=request.expand,
                strict_tags=request.strict_tags,
                folder_keys=resolve_folder_keys(request.folder_keys),
                mtime_after=request.mtime_after,
                mtime_before=request.mtime_before,
                offset=request.offset,
                exclude_ids=request.exclude_ids,
                dedupe=request.dedupe,
                restrict_ids=restrict_ids,
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return persist_search_snapshot(
            snapshot_id=request.snapshot_id,
            query=request.query,
            text_weight=request.text_weight,
            references=specs,
            expand=request.expand,
            strict_tags=request.strict_tags,
            dedupe=request.dedupe,
            limit=request.limit,
            folder_keys=resolve_folder_keys(request.folder_keys),
            mtime_after=request.mtime_after,
            mtime_before=request.mtime_before,
            offset=request.offset,
            payload=dataclasses.asdict(response),
        )

    @app.post("/api/search-upload")
    async def search_upload(
        files: list[UploadFile] = File(default=[]),
        file_weights: str = Form(default="[]"),
        references: str = Form(default="[]"),
        query: str = Form(default=""),
        text_weight: float = Form(default=1.0),
        limit: int = Form(default=40),
        expand: bool = Form(default=True),
        strict_tags: bool = Form(default=False),
        folder_keys: str = Form(default=""),
        mtime_after: str = Form(default=""),
        mtime_before: str = Form(default=""),
        offset: int = Form(default=0),
        exclude_ids: str = Form(default=""),
        dedupe: bool = Form(default=False),
        snapshot_id: str = Form(default=""),
        author_key: str = Form(default=""),
    ) -> dict:
        if app.state.index_job.running:
            raise HTTPException(409, "正在索引图片，请稍后再检索")
        specs = parse_reference_specs(references)
        library_paths, library_weights = resolve_library_references(specs)
        upload_weights = _parse_weight_list(file_weights, len(files))
        temps: list[Path] = []
        try:
            for file in files:
                suffix = Path(file.filename or "query.png").suffix or ".png"
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
                    temps.append(Path(handle.name))
                    handle.write(await file.read())
            folder_keys_list = resolve_folder_keys(parse_folder_keys(folder_keys))
            response = app.state.engine.search(
                query,
                images=library_paths + temps,
                text_weight=max(0.0, min(text_weight, 8.0)),
                image_weights=library_weights + upload_weights,
                limit=max(1, min(limit, 200)),
                expand=expand,
                strict_tags=strict_tags,
                folder_keys=folder_keys_list,
                mtime_after=mtime_after or None,
                mtime_before=mtime_before or None,
                offset=max(0, min(offset, 4000)),
                exclude_ids=_parse_id_list(exclude_ids),
                dedupe=dedupe,
                restrict_ids=(
                    app.state.provenance.point_ids_for_author(author_key)
                    if author_key
                    else None
                ),
            )
            return persist_search_snapshot(
                snapshot_id=snapshot_id or None,
                query=query,
                text_weight=max(0.0, min(text_weight, 8.0)),
                references=specs,
                expand=expand,
                strict_tags=strict_tags,
                dedupe=dedupe,
                limit=max(1, min(limit, 200)),
                folder_keys=folder_keys_list,
                mtime_after=mtime_after or None,
                mtime_before=mtime_before or None,
                offset=max(0, min(offset, 4000)),
                payload=dataclasses.asdict(response),
                had_file_refs=bool(files),
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        finally:
            for path in temps:
                path.unlink(missing_ok=True)

    def resolve_item(point_id: str) -> tuple[dict, Path]:
        payload = app.state.engine.store.get(point_id)
        if not payload:
            raise HTTPException(404, "图片不在索引中")
        path = Path(payload["path"])
        if not path.exists():
            raise HTTPException(404, "原图片已移动或删除，请重新索引")
        return payload, path

    @app.get("/thumb/{point_id}")
    def thumbnail(point_id: str) -> FileResponse:
        _, source = resolve_item(point_id)
        destination = config.thumbnail_dir / f"{point_id}.webp"
        if not destination.exists() or destination.stat().st_mtime_ns < source.stat().st_mtime_ns:
            make_thumbnail(source, destination)
        return FileResponse(destination, media_type="image/webp")

    @app.get("/image/{point_id}")
    def full_image(point_id: str) -> FileResponse:
        _, path = resolve_item(point_id)
        return FileResponse(path)

    @app.get("/api/item/{point_id}")
    def item(point_id: str) -> dict:
        payload, _ = resolve_item(point_id)
        return payload

    # --------------------------------------------------------------- provenance

    def provenance_budgets() -> list[dict[str, Any]]:
        store: ProvenanceStore = app.state.provenance
        cfg = config.provenance
        return [
            RateLimiter(
                store,
                PIXIV_DOMAIN,
                min_interval=max(0.0, cfg.pixiv_min_interval),
                daily_limit=max(1, cfg.pixiv_daily_limit),
            ).snapshot().as_dict(),
            RateLimiter(
                store,
                TWITTER_DOMAIN,
                min_interval=max(0.0, cfg.twitter_min_interval),
                daily_limit=max(1, cfg.twitter_daily_limit),
            ).snapshot().as_dict(),
        ]

    def provenance_job_payload() -> dict[str, Any]:
        job: ProvenanceJob = app.state.provenance_job
        return {
            "running": job.running,
            "paused": job.paused,
            "kind": job.kind,
            "done": job.done,
            "total": job.total,
            "message": job.message,
            "error": job.error,
            "summary": job.summary,
            "budgets": provenance_budgets(),
        }

    def ensure_provenance_record(point_id: str, payload: dict[str, Any], path: Path):
        store: ProvenanceStore = app.state.provenance
        record = store.get(point_id)
        if record is not None:
            return record
        parsed = parse_payload_path(path, deep=True)
        if parsed is not None:
            store.upsert_scan(
                point_id,
                payload,
                parsed,
                path_key=str(payload.get("path_key") or ""),
                origin=parsed.primary.origin,
            )
        else:
            store.upsert_unresolved(
                point_id, payload, path_key=str(payload.get("path_key") or "")
            )
        return store.get(point_id)

    def provenance_payload(
        point_id: str,
        payload: dict[str, Any],
        path: Path,
    ) -> dict[str, Any]:
        store: ProvenanceStore = app.state.provenance
        record = ensure_provenance_record(point_id, payload, path)
        if record is None:
            return {"point_id": point_id, "record": None, "author": None}
        author = None
        if record.get("author_key"):
            author = store.author_summary(str(record["author_key"]))
        page = record.get("page")
        count = record.get("page_count")
        if record.get("platform") == "pixiv" and page is not None:
            page_label = f"p{page}" + (f" / {count}" if count else "")
        elif record.get("platform") == "twitter" and page is not None:
            page_label = f"{page} / {count}" if count else str(page)
        else:
            page_label = ""
        clean = dict(record)
        clean.pop("raw_json", None)
        return {
            "point_id": point_id,
            "record": clean,
            "author": author,
            "page_label": page_label,
            "path": str(path),
            "reverse_search": {
                "saucenao": "https://saucenao.com/",
                "ascii2d": "https://ascii2d.net/",
            },
        }

    def make_service() -> ProvenanceService:
        store: ProvenanceStore = app.state.provenance
        cfg = config.provenance
        return ProvenanceService(
            store,
            pixiv=PixivClient(),
            twitter=FxTwitterClient(),
            pixiv_limiter=RateLimiter(
                store,
                PIXIV_DOMAIN,
                min_interval=max(0.0, cfg.pixiv_min_interval),
                daily_limit=max(1, cfg.pixiv_daily_limit),
            ),
            twitter_limiter=RateLimiter(
                store,
                TWITTER_DOMAIN,
                min_interval=max(0.0, cfg.twitter_min_interval),
                daily_limit=max(1, cfg.twitter_daily_limit),
            ),
        )

    def start_enrich_job(
        *,
        platforms: list[str],
        limit: int | None = None,
        point_ids: list[str] | None = None,
        reset_stop: bool = True,
    ) -> bool:
        store: ProvenanceStore = app.state.provenance
        job: ProvenanceJob = app.state.provenance_job
        clean_platforms = [item for item in platforms if item in {"pixiv", "twitter"}]
        if not clean_platforms:
            return False
        if point_ids:
            records = [
                record
                for point_id in dict.fromkeys(point_ids)
                if (record := store.get(point_id)) is not None
                and record.get("platform") in clean_platforms
            ]
        else:
            records = store.pending_records(
                platforms=tuple(clean_platforms), limit=limit
            )
        if limit is not None:
            records = records[: max(0, int(limit))]
        with app.state.provenance_lock:
            if job.running:
                return False
            if not reset_stop and job.stop_event.is_set():
                return False
            job.running = True
            job.paused = False
            job.kind = "enrich"
            job.done = 0
            job.total = len(records)
            job.message = (
                f"待补全 {len(records)} 条来源"
                if records
                else "没有待补全的来源"
            )
            job.error = None
            job.summary = None
            if reset_stop:
                job.stop_event.clear()
            job.pause_event.clear()
        if not records:
            job.running = False
            return True

        def progress(done: int, total: int, label: str) -> None:
            job.done = done
            job.total = total
            job.message = f"正在补全来源 {done}/{total}" + (f"：{label}" if label else "")

        def run() -> None:
            service = make_service()
            try:
                summary = service.enrich_records(records, progress=progress)
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
                service.close()
                job.running = False
                job.paused = False

        threading.Thread(target=run, daemon=True, name="artseek-provenance").start()
        return True

    def start_scan_job(
        *,
        deep: bool = False,
        only_new: bool = False,
        auto_enrich: bool | None = None,
    ) -> bool:
        store: ProvenanceStore = app.state.provenance
        job: ProvenanceJob = app.state.provenance_job
        with app.state.provenance_lock:
            if job.running:
                return False
            job.running = True
            job.paused = False
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

        should_enrich = config.provenance.auto_enrich if auto_enrich is None else auto_enrich

        def run() -> None:
            summary: dict[str, Any] | None = None
            try:
                payloads = app.state.engine.store.iter_payloads()
                total = app.state.engine.store.count()
                summary = scan_payloads(
                    store,
                    payloads,
                    total=total,
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
                job.running = False
                job.paused = False
                if should_enrich and not job.stop_event.is_set() and job.error is None:
                    start_enrich_job(
                        platforms=["pixiv", "twitter"], reset_stop=False
                    )

        threading.Thread(target=run, daemon=True, name="artseek-provenance-scan").start()
        return True

    @app.get("/api/provenance/stats")
    def provenance_stats() -> dict[str, Any]:
        stats = app.state.provenance.stats()
        stats["job"] = provenance_job_payload()
        return stats

    @app.get("/api/provenance/job")
    def provenance_job() -> dict[str, Any]:
        return provenance_job_payload()

    @app.post("/api/provenance/scan")
    def provenance_scan(request: ProvenanceScanRequest) -> dict[str, Any]:
        if app.state.index_job.running:
            raise HTTPException(409, "正在索引图片，请稍后再扫描来源")
        started = start_scan_job(
            deep=request.deep,
            only_new=request.only_new,
            auto_enrich=request.auto_enrich,
        )
        if not started:
            raise HTTPException(409, "已有来源任务在运行")
        return provenance_job_payload()

    @app.post("/api/provenance/enrich")
    def provenance_enrich(request: ProvenanceEnrichRequest) -> dict[str, Any]:
        if app.state.index_job.running:
            raise HTTPException(409, "正在索引图片，请稍后再补全来源")
        platforms = list(request.platforms)
        if not request.point_ids and app.state.provenance.stats()["total"] == 0:
            # First run: discover filename/embedded references, then
            # automatically continue into the budgeted network queue.
            started = start_scan_job(deep=False, only_new=True, auto_enrich=True)
            if not started:
                raise HTTPException(409, "已有来源任务在运行")
            return provenance_job_payload()
        started = start_enrich_job(
            platforms=platforms,
            limit=request.limit,
            point_ids=list(request.point_ids),
        )
        if not started:
            raise HTTPException(409, "已有来源任务在运行")
        return provenance_job_payload()

    @app.post("/api/provenance/job/pause")
    def provenance_pause() -> dict[str, Any]:
        job: ProvenanceJob = app.state.provenance_job
        if job.running:
            job.pause_event.set()
            job.paused = True
            job.message = "来源任务已暂停"
        return provenance_job_payload()

    @app.post("/api/provenance/job/resume")
    def provenance_resume() -> dict[str, Any]:
        job: ProvenanceJob = app.state.provenance_job
        if job.running:
            job.pause_event.clear()
            job.paused = False
            job.message = "来源任务已继续"
        return provenance_job_payload()

    @app.post("/api/provenance/job/stop")
    def provenance_stop() -> dict[str, Any]:
        job: ProvenanceJob = app.state.provenance_job
        if job.running:
            job.stop_event.set()
            job.pause_event.clear()
            job.paused = False
            job.message = "正在停止来源任务…"
        return provenance_job_payload()

    @app.get("/api/images/{point_id}/provenance")
    def image_provenance(point_id: str) -> dict[str, Any]:
        payload, path = resolve_item(point_id)
        return provenance_payload(point_id, payload, path)

    @app.put("/api/images/{point_id}/provenance")
    def update_image_provenance(point_id: str, request: ProvenanceManualRequest) -> dict[str, Any]:
        payload, path = resolve_item(point_id)
        store: ProvenanceStore = app.state.provenance
        record = ensure_provenance_record(point_id, payload, path)
        if record is None:
            raise HTTPException(404, "图片不在索引中")
        if request.clear_manual:
            store.clear_manual(point_id)
            parsed = parse_payload_path(path, deep=True)
            if parsed is not None:
                store.upsert_scan(
                    point_id,
                    payload,
                    parsed,
                    path_key=str(payload.get("path_key") or ""),
                    origin=parsed.primary.origin,
                )
            record = store.get(point_id)
            return provenance_payload(point_id, payload, path)
        fields: dict[str, Any] = {}
        ref = None
        if request.url:
            ref = parse_source_text(request.url)
            if ref is None:
                raise HTTPException(400, "请输入 Pixiv/X 作品 URL 或任意 http(s) 来源 URL")
        if request.title is not None:
            fields["title"] = request.title.strip() or None
        if request.description is not None:
            fields["description"] = request.description
        if request.tags is not None:
            fields["tags"] = [tag.strip() for tag in request.tags if tag.strip()]
        if request.author_name is not None or request.author_handle is not None:
            name = (request.author_name or "").strip()
            handle = (request.author_handle or "").strip().lstrip("@")
            if handle:
                key = store.ensure_author(
                    "twitter", handle, name=name or handle, handle=handle
                )
            elif name:
                key = manual_author_key(name)
                store.ensure_author("manual", key.split(":", 1)[1], name=name)
            else:
                key = ""
            if key:
                fields["author_key"] = key
        if ref is not None:
            store.set_manual(point_id, fields, ref=ref)
            record = store.get(point_id)
            if record is not None and record.get("platform") in {"pixiv", "twitter"}:
                domain = PIXIV_DOMAIN if record["platform"] == "pixiv" else TWITTER_DOMAIN
                state = store.domain_state(domain)
                if int(state.get("blocked_until") or 0) > int(time.time()):
                    return provenance_payload(point_id, payload, path)
                service = make_service()
                try:
                    service.enrich_record(record)
                except (FetchError, BudgetExhausted) as exc:
                    store.set_status(point_id, getattr(exc, "status", "error"), str(exc))
                finally:
                    service.close()
        elif fields:
            store.set_manual(point_id, fields)
        return provenance_payload(point_id, payload, path)

    @app.post("/api/images/{point_id}/provenance/resolve")
    def resolve_image_provenance(
        point_id: str, request: ProvenanceUrlRequest
    ) -> dict[str, Any]:
        payload, path = resolve_item(point_id)
        ref = parse_source_text(request.url)
        if ref is None:
            raise HTTPException(400, "无法识别该 URL")
        store: ProvenanceStore = app.state.provenance
        ensure_provenance_record(point_id, payload, path)
        store.set_manual(point_id, {}, ref=ref)
        record = store.get(point_id)
        if record is not None:
            domain = PIXIV_DOMAIN if ref.platform == "pixiv" else TWITTER_DOMAIN
            state = store.domain_state(domain)
            if int(state.get("blocked_until") or 0) <= int(time.time()):
                service = make_service()
                try:
                    service.enrich_record(record)
                except (FetchError, BudgetExhausted) as exc:
                    store.set_status(point_id, getattr(exc, "status", "error"), str(exc))
                finally:
                    service.close()
        return provenance_payload(point_id, payload, path)

    @app.get("/api/authors")
    def list_authors(q: str = "", limit: int = 40) -> dict[str, Any]:
        limit = max(1, min(int(limit), 200))
        return {"authors": app.state.provenance.list_authors(q, limit=limit)}

    @app.get("/api/authors/{author_key}")
    def get_author(author_key: str) -> dict[str, Any]:
        summary = app.state.provenance.author_summary(author_key)
        if summary is None:
            raise HTTPException(404, "作者不存在")
        ids = app.state.provenance.point_ids_for_author(author_key)
        return {"author": summary, "point_ids": ids, "total": len(ids)}

    @app.get("/api/authors/{author_key}/images")
    def author_images(
        author_key: str,
        limit: int = 40,
        offset: int = Query(default=0, ge=0, le=50_000),
    ) -> dict[str, Any]:
        store: ProvenanceStore = app.state.provenance
        summary = store.author_summary(author_key)
        if summary is None:
            raise HTTPException(404, "作者不存在")
        ids = store.point_ids_for_author(author_key)
        total = len(ids)
        page_size = max(1, min(int(limit), 200))
        page_ids = ids[offset : offset + page_size]
        payloads = retrieve_collection_payloads(page_ids)
        results = [
            gallery_result(point_id, payloads[point_id])
            for point_id in page_ids
            if isinstance(payloads.get(point_id), dict)
        ]
        return {
            "author": summary,
            "offset": offset,
            "limit": page_size,
            "total": total,
            "has_more": offset + len(page_ids) < total,
            "results": results,
        }

    def ensure_tagger():
        if app.state.tagger is None and config.tagger.enabled:
            from ..models.tagger import WDTagger

            app.state.tagger = WDTagger(config.tagger, config.embedding)
        return app.state.tagger

    def close_tagger() -> None:
        tagger = app.state.tagger
        app.state.tagger = None
        closer = getattr(tagger, "close", None)
        if callable(closer):
            closer()

    def job_status() -> dict[str, Any]:
        job: IndexJob = app.state.index_job
        return {
            "running": job.running,
            "path": job.path,
            "done": job.done,
            "total": job.total,
            "message": job.message,
            "error": job.error,
            "summary": job.summary,
        }

    def run_index_job(directories: list[Path], recursive: bool) -> None:
        job: IndexJob = app.state.index_job
        # Reset before work starts and again in finally: an indexer may write
        # a partial collection before raising, which must remain browseable.
        reset_collection_index_state()
        job.path = "、".join(str(item) for item in directories)
        job.done = 0
        job.total = 0
        job.error = None
        job.summary = None
        job.message = "正在准备索引…"
        try:
            job.message = "正在载入图文向量模型…"
            with app.state.engine.hold_model():
                job.message = "正在载入标签器…"
                indexer = ImageIndexer(
                    config,
                    embedder=app.state.engine.embedder,
                    store=app.state.engine.store,
                    tagger=ensure_tagger(),
                    llm=app.state.engine.llm,
                )
                job.message = "正在扫描并编码图片…"

                def progress(done: int, total: int, root: str) -> None:
                    job.done = done
                    job.total = total
                    job.message = f"正在更新 {Path(root).name or root}（{done}/{max(total, 1)}）"

                summary = indexer.index(directories, recursive=recursive, progress=progress)
                if indexer.folder_catalog is not None:
                    app.state.engine.replace_folder_catalog(indexer.folder_catalog)
                job.summary = {
                    "discovered": summary.discovered,
                    "indexed": summary.indexed,
                    "unchanged": summary.unchanged,
                    "deleted": summary.deleted,
                    "failed": len(summary.failed),
                }
                job.message = "索引完成"
                if summary.llm_warning:
                    job.message = f"索引完成（{summary.llm_warning}）"
        except Exception as exc:
            job.error = str(exc)
            job.message = "索引失败"
        finally:
            reset_collection_index_state()
            close_tagger()
            job.running = False
            if job.error is None:
                # Local parsing is cheap; network enrichment is gated by the
                # provenance job's budgets and never blocks index completion.
                try:
                    start_scan_job(deep=False, only_new=True)
                except Exception:
                    pass

    @app.get("/api/index-status")
    def index_status() -> dict:
        return job_status()

    def resolve_index_directories(request: IndexAddRequest) -> list[Path]:
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

    @app.post("/api/index")
    def add_index(request: IndexAddRequest) -> dict:
        job: IndexJob = app.state.index_job
        directories = resolve_index_directories(request)
        with app.state.index_lock:
            if job.running:
                raise HTTPException(409, "已有索引任务正在进行")
            job.running = True
            job.message = "正在排队…"
            job.error = None
            job.summary = None
            reset_collection_index_state()
        thread = threading.Thread(
            target=run_index_job,
            args=(directories, request.recursive),
            daemon=True,
            name="artseek-index",
        )
        thread.start()
        return job_status()

    @app.post("/api/index-remove")
    def remove_index(request: IndexRemoveRequest) -> dict:
        job: IndexJob = app.state.index_job
        if job.running:
            raise HTTPException(409, "正在索引，请稍后再移出目录")
        catalog = app.state.engine.folder_catalog()
        key = canonical_path(request.folder_key)
        if key not in catalog.folders:
            raise HTTPException(404, "该目录不在索引中")
        drop_keys = catalog.subtree_keys(key)
        ids = catalog.subtree_ids(key)
        state = IncrementalIndexState(config.index_state_path)
        if ids:
            state.mark_all_dirty()
        app.state.engine.store.delete_ids(ids)
        state.invalidate_all()
        for point_id in ids:
            thumb = config.thumbnail_dir / f"{point_id}.webp"
            thumb.unlink(missing_ok=True)
        if ids:
            catalog = rebuild_catalog(app.state.engine.store, config)
            app.state.engine.replace_folder_catalog(catalog)
        reset_collection_index_state()
        scope = load_scope(config)
        scope.selected = [item for item in scope.selected if item not in drop_keys]
        scope.known = [item for item in scope.known if item not in drop_keys]
        save_scope(config, scope)
        return current_library()

    @app.get("/api/model")
    def model_status() -> dict:
        return app.state.engine.model_status()

    @app.post("/api/model")
    def update_model(request: ModelControlRequest) -> dict:
        engine = app.state.engine
        if request.auto_unload_minutes is not None:
            engine.set_auto_unload_seconds(request.auto_unload_minutes * 60)
        if request.loaded is True:
            try:
                engine.load_model()
            except Exception as exc:
                raise HTTPException(500, f"载入模型失败：{exc}") from exc
        elif request.loaded is False:
            if not engine.unload_model():
                status = engine.model_status()
                status["message"] = "正在检索或索引，稍后再卸载"
                return status
        return engine.model_status()

    @app.get("/api/web-settings")
    def web_settings() -> dict:
        return dataclasses.asdict(load_web_search_settings(config))

    @app.put("/api/web-settings")
    def update_web_settings(request: WebSearchSettingsRequest) -> dict:
        settings = WebSearchSettings(
            use_llm=request.use_llm,
            strict_tags=request.strict_tags,
            dedupe=request.dedupe,
            limit=request.limit,
        )
        save_web_search_settings(config, settings)
        return dataclasses.asdict(settings)

    def storage_paths_payload(*, qdrant_restart_required: bool | None = None) -> dict[str, Any]:
        active = app.state.active_qdrant_path
        restart = (
            qdrant_restart_required
            if qdrant_restart_required is not None
            else not paths_equal(config.qdrant_path, active)
        )
        return {
            "qdrant_path": str(config.qdrant_path),
            "notes_dir": str(config.notes_dir),
            "cache_dir": str(config.cache_dir),
            "index_dir": str(config.index_dir),
            "active_qdrant_path": str(active),
            "qdrant_restart_required": restart,
        }

    def _payload_phash(payload: dict[str, Any] | None) -> str:
        if not payload:
            return ""
        stored = str(payload.get("phash") or "").strip().casefold()
        if len(stored) == PHASH_LENGTH:
            return stored
        raw_path = Path(str(payload.get("path") or ""))
        if not raw_path.is_file():
            return ""
        try:
            return perceptual_hash(raw_path)
        except (OSError, ValueError):
            return ""

    def notes_context(point_id: str) -> tuple[dict[str, Any], str]:
        getter = getattr(app.state.engine.store, "get", None)
        payload = getter(point_id) if callable(getter) else None
        digest = ""
        if payload:
            try:
                digest = resolve_content_hash(payload)
            except FileNotFoundError:
                digest = app.state.notes.lookup_by_point_id(point_id) or ""
            except NoteError as exc:
                raise HTTPException(400, str(exc)) from exc
        if not digest:
            digest = app.state.notes.lookup_by_point_id(point_id) or ""
        if not digest:
            detail = "图片不在索引中" if not payload else "原图片已移动或删除，且没有已保存的笔记"
            raise HTTPException(404, detail)
        identity = note_identity(payload or {}, point_id)
        try:
            canonical = app.state.notes.resolve_canonical(digest, **identity)
        except NoteError as exc:
            raise HTTPException(400, str(exc)) from exc
        if not payload:
            meta = app.state.notes.image_meta(canonical)
            payload = {
                "name": meta.get("name") or "",
                "path": meta.get("path") or "",
                "content_hash": canonical,
            }
        payload = dict(payload)
        payload["phash"] = identity.get("phash") or payload.get("phash") or ""
        return payload, canonical

    def note_identity(payload: dict[str, Any], point_id: str) -> dict[str, Any]:
        try:
            width = int(payload.get("width") or 0)
        except (TypeError, ValueError):
            width = 0
        try:
            height = int(payload.get("height") or 0)
        except (TypeError, ValueError):
            height = 0
        return {
            "name": str(payload.get("name") or ""),
            "path": str(payload.get("path") or ""),
            "point_id": point_id,
            "phash": _payload_phash(payload),
            "width": width,
            "height": height,
        }

    @app.get("/api/storage-paths")
    def storage_paths() -> dict:
        return storage_paths_payload()

    @app.put("/api/storage-paths")
    def update_storage_paths(request: StoragePathsRequest) -> dict:
        update_qdrant = request.qdrant_path is not None
        update_notes = request.notes_dir is not None
        update_cache = request.cache_dir is not None
        update_index = request.index_dir is not None
        if not update_qdrant and not update_notes and not update_cache and not update_index:
            raise HTTPException(400, "请提供要保存的存储路径")
        try:
            save_storage_paths(
                config,
                qdrant_path=request.qdrant_path,
                notes_dir=request.notes_dir,
                cache_dir=request.cache_dir,
                index_dir=request.index_dir,
                update_qdrant=update_qdrant,
                update_notes=update_notes,
                update_cache=update_cache,
                update_index=update_index,
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        if update_notes or update_cache or update_index:
            config.ensure_dirs()
        if update_notes:
            app.state.notes.set_root(config.notes_dir)
        if update_cache:
            app.state.search_snapshots.set_root(config.search_snapshot_dir)
        if update_index:
            app.state.collections = CollectionStore(config.collections_path)
        return storage_paths_payload()

    def _reopen_vector_store(path: Path) -> None:
        app.state.engine.store = VectorStore(config)
        app.state.active_qdrant_path = path

    @app.post("/api/storage-paths/migrate")
    def migrate_storage_path(request: StorageMigrateRequest) -> dict:
        if app.state.index_job.running:
            raise HTTPException(409, "正在索引图片，请稍后再迁移")
        if not app.state.storage_lock.acquire(blocking=False):
            raise HTTPException(409, "正在迁移存储目录")
        try:
            if request.kind == "notes":
                source = Path(app.state.notes.root)
                try:
                    dest, copied = migrate_storage_tree(source, request.destination)
                    save_storage_paths(
                        config, notes_dir=dest, update_notes=True
                    )
                except ValueError as exc:
                    raise HTTPException(400, str(exc)) from exc
                app.state.notes.set_root(config.notes_dir)
                return {
                    **storage_paths_payload(),
                    "kind": "notes",
                    "copied": copied,
                    "source": str(source),
                    "destination": str(config.notes_dir),
                }

            if request.kind == "cache":
                source = Path(config.cache_dir)
                try:
                    dest, copied = migrate_storage_tree(source, request.destination)
                    save_storage_paths(config, cache_dir=dest, update_cache=True)
                except ValueError as exc:
                    raise HTTPException(400, str(exc)) from exc
                config.ensure_dirs()
                app.state.search_snapshots.set_root(config.search_snapshot_dir)
                return {
                    **storage_paths_payload(),
                    "kind": "cache",
                    "copied": copied,
                    "source": str(source),
                    "destination": str(config.cache_dir),
                }

            if request.kind == "index":
                source = Path(config.index_dir)
                try:
                    dest, copied = migrate_index_sidecars(source, request.destination)
                    save_storage_paths(config, index_dir=dest, update_index=True)
                except ValueError as exc:
                    raise HTTPException(400, str(exc)) from exc
                config.ensure_dirs()
                app.state.collections = CollectionStore(config.collections_path)
                return {
                    **storage_paths_payload(),
                    "kind": "index",
                    "copied": copied,
                    "source": str(source),
                    "destination": str(config.index_dir),
                }

            source = Path(app.state.active_qdrant_path)
            live = isinstance(app.state.engine.store, VectorStore)
            if live:
                app.state.engine.store.close()
            switched = False
            try:
                dest, copied = migrate_storage_tree(source, request.destination)
                save_storage_paths(config, qdrant_path=dest, update_qdrant=True)
                if live:
                    _reopen_vector_store(config.qdrant_path)
                else:
                    app.state.active_qdrant_path = config.qdrant_path
                switched = True
            except Exception:
                if live and not switched:
                    if not paths_equal(config.qdrant_path, source):
                        save_storage_paths(config, qdrant_path=source, update_qdrant=True)
                    _reopen_vector_store(source)
                raise
            return {
                **storage_paths_payload(),
                "kind": "qdrant",
                "copied": copied,
                "source": str(source),
                "destination": str(config.qdrant_path),
            }
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except Exception as exc:
            raise HTTPException(500, f"迁移失败：{exc}") from exc
        finally:
            app.state.storage_lock.release()

    @app.get("/api/images/{point_id}/notes")
    def list_image_notes(point_id: str) -> dict:
        payload, digest = notes_context(point_id)
        try:
            notes = app.state.notes.list_notes(digest, **note_identity(payload, point_id))
        except NoteError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {
            "image": {
                "id": point_id,
                "content_hash": digest,
                "name": str(payload.get("name") or ""),
                "path": str(payload.get("path") or ""),
            },
            "notes": [item.as_dict() for item in notes],
        }

    @app.post("/api/images/{point_id}/notes")
    def create_image_note(point_id: str, request: NoteCreateRequest) -> dict:
        payload, digest = notes_context(point_id)
        try:
            note = app.state.notes.create_note(
                digest,
                title=request.title,
                content=request.content,
                **note_identity(payload, point_id),
            )
        except NoteError as exc:
            raise HTTPException(400, str(exc)) from exc
        return note.as_dict()

    @app.put("/api/images/{point_id}/note-order")
    def reorder_image_notes(point_id: str, request: NoteOrderRequest) -> dict:
        payload, digest = notes_context(point_id)
        try:
            notes = app.state.notes.reorder_notes(
                digest,
                request.names,
                **note_identity(payload, point_id),
            )
        except NoteError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"notes": [item.as_dict() for item in notes]}

    @app.get("/api/images/{point_id}/notes/{name}")
    def read_image_note(point_id: str, name: str) -> dict:
        _payload, digest = notes_context(point_id)
        try:
            return app.state.notes.read_note(digest, name).as_dict()
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        except NoteError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.put("/api/images/{point_id}/notes/{name}")
    def write_image_note(point_id: str, name: str, request: NoteWriteRequest) -> dict:
        payload, digest = notes_context(point_id)
        try:
            note = app.state.notes.write_note(
                digest,
                name,
                title=request.title,
                content=request.content,
                **note_identity(payload, point_id),
            )
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        except NoteError as exc:
            raise HTTPException(400, str(exc)) from exc
        return note.as_dict()

    @app.delete("/api/images/{point_id}/notes/{name}")
    def delete_image_note(point_id: str, name: str) -> dict:
        _payload, digest = notes_context(point_id)
        try:
            removed = app.state.notes.delete_note(digest, name)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        except NoteError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"deleted": True, "removed_assets": removed}

    @app.delete("/api/images/{point_id}/notes")
    def delete_all_image_notes(point_id: str) -> dict:
        _payload, digest = notes_context(point_id)
        try:
            app.state.notes.delete_folder(digest)
        except NoteError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"deleted": True, "folder": digest}

    @app.post("/api/images/{point_id}/assets")
    async def upload_note_asset(point_id: str, file: UploadFile = File(...)) -> dict:
        payload, digest = notes_context(point_id)
        data = await file.read()
        filename = file.filename or "paste"
        if not Path(filename).suffix:
            filename += mimetypes.guess_extension(file.content_type or "") or ".png"
        try:
            relative = app.state.notes.save_asset(
                digest,
                filename,
                data,
                **note_identity(payload, point_id),
            )
        except NoteError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {
            "path": relative,
            "url": f"/api/note-files/{digest}/{relative}",
        }

    @app.get("/api/note-files/{content_hash}/{relative:path}")
    def note_file(content_hash: str, relative: str) -> FileResponse:
        try:
            path = app.state.notes.public_file(content_hash, relative)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        except NoteError as exc:
            raise HTTPException(400, str(exc)) from exc
        media_type, _ = mimetypes.guess_type(path.name)
        return FileResponse(path, media_type=media_type or "application/octet-stream")


    static_dir = Path(__file__).with_name("static")
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    return app
