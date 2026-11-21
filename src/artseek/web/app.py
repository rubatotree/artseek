from __future__ import annotations

import dataclasses
import json
import mimetypes
import random
import re
import tempfile
import threading
import time
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from ..api.app import create_app as create_v2_app
from ..api.schemas import (
    ArtistCoverRequest,
    ArtistGalleryRequest,
    ArtistIdentityRemoveRequest,
    ArtistIdentityRequest,
    ArtistImagesRequest,
    ArtistMembersRequest,
    ArtistMergeRequest,
    ArtistNameRequest,
    ArtistScopeRequest,
    AutoTagRequest,
    CollectionItemOrderRequest,
    CollectionItemRequest,
    CollectionNameRequest,
    CollectionOrderRequest,
    CollectionUpdateRequest,
    FolderScopeRequest,
    IndexAddRequest,
    IndexRemoveRequest,
    ModelControlRequest,
    NoteCreateRequest,
    NoteOrderRequest,
    NoteWriteRequest,
    ProvenanceEnrichRequest,
    ProvenanceManualRequest,
    ProvenanceScanRequest,
    ProvenanceUrlRequest,
    RandomBrowseRequest,
    ReferenceSpec,
    ScopeImagesRequest,
    StorageMigrateRequest,
    StoragePathsRequest,
    SuggestionRequest,
    TextSearchRequest,
    WebSearchSettingsRequest,
)
from ..collections import (
    COLLECTION_SORT_KEYS,
    DEFAULT_COLLECTION_SORT,
    PAYLOAD_SORT_KEYS,
    SORT_ADDED_DESC,
    SORT_MANUAL,
    Collection,
    CollectionStore,
    HistoryAction,
    artist_item_key,
    is_artist_item,
)
from ..config import (
    AppConfig,
    WebSearchSettings,
    load_web_search_settings,
    migrate_index_sidecars,
    migrate_storage_tree,
    paths_equal,
    save_storage_paths,
    save_web_search_settings,
)
from ..image_io import (
    PHASH_LENGTH,
    canonical_path,
    make_search_image,
    make_thumbnail,
    perceptual_hash,
)
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
    natural_key,
    rebuild_catalog,
    save_scope,
)
from ..notes import NoteError, NoteStore, resolve_content_hash
from ..provenance import (
    PIXIV_DOMAIN,
    TWITTER_DOMAIN,
    UNASSIGNED_AUTHOR_KEY,
    BudgetExhausted,
    FetchError,
    FxTwitterClient,
    PixivClient,
    ProvenanceService,
    ProvenanceStore,
    RateLimiter,
    normalize_pixiv_user_id,
    normalize_social_link,
    normalize_twitter_handle,
    normalize_xiaohongshu_user_id,
    parse_payload_path,
    parse_source_text,
    record_unresolved,
    scan_payloads,
)
from ..query import QueryPlan
from ..search import SearchEngine
from ..search_cache import SearchSnapshotStore, is_snapshot_id
from ..store import VectorStore
from ..tag_dictionary import expand_tag_aliases, search_tags, tag_info
from ..tag_query import (
    TagQueryError,
    evaluate_tag_expression_with_universe,
    parse_tag_expression,
)


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
    if engine is None:
        return create_v2_app(config)
    app = FastAPI(title="ArtSeek", docs_url="/api/docs", redoc_url=None)
    app.state.config = config
    app.state.engine = engine
    app.state.tagger = None
    app.state.index_job = IndexJob()
    app.state.index_lock = threading.Lock()
    app.state.collections = CollectionStore(config.collections_path)
    app.state.notes = NoteStore(config.notes_dir)
    app.state.provenance = ProvenanceStore(config.provenance_path)
    app.state.provenance_job = ProvenanceJob()
    app.state.provenance_lock = threading.RLock()
    app.state.provenance_refresh_pending = False
    app.state.search_snapshots = SearchSnapshotStore(config.search_snapshot_dir)
    app.state.active_qdrant_path = config.qdrant_path
    app.state.storage_lock = threading.Lock()
    app.state.collection_index_empty = False
    app.state.collection_index_generation = 0
    app.state.collection_id_cursors: OrderedDict[str, CollectionIdCursor] = OrderedDict()
    app.state.collection_sort_cursors: OrderedDict[str, CollectionSortCursor] = OrderedDict()
    app.state.collection_id_cursors_lock = threading.RLock()
    app.state.tag_suggestion_lock = threading.RLock()
    app.state.image_tag_suggestions = {"generation": -1, "items": []}
    app.state.author_scope_cache: OrderedDict[
        tuple[Any, ...], tuple[float, dict[str, int]]
    ] = OrderedDict()
    app.state.author_scope_cache_lock = threading.RLock()
    app.state.artist_catalog_cache: OrderedDict[
        tuple[Any, ...],
        tuple[float, list[dict[str, Any]], dict[str, list[dict[str, Any]]]],
    ] = OrderedDict()
    static_path = Path(__file__).with_name("static") / "index.html"

    def invalidate_author_cache() -> None:
        with app.state.author_scope_cache_lock:
            app.state.author_scope_cache.clear()
            app.state.artist_catalog_cache.clear()

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
    @app.get("/scope", response_class=HTMLResponse)
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

    @app.get("/tag/{tag:path}", response_class=HTMLResponse)
    def spa_tag(tag: str) -> HTMLResponse:
        del tag
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
            "mtime_ns": int(payload.get("mtime_ns") or 0),
            "tags": list(payload.get("tags") or []),
            "tag_hits": [],
            "caption": payload.get("caption"),
        }

    def attach_provenance(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Attach a compact source snapshot so cards/viewer need no extra fetch."""
        if not results:
            return results
        store: ProvenanceStore = app.state.provenance
        ids = [str(item.get("id") or "") for item in results]
        records = store.get_many([item for item in ids if item])
        author_keys = [
            str(record["author_key"])
            for record in records.values()
            if record.get("author_key")
        ]
        groups = store.author_groups(author_keys)
        names = store.author_names(author_keys)
        for item in results:
            record = records.get(str(item.get("id") or ""))
            if record is None:
                continue
            author_key = record.get("author_key")
            item["source"] = {
                "title": record.get("title"),
                "description": record.get("description"),
                "url": record.get("url"),
                "tags": list(record.get("tags") or []),
                "manual_tags": list(record.get("manual_tags") or []),
                "auto_tags": list(record.get("auto_tags") or []),
                "platform": record.get("platform"),
                "status": record.get("status"),
                "author_key": author_key,
                "author_name": names.get(str(author_key), "") if author_key else "",
                "author_members": groups.get(str(author_key), []) if author_key else [],
            }
        return results

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

    def artist_collection_result(
        key: str,
        summary: dict[str, Any],
        latest_mtime: int,
        cover: dict[str, Any] | None,
    ) -> dict[str, Any]:
        return {
            "id": f"artist:{key}",
            "kind": "artist",
            "artist_key": key,
            "name": summary.get("name") or summary.get("handle") or key,
            "handle": summary.get("handle") or "",
            "members": summary.get("members") or [],
            "image_count": int(summary.get("image_count") or 0),
            "latest_mtime": int(latest_mtime or 0),
            "width": int((cover or {}).get("width") or 0),
            "height": int((cover or {}).get("height") or 0),
            "cover": cover,
        }

    def mixed_collection_page(
        collection: Collection,
        method: str,
        start: int,
        page_size: int,
        needle: str,
        allowed_author_key: str | None,
        seed: int | None,
    ) -> tuple[list[dict[str, Any]], int, bool]:
        """Materialize collections that mix image IDs and artist IDs."""
        store: ProvenanceStore = app.state.provenance
        item_ids = list(collection.item_ids)
        image_ids = [item_id for item_id in item_ids if not is_artist_item(item_id)]
        artist_keys = [
            artist_item_key(item_id) for item_id in item_ids if is_artist_item(item_id)
        ]
        image_payloads: dict[str, dict[str, Any]] = {}
        for batch_start in range(0, len(image_ids), 256):
            batch = image_ids[batch_start : batch_start + 256]
            image_payloads.update(retrieve_collection_payloads(batch))
        summaries = {
            key: summary
            for key in artist_keys
            if (summary := store.author_summary(key)) is not None
        }
        latest = store.author_latest_mtimes(list(summaries))
        image_ids_by_artist = store.point_ids_for_authors(list(summaries))
        explicit_covers = store.artist_cover_ids(list(summaries))
        cover_ids = [
            explicit_covers.get(key) or ids[0]
            for key, ids in image_ids_by_artist.items()
            if explicit_covers.get(key) or ids
        ]
        cover_payloads = retrieve_collection_payloads(list(dict.fromkeys(cover_ids)))
        allowed_image_ids = (
            set(store.point_ids_for_author(allowed_author_key))
            if allowed_author_key
            else None
        )
        entries: list[tuple[int | str, str, dict[str, Any] | None]] = []
        for item_id in item_ids:
            if is_artist_item(item_id):
                key = artist_item_key(item_id)
                summary = summaries.get(key)
                if summary is None:
                    continue
                if allowed_author_key and key != allowed_author_key:
                    continue
                searchable = "\n".join(
                    [
                        str(summary.get("name") or ""),
                        str(summary.get("handle") or ""),
                        key,
                        *(
                            str(member.get("handle") or member.get("platform_id") or "")
                            for member in summary.get("members") or []
                        ),
                    ]
                ).casefold()
                if needle and needle not in searchable:
                    continue
                cover_id = explicit_covers.get(key) or next(
                    (
                        point_id
                        for point_id in image_ids_by_artist.get(key, [])
                        if point_id in cover_payloads
                    ),
                    "",
                )
                cover_payload = cover_payloads.get(cover_id) if cover_id else None
                cover = (
                    {
                        "id": cover_id,
                        "name": cover_payload.get("name"),
                        "width": cover_payload.get("width"),
                        "height": cover_payload.get("height"),
                    }
                    if isinstance(cover_payload, dict)
                    else None
                )
                result = artist_collection_result(
                    key, summary, latest.get(key, 0), cover
                )
                sort_key: int | str = (
                    result["latest_mtime"]
                    if method.startswith("mtime")
                    else str(result["name"]).casefold()
                )
                entries.append((sort_key, item_id, result))
            else:
                payload = image_payloads.get(item_id)
                if not isinstance(payload, dict):
                    continue
                if allowed_image_ids is not None and item_id not in allowed_image_ids:
                    continue
                if needle:
                    tags = payload.get("tags") or []
                    searchable = "\n".join(
                        [
                            str(payload.get("name") or ""),
                            str(payload.get("path") or ""),
                            str(payload.get("caption") or ""),
                            *(str(tag) for tag in tags),
                        ]
                    ).casefold()
                    if needle not in searchable:
                        continue
                entries.append((collection_sort_key(payload, method), item_id, None))
        if method == "random":
            seed = seed if seed is not None else random.SystemRandom().randrange(1 << 53)
            random.Random(seed).shuffle(entries)
        elif method == "manual":
            rank = {
                item_id: index
                for index, item_id in enumerate(collection.ordered_item_ids())
            }
            entries.sort(key=lambda entry: rank.get(entry[1], len(rank)))
        elif method == SORT_ADDED_DESC:
            entries.reverse()
        elif method in PAYLOAD_SORT_KEYS:
            entries.sort(
                key=lambda entry: entry[0],
                reverse=method.endswith("_desc"),
            )
        page_entries = entries[start : start + page_size]
        page_image_ids = [
            item_id for _key, item_id, result in page_entries if result is None
        ]
        image_results = {
            str(item["id"]): item
            for item in attach_provenance(
                [
                    gallery_result(item_id, image_payloads[item_id])
                    for item_id in page_image_ids
                    if isinstance(image_payloads.get(item_id), dict)
                ]
            )
        }
        page = [
            result if result is not None else image_results.get(item_id)
            for _key, item_id, result in page_entries
        ]
        return (
            [item for item in page if isinstance(item, dict)],
            len(entries),
            start + len(page_entries) < len(entries),
        )

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
        if any(is_artist_item(item_id) for item_id in collection.item_ids):
            results, _total, _has_more = mixed_collection_page(
                collection,
                sort,
                0,
                MAX_CACHED_COLLECTION_IDS,
                "",
                None,
                None,
            )
            return [str(item["id"]) for item in results]
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
        seed: int | None = None,
    ) -> dict:
        key = canonical_path(folder_key)
        catalog = app.state.engine.folder_catalog()
        acc = catalog.folders.get(key)
        if acc is None:
            raise HTTPException(404, "该目录不在索引中")
        method = sort if sort in SORT_KEYS or sort == "random" else DEFAULT_SORT
        page_size = max(1, min(limit, 200))
        start = max(0, offset)
        refs = catalog.list_folder_images(key, method)
        if author_key:
            allowed = set(app.state.provenance.point_ids_for_author(author_key))
            refs = [item for item in refs if item[0] in allowed]
        if method == "random":
            seed = (
                seed
                if seed is not None
                else random.SystemRandom().randrange(1 << 53)
            )
            random.Random(seed).shuffle(refs)
        total = len(refs)
        page = refs[start : start + page_size]
        payloads = app.state.engine.store.retrieve_payloads([point_id for point_id, _, _ in page])
        results: list[dict[str, Any]] = []
        for point_id, name, _mtime in page:
            payload = payloads.get(point_id) or {}
            results.append(gallery_result(point_id, payload, name))
        attach_provenance(results)
        return {
            "folder_key": key,
            "folder_name": acc.name,
            "folder_path": acc.path,
            "sort": method,
            "seed": seed if method == "random" else None,
            "offset": start,
            "limit": page_size,
            "total": total,
            "has_more": start + len(results) < total,
            "results": results,
        }

    @app.get("/api/scope-images")
    def scope_images(
        folder_keys: str = "",
        sort: str = DEFAULT_SORT,
        seed: int | None = None,
        limit: int = 40,
        offset: int = 0,
        author_key: str | None = None,
        mtime_after: str | None = None,
        mtime_before: str | None = None,
    ) -> dict[str, Any]:
        keys = parse_folder_keys(folder_keys) or []
        if not keys:
            raise HTTPException(400, "请先勾选要查看的目录")
        catalog = app.state.engine.folder_catalog()
        ids = catalog.point_ids(
            keys,
            after_ns=day_start_ns(mtime_after),
            before_ns=day_end_ns(mtime_before),
        )
        if ids is None:
            ids = [point_id for acc in catalog.folders.values() for point_id in acc.ids]
        if author_key:
            allowed = set(app.state.provenance.point_ids_for_author(author_key))
            ids = [point_id for point_id in ids if point_id in allowed]
        ids = list(dict.fromkeys(ids))
        method = sort if sort in SORT_KEYS or sort == "random" else DEFAULT_SORT
        if method == "random":
            seed = (
                seed
                if seed is not None
                else random.SystemRandom().randrange(1 << 53)
            )
            random.Random(seed).shuffle(ids)
        elif method in {"name", "name_desc", "mtime", "mtime_asc"}:
            wanted = set(ids)
            names: dict[str, str] = {}
            mtimes: dict[str, int] = {}
            for acc in catalog.folders.values():
                for index, point_id in enumerate(acc.ids):
                    if point_id not in wanted:
                        continue
                    names[point_id] = (
                        acc.id_names[index] if index < len(acc.id_names) else point_id
                    )
                    mtimes[point_id] = (
                        acc.id_mtimes[index] if index < len(acc.id_mtimes) else 0
                    )
            if method.startswith("name"):
                ids.sort(
                    key=lambda point_id: (
                        names.get(point_id, point_id).casefold(),
                        point_id,
                    ),
                    reverse=method == "name_desc",
                )
            else:
                ids.sort(
                    key=lambda point_id: (mtimes.get(point_id, 0), point_id),
                    reverse=method == "mtime",
                )
        total = len(ids)
        page_size = max(1, min(limit, 200))
        start = max(0, min(offset, 50_000))
        page_ids = ids[start : start + page_size]
        payloads = app.state.engine.store.retrieve_payloads(page_ids)
        results = attach_provenance(
            [
                gallery_result(point_id, payloads[point_id])
                for point_id in page_ids
                if isinstance(payloads.get(point_id), dict)
            ]
        )
        return {
            "folder_keys": keys,
            "folder_count": len(keys),
            "sort": method,
            "seed": seed if method == "random" else None,
            "offset": start,
            "limit": page_size,
            "total": total,
            "has_more": start + len(page_ids) < total,
            "results": results,
        }

    @app.post("/api/scope-images")
    def scope_images_post(request: ScopeImagesRequest) -> dict[str, Any]:
        return scope_images(
            folder_keys=json.dumps(request.folder_keys, ensure_ascii=False),
            sort=request.sort,
            seed=request.seed,
            limit=request.limit,
            offset=request.offset,
            author_key=request.author_key,
            mtime_after=request.mtime_after,
            mtime_before=request.mtime_before,
        )

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

    def collection_item_name(item_id: str) -> str:
        if is_artist_item(item_id):
            summary = app.state.provenance.author_summary(artist_item_key(item_id))
            if summary is not None:
                return str(summary.get("name") or summary.get("handle") or artist_item_key(item_id))
            return ""
        payload = app.state.engine.store.get(item_id)
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
        if is_artist_item(request.image_id):
            if app.state.provenance.author_summary(artist_item_key(request.image_id)) is None:
                raise HTTPException(404, "画师不存在，无法收藏")
        elif not app.state.engine.store.get(request.image_id):
            raise HTTPException(404, "图片不在索引中，无法收藏")
        collection, added = app.state.collections.add_item(
            collection_id,
            request.image_id,
            item_name=collection_item_name(request.image_id),
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
            item_name=collection_item_name(point_id),
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
        seed: int | None = None,
    ) -> dict[str, Any]:
        collection = require_collection(collection_id)
        # A query-level sort is a temporary view override. It never persists.
        random_requested = sort == "random"
        method = (
            "random"
            if random_requested
            else (sort if sort in COLLECTION_SORT_KEYS else collection.sort)
        )
        if method not in COLLECTION_SORT_KEYS and method != "random":
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

        if any(is_artist_item(item_id) for item_id in collection.item_ids):
            if method == "random" and seed is None:
                seed = random.SystemRandom().randrange(1 << 53)
            mixed_results, mixed_total, mixed_has_more = mixed_collection_page(
                collection,
                method,
                start,
                page_size,
                needle,
                author_key,
                seed,
            )
            return {
                "collection": collection_summary(collection),
                "query": query.strip(),
                "sort": method,
                "seed": seed if method == "random" else None,
                "saved_sort": collection.sort,
                "offset": start,
                "limit": page_size,
                "total": mixed_total,
                "total_exact": True,
                "has_more": mixed_has_more,
                "results": mixed_results,
            }

        if method == "random":
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
            seed = (
                seed
                if seed is not None
                else random.SystemRandom().randrange(1 << 53)
            )
            random.Random(seed).shuffle(filtered)
            page = filtered[start : start + page_size]
            total = len(filtered)
            has_more = start + len(page) < total
            total_exact = True
        elif needle or allowed is not None:
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
            "seed": seed if method == "random" else None,
            "saved_sort": collection.sort,
            "offset": start,
            "limit": page_size,
            "total": total,
            "total_exact": total_exact,
            "has_more": has_more,
            "results": attach_provenance(
                [gallery_result(point_id, payload) for point_id, payload in page]
            ),
        }

    @app.post("/api/random-browse")
    def random_browse(request: RandomBrowseRequest) -> dict:
        """Return one deterministic page from a random ordering in the selected scope."""
        # Browsing is read-only (WAL snapshots, locked vector store), so it stays
        # available while an index job is queued or running.
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
        attach_provenance(results)
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
        attach_provenance(live)
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
        specs = list(request.references)
        if request.reference_id:
            specs.append(ReferenceSpec(id=request.reference_id, weight=1.0))
        paths, weights = resolve_library_references(specs)
        cleaned_query, author_token, tag_tokens = parse_query_filters(request.query)
        requested_authors = [*request.author_keys]
        if request.author_key:
            requested_authors.append(request.author_key)
        if author_token:
            requested_authors.append(author_token)
        resolved_authors = list(dict.fromkeys(
            resolve_author_token(token)
            for token in requested_authors
            if str(token or "").strip()
        ))
        author_key = resolved_authors[0] if len(resolved_authors) == 1 else None
        text_tags = [*tag_tokens, *request.text_tags]
        restrict_ids = restricted_ids(
            author_keys=resolved_authors,
            tag_tokens=text_tags,
            title_query=request.title_query,
            tag_text=request.tag_text,
            image_tags=request.image_tags,
            tag_expression=request.tag_expression,
        )
        search_text = cleaned_query
        if tag_tokens:
            search_text = " ".join([search_text, *tag_tokens]).strip()
        if request.text_label.strip():
            search_text = " ".join([search_text, request.text_label.strip()]).strip()
        folder_keys = resolve_folder_keys(request.folder_keys)
        if not search_text and not specs and restrict_ids is not None:
            payload = filter_only_search(
                query=request.query,
                author_key=author_key,
                author_keys=resolved_authors,
                tag_tokens=text_tags,
                point_ids=restrict_ids,
                title_query=request.title_query,
                tag_text=request.tag_text,
                image_tags=request.image_tags,
                tag_expression=request.tag_expression,
                sort=request.sort,
                seed=request.seed,
                offset=max(0, request.offset),
                limit=max(1, min(request.limit, 200)),
                folder_keys=folder_keys,
                mtime_after=request.mtime_after,
                mtime_before=request.mtime_before,
            )
        else:
            try:
                response = app.state.engine.search(
                    search_text,
                    images=paths,
                    text_weight=request.text_weight,
                    image_weights=weights,
                    limit=request.limit,
                    expand=request.expand,
                    strict_tags=request.strict_tags,
                    folder_keys=folder_keys,
                    mtime_after=request.mtime_after,
                    mtime_before=request.mtime_before,
                    offset=request.offset,
                    exclude_ids=request.exclude_ids,
                    dedupe=request.dedupe,
                    restrict_ids=restrict_ids,
                )
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from exc
            payload = dataclasses.asdict(response)
            payload["applied_filters"] = {
                "author_key": author_key,
                "author_keys": resolved_authors,
                "tags": text_tags,
                "title_query": request.title_query,
                "text_label": request.text_label,
                "tag_expression": request.tag_expression,
                "tag_text": request.tag_text,
                "image_tags": request.image_tags,
            }
        attach_provenance(payload.get("results") or [])
        payload = persist_search_snapshot(
            snapshot_id=request.snapshot_id,
            query=request.query,
            text_weight=request.text_weight,
            references=specs,
            expand=request.expand,
            strict_tags=request.strict_tags,
            dedupe=request.dedupe,
            limit=request.limit,
            folder_keys=folder_keys,
            mtime_after=request.mtime_after,
            mtime_before=request.mtime_before,
            offset=request.offset,
            payload=payload,
        )
        payload.setdefault(
            "applied_filters",
            {"author_key": author_key, "tags": tag_tokens},
        )
        return payload

    @app.post("/api/local-search")
    def local_search(request: TextSearchRequest) -> dict:
        terms = [
            term.lstrip("#")
            for term in re.split(r"\s+", request.query.strip())
            if term.lstrip("#")
        ]
        if not terms:
            raise HTTPException(400, "请输入 tag、标题或作者关键词")
        store: ProvenanceStore = app.state.provenance
        point_ids = store.point_ids_for_metadata_search(
            [sorted(expand_tag_aliases(term)) for term in terms]
        )
        folder_keys = resolve_folder_keys(request.folder_keys)
        payload = filter_only_search(
            query=request.query,
            author_key=None,
            author_keys=[],
            tag_tokens=[],
            point_ids=point_ids,
            offset=max(0, request.offset),
            limit=max(1, min(request.limit, 200)),
            folder_keys=folder_keys,
            mtime_after=request.mtime_after,
            mtime_before=request.mtime_before,
            sort=request.sort,
            seed=request.seed,
        )
        payload["metadata_search"] = True
        payload["applied_filters"]["mode"] = "metadata"
        return payload

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
        author_keys: str = Form(default="[]"),
        title_query: str = Form(default=""),
        text_label: str = Form(default=""),
        tag_expression: str = Form(default=""),
        sort: str = Form(default="mtime_desc"),
        seed: int | None = Form(default=None),
        text_tags: str = Form(default="[]"),
        tag_text: str = Form(default="[]"),
        image_tags: str = Form(default="[]"),
    ) -> dict:
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
            cleaned_query, author_token, tag_tokens = parse_query_filters(query)
            try:
                extra_authors = json.loads(author_keys or "[]")
            except json.JSONDecodeError as exc:
                raise HTTPException(400, "作者筛选参数不合法") from exc
            if not isinstance(extra_authors, list):
                raise HTTPException(400, "作者筛选参数不合法")
            requested_authors = [*extra_authors]
            if author_key.strip():
                requested_authors.append(author_key)
            if author_token:
                requested_authors.append(author_token)
            resolved_authors = list(dict.fromkeys(
                resolve_author_token(token)
                for token in requested_authors
                if str(token or "").strip()
            ))
            resolved_author = resolved_authors[0] if len(resolved_authors) == 1 else None
            try:
                extra_text_tags = json.loads(text_tags or "[]")
                extra_tag_text = json.loads(tag_text or "[]")
                extra_image_tags = json.loads(image_tags or "[]")
            except json.JSONDecodeError as exc:
                raise HTTPException(400, "标签筛选参数不合法") from exc
            if (
                not isinstance(extra_text_tags, list)
                or not isinstance(extra_tag_text, list)
                or not isinstance(extra_image_tags, list)
            ):
                raise HTTPException(400, "标签筛选参数不合法")
            text_tags = [*tag_tokens, *[str(tag) for tag in extra_text_tags if str(tag).strip()]]
            tag_text = [str(term) for term in extra_tag_text if str(term).strip()]
            image_tags = [str(tag) for tag in extra_image_tags if str(tag).strip()]
            restrict_ids = restricted_ids(
                author_key=resolved_author,
                tag_tokens=text_tags,
                title_query=title_query,
                tag_text=tag_text,
                image_tags=image_tags,
                tag_expression=tag_expression,
            )
            search_text = " ".join(
                [cleaned_query, *tag_tokens]
            ).strip()
            if text_label.strip():
                search_text = " ".join([search_text, text_label.strip()]).strip()
            if (
                not search_text
                and not library_paths
                and not temps
                and restrict_ids is not None
            ):
                payload = filter_only_search(
                    query=query,
                    author_key=resolved_author,
                    author_keys=resolved_authors,
                    tag_tokens=text_tags,
                    point_ids=restrict_ids,
                    offset=max(0, min(offset, 4000)),
                    limit=max(1, min(limit, 200)),
                    folder_keys=folder_keys_list,
                    mtime_after=mtime_after or None,
                    mtime_before=mtime_before or None,
                    title_query=title_query,
                    tag_text=tag_text,
                    image_tags=image_tags,
                    tag_expression=tag_expression,
                    sort=sort,
                    seed=seed,
                )
            else:
                response = app.state.engine.search(
                    search_text,
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
                    restrict_ids=restrict_ids,
                )
                payload = dataclasses.asdict(response)
                payload["applied_filters"] = {
                    "author_key": resolved_author,
                    "author_keys": resolved_authors,
                    "tags": text_tags,
                    "title_query": title_query,
                    "text_label": text_label,
                    "tag_expression": tag_expression,
                    "tag_text": tag_text,
                    "image_tags": image_tags,
                }
            attach_provenance(payload.get("results") or [])
            payload = persist_search_snapshot(
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
                payload=payload,
                had_file_refs=bool(files),
            )
            payload.setdefault(
                "applied_filters",
                {"author_key": resolved_author, "tags": tag_tokens},
            )
            return payload
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

    @app.get("/search-image/{point_id}")
    def search_image(point_id: str) -> FileResponse:
        _, source = resolve_item(point_id)
        destination = config.thumbnail_dir / f"{point_id}-search.jpg"
        if not destination.exists() or destination.stat().st_mtime_ns < source.stat().st_mtime_ns:
            make_search_image(source, destination)
        return FileResponse(destination, media_type="image/jpeg")

    @app.get("/api/item/{point_id}")
    def item(point_id: str) -> dict:
        payload, _ = resolve_item(point_id)
        return payload

    # --------------------------------------------------------------- provenance

    filter_token_re = re.compile(
        r'(?:^|\s)(author|artist|tag):(?:"([^"]+)"|(\S+))',
        re.IGNORECASE,
    )

    def parse_query_filters(query: str) -> tuple[str, str | None, list[str]]:
        author: str | None = None
        tags: list[str] = []

        def take(match: re.Match[str]) -> str:
            nonlocal author
            key = match.group(1).casefold()
            value = (match.group(2) or match.group(3) or "").strip()
            if key in {"author", "artist"} and value:
                author = value
            elif key == "tag" and value:
                tags.append(value)
            return " "

        cleaned = re.sub(r"\s+", " ", filter_token_re.sub(take, query or "")).strip()
        return cleaned, author, tags

    def resolve_author_token(value: str) -> str:
        store: ProvenanceStore = app.state.provenance
        token = str(value or "").strip().lstrip("@")
        if token.casefold() in {
            UNASSIGNED_AUTHOR_KEY,
            "unassigned",
            "无画师",
            "未标注",
            "作者未标注",
            "无作者",
        }:
            return UNASSIGNED_AUTHOR_KEY
        summary = store.author_summary(token)
        if summary is not None:
            return str(summary["key"])
        matches = store.list_authors(token, limit=12)
        if len(matches) == 1:
            return str(matches[0]["key"])
        if not matches:
            raise HTTPException(400, f"没有找到作者：{value}")
        key_to_root = {
            str(item["key"]): store.canonical_author_key(str(item["key"]))
            for item in matches
        }
        grouped: dict[str, list[dict]] = {}
        for item in matches:
            root = key_to_root.get(str(item["key"]), str(item["key"]))
            grouped.setdefault(root, []).append(item)
        ranked = sorted(
            grouped.items(),
            key=lambda pair: (
                -max(float(item.get("score") or 0) for item in pair[1]),
                -len(pair[1]),
                pair[0],
            ),
        )
        if len(ranked) == 1:
            return ranked[0][0]
        top_score = max(float(item.get("score") or 0) for item in ranked[0][1])
        runner_up = max(float(item.get("score") or 0) for item in ranked[1][1])
        if top_score >= 0.8 and top_score - runner_up >= 0.08:
            return ranked[0][0]
        candidates = [
            {
                "key": root,
                "name": members[0].get("name") or members[0].get("handle") or root,
            }
            for root, members in ranked
        ]
        raise HTTPException(
            400,
            detail={
                "message": "作者不唯一，请选择完整作者。",
                "author_candidates": candidates,
            },
        )

    def resolve_author_reference(value: str) -> str:
        """Resolve an exact artist UID, Pixiv ID, X handle, or unique name."""
        store: ProvenanceStore = app.state.provenance
        token = str(value or "").strip().lstrip("@")
        if not token:
            raise HTTPException(400, "请填写作者 UID、Pixiv ID 或 X handle")
        if token.casefold() in {
            UNASSIGNED_AUTHOR_KEY,
            "unassigned",
            "无画师",
            "未标注",
            "作者未标注",
            "无作者",
        }:
            raise HTTPException(400, "无画师不能作为标注作者")
        summary = store.author_summary(token)
        if summary is not None:
            return str(summary["key"])
        owner = store.identity_owner("pixiv", token)
        if owner:
            return owner
        owner = store.identity_owner("twitter", token)
        if owner:
            return owner
        matches = store.list_authors(token, limit=3)
        if len(matches) == 1:
            return store.canonical_author_key(str(matches[0]["key"]))
        if not matches:
            raise HTTPException(400, f"没有找到作者：{value}")
        candidates = [
            {
                "key": str(item["key"]),
                "name": item.get("name") or item.get("handle") or str(item["key"]),
            }
            for item in matches
        ]
        raise HTTPException(
            400,
            detail={
                "message": "作者不唯一，请选择列表中的作者。",
                "author_candidates": candidates,
            },
        )

    def restricted_ids(
        *,
        author_key: str | None = None,
        author_keys: list[str] | None = None,
        tag_tokens: list[str],
        title_query: str = "",
        tag_text: list[str] | None = None,
        image_tags: list[str] | None = None,
        tag_expression: str = "",
    ) -> list[str] | None:
        store: ProvenanceStore = app.state.provenance
        result: set[str] | None = None
        authors = list(dict.fromkeys([
            *(author_keys or []),
            *([author_key] if author_key else []),
        ]))
        if authors:
            allowed: set[str] = set()
            for author in authors:
                allowed.update(store.point_ids_for_author(author))
            result = allowed
        if tag_tokens:
            for tag in tag_tokens:
                group = expand_tag_aliases(tag)
                matching = set(store.point_ids_for_any_tags(sorted(group)))
                result = matching if result is None else result & matching
        if title_query.strip():
            titled = set(store.point_ids_for_title(title_query))
            result = titled if result is None else result & titled
        if tag_text:
            matching = set(store.point_ids_for_tag_text(tag_text))
            result = matching if result is None else result & matching
        if tag_expression.strip():
            try:
                node = parse_tag_expression(tag_expression)
            except TagQueryError as exc:
                raise HTTPException(400, str(exc)) from exc
            universe = set(store.all_point_ids())
            matching = evaluate_tag_expression_with_universe(
                node,
                lambda atom: set(
                    store.point_ids_for_any_tags(sorted(expand_tag_aliases(atom)))
                ),
                universe,
            )
            result = matching if result is None else result & matching
        if image_tags:
            tagged = set(app.state.engine.store.point_ids_for_image_tags(image_tags))
            result = tagged if result is None else result & tagged
        return list(result) if result is not None else None

    def sort_gallery_ids(
        point_ids: Sequence[str], sort: str, seed: int | None
    ) -> tuple[list[str], str, int | None]:
        scoped = list(point_ids)
        order = "mtime_desc" if sort == "mtime" else sort
        if order not in {"mtime_desc", "mtime_asc", "name_asc", "name_desc", "random"}:
            order = "mtime_desc"
        if order == "random":
            seed = seed if seed is not None else random.SystemRandom().randrange(1 << 53)
            random.Random(seed).shuffle(scoped)
        elif order in {"name_asc", "name_desc"}:
            names = app.state.provenance.point_file_names(scoped)
            scoped.sort(
                key=lambda item_id: (natural_key(names.get(item_id, item_id)), item_id),
                reverse=order == "name_desc",
            )
        else:
            mtimes = app.state.provenance.point_mtimes(scoped)
            scoped.sort(
                key=lambda item_id: (mtimes.get(item_id, 0), item_id),
                reverse=order == "mtime_desc",
            )
        return scoped, order, seed if order == "random" else None

    def filter_only_search(
        *,
        query: str,
        author_key: str | None,
        author_keys: list[str] | None = None,
        tag_tokens: list[str],
        point_ids: list[str],
        offset: int,
        limit: int,
        folder_keys: list[str] | None,
        mtime_after: str | None,
        mtime_before: str | None,
        title_query: str = "",
        tag_text: list[str] | None = None,
        image_tags: list[str] | None = None,
        tag_expression: str = "",
        sort: str = "mtime_desc",
        seed: int | None = None,
    ) -> dict[str, Any]:
        scoped = list(point_ids)
        if folder_keys is not None:
            catalog = app.state.engine.folder_catalog()
            scope_ids = catalog.point_ids(
                folder_keys,
                after_ns=day_start_ns(mtime_after),
                before_ns=day_end_ns(mtime_before),
            )
            if scope_ids is not None:
                allowed = set(scope_ids)
                scoped = [item_id for item_id in scoped if item_id in allowed]
        scoped, order, seed = sort_gallery_ids(scoped, sort, seed)
        page_ids = scoped[offset : offset + limit]
        payloads = app.state.engine.store.retrieve_payloads(page_ids)
        results = attach_provenance(
            [
                gallery_result(point_id, payloads[point_id])
                for point_id in page_ids
                if isinstance(payloads.get(point_id), dict)
            ]
        )
        plan = QueryPlan(original=query, semantic_zh="", positive_tags=tag_tokens)
        return {
            "query_plan": dataclasses.asdict(plan),
            "results": results,
            "total": len(scoped),
            "has_more": offset + len(page_ids) < len(scoped),
            "offset": offset,
            "sort": order,
            "seed": seed if order == "random" else None,
            "applied_filters": {
                "author_key": author_key,
                "author_keys": author_keys or [],
                "tags": tag_tokens,
                "title_query": title_query,
                "tag_text": tag_text or [],
                "image_tags": image_tags or [],
            },
        }

    def provenance_budgets() -> list[dict[str, Any]]:
        store: ProvenanceStore = app.state.provenance
        cfg = config.provenance
        return [
            RateLimiter(
                store,
                PIXIV_DOMAIN,
                min_interval=max(0.0, cfg.pixiv_min_interval),
                daily_limit=cfg.pixiv_daily_limit,
                concurrency=max(1, cfg.pixiv_concurrency),
            ).snapshot().as_dict(),
            RateLimiter(
                store,
                TWITTER_DOMAIN,
                min_interval=max(0.0, cfg.twitter_min_interval),
                daily_limit=cfg.twitter_daily_limit,
                concurrency=max(1, cfg.twitter_concurrency),
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
        if record is not None and record.get("status") != "unresolved":
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
        elif record is None:
            record_unresolved(
                store,
                point_id,
                payload,
                path_key=str(payload.get("path_key") or ""),
                path=path,
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
        tag_aliases: dict[str, list[str]] = {}
        for tag in {
            *record.get("tags", []),
            *record.get("manual_tags", []),
            *record.get("auto_tags", []),
        }:
            info = tag_info(str(tag))
            if info is None:
                continue
            values = [info.zh, *info.aliases, info.tag, str(tag)]
            tag_aliases[str(tag)] = list(dict.fromkeys(
                value for value in values if str(value).strip()
            ))
        clean["tag_aliases"] = tag_aliases
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
                daily_limit=cfg.pixiv_daily_limit,
                concurrency=max(1, cfg.pixiv_concurrency),
            ),
            twitter_limiter=RateLimiter(
                store,
                TWITTER_DOMAIN,
                min_interval=max(0.0, cfg.twitter_min_interval),
                daily_limit=cfg.twitter_daily_limit,
                concurrency=max(1, cfg.twitter_concurrency),
            ),
        )

    def start_enrich_job(
        *,
        platforms: list[str],
        limit: int | None = None,
        point_ids: list[str] | None = None,
        reset_stop: bool = True,
        allow_running: bool = False,
        then_auto_tag: bool = False,
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
            if job.running and not allow_running:
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
            if then_auto_tag:
                return start_auto_tag_job(limit=None)
            drain_provenance_refresh()
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
                invalidate_author_cache()
                if then_auto_tag and not job.stop_event.is_set():
                    start_auto_tag_job(limit=None)
                else:
                    drain_provenance_refresh()

        threading.Thread(target=run, daemon=True, name="artseek-provenance").start()
        return True

    def start_auto_tag_job(*, limit: int | None = 500) -> bool:
        store: ProvenanceStore = app.state.provenance
        job: ProvenanceJob = app.state.provenance_job
        point_ids = store.point_ids_without_auto_tags(limit)
        with app.state.provenance_lock:
            if job.running:
                return False
            job.running = True
            job.paused = False
            job.kind = "auto_tag"
            job.done = 0
            job.total = len(point_ids)
            job.message = (
                f"待补 Tag {len(point_ids)} 张"
                if point_ids
                else "没有待自动 Tag 的图片"
            )
            job.error = None
            job.summary = None
            job.stop_event.clear()
            job.pause_event.clear()
        if not point_ids:
            job.running = False
            drain_provenance_refresh()
            return True

        def progress(done: int) -> None:
            job.done = done
            job.message = f"正在用视觉近邻补 Tag {done}/{job.total}"

        def run() -> None:
            summary = {"ok": 0, "skipped": 0, "failed": 0}
            batch_size = 64
            neighbor_limit = 24
            top_tags = 5

            def rank_neighbors(
                point_id: str, neighbors: list[Any]
            ) -> tuple[list[str], dict[str, float]]:
                votes: dict[str, float] = {}
                for neighbor in neighbors:
                    if neighbor.id == point_id:
                        continue
                    for tag in neighbor.payload.get("tags") or []:
                        text = str(tag).strip()
                        if text:
                            votes[text] = votes.get(text, 0.0) + float(neighbor.score)
                ranked = sorted(votes.items(), key=lambda item: item[1], reverse=True)
                top = ranked[:top_tags]
                return (
                    [tag for tag, _ in top],
                    {
                        tag: round(score / max(ranked[0][1], 1e-6), 4)
                        for tag, score in top
                    },
                )

            matrix_data = None
            if len(point_ids) >= 64 and app.state.engine.store.count() <= 200_000:
                job.message = "正在加载视觉向量…"
                try:
                    matrix_data = app.state.engine.store.visual_tag_data()
                except Exception:
                    matrix_data = None

            try:
                for start in range(0, len(point_ids), batch_size):
                    if job.stop_event.is_set():
                        break
                    while job.pause_event.is_set() and not job.stop_event.is_set():
                        time.sleep(0.25)
                    if job.stop_event.is_set():
                        break
                    chunk = point_ids[start : start + batch_size]
                    try:
                        updates: list[tuple[str, list[str], dict[str, float]]] = []
                        if matrix_data is not None:
                            votes = app.state.engine.store.neighbor_tag_votes_from_data(
                                matrix_data,
                                chunk,
                                limit=neighbor_limit,
                                top_tags=top_tags,
                            )
                            for point_id in chunk:
                                ranked = votes.get(point_id)
                                if ranked is None:
                                    summary["skipped"] += 1
                                    continue
                                tags, scores = ranked
                                updates.append((point_id, tags, scores))
                                if tags:
                                    summary["ok"] += 1
                                else:
                                    summary["skipped"] += 1
                        else:
                            vectors = app.state.engine.store.visual_vectors(chunk)
                            query_ids = [point_id for point_id in chunk if point_id in vectors]
                            responses = app.state.engine.store.query_visual_tag_neighbors(
                                [vectors[point_id] for point_id in query_ids],
                                neighbor_limit,
                            )
                            for point_id, neighbors in zip(query_ids, responses, strict=True):
                                tags, scores = rank_neighbors(point_id, neighbors)
                                updates.append((point_id, tags, scores))
                                if tags:
                                    summary["ok"] += 1
                                else:
                                    summary["skipped"] += 1
                            summary["skipped"] += len(chunk) - len(query_ids)
                        app.state.provenance.set_auto_tags_many(updates)
                    except Exception:
                        summary["failed"] += len(chunk)
                    progress(min(start + len(chunk), job.total))
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
                drain_provenance_refresh()

        threading.Thread(target=run, daemon=True, name="artseek-auto-tag").start()
        return True

    def drain_provenance_refresh() -> None:
        if not app.state.provenance_refresh_pending:
            return
        if app.state.provenance_job.running:
            return
        try:
            start_scan_job(deep=False, only_new=True)
        except Exception:
            pass

    def request_provenance_refresh() -> None:
        app.state.provenance_refresh_pending = True
        drain_provenance_refresh()

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
            app.state.provenance_refresh_pending = False
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
                started_next = False
                invalidate_author_cache()
                if not job.stop_event.is_set() and job.error is None:
                    if should_enrich:
                        started_next = start_enrich_job(
                            platforms=["pixiv", "twitter"],
                            reset_stop=False,
                            allow_running=True,
                            then_auto_tag=True,
                        )
                    else:
                        job.running = False
                        job.paused = False
                        started_next = start_auto_tag_job(limit=None)
                if not started_next:
                    job.running = False
                    job.paused = False
                    drain_provenance_refresh()

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

    @app.post("/api/provenance/auto-tag")
    def provenance_auto_tag(request: AutoTagRequest) -> dict[str, Any]:
        started = start_auto_tag_job(limit=request.limit)
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
            invalidate_author_cache()
            return provenance_payload(point_id, payload, path)
        fields: dict[str, Any] = {}
        ref = None
        if request.url:
            ref = parse_source_text(request.url)
            if ref is None:
                raise HTTPException(400, "请输入 Pixiv/X/小红书作品 URL 或任意 http(s) 来源 URL")
        if request.title is not None:
            fields["title"] = request.title.strip() or None
        if request.description is not None:
            fields["description"] = request.description
        if request.tags is not None:
            fields["tags"] = [tag.strip() for tag in request.tags if tag.strip()]
        if request.manual_tags is not None:
            fields["manual_tags"] = [
                tag.strip() for tag in request.manual_tags if tag.strip()
            ]
        author_key = (request.author_key or "").strip()
        author_query = (request.author_query or "").strip()
        if author_key or author_query:
            fields["author_key"] = resolve_author_reference(author_key or author_query)
        elif (
            request.author_name is not None
            or request.author_handle is not None
            or request.pixiv_id is not None
            or request.x_handle is not None
        ):
            name = (request.author_name or "").strip()
            raw_pixiv = (request.pixiv_id or "").strip()
            raw_x = (request.x_handle or request.author_handle or "").strip()
            pixiv_id = normalize_pixiv_user_id(raw_pixiv) if raw_pixiv else ""
            x_handle = normalize_twitter_handle(raw_x) if raw_x else ""
            if raw_pixiv and not pixiv_id:
                raise HTTPException(400, "Pixiv ID 应为数字 ID 或作者主页 URL")
            if raw_x and not re.fullmatch(r"[A-Za-z0-9_]{1,15}", x_handle):
                raise HTTPException(400, "X handle 格式无效")
            key = ""
            if pixiv_id or x_handle:
                base_key = str(record.get("author_key") or "")
                single_platform = ""
                single_value = ""
                if bool(pixiv_id) != bool(x_handle):
                    single_platform = "pixiv" if pixiv_id else "twitter"
                    single_value = pixiv_id or x_handle
                owner = (
                    store.identity_owner(single_platform, single_value)
                    if single_platform
                    else None
                )
                if owner is not None:
                    key = owner
                elif base_key.isdigit() and single_platform:
                    key = store.attach_identity(
                        base_key,
                        single_platform,
                        single_value,
                        name=name or None,
                    )
                else:
                    key = store.ensure_linked_author(
                        name=name or None,
                        pixiv_id=pixiv_id or None,
                        x_handle=x_handle or None,
                    )
            elif name:
                key = store.ensure_manual_author(name)
            if key:
                fields["author_key"] = key
        should_fetch = ref is not None and (
            ref.url != str(record.get("url") or "")
            or str(record.get("status") or "") == "pending"
        )
        if should_fetch:
            store.set_manual(point_id, fields, ref=ref)
            record = store.get(point_id)
            if record is not None and record.get("platform") in {"pixiv", "twitter"}:
                domain = PIXIV_DOMAIN if record["platform"] == "pixiv" else TWITTER_DOMAIN
                state = store.domain_state(domain)
                if int(state.get("blocked_until") or 0) > int(time.time()):
                    invalidate_author_cache()
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
        invalidate_author_cache()
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
        if record is not None and ref.platform in {"pixiv", "twitter"}:
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
        invalidate_author_cache()
        return provenance_payload(point_id, payload, path)

    def suggestion_scope_ids(
        folder_keys: list[str] | None,
        mtime_after: str | None,
        mtime_before: str | None,
    ) -> list[str] | None:
        if folder_keys is None:
            return None
        if not folder_keys:
            return []
        catalog = app.state.engine.folder_catalog()
        ids = catalog.point_ids(
            folder_keys,
            after_ns=day_start_ns(mtime_after),
            before_ns=day_end_ns(mtime_before),
        )
        if ids is None:
            ids = [point_id for acc in catalog.folders.values() for point_id in acc.ids]
        return ids

    def authors_payload(
        q: str,
        limit: int,
        offset: int,
        folder_keys: list[str] | None = None,
        mtime_after: str | None = None,
        mtime_before: str | None = None,
    ) -> dict[str, Any]:
        page_size = max(1, min(int(limit), 500))
        start = max(0, int(offset))
        scope_ids = suggestion_scope_ids(folder_keys, mtime_after, mtime_before)
        counts = None
        if scope_ids is not None:
            cache_key = (
                tuple(sorted(folder_keys or [])),
                mtime_after or "",
                mtime_before or "",
            )
            now = time.monotonic()
            with app.state.author_scope_cache_lock:
                cached = app.state.author_scope_cache.get(cache_key)
                if cached is not None and now - cached[0] <= 20.0:
                    counts = cached[1]
            if counts is None:
                counts = app.state.provenance.author_counts_for_points(scope_ids)
                with app.state.author_scope_cache_lock:
                    app.state.author_scope_cache[cache_key] = (now, counts)
                    while len(app.state.author_scope_cache) > 16:
                        app.state.author_scope_cache.popitem(last=False)
        unassigned_count = (
            int(counts.get(UNASSIGNED_AUTHOR_KEY, 0))
            if counts is not None
            else app.state.provenance.unassigned_count()
        )
        if counts is not None:
            counts = {
                key: value
                for key, value in counts.items()
                if key != UNASSIGNED_AUTHOR_KEY
            }
        authors = app.state.provenance.list_authors(
            q,
            limit=page_size + 1,
            offset=start,
            image_counts=counts,
        )
        return {
            "authors": authors[:page_size],
            "has_more": len(authors) > page_size,
            "offset": start,
            "unassigned_count": unassigned_count,
        }

    @app.get("/api/authors")
    def list_authors(q: str = "", limit: int = 40, offset: int = 0) -> dict[str, Any]:
        return authors_payload(q, limit, offset)

    @app.post("/api/authors")
    def list_authors_post(request: SuggestionRequest) -> dict[str, Any]:
        return authors_payload(
            request.q,
            request.limit,
            request.offset,
            request.folder_keys,
            request.mtime_after,
            request.mtime_before,
        )

    @app.get("/api/authors/{author_key}")
    def get_author(author_key: str) -> dict[str, Any]:
        summary = app.state.provenance.author_summary(author_key)
        if summary is None:
            raise HTTPException(404, "作者不存在")
        ids = app.state.provenance.point_ids_for_author(author_key)
        return {"author": summary, "point_ids": ids, "total": len(ids)}

    @app.post("/api/authors/{author_key}")
    def get_scoped_author(
        author_key: str, request: SuggestionRequest
    ) -> dict[str, Any]:
        summary = app.state.provenance.author_summary(author_key)
        if summary is None:
            raise HTTPException(404, "作者不存在")
        ids = app.state.provenance.point_ids_for_author(author_key)
        scope_ids = suggestion_scope_ids(
            request.folder_keys, request.mtime_after, request.mtime_before
        )
        if scope_ids is not None:
            allowed = set(scope_ids)
            ids = [point_id for point_id in ids if point_id in allowed]
            summary = dict(summary)
            summary["image_count"] = len(ids)
        return {"author": summary, "point_ids": ids, "total": len(ids)}

    @app.get("/api/authors/{author_key}/images")
    def author_images(
        author_key: str,
        limit: int = 40,
        offset: int = Query(default=0, ge=0, le=50_000),
        sort: str = "mtime_desc",
        seed: int | None = None,
    ) -> dict[str, Any]:
        store: ProvenanceStore = app.state.provenance
        summary = store.author_summary(author_key)
        if summary is None:
            raise HTTPException(404, "作者不存在")
        ids = store.point_ids_for_author(author_key)
        ids, order, seed = sort_gallery_ids(ids, sort, seed)
        total = len(ids)
        page_size = max(1, min(int(limit), 200))
        page_ids = ids[offset : offset + page_size]
        payloads = retrieve_collection_payloads(page_ids)
        results = attach_provenance(
            [
                gallery_result(point_id, payloads[point_id])
                for point_id in page_ids
                if isinstance(payloads.get(point_id), dict)
            ]
        )
        return {
            "author": summary,
            "offset": offset,
            "limit": page_size,
            "sort": order,
            "seed": seed,
            "total": total,
            "has_more": offset + len(page_ids) < total,
            "results": results,
        }

    def resolve_artist_scope(request: ArtistScopeRequest) -> set[str] | None:
        """Resolve the current UI filter context into one point-ID universe."""
        scope: set[str] | None = None

        def intersect(values: Sequence[str]) -> None:
            nonlocal scope
            cleaned = {str(value) for value in values if value}
            scope = cleaned if scope is None else scope & cleaned

        if request.snapshot_id:
            if not is_snapshot_id(request.snapshot_id):
                raise HTTPException(400, "检索快照编号不合法")
            snapshot = app.state.search_snapshots.get(request.snapshot_id)
            if snapshot is None:
                raise HTTPException(404, "检索快照不存在或已过期")
            intersect([str(item.get("id") or "") for item in snapshot.get("results") or []])
        if request.folder_keys is not None:
            catalog = app.state.engine.folder_catalog()
            ids = catalog.point_ids(
                request.folder_keys,
                after_ns=day_start_ns(request.mtime_after),
                before_ns=day_end_ns(request.mtime_before),
            )
            if ids is None:
                ids = [point_id for acc in catalog.folders.values() for point_id in acc.ids]
            intersect(ids)
        if request.collection_id:
            collection = require_collection(request.collection_id)
            intersect(collection.item_ids)
        filters_present = bool(
            request.author_keys
            or request.title_query.strip()
            or request.text_tags
            or request.tag_expression.strip()
        )
        if filters_present:
            authors = [
                resolve_author_token(value)
                for value in request.author_keys
                if str(value or "").strip()
            ]
            restrict = restricted_ids(
                author_keys=authors,
                tag_tokens=request.text_tags,
                title_query=request.title_query,
                tag_expression=request.tag_expression,
            )
            intersect(restrict or [])
        return scope

    def artist_collection_order(request: ArtistScopeRequest) -> tuple[str, ...] | None:
        if not request.collection_id:
            return None
        method = str(
            getattr(request, "image_sort", "") or getattr(request, "sort", "")
        )
        if method not in {"manual", "added_asc", "added_desc"}:
            return None
        collection = require_collection(request.collection_id)
        if method == "manual":
            return collection.ordered_item_ids()
        if method == "added_desc":
            return tuple(reversed(collection.item_ids))
        return collection.item_ids

    def scoped_author_images(
        author_key: str,
        scope: set[str] | None,
        *,
        sort: str,
        seed: int | None,
        offset: int,
        limit: int,
        order_ids: Sequence[str] | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]], str, int | None, int, bool]:
        store: ProvenanceStore = app.state.provenance
        summary = store.author_summary(author_key)
        if summary is None:
            raise HTTPException(404, "作者不存在")
        ids = store.point_ids_for_author(author_key)
        if scope is not None:
            ids = [point_id for point_id in ids if point_id in scope]
        if order_ids is not None:
            rank = {point_id: index for index, point_id in enumerate(order_ids)}
            ids.sort(key=lambda point_id: rank.get(point_id, len(rank)))
            order, used_seed = sort, None
        else:
            ids, order, used_seed = sort_gallery_ids(ids, sort, seed)
        total = len(ids)
        page_size = max(1, min(int(limit), 200))
        page_ids = ids[offset : offset + page_size]
        payloads = retrieve_collection_payloads(page_ids)
        results = attach_provenance(
            [
                gallery_result(point_id, payloads[point_id])
                for point_id in page_ids
                if isinstance(payloads.get(point_id), dict)
            ]
        )
        return (
            summary,
            results,
            order,
            used_seed,
            total,
            offset + len(page_ids) < total,
        )

    artist_sort_keys = (
        "name_asc",
        "name_desc",
        "latest_asc",
        "latest_desc",
        "count_asc",
        "count_desc",
    )

    def artist_catalog_for(
        request: ArtistGalleryRequest,
        scope: set[str] | None,
        sort: str,
    ) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
        """Return one cached artist catalog; sort orders fill lazily/in background."""
        store: ProvenanceStore = app.state.provenance
        cache_key = (
            tuple(sorted(request.folder_keys)) if request.folder_keys is not None else None,
            request.mtime_after or "",
            request.mtime_before or "",
            tuple(sorted(request.author_keys)),
            request.title_query,
            tuple(request.text_tags),
            request.tag_expression,
            request.collection_id,
            request.snapshot_id,
        )
        now = time.monotonic()
        with app.state.author_scope_cache_lock:
            cached = app.state.artist_catalog_cache.get(cache_key)
            if cached is not None and now - cached[0] <= 120.0:
                app.state.artist_catalog_cache.move_to_end(cache_key)
                return cached[1], cached[2]
        entries = store.artist_entries(list(scope) if scope is not None else None)
        orders = {
            sort: store.sort_artist_entries(entries, sort, include_unassigned=True)
        }
        with app.state.author_scope_cache_lock:
            app.state.artist_catalog_cache[cache_key] = (now, entries, orders)
            while len(app.state.artist_catalog_cache) > 8:
                app.state.artist_catalog_cache.popitem(last=False)

        def warm_orders() -> None:
            for other in artist_sort_keys:
                if other == sort:
                    continue
                ordered = store.sort_artist_entries(
                    entries, other, include_unassigned=True
                )
                with app.state.author_scope_cache_lock:
                    cached = app.state.artist_catalog_cache.get(cache_key)
                    if cached is None or cached[1] is not entries:
                        return
                    cached[2].setdefault(other, ordered)

        threading.Thread(
            target=warm_orders, daemon=True, name="artseek-artist-sort-warm"
        ).start()
        return entries, orders

    def public_artist_entry(entry: dict[str, Any]) -> dict[str, Any]:
        clean = dict(entry)
        clean.pop("_unassigned", None)
        clean.pop("_search_values", None)
        clean.pop("_search_text", None)
        return clean

    @app.post("/api/artists/gallery")
    def artist_gallery(request: ArtistGalleryRequest) -> dict[str, Any]:
        store: ProvenanceStore = app.state.provenance
        scope = resolve_artist_scope(request)
        sort = request.sort if request.sort in artist_sort_keys else "latest_desc"
        entries, orders = artist_catalog_for(request, scope, sort)
        if request.q.strip():
            ranked = store.search_artist_entries(
                entries,
                request.q,
                sort,
                include_unassigned=request.include_unassigned,
            )
        else:
            ranked = orders.get(sort) or store.sort_artist_entries(
                entries, sort, include_unassigned=True
            )
            if not request.include_unassigned:
                ranked = [
                    item for item in ranked if item["key"] != UNASSIGNED_AUTHOR_KEY
                ]
        total = len(ranked)
        page = {
            "authors": [
                public_artist_entry(item)
                for item in ranked[
                    max(0, request.offset) : max(0, request.offset) + request.limit
                ]
            ],
            "total": total,
        }
        image_sort = {
            "name": "name_asc",
            "count": "mtime_desc",
            "count_asc": "mtime_desc",
            "mtime": "mtime_desc",
        }.get(request.image_sort, request.image_sort)
        if image_sort not in {
            "mtime_desc",
            "mtime_asc",
            "name_asc",
            "name_desc",
            "random",
            "manual",
            "added_asc",
            "added_desc",
        }:
            image_sort = "mtime_desc"
        order_ids = artist_collection_order(request)
        if order_ids is not None and image_sort not in {"manual", "added_asc", "added_desc"}:
            order_ids = None
        base_seed = request.image_seed
        if image_sort == "random" and base_seed is None:
            base_seed = random.SystemRandom().randrange(1 << 53)
        entries = [dict(item) for item in page["authors"]]
        entry_keys = [str(item["key"]) for item in entries]
        id_map = store.point_ids_for_authors(entry_keys)
        explicit_covers = store.artist_cover_ids(entry_keys)
        rank = (
            {point_id: index for index, point_id in enumerate(order_ids)}
            if order_ids is not None
            else None
        )
        previews: dict[str, list[str]] = {}
        totals: dict[str, int] = {}
        seeds: dict[str, int | None] = {}
        all_ids: list[str] = []
        for entry in entries:
            key = str(entry["key"])
            row_seed = base_seed
            if image_sort == "random" and base_seed is not None:
                row_seed = random.Random(f"{base_seed}:{key}").randrange(1 << 53)
            ids = id_map.get(key, [])
            if scope is not None:
                ids = [point_id for point_id in ids if point_id in scope]
            if rank is not None:
                ids.sort(key=lambda point_id: rank.get(point_id, len(rank)))
                used_seed = None
            else:
                ids, _, used_seed = sort_gallery_ids(ids, image_sort, row_seed)
            previews[key] = ids[: request.image_limit]
            totals[key] = len(ids)
            seeds[key] = used_seed
            all_ids.extend(previews[key])
            if explicit_covers.get(key):
                all_ids.append(explicit_covers[key])
        payloads = retrieve_collection_payloads(list(dict.fromkeys(all_ids)))
        result_rows = attach_provenance(
            [
                gallery_result(point_id, payloads[point_id])
                for point_id in dict.fromkeys(all_ids)
                if isinstance(payloads.get(point_id), dict)
            ]
        )
        by_id = {str(item["id"]): item for item in result_rows}
        for entry in entries:
            key = str(entry["key"])
            images = [by_id[point_id] for point_id in previews[key] if point_id in by_id]
            entry["images"] = images
            entry["images_has_more"] = totals[key] > len(previews[key])
            entry["images_offset"] = len(images)
            entry["images_total"] = totals[key]
            entry["image_seed"] = seeds[key]
            cover_id = explicit_covers.get(key) or (previews[key][0] if previews[key] else "")
            cover_payload = payloads.get(cover_id) if cover_id else None
            entry["cover"] = (
                {
                    "id": cover_id,
                    "name": cover_payload.get("name"),
                    "width": cover_payload.get("width"),
                    "height": cover_payload.get("height"),
                }
                if isinstance(cover_payload, dict)
                else None
            )
            entry["cover_explicit"] = bool(
                explicit_covers.get(key) and cover_id == explicit_covers.get(key)
            )
            members = entry.get("members") or []
            entry["pixiv_ids"] = [
                str(member.get("platform_id") or "")
                for member in members
                if member.get("platform") == "pixiv"
            ]
            entry["x_handles"] = [
                str(member.get("handle") or "")
                for member in members
                if member.get("platform") == "twitter"
            ]
        authors = entries
        page["authors"] = authors
        page["image_sort"] = image_sort
        page["image_seed"] = base_seed
        return page

    @app.post("/api/artists/{author_key}/images")
    def scoped_artist_images(
        author_key: str, request: ArtistImagesRequest
    ) -> dict[str, Any]:
        scope = resolve_artist_scope(request)
        sort = {
            "name": "name_asc",
            "count": "mtime_desc",
            "count_asc": "mtime_desc",
            "mtime": "mtime_desc",
        }.get(request.sort, request.sort)
        if sort not in {
            "mtime_desc",
            "mtime_asc",
            "name_asc",
            "name_desc",
            "random",
            "manual",
            "added_asc",
            "added_desc",
        }:
            sort = "mtime_desc"
        order_ids = artist_collection_order(request)
        if order_ids is not None and sort not in {"manual", "added_asc", "added_desc"}:
            order_ids = None
        summary, results, order, seed, total, has_more = scoped_author_images(
            author_key,
            scope,
            sort=sort,
            seed=request.seed,
            offset=request.offset,
            limit=request.limit,
            order_ids=order_ids,
        )
        return {
            "author": summary,
            "offset": request.offset,
            "limit": request.limit,
            "sort": order,
            "seed": seed,
            "total": total,
            "has_more": has_more,
            "results": results,
        }

    @app.post("/api/authors/{author_key}/identities")
    def add_author_identity(
        author_key: str, request: ArtistIdentityRequest
    ) -> dict[str, Any]:
        store: ProvenanceStore = app.state.provenance
        if author_key.casefold() == UNASSIGNED_AUTHOR_KEY:
            raise HTTPException(400, "无画师不能添加身份")
        kind = request.kind.strip().casefold()
        try:
            if kind == "link":
                if not normalize_social_link(request.value):
                    raise ValueError("请输入有效的 http(s) 链接")
                store.add_link(author_key, request.value, label=request.label)
            elif kind in {"alias", "name_alias"}:
                store.add_alias(author_key, request.value)
            elif kind in {"pixiv", "pixiv_id"}:
                store.attach_identity(author_key, "pixiv", request.value)
            elif kind in {"twitter", "x", "x_id"}:
                store.attach_identity(author_key, "twitter", request.value)
            elif kind in {"xiaohongshu", "xhs", "rednote"}:
                if not normalize_xiaohongshu_user_id(request.value):
                    raise ValueError("小红书主页链接或 UID 无效")
                store.attach_identity(author_key, "xiaohongshu", request.value)
            else:
                raise HTTPException(400, "不支持的身份类型")
        except ValueError as exc:
            status = 409 if "已属于" in str(exc) else 400
            raise HTTPException(status, str(exc)) from exc
        invalidate_author_cache()
        summary = store.author_summary(author_key)
        if summary is None:
            raise HTTPException(404, "作者不存在")
        return {"author": summary}

    @app.post("/api/authors/{author_key}/name")
    def rename_author(author_key: str, request: ArtistNameRequest) -> dict[str, Any]:
        store: ProvenanceStore = app.state.provenance
        try:
            store.set_author_name(author_key, request.name)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        invalidate_author_cache()
        summary = store.author_summary(author_key)
        if summary is None:
            raise HTTPException(404, "作者不存在")
        return {"author": summary}

    @app.post("/api/authors/{author_key}/identities/remove")
    def remove_author_identity(
        author_key: str, request: ArtistIdentityRemoveRequest
    ) -> dict[str, Any]:
        store: ProvenanceStore = app.state.provenance
        try:
            store.remove_identity(author_key, request.kind, request.value)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        invalidate_author_cache()
        summary = store.author_summary(author_key)
        if summary is None:
            raise HTTPException(404, "作者不存在")
        return {"author": summary}

    @app.post("/api/authors/{author_key}/members")
    def set_author_members(
        author_key: str, request: ArtistMembersRequest
    ) -> dict[str, Any]:
        """Reorder an artist's social accounts and toggle which ones are shown."""
        store: ProvenanceStore = app.state.provenance
        try:
            store.set_member_preferences(
                author_key, [item.model_dump() for item in request.members]
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        invalidate_author_cache()
        summary = store.author_summary(author_key)
        if summary is None:
            raise HTTPException(404, "作者不存在")
        return {"author": summary}

    @app.post("/api/authors/{author_key}/cover")
    def set_artist_cover(
        author_key: str, request: ArtistCoverRequest
    ) -> dict[str, Any]:
        store: ProvenanceStore = app.state.provenance
        try:
            store.set_artist_cover(author_key, request.point_id)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        invalidate_author_cache()
        summary = store.author_summary(author_key)
        if summary is None:
            raise HTTPException(404, "作者不存在")
        return {"author": summary}

    @app.post("/api/authors/{author_key}/merge")
    def merge_author(author_key: str, request: ArtistMergeRequest) -> dict[str, Any]:
        store: ProvenanceStore = app.state.provenance
        target = author_key.strip().lstrip("@")
        if target.casefold() == UNASSIGNED_AUTHOR_KEY:
            raise HTTPException(400, "无画师不能合并")
        target_summary = store.author_summary(target)
        if target_summary is None:
            raise HTTPException(404, "作者不存在")
        token = request.source.strip().lstrip("@")
        if not token:
            raise HTTPException(400, "请输入要合并的 UID、Pixiv ID 或 X handle")
        if token.casefold() in {UNASSIGNED_AUTHOR_KEY, "无画师", "未标注", "作者未标注", "无作者"}:
            raise HTTPException(400, "无画师不能合并")
        source_key = ""
        if token.isdigit():
            summary = store.author_summary(token)
            if summary is not None:
                source_key = str(summary["key"])
        if not source_key:
            owner = store.identity_owner("pixiv", token)
            if owner:
                source_key = owner
        if not source_key:
            owner = store.identity_owner("twitter", token)
            if owner:
                source_key = owner
        if not source_key:
            owner = store.identity_owner("xiaohongshu", token)
            if owner:
                source_key = owner
        if not source_key:
            matches = store.list_authors(token, limit=5)
            grouped: dict[str, list[dict[str, Any]]] = {}
            for item in matches:
                root = store.canonical_author_key(str(item["key"]))
                grouped.setdefault(root, []).append(item)
            if len(grouped) == 1:
                source_key = next(iter(grouped))
            elif len(grouped) > 1:
                raise HTTPException(
                    409,
                    detail={
                        "message": "作者不唯一，请选择完整 UID",
                        "author_candidates": [
                            {
                                "key": root,
                                "name": members[0].get("name")
                                or members[0].get("handle")
                                or root,
                            }
                            for root, members in grouped.items()
                        ],
                    },
                )
        if not source_key:
            raise HTTPException(404, "没有找到要合并的作者")
        source_summary = store.author_summary(source_key)
        if source_summary is None:
            raise HTTPException(404, "没有找到要合并的作者")
        if str(source_summary["key"]) == str(target_summary["key"]):
            return {"author": target_summary, "merged_from": None}
        store.merge_authors(source_key, target)
        invalidate_author_cache()
        return {
            "author": store.author_summary(target),
            "merged_from": source_summary,
        }

    def image_tag_suggestions() -> list[dict[str, Any]]:
        generation = int(app.state.collection_index_generation)
        cache = app.state.image_tag_suggestions
        with app.state.tag_suggestion_lock:
            if cache.get("generation") == generation:
                return list(cache.get("items") or [])
            counts: dict[str, int] = {}
            for _, payload in app.state.engine.store.iter_payloads():
                for tag in payload.get("tags") or []:
                    text = str(tag or "").strip()
                    if text:
                        counts[text] = counts.get(text, 0) + 1
            items = [
                {"tag": tag, "count": count}
                for tag, count in sorted(
                    counts.items(), key=lambda item: (-item[1], item[0].casefold())
                )
            ]
            app.state.image_tag_suggestions = {"generation": generation, "items": items}
            return list(items)

    def group_source_tag_items(
        items: list[dict[str, Any]], query: str = "", scope_ids: list[str] | None = None
    ) -> list[dict[str, Any]]:
        groups: dict[str, dict[str, Any]] = {}
        for item in items:
            tag = str(item.get("tag") or "").strip()
            if not tag:
                continue
            info = tag_info(tag)
            canonical = info.tag if info is not None else tag
            group = groups.setdefault(
                canonical,
                {"tag": canonical, "count": 0, "display_zh": "", "aliases": set()},
            )
            group["count"] += int(item.get("count") or 0)
            if info is not None:
                group["display_zh"] = group["display_zh"] or info.zh
                group["aliases"].update(info.aliases)
            group["aliases"].add(tag)
        result: list[dict[str, Any]] = []
        for group in groups.values():
            aliases = sorted(
                {alias for alias in group["aliases"] if alias and alias != group["tag"]},
                key=lambda value: (len(value), value.casefold()),
            )
            result.append({
                "tag": group["tag"],
                "count": group["count"],
                "display_zh": group["display_zh"],
                "aliases": aliases,
            })
        if not result and str(query or "").strip():
            seen = {item["tag"].casefold() for item in result}
            scope = set(scope_ids) if scope_ids is not None else None
            for entry in search_tags(query, limit=20):
                if entry.tag.casefold() in seen:
                    continue
                points = set(app.state.provenance.point_ids_for_tags([entry.tag]))
                if scope is not None:
                    points &= scope
                result.append({
                    "tag": entry.tag,
                    "count": len(points),
                    "display_zh": entry.zh,
                    "aliases": list(entry.aliases),
                })
        result.sort(key=lambda item: (-int(item["count"]), item["tag"].casefold()))
        return result

    def decorate_tag_items(
        items: list[dict[str, Any]], query: str = "", scope_ids: list[str] | None = None
    ) -> list[dict[str, Any]]:
        decorated: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in items:
            tag = str(item.get("tag") or "")
            if not tag:
                continue
            info = tag_info(tag)
            enriched = dict(item)
            if info is not None:
                enriched["display_zh"] = info.zh
                enriched["aliases"] = list(info.aliases)
            decorated.append(enriched)
            seen.add(tag.casefold())
        if str(query or "").strip():
            scope = set(scope_ids) if scope_ids is not None else None
            for entry in search_tags(query, limit=20):
                if entry.tag.casefold() in seen:
                    continue
                points = set(app.state.provenance.point_ids_for_tags([entry.tag]))
                if scope is not None:
                    points &= scope
                decorated.append({
                    "tag": entry.tag,
                    "count": len(points),
                    "display_zh": entry.zh,
                    "aliases": list(entry.aliases),
                })
        return decorated

    def tags_payload(
        q: str,
        kind: str,
        limit: int,
        offset: int,
        folder_keys: list[str] | None = None,
        mtime_after: str | None = None,
        mtime_before: str | None = None,
    ) -> dict[str, Any]:
        page_size = max(1, min(int(limit), 500))
        start = max(0, int(offset))
        if kind == "image":
            needle = str(q or "").strip().casefold()
            items = [
                item
                for item in image_tag_suggestions()
                if not needle or needle in item["tag"].casefold()
            ]
            page = items[start : start + page_size + 1]
            has_more = len(page) > page_size
        elif folder_keys is not None:
            ids = suggestion_scope_ids(folder_keys, mtime_after, mtime_before) or []
            raw_page = app.state.provenance.source_tag_counts_for_points(
                ids, q, limit=page_size + 1, offset=start
            )
            has_more = len(raw_page) > page_size
            page = group_source_tag_items(raw_page, q, ids)
        else:
            raw_page = app.state.provenance.list_source_tags(
                q, limit=page_size + 1, offset=start
            )
            has_more = len(raw_page) > page_size
            page = group_source_tag_items(raw_page, q)
        return {
            "kind": kind,
            "tags": page[:page_size],
            "has_more": has_more,
            "offset": start,
        }

    @app.get("/api/tags")
    def tag_suggestions(
        q: str = "",
        kind: str = "source",
        limit: int = 60,
        offset: int = 0,
    ) -> dict[str, Any]:
        return tags_payload(q, kind, limit, offset)

    @app.post("/api/tags")
    def tag_suggestions_post(request: SuggestionRequest) -> dict[str, Any]:
        return tags_payload(
            request.q,
            request.kind,
            request.limit,
            request.offset,
            request.folder_keys,
            request.mtime_after,
            request.mtime_before,
        )

    @app.get("/api/tag-dictionary")
    def tag_dictionary(q: str = "", limit: int = 40) -> dict[str, Any]:
        return {
            "tags": [
                {
                    "tag": entry.tag,
                    "display_zh": entry.zh,
                    "aliases": list(entry.aliases),
                }
                for entry in search_tags(q, limit=limit)
            ]
        }

    @app.get("/api/tags/{tag}/images")
    def tag_images(
        tag: str,
        limit: int = 40,
        offset: int = Query(default=0, ge=0, le=50_000),
        sort: str = "mtime_desc",
        seed: int | None = None,
    ) -> dict[str, Any]:
        store: ProvenanceStore = app.state.provenance
        clean = str(tag or "").strip()
        if not clean:
            raise HTTPException(400, "标签不能为空")
        ids = store.point_ids_for_tags([clean])
        ids, order, seed = sort_gallery_ids(ids, sort, seed)
        total = len(ids)
        page_size = max(1, min(int(limit), 200))
        page_ids = ids[offset : offset + page_size]
        payloads = retrieve_collection_payloads(page_ids)
        results = attach_provenance(
            [
                gallery_result(point_id, payloads[point_id])
                for point_id in page_ids
                if isinstance(payloads.get(point_id), dict)
            ]
        )
        return {
            "tag": clean,
            "offset": offset,
            "limit": page_size,
            "sort": order,
            "seed": seed,
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
                    request_provenance_refresh()
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
