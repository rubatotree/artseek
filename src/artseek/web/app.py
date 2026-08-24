from __future__ import annotations

import dataclasses
import json
import random
import tempfile
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..collections import Collection, CollectionStore
from ..config import (
    DEFAULT_WEB_SEARCH_LIMIT,
    MAX_WEB_SEARCH_LIMIT,
    MIN_WEB_SEARCH_LIMIT,
    WEB_SEARCH_LIMIT_STEP,
    AppConfig,
    WebSearchSettings,
    load_web_search_settings,
    save_web_search_settings,
)
from ..image_io import canonical_path, make_thumbnail
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
from ..search import SearchEngine


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


class RandomBrowseRequest(BaseModel):
    """A stable random ordering for one paginated gallery session."""

    folder_keys: list[str] | None = None
    mtime_after: str | None = None
    mtime_before: str | None = None
    seed: int | None = None
    limit: int = Field(default=40, ge=1, le=200)
    offset: int = Field(default=0, ge=0, le=50_000)


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
        if not name:
            raise ValueError("收藏夹名称不能为空")
        if len(name) > 80:
            raise ValueError("收藏夹名称不能超过 80 个字符")
        return name


class CollectionUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = None

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        name = value.strip()
        if not name:
            raise ValueError("收藏夹名称不能为空")
        if len(name) > 80:
            raise ValueError("收藏夹名称不能超过 80 个字符")
        return name

    @model_validator(mode="after")
    def validate_update(self) -> CollectionUpdateRequest:
        if self.name is None:
            raise ValueError("请提供新的收藏夹名称")
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
    live_item_ids: list[str] = field(default_factory=list)
    source_cursor: int = 0
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
    app.state.collection_index_empty = False
    app.state.collection_index_generation = 0
    app.state.collection_id_cursors: OrderedDict[str, CollectionIdCursor] = OrderedDict()
    app.state.collection_id_cursors_lock = threading.RLock()
    static_path = Path(__file__).with_name("static") / "index.html"

    @app.on_event("shutdown")
    def shutdown() -> None:
        app.state.engine.close()

    @app.get("/", response_class=HTMLResponse)
    def home() -> HTMLResponse:
        return HTMLResponse(
            static_path.read_text(encoding="utf-8"),
            headers={"Cache-Control": "no-store"},
        )

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

    def invalidate_collection_id_cursor(collection_id: str) -> None:
        with app.state.collection_id_cursors_lock:
            app.state.collection_id_cursors.pop(collection_id, None)

    def collection_id_cursor(collection: Collection) -> CollectionIdCursor:
        """Return one LRU cursor whose data fits the public pagination window."""
        membership_token = hash(collection.item_ids)
        with app.state.collection_id_cursors_lock:
            cursor = app.state.collection_id_cursors.get(collection.id)
            generation = app.state.collection_index_generation
            if (
                cursor is None
                or cursor.membership_token != membership_token
                or cursor.index_generation != generation
            ):
                cursor = CollectionIdCursor(
                    membership_token=membership_token,
                    index_generation=generation,
                )
                app.state.collection_id_cursors[collection.id] = cursor
            app.state.collection_id_cursors.move_to_end(collection.id)
            while len(app.state.collection_id_cursors) > MAX_CACHED_COLLECTIONS:
                app.state.collection_id_cursors.popitem(last=False)
            return cursor

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
                return {}
            raise

    def materialize_unfiltered_collection_page(
        collection: Collection,
        start: int,
        page_size: int,
    ) -> tuple[list[tuple[str, dict[str, Any]]], int | None, bool]:
        """Read an ordered page with a bounded ID cursor, never cached payloads."""
        cursor = collection_id_cursor(collection)
        item_ids = collection.item_ids
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

    @app.get("/api/collections")
    def list_collections(image_id: str | None = None) -> dict[str, list[dict[str, Any]]]:
        collections = app.state.collections.list()
        summaries: list[dict[str, Any]] = []
        for collection in collections:
            summary = collection_summary(collection)
            if image_id:
                summary["contains"] = image_id in collection.item_ids
            summaries.append(summary)
        return {"collections": summaries}

    @app.post("/api/collections", status_code=201)
    def create_collection(request: CollectionNameRequest) -> dict[str, Any]:
        collection = app.state.collections.create(request.name)
        return {"collection": collection_summary(collection)}

    @app.put("/api/collections/order")
    def reorder_collections(request: CollectionOrderRequest) -> dict[str, list[dict[str, Any]]]:
        collections = app.state.collections.reorder(request.collection_ids)
        if collections is None:
            raise HTTPException(409, "收藏夹已更新，请刷新后重试")
        return {"collections": [collection_summary(collection) for collection in collections]}

    @app.patch("/api/collections/{collection_id}")
    def update_collection(collection_id: str, request: CollectionUpdateRequest) -> dict[str, Any]:
        collection = app.state.collections.update(
            collection_id,
            name=request.name,
        )
        if collection is None:
            raise HTTPException(404, "收藏夹不存在")
        return {"collection": collection_summary(collection)}

    @app.delete("/api/collections/{collection_id}")
    def delete_collection(collection_id: str) -> dict[str, Any]:
        collection = app.state.collections.delete(collection_id)
        if collection is None:
            raise HTTPException(404, "收藏夹不存在")
        invalidate_collection_id_cursor(collection_id)
        return {"deleted": True, "collection": collection_summary(collection)}

    @app.post("/api/collections/{collection_id}/items")
    def add_collection_item(collection_id: str, request: CollectionItemRequest) -> dict[str, Any]:
        require_collection(collection_id)
        if not app.state.engine.store.get(request.image_id):
            raise HTTPException(404, "图片不在索引中，无法收藏")
        collection, added = app.state.collections.add_item(collection_id, request.image_id)
        if collection is None:
            raise HTTPException(404, "收藏夹不存在")
        if added:
            invalidate_collection_id_cursor(collection_id)
        return {"collection": collection_summary(collection), "added": added}

    @app.delete("/api/collections/{collection_id}/items/{point_id}")
    def remove_collection_item(collection_id: str, point_id: str) -> dict[str, Any]:
        collection, removed = app.state.collections.remove_item(collection_id, point_id)
        if collection is None:
            raise HTTPException(404, "收藏夹不存在")
        if removed:
            invalidate_collection_id_cursor(collection_id)
        return {"collection": collection_summary(collection), "removed": removed}

    @app.get("/api/collections/{collection_id}/images")
    def collection_images(
        collection_id: str,
        query: str = "",
        limit: int = 40,
        offset: int = Query(default=0, ge=0, le=10_000),
    ) -> dict[str, Any]:
        collection = require_collection(collection_id)
        needle = query.strip().casefold()
        page_size = max(1, min(limit, 200))
        start = offset

        def matches(payload: dict[str, Any]) -> bool:
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

        if needle:
            filtered: list[tuple[str, dict[str, Any]]] = []
            for batch_start in range(0, len(collection.item_ids), 256):
                batch = list(collection.item_ids[batch_start : batch_start + 256])
                payloads = retrieve_collection_payloads(batch)
                filtered.extend(
                    (point_id, payloads[point_id])
                    for point_id in batch
                    if point_id in payloads
                    and isinstance(payloads[point_id], dict)
                    and matches(payloads[point_id])
                )
            page = filtered[start : start + page_size]
            total = len(filtered)
            has_more = start + len(page) < total
            total_exact = True
        else:
            page, total, has_more = materialize_unfiltered_collection_page(
                collection,
                start,
                page_size,
            )
            total_exact = total is not None
        return {
            "collection": collection_summary(collection),
            "query": query.strip(),
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

    @app.post("/api/search")
    def search(request: TextSearchRequest) -> dict:
        if app.state.index_job.running:
            raise HTTPException(409, "正在索引图片，请稍后再检索")
        specs = list(request.references)
        if request.reference_id:
            specs.append(ReferenceSpec(id=request.reference_id, weight=1.0))
        paths, weights = resolve_library_references(specs)
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
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return dataclasses.asdict(response)

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
    ) -> dict:
        if app.state.index_job.running:
            raise HTTPException(409, "正在索引图片，请稍后再检索")
        library_paths, library_weights = resolve_library_references(
            parse_reference_specs(references)
        )
        upload_weights = _parse_weight_list(file_weights, len(files))
        temps: list[Path] = []
        try:
            for file in files:
                suffix = Path(file.filename or "query.png").suffix or ".png"
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
                    temps.append(Path(handle.name))
                    handle.write(await file.read())
            response = app.state.engine.search(
                query,
                images=library_paths + temps,
                text_weight=max(0.0, min(text_weight, 8.0)),
                image_weights=library_weights + upload_weights,
                limit=max(1, min(limit, 200)),
                expand=expand,
                strict_tags=strict_tags,
                folder_keys=resolve_folder_keys(parse_folder_keys(folder_keys)),
                mtime_after=mtime_after or None,
                mtime_before=mtime_before or None,
                offset=max(0, min(offset, 4000)),
                exclude_ids=_parse_id_list(exclude_ids),
                dedupe=dedupe,
            )
            return dataclasses.asdict(response)
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

    return app
