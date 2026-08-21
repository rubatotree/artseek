from __future__ import annotations

import dataclasses
import json
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field, field_validator

from ..config import AppConfig
from ..image_io import canonical_path, make_thumbnail
from ..indexer import ImageIndexer
from ..library import (
    DEFAULT_SORT,
    SORT_KEYS,
    apply_scope,
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


@dataclass
class IndexJob:
    running: bool = False
    path: str = ""
    done: int = 0
    total: int = 0
    message: str = ""
    error: str | None = None
    summary: dict[str, Any] | None = None


def create_app(config: AppConfig) -> FastAPI:
    app = FastAPI(title="ArtSeek", docs_url="/api/docs", redoc_url=None)
    app.state.config = config
    app.state.engine = SearchEngine(config)
    app.state.tagger = None
    app.state.index_job = IndexJob()
    app.state.index_lock = threading.Lock()
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
            results.append(
                {
                    "id": point_id,
                    "score": None,
                    "path": str(payload.get("path") or ""),
                    "name": str(payload.get("name") or name),
                    "width": int(payload.get("width") or 0),
                    "height": int(payload.get("height") or 0),
                    "tags": list(payload.get("tags") or []),
                    "tag_hits": [],
                    "caption": payload.get("caption"),
                }
            )
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
        library_paths, library_weights = resolve_library_references(parse_reference_specs(references))
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
        job.path = "、".join(str(item) for item in directories)
        job.done = 0
        job.total = 0
        job.error = None
        job.summary = None
        job.message = "正在准备索引…"
        try:
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
            app.state.engine.invalidate_folder_catalog()
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
        app.state.engine.store.delete_ids(ids)
        for point_id in ids:
            thumb = config.thumbnail_dir / f"{point_id}.webp"
            thumb.unlink(missing_ok=True)
        if ids:
            rebuild_catalog(app.state.engine.store, config)
        app.state.engine.invalidate_folder_catalog()
        scope = load_scope(config)
        scope.selected = [item for item in scope.selected if item not in drop_keys]
        scope.known = [item for item in scope.known if item not in drop_keys]
        save_scope(config, scope)
        return current_library()

    return app
