"""Search, non-blocking query plan poll, image-upload search, local search, and snapshot routes."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from ..deps import AppContainer, get_container
from ..schemas import (
    CharacterSearchRequest,
    TextSearchRequest,
    parse_folder_keys,
    parse_id_list,
    parse_reference_specs,
    parse_weight_list,
)

router = APIRouter()


# Reads stay open while an index job is queued, preparing, or running: SQLite is
# in WAL mode (readers never block the writer) and VectorIndex guards every
# matrix access with its own RLock, so a query during indexing returns a slightly
# stale-but-consistent slice instead of failing. Only mutations that would
# interleave with the scan (directory removal, task submissions, storage moves)
# are gated in their own routers.
@router.post("/api/search")
def search(
    request: TextSearchRequest,
    c: AppContainer = Depends(get_container),
) -> dict[str, Any]:
    return c.search.search(
        query=request.query,
        references=[{"id": r.id, "weight": r.weight} for r in request.references],
        reference_id=request.reference_id,
        text_weight=request.text_weight,
        limit=request.limit,
        expand=request.expand,
        wait_for_llm=request.wait_for_llm,
        strict_tags=request.strict_tags,
        folder_keys=request.folder_keys,
        mtime_after=request.mtime_after,
        mtime_before=request.mtime_before,
        offset=request.offset,
        exclude_ids=request.exclude_ids,
        dedupe=request.dedupe,
        snapshot_id=request.snapshot_id,
        author_key=request.author_key,
        author_keys=request.author_keys,
        title_query=request.title_query,
        text_label=request.text_label,
        tag_expression=request.tag_expression,
        sort=request.sort,
        seed=request.seed,
        text_tags=request.text_tags,
        tag_text=request.tag_text,
        image_tags=request.image_tags,
        media_types=request.media_types,
        ratings=request.ratings,
        monochrome=request.monochrome,
        exclude_ai=request.exclude_ai,
    )


@router.get("/api/search-plan")
def search_plan_status(
    query: str = "",
    text_label: str = "",
    expand: bool = True,
    c: AppContainer = Depends(get_container),
) -> dict[str, Any]:
    return c.search.get_query_plan_status(query, text_label=text_label, expand=expand)


@router.post("/api/local-search")
def local_search(
    request: TextSearchRequest,
    c: AppContainer = Depends(get_container),
) -> dict[str, Any]:
    return c.search.local_search(
        query=request.query,
        limit=request.limit,
        offset=request.offset,
        folder_keys=request.folder_keys,
        mtime_after=request.mtime_after,
        mtime_before=request.mtime_before,
        media_types=request.media_types,
        ratings=request.ratings,
        monochrome=request.monochrome,
        exclude_ai=request.exclude_ai,
        sort=request.sort,
        seed=request.seed,
    )


@router.post("/api/search/character")
def search_by_character(
    request: CharacterSearchRequest,
    c: AppContainer = Depends(get_container),
) -> dict[str, Any]:
    return c.search.search_by_character(
        reference_id=request.reference_id,
        bbox=request.bbox,
        folder_keys=request.folder_keys,
        mtime_after=request.mtime_after,
        mtime_before=request.mtime_before,
        media_types=request.media_types,
        ratings=request.ratings,
        monochrome=request.monochrome,
        exclude_ai=request.exclude_ai,
        dedupe=request.dedupe,
        limit=request.limit,
        offset=request.offset,
    )


@router.post("/api/search-upload")
async def search_upload(
    files: list[UploadFile] = File(default=[]),
    file_weights: str = Form(default="[]"),
    references: str = Form(default="[]"),
    query: str = Form(default=""),
    text_weight: float = Form(default=1.0),
    limit: int = Form(default=40),
    expand: bool = Form(default=True),
    wait_for_llm: bool = Form(default=False),
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
    media_types: str = Form(default="[]"),
    ratings: str = Form(default="[]"),
    monochrome: bool | None = Form(default=None),
    exclude_ai: bool | None = Form(default=None),
    c: AppContainer = Depends(get_container),
) -> dict[str, Any]:
    specs = parse_reference_specs(references)
    upload_weights = parse_weight_list(file_weights, len(files))
    temps: list[Path] = []
    try:
        for file in files:
            suffix = Path(file.filename or "query.png").suffix or ".png"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
                temps.append(Path(handle.name))
                handle.write(await file.read())
        try:
            extra_authors = json.loads(author_keys or "[]")
        except json.JSONDecodeError as exc:
            raise HTTPException(400, "作者筛选参数不合法") from exc
        if not isinstance(extra_authors, list):
            raise HTTPException(400, "作者筛选参数不合法")
        try:
            extra_text_tags = json.loads(text_tags or "[]")
            extra_tag_text = json.loads(tag_text or "[]")
            extra_image_tags = json.loads(image_tags or "[]")
            extra_media_types = json.loads(media_types or "[]")
            extra_ratings = json.loads(ratings or "[]")
        except json.JSONDecodeError as exc:
            raise HTTPException(400, "标签筛选参数不合法") from exc
        if (
            not isinstance(extra_text_tags, list)
            or not isinstance(extra_tag_text, list)
            or not isinstance(extra_image_tags, list)
            or not isinstance(extra_media_types, list)
            or not isinstance(extra_ratings, list)
        ):
            raise HTTPException(400, "标签筛选参数不合法")
        return c.search.search(
            query=query,
            references=[{"id": s.id, "weight": s.weight} for s in specs],
            upload_paths=temps,
            upload_weights=upload_weights,
            text_weight=max(0.0, min(text_weight, 8.0)),
            limit=max(1, min(limit, 200)),
            expand=expand,
            wait_for_llm=wait_for_llm,
            strict_tags=strict_tags,
            folder_keys=parse_folder_keys(folder_keys),
            mtime_after=mtime_after or None,
            mtime_before=mtime_before or None,
            offset=max(0, min(offset, 4000)),
            exclude_ids=parse_id_list(exclude_ids),
            dedupe=dedupe,
            snapshot_id=snapshot_id or None,
            author_key=author_key or None,
            author_keys=[str(a) for a in extra_authors if str(a or "").strip()],
            title_query=title_query,
            text_label=text_label,
            tag_expression=tag_expression,
            sort=sort,
            seed=seed,
            text_tags=[str(t) for t in extra_text_tags if str(t).strip()],
            tag_text=[str(t) for t in extra_tag_text if str(t).strip()],
            image_tags=[str(t) for t in extra_image_tags if str(t).strip()],
            media_types=[str(m) for m in extra_media_types if str(m or "").strip()],
            ratings=[str(r) for r in extra_ratings if str(r or "").strip()],
            monochrome=monochrome,
            exclude_ai=exclude_ai,
            had_file_refs=bool(files),
        )
    finally:
        for path in temps:
            path.unlink(missing_ok=True)


@router.get("/api/search-snapshots/{snapshot_id}")
def read_search_snapshot(
    snapshot_id: str,
    c: AppContainer = Depends(get_container),
) -> dict[str, Any]:
    return c.search.read_snapshot(snapshot_id)
