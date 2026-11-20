"""Tag suggestion, tag dictionary, and tag image listing routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query

from ..deps import AppContainer, get_container
from ..schemas import SuggestionRequest

router = APIRouter()


@router.get("/api/tags")
def tag_suggestions(
    q: str = "",
    kind: str = "source",
    limit: int = 60,
    offset: int = 0,
    c: AppContainer = Depends(get_container),
) -> dict[str, Any]:
    return c.gallery.tag_suggestions(q=q, kind=kind, limit=limit, offset=offset)


@router.post("/api/tags")
def tag_suggestions_post(
    request: SuggestionRequest,
    c: AppContainer = Depends(get_container),
) -> dict[str, Any]:
    return c.gallery.tag_suggestions(
        q=request.q,
        kind=request.kind,
        limit=request.limit,
        offset=request.offset,
        folder_keys=request.folder_keys,
        mtime_after=request.mtime_after,
        mtime_before=request.mtime_before,
    )


@router.get("/api/tag-dictionary")
def tag_dictionary(
    q: str = "",
    limit: int = 40,
    c: AppContainer = Depends(get_container),
) -> dict[str, Any]:
    return c.gallery.tag_dictionary(q=q, limit=limit)


@router.get("/api/tags/{tag}/images")
def tag_images(
    tag: str,
    limit: int = 40,
    offset: int = Query(default=0, ge=0, le=50_000),
    sort: str = "mtime_desc",
    seed: int | None = None,
    c: AppContainer = Depends(get_container),
) -> dict[str, Any]:
    return c.gallery.tag_images(
        tag,
        limit=limit,
        offset=offset,
        sort=sort,
        seed=seed,
    )
