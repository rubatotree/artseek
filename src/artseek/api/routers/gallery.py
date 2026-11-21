"""Library folder tree, scope browsing, folder browsing, and random browse routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from ...library import DEFAULT_SORT
from ..deps import AppContainer, get_container
from ..schemas import (
    FolderScopeRequest,
    RandomBrowseRequest,
    ScopeImagesRequest,
    parse_folder_keys,
)

router = APIRouter()


@router.get("/api/library-tree")
def library_tree(c: AppContainer = Depends(get_container)) -> dict[str, Any]:
    try:
        return c.gallery.library_tree()
    except Exception as exc:
        raise HTTPException(500, f"读取目录树失败：{exc}") from exc


@router.put("/api/library-tree")
def update_library_tree(
    request: FolderScopeRequest,
    c: AppContainer = Depends(get_container),
) -> dict[str, Any]:
    return c.gallery.update_scope(
        sort=request.sort,
        selected=request.selected,
        mtime_after=request.mtime_after,
        mtime_before=request.mtime_before,
    )


@router.get("/api/folder-images")
def folder_images(
    folder_key: str,
    sort: str = DEFAULT_SORT,
    limit: int = 40,
    offset: int = 0,
    author_key: str | None = None,
    seed: int | None = None,
    ratings: list[str] = Query(default=[]),
    c: AppContainer = Depends(get_container),
) -> dict[str, Any]:
    return c.gallery.folder_images(
        folder_key,
        sort=sort,
        limit=limit,
        offset=offset,
        author_key=author_key,
        seed=seed,
        ratings=ratings,
    )


@router.get("/api/scope-images")
def scope_images(
    folder_keys: str = "",
    sort: str = DEFAULT_SORT,
    seed: int | None = None,
    limit: int = 40,
    offset: int = 0,
    author_key: str | None = None,
    mtime_after: str | None = None,
    mtime_before: str | None = None,
    ratings: list[str] = Query(default=[]),
    c: AppContainer = Depends(get_container),
) -> dict[str, Any]:
    keys = parse_folder_keys(folder_keys) or []
    return c.gallery.scope_images(
        keys,
        sort=sort,
        seed=seed,
        limit=limit,
        offset=offset,
        author_key=author_key,
        mtime_after=mtime_after,
        mtime_before=mtime_before,
        ratings=ratings,
    )


@router.post("/api/scope-images")
def scope_images_post(
    request: ScopeImagesRequest,
    c: AppContainer = Depends(get_container),
) -> dict[str, Any]:
    return c.gallery.scope_images(
        request.folder_keys,
        sort=request.sort,
        seed=request.seed,
        limit=request.limit,
        offset=request.offset,
        author_key=request.author_key,
        mtime_after=request.mtime_after,
        mtime_before=request.mtime_before,
        ratings=request.ratings,
    )


@router.post("/api/random-browse")
def random_browse(
    request: RandomBrowseRequest,
    c: AppContainer = Depends(get_container),
) -> dict[str, Any]:
    # Random paging reads the same WAL snapshot + locked vector matrices as any
    # other browse route, so it stays usable while indexing; the waterfall must
    # not freeze just because a scan is queued or preparing.
    return c.gallery.random_browse(
        folder_keys=request.folder_keys,
        seed=request.seed,
        limit=request.limit,
        offset=request.offset,
        author_key=request.author_key,
        mtime_after=request.mtime_after,
        mtime_before=request.mtime_before,
        ratings=request.ratings,
    )
