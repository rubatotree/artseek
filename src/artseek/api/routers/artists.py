"""Artist directory, author gallery, author images, and identity management routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, File, Query, UploadFile
from fastapi.responses import FileResponse

from ..deps import AppContainer, get_container
from ..schemas import (
    ArtistCoverRequest,
    ArtistGalleryRequest,
    ArtistIdentityRemoveRequest,
    ArtistIdentityRequest,
    ArtistImagesRequest,
    ArtistMembersRequest,
    ArtistMergeRequest,
    ArtistNameRequest,
    SuggestionRequest,
)

router = APIRouter()


@router.get("/api/authors")
def list_authors(
    q: str = "",
    limit: int = 40,
    offset: int = 0,
    c: AppContainer = Depends(get_container),
) -> dict[str, Any]:
    return c.artists.list_authors(q=q, limit=limit, offset=offset)


@router.post("/api/authors")
def list_authors_post(
    request: SuggestionRequest,
    c: AppContainer = Depends(get_container),
) -> dict[str, Any]:
    return c.artists.list_authors(
        q=request.q,
        limit=request.limit,
        offset=request.offset,
        folder_keys=request.folder_keys,
        mtime_after=request.mtime_after,
        mtime_before=request.mtime_before,
    )


@router.get("/api/authors/{author_key}")
def get_author(
    author_key: str,
    c: AppContainer = Depends(get_container),
) -> dict[str, Any]:
    return c.artists.get_author(author_key)


@router.post("/api/authors/{author_key}")
def get_scoped_author(
    author_key: str,
    request: SuggestionRequest,
    c: AppContainer = Depends(get_container),
) -> dict[str, Any]:
    return c.artists.get_author(
        author_key,
        folder_keys=request.folder_keys,
        mtime_after=request.mtime_after,
        mtime_before=request.mtime_before,
    )


@router.get("/api/authors/{author_key}/images")
def author_images(
    author_key: str,
    limit: int = 40,
    offset: int = Query(default=0, ge=0, le=50_000),
    sort: str = "mtime_desc",
    seed: int | None = None,
    ratings: list[str] = Query(default=[]),
    c: AppContainer = Depends(get_container),
) -> dict[str, Any]:
    return c.artists.author_images(
        author_key,
        limit=limit,
        offset=offset,
        sort=sort,
        seed=seed,
        ratings=ratings,
    )


@router.post("/api/artists/gallery")
def artist_gallery(
    request: ArtistGalleryRequest,
    c: AppContainer = Depends(get_container),
) -> dict[str, Any]:
    return c.artists.artists_gallery(
        q=request.q,
        sort=request.sort,
        include_unassigned=request.include_unassigned,
        offset=request.offset,
        limit=request.limit,
        image_limit=request.image_limit,
        image_sort=request.image_sort,
        image_seed=request.image_seed,
        snapshot_id=request.snapshot_id,
        folder_keys=request.folder_keys,
        mtime_after=request.mtime_after,
        mtime_before=request.mtime_before,
        collection_id=request.collection_id,
        author_keys=request.author_keys,
        title_query=request.title_query,
        text_tags=request.text_tags,
        tag_expression=request.tag_expression,
        ratings=request.ratings,
    )


@router.post("/api/artists/{author_key}/images")
def scoped_artist_images(
    author_key: str,
    request: ArtistImagesRequest,
    c: AppContainer = Depends(get_container),
) -> dict[str, Any]:
    return c.artists.scoped_artist_images(
        author_key,
        sort=request.sort,
        seed=request.seed,
        offset=request.offset,
        limit=request.limit,
        snapshot_id=request.snapshot_id,
        folder_keys=request.folder_keys,
        mtime_after=request.mtime_after,
        mtime_before=request.mtime_before,
        collection_id=request.collection_id,
        author_keys=request.author_keys,
        title_query=request.title_query,
        text_tags=request.text_tags,
        tag_expression=request.tag_expression,
        ratings=request.ratings,
    )


@router.post("/api/authors/{author_key}/identities")
def add_author_identity(
    author_key: str,
    request: ArtistIdentityRequest,
    c: AppContainer = Depends(get_container),
) -> dict[str, Any]:
    return c.artists.add_identity(
        author_key,
        kind=request.kind,
        value=request.value,
        label=request.label,
    )


@router.post("/api/authors/{author_key}/name")
def rename_author(
    author_key: str,
    request: ArtistNameRequest,
    c: AppContainer = Depends(get_container),
) -> dict[str, Any]:
    return c.artists.rename_author(author_key, request.name)


@router.post("/api/authors/{author_key}/identities/remove")
def remove_author_identity(
    author_key: str,
    request: ArtistIdentityRemoveRequest,
    c: AppContainer = Depends(get_container),
) -> dict[str, Any]:
    return c.artists.remove_identity(author_key, request.kind, request.value)


@router.post("/api/authors/{author_key}/cover")
def set_artist_cover(
    author_key: str,
    request: ArtistCoverRequest,
    c: AppContainer = Depends(get_container),
) -> dict[str, Any]:
    return c.artists.set_cover(author_key, request.point_id)


@router.delete("/api/authors/{author_key}/cover")
def clear_artist_cover(
    author_key: str,
    c: AppContainer = Depends(get_container),
) -> dict[str, Any]:
    return c.artists.clear_cover(author_key)


@router.post("/api/authors/{author_key}/cover/upload")
async def upload_artist_cover(
    author_key: str,
    file: UploadFile = File(...),
    c: AppContainer = Depends(get_container),
) -> dict[str, Any]:
    return c.artists.upload_cover(author_key, await file.read())


@router.get("/api/authors/{author_key}/cover-image")
def get_artist_cover_image(
    author_key: str,
    c: AppContainer = Depends(get_container),
) -> FileResponse:
    path = c.artists.resolve_asset_path(author_key, "cover")
    return FileResponse(
        path,
        media_type="image/webp",
        headers={"Cache-Control": "public, max-age=31536000"},
    )


@router.post("/api/authors/{author_key}/avatar/upload")
async def upload_artist_avatar(
    author_key: str,
    file: UploadFile = File(...),
    c: AppContainer = Depends(get_container),
) -> dict[str, Any]:
    return c.artists.upload_avatar(author_key, await file.read())


@router.delete("/api/authors/{author_key}/avatar")
def clear_artist_avatar(
    author_key: str,
    c: AppContainer = Depends(get_container),
) -> dict[str, Any]:
    return c.artists.clear_avatar(author_key)


@router.get("/api/authors/{author_key}/avatar")
def get_artist_avatar(
    author_key: str,
    c: AppContainer = Depends(get_container),
) -> FileResponse:
    path = c.artists.resolve_asset_path(author_key, "avatar")
    return FileResponse(
        path,
        media_type="image/webp",
        headers={"Cache-Control": "public, max-age=31536000"},
    )


@router.post("/api/authors/{author_key}/merge")
def merge_author(
    author_key: str,
    request: ArtistMergeRequest,
    c: AppContainer = Depends(get_container),
) -> dict[str, Any]:
    return c.artists.merge_author(author_key, request.source)


@router.post("/api/authors/{author_key}/members")
def set_author_members(
    author_key: str,
    request: ArtistMembersRequest,
    c: AppContainer = Depends(get_container),
) -> dict[str, Any]:
    """Reorder an artist's social accounts and toggle which ones are shown."""
    return c.artists.set_member_preferences(
        author_key,
        [item.model_dump() for item in request.members],
    )
