"""Thumbnail, search-image, full-image, and item metadata routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse

from ...image_io import make_search_image, make_thumbnail
from ..deps import AppContainer, get_container

router = APIRouter()


@router.get("/thumb/{point_id}")
def thumbnail(point_id: str, c: AppContainer = Depends(get_container)) -> FileResponse:
    _, source = c.gallery.resolve_item(point_id)
    destination = c.config.thumbnail_dir / f"{point_id}.webp"
    if not destination.exists() or destination.stat().st_mtime_ns < source.stat().st_mtime_ns:
        make_thumbnail(source, destination)
    return FileResponse(destination, media_type="image/webp")


@router.get("/image/{point_id}")
def full_image(point_id: str, c: AppContainer = Depends(get_container)) -> FileResponse:
    _, path = c.gallery.resolve_item(point_id)
    return FileResponse(path)


@router.get("/search-image/{point_id}")
def search_image(point_id: str, c: AppContainer = Depends(get_container)) -> FileResponse:
    _, source = c.gallery.resolve_item(point_id)
    destination = c.config.thumbnail_dir / f"{point_id}-search.jpg"
    if not destination.exists() or destination.stat().st_mtime_ns < source.stat().st_mtime_ns:
        make_search_image(source, destination)
    return FileResponse(destination, media_type="image/jpeg")


@router.get("/api/item/{point_id}")
def item_metadata(point_id: str, c: AppContainer = Depends(get_container)) -> dict[str, Any]:
    payload, _ = c.gallery.resolve_item(point_id)
    return payload
