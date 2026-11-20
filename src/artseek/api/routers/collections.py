"""Collection CRUD, mixed item management, reorder, sort, and undo/redo routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query

from ..deps import AppContainer, get_container
from ..schemas import (
    CollectionItemOrderRequest,
    CollectionItemRequest,
    CollectionNameRequest,
    CollectionOrderRequest,
    CollectionUpdateRequest,
)

router = APIRouter()


@router.get("/api/collections")
def list_collections(
    image_id: str | None = None,
    c: AppContainer = Depends(get_container),
) -> dict[str, Any]:
    return c.collections.list_collections(image_id)


@router.post("/api/collections", status_code=201)
def create_collection(
    request: CollectionNameRequest,
    c: AppContainer = Depends(get_container),
) -> dict[str, Any]:
    return c.collections.create_collection(request.name)


@router.put("/api/collections/order")
def reorder_collections(
    request: CollectionOrderRequest,
    c: AppContainer = Depends(get_container),
) -> dict[str, Any]:
    return c.collections.reorder_collections(request.collection_ids)


@router.post("/api/collections/undo")
def undo_collection_history(c: AppContainer = Depends(get_container)) -> dict[str, Any]:
    return c.collections.undo()


@router.post("/api/collections/redo")
def redo_collection_history(c: AppContainer = Depends(get_container)) -> dict[str, Any]:
    return c.collections.redo()


@router.patch("/api/collections/{collection_id}")
def update_collection(
    collection_id: str,
    request: CollectionUpdateRequest,
    c: AppContainer = Depends(get_container),
) -> dict[str, Any]:
    return c.collections.update_collection(
        collection_id,
        name=request.name,
        sort=request.sort,
        seed_sort=request.seed_sort,
    )


@router.delete("/api/collections/{collection_id}")
def delete_collection(
    collection_id: str,
    discard: bool = False,
    c: AppContainer = Depends(get_container),
) -> dict[str, Any]:
    return c.collections.delete_collection(collection_id, discard=discard)


@router.post("/api/collections/{collection_id}/items")
def add_collection_item(
    collection_id: str,
    request: CollectionItemRequest,
    c: AppContainer = Depends(get_container),
) -> dict[str, Any]:
    return c.collections.add_item(collection_id, request.image_id)


@router.delete("/api/collections/{collection_id}/items/{point_id}")
def remove_collection_item(
    collection_id: str,
    point_id: str,
    c: AppContainer = Depends(get_container),
) -> dict[str, Any]:
    return c.collections.remove_item(collection_id, point_id)


@router.put("/api/collections/{collection_id}/items/order")
def reorder_collection_item(
    collection_id: str,
    request: CollectionItemOrderRequest,
    c: AppContainer = Depends(get_container),
) -> dict[str, Any]:
    return c.collections.reorder_item(
        collection_id,
        request.image_id,
        before_id=request.before_id,
        after_id=request.after_id,
    )


@router.get("/api/collections/{collection_id}/images")
def collection_images(
    collection_id: str,
    query: str = "",
    sort: str = "",
    limit: int = 40,
    offset: int = Query(default=0, ge=0, le=10_000),
    author_key: str | None = None,
    seed: int | None = None,
    ratings: list[str] = Query(default=[]),
    c: AppContainer = Depends(get_container),
) -> dict[str, Any]:
    return c.collections.collection_images(
        collection_id,
        query=query,
        sort=sort,
        limit=limit,
        offset=offset,
        author_key=author_key,
        seed=seed,
        ratings=ratings,
    )
