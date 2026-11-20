"""Model lifecycle, web settings, and storage path management routes."""

from __future__ import annotations

import dataclasses
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from ...config import WebSearchSettings, load_web_search_settings, save_web_search_settings
from ..deps import AppContainer, get_container
from ..schemas import (
    ModelControlRequest,
    StorageMigrateRequest,
    StoragePathsRequest,
    WebSearchSettingsRequest,
)

router = APIRouter()


@router.get("/api/model")
def model_status(c: AppContainer = Depends(get_container)) -> dict[str, Any]:
    return c.lifecycle.status()


@router.post("/api/model")
def update_model(
    request: ModelControlRequest,
    c: AppContainer = Depends(get_container),
) -> dict[str, Any]:
    if request.auto_unload_minutes is not None:
        c.lifecycle.set_auto_unload_seconds(request.auto_unload_minutes * 60)
    if request.loaded is True:
        try:
            c.lifecycle.load_model()
        except Exception as exc:
            raise HTTPException(500, f"载入模型失败：{exc}") from exc
    elif request.loaded is False:
        if not c.lifecycle.unload_model():
            status = c.lifecycle.status()
            status["message"] = "正在检索或索引，稍后再卸载"
            return status
    return c.lifecycle.status()


@router.get("/api/web-settings")
def web_settings(c: AppContainer = Depends(get_container)) -> dict[str, Any]:
    return dataclasses.asdict(load_web_search_settings(c.config))


@router.put("/api/web-settings")
def update_web_settings(
    request: WebSearchSettingsRequest,
    c: AppContainer = Depends(get_container),
) -> dict[str, Any]:
    settings = WebSearchSettings(
        use_llm=request.use_llm,
        strict_tags=request.strict_tags,
        dedupe=request.dedupe,
        limit=request.limit,
    )
    save_web_search_settings(c.config, settings)
    return dataclasses.asdict(settings)


@router.get("/api/storage-paths")
def storage_paths(c: AppContainer = Depends(get_container)) -> dict[str, Any]:
    return c.storage.get_paths()


@router.put("/api/storage-paths")
def update_storage_paths(
    request: StoragePathsRequest,
    c: AppContainer = Depends(get_container),
) -> dict[str, Any]:
    return c.storage.update_paths(
        qdrant_path=request.qdrant_path,
        notes_dir=request.notes_dir,
        cache_dir=request.cache_dir,
        index_dir=request.index_dir,
    )


@router.post("/api/storage-paths/migrate")
def migrate_storage_path(
    request: StorageMigrateRequest,
    c: AppContainer = Depends(get_container),
) -> dict[str, Any]:
    return c.storage.migrate_path(request.kind, request.destination)
