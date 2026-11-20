"""Per-image Markdown note and attachment asset routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, File, UploadFile
from fastapi.responses import FileResponse

from ..deps import AppContainer, get_container
from ..schemas import NoteCreateRequest, NoteOrderRequest, NoteWriteRequest

router = APIRouter()


@router.get("/api/images/{point_id}/notes")
def list_image_notes(
    point_id: str,
    c: AppContainer = Depends(get_container),
) -> dict[str, Any]:
    return c.notes.list_notes(point_id)


@router.post("/api/images/{point_id}/notes")
def create_image_note(
    point_id: str,
    request: NoteCreateRequest,
    c: AppContainer = Depends(get_container),
) -> dict[str, Any]:
    return c.notes.create_note(point_id, title=request.title, content=request.content)


@router.put("/api/images/{point_id}/note-order")
def reorder_image_notes(
    point_id: str,
    request: NoteOrderRequest,
    c: AppContainer = Depends(get_container),
) -> dict[str, Any]:
    return c.notes.reorder_notes(point_id, request.names)


@router.get("/api/images/{point_id}/notes/{name}")
def read_image_note(
    point_id: str,
    name: str,
    c: AppContainer = Depends(get_container),
) -> dict[str, Any]:
    return c.notes.read_note(point_id, name)


@router.put("/api/images/{point_id}/notes/{name}")
def write_image_note(
    point_id: str,
    name: str,
    request: NoteWriteRequest,
    c: AppContainer = Depends(get_container),
) -> dict[str, Any]:
    return c.notes.write_note(point_id, name, title=request.title, content=request.content)


@router.delete("/api/images/{point_id}/notes/{name}")
def delete_image_note(
    point_id: str,
    name: str,
    c: AppContainer = Depends(get_container),
) -> dict[str, Any]:
    return c.notes.delete_note(point_id, name)


@router.delete("/api/images/{point_id}/notes")
def delete_all_image_notes(
    point_id: str,
    c: AppContainer = Depends(get_container),
) -> dict[str, Any]:
    return c.notes.delete_all_notes(point_id)


@router.post("/api/images/{point_id}/assets")
async def upload_note_asset(
    point_id: str,
    file: UploadFile = File(...),
    c: AppContainer = Depends(get_container),
) -> dict[str, Any]:
    data = await file.read()
    return c.notes.save_asset(
        point_id,
        filename=file.filename or "paste",
        content_type=file.content_type,
        data=data,
    )


@router.get("/api/note-files/{content_hash}/{relative:path}")
def note_file(
    content_hash: str,
    relative: str,
    c: AppContainer = Depends(get_container),
) -> FileResponse:
    path, media_type = c.notes.public_file(content_hash, relative)
    return FileResponse(path, media_type=media_type)
