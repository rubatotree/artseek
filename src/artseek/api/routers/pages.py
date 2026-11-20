"""SPA HTML page routes."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()

_LEGACY_INDEX_PATH = Path(__file__).resolve().parents[2] / "web" / "static" / "index.html"


def _spa_html() -> HTMLResponse:
    return HTMLResponse(
        _LEGACY_INDEX_PATH.read_text(encoding="utf-8"),
        headers={"Cache-Control": "no-store"},
    )


@router.get("/", response_class=HTMLResponse)
def home() -> HTMLResponse:
    return _spa_html()


@router.get("/search", response_class=HTMLResponse)
@router.get("/folder", response_class=HTMLResponse)
@router.get("/random", response_class=HTMLResponse)
@router.get("/scope", response_class=HTMLResponse)
def spa_gallery() -> HTMLResponse:
    return _spa_html()


@router.get("/c/{collection_id}", response_class=HTMLResponse)
def spa_collection(collection_id: str) -> HTMLResponse:
    del collection_id
    return _spa_html()


@router.get("/author/{author_key}", response_class=HTMLResponse)
def spa_author(author_key: str) -> HTMLResponse:
    del author_key
    return _spa_html()


@router.get("/tag/{tag:path}", response_class=HTMLResponse)
def spa_tag(tag: str) -> HTMLResponse:
    del tag
    return _spa_html()


@router.get("/i/{point_id}", response_class=HTMLResponse)
@router.get("/i/{point_id}/notes", response_class=HTMLResponse)
def spa_image(point_id: str) -> HTMLResponse:
    del point_id
    return _spa_html()


@router.get("/i/{point_id}/notes/{name:path}", response_class=HTMLResponse)
def spa_note(point_id: str, name: str) -> HTMLResponse:
    del point_id, name
    return _spa_html()
