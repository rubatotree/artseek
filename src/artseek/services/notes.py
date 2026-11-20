"""Notes service syncing Markdown vault storage with SQLite `notes_index`."""

from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Any

from ..db.repos.images import ImageRepo
from ..db.repos.notes import NotesIndexRepo
from ..image_io import PHASH_LENGTH, perceptual_hash
from ..notes import HASH_RE, NoteError, NoteStore, normalize_content_hash, resolve_content_hash
from .common import ServiceError


class NoteService:
    """Manages per-image Markdown notes & assets while keeping `notes_index` synchronized."""

    def __init__(
        self,
        store: NoteStore,
        notes_index: NotesIndexRepo,
        images: ImageRepo,
    ) -> None:
        self.store = store
        self.notes_index = notes_index
        self.images = images

    def set_root(self, root: Path) -> None:
        self.store.set_root(root)
        self.rebuild_index()

    def rebuild_index(self) -> int:
        """Scan `store.root` and sync all markdown note headers into `notes_index`."""
        count = 0
        root = Path(self.store.root)
        if not root.is_dir():
            return 0
        for child in root.iterdir():
            if not child.is_dir():
                continue
            digest = child.name.strip().casefold()
            if not HASH_RE.fullmatch(digest):
                continue
            notes = self.store.list_notes(digest)
            entries = [
                (n.name, n.title, int(n.mtime_ns // 1_000_000_000))
                for n in notes
            ]
            self.notes_index.sync_hash(digest, entries)
            count += len(entries)
        return count

    def _payload_phash(self, payload: dict[str, Any] | None) -> str:
        if not payload:
            return ""
        stored = str(payload.get("phash") or "").strip().casefold()
        if len(stored) == PHASH_LENGTH:
            return stored
        raw_path = Path(str(payload.get("path") or ""))
        if not raw_path.is_file():
            return ""
        try:
            return perceptual_hash(raw_path)
        except (OSError, ValueError):
            return ""

    def _note_identity(self, payload: dict[str, Any], point_id: str) -> dict[str, Any]:
        try:
            width = int(payload.get("width") or 0)
        except (TypeError, ValueError):
            width = 0
        try:
            height = int(payload.get("height") or 0)
        except (TypeError, ValueError):
            height = 0
        return {
            "name": str(payload.get("name") or ""),
            "path": str(payload.get("path") or ""),
            "point_id": point_id,
            "phash": self._payload_phash(payload),
            "width": width,
            "height": height,
        }

    def _context(self, point_id: str) -> tuple[dict[str, Any], str]:
        payload = self.images.get_by_point_id(point_id)
        digest = ""
        if payload:
            try:
                digest = resolve_content_hash(payload)
            except FileNotFoundError:
                digest = self.store.lookup_by_point_id(point_id) or ""
            except NoteError as exc:
                raise ServiceError(400, str(exc)) from exc
        if not digest:
            digest = self.store.lookup_by_point_id(point_id) or ""
        if not digest:
            detail = (
                "图片不在索引中"
                if not payload
                else "原图片已移动或删除，且没有已保存的笔记"
            )
            raise ServiceError(404, detail)
        identity = self._note_identity(payload or {}, point_id)
        try:
            canonical = self.store.resolve_canonical(digest, **identity)
        except NoteError as exc:
            raise ServiceError(400, str(exc)) from exc
        if not payload:
            meta = self.store.image_meta(canonical)
            payload = {
                "name": meta.get("name") or "",
                "path": meta.get("path") or "",
                "content_hash": canonical,
            }
        payload = dict(payload)
        payload["phash"] = identity.get("phash") or payload.get("phash") or ""
        return payload, canonical

    def _sync_notes_index(self, digest: str, identity: dict[str, Any]) -> None:
        canonical = normalize_content_hash(digest)
        notes = self.store.list_notes(canonical, **identity)
        self.notes_index.sync_hash(
            canonical,
            [(n.name, n.title, int(n.mtime_ns // 1_000_000_000)) for n in notes],
        )

    def list_notes(self, point_id: str) -> dict[str, Any]:
        payload, digest = self._context(point_id)
        identity = self._note_identity(payload, point_id)
        try:
            notes = self.store.list_notes(digest, **identity)
        except NoteError as exc:
            raise ServiceError(400, str(exc)) from exc
        self.notes_index.sync_hash(
            digest,
            [(n.name, n.title, int(n.mtime_ns // 1_000_000_000)) for n in notes],
        )
        return {
            "image": {
                "id": point_id,
                "content_hash": digest,
                "name": str(payload.get("name") or ""),
                "path": str(payload.get("path") or ""),
            },
            "notes": [item.as_dict() for item in notes],
        }

    def create_note(
        self, point_id: str, *, title: str | None = None, content: str = ""
    ) -> dict[str, Any]:
        payload, digest = self._context(point_id)
        identity = self._note_identity(payload, point_id)
        try:
            doc = self.store.create_note(digest, title=title, content=content, **identity)
        except NoteError as exc:
            raise ServiceError(400, str(exc)) from exc
        self._sync_notes_index(digest, identity)
        return doc.as_dict()

    def reorder_notes(self, point_id: str, names: list[str]) -> dict[str, Any]:
        payload, digest = self._context(point_id)
        identity = self._note_identity(payload, point_id)
        try:
            notes = self.store.reorder_notes(digest, names, **identity)
        except NoteError as exc:
            raise ServiceError(400, str(exc)) from exc
        self.notes_index.sync_hash(
            digest,
            [(n.name, n.title, int(n.mtime_ns // 1_000_000_000)) for n in notes],
        )
        return {"notes": [item.as_dict() for item in notes]}

    def read_note(self, point_id: str, name: str) -> dict[str, Any]:
        _payload, digest = self._context(point_id)
        try:
            return self.store.read_note(digest, name).as_dict()
        except FileNotFoundError as exc:
            raise ServiceError(404, str(exc)) from exc
        except NoteError as exc:
            raise ServiceError(400, str(exc)) from exc

    def write_note(
        self,
        point_id: str,
        name: str,
        *,
        title: str | None = None,
        content: str | None = None,
    ) -> dict[str, Any]:
        payload, digest = self._context(point_id)
        identity = self._note_identity(payload, point_id)
        try:
            doc = self.store.write_note(digest, name, title=title, content=content, **identity)
        except FileNotFoundError as exc:
            raise ServiceError(404, str(exc)) from exc
        except NoteError as exc:
            raise ServiceError(400, str(exc)) from exc
        self._sync_notes_index(digest, identity)
        return doc.as_dict()

    def delete_note(self, point_id: str, name: str) -> dict[str, Any]:
        payload, digest = self._context(point_id)
        try:
            removed = self.store.delete_note(digest, name)
        except FileNotFoundError as exc:
            raise ServiceError(404, str(exc)) from exc
        except NoteError as exc:
            raise ServiceError(400, str(exc)) from exc
        self.notes_index.delete_note(digest, name)
        return {"deleted": True, "removed_assets": removed}

    def delete_all_notes(self, point_id: str) -> dict[str, Any]:
        _payload, digest = self._context(point_id)
        try:
            self.store.delete_folder(digest)
        except NoteError as exc:
            raise ServiceError(400, str(exc)) from exc
        self.notes_index.delete_hash(digest)
        return {"deleted": True, "folder": digest}

    def save_asset(
        self,
        point_id: str,
        *,
        filename: str,
        content_type: str | None,
        data: bytes,
    ) -> dict[str, Any]:
        payload, digest = self._context(point_id)
        name = filename or "paste"
        if not Path(name).suffix:
            name += mimetypes.guess_extension(content_type or "") or ".png"
        try:
            relative = self.store.save_asset(
                digest,
                name,
                data,
                **self._note_identity(payload, point_id),
            )
        except NoteError as exc:
            raise ServiceError(400, str(exc)) from exc
        return {
            "path": relative,
            "url": f"/api/note-files/{digest}/{relative}",
        }

    def public_file(self, content_hash: str, relative: str) -> tuple[Path, str]:
        try:
            path = self.store.public_file(content_hash, relative)
        except FileNotFoundError as exc:
            raise ServiceError(404, str(exc)) from exc
        except NoteError as exc:
            raise ServiceError(400, str(exc)) from exc
        media_type, _ = mimetypes.guess_type(path.name)
        return path, media_type or "application/octet-stream"
