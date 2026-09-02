from __future__ import annotations

import json
import re
import shutil
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .image_io import (
    PHASH_LENGTH,
    PHASH_MAX_DISTANCE,
    aspect_ratio_compatible,
    file_digest,
    hamming_distance,
)

NOTES_INDEX_VERSION = 3
INDEX_NAME = ".artseek-index.json"
META_NAME = ".artseek.json"
ASSETS_DIR = "assets"
DEFAULT_NOTE_TITLE = "笔记"
MAX_NOTE_TITLE_LENGTH = 80
MAX_NOTE_BYTES = 2 * 1024 * 1024
MAX_ASSET_BYTES = 8 * 1024 * 1024
HASH_RE = re.compile(r"^[0-9a-f]{64}$")
PHASH_RE = re.compile(rf"^[0-9a-f]{{{PHASH_LENGTH}}}$")
IMAGE_MARKDOWN_RE = re.compile(r"!\[[^\]]*]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
IMAGE_HTML_RE = re.compile(r"<img\b[^>]*\bsrc=['\"]([^'\"]+)['\"]", re.IGNORECASE)
FORBIDDEN_NAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
ATX_H1_RE = re.compile(r"^ {0,3}#(?!#)[ \t]*(.+?)(?:[ \t]+#+)?[ \t]*$")
INLINE_LINK_RE = re.compile(r"!\[([^\]]*)\]\([^)]*\)|\[([^\]]+)\]\([^)]*\)")
INLINE_MARK_RE = re.compile(r"[*_`]+")
RESERVED_DEVICE_NAMES = frozenset(
    {
        "con",
        "prn",
        "aux",
        "nul",
        *(f"com{index}" for index in range(1, 10)),
        *(f"lpt{index}" for index in range(1, 10)),
    }
)
_PATH_LOCKS: dict[str, threading.RLock] = {}
_PATH_LOCKS_GUARD = threading.Lock()


class NoteError(ValueError):
    """User-facing validation failure for note names or payloads."""


def _lock_for_path(path: Path) -> threading.RLock:
    try:
        key = str(path.resolve()).casefold()
    except OSError:
        key = str(path.absolute()).casefold()
    with _PATH_LOCKS_GUARD:
        return _PATH_LOCKS.setdefault(key, threading.RLock())


def normalize_content_hash(value: str) -> str:
    digest = value.strip().casefold()
    if not HASH_RE.fullmatch(digest):
        raise NoteError("图片内容哈希不合法")
    return digest


def resolve_content_hash(payload: dict[str, Any], *, path: Path | None = None) -> str:
    """Prefer the indexed digest; hash the file only when the payload lacks one."""
    stored = str(payload.get("content_hash") or "").strip().casefold()
    if HASH_RE.fullmatch(stored):
        return stored
    source = path if path is not None else Path(str(payload.get("path") or ""))
    if not source or not source.exists():
        raise FileNotFoundError("原图片已移动或删除，无法关联笔记")
    return file_digest(source)


def sanitize_note_filename(title: str) -> str:
    text = str(title or "").strip()
    if text.lower().endswith(".md"):
        text = text[:-3].rstrip()
    text = FORBIDDEN_NAME_CHARS.sub("_", text)
    text = text.strip(" .")
    if not text:
        text = DEFAULT_NOTE_TITLE
    stem = text.split(".", 1)[0].casefold()
    if stem in RESERVED_DEVICE_NAMES:
        text = f"_{text}"
    if len(text) > MAX_NOTE_TITLE_LENGTH:
        text = text[:MAX_NOTE_TITLE_LENGTH].rstrip(" .") or DEFAULT_NOTE_TITLE
    if text.startswith("."):
        text = f"_{text.lstrip('.')}" or DEFAULT_NOTE_TITLE
    return f"{text}.md"


def title_from_filename(filename: str) -> str:
    return Path(filename).stem


def first_heading_title(markdown: str) -> str:
    """Return the ATX H1 on the first non-empty line, or empty if that line is not H1."""
    first = ""
    for line in (markdown or "").splitlines():
        if line.strip():
            first = line
            break
    if not first:
        return ""
    match = ATX_H1_RE.match(first)
    if not match:
        return ""

    def _link_text(item: re.Match[str]) -> str:
        return item.group(1) or item.group(2) or ""

    text = INLINE_LINK_RE.sub(_link_text, match.group(1))
    text = INLINE_MARK_RE.sub("", text).strip()
    return text


def vault_asset_refs(markdown: str, content_hash: str) -> set[str]:
    """Return `assets/...` paths this note points at inside the notes vault."""
    digest = content_hash.strip().casefold()
    found: set[str] = set()
    for match in IMAGE_MARKDOWN_RE.finditer(markdown or ""):
        relative = _normalize_vault_asset(match.group(1), digest)
        if relative:
            found.add(relative)
    for match in IMAGE_HTML_RE.finditer(markdown or ""):
        relative = _normalize_vault_asset(match.group(1), digest)
        if relative:
            found.add(relative)
    return found


def _normalize_vault_asset(url: str, content_hash: str) -> str | None:
    text = (url or "").strip().strip("<>").replace("\\", "/")
    if not text or text.startswith(("#", "data:", "file:")):
        return None
    lowered = text.casefold()
    if lowered.startswith(("http://", "https://", "mailto:")):
        return None
    api_prefix = f"/api/note-files/{content_hash}/"
    if lowered.startswith(api_prefix):
        text = text[len(api_prefix) :]
    text = text.lstrip("./")
    relative = Path(text)
    if relative.is_absolute() or ".." in relative.parts:
        return None
    if relative.parts[:1] != (ASSETS_DIR,):
        return None
    return relative.as_posix()


def _is_inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (OSError, ValueError):
        return False


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_bytes(data)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def _iso_mtime(path: Path) -> str:
    try:
        stamp = path.stat().st_mtime
    except OSError:
        stamp = 0.0
    return datetime.fromtimestamp(stamp, tz=UTC).replace(microsecond=0).isoformat()


@dataclass(frozen=True, slots=True)
class NoteInfo:
    name: str
    title: str
    updated_at: str
    size_bytes: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "title": self.title,
            "updated_at": self.updated_at,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True, slots=True)
class NoteDocument:
    name: str
    title: str
    content: str
    updated_at: str
    content_hash: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "title": self.title,
            "content": self.content,
            "updated_at": self.updated_at,
            "content_hash": self.content_hash,
        }


class NoteStore:
    """Filesystem vault of per-image Markdown folders.

    Folders are named by file SHA-256. Near-duplicate frames (same picture at
    another resolution) are aliased onto one canonical folder via dHash.
    Hamming search only looks at hashes already registered in this vault.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self._lock = _lock_for_path(root)

    def set_root(self, root: Path) -> None:
        root.mkdir(parents=True, exist_ok=True)
        self.root = root
        self._lock = _lock_for_path(root)

    def image_dir(self, content_hash: str) -> Path:
        digest = normalize_content_hash(content_hash)
        return self.root / digest

    def lookup_by_point_id(self, point_id: str) -> str | None:
        key = str(point_id or "").strip()
        if not key:
            return None
        with self._lock:
            payload = self._read_index(self.root / INDEX_NAME)
            digest = str((payload.get("points") or {}).get(key) or "").strip().casefold()
            if not HASH_RE.fullmatch(digest):
                return None
            return self._follow_alias(payload, digest)

    def resolve_canonical(
        self,
        content_hash: str,
        *,
        phash: str = "",
        width: int = 0,
        height: int = 0,
        name: str = "",
        path: str = "",
        point_id: str = "",
    ) -> str:
        digest = normalize_content_hash(content_hash)
        with self._lock:
            return self._resolve_canonical_locked(
                digest,
                phash=phash,
                width=width,
                height=height,
                name=name,
                path=path,
                point_id=point_id,
            )

    def image_meta(self, content_hash: str) -> dict[str, str]:
        digest = normalize_content_hash(content_hash)
        meta_path = self.image_dir(digest) / META_NAME
        if not meta_path.is_file():
            return {"content_hash": digest, "name": "", "path": "", "point_id": ""}
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = {}
        if not isinstance(data, dict):
            data = {}
        return {
            "content_hash": digest,
            "name": str(data.get("name") or ""),
            "path": str(data.get("path") or ""),
            "point_id": str(data.get("point_id") or ""),
        }

    def list_notes(
        self,
        content_hash: str,
        *,
        name: str = "",
        path: str = "",
        point_id: str = "",
        phash: str = "",
        width: int = 0,
        height: int = 0,
    ) -> list[NoteInfo]:
        digest = normalize_content_hash(content_hash)
        with self._lock:
            canonical = self._resolve_canonical_locked(
                digest,
                phash=phash,
                width=width,
                height=height,
                name=name,
                path=path,
                point_id=point_id,
            )
            directory = self._ensure_image_dir(
                canonical, name=name, path=path, point_id=point_id
            )
            notes = self._list_markdown(directory)
            self._remember_note_order(directory, notes)
            self._update_index(
                canonical,
                name=name,
                path=path,
                point_id=point_id,
                note_count=len(notes),
                phash=phash,
                width=width,
                height=height,
                source_hash=digest,
            )
            return notes

    def create_note(
        self,
        content_hash: str,
        *,
        title: str | None = None,
        content: str = "",
        name: str = "",
        path: str = "",
        point_id: str = "",
        phash: str = "",
        width: int = 0,
        height: int = 0,
    ) -> NoteDocument:
        digest = normalize_content_hash(content_hash)
        body = content if content is not None else ""
        self._check_content_size(body)
        chosen = title if str(title or "").strip() else None
        if chosen is None or sanitize_note_filename(chosen) == sanitize_note_filename(
            DEFAULT_NOTE_TITLE
        ):
            heading = first_heading_title(body)
            if heading:
                chosen = heading
        filename = sanitize_note_filename(chosen or DEFAULT_NOTE_TITLE)
        with self._lock:
            canonical = self._resolve_canonical_locked(
                digest,
                phash=phash,
                width=width,
                height=height,
                name=name,
                path=path,
                point_id=point_id,
            )
            directory = self._ensure_image_dir(
                canonical, name=name, path=path, point_id=point_id
            )
            filename = self._unique_name(directory, filename)
            note_path = directory / filename
            _atomic_write_text(note_path, body)
            notes = self._list_markdown(directory)
            self._remember_note_order(directory, notes)
            self._update_index(
                canonical,
                name=name,
                path=path,
                point_id=point_id,
                note_count=len(notes),
                phash=phash,
                width=width,
                height=height,
                source_hash=digest,
            )
            return self._read_document(canonical, note_path)

    def read_note(self, content_hash: str, filename: str) -> NoteDocument:
        digest = normalize_content_hash(content_hash)
        with self._lock:
            note_path = self._note_path(digest, filename)
            if not note_path.is_file():
                raise FileNotFoundError("笔记不存在")
            return self._read_document(digest, note_path)

    def write_note(
        self,
        content_hash: str,
        filename: str,
        *,
        content: str | None = None,
        title: str | None = None,
        name: str = "",
        path: str = "",
        point_id: str = "",
        phash: str = "",
        width: int = 0,
        height: int = 0,
    ) -> NoteDocument:
        digest = normalize_content_hash(content_hash)
        if content is not None:
            self._check_content_size(content)
        with self._lock:
            canonical = self._resolve_canonical_locked(
                digest,
                phash=phash,
                width=width,
                height=height,
                name=name,
                path=path,
                point_id=point_id,
            )
            directory = self._ensure_image_dir(
                canonical, name=name, path=path, point_id=point_id
            )
            note_path = self._note_path(canonical, filename, directory=directory)
            if not note_path.is_file():
                raise FileNotFoundError("笔记不存在")
            auto_title = False
            next_title = title
            heading = first_heading_title(content) if content is not None else ""
            if next_title is None and heading:
                next_title = heading
                auto_title = True
            elif (
                next_title is not None
                and heading
                and sanitize_note_filename(next_title) == sanitize_note_filename(heading)
            ):
                auto_title = True
            if next_title is not None:
                next_name = sanitize_note_filename(next_title)
                destination = directory / next_name
                if destination.name != note_path.name:
                    if destination.exists():
                        if not auto_title:
                            raise NoteError("已有同名笔记")
                    else:
                        self._rename_note_order(directory, note_path.name, destination.name)
                        note_path.replace(destination)
                        note_path = destination
            if content is not None:
                _atomic_write_text(note_path, content)
            notes = self._list_markdown(directory)
            self._remember_note_order(directory, notes)
            self._update_index(
                canonical,
                name=name,
                path=path,
                point_id=point_id,
                note_count=len(notes),
                phash=phash,
                width=width,
                height=height,
                source_hash=digest,
            )
            return self._read_document(canonical, note_path)

    def delete_note(self, content_hash: str, filename: str) -> list[str]:
        """Delete one Markdown file and vault images it uniquely referenced."""
        digest = normalize_content_hash(content_hash)
        with self._lock:
            note_path = self._note_path(digest, filename)
            if not note_path.is_file():
                raise FileNotFoundError("笔记不存在")
            try:
                body = note_path.read_text(encoding="utf-8")
            except OSError:
                body = ""
            owned = vault_asset_refs(body, digest)
            note_path.unlink()
            directory = note_path.parent
            remaining: set[str] = set()
            for item in self._list_markdown(directory):
                other = directory / item.name
                try:
                    remaining |= vault_asset_refs(other.read_text(encoding="utf-8"), digest)
                except OSError:
                    continue
            removed = self._delete_unreferenced_assets(directory, owned - remaining)
            notes = self._list_markdown(directory)
            self._remember_note_order(directory, notes)
            self._update_index(digest, name="", path="", point_id="", note_count=len(notes))
            return removed

    def delete_folder(self, content_hash: str) -> None:
        """Remove the whole per-image notes directory. Originals are untouched."""
        digest = normalize_content_hash(content_hash)
        with self._lock:
            directory = self.image_dir(digest)
            if directory.exists():
                shutil.rmtree(directory)
            self._drop_index(digest)

    def reorder_notes(
        self,
        content_hash: str,
        names: list[str],
        *,
        name: str = "",
        path: str = "",
        point_id: str = "",
        phash: str = "",
        width: int = 0,
        height: int = 0,
    ) -> list[NoteInfo]:
        digest = normalize_content_hash(content_hash)
        requested = [sanitize_note_filename(item) for item in names]
        if len(requested) != len(set(requested)):
            raise NoteError("笔记顺序不能重复")
        with self._lock:
            canonical = self._resolve_canonical_locked(
                digest,
                phash=phash,
                width=width,
                height=height,
                name=name,
                path=path,
                point_id=point_id,
            )
            directory = self._ensure_image_dir(
                canonical, name=name, path=path, point_id=point_id
            )
            current = {item.name for item in self._scan_markdown(directory)}
            if set(requested) != current:
                raise NoteError("笔记顺序必须包含当前全部笔记")
            self._write_note_order(directory, requested)
            notes = self._list_markdown(directory)
            self._update_index(
                canonical,
                name=name,
                path=path,
                point_id=point_id,
                note_count=len(notes),
                phash=phash,
                width=width,
                height=height,
                source_hash=digest,
            )
            return notes

    def save_asset(
        self,
        content_hash: str,
        filename: str,
        data: bytes,
        *,
        name: str = "",
        path: str = "",
        point_id: str = "",
        phash: str = "",
        width: int = 0,
        height: int = 0,
    ) -> str:
        digest = normalize_content_hash(content_hash)
        if len(data) > MAX_ASSET_BYTES:
            raise NoteError("附件过大")
        safe_name = self._sanitize_asset_name(filename)
        with self._lock:
            canonical = self._resolve_canonical_locked(
                digest,
                phash=phash,
                width=width,
                height=height,
                name=name,
                path=path,
                point_id=point_id,
            )
            directory = self._ensure_image_dir(
                canonical, name=name, path=path, point_id=point_id
            )
            assets = directory / ASSETS_DIR
            assets.mkdir(parents=True, exist_ok=True)
            destination = assets / self._unique_name(assets, safe_name)
            _atomic_write_bytes(destination, data)
            self._update_index(
                canonical,
                name=name,
                path=path,
                point_id=point_id,
                note_count=len(self._list_markdown(directory)),
                phash=phash,
                width=width,
                height=height,
                source_hash=digest,
            )
            return f"{ASSETS_DIR}/{destination.name}"

    def public_file(self, content_hash: str, relative: str) -> Path:
        digest = normalize_content_hash(content_hash)
        relative_path = Path(relative.replace("\\", "/"))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise NoteError("附件路径不合法")
        if relative_path.parts[:1] != (ASSETS_DIR,):
            raise NoteError("只能读取笔记附件")
        directory = self.image_dir(digest)
        path = (directory / relative_path).resolve()
        if not _is_inside(path, directory / ASSETS_DIR):
            raise NoteError("附件路径不合法")
        if not path.is_file():
            raise FileNotFoundError("附件不存在")
        return path

    def _ensure_image_dir(
        self,
        digest: str,
        *,
        name: str,
        path: str,
        point_id: str,
    ) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        directory = self.root / digest
        directory.mkdir(parents=True, exist_ok=True)
        meta_path = directory / META_NAME
        meta = {
            "content_hash": digest,
            "name": name,
            "path": path,
            "point_id": point_id,
        }
        existing = self._read_meta(directory)
        if existing:
            if not name:
                meta["name"] = str(existing.get("name") or "")
            if not path:
                meta["path"] = str(existing.get("path") or "")
            if not point_id:
                meta["point_id"] = str(existing.get("point_id") or "")
            order = existing.get("note_order")
            if isinstance(order, list):
                meta["note_order"] = [str(item) for item in order if str(item).endswith(".md")]
        _atomic_write_text(meta_path, json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
        return directory

    def _read_meta(self, directory: Path) -> dict[str, Any]:
        path = directory / META_NAME
        if not path.is_file():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    def _write_meta(self, directory: Path, meta: dict[str, Any]) -> None:
        _atomic_write_text(
            directory / META_NAME,
            json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        )

    def _write_note_order(self, directory: Path, names: list[str]) -> None:
        meta = self._read_meta(directory)
        meta["note_order"] = names
        self._write_meta(directory, meta)

    def _remember_note_order(self, directory: Path, notes: list[NoteInfo]) -> None:
        names = [item.name for item in notes]
        if self._read_meta(directory).get("note_order") == names:
            return
        self._write_note_order(directory, names)

    def _rename_note_order(self, directory: Path, old_name: str, new_name: str) -> None:
        meta = self._read_meta(directory)
        raw = meta.get("note_order")
        if not isinstance(raw, list):
            return
        meta["note_order"] = [new_name if str(item) == old_name else str(item) for item in raw]
        self._write_meta(directory, meta)

    def _ordered_names(self, directory: Path, names: list[str]) -> list[str]:
        known = set(names)
        stored = self._read_meta(directory).get("note_order")
        ordered: list[str] = []
        seen: set[str] = set()
        if isinstance(stored, list):
            for item in stored:
                candidate = str(item or "")
                if candidate in known and candidate not in seen:
                    ordered.append(candidate)
                    seen.add(candidate)
        extras = sorted((name for name in names if name not in seen), key=str.casefold)
        return ordered + extras

    def _scan_markdown(self, directory: Path) -> list[NoteInfo]:
        notes: list[NoteInfo] = []
        try:
            entries = list(directory.iterdir())
        except OSError:
            return notes
        for entry in entries:
            if not entry.is_file() or entry.suffix.lower() != ".md":
                continue
            if entry.name.startswith(".") or entry.name.endswith(".tmp"):
                continue
            try:
                size = entry.stat().st_size
            except OSError:
                continue
            notes.append(
                NoteInfo(
                    name=entry.name,
                    title=title_from_filename(entry.name),
                    updated_at=_iso_mtime(entry),
                    size_bytes=size,
                )
            )
        return notes

    def _list_markdown(self, directory: Path) -> list[NoteInfo]:
        notes = self._scan_markdown(directory)
        by_name = {item.name: item for item in notes}
        order = self._ordered_names(directory, list(by_name))
        return [by_name[name] for name in order if name in by_name]

    def _note_path(self, digest: str, filename: str, *, directory: Path | None = None) -> Path:
        safe = sanitize_note_filename(filename)
        folder = directory if directory is not None else self.image_dir(digest)
        path = (folder / safe).resolve()
        if path.name != safe or not _is_inside(path, folder):
            raise NoteError("笔记名称不合法")
        return path

    def _read_document(self, digest: str, path: Path) -> NoteDocument:
        content = path.read_text(encoding="utf-8")
        return NoteDocument(
            name=path.name,
            title=title_from_filename(path.name),
            content=content,
            updated_at=_iso_mtime(path),
            content_hash=digest,
        )

    def _unique_name(self, directory: Path, filename: str) -> str:
        if not (directory / filename).exists():
            return filename
        stem = Path(filename).stem
        suffix = Path(filename).suffix
        index = 2
        while True:
            candidate = f"{stem}-{index}{suffix}"
            if not (directory / candidate).exists():
                return candidate
            index += 1

    def _follow_alias(self, payload: dict[str, Any], digest: str) -> str:
        aliases = payload.get("aliases")
        if not isinstance(aliases, dict):
            return digest
        seen: set[str] = set()
        current = digest
        while current in aliases and current not in seen:
            seen.add(current)
            nxt = str(aliases.get(current) or "").strip().casefold()
            if not HASH_RE.fullmatch(nxt):
                break
            current = nxt
        return current

    def _folder_has_notes(self, digest: str) -> bool:
        directory = self.image_dir(digest)
        if not directory.is_dir():
            return False
        return bool(self._list_markdown(directory))

    def _resolve_canonical_locked(
        self,
        digest: str,
        *,
        phash: str,
        width: int,
        height: int,
        name: str,
        path: str,
        point_id: str,
    ) -> str:
        payload = self._read_index(self.root / INDEX_NAME)
        digest = self._follow_alias(payload, digest)
        images = payload.get("images") if isinstance(payload.get("images"), dict) else {}
        if digest in images and self._folder_has_notes(digest):
            self._update_index(
                digest,
                name=name,
                path=path,
                point_id=point_id,
                note_count=len(self._list_markdown(self.image_dir(digest))),
                phash=phash,
                width=width,
                height=height,
                source_hash=digest,
            )
            return digest
        match = self._nearest_canonical(payload, phash, width=width, height=height)
        if match and match != digest:
            empty = self.image_dir(digest)
            if empty.is_dir() and not self._folder_has_notes(digest):
                shutil.rmtree(empty, ignore_errors=True)
            self._update_index(
                match,
                name=name,
                path=path,
                point_id=point_id,
                note_count=len(self._list_markdown(self.image_dir(match)))
                if self.image_dir(match).is_dir()
                else 0,
                phash=phash,
                width=width,
                height=height,
                source_hash=digest,
            )
            return match
        return digest

    def _nearest_canonical(
        self,
        payload: dict[str, Any],
        phash: str,
        *,
        width: int,
        height: int,
    ) -> str | None:
        folded = str(phash or "").strip().casefold()
        if not PHASH_RE.fullmatch(folded):
            return None
        phashes = payload.get("phashes")
        if not isinstance(phashes, dict) or not phashes:
            return None
        images = payload.get("images") if isinstance(payload.get("images"), dict) else {}
        exact = str(phashes.get(folded) or "").strip().casefold()
        if HASH_RE.fullmatch(exact) and self._phash_geometry_ok(images, exact, width, height):
            return self._follow_alias(payload, exact)
        best: str | None = None
        best_distance = PHASH_MAX_DISTANCE + 1
        for other, canonical in phashes.items():
            key = str(other or "").strip().casefold()
            target = str(canonical or "").strip().casefold()
            if not PHASH_RE.fullmatch(key) or not HASH_RE.fullmatch(target):
                continue
            distance = hamming_distance(folded, key)
            if distance >= best_distance:
                continue
            if not self._phash_geometry_ok(images, target, width, height):
                continue
            best = target
            best_distance = distance
        if best is None or best_distance > PHASH_MAX_DISTANCE:
            return None
        return self._follow_alias(payload, best)

    def _phash_geometry_ok(
        self,
        images: dict[str, Any],
        canonical: str,
        width: int,
        height: int,
    ) -> bool:
        record = images.get(canonical) if isinstance(images.get(canonical), dict) else {}
        other_w = int(record.get("width") or 0) if record else 0
        other_h = int(record.get("height") or 0) if record else 0
        return aspect_ratio_compatible(width, height, other_w, other_h)

    def _update_index(
        self,
        digest: str,
        *,
        name: str,
        path: str,
        point_id: str,
        note_count: int,
        phash: str = "",
        width: int = 0,
        height: int = 0,
        source_hash: str = "",
    ) -> None:
        index_path = self.root / INDEX_NAME
        payload = self._read_index(index_path)
        images = payload.setdefault("images", {})
        points = payload.setdefault("points", {})
        phashes = payload.setdefault("phashes", {})
        aliases = payload.setdefault("aliases", {})
        current = images.get(digest) if isinstance(images.get(digest), dict) else {}
        stored_point = point_id or str(current.get("point_id") or "")
        previous_point = str(current.get("point_id") or "")
        stored_phash = str(phash or current.get("phash") or "").strip().casefold()
        record = {
            "name": name or str(current.get("name") or ""),
            "path": path or str(current.get("path") or ""),
            "point_id": stored_point,
            "note_count": note_count,
            "phash": stored_phash,
            "width": width or int(current.get("width") or 0),
            "height": height or int(current.get("height") or 0),
        }
        images[digest] = record
        if (
            previous_point
            and previous_point != stored_point
            and points.get(previous_point) == digest
        ):
            points.pop(previous_point, None)
        if stored_point:
            points[stored_point] = digest
        if PHASH_RE.fullmatch(stored_phash):
            phashes[stored_phash] = digest
        source = str(source_hash or "").strip().casefold()
        if HASH_RE.fullmatch(source) and source != digest:
            aliases[source] = digest
            if stored_point:
                points[stored_point] = digest
        payload["version"] = NOTES_INDEX_VERSION
        payload["images"] = images
        payload["points"] = points
        payload["phashes"] = phashes
        payload["aliases"] = aliases
        _atomic_write_text(index_path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")

    def _drop_index(self, digest: str) -> None:
        index_path = self.root / INDEX_NAME
        payload = self._read_index(index_path)
        images = payload.setdefault("images", {})
        points = payload.setdefault("points", {})
        current = images.pop(digest, None)
        if isinstance(current, dict):
            point_id = str(current.get("point_id") or "")
            if point_id and points.get(point_id) == digest:
                points.pop(point_id, None)
        stale = [key for key, value in points.items() if value == digest]
        for key in stale:
            points.pop(key, None)
        aliases = payload.setdefault("aliases", {})
        phashes = payload.setdefault("phashes", {})
        stale_aliases = [key for key, value in aliases.items() if value == digest]
        for key in stale_aliases:
            aliases.pop(key, None)
        stale_phashes = [key for key, value in phashes.items() if value == digest]
        for key in stale_phashes:
            phashes.pop(key, None)
        payload["images"] = images
        payload["points"] = points
        payload["aliases"] = aliases
        payload["phashes"] = phashes
        payload["version"] = NOTES_INDEX_VERSION
        _atomic_write_text(index_path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")

    def _read_index(self, index_path: Path) -> dict[str, Any]:
        empty = {
            "version": NOTES_INDEX_VERSION,
            "images": {},
            "points": {},
            "phashes": {},
            "aliases": {},
        }
        if not index_path.exists():
            return empty
        try:
            data = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return empty
        if not isinstance(data, dict):
            return empty
        images = data.get("images")
        if not isinstance(images, dict):
            images = {}
            data["images"] = images
        points = data.get("points")
        if not isinstance(points, dict):
            points = {}
        phashes = data.get("phashes")
        if not isinstance(phashes, dict):
            phashes = {}
        aliases = data.get("aliases")
        if not isinstance(aliases, dict):
            aliases = {}
        for digest, record in images.items():
            if not isinstance(record, dict):
                continue
            folded = str(digest).casefold()
            point_id = str(record.get("point_id") or "").strip()
            if point_id and HASH_RE.fullmatch(folded) and point_id not in points:
                points[point_id] = folded
            stored_phash = str(record.get("phash") or "").strip().casefold()
            if PHASH_RE.fullmatch(stored_phash) and stored_phash not in phashes:
                phashes[stored_phash] = folded
        data["points"] = points
        data["phashes"] = phashes
        data["aliases"] = aliases
        return data

    def _delete_unreferenced_assets(self, directory: Path, relatives: set[str]) -> list[str]:
        assets_root = (directory / ASSETS_DIR).resolve()
        removed: list[str] = []
        for relative in sorted(relatives):
            path = (directory / relative).resolve()
            if not _is_inside(path, assets_root):
                continue
            if not path.is_file():
                continue
            path.unlink()
            removed.append(relative)
        try:
            if assets_root.is_dir() and not any(assets_root.iterdir()):
                assets_root.rmdir()
        except OSError:
            pass
        return removed

    def _check_content_size(self, content: str) -> None:
        if len(content.encode("utf-8")) > MAX_NOTE_BYTES:
            raise NoteError("笔记内容过大")

    def _sanitize_asset_name(self, filename: str) -> str:
        raw = Path(str(filename or "")).name.strip()
        raw = FORBIDDEN_NAME_CHARS.sub("_", raw)
        raw = raw.strip(" .")
        if not raw or raw in {".", ".."}:
            raw = "paste.bin"
        stem = Path(raw).stem.casefold()
        if stem in RESERVED_DEVICE_NAMES:
            raw = f"_{raw}"
        if raw.startswith("."):
            raw = f"_{raw.lstrip('.')}" or "paste.bin"
        if len(raw) > 120:
            suffix = Path(raw).suffix[:12]
            raw = raw[: 120 - len(suffix)] + suffix
        return raw
