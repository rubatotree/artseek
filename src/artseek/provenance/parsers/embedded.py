"""EXIF/XMP/PNG embedded metadata extraction for image provenance."""

from __future__ import annotations

import html
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from .filename import parse_filename
from .urls import (
    _PIXIV_URL_RES,
    _TWITTER_HANDLE_RE,
    _TWITTER_RESERVED_HANDLES,
    _TWITTER_URL_RE,
    ParsedSource,
    SourceRef,
    canonical_pixiv_url,
    twitter_photo_url,
)


def _decode(value: object) -> str:
    if not isinstance(value, bytes):
        return str(value)
    data = value
    if not data:
        return ""
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        try:
            return data.decode("utf-16").rstrip("\x00")
        except UnicodeDecodeError:
            pass
    if len(data) >= 4 and len(data) % 2 == 0:
        even_nuls = data[0::2].count(0)
        odd_nuls = data[1::2].count(0)
        half = len(data) // 2
        if odd_nuls > half * 0.4 and even_nuls < half * 0.1:
            return data.decode("utf-16-le", "replace").rstrip("\x00")
        if even_nuls > half * 0.4 and odd_nuls < half * 0.1:
            return data.decode("utf-16-be", "replace").rstrip("\x00")
    data = data.rstrip(b"\x00")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1", "replace")


def _metadata_strings(path: Path) -> list[str]:
    strings: list[str] = []
    try:
        with Image.open(path) as image:
            for key, value in (image.info or {}).items():
                if key in {"icc_profile", "exif"}:
                    continue
                if isinstance(value, (str, bytes)):
                    text = _decode(value)
                    if len(text) <= 200_000:
                        strings.append(text)
            try:
                exif = image.getexif()
            except Exception:
                exif = None
            if exif:
                for value in exif.values():
                    if isinstance(value, (str, bytes)):
                        strings.append(_decode(value))
                for tag in (0x8769, 0x8825):
                    try:
                        for value in exif.get_ifd(tag).values():
                            if isinstance(value, (str, bytes)):
                                strings.append(_decode(value))
                    except Exception:
                        continue
    except (OSError, UnidentifiedImageError, ValueError):
        return []
    return strings


def twitter_authors_from_metadata(
    path: Path,
) -> tuple[dict[str, set[str]], list[str]]:
    """Extract status-ID->handle mappings and profile handles from metadata."""
    status_handles: dict[str, set[str]] = {}
    profile_handles: list[str] = []
    seen_profiles: set[str] = set()
    for text in _metadata_strings(path):
        decoded = html.unescape(text)
        for match in _TWITTER_URL_RE.finditer(decoded):
            author = (match.group("author") or "").strip().lstrip("@")
            if author and author.casefold() not in _TWITTER_RESERVED_HANDLES:
                status_handles.setdefault(match.group("id"), set()).add(author)
        for match in _TWITTER_HANDLE_RE.finditer(decoded):
            handle = match.group("handle").lstrip("@")
            key = handle.casefold()
            if handle and key not in _TWITTER_RESERVED_HANDLES and key not in seen_profiles:
                seen_profiles.add(key)
                profile_handles.append(handle)
    return status_handles, profile_handles


def extract_embedded_refs(path: Path) -> list[SourceRef]:
    """Return distinct supported URLs found in XMP/EXIF/PNG text metadata."""
    refs: list[SourceRef] = []
    seen: set[tuple[str, str, int | None]] = set()
    for text in _metadata_strings(path):
        for match in _TWITTER_URL_RE.finditer(html.unescape(text)):
            source_id = match.group("id")
            photo = int(match.group("photo")) if match.group("photo") else None
            key = ("twitter", source_id, photo)
            if key in seen:
                continue
            seen.add(key)
            refs.append(
                SourceRef(
                    platform="twitter",
                    source_id=source_id,
                    page=photo,
                    url=twitter_photo_url(source_id, photo),
                    origin="embedded",
                    confidence=0.95,
                )
            )
        for pattern in _PIXIV_URL_RES:
            for match in pattern.finditer(html.unescape(text)):
                source_id = match.group("id")
                key = ("pixiv", source_id, None)
                if key in seen:
                    continue
                seen.add(key)
                refs.append(
                    SourceRef(
                        platform="pixiv",
                        source_id=source_id,
                        page=None,
                        url=canonical_pixiv_url(source_id),
                        origin="embedded",
                        confidence=0.95,
                    )
                )
    refs.sort(key=lambda item: (item.platform, item.source_id, item.page or 0))
    return refs


def parsed_sources(
    path: Path, *, embedded: bool = True, filename: bool = True
) -> ParsedSource | None:
    """Prefer embedded URLs, then the filename, while keeping conflicts visible."""
    embedded_refs = extract_embedded_refs(path) if embedded else []
    filename_ref = parse_filename(path) if filename else None
    if embedded_refs:
        primary = embedded_refs[0]
        candidates = tuple(
            ref
            for ref in (*embedded_refs[1:], filename_ref)
            if ref is not None
            and (ref.platform, ref.source_id, ref.page)
            != (primary.platform, primary.source_id, primary.page)
        )
        return ParsedSource(primary=primary, candidates=candidates)
    if filename_ref is not None:
        return ParsedSource(primary=filename_ref)
    return None
