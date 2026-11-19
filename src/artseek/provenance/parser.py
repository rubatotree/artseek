"""Offline source parsing for illustration filenames and embedded metadata.

The parser never touches the network and never rewrites an image. It returns
the best-known source reference plus lower-priority candidates so callers can
surface conflicts instead of silently overwriting one convention with another.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from PIL import Image, UnidentifiedImageError

Platform = Literal["pixiv", "twitter", "other"]
Origin = Literal["filename", "embedded", "manual", "saucy", "ascii2d"]

PIXIV_FILENAME_RE = re.compile(
    r"^(?P<id>\d{5,10})"
    r"(?:[-_][0-9a-fA-F]{8,})?"
    r"_p(?P<page>\d+)"
    r"(?:[ _](?:master\d+|square\d+|.*))?$"
)
TWITTER_FILENAME_RE = re.compile(r"^(?P<id>\d{15,19})_(?P<index>\d+)(?:[ _].*)?$")
PIXIV_NAMED_RE = re.compile(r"pixiv[\s_-]*id[\s_-]*(?P<id>\d+)", re.IGNORECASE)
DUPLICATE_SUFFIX_RE = re.compile(r"\s*\(\d+\)$")

_PIXIV_URL_RES = (
    re.compile(r"https?://(?:www\.)?pixiv\.net/(?:en/)?artworks/(?P<id>\d+)", re.I),
    re.compile(r"https?://(?:www\.)?pixiv\.net/(?:en/)?i/(?P<id>\d+)", re.I),
    re.compile(
        r"https?://(?:www\.)?pixiv\.net/member_illust\.php"
        r"[^\s\"'<>]*?illust_id=(?P<id>\d+)",
        re.I,
    ),
)
_TWITTER_URL_RE = re.compile(
    r"https?://(?:www\.|mobile\.)?"
    r"(?:x\.com|twitter\.com|vxtwitter\.com|fxtwitter\.com|fixupx\.com|fixvx\.com)"
    r"/(?:(?P<author>[A-Za-z0-9_]{1,15})|i/web)/status/"
    r"(?P<id>\d+)(?:/photo/(?P<photo>\d+))?",
    re.IGNORECASE,
)
_TWITTER_RESERVED_HANDLES = frozenset({"i", "home", "intent", "share", "search"})
_TWITTER_HANDLE_RE = re.compile(
    r"https?://(?:www\.)?(?:twitter\.com|x\.com)/(?P<handle>[A-Za-z0-9_]{1,15})",
    re.IGNORECASE,
)
_PIXIV_USER_RE = re.compile(
    r"https?://(?:www\.)?pixiv\.net/(?:en/)?users/(?P<id>\d+)",
    re.IGNORECASE,
)
_HASHTAG_RE = re.compile(r"(?<!\w)#([^\s#.,，。;；:：!?！？()（）\[\]【】]+)")


@dataclass(frozen=True, slots=True)
class SourceRef:
    platform: Platform
    source_id: str
    page: int | None
    url: str
    origin: Origin = "filename"
    confidence: float = 0.9

    @property
    def key(self) -> str:
        return f"{self.platform}:{self.source_id}"


@dataclass(frozen=True, slots=True)
class ParsedSource:
    """A primary reference plus any additional distinct references."""

    primary: SourceRef
    candidates: tuple[SourceRef, ...] = ()

    @property
    def all(self) -> tuple[SourceRef, ...]:
        return (self.primary, *self.candidates)


def canonical_pixiv_url(source_id: str) -> str:
    return f"https://www.pixiv.net/artworks/{source_id}"


def canonical_twitter_url(source_id: str) -> str:
    return f"https://x.com/i/status/{source_id}"


def twitter_photo_url(source_id: str, photo: int | None) -> str:
    if photo:
        return f"https://x.com/i/status/{source_id}/photo/{photo}"
    return canonical_twitter_url(source_id)


def strip_duplicate_suffix(stem: str) -> str:
    return DUPLICATE_SUFFIX_RE.sub("", stem)


def _stem(value: str | Path) -> str:
    text = value.name if isinstance(value, Path) else str(value)
    return strip_duplicate_suffix(Path(text).stem)


def parse_filename(value: str | Path) -> SourceRef | None:
    """Parse a Pixiv or X/Twitter download name.

    Pixiv pages are 0-based; X photo indexes are 1-based. The returned `page`
    preserves that native convention and `None` means "unknown".
    """
    stem = _stem(value)
    match = PIXIV_FILENAME_RE.match(stem)
    if match:
        source_id = match.group("id")
        return SourceRef(
            platform="pixiv",
            source_id=source_id,
            page=int(match.group("page")),
            url=canonical_pixiv_url(source_id),
            origin="filename",
            confidence=0.9,
        )
    match = TWITTER_FILENAME_RE.match(stem)
    if match:
        source_id = match.group("id")
        index = int(match.group("index"))
        return SourceRef(
            platform="twitter",
            source_id=source_id,
            page=index if index > 0 else None,
            url=twitter_photo_url(source_id, index if index > 0 else None),
            origin="filename",
            confidence=0.9,
        )
    match = PIXIV_NAMED_RE.search(stem)
    if match:
        source_id = match.group("id")
        page_match = re.search(r"[ _]p(\d+)\b", stem, re.IGNORECASE)
        return SourceRef(
            platform="pixiv",
            source_id=source_id,
            page=int(page_match.group(1)) if page_match else None,
            url=canonical_pixiv_url(source_id),
            origin="filename",
            confidence=0.6,
        )
    return None


def parse_url(value: str) -> SourceRef | None:
    """Normalize one supported source URL (or a string containing one)."""
    if not value:
        return None
    text = html.unescape(str(value)).strip()
    for pattern in _PIXIV_URL_RES:
        match = pattern.search(text)
        if match:
            source_id = match.group("id")
            return SourceRef(
                platform="pixiv",
                source_id=source_id,
                page=None,
                url=canonical_pixiv_url(source_id),
                origin="embedded",
                confidence=0.95,
            )
    match = _TWITTER_URL_RE.search(text)
    if match:
        source_id = match.group("id")
        photo = int(match.group("photo")) if match.group("photo") else None
        return SourceRef(
            platform="twitter",
            source_id=source_id,
            page=photo if photo and photo > 0 else None,
            url=twitter_photo_url(source_id, photo),
            origin="embedded",
            confidence=0.95,
        )
    return None


def parse_source_text(value: str, *, origin: Origin = "manual") -> SourceRef | None:
    """Parse user input, accepting a bare ID or a supported URL."""
    text = str(value or "").strip()
    if not text:
        return None
    ref = parse_url(text)
    if ref is not None:
        return SourceRef(
            platform=ref.platform,
            source_id=ref.source_id,
            page=ref.page,
            url=text if ref.platform == "other" else ref.url,
            origin=origin,
            confidence=1.0,
        )
    if text.startswith(("http://", "https://")):
        return SourceRef(
            platform="other",
            source_id=text,
            page=None,
            url=text,
            origin=origin,
            confidence=0.8,
        )
    if re.fullmatch(r"\d{5,10}(?:_p\d+)?", text):
        source_id, _, page = text.partition("_p")
        return SourceRef(
            platform="pixiv",
            source_id=source_id,
            page=int(page) if page else None,
            url=canonical_pixiv_url(source_id),
            origin=origin,
            confidence=1.0,
        )
    if re.fullmatch(r"\d{15,19}(?:_\d+)?", text):
        source_id, _, index = text.partition("_")
        photo = int(index) if index else None
        return SourceRef(
            platform="twitter",
            source_id=source_id,
            page=photo,
            url=twitter_photo_url(source_id, photo),
            origin=origin,
            confidence=1.0,
        )
    return None


def twitter_handle_from_url(value: str) -> str | None:
    """Extract an X/Twitter screen name from a profile URL."""
    match = _TWITTER_HANDLE_RE.search(html.unescape(str(value or "")))
    if not match:
        return None
    handle = match.group("handle")
    if handle.casefold() in _TWITTER_RESERVED_HANDLES:
        return None
    return handle


def twitter_authors_from_metadata(
    path: Path,
) -> tuple[dict[str, set[str]], list[str]]:
    """Extract status-ID→handle mappings and profile handles from metadata."""
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


def pixiv_user_id_from_url(value: str) -> str | None:
    match = _PIXIV_USER_RE.search(html.unescape(str(value or "")))
    return match.group("id") if match else None


def hashtags(value: str) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for match in _HASHTAG_RE.finditer(value or ""):
        tag = match.group(1).strip()
        key = tag.casefold()
        if tag and key not in seen:
            seen.add(key)
            result.append(tag)
    return result


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
    # Windows XPComment/XPTitle tags are UTF-16LE without a BOM. UTF-8 would
    # happily decode the interleaved NULs into an unusable string, so detect
    # the alternating-zero pattern before stripping the terminator: one trailing
    # NUL belongs to the UTF-16 code unit and must not make the length odd.
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
            if ref is not None and (ref.platform, ref.source_id, ref.page) != (
                primary.platform,
                primary.source_id,
                primary.page,
            )
        )
        return ParsedSource(primary=primary, candidates=candidates)
    if filename_ref is not None:
        return ParsedSource(primary=filename_ref)
    return None


def platform_label(platform: str) -> str:
    return {"pixiv": "Pixiv", "twitter": "X/Twitter", "other": "其他"}.get(platform, platform)
