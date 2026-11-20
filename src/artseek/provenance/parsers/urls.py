"""Pure URL, ID, and hashtag parsers for Pixiv, X/Twitter, and Xiaohongshu."""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from typing import Literal

Platform = Literal["pixiv", "twitter", "xiaohongshu", "other"]
Origin = Literal["filename", "embedded", "manual", "saucy", "ascii2d"]

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
_XIAOHONGSHU_NOTE_URL_RE = re.compile(
    r"https?://(?:www\.)?xiaohongshu\.com/(?:explore|discovery/item)/(?P<id>[0-9a-f]{24})",
    re.IGNORECASE,
)
_XIAOHONGSHU_USER_URL_RE = re.compile(
    r"https?://(?:www\.)?xiaohongshu\.com/user/profile/(?P<id>[0-9a-f]{24})",
    re.IGNORECASE,
)
_XIAOHONGSHU_ID_RE = re.compile(r"^[0-9a-f]{24}$", re.IGNORECASE)
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


def canonical_xiaohongshu_url(note_id: str) -> str:
    return f"https://www.xiaohongshu.com/explore/{note_id}"


def xiaohongshu_user_url(user_id: str) -> str:
    return f"https://www.xiaohongshu.com/user/profile/{user_id}"


def twitter_photo_url(source_id: str, photo: int | None) -> str:
    if photo:
        return f"https://x.com/i/status/{source_id}/photo/{photo}"
    return canonical_twitter_url(source_id)


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
    match = _XIAOHONGSHU_NOTE_URL_RE.search(text)
    if match:
        source_id = match.group("id")
        return SourceRef(
            platform="xiaohongshu",
            source_id=source_id,
            page=None,
            url=canonical_xiaohongshu_url(source_id),
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
            url=text if ref.platform in {"other", "xiaohongshu"} else ref.url,
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
    if _XIAOHONGSHU_ID_RE.fullmatch(text):
        return SourceRef(
            platform="xiaohongshu",
            source_id=text.casefold(),
            page=None,
            url=canonical_xiaohongshu_url(text.casefold()),
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


def pixiv_user_id_from_url(value: str) -> str | None:
    match = _PIXIV_USER_RE.search(html.unescape(str(value or "")))
    return match.group("id") if match else None


def xiaohongshu_user_id(value: str | None) -> str | None:
    """Return the 24-hex user ID from a bare ID or profile URL."""
    text = html.unescape(str(value or "")).strip()
    match = _XIAOHONGSHU_USER_URL_RE.search(text)
    if match:
        return match.group("id").casefold()
    return text.casefold() if _XIAOHONGSHU_ID_RE.fullmatch(text) else None


def xiaohongshu_note_id(value: str | None) -> str | None:
    text = html.unescape(str(value or "")).strip()
    match = _XIAOHONGSHU_NOTE_URL_RE.search(text)
    if match:
        return match.group("id").casefold()
    return text.casefold() if _XIAOHONGSHU_ID_RE.fullmatch(text) else None


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


def platform_label(platform: str) -> str:
    return {
        "pixiv": "Pixiv",
        "twitter": "X/Twitter",
        "xiaohongshu": "小红书",
        "other": "其他",
    }.get(platform, platform)
