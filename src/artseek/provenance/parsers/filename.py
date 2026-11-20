"""Pure filename parsers for Pixiv, X/Twitter, and Xiaohongshu downloads."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .urls import SourceRef, canonical_pixiv_url, twitter_photo_url

PIXIV_FILENAME_RE = re.compile(
    r"^(?P<id>\d{5,10})"
    r"(?:[-_][0-9a-fA-F]{8,})?"
    r"_p(?P<page>\d+)"
    r"(?:[ _](?:master\d+|square\d+|.*))?$"
)
TWITTER_FILENAME_RE = re.compile(r"^(?P<id>\d{15,19})_(?P<index>\d+)(?:[ _].*)?$")
XIAOHONGSHU_FILENAME_RE = re.compile(
    r"^(?P<title>.+)_(?P<index>\d+)_(?P<author>[^_]+?)_来自小红书(?:网页版)?$"
)
PIXIV_NAMED_RE = re.compile(r"pixiv[\s_-]*id[\s_-]*(?P<id>\d+)", re.IGNORECASE)
DUPLICATE_SUFFIX_RE = re.compile(r"\s*\(\d+\)$")


def strip_duplicate_suffix(stem: str) -> str:
    return DUPLICATE_SUFFIX_RE.sub("", stem)


def _stem(value: str | Path) -> str:
    text = value.name if isinstance(value, Path) else str(value)
    return strip_duplicate_suffix(Path(text).stem)


@dataclass(frozen=True, slots=True)
class XiaohongshuName:
    """What a Xiaohongshu download name still tells us, without a note ID."""

    title: str
    index: int
    author: str


def parse_xiaohongshu_filename(value: str | Path) -> XiaohongshuName | None:
    """Parse the Xiaohongshu web downloader's ``title_index_author`` name."""
    stem = _stem(value)
    match = XIAOHONGSHU_FILENAME_RE.match(stem)
    if not match:
        return None
    title = match.group("title").strip()
    author = match.group("author").strip()
    if not title or not author:
        return None
    return XiaohongshuName(title=title, index=int(match.group("index")), author=author)


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
