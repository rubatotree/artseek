from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
import unicodedata
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core import name_match
from ..core.name_match import NameKey
from ..core.transliterate import fold_text
from .parser import (
    ParsedSource,
    SourceRef,
    XiaohongshuName,
    canonical_pixiv_url,
    twitter_photo_url,
    xiaohongshu_user_id,
    xiaohongshu_user_url,
)

SCHEMA_VERSION = 5
ORIGIN_PRIORITY = {"filename": 1, "embedded": 2, "manual": 3, "saucy": 4, "ascii2d": 4}
AUTHOR_TTL_SECONDS = 30 * 24 * 60 * 60
# Identity kinds form one shared namespace per kind, so a handle discovered on
# Pixiv and the same handle discovered on X resolve to a single artist row.
IDENTITY_PIXIV = "pixiv_id"
IDENTITY_TWITTER = "x_id"
IDENTITY_HANDLE = "handle"
IDENTITY_NAME = "name"
IDENTITY_NAME_ALIAS = "name_alias"
IDENTITY_LINK = "link"
IDENTITY_COVER = "cover"
IDENTITY_XIAOHONGSHU = "xhs_uid"
UNASSIGNED_AUTHOR_KEY = "none"
_STATUSES = frozenset(
    {"pending", "ok", "not_found", "protected", "rate_limited", "error", "unresolved", "stale"}
)


def _now() -> int:
    return int(time.time())


def _json(value: Any, fallback: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return json.dumps(fallback, ensure_ascii=False)


def _loads(value: Any, fallback: Any) -> Any:
    if value in (None, ""):
        return fallback
    try:
        return json.loads(str(value))
    except (TypeError, ValueError):
        return fallback


def _normalized_handle(value: str | None) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).strip().lstrip("@")
    return re.sub(r"\s+", " ", text).casefold()


def _author_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = text.lstrip("@")
    return re.sub(r"\s+", " ", text).strip()


@dataclass(slots=True)
class _ArtistSearchIndex:
    """Folded + pinyin/romaji keys for every artist name and identity.

    Built lazily from ``artist`` / ``artist_identity`` and reused until the
    tables change (see ``ProvenanceStore._artist_search_index``).
    """

    signature: tuple[int, int, int, int, int]
    built_at: float
    values: dict[int, tuple[NameKey, ...]]
    entries: tuple[tuple[str, set[int]], ...]
    initials: dict[str, set[int]]

    @property
    def newest_id(self) -> int:
        return max(self.values) if self.values else 0

    def candidates(self, needle: NameKey) -> set[int]:
        """Artist ids whose folded/romanized keys overlap the query keys."""
        found: set[int] = set()
        for wanted in name_match.search_keys(needle):
            for key, artist_ids in self.entries:
                if wanted in key:
                    found.update(artist_ids)
        return found

    def typo_candidates(self, needle: NameKey) -> set[int]:
        """Artist ids sharing the query's first folded character (typo pass)."""
        initial = needle.compact[:1]
        return set(self.initials.get(initial, ())) if initial else set()


def normalize_pixiv_user_id(value: str | None) -> str:
    """Accept a bare Pixiv user ID or a profile URL."""
    text = unicodedata.normalize("NFKC", str(value or "")).strip()
    if not text:
        return ""
    if text.casefold().startswith("pixiv:"):
        text = text.split(":", 1)[1].strip()
    match = re.search(r"pixiv\.net/(?:en/)?users/(\d+)", text, re.IGNORECASE)
    if match:
        return match.group(1)
    return text if text.isdigit() else ""


def normalize_twitter_handle(value: str | None) -> str:
    """Accept a bare X handle, @handle, or an x.com/twitter.com profile URL."""
    text = unicodedata.normalize("NFKC", str(value or "")).strip()
    if not text:
        return ""
    match = re.search(
        r"(?:x|twitter)\.com/(?:#!/)?@?([A-Za-z0-9_]{1,15})", text, re.IGNORECASE
    )
    if match:
        text = match.group(1)
    return text.lstrip("@").strip()


def normalize_xiaohongshu_user_id(value: str | None) -> str:
    """Accept a bare 24-hex Xiaohongshu user ID or a profile URL."""
    return xiaohongshu_user_id(value) or ""


def normalize_social_link(value: str | None) -> str:
    """Return a canonical http(s) URL, or an empty string when invalid."""
    from urllib.parse import urlsplit, urlunsplit

    text = unicodedata.normalize("NFKC", str(value or "")).strip()
    if not text:
        return ""
    try:
        parts = urlsplit(text)
    except ValueError:
        return ""
    if parts.scheme.casefold() not in {"http", "https"} or not parts.netloc:
        return ""
    path = parts.path.rstrip("/")
    return urlunsplit(
        (
            parts.scheme.casefold(),
            parts.netloc.casefold(),
            path,
            parts.query,
            "",
        )
    )


def _identity_kind_platform(kind: str) -> str:
    return {
        IDENTITY_PIXIV: "pixiv",
        IDENTITY_TWITTER: "twitter",
        IDENTITY_HANDLE: "twitter",
        IDENTITY_NAME: "manual",
        IDENTITY_NAME_ALIAS: "name",
        IDENTITY_LINK: "link",
        IDENTITY_COVER: "cover",
        IDENTITY_XIAOHONGSHU: "xiaohongshu",
    }.get(kind, "other")


class ProvenanceStore:
    """Transactional, user-owned source metadata keyed by Qdrant point ID.

    The store is deliberately independent of the vector payload. It survives a
    re-index, a model change, and an index-directory move, and it never mutates
    an original image.
    """

    def __init__(self, path: Path):
        self.path = path
        self._init_caches()

    def _init_caches(self) -> None:
        """Initialise the schema guards shared by the store subclasses."""
        self._schema_lock = threading.Lock()
        self._schema_ready = False
        self._artist_index_lock = threading.Lock()
        self._artist_index: _ArtistSearchIndex | None = None

    def _ensure_artist_index_cache(self) -> threading.Lock:
        """Create the artist index cache on demand.

        ``ProvenanceRepo`` builds its own connection and never runs
        ``ProvenanceStore.__init__``, so the cache fields have to be created
        lazily instead of assuming construction went through this class.
        """
        lock = getattr(self, "_artist_index_lock", None)
        if lock is None:
            lock = threading.Lock()
            self._artist_index_lock = lock
            self._artist_index = None
        return lock

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(self.path), timeout=15)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 15000")
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA temp_store = MEMORY")
        try:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = NORMAL")
        except sqlite3.DatabaseError:
            pass
        return connection

    def _ensure_schema(self, connection: sqlite3.Connection) -> None:
        if self._schema_ready:
            return
        with self._schema_lock:
            if self._schema_ready:
                return
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS image_source (
                    point_id TEXT PRIMARY KEY,
                    content_hash TEXT,
                    path_key TEXT,
                    platform TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    page INTEGER,
                    url TEXT NOT NULL,
                    candidates_json TEXT,
                    status TEXT NOT NULL DEFAULT 'pending',
                    confidence REAL NOT NULL DEFAULT 0.9,
                    title TEXT,
                    description TEXT,
                    author_key TEXT,
                    posted_at TEXT,
                    width INTEGER,
                    height INTEGER,
                    mtime_ns INTEGER,
                    page_count INTEGER,
                    tags_json TEXT,
                    manual_tags_json TEXT,
                    auto_tags_json TEXT,
                    auto_tag_scores_json TEXT,
                    auto_tagged_at INTEGER,
                    raw_json TEXT,
                    source_origin TEXT NOT NULL DEFAULT 'filename',
                    manual_fields TEXT,
                    fetched_at INTEGER,
                    error TEXT,
                    updated_at INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_source_hash
                    ON image_source(content_hash);
                CREATE INDEX IF NOT EXISTS idx_source_author
                    ON image_source(author_key);
                CREATE INDEX IF NOT EXISTS idx_source_platform_status
                    ON image_source(platform, status);
                CREATE TABLE IF NOT EXISTS artist (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    name_key TEXT,
                    name_source TEXT,
                    avatar_url TEXT,
                    note TEXT,
                    profile_fetched INTEGER NOT NULL DEFAULT 0,
                    image_count INTEGER NOT NULL DEFAULT 0,
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_artist_name_key
                    ON artist(name_key);
                CREATE INDEX IF NOT EXISTS idx_artist_image_count
                    ON artist(image_count DESC, id);
                CREATE TABLE IF NOT EXISTS artist_identity (
                    kind TEXT NOT NULL,
                    identity TEXT NOT NULL,
                    display TEXT,
                    profile_url TEXT,
                    source TEXT,
                    artist_id INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL,
                    sort_order INTEGER,
                    inactive INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (kind, identity)
                );
                CREATE INDEX IF NOT EXISTS idx_artist_identity_artist
                    ON artist_identity(artist_id);
                CREATE TABLE IF NOT EXISTS artist_legacy (
                    legacy_key TEXT PRIMARY KEY,
                    artist_id INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_artist_legacy_artist
                    ON artist_legacy(artist_id);
                CREATE TRIGGER IF NOT EXISTS trg_artist_count_insert
                AFTER INSERT ON image_source
                WHEN NEW.author_key IS NOT NULL AND NEW.author_key <> ''
                BEGIN
                    UPDATE artist SET image_count = image_count + 1
                    WHERE id = CAST(NEW.author_key AS INTEGER);
                END;
                CREATE TRIGGER IF NOT EXISTS trg_artist_count_update
                AFTER UPDATE OF author_key ON image_source
                WHEN OLD.author_key IS NOT NEW.author_key
                BEGIN
                    UPDATE artist SET image_count = MAX(image_count - 1, 0)
                    WHERE OLD.author_key IS NOT NULL
                      AND id = CAST(OLD.author_key AS INTEGER);
                    UPDATE artist SET image_count = image_count + 1
                    WHERE NEW.author_key IS NOT NULL AND NEW.author_key <> ''
                      AND id = CAST(NEW.author_key AS INTEGER);
                END;
                CREATE TRIGGER IF NOT EXISTS trg_artist_count_delete
                AFTER DELETE ON image_source
                WHEN OLD.author_key IS NOT NULL AND OLD.author_key <> ''
                BEGIN
                    UPDATE artist SET image_count = MAX(image_count - 1, 0)
                    WHERE id = CAST(OLD.author_key AS INTEGER);
                END;
                CREATE TABLE IF NOT EXISTS fetch_state (
                    domain TEXT PRIMARY KEY,
                    blocked_until INTEGER,
                    last_request REAL,
                    day TEXT,
                    day_count INTEGER,
                    long_remaining INTEGER,
                    last_error TEXT
                );
                """
            )
            source_columns = {
                str(row[1])
                for row in connection.execute("PRAGMA table_info(image_source)").fetchall()
            }
            artist_columns = {
                str(row[1])
                for row in connection.execute("PRAGMA table_info(artist)").fetchall()
            }
            identity_columns = {
                str(row[1])
                for row in connection.execute(
                    "PRAGMA table_info(artist_identity)"
                ).fetchall()
            }
            if "sort_order" not in identity_columns:
                # Manual display order of social accounts on the artist card.
                connection.execute("ALTER TABLE artist_identity ADD COLUMN sort_order INTEGER")
            if "inactive" not in identity_columns:
                # Inactive accounts stay searchable but are hidden next to names.
                connection.execute(
                    "ALTER TABLE artist_identity ADD COLUMN inactive INTEGER NOT NULL DEFAULT 0"
                )
            if "name_source" not in artist_columns:
                connection.execute("ALTER TABLE artist ADD COLUMN name_source TEXT")
            if "mtime_ns" not in source_columns:
                connection.execute("ALTER TABLE image_source ADD COLUMN mtime_ns INTEGER")
            if "manual_tags_json" not in source_columns:
                connection.execute("ALTER TABLE image_source ADD COLUMN manual_tags_json TEXT")
            if "auto_tags_json" not in source_columns:
                connection.execute("ALTER TABLE image_source ADD COLUMN auto_tags_json TEXT")
            if "auto_tag_scores_json" not in source_columns:
                connection.execute("ALTER TABLE image_source ADD COLUMN auto_tag_scores_json TEXT")
            if "auto_tagged_at" not in source_columns:
                connection.execute("ALTER TABLE image_source ADD COLUMN auto_tagged_at INTEGER")
            version_row = connection.execute(
                "SELECT value FROM meta WHERE key = 'schema_version'"
            ).fetchone()
            previous_version = 0
            if version_row is not None:
                try:
                    previous_version = int(version_row[0])
                except (TypeError, ValueError):
                    previous_version = 0
            if previous_version < 2:
                self._repair_twitter_bio_descriptions(connection)
            if previous_version < 4:
                # Older builds marked R18/login-only Pixiv works as not_found
                # because /pages returned 404. Retry them once with the new
                # metadata-only fallback.
                connection.execute(
                    "UPDATE image_source SET status = 'pending', error = NULL, "
                    "updated_at = ? "
                    "WHERE platform = 'pixiv' AND status = 'not_found'",
                    (_now(),),
                )
            if previous_version < 5:
                self._migrate_author_schema(connection)
            connection.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )
            connection.commit()
            self._schema_ready = True

    @staticmethod
    def _repair_twitter_bio_descriptions(connection: sqlite3.Connection) -> None:
        """Clear the legacy fallback that copied an author bio into tweet text."""
        rows = connection.execute(
            "SELECT point_id, description, raw_json FROM image_source "
            "WHERE platform = 'twitter' AND description IS NOT NULL "
            "AND raw_json IS NOT NULL"
        ).fetchall()
        for row in rows:
            raw = _loads(row["raw_json"], {})
            bio = raw.get("author_description") if isinstance(raw, dict) else None
            if bio and str(bio) == str(row["description"]):
                connection.execute(
                    "UPDATE image_source SET description = NULL, updated_at = ? "
                    "WHERE point_id = ?",
                    (_now(), row["point_id"]),
                )

    def _migrate_author_schema(self, connection: sqlite3.Connection) -> None:
        """Convert the legacy ``author``/``author_alias`` schema to artist IDs.

        Every legacy key becomes a legacy lookup entry.  Identities are attached
        in a shared handle namespace, so two old author rows that share a Pixiv
        ``userAccount`` or an X screen name are merged into one artist.
        """
        exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'author'"
        ).fetchone()
        if exists is None:
            return
        connection.commit()
        self._backup_before_migration(connection)
        alias_table = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'author_alias'"
        ).fetchone()
        aliases: dict[str, str] = {}
        if alias_table is not None:
            aliases = {
                str(row["alias_key"]): str(row["canonical_key"])
                for row in connection.execute(
                    "SELECT alias_key, canonical_key FROM author_alias"
                )
            }
        rows = [dict(row) for row in connection.execute("SELECT * FROM author")]

        def root(key: str) -> str:
            current = key
            seen: set[str] = set()
            for _ in range(16):
                if current in seen:
                    break
                seen.add(current)
                target = aliases.get(current)
                if target is None:
                    break
                current = target
            return current

        groups: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            groups.setdefault(root(str(row["key"])), []).append(row)
        ordered = sorted(
            groups.items(),
            key=lambda item: (
                -sum(1 for row in item[1] if row.get("platform") == "pixiv"),
                -len(item[1]),
                item[0],
            ),
        )
        for root_key, items in ordered:
            specs: list[tuple[str, str, str | None, str | None, str | None]] = []
            name: str | None = None
            pixiv_name: str | None = None
            twitter_name: str | None = None
            manual_name: str | None = None
            avatar: str | None = None
            fetched = False
            keys: list[str] = []
            for row in items:
                keys.append(str(row["key"]))
                platform = str(row.get("platform") or "")
                platform_id = str(row.get("platform_id") or "")
                handle = str(row.get("handle") or "").strip()
                row_name = str(row.get("name") or "").strip() or None
                if platform == "pixiv":
                    specs.extend(
                        self._identity_specs(
                            "pixiv",
                            platform_id,
                            handle=handle or None,
                            profile_url=row.get("profile_url"),
                        )
                    )
                    pixiv_name = pixiv_name or row_name
                elif platform == "twitter":
                    specs.extend(
                        self._identity_specs(
                            "twitter",
                            platform_id,
                            handle=handle or platform_id,
                            profile_url=row.get("profile_url"),
                        )
                    )
                    twitter_name = twitter_name or row_name
                elif platform == "manual":
                    specs.extend(
                        self._identity_specs(
                            "manual", row_name or platform_id, name=row_name
                        )
                    )
                    manual_name = manual_name or row_name
                avatar = avatar or row.get("avatar_url")
                fetched = fetched or bool(row.get("profile_fetched"))
            name = pixiv_name or twitter_name or manual_name
            artist_id = self._ensure_artist_locked(
                connection,
                specs,
                name=name,
                avatar_url=avatar,
                profile_fetched=fetched,
                overwrite_name=False,
                legacy_keys=keys,
                commit=False,
            )
            for key in keys:
                connection.execute(
                    "UPDATE image_source SET author_key = ? WHERE author_key = ?",
                    (str(artist_id), key),
                )
            for alias_key in aliases:
                if root(alias_key) == root_key:
                    self._record_legacy_locked(connection, alias_key, artist_id)
                    connection.execute(
                        "UPDATE image_source SET author_key = ? WHERE author_key = ?",
                        (str(artist_id), alias_key),
                    )
        connection.execute("DROP TABLE IF EXISTS author_alias")
        connection.execute("DROP TABLE IF EXISTS author")
        connection.execute(
            "UPDATE artist SET image_count = ("
            "SELECT COUNT(*) FROM image_source "
            "WHERE image_source.author_key = CAST(artist.id AS TEXT)"
            ")"
        )
        connection.commit()

    def _backup_before_migration(self, connection: sqlite3.Connection) -> None:
        if not self.path.exists():
            return
        stamp = time.strftime("%Y%m%d-%H%M%S")
        target = self.path.with_name(f"{self.path.name}.bak-v4-{stamp}")
        if target.exists():
            return
        try:
            backup = sqlite3.connect(str(target))
            try:
                connection.backup(backup)
            finally:
                backup.close()
        except sqlite3.DatabaseError:
            pass

    @contextmanager
    def _read(self):
        connection = self._connect()
        try:
            self._ensure_schema(connection)
            yield connection
        finally:
            connection.close()

    def get(self, point_id: str) -> dict[str, Any] | None:
        with self._read() as connection:
            row = connection.execute(
                "SELECT * FROM image_source WHERE point_id = ?", (point_id,)
            ).fetchone()
        return self._row_to_record(row) if row is not None else None

    def get_many(self, point_ids: Sequence[str]) -> dict[str, dict[str, Any]]:
        unique = list(dict.fromkeys(item for item in point_ids if item))
        if not unique:
            return {}
        result: dict[str, dict[str, Any]] = {}
        with self._read() as connection:
            for start in range(0, len(unique), 256):
                batch = unique[start : start + 256]
                placeholders = ",".join("?" for _ in batch)
                rows = connection.execute(
                    f"SELECT * FROM image_source WHERE point_id IN ({placeholders})",
                    batch,
                ).fetchall()
                for row in rows:
                    result[str(row["point_id"])] = self._row_to_record(row)
        return result

    def known_points(self) -> dict[str, tuple[str | None, int, int | None, str | None]]:
        with self._read() as connection:
            rows = connection.execute(
                "SELECT point_id, content_hash, updated_at, mtime_ns, path_key "
                "FROM image_source"
            ).fetchall()
        return {
            str(row["point_id"]): (
                row["content_hash"],
                int(row["updated_at"]),
                _int_or_none(row["mtime_ns"]),
                row["path_key"],
            )
            for row in rows
        }

    def upsert_scan(
        self,
        point_id: str,
        payload: dict[str, Any],
        parsed: ParsedSource | None,
        *,
        path_key: str | None = None,
        origin: str = "filename",
    ) -> dict[str, Any] | None:
        """Record the best offline reference, preserving manual and fetched data."""
        if parsed is None:
            return None
        primary = parsed.primary
        now = _now()
        width = _int_or_none(payload.get("width"))
        height = _int_or_none(payload.get("height"))
        mtime_ns = _int_or_none(payload.get("mtime_ns"))
        digest = str(payload.get("content_hash") or "") or None
        with self._read() as connection:
            existing = connection.execute(
                "SELECT * FROM image_source WHERE point_id = ?", (point_id,)
            ).fetchone()
            if existing is not None:
                record = self._row_to_record(existing)
                manual = set(record.get("manual_fields") or [])
                same_source = (
                    record.get("platform") == primary.platform
                    and record.get("source_id") == primary.source_id
                    and record.get("page") == primary.page
                )
                old_priority = ORIGIN_PRIORITY.get(str(record.get("source_origin")), 0)
                new_priority = ORIGIN_PRIORITY.get(origin, 0)
                keep_source = same_source or old_priority > new_priority
                if manual & {"platform", "source_id", "page", "url"}:
                    keep_source = True
                if keep_source:
                    connection.execute(
                        """
                        UPDATE image_source
                        SET content_hash = COALESCE(?, content_hash),
                            path_key = COALESCE(?, path_key),
                            width = COALESCE(?, width),
                            height = COALESCE(?, height),
                            mtime_ns = COALESCE(?, mtime_ns),
                            candidates_json = ?,
                            updated_at = ?
                        WHERE point_id = ?
                        """,
                        (
                            digest,
                            path_key,
                            width,
                            height,
                            mtime_ns,
                            _json([_ref_dict(item) for item in parsed.candidates], []),
                            now,
                            point_id,
                        ),
                    )
                else:
                    connection.execute(
                        """
                        UPDATE image_source
                        SET content_hash = COALESCE(?, content_hash),
                            path_key = COALESCE(?, path_key),
                            platform = ?,
                            source_id = ?,
                            page = ?,
                            url = ?,
                            candidates_json = ?,
                            status = 'pending',
                            confidence = ?,
                            source_origin = ?,
                            title = NULL,
                            description = NULL,
                            author_key = NULL,
                            posted_at = NULL,
                            width = COALESCE(?, width),
                            height = COALESCE(?, height),
                            mtime_ns = COALESCE(?, mtime_ns),
                            page_count = NULL,
                            tags_json = NULL,
                            manual_tags_json = NULL,
                            auto_tags_json = NULL,
                            auto_tag_scores_json = NULL,
                            auto_tagged_at = NULL,
                            raw_json = NULL,
                            fetched_at = NULL,
                            error = NULL,
                            updated_at = ?
                        WHERE point_id = ?
                        """,
                        (
                            digest,
                            path_key,
                            primary.platform,
                            primary.source_id,
                            primary.page,
                            primary.url,
                            _json([_ref_dict(item) for item in parsed.candidates], []),
                            primary.confidence,
                            origin,
                            width,
                            height,
                            mtime_ns,
                            now,
                            point_id,
                        ),
                    )
            else:
                connection.execute(
                    """
                    INSERT INTO image_source (
                        point_id, content_hash, path_key, platform, source_id, page,
                        url, candidates_json, status, confidence, width, height,
                        mtime_ns, source_origin, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        point_id,
                        digest,
                        path_key,
                        primary.platform,
                        primary.source_id,
                        primary.page,
                        primary.url,
                        _json([_ref_dict(item) for item in parsed.candidates], []),
                        primary.confidence,
                        width,
                        height,
                        mtime_ns,
                        origin,
                        now,
                    ),
                )
            connection.commit()
        return self.get(point_id)

    def upsert_unresolved(
        self,
        point_id: str,
        payload: dict[str, Any],
        *,
        path_key: str | None = None,
        hint: XiaohongshuName | None = None,
        author_key: str | None = None,
    ) -> dict[str, Any] | None:
        """Create a visible "unidentified" row without guessing a source.

        A Xiaohongshu download name carries a title, a 1-based image index and
        the author nickname but no note ID.  Those fields are recorded as an
        unresolved Xiaohongshu row so the author group is still usable; the
        note itself stays unidentified until a URL or reverse search resolves it.
        """
        now = _now()
        digest = str(payload.get("content_hash") or "") or None
        width = _int_or_none(payload.get("width"))
        height = _int_or_none(payload.get("height"))
        mtime_ns = _int_or_none(payload.get("mtime_ns"))
        with self._read() as connection:
            existing = connection.execute(
                "SELECT * FROM image_source WHERE point_id = ?", (point_id,)
            ).fetchone()
            if existing is not None:
                manual = set(_loads(existing["manual_fields"], []))
                anonymous = str(existing["platform"] or "") in {"other", "xiaohongshu"} and (
                    not str(existing["source_id"] or "")
                    or str(existing["source_origin"] or "") != "manual"
                )
                adopt_hint = (
                    hint is not None
                    and anonymous
                    and str(existing["status"] or "") in {"unresolved", "pending", "error"}
                    and not manual & {"platform", "source_id", "url", "title", "author_key"}
                )
                if adopt_hint:
                    connection.execute(
                        """
                        UPDATE image_source
                        SET content_hash = COALESCE(?, content_hash),
                            path_key = COALESCE(?, path_key),
                            platform = 'xiaohongshu',
                            source_id = '',
                            page = ?,
                            url = '',
                            title = ?,
                            author_key = COALESCE(?, author_key),
                            status = 'unresolved',
                            confidence = 0.5,
                            source_origin = 'filename',
                            error = NULL,
                            width = COALESCE(?, width),
                            height = COALESCE(?, height),
                            mtime_ns = COALESCE(?, mtime_ns),
                            updated_at = ?
                        WHERE point_id = ?
                        """,
                        (
                            digest,
                            path_key,
                            hint.index,
                            hint.title,
                            author_key,
                            width,
                            height,
                            mtime_ns,
                            now,
                            point_id,
                        ),
                    )
                else:
                    connection.execute(
                        """
                        UPDATE image_source
                        SET content_hash = COALESCE(?, content_hash),
                            path_key = COALESCE(?, path_key),
                            width = COALESCE(?, width),
                            height = COALESCE(?, height),
                            mtime_ns = COALESCE(?, mtime_ns),
                            status = CASE
                                WHEN status = 'pending'
                                 AND platform IN ('other', 'xiaohongshu')
                                THEN 'unresolved'
                                ELSE status
                            END,
                            updated_at = ?
                        WHERE point_id = ?
                        """,
                        (digest, path_key, width, height, mtime_ns, now, point_id),
                    )
            elif hint is not None:
                connection.execute(
                    """
                    INSERT INTO image_source (
                        point_id, content_hash, path_key, platform, source_id,
                        page, url, status, confidence, title, author_key,
                        width, height, mtime_ns, source_origin, updated_at
                    ) VALUES (
                        ?, ?, ?, 'xiaohongshu', ?, ?, '', 'unresolved', 0.5, ?, ?,
                        ?, ?, ?, 'filename', ?
                    )
                    """,
                    (
                        point_id,
                        digest,
                        path_key,
                        "",
                        hint.index,
                        hint.title,
                        author_key,
                        width,
                        height,
                        mtime_ns,
                        now,
                    ),
                )
            else:
                connection.execute(
                    """
                    INSERT INTO image_source (
                        point_id, content_hash, path_key, platform, source_id,
                        page, url, status, confidence, width, height,
                        mtime_ns, source_origin, updated_at
                    ) VALUES (
                        ?, ?, ?, 'other', ?, NULL, '', 'unresolved', 0,
                        ?, ?, ?, 'filename', ?
                    )
                    """,
                    (point_id, digest, path_key, point_id, width, height, mtime_ns, now),
                )
            connection.commit()
        return self.get(point_id)

    def reset_for_replacement(self, point_id: str) -> None:
        """Drop metadata that belonged to the previous bytes at the same path."""
        with self._read() as connection:
            connection.execute(
                """
                UPDATE image_source
                SET content_hash = NULL, title = NULL, description = NULL,
                    author_key = NULL, posted_at = NULL, page_count = NULL,
                    tags_json = NULL, manual_tags_json = NULL,
                    auto_tags_json = NULL, auto_tag_scores_json = NULL,
                    auto_tagged_at = NULL, raw_json = NULL, fetched_at = NULL,
                    manual_fields = NULL, source_origin = 'filename',
                    status = 'pending', error = NULL, updated_at = ?
                WHERE point_id = ?
                """,
                (_now(), point_id),
            )
            connection.commit()

    def set_status(self, point_id: str, status: str, error: str | None = None) -> None:
        if status not in _STATUSES:
            status = "error"
        with self._read() as connection:
            connection.execute(
                "UPDATE image_source SET status = ?, error = ?, updated_at = ? WHERE point_id = ?",
                (status, error, _now(), point_id),
            )
            connection.commit()

    def set_fetch_result(
        self,
        point_id: str,
        result: dict[str, Any],
        *,
        author_key: str | None = None,
    ) -> None:
        page = result.get("page")
        with self._read() as connection:
            existing = connection.execute(
                "SELECT page, manual_fields FROM image_source WHERE point_id = ?",
                (point_id,),
            ).fetchone()
            if existing is None:
                return
            manual = set(_loads(existing["manual_fields"], []))
            if page is None and existing["page"] is not None:
                page = int(existing["page"])
            updates = [
                "status = ?",
                "confidence = ?",
                "posted_at = ?",
                "page_count = ?",
                "raw_json = ?",
                "fetched_at = ?",
                "error = NULL",
                "updated_at = ?",
            ]
            values: list[Any] = [
                result.get("status", "ok"),
                result.get("confidence", 1.0),
                result.get("posted_at"),
                result.get("page_count"),
                _json(result.get("raw") or {}, {}),
                _now(),
                _now(),
            ]
            for field, value in (
                ("platform", result.get("platform")),
                ("source_id", result.get("source_id")),
                ("page", page),
                ("url", result.get("url")),
                ("title", result.get("title")),
                ("description", result.get("description")),
                ("author_key", author_key),
            ):
                if field not in manual:
                    updates.append(f"{field} = ?")
                    values.append(value)
            if "tags" not in manual:
                updates.append("tags_json = ?")
                values.append(_json(result.get("tags") or [], []))
            values.append(point_id)
            connection.execute(
                f"UPDATE image_source SET {', '.join(updates)} WHERE point_id = ?",
                values,
            )
            connection.commit()

    def set_manual(
        self,
        point_id: str,
        fields: dict[str, Any],
        *,
        ref: SourceRef | None = None,
    ) -> dict[str, Any] | None:
        record = self.get(point_id)
        if record is None:
            return None
        manual = set(record.get("manual_fields") or [])
        updates: list[str] = []
        values: list[Any] = []
        if ref is not None:
            updates.extend(["platform", "source_id", "page", "url", "status"])
            values.extend(
                [
                    ref.platform,
                    ref.source_id,
                    ref.page,
                    ref.url,
                    "ok" if ref.platform in {"other", "xiaohongshu"} else "pending",
                ]
            )
            manual.update({"platform", "source_id", "page", "url"})
        for key in ("title", "description", "author_key"):
            if key in fields:
                updates.append(key)
                values.append(fields[key])
                manual.add(key)
        if "tags" in fields:
            updates.append("tags_json")
            values.append(_json(fields["tags"] or [], []))
            manual.add("tags")
        if "manual_tags" in fields:
            updates.append("manual_tags_json")
            values.append(_json(fields["manual_tags"] or [], []))
            manual.add("manual_tags")
        if not updates:
            return record
        updates.extend(["source_origin", "manual_fields", "updated_at"])
        values.extend(["manual", _json(sorted(manual), []), _now()])
        values.append(point_id)
        with self._read() as connection:
            connection.execute(
                f"UPDATE image_source SET {', '.join(f'{item} = ?' for item in updates)} "
                "WHERE point_id = ?",
                values,
            )
            connection.commit()
        return self.get(point_id)

    def clear_manual(self, point_id: str) -> dict[str, Any] | None:
        record = self.get(point_id)
        if record is None:
            return None
        with self._read() as connection:
            connection.execute(
                "UPDATE image_source SET manual_fields = NULL, source_origin = 'filename', "
                "title = NULL, description = NULL, author_key = NULL, "
                "manual_tags_json = NULL, status = 'pending', updated_at = ? WHERE point_id = ?",
                (_now(), point_id),
            )
            connection.commit()
        return self.get(point_id)

    def pending_records(
        self,
        *,
        platforms: Sequence[str] = ("pixiv", "twitter"),
        limit: int | None = None,
        include_errors: bool = True,
        retry_not_found_after: int = 7 * 24 * 60 * 60,
    ) -> list[dict[str, Any]]:
        statuses = ["pending"]
        if include_errors:
            statuses.extend(["error", "rate_limited"])
        placeholders = ",".join("?" for _ in platforms)
        status_placeholders = ",".join("?" for _ in statuses)
        sql = (
            "SELECT * FROM image_source "
            f"WHERE platform IN ({placeholders}) AND ("
            f"status IN ({status_placeholders}) "
            "OR (status = 'not_found' AND updated_at < ?)"
            ") ORDER BY COALESCE(mtime_ns, 0) DESC, updated_at DESC"
        )
        params: list[Any] = [
            *platforms,
            *statuses,
            _now() - max(0, int(retry_not_found_after)),
        ]
        if limit is not None:
            sql += " LIMIT ?"
            params.append(max(0, int(limit)))
        with self._read() as connection:
            rows = connection.execute(sql, params).fetchall()
        return [self._row_to_record(row) for row in rows]

    def stale_points(self, live_hashes: dict[str, str | None]) -> list[str]:
        """Mark replaced paths stale and return their IDs."""
        stale: list[str] = []
        with self._read() as connection:
            rows = connection.execute(
                "SELECT point_id, content_hash FROM image_source "
                "WHERE content_hash IS NOT NULL"
            ).fetchall()
            for row in rows:
                point_id = str(row["point_id"])
                live = live_hashes.get(point_id)
                if live and row["content_hash"] and live != row["content_hash"]:
                    stale.append(point_id)
            for point_id in stale:
                connection.execute(
                    "UPDATE image_source SET status = 'stale', updated_at = ? WHERE point_id = ?",
                    (_now(), point_id),
                )
            connection.commit()
        return stale

    def list_source_tags(
        self, query: str = "", *, limit: int = 60, offset: int = 0
    ) -> list[dict[str, Any]]:
        needle = f"%{str(query or '').strip().casefold()}%"
        params: list[Any] = []
        where = "TRIM(value) <> ''"
        if str(query or "").strip():
            where += " AND LOWER(value) LIKE ?"
            params.append(needle)
        params.extend([max(1, min(int(limit), 500)), max(0, int(offset))])
        with self._read() as connection:
            rows = connection.execute(
                "WITH all_tags AS ("
                "SELECT point_id, value FROM image_source, json_each(tags_json) "
                "UNION ALL SELECT point_id, value FROM image_source, json_each(manual_tags_json) "
                "UNION ALL SELECT point_id, value FROM image_source, json_each(auto_tags_json)"
                ") SELECT value AS tag, COUNT(DISTINCT point_id) AS count FROM all_tags "
                f"WHERE {where} GROUP BY value "
                "ORDER BY count DESC, value COLLATE NOCASE LIMIT ? OFFSET ?",
                params,
            ).fetchall()
        return [{"tag": str(row["tag"]), "count": int(row["count"])} for row in rows]

    def set_auto_tags(
        self,
        point_id: str,
        tags: Sequence[str],
        scores: dict[str, float] | None = None,
    ) -> None:
        self.set_auto_tags_many([(point_id, tags, scores)])

    def set_auto_tags_many(
        self,
        entries: Sequence[tuple[str, Sequence[str], dict[str, float] | None]],
    ) -> None:
        if not entries:
            return
        now = _now()
        rows = [
            (
                _json(list(tags), []),
                _json(scores or {}, {}),
                now,
                now,
                point_id,
            )
            for point_id, tags, scores in entries
            if point_id
        ]
        if not rows:
            return
        with self._read() as connection:
            connection.executemany(
                "UPDATE image_source SET auto_tags_json = ?, auto_tag_scores_json = ?, "
                "auto_tagged_at = ?, updated_at = ? WHERE point_id = ?",
                rows,
            )
            connection.commit()

    def point_ids_without_auto_tags(self, limit: int | None = None) -> list[str]:
        sql = (
            "SELECT point_id FROM image_source "
            "WHERE auto_tagged_at IS NULL "
            "ORDER BY mtime_ns IS NULL, mtime_ns DESC, point_id"
        )
        params: list[Any] = []
        if limit is not None:
            sql += " LIMIT ?"
            params.append(max(1, int(limit)))
        with self._read() as connection:
            rows = connection.execute(sql, params).fetchall()
        return [str(row["point_id"]) for row in rows]

    def _scope_temp_table(self, connection: sqlite3.Connection, point_ids: Sequence[str]) -> None:
        connection.execute(
            "CREATE TEMP TABLE IF NOT EXISTS _scope_points "
            "(point_id TEXT PRIMARY KEY)"
        )
        connection.execute("DELETE FROM _scope_points")
        rows = [(str(item),) for item in dict.fromkeys(point_ids) if item]
        if rows:
            connection.executemany(
                "INSERT OR IGNORE INTO _scope_points(point_id) VALUES (?)", rows
            )

    def author_counts_for_points(self, point_ids: Sequence[str]) -> dict[str, int]:
        with self._read() as connection:
            self._scope_temp_table(connection, point_ids)
            rows = connection.execute(
                "SELECT image_source.author_key AS key, COUNT(*) AS count "
                "FROM image_source JOIN _scope_points USING(point_id) "
                "GROUP BY image_source.author_key"
            ).fetchall()
        counts: dict[str, int] = {}
        for row in rows:
            key = row["key"]
            if key in (None, ""):
                counts[UNASSIGNED_AUTHOR_KEY] = int(row["count"])
            else:
                counts[str(key)] = int(row["count"])
        return counts

    def source_tag_counts_for_points(
        self,
        point_ids: Sequence[str],
        query: str = "",
        *,
        limit: int = 60,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        needle = f"%{str(query or '').strip().casefold()}%"
        params: list[Any] = []
        where = "TRIM(value) <> ''"
        if str(query or "").strip():
            where += " AND LOWER(value) LIKE ?"
            params.append(needle)
        params.extend([max(1, min(int(limit), 500)), max(0, int(offset))])
        with self._read() as connection:
            self._scope_temp_table(connection, point_ids)
            rows = connection.execute(
                "WITH all_tags AS ("
                "SELECT point_id, value FROM image_source, json_each(tags_json) "
                "UNION ALL SELECT point_id, value FROM image_source, json_each(manual_tags_json) "
                "UNION ALL SELECT point_id, value FROM image_source, json_each(auto_tags_json)"
                ") SELECT value AS tag, COUNT(DISTINCT point_id) AS count FROM all_tags "
                "JOIN _scope_points USING(point_id) "
                f"WHERE {where} GROUP BY value "
                "ORDER BY count DESC, value COLLATE NOCASE LIMIT ? OFFSET ?",
                params,
            ).fetchall()
        return [{"tag": str(row["tag"]), "count": int(row["count"])} for row in rows]

    def point_file_names(self, point_ids: Sequence[str]) -> dict[str, str]:
        clean = [str(item) for item in dict.fromkeys(point_ids) if item]
        result: dict[str, str] = {}
        with self._read() as connection:
            for start in range(0, len(clean), 512):
                batch = clean[start : start + 512]
                placeholders = ",".join("?" for _ in batch)
                rows = connection.execute(
                    f"SELECT point_id, path_key FROM image_source "
                    f"WHERE point_id IN ({placeholders})",
                    batch,
                ).fetchall()
                for row in rows:
                    path_key = str(row["path_key"] or "").replace("\\", "/")
                    result[str(row["point_id"])] = path_key.rsplit("/", 1)[-1]
        return result

    def point_mtimes(self, point_ids: Sequence[str]) -> dict[str, int]:
        clean = [str(item) for item in dict.fromkeys(point_ids) if item]
        result: dict[str, int] = {}
        with self._read() as connection:
            for start in range(0, len(clean), 512):
                batch = clean[start : start + 512]
                placeholders = ",".join("?" for _ in batch)
                rows = connection.execute(
                    f"SELECT point_id, mtime_ns FROM image_source "
                    f"WHERE point_id IN ({placeholders})",
                    batch,
                ).fetchall()
                for row in rows:
                    result[str(row["point_id"])] = _int_or_none(row["mtime_ns"]) or 0
        return result

    def point_ids_for_any_tags(self, tags: Sequence[str]) -> list[str]:
        clean = [str(tag).strip() for tag in tags if str(tag).strip()]
        if not clean:
            return []
        clauses = " OR ".join(
            "EXISTS (SELECT 1 FROM all_tags WHERE all_tags.point_id = image_source.point_id "
            "AND value = ? COLLATE NOCASE)"
            for _ in clean
        )
        with self._read() as connection:
            rows = connection.execute(
                "WITH all_tags AS ("
                "SELECT point_id, value FROM image_source, json_each(tags_json) "
                "UNION ALL SELECT point_id, value FROM image_source, json_each(manual_tags_json) "
                "UNION ALL SELECT point_id, value FROM image_source, json_each(auto_tags_json)"
                ") SELECT point_id FROM image_source WHERE "
                f"{clauses} ORDER BY mtime_ns IS NULL, mtime_ns DESC, updated_at DESC, point_id",
                clean,
            ).fetchall()
        return [str(row["point_id"]) for row in rows]

    def all_point_ids(self) -> list[str]:
        with self._read() as connection:
            rows = connection.execute("SELECT point_id FROM image_source").fetchall()
        return [str(row["point_id"]) for row in rows]

    def point_ids_for_title(self, query: str) -> list[str]:
        terms = [term for term in re.split(r"\s+", str(query or "").strip()) if term]
        if not terms:
            return []
        clauses = " AND ".join(
            "instr(LOWER(COALESCE(title, '')), ?) > 0" for _ in terms
        )
        with self._read() as connection:
            rows = connection.execute(
                f"SELECT point_id FROM image_source WHERE {clauses} "
                "ORDER BY mtime_ns IS NULL, mtime_ns DESC, updated_at DESC, point_id",
                [term.casefold() for term in terms],
            ).fetchall()
        return [str(row["point_id"]) for row in rows]

    def point_ids_for_tag_text(self, terms: Sequence[str]) -> list[str]:
        clean = [str(term).strip() for term in terms if str(term).strip()]
        if not clean:
            return []
        clauses = " AND ".join(
            "EXISTS (SELECT 1 FROM all_tags WHERE all_tags.point_id = image_source.point_id "
            "AND value LIKE ? COLLATE NOCASE)"
            for _ in clean
        )
        with self._read() as connection:
            rows = connection.execute(
                "WITH all_tags AS ("
                "SELECT point_id, value FROM image_source, json_each(tags_json) "
                "UNION ALL SELECT point_id, value FROM image_source, json_each(manual_tags_json) "
                "UNION ALL SELECT point_id, value FROM image_source, json_each(auto_tags_json)"
                ") SELECT point_id FROM image_source WHERE "
                f"{clauses} ORDER BY mtime_ns IS NULL, mtime_ns DESC, updated_at DESC, point_id",
                [f"%{term}%" for term in clean],
            ).fetchall()
        return [str(row["point_id"]) for row in rows]

    def point_ids_for_metadata_search(
        self, term_groups: Sequence[str | Sequence[str]]
    ) -> list[str]:
        groups: list[list[str]] = []
        for group in term_groups:
            values = [group] if isinstance(group, str) else list(group)
            clean = [str(value).strip().lstrip("#") for value in values]
            clean = list(dict.fromkeys(value for value in clean if value))
            if clean:
                groups.append(clean)
        if not groups:
            return []
        clauses: list[str] = []
        params: list[str] = []
        for group in groups:
            alternatives: list[str] = []
            for term in group:
                like = f"%{term.casefold()}%"
                alternatives.append(
                    "(EXISTS (SELECT 1 FROM all_tags "
                    "WHERE all_tags.point_id = image_source.point_id "
                    "AND LOWER(value) LIKE ?) "
                    "OR LOWER(COALESCE(title, '')) LIKE ? "
                    "OR EXISTS (SELECT 1 FROM artist "
                    "WHERE CAST(artist.id AS TEXT) = image_source.author_key "
                    "AND (LOWER(COALESCE(artist.name, '')) LIKE ? "
                    "OR LOWER(COALESCE(artist.name_key, '')) LIKE ? "
                    "OR EXISTS (SELECT 1 FROM artist_identity "
                    "WHERE artist_identity.artist_id = artist.id "
                    "AND (artist_identity.identity LIKE ? "
                    "OR LOWER(artist_identity.display) LIKE ?)))))"
                )
                params.extend([like, like, like, like, like, like])
            clauses.append("(" + " OR ".join(alternatives) + ")")
        with self._read() as connection:
            rows = connection.execute(
                "WITH all_tags AS ("
                "SELECT point_id, value FROM image_source, json_each(tags_json) "
                "UNION ALL SELECT point_id, value FROM image_source, json_each(manual_tags_json) "
                "UNION ALL SELECT point_id, value FROM image_source, json_each(auto_tags_json)"
                ") SELECT point_id FROM image_source WHERE "
                + " AND ".join(clauses)
                + " ORDER BY mtime_ns IS NULL, mtime_ns DESC, updated_at DESC, point_id",
                params,
            ).fetchall()
        return [str(row["point_id"]) for row in rows]

    def point_ids_for_tags(self, tags: Sequence[str]) -> list[str]:
        clean = [str(tag).strip() for tag in tags if str(tag).strip()]
        if not clean:
            return []
        clauses = " AND ".join(
            "EXISTS (SELECT 1 FROM all_tags WHERE all_tags.point_id = image_source.point_id "
            "AND value = ? COLLATE NOCASE)"
            for _ in clean
        )
        with self._read() as connection:
            rows = connection.execute(
                "WITH all_tags AS ("
                "SELECT point_id, value FROM image_source, json_each(tags_json) "
                "UNION ALL SELECT point_id, value FROM image_source, json_each(manual_tags_json) "
                "UNION ALL SELECT point_id, value FROM image_source, json_each(auto_tags_json)"
                ") SELECT point_id FROM image_source WHERE "
                f"{clauses} ORDER BY mtime_ns IS NULL, mtime_ns DESC, updated_at DESC, point_id",
                clean,
            ).fetchall()
        return [str(row["point_id"]) for row in rows]

    # --------------------------------------------------------- artist registry

    @staticmethod
    def _identity_specs(
        platform: str,
        platform_id: str,
        *,
        name: str | None = None,
        handle: str | None = None,
        profile_url: str | None = None,
    ) -> list[tuple[str, str, str | None, str | None, str | None]]:
        """Return ``(kind, identity, display, profile_url, source)`` specs.

        Handle identities share one namespace across Pixiv and X.  That is what
        makes a Pixiv ``userAccount`` and an X screen name with the same text
        resolve to one artist instead of two disconnected rows.
        """
        platform = str(platform or "").strip().casefold()
        raw = str(platform_id or "").strip()
        display = str(handle or "").strip().lstrip("@")
        specs: list[tuple[str, str, str | None, str | None, str | None]] = []
        if platform == "pixiv":
            pixiv_id = normalize_pixiv_user_id(raw)
            if pixiv_id:
                specs.append(
                    (
                        IDENTITY_PIXIV,
                        pixiv_id,
                        pixiv_id,
                        profile_url or f"https://www.pixiv.net/users/{pixiv_id}",
                        "pixiv",
                    )
                )
            handle_text = display or str(handle or "").strip()
            handle_key = _normalized_handle(handle_text)
            if handle_key:
                specs.append((IDENTITY_HANDLE, handle_key, handle_text, None, "pixiv"))
        elif platform == "twitter":
            handle_text = display or raw.lstrip("@")
            handle_key = _normalized_handle(handle_text)
            if raw.isdigit():
                specs.append(
                    (
                        IDENTITY_TWITTER,
                        raw,
                        handle_text or raw,
                        f"https://x.com/{handle_text}" if handle_text else None,
                        "twitter",
                    )
                )
            if handle_key:
                specs.append(
                    (
                        IDENTITY_HANDLE,
                        handle_key,
                        handle_text,
                        f"https://x.com/{handle_text}" if handle_text else None,
                        "twitter",
                    )
                )
        elif platform == "xiaohongshu":
            uid = normalize_xiaohongshu_user_id(raw)
            label = str(name or display or "").strip()
            if uid:
                specs.append(
                    (
                        IDENTITY_XIAOHONGSHU,
                        uid,
                        label or uid,
                        profile_url or xiaohongshu_user_url(uid),
                        "xiaohongshu",
                    )
                )
            name_key = _author_text(name or display)
            if name_key:
                specs.append(
                    (IDENTITY_NAME, name_key, label or None, None, "xiaohongshu")
                )
        elif platform == "manual":
            label = str(name or raw or handle or "").strip()
            name_key = _author_text(label)
            if name_key:
                specs.append((IDENTITY_NAME, name_key, label, None, "manual"))
        elif raw:
            identity = _normalized_handle(raw)
            specs.append((platform, identity, display or raw, profile_url, platform))
        unique: list[tuple[str, str, str | None, str | None, str | None]] = []
        seen: set[tuple[str, str]] = set()
        for spec in specs:
            key = (spec[0], spec[1])
            if key in seen:
                continue
            seen.add(key)
            unique.append(spec)
        return unique

    @staticmethod
    def _should_replace_avatar(current: str | None, incoming: str | None) -> bool:
        new_url = str(incoming or "").strip()
        if not new_url:
            return False
        cur_url = str(current or "").strip()
        if not cur_url:
            return True
        if cur_url.startswith("/api/authors/"):
            return False
        if new_url.startswith("/api/authors/"):
            return True
        # Pixiv's pximg.net rejects browser requests without a pixiv.net referrer;
        # prefer directly loadable social avatars such as Twitter's pbs.twimg.com.
        return "pximg.net" in cur_url.casefold() and "pximg.net" not in new_url.casefold()

    @staticmethod
    def _name_priority(value: Any) -> tuple[int, int, int]:
        text = str(value or "").strip()
        if not text:
            return (0, 0, 0)
        non_ascii = any(ord(char) > 127 for char in text)
        human_script = 1 if non_ascii and len(text) >= 2 else 0
        return (1, human_script, min(len(text), 40))

    def _canonical_locked(self, connection: sqlite3.Connection, key: str) -> str:
        raw = str(key or "").strip()
        if not raw:
            return raw
        if raw.isdigit():
            row = connection.execute(
                "SELECT id FROM artist WHERE id = ?", (int(raw),)
            ).fetchone()
            if row is not None:
                return str(row["id"])
        row = connection.execute(
            "SELECT artist_id FROM artist_legacy WHERE legacy_key = ?", (raw,)
        ).fetchone()
        if row is not None:
            return str(row["artist_id"])
        if ":" in raw:
            prefix, value = raw.split(":", 1)
            prefix = prefix.strip().casefold()
            candidates: list[tuple[str, str]] = []
            if prefix in {"twitter", "x", "handle"}:
                candidates.append((IDENTITY_HANDLE, _normalized_handle(value)))
                if value.strip().isdigit():
                    candidates.append((IDENTITY_TWITTER, value.strip()))
            elif prefix == "pixiv":
                candidates.append((IDENTITY_PIXIV, normalize_pixiv_user_id(value)))
            elif prefix in {"manual", "artist"}:
                candidates.append((IDENTITY_NAME, _author_text(value)))
            for kind, identity in candidates:
                if not identity:
                    continue
                row = connection.execute(
                    "SELECT artist_id FROM artist_identity WHERE kind = ? AND identity = ?",
                    (kind, identity),
                ).fetchone()
                if row is not None:
                    return str(row["artist_id"])
        if raw.startswith("@"):
            row = connection.execute(
                "SELECT artist_id FROM artist_identity WHERE kind = ? AND identity = ?",
                (IDENTITY_HANDLE, _normalized_handle(raw)),
            ).fetchone()
            if row is not None:
                return str(row["artist_id"])
        return raw

    def _record_legacy_locked(
        self, connection: sqlite3.Connection, legacy_key: str, artist_id: int
    ) -> None:
        key = str(legacy_key or "").strip()
        if not key:
            return
        connection.execute(
            """
            INSERT INTO artist_legacy (legacy_key, artist_id, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(legacy_key) DO UPDATE SET
                artist_id = excluded.artist_id,
                updated_at = excluded.updated_at
            """,
            (key, int(artist_id), _now()),
        )

    _NAME_SOURCE_RANK = {"manual": 3, "pixiv": 2, "xiaohongshu": 2, "twitter": 1}

    def _record_name_alias_locked(
        self,
        connection: sqlite3.Connection,
        artist_id: int,
        name: str | None,
        source: str | None,
        now: int,
    ) -> None:
        label = str(name or "").strip()
        key = _author_text(label)
        if not label or not key:
            return
        row = connection.execute(
            "SELECT artist_id FROM artist_identity WHERE kind = ? AND identity = ?",
            (IDENTITY_NAME_ALIAS, key),
        ).fetchone()
        if row is None:
            connection.execute(
                "INSERT INTO artist_identity"
                "(kind, identity, display, profile_url, source, artist_id, updated_at) "
                "VALUES (?, ?, ?, NULL, ?, ?, ?)",
                (IDENTITY_NAME_ALIAS, key, label, source or "manual", int(artist_id), now),
            )
        elif int(row["artist_id"]) == int(artist_id):
            connection.execute(
                "UPDATE artist_identity SET display = ?, source = ?, updated_at = ? "
                "WHERE kind = ? AND identity = ? AND artist_id = ?",
                (label, source or "manual", now, IDENTITY_NAME_ALIAS, key, int(artist_id)),
            )

    def _update_artist_locked(
        self,
        connection: sqlite3.Connection,
        artist_id: int,
        *,
        name: str | None,
        name_source: str | None,
        avatar_url: str | None,
        profile_fetched: bool,
        overwrite_name: bool,
        now: int,
    ) -> None:
        row = connection.execute(
            "SELECT name, name_source, avatar_url, profile_fetched FROM artist WHERE id = ?",
            (int(artist_id),),
        ).fetchone()
        if row is None:
            return
        updates: list[str] = ["updated_at = ?"]
        values: list[Any] = [now]
        current_name = str(row["name"] or "").strip()
        incoming = str(name or "").strip()
        incoming_source = str(name_source or "")
        current_source = str(row["name_source"] or "")
        if current_name and incoming and current_name != incoming:
            self._record_name_alias_locked(
                connection, artist_id, current_name, current_source, now
            )
        if incoming and (
            (
                overwrite_name
                and self._NAME_SOURCE_RANK.get(incoming_source, 0)
                >= self._NAME_SOURCE_RANK.get(current_source, 0)
            )
            or not current_name
            or self._NAME_SOURCE_RANK.get(incoming_source, 0)
            > self._NAME_SOURCE_RANK.get(current_source, 0)
            or (
                self._NAME_SOURCE_RANK.get(incoming_source, 0)
                == self._NAME_SOURCE_RANK.get(current_source, 0)
                and self._name_priority(incoming) > self._name_priority(current_name)
            )
        ):
            updates.extend(["name = ?", "name_key = ?", "name_source = ?"])
            values.extend([incoming, _author_text(incoming), incoming_source or None])
        if self._should_replace_avatar(row["avatar_url"], avatar_url):
            updates.append("avatar_url = ?")
            values.append(avatar_url)
        if profile_fetched:
            updates.append("profile_fetched = 1")
        values.append(int(artist_id))
        connection.execute(
            f"UPDATE artist SET {', '.join(updates)} WHERE id = ?", values
        )

    def _merge_locked(
        self,
        connection: sqlite3.Connection,
        source_id: int,
        target_id: int,
        now: int,
    ) -> None:
        source_id = int(source_id)
        target_id = int(target_id)
        if source_id == target_id:
            return
        source = connection.execute(
            "SELECT * FROM artist WHERE id = ?", (source_id,)
        ).fetchone()
        target = connection.execute(
            "SELECT * FROM artist WHERE id = ?", (target_id,)
        ).fetchone()
        if source is None:
            return
        if target is None:
            connection.execute(
                "UPDATE artist SET id = ?, updated_at = ? WHERE id = ?",
                (target_id, now, source_id),
            )
            return
        name = target["name"]
        name_source = target["name_source"]
        source_rank = self._NAME_SOURCE_RANK.get(str(source["name_source"] or ""), 0)
        target_rank = self._NAME_SOURCE_RANK.get(str(name_source or ""), 0)
        if source_rank > target_rank or (
            source_rank == target_rank
            and self._name_priority(source["name"]) > self._name_priority(name)
        ):
            name = source["name"]
            name_source = source["name_source"]
        avatar = (
            source["avatar_url"]
            if self._should_replace_avatar(target["avatar_url"], source["avatar_url"])
            else (target["avatar_url"] or source["avatar_url"])
        )
        note = target["note"] or source["note"]
        profile_fetched = max(
            int(target["profile_fetched"] or 0), int(source["profile_fetched"] or 0)
        )
        created_at = min(int(target["created_at"]), int(source["created_at"]))
        connection.execute(
            """
            UPDATE artist SET name = ?, name_key = ?, name_source = ?, avatar_url = ?, note = ?,
                profile_fetched = ?, created_at = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                name,
                _author_text(name) or None,
                name_source,
                avatar,
                note,
                profile_fetched,
                created_at,
                now,
                target_id,
            ),
        )
        connection.execute(
            "UPDATE OR IGNORE artist_identity SET artist_id = ?, updated_at = ? "
            "WHERE artist_id = ?",
            (target_id, now, source_id),
        )
        connection.execute(
            "DELETE FROM artist_identity WHERE artist_id = ?", (source_id,)
        )
        connection.execute(
            "UPDATE image_source SET author_key = ?, updated_at = ? WHERE author_key = ?",
            (str(target_id), now, str(source_id)),
        )
        connection.execute(
            "UPDATE artist_legacy SET artist_id = ?, updated_at = ? WHERE artist_id = ?",
            (target_id, now, source_id),
        )
        connection.execute("DELETE FROM artist WHERE id = ?", (source_id,))
        self._record_legacy_locked(connection, str(source_id), target_id)
        connection.execute(
            """
            UPDATE artist SET image_count = (
                SELECT COUNT(*) FROM image_source WHERE author_key = CAST(? AS TEXT)
            ) WHERE id = ?
            """,
            (target_id, target_id),
        )

    def _ensure_artist_locked(
        self,
        connection: sqlite3.Connection,
        specs: Sequence[tuple[str, str, str | None, str | None, str | None]],
        *,
        name: str | None,
        name_source: str | None = None,
        avatar_url: str | None,
        profile_fetched: bool,
        overwrite_name: bool,
        legacy_keys: Sequence[str] = (),
        base_artist_id: int | None = None,
        commit: bool = True,
    ) -> int:
        now = _now()
        if name_source is None and name:
            if any(kind == IDENTITY_PIXIV for kind, *_ in specs):
                name_source = "pixiv"
            elif any(kind in {IDENTITY_TWITTER, IDENTITY_HANDLE} for kind, *_ in specs):
                name_source = "twitter"
            elif any(kind == IDENTITY_NAME for kind, *_ in specs):
                name_source = "manual"
        owners: list[int] = []
        if base_artist_id is not None:
            owners.append(int(base_artist_id))
        for kind, identity, _display, _url, _source in specs:
            row = connection.execute(
                "SELECT artist_id FROM artist_identity WHERE kind = ? AND identity = ?",
                (kind, identity),
            ).fetchone()
            if row is not None:
                owner = int(row["artist_id"])
                if owner not in owners:
                    owners.append(owner)
        artist_id = (
            int(base_artist_id)
            if base_artist_id is not None
            else (min(owners) if owners else None)
        )
        if artist_id is None:
            cursor = connection.execute(
                """
                INSERT INTO artist (
                    name, name_key, name_source, avatar_url, profile_fetched,
                    image_count, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, 0, ?, ?)
                """,
                (
                    str(name or "").strip() or None,
                    _author_text(name) or None,
                    name_source,
                    avatar_url,
                    1 if profile_fetched else 0,
                    now,
                    now,
                ),
            )
            artist_id = int(cursor.lastrowid)
        else:
            for other in owners:
                if other != artist_id:
                    self._merge_locked(connection, other, artist_id, now)
            self._update_artist_locked(
                connection,
                artist_id,
                name=name,
                name_source=name_source,
                avatar_url=avatar_url,
                profile_fetched=profile_fetched,
                overwrite_name=overwrite_name,
                now=now,
            )
        self._record_name_alias_locked(
            connection, artist_id, name, name_source, now
        )
        for kind, identity, display, url, source in specs:
            connection.execute(
                """
                INSERT INTO artist_identity (
                    kind, identity, display, profile_url, source, artist_id, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(kind, identity) DO UPDATE SET
                    artist_id = excluded.artist_id,
                    display = COALESCE(excluded.display, artist_identity.display),
                    profile_url = COALESCE(excluded.profile_url, artist_identity.profile_url),
                    source = CASE
                        WHEN artist_identity.source = 'twitter'
                          OR excluded.source = 'twitter' THEN 'twitter'
                        ELSE COALESCE(excluded.source, artist_identity.source)
                    END,
                    updated_at = excluded.updated_at
                """,
                (kind, identity, display, url, source, artist_id, now),
            )
        for legacy in legacy_keys:
            self._record_legacy_locked(connection, legacy, artist_id)
        if commit:
            connection.commit()
        return artist_id

    def ensure_author(
        self,
        platform: str,
        platform_id: str,
        *,
        name: str | None = None,
        handle: str | None = None,
        profile_url: str | None = None,
        avatar_url: str | None = None,
        social: dict[str, Any] | None = None,
        profile_fetched: bool = False,
    ) -> str:
        del social  # Social links are resolved into identities before reaching the store.
        specs = self._identity_specs(
            platform,
            platform_id,
            name=name,
            handle=handle,
            profile_url=profile_url,
        )
        with self._read() as connection:
            artist_id = self._ensure_artist_locked(
                connection,
                specs,
                name=name,
                avatar_url=avatar_url,
                profile_fetched=profile_fetched,
                overwrite_name=bool(profile_fetched) or platform == "manual",
                legacy_keys=[
                    key
                    for key in (
                        f"{platform}:{platform_id}".strip(":"),
                        f"{platform}:{handle}".strip(":") if handle else "",
                    )
                    if key
                ],
            )
        return str(artist_id)

    def ensure_manual_author(self, name: str) -> str:
        """Find or create the artist identified by a manually entered name."""
        label = str(name or "").strip()
        if not label:
            raise ValueError("作者名不能为空")
        return self.ensure_author("manual", label, name=label)

    def ensure_xhs_author(self, name: str) -> str:
        """Group Xiaohongshu images by the nickname written into file names.

        The download name has no user ID, so the normalized nickname is the only
        identity we can offer.  It shares the ``name`` namespace with manual
        artists, which is what makes same-named files land in one group.
        """
        label = str(name or "").strip()
        if not label:
            raise ValueError("作者名不能为空")
        specs = [(IDENTITY_NAME, _author_text(label), label, None, "xiaohongshu")]
        with self._read() as connection:
            artist_id = self._ensure_artist_locked(
                connection,
                specs,
                name=label,
                name_source="xiaohongshu",
                avatar_url=None,
                profile_fetched=False,
                overwrite_name=True,
            )
        return str(artist_id)

    def ensure_linked_author(
        self,
        *,
        name: str | None = None,
        pixiv_id: str | None = None,
        x_handle: str | None = None,
    ) -> str:
        """Attach Pixiv and/or X identities to one artist, merging conflicts."""
        specs: list[tuple[str, str, str | None, str | None, str | None]] = []
        legacy: list[str] = []
        pixiv_value = normalize_pixiv_user_id(pixiv_id)
        if pixiv_value:
            specs.extend(self._identity_specs("pixiv", pixiv_value))
            legacy.append(f"pixiv:{pixiv_value}")
        handle_value = normalize_twitter_handle(x_handle)
        if handle_value:
            specs.extend(self._identity_specs("twitter", handle_value, handle=handle_value))
            legacy.extend([f"twitter:{handle_value}", f"twitter:{handle_value.casefold()}"])
        if not specs:
            return self.ensure_manual_author(name or "")
        with self._read() as connection:
            artist_id = self._ensure_artist_locked(
                connection,
                specs,
                name=name,
                name_source="manual" if name else None,
                avatar_url=None,
                profile_fetched=False,
                overwrite_name=bool(name),
                legacy_keys=legacy,
            )
        return str(artist_id)

    def identity_owner(self, platform: str, value: str) -> str | None:
        """Return the artist ID that already owns a Pixiv ID, handle, or XHS UID."""
        platform = str(platform or "").strip().casefold()
        if platform == "pixiv":
            kind = IDENTITY_PIXIV
            identity = normalize_pixiv_user_id(value)
        elif platform in {"xiaohongshu", "xhs", "rednote"}:
            kind = IDENTITY_XIAOHONGSHU
            identity = normalize_xiaohongshu_user_id(value)
        else:
            kind = IDENTITY_HANDLE
            identity = _normalized_handle(normalize_twitter_handle(value))
        if not identity:
            return None
        with self._read() as connection:
            row = connection.execute(
                "SELECT artist_id FROM artist_identity WHERE kind = ? AND identity = ?",
                (kind, identity),
            ).fetchone()
        return str(row["artist_id"]) if row is not None else None

    def attach_identity(
        self,
        artist_key: str,
        platform: str,
        value: str,
        *,
        name: str | None = None,
    ) -> str:
        """Attach an unowned Pixiv ID, X handle, or Xiaohongshu UID."""
        platform = str(platform or "").strip().casefold()
        if platform == "pixiv":
            pixiv_id = normalize_pixiv_user_id(value)
            if not pixiv_id:
                raise ValueError("Pixiv ID 无效")
            specs = self._identity_specs("pixiv", pixiv_id)
            legacy = [f"pixiv:{pixiv_id}"]
        elif platform in {"xiaohongshu", "xhs", "rednote"}:
            uid = normalize_xiaohongshu_user_id(value)
            if not uid:
                raise ValueError("小红书主页链接或 UID 无效")
            specs = self._identity_specs("xiaohongshu", uid, name=name)
            legacy = [f"xiaohongshu:{uid}"]
        elif platform in {"twitter", "x", "x_id"}:
            handle = normalize_twitter_handle(value)
            if not handle:
                raise ValueError("X handle 无效")
            specs = self._identity_specs("twitter", handle, handle=handle)
            legacy = [f"twitter:{handle}", f"twitter:{handle.casefold()}"]
        else:
            raise ValueError("不支持的社交平台")
        with self._read() as connection:
            base = self._canonical_locked(connection, artist_key)
            if not base.isdigit():
                raise ValueError("作者不存在")
            artist_id = self._ensure_artist_locked(
                connection,
                specs,
                name=name,
                name_source=(
                    "xiaohongshu"
                    if platform in {"xiaohongshu", "xhs", "rednote"}
                    else ("manual" if name else None)
                ),
                avatar_url=None,
                profile_fetched=False,
                overwrite_name=bool(name),
                legacy_keys=legacy,
                base_artist_id=int(base),
            )
        return str(artist_id)

    def add_link(self, artist_key: str, url: str, *, label: str = "") -> str:
        """Attach a normalized http(s) social link to an artist."""
        if self._is_unassigned_key(artist_key):
            raise ValueError("无画师不能添加社交链接")
        normalized = normalize_social_link(url)
        if not normalized:
            raise ValueError("请输入有效的 http(s) 链接")
        display = str(label or "").strip() or normalized
        with self._read() as connection:
            base = self._canonical_locked(connection, artist_key)
            if not base.isdigit():
                raise ValueError("作者不存在")
            owner = connection.execute(
                "SELECT artist_id FROM artist_identity WHERE kind = ? AND identity = ?",
                (IDENTITY_LINK, normalized),
            ).fetchone()
            if owner is not None and str(owner["artist_id"]) != base:
                owner_id = int(owner["artist_id"])
                owner_row = connection.execute(
                    "SELECT name FROM artist WHERE id = ?", (owner_id,)
                ).fetchone()
                owner_name = str(owner_row["name"] or owner_id) if owner_row else str(owner_id)
                raise ValueError(f"该链接已属于画师「{owner_name}」")
            connection.execute(
                "INSERT OR REPLACE INTO artist_identity"
                "(kind, identity, display, profile_url, source, artist_id, updated_at) "
                "VALUES (?, ?, ?, ?, 'manual', ?, ?)",
                (IDENTITY_LINK, normalized, display, normalized, int(base), _now()),
            )
            connection.execute(
                "UPDATE artist SET updated_at = ? WHERE id = ?", (_now(), int(base))
            )
            connection.commit()
        return base

    def add_alias(self, artist_key: str, alias: str) -> str:
        """Attach a searchable former/platform name to an artist."""
        if self._is_unassigned_key(artist_key):
            raise ValueError("无画师不能添加别名")
        label = str(alias or "").strip()
        key = _author_text(label)
        if not key:
            raise ValueError("作者别名不能为空")
        with self._read() as connection:
            base = self._canonical_locked(connection, artist_key)
            if not base.isdigit():
                raise ValueError("作者不存在")
            owner = connection.execute(
                "SELECT artist_id FROM artist_identity WHERE kind = ? AND identity = ?",
                (IDENTITY_NAME_ALIAS, key),
            ).fetchone()
            if owner is not None and str(owner["artist_id"]) != base:
                owner_row = connection.execute(
                    "SELECT name FROM artist WHERE id = ?", (int(owner["artist_id"]),)
                ).fetchone()
                owner_name = (
                    str(owner_row["name"] or owner["artist_id"])
                    if owner_row is not None
                    else str(owner["artist_id"])
                )
                raise ValueError(f"该别名已属于画师「{owner_name}」")
            self._record_name_alias_locked(connection, int(base), label, "manual", _now())
            connection.execute(
                "UPDATE artist SET updated_at = ? WHERE id = ?", (_now(), int(base))
            )
            connection.commit()
        return base

    def set_artist_cover(self, artist_key: str, point_id: str) -> str:
        """Mark one of the artist's images as their explicit representative."""
        if self._is_unassigned_key(artist_key):
            raise ValueError("无画师不能设置代表图")
        with self._read() as connection:
            base = self._canonical_locked(connection, artist_key)
            if not base.isdigit():
                raise ValueError("作者不存在")
            row = connection.execute(
                "SELECT author_key FROM image_source WHERE point_id = ?",
                (point_id,),
            ).fetchone()
            if row is None:
                raise ValueError("图片不在来源记录中")
            owner = self._canonical_locked(connection, str(row["author_key"] or ""))
            if owner != base:
                raise ValueError("该图片不属于这个画师")
            connection.execute(
                "DELETE FROM artist_identity WHERE artist_id = ? AND kind = ?",
                (int(base), IDENTITY_COVER),
            )
            connection.execute(
                "INSERT INTO artist_identity"
                "(kind, identity, display, profile_url, source, artist_id, updated_at) "
                "VALUES (?, ?, ?, NULL, 'manual', ?, ?)",
                (IDENTITY_COVER, point_id, point_id, int(base), _now()),
            )
            connection.execute(
                "UPDATE artist SET updated_at = ? WHERE id = ?", (_now(), int(base))
            )
            connection.commit()
        return base

    def clear_artist_cover(self, artist_key: str) -> str:
        """Remove any explicit or uploaded representative cover for an artist."""
        if self._is_unassigned_key(artist_key):
            raise ValueError("无画师不能取消代表图")
        with self._read() as connection:
            base = self._canonical_locked(connection, artist_key)
            if not base.isdigit():
                raise ValueError("作者不存在")
            connection.execute(
                "DELETE FROM artist_identity WHERE artist_id = ? AND kind = ?",
                (int(base), IDENTITY_COVER),
            )
            connection.execute(
                "UPDATE artist SET updated_at = ? WHERE id = ?", (_now(), int(base))
            )
            connection.commit()
        return base

    def set_uploaded_artist_cover(
        self, artist_key: str, asset_token: str, url: str
    ) -> str:
        """Store a manually uploaded custom representative cover for an artist."""
        if self._is_unassigned_key(artist_key):
            raise ValueError("无画师不能设置代表图")
        with self._read() as connection:
            base = self._canonical_locked(connection, artist_key)
            if not base.isdigit():
                raise ValueError("作者不存在")
            connection.execute(
                "DELETE FROM artist_identity WHERE artist_id = ? AND kind = ?",
                (int(base), IDENTITY_COVER),
            )
            connection.execute(
                "INSERT INTO artist_identity"
                "(kind, identity, display, profile_url, source, artist_id, updated_at) "
                "VALUES (?, ?, ?, ?, 'upload', ?, ?)",
                (IDENTITY_COVER, asset_token, asset_token, url, int(base), _now()),
            )
            connection.execute(
                "UPDATE artist SET updated_at = ? WHERE id = ?", (_now(), int(base))
            )
            connection.commit()
        return base

    def set_artist_avatar(self, artist_key: str, avatar_url: str | None) -> str:
        """Set or clear the artist's avatar URL."""
        if self._is_unassigned_key(artist_key):
            raise ValueError("无画师不能设置头像")
        with self._read() as connection:
            base = self._canonical_locked(connection, artist_key)
            if not base.isdigit():
                raise ValueError("作者不存在")
            connection.execute(
                "UPDATE artist SET avatar_url = ?, updated_at = ? WHERE id = ?",
                (avatar_url, _now(), int(base)),
            )
            connection.commit()
        return base

    def artist_cover_entries(self, keys: Sequence[str]) -> dict[str, dict[str, str]]:
        clean = list(dict.fromkeys(str(key) for key in keys if key))
        numeric = sorted({key for key in clean if key.isdigit()})
        result: dict[str, dict[str, str]] = {}
        if not numeric:
            return result
        with self._read() as connection:
            for start in range(0, len(numeric), 400):
                batch = numeric[start : start + 400]
                placeholders = ",".join("?" for _ in batch)
                for row in connection.execute(
                    "SELECT artist_id, identity, profile_url, source FROM artist_identity "
                    f"WHERE kind = ? AND artist_id IN ({placeholders})",
                    [IDENTITY_COVER, *batch],
                ).fetchall():
                    result[str(row["artist_id"])] = {
                        "identity": str(row["identity"] or ""),
                        "profile_url": str(row["profile_url"] or ""),
                        "source": str(row["source"] or "manual"),
                    }
        return result

    def artist_cover_ids(self, keys: Sequence[str]) -> dict[str, str]:
        return {
            key: entry["identity"]
            for key, entry in self.artist_cover_entries(keys).items()
            if entry.get("source") != "upload" and entry.get("identity")
        }

    def set_author_name(self, artist_key: str, name: str) -> str:
        """Set the display name for an artist without merging other artists."""
        if self._is_unassigned_key(artist_key):
            raise ValueError("无画师不能改名")
        label = str(name or "").strip()
        if not label:
            raise ValueError("作者名不能为空")
        name_key = _author_text(label)
        with self._read() as connection:
            base = self._canonical_locked(connection, artist_key)
            if not base.isdigit():
                raise ValueError("作者不存在")
            artist_id = int(base)
            current = connection.execute(
                "SELECT name, name_source FROM artist WHERE id = ?", (artist_id,)
            ).fetchone()
            now = _now()
            if current is not None:
                self._record_name_alias_locked(
                    connection,
                    artist_id,
                    str(current["name"] or ""),
                    str(current["name_source"] or ""),
                    now,
                )
            self._record_name_alias_locked(connection, artist_id, label, "manual", now)
            connection.execute(
                "UPDATE artist SET name = ?, name_key = ?, name_source = 'manual', "
                "updated_at = ? WHERE id = ?",
                (label, name_key or None, now, artist_id),
            )
            connection.execute(
                "DELETE FROM artist_identity WHERE artist_id = ? AND kind = ?",
                (artist_id, IDENTITY_NAME),
            )
            if name_key:
                owner = connection.execute(
                    "SELECT artist_id FROM artist_identity WHERE kind = ? AND identity = ?",
                    (IDENTITY_NAME, name_key),
                ).fetchone()
                if owner is None:
                    connection.execute(
                        "INSERT INTO artist_identity"
                        "(kind, identity, display, profile_url, source, artist_id, updated_at) "
                        "VALUES (?, ?, ?, NULL, 'manual', ?, ?)",
                        (IDENTITY_NAME, name_key, label, artist_id, _now()),
                    )
            connection.commit()
        return base

    def remove_identity(self, artist_key: str, platform: str, value: str) -> str:
        """Remove one social identity from an artist without touching images."""
        if self._is_unassigned_key(artist_key):
            raise ValueError("无画师没有社交身份")
        platform = str(platform or "").strip().casefold()
        with self._read() as connection:
            base = self._canonical_locked(connection, artist_key)
            if not base.isdigit():
                raise ValueError("作者不存在")
            artist_id = int(base)
            clauses: list[tuple[str, str]] = []
            if platform == "link":
                normalized = normalize_social_link(value)
                if not normalized:
                    raise ValueError("链接无效")
                clauses.append((IDENTITY_LINK, normalized))
            elif platform == "pixiv":
                normalized = normalize_pixiv_user_id(value)
                if not normalized:
                    raise ValueError("Pixiv ID 无效")
                clauses.append((IDENTITY_PIXIV, normalized))
            elif platform in {"alias", "name_alias"}:
                normalized = _author_text(value)
                if not normalized:
                    raise ValueError("作者别名无效")
                clauses.append((IDENTITY_NAME_ALIAS, normalized))
            elif platform == "twitter":
                handle = normalize_twitter_handle(value)
                if not handle:
                    raise ValueError("X handle 无效")
                if handle.isdigit():
                    clauses.append((IDENTITY_TWITTER, handle))
                else:
                    normalized_handle = _normalized_handle(handle)
                    if normalized_handle:
                        clauses.append((IDENTITY_HANDLE, normalized_handle))
            elif platform in {"xiaohongshu", "xhs", "rednote"}:
                uid = normalize_xiaohongshu_user_id(value)
                if not uid:
                    raise ValueError("小红书主页链接或 UID 无效")
                clauses.append((IDENTITY_XIAOHONGSHU, uid))
            else:
                raise ValueError("不支持的社交平台")
            removed = 0
            for kind, identity in clauses:
                cursor = connection.execute(
                    "DELETE FROM artist_identity "
                    "WHERE artist_id = ? AND kind = ? AND identity = ?",
                    (artist_id, kind, identity),
                )
                removed += cursor.rowcount
            if platform == "twitter":
                handle = normalize_twitter_handle(value)
                if handle and not handle.isdigit():
                    cursor = connection.execute(
                        "DELETE FROM artist_identity "
                        "WHERE artist_id = ? AND kind = ? "
                        "AND LOWER(COALESCE(display, '')) = LOWER(?)",
                        (artist_id, IDENTITY_TWITTER, handle),
                    )
                    removed += cursor.rowcount
            if removed:
                connection.execute(
                    "UPDATE artist SET updated_at = ? WHERE id = ?", (_now(), artist_id)
                )
            connection.commit()
        return base

    def get_author(self, key: str) -> dict[str, Any] | None:
        with self._read() as connection:
            artist_id = self._canonical_locked(connection, key)
            if not artist_id.isdigit():
                return None
            row = connection.execute(
                "SELECT * FROM artist WHERE id = ?", (int(artist_id),)
            ).fetchone()
            if row is None:
                return None
            identities = connection.execute(
                "SELECT * FROM artist_identity WHERE artist_id = ? "
                "ORDER BY kind, identity",
                (int(artist_id),),
            ).fetchall()
        return self._author_view(row, identities)

    def author_is_fresh(self, key: str, *, max_age: int = AUTHOR_TTL_SECONDS) -> bool:
        author = self.get_author(key)
        return bool(author and _now() - int(author["updated_at"]) <= max_age)

    def author_profile_fresh(self, key: str, *, max_age: int = AUTHOR_TTL_SECONDS) -> bool:
        author = self.get_author(key)
        return bool(
            author
            and author.get("profile_fetched")
            and _now() - int(author["updated_at"]) <= max_age
        )

    def link_authors(
        self,
        alias_key: str,
        canonical_key: str,
        *,
        evidence: str,
        confidence: float = 0.8,
    ) -> str:
        del evidence, confidence
        with self._read() as connection:
            alias = self._canonical_locked(connection, alias_key)
            canonical = self._canonical_locked(connection, canonical_key)
            if alias.isdigit() and canonical.isdigit() and alias != canonical:
                self._merge_locked(connection, int(alias), int(canonical), _now())
                self._record_legacy_locked(connection, str(alias_key), int(canonical))
                connection.commit()
            return canonical

    def merge_authors(self, source_key: str, target_key: str) -> str:
        """Merge one artist into another; every identity and image follows."""
        with self._read() as connection:
            source = self._canonical_locked(connection, source_key)
            target = self._canonical_locked(connection, target_key)
            if not source.isdigit() or not target.isdigit():
                raise ValueError("作者不存在")
            if source != target:
                self._merge_locked(connection, int(source), int(target), _now())
                self._record_legacy_locked(connection, source_key, int(target))
                connection.commit()
            return target

    def canonical_author_key(self, key: str) -> str:
        with self._read() as connection:
            return self._canonical_locked(connection, key)

    def author_members(self, key: str) -> list[str]:
        with self._read() as connection:
            artist_id = self._canonical_locked(connection, key)
            if not artist_id.isdigit():
                return []
            rows = connection.execute(
                "SELECT kind, identity FROM artist_identity WHERE artist_id = ? "
                "AND kind IN (?, ?, ?, ?, ?)",
                (
                    int(artist_id),
                    IDENTITY_PIXIV,
                    IDENTITY_TWITTER,
                    IDENTITY_HANDLE,
                    IDENTITY_LINK,
                    IDENTITY_XIAOHONGSHU,
                ),
            ).fetchall()
        return sorted(f"{row['kind']}:{row['identity']}" for row in rows)

    def point_ids_for_author(self, key: str) -> list[str]:
        with self._read() as connection:
            if self._is_unassigned_key(key):
                rows = connection.execute(
                    "SELECT point_id FROM image_source "
                    "WHERE author_key IS NULL OR author_key = '' "
                    "ORDER BY mtime_ns IS NULL, mtime_ns DESC, updated_at DESC, point_id"
                ).fetchall()
                return [str(row["point_id"]) for row in rows]
            artist_id = self._canonical_locked(connection, key)
            if not artist_id.isdigit():
                return []
            rows = connection.execute(
                "SELECT point_id FROM image_source WHERE author_key = ? "
                "ORDER BY mtime_ns IS NULL, mtime_ns DESC, updated_at DESC, point_id",
                (artist_id,),
            ).fetchall()
        return [str(row["point_id"]) for row in rows]

    def point_ids_for_authors(self, keys: Sequence[str]) -> dict[str, list[str]]:
        """Return per-author image IDs for many authors in one read.

        The result preserves each author's newest-first default order; callers
        that need another order sort their own page slice.
        """
        clean = list(dict.fromkeys(str(key) for key in keys if key))
        result: dict[str, list[str]] = {key: [] for key in clean}
        numeric = sorted({key for key in clean if key.isdigit()})
        with self._read() as connection:
            for start in range(0, len(numeric), 400):
                batch = numeric[start : start + 400]
                placeholders = ",".join("?" for _ in batch)
                rows = connection.execute(
                    "SELECT author_key, point_id FROM image_source "
                    f"WHERE author_key IN ({placeholders}) "
                    "ORDER BY author_key, mtime_ns IS NULL, mtime_ns DESC, "
                    "updated_at DESC, point_id",
                    batch,
                ).fetchall()
                for row in rows:
                    key = str(row["author_key"])
                    if key in result:
                        result[key].append(str(row["point_id"]))
            if any(self._is_unassigned_key(key) for key in clean):
                rows = connection.execute(
                    "SELECT point_id FROM image_source "
                    "WHERE author_key IS NULL OR author_key = '' "
                    "ORDER BY mtime_ns IS NULL, mtime_ns DESC, updated_at DESC, point_id"
                ).fetchall()
                unassigned = [str(row["point_id"]) for row in rows]
                for key in clean:
                    if self._is_unassigned_key(key):
                        result[key] = list(unassigned)
        return result

    def author_latest_mtimes(self, keys: Sequence[str]) -> dict[str, int]:
        clean = list(dict.fromkeys(str(key) for key in keys if key))
        numeric = sorted({key for key in clean if key.isdigit()})
        result: dict[str, int] = {}
        with self._read() as connection:
            for start in range(0, len(numeric), 400):
                batch = numeric[start : start + 400]
                placeholders = ",".join("?" for _ in batch)
                for row in connection.execute(
                    "SELECT author_key, MAX(COALESCE(mtime_ns, 0)) AS latest "
                    "FROM image_source "
                    f"WHERE author_key IN ({placeholders}) GROUP BY author_key",
                    batch,
                ).fetchall():
                    result[str(row["author_key"])] = int(row["latest"] or 0)
            if any(self._is_unassigned_key(key) for key in clean):
                row = connection.execute(
                    "SELECT MAX(COALESCE(mtime_ns, 0)) FROM image_source "
                    "WHERE author_key IS NULL OR author_key = ''"
                ).fetchone()
                unassigned = int(row[0] or 0) if row is not None else 0
                for key in clean:
                    if self._is_unassigned_key(key):
                        result[key] = unassigned
        return result

    def point_ids_without_author(self, platform: str | None = None) -> list[str]:
        with self._read() as connection:
            if platform:
                rows = connection.execute(
                    "SELECT point_id FROM image_source "
                    "WHERE platform = ? AND (author_key IS NULL OR author_key = '') "
                    "ORDER BY point_id",
                    (str(platform),),
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT point_id FROM image_source "
                    "WHERE author_key IS NULL OR author_key = '' "
                    "ORDER BY point_id"
                ).fetchall()
        return [str(row["point_id"]) for row in rows]

    def assign_author_if_missing(self, point_id: str, author_key: str) -> bool:
        """Attach an artist only when the point has no author yet."""
        if self._is_unassigned_key(author_key):
            return False
        with self._read() as connection:
            artist = self._canonical_locked(connection, author_key)
            if not artist.isdigit():
                return False
            cursor = connection.execute(
                "UPDATE image_source SET author_key = ?, updated_at = ? "
                "WHERE point_id = ? AND (author_key IS NULL OR author_key = '')",
                (artist, _now(), point_id),
            )
            connection.commit()
        return cursor.rowcount > 0

    @staticmethod
    def _is_unassigned_key(key: str | None) -> bool:
        return str(key or "").strip().casefold() == UNASSIGNED_AUTHOR_KEY

    def author_names(self, keys: Sequence[str]) -> dict[str, str]:
        clean = list(dict.fromkeys(str(key) for key in keys if key))
        if not clean:
            return {}
        result: dict[str, str] = {}
        for key in clean:
            if self._is_unassigned_key(key):
                result[key] = "无画师"
        clean = [key for key in clean if not self._is_unassigned_key(key)]
        if not clean:
            return result
        with self._read() as connection:
            id_map = {
                key: self._canonical_locked(connection, key) for key in clean
            }
            ids = sorted({int(value) for value in id_map.values() if value.isdigit()})
            names: dict[int, str] = {}
            for start in range(0, len(ids), 400):
                batch = ids[start : start + 400]
                placeholders = ",".join("?" for _ in batch)
                rows = connection.execute(
                    "SELECT id, name FROM artist WHERE id IN "
                    f"({placeholders})",
                    batch,
                ).fetchall()
                for row in rows:
                    names[int(row["id"])] = str(row["name"] or "")
            handles = self._primary_handles_locked(connection, ids)
            for key, value in id_map.items():
                if not value.isdigit():
                    continue
                artist_id = int(value)
                result[key] = (
                    names.get(artist_id)
                    or handles.get(artist_id)
                    or str(artist_id)
                )
        return result

    def author_groups(self, keys: Sequence[str]) -> dict[str, list[dict[str, Any]]]:
        """Return each artist key mapped to its linked platform accounts."""
        wanted = list(dict.fromkeys(str(key) for key in keys if key))
        if not wanted:
            return {}
        with self._read() as connection:
            roots = {
                key: self._canonical_locked(connection, key) for key in wanted
            }
            ids = sorted({int(value) for value in roots.values() if value.isdigit()})
            groups: dict[int, list[dict[str, Any]]] = {}
            if ids:
                artist_rows_all: list[sqlite3.Row] = []
                identity_rows_all: list[sqlite3.Row] = []
                for start in range(0, len(ids), 400):
                    batch = ids[start : start + 400]
                    placeholders = ",".join("?" for _ in batch)
                    artist_rows_all.extend(
                        connection.execute(
                            "SELECT id, name, avatar_url FROM artist WHERE id IN "
                            f"({placeholders})",
                            batch,
                        ).fetchall()
                    )
                    identity_rows_all.extend(
                        connection.execute(
                            "SELECT * FROM artist_identity WHERE artist_id IN "
                            f"({placeholders}) ORDER BY artist_id, kind, identity",
                            batch,
                        ).fetchall()
                    )
                artists = {int(row["id"]): row for row in artist_rows_all}
                by_artist: dict[int, list[sqlite3.Row]] = {}
                for row in identity_rows_all:
                    by_artist.setdefault(int(row["artist_id"]), []).append(row)
                for artist_id, rows in by_artist.items():
                    artist = artists.get(artist_id)
                    groups[artist_id] = self._member_views(
                        artist, rows
                    )
        result: dict[str, list[dict[str, Any]]] = {}
        for key, root in roots.items():
            if self._is_unassigned_key(key):
                result[key] = []
            elif root.isdigit() and int(root) in groups:
                result[key] = groups[int(root)]
        if ids:
            placeholders = ",".join("?" for _ in ids)
            with self._read() as connection:
                for row in connection.execute(
                    "SELECT legacy_key, artist_id FROM artist_legacy "
                    f"WHERE artist_id IN ({placeholders})",
                    ids,
                ).fetchall():
                    artist_id = int(row["artist_id"])
                    if artist_id in groups:
                        result.setdefault(str(row["legacy_key"]), groups[artist_id])
        return result

    def author_summary(self, key: str) -> dict[str, Any] | None:
        if self._is_unassigned_key(key):
            point_ids = self.point_ids_for_author(UNASSIGNED_AUTHOR_KEY)
            return {
                "id": 0,
                "key": UNASSIGNED_AUTHOR_KEY,
                "name": "无画师",
                "handle": "",
                "avatar_url": None,
                "note": None,
                "profile_fetched": 0,
                "image_count": len(point_ids),
                "created_at": 0,
                "updated_at": 0,
                "identities": [],
                "members": [],
                "pixiv_ids": [],
                "x_handles": [],
                "handles": [],
                "xiaohongshu_ids": [],
                "xiaohongshu_names": [],
                "links": [],
                "cover_point_id": None,
                "cover_url": None,
                "cover_source": None,
            }
        with self._read() as connection:
            artist_id = self._canonical_locked(connection, key)
            if not artist_id.isdigit():
                return None
            row = connection.execute(
                "SELECT * FROM artist WHERE id = ?", (int(artist_id),)
            ).fetchone()
            if row is None:
                return None
            identities = connection.execute(
                "SELECT * FROM artist_identity WHERE artist_id = ? "
                "ORDER BY kind, identity",
                (int(artist_id),),
            ).fetchall()
        summary = self._author_view(row, identities)
        summary["members"] = self._member_views(row, identities)
        summary["image_count"] = int(row["image_count"] or 0)
        cover_identity = next(
            (
                identity
                for identity in summary["identities"]
                if identity["kind"] == IDENTITY_COVER
            ),
            None,
        )
        if cover_identity and cover_identity.get("source") == "upload":
            summary["cover_point_id"] = None
            summary["cover_url"] = cover_identity.get("profile_url") or None
            summary["cover_source"] = "upload"
        elif cover_identity:
            summary["cover_point_id"] = cover_identity["identity"]
            summary["cover_url"] = None
            summary["cover_source"] = "manual"
        else:
            summary["cover_point_id"] = None
            summary["cover_url"] = None
            summary["cover_source"] = None
        return summary

    def unassigned_count(self) -> int:
        with self._read() as connection:
            row = connection.execute(
                "SELECT COUNT(*) FROM image_source "
                "WHERE author_key IS NULL OR author_key = ''"
            ).fetchone()
        return int(row[0]) if row is not None else 0

    def unassigned_count_for_points(self, point_ids: Sequence[str]) -> int:
        with self._read() as connection:
            self._scope_temp_table(connection, point_ids)
            row = connection.execute(
                "SELECT COUNT(*) FROM image_source JOIN _scope_points USING(point_id) "
                "WHERE image_source.author_key IS NULL OR image_source.author_key = ''"
            ).fetchone()
        return int(row[0]) if row is not None else 0

    @staticmethod
    def _identity_dict(row: sqlite3.Row) -> dict[str, Any]:
        kind = str(row["kind"])
        return {
            "kind": kind,
            "identity": str(row["identity"]),
            "platform": _identity_kind_platform(kind),
            "handle": str(row["display"] or row["identity"]),
            "display": str(row["display"] or row["identity"]),
            "profile_url": str(row["profile_url"] or ""),
            "source": str(row["source"] or ""),
        }

    def _author_view(
        self, row: sqlite3.Row, identities: Sequence[sqlite3.Row]
    ) -> dict[str, Any]:
        artist_id = int(row["id"])
        identity_dicts = [self._identity_dict(item) for item in identities]
        pixiv_ids = sorted(
            {
                item["identity"]
                for item in identity_dicts
                if item["kind"] == IDENTITY_PIXIV
            }
        )
        x_handles = sorted(
            {
                item["display"]
                for item in identity_dicts
                if item["kind"] == IDENTITY_TWITTER
                or (
                    item["kind"] == IDENTITY_HANDLE
                    and item["source"] == "twitter"
                )
            },
            key=str.casefold,
        )
        handles = sorted(
            {
                item["display"]
                for item in identity_dicts
                if item["kind"] == IDENTITY_HANDLE
            },
            key=str.casefold,
        )
        links = [item for item in identity_dicts if item["kind"] == IDENTITY_LINK]
        xhs_names = sorted(
            {
                item["display"]
                for item in identity_dicts
                if item["kind"] == IDENTITY_XIAOHONGSHU
            },
            key=str.casefold,
        )
        primary = (
            (x_handles[0] if x_handles else "")
            or (handles[0] if handles else "")
            or (xhs_names[0] if xhs_names else "")
            or str(row["name"] or "")
        )
        return {
            "id": artist_id,
            "key": str(artist_id),
            "name": row["name"],
            "handle": primary,
            "avatar_url": row["avatar_url"],
            "note": row["note"],
            "profile_fetched": int(row["profile_fetched"] or 0),
            "image_count": int(row["image_count"] or 0),
            "created_at": int(row["created_at"]),
            "updated_at": int(row["updated_at"]),
            "identities": identity_dicts,
            "pixiv_ids": pixiv_ids,
            "x_handles": x_handles,
            "handles": handles,
            "xiaohongshu_ids": sorted(
                {
                    item["identity"]
                    for item in identity_dicts
                    if item["kind"] == IDENTITY_XIAOHONGSHU
                }
            ),
            "xiaohongshu_names": xhs_names,
            "links": links,
        }

    def _member_views(
        self, artist: sqlite3.Row | None, identities: Sequence[sqlite3.Row]
    ) -> list[dict[str, Any]]:
        return [member for member, _rows in self._member_groups(artist, identities)]

    def _member_groups(
        self, artist: sqlite3.Row | None, identities: Sequence[sqlite3.Row]
    ) -> list[tuple[dict[str, Any], list[sqlite3.Row]]]:
        """Group identity rows into displayable social members.

        One member may be backed by several rows (an X numeric ID and its
        handle), so manual order/activity is stored on every backing row and
        read back as ``min(sort_order)`` / ``any(inactive)``.  Members without a
        manual position keep the natural ``kind, identity`` order after the
        positioned ones.
        """
        name = str(artist["name"] or "") if artist is not None else ""
        avatar = str(artist["avatar_url"] or "") if artist is not None else ""
        groups: dict[tuple[str, str], tuple[dict[str, Any], list[sqlite3.Row]]] = {}
        for row in identities:
            kind = str(row["kind"])
            identity = str(row["identity"])
            display = str(row["display"] or identity)
            if kind == IDENTITY_PIXIV:
                platform = "pixiv"
                platform_id = identity
                handle = display
                profile = str(row["profile_url"] or f"https://www.pixiv.net/users/{identity}")
                key = (platform, platform_id)
            elif kind == IDENTITY_TWITTER:
                platform = "twitter"
                platform_id = identity
                handle = display
                profile = str(row["profile_url"] or f"https://x.com/{display}")
                key = (platform, display.casefold())
            elif kind == IDENTITY_HANDLE and str(row["source"] or "") == "twitter":
                platform = "twitter"
                platform_id = display
                handle = display
                profile = str(row["profile_url"] or f"https://x.com/{display}")
                key = (platform, display.casefold())
            elif kind == IDENTITY_XIAOHONGSHU:
                platform = "xiaohongshu"
                platform_id = identity
                handle = display
                profile = str(row["profile_url"] or xiaohongshu_user_url(identity))
                key = (platform, identity)
            elif kind == IDENTITY_LINK:
                platform = "link"
                platform_id = identity
                handle = display
                profile = str(row["profile_url"] or identity)
                key = (platform, identity.casefold())
            else:
                continue
            if key in groups:
                groups[key][1].append(row)
                continue
            groups[key] = (
                {
                    "key": f"{platform}:{platform_id}",
                    "platform": platform,
                    "platform_id": platform_id,
                    "handle": handle,
                    "name": name,
                    "profile_url": profile,
                    "avatar_url": avatar,
                },
                [row],
            )
        ordered: list[tuple[int, int, int, dict[str, Any], list[sqlite3.Row]]] = []
        for index, (member, rows) in enumerate(groups.values()):
            positions = [
                int(value)
                for value in (_row_get(row, "sort_order") for row in rows)
                if value is not None
            ]
            member["active"] = not any(int(_row_get(row, "inactive") or 0) for row in rows)
            if positions:
                ordered.append((0, min(positions), index, member, rows))
            else:
                ordered.append((1, 0, index, member, rows))
        ordered.sort(key=lambda item: item[:3])
        result: list[tuple[dict[str, Any], list[sqlite3.Row]]] = []
        for position, (_, _, _, member, rows) in enumerate(ordered):
            member["position"] = position
            result.append((member, rows))
        return result

    def set_member_preferences(
        self, artist_key: str, members: Sequence[dict[str, Any]]
    ) -> str:
        """Persist the display order and active flag of an artist's members.

        ``members`` is the desired order; each item carries ``key`` (the
        member key returned by :meth:`author_summary`) and ``active``.  Members
        that are not listed keep their flags and move after the listed ones.
        """
        if self._is_unassigned_key(artist_key):
            raise ValueError("无画师没有社交身份")
        with self._read() as connection:
            base = self._canonical_locked(connection, artist_key)
            if not base.isdigit():
                raise ValueError("作者不存在")
            artist_id = int(base)
            artist = connection.execute(
                "SELECT * FROM artist WHERE id = ?", (artist_id,)
            ).fetchone()
            identities = connection.execute(
                "SELECT * FROM artist_identity WHERE artist_id = ? ORDER BY kind, identity",
                (artist_id,),
            ).fetchall()
            groups = self._member_groups(artist, identities)
            by_key = {str(member["key"]): rows for member, rows in groups}
            listed: list[str] = []
            flags: dict[str, bool] = {}
            for item in members:
                key = str(item.get("key") or "")
                if key not in by_key:
                    raise ValueError(f"社交账号不存在：{key}")
                if key in flags:
                    continue
                listed.append(key)
                flags[key] = bool(item.get("active", True))
            rest = [str(member["key"]) for member, _ in groups if str(member["key"]) not in flags]
            now = _now()
            for position, key in enumerate(listed + rest):
                for row in by_key[key]:
                    if key in flags:
                        connection.execute(
                            "UPDATE artist_identity SET sort_order = ?, inactive = ?, "
                            "updated_at = ? WHERE kind = ? AND identity = ?",
                            (position, 0 if flags[key] else 1, now, row["kind"], row["identity"]),
                        )
                    else:
                        connection.execute(
                            "UPDATE artist_identity SET sort_order = ?, updated_at = ? "
                            "WHERE kind = ? AND identity = ?",
                            (position, now, row["kind"], row["identity"]),
                        )
            connection.execute(
                "UPDATE artist SET updated_at = ? WHERE id = ?", (now, artist_id)
            )
            connection.commit()
        return base

    def _primary_handles_locked(
        self, connection: sqlite3.Connection, ids: Sequence[int]
    ) -> dict[int, str]:
        if not ids:
            return {}
        rows: list[sqlite3.Row] = []
        for start in range(0, len(ids), 400):
            batch = list(ids[start : start + 400])
            placeholders = ",".join("?" for _ in batch)
            rows.extend(
                connection.execute(
                    "SELECT artist_id, kind, identity, display, source "
                    "FROM artist_identity WHERE artist_id IN "
                    f"({placeholders})",
                    batch,
                ).fetchall()
            )
        best: dict[int, tuple[int, str]] = {}
        for row in rows:
            artist_id = int(row["artist_id"])
            kind = str(row["kind"])
            identity = str(row["identity"])
            display = str(row["display"] or identity)
            source = str(row["source"] or "")
            if kind == IDENTITY_TWITTER:
                rank = 4
            elif kind == IDENTITY_HANDLE and source == "twitter":
                rank = 3
            elif kind == IDENTITY_HANDLE:
                rank = 2
            elif kind == IDENTITY_PIXIV:
                rank = 1
            else:
                rank = 0
            if rank > best.get(artist_id, (0, ""))[0]:
                best[artist_id] = (rank, display)
        return {artist_id: value[1] for artist_id, value in best.items()}

    @staticmethod
    def _author_text(value: Any) -> str:
        return _author_text(value)

    # ------------------------------------------------------ artist search index
    @staticmethod
    def _artist_index_signature(
        connection: sqlite3.Connection,
    ) -> tuple[int, int, int, int, int]:
        """Cheap fingerprint of every table the artist index is built from."""
        artist = connection.execute(
            "SELECT COUNT(*) AS total, COALESCE(MAX(id), 0) AS newest, "
            "COALESCE(SUM(updated_at), 0) AS touched, "
            "COALESCE(SUM(LENGTH(COALESCE(name, ''))), 0) AS name_chars "
            "FROM artist"
        ).fetchone()
        identity = connection.execute(
            "SELECT COUNT(*) AS total, "
            "COALESCE(SUM(LENGTH(COALESCE(display, identity, ''))), 0) AS chars "
            "FROM artist_identity"
        ).fetchone()
        return (
            int(artist["total"]),
            int(artist["newest"]),
            int(artist["touched"]),
            int(artist["name_chars"]),
            int(identity["total"]) * 1_000_003 + int(identity["chars"]),
        )

    def _build_artist_index(
        self,
        connection: sqlite3.Connection,
        signature: tuple[int, int, int, int, int],
    ) -> _ArtistSearchIndex:
        identities: dict[int, list[str]] = {}
        for row in connection.execute(
            "SELECT artist_id, identity, display FROM artist_identity"
        ):
            bucket = identities.setdefault(int(row["artist_id"]), [])
            bucket.extend(
                str(value) for value in (row["display"], row["identity"]) if value
            )
        values: dict[int, tuple[NameKey, ...]] = {}
        by_key: dict[str, set[int]] = {}
        initials: dict[str, set[int]] = {}
        for row in connection.execute("SELECT id, name FROM artist"):
            artist_id = int(row["id"])
            keys = name_match.build_all(
                [row["name"], str(artist_id), *(identities.get(artist_id) or ())]
            )
            if not keys:
                continue
            values[artist_id] = keys
            for key in keys:
                for token in (key.compact, *key.keys):
                    if len(token) < 2:
                        continue
                    by_key.setdefault(token, set()).add(artist_id)
                    initials.setdefault(token[:1], set()).add(artist_id)
        return _ArtistSearchIndex(
            signature=signature,
            built_at=time.monotonic(),
            values=values,
            entries=tuple(by_key.items()),
            initials=initials,
        )

    def _artist_search_index(
        self, connection: sqlite3.Connection
    ) -> tuple[_ArtistSearchIndex, bool]:
        """Return the artist index and whether it reflects the current tables."""
        lock = self._ensure_artist_index_cache()
        signature = self._artist_index_signature(connection)
        cached = self._artist_index
        if cached is not None and cached.signature == signature:
            return cached, True
        with lock:
            cached = self._artist_index
            if cached is not None and cached.signature == signature:
                return cached, True
            # Bulk imports touch the artist tables constantly; rebuilding on
            # every keystroke would be wasteful, so serve the previous index
            # for a short window instead of thrashing. Callers compensate for
            # the lag by also looking at rows the index has never seen.
            if cached is not None and time.monotonic() - cached.built_at < 1.5:
                return cached, False
            index = self._build_artist_index(connection, signature)
            self._artist_index = index
            return index, True

    @staticmethod
    def _author_like_ids(connection: sqlite3.Connection, needle: str) -> set[int]:
        search = needle.split(":", 1)[-1] if ":" in needle else needle
        pattern = (
            "%" + search.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
        )
        rows = connection.execute(
            """
            SELECT DISTINCT artist.id
            FROM artist
            LEFT JOIN artist_identity
              ON artist_identity.artist_id = artist.id
            WHERE artist.name_key LIKE ? ESCAPE '\\'
               OR artist_identity.identity LIKE ? ESCAPE '\\'
               OR LOWER(artist_identity.display) LIKE ? ESCAPE '\\'
            LIMIT 8000
            """,
            (pattern, pattern, pattern),
        ).fetchall()
        return {int(row[0]) for row in rows}

    @staticmethod
    def _author_rows(
        connection: sqlite3.Connection,
        artist_ids: Sequence[int],
    ) -> list[sqlite3.Row]:
        rows: list[sqlite3.Row] = []
        ids = list(dict.fromkeys(int(value) for value in artist_ids))
        for chunk_start in range(0, len(ids), 800):
            batch = ids[chunk_start : chunk_start + 800]
            placeholders = ",".join("?" for _ in batch)
            rows.extend(
                connection.execute(
                    "SELECT id, name, image_count FROM artist "
                    f"WHERE id IN ({placeholders})",
                    batch,
                ).fetchall()
            )
        return rows

    def list_authors(
        self,
        query: str = "",
        *,
        limit: int = 40,
        offset: int = 0,
        image_counts: dict[str, int] | None = None,
    ) -> list[dict[str, Any]]:
        needle_text = _author_text(query)
        page_size = max(1, min(int(limit), 500))
        start = max(0, int(offset))
        needle = name_match.build(needle_text)
        with self._read() as connection:
            rows: list[sqlite3.Row]
            index: _ArtistSearchIndex | None = None
            if needle.empty:
                rows = connection.execute(
                    "SELECT id, name, image_count FROM artist"
                ).fetchall()
            else:
                index, fresh = self._artist_search_index(connection)
                candidate_ids = index.candidates(needle)
                candidate_ids |= self._author_like_ids(connection, needle_text)
                # Typo tolerance: candidates that share the query's first
                # character still get a difflib pass, which keeps the scan
                # bounded instead of walking the whole artist table.
                candidate_ids |= index.typo_candidates(needle)
                if not fresh:
                    # Artists created after the cached index was built.
                    candidate_ids.update(
                        int(row[0])
                        for row in connection.execute(
                            "SELECT id FROM artist WHERE id > ?", (index.newest_id,)
                        )
                    )
                rows = self._author_rows(connection, candidate_ids)
                rows.sort(key=lambda row: -int(row["image_count"] or 0))
                rows = rows[:4000]
            scored: list[tuple[float, sqlite3.Row, int]] = []
            for row in rows:
                artist_id = int(row["id"])
                scoped_count = (
                    int(image_counts.get(str(artist_id), 0))
                    if image_counts is not None
                    else int(row["image_count"] or 0)
                )
                score = 1.0
                if index is not None:
                    prepared = index.values.get(artist_id) or ()
                    score = name_match.best_score(needle, prepared)
                    # The index may lag behind a fresh rename; the row itself is
                    # always authoritative.
                    score = max(
                        score,
                        name_match.score(needle, name_match.build_cached(str(row["name"] or ""))),
                    )
                    if score <= 0:
                        continue
                scored.append((score, row, scoped_count))
            if index is not None:
                scored.sort(
                    key=lambda item: (
                        -item[0],
                        -item[2],
                        str(item[1]["name"] or item[1]["id"]).casefold(),
                    )
                )
            else:
                scored.sort(
                    key=lambda item: (
                        -item[2],
                        str(item[1]["name"] or item[1]["id"]).casefold(),
                    )
                )
            page = scored[start : start + page_size]
            page_ids = [int(item[1]["id"]) for item in page]
            views = self._artist_views_locked(connection, page_ids)
        result: list[dict[str, Any]] = []
        for score, row, count in page:
            view = views.get(int(row["id"]))
            if view is None:
                continue
            view["image_count"] = count
            if index is not None:
                view["score"] = round(score, 3)
            result.append(view)
        return result

    def artist_entries(
        self,
        point_ids: Sequence[str] | None,
    ) -> list[dict[str, Any]]:
        """Aggregate author counts/latest/covers for one scope, unsorted."""
        with self._read() as connection:
            if point_ids is None:
                aggregate = connection.execute(
                    "SELECT author_key, COUNT(*) AS count, "
                    "MAX(COALESCE(mtime_ns, 0)) AS latest "
                    "FROM image_source GROUP BY author_key"
                ).fetchall()
            else:
                self._scope_temp_table(connection, point_ids)
                aggregate = connection.execute(
                    "SELECT image_source.author_key AS author_key, COUNT(*) AS count, "
                    "MAX(COALESCE(image_source.mtime_ns, 0)) AS latest "
                    "FROM image_source JOIN _scope_points USING(point_id) "
                    "GROUP BY image_source.author_key"
                ).fetchall()
            raw: dict[str, tuple[int, int]] = {}
            for row in aggregate:
                key = row["author_key"]
                target = UNASSIGNED_AUTHOR_KEY if key in (None, "") else str(key)
                count, latest = raw.get(target, (0, 0))
                raw[target] = (count + int(row["count"]), max(latest, int(row["latest"] or 0)))
            numeric_ids = sorted(
                int(key) for key in raw if key.isdigit() and key != UNASSIGNED_AUTHOR_KEY
            )
            artist_rows: dict[int, sqlite3.Row] = {}
            identities: dict[int, list[sqlite3.Row]] = {}
            for chunk_start in range(0, len(numeric_ids), 400):
                batch = numeric_ids[chunk_start : chunk_start + 400]
                placeholders = ",".join("?" for _ in batch)
                for row in connection.execute(
                    f"SELECT * FROM artist WHERE id IN ({placeholders})", batch
                ).fetchall():
                    artist_rows[int(row["id"])] = row
                for row in connection.execute(
                    "SELECT * FROM artist_identity WHERE artist_id IN "
                    f"({placeholders}) ORDER BY artist_id, kind, identity",
                    batch,
                ).fetchall():
                    identities.setdefault(int(row["artist_id"]), []).append(row)
            entries: list[dict[str, Any]] = []
            for key, (count, latest) in raw.items():
                if key == UNASSIGNED_AUTHOR_KEY:
                    if count <= 0:
                        continue
                    name = "无画师"
                    handle = ""
                    members: list[dict[str, Any]] = []
                    search_values = ["无画师", "none", "未标注", "unassigned"]
                    is_unassigned = True
                else:
                    artist_id = int(key)
                    row = artist_rows.get(artist_id)
                    if row is None:
                        continue
                    identity_rows = identities.get(artist_id, [])
                    view = self._author_view(row, identity_rows)
                    name = str(view["name"] or view["handle"] or key)
                    handle = str(view["handle"] or "")
                    members = self._member_views(row, identity_rows)
                    search_values = [
                        name,
                        key,
                        handle,
                        *[
                            value
                            for member in members
                            for value in (
                                member.get("handle"),
                                member.get("platform_id"),
                            )
                        ],
                        *[
                            value
                            for identity in view["identities"]
                            for value in (identity.get("display"), identity.get("identity"))
                        ],
                    ]
                    is_unassigned = False
                clean_values: list[Any] = []
                seen_values: set[str] = set()
                for value in search_values:
                    text = str(value or "").strip()
                    if not text:
                        continue
                    folded = fold_text(text)
                    if folded in seen_values:
                        continue
                    seen_values.add(folded)
                    clean_values.append(text)
                entries.append(
                    {
                        "key": key,
                        "name": name,
                        "handle": handle,
                        "members": members,
                        "image_count": count,
                        "latest_mtime": latest,
                        "_unassigned": is_unassigned,
                        "_search_values": clean_values,
                        "_search_text": "\n".join(fold_text(value) for value in clean_values),
                    }
                )
        return entries

    @classmethod
    def sort_artist_entries(
        cls,
        entries: Sequence[dict[str, Any]],
        sort: str = "latest_desc",
        *,
        include_unassigned: bool = True,
    ) -> list[dict[str, Any]]:
        order = str(sort or "latest_desc")
        if order not in {
            "name_asc",
            "name_desc",
            "latest_asc",
            "latest_desc",
            "count_asc",
            "count_desc",
        }:
            order = "latest_desc"
        ordered = [
            item
            for item in entries
            if include_unassigned or item["key"] != UNASSIGNED_AUTHOR_KEY
        ]
        if order in {"name_asc", "name_desc"}:
            ordered.sort(
                key=lambda item: (_author_text(item["name"] or item["key"]), item["key"]),
                reverse=order == "name_desc",
            )
        elif order in {"latest_asc", "latest_desc"}:
            ordered.sort(
                key=lambda item: (
                    item["latest_mtime"],
                    _author_text(item["name"] or item["key"]),
                    item["key"],
                ),
                reverse=order == "latest_desc",
            )
        else:
            ordered.sort(
                key=lambda item: (
                    item["image_count"],
                    _author_text(item["name"] or item["key"]),
                    item["key"],
                ),
                reverse=order == "count_desc",
            )
        ordered.sort(key=lambda item: not item["_unassigned"])
        return ordered

    def search_artist_entries(
        self,
        entries: Sequence[dict[str, Any]],
        query: str,
        sort: str = "latest_desc",
        *,
        include_unassigned: bool = True,
    ) -> list[dict[str, Any]]:
        needle = name_match.build(_author_text(query))
        ordered = self.sort_artist_entries(
            entries, sort, include_unassigned=include_unassigned
        )
        if needle.empty:
            return ordered
        with self._read() as connection:
            index, fresh = self._artist_search_index(connection)
        pooled = index.candidates(needle) | index.typo_candidates(needle)
        scored: list[dict[str, Any]] = []
        for item in ordered:
            key = str(item.get("key") or "")
            artist_id = int(key) if key.isdigit() else 0
            prepared = index.values.get(artist_id) if artist_id else None
            if prepared is None or not fresh:
                # Unindexed rows, and every row while the index is throttled
                # after a rename/import, are scored from the authoritative
                # entry values so a fresh name is never hidden by a stale key.
                value = name_match.best_score(
                    needle,
                    name_match.build_all(
                        item.get("_search_values") or [item.get("name"), key]
                    ),
                )
            else:
                if artist_id not in pooled:
                    # The index is current, so staying out of the candidate
                    # pool means this artist cannot match; skip the costly
                    # per-entry scoring pass.
                    continue
                value = name_match.best_score(needle, prepared)
            if value <= 0:
                continue
            copy = dict(item)
            copy["score"] = round(value, 3)
            scored.append(copy)
        scored.sort(key=lambda item: (-item["score"], not item["_unassigned"]))
        return scored

    def artist_stats(
        self,
        point_ids: Sequence[str] | None,
        *,
        query: str = "",
        sort: str = "latest_desc",
        include_unassigned: bool = False,
        offset: int = 0,
        limit: int = 30,
    ) -> dict[str, Any]:
        """Aggregate artists for one filter scope, ordered for the artist view."""
        entries = self.artist_entries(point_ids)
        if query.strip():
            ranked = self.search_artist_entries(
                entries,
                query,
                sort,
                include_unassigned=include_unassigned,
            )
        else:
            ranked = self.sort_artist_entries(
                entries,
                sort,
                include_unassigned=include_unassigned,
            )
        start = max(0, int(offset))
        page_size = max(1, int(limit))
        total = len(ranked)
        page: list[dict[str, Any]] = []
        for item in ranked[start : start + page_size]:
            clean = dict(item)
            clean.pop("_unassigned", None)
            clean.pop("_search_values", None)
            clean.pop("_search_text", None)
            page.append(clean)
        return {"authors": page, "total": total}

    def _artist_views_locked(
        self, connection: sqlite3.Connection, ids: Sequence[int]
    ) -> dict[int, dict[str, Any]]:
        clean = sorted({int(value) for value in ids})
        if not clean:
            return {}
        placeholders = ",".join("?" for _ in clean)
        rows = connection.execute(
            f"SELECT * FROM artist WHERE id IN ({placeholders})", clean
        ).fetchall()
        identity_rows = connection.execute(
            "SELECT * FROM artist_identity WHERE artist_id IN "
            f"({placeholders}) ORDER BY artist_id, kind, identity",
            clean,
        ).fetchall()
        by_artist: dict[int, list[sqlite3.Row]] = {}
        for row in identity_rows:
            by_artist.setdefault(int(row["artist_id"]), []).append(row)
        return {
            int(row["id"]): self._author_view(row, by_artist.get(int(row["id"]), []))
            for row in rows
        }

    def count_authors(self) -> int:
        with self._read() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM artist").fetchone()[0])

    # -------------------------------------------------------------- fetch state

    def domain_state(self, domain: str) -> dict[str, Any]:
        with self._read() as connection:
            row = connection.execute(
                "SELECT * FROM fetch_state WHERE domain = ?", (domain,)
            ).fetchone()
        if row is None:
            return {
                "domain": domain,
                "blocked_until": 0,
                "last_request": 0.0,
                "day": "",
                "day_count": 0,
                "long_remaining": None,
                "last_error": None,
            }
        return dict(row)

    def note_request(self, domain: str) -> None:
        day = time.strftime("%Y-%m-%d", time.gmtime())
        now = time.time()
        with self._read() as connection:
            row = connection.execute(
                "SELECT day, day_count FROM fetch_state WHERE domain = ?", (domain,)
            ).fetchone()
            count = int(row["day_count"] or 0) if row and row["day"] == day else 0
            connection.execute(
                """
                INSERT INTO fetch_state(domain, last_request, day, day_count)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(domain) DO UPDATE SET
                    last_request = excluded.last_request,
                    day = excluded.day,
                    day_count = excluded.day_count
                """,
                (domain, now, day, count + 1),
            )
            connection.commit()

    def set_blocked(self, domain: str, seconds: int, error: str | None = None) -> int:
        blocked_until = _now() + max(0, int(seconds))
        with self._read() as connection:
            connection.execute(
                """
                INSERT INTO fetch_state(domain, blocked_until, last_error)
                VALUES (?, ?, ?)
                ON CONFLICT(domain) DO UPDATE SET
                    blocked_until = excluded.blocked_until,
                    last_error = excluded.last_error
                """,
                (domain, blocked_until, error),
            )
            connection.commit()
        return blocked_until

    def set_long_remaining(self, domain: str, remaining: int | None) -> None:
        with self._read() as connection:
            connection.execute(
                """
                INSERT INTO fetch_state(domain, long_remaining)
                VALUES (?, ?)
                ON CONFLICT(domain) DO UPDATE SET
                    long_remaining = excluded.long_remaining
                """,
                (domain, remaining),
            )
            connection.commit()

    # ------------------------------------------------------------------- stats

    def stats(self) -> dict[str, Any]:
        with self._read() as connection:
            rows = connection.execute(
                "SELECT platform, status, COUNT(*) AS count FROM image_source "
                "GROUP BY platform, status"
            ).fetchall()
            total = connection.execute("SELECT COUNT(*) FROM image_source").fetchone()[0]
            authors = connection.execute("SELECT COUNT(*) FROM artist").fetchone()[0]
            with_metadata = connection.execute(
                "SELECT COUNT(*) FROM image_source "
                "WHERE title IS NOT NULL OR author_key IS NOT NULL"
            ).fetchone()[0]
            domains = connection.execute("SELECT * FROM fetch_state").fetchall()
        by_platform: dict[str, dict[str, int]] = {}
        for row in rows:
            platform = str(row["platform"])
            by_platform.setdefault(platform, {})[str(row["status"])] = int(row["count"])
        return {
            "total": int(total),
            "with_metadata": int(with_metadata),
            "authors": int(authors),
            "by_platform": by_platform,
            "domains": {str(row["domain"]): dict(row) for row in domains},
        }

    # --------------------------------------------------------------- internals

    def _row_to_record(self, row: sqlite3.Row) -> dict[str, Any]:
        record = dict(row)
        record["tags"] = _loads(record.pop("tags_json", None), [])
        record["manual_tags"] = _loads(record.pop("manual_tags_json", None), [])
        record["auto_tags"] = _loads(record.pop("auto_tags_json", None), [])
        record["auto_tag_scores"] = _loads(record.pop("auto_tag_scores_json", None), {})
        record["candidates"] = _loads(record.pop("candidates_json", None), [])
        record["manual_fields"] = _loads(record.pop("manual_fields", None), [])
        return record


def _row_get(row: sqlite3.Row, name: str) -> Any:
    """Read an optional column from rows selected before a schema upgrade."""
    try:
        return row[name]
    except (IndexError, KeyError):
        return None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _ref_dict(ref: SourceRef) -> dict[str, Any]:
    return {
        "platform": ref.platform,
        "source_id": ref.source_id,
        "page": ref.page,
        "url": ref.url,
        "origin": ref.origin,
        "confidence": ref.confidence,
    }


def ref_from_dict(value: dict[str, Any]) -> SourceRef | None:
    platform = str(value.get("platform") or "")
    source_id = str(value.get("source_id") or "")
    if platform not in {"pixiv", "twitter", "other"} or not source_id:
        return None
    page = value.get("page")
    return SourceRef(
        platform=platform,  # type: ignore[arg-type]
        source_id=source_id,
        page=int(page) if page is not None else None,
        url=str(value.get("url") or ""),
        origin=str(value.get("origin") or "filename"),  # type: ignore[arg-type]
        confidence=float(value.get("confidence") or 0.5),
    )


def valid_source_url(platform: str, source_id: str, page: int | None) -> str:
    if platform == "pixiv":
        return canonical_pixiv_url(source_id)
    if platform == "twitter":
        return twitter_photo_url(source_id, page)
    return ""

