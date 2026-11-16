from __future__ import annotations

import difflib
import hashlib
import json
import re
import sqlite3
import threading
import time
import unicodedata
from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .parser import ParsedSource, SourceRef, canonical_pixiv_url, twitter_photo_url

SCHEMA_VERSION = 4
ORIGIN_PRIORITY = {"filename": 1, "embedded": 2, "manual": 3, "saucy": 4, "ascii2d": 4}
AUTHOR_TTL_SECONDS = 30 * 24 * 60 * 60
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
    return str(value or "").strip().lstrip("@").casefold()


class ProvenanceStore:
    """Transactional, user-owned source metadata keyed by Qdrant point ID.

    The store is deliberately independent of the vector payload. It survives a
    re-index, a model change, and an index-directory move, and it never mutates
    an original image.
    """

    def __init__(self, path: Path):
        self.path = path
        self._schema_lock = threading.Lock()
        self._schema_ready = False

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(self.path), timeout=15)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 15000")
        connection.execute("PRAGMA foreign_keys = ON")
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
                CREATE TABLE IF NOT EXISTS author (
                    key TEXT PRIMARY KEY,
                    platform TEXT NOT NULL,
                    platform_id TEXT NOT NULL,
                    name TEXT,
                    handle TEXT,
                    profile_url TEXT,
                    avatar_url TEXT,
                    social_json TEXT,
                    profile_fetched INTEGER NOT NULL DEFAULT 0,
                    updated_at INTEGER NOT NULL,
                    UNIQUE(platform, platform_id)
                );
                CREATE TABLE IF NOT EXISTS author_alias (
                    alias_key TEXT PRIMARY KEY,
                    canonical_key TEXT NOT NULL,
                    evidence TEXT,
                    confidence REAL NOT NULL DEFAULT 0.5,
                    updated_at INTEGER NOT NULL
                );
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
            columns = {
                str(row[1])
                for row in connection.execute("PRAGMA table_info(author)").fetchall()
            }
            if "profile_fetched" not in columns:
                connection.execute(
                    "ALTER TABLE author ADD COLUMN profile_fetched INTEGER NOT NULL DEFAULT 0"
                )
            source_columns = {
                str(row[1])
                for row in connection.execute("PRAGMA table_info(image_source)").fetchall()
            }
            if "mtime_ns" not in source_columns:
                connection.execute("ALTER TABLE image_source ADD COLUMN mtime_ns INTEGER")
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
    ) -> dict[str, Any] | None:
        """Create a visible "unidentified" row without guessing a source."""
        now = _now()
        digest = str(payload.get("content_hash") or "") or None
        width = _int_or_none(payload.get("width"))
        height = _int_or_none(payload.get("height"))
        mtime_ns = _int_or_none(payload.get("mtime_ns"))
        with self._read() as connection:
            existing = connection.execute(
                "SELECT platform FROM image_source WHERE point_id = ?", (point_id,)
            ).fetchone()
            if existing is not None:
                connection.execute(
                    """
                    UPDATE image_source
                    SET content_hash = COALESCE(?, content_hash),
                        path_key = COALESCE(?, path_key),
                        width = COALESCE(?, width),
                        height = COALESCE(?, height),
                        mtime_ns = COALESCE(?, mtime_ns),
                        updated_at = ?
                    WHERE point_id = ?
                    """,
                    (digest, path_key, width, height, mtime_ns, now, point_id),
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
                    tags_json = NULL, raw_json = NULL, fetched_at = NULL,
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
                    "ok" if ref.platform == "other" else "pending",
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
                "title = NULL, description = NULL, author_key = NULL, tags_json = NULL, "
                "status = 'pending', updated_at = ? WHERE point_id = ?",
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

    # ------------------------------------------------------------------ authors

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
        stored_id = _normalized_handle(platform_id) if platform == "twitter" else platform_id
        key = f"{platform}:{stored_id}"
        now = _now()
        with self._read() as connection:
            connection.execute(
                """
                INSERT INTO author (
                    key, platform, platform_id, name, handle, profile_url,
                    avatar_url, social_json, profile_fetched, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    name = COALESCE(excluded.name, author.name),
                    handle = COALESCE(excluded.handle, author.handle),
                    profile_url = COALESCE(excluded.profile_url, author.profile_url),
                    avatar_url = COALESCE(excluded.avatar_url, author.avatar_url),
                    social_json = COALESCE(excluded.social_json, author.social_json),
                    profile_fetched = MAX(author.profile_fetched, excluded.profile_fetched),
                    updated_at = excluded.updated_at
                """,
                (
                    key,
                    platform,
                    stored_id,
                    name,
                    handle,
                    profile_url,
                    avatar_url,
                    _json(social or {}, {}),
                    1 if profile_fetched else 0,
                    now,
                ),
            )
            connection.commit()
        return key

    def get_author(self, key: str) -> dict[str, Any] | None:
        with self._read() as connection:
            row = connection.execute("SELECT * FROM author WHERE key = ?", (key,)).fetchone()
        return self._row_to_author(row) if row is not None else None

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
        alias = self.canonical_author_key(alias_key)
        canonical = self.canonical_author_key(canonical_key)
        if alias == canonical:
            return canonical
        now = _now()
        with self._read() as connection:
            connection.execute(
                """
                INSERT INTO author_alias (
                    alias_key, canonical_key, evidence, confidence, updated_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(alias_key) DO UPDATE SET
                    canonical_key = excluded.canonical_key,
                    evidence = excluded.evidence,
                    confidence = excluded.confidence,
                    updated_at = excluded.updated_at
                """,
                (alias, canonical, evidence, confidence, now),
            )
            # Move any existing aliases that pointed at the old root.
            connection.execute(
                "UPDATE author_alias SET canonical_key = ?, updated_at = ? "
                "WHERE canonical_key = ? AND alias_key != ?",
                (canonical, now, alias, alias),
            )
            connection.commit()
        return canonical

    def canonical_author_key(self, key: str) -> str:
        current = key
        seen: set[str] = set()
        with self._read() as connection:
            for _ in range(8):
                if current in seen:
                    break
                seen.add(current)
                row = connection.execute(
                    "SELECT canonical_key FROM author_alias WHERE alias_key = ?", (current,)
                ).fetchone()
                if row is None:
                    break
                current = str(row["canonical_key"])
        return current

    def author_members(self, key: str) -> list[str]:
        canonical = self.canonical_author_key(key)
        with self._read() as connection:
            rows = connection.execute(
                "SELECT alias_key FROM author_alias WHERE canonical_key = ?", (canonical,)
            ).fetchall()
        members = {canonical, *(str(row["alias_key"]) for row in rows)}
        return sorted(members)

    def point_ids_for_author(self, key: str) -> list[str]:
        members = self.author_members(key)
        placeholders = ",".join("?" for _ in members)
        with self._read() as connection:
            rows = connection.execute(
                f"SELECT point_id FROM image_source WHERE author_key IN ({placeholders}) "
                "ORDER BY mtime_ns IS NULL, mtime_ns DESC, updated_at DESC, point_id",
                members,
            ).fetchall()
        return [str(row["point_id"]) for row in rows]

    def list_source_tags(self, query: str = "", *, limit: int = 60) -> list[dict[str, Any]]:
        needle = f"%{str(query or '').strip().casefold()}%"
        params: list[Any] = []
        where = "image_source.tags_json IS NOT NULL AND TRIM(value) <> ''"
        if str(query or "").strip():
            where += " AND LOWER(value) LIKE ?"
            params.append(needle)
        params.append(max(1, min(int(limit), 200)))
        with self._read() as connection:
            rows = connection.execute(
                "SELECT value AS tag, COUNT(*) AS count "
                "FROM image_source, json_each(image_source.tags_json) "
                f"WHERE {where} GROUP BY value "
                "ORDER BY count DESC, value COLLATE NOCASE LIMIT ?",
                params,
            ).fetchall()
        return [{"tag": str(row["tag"]), "count": int(row["count"])} for row in rows]

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
            "EXISTS (SELECT 1 FROM json_each(image_source.tags_json) "
            "WHERE value LIKE ? COLLATE NOCASE)"
            for _ in clean
        )
        with self._read() as connection:
            rows = connection.execute(
                f"SELECT point_id FROM image_source WHERE {clauses} "
                "ORDER BY mtime_ns IS NULL, mtime_ns DESC, updated_at DESC, point_id",
                [f"%{term}%" for term in clean],
            ).fetchall()
        return [str(row["point_id"]) for row in rows]

    def point_ids_for_tags(self, tags: Sequence[str]) -> list[str]:
        clean = [str(tag).strip() for tag in tags if str(tag).strip()]
        if not clean:
            return []
        clauses = " AND ".join(
            "EXISTS (SELECT 1 FROM json_each(image_source.tags_json) "
            "WHERE value = ? COLLATE NOCASE)"
            for _ in clean
        )
        with self._read() as connection:
            rows = connection.execute(
                f"SELECT point_id FROM image_source WHERE {clauses} "
                "ORDER BY mtime_ns IS NULL, mtime_ns DESC, updated_at DESC, point_id",
                clean,
            ).fetchall()
        return [str(row["point_id"]) for row in rows]

    def author_names(self, keys: Sequence[str]) -> dict[str, str]:
        clean = list(dict.fromkeys(str(key) for key in keys if key))
        if not clean:
            return {}
        result: dict[str, str] = {}
        with self._read() as connection:
            for start in range(0, len(clean), 256):
                batch = clean[start : start + 256]
                placeholders = ",".join("?" for _ in batch)
                rows = connection.execute(
                    f"SELECT key, name, handle FROM author WHERE key IN ({placeholders})",
                    batch,
                ).fetchall()
                for row in rows:
                    result[str(row["key"])] = str(
                        row["name"] or row["handle"] or row["key"]
                    )
        return result

    def author_groups(self, keys: Sequence[str]) -> dict[str, list[dict[str, Any]]]:
        """Return canonical/alias keys mapped to their merged author members."""
        wanted = list(dict.fromkeys(str(key) for key in keys if key))
        if not wanted:
            return {}
        with self._read() as connection:
            alias_rows = connection.execute(
                "SELECT alias_key, canonical_key FROM author_alias"
            ).fetchall()
        aliases = {
            str(row["alias_key"]): str(row["canonical_key"]) for row in alias_rows
        }

        def canonical(key: str) -> str:
            current = key
            seen: set[str] = set()
            while current in aliases and current not in seen:
                seen.add(current)
                current = aliases[current]
            return current

        canonical_to_keys: dict[str, set[str]] = {}
        for key in wanted:
            root = canonical(key)
            canonical_to_keys.setdefault(root, {root}).add(key)
        for alias_key in aliases:
            root = canonical(alias_key)
            if root in canonical_to_keys:
                canonical_to_keys[root].add(alias_key)
        all_keys = sorted({key for group in canonical_to_keys.values() for key in group})
        authors: dict[str, dict[str, Any]] = {}
        if all_keys:
            with self._read() as connection:
                for start in range(0, len(all_keys), 256):
                    batch = all_keys[start : start + 256]
                    placeholders = ",".join("?" for _ in batch)
                    rows = connection.execute(
                        "SELECT key, platform, platform_id, name, handle, profile_url "
                        f"FROM author WHERE key IN ({placeholders})",
                        batch,
                    ).fetchall()
                    for row in rows:
                        authors[str(row["key"])] = dict(row)
        result: dict[str, list[dict[str, Any]]] = {}
        for root, members in canonical_to_keys.items():
            group = [authors[key] for key in sorted(members) if key in authors]
            result[root] = group
            for key in members:
                result[key] = group
        return result

    def author_summary(self, key: str) -> dict[str, Any] | None:
        canonical = self.canonical_author_key(key)
        author = self.get_author(canonical)
        if author is None:
            return None
        members = self.author_members(canonical)
        placeholders = ",".join("?" for _ in members)
        with self._read() as connection:
            count = connection.execute(
                f"SELECT COUNT(*) FROM image_source WHERE author_key IN ({placeholders})",
                members,
            ).fetchone()[0]
            member_rows = connection.execute(
                f"SELECT * FROM author WHERE key IN ({placeholders})", members
            ).fetchall()
        author["members"] = [self._row_to_author(row) for row in member_rows]
        author["image_count"] = int(count)
        return author

    @staticmethod
    def _author_text(value: Any) -> str:
        text = unicodedata.normalize("NFKC", str(value or "")).casefold()
        text = text.lstrip("@")
        return re.sub(r"\s+", " ", text).strip()

    @classmethod
    def _author_score(cls, query: str, *values: Any) -> float:
        needle = cls._author_text(query)
        if not needle:
            return 1.0
        best = 0.0
        for raw in values:
            text = cls._author_text(raw)
            if not text:
                continue
            if text == needle:
                return 1.0
            if needle in text:
                best = max(
                    best,
                    0.85 + 0.15 * (len(needle) / max(len(text), 1)),
                )
                continue
            ratio = difflib.SequenceMatcher(None, needle, text).ratio()
            if ratio >= 0.45:
                best = max(best, ratio * 0.8)
        return best

    def list_authors(self, query: str = "", *, limit: int = 40) -> list[dict[str, Any]]:
        with self._read() as connection:
            rows = connection.execute(
                """
                SELECT author.*, COUNT(image_source.point_id) AS image_count
                FROM author
                LEFT JOIN image_source ON image_source.author_key = author.key
                GROUP BY author.key
                """
            ).fetchall()
        authors = [
            self._row_to_author(row) | {"image_count": int(row["image_count"])}
            for row in rows
        ]
        needle = query.strip()
        if not needle:
            authors.sort(
                key=lambda item: (
                    -int(item.get("image_count") or 0),
                    str(item.get("name") or item.get("handle") or item["key"]).casefold(),
                )
            )
            return authors[: max(1, min(int(limit), 200))]
        scored: list[dict[str, Any]] = []
        for author in authors:
            score = self._author_score(
                needle,
                author.get("key"),
                author.get("name"),
                author.get("handle"),
                author.get("platform_id"),
            )
            if score <= 0:
                continue
            author["score"] = round(score, 3)
            scored.append(author)
        scored.sort(
            key=lambda item: (
                -float(item.get("score") or 0),
                -int(item.get("image_count") or 0),
                str(item.get("name") or item.get("handle") or item["key"]).casefold(),
            )
        )
        return scored[: max(1, min(int(limit), 200))]

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
            authors = connection.execute("SELECT COUNT(*) FROM author").fetchone()[0]
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
        record["candidates"] = _loads(record.pop("candidates_json", None), [])
        record["manual_fields"] = _loads(record.pop("manual_fields", None), [])
        return record

    def _row_to_author(self, row: sqlite3.Row) -> dict[str, Any]:
        author = dict(row)
        author["social"] = _loads(author.pop("social_json", None), {})
        return author


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


def manual_author_key(name: str, handle: str | None = None) -> str:
    normalized_handle = _normalized_handle(handle)
    if normalized_handle:
        return f"twitter:{normalized_handle}"
    digest = hashlib.sha1(str(name or "").strip().casefold().encode("utf-8")).hexdigest()
    return f"manual:{digest[:16]}"

