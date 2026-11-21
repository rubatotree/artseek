"""SQLite-backed provenance repository on the unified v2 database."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ...core.query_plan import normalize_tag
from ...provenance.parser import ParsedSource, SourceRef, XiaohongshuName
from ...provenance.store import ProvenanceStore, _loads
from ..connection import Database
from .task_state import sync_task_state_from_provenance_locked


class _NoCommitConn:
    """Proxy delegating to an active SQLite connection while suppressing early commits."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def commit(self) -> None:
        return

    def close(self) -> None:
        return

    def __getattr__(self, name: str) -> Any:
        return getattr(self._conn, name)


def _sync_point_links_and_tags(
    conn: sqlite3.Connection, point_ids: Sequence[str]
) -> None:
    """Keep image_source.image_id, artist_id, and indexed source_tags in sync."""
    unique = [pid for pid in dict.fromkeys(point_ids) if pid]
    if not unique:
        return
    for start in range(0, len(unique), 400):
        batch = unique[start : start + 400]
        marks = ",".join("?" for _ in batch)
        rows = conn.execute(
            f"""
            SELECT s.point_id, s.image_id, s.author_key, s.artist_id,
                   s.tags_json, s.manual_tags_json, s.auto_tags_json,
                   s.auto_tag_scores_json, i.id AS live_image_id
            FROM image_source AS s
            LEFT JOIN images AS i ON i.point_id = s.point_id AND i.deleted = 0
            WHERE s.point_id IN ({marks})
            """,
            tuple(batch),
        ).fetchall()

        conn.execute(f"DELETE FROM source_tags WHERE point_id IN ({marks})", tuple(batch))

        tag_rows: list[tuple[str, int | None, str, str, str, float | None]] = []
        for r in rows:
            pid = str(r["point_id"])
            live_img_id = int(r["live_image_id"]) if r["live_image_id"] is not None else None
            author_key = str(r["author_key"] or "").strip()
            artist_id = int(author_key) if author_key.isdigit() else None
            if r["image_id"] != live_img_id or r["artist_id"] != artist_id:
                conn.execute(
                    "UPDATE image_source SET image_id = ?, artist_id = ? WHERE point_id = ?",
                    (live_img_id, artist_id, pid),
                )

            scores = _loads(r["auto_tag_scores_json"], {})
            if not isinstance(scores, dict):
                scores = {}
            for origin, raw_json in (
                ("source", r["tags_json"]),
                ("manual", r["manual_tags_json"]),
                ("auto", r["auto_tags_json"]),
            ):
                seen_norm: set[str] = set()
                for item in _loads(raw_json, []):
                    tag = str(item or "").strip()
                    if not tag:
                        continue
                    norm = tag.casefold()
                    if norm in seen_norm:
                        continue
                    seen_norm.add(norm)
                    score = float(scores[tag]) if origin == "auto" and tag in scores else None
                    tag_rows.append((pid, live_img_id, tag, norm, origin, score))

        if tag_rows:
            conn.executemany(
                """
                INSERT OR REPLACE INTO source_tags(
                    point_id, image_id, tag, tag_norm, origin, score
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                tag_rows,
            )
        sync_task_state_from_provenance_locked(conn, batch)


class ProvenanceRepo(ProvenanceStore):
    """Unified-DB provenance repository preserving v1 ProvenanceStore semantics."""

    def __init__(self, db: Database) -> None:
        self.db = db
        self._schema_lock = db._write_lock
        self._schema_ready = True
        self._tx_local = threading.local()
        # The store's __init__ never runs here, so seed its caches explicitly.
        self._ensure_artist_index_cache()

    @property
    def path(self) -> Path:
        return Path(self.db.path)

    @contextmanager
    def _read(self) -> Iterator[Any]:
        active = getattr(self._tx_local, "conn", None)
        if active is not None:
            yield active
            return
        yield self.db.read_conn()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        with self.db.write_tx() as conn:
            prev = getattr(self._tx_local, "conn", None)
            self._tx_local.conn = _NoCommitConn(conn)
            try:
                yield conn
            finally:
                self._tx_local.conn = prev

    # ------------------------------------------------------------------ writes wrapped in write_tx
    def upsert_scan(
        self,
        point_id: str,
        payload: dict[str, Any],
        parsed: ParsedSource | None,
        *,
        path_key: str | None = None,
        origin: str = "filename",
    ) -> dict[str, Any] | None:
        with self._tx() as conn:
            res = super().upsert_scan(
                point_id, payload, parsed, path_key=path_key, origin=origin
            )
            _sync_point_links_and_tags(conn, [point_id])
        return self.get(point_id) if res is not None else None

    def upsert_unresolved(
        self,
        point_id: str,
        payload: dict[str, Any],
        *,
        path_key: str | None = None,
        hint: XiaohongshuName | None = None,
        author_key: str | None = None,
    ) -> dict[str, Any] | None:
        with self._tx() as conn:
            res = super().upsert_unresolved(
                point_id, payload, path_key=path_key, hint=hint, author_key=author_key
            )
            _sync_point_links_and_tags(conn, [point_id])
        return self.get(point_id) if res is not None else None

    def reset_for_replacement(self, point_id: str) -> None:
        with self._tx() as conn:
            super().reset_for_replacement(point_id)
            _sync_point_links_and_tags(conn, [point_id])

    def set_status(self, point_id: str, status: str, error: str | None = None) -> None:
        with self._tx():
            super().set_status(point_id, status, error)

    def set_fetch_result(
        self,
        point_id: str,
        result: dict[str, Any],
        *,
        author_key: str | None = None,
    ) -> None:
        with self._tx() as conn:
            super().set_fetch_result(point_id, result, author_key=author_key)
            _sync_point_links_and_tags(conn, [point_id])

    def set_manual(
        self,
        point_id: str,
        fields: dict[str, Any],
        *,
        ref: SourceRef | None = None,
    ) -> dict[str, Any] | None:
        with self._tx() as conn:
            res = super().set_manual(point_id, fields, ref=ref)
            _sync_point_links_and_tags(conn, [point_id])
        return self.get(point_id) if res is not None else None

    def clear_manual(self, point_id: str) -> dict[str, Any] | None:
        with self._tx() as conn:
            res = super().clear_manual(point_id)
            _sync_point_links_and_tags(conn, [point_id])
        return self.get(point_id) if res is not None else None

    def stale_points(self, live_hashes: dict[str, str | None]) -> list[str]:
        with self._tx():
            return super().stale_points(live_hashes)

    def set_auto_tags_many(
        self,
        entries: Sequence[tuple[str, Sequence[str], dict[str, float] | None]],
    ) -> None:
        with self._tx() as conn:
            super().set_auto_tags_many(entries)
            _sync_point_links_and_tags(conn, [pid for pid, _, _ in entries if pid])

    def set_url_ignored(self, point_id: str, ignored: bool = True) -> dict[str, Any] | None:
        """Mark or unmark a point_id as local-only (no remote source URL needed)."""
        with self._tx() as conn:
            row = conn.execute(
                "SELECT manual_fields FROM image_source WHERE point_id = ?",
                (point_id,),
            ).fetchone()
            if row is None:
                return None
            manual = set(_loads(row["manual_fields"], []))
            if ignored:
                manual.add("url_ignored")
            else:
                manual.discard("url_ignored")
            conn.execute(
                "UPDATE image_source SET manual_fields = ?, updated_at = ? WHERE point_id = ?",
                (json.dumps(sorted(manual), ensure_ascii=False), int(time.time()), point_id),
            )
            sync_task_state_from_provenance_locked(conn, [point_id])
        return self.get(point_id)

    @staticmethod
    def _sync_artist_id_locked(conn: sqlite3.Connection, artist_key: str) -> None:
        if artist_key.isdigit():
            conn.execute(
                "UPDATE image_source SET artist_id = ? "
                "WHERE author_key = ? AND (artist_id IS NULL OR artist_id <> ?)",
                (int(artist_key), artist_key, int(artist_key)),
            )

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
        with self._tx() as conn:
            res = super().ensure_author(
                platform,
                platform_id,
                name=name,
                handle=handle,
                profile_url=profile_url,
                avatar_url=avatar_url,
                social=social,
                profile_fetched=profile_fetched,
            )
            self._sync_artist_id_locked(conn, res)
            return res

    def ensure_xhs_author(self, name: str) -> str:
        with self._tx() as conn:
            res = super().ensure_xhs_author(name)
            self._sync_artist_id_locked(conn, res)
            return res

    def ensure_linked_author(
        self,
        *,
        name: str | None = None,
        pixiv_id: str | None = None,
        x_handle: str | None = None,
    ) -> str:
        with self._tx() as conn:
            res = super().ensure_linked_author(
                name=name, pixiv_id=pixiv_id, x_handle=x_handle
            )
            self._sync_artist_id_locked(conn, res)
            return res

    def attach_identity(
        self,
        artist_key: str,
        platform: str,
        value: str,
        *,
        name: str | None = None,
    ) -> str:
        with self._tx() as conn:
            res = super().attach_identity(artist_key, platform, value, name=name)
            self._sync_artist_id_locked(conn, res)
            return res

    def add_link(self, artist_key: str, url: str, *, label: str = "") -> str:
        with self._tx():
            return super().add_link(artist_key, url, label=label)

    def add_alias(self, artist_key: str, alias: str) -> str:
        with self._tx():
            return super().add_alias(artist_key, alias)

    def set_artist_cover(self, artist_key: str, point_id: str) -> str:
        with self._tx():
            return super().set_artist_cover(artist_key, point_id)

    def set_author_name(self, artist_key: str, name: str) -> str:
        with self._tx():
            return super().set_author_name(artist_key, name)

    def remove_identity(self, artist_key: str, platform: str, value: str) -> str:
        with self._tx():
            return super().remove_identity(artist_key, platform, value)

    def set_member_preferences(
        self, artist_key: str, members: Sequence[dict[str, Any]]
    ) -> str:
        with self._tx():
            return super().set_member_preferences(artist_key, members)

    def link_authors(
        self,
        alias_key: str,
        canonical_key: str,
        *,
        evidence: str,
        confidence: float = 0.8,
    ) -> str:
        with self._tx() as conn:
            res = super().link_authors(
                alias_key, canonical_key, evidence=evidence, confidence=confidence
            )
            if res.isdigit():
                conn.execute(
                    "UPDATE image_source SET artist_id = ? WHERE author_key = ?",
                    (int(res), res),
                )
            return res

    def merge_authors(self, source_key: str, target_key: str) -> str:
        with self._tx() as conn:
            res = super().merge_authors(source_key, target_key)
            if res.isdigit():
                conn.execute(
                    "UPDATE image_source SET artist_id = ? WHERE author_key = ?",
                    (int(res), res),
                )
            return res

    def assign_author_if_missing(self, point_id: str, author_key: str) -> bool:
        with self._tx() as conn:
            ok = super().assign_author_if_missing(point_id, author_key)
            if ok:
                _sync_point_links_and_tags(conn, [point_id])
            return ok

    def note_request(self, domain: str) -> None:
        with self._tx():
            super().note_request(domain)

    def set_blocked(self, domain: str, seconds: int, error: str | None = None) -> int:
        with self._tx():
            return super().set_blocked(domain, seconds, error)

    def set_long_remaining(self, domain: str, remaining: int | None) -> None:
        with self._tx():
            super().set_long_remaining(domain, remaining)

    # ------------------------------------------------------------------ indexed source_tags queries
    def list_source_tags(
        self, query: str = "", *, limit: int = 60, offset: int = 0
    ) -> list[dict[str, Any]]:
        clean = str(query or "").strip().casefold()
        params: list[Any] = []
        where = "1=1"
        if clean:
            where = "tag_norm LIKE ?"
            params.append(f"%{clean}%")
        params.extend([max(1, min(int(limit), 500)), max(0, int(offset))])
        with self._read() as conn:
            rows = conn.execute(
                f"SELECT tag, COUNT(DISTINCT point_id) AS count "
                f"FROM source_tags WHERE {where} "
                "GROUP BY tag "
                "ORDER BY count DESC, tag COLLATE NOCASE LIMIT ? OFFSET ?",
                tuple(params),
            ).fetchall()
        return [{"tag": str(r["tag"]), "count": int(r["count"])} for r in rows]

    def source_tag_counts_for_points(
        self,
        point_ids: Sequence[str],
        query: str = "",
        *,
        limit: int = 60,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        clean = str(query or "").strip().casefold()
        params: list[Any] = []
        where = "1=1"
        if clean:
            where = "st.tag_norm LIKE ?"
            params.append(f"%{clean}%")
        params.extend([max(1, min(int(limit), 500)), max(0, int(offset))])
        with self._read() as conn:
            self._scope_temp_table(conn, point_ids)
            rows = conn.execute(
                "SELECT st.tag AS tag, COUNT(DISTINCT st.point_id) AS count "
                "FROM source_tags AS st "
                "JOIN _scope_points AS sp ON sp.point_id = st.point_id "
                f"WHERE {where} "
                "GROUP BY st.tag "
                "ORDER BY count DESC, st.tag COLLATE NOCASE LIMIT ? OFFSET ?",
                tuple(params),
            ).fetchall()
        return [{"tag": str(r["tag"]), "count": int(r["count"])} for r in rows]

    def point_ids_for_any_tags(self, tags: Sequence[str]) -> list[str]:
        clean = sorted({str(tag).strip().casefold() for tag in tags if str(tag).strip()})
        if not clean:
            return []
        marks = ",".join("?" for _ in clean)
        with self._read() as conn:
            rows = conn.execute(
                "SELECT DISTINCT s.point_id "
                "FROM source_tags AS st "
                "JOIN image_source AS s ON s.point_id = st.point_id "
                f"WHERE st.tag_norm IN ({marks}) "
                "ORDER BY s.mtime_ns IS NULL, s.mtime_ns DESC, s.updated_at DESC, s.point_id",
                tuple(clean),
            ).fetchall()
        return [str(r["point_id"]) for r in rows]

    def point_ids_for_tags(self, tags: Sequence[str]) -> list[str]:
        clean = sorted({str(tag).strip().casefold() for tag in tags if str(tag).strip()})
        if not clean:
            return []
        marks = ",".join("?" for _ in clean)
        with self._read() as conn:
            rows = conn.execute(
                "SELECT s.point_id "
                "FROM source_tags AS st "
                "JOIN image_source AS s ON s.point_id = st.point_id "
                f"WHERE st.tag_norm IN ({marks}) "
                "GROUP BY s.point_id "
                "HAVING COUNT(DISTINCT st.tag_norm) = ? "
                "ORDER BY s.mtime_ns IS NULL, s.mtime_ns DESC, s.updated_at DESC, s.point_id",
                (*clean, len(clean)),
            ).fetchall()
        return [str(r["point_id"]) for r in rows]

    def point_ids_for_tag_text(self, terms: Sequence[str]) -> list[str]:
        clean = [str(term).strip().casefold() for term in terms if str(term).strip()]
        if not clean:
            return []
        clauses = " AND ".join(
            "EXISTS (SELECT 1 FROM source_tags AS st "
            "WHERE st.point_id = image_source.point_id AND st.tag_norm LIKE ?)"
            for _ in clean
        )
        with self._read() as conn:
            rows = conn.execute(
                f"SELECT point_id FROM image_source WHERE {clauses} "
                "ORDER BY mtime_ns IS NULL, mtime_ns DESC, updated_at DESC, point_id",
                [f"%{term}%" for term in clean],
            ).fetchall()
        return [str(r["point_id"]) for r in rows]

    def point_ids_for_metadata_search(
        self, term_groups: Sequence[str | Sequence[str] | tuple[str, Sequence[str]]]
    ) -> list[str]:
        groups: list[tuple[str, list[str]]] = []
        for group in term_groups:
            if isinstance(group, str):
                original = group
                aliases = [group]
            elif (
                isinstance(group, tuple)
                and len(group) == 2
                and not isinstance(group[1], str)
            ):
                original = str(group[0])
                aliases = [str(value) for value in group[1]]
            else:
                aliases = [str(value) for value in group]
                original = aliases[0] if aliases else ""
            values = [
                value.strip().lstrip("#")
                for value in (original, *aliases)
                if value.strip().lstrip("#")
            ]
            values = list(dict.fromkeys(values))
            if values:
                groups.append((original.strip().lstrip("#"), values))
        if not groups:
            return []
        group_sqls: list[str] = []
        params: list[str] = []
        for original, values in groups:
            tag_names = sorted(
                {normalize_tag(value) for value in values if normalize_tag(value)}
            )
            tag_norms = sorted(
                {
                    value.casefold()
                    for value in (*values, *tag_names)
                    if value.casefold()
                }
            )
            selects: list[str] = []
            if tag_names:
                marks = ",".join("?" for _ in tag_names)
                selects.append(
                    "SELECT it.image_id AS id FROM image_tags AS it "
                    "JOIN tags AS t ON t.id = it.tag_id "
                    f"WHERE t.name IN ({marks})"
                )
                params.extend(tag_names)
            if tag_norms:
                marks = ",".join("?" for _ in tag_norms)
                selects.append(
                    "SELECT st.image_id AS id FROM source_tags AS st "
                    f"WHERE st.image_id IS NOT NULL AND st.tag_norm IN ({marks})"
                )
                params.extend(tag_norms)
                selects.append(
                    "SELECT i2.id AS id FROM source_tags AS st "
                    "JOIN images AS i2 ON i2.point_id = st.point_id "
                    f"WHERE st.tag_norm IN ({marks})"
                )
                params.extend(tag_norms)
            like = f"%{original.casefold()}%"
            selects.append(
                "SELECT i2.id AS id FROM images AS i2 "
                "WHERE LOWER(COALESCE(i2.name, '')) LIKE ?"
            )
            params.append(like)
            selects.append(
                "SELECT s.image_id AS id FROM image_source AS s "
                "WHERE s.image_id IS NOT NULL "
                "AND LOWER(COALESCE(s.title, '')) LIKE ?"
            )
            params.append(like)
            group_sqls.append("SELECT id FROM (" + " UNION ".join(selects) + ")")
        combined = " INTERSECT ".join(group_sqls)
        with self._read() as conn:
            rows = conn.execute(
                "SELECT i.point_id FROM images AS i "
                f"JOIN ({combined}) AS m ON m.id = i.id "
                "WHERE i.deleted = 0 "
                "ORDER BY i.mtime_ns IS NULL, i.mtime_ns DESC, i.id DESC",
                params,
            ).fetchall()
        return [str(r["point_id"]) for r in rows]

    def resolve_author_keys_for_query(self, raw_tokens: Sequence[str]) -> list[str]:
        """Resolve canonical author keys (preserving 'none') using _canonical_locked."""
        out: list[str] = []
        with self._read() as conn:
            for token in raw_tokens:
                if self._is_unassigned_key(token):
                    out.append("none")
                else:
                    canon = self._canonical_locked(conn, str(token))
                    out.append(canon)
        return list(dict.fromkeys(out))
