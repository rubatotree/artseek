"""Per-image completion state across indexing, provenance, tagging, vision, and CCIP."""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Sequence
from typing import Any

from ..connection import Database

_SQL_CHUNK = 400


def _parse_tag_models(raw: str | None) -> dict[str, int]:
    if not raw:
        return {}
    try:
        loaded = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if not isinstance(loaded, dict):
        return {}
    out: dict[str, int] = {}
    for key, val in loaded.items():
        try:
            out[str(key)] = int(val)
        except (TypeError, ValueError):
            continue
    return out


def sync_task_state_from_index_locked(
    conn: sqlite3.Connection,
    entries: Sequence[tuple[str, int, str, bool]],
    *,
    tag_model_versions: dict[str, int] | None = None,
) -> None:
    """Update `image_task_state` after indexing."""
    if not entries:
        return
    now = int(time.time())
    pids = [pid for pid, _, _, _ in entries if pid]
    existing: dict[str, tuple[str, str]] = {}
    for start in range(0, len(pids), _SQL_CHUNK):
        chunk = pids[start : start + _SQL_CHUNK]
        marks = ",".join("?" for _ in chunk)
        for r in conn.execute(
            "SELECT point_id, content_hash, tag_models_json "
            f"FROM image_task_state WHERE point_id IN ({marks})",
            tuple(chunk),
        ).fetchall():
            existing[str(r["point_id"])] = (
                str(r["content_hash"] or ""),
                str(r["tag_models_json"] or "{}"),
            )

    new_models = dict(tag_model_versions or {})
    rows_to_upsert: list[tuple[Any, ...]] = []
    for pid, img_id, chash, desc_done in entries:
        prev = existing.get(pid)
        if prev is None:
            models_json = json.dumps(new_models, sort_keys=True)
            rows_to_upsert.append(
                (pid, img_id, chash, 1, 0, 0, models_json, int(desc_done), 0, 0, 0, now)
            )
            continue

        prev_hash, prev_models_raw = prev
        if prev_hash and chash and prev_hash != chash:
            # File bytes changed: reset derived visual/tag/ccip/caption completions.
            models_json = json.dumps(new_models, sort_keys=True)
            conn.execute(
                """
                UPDATE image_task_state
                SET image_id = ?, content_hash = ?, index_done = 1,
                    tag_models_json = ?, descriptor_done = ?,
                    vision_done = 0, ccip_done = 0, caption_done = 0,
                    updated_at = ?
                WHERE point_id = ?
                """,
                (img_id, chash, models_json, int(desc_done), now, pid),
            )
        else:
            merged = _parse_tag_models(prev_models_raw)
            merged.update(new_models)
            models_json = json.dumps(merged, sort_keys=True)
            conn.execute(
                """
                UPDATE image_task_state
                SET image_id = ?, content_hash = COALESCE(NULLIF(?, ''), content_hash),
                    index_done = 1, tag_models_json = ?,
                    descriptor_done = MAX(descriptor_done, ?),
                    updated_at = ?
                WHERE point_id = ?
                """,
                (img_id, chash, models_json, int(desc_done), now, pid),
            )

    if rows_to_upsert:
        conn.executemany(
            """
            INSERT INTO image_task_state(
                point_id, image_id, content_hash, index_done, url_done, author_done,
                tag_models_json, descriptor_done, vision_done, ccip_done, caption_done, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(point_id) DO UPDATE SET
                image_id = excluded.image_id,
                content_hash = excluded.content_hash,
                index_done = 1,
                tag_models_json = excluded.tag_models_json,
                descriptor_done = MAX(image_task_state.descriptor_done, excluded.descriptor_done),
                updated_at = excluded.updated_at
            """,
            rows_to_upsert,
        )


def sync_task_state_from_provenance_locked(
    conn: sqlite3.Connection, point_ids: Sequence[str]
) -> None:
    """Sync `url_done` and `author_done` from `image_source` for the given `point_ids`."""
    unique = [pid for pid in dict.fromkeys(point_ids) if pid]
    if not unique:
        return
    now = int(time.time())
    for start in range(0, len(unique), _SQL_CHUNK):
        batch = unique[start : start + _SQL_CHUNK]
        marks = ",".join("?" for _ in batch)
        rows = conn.execute(
            f"""
            SELECT s.point_id, s.image_id, s.content_hash, s.status, s.source_origin,
                   s.author_key, s.manual_fields, i.id AS live_image_id,
                   COALESCE(i.content_hash, s.content_hash, '') AS effective_hash
            FROM image_source AS s
            LEFT JOIN images AS i ON i.point_id = s.point_id AND i.deleted = 0
            WHERE s.point_id IN ({marks})
            """,
            tuple(batch),
        ).fetchall()
        updates: list[tuple[Any, ...]] = []
        for r in rows:
            pid = str(r["point_id"])
            img_id = (
                int(r["live_image_id"])
                if r["live_image_id"] is not None
                else (int(r["image_id"]) if r["image_id"] is not None else None)
            )
            manual_raw = str(r["manual_fields"] or "")
            url_done = int(
                r["status"] == "ok"
                or r["source_origin"] == "manual"
                or "url_ignored" in manual_raw
            )
            author_done = int(bool(str(r["author_key"] or "").strip()))
            chash = str(r["effective_hash"] or "")
            updates.append((pid, img_id, chash, url_done, author_done, now))
        if updates:
            conn.executemany(
                """
                INSERT INTO image_task_state(
                    point_id, image_id, content_hash, index_done,
                    url_done, author_done, updated_at
                ) VALUES (?, ?, ?, CASE WHEN ? IS NOT NULL THEN 1 ELSE 0 END, ?, ?, ?)
                ON CONFLICT(point_id) DO UPDATE SET
                    image_id = COALESCE(excluded.image_id, image_task_state.image_id),
                    content_hash = COALESCE(
                        NULLIF(excluded.content_hash, ''), image_task_state.content_hash
                    ),
                    url_done = excluded.url_done,
                    author_done = excluded.author_done,
                    updated_at = excluded.updated_at
                """,
                [(u[0], u[1], u[2], u[1], u[3], u[4], u[5]) for u in updates],
            )


class TaskStateRepo:
    """Queries and updates per-image task completion metadata."""

    def __init__(self, db: Database) -> None:
        self.db = db

    def get(self, point_id: str) -> dict[str, Any] | None:
        row = self.db.read_conn().execute(
            "SELECT * FROM image_task_state WHERE point_id = ?", (point_id,)
        ).fetchone()
        if row is None:
            return None
        return {
            "point_id": str(row["point_id"]),
            "image_id": int(row["image_id"]) if row["image_id"] is not None else None,
            "content_hash": str(row["content_hash"] or ""),
            "index_done": bool(row["index_done"]),
            "url_done": bool(row["url_done"]),
            "author_done": bool(row["author_done"]),
            "tag_models": _parse_tag_models(row["tag_models_json"]),
            "descriptor_done": bool(row["descriptor_done"]),
            "vision_done": bool(row["vision_done"]),
            "ccip_done": bool(row["ccip_done"]),
            "caption_done": bool(row["caption_done"]),
            "updated_at": int(row["updated_at"]),
        }

    def mark_tag_models_done(
        self,
        entries: Sequence[tuple[str, int, str]],
        model_versions: dict[str, int],
        *,
        descriptor_done: bool = True,
    ) -> None:
        if not entries or not model_versions:
            return
        now = int(time.time())
        with self.db.write_tx() as conn:
            pids = [pid for pid, _, _ in entries]
            existing: dict[str, str] = {}
            for start in range(0, len(pids), _SQL_CHUNK):
                chunk = pids[start : start + _SQL_CHUNK]
                marks = ",".join("?" for _ in chunk)
                for r in conn.execute(
                    "SELECT point_id, tag_models_json "
                    f"FROM image_task_state WHERE point_id IN ({marks})",
                    tuple(chunk),
                ).fetchall():
                    existing[str(r["point_id"])] = str(r["tag_models_json"] or "{}")
            rows: list[tuple[Any, ...]] = []
            for pid, img_id, chash in entries:
                merged = _parse_tag_models(existing.get(pid))
                merged.update(model_versions)
                rows.append(
                    (
                        pid,
                        img_id,
                        chash,
                        1,
                        json.dumps(merged, sort_keys=True),
                        int(descriptor_done),
                        now,
                    )
                )
            conn.executemany(
                """
                INSERT INTO image_task_state(
                    point_id, image_id, content_hash, index_done,
                    tag_models_json, descriptor_done, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(point_id) DO UPDATE SET
                    image_id = excluded.image_id,
                    content_hash = COALESCE(
                        NULLIF(excluded.content_hash, ''), image_task_state.content_hash
                    ),
                    tag_models_json = excluded.tag_models_json,
                    descriptor_done = MAX(
                        image_task_state.descriptor_done, excluded.descriptor_done
                    ),
                    updated_at = excluded.updated_at
                """,
                rows,
            )

    def mark_vision_done(
        self, entries: Sequence[tuple[str, int, str]], *, version: int = 1
    ) -> None:
        if not entries:
            return
        now = int(time.time())
        with self.db.write_tx() as conn:
            conn.executemany(
                """
                INSERT INTO image_task_state(
                    point_id, image_id, content_hash, index_done, vision_done, updated_at
                ) VALUES (?, ?, ?, 1, ?, ?)
                ON CONFLICT(point_id) DO UPDATE SET
                    image_id = excluded.image_id,
                    vision_done = excluded.vision_done,
                    updated_at = excluded.updated_at
                """,
                [(pid, img_id, chash, int(version), now) for pid, img_id, chash in entries],
            )

    def mark_ccip_done(self, entries: Sequence[tuple[str, int, str]]) -> None:
        if not entries:
            return
        now = int(time.time())
        with self.db.write_tx() as conn:
            conn.executemany(
                """
                INSERT INTO image_task_state(
                    point_id, image_id, content_hash, index_done, ccip_done, updated_at
                ) VALUES (?, ?, ?, 1, 1, ?)
                ON CONFLICT(point_id) DO UPDATE SET
                    image_id = excluded.image_id,
                    ccip_done = 1,
                    updated_at = excluded.updated_at
                """,
                [(pid, img_id, chash, now) for pid, img_id, chash in entries],
            )

    def reset_ccip(self, *, scope_row_ids: Sequence[int] | None = None) -> int:
        """Clear `ccip_done` so the next CCIP job recomputes the affected rows.

        Returns the number of task-state rows actually flipped back to pending.
        """
        now = int(time.time())
        with self.db.write_tx() as conn:
            if scope_row_ids is None:
                cur = conn.execute(
                    """
                    UPDATE image_task_state
                    SET ccip_done = 0, updated_at = ?
                    WHERE ccip_done <> 0
                    """,
                    (now,),
                )
                return int(cur.rowcount or 0)
            ids = [int(r) for r in dict.fromkeys(scope_row_ids)]
            if not ids:
                return 0
            total = 0
            for chunk in range(0, len(ids), _SQL_CHUNK):
                batch = ids[chunk : chunk + _SQL_CHUNK]
                cur = conn.execute(
                    f"""
                    UPDATE image_task_state
                    SET ccip_done = 0, updated_at = ?
                    WHERE ccip_done <> 0
                      AND image_id IN ({",".join("?" for _ in batch)})
                    """,
                    (now, *batch),
                )
                total += int(cur.rowcount or 0)
            return total

    def mark_caption_done(self, image_id: int) -> None:
        now = int(time.time())
        with self.db.write_tx() as conn:
            conn.execute(
                """
                UPDATE image_task_state
                SET caption_done = 1, descriptor_done = 1, updated_at = ?
                WHERE image_id = ?
                """,
                (now, int(image_id)),
            )

    def pending_for_tagger(
        self,
        required_models: dict[str, int],
        *,
        scope_row_ids: Sequence[int] | None = None,
        skip_completed: bool = True,
        limit: int | None = None,
    ) -> list[tuple[int, str, str, str]]:
        """Return `[(image_id, point_id, path, content_hash)]` needing tagging."""
        conn = self.db.read_conn()
        sql = """
            SELECT i.id, i.point_id, i.path, i.content_hash, ts.tag_models_json
            FROM images AS i
            LEFT JOIN image_task_state AS ts ON ts.point_id = i.point_id
            WHERE i.deleted = 0
        """
        params: list[Any] = []
        if scope_row_ids is not None:
            ids = [int(r) for r in dict.fromkeys(scope_row_ids)]
            if not ids:
                return []
            sql += " AND i.id IN (SELECT value FROM json_each(?))"
            params.append(json.dumps(ids))
        sql += " ORDER BY i.mtime_ns DESC, i.id DESC"
        rows = conn.execute(sql, tuple(params)).fetchall()

        pending: list[tuple[int, str, str, str]] = []
        for r in rows:
            if skip_completed:
                done_map = _parse_tag_models(r["tag_models_json"])
                if all(done_map.get(k, 0) >= ver for k, ver in required_models.items()):
                    continue
            pending.append(
                (int(r["id"]), str(r["point_id"]), str(r["path"]), str(r["content_hash"] or ""))
            )
            if limit is not None and len(pending) >= max(1, int(limit)):
                break
        return pending

    def pending_for_vision(
        self,
        *,
        min_version: int = 1,
        scope_row_ids: Sequence[int] | None = None,
        skip_completed: bool = True,
        limit: int | None = None,
    ) -> list[tuple[int, str, str, str]]:
        conn = self.db.read_conn()
        sql = """
            SELECT i.id, i.point_id, i.path, i.content_hash,
                   COALESCE(ts.vision_done, 0) AS vision_done,
                   COALESCE(i.vision_meta_version, 0) AS vm_ver
            FROM images AS i
            LEFT JOIN image_task_state AS ts ON ts.point_id = i.point_id
            WHERE i.deleted = 0
        """
        params: list[Any] = []
        if scope_row_ids is not None:
            ids = [int(r) for r in dict.fromkeys(scope_row_ids)]
            if not ids:
                return []
            sql += " AND i.id IN (SELECT value FROM json_each(?))"
            params.append(json.dumps(ids))
        if skip_completed:
            sql += (
                " AND (COALESCE(ts.vision_done, 0) < ? "
                "OR COALESCE(i.vision_meta_version, 0) < ?)"
            )
            params.extend([int(min_version), int(min_version)])
        sql += " ORDER BY i.mtime_ns DESC, i.id DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(max(1, int(limit)))
        rows = conn.execute(sql, tuple(params)).fetchall()
        return [
            (int(r["id"]), str(r["point_id"]), str(r["path"]), str(r["content_hash"] or ""))
            for r in rows
        ]

    def pending_for_ccip(
        self,
        *,
        scope_row_ids: Sequence[int] | None = None,
        skip_completed: bool = True,
        limit: int | None = None,
    ) -> list[tuple[int, str, str, str, str | None]]:
        conn = self.db.read_conn()
        sql = """
            SELECT i.id, i.point_id, i.path, i.content_hash, i.character_crop_json
            FROM images AS i
            LEFT JOIN image_task_state AS ts ON ts.point_id = i.point_id
            WHERE i.deleted = 0
        """
        params: list[Any] = []
        if scope_row_ids is not None:
            ids = [int(r) for r in dict.fromkeys(scope_row_ids)]
            if not ids:
                return []
            sql += " AND i.id IN (SELECT value FROM json_each(?))"
            params.append(json.dumps(ids))
        if skip_completed:
            sql += " AND COALESCE(ts.ccip_done, 0) = 0"
        sql += " ORDER BY i.mtime_ns DESC, i.id DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(max(1, int(limit)))
        rows = conn.execute(sql, tuple(params)).fetchall()
        return [
            (
                int(r["id"]),
                str(r["point_id"]),
                str(r["path"]),
                str(r["content_hash"] or ""),
                str(r["character_crop_json"]) if r["character_crop_json"] else None,
            )
            for r in rows
        ]

    def summary_counts(
        self,
        scope_row_ids: Sequence[int] | None = None,
        *,
        min_vision_version: int = 1,
    ) -> dict[str, int]:
        conn = self.db.read_conn()
        sql = """
            SELECT i.id,
                   COALESCE(ts.index_done, 1) AS index_done,
                   COALESCE(ts.url_done, 0) AS url_done,
                   COALESCE(ts.author_done, 0) AS author_done,
                   COALESCE(ts.tag_models_json, '{}') AS tag_models_json,
                   COALESCE(ts.descriptor_done, 0) AS descriptor_done,
                   COALESCE(ts.vision_done, 0) AS vision_done,
                   COALESCE(i.vision_meta_version, 0) AS vision_meta_version,
                   COALESCE(ts.ccip_done, 0) AS ccip_done,
                   COALESCE(ts.caption_done, 0) AS caption_done
            FROM images AS i
            LEFT JOIN image_task_state AS ts ON ts.point_id = i.point_id
            WHERE i.deleted = 0
        """
        params: list[Any] = []
        if scope_row_ids is not None:
            ids = [int(r) for r in dict.fromkeys(scope_row_ids)]
            if not ids:
                return {
                    "total": 0,
                    "index_done": 0,
                    "url_done": 0,
                    "author_done": 0,
                    "default_tag_done": 0,
                    "camie_v2_done": 0,
                    "pixai_v09_done": 0,
                    "pixai_v10_done": 0,
                    "vision_done": 0,
                    "ccip_done": 0,
                    "caption_done": 0,
                }
            sql += " AND i.id IN (SELECT value FROM json_each(?))"
            params.append(json.dumps(ids))

        rows = conn.execute(sql, tuple(params)).fetchall()
        counts = {
            "total": len(rows),
            "index_done": 0,
            "url_done": 0,
            "author_done": 0,
            "default_tag_done": 0,
            "camie_v2_done": 0,
            "pixai_v09_done": 0,
            "pixai_v10_done": 0,
            "vision_done": 0,
            "ccip_done": 0,
            "caption_done": 0,
        }
        req_vision = max(1, int(min_vision_version))
        for r in rows:
            counts["index_done"] += int(bool(r["index_done"]))
            counts["url_done"] += int(bool(r["url_done"]))
            counts["author_done"] += int(bool(r["author_done"]))
            counts["vision_done"] += int(
                int(r["vision_done"] or 0) >= req_vision
                and int(r["vision_meta_version"] or 0) >= req_vision
            )
            counts["ccip_done"] += int(bool(r["ccip_done"]))
            counts["caption_done"] += int(bool(r["caption_done"]))
            models = _parse_tag_models(r["tag_models_json"])
            has_camie = models.get("camie_v2", 0) >= 1
            has_pixai09 = models.get("pixai_v09", 0) >= 1
            has_pixai10 = models.get("pixai_v10", 0) >= 1
            counts["camie_v2_done"] += int(has_camie)
            counts["pixai_v09_done"] += int(has_pixai09)
            counts["default_tag_done"] += int(has_camie and has_pixai09)
            counts["pixai_v10_done"] += int(has_pixai10)
        return counts
