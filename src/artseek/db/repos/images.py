"""Repository for roots, folders, images, tags, scope preferences, and metadata."""

from __future__ import annotations

import json
import os
import sqlite3
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ...image_io import canonical_path
from ...query import normalize_tag
from ..connection import Database

_SQL_CHUNK = 400


@dataclass(slots=True)
class ImageUpsertRecord:
    point_id: str
    root_key: str
    root_path: str
    folder_key: str
    folder_path: str
    folder_mtime_ns: int
    path: str
    path_key: str
    name: str
    ext: str
    mtime_ns: int
    size_bytes: int
    width: int
    height: int
    content_hash: str
    phash: str
    descriptor_text: str = ""
    caption: dict[str, Any] | None = None
    tags: list[str] = field(default_factory=list)
    tag_scores: dict[str, float] = field(default_factory=dict)
    indexed_at: int | None = None


def _chunks(values: Sequence[Any], size: int = _SQL_CHUNK) -> Sequence[Sequence[Any]]:
    return [values[i : i + size] for i in range(0, len(values), size)]


def _decode_caption(raw: str | None) -> Any:
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _encode_caption(caption: Any) -> str | None:
    if caption is None:
        return None
    if isinstance(caption, str):
        return caption
    return json.dumps(caption, ensure_ascii=False)


class ImageRepo:
    def __init__(self, db: Database) -> None:
        self.db = db

    # ------------------------------------------------------------------ meta
    def meta_get(self, key: str, default: str | None = None) -> str | None:
        row = self.db.read_conn().execute(
            "SELECT value FROM meta WHERE key = ?", (key,)
        ).fetchone()
        return str(row["value"]) if row is not None else default

    def meta_set(self, key: str, value: str) -> None:
        with self.db.write_tx() as conn:
            conn.execute(
                "INSERT INTO meta(key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, str(value)),
            )

    def meta_delete(self, key: str) -> None:
        with self.db.write_tx() as conn:
            conn.execute("DELETE FROM meta WHERE key = ?", (key,))

    def known_dimension(self) -> int | None:
        raw = self.meta_get("dimension")
        if raw is None:
            return None
        try:
            dim = int(raw)
        except ValueError:
            return None
        return dim if dim > 0 else None

    # ---------------------------------------------------------------- counts & IDs
    def count(self) -> int:
        row = self.db.read_conn().execute(
            "SELECT COUNT(*) AS cnt FROM images WHERE deleted = 0"
        ).fetchone()
        return int(row["cnt"]) if row else 0

    def active_row_ids(self) -> np.ndarray:
        rows = self.db.read_conn().execute(
            "SELECT id FROM images WHERE deleted = 0 ORDER BY id"
        ).fetchall()
        return np.array([int(r[0]) for r in rows], dtype=np.int64)

    def row_id_for_point(self, point_id: str) -> int | None:
        row = self.db.read_conn().execute(
            "SELECT id FROM images WHERE point_id = ? AND deleted = 0", (point_id,)
        ).fetchone()
        return int(row["id"]) if row is not None else None

    def row_ids_for_points(self, point_ids: Sequence[str]) -> dict[str, int]:
        unique = [pid for pid in dict.fromkeys(point_ids) if pid]
        if not unique:
            return {}
        conn = self.db.read_conn()
        found: dict[str, int] = {}
        for chunk in _chunks(unique):
            marks = ",".join("?" for _ in chunk)
            rows = conn.execute(
                f"SELECT point_id, id FROM images WHERE deleted = 0 AND point_id IN ({marks})",
                tuple(chunk),
            ).fetchall()
            for r in rows:
                found[str(r["point_id"])] = int(r["id"])
        return found

    def point_ids_for_rows(self, row_ids: Sequence[int]) -> dict[int, str]:
        unique = [int(r) for r in dict.fromkeys(row_ids)]
        if not unique:
            return {}
        conn = self.db.read_conn()
        found: dict[int, str] = {}
        for chunk in _chunks(unique):
            marks = ",".join("?" for _ in chunk)
            rows = conn.execute(
                f"SELECT id, point_id FROM images WHERE deleted = 0 AND id IN ({marks})",
                tuple(chunk),
            ).fetchall()
            for r in rows:
                found[int(r["id"])] = str(r["point_id"])
        return found

    # ---------------------------------------------------------------- payload hydration
    def _load_tags_for_rows(
        self, conn: sqlite3.Connection, row_ids: Sequence[int]
    ) -> dict[int, tuple[list[str], dict[str, float]]]:
        result: dict[int, tuple[list[str], dict[str, float]]] = {}
        for chunk in _chunks(list(dict.fromkeys(row_ids))):
            marks = ",".join("?" for _ in chunk)
            rows = conn.execute(
                f"""
                SELECT it.image_id, t.name, it.score
                FROM image_tags AS it
                JOIN tags AS t ON t.id = it.tag_id
                WHERE it.image_id IN ({marks})
                ORDER BY it.image_id, it.score DESC, t.name
                """,
                tuple(chunk),
            ).fetchall()
            for r in rows:
                img_id = int(r["image_id"])
                tag = str(r["name"])
                score = float(r["score"])
                tag_list, score_map = result.setdefault(img_id, ([], {}))
                tag_list.append(tag)
                score_map[tag] = score
        return result

    def _row_to_payload(
        self,
        row: sqlite3.Row,
        tags_entry: tuple[list[str], dict[str, float]] | None,
    ) -> dict[str, Any]:
        tags, scores = tags_entry if tags_entry is not None else ([], {})
        return {
            "row_id": int(row["id"]),
            "id": str(row["point_id"]),
            "point_id": str(row["point_id"]),
            "path": str(row["path"]),
            "path_key": str(row["path_key"]),
            "name": str(row["name"]),
            "extension": str(row["ext"]),
            "mtime_ns": int(row["mtime_ns"]),
            "size_bytes": int(row["size_bytes"]),
            "width": int(row["width"]),
            "height": int(row["height"]),
            "root": str(row["root_path"]),
            "root_key": str(row["root_key"]),
            "folder_key": str(row["folder_key"]),
            "content_hash": str(row["content_hash"]),
            "phash": str(row["phash"]),
            "tags": list(tags),
            "tag_scores": dict(scores),
            "caption": _decode_caption(row["caption_json"]),
            "descriptor_text": str(row["descriptor_text"] or ""),
        }

    def get_by_point_id(self, point_id: str) -> dict[str, Any] | None:
        conn = self.db.read_conn()
        row = conn.execute(
            """
            SELECT i.*, r.root_key, r.root_path, f.folder_key
            FROM images AS i
            JOIN roots AS r ON r.id = i.root_id
            JOIN folders AS f ON f.id = i.folder_id
            WHERE i.point_id = ? AND i.deleted = 0
            """,
            (point_id,),
        ).fetchone()
        if row is None:
            return None
        tags = self._load_tags_for_rows(conn, [int(row["id"])]).get(int(row["id"]))
        return self._row_to_payload(row, tags)

    def get_payloads_by_ids(
        self, row_ids: Sequence[int], *, include_tags: bool = True
    ) -> dict[int, dict[str, Any]]:
        unique = [int(r) for r in dict.fromkeys(row_ids)]
        if not unique:
            return {}
        conn = self.db.read_conn()
        tags_map = self._load_tags_for_rows(conn, unique) if include_tags else {}
        result: dict[int, dict[str, Any]] = {}
        for chunk in _chunks(unique):
            marks = ",".join("?" for _ in chunk)
            rows = conn.execute(
                f"""
                SELECT i.*, r.root_key, r.root_path, f.folder_key
                FROM images AS i
                JOIN roots AS r ON r.id = i.root_id
                JOIN folders AS f ON f.id = i.folder_id
                WHERE i.deleted = 0 AND i.id IN ({marks})
                """,
                tuple(chunk),
            ).fetchall()
            for row in rows:
                img_id = int(row["id"])
                result[img_id] = self._row_to_payload(row, tags_map.get(img_id))
        return result

    def get_payloads_by_point_ids(
        self, point_ids: Sequence[str], *, include_tags: bool = True
    ) -> dict[str, dict[str, Any]]:
        unique = [pid for pid in dict.fromkeys(point_ids) if pid]
        if not unique:
            return {}
        conn = self.db.read_conn()
        rows_found: list[sqlite3.Row] = []
        for chunk in _chunks(unique):
            marks = ",".join("?" for _ in chunk)
            rows_found.extend(
                conn.execute(
                    f"""
                    SELECT i.*, r.root_key, r.root_path, f.folder_key
                    FROM images AS i
                    JOIN roots AS r ON r.id = i.root_id
                    JOIN folders AS f ON f.id = i.folder_id
                    WHERE i.deleted = 0 AND i.point_id IN ({marks})
                    """,
                    tuple(chunk),
                ).fetchall()
            )
        if not rows_found:
            return {}
        tags_map = (
            self._load_tags_for_rows(conn, [int(r["id"]) for r in rows_found])
            if include_tags
            else {}
        )
        return {
            str(row["point_id"]): self._row_to_payload(row, tags_map.get(int(row["id"])))
            for row in rows_found
        }

    def iter_payloads(self) -> list[tuple[str, dict[str, Any]]]:
        conn = self.db.read_conn()
        rows = conn.execute(
            """
            SELECT i.*, r.root_key, r.root_path, f.folder_key
            FROM images AS i
            JOIN roots AS r ON r.id = i.root_id
            JOIN folders AS f ON f.id = i.folder_id
            WHERE i.deleted = 0
            ORDER BY i.id
            """
        ).fetchall()
        if not rows:
            return []
        tags_map = self._load_tags_for_rows(conn, [int(r["id"]) for r in rows])
        return [
            (str(row["point_id"]), self._row_to_payload(row, tags_map.get(int(row["id"]))))
            for row in rows
        ]

    def iter_scan_payloads(self, *, batch_size: int = 1000):
        """Stream minimal `(point_id, payload)` tuples in keyset batches for provenance scans."""
        last_id = -1
        while True:
            conn = self.db.read_conn()
            rows = conn.execute(
                """
                SELECT i.id, i.point_id, i.path, i.path_key, i.name, i.mtime_ns,
                       i.width, i.height, i.content_hash, i.phash,
                       r.root_key, r.root_path, f.folder_key
                FROM images AS i
                JOIN roots AS r ON r.id = i.root_id
                JOIN folders AS f ON f.id = i.folder_id
                WHERE i.deleted = 0 AND i.id > ?
                ORDER BY i.id
                LIMIT ?
                """,
                (last_id, int(batch_size)),
            ).fetchall()
            if not rows:
                return
            for row in rows:
                last_id = int(row["id"])
                pid = str(row["point_id"])
                yield pid, {
                    "row_id": last_id,
                    "id": pid,
                    "point_id": pid,
                    "path": str(row["path"]),
                    "path_key": str(row["path_key"]),
                    "name": str(row["name"]),
                    "mtime_ns": int(row["mtime_ns"]),
                    "width": int(row["width"]),
                    "height": int(row["height"]),
                    "content_hash": str(row["content_hash"]),
                    "phash": str(row["phash"]),
                    "root": str(row["root_path"]),
                    "root_key": str(row["root_key"]),
                    "folder_key": str(row["folder_key"]),
                }

    # ---------------------------------------------------------------- roots & folders
    def _ensure_root_locked(
        self,
        conn: sqlite3.Connection,
        root_key: str,
        root_path: str,
        *,
        recursive: bool = True,
    ) -> int:
        row = conn.execute(
            "SELECT id FROM roots WHERE root_key = ?", (root_key,)
        ).fetchone()
        if row is not None:
            conn.execute(
                "UPDATE roots SET root_path = ?, recursive = ? WHERE id = ?",
                (root_path, int(recursive), int(row["id"])),
            )
            return int(row["id"])
        cur = conn.execute(
            "INSERT INTO roots(root_key, root_path, recursive) VALUES (?, ?, ?)",
            (root_key, root_path, int(recursive)),
        )
        return int(cur.lastrowid)

    def _ensure_folder_chain_locked(
        self,
        conn: sqlite3.Connection,
        root_id: int,
        root_key: str,
        root_path: str,
        folder_key: str,
        folder_path: str,
        mtime_ns: int = 0,
        cache: dict[str, int] | None = None,
    ) -> int:
        if cache is not None and folder_key in cache:
            return cache[folder_key]

        row = conn.execute(
            "SELECT id FROM folders WHERE folder_key = ?", (folder_key,)
        ).fetchone()
        if row is not None:
            fid = int(row["id"])
            if mtime_ns:
                conn.execute(
                    "UPDATE folders SET mtime_ns = ?, path = ? WHERE id = ?",
                    (mtime_ns, folder_path, fid),
                )
            if cache is not None:
                cache[folder_key] = fid
            return fid

        # Build chain from folder up to root_key (case-insensitive path prefix).
        chain: list[tuple[str, str]] = [(folder_key, folder_path)]
        cur_key = folder_key
        cur_path = folder_path
        while cur_key != root_key:
            parent_path = os.path.dirname(cur_path)
            parent_key = os.path.dirname(cur_key)
            if not parent_key or parent_key == cur_key:
                break
            if parent_key == root_key:
                chain.append((root_key, root_path))
                break
            chain.append((parent_key, parent_path))
            cur_key = parent_key
            cur_path = parent_path

        parent_id: int | None = None
        for idx, (fkey, fpath) in enumerate(reversed(chain)):
            existing = conn.execute(
                "SELECT id FROM folders WHERE folder_key = ?", (fkey,)
            ).fetchone()
            if existing is not None:
                parent_id = int(existing["id"])
                if cache is not None:
                    cache[fkey] = parent_id
                continue
            name = os.path.basename(fpath.rstrip("\\/")) or fpath or fkey
            is_leaf = idx == len(chain) - 1
            cur = conn.execute(
                """
                INSERT INTO folders(
                    root_id, folder_key, path, name, parent_id, mtime_ns, selected, known
                )
                VALUES (?, ?, ?, ?, ?, ?, 0, 1)
                """,
                (root_id, fkey, fpath, name, parent_id, mtime_ns if is_leaf else 0),
            )
            parent_id = int(cur.lastrowid)
            if cache is not None:
                cache[fkey] = parent_id
        assert parent_id is not None
        return parent_id

    # ---------------------------------------------------------------- mutations
    def upsert_images_batch(self, records: Sequence[ImageUpsertRecord]) -> list[int]:
        """Insert or update a batch of images atomically, returning 0-based row IDs."""
        if not records:
            return []
        now = int(time.time())
        with self.db.write_tx() as conn:
            row_max = conn.execute("SELECT MAX(id) FROM images").fetchone()
            next_id = 0 if row_max is None or row_max[0] is None else int(row_max[0]) + 1

            root_cache: dict[str, int] = {}
            folder_cache: dict[str, int] = {}
            tag_cache: dict[str, int] = {}

            all_tags = {
                normalize_tag(tag)
                for rec in records
                for tag in rec.tags
                if normalize_tag(tag)
            }
            if all_tags:
                for chunk in _chunks(sorted(all_tags)):
                    marks = ",".join("?" for _ in chunk)
                    for r in conn.execute(
                        f"SELECT id, name FROM tags WHERE name IN ({marks})", tuple(chunk)
                    ).fetchall():
                        tag_cache[str(r["name"])] = int(r["id"])
                missing = sorted(t for t in all_tags if t not in tag_cache)
                if missing:
                    conn.executemany(
                        "INSERT OR IGNORE INTO tags(name) VALUES (?)",
                        [(t,) for t in missing],
                    )
                    for chunk in _chunks(missing):
                        marks = ",".join("?" for _ in chunk)
                        for r in conn.execute(
                            f"SELECT id, name FROM tags WHERE name IN ({marks})", tuple(chunk)
                        ).fetchall():
                            tag_cache[str(r["name"])] = int(r["id"])

            assigned_ids: list[int] = []
            for rec in records:
                root_id = root_cache.get(rec.root_key)
                if root_id is None:
                    root_id = self._ensure_root_locked(conn, rec.root_key, rec.root_path)
                    root_cache[rec.root_key] = root_id
                folder_id = self._ensure_folder_chain_locked(
                    conn,
                    root_id,
                    rec.root_key,
                    rec.root_path,
                    rec.folder_key,
                    rec.folder_path,
                    rec.folder_mtime_ns,
                    cache=folder_cache,
                )
                existing = conn.execute(
                    "SELECT id FROM images WHERE point_id = ? OR path_key = ?",
                    (rec.point_id, rec.path_key),
                ).fetchone()
                if existing is not None:
                    img_id = int(existing["id"])
                else:
                    img_id = next_id
                    next_id += 1

                caption_json = _encode_caption(rec.caption)
                indexed_at = rec.indexed_at if rec.indexed_at is not None else now
                conn.execute(
                    """
                    INSERT INTO images(
                        id, point_id, root_id, folder_id, path, path_key, name, ext,
                        mtime_ns, size_bytes, width, height, content_hash, phash,
                        indexed_at, descriptor_text, caption_json, deleted
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                    ON CONFLICT(id) DO UPDATE SET
                        point_id = excluded.point_id,
                        root_id = excluded.root_id,
                        folder_id = excluded.folder_id,
                        path = excluded.path,
                        path_key = excluded.path_key,
                        name = excluded.name,
                        ext = excluded.ext,
                        mtime_ns = excluded.mtime_ns,
                        size_bytes = excluded.size_bytes,
                        width = excluded.width,
                        height = excluded.height,
                        content_hash = excluded.content_hash,
                        phash = excluded.phash,
                        indexed_at = excluded.indexed_at,
                        descriptor_text = excluded.descriptor_text,
                        caption_json = excluded.caption_json,
                        deleted = 0
                    """,
                    (
                        img_id,
                        rec.point_id,
                        root_id,
                        folder_id,
                        rec.path,
                        rec.path_key,
                        rec.name,
                        rec.ext,
                        int(rec.mtime_ns),
                        int(rec.size_bytes),
                        int(rec.width),
                        int(rec.height),
                        rec.content_hash,
                        rec.phash,
                        int(indexed_at),
                        rec.descriptor_text,
                        caption_json,
                    ),
                )
                conn.execute("DELETE FROM image_tags WHERE image_id = ?", (img_id,))
                tag_rows: list[tuple[int, int, float]] = []
                seen_tags: set[int] = set()
                for raw_tag in rec.tags:
                    norm = normalize_tag(raw_tag)
                    if not norm:
                        continue
                    tid = tag_cache.get(norm)
                    if tid is None or tid in seen_tags:
                        continue
                    seen_tags.add(tid)
                    score = float(rec.tag_scores.get(raw_tag, rec.tag_scores.get(norm, 1.0)))
                    tag_rows.append((img_id, tid, score))
                if tag_rows:
                    conn.executemany(
                        "INSERT INTO image_tags(image_id, tag_id, score) VALUES (?, ?, ?)",
                        tag_rows,
                    )
                conn.execute(
                    "UPDATE image_source SET image_id = ? WHERE point_id = ?",
                    (img_id, rec.point_id),
                )
                conn.execute(
                    "UPDATE source_tags SET image_id = ? WHERE point_id = ?",
                    (img_id, rec.point_id),
                )
                conn.execute(
                    "UPDATE collection_items SET image_id = ? WHERE item_id = ?",
                    (img_id, rec.point_id),
                )
                assigned_ids.append(img_id)
            return assigned_ids

    def soft_delete_by_point_ids(self, point_ids: Sequence[str]) -> list[int]:
        unique = [pid for pid in dict.fromkeys(point_ids) if pid]
        if not unique:
            return []
        deleted_rows: list[int] = []
        with self.db.write_tx() as conn:
            for chunk in _chunks(unique):
                marks = ",".join("?" for _ in chunk)
                rows = conn.execute(
                    f"SELECT id FROM images WHERE deleted = 0 AND point_id IN ({marks})",
                    tuple(chunk),
                ).fetchall()
                ids = [int(r["id"]) for r in rows]
                if not ids:
                    continue
                id_marks = ",".join("?" for _ in ids)
                conn.execute(
                    f"UPDATE images SET deleted = 1 WHERE id IN ({id_marks})",
                    tuple(ids),
                )
                conn.execute(
                    f"DELETE FROM image_tags WHERE image_id IN ({id_marks})",
                    tuple(ids),
                )
                conn.execute(
                    f"UPDATE source_tags SET image_id = NULL WHERE image_id IN ({id_marks})",
                    tuple(ids),
                )
                conn.execute(
                    f"UPDATE image_source SET image_id = NULL WHERE image_id IN ({id_marks})",
                    tuple(ids),
                )
                conn.execute(
                    f"UPDATE collection_items SET image_id = NULL WHERE image_id IN ({id_marks})",
                    tuple(ids),
                )
                deleted_rows.extend(ids)
        return deleted_rows

    def soft_delete_folder_subtree(self, folder_key: str) -> tuple[list[int], list[str]]:
        key = canonical_path(folder_key) if os.path.isabs(folder_key) else folder_key.casefold()
        with self.db.write_tx() as conn:
            folder_rows = conn.execute(
                """
                WITH RECURSIVE tree(id) AS (
                    SELECT id FROM folders WHERE folder_key = ?
                    UNION ALL
                    SELECT f.id FROM folders AS f JOIN tree AS t ON f.parent_id = t.id
                )
                SELECT id FROM tree
                """,
                (key,),
            ).fetchall()
            fids = [int(r["id"]) for r in folder_rows]
            if not fids:
                return [], []
            f_marks = ",".join("?" for _ in fids)
            imgs = conn.execute(
                f"SELECT id, point_id FROM images WHERE deleted = 0 AND folder_id IN ({f_marks})",
                tuple(fids),
            ).fetchall()
            row_ids = [int(r["id"]) for r in imgs]
            point_ids = [str(r["point_id"]) for r in imgs]
            if row_ids:
                i_marks = ",".join("?" for _ in row_ids)
                conn.execute(
                    f"UPDATE images SET deleted = 1 WHERE id IN ({i_marks})",
                    tuple(row_ids),
                )
                conn.execute(
                    f"DELETE FROM image_tags WHERE image_id IN ({i_marks})",
                    tuple(row_ids),
                )
                conn.execute(
                    f"UPDATE source_tags SET image_id = NULL WHERE image_id IN ({i_marks})",
                    tuple(row_ids),
                )
                conn.execute(
                    f"UPDATE image_source SET image_id = NULL WHERE image_id IN ({i_marks})",
                    tuple(row_ids),
                )
                conn.execute(
                    f"UPDATE collection_items SET image_id = NULL WHERE image_id IN ({i_marks})",
                    tuple(row_ids),
                )
            conn.execute(
                f"UPDATE folders SET selected = 0 WHERE id IN ({f_marks})",
                tuple(fids),
            )
            return row_ids, point_ids

    def update_caption_and_descriptor(
        self, image_id: int, caption: dict[str, Any] | None, descriptor_text: str
    ) -> None:
        with self.db.write_tx() as conn:
            conn.execute(
                "UPDATE images SET caption_json = ?, descriptor_text = ? WHERE id = ?",
                (_encode_caption(caption), descriptor_text, int(image_id)),
            )

    def clear_all_images(self) -> None:
        with self.db.write_tx() as conn:
            conn.execute("UPDATE collection_items SET image_id = NULL")
            conn.execute("UPDATE image_source SET image_id = NULL")
            conn.execute("UPDATE source_tags SET image_id = NULL")
            conn.execute("DELETE FROM image_tags")
            conn.execute("DELETE FROM images")
            conn.execute("DELETE FROM folders")
            conn.execute("DELETE FROM roots")
            conn.execute(
                "DELETE FROM meta WHERE key IN ('dimension', 'embedding_model', 'tagger_model')"
            )

    # ---------------------------------------------------------------- queries & aggregates
    def list_folder_rows(self) -> list[sqlite3.Row]:
        return self.db.read_conn().execute(
            """
            WITH agg AS (
                SELECT folder_id, COUNT(*) AS direct_count, MAX(mtime_ns) AS max_mtime_ns
                FROM images
                WHERE deleted = 0
                GROUP BY folder_id
            )
            SELECT f.id, f.root_id, f.folder_key, f.path, f.name, f.parent_id,
                   f.mtime_ns, f.selected, f.known, r.root_key, r.root_path,
                   COALESCE(agg.direct_count, 0) AS direct_count,
                   MAX(f.mtime_ns, COALESCE(agg.max_mtime_ns, 0)) AS latest_mtime_ns
            FROM folders AS f
            JOIN roots AS r ON r.id = f.root_id
            LEFT JOIN agg ON agg.folder_id = f.id
            """
        ).fetchall()

    def root_image_counts(self) -> list[tuple[str, int]]:
        rows = self.db.read_conn().execute(
            """
            SELECT r.root_path, COUNT(i.id) AS cnt
            FROM roots AS r
            LEFT JOIN images AS i ON i.root_id = r.id AND i.deleted = 0
            GROUP BY r.id
            ORDER BY r.root_key
            """
        ).fetchall()
        return [(str(r["root_path"]), int(r["cnt"])) for r in rows]

    def time_bounds(self) -> tuple[int, int]:
        row = self.db.read_conn().execute(
            "SELECT COALESCE(MIN(mtime_ns), 0), COALESCE(MAX(mtime_ns), 0) "
            "FROM images WHERE deleted = 0 AND mtime_ns > 0"
        ).fetchone()
        return (int(row[0]), int(row[1])) if row else (0, 0)

    def image_tag_counts(
        self, query: str = "", *, limit: int = 60, offset: int = 0
    ) -> list[dict[str, Any]]:
        needle = str(query or "").strip()
        sql = """
            SELECT t.name AS tag, COUNT(*) AS cnt
            FROM tags AS t
            JOIN image_tags AS it ON it.tag_id = t.id
        """
        params: list[Any] = []
        if needle:
            sql += " WHERE t.name LIKE ? ESCAPE '\\' "
            escaped = needle.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            params.append(f"%{escaped}%")
        sql += " GROUP BY t.id ORDER BY cnt DESC, t.name COLLATE NOCASE LIMIT ? OFFSET ?"
        params.extend([int(limit), int(offset)])
        rows = self.db.read_conn().execute(sql, tuple(params)).fetchall()
        return [{"tag": str(r["tag"]), "count": int(r["cnt"])} for r in rows]

    def point_ids_for_image_tags(self, tags: Sequence[str]) -> list[str]:
        clean = sorted({normalize_tag(str(t)) for t in tags if normalize_tag(str(t))})
        if not clean:
            return []
        marks = ",".join("?" for _ in clean)
        rows = self.db.read_conn().execute(
            f"""
            SELECT i.point_id
            FROM images AS i
            JOIN image_tags AS it ON it.image_id = i.id
            JOIN tags AS t ON t.id = it.tag_id
            WHERE i.deleted = 0 AND t.name IN ({marks})
            GROUP BY i.id
            HAVING COUNT(DISTINCT t.name) = ?
            ORDER BY i.mtime_ns DESC, i.point_id
            """,
            (*clean, len(clean)),
        ).fetchall()
        return [str(r["point_id"]) for r in rows]

    def row_tags_map(self) -> dict[int, list[str]]:
        rows = self.db.read_conn().execute(
            """
            SELECT it.image_id, t.name
            FROM image_tags AS it
            JOIN tags AS t ON t.id = it.tag_id
            JOIN images AS i ON i.id = it.image_id AND i.deleted = 0
            ORDER BY it.image_id, it.score DESC
            """
        ).fetchall()
        out: dict[int, list[str]] = {}
        for r in rows:
            out.setdefault(int(r["image_id"]), []).append(str(r["name"]))
        return out

    def load_scope_state(self) -> dict[str, Any]:
        conn = self.db.read_conn()
        folders = conn.execute("SELECT folder_key, selected, known FROM folders").fetchall()
        selected = [str(r["folder_key"]) for r in folders if int(r["selected"]) == 1]
        known = [str(r["folder_key"]) for r in folders if int(r["known"]) == 1]
        return {
            "sort": self.meta_get("scope.sort", "name") or "name",
            "selected": selected,
            "known": known,
            "mtime_after": self.meta_get("scope.mtime_after"),
            "mtime_before": self.meta_get("scope.mtime_before"),
        }

    def save_scope_state(
        self,
        *,
        sort: str,
        selected: Sequence[str],
        known: Sequence[str],
        mtime_after: str | None,
        mtime_before: str | None,
    ) -> None:
        sel_set = set(selected)
        known_set = set(known) | sel_set
        with self.db.write_tx() as conn:
            conn.execute(
                "INSERT INTO meta(key, value) VALUES ('scope.sort', ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (sort,),
            )
            if mtime_after:
                conn.execute(
                    "INSERT INTO meta(key, value) VALUES ('scope.mtime_after', ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (mtime_after,),
                )
            else:
                conn.execute("DELETE FROM meta WHERE key = 'scope.mtime_after'")
            if mtime_before:
                conn.execute(
                    "INSERT INTO meta(key, value) VALUES ('scope.mtime_before', ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (mtime_before,),
                )
            else:
                conn.execute("DELETE FROM meta WHERE key = 'scope.mtime_before'")
            conn.execute("UPDATE folders SET selected = 0")
            for chunk in _chunks(sorted(sel_set)):
                marks = ",".join("?" for _ in chunk)
                conn.execute(
                    f"UPDATE folders SET selected = 1, known = 1 WHERE folder_key IN ({marks})",
                    tuple(chunk),
                )
            for chunk in _chunks(sorted(known_set)):
                marks = ",".join("?" for _ in chunk)
                conn.execute(
                    f"UPDATE folders SET known = 1 WHERE folder_key IN ({marks})",
                    tuple(chunk),
                )

    def compact_ids(self) -> list[int]:
        """Renumber live images densely to 0..M-1 inside a transaction; returns old row IDs."""
        with self.db._write_lock:
            conn = self.db.read_conn()
            if conn.in_transaction:
                raise RuntimeError("compact_ids() 不能在嵌套事务中调用")
            live = conn.execute(
                "SELECT id FROM images WHERE deleted = 0 ORDER BY id"
            ).fetchall()
            old_ids = [int(r["id"]) for r in live]
            if old_ids == list(range(len(old_ids))):
                with self.db.write_tx():
                    conn.execute("DELETE FROM images WHERE deleted = 1")
                return old_ids

            # SQLite ignores PRAGMA foreign_keys = OFF inside a transaction, so toggle outside.
            conn.execute("PRAGMA foreign_keys = OFF")
            try:
                with self.db.write_tx():
                    conn.execute("DELETE FROM images WHERE deleted = 1")
                    for new_id, old_id in enumerate(old_ids):
                        tmp_id = -(new_id + 1)
                        conn.execute("UPDATE images SET id = ? WHERE id = ?", (tmp_id, old_id))
                        conn.execute(
                            "UPDATE image_tags SET image_id = ? WHERE image_id = ?",
                            (tmp_id, old_id),
                        )
                        conn.execute(
                            "UPDATE collection_items SET image_id = ? WHERE image_id = ?",
                            (tmp_id, old_id),
                        )
                        conn.execute(
                            "UPDATE image_source SET image_id = ? WHERE image_id = ?",
                            (tmp_id, old_id),
                        )
                        conn.execute(
                            "UPDATE source_tags SET image_id = ? WHERE image_id = ?",
                            (tmp_id, old_id),
                        )
                    conn.execute("UPDATE images SET id = -id - 1 WHERE id < 0")
                    conn.execute(
                        "UPDATE image_tags SET image_id = -image_id - 1 WHERE image_id < 0"
                    )
                    conn.execute(
                        "UPDATE collection_items SET image_id = -image_id - 1 WHERE image_id < 0"
                    )
                    conn.execute(
                        "UPDATE image_source SET image_id = -image_id - 1 WHERE image_id < 0"
                    )
                    conn.execute(
                        "UPDATE source_tags SET image_id = -image_id - 1 WHERE image_id < 0"
                    )
            finally:
                conn.execute("PRAGMA foreign_keys = ON")
            return old_ids

    @staticmethod
    def build_upsert_record(
        path: Path,
        *,
        root: Path,
        point_id: str,
        width: int,
        height: int,
        mtime_ns: int,
        size_bytes: int,
        content_hash: str,
        phash: str,
        tags: Sequence[str] = (),
        tag_scores: dict[str, float] | None = None,
        caption: dict[str, Any] | None = None,
        descriptor_text: str = "",
        folder_mtime_ns: int = 0,
    ) -> ImageUpsertRecord:
        return ImageUpsertRecord(
            point_id=point_id,
            root_key=canonical_path(root),
            root_path=str(root.resolve()),
            folder_key=canonical_path(path.parent),
            folder_path=str(path.parent.resolve()),
            folder_mtime_ns=folder_mtime_ns,
            path=str(path.resolve()),
            path_key=canonical_path(path),
            name=path.name,
            ext=path.suffix.lower(),
            mtime_ns=mtime_ns,
            size_bytes=size_bytes,
            width=width,
            height=height,
            content_hash=content_hash,
            phash=phash,
            descriptor_text=descriptor_text,
            caption=caption,
            tags=list(tags),
            tag_scores=dict(tag_scores or {}),
        )
