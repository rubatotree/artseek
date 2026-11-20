"""One-shot, idempotent, strictly read-only migration from ArtSeek v1 to v2."""

from __future__ import annotations

import io
import json
import os
import pickle
import random
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..collections import CollectionStore, is_artist_item
from ..config import AppConfig
from ..core.sort_keys import natural_sort_text
from ..image_io import canonical_path
from ..library import DEFAULT_SORT, SORT_KEYS, normalize_day
from ..notes import HASH_RE, display_title
from ..provenance.store import _loads
from ..query import normalize_tag
from ..vectors.index import VectorIndex, _save_npy_atomic
from .connection import Database

QDRANT_COLLECTION = "artseek_images"
StatusCallback = Callable[[str], None]


class _DummyQdrantObj:
    """Stub object for unpickling Qdrant Pydantic v2 records without qdrant-client."""

    def __setstate__(self, state: Any) -> None:
        if isinstance(state, dict):
            inner = state.get("__dict__")
            if isinstance(inner, dict):
                self.__dict__.update(inner)
            else:
                self.__dict__.update(state)
        elif isinstance(state, tuple):
            for part in state:
                if isinstance(part, dict):
                    self.__dict__.update(part)


class _QdrantPointUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if module.startswith("qdrant_client") or module.startswith("pydantic"):
            return _DummyQdrantObj
        return super().find_class(module, name)


def _unpickle_point(blob: bytes) -> Any:
    return _QdrantPointUnpickler(io.BytesIO(blob)).load()


def _connect_ro(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True, timeout=15.0)
    conn.row_factory = sqlite3.Row
    return conn


def _normalize_vector(raw: Any, dim: int) -> np.ndarray:
    arr = np.asarray(raw, dtype=np.float32).reshape(-1)
    if arr.shape[0] != dim:
        raise ValueError(f"向量维度不匹配：期望 {dim}，实际 {arr.shape[0]}")
    norm = float(np.linalg.norm(arr))
    if norm > 1e-12:
        arr = arr / norm
    return arr.astype(np.float32, copy=False)


def _extract_tag_scores(raw_tags: Any, raw_scores: Any) -> tuple[list[str], dict[str, float]]:
    scores_map: dict[str, float] = {}
    if isinstance(raw_scores, dict):
        for k, v in raw_scores.items():
            norm = normalize_tag(str(k))
            if norm:
                scores_map[norm] = float(v)
    elif isinstance(raw_scores, (list, tuple)):
        for entry in raw_scores:
            if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                norm = normalize_tag(str(entry[0]))
                if norm:
                    scores_map[norm] = float(entry[1])

    tags: list[str] = []
    seen: set[str] = set()
    for raw in raw_tags or ():
        norm = normalize_tag(str(raw))
        if norm and norm not in seen:
            seen.add(norm)
            tags.append(norm)
    return tags, scores_map


def _encode_caption(caption: Any) -> str | None:
    if caption is None:
        return None
    if isinstance(caption, str):
        return caption
    return json.dumps(caption, ensure_ascii=False)


@dataclass(slots=True)
class MigrationReport:
    images_v1: int = 0
    images_v2: int = 0
    dimension: int = 0
    collections_v1: int = 0
    collections_v2: int = 0
    collection_items_mismatches: list[str] = field(default_factory=list)
    artists_v1: int = 0
    artists_v2: int = 0
    identities_v1: int = 0
    identities_v2: int = 0
    legacy_v1: int = 0
    legacy_v2: int = 0
    image_source_v1: int = 0
    image_source_v2: int = 0
    notes_folders: int = 0
    notes_files_v1: int = 0
    notes_files_v2: int = 0
    query_overlap_min: float = 1.0
    query_overlap_mean: float = 1.0
    elapsed_seconds: float = 0.0

    def assert_valid(self) -> None:
        checks = [
            (self.images_v1 == self.images_v2, f"images {self.images_v1} != {self.images_v2}"),
            (
                self.collections_v1 == self.collections_v2,
                f"collections {self.collections_v1} != {self.collections_v2}",
            ),
            (
                not self.collection_items_mismatches,
                f"collection_items mismatch: {self.collection_items_mismatches}",
            ),
            (self.artists_v1 == self.artists_v2, f"artists {self.artists_v1} != {self.artists_v2}"),
            (
                self.identities_v1 == self.identities_v2,
                f"identities {self.identities_v1} != {self.identities_v2}",
            ),
            (self.legacy_v1 == self.legacy_v2, f"legacy {self.legacy_v1} != {self.legacy_v2}"),
            (
                self.image_source_v1 == self.image_source_v2,
                f"image_source {self.image_source_v1} != {self.image_source_v2}",
            ),
            (
                self.notes_files_v1 == self.notes_files_v2,
                f"notes {self.notes_files_v1} != {self.notes_files_v2}",
            ),
            (
                self.query_overlap_min >= 0.95,
                f"top20 overlap {self.query_overlap_min:.3f} < 0.95",
            ),
        ]
        errors = [msg for ok, msg in checks if not ok]
        if errors:
            raise RuntimeError("迁移校验失败：" + "; ".join(errors))

    def to_dict(self) -> dict[str, Any]:
        return {
            "images_v1": self.images_v1,
            "images_v2": self.images_v2,
            "dimension": self.dimension,
            "collections_v1": self.collections_v1,
            "collections_v2": self.collections_v2,
            "collection_items_mismatches": list(self.collection_items_mismatches),
            "artists_v1": self.artists_v1,
            "artists_v2": self.artists_v2,
            "identities_v1": self.identities_v1,
            "identities_v2": self.identities_v2,
            "legacy_v1": self.legacy_v1,
            "legacy_v2": self.legacy_v2,
            "image_source_v1": self.image_source_v1,
            "image_source_v2": self.image_source_v2,
            "notes_folders": self.notes_folders,
            "notes_files_v1": self.notes_files_v1,
            "notes_files_v2": self.notes_files_v2,
            "query_overlap_min": round(self.query_overlap_min, 4),
            "query_overlap_mean": round(self.query_overlap_mean, 4),
            "elapsed_seconds": round(self.elapsed_seconds, 2),
        }


def _read_qdrant_meta(qdrant_path: Path) -> int:
    meta_file = qdrant_path / "meta.json"
    if meta_file.is_file():
        data = json.loads(meta_file.read_text(encoding="utf-8"))
        col = (data.get("collections") or {}).get(QDRANT_COLLECTION) or {}
        vectors = col.get("vectors") or {}
        visual = vectors.get("visual") or {}
        size = int(visual.get("size") or 0)
        if size > 0:
            return size
    return 0


def _ensure_folder_chain(
    conn: sqlite3.Connection,
    *,
    root_id: int,
    root_key: str,
    root_path: str,
    folder_key: str,
    folder_path: str,
    folder_cache: dict[str, int],
    dir_mtimes: dict[str, int],
) -> int:
    cached = folder_cache.get(folder_key)
    if cached is not None:
        return cached

    chain: list[tuple[str, str]] = [(folder_key, folder_path)]
    cur_key = folder_key
    cur_path = folder_path
    while cur_key != root_key:
        parent_key = os.path.dirname(cur_key)
        parent_path = os.path.dirname(cur_path)
        if not parent_key or parent_key == cur_key:
            break
        if parent_key == root_key:
            chain.append((root_key, root_path))
            break
        chain.append((parent_key, parent_path))
        cur_key = parent_key
        cur_path = parent_path

    parent_id: int | None = None
    for fkey, fpath in reversed(chain):
        existing = folder_cache.get(fkey)
        if existing is not None:
            parent_id = existing
            continue
        row = conn.execute(
            "SELECT id FROM folders WHERE folder_key = ?", (fkey,)
        ).fetchone()
        if row is not None:
            parent_id = int(row["id"])
            folder_cache[fkey] = parent_id
            continue
        name = os.path.basename(fpath.rstrip("\\/")) or fpath or fkey
        mtime_ns = int(dir_mtimes.get(fkey, 0))
        cur = conn.execute(
            """
            INSERT INTO folders(
                root_id, folder_key, path, name, parent_id, mtime_ns, selected, known
            )
            VALUES (?, ?, ?, ?, ?, ?, 0, 1)
            """,
            (root_id, fkey, fpath, name, parent_id, mtime_ns),
        )
        parent_id = int(cur.lastrowid)
        folder_cache[fkey] = parent_id
    assert parent_id is not None
    return parent_id


def migrate_v1(
    config: AppConfig,
    *,
    qdrant_path: Path | None = None,
    index_dir: Path | None = None,
    notes_dir: Path | None = None,
    output_dir: Path | None = None,
    status: StatusCallback | None = None,
) -> MigrationReport:
    """Read v1 files read-only and atomically create v2 database and vector files."""
    started = time.perf_counter()
    notify = status or (lambda _msg: None)
    src_qdrant = Path(qdrant_path) if qdrant_path is not None else config.qdrant_path
    src_index = Path(index_dir) if index_dir is not None else config.index_dir
    src_notes = Path(notes_dir) if notes_dir is not None else config.notes_dir
    dst_dir = Path(output_dir) if output_dir is not None else config.index_dir
    dst_dir.mkdir(parents=True, exist_ok=True)

    storage_sqlite = src_qdrant / "collection" / QDRANT_COLLECTION / "storage.sqlite"
    if not storage_sqlite.is_file():
        raise FileNotFoundError(f"未找到 v1 Qdrant 数据库：{storage_sqlite}")

    report = MigrationReport()

    tmp_db_path = dst_dir / "artseek.sqlite3.migrating"
    tmp_vis_path = dst_dir / "visual.f32.npy.migrating"
    tmp_desc_path = dst_dir / "descriptor.f32.npy.migrating"
    for p in (
        tmp_db_path,
        tmp_db_path.with_name(tmp_db_path.name + "-wal"),
        tmp_db_path.with_name(tmp_db_path.name + "-shm"),
        tmp_vis_path,
        tmp_desc_path,
    ):
        p.unlink(missing_ok=True)

    db = Database(tmp_db_path)
    sample_queries: list[tuple[int, np.ndarray, np.ndarray]] = []
    visual_mat = np.zeros((0, 0), dtype=np.float32)
    descriptor_mat = np.zeros((0, 0), dtype=np.float32)

    try:
        # 1. Read manifest & meta
        manifest_path = src_index / "manifest.json"
        manifest: dict[str, Any] = {}
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        dim = int(manifest.get("dimension") or 0) or _read_qdrant_meta(src_qdrant)

        # 2. Read index_state.sqlite3 (read-only)
        dir_mtimes: dict[str, int] = {}
        root_records: list[tuple[str, str, int, int, int]] = []
        state_path = src_index / "index_state.sqlite3"
        if state_path.is_file():
            state_conn = _connect_ro(state_path)
            try:
                dirty_keys = {
                    str(r["root_key"])
                    for r in state_conn.execute("SELECT root_key FROM dirty_roots").fetchall()
                }
                for r in state_conn.execute(
                    "SELECT root_key, root_path, recursive, last_full_verification_ns FROM roots"
                ).fetchall():
                    rk = str(r["root_key"])
                    root_records.append(
                        (
                            rk,
                            str(r["root_path"]),
                            int(r["recursive"]),
                            int(r["last_full_verification_ns"]),
                            1 if rk in dirty_keys else 0,
                        )
                    )
                for r in state_conn.execute(
                    "SELECT path_key, mtime_ns FROM directories"
                ).fetchall():
                    dir_mtimes[str(r["path_key"])] = int(r["mtime_ns"])
            finally:
                state_conn.close()

        # 3. Read Qdrant storage.sqlite (read-only)
        notify("正在只读迁移 Qdrant 向量与图片元数据…")
        q_conn = _connect_ro(storage_sqlite)
        point_to_row: dict[str, int] = {}
        try:
            total_points = int(q_conn.execute("SELECT COUNT(*) FROM points").fetchone()[0])
            report.images_v1 = total_points

            if dim <= 0 and total_points > 0:
                first_blob = q_conn.execute("SELECT point FROM points LIMIT 1").fetchone()[0]
                first_obj = _unpickle_point(first_blob)
                dim = len(first_obj.vector["visual"])
            report.dimension = dim

            visual_mat = np.zeros((total_points, dim), dtype=np.float32)
            descriptor_mat = np.zeros((total_points, dim), dtype=np.float32)

            sample_indices = set(
                random.Random(20261120).sample(range(total_points), min(20, total_points))
            )

            with db.write_tx() as conn:
                conn.executemany(
                    """
                    INSERT OR IGNORE INTO roots(
                        root_key, root_path, recursive, last_full_verification_ns, dirty
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    root_records,
                )
                root_cache: dict[str, int] = {
                    str(r["root_key"]): int(r["id"])
                    for r in conn.execute("SELECT id, root_key FROM roots").fetchall()
                }
                folder_cache: dict[str, int] = {}
                tag_cache: dict[str, int] = {}

                cursor = q_conn.execute("SELECT point FROM points")
                row_idx = 0
                now_ts = int(time.time())
                while True:
                    batch = cursor.fetchmany(1024)
                    if not batch:
                        break
                    img_rows: list[tuple[Any, ...]] = []
                    pending_tags: list[tuple[int, list[str], dict[str, float]]] = []
                    for (blob,) in batch:
                        obj = _unpickle_point(blob)
                        pid = str(obj.id)
                        payload = dict(obj.payload or {})
                        vecs = dict(obj.vector or {})
                        vis_vec = _normalize_vector(vecs["visual"], dim)
                        desc_raw = vecs.get("descriptor")
                        desc_vec = (
                            _normalize_vector(desc_raw, dim)
                            if desc_raw is not None
                            else vis_vec.copy()
                        )
                        visual_mat[row_idx] = vis_vec
                        descriptor_mat[row_idx] = desc_vec
                        if row_idx in sample_indices:
                            sample_queries.append((row_idx, vis_vec.copy(), desc_vec.copy()))

                        raw_path = str(payload.get("path") or "")
                        path_key = str(payload.get("path_key") or raw_path.casefold())
                        root_path = str(payload.get("root") or os.path.dirname(raw_path))
                        root_key = str(payload.get("root_key") or root_path.casefold())
                        folder_path = os.path.dirname(raw_path)
                        folder_key = str(
                            payload.get("folder_key") or os.path.dirname(path_key)
                        )

                        root_id = root_cache.get(root_key)
                        if root_id is None:
                            cur = conn.execute(
                                "INSERT INTO roots(root_key, root_path) VALUES (?, ?)",
                                (root_key, root_path),
                            )
                            root_id = int(cur.lastrowid)
                            root_cache[root_key] = root_id

                        folder_id = _ensure_folder_chain(
                            conn,
                            root_id=root_id,
                            root_key=root_key,
                            root_path=root_path,
                            folder_key=folder_key,
                            folder_path=folder_path,
                            folder_cache=folder_cache,
                            dir_mtimes=dir_mtimes,
                        )

                        tags, scores_map = _extract_tag_scores(
                            payload.get("tags"), payload.get("tag_scores")
                        )
                        for t in tags:
                            if t not in tag_cache:
                                conn.execute(
                                    "INSERT OR IGNORE INTO tags(name) VALUES (?)", (t,)
                                )
                                tid = int(
                                    conn.execute(
                                        "SELECT id FROM tags WHERE name = ?", (t,)
                                    ).fetchone()[0]
                                )
                                tag_cache[t] = tid

                        img_rows.append(
                            (
                                row_idx,
                                pid,
                                root_id,
                                folder_id,
                                raw_path,
                                path_key,
                                str(payload.get("name") or os.path.basename(raw_path)),
                                str(payload.get("extension") or ""),
                                int(payload.get("mtime_ns") or 0),
                                int(payload.get("size_bytes") or 0),
                                int(payload.get("width") or 0),
                                int(payload.get("height") or 0),
                                str(payload.get("content_hash") or ""),
                                str(payload.get("phash") or ""),
                                now_ts,
                                str(payload.get("descriptor_text") or ""),
                                _encode_caption(payload.get("caption")),
                                natural_sort_text(
                                    str(payload.get("name") or os.path.basename(raw_path))
                                ),
                            )
                        )
                        if tags:
                            pending_tags.append((row_idx, tags, scores_map))
                        point_to_row[pid] = row_idx
                        row_idx += 1

                    conn.executemany(
                        """
                        INSERT INTO images(
                            id, point_id, root_id, folder_id, path, path_key, name, ext,
                            mtime_ns, size_bytes, width, height, content_hash, phash,
                            indexed_at, descriptor_text, caption_json, deleted, sort_name
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
                        """,
                        img_rows,
                    )
                    it_rows: list[tuple[int, int, float]] = []
                    for img_id, tags, scores_map in pending_tags:
                        for t in tags:
                            it_rows.append((img_id, tag_cache[t], scores_map.get(t, 1.0)))
                    if it_rows:
                        conn.executemany(
                            "INSERT INTO image_tags(image_id, tag_id, score) VALUES (?, ?, ?)",
                            it_rows,
                        )

                # Store manifest & vector commit metadata
                meta_items = [
                    ("v2_schema_version", "1"),
                    ("dimension", str(dim)),
                    (
                        "embedding_model",
                        str(manifest.get("embedding_model") or config.embedding.model),
                    ),
                    ("vector_rows_committed", str(total_points)),
                ]
                if manifest.get("tagger_model") is not None:
                    meta_items.append(("tagger_model", str(manifest["tagger_model"])))
                conn.executemany(
                    "INSERT INTO meta(key, value) VALUES (?, ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    meta_items,
                )
        finally:
            q_conn.close()

        # 4. Import folder_scope.json
        scope_path = src_index / "folder_scope.json"
        if scope_path.is_file():
            raw_scope = json.loads(scope_path.read_text(encoding="utf-8"))
            scope_sort = str(raw_scope.get("sort") or DEFAULT_SORT)
            if scope_sort not in SORT_KEYS:
                scope_sort = DEFAULT_SORT
            scope_selected = [
                canonical_path(item) for item in raw_scope.get("selected", []) if item
            ]
            scope_known = [canonical_path(item) for item in raw_scope.get("known", []) if item]
            mtime_after = normalize_day(raw_scope.get("mtime_after"))
            mtime_before = normalize_day(raw_scope.get("mtime_before"))
            if mtime_after and mtime_before and mtime_after > mtime_before:
                mtime_after, mtime_before = mtime_before, mtime_after
            with db.write_tx() as conn:
                conn.execute(
                    "INSERT INTO meta(key, value) VALUES ('scope.sort', ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (scope_sort,),
                )
                if mtime_after:
                    conn.execute(
                        "INSERT INTO meta(key, value) VALUES ('scope.mtime_after', ?) "
                        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                        (mtime_after,),
                    )
                if mtime_before:
                    conn.execute(
                        "INSERT INTO meta(key, value) VALUES ('scope.mtime_before', ?) "
                        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                        (mtime_before,),
                    )
                if scope_selected:
                    for start in range(0, len(scope_selected), 400):
                        chunk = scope_selected[start : start + 400]
                        marks = ",".join("?" for _ in chunk)
                        conn.execute(
                            f"UPDATE folders SET selected = 1, known = 1 "
                            f"WHERE folder_key IN ({marks})",
                            tuple(chunk),
                        )
                if scope_known:
                    for start in range(0, len(scope_known), 400):
                        chunk = scope_known[start : start + 400]
                        marks = ",".join("?" for _ in chunk)
                        conn.execute(
                            f"UPDATE folders SET known = 1 WHERE folder_key IN ({marks})",
                            tuple(chunk),
                        )

        # 5. Import provenance.sqlite3 (read-only)
        prov_path = src_index / "provenance.sqlite3"
        if prov_path.is_file():
            notify("正在只读迁移 provenance 来源与画师数据…")
            p_conn = _connect_ro(prov_path)
            try:
                report.artists_v1 = int(p_conn.execute("SELECT COUNT(*) FROM artist").fetchone()[0])
                report.identities_v1 = int(
                    p_conn.execute("SELECT COUNT(*) FROM artist_identity").fetchone()[0]
                )
                report.legacy_v1 = int(
                    p_conn.execute("SELECT COUNT(*) FROM artist_legacy").fetchone()[0]
                )
                report.image_source_v1 = int(
                    p_conn.execute("SELECT COUNT(*) FROM image_source").fetchone()[0]
                )

                with db.write_tx() as conn:
                    artist_rows = [
                        (
                            int(r["id"]),
                            r["name"],
                            r["name_key"],
                            r["name_source"],
                            r["avatar_url"],
                            r["note"],
                            int(r["profile_fetched"]),
                            0,  # Recomputed after triggers/bulk insert
                            int(r["created_at"]),
                            int(r["updated_at"]),
                        )
                        for r in p_conn.execute("SELECT * FROM artist").fetchall()
                    ]
                    if artist_rows:
                        conn.executemany(
                            """
                            INSERT INTO artist(
                                id, name, name_key, name_source, avatar_url, note,
                                profile_fetched, image_count, created_at, updated_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            artist_rows,
                        )
                        max_artist_id = max(r[0] for r in artist_rows)
                        conn.execute(
                            "INSERT OR REPLACE INTO sqlite_sequence(name, seq) "
                            "VALUES ('artist', ?)",
                            (max_artist_id,),
                        )

                    id_rows = [
                        (
                            str(r["kind"]),
                            str(r["identity"]),
                            r["display"],
                            r["profile_url"],
                            r["source"],
                            int(r["artist_id"]),
                            int(r["updated_at"]),
                        )
                        for r in p_conn.execute("SELECT * FROM artist_identity").fetchall()
                    ]
                    if id_rows:
                        conn.executemany(
                            """
                            INSERT INTO artist_identity(
                                kind, identity, display, profile_url, source, artist_id, updated_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?)
                            """,
                            id_rows,
                        )

                    legacy_rows = [
                        (str(r["legacy_key"]), int(r["artist_id"]), int(r["updated_at"]))
                        for r in p_conn.execute("SELECT * FROM artist_legacy").fetchall()
                    ]
                    if legacy_rows:
                        conn.executemany(
                            "INSERT INTO artist_legacy(legacy_key, artist_id, updated_at) "
                            "VALUES (?, ?, ?)",
                            legacy_rows,
                        )

                    fetch_rows = [
                        (
                            str(r["domain"]),
                            r["blocked_until"],
                            r["last_request"],
                            r["day"],
                            r["day_count"],
                            r["long_remaining"],
                            r["last_error"],
                        )
                        for r in p_conn.execute("SELECT * FROM fetch_state").fetchall()
                    ]
                    if fetch_rows:
                        conn.executemany(
                            """
                            INSERT INTO fetch_state(
                                domain, blocked_until, last_request, day, day_count,
                                long_remaining, last_error
                            ) VALUES (?, ?, ?, ?, ?, ?, ?)
                            """,
                            fetch_rows,
                        )

                    src_cursor = p_conn.execute("SELECT * FROM image_source")
                    while True:
                        batch = src_cursor.fetchmany(1024)
                        if not batch:
                            break
                        src_inserts: list[tuple[Any, ...]] = []
                        st_inserts: list[tuple[Any, ...]] = []
                        hash_backfills: list[tuple[str, int]] = []
                        for r in batch:
                            pid = str(r["point_id"])
                            img_id = point_to_row.get(pid)
                            ak = str(r["author_key"] or "").strip() or None
                            artist_id = int(ak) if ak and ak.isdigit() else None
                            src_inserts.append(
                                (
                                    pid,
                                    img_id,
                                    r["content_hash"],
                                    r["path_key"],
                                    str(r["platform"]),
                                    str(r["source_id"]),
                                    r["page"],
                                    str(r["url"]),
                                    r["candidates_json"],
                                    str(r["status"]),
                                    float(r["confidence"]),
                                    r["title"],
                                    r["description"],
                                    ak,
                                    artist_id,
                                    r["posted_at"],
                                    r["width"],
                                    r["height"],
                                    r["mtime_ns"],
                                    r["page_count"],
                                    r["tags_json"],
                                    r["manual_tags_json"],
                                    r["auto_tags_json"],
                                    r["auto_tag_scores_json"],
                                    r["auto_tagged_at"],
                                    r["raw_json"],
                                    str(r["source_origin"]),
                                    r["manual_fields"],
                                    r["fetched_at"],
                                    r["error"],
                                    int(r["updated_at"]),
                                )
                            )
                            if img_id is not None and r["content_hash"]:
                                hash_backfills.append((str(r["content_hash"]), img_id))

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
                                    sc = (
                                        float(scores[tag])
                                        if origin == "auto" and tag in scores
                                        else None
                                    )
                                    st_inserts.append((pid, img_id, tag, norm, origin, sc))

                        conn.executemany(
                            """
                            INSERT INTO image_source(
                                point_id, image_id, content_hash, path_key, platform, source_id,
                                page, url, candidates_json, status, confidence, title, description,
                                author_key, artist_id, posted_at, width, height, mtime_ns,
                                page_count, tags_json, manual_tags_json, auto_tags_json,
                                auto_tag_scores_json, auto_tagged_at, raw_json, source_origin,
                                manual_fields, fetched_at, error, updated_at
                            ) VALUES (
                                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                            )
                            """,
                            src_inserts,
                        )
                        if st_inserts:
                            conn.executemany(
                                """
                                INSERT OR REPLACE INTO source_tags(
                                    point_id, image_id, tag, tag_norm, origin, score
                                ) VALUES (?, ?, ?, ?, ?, ?)
                                """,
                                st_inserts,
                            )
                        if hash_backfills:
                            conn.executemany(
                                "UPDATE images SET content_hash = ? "
                                "WHERE id = ? AND (content_hash IS NULL OR content_hash = '')",
                                hash_backfills,
                            )

                    # Reconcile artist.image_count cleanly
                    conn.execute(
                        """
                        UPDATE artist
                        SET image_count = COALESCE(
                            (SELECT COUNT(*) FROM image_source WHERE artist_id = artist.id),
                            0
                        )
                        """
                    )
            finally:
                p_conn.close()

        # 6. Import collections.json (read-only)
        collections_path = src_index / "collections.json"
        expected_col_counts: dict[str, int] = {}
        if collections_path.is_file():
            notify("正在迁移收藏夹与历史栈…")
            doc = CollectionStore(collections_path)._load_document()
            report.collections_v1 = len(doc.collections)
            for col in doc.collections:
                expected_col_counts[col.id] = len(col.item_ids)
            with db.write_tx() as conn:
                conn.executemany(
                    "INSERT INTO collections(id, name, position, sort) VALUES (?, ?, ?, ?)",
                    [(col.id, col.name, idx, col.sort) for idx, col in enumerate(doc.collections)],
                )
                item_rows: list[tuple[str, str, int | None, int, int | None]] = []
                for col in doc.collections:
                    manual_pos = {item_id: idx for idx, item_id in enumerate(col.manual_order)}
                    for added_idx, item_id in enumerate(col.item_ids):
                        img_id = None if is_artist_item(item_id) else point_to_row.get(item_id)
                        item_rows.append(
                            (col.id, item_id, img_id, added_idx, manual_pos.get(item_id))
                        )
                if item_rows:
                    conn.executemany(
                        """
                        INSERT INTO collection_items(
                            collection_id, item_id, image_id, added_position, manual_position
                        ) VALUES (?, ?, ?, ?, ?)
                        """,
                        item_rows,
                    )
                hist_rows: list[tuple[str, int, str]] = []
                for seq, entry in enumerate(doc.undo_stack):
                    hist_rows.append(("undo", seq, json.dumps(entry.encode(), ensure_ascii=False)))
                for seq, entry in enumerate(doc.redo_stack):
                    hist_rows.append(("redo", seq, json.dumps(entry.encode(), ensure_ascii=False)))
                if hist_rows:
                    conn.executemany(
                        "INSERT INTO collection_history(stack, seq, entry_json) VALUES (?, ?, ?)",
                        hist_rows,
                    )

        # 7. Scan notes_dir (strictly read-only)
        if src_notes.is_dir():
            notify("正在扫描笔记索引…")
            notes_rows: list[tuple[str, str, str, int]] = []
            folder_count = 0
            with os.scandir(src_notes) as entries:
                for entry in entries:
                    if not entry.is_dir(follow_symlinks=False):
                        continue
                    digest = entry.name.lower()
                    if not HASH_RE.match(digest):
                        continue
                    folder_count += 1
                    with os.scandir(entry.path) as files:
                        for f in files:
                            if (
                                not f.is_file(follow_symlinks=False)
                                or f.name.startswith(".")
                                or not f.name.lower().endswith(".md")
                            ):
                                continue
                            try:
                                text = Path(f.path).read_text(encoding="utf-8")
                                stat = f.stat()
                            except OSError:
                                continue
                            title = display_title(text, f.name)
                            notes_rows.append((digest, f.name, title, int(stat.st_mtime)))
            report.notes_folders = folder_count
            report.notes_files_v1 = len(notes_rows)
            if notes_rows:
                with db.write_tx() as conn:
                    conn.executemany(
                        """
                        INSERT OR REPLACE INTO notes_index(
                            content_hash, name, title, updated_at
                        ) VALUES (?, ?, ?, ?)
                        """,
                        notes_rows,
                    )

        # 8. Write .npy matrices atomically to .migrating paths
        notify("正在写入向量矩阵 .npy…")
        _save_npy_atomic(tmp_vis_path, visual_mat)
        _save_npy_atomic(tmp_desc_path, descriptor_mat)

        db.checkpoint()
    finally:
        db.close()

    # 9. Atomic swap into dst_dir
    final_db = dst_dir / "artseek.sqlite3"
    final_vis = dst_dir / "visual.f32.npy"
    final_desc = dst_dir / "descriptor.f32.npy"
    for suffix in ("-wal", "-shm", "-journal"):
        final_db.with_name(final_db.name + suffix).unlink(missing_ok=True)
        tmp_db_path.with_name(tmp_db_path.name + suffix).unlink(missing_ok=True)
    os.replace(tmp_vis_path, final_vis)
    os.replace(tmp_desc_path, final_desc)
    os.replace(tmp_db_path, final_db)

    # 10. Verify migrated database and top-20 query overlap
    notify("正在校验迁移计数与向量检索一致性…")
    verify_db = Database(final_db)
    try:
        conn = verify_db.read_conn()
        report.images_v2 = int(
            conn.execute("SELECT COUNT(*) FROM images WHERE deleted = 0").fetchone()[0]
        )
        report.collections_v2 = int(conn.execute("SELECT COUNT(*) FROM collections").fetchone()[0])
        for cid, exp_cnt in expected_col_counts.items():
            got_cnt = int(
                conn.execute(
                    "SELECT COUNT(*) FROM collection_items WHERE collection_id = ?", (cid,)
                ).fetchone()[0]
            )
            if got_cnt != exp_cnt:
                report.collection_items_mismatches.append(f"{cid}: {got_cnt} != {exp_cnt}")
        report.artists_v2 = int(conn.execute("SELECT COUNT(*) FROM artist").fetchone()[0])
        report.identities_v2 = int(
            conn.execute("SELECT COUNT(*) FROM artist_identity").fetchone()[0]
        )
        report.legacy_v2 = int(conn.execute("SELECT COUNT(*) FROM artist_legacy").fetchone()[0])
        report.image_source_v2 = int(
            conn.execute("SELECT COUNT(*) FROM image_source").fetchone()[0]
        )
        report.notes_files_v2 = int(conn.execute("SELECT COUNT(*) FROM notes_index").fetchone()[0])
    finally:
        verify_db.close()

    if sample_queries and report.images_v2 > 0:
        v_index = VectorIndex(final_vis, final_desc)
        overlaps: list[float] = []
        k = min(20, report.images_v2)
        for _row_idx, vis_q, _desc_q in sample_queries:
            # Exact brute-force baseline on the source float32 vectors (identical to Qdrant Cosine)
            ref_scores = visual_mat @ vis_q
            ref_top = set(np.argsort(-ref_scores, kind="stable")[:k].tolist())
            v2_rows, _ = v_index.search(vis_q, "visual", k)
            v2_top = set(v2_rows.tolist())
            overlaps.append(len(ref_top & v2_top) / float(k))
        report.query_overlap_min = min(overlaps)
        report.query_overlap_mean = sum(overlaps) / len(overlaps)

    report.elapsed_seconds = time.perf_counter() - started
    report.assert_valid()
    return report
