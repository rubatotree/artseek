"""SQLite CTE-backed ImageScope resolver and paged row fetcher."""

from __future__ import annotations

import json
import random
import re
import threading
from collections import OrderedDict
from typing import Any

import numpy as np

from ..core.scope import ImageScope
from ..query import normalize_tag
from ..tag_dictionary import expand_tag_aliases
from ..tag_query import And, Atom, Node, Not, Or, parse_tag_expression
from .connection import Database
from .repos.provenance import ProvenanceRepo


def _compile_tag_expr(node: Node, params: list[Any]) -> str:
    if isinstance(node, Atom):
        aliases = sorted(
            {a.strip().casefold() for a in expand_tag_aliases(node.value) if a.strip()}
        )
        if not aliases:
            return "0"
        params.append(json.dumps(aliases, ensure_ascii=False))
        return (
            "i.id IN (SELECT st.image_id FROM source_tags AS st "
            "WHERE st.image_id IS NOT NULL "
            "AND st.tag_norm IN (SELECT value FROM json_each(?)))"
        )
    if isinstance(node, Not):
        inner = _compile_tag_expr(node.child, params)
        return (
            f"(i.id IN (SELECT s.image_id FROM image_source AS s WHERE s.image_id IS NOT NULL) "
            f"AND NOT ({inner}))"
        )
    if isinstance(node, And):
        left = _compile_tag_expr(node.left, params)
        right = _compile_tag_expr(node.right, params)
        return f"({left} AND {right})"
    if isinstance(node, Or):
        left = _compile_tag_expr(node.left, params)
        right = _compile_tag_expr(node.right, params)
        return f"({left} OR {right})"
    raise ValueError("未知表达式节点")


def _deterministic_random_order(ids: np.ndarray, seed: int) -> np.ndarray:
    """Order row IDs deterministically from `(seed, id)` using SplitMix64 mixing."""
    if ids.size <= 1:
        return ids
    x = ids.astype(np.uint64) ^ np.uint64(seed & 0xFFFFFFFFFFFFFFFF)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    keys = x ^ (x >> np.uint64(31))
    order = np.argsort(keys, kind="stable")
    return ids[order]


class ScopeResolver:
    """Resolves unified `ImageScope` filters into SQLite predicates and row arrays."""

    def __init__(self, db: Database) -> None:
        self.db = db
        self._prov = ProvenanceRepo(db)
        self._cache_lock = threading.Lock()
        self._rows_cache: OrderedDict[tuple[int, ImageScope], np.ndarray] = OrderedDict()

    def _build_query_parts(
        self, scope: ImageScope
    ) -> tuple[bool, list[str], list[str], list[Any]]:
        """Return `(definitely_empty, ctes, where_clauses, params)` for `images AS i`."""
        if scope.folders is not None and len(scope.folders) == 0:
            return True, [], [], []
        if scope.snapshot_ids is not None and len(scope.snapshot_ids) == 0:
            return True, [], [], []
        if scope.restrict_point_ids is not None and len(scope.restrict_point_ids) == 0:
            return True, [], [], []

        ctes: list[str] = []
        where: list[str] = ["i.deleted = 0"]
        params: list[Any] = []

        # 1. Folders (direct or recursive subtree)
        if scope.folders is not None:
            folder_json = json.dumps(list(dict.fromkeys(scope.folders)), ensure_ascii=False)
            if scope.folder_recursive:
                ctes.append(
                    "_scope_folders(id) AS ("
                    "SELECT id FROM folders WHERE folder_key IN (SELECT value FROM json_each(?)) "
                    "UNION ALL "
                    "SELECT f.id FROM folders AS f JOIN _scope_folders AS sf ON f.parent_id = sf.id"
                    ")"
                )
                params.append(folder_json)
                where.append("i.folder_id IN (SELECT id FROM _scope_folders)")
            else:
                where.append(
                    "i.folder_id IN (SELECT id FROM folders "
                    "WHERE folder_key IN (SELECT value FROM json_each(?)))"
                )
                params.append(folder_json)

        # 2. Time range
        if scope.time_range is not None:
            after_ns, before_ns = scope.time_range
            if after_ns is not None:
                where.append("i.mtime_ns >= ?")
                params.append(int(after_ns))
            if before_ns is not None:
                where.append("i.mtime_ns <= ?")
                params.append(int(before_ns))

        # 3. Collection membership
        if scope.collection_id is not None:
            where.append(
                "i.id IN (SELECT image_id FROM collection_items "
                "WHERE collection_id = ? AND image_id IS NOT NULL)"
            )
            params.append(scope.collection_id)

        # 4. Authors (supports "none" pseudo-author + union across keys)
        if scope.author_keys:
            resolved = self._prov.resolve_author_keys_for_query(scope.author_keys)
            include_none = "none" in resolved
            canon_keys = [k for k in resolved if k != "none"]
            author_clauses: list[str] = []
            if canon_keys:
                if len(canon_keys) == 1:
                    author_clauses.append(
                        "i.id IN (SELECT s.image_id FROM image_source AS s "
                        "WHERE s.author_key = ? AND s.image_id IS NOT NULL)"
                    )
                    params.append(canon_keys[0])
                else:
                    author_clauses.append(
                        "i.id IN (SELECT s.image_id FROM image_source AS s "
                        "WHERE s.image_id IS NOT NULL "
                        "AND s.author_key IN (SELECT value FROM json_each(?)))"
                    )
                    params.append(json.dumps(canon_keys, ensure_ascii=False))
            if include_none:
                author_clauses.append(
                    "i.id IN (SELECT s.image_id FROM image_source AS s "
                    "WHERE s.image_id IS NOT NULL "
                    "AND (s.author_key IS NULL OR s.author_key = ''))"
                )
            if author_clauses:
                where.append("(" + " OR ".join(author_clauses) + ")")
            else:
                return True, [], [], []

        # 5. WD image tags (tags_all / tags_none)
        for raw_tag in scope.tags_all:
            norm = normalize_tag(raw_tag)
            if not norm:
                continue
            where.append(
                "i.id IN (SELECT it.image_id FROM image_tags AS it "
                "JOIN tags AS t ON t.id = it.tag_id WHERE t.name = ?)"
            )
            params.append(norm)

        if scope.tags_none:
            neg = sorted({normalize_tag(t) for t in scope.tags_none if normalize_tag(t)})
            if neg:
                where.append(
                    "i.id NOT IN (SELECT it.image_id FROM image_tags AS it "
                    "JOIN tags AS t ON t.id = it.tag_id "
                    "WHERE t.name IN (SELECT value FROM json_each(?)))"
                )
                params.append(json.dumps(neg, ensure_ascii=False))

        # 6. Provenance / source tags (AND with alias expansion; ANY; substring tag text)
        for raw_tag in scope.source_tags_all:
            aliases = sorted(
                {a.strip().casefold() for a in expand_tag_aliases(raw_tag) if a.strip()}
            )
            if not aliases:
                continue
            where.append(
                "i.id IN (SELECT st.image_id FROM source_tags AS st "
                "WHERE st.image_id IS NOT NULL "
                "AND st.tag_norm IN (SELECT value FROM json_each(?)))"
            )
            params.append(json.dumps(aliases, ensure_ascii=False))

        if scope.source_tags_any:
            any_norm = sorted(
                {str(t).strip().casefold() for t in scope.source_tags_any if str(t).strip()}
            )
            if any_norm:
                where.append(
                    "i.id IN (SELECT st.image_id FROM source_tags AS st "
                    "WHERE st.image_id IS NOT NULL "
                    "AND st.tag_norm IN (SELECT value FROM json_each(?)))"
                )
                params.append(json.dumps(any_norm, ensure_ascii=False))

        for term in scope.source_tag_text:
            clean = str(term).strip().casefold()
            if not clean:
                continue
            where.append(
                "i.id IN (SELECT st.image_id FROM source_tags AS st "
                "WHERE st.image_id IS NOT NULL AND st.tag_norm LIKE ?)"
            )
            params.append(f"%{clean}%")

        # 7. Boolean tag expression
        expr_text = str(scope.tag_expression or "").strip()
        if expr_text:
            node = parse_tag_expression(expr_text)
            expr_sql = _compile_tag_expr(node, params)
            where.append(expr_sql)

        # 8. Provenance title terms
        for term in scope.title_terms:
            for word in re.split(r"\s+", str(term).strip()):
                if not word:
                    continue
                where.append(
                    "i.id IN (SELECT s.image_id FROM image_source AS s "
                    "WHERE s.image_id IS NOT NULL "
                    "AND instr(LOWER(COALESCE(s.title, '')), ?) > 0)"
                )
                params.append(word.casefold())

        # 9. Snapshot / explicit point ID restrictions
        if scope.snapshot_ids is not None:
            where.append("i.point_id IN (SELECT value FROM json_each(?))")
            params.append(json.dumps(list(dict.fromkeys(scope.snapshot_ids)), ensure_ascii=False))

        if scope.restrict_point_ids is not None:
            where.append("i.point_id IN (SELECT value FROM json_each(?))")
            params.append(
                json.dumps(list(dict.fromkeys(scope.restrict_point_ids)), ensure_ascii=False)
            )

        return False, ctes, where, params

    def rows(self, scope: ImageScope) -> np.ndarray | None:
        """Return matching live `images.id` array, or `None` when completely unfiltered."""
        if scope.is_unfiltered:
            return None
        cache_key = (self.db.write_generation, scope)
        with self._cache_lock:
            cached = self._rows_cache.get(cache_key)
            if cached is not None:
                self._rows_cache.move_to_end(cache_key)
                return cached
        empty, ctes, where, params = self._build_query_parts(scope)
        if empty:
            arr = np.empty(0, dtype=np.int64)
        else:
            with_prefix = f"WITH {', '.join(ctes)} " if ctes else ""
            sql = (
                f"{with_prefix}SELECT i.id FROM images AS i "
                f"WHERE {' AND '.join(where)} ORDER BY i.id"
            )
            cur = self.db.read_conn().cursor()
            cur.row_factory = None
            cur.execute(sql, tuple(params))
            arr = np.fromiter((r[0] for r in cur), dtype=np.int64)
        with self._cache_lock:
            self._rows_cache[cache_key] = arr
            while len(self._rows_cache) > 32:
                self._rows_cache.popitem(last=False)
        return arr

    def resolve_rows(self, scope: ImageScope) -> np.ndarray | None:
        """Alias of `rows(scope)`."""
        return self.rows(scope)

    def point_ids(self, scope: ImageScope) -> list[str] | None:
        """Return matching live `point_id`s, or `None` when completely unfiltered."""
        if scope.is_unfiltered:
            return None
        empty, ctes, where, params = self._build_query_parts(scope)
        if empty:
            return []
        with_prefix = f"WITH {', '.join(ctes)} " if ctes else ""
        sql = (
            f"{with_prefix}SELECT i.point_id FROM images AS i "
            f"WHERE {' AND '.join(where)} ORDER BY i.mtime_ns DESC, i.id ASC"
        )
        cur = self.db.read_conn().cursor()
        cur.row_factory = None
        cur.execute(sql, tuple(params))
        return [str(r[0]) for r in cur]

    def count(self, scope: ImageScope) -> int:
        empty, ctes, where, params = self._build_query_parts(scope)
        if empty:
            return 0
        with_prefix = f"WITH {', '.join(ctes)} " if ctes else ""
        sql = f"{with_prefix}SELECT COUNT(*) FROM images AS i WHERE {' AND '.join(where)}"
        row = self.db.read_conn().execute(sql, tuple(params)).fetchone()
        return int(row[0]) if row else 0

    def page_rows(
        self,
        scope: ImageScope,
        *,
        sort: str = "mtime_desc",
        limit: int = 40,
        offset: int = 0,
        seed: int | None = None,
    ) -> tuple[list[int], int, bool, str, int | None]:
        """Return `(row_ids, total, has_more, normalized_sort, seed)`."""
        limit = max(1, int(limit))
        offset = max(0, int(offset))
        order = "mtime_desc" if sort in {"mtime", ""} else sort
        if order not in {"mtime_desc", "mtime_asc", "name_asc", "name_desc", "name", "random"}:
            order = "mtime_desc"
        if order == "name":
            order = "name_asc"

        empty, ctes, where, params = self._build_query_parts(scope)
        if empty:
            used_seed = seed if order == "random" else None
            if order == "random" and used_seed is None:
                used_seed = random.SystemRandom().randrange(1 << 53)
            return [], 0, False, order, used_seed

        with_prefix = f"WITH {', '.join(ctes)} " if ctes else ""
        where_sql = " AND ".join(where)
        cur = self.db.read_conn().cursor()
        cur.row_factory = None

        if order in {"mtime_desc", "mtime_asc"}:
            has_selective_filter = bool(
                scope.tags_all
                or scope.tags_none
                or scope.source_tags_all
                or scope.source_tags_any
                or scope.source_tag_text
                or scope.tag_expression
                or scope.author_keys
                or scope.title_terms
                or scope.collection_id
                or scope.snapshot_ids is not None
                or scope.restrict_point_ids is not None
            )
            if has_selective_filter:
                # Unary + prevents SQLite from scanning all 45k rows of idx_images_mtime
                # when a selective tag/author/collection index can drive the query in <5ms.
                order_clause = (
                    "+i.mtime_ns DESC, +i.id ASC"
                    if order == "mtime_desc"
                    else "+i.mtime_ns ASC, +i.id DESC"
                )
                all_rows = cur.execute(
                    f"{with_prefix}SELECT i.id FROM images AS i WHERE {where_sql} "
                    f"ORDER BY {order_clause}",
                    tuple(params),
                ).fetchall()
                total = len(all_rows)
                if offset >= total:
                    return [], total, False, order, None
                page_ids = [int(r[0]) for r in all_rows[offset : offset + limit]]
                return page_ids, total, offset + len(page_ids) < total, order, None

            order_clause = (
                "i.mtime_ns DESC, i.id ASC"
                if order == "mtime_desc"
                else "i.mtime_ns ASC, i.id DESC"
            )
            total_row = cur.execute(
                f"{with_prefix}SELECT COUNT(*) FROM images AS i WHERE {where_sql}",
                tuple(params),
            ).fetchone()
            total = int(total_row[0]) if total_row else 0
            if offset >= total:
                return [], total, False, order, None
            rows = cur.execute(
                f"{with_prefix}SELECT i.id FROM images AS i WHERE {where_sql} "
                f"ORDER BY {order_clause} LIMIT ? OFFSET ?",
                (*params, limit, offset),
            ).fetchall()
            page_ids = [int(r[0]) for r in rows]
            return page_ids, total, offset + len(page_ids) < total, order, None

        if order == "random":
            used_seed = seed if seed is not None else random.SystemRandom().randrange(1 << 53)
            cur.execute(
                f"{with_prefix}SELECT i.id FROM images AS i WHERE {where_sql}",
                tuple(params),
            )
            all_ids = np.fromiter((r[0] for r in cur), dtype=np.int64)
            total = int(all_ids.size)
            if offset >= total:
                return [], total, False, order, used_seed
            shuffled = _deterministic_random_order(all_ids, used_seed)
            page_ids = shuffled[offset : offset + limit].tolist()
            return page_ids, total, offset + len(page_ids) < total, order, used_seed

        # Natural filename sorting is precomputed into `sort_name` so SQLite can
        # order large scopes in C instead of sorting every row in Python.
        direction = "DESC" if order == "name_desc" else "ASC"
        total_row = cur.execute(
            f"{with_prefix}SELECT COUNT(*) FROM images AS i WHERE {where_sql}",
            tuple(params),
        ).fetchone()
        total = int(total_row[0]) if total_row else 0
        if offset >= total:
            return [], total, False, order, None
        rows = cur.execute(
            f"{with_prefix}SELECT i.id FROM images AS i WHERE {where_sql} "
            f"ORDER BY i.sort_name {direction}, i.point_id {direction} LIMIT ? OFFSET ?",
            (*params, limit, offset),
        ).fetchall()
        page_ids = [int(r[0]) for r in rows]
        return page_ids, total, offset + len(page_ids) < total, order, None
