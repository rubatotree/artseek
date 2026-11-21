"""Artist directory, scoped waterfall gallery, author page, and identity management."""

from __future__ import annotations

import io
import random
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from PIL import Image, ImageOps, UnidentifiedImageError

from ..collections import ARTIST_ITEM_PREFIX
from ..core.scope import ImageScope
from ..core.tag_query import TagQueryError, parse_tag_expression
from ..db.repos.collections import CollectionRepo
from ..db.repos.images import ImageRepo
from ..db.repos.provenance import ProvenanceRepo
from ..db.scope_resolver import ScopeResolver
from ..library import day_end_ns, day_start_ns, natural_key
from ..provenance.store import (
    UNASSIGNED_AUTHOR_KEY,
    normalize_social_link,
    normalize_xiaohongshu_user_id,
)
from ..search_cache import SearchSnapshotStore, is_snapshot_id
from .common import (
    ServiceError,
    attach_provenance,
    fast_folder_key,
    gallery_result,
    normalize_ratings,
    resolve_author_token,
)

ARTIST_SORT_KEYS = (
    "name_asc",
    "name_desc",
    "latest_asc",
    "latest_desc",
    "count_asc",
    "count_desc",
)
_UNASSIGNED_MERGE_TOKENS = frozenset(
    {UNASSIGNED_AUTHOR_KEY, "无画师", "未标注", "作者未标注", "无作者"}
)


def _normalize_image_sort(raw_sort: str) -> str:
    mapped = {
        "name": "name_asc",
        "count": "mtime_desc",
        "count_asc": "mtime_desc",
        "mtime": "mtime_desc",
    }.get(raw_sort, raw_sort)
    if mapped not in {
        "mtime_desc",
        "mtime_asc",
        "name_asc",
        "name_desc",
        "random",
        "manual",
        "added_asc",
        "added_desc",
    }:
        return "mtime_desc"
    return mapped


def _cover_rating_rank(rating: object) -> int:
    """Prefer confirmed safe -> R15 -> R18 -> unrated when auto-picking a cover."""
    value = str(rating or "").strip().casefold()
    if value == "safe":
        return 0
    if value == "r15":
        return 1
    if value == "r18":
        return 2
    return 3


class ArtistService:
    """Author listing, paginated artist waterfall, identity linking, and merge service."""

    def __init__(
        self,
        images: ImageRepo,
        prov: ProvenanceRepo,
        resolver: ScopeResolver,
        collections: CollectionRepo,
        snapshots: SearchSnapshotStore,
        *,
        assets_dir: Path | None = None,
    ) -> None:
        self.images = images
        self.prov = prov
        self.resolver = resolver
        self.collections = collections
        self.snapshots = snapshots
        self.assets_dir = assets_dir
        self._cache_lock = threading.RLock()
        self._author_scope_cache: OrderedDict[tuple[Any, ...], tuple[float, dict[str, int]]] = (
            OrderedDict()
        )
        self._artist_catalog_cache: OrderedDict[
            tuple[Any, ...],
            tuple[float, list[dict[str, Any]], dict[str, list[dict[str, Any]]]],
        ] = OrderedDict()

    def invalidate_cache(self) -> None:
        with self._cache_lock:
            self._author_scope_cache.clear()
            self._artist_catalog_cache.clear()

    # ---------------------------------------------------------------- helpers
    def _scope_point_ids(
        self,
        folder_keys: Sequence[str] | None,
        mtime_after: str | None,
        mtime_before: str | None,
    ) -> list[str] | None:
        if folder_keys is None:
            return None
        if not folder_keys:
            return []
        after_ns = day_start_ns(mtime_after)
        before_ns = day_end_ns(mtime_before)
        time_range = (
            (after_ns, before_ns) if (after_ns is not None or before_ns is not None) else None
        )
        return self.resolver.point_ids(
            ImageScope(
                folders=tuple(fast_folder_key(k) for k in folder_keys if k),
                folder_recursive=False,
                time_range=time_range,
            )
        )

    def _hydrate_point_ids(self, point_ids: Sequence[str]) -> list[dict[str, Any]]:
        if not point_ids:
            return []
        row_map = self.images.row_ids_for_points(point_ids)
        unique_rows = list(dict.fromkeys(row_map.values()))
        payloads = self.images.get_payloads_by_ids(unique_rows)
        items: list[dict[str, Any]] = []
        for pid in point_ids:
            rid = row_map.get(pid)
            if rid is None:
                continue
            payload = payloads.get(rid)
            if payload is not None:
                items.append(gallery_result(pid, payload))
        return attach_provenance(self.prov, items)

    def _batch_live_meta(
        self, point_ids: Sequence[str]
    ) -> dict[str, tuple[int, str, int, str, int]]:
        """Return `{point_id: (row_id, name, mtime_ns, rating, pixels)}` for live images."""
        unique = [pid for pid in dict.fromkeys(point_ids) if pid]
        if not unique:
            return {}
        conn = self.images.db.read_conn()
        found: dict[str, tuple[int, str, int, str, int]] = {}
        for start in range(0, len(unique), 400):
            chunk = unique[start : start + 400]
            marks = ",".join("?" for _ in chunk)
            rows = conn.execute(
                f"SELECT id, point_id, name, mtime_ns, rating, width, height FROM images "
                f"WHERE deleted = 0 AND point_id IN ({marks})",
                tuple(chunk),
            ).fetchall()
            for r in rows:
                pixels = int(r["width"] or 0) * int(r["height"] or 0)
                found[str(r["point_id"])] = (
                    int(r["id"]),
                    str(r["name"]),
                    int(r["mtime_ns"] or 0),
                    str(r["rating"] or ""),
                    pixels,
                )
        return found

    def _sort_point_ids_in_mem(
        self,
        point_ids: Sequence[str],
        meta: dict[str, tuple[int, ...]],
        sort: str,
        seed: int | None,
    ) -> tuple[list[str], str, int | None]:
        scoped = [pid for pid in point_ids if pid in meta]
        order = "mtime_desc" if sort == "mtime" else sort
        if order not in {"mtime_desc", "mtime_asc", "name_asc", "name_desc", "random"}:
            order = "mtime_desc"
        if order == "random":
            used_seed = seed if seed is not None else random.SystemRandom().randrange(1 << 53)
            random.Random(used_seed).shuffle(scoped)
            return scoped, order, used_seed
        if order in {"name_asc", "name_desc"}:
            scoped.sort(
                key=lambda pid: (natural_key(meta[pid][1] or pid), pid),
                reverse=order == "name_desc",
            )
            return scoped, order, None
        scoped.sort(
            key=lambda pid: (meta[pid][2], pid),
            reverse=order == "mtime_desc",
        )
        return scoped, order, None

    # ---------------------------------------------------------------- author queries
    def list_authors(
        self,
        *,
        q: str = "",
        limit: int = 40,
        offset: int = 0,
        folder_keys: Sequence[str] | None = None,
        mtime_after: str | None = None,
        mtime_before: str | None = None,
    ) -> dict[str, Any]:
        page_size = max(1, min(int(limit), 500))
        start = max(0, int(offset))
        scope_ids = self._scope_point_ids(folder_keys, mtime_after, mtime_before)
        counts: dict[str, int] | None = None
        if scope_ids is not None:
            cache_key = (
                tuple(sorted(folder_keys or ())),
                mtime_after or "",
                mtime_before or "",
            )
            now = time.monotonic()
            with self._cache_lock:
                cached = self._author_scope_cache.get(cache_key)
                if cached is not None and now - cached[0] <= 20.0:
                    counts = cached[1]
            if counts is None:
                counts = self.prov.author_counts_for_points(scope_ids)
                with self._cache_lock:
                    self._author_scope_cache[cache_key] = (now, counts)
                    while len(self._author_scope_cache) > 16:
                        self._author_scope_cache.popitem(last=False)
        unassigned_count = (
            int(counts.get(UNASSIGNED_AUTHOR_KEY, 0))
            if counts is not None
            else self.prov.unassigned_count()
        )
        if counts is not None:
            counts = {k: v for k, v in counts.items() if k != UNASSIGNED_AUTHOR_KEY}
        authors = self.prov.list_authors(
            q,
            limit=page_size + 1,
            offset=start,
            image_counts=counts,
        )
        return {
            "authors": authors[:page_size],
            "has_more": len(authors) > page_size,
            "offset": start,
            "unassigned_count": unassigned_count,
        }

    def get_author(
        self,
        author_key: str,
        *,
        folder_keys: Sequence[str] | None = None,
        mtime_after: str | None = None,
        mtime_before: str | None = None,
    ) -> dict[str, Any]:
        summary = self.prov.author_summary(author_key)
        if summary is None:
            raise ServiceError(404, "作者不存在")
        ids = self.prov.point_ids_for_author(author_key)
        scope_ids = self._scope_point_ids(folder_keys, mtime_after, mtime_before)
        if scope_ids is not None:
            allowed = set(scope_ids)
            ids = [pid for pid in ids if pid in allowed]
            summary = dict(summary)
            summary["image_count"] = len(ids)
        return {"author": summary, "point_ids": ids, "total": len(ids)}

    def author_images(
        self,
        author_key: str,
        *,
        limit: int = 40,
        offset: int = 0,
        sort: str = "mtime_desc",
        seed: int | None = None,
        ratings: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        summary = self.prov.author_summary(author_key)
        if summary is None:
            raise ServiceError(404, "作者不存在")
        page_size = max(1, min(int(limit), 200))
        start = max(0, min(int(offset), 50_000))
        canonical = (
            UNASSIGNED_AUTHOR_KEY
            if str(author_key).casefold() == UNASSIGNED_AUTHOR_KEY
            else self.prov.canonical_author_key(author_key)
        )
        row_ids, total, has_more, order, used_seed = self.resolver.page_rows(
            ImageScope(
                author_keys=(canonical,),
                ratings=normalize_ratings(ratings),
            ),
            sort=sort,
            limit=page_size,
            offset=start,
            seed=seed,
        )
        payloads = self.images.get_payloads_by_ids(row_ids)
        items = [
            gallery_result(str(payloads[rid]["point_id"]), payloads[rid])
            for rid in row_ids
            if rid in payloads
        ]
        results = attach_provenance(self.prov, items)
        return {
            "author": summary,
            "offset": start,
            "limit": page_size,
            "sort": order,
            "seed": used_seed,
            "total": total,
            "has_more": has_more,
            "results": results,
        }

    # ---------------------------------------------------------------- artists gallery & images
    def _resolve_artist_scope_ids(
        self,
        *,
        snapshot_id: str | None = None,
        folder_keys: Sequence[str] | None = None,
        mtime_after: str | None = None,
        mtime_before: str | None = None,
        collection_id: str | None = None,
        author_keys: Sequence[str] = (),
        title_query: str = "",
        text_tags: Sequence[str] = (),
        tag_expression: str = "",
        ratings: Sequence[str] = (),
    ) -> set[str] | None:
        scope: set[str] | None = None

        def intersect(values: Sequence[str]) -> None:
            nonlocal scope
            cleaned = {str(v) for v in values if v}
            scope = cleaned if scope is None else scope & cleaned

        if snapshot_id:
            if not is_snapshot_id(snapshot_id):
                raise ServiceError(400, "检索快照编号不合法")
            snapshot = self.snapshots.get(snapshot_id)
            if snapshot is None:
                raise ServiceError(404, "检索快照不存在或已过期")
            intersect([str(item.get("id") or "") for item in snapshot.get("results") or []])

        if collection_id:
            collection = self.collections.get(collection_id)
            if collection is None:
                raise ServiceError(404, "收藏夹不存在")
            intersect(
                item_id
                for item_id in collection.item_ids
                if not item_id.startswith(ARTIST_ITEM_PREFIX)
            )

        expr_text = str(tag_expression or "").strip()
        if expr_text:
            try:
                parse_tag_expression(expr_text)
            except TagQueryError as exc:
                raise ServiceError(400, str(exc)) from exc

        resolved_authors = tuple(
            resolve_author_token(self.prov, value)
            for value in author_keys
            if str(value or "").strip()
        )
        after_ns = day_start_ns(mtime_after)
        before_ns = day_end_ns(mtime_before)
        time_range = (
            (after_ns, before_ns) if (after_ns is not None or before_ns is not None) else None
        )

        img_scope = ImageScope(
            folders=(
                tuple(fast_folder_key(k) for k in folder_keys if k)
                if folder_keys is not None
                else None
            ),
            folder_recursive=False,
            time_range=time_range,
            author_keys=resolved_authors,
            source_tags_all=tuple(str(t).strip() for t in text_tags if str(t).strip()),
            tag_expression=expr_text,
            title_terms=(title_query.strip(),) if title_query.strip() else (),
            ratings=normalize_ratings(ratings),
        )
        if not img_scope.is_unfiltered:
            pids = self.resolver.point_ids(img_scope)
            if pids is not None:
                intersect(pids)
        return scope

    def _collection_order_ids(
        self, collection_id: str | None, image_sort: str
    ) -> tuple[str, ...] | None:
        if not collection_id or image_sort not in {"manual", "added_asc", "added_desc"}:
            return None
        collection = self.collections.get(collection_id)
        if collection is None:
            raise ServiceError(404, "收藏夹不存在")
        if image_sort == "manual":
            return collection.ordered_item_ids()
        if image_sort == "added_desc":
            return tuple(reversed(collection.item_ids))
        return collection.item_ids

    def _artist_catalog_for(
        self,
        cache_key: tuple[Any, ...],
        scope: set[str] | None,
        sort: str,
    ) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
        now = time.monotonic()
        with self._cache_lock:
            cached = self._artist_catalog_cache.get(cache_key)
            if cached is not None and now - cached[0] <= 120.0:
                self._artist_catalog_cache.move_to_end(cache_key)
                return cached[1], cached[2]

        entries = self.prov.artist_entries(list(scope) if scope is not None else None)
        orders = {sort: self.prov.sort_artist_entries(entries, sort, include_unassigned=True)}
        with self._cache_lock:
            self._artist_catalog_cache[cache_key] = (now, entries, orders)
            while len(self._artist_catalog_cache) > 8:
                self._artist_catalog_cache.popitem(last=False)

        def warm_orders() -> None:
            for other in ARTIST_SORT_KEYS:
                if other == sort:
                    continue
                ordered = self.prov.sort_artist_entries(entries, other, include_unassigned=True)
                with self._cache_lock:
                    cur = self._artist_catalog_cache.get(cache_key)
                    if cur is None or cur[1] is not entries:
                        return
                    cur[2].setdefault(other, ordered)

        threading.Thread(
            target=warm_orders, daemon=True, name="artseek-artist-sort-warm"
        ).start()
        return entries, orders

    @staticmethod
    def _public_artist_entry(entry: dict[str, Any]) -> dict[str, Any]:
        clean = dict(entry)
        clean.pop("_unassigned", None)
        clean.pop("_search_values", None)
        clean.pop("_search_text", None)
        return clean

    def artists_gallery(
        self,
        *,
        q: str = "",
        sort: str = "latest_desc",
        include_unassigned: bool = False,
        offset: int = 0,
        limit: int = 30,
        image_limit: int = 12,
        image_sort: str = "mtime_desc",
        image_seed: int | None = None,
        snapshot_id: str | None = None,
        folder_keys: Sequence[str] | None = None,
        mtime_after: str | None = None,
        mtime_before: str | None = None,
        collection_id: str | None = None,
        author_keys: Sequence[str] = (),
        title_query: str = "",
        text_tags: Sequence[str] = (),
        tag_expression: str = "",
        ratings: Sequence[str] = (),
    ) -> dict[str, Any]:
        clean_ratings = normalize_ratings(ratings)
        scope = self._resolve_artist_scope_ids(
            snapshot_id=snapshot_id,
            folder_keys=folder_keys,
            mtime_after=mtime_after,
            mtime_before=mtime_before,
            collection_id=collection_id,
            author_keys=author_keys,
            title_query=title_query,
            text_tags=text_tags,
            tag_expression=tag_expression,
            ratings=clean_ratings,
        )
        norm_sort = sort if sort in ARTIST_SORT_KEYS else "latest_desc"
        cache_key = (
            tuple(sorted(folder_keys)) if folder_keys is not None else None,
            mtime_after or "",
            mtime_before or "",
            tuple(sorted(author_keys)),
            title_query,
            tuple(text_tags),
            tag_expression,
            collection_id,
            snapshot_id,
            clean_ratings,
        )
        entries, orders = self._artist_catalog_for(cache_key, scope, norm_sort)
        if q.strip():
            ranked = self.prov.search_artist_entries(
                entries,
                q,
                norm_sort,
                include_unassigned=include_unassigned,
            )
        else:
            ranked = orders.get(norm_sort) or self.prov.sort_artist_entries(
                entries, norm_sort, include_unassigned=True
            )
            if not include_unassigned:
                ranked = [item for item in ranked if item["key"] != UNASSIGNED_AUTHOR_KEY]

        total = len(ranked)
        start = max(0, int(offset))
        page_entries = [
            dict(self._public_artist_entry(item))
            for item in ranked[start : start + max(1, int(limit))]
        ]

        norm_image_sort = _normalize_image_sort(image_sort)
        order_ids = self._collection_order_ids(collection_id, norm_image_sort)
        base_seed = image_seed
        if norm_image_sort == "random" and base_seed is None:
            base_seed = random.SystemRandom().randrange(1 << 53)

        entry_keys = [str(item["key"]) for item in page_entries]
        id_map = self.prov.point_ids_for_authors(entry_keys) if entry_keys else {}
        explicit_covers = self.prov.artist_cover_entries(entry_keys) if entry_keys else {}

        candidate_ids: list[str] = []
        for key in entry_keys:
            candidate_ids.extend(id_map.get(key, ()))
            explicit = explicit_covers.get(key)
            if explicit and explicit.get("source") != "upload" and explicit.get("identity"):
                candidate_ids.append(explicit["identity"])
        live_meta = self._batch_live_meta(candidate_ids)

        rank_map = (
            {pid: idx for idx, pid in enumerate(order_ids)}
            if order_ids is not None
            else None
        )
        previews: dict[str, list[str]] = {}
        totals: dict[str, int] = {}
        seeds: dict[str, int | None] = {}
        cover_plans: dict[str, tuple[dict[str, Any] | None, str, bool, bool]] = {}
        fetch_ids: list[str] = []

        for entry in page_entries:
            key = str(entry["key"])
            raw_ids = id_map.get(key, [])
            all_live_ids = [pid for pid in raw_ids if pid in live_meta]
            scoped_ids = (
                [pid for pid in all_live_ids if pid in scope]
                if scope is not None
                else list(all_live_ids)
            )
            if rank_map is not None:
                ids = list(scoped_ids)
                ids.sort(key=lambda pid: rank_map.get(pid, len(rank_map)))
                used_seed = None
            else:
                row_seed = base_seed
                if norm_image_sort == "random" and base_seed is not None:
                    row_seed = random.Random(f"{base_seed}:{key}").randrange(1 << 53)
                ids, _, used_seed = self._sort_point_ids_in_mem(
                    scoped_ids, live_meta, norm_image_sort, row_seed
                )
            previews[key] = ids[: max(1, int(image_limit))]
            totals[key] = len(ids)
            seeds[key] = used_seed
            fetch_ids.extend(previews[key])

            explicit = explicit_covers.get(key)
            if explicit and explicit.get("source") == "upload" and explicit.get("profile_url"):
                cover_plans[key] = (
                    {
                        "id": explicit["identity"],
                        "url": explicit["profile_url"],
                        "uploaded": True,
                        "blurred": False,
                        "name": None,
                        "width": 0,
                        "height": 0,
                    },
                    "",
                    True,
                    False,
                )
            elif (
                explicit
                and explicit.get("source") != "upload"
                and explicit.get("identity") in live_meta
            ):
                cid = explicit["identity"]
                blurred = scope is not None and cid not in scope
                fetch_ids.append(cid)
                cover_plans[key] = (None, cid, True, blurred)
            else:
                pool = scoped_ids or all_live_ids
                blurred = (not scoped_ids) and bool(all_live_ids)
                cid = (
                    min(
                        pool,
                        key=lambda pid: (
                            _cover_rating_rank(live_meta[pid][3]),
                            -live_meta[pid][4],
                            pid,
                        ),
                    )
                    if pool
                    else ""
                )
                if cid:
                    fetch_ids.append(cid)
                cover_plans[key] = (None, cid, False, blurred)

        hydrated = self._hydrate_point_ids(list(dict.fromkeys(fetch_ids)))
        by_id = {str(item["id"]): item for item in hydrated}
        for entry in page_entries:
            key = str(entry["key"])
            images = [by_id[pid] for pid in previews[key] if pid in by_id]
            entry["images"] = images
            entry["images_has_more"] = totals[key] > len(previews[key])
            entry["images_offset"] = len(images)
            entry["images_total"] = totals[key]
            entry["image_seed"] = seeds[key]
            uploaded_cover, cover_id, is_explicit, is_blurred = cover_plans[key]
            if uploaded_cover is not None:
                entry["cover"] = uploaded_cover
                entry["cover_explicit"] = True
            else:
                cover_item = by_id.get(cover_id) if cover_id else None
                entry["cover"] = (
                    {
                        "id": cover_id,
                        "name": cover_item.get("name"),
                        "width": cover_item.get("width"),
                        "height": cover_item.get("height"),
                        "blurred": is_blurred,
                    }
                    if cover_item is not None
                    else None
                )
                entry["cover_explicit"] = is_explicit and cover_item is not None
            members = entry.get("members") or []
            entry["pixiv_ids"] = [
                str(m.get("platform_id") or "")
                for m in members
                if m.get("platform") == "pixiv"
            ]
            entry["x_handles"] = [
                str(m.get("handle") or "")
                for m in members
                if m.get("platform") == "twitter"
            ]

        return {
            "authors": page_entries,
            "total": total,
            "has_more": start + len(page_entries) < total,
            "image_sort": norm_image_sort,
            "image_seed": base_seed,
        }

    def scoped_artist_images(
        self,
        author_key: str,
        *,
        sort: str = "mtime_desc",
        seed: int | None = None,
        offset: int = 0,
        limit: int = 24,
        snapshot_id: str | None = None,
        folder_keys: Sequence[str] | None = None,
        mtime_after: str | None = None,
        mtime_before: str | None = None,
        collection_id: str | None = None,
        author_keys: Sequence[str] = (),
        title_query: str = "",
        text_tags: Sequence[str] = (),
        tag_expression: str = "",
        ratings: Sequence[str] = (),
    ) -> dict[str, Any]:
        summary = self.prov.author_summary(author_key)
        if summary is None:
            raise ServiceError(404, "作者不存在")
        scope = self._resolve_artist_scope_ids(
            snapshot_id=snapshot_id,
            folder_keys=folder_keys,
            mtime_after=mtime_after,
            mtime_before=mtime_before,
            collection_id=collection_id,
            author_keys=author_keys,
            title_query=title_query,
            text_tags=text_tags,
            tag_expression=tag_expression,
            ratings=ratings,
        )
        norm_sort = _normalize_image_sort(sort)
        order_ids = self._collection_order_ids(collection_id, norm_sort)
        ids = self.prov.point_ids_for_author(author_key)
        if scope is not None:
            ids = [pid for pid in ids if pid in scope]
        live_meta = self._batch_live_meta(ids)
        if order_ids is not None:
            rank = {pid: idx for idx, pid in enumerate(order_ids)}
            ids = [pid for pid in ids if pid in live_meta]
            ids.sort(key=lambda pid: rank.get(pid, len(rank)))
            order, used_seed = norm_sort, None
        else:
            ids, order, used_seed = self._sort_point_ids_in_mem(ids, live_meta, norm_sort, seed)
        total = len(ids)
        page_size = max(1, min(int(limit), 200))
        start = max(0, int(offset))
        page_ids = ids[start : start + page_size]
        results = self._hydrate_point_ids(page_ids)
        return {
            "author": summary,
            "offset": start,
            "limit": page_size,
            "sort": order,
            "seed": used_seed,
            "total": total,
            "has_more": start + len(page_ids) < total,
            "results": results,
        }

    # ---------------------------------------------------------------- author mutations
    def add_identity(
        self,
        author_key: str,
        *,
        kind: str,
        value: str,
        label: str = "",
    ) -> dict[str, Any]:
        if author_key.casefold() == UNASSIGNED_AUTHOR_KEY:
            raise ServiceError(400, "无画师不能添加身份")
        norm_kind = kind.strip().casefold()
        try:
            if norm_kind == "link":
                if not normalize_social_link(value):
                    raise ValueError("请输入有效的 http(s) 链接")
                self.prov.add_link(author_key, value, label=label)
            elif norm_kind in {"alias", "name_alias"}:
                self.prov.add_alias(author_key, value)
            elif norm_kind in {"pixiv", "pixiv_id"}:
                self.prov.attach_identity(author_key, "pixiv", value)
            elif norm_kind in {"twitter", "x", "x_id"}:
                self.prov.attach_identity(author_key, "twitter", value)
            elif norm_kind in {"xiaohongshu", "xhs", "rednote"}:
                if not normalize_xiaohongshu_user_id(value):
                    raise ValueError("小红书主页链接或 UID 无效")
                self.prov.attach_identity(author_key, "xiaohongshu", value)
            else:
                raise ServiceError(400, "不支持的身份类型")
        except ValueError as exc:
            status = 409 if "已属于" in str(exc) else 400
            raise ServiceError(status, str(exc)) from exc
        self.invalidate_cache()
        summary = self.prov.author_summary(author_key)
        if summary is None:
            raise ServiceError(404, "作者不存在")
        return {"author": summary}

    def rename_author(self, author_key: str, name: str) -> dict[str, Any]:
        try:
            self.prov.set_author_name(author_key, name)
        except ValueError as exc:
            raise ServiceError(400, str(exc)) from exc
        self.invalidate_cache()
        summary = self.prov.author_summary(author_key)
        if summary is None:
            raise ServiceError(404, "作者不存在")
        return {"author": summary}

    def remove_identity(self, author_key: str, kind: str, value: str) -> dict[str, Any]:
        try:
            self.prov.remove_identity(author_key, kind, value)
        except ValueError as exc:
            raise ServiceError(400, str(exc)) from exc
        self.invalidate_cache()
        summary = self.prov.author_summary(author_key)
        if summary is None:
            raise ServiceError(404, "作者不存在")
        return {"author": summary}

    def set_member_preferences(
        self, author_key: str, members: Sequence[dict[str, Any]]
    ) -> dict[str, Any]:
        try:
            self.prov.set_member_preferences(author_key, members)
        except ValueError as exc:
            raise ServiceError(400, str(exc)) from exc
        self.invalidate_cache()
        summary = self.prov.author_summary(author_key)
        if summary is None:
            raise ServiceError(404, "作者不存在")
        return {"author": summary}

    def _require_assets_dir(self) -> Path:
        if self.assets_dir is None:
            raise ServiceError(500, "未配置画师资源目录")
        self.assets_dir.mkdir(parents=True, exist_ok=True)
        return self.assets_dir

    def _delete_artist_assets(self, canonical_key: str, kind: Literal["avatar", "cover"]) -> None:
        if self.assets_dir is None or not self.assets_dir.exists():
            return
        for path in self.assets_dir.glob(f"{kind}-{canonical_key}-*.webp"):
            path.unlink(missing_ok=True)

    def _save_artist_asset(
        self,
        author_key: str,
        kind: Literal["avatar", "cover"],
        raw_bytes: bytes,
        max_dim: int,
    ) -> tuple[str, str]:
        if not raw_bytes:
            raise ServiceError(400, "请选择要上传的图片文件")
        if len(raw_bytes) > 20 * 1024 * 1024:
            raise ServiceError(400, "图片文件过大（上限 20 MB）")
        summary = self.prov.author_summary(author_key)
        if summary is None or str(author_key).casefold() == UNASSIGNED_AUTHOR_KEY:
            raise ServiceError(
                404 if summary is None else 400,
                "作者不存在" if summary is None else "无画师不能上传图片",
            )
        canonical = str(summary["key"])
        try:
            with Image.open(io.BytesIO(raw_bytes)) as img:
                prepared = ImageOps.exif_transpose(img)
                if prepared.mode not in ("RGB", "RGBA"):
                    prepared = prepared.convert("RGBA" if "A" in prepared.getbands() else "RGB")
                prepared.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
                assets_dir = self._require_assets_dir()
                token = uuid.uuid4().hex[:10]
                dest = assets_dir / f"{kind}-{canonical}-{token}.webp"
                tmp = assets_dir / f".{kind}-{canonical}-{token}.webp.tmp"
                prepared.save(tmp, format="WEBP", quality=90, method=4)
                self._delete_artist_assets(canonical, kind)
                tmp.replace(dest)
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise ServiceError(400, "无法读取该图片文件，请换一张有效的图片") from exc
        return canonical, token

    def resolve_asset_path(self, author_key: str, kind: Literal["avatar", "cover"]) -> Path:
        summary = self.prov.author_summary(author_key)
        if summary is None:
            raise ServiceError(404, "作者不存在")
        canonical = str(summary["key"])
        assets_dir = self.assets_dir
        if assets_dir is None or not assets_dir.exists():
            raise ServiceError(404, "资源不存在")
        matches = sorted(assets_dir.glob(f"{kind}-{canonical}-*.webp"), reverse=True)
        if not matches:
            raise ServiceError(404, "资源不存在")
        return matches[0]

    def set_cover(self, author_key: str, point_id: str) -> dict[str, Any]:
        if not str(point_id or "").strip():
            return self.clear_cover(author_key)
        try:
            canonical = self.prov.set_artist_cover(author_key, point_id)
        except ValueError as exc:
            raise ServiceError(400, str(exc)) from exc
        self._delete_artist_assets(canonical, "cover")
        self.invalidate_cache()
        summary = self.prov.author_summary(author_key)
        if summary is None:
            raise ServiceError(404, "作者不存在")
        return {"author": summary}

    def clear_cover(self, author_key: str) -> dict[str, Any]:
        try:
            canonical = self.prov.clear_artist_cover(author_key)
        except ValueError as exc:
            raise ServiceError(400, str(exc)) from exc
        self._delete_artist_assets(canonical, "cover")
        self.invalidate_cache()
        summary = self.prov.author_summary(author_key)
        if summary is None:
            raise ServiceError(404, "作者不存在")
        return {"author": summary}

    def upload_cover(self, author_key: str, raw_bytes: bytes) -> dict[str, Any]:
        canonical, token = self._save_artist_asset(author_key, "cover", raw_bytes, 1600)
        url = f"/api/authors/{canonical}/cover-image?v={token}"
        try:
            self.prov.set_uploaded_artist_cover(canonical, f"upload:{token}", url)
        except ValueError as exc:
            raise ServiceError(400, str(exc)) from exc
        self.invalidate_cache()
        summary = self.prov.author_summary(canonical)
        if summary is None:
            raise ServiceError(404, "作者不存在")
        return {"author": summary}

    def upload_avatar(self, author_key: str, raw_bytes: bytes) -> dict[str, Any]:
        canonical, token = self._save_artist_asset(author_key, "avatar", raw_bytes, 512)
        url = f"/api/authors/{canonical}/avatar?v={token}"
        try:
            self.prov.set_artist_avatar(canonical, url)
        except ValueError as exc:
            raise ServiceError(400, str(exc)) from exc
        self.invalidate_cache()
        summary = self.prov.author_summary(canonical)
        if summary is None:
            raise ServiceError(404, "作者不存在")
        return {"author": summary}

    def clear_avatar(self, author_key: str) -> dict[str, Any]:
        try:
            canonical = self.prov.set_artist_avatar(author_key, None)
        except ValueError as exc:
            raise ServiceError(400, str(exc)) from exc
        self._delete_artist_assets(canonical, "avatar")
        self.invalidate_cache()
        summary = self.prov.author_summary(canonical)
        if summary is None:
            raise ServiceError(404, "作者不存在")
        return {"author": summary}

    def merge_author(self, author_key: str, source: str) -> dict[str, Any]:
        target = author_key.strip().lstrip("@")
        if target.casefold() == UNASSIGNED_AUTHOR_KEY:
            raise ServiceError(400, "无画师不能合并")
        target_summary = self.prov.author_summary(target)
        if target_summary is None:
            raise ServiceError(404, "作者不存在")
        token = source.strip().lstrip("@")
        if not token:
            raise ServiceError(400, "请输入要合并的 UID、Pixiv ID 或 X handle")
        if token.casefold() in _UNASSIGNED_MERGE_TOKENS:
            raise ServiceError(400, "无画师不能合并")

        source_key = ""
        if token.isdigit():
            summary = self.prov.author_summary(token)
            if summary is not None:
                source_key = str(summary["key"])
        for platform in ("pixiv", "twitter", "xiaohongshu"):
            if source_key:
                break
            owner = self.prov.identity_owner(platform, token)
            if owner:
                source_key = owner
        if not source_key:
            matches = self.prov.list_authors(token, limit=5)
            grouped: dict[str, list[dict[str, Any]]] = {}
            for item in matches:
                root = self.prov.canonical_author_key(str(item["key"]))
                grouped.setdefault(root, []).append(item)
            if len(grouped) == 1:
                source_key = next(iter(grouped))
            elif len(grouped) > 1:
                raise ServiceError(
                    409,
                    {
                        "message": "作者不唯一，请选择完整 UID",
                        "author_candidates": [
                            {
                                "key": root,
                                "name": members[0].get("name")
                                or members[0].get("handle")
                                or root,
                            }
                            for root, members in grouped.items()
                        ],
                    },
                )
        if not source_key:
            raise ServiceError(404, "没有找到要合并的作者")
        source_summary = self.prov.author_summary(source_key)
        if source_summary is None:
            raise ServiceError(404, "没有找到要合并的作者")
        if str(source_summary["key"]) == str(target_summary["key"]):
            return {"author": target_summary, "merged_from": None}
        self.prov.merge_authors(source_key, target)
        self.invalidate_cache()
        return {
            "author": self.prov.author_summary(target),
            "merged_from": source_summary,
        }
