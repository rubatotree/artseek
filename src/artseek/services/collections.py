"""SQLite-backed collections service preserving mixed artist+image items and undo/redo."""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any

from ..collections import (
    COLLECTION_SORT_KEYS,
    DEFAULT_COLLECTION_SORT,
    PAYLOAD_SORT_KEYS,
    SORT_ADDED_DESC,
    SORT_MANUAL,
    Collection,
    artist_item_key,
    is_artist_item,
)
from ..db.repos.collections import CollectionRepo
from ..db.repos.images import ImageRepo
from ..db.repos.provenance import ProvenanceRepo
from .common import ServiceError, attach_provenance, gallery_result

_MAX_ORDERED_COLLECTION_IDS = 10_240


def _artist_collection_result(
    key: str,
    summary: dict[str, Any],
    latest_mtime: int,
    cover: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "id": f"artist:{key}",
        "kind": "artist",
        "artist_key": key,
        "name": summary.get("name") or summary.get("handle") or key,
        "handle": summary.get("handle") or "",
        "members": summary.get("members") or [],
        "image_count": int(summary.get("image_count") or 0),
        "latest_mtime": int(latest_mtime or 0),
        "width": int((cover or {}).get("width") or 0),
        "height": int((cover or {}).get("height") or 0),
        "cover": cover,
    }


def _collection_sort_key(payload: dict[str, Any], sort: str) -> int | str:
    if sort.startswith("mtime"):
        return int(payload.get("mtime_ns") or 0)
    return str(payload.get("name") or "").casefold()


class CollectionService:
    """Manages user collections, mixed artist+image items, temporary vs saved sorts, and history."""

    def __init__(
        self,
        collections: CollectionRepo,
        images: ImageRepo,
        prov: ProvenanceRepo,
    ) -> None:
        self.collections = collections
        self.images = images
        self.prov = prov

    def _require_collection(self, collection_id: str) -> Collection:
        col = self.collections.get(collection_id)
        if col is None:
            raise ServiceError(404, "收藏夹不存在")
        return col

    def _current_history(self) -> dict[str, Any]:
        _, history = self.collections.snapshot()
        return history

    def _collections_payload(
        self,
        collections: Sequence[Collection],
        history: dict[str, Any],
        image_id: str | None = None,
    ) -> dict[str, Any]:
        summaries: list[dict[str, Any]] = []
        for col in collections:
            summary = col.summary()
            if image_id:
                summary["contains"] = image_id in col.item_ids
            summaries.append(summary)
        return {"collections": summaries, "history": history}

    def _item_name(self, item_id: str) -> str:
        if is_artist_item(item_id):
            key = artist_item_key(item_id)
            summary = self.prov.author_summary(key)
            if summary is not None:
                return str(summary.get("name") or summary.get("handle") or key)
            return ""
        payload = self.images.get_by_point_id(item_id)
        return str(payload.get("name") or "") if payload else ""

    def _retrieve_payloads_by_point(
        self, point_ids: Sequence[str]
    ) -> dict[str, dict[str, Any]]:
        if not point_ids:
            return {}
        row_map = self.images.row_ids_for_points(point_ids)
        if not row_map:
            return {}
        by_row = self.images.get_payloads_by_ids(list(row_map.values()))
        return {
            pid: by_row[rid]
            for pid, rid in row_map.items()
            if rid in by_row
        }

    # ---------------------------------------------------------------- CRUD & History
    def list_collections(self, image_id: str | None = None) -> dict[str, Any]:
        collections, history = self.collections.snapshot()
        return self._collections_payload(collections, history, image_id)

    def create_collection(self, name: str) -> dict[str, Any]:
        col = self.collections.create(name)
        return {"collection": col.summary()}

    def reorder_collections(self, collection_ids: list[str]) -> dict[str, Any]:
        collections = self.collections.reorder(collection_ids)
        if collections is None:
            raise ServiceError(409, "收藏夹已更新，请刷新后重试")
        _, history = self.collections.snapshot()
        return self._collections_payload(collections, history)

    def undo(self) -> dict[str, Any]:
        result = self.collections.undo()
        if result is None:
            raise ServiceError(409, "没有可撤销的操作")
        return result.payload()

    def redo(self) -> dict[str, Any]:
        result = self.collections.redo()
        if result is None:
            raise ServiceError(409, "没有可重做的操作")
        return result.payload()

    def _ordered_live_ids(self, collection: Collection, sort: str) -> list[str]:
        if any(is_artist_item(item_id) for item_id in collection.item_ids):
            results, _, _ = self._mixed_collection_page(
                collection, sort, 0, _MAX_ORDERED_COLLECTION_IDS, "", None, None
            )
            return [str(item["id"]) for item in results]
        if sort == SORT_ADDED_DESC:
            return list(reversed(collection.item_ids))
        if sort == SORT_MANUAL:
            return list(collection.ordered_item_ids())
        if sort not in PAYLOAD_SORT_KEYS:
            return list(collection.item_ids)
        payloads = self._retrieve_payloads_by_point(collection.item_ids)
        entries: list[tuple[Any, str]] = []
        for pid in collection.item_ids:
            payload = payloads.get(pid)
            if payload is not None:
                entries.append((_collection_sort_key(payload, sort), pid))
        entries.sort(key=lambda pair: pair[0], reverse=sort.endswith("_desc"))
        return [pid for _, pid in entries[:_MAX_ORDERED_COLLECTION_IDS]]

    def update_collection(
        self,
        collection_id: str,
        *,
        name: str | None = None,
        sort: str | None = None,
        seed_sort: str | None = None,
    ) -> dict[str, Any]:
        current = self._require_collection(collection_id)
        manual_order: tuple[str, ...] | None = None
        if sort == SORT_MANUAL and current.sort != SORT_MANUAL and current.item_ids:
            seed = seed_sort or current.sort
            if seed not in COLLECTION_SORT_KEYS:
                seed = DEFAULT_COLLECTION_SORT
            live_ids = self._ordered_live_ids(current, seed)
            seen = set(live_ids)
            manual_order = tuple(live_ids) + tuple(
                pid for pid in current.item_ids if pid not in seen
            )
        col = self.collections.update(
            collection_id,
            name=name,
            sort=sort,
            manual_order=manual_order,
        )
        if col is None:
            raise ServiceError(404, "收藏夹不存在")
        return {
            "collection": col.summary(),
            "history": self._current_history(),
        }

    def delete_collection(self, collection_id: str, *, discard: bool = False) -> dict[str, Any]:
        col = self.collections.delete(collection_id, record_history=not discard)
        if col is None:
            raise ServiceError(404, "收藏夹不存在")
        return {
            "deleted": True,
            "collection": col.summary(),
            "history": self._current_history(),
        }

    def add_item(self, collection_id: str, image_id: str) -> dict[str, Any]:
        self._require_collection(collection_id)
        if is_artist_item(image_id):
            if self.prov.author_summary(artist_item_key(image_id)) is None:
                raise ServiceError(404, "画师不存在，无法收藏")
        elif self.images.row_id_for_point(image_id) is None:
            raise ServiceError(404, "图片不在索引中，无法收藏")
        col, added = self.collections.add_item(
            collection_id,
            image_id,
            item_name=self._item_name(image_id),
        )
        if col is None:
            raise ServiceError(404, "收藏夹不存在")
        return {
            "collection": col.summary(),
            "added": added,
            "history": self._current_history(),
        }

    def remove_item(self, collection_id: str, point_id: str) -> dict[str, Any]:
        col, removed = self.collections.remove_item(
            collection_id,
            point_id,
            item_name=self._item_name(point_id),
        )
        if col is None:
            raise ServiceError(404, "收藏夹不存在")
        return {
            "collection": col.summary(),
            "removed": removed,
            "history": self._current_history(),
        }

    def reorder_item(
        self,
        collection_id: str,
        image_id: str,
        *,
        before_id: str | None = None,
        after_id: str | None = None,
    ) -> dict[str, Any]:
        self._require_collection(collection_id)
        col, moved = self.collections.reorder_item(
            collection_id,
            image_id,
            before_id=before_id,
            after_id=after_id,
        )
        if col is None:
            raise ServiceError(409, "收藏夹内容已更新，请刷新后重试")
        return {
            "collection": col.summary(),
            "moved": moved,
            "history": self._current_history(),
        }

    # ---------------------------------------------------------------- Images & Mixed Pages
    def _mixed_collection_page(
        self,
        collection: Collection,
        method: str,
        start: int,
        page_size: int,
        needle: str,
        allowed_author_key: str | None,
        seed: int | None,
    ) -> tuple[list[dict[str, Any]], int, bool]:
        item_ids = list(collection.item_ids)
        image_ids = [item_id for item_id in item_ids if not is_artist_item(item_id)]
        artist_keys = [artist_item_key(item_id) for item_id in item_ids if is_artist_item(item_id)]

        image_payloads = self._retrieve_payloads_by_point(image_ids)
        summaries = {
            key: summary
            for key in artist_keys
            if (summary := self.prov.author_summary(key)) is not None
        }
        latest = self.prov.author_latest_mtimes(list(summaries))
        image_ids_by_artist = self.prov.point_ids_for_authors(list(summaries))
        explicit_covers = self.prov.artist_cover_ids(list(summaries))
        cover_ids = [
            explicit_covers.get(key) or ids[0]
            for key, ids in image_ids_by_artist.items()
            if explicit_covers.get(key) or ids
        ]
        cover_payloads = self._retrieve_payloads_by_point(list(dict.fromkeys(cover_ids)))
        allowed_image_ids = (
            set(self.prov.point_ids_for_author(allowed_author_key))
            if allowed_author_key
            else None
        )

        entries: list[tuple[int | str, str, dict[str, Any] | None]] = []
        for item_id in item_ids:
            if is_artist_item(item_id):
                key = artist_item_key(item_id)
                summary = summaries.get(key)
                if summary is None:
                    continue
                if allowed_author_key and key != allowed_author_key:
                    continue
                searchable = "\n".join(
                    [
                        str(summary.get("name") or ""),
                        str(summary.get("handle") or ""),
                        key,
                        *(
                            str(m.get("handle") or m.get("platform_id") or "")
                            for m in summary.get("members") or []
                        ),
                    ]
                ).casefold()
                if needle and needle not in searchable:
                    continue
                cover_id = explicit_covers.get(key) or next(
                    (
                        pid
                        for pid in image_ids_by_artist.get(key, [])
                        if pid in cover_payloads
                    ),
                    "",
                )
                cover_payload = cover_payloads.get(cover_id) if cover_id else None
                cover = (
                    {
                        "id": cover_id,
                        "name": cover_payload.get("name"),
                        "width": cover_payload.get("width"),
                        "height": cover_payload.get("height"),
                    }
                    if isinstance(cover_payload, dict)
                    else None
                )
                res = _artist_collection_result(key, summary, latest.get(key, 0), cover)
                sort_key: int | str = (
                    res["latest_mtime"]
                    if method.startswith("mtime")
                    else str(res["name"]).casefold()
                )
                entries.append((sort_key, item_id, res))
            else:
                payload = image_payloads.get(item_id)
                if payload is None:
                    continue
                if allowed_image_ids is not None and item_id not in allowed_image_ids:
                    continue
                if needle:
                    tags = payload.get("tags") or []
                    searchable = "\n".join(
                        [
                            str(payload.get("name") or ""),
                            str(payload.get("path") or ""),
                            str(payload.get("caption") or ""),
                            *(str(tag) for tag in tags),
                        ]
                    ).casefold()
                    if needle not in searchable:
                        continue
                entries.append((_collection_sort_key(payload, method), item_id, None))

        if method == "random":
            used_seed = seed if seed is not None else random.SystemRandom().randrange(1 << 53)
            random.Random(used_seed).shuffle(entries)
        elif method == SORT_MANUAL:
            rank = {item_id: idx for idx, item_id in enumerate(collection.ordered_item_ids())}
            entries.sort(key=lambda entry: rank.get(entry[1], len(rank)))
        elif method == SORT_ADDED_DESC:
            entries.reverse()
        elif method in PAYLOAD_SORT_KEYS:
            entries.sort(key=lambda entry: entry[0], reverse=method.endswith("_desc"))

        page_entries = entries[start : start + page_size]
        page_image_ids = [
            item_id for _key, item_id, result in page_entries if result is None
        ]
        image_results = {
            str(item["id"]): item
            for item in attach_provenance(
                self.prov,
                [
                    gallery_result(item_id, image_payloads[item_id])
                    for item_id in page_image_ids
                    if item_id in image_payloads
                ],
            )
        }
        page = [
            result if result is not None else image_results.get(item_id)
            for _key, item_id, result in page_entries
        ]
        return (
            [item for item in page if isinstance(item, dict)],
            len(entries),
            start + len(page_entries) < len(entries),
        )

    def collection_images(
        self,
        collection_id: str,
        *,
        query: str = "",
        sort: str = "",
        limit: int = 40,
        offset: int = 0,
        author_key: str | None = None,
        seed: int | None = None,
    ) -> dict[str, Any]:
        collection = self._require_collection(collection_id)
        random_requested = sort == "random"
        method = (
            "random"
            if random_requested
            else (sort if sort in COLLECTION_SORT_KEYS else collection.sort)
        )
        if method not in COLLECTION_SORT_KEYS and method != "random":
            method = DEFAULT_COLLECTION_SORT
        needle = query.strip().casefold()
        page_size = max(1, min(int(limit), 200))
        start = max(0, int(offset))

        if any(is_artist_item(item_id) for item_id in collection.item_ids):
            used_seed = seed
            if method == "random" and used_seed is None:
                used_seed = random.SystemRandom().randrange(1 << 53)
            mixed_results, mixed_total, mixed_has_more = self._mixed_collection_page(
                collection, method, start, page_size, needle, author_key, used_seed
            )
            return {
                "collection": collection.summary(),
                "query": query.strip(),
                "sort": method,
                "seed": used_seed if method == "random" else None,
                "saved_sort": collection.sort,
                "offset": start,
                "limit": page_size,
                "total": mixed_total,
                "total_exact": True,
                "has_more": mixed_has_more,
                "results": mixed_results,
            }

        allowed_ids = (
            set(self.prov.point_ids_for_author(author_key)) if author_key else None
        )
        if not needle and allowed_ids is None and method != "random":
            return self._unfiltered_sql_collection_page(
                collection, method, start, page_size, query.strip()
            )

        payloads = self._retrieve_payloads_by_point(collection.item_ids)
        filtered: list[tuple[str, dict[str, Any]]] = []
        for pid in collection.item_ids:
            payload = payloads.get(pid)
            if payload is None:
                continue
            if allowed_ids is not None and pid not in allowed_ids:
                continue
            if needle:
                tags = payload.get("tags") or []
                searchable = "\n".join(
                    [
                        str(payload.get("name") or ""),
                        str(payload.get("path") or ""),
                        str(payload.get("caption") or ""),
                        *(str(tag) for tag in tags),
                    ]
                ).casefold()
                if needle not in searchable:
                    continue
            filtered.append((pid, payload))

        used_seed: int | None = None
        if method == "random":
            used_seed = seed if seed is not None else random.SystemRandom().randrange(1 << 53)
            random.Random(used_seed).shuffle(filtered)
        elif method in PAYLOAD_SORT_KEYS:
            filtered.sort(
                key=lambda item: _collection_sort_key(item[1], method),
                reverse=method.endswith("_desc"),
            )
        elif method == SORT_ADDED_DESC:
            filtered.reverse()
        elif method == SORT_MANUAL:
            rank = {pid: idx for idx, pid in enumerate(collection.ordered_item_ids())}
            filtered.sort(key=lambda item: rank.get(item[0], len(rank)))

        total = len(filtered)
        page = filtered[start : start + page_size]
        results = attach_provenance(
            self.prov,
            [gallery_result(pid, payload) for pid, payload in page],
        )
        return {
            "collection": collection.summary(),
            "query": query.strip(),
            "sort": method,
            "seed": used_seed if method == "random" else None,
            "saved_sort": collection.sort,
            "offset": start,
            "limit": page_size,
            "total": total,
            "total_exact": True,
            "has_more": start + len(page) < total,
            "results": results,
        }

    def _unfiltered_sql_collection_page(
        self,
        collection: Collection,
        method: str,
        start: int,
        page_size: int,
        query: str,
    ) -> dict[str, Any]:
        conn = self.images.db.read_conn()
        total_row = conn.execute(
            """
            SELECT COUNT(*)
            FROM collection_items AS ci
            JOIN images AS i ON i.id = ci.image_id AND i.deleted = 0
            WHERE ci.collection_id = ?
            """,
            (collection.id,),
        ).fetchone()
        total = int(total_row[0]) if total_row else 0
        if start >= total or total == 0:
            return {
                "collection": collection.summary(),
                "query": query,
                "sort": method,
                "seed": None,
                "saved_sort": collection.sort,
                "offset": start,
                "limit": page_size,
                "total": total,
                "total_exact": True,
                "has_more": False,
                "results": [],
            }

        order_clause = {
            "added_asc": "ci.added_position ASC",
            "added_desc": "ci.added_position DESC",
            "manual": "COALESCE(ci.manual_position, 1000000000 + ci.added_position) ASC",
            "mtime_asc": "i.mtime_ns ASC, ci.added_position ASC",
            "mtime_desc": "i.mtime_ns DESC, ci.added_position ASC",
            "name_asc": "LOWER(i.name) ASC, ci.added_position ASC",
            "name_desc": "LOWER(i.name) DESC, ci.added_position ASC",
        }.get(method, "ci.added_position ASC")

        rows = conn.execute(
            f"""
            SELECT i.id, i.point_id
            FROM collection_items AS ci
            JOIN images AS i ON i.id = ci.image_id AND i.deleted = 0
            WHERE ci.collection_id = ?
            ORDER BY {order_clause}
            LIMIT ? OFFSET ?
            """,
            (collection.id, page_size, start),
        ).fetchall()
        row_ids = [int(r["id"]) for r in rows]
        payloads = self.images.get_payloads_by_ids(row_ids)
        results = attach_provenance(
            self.prov,
            [
                gallery_result(str(payloads[rid]["point_id"]), payloads[rid])
                for rid in row_ids
                if rid in payloads
            ],
        )
        return {
            "collection": collection.summary(),
            "query": query,
            "sort": method,
            "seed": None,
            "saved_sort": collection.sort,
            "offset": start,
            "limit": page_size,
            "total": total,
            "total_exact": True,
            "has_more": start + len(results) < total,
            "results": results,
        }
