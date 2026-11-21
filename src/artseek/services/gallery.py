"""Gallery, folder tree, scope browsing, random browse, tags, and item services."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..core.scope import ImageScope
from ..core.tag_dictionary import search_tags, tag_info
from ..db.repos.images import ImageRepo
from ..db.repos.provenance import ProvenanceRepo
from ..db.scope_resolver import ScopeResolver
from ..library import (
    DEFAULT_SORT,
    SORT_KEYS,
    SORT_OPTIONS,
    day_end_ns,
    day_start_ns,
    normalize_day,
    ns_to_day,
)
from .common import (
    FolderTreeIndex,
    ServiceError,
    attach_provenance,
    build_folder_tree_from_db,
    fast_folder_key,
    gallery_result,
    normalize_folder_sort,
    normalize_ratings,
    reconcile_scope_with_tree,
    resolve_author_token,
    resolve_folder_keys_from_db,
)


def _hydrating_gallery_rows(
    images: ImageRepo,
    prov: ProvenanceRepo,
    row_ids: Sequence[int],
) -> list[dict[str, Any]]:
    if not row_ids:
        return []
    payloads = images.get_payloads_by_ids(row_ids)
    items = [
        gallery_result(str(payloads[rid]["point_id"]), payloads[rid])
        for rid in row_ids
        if rid in payloads
    ]
    return attach_provenance(prov, items)


class GalleryService:
    """High-speed SQL-backed gallery, library tree, scope, random, and tag queries."""

    def __init__(
        self,
        images: ImageRepo,
        prov: ProvenanceRepo,
        resolver: ScopeResolver,
    ) -> None:
        self.images = images
        self.prov = prov
        self.resolver = resolver
        self._tag_lock = threading.RLock()
        self._image_tags_cache_count: int | None = None
        self._image_tags_cache_items: list[dict[str, Any]] = []

    def invalidate_tag_cache(self) -> None:
        with self._tag_lock:
            self._image_tags_cache_count = None
            self._image_tags_cache_items = []

    # ---------------------------------------------------------------- library tree & scope
    def _library_payload_from_tree(
        self,
        tree: FolderTreeIndex,
        scope_state: dict[str, Any],
        new_keys: Sequence[str],
    ) -> dict[str, Any]:
        after_ns = day_start_ns(scope_state.get("mtime_after"))
        before_ns = day_end_ns(scope_state.get("mtime_before"))
        low, high = self.images.time_bounds()
        return {
            "sort": scope_state["sort"],
            "sort_options": [{"id": key, "label": label} for key, label in SORT_OPTIONS],
            "selected": list(scope_state["selected"]),
            "new_keys": list(new_keys),
            "mtime_after": scope_state.get("mtime_after"),
            "mtime_before": scope_state.get("mtime_before"),
            "mtime_min": ns_to_day(low),
            "mtime_max": ns_to_day(high),
            "image_count": tree.image_count,
            "selected_image_count": tree.selected_image_count(
                self.resolver,
                scope_state["selected"],
                after_ns=after_ns,
                before_ns=before_ns,
            ),
            "roots": tree.to_roots(scope_state["sort"]),
        }

    def library_tree(self) -> dict[str, Any]:
        tree = build_folder_tree_from_db(self.images)
        scope_state, new_keys = reconcile_scope_with_tree(self.images, tree)
        return self._library_payload_from_tree(tree, scope_state, new_keys)

    def update_scope(
        self,
        *,
        sort: str | None = None,
        selected: Sequence[str] = (),
        mtime_after: str | None = None,
        mtime_before: str | None = None,
    ) -> dict[str, Any]:
        tree = build_folder_tree_from_db(self.images)
        prev, _ = reconcile_scope_with_tree(self.images, tree)
        current = tree.all_keys
        norm_sort = normalize_folder_sort(sort)
        cleaned_selected = [
            fk
            for k in selected
            if k and (fk := fast_folder_key(k, current)) in current
        ]
        after = normalize_day(mtime_after)
        before = normalize_day(mtime_before)
        if after and before and after > before:
            after, before = before, after
        known = sorted(set(prev.get("known", ())) | current)
        new_keys = sorted(current - set(prev.get("known", ()))) if prev.get("known") else []
        self.images.save_scope_state(
            sort=norm_sort,
            selected=cleaned_selected,
            known=known,
            mtime_after=after,
            mtime_before=before,
        )
        next_state = {
            "sort": norm_sort,
            "selected": cleaned_selected,
            "known": known,
            "mtime_after": after,
            "mtime_before": before,
        }
        return self._library_payload_from_tree(tree, next_state, new_keys)

    # ---------------------------------------------------------------- folder & scope browsing
    def _folder_dfs_order(self, tree: FolderTreeIndex, root_key: str, sort: str) -> list[str]:
        for root in tree.sorted_root_nodes(sort):
            found = root.find(root_key)
            if found is not None:
                return list(found.iter_keys())
        return [root_key]

    def _folder_images_count_ordered(
        self,
        tree: FolderTreeIndex,
        folder_keys_in_order: Sequence[str],
        *,
        author_keys: tuple[str, ...],
        ratings: tuple[str, ...] = (),
        start: int,
        page_size: int,
    ) -> tuple[list[int], int]:
        """Slice DFS-by-folder-count without scanning every folder's rows into RAM."""
        if not author_keys and not ratings:
            counts_by_key = {k: tree.direct_count_by_key.get(k, 0) for k in folder_keys_in_order}
        else:
            counts_by_key = {}
            for k in folder_keys_in_order:
                counts_by_key[k] = self.resolver.count(
                    ImageScope(
                        folders=(k,),
                        folder_recursive=False,
                        author_keys=author_keys,
                        ratings=ratings,
                    )
                )
        total = sum(counts_by_key.values())
        if start >= total or total == 0:
            return [], total

        chosen_rows: list[int] = []
        cursor = 0
        end = start + page_size
        for fkey in folder_keys_in_order:
            cnt = counts_by_key.get(fkey, 0)
            if cnt <= 0:
                continue
            folder_end = cursor + cnt
            if folder_end <= start:
                cursor = folder_end
                continue
            if cursor >= end:
                break
            local_offset = max(0, start - cursor)
            local_limit = min(cnt - local_offset, end - (cursor + local_offset))
            rows, _, _, _, _ = self.resolver.page_rows(
                ImageScope(
                    folders=(fkey,),
                    folder_recursive=False,
                    author_keys=author_keys,
                    ratings=ratings,
                ),
                sort="name_asc",
                limit=max(1, local_limit),
                offset=local_offset,
            )
            chosen_rows.extend(rows)
            cursor = folder_end
        return chosen_rows, total

    def folder_images(
        self,
        folder_key: str,
        *,
        sort: str = DEFAULT_SORT,
        limit: int = 40,
        offset: int = 0,
        author_key: str | None = None,
        seed: int | None = None,
        ratings: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        tree = build_folder_tree_from_db(self.images)
        key = fast_folder_key(folder_key, tree.all_keys)
        node = tree.nodes_by_key.get(key)
        if node is None:
            raise ServiceError(404, "该目录不在索引中")
        method = (
            sort
            if sort in SORT_KEYS or sort in {"random", "indexed"}
            else DEFAULT_SORT
        )
        page_size = max(1, min(int(limit), 200))
        start = max(0, int(offset))
        authors: tuple[str, ...] = ()
        if author_key:
            authors = (resolve_author_token(self.prov, author_key),)
        clean_ratings = normalize_ratings(ratings)

        used_seed: int | None = None
        if method in {"count", "count_asc"}:
            ordered_folders = self._folder_dfs_order(tree, key, method)
            row_ids, total = self._folder_images_count_ordered(
                tree,
                ordered_folders,
                author_keys=authors,
                ratings=clean_ratings,
                start=start,
                page_size=page_size,
            )
        else:
            sql_sort = {
                "name": "name_asc",
                "mtime": "mtime_desc",
                "indexed": "indexed_desc",
            }.get(method, method)
            row_ids, total, _has_more, _order, used_seed = self.resolver.page_rows(
                ImageScope(
                    folders=(key,),
                    folder_recursive=True,
                    author_keys=authors,
                    ratings=clean_ratings,
                ),
                sort=sql_sort,
                limit=page_size,
                offset=start,
                seed=seed,
            )
        results = _hydrating_gallery_rows(self.images, self.prov, row_ids)
        return {
            "folder_key": key,
            "folder_name": node.name,
            "folder_path": node.path,
            "sort": method,
            "seed": used_seed if method == "random" else None,
            "offset": start,
            "limit": page_size,
            "total": total,
            "has_more": start + len(results) < total,
            "results": results,
        }

    def scope_images(
        self,
        folder_keys: Sequence[str] | None,
        *,
        sort: str = DEFAULT_SORT,
        seed: int | None = None,
        limit: int = 40,
        offset: int = 0,
        author_key: str | None = None,
        mtime_after: str | None = None,
        mtime_before: str | None = None,
        ratings: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        keys = resolve_folder_keys_from_db(self.images, folder_keys)
        if not keys:
            raise ServiceError(400, "请先勾选要查看的目录")
        method = (
            sort
            if sort in SORT_KEYS or sort in {"random", "indexed"}
            else DEFAULT_SORT
        )
        sql_sort = (
            "name_asc"
            if method in {"name", "count", "count_asc"}
            else {
                "mtime": "mtime_desc",
                "indexed": "indexed_desc",
            }.get(method, method)
        )
        page_size = max(1, min(int(limit), 200))
        start = max(0, min(int(offset), 50_000))
        after_ns = day_start_ns(mtime_after)
        before_ns = day_end_ns(mtime_before)
        time_range = (
            (after_ns, before_ns) if (after_ns is not None or before_ns is not None) else None
        )
        authors = (resolve_author_token(self.prov, author_key),) if author_key else ()
        row_ids, total, _has_more, _order, used_seed = self.resolver.page_rows(
            ImageScope(
                folders=tuple(keys),
                folder_recursive=False,
                time_range=time_range,
                author_keys=authors,
                ratings=normalize_ratings(ratings),
            ),
            sort=sql_sort,
            limit=page_size,
            offset=start,
            seed=seed,
        )
        results = _hydrating_gallery_rows(self.images, self.prov, row_ids)
        return {
            "folder_keys": keys,
            "folder_count": len(keys),
            "sort": method,
            "seed": used_seed if method == "random" else None,
            "offset": start,
            "limit": page_size,
            "total": total,
            "has_more": start + len(results) < total,
            "results": results,
        }

    def random_browse(
        self,
        *,
        folder_keys: Sequence[str] | None = None,
        seed: int | None = None,
        limit: int = 40,
        offset: int = 0,
        author_key: str | None = None,
        mtime_after: str | None = None,
        mtime_before: str | None = None,
        ratings: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        keys = resolve_folder_keys_from_db(self.images, folder_keys)
        if not keys:
            raise ServiceError(400, "请先勾选要随机浏览的文件夹")
        page_size = max(1, min(int(limit), 200))
        start = max(0, int(offset))
        after_ns = day_start_ns(mtime_after)
        before_ns = day_end_ns(mtime_before)
        time_range = (
            (after_ns, before_ns) if (after_ns is not None or before_ns is not None) else None
        )
        authors = (resolve_author_token(self.prov, author_key),) if author_key else ()
        row_ids, total, has_more, _order, used_seed = self.resolver.page_rows(
            ImageScope(
                folders=tuple(keys),
                folder_recursive=False,
                time_range=time_range,
                author_keys=authors,
                ratings=normalize_ratings(ratings),
            ),
            sort="random",
            limit=page_size,
            offset=start,
            seed=seed,
        )
        results = _hydrating_gallery_rows(self.images, self.prov, row_ids)
        return {
            "seed": used_seed,
            "offset": min(start, total),
            "limit": page_size,
            "total": total,
            "has_more": has_more,
            "results": results,
        }

    # ---------------------------------------------------------------- tags & item
    def _all_image_tag_counts(self) -> list[dict[str, Any]]:
        count = self.images.count()
        with self._tag_lock:
            if self._image_tags_cache_count == count:
                return self._image_tags_cache_items
            items = self.images.image_tag_counts("", limit=100_000, offset=0)
            self._image_tags_cache_count = count
            self._image_tags_cache_items = items
            return items

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

    def _group_source_tag_items(
        self,
        items: Sequence[dict[str, Any]],
        query: str = "",
        scope_ids: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        groups: dict[str, dict[str, Any]] = {}
        for item in items:
            tag = str(item.get("tag") or "").strip()
            if not tag:
                continue
            info = tag_info(tag)
            canonical = info.tag if info is not None else tag
            group = groups.setdefault(
                canonical,
                {"tag": canonical, "count": 0, "display_zh": "", "aliases": set()},
            )
            group["count"] += int(item.get("count") or 0)
            if info is not None:
                group["display_zh"] = group["display_zh"] or info.zh
                group["aliases"].update(info.aliases)
            group["aliases"].add(tag)

        result: list[dict[str, Any]] = []
        for group in groups.values():
            aliases = sorted(
                {alias for alias in group["aliases"] if alias and alias != group["tag"]},
                key=lambda value: (len(value), value.casefold()),
            )
            result.append(
                {
                    "tag": group["tag"],
                    "count": group["count"],
                    "display_zh": group["display_zh"],
                    "aliases": aliases,
                }
            )
        if not result and str(query or "").strip():
            seen = {item["tag"].casefold() for item in result}
            scope = set(scope_ids) if scope_ids is not None else None
            for entry in search_tags(query, limit=20):
                if entry.tag.casefold() in seen:
                    continue
                points = set(self.prov.point_ids_for_tags([entry.tag]))
                if scope is not None:
                    points &= scope
                result.append(
                    {
                        "tag": entry.tag,
                        "count": len(points),
                        "display_zh": entry.zh,
                        "aliases": list(entry.aliases),
                    }
                )
        result.sort(key=lambda item: (-int(item["count"]), item["tag"].casefold()))
        return result

    def tag_suggestions(
        self,
        *,
        q: str = "",
        kind: str = "source",
        limit: int = 60,
        offset: int = 0,
        folder_keys: Sequence[str] | None = None,
        mtime_after: str | None = None,
        mtime_before: str | None = None,
    ) -> dict[str, Any]:
        page_size = max(1, min(int(limit), 500))
        start = max(0, int(offset))
        if kind == "image":
            needle = str(q or "").strip().casefold()
            items = [
                item
                for item in self._all_image_tag_counts()
                if not needle or needle in item["tag"].casefold()
            ]
            page = items[start : start + page_size + 1]
            has_more = len(page) > page_size
            result_tags = page[:page_size]
        elif folder_keys is not None:
            ids = self._scope_point_ids(folder_keys, mtime_after, mtime_before) or []
            raw_page = self.prov.source_tag_counts_for_points(
                ids, q, limit=page_size + 1, offset=start
            )
            has_more = len(raw_page) > page_size
            result_tags = self._group_source_tag_items(raw_page, q, ids)[:page_size]
        else:
            raw_page = self.prov.list_source_tags(q, limit=page_size + 1, offset=start)
            has_more = len(raw_page) > page_size
            result_tags = self._group_source_tag_items(raw_page, q)[:page_size]
        return {
            "kind": kind,
            "tags": result_tags,
            "has_more": has_more,
            "offset": start,
        }

    def tag_dictionary(self, q: str = "", limit: int = 40) -> dict[str, Any]:
        return {
            "tags": [
                {
                    "tag": entry.tag,
                    "display_zh": entry.zh,
                    "aliases": list(entry.aliases),
                }
                for entry in search_tags(q, limit=limit)
            ]
        }

    def tag_images(
        self,
        tag: str,
        *,
        limit: int = 40,
        offset: int = 0,
        sort: str = "mtime_desc",
        seed: int | None = None,
        ratings: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        clean = str(tag or "").strip()
        if not clean:
            raise ServiceError(400, "标签不能为空")
        page_size = max(1, min(int(limit), 200))
        start = max(0, min(int(offset), 50_000))
        row_ids, total, has_more, order, used_seed = self.resolver.page_rows(
            ImageScope(
                source_tags_any=(clean,),
                ratings=normalize_ratings(ratings),
            ),
            sort=sort,
            limit=page_size,
            offset=start,
            seed=seed,
        )
        results = _hydrating_gallery_rows(self.images, self.prov, row_ids)
        return {
            "tag": clean,
            "offset": start,
            "limit": page_size,
            "sort": order,
            "seed": used_seed,
            "total": total,
            "has_more": has_more,
            "results": results,
        }

    def resolve_item(self, point_id: str) -> tuple[dict[str, Any], Path]:
        payload = self.images.get_by_point_id(point_id)
        if not payload:
            raise ServiceError(404, "图片不在索引中")
        path = Path(str(payload["path"]))
        if not path.exists():
            raise ServiceError(404, "原图片已移动或删除，请重新索引")
        return payload, path
