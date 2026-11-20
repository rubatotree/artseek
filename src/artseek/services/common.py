"""Shared service projection, fast SQL folder tree, author resolution, and error helpers."""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.scope import ImageScope
from ..db.repos.images import ImageRepo
from ..db.repos.provenance import ProvenanceRepo
from ..db.scope_resolver import ScopeResolver
from ..image_io import canonical_path
from ..library import (
    DEFAULT_SORT,
    SORT_KEYS,
    FolderCatalog,
    FolderNode,
    _FolderAcc,
    natural_key,
    normalize_day,
)
from ..provenance.store import UNASSIGNED_AUTHOR_KEY

_UNASSIGNED_TOKENS = frozenset(
    {
        UNASSIGNED_AUTHOR_KEY,
        "unassigned",
        "无画师",
        "未标注",
        "作者未标注",
        "无作者",
    }
)


class ServiceError(Exception):
    """Domain error carrying an HTTP-friendly status code and optional structured detail."""

    def __init__(self, status_code: int, detail: Any) -> None:
        message = detail.get("message", str(detail)) if isinstance(detail, dict) else str(detail)
        super().__init__(message)
        self.status_code = int(status_code)
        self.detail = detail


def gallery_result(
    point_id: str,
    payload: dict[str, Any],
    fallback_name: str = "",
) -> dict[str, Any]:
    """Project an indexed image payload into the gallery card response shape."""
    return {
        "id": point_id,
        "score": None,
        "path": str(payload.get("path") or ""),
        "name": str(payload.get("name") or fallback_name),
        "width": int(payload.get("width") or 0),
        "height": int(payload.get("height") or 0),
        "mtime_ns": int(payload.get("mtime_ns") or 0),
        "tags": list(payload.get("tags") or []),
        "tag_hits": [],
        "caption": payload.get("caption"),
    }


def attach_provenance(
    prov: ProvenanceRepo,
    results: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Hydrate `source` metadata and resolved author display names onto gallery items."""
    items = list(results)
    if not items:
        return items
    point_ids = [
        str(item["id"])
        for item in items
        if item.get("id") and item.get("kind") != "artist"
    ]
    if not point_ids:
        return items
    records = prov.get_many(point_ids)
    author_keys = list(
        dict.fromkeys(
            str(record["author_key"])
            for record in records.values()
            if record.get("author_key")
        )
    )
    groups = prov.author_groups(author_keys) if author_keys else {}
    names = prov.author_names(author_keys) if author_keys else {}
    for item in items:
        if item.get("kind") == "artist":
            continue
        record = records.get(str(item.get("id") or ""))
        if not record:
            continue
        author_key = record.get("author_key")
        key_str = str(author_key) if author_key else ""
        item["source"] = {
            "title": record.get("title"),
            "description": record.get("description"),
            "url": record.get("url"),
            "tags": list(record.get("tags") or []),
            "manual_tags": list(record.get("manual_tags") or []),
            "auto_tags": list(record.get("auto_tags") or []),
            "platform": record.get("platform"),
            "status": record.get("status"),
            "author_key": author_key,
            "author_name": names.get(key_str, "") if key_str else "",
            "author_members": groups.get(key_str, []) if key_str else [],
        }
    return items


def normalize_folder_sort(sort: str | None) -> str:
    value = str(sort or DEFAULT_SORT)
    return value if value in SORT_KEYS else DEFAULT_SORT


def fast_folder_key(value: str | Path, known_keys: set[str] | None = None) -> str:
    """Normalize a folder key in-memory without filesystem syscalls when possible."""
    text = str(value or "").strip()
    if not text:
        return ""
    if not text.startswith("~") and (
        os.path.isabs(text) or (len(text) >= 2 and text[1] == ":")
    ):
        norm = os.path.normcase(os.path.normpath(text))
        if known_keys is None or norm in known_keys:
            return norm
    return canonical_path(text)


@dataclass
class FolderTreeIndex:
    """Lightweight SQL-backed folder hierarchy (~600 folders, never scans 45k images)."""

    nodes_by_key: dict[str, FolderNode]
    parent_by_key: dict[str, str | None]
    child_keys_by_key: dict[str, list[str]]
    direct_count_by_key: dict[str, int]
    known_by_key: dict[str, bool]
    selected_by_key: dict[str, bool]
    root_keys: list[str] = field(default_factory=list)
    image_count: int = 0

    @property
    def all_keys(self) -> set[str]:
        return set(self.nodes_by_key)

    def subtree_keys(self, folder_key: str) -> list[str]:
        start = fast_folder_key(folder_key, self.all_keys)
        if start not in self.nodes_by_key:
            return []
        found: list[str] = []
        seen: set[str] = set()
        stack = [start]
        while stack:
            current = stack.pop()
            if current in seen or current not in self.nodes_by_key:
                continue
            seen.add(current)
            found.append(current)
            for child in reversed(self.child_keys_by_key.get(current, ())):
                stack.append(child)
        return found

    def _sort_nodes(self, nodes: list[FolderNode], method: str) -> None:
        for node in nodes:
            self._sort_nodes(node.children, method)
        if method == "name_desc":
            nodes.sort(key=lambda n: natural_key(n.name), reverse=True)
        elif method == "count":
            nodes.sort(key=lambda n: (-n.total, natural_key(n.name)))
        elif method == "count_asc":
            nodes.sort(key=lambda n: (n.total, natural_key(n.name)))
        elif method == "mtime":
            nodes.sort(key=lambda n: (-n.mtime_ns, natural_key(n.name)))
        elif method == "mtime_asc":
            nodes.sort(key=lambda n: (n.mtime_ns, natural_key(n.name)))
        else:
            nodes.sort(key=lambda n: natural_key(n.name))

    def sorted_root_nodes(self, sort: str = DEFAULT_SORT) -> list[FolderNode]:
        method = normalize_folder_sort(sort)
        clones = {
            key: FolderNode(
                key=node.key,
                name=node.name,
                path=node.path,
                count=node.count,
                total=node.total,
                mtime_ns=node.mtime_ns,
            )
            for key, node in self.nodes_by_key.items()
        }
        roots: list[FolderNode] = []
        for key, clone in clones.items():
            parent = self.parent_by_key.get(key)
            if parent and parent in clones:
                clones[parent].children.append(clone)
            else:
                roots.append(clone)
        self._sort_nodes(roots, method)
        return roots

    def to_roots(self, sort: str = DEFAULT_SORT) -> list[dict[str, Any]]:
        return [node.to_dict() for node in self.sorted_root_nodes(sort)]

    def selected_image_count(
        self,
        resolver: ScopeResolver,
        selected: Sequence[str],
        *,
        after_ns: int | None = None,
        before_ns: int | None = None,
    ) -> int:
        all_keys = self.all_keys
        selected_set = {fast_folder_key(key, all_keys) for key in selected if key}
        if not selected_set:
            return 0
        if after_ns is None and before_ns is None:
            return sum(self.direct_count_by_key.get(key, 0) for key in selected_set)
        return resolver.count(
            ImageScope(
                folders=tuple(sorted(selected_set)),
                folder_recursive=False,
                time_range=(after_ns, before_ns),
            )
        )


def build_folder_tree_from_db(images: ImageRepo) -> FolderTreeIndex:
    """Build folder hierarchy & bottom-up totals, cached by DB write generation."""
    gen = images.db.write_generation
    cached = getattr(images, "_cached_folder_tree", None)
    if cached is not None and cached[0] == gen:
        return cached[1]
    rows = images.list_folder_rows()
    by_id: dict[int, str] = {int(r["id"]): str(r["folder_key"]) for r in rows}

    nodes: dict[str, FolderNode] = {}
    parent_by_key: dict[str, str | None] = {}
    child_keys_by_key: dict[str, list[str]] = {}
    direct_count_by_key: dict[str, int] = {}
    known_by_key: dict[str, bool] = {}
    selected_by_key: dict[str, bool] = {}
    root_keys: list[str] = []
    total_images = 0

    for r in rows:
        fkey = str(r["folder_key"])
        parent_id = r["parent_id"]
        parent_key = by_id.get(int(parent_id)) if parent_id is not None else None
        direct = int(r["direct_count"] or 0)
        latest_mtime = int(r["latest_mtime_ns"] or 0)
        total_images += direct
        nodes[fkey] = FolderNode(
            key=fkey,
            name=str(r["name"]),
            path=str(r["path"]),
            count=direct,
            total=direct,
            mtime_ns=latest_mtime,
        )
        parent_by_key[fkey] = parent_key
        child_keys_by_key[fkey] = []
        direct_count_by_key[fkey] = direct
        known_by_key[fkey] = int(r["known"] or 0) == 1
        selected_by_key[fkey] = int(r["selected"] or 0) == 1

    roots: list[FolderNode] = []
    for key, node in nodes.items():
        parent = parent_by_key[key]
        if parent and parent in nodes:
            nodes[parent].children.append(node)
            child_keys_by_key[parent].append(key)
        else:
            roots.append(node)
            root_keys.append(key)

    def accumulate(node: FolderNode) -> None:
        for child in node.children:
            accumulate(child)
        node.total = node.count + sum(child.total for child in node.children)
        node.mtime_ns = max(
            [node.mtime_ns, *(child.mtime_ns for child in node.children)],
            default=node.mtime_ns,
        )

    for root in roots:
        accumulate(root)
        root.children.clear()
    for node in nodes.values():
        node.children.clear()

    tree = FolderTreeIndex(
        nodes_by_key=nodes,
        parent_by_key=parent_by_key,
        child_keys_by_key=child_keys_by_key,
        direct_count_by_key=direct_count_by_key,
        known_by_key=known_by_key,
        selected_by_key=selected_by_key,
        root_keys=sorted(root_keys),
        image_count=total_images,
    )
    images._cached_folder_tree = (gen, tree)  # type: ignore[attr-defined]
    return tree


def reconcile_scope_with_tree(
    images: ImageRepo, tree: FolderTreeIndex, *, persist: bool = True
) -> tuple[dict[str, Any], list[str]]:
    """Reconcile persisted scope state against current DB folders without disk Path.resolve."""
    gen = images.db.write_generation
    cached = getattr(images, "_cached_reconciled_scope", None)
    if cached is not None and cached[0] == gen:
        return dict(cached[1]), list(cached[2])

    raw = images.load_scope_state()
    current = tree.all_keys
    known = {fast_folder_key(k, current) for k in raw.get("known", ()) if k}
    new_keys = sorted(current - known) if known else []
    selected = [
        fk
        for k in raw.get("selected", ())
        if k and (fk := fast_folder_key(k, current)) in current
    ]
    merged_known = sorted(known | current)
    sort = normalize_folder_sort(raw.get("sort"))
    after = normalize_day(raw.get("mtime_after"))
    before = normalize_day(raw.get("mtime_before"))
    if after and before and after > before:
        after, before = before, after

    merged = {
        "sort": sort,
        "selected": selected,
        "known": merged_known,
        "mtime_after": after,
        "mtime_before": before,
    }
    raw_cmp = {
        "sort": raw.get("sort"),
        "selected": [fast_folder_key(k, current) for k in raw.get("selected", ()) if k],
        "known": sorted(known),
        "mtime_after": raw.get("mtime_after"),
        "mtime_before": raw.get("mtime_before"),
    }
    if persist and merged != raw_cmp and not images.db._write_lock._is_owned():
        # Skip persisting during a concurrent writer's transaction so reads stay lock-free.
        acquired = images.db._write_lock.acquire(blocking=False)
        if acquired:
            try:
                images.save_scope_state(
                    sort=sort,
                    selected=selected,
                    known=merged_known,
                    mtime_after=after,
                    mtime_before=before,
                )
            finally:
                images.db._write_lock.release()
    images._cached_reconciled_scope = (  # type: ignore[attr-defined]
        images.db.write_generation,
        merged,
        new_keys,
    )
    return merged, new_keys


def resolve_folder_keys_from_db(
    images: ImageRepo, explicit: Sequence[str] | None
) -> list[str]:
    """Resolve explicit folder keys (preserving empty `[]`) or fall back to persisted scope."""
    tree = build_folder_tree_from_db(images)
    all_keys = tree.all_keys
    if explicit is not None:
        return [fast_folder_key(str(k), all_keys) for k in explicit if str(k).strip()]
    merged, _ = reconcile_scope_with_tree(images, tree)
    return list(merged["selected"])


def resolve_author_token(prov: ProvenanceRepo, value: str) -> str:
    """Resolve user query author token to a canonical author key or 'none'."""
    token = str(value or "").strip().lstrip("@")
    if token.casefold() in _UNASSIGNED_TOKENS:
        return UNASSIGNED_AUTHOR_KEY
    summary = prov.author_summary(token)
    if summary is not None:
        return str(summary["key"])
    matches = prov.list_authors(token, limit=12)
    if len(matches) == 1:
        return str(matches[0]["key"])
    if not matches:
        raise ServiceError(400, f"没有找到作者：{value}")
    key_to_root = {
        str(item["key"]): prov.canonical_author_key(str(item["key"]))
        for item in matches
    }
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in matches:
        root = key_to_root.get(str(item["key"]), str(item["key"]))
        grouped.setdefault(root, []).append(item)
    ranked = sorted(
        grouped.items(),
        key=lambda pair: (
            -max(float(item.get("score") or 0) for item in pair[1]),
            -len(pair[1]),
            pair[0],
        ),
    )
    if len(ranked) == 1:
        return ranked[0][0]
    top_score = max(float(item.get("score") or 0) for item in ranked[0][1])
    runner_up = max(float(item.get("score") or 0) for item in ranked[1][1])
    if top_score >= 0.8 and top_score - runner_up >= 0.08:
        return ranked[0][0]
    candidates = [
        {
            "key": root,
            "name": members[0].get("name") or members[0].get("handle") or root,
        }
        for root, members in ranked
    ]
    raise ServiceError(
        400,
        {
            "message": "作者不唯一，请选择完整作者。",
            "author_candidates": candidates,
        },
    )


def resolve_author_reference(prov: ProvenanceRepo, value: str) -> str:
    """Resolve an exact artist UID, Pixiv ID, X handle, or unique name (rejecting 'none')."""
    token = str(value or "").strip().lstrip("@")
    if not token:
        raise ServiceError(400, "请填写作者 UID、Pixiv ID 或 X handle")
    if token.casefold() in _UNASSIGNED_TOKENS:
        raise ServiceError(400, "无画师不能作为标注作者")
    summary = prov.author_summary(token)
    if summary is not None:
        return str(summary["key"])
    owner = prov.identity_owner("pixiv", token)
    if owner:
        return owner
    owner = prov.identity_owner("twitter", token)
    if owner:
        return owner
    matches = prov.list_authors(token, limit=3)
    if len(matches) == 1:
        return prov.canonical_author_key(str(matches[0]["key"]))
    if not matches:
        raise ServiceError(400, f"没有找到作者：{value}")
    candidates = [
        {
            "key": str(item["key"]),
            "name": item.get("name") or item.get("handle") or str(item["key"]),
        }
        for item in matches
    ]
    raise ServiceError(
        400,
        {
            "message": "作者不唯一，请选择列表中的作者。",
            "author_candidates": candidates,
        },
    )


def build_folder_catalog_from_db(images: ImageRepo) -> FolderCatalog:
    """Offline utility building full legacy `FolderCatalog`; avoid on web hot paths."""
    rows = images.list_folder_rows()
    by_id: dict[int, str] = {int(r["id"]): str(r["folder_key"]) for r in rows}
    folders: dict[str, _FolderAcc] = {}
    root_keys_set: set[str] = set()

    for r in rows:
        fkey = str(r["folder_key"])
        rkey = str(r["root_key"])
        root_keys_set.add(rkey)
        parent_id = r["parent_id"]
        parent_key = by_id.get(int(parent_id)) if parent_id is not None else None
        folders[fkey] = _FolderAcc(
            key=fkey,
            name=str(r["name"]),
            path=str(r["path"]),
            parent_key=parent_key,
            mtime_ns=int(r["latest_mtime_ns"] or 0),
        )

    for acc in folders.values():
        if acc.parent_key and acc.parent_key in folders:
            folders[acc.parent_key].child_keys.append(acc.key)

    img_rows = images.db.read_conn().execute(
        """
        SELECT f.folder_key, i.point_id, i.name, i.mtime_ns
        FROM images AS i
        JOIN folders AS f ON f.id = i.folder_id
        WHERE i.deleted = 0
        """
    ).fetchall()
    for r in img_rows:
        acc = folders.get(str(r["folder_key"]))
        if acc is not None:
            acc.ids.append(str(r["point_id"]))
            acc.id_names.append(str(r["name"]))
            acc.id_mtimes.append(int(r["mtime_ns"] or 0))
            acc.mtime_ns = max(acc.mtime_ns, int(r["mtime_ns"] or 0))

    return FolderCatalog(folders=folders, root_keys=sorted(root_keys_set))
