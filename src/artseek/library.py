from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, time
from pathlib import Path
from typing import Any

from .config import AppConfig
from .image_io import canonical_path

SORT_OPTIONS: tuple[tuple[str, str], ...] = (
    ("name", "名称 A→Z"),
    ("name_desc", "名称 Z→A"),
    ("count", "图片多→少"),
    ("count_asc", "图片少→多"),
    ("mtime", "最近更新"),
    ("mtime_asc", "最早更新"),
)
DEFAULT_SORT = "name"
SORT_KEYS = {key for key, _ in SORT_OPTIONS}
SCOPE_VERSION = 2
CATALOG_VERSION = 3


def folder_key_of(payload: dict[str, Any]) -> str:
    stored = payload.get("folder_key")
    if stored:
        return str(stored)
    raw = payload.get("path") or payload.get("path_key") or ""
    if not raw:
        return ""
    return canonical_path(Path(raw).parent)


def normalize_day(value: str | None) -> str | None:
    if not value or not str(value).strip():
        return None
    try:
        return date.fromisoformat(str(value).strip()[:10]).isoformat()
    except ValueError:
        return None


def day_start_ns(value: str | None) -> int | None:
    day = normalize_day(value)
    if day is None:
        return None
    return int(datetime.combine(date.fromisoformat(day), time.min).timestamp() * 1_000_000_000)


def day_end_ns(value: str | None) -> int | None:
    day = normalize_day(value)
    if day is None:
        return None
    return int(datetime.combine(date.fromisoformat(day), time.max).timestamp() * 1_000_000_000)


def ns_to_day(value: int) -> str | None:
    if value <= 0:
        return None
    try:
        return datetime.fromtimestamp(value / 1_000_000_000).date().isoformat()
    except (OSError, OverflowError, ValueError):
        return None


def _natural_key(name: str) -> tuple[tuple[int, int | str], ...]:
    # Tag numbers vs text so "133" and "bjd" stay comparable in Python 3.
    parts = re.split(r"(\d+)", name.casefold())
    key: list[tuple[int, int | str]] = []
    for part in parts:
        if not part:
            continue
        if part.isdigit():
            key.append((0, int(part)))
        else:
            key.append((1, part))
    return tuple(key)


def _sort_method(sort: str) -> str:
    return sort if sort in SORT_KEYS else DEFAULT_SORT


def _folder_image_refs(acc: _FolderAcc) -> list[tuple[str, str, int]]:
    refs: list[tuple[str, str, int]] = []
    for index, point_id in enumerate(acc.ids):
        name = acc.id_names[index] if index < len(acc.id_names) else ""
        stamp = acc.id_mtimes[index] if index < len(acc.id_mtimes) else 0
        refs.append((point_id, name, stamp))
    return refs


def _sort_image_refs(refs: list[tuple[str, str, int]], method: str) -> list[tuple[str, str, int]]:
    def name_key(item: tuple[str, str, int]) -> tuple[tuple[int, int | str], ...]:
        return _natural_key(item[1] or item[0])

    ordered = list(refs)
    if method == "name_desc":
        ordered.sort(key=name_key, reverse=True)
    elif method == "mtime":
        ordered.sort(key=lambda item: (-item[2], name_key(item)))
    elif method == "mtime_asc":
        ordered.sort(key=lambda item: (item[2], name_key(item)))
    else:
        ordered.sort(key=name_key)
    return ordered


@dataclass(slots=True)
class FolderNode:
    key: str
    name: str
    path: str
    count: int
    total: int
    mtime_ns: int
    children: list[FolderNode] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "name": self.name,
            "path": self.path,
            "count": self.count,
            "total": self.total,
            "mtime_ns": self.mtime_ns,
            "children": [child.to_dict() for child in self.children],
        }

    def find(self, key: str) -> FolderNode | None:
        if self.key == key:
            return self
        for child in self.children:
            found = child.find(key)
            if found is not None:
                return found
        return None

    def iter_keys(self) -> Iterable[str]:
        yield self.key
        for child in self.children:
            yield from child.iter_keys()


@dataclass(slots=True)
class FolderScope:
    sort: str = DEFAULT_SORT
    selected: list[str] = field(default_factory=list)
    known: list[str] = field(default_factory=list)
    mtime_after: str | None = None
    mtime_before: str | None = None


@dataclass
class _FolderAcc:
    key: str
    name: str
    path: str
    parent_key: str | None
    count: int = 0
    mtime_ns: int = 0
    ids: list[str] = field(default_factory=list)
    id_names: list[str] = field(default_factory=list)
    id_mtimes: list[int] = field(default_factory=list)
    child_keys: list[str] = field(default_factory=list)


@dataclass
class FolderCatalog:
    folders: dict[str, _FolderAcc]
    root_keys: list[str]

    def all_keys(self) -> set[str]:
        return set(self.folders)

    def subtree_keys(self, folder_key: str) -> set[str]:
        start = canonical_path(folder_key)
        found: set[str] = set()
        stack = [start]
        while stack:
            current = stack.pop()
            if current in found or current not in self.folders:
                continue
            found.add(current)
            stack.extend(self.folders[current].child_keys)
        return found

    def subtree_ids(self, folder_key: str) -> list[str]:
        ids: list[str] = []
        for key in self.subtree_keys(folder_key):
            ids.extend(self.folders[key].ids)
        return ids

    def point_ids(
        self,
        selected: Sequence[str],
        *,
        after_ns: int | None = None,
        before_ns: int | None = None,
    ) -> list[str] | None:
        """None means every indexed image is in scope (no query filter)."""
        selected_set = {canonical_path(key) for key in selected if key}
        if not selected_set:
            return []
        current = self.all_keys()
        all_folders = bool(current) and current <= selected_set
        time_filter = after_ns is not None or before_ns is not None
        if all_folders and not time_filter:
            return None
        ids: list[str] = []
        times: list[int] = []
        sources = self.folders.values() if all_folders else (
            self.folders[key] for key in selected_set if key in self.folders
        )
        for acc in sources:
            ids.extend(acc.ids)
            times.extend(acc.id_mtimes)
        if not time_filter:
            return ids
        lo = after_ns if after_ns is not None else -1
        hi = before_ns if before_ns is not None else (1 << 63) - 1
        if len(times) != len(ids):
            times = [*times, *([0] * (len(ids) - len(times)))][: len(ids)]
        return [point_id for point_id, stamp in zip(ids, times, strict=True) if lo <= stamp <= hi]

    def selected_image_count(
        self,
        selected: Sequence[str],
        *,
        after_ns: int | None = None,
        before_ns: int | None = None,
    ) -> int:
        ids = self.point_ids(selected, after_ns=after_ns, before_ns=before_ns)
        if ids is None:
            return sum(len(acc.ids) for acc in self.folders.values())
        return len(ids)

    def time_bounds(self) -> tuple[int, int]:
        stamps = [stamp for acc in self.folders.values() for stamp in acc.id_mtimes if stamp > 0]
        if not stamps:
            return 0, 0
        return min(stamps), max(stamps)

    def tree(self, sort: str) -> list[FolderNode]:
        method = _sort_method(sort)
        nodes = {key: self._to_node(acc) for key, acc in self.folders.items()}
        roots: list[FolderNode] = []
        for key, acc in self.folders.items():
            node = nodes[key]
            if acc.parent_key and acc.parent_key in nodes:
                nodes[acc.parent_key].children.append(node)
            else:
                roots.append(node)
        for root in roots:
            self._accumulate(root)
        self._sort_nodes(roots, method)
        return roots

    def find_node(self, folder_key: str, sort: str) -> FolderNode | None:
        key = canonical_path(folder_key)
        for root in self.tree(sort):
            found = root.find(key)
            if found is not None:
                return found
        return None

    def list_folder_images(self, folder_key: str, sort: str) -> list[tuple[str, str, int]]:
        """Subtree images as (id, name, mtime_ns), ordered like the query-directory sort."""
        key = canonical_path(folder_key)
        if key not in self.folders:
            return []
        method = _sort_method(sort)
        node = self.find_node(key, method)
        if node is None:
            return _sort_image_refs(_folder_image_refs(self.folders[key]), method)
        if method in {"count", "count_asc"}:
            return self._walk_image_refs(node)
        refs: list[tuple[str, str, int]] = []
        stack = [node]
        while stack:
            current = stack.pop()
            acc = self.folders.get(current.key)
            if acc is not None:
                refs.extend(_folder_image_refs(acc))
            stack.extend(reversed(current.children))
        return _sort_image_refs(refs, method)

    def _walk_image_refs(self, node: FolderNode) -> list[tuple[str, str, int]]:
        acc = self.folders.get(node.key)
        refs = _sort_image_refs(_folder_image_refs(acc), "name") if acc is not None else []
        for child in node.children:
            refs.extend(self._walk_image_refs(child))
        return refs

    def _accumulate(self, node: FolderNode) -> None:
        for child in node.children:
            self._accumulate(child)
        node.total = node.count + sum(child.total for child in node.children)
        node.mtime_ns = max(
            [node.mtime_ns, *(child.mtime_ns for child in node.children)],
            default=node.mtime_ns,
        )

    def _to_node(self, acc: _FolderAcc) -> FolderNode:
        return FolderNode(
            key=acc.key,
            name=acc.name,
            path=acc.path,
            count=acc.count,
            total=acc.count,
            mtime_ns=acc.mtime_ns,
        )

    def _sort_nodes(self, nodes: list[FolderNode], method: str) -> None:
        for node in nodes:
            self._sort_nodes(node.children, method)
        if method == "name_desc":
            nodes.sort(key=lambda node: _natural_key(node.name), reverse=True)
            return
        if method == "count":
            nodes.sort(key=lambda node: (-node.total, _natural_key(node.name)))
            return
        if method == "count_asc":
            nodes.sort(key=lambda node: (node.total, _natural_key(node.name)))
            return
        if method == "mtime":
            nodes.sort(key=lambda node: (-node.mtime_ns, _natural_key(node.name)))
            return
        if method == "mtime_asc":
            nodes.sort(key=lambda node: (node.mtime_ns, _natural_key(node.name)))
            return
        nodes.sort(key=lambda node: _natural_key(node.name))


def build_catalog(payloads: Iterable[tuple[str, dict[str, Any]]]) -> FolderCatalog:
    folders: dict[str, _FolderAcc] = {}
    root_keys: list[str] = []
    seen_roots: set[str] = set()

    for point_id, payload in payloads:
        image_path = Path(str(payload.get("path") or payload.get("path_key") or ""))
        if not image_path.parts:
            continue
        root = Path(str(payload.get("root") or image_path.anchor or image_path.parent))
        chain = _folder_chain(image_path, root)
        if not chain:
            continue
        root_key = canonical_path(chain[0])
        if root_key not in seen_roots:
            seen_roots.add(root_key)
            root_keys.append(root_key)
        parent_key: str | None = None
        for folder in chain:
            key = canonical_path(folder)
            acc = folders.get(key)
            if acc is None:
                acc = _FolderAcc(
                    key=key,
                    name=folder.name or str(folder),
                    path=str(folder),
                    parent_key=parent_key,
                )
                folders[key] = acc
                if parent_key and parent_key in folders:
                    parent = folders[parent_key]
                    if key not in parent.child_keys:
                        parent.child_keys.append(key)
            parent_key = key
        leaf_key = canonical_path(chain[-1])
        leaf = folders[leaf_key]
        leaf.count += 1
        leaf.ids.append(point_id)
        leaf.id_names.append(str(payload.get("name") or image_path.name))
        stamp = int(payload.get("mtime_ns", 0))
        leaf.id_mtimes.append(stamp)
        leaf.mtime_ns = max(leaf.mtime_ns, stamp)

    _restore_child_keys(folders)
    return FolderCatalog(folders=folders, root_keys=root_keys)


def _folder_chain(image_path: Path, root: Path) -> list[Path]:
    folder = image_path.parent
    root_key = canonical_path(root)
    chain: list[Path] = []
    current = folder
    while True:
        chain.append(current)
        if canonical_path(current) == root_key:
            break
        nxt = current.parent
        if nxt == current:
            break
        current = nxt
    chain.reverse()
    return chain


def load_scope(config: AppConfig) -> FolderScope:
    path = config.folder_scope_path
    if not path.exists():
        return FolderScope()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return FolderScope()
    sort = str(raw.get("sort") or DEFAULT_SORT)
    if sort not in SORT_KEYS:
        sort = DEFAULT_SORT
    selected = [canonical_path(item) for item in raw.get("selected", []) if item]
    known = [canonical_path(item) for item in raw.get("known", []) if item]
    after = normalize_day(raw.get("mtime_after"))
    before = normalize_day(raw.get("mtime_before"))
    if after and before and after > before:
        after, before = before, after
    return FolderScope(
        sort=sort,
        selected=selected,
        known=known,
        mtime_after=after,
        mtime_before=before,
    )


def save_scope(config: AppConfig, scope: FolderScope) -> None:
    config.ensure_dirs()
    payload = {
        "version": SCOPE_VERSION,
        "sort": scope.sort if scope.sort in SORT_KEYS else DEFAULT_SORT,
        "selected": list(dict.fromkeys(canonical_path(key) for key in scope.selected if key)),
        "known": list(dict.fromkeys(canonical_path(key) for key in scope.known if key)),
        "mtime_after": normalize_day(scope.mtime_after),
        "mtime_before": normalize_day(scope.mtime_before),
    }
    path = config.folder_scope_path
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def apply_scope(catalog: FolderCatalog, scope: FolderScope) -> tuple[FolderScope, list[str]]:
    current = catalog.all_keys()
    known = set(scope.known)
    new_keys = sorted(current - known, key=str) if known else []
    selected = [key for key in scope.selected if key in current]
    merged_known = sorted(known | current)
    sort = _sort_method(scope.sort)
    after = normalize_day(scope.mtime_after)
    before = normalize_day(scope.mtime_before)
    if after and before and after > before:
        after, before = before, after
    return (
        FolderScope(
            sort=sort,
            selected=selected,
            known=merged_known,
            mtime_after=after,
            mtime_before=before,
        ),
        new_keys,
    )


def load_catalog(path: Path, expected_count: int | None = None) -> FolderCatalog | None:
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if int(raw.get("version", 0)) != CATALOG_VERSION:
        return None
    if expected_count is not None and int(raw.get("count", -1)) != expected_count:
        return None
    folders: dict[str, _FolderAcc] = {}
    for item in raw.get("folders", []):
        key = str(item.get("key", ""))
        if not key:
            continue
        parent = item.get("parent_key")
        ids = [str(point_id) for point_id in item.get("ids", [])]
        names = [str(name) for name in item.get("id_names", [])]
        mtimes = [int(stamp) for stamp in item.get("id_mtimes", [])]
        if len(names) < len(ids):
            names.extend([""] * (len(ids) - len(names)))
        if len(mtimes) < len(ids):
            mtimes.extend([0] * (len(ids) - len(mtimes)))
        folders[key] = _FolderAcc(
            key=key,
            name=str(item.get("name") or key),
            path=str(item.get("path") or key),
            parent_key=str(parent) if parent else None,
            count=int(item.get("count", 0)),
            mtime_ns=int(item.get("mtime_ns", 0)),
            ids=ids,
            id_names=names[: len(ids)],
            id_mtimes=mtimes[: len(ids)],
        )
    _restore_child_keys(folders)
    return FolderCatalog(folders=folders, root_keys=[str(key) for key in raw.get("root_keys", [])])


def save_catalog(path: Path, catalog: FolderCatalog, count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": CATALOG_VERSION,
        "count": count,
        "root_keys": catalog.root_keys,
        "folders": [
            {
                "key": acc.key,
                "name": acc.name,
                "path": acc.path,
                "parent_key": acc.parent_key,
                "count": acc.count,
                "mtime_ns": acc.mtime_ns,
                "ids": acc.ids,
                "id_names": acc.id_names,
                "id_mtimes": acc.id_mtimes,
            }
            for acc in catalog.folders.values()
        ],
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    tmp.replace(path)


def rebuild_catalog(store: Any, config: AppConfig) -> FolderCatalog:
    rows = store.iter_folder_payloads() if hasattr(store, "iter_folder_payloads") else store.iter_payloads()
    catalog = build_catalog(rows)
    count = int(store.count()) if hasattr(store, "count") else sum(len(acc.ids) for acc in catalog.folders.values())
    save_catalog(config.folder_catalog_path, catalog, count)
    return catalog


def _restore_child_keys(folders: dict[str, _FolderAcc]) -> None:
    for acc in folders.values():
        acc.child_keys.clear()
    for acc in folders.values():
        parent = acc.parent_key
        if parent and parent in folders:
            siblings = folders[parent].child_keys
            if acc.key not in siblings:
                siblings.append(acc.key)


def root_image_counts(catalog: FolderCatalog) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    for key in catalog.root_keys:
        acc = catalog.folders.get(key)
        path = acc.path if acc is not None else key
        count = sum(len(catalog.folders[item].ids) for item in catalog.subtree_keys(key))
        rows.append((path, count))
    rows.sort(key=lambda item: item[0].casefold())
    return rows


def index_roots_from_scope(scope: FolderScope) -> list[str]:
    known = {canonical_path(key) for key in scope.known if key}
    roots = [key for key in known if canonical_path(Path(key).parent) not in known]
    roots.sort(key=str.casefold)
    return roots


def library_payload(
    catalog: FolderCatalog,
    scope: FolderScope,
    new_keys: Sequence[str],
) -> dict[str, Any]:
    after_ns = day_start_ns(scope.mtime_after)
    before_ns = day_end_ns(scope.mtime_before)
    low, high = catalog.time_bounds()
    return {
        "sort": scope.sort,
        "sort_options": [{"id": key, "label": label} for key, label in SORT_OPTIONS],
        "selected": scope.selected,
        "new_keys": list(new_keys),
        "mtime_after": scope.mtime_after,
        "mtime_before": scope.mtime_before,
        "mtime_min": ns_to_day(low),
        "mtime_max": ns_to_day(high),
        "image_count": sum(len(acc.ids) for acc in catalog.folders.values()),
        "selected_image_count": catalog.selected_image_count(
            scope.selected, after_ns=after_ns, before_ns=before_ns
        ),
        "roots": [node.to_dict() for node in catalog.tree(scope.sort)],
    }
