from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

COLLECTIONS_VERSION = 3
MAX_COLLECTION_HISTORY = 20
HISTORY_DELETE_COLLECTION = "delete_collection"
HISTORY_CREATE_COLLECTION = "create_collection"
HISTORY_ADD_ITEM = "add_item"
HISTORY_REMOVE_ITEM = "remove_item"
HISTORY_RENAME_COLLECTION = "rename_collection"
HISTORY_REORDER_COLLECTIONS = "reorder_collections"
HISTORY_REORDER_ITEMS = "reorder_items"
HISTORY_COLLECTION_ORDER_ID = "__collection_order__"
_HISTORY_KINDS = frozenset(
    {
        HISTORY_DELETE_COLLECTION,
        HISTORY_CREATE_COLLECTION,
        HISTORY_ADD_ITEM,
        HISTORY_REMOVE_ITEM,
        HISTORY_RENAME_COLLECTION,
        HISTORY_REORDER_COLLECTIONS,
        HISTORY_REORDER_ITEMS,
    }
)
SORT_ADDED_ASC = "added_asc"
SORT_ADDED_DESC = "added_desc"
SORT_MTIME_ASC = "mtime_asc"
SORT_MTIME_DESC = "mtime_desc"
SORT_NAME_ASC = "name_asc"
SORT_NAME_DESC = "name_desc"
SORT_MANUAL = "manual"
DEFAULT_COLLECTION_SORT = SORT_MANUAL
COLLECTION_SORT_KEYS = frozenset(
    {
        SORT_ADDED_ASC,
        SORT_ADDED_DESC,
        SORT_MTIME_ASC,
        SORT_MTIME_DESC,
        SORT_NAME_ASC,
        SORT_NAME_DESC,
        SORT_MANUAL,
    }
)
PAYLOAD_SORT_KEYS = frozenset({SORT_MTIME_ASC, SORT_MTIME_DESC, SORT_NAME_ASC, SORT_NAME_DESC})
ARTIST_ITEM_PREFIX = "artist:"


def is_artist_item(item_id: str) -> bool:
    return str(item_id or "").startswith(ARTIST_ITEM_PREFIX)


def artist_item_key(item_id: str) -> str:
    return str(item_id or "")[len(ARTIST_ITEM_PREFIX):]


def artist_item_id(artist_key: str) -> str:
    return f"{ARTIST_ITEM_PREFIX}{artist_key}"
_PATH_LOCKS: dict[str, threading.RLock] = {}
_PATH_LOCKS_GUARD = threading.Lock()


def _lock_for_path(path: Path) -> threading.RLock:
    """Return one process-wide lock for every spelling of the same JSON file."""
    try:
        key = str(path.resolve()).casefold()
    except OSError:
        key = str(path.absolute()).casefold()
    with _PATH_LOCKS_GUARD:
        return _PATH_LOCKS.setdefault(key, threading.RLock())


@dataclass(frozen=True, slots=True)
class Collection:
    """A user-owned ordered set of indexed image point IDs.

    ``item_ids`` is the immutable insertion sequence; ``manual_order`` only
    stores a user-defined permutation of that sequence, so switching between
    "added order" and "manual order" never loses either one.
    """

    id: str
    name: str
    item_ids: tuple[str, ...]
    sort: str = DEFAULT_COLLECTION_SORT
    manual_order: tuple[str, ...] = ()

    @property
    def count(self) -> int:
        return len(self.item_ids)

    def ordered_item_ids(self) -> tuple[str, ...]:
        """Manual ordering reconciled with the current membership.

        Stale IDs (kept across index removals) stay in the permutation, and new
        items are appended in insertion order until the user moves them.
        """
        if not self.manual_order:
            return self.item_ids
        known = set(self.item_ids)
        ordered: list[str] = []
        seen: set[str] = set()
        for point_id in self.manual_order:
            if point_id in known and point_id not in seen:
                ordered.append(point_id)
                seen.add(point_id)
        ordered.extend(point_id for point_id in self.item_ids if point_id not in seen)
        return tuple(ordered)

    def summary(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "count": self.count,
            "sort": self.sort,
        }


@dataclass(frozen=True, slots=True)
class HistoryEntry:
    """One reversible collection mutation, stored oldest-first."""

    kind: str
    collection_id: str
    name: str
    index: int
    item_ids: tuple[str, ...] = ()
    point_id: str = ""
    item_name: str = ""
    next_name: str = ""
    order_ids: tuple[str, ...] = ()
    next_order_ids: tuple[str, ...] = ()
    manual_order: tuple[str, ...] = ()
    next_manual_order: tuple[str, ...] = ()
    sort_mode: str = ""
    next_sort_mode: str = ""

    def label(self, collections: list[Collection]) -> str:
        current = CollectionStore._find(collections, self.collection_id)
        name = current.name if current is not None else self.name
        item = self.item_name
        if self.kind == HISTORY_DELETE_COLLECTION:
            return f"删除「{name}」"
        if self.kind == HISTORY_CREATE_COLLECTION:
            return f"新建「{name}」" if name else "新建收藏夹"
        if self.kind == HISTORY_RENAME_COLLECTION:
            return f"重命名「{self.name}」为「{self.next_name}」"
        if self.kind == HISTORY_REORDER_COLLECTIONS:
            return "调整收藏夹顺序"
        if self.kind == HISTORY_REORDER_ITEMS:
            return f"调整「{name}」内的图片顺序"
        if self.kind == HISTORY_ADD_ITEM:
            return f"收藏「{item}」到「{name}」" if item else f"收藏到「{name}」"
        if item:
            return f"从「{name}」移出「{item}」"
        return f"从「{name}」移出"

    def encode(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "collection_id": self.collection_id,
            "name": self.name,
            "index": self.index,
            "item_ids": list(self.item_ids),
            "point_id": self.point_id,
            "item_name": self.item_name,
            "next_name": self.next_name,
            "order_ids": list(self.order_ids),
            "next_order_ids": list(self.next_order_ids),
            "manual_order": list(self.manual_order),
            "next_manual_order": list(self.next_manual_order),
            "sort_mode": self.sort_mode,
            "next_sort_mode": self.next_sort_mode,
        }


@dataclass(frozen=True, slots=True)
class HistoryAction:
    """The mutation that undo or redo just applied."""

    action: str
    kind: str
    collection_id: str
    name: str
    label: str
    collection: Collection | None
    collections: tuple[Collection, ...]
    history: dict[str, Any]

    def payload(self) -> dict[str, Any]:
        return {
            "applied": {
                "action": self.action,
                "kind": self.kind,
                "collection_id": self.collection_id,
                "name": self.name,
                "label": self.label,
            },
            "collection": self.collection.summary() if self.collection else None,
            "collections": [item.summary() for item in self.collections],
            "history": self.history,
        }


@dataclass(slots=True)
class _Document:
    collections: list[Collection] = field(default_factory=list)
    undo_stack: list[HistoryEntry] = field(default_factory=list)
    redo_stack: list[HistoryEntry] = field(default_factory=list)

    def history_payload(self) -> dict[str, Any]:
        undo = self.undo_stack[-1] if self.undo_stack else None
        redo = self.redo_stack[-1] if self.redo_stack else None
        return {
            "can_undo": undo is not None,
            "can_redo": redo is not None,
            "undo_label": undo.label(self.collections) if undo else None,
            "redo_label": redo.label(self.collections) if redo else None,
        }


class CollectionStore:
    """Small, process-local locked JSON store for user collection state.

    Collections deliberately do not share Qdrant's lifecycle. A point which is
    temporarily absent from the index remains in this file and can reappear in
    its original position after a later re-index.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = _lock_for_path(path)

    def list(self) -> list[Collection]:
        with self._lock:
            return self._load_document().collections

    def snapshot(self) -> tuple[list[Collection], dict[str, Any]]:
        with self._lock:
            document = self._load_document()
            return document.collections, document.history_payload()

    def get(self, collection_id: str) -> Collection | None:
        with self._lock:
            return self._find(self._load_document().collections, collection_id)

    def create(self, name: str) -> Collection:
        with self._lock:
            document = self._load_document()
            collection = Collection(
                id=uuid.uuid4().hex,
                name=name,
                item_ids=(),
                sort=DEFAULT_COLLECTION_SORT,
            )
            document.collections.append(collection)
            self._record_undo(
                document,
                HistoryEntry(
                    kind=HISTORY_CREATE_COLLECTION,
                    collection_id=collection.id,
                    name=collection.name,
                    index=len(document.collections) - 1,
                ),
            )
            self._save(document)
            return collection

    def rename(self, collection_id: str, name: str) -> Collection | None:
        return self.update(collection_id, name=name)

    def update(
        self,
        collection_id: str,
        *,
        name: str | None = None,
        sort: str | None = None,
        manual_order: tuple[str, ...] | None = None,
    ) -> Collection | None:
        """Update collection metadata; ``sort`` is a preference, not history."""
        if sort is not None and sort not in COLLECTION_SORT_KEYS:
            raise ValueError("未知的收藏夹排序方式")
        with self._lock:
            document = self._load_document()
            index = self._index_of(document.collections, collection_id)
            if index is None:
                return None
            current = document.collections[index]
            next_name = current.name if name is None else name
            next_sort = current.sort if sort is None else sort
            next_manual_order = current.manual_order if manual_order is None else manual_order
            if (
                current.name == next_name
                and current.sort == next_sort
                and current.manual_order == next_manual_order
            ):
                return current
            updated = replace(
                current,
                name=next_name,
                sort=next_sort,
                manual_order=next_manual_order,
            )
            document.collections[index] = updated
            if current.name != next_name:
                self._record_undo(
                    document,
                    HistoryEntry(
                        kind=HISTORY_RENAME_COLLECTION,
                        collection_id=current.id,
                        name=current.name,
                        next_name=next_name,
                        index=index,
                    ),
                )
            self._save(document)
            return updated

    def reorder_item(
        self,
        collection_id: str,
        point_id: str,
        *,
        before_id: str | None = None,
        after_id: str | None = None,
    ) -> tuple[Collection | None, bool]:
        """Move one item to an anchor in the manual order.

        ``None`` anchors move the item to the end. A missing anchor returns
        ``None`` so the caller can report a stale view instead of silently
        appending, which would surprise a concurrent editor.
        """
        with self._lock:
            document = self._load_document()
            index = self._index_of(document.collections, collection_id)
            if index is None:
                return None, False
            current = document.collections[index]
            if point_id not in current.item_ids:
                return None, False
            order = list(current.ordered_item_ids())
            previous_order = tuple(order)
            sort_mode = current.sort
            order.remove(point_id)
            if before_id is not None:
                if before_id == point_id:
                    return current, False
                if before_id not in order:
                    return None, False
                insert_at = order.index(before_id)
            elif after_id is not None:
                if after_id == point_id:
                    return current, False
                if after_id not in order:
                    return None, False
                insert_at = order.index(after_id) + 1
            else:
                insert_at = len(order)
            order.insert(insert_at, point_id)
            next_order = tuple(order)
            if next_order == previous_order and sort_mode == SORT_MANUAL:
                return current, False
            updated = replace(
                current,
                sort=SORT_MANUAL,
                manual_order=next_order,
            )
            document.collections[index] = updated
            self._record_undo(
                document,
                HistoryEntry(
                    kind=HISTORY_REORDER_ITEMS,
                    collection_id=current.id,
                    name=current.name,
                    index=insert_at,
                    point_id=point_id,
                    sort_mode=sort_mode,
                    next_sort_mode=SORT_MANUAL,
                    manual_order=current.manual_order,
                    next_manual_order=next_order,
                ),
            )
            self._save(document)
            return updated, True

    def reorder(self, collection_ids: list[str]) -> list[Collection] | None:
        """Persist a complete collection ordering without dropping concurrent rows."""
        with self._lock:
            document = self._load_document()
            current_ids = [collection.id for collection in document.collections]
            if len(collection_ids) != len(current_ids) or set(collection_ids) != set(current_ids):
                return None
            if collection_ids == current_ids:
                return document.collections
            by_id = {collection.id: collection for collection in document.collections}
            document.collections = [by_id[collection_id] for collection_id in collection_ids]
            self._record_undo(
                document,
                HistoryEntry(
                    kind=HISTORY_REORDER_COLLECTIONS,
                    collection_id=HISTORY_COLLECTION_ORDER_ID,
                    name="收藏夹",
                    index=0,
                    order_ids=tuple(current_ids),
                    next_order_ids=tuple(collection_ids),
                ),
            )
            self._save(document)
            return document.collections

    def delete(
        self,
        collection_id: str,
        *,
        record_history: bool = True,
    ) -> Collection | None:
        with self._lock:
            document = self._load_document()
            index = self._index_of(document.collections, collection_id)
            if index is None:
                return None
            removed = document.collections.pop(index)
            if record_history:
                self._record_undo(
                    document,
                    HistoryEntry(
                        kind=HISTORY_DELETE_COLLECTION,
                        collection_id=removed.id,
                        name=removed.name,
                        index=index,
                        item_ids=removed.item_ids,
                        manual_order=removed.manual_order,
                        sort_mode=removed.sort,
                    ),
                )
            else:
                document.undo_stack = [
                    entry
                    for entry in document.undo_stack
                    if not (
                        entry.kind == HISTORY_CREATE_COLLECTION
                        and entry.collection_id == collection_id
                    )
                ]
                document.redo_stack = [
                    entry
                    for entry in document.redo_stack
                    if not (
                        entry.kind == HISTORY_CREATE_COLLECTION
                        and entry.collection_id == collection_id
                    )
                ]
            self._save(document)
            return removed

    def add_item(
        self,
        collection_id: str,
        point_id: str,
        *,
        item_name: str = "",
    ) -> tuple[Collection | None, bool]:
        """Add an item once, returning the collection and whether it was new."""
        with self._lock:
            document = self._load_document()
            index = self._index_of(document.collections, collection_id)
            if index is None:
                return None, False
            current = document.collections[index]
            if point_id in current.item_ids:
                return current, False
            if current.manual_order:
                next_manual = (
                    point_id,
                    *(item for item in current.manual_order if item != point_id),
                )
            elif current.sort == SORT_MANUAL:
                next_manual = (point_id, *reversed(current.item_ids))
            else:
                next_manual = current.manual_order
            updated = replace(
                current,
                item_ids=(*current.item_ids, point_id),
                sort=SORT_MANUAL,
                manual_order=next_manual,
            )
            document.collections[index] = updated
            self._record_undo(
                document,
                HistoryEntry(
                    kind=HISTORY_ADD_ITEM,
                    collection_id=current.id,
                    name=current.name,
                    index=len(updated.item_ids) - 1,
                    point_id=point_id,
                    item_name=item_name.strip()[:80],
                    manual_order=current.manual_order,
                    next_manual_order=next_manual,
                    sort_mode=current.sort,
                    next_sort_mode=SORT_MANUAL,
                ),
            )
            self._save(document)
            return updated, True

    def remove_item(
        self,
        collection_id: str,
        point_id: str,
        *,
        item_name: str = "",
    ) -> tuple[Collection | None, bool]:
        """Remove an item if present without changing the remaining order."""
        with self._lock:
            document = self._load_document()
            index = self._index_of(document.collections, collection_id)
            if index is None:
                return None, False
            current = document.collections[index]
            if point_id not in current.item_ids:
                return current, False
            item_index = current.item_ids.index(point_id)
            next_manual = tuple(item for item in current.manual_order if item != point_id)
            updated = replace(
                current,
                item_ids=tuple(item for item in current.item_ids if item != point_id),
                manual_order=next_manual,
            )
            document.collections[index] = updated
            self._record_undo(
                document,
                HistoryEntry(
                    kind=HISTORY_REMOVE_ITEM,
                    collection_id=current.id,
                    name=current.name,
                    index=item_index,
                    point_id=point_id,
                    item_name=item_name.strip()[:80],
                    manual_order=current.manual_order,
                    next_manual_order=next_manual,
                    sort_mode=current.sort,
                    next_sort_mode=current.sort,
                ),
            )
            self._save(document)
            return updated, True

    def undo(self) -> HistoryAction | None:
        with self._lock:
            document = self._load_document()
            if not document.undo_stack:
                return None
            entry = document.undo_stack.pop()
            collection = self._apply_undo(document, entry)
            document.redo_stack.append(entry)
            self._save(document)
            return self._history_action("undo", document, entry, collection)

    def redo(self) -> HistoryAction | None:
        with self._lock:
            document = self._load_document()
            if not document.redo_stack:
                return None
            entry = self._refresh_redo_entry(document, document.redo_stack.pop())
            collection = self._apply_redo(document, entry)
            document.undo_stack.append(entry)
            overflow = len(document.undo_stack) - MAX_COLLECTION_HISTORY
            if overflow > 0:
                del document.undo_stack[:overflow]
            self._save(document)
            return self._history_action("redo", document, entry, collection)

    def _load_document(self) -> _Document:
        if not self.path.exists():
            return _Document()
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return _Document()
        if not isinstance(raw, dict) or not isinstance(raw.get("collections"), list):
            return _Document()

        version = raw.get("version")
        legacy_order = not isinstance(version, int) or version < COLLECTIONS_VERSION
        collections: list[Collection] = []
        seen_ids: set[str] = set()
        for value in raw["collections"]:
            collection = self._decode_collection(value, seen_ids)
            if collection is not None:
                seen_ids.add(collection.id)
                if legacy_order:
                    manual_order = collection.manual_order or tuple(reversed(collection.item_ids))
                    collection = replace(
                        collection,
                        sort=SORT_MANUAL,
                        manual_order=manual_order,
                    )
                collections.append(collection)
        return _Document(
            collections=collections,
            undo_stack=self._decode_history(raw.get("undo_history")),
            redo_stack=self._decode_history(raw.get("redo_history")),
        )

    def _decode_collection(self, value: object, seen_ids: set[str]) -> Collection | None:
        if not isinstance(value, dict):
            return None
        collection_id = value.get("id")
        name = value.get("name")
        raw_items = value.get("item_ids")
        raw_manual_order = value.get("manual_order")
        raw_sort = value.get("sort")
        if (
            not isinstance(collection_id, str)
            or not collection_id
            or collection_id in seen_ids
            or not isinstance(name, str)
            or not isinstance(raw_items, list)
            or not isinstance(raw_manual_order, (list, type(None)))
        ):
            return None
        item_ids: list[str] = []
        seen_items: set[str] = set()
        for point_id in raw_items:
            if not isinstance(point_id, str) or not point_id or point_id in seen_items:
                continue
            seen_items.add(point_id)
            item_ids.append(point_id)
        sort = (
            raw_sort
            if isinstance(raw_sort, str) and raw_sort in COLLECTION_SORT_KEYS
            else DEFAULT_COLLECTION_SORT
        )
        manual_order = tuple(self._decode_history_ids(raw_manual_order or []))
        return Collection(
            id=collection_id,
            name=name.strip(),
            item_ids=tuple(item_ids),
            sort=sort,
            manual_order=manual_order,
        )

    def _decode_history(self, raw: object) -> list[HistoryEntry]:
        if not isinstance(raw, list):
            return []
        entries: list[HistoryEntry] = []
        for value in raw:
            entry = self._decode_history_entry(value)
            if entry is not None:
                entries.append(entry)
            if len(entries) >= MAX_COLLECTION_HISTORY:
                break
        return entries

    def _decode_history_entry(self, value: object) -> HistoryEntry | None:
        if not isinstance(value, dict):
            return None
        kind = value.get("kind")
        collection_id = value.get("collection_id")
        name = value.get("name")
        index = value.get("index")
        point_id = value.get("point_id") or ""
        item_name = value.get("item_name") or ""
        next_name = value.get("next_name") or ""
        raw_order_ids = value.get("order_ids") or []
        raw_next_order_ids = value.get("next_order_ids") or []
        raw_items = value.get("item_ids") or []
        raw_manual_order = value.get("manual_order") or []
        raw_next_manual_order = value.get("next_manual_order") or []
        sort_mode = value.get("sort_mode") or ""
        next_sort_mode = value.get("next_sort_mode") or ""
        if (
            kind not in _HISTORY_KINDS
            or not isinstance(collection_id, str)
            or not collection_id
            or not isinstance(name, str)
            or not isinstance(index, int)
            or isinstance(index, bool)
            or index < 0
            or not isinstance(point_id, str)
            or not isinstance(item_name, str)
            or not isinstance(next_name, str)
            or not isinstance(raw_order_ids, list)
            or not isinstance(raw_next_order_ids, list)
            or not isinstance(raw_items, list)
            or not isinstance(raw_manual_order, list)
            or not isinstance(raw_next_manual_order, list)
            or not isinstance(sort_mode, str)
            or not isinstance(next_sort_mode, str)
        ):
            return None
        if kind in {HISTORY_ADD_ITEM, HISTORY_REMOVE_ITEM, HISTORY_REORDER_ITEMS} and not point_id:
            return None
        sort_mode = sort_mode if sort_mode in COLLECTION_SORT_KEYS else ""
        next_sort_mode = next_sort_mode if next_sort_mode in COLLECTION_SORT_KEYS else ""
        if kind == HISTORY_REORDER_ITEMS and (
            sort_mode not in COLLECTION_SORT_KEYS
            or next_sort_mode not in COLLECTION_SORT_KEYS
        ):
            return None
        order_ids = self._decode_history_ids(raw_order_ids)
        next_order_ids = self._decode_history_ids(raw_next_order_ids)
        if kind == HISTORY_REORDER_COLLECTIONS and (
            not order_ids
            or set(order_ids) != set(next_order_ids)
        ):
            return None
        item_ids: list[str] = []
        seen_items: set[str] = set()
        for item_id in raw_items:
            if not isinstance(item_id, str) or not item_id or item_id in seen_items:
                continue
            seen_items.add(item_id)
            item_ids.append(item_id)
        return HistoryEntry(
            kind=kind,
            collection_id=collection_id,
            name=name.strip(),
            index=index,
            item_ids=tuple(item_ids),
            point_id=point_id,
            item_name=item_name.strip()[:80],
            next_name=next_name.strip(),
            order_ids=tuple(order_ids),
            next_order_ids=tuple(next_order_ids),
            manual_order=tuple(self._decode_history_ids(raw_manual_order)),
            next_manual_order=tuple(self._decode_history_ids(raw_next_manual_order)),
            sort_mode=sort_mode,
            next_sort_mode=next_sort_mode,
        )

    @staticmethod
    def _decode_history_ids(raw: list[object]) -> list[str]:
        ids: list[str] = []
        seen: set[str] = set()
        for value in raw:
            if not isinstance(value, str) or not value or value in seen:
                continue
            seen.add(value)
            ids.append(value)
        return ids

    def _save(self, document: _Document) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": COLLECTIONS_VERSION,
            "collections": [
                {
                    "id": item.id,
                    "name": item.name,
                    "item_ids": list(item.item_ids),
                    "sort": item.sort,
                    "manual_order": list(item.manual_order),
                }
                for item in document.collections
            ],
            "undo_history": [entry.encode() for entry in document.undo_stack],
            "redo_history": [entry.encode() for entry in document.redo_stack],
        }
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        try:
            tmp.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            tmp.replace(self.path)
        finally:
            if tmp.exists():
                tmp.unlink(missing_ok=True)

    def _record_undo(self, document: _Document, entry: HistoryEntry) -> None:
        document.undo_stack.append(entry)
        document.redo_stack.clear()
        overflow = len(document.undo_stack) - MAX_COLLECTION_HISTORY
        if overflow > 0:
            del document.undo_stack[:overflow]

    def _refresh_redo_entry(self, document: _Document, entry: HistoryEntry) -> HistoryEntry:
        if entry.kind != HISTORY_DELETE_COLLECTION:
            return entry
        index = self._index_of(document.collections, entry.collection_id)
        if index is None:
            return entry
        current = document.collections[index]
        return HistoryEntry(
            kind=entry.kind,
            collection_id=current.id,
            name=current.name,
            index=index,
            item_ids=current.item_ids,
            item_name=entry.item_name,
            manual_order=current.manual_order,
            sort_mode=current.sort,
        )

    def _apply_undo(self, document: _Document, entry: HistoryEntry) -> Collection | None:
        if entry.kind == HISTORY_REORDER_COLLECTIONS:
            return self._apply_collection_order(document, entry.order_ids)
        if entry.kind == HISTORY_REORDER_ITEMS:
            return self._apply_manual_order(document, entry, next_order=False)
        if entry.kind == HISTORY_CREATE_COLLECTION:
            return self._remove_collection(document, entry)
        if entry.kind == HISTORY_DELETE_COLLECTION:
            return self._insert_collection(document, entry)
        if entry.kind == HISTORY_ADD_ITEM:
            return self._remove_point(
                document,
                entry,
                manual_order=entry.manual_order,
                sort_mode=entry.sort_mode,
            )
        if entry.kind == HISTORY_REMOVE_ITEM:
            return self._insert_point(
                document,
                entry,
                manual_order=entry.manual_order,
                sort_mode=entry.sort_mode,
            )
        return self._rename_collection(document, entry, entry.name)

    def _apply_redo(self, document: _Document, entry: HistoryEntry) -> Collection | None:
        if entry.kind == HISTORY_REORDER_COLLECTIONS:
            return self._apply_collection_order(document, entry.next_order_ids)
        if entry.kind == HISTORY_REORDER_ITEMS:
            return self._apply_manual_order(document, entry, next_order=True)
        if entry.kind == HISTORY_CREATE_COLLECTION:
            return self._insert_collection(document, entry)
        if entry.kind == HISTORY_DELETE_COLLECTION:
            return self._remove_collection(document, entry)
        if entry.kind == HISTORY_ADD_ITEM:
            return self._insert_point(
                document,
                entry,
                manual_order=entry.next_manual_order,
                sort_mode=entry.next_sort_mode,
            )
        if entry.kind == HISTORY_REMOVE_ITEM:
            return self._remove_point(
                document,
                entry,
                manual_order=entry.next_manual_order,
                sort_mode=entry.next_sort_mode,
            )
        return self._rename_collection(document, entry, entry.next_name)

    def _insert_collection(self, document: _Document, entry: HistoryEntry) -> Collection | None:
        if self._index_of(document.collections, entry.collection_id) is not None:
            return self._find(document.collections, entry.collection_id)
        restored = Collection(
            id=entry.collection_id,
            name=entry.name,
            item_ids=entry.item_ids,
            sort=(
                entry.sort_mode
                if entry.sort_mode in COLLECTION_SORT_KEYS
                else DEFAULT_COLLECTION_SORT
            ),
            manual_order=entry.manual_order,
        )
        insert_at = min(entry.index, len(document.collections))
        document.collections.insert(insert_at, restored)
        return restored

    def _remove_collection(self, document: _Document, entry: HistoryEntry) -> Collection | None:
        index = self._index_of(document.collections, entry.collection_id)
        if index is None:
            return Collection(
                id=entry.collection_id,
                name=entry.name,
                item_ids=entry.item_ids,
                sort=(
                    entry.sort_mode
                    if entry.sort_mode in COLLECTION_SORT_KEYS
                    else DEFAULT_COLLECTION_SORT
                ),
                manual_order=entry.manual_order,
            )
        return document.collections.pop(index)

    def _apply_manual_order(
        self,
        document: _Document,
        entry: HistoryEntry,
        *,
        next_order: bool,
    ) -> Collection | None:
        index = self._index_of(document.collections, entry.collection_id)
        if index is None:
            return None
        current = document.collections[index]
        if entry.point_id not in current.item_ids:
            return current
        sort_mode = entry.next_sort_mode if next_order else entry.sort_mode
        manual_order = entry.next_manual_order if next_order else entry.manual_order
        updated = replace(
            current,
            sort=sort_mode if sort_mode in COLLECTION_SORT_KEYS else DEFAULT_COLLECTION_SORT,
            manual_order=manual_order,
        )
        document.collections[index] = updated
        return updated

    def _insert_point(
        self,
        document: _Document,
        entry: HistoryEntry,
        *,
        manual_order: tuple[str, ...] | None = None,
        sort_mode: str = "",
    ) -> Collection | None:
        index = self._index_of(document.collections, entry.collection_id)
        if index is None:
            return None
        current = document.collections[index]
        if entry.point_id in current.item_ids:
            return current
        items = list(current.item_ids)
        items.insert(min(entry.index, len(items)), entry.point_id)
        changes: dict[str, Any] = {"item_ids": tuple(items)}
        if manual_order is not None:
            changes["manual_order"] = manual_order
        if sort_mode in COLLECTION_SORT_KEYS:
            changes["sort"] = sort_mode
        updated = replace(current, **changes)
        document.collections[index] = updated
        return updated

    def _remove_point(
        self,
        document: _Document,
        entry: HistoryEntry,
        *,
        manual_order: tuple[str, ...] | None = None,
        sort_mode: str = "",
    ) -> Collection | None:
        index = self._index_of(document.collections, entry.collection_id)
        if index is None:
            return None
        current = document.collections[index]
        if entry.point_id not in current.item_ids:
            return current
        changes: dict[str, Any] = {
            "item_ids": tuple(item for item in current.item_ids if item != entry.point_id),
        }
        if manual_order is not None:
            changes["manual_order"] = manual_order
        if sort_mode in COLLECTION_SORT_KEYS:
            changes["sort"] = sort_mode
        updated = replace(current, **changes)
        document.collections[index] = updated
        return updated

    def _rename_collection(
        self,
        document: _Document,
        entry: HistoryEntry,
        name: str,
    ) -> Collection | None:
        index = self._index_of(document.collections, entry.collection_id)
        if index is None:
            return None
        current = document.collections[index]
        updated = replace(current, name=name)
        document.collections[index] = updated
        return updated

    @staticmethod
    def _apply_collection_order(
        document: _Document,
        ordered_ids: tuple[str, ...],
    ) -> None:
        by_id = {collection.id: collection for collection in document.collections}
        ordered = [by_id[collection_id] for collection_id in ordered_ids if collection_id in by_id]
        ordered_ids_set = set(ordered_ids)
        document.collections = ordered + [
            collection
            for collection in document.collections
            if collection.id not in ordered_ids_set
        ]
        return None

    @staticmethod
    def _history_action(
        action: str,
        document: _Document,
        entry: HistoryEntry,
        collection: Collection | None,
    ) -> HistoryAction:
        return HistoryAction(
            action=action,
            kind=entry.kind,
            collection_id=entry.collection_id,
            name=collection.name if collection is not None else entry.name,
            label=entry.label(document.collections),
            collection=collection,
            collections=tuple(document.collections),
            history=document.history_payload(),
        )

    @staticmethod
    def _find(collections: list[Collection], collection_id: str) -> Collection | None:
        index = CollectionStore._index_of(collections, collection_id)
        return collections[index] if index is not None else None

    @staticmethod
    def _index_of(collections: list[Collection], collection_id: str) -> int | None:
        for index, collection in enumerate(collections):
            if collection.id == collection_id:
                return index
        return None
