from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

COLLECTIONS_VERSION = 1
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
    """A user-owned ordered set of indexed image point IDs."""

    id: str
    name: str
    item_ids: tuple[str, ...]
    pinned: bool = False

    @property
    def count(self) -> int:
        return len(self.item_ids)

    def summary(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "count": self.count,
            "pinned": self.pinned,
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
            collections = self._load()
            return [
                *[collection for collection in collections if collection.pinned],
                *[collection for collection in collections if not collection.pinned],
            ]

    def get(self, collection_id: str) -> Collection | None:
        with self._lock:
            return self._find(self._load(), collection_id)

    def create(self, name: str) -> Collection:
        with self._lock:
            collections = self._load()
            collection = Collection(id=uuid.uuid4().hex, name=name, item_ids=(), pinned=False)
            collections.append(collection)
            self._save(collections)
            return collection

    def rename(self, collection_id: str, name: str) -> Collection | None:
        return self.update(collection_id, name=name)

    def set_pinned(self, collection_id: str, pinned: bool) -> Collection | None:
        """Set one collection's dock visibility without changing its order."""
        return self.update(collection_id, pinned=pinned)

    def update(
        self,
        collection_id: str,
        *,
        name: str | None = None,
        pinned: bool | None = None,
    ) -> Collection | None:
        with self._lock:
            collections = self._load()
            index = self._index_of(collections, collection_id)
            if index is None:
                return None
            current = collections[index]
            updated = Collection(
                id=current.id,
                name=current.name if name is None else name,
                item_ids=current.item_ids,
                pinned=current.pinned if pinned is None else pinned,
            )
            collections[index] = updated
            self._save(collections)
            return updated

    def delete(self, collection_id: str) -> Collection | None:
        with self._lock:
            collections = self._load()
            index = self._index_of(collections, collection_id)
            if index is None:
                return None
            removed = collections.pop(index)
            self._save(collections)
            return removed

    def add_item(self, collection_id: str, point_id: str) -> tuple[Collection | None, bool]:
        """Add an item once, returning the collection and whether it was new."""
        with self._lock:
            collections = self._load()
            index = self._index_of(collections, collection_id)
            if index is None:
                return None, False
            current = collections[index]
            if point_id in current.item_ids:
                return current, False
            updated = Collection(
                id=current.id,
                name=current.name,
                item_ids=(*current.item_ids, point_id),
                pinned=current.pinned,
            )
            collections[index] = updated
            self._save(collections)
            return updated, True

    def remove_item(self, collection_id: str, point_id: str) -> tuple[Collection | None, bool]:
        """Remove an item if present without changing the remaining order."""
        with self._lock:
            collections = self._load()
            index = self._index_of(collections, collection_id)
            if index is None:
                return None, False
            current = collections[index]
            if point_id not in current.item_ids:
                return current, False
            updated = Collection(
                id=current.id,
                name=current.name,
                item_ids=tuple(item for item in current.item_ids if item != point_id),
                pinned=current.pinned,
            )
            collections[index] = updated
            self._save(collections)
            return updated, True

    def _load(self) -> list[Collection]:
        if not self.path.exists():
            return []
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        if not isinstance(raw, dict) or not isinstance(raw.get("collections"), list):
            return []

        collections: list[Collection] = []
        seen_ids: set[str] = set()
        for value in raw["collections"]:
            collection = self._decode_collection(value, seen_ids)
            if collection is not None:
                seen_ids.add(collection.id)
                collections.append(collection)
        return collections

    def _decode_collection(self, value: object, seen_ids: set[str]) -> Collection | None:
        if not isinstance(value, dict):
            return None
        collection_id = value.get("id")
        name = value.get("name")
        raw_items = value.get("item_ids")
        pinned = value.get("pinned", False)
        if (
            not isinstance(collection_id, str)
            or not collection_id
            or collection_id in seen_ids
            or not isinstance(name, str)
            or not name.strip()
            or not isinstance(raw_items, list)
        ):
            return None
        item_ids: list[str] = []
        seen_items: set[str] = set()
        for point_id in raw_items:
            if not isinstance(point_id, str) or not point_id or point_id in seen_items:
                continue
            seen_items.add(point_id)
            item_ids.append(point_id)
        return Collection(
            id=collection_id,
            name=name.strip(),
            item_ids=tuple(item_ids),
            pinned=pinned if isinstance(pinned, bool) else False,
        )

    def _save(self, collections: list[Collection]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": COLLECTIONS_VERSION,
            "collections": [
                {
                    "id": item.id,
                    "name": item.name,
                    "item_ids": list(item.item_ids),
                    "pinned": item.pinned,
                }
                for item in collections
            ],
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
