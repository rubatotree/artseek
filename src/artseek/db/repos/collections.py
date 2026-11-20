"""SQLite-backed collection repository preserving full v1 CollectionStore semantics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ...collections import (
    Collection,
    CollectionStore,
    HistoryAction,
    _Document,
    is_artist_item,
)
from ..connection import Database


class CollectionRepo(CollectionStore):
    """Stores collections, ordered items, manual order, and undo/redo history in SQLite."""

    def __init__(self, db: Database) -> None:
        self.db = db
        self._lock = db._write_lock

    @property
    def path(self) -> Path:
        return Path(self.db.path)

    def create(self, name: str) -> Collection:
        with self.db.write_tx():
            return super().create(name)

    def update(
        self,
        collection_id: str,
        *,
        name: str | None = None,
        sort: str | None = None,
        manual_order: tuple[str, ...] | None = None,
    ) -> Collection | None:
        with self.db.write_tx():
            return super().update(collection_id, name=name, sort=sort, manual_order=manual_order)

    def reorder_item(
        self,
        collection_id: str,
        point_id: str,
        *,
        before_id: str | None = None,
        after_id: str | None = None,
    ) -> tuple[Collection | None, bool]:
        with self.db.write_tx():
            return super().reorder_item(
                collection_id, point_id, before_id=before_id, after_id=after_id
            )

    def reorder(self, collection_ids: list[str]) -> list[Collection] | None:
        with self.db.write_tx():
            return super().reorder(collection_ids)

    def delete(
        self,
        collection_id: str,
        *,
        record_history: bool = True,
    ) -> Collection | None:
        with self.db.write_tx():
            return super().delete(collection_id, record_history=record_history)

    def add_item(
        self,
        collection_id: str,
        point_id: str,
        *,
        item_name: str = "",
    ) -> tuple[Collection | None, bool]:
        with self.db.write_tx():
            return super().add_item(collection_id, point_id, item_name=item_name)

    def remove_item(
        self,
        collection_id: str,
        point_id: str,
        *,
        item_name: str = "",
    ) -> tuple[Collection | None, bool]:
        with self.db.write_tx():
            return super().remove_item(collection_id, point_id, item_name=item_name)

    def undo(self) -> HistoryAction | None:
        with self.db.write_tx():
            return super().undo()

    def redo(self) -> HistoryAction | None:
        with self.db.write_tx():
            return super().redo()

    def _load_document(self) -> _Document:
        conn = self.db.read_conn()
        col_rows = conn.execute(
            "SELECT id, name, sort FROM collections ORDER BY position"
        ).fetchall()
        item_rows = conn.execute(
            """
            SELECT collection_id, item_id, added_position, manual_position
            FROM collection_items
            ORDER BY collection_id, added_position
            """
        ).fetchall()
        by_col: dict[str, list[tuple[str, int | None]]] = {}
        for r in item_rows:
            manual_pos = int(r["manual_position"]) if r["manual_position"] is not None else None
            by_col.setdefault(str(r["collection_id"]), []).append((str(r["item_id"]), manual_pos))

        collections: list[Collection] = []
        for r in col_rows:
            cid = str(r["id"])
            entries = by_col.get(cid, [])
            item_ids = tuple(item_id for item_id, _ in entries)
            manual_entries = sorted(
                ((pos, item_id) for item_id, pos in entries if pos is not None),
                key=lambda pair: pair[0],
            )
            manual_order = tuple(item_id for _, item_id in manual_entries)
            collections.append(
                Collection(
                    id=cid,
                    name=str(r["name"]),
                    item_ids=item_ids,
                    sort=str(r["sort"]),
                    manual_order=manual_order,
                )
            )

        hist_rows = conn.execute(
            "SELECT stack, entry_json FROM collection_history ORDER BY stack, seq"
        ).fetchall()
        raw_undo: list[Any] = []
        raw_redo: list[Any] = []
        for r in hist_rows:
            try:
                decoded = json.loads(str(r["entry_json"]))
            except json.JSONDecodeError:
                continue
            if str(r["stack"]) == "undo":
                raw_undo.append(decoded)
            elif str(r["stack"]) == "redo":
                raw_redo.append(decoded)
        return _Document(
            collections=collections,
            undo_stack=self._decode_history(raw_undo),
            redo_stack=self._decode_history(raw_redo),
        )

    def _save(self, document: _Document) -> None:
        with self.db.write_tx() as conn:
            all_point_ids = sorted(
                {
                    item_id
                    for col in document.collections
                    for item_id in col.item_ids
                    if not is_artist_item(item_id)
                }
            )
            image_map: dict[str, int] = {}
            for start in range(0, len(all_point_ids), 400):
                chunk = all_point_ids[start : start + 400]
                marks = ",".join("?" for _ in chunk)
                for row in conn.execute(
                    f"SELECT point_id, id FROM images WHERE deleted = 0 AND point_id IN ({marks})",
                    tuple(chunk),
                ).fetchall():
                    image_map[str(row["point_id"])] = int(row["id"])

            conn.execute("DELETE FROM collection_items")
            conn.execute("DELETE FROM collections")
            conn.execute("DELETE FROM collection_history")

            conn.executemany(
                "INSERT INTO collections(id, name, position, sort) VALUES (?, ?, ?, ?)",
                [
                    (col.id, col.name, pos, col.sort)
                    for pos, col in enumerate(document.collections)
                ],
            )
            item_rows: list[tuple[str, str, int | None, int, int | None]] = []
            for col in document.collections:
                manual_pos = {item_id: idx for idx, item_id in enumerate(col.manual_order)}
                for added_idx, item_id in enumerate(col.item_ids):
                    img_id = None if is_artist_item(item_id) else image_map.get(item_id)
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

            history_rows: list[tuple[str, int, str]] = []
            for seq, entry in enumerate(document.undo_stack):
                history_rows.append(
                    ("undo", seq, json.dumps(entry.encode(), ensure_ascii=False))
                )
            for seq, entry in enumerate(document.redo_stack):
                history_rows.append(
                    ("redo", seq, json.dumps(entry.encode(), ensure_ascii=False))
                )
            if history_rows:
                conn.executemany(
                    "INSERT INTO collection_history(stack, seq, entry_json) VALUES (?, ?, ?)",
                    history_rows,
                )
