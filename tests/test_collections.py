from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path

from artseek.collections import MAX_COLLECTION_HISTORY, CollectionStore


class CollectionStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.path = Path(self._temporary_directory.name) / "collections.json"
        self.store = CollectionStore(self.path)

    def tearDown(self) -> None:
        self._temporary_directory.cleanup()

    def test_persists_collection_and_item_insertion_order_without_duplicates(self) -> None:
        first = self.store.create("灵感")
        self.store.create("待研究")
        self.store.add_item(first.id, "first")
        self.store.add_item(first.id, "second")
        duplicate, added = self.store.add_item(first.id, "first")

        self.assertFalse(added)
        self.assertEqual(duplicate.item_ids if duplicate else (), ("first", "second"))
        loaded = CollectionStore(self.path).list()
        self.assertEqual([item.name for item in loaded], ["灵感", "待研究"])
        self.assertEqual(loaded[0].item_ids, ("first", "second"))
        self.assertFalse(self.path.with_suffix(".json.tmp").exists())

        raw = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(raw["collections"][0]["item_ids"], ["first", "second"])

    def test_preserves_a_blank_default_collection_name(self) -> None:
        collection = self.store.create("")

        restored = CollectionStore(self.path).get(collection.id)
        self.assertIsNotNone(restored)
        self.assertEqual(restored.name if restored else None, "")
        self.assertEqual(self.store.snapshot()[1]["undo_label"], "新建收藏夹")
        undone = self.store.undo()
        self.assertEqual(undone.kind if undone else "", "create_collection")
        self.assertIsNone(self.store.get(collection.id))

        redone = self.store.redo()
        self.assertEqual(redone.kind if redone else "", "create_collection")
        self.assertIsNotNone(self.store.get(collection.id))

        self.assertIsNotNone(self.store.delete(collection.id, record_history=False))
        self.assertFalse(self.store.snapshot()[1]["can_undo"])

    def test_corrupt_and_duplicate_on_disk_data_is_safe_to_load(self) -> None:
        self.path.write_text("not json", encoding="utf-8")
        self.assertEqual(self.store.list(), [])

        self.path.write_text(
            json.dumps(
                {
                    "collections": [
                        {"id": "one", "name": "  保留  ", "item_ids": ["a", "a", 7, "b"]},
                        {"id": "one", "name": "忽略", "item_ids": []},
                        {"id": "bad", "name": None, "item_ids": []},
                    ]
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        loaded = self.store.list()
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].name, "保留")
        self.assertEqual(loaded[0].item_ids, ("a", "b"))

    def test_remove_keeps_remaining_order_and_is_idempotent(self) -> None:
        collection = self.store.create("素材")
        for point_id in ("a", "b", "c"):
            self.store.add_item(collection.id, point_id)

        updated, removed = self.store.remove_item(collection.id, "b")
        repeated, removed_again = self.store.remove_item(collection.id, "b")

        self.assertTrue(removed)
        self.assertFalse(removed_again)
        self.assertEqual(updated.item_ids if updated else (), ("a", "c"))
        self.assertEqual(repeated.item_ids if repeated else (), ("a", "c"))

    def test_legacy_pinned_state_is_ignored_and_removed_on_next_write(self) -> None:
        self.path.write_text(
            json.dumps(
                {
                    "collections": [
                        {"id": "older", "name": "旧格式", "item_ids": []},
                        {"id": "legacy", "name": "旧置顶", "item_ids": [], "pinned": True},
                        {"id": "newer", "name": "新建", "item_ids": []},
                    ]
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        self.assertEqual([item.id for item in self.store.list()], ["older", "legacy", "newer"])
        self.assertIsNone(self.store.reorder(["older", "legacy"]))
        reordered = self.store.reorder(["newer", "older", "legacy"])

        self.assertEqual([item.id for item in reordered or []], ["newer", "older", "legacy"])
        restored = CollectionStore(self.path).list()
        self.assertEqual([item.id for item in restored], ["newer", "older", "legacy"])
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertTrue(all("pinned" not in item for item in raw["collections"]))

    def test_stores_for_the_same_path_share_one_read_modify_write_lock(self) -> None:
        first = CollectionStore(self.path)
        second = CollectionStore(self.path)

        self.assertIs(first._lock, second._lock)
        collection = first.create("并发写入")
        start = threading.Barrier(2)

        def add(store: CollectionStore, point_id: str) -> None:
            start.wait()
            store.add_item(collection.id, point_id)

        threads = [
            threading.Thread(target=add, args=(first, "first")),
            threading.Thread(target=add, args=(second, "second")),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        updated = first.get(collection.id)
        self.assertEqual(set(updated.item_ids if updated else ()), {"first", "second"})

    def test_delete_and_membership_changes_are_undoable_in_order(self) -> None:
        first = self.store.create("灵感")
        second = self.store.create("待研究")
        self.store.add_item(first.id, "a")
        self.store.add_item(first.id, "b")
        self.store.add_item(first.id, "c")
        self.store.remove_item(first.id, "b")
        removed = self.store.delete(second.id)
        self.assertIsNotNone(removed)
        self.assertEqual([item.id for item in self.store.list()], [first.id])

        _, history = self.store.snapshot()
        self.assertTrue(history["can_undo"])
        self.assertFalse(history["can_redo"])
        self.assertEqual(history["undo_label"], "删除「待研究」")

        undone_delete = self.store.undo()
        self.assertIsNotNone(undone_delete)
        self.assertEqual(undone_delete.kind if undone_delete else "", "delete_collection")
        self.assertEqual([item.id for item in self.store.list()], [first.id, second.id])
        restored = self.store.get(second.id)
        self.assertEqual(restored.name if restored else "", "待研究")
        first_items = self.store.get(first.id)
        self.assertEqual(first_items.item_ids if first_items else (), ("a", "c"))

        undone_remove = self.store.undo()
        self.assertEqual(undone_remove.kind if undone_remove else "", "remove_item")
        first_items = self.store.get(first.id)
        self.assertEqual(first_items.item_ids if first_items else (), ("a", "b", "c"))
        undone_add = self.store.undo()
        self.assertEqual(undone_add.kind if undone_add else "", "add_item")
        first_items = self.store.get(first.id)
        self.assertEqual(first_items.item_ids if first_items else (), ("a", "b"))

        redone_add = self.store.redo()
        self.assertEqual(redone_add.kind if redone_add else "", "add_item")
        first_items = self.store.get(first.id)
        self.assertEqual(first_items.item_ids if first_items else (), ("a", "b", "c"))
        self.store.redo()
        self.store.redo()
        self.assertEqual([item.id for item in self.store.list()], [first.id])
        self.assertIsNone(self.store.get(second.id))

        raw = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(len(raw["undo_history"]), 7)
        self.assertEqual(raw["redo_history"], [])

    def test_undoable_mutations_clear_redo_and_rename_is_reversible(self) -> None:
        collection = self.store.create("灵感")
        other = self.store.create("参考")
        self.store.add_item(collection.id, "a", item_name="alpha.png")
        self.assertEqual(self.store.snapshot()[1]["undo_label"], "收藏「alpha.png」到「灵感」")
        self.store.undo()
        self.assertTrue(self.store.snapshot()[1]["can_redo"])
        self.assertEqual(self.store.snapshot()[1]["redo_label"], "收藏「alpha.png」到「灵感」")

        self.store.add_item(collection.id, "b")
        _, history = self.store.snapshot()
        self.assertFalse(history["can_redo"])
        self.assertEqual(history["undo_label"], "收藏到「灵感」")
        self.assertFalse(self.store.add_item(collection.id, "b")[1])
        self.assertFalse(self.store.remove_item(collection.id, "missing")[1])
        self.assertEqual(self.store.snapshot()[1]["undo_label"], "收藏到「灵感」")

        self.store.update(collection.id, name="角色")
        self.store.reorder([other.id, collection.id])
        _, after_rename = self.store.snapshot()
        self.assertEqual(after_rename["undo_label"], "调整收藏夹顺序")
        self.assertEqual([item.id for item in self.store.list()], [other.id, collection.id])
        renamed = self.store.get(collection.id)
        self.assertEqual(renamed.item_ids if renamed else (), ("b",))
        self.assertEqual(renamed.name if renamed else "", "角色")

        undone = CollectionStore(self.path).undo()
        self.assertEqual(undone.kind if undone else "", "reorder_collections")
        self.assertEqual([item.id for item in self.store.list()], [collection.id, other.id])
        undone = self.store.undo()
        self.assertEqual(undone.kind if undone else "", "rename_collection")
        restored = self.store.get(collection.id)
        self.assertEqual(restored.name if restored else "", "灵感")
        redone = self.store.redo()
        self.assertEqual(redone.kind if redone else "", "rename_collection")
        renamed_again = self.store.get(collection.id)
        self.assertEqual(renamed_again.name if renamed_again else "", "角色")
        redone = self.store.redo()
        self.assertEqual(redone.kind if redone else "", "reorder_collections")
        self.assertEqual([item.id for item in self.store.list()], [other.id, collection.id])

    def test_history_is_capped_and_skips_corrupt_entries(self) -> None:
        first = self.store.create("保留")
        for index in range(MAX_COLLECTION_HISTORY + 5):
            extra = self.store.create(f"临时 {index}")
            self.store.delete(extra.id)

        _, history = self.store.snapshot()
        self.assertTrue(history["can_undo"])
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(len(raw["undo_history"]), MAX_COLLECTION_HISTORY)
        self.assertEqual(len(self.store.list()), 1)

        restored_count = 0
        while self.store.undo() is not None:
            restored_count += 1
        self.assertEqual(restored_count, MAX_COLLECTION_HISTORY)
        self.assertEqual(len(self.store.list()), 1)
        self.assertEqual(self.store.list()[0].id, first.id)

        self.path.write_text(
            json.dumps(
                {
                    "collections": [
                        {"id": "keep", "name": "有效", "item_ids": []},
                    ],
                    "undo_history": [
                        {"kind": "nope", "collection_id": "keep", "name": "有效", "index": 0},
                        {
                            "kind": "delete_collection",
                            "collection_id": "gone",
                            "name": "已删",
                            "index": 1,
                            "item_ids": ["a"],
                        },
                        {"kind": "add_item", "collection_id": "keep", "name": "有效", "index": 0},
                    ],
                    "redo_history": "bad",
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        _, loaded = CollectionStore(self.path).snapshot()
        self.assertTrue(loaded["can_undo"])
        self.assertEqual(loaded["undo_label"], "删除「已删」")
        self.assertFalse(loaded["can_redo"])
        undone = CollectionStore(self.path).undo()
        self.assertIsNotNone(undone)
        self.assertEqual([item.id for item in CollectionStore(self.path).list()], ["keep", "gone"])


if __name__ == "__main__":
    unittest.main()
