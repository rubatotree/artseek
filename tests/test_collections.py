from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path

from artseek.collections import CollectionStore


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

    def test_corrupt_and_duplicate_on_disk_data_is_safe_to_load(self) -> None:
        self.path.write_text("not json", encoding="utf-8")
        self.assertEqual(self.store.list(), [])

        self.path.write_text(
            json.dumps(
                {
                    "collections": [
                        {"id": "one", "name": "  保留  ", "item_ids": ["a", "a", 7, "b"]},
                        {"id": "one", "name": "忽略", "item_ids": []},
                        {"id": "bad", "name": "", "item_ids": []},
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

    def test_pinned_state_is_backward_compatible_and_survives_updates(self) -> None:
        self.path.write_text(
            json.dumps(
                {
                    "collections": [
                        {"id": "older", "name": "旧格式", "item_ids": []},
                        {"id": "pinned", "name": "固定", "item_ids": [], "pinned": True},
                        {"id": "newer", "name": "新建", "item_ids": []},
                    ]
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        self.assertEqual([item.id for item in self.store.list()], ["pinned", "older", "newer"])
        self.assertFalse(self.store.get("older").pinned)
        self.assertTrue(self.store.set_pinned("older", True).pinned)
        self.store.rename("older", "已重命名")
        self.store.add_item("older", "image")
        self.store.remove_item("older", "image")

        restored = CollectionStore(self.path).get("older")
        self.assertEqual(restored.name if restored else "", "已重命名")
        self.assertTrue(restored.pinned if restored else False)
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        older = next(item for item in raw["collections"] if item["id"] == "older")
        self.assertTrue(older["pinned"])

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


if __name__ == "__main__":
    unittest.main()
