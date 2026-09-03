from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from artseek.search_cache import (
    MAX_SNAPSHOTS,
    SearchSnapshotStore,
    is_snapshot_id,
    sanitize_reference,
)


class SearchSnapshotStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.store = SearchSnapshotStore(Path(self._temporary_directory.name) / "snaps")

    def tearDown(self) -> None:
        self._temporary_directory.cleanup()

    def _create(self, *, query: str = "白袜", extra: dict[str, object] | None = None) -> str:
        kwargs = dict(
            query=query,
            text_weight=1.0,
            references=[{"id": "img-1", "weight": 1.2}],
            expand=False,
            strict_tags=False,
            dedupe=True,
            limit=40,
            folder_keys=["D:/pics"],
            mtime_after=None,
            mtime_before=None,
            query_plan={
                "original": query,
                "positive_tags": ["white_socks"],
                "negative_tags": [],
            },
            llm_warning=None,
            results=[
                {
                    "id": "hit-1",
                    "score": 100,
                    "name": "a.png",
                    "path": "a.png",
                    "width": 8,
                    "height": 8,
                    "tags": [],
                    "tag_hits": ["white_socks"],
                }
            ],
            has_more=True,
            had_file_refs=False,
        )
        if extra:
            kwargs.update(extra)
        return self.store.create(**kwargs)

    def test_create_round_trip_and_append(self) -> None:
        snapshot_id = self._create()
        self.assertTrue(is_snapshot_id(snapshot_id))
        loaded = self.store.get(snapshot_id)
        assert loaded is not None
        self.assertEqual(loaded["query"], "白袜")
        self.assertEqual(loaded["references"], [{"id": "img-1", "weight": 1.2}])
        self.assertEqual(loaded["results"][0]["id"], "hit-1")
        self.assertTrue(loaded["has_more"])
        self.assertTrue(
            self.store.append(
                snapshot_id,
                [{"id": "hit-2", "score": 80, "name": "b.png"}],
                has_more=False,
            )
        )
        again = self.store.get(snapshot_id)
        assert again is not None
        self.assertEqual([row["id"] for row in again["results"]], ["hit-1", "hit-2"])
        self.assertFalse(again["has_more"])

    def test_missing_and_invalid_ids(self) -> None:
        self.assertIsNone(self.store.get("not-a-snapshot"))
        self.assertIsNone(self.store.get("ab" * 6))
        self.assertFalse(self.store.append("0" * 12, [{"id": "x"}], has_more=False))
        self.assertIsNone(sanitize_reference({"weight": 1}))

    def test_evicts_oldest_beyond_cap(self) -> None:
        ids = [self._create(query=f"q{index}") for index in range(MAX_SNAPSHOTS + 3)]
        remaining = [snapshot_id for snapshot_id in ids if self.store.get(snapshot_id) is not None]
        self.assertEqual(len(remaining), MAX_SNAPSHOTS)
        self.assertIsNone(self.store.get(ids[0]))
        self.assertIsNotNone(self.store.get(ids[-1]))
