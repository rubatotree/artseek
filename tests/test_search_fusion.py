from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from artseek.config import AppConfig
from artseek.image_io import canonical_path
from artseek.library import day_start_ns
from artseek.search import (
    SearchEngine,
    compose_visual_vector,
    duplicate_key,
    duplicate_keys,
    keep_first_unique,
)
from artseek.store import Neighbor, query_mode

LIB = Path("lib")
WIDE = LIB / "wide"
SHIRTS = LIB / "shirts"
MTIME_OLD = day_start_ns("2020-01-15") or 0
MTIME_NEW = day_start_ns("2024-06-01") or 0


class FakeEmbedder:
    def encode_texts(self, texts: list[str]) -> np.ndarray:
        return np.array([[1.0, 0.0, 0.0] for _ in texts], dtype=np.float32)

    def encode_images(self, images: list) -> np.ndarray:
        rows = []
        for index, _image in enumerate(images):
            if index == 0:
                rows.append([0.0, 1.0, 0.0])
            else:
                rows.append([0.0, 0.0, 1.0])
        return np.array(rows, dtype=np.float32)


class FakeStore:
    payloads = {
        "a": {
            "path": str(WIDE / "a.png"),
            "name": "a.png",
            "width": 100,
            "height": 200,
            "root": str(LIB),
            "folder_key": canonical_path(WIDE),
            "mtime_ns": MTIME_OLD,
            "size_bytes": 111,
            "tags": ["wide_sleeves", "sitting"],
        },
        "b": {
            "path": str(WIDE / "b.png"),
            "name": "b.png",
            "width": 100,
            "height": 200,
            "root": str(LIB),
            "folder_key": canonical_path(WIDE),
            "mtime_ns": MTIME_NEW,
            "size_bytes": 222,
            "tags": ["wide_sleeves", "standing"],
        },
        "c": {
            "path": str(SHIRTS / "c.png"),
            "name": "c.png",
            "width": 100,
            "height": 200,
            "root": str(LIB),
            "folder_key": canonical_path(SHIRTS),
            "mtime_ns": MTIME_NEW,
            "size_bytes": 333,
            "tags": ["shirt"],
        },
    }

    def iter_payloads(self):
        yield from self.payloads.items()

    def retrieve_payloads(self, ids: list[str]) -> dict:
        return {item_id: self.payloads[item_id] for item_id in ids if item_id in self.payloads}

    def query(
        self,
        vector: list[float],
        using: str,
        limit: int,
        point_ids: list[str] | None = None,
        universe: int | None = None,
    ) -> list[Neighbor]:
        del vector, limit, universe
        order = ["a", "b", "c"] if using == "visual" else ["c", "a", "b"]
        if point_ids is not None:
            allowed = set(point_ids)
            order = [key for key in order if key in allowed]
        return [
            Neighbor(key, 0.9 - index * 0.1, self.payloads[key])
            for index, key in enumerate(order)
        ]


class DuplicateStore(FakeStore):
    payloads = {
        **FakeStore.payloads,
        "a_copy": {
            **FakeStore.payloads["a"],
            "path": str(WIDE / "a_copy.png"),
            "name": "a_copy.png",
        },
    }

    def query(
        self,
        vector: list[float],
        using: str,
        limit: int,
        point_ids: list[str] | None = None,
        universe: int | None = None,
    ) -> list[Neighbor]:
        del vector, limit, universe
        order = ["a", "a_copy", "b", "c"] if using == "visual" else ["c", "a", "a_copy", "b"]
        if point_ids is not None:
            allowed = set(point_ids)
            order = [key for key in order if key in allowed]
        return [
            Neighbor(key, 0.9 - index * 0.1, self.payloads[key])
            for index, key in enumerate(order)
        ]


class HashDuplicateStore(DuplicateStore):
    payloads = {
        **DuplicateStore.payloads,
        "a": {**DuplicateStore.payloads["a"], "content_hash": "aa" * 32},
        "a_copy": {
            **DuplicateStore.payloads["a_copy"],
            "content_hash": "AA" * 32,
            "size_bytes": 999,
            "width": 12,
            "height": 12,
        },
    }


class SearchFusionTests(unittest.TestCase):
    def test_strict_positive_and_negative_tags(self) -> None:
        engine = object.__new__(SearchEngine)
        engine.config = AppConfig()
        engine.config.llm.enabled = False
        engine.embedder = FakeEmbedder()
        engine.store = FakeStore()

        response = engine.search("姬袖 -站立", expand=False, strict_tags=True, limit=10)
        self.assertEqual([item.id for item in response.results], ["a"])
        self.assertEqual(response.results[0].tag_hits, ["wide_sleeves"])

    def test_folder_keys_filter_candidates(self) -> None:
        engine = object.__new__(SearchEngine)
        engine.config = AppConfig()
        engine.config.llm.enabled = False
        engine.embedder = FakeEmbedder()
        engine.store = FakeStore()

        response = engine.search(
            "衬衫", expand=False, folder_keys=[canonical_path(SHIRTS)], limit=10
        )
        self.assertEqual([item.id for item in response.results], ["c"])

        empty = engine.search("衬衫", expand=False, folder_keys=[], limit=10)
        self.assertEqual(empty.results, [])

    def test_restrict_ids_filters_candidates_outside_a_folder_scope(self) -> None:
        engine = object.__new__(SearchEngine)
        engine.config = AppConfig()
        engine.config.llm.enabled = False
        engine.embedder = FakeEmbedder()
        engine.store = FakeStore()

        restricted = engine.search(
            "姬袖", expand=False, restrict_ids=["b"], limit=10
        )
        self.assertEqual([item.id for item in restricted.results], ["b"])

        empty = engine.search(
            "姬袖", expand=False, restrict_ids=[], limit=10
        )
        self.assertEqual(empty.results, [])

    def test_mtime_range_filters_candidates(self) -> None:
        engine = object.__new__(SearchEngine)
        engine.config = AppConfig()
        engine.config.llm.enabled = False
        engine.embedder = FakeEmbedder()
        engine.store = FakeStore()

        wide = canonical_path(WIDE)
        recent = engine.search(
            "姬袖",
            expand=False,
            folder_keys=[wide],
            mtime_after="2023-01-01",
            limit=10,
        )
        self.assertEqual([item.id for item in recent.results], ["b"])

        older = engine.search(
            "姬袖",
            expand=False,
            folder_keys=[wide],
            mtime_before="2021-01-01",
            limit=10,
        )
        self.assertEqual([item.id for item in older.results], ["a"])

    def test_exclude_ids_does_not_reshuffle_loaded(self) -> None:
        engine = object.__new__(SearchEngine)
        engine.config = AppConfig()
        engine.config.llm.enabled = False
        engine.embedder = FakeEmbedder()
        engine.store = FakeStore()

        first = engine.search("姬袖", expand=False, limit=1)
        second = engine.search(
            "姬袖",
            expand=False,
            limit=1,
            exclude_ids=[first.results[0].id],
        )
        self.assertEqual(first.results[0].id, "a")
        self.assertEqual(second.results[0].id, "b")
        self.assertNotIn("a", [item.id for item in second.results])

    def test_offset_returns_next_page(self) -> None:
        engine = object.__new__(SearchEngine)
        engine.config = AppConfig()
        engine.config.llm.enabled = False
        engine.embedder = FakeEmbedder()
        engine.store = FakeStore()

        first = engine.search("姬袖", expand=False, limit=1, offset=0)
        second = engine.search("姬袖", expand=False, limit=1, offset=1)
        third = engine.search("姬袖", expand=False, limit=1, offset=2)
        done = engine.search("姬袖", expand=False, limit=1, offset=3)
        self.assertEqual([item.id for item in first.results], ["a"])
        self.assertEqual([item.id for item in second.results], ["b"])
        self.assertEqual([item.id for item in third.results], ["c"])
        self.assertEqual(done.results, [])
        self.assertTrue(first.has_more)
        self.assertFalse(done.has_more)

    def test_duplicate_key_prefers_content_hash(self) -> None:
        hashed = duplicate_key({"content_hash": "ABC", "size_bytes": 1, "width": 2, "height": 3})
        sized = duplicate_key({"size_bytes": 111, "width": 100, "height": 200})
        self.assertEqual(hashed, ("h", "abc"))
        self.assertEqual(sized, ("s", 111, 100, 200))
        self.assertIsNone(duplicate_key({"width": 100, "height": 200}))
        self.assertEqual(
            duplicate_keys(
                {
                    "content_hash": "AB" * 32,
                    "phash": "FF" * 8,
                    "size_bytes": 9,
                    "width": 1,
                    "height": 1,
                }
            ),
            (("h", "ab" * 32), ("p", "ff" * 8)),
        )

    def test_keep_first_unique_drops_later_copies(self) -> None:
        payload_a = {"size_bytes": 111, "width": 100, "height": 200}
        payload_b = {"size_bytes": 222, "width": 100, "height": 200}
        ranked = [
            ("a", {"payload": payload_a}, []),
            ("a_copy", {"payload": payload_a}, []),
            ("b", {"payload": payload_b}, []),
        ]
        kept = keep_first_unique(ranked)
        self.assertEqual([row[0] for row in kept], ["a", "b"])

    def test_keep_first_unique_drops_same_content_or_perceptual_hash(self) -> None:
        digest = "ab" * 32
        ranked = [
            (
                "a",
                {"payload": {"content_hash": digest, "size_bytes": 10, "width": 8, "height": 8}},
                [],
            ),
            (
                "a_copy",
                {
                    "payload": {
                        "content_hash": digest.upper(),
                        "size_bytes": 99,
                        "width": 64,
                        "height": 64,
                    }
                },
                [],
            ),
            (
                "b",
                {"payload": {"phash": "aa" * 8, "size_bytes": 11, "width": 9, "height": 9}},
                [],
            ),
            (
                "b_copy",
                {"payload": {"phash": "AA" * 8, "size_bytes": 12, "width": 16, "height": 16}},
                [],
            ),
            (
                "c",
                {"payload": {"content_hash": "cd" * 32, "size_bytes": 10, "width": 8, "height": 8}},
                [],
            ),
        ]
        kept = keep_first_unique(ranked)
        self.assertEqual([row[0] for row in kept], ["a", "b", "c"])

    def test_dedupe_keeps_first_identical_hit(self) -> None:
        engine = object.__new__(SearchEngine)
        engine.config = AppConfig()
        engine.config.llm.enabled = False
        engine.embedder = FakeEmbedder()
        engine.store = DuplicateStore()

        all_hits = engine.search("姬袖", expand=False, limit=10)
        unique = engine.search("姬袖", expand=False, limit=10, dedupe=True)
        all_ids = [item.id for item in all_hits.results]
        unique_ids = [item.id for item in unique.results]
        self.assertEqual(all_ids[0], "a")
        self.assertIn("a_copy", all_ids)
        self.assertEqual(unique_ids[0], "a")
        self.assertNotIn("a_copy", unique_ids)
        self.assertEqual(set(unique_ids), {"a", "b", "c"})

    def test_dedupe_skips_copies_of_already_shown(self) -> None:
        engine = object.__new__(SearchEngine)
        engine.config = AppConfig()
        engine.config.llm.enabled = False
        engine.embedder = FakeEmbedder()
        engine.store = DuplicateStore()

        unique = engine.search("姬袖", expand=False, limit=10, dedupe=True)
        first = engine.search("姬袖", expand=False, limit=1, dedupe=True)
        second = engine.search(
            "姬袖",
            expand=False,
            limit=1,
            dedupe=True,
            exclude_ids=[first.results[0].id],
        )
        self.assertEqual(first.results[0].id, unique.results[0].id)
        self.assertEqual(second.results[0].id, unique.results[1].id)
        self.assertNotIn("a_copy", [item.id for item in second.results])

    def test_dedupe_drops_same_content_hash_despite_size(self) -> None:
        engine = object.__new__(SearchEngine)
        engine.config = AppConfig()
        engine.config.llm.enabled = False
        engine.embedder = FakeEmbedder()
        engine.store = HashDuplicateStore()

        unique = engine.search("姬袖", expand=False, limit=10, dedupe=True)
        self.assertEqual([item.id for item in unique.results], ["a", "b", "c"])

    def test_query_mode_avoids_large_has_id(self) -> None:
        self.assertEqual(query_mode(0, 24541), "empty")
        self.assertEqual(query_mode(24541, 24541), "all")
        self.assertEqual(query_mode(20000, 24541), "postfilter")
        self.assertEqual(query_mode(80, 24541), "subset")


class ComposeVisualTests(unittest.TestCase):
    def test_equal_weights_normalize(self) -> None:
        text = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        images = [np.array([0.0, 1.0, 0.0], dtype=np.float32)]
        out = compose_visual_vector(text, images, text_weight=1.0, image_weights=[1.0])
        np.testing.assert_allclose(
            out, np.array([0.70710678, 0.70710678, 0.0], dtype=np.float32), atol=1e-5
        )

    def test_image_weight_scales_contribution(self) -> None:
        text = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        images = [
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
            np.array([0.0, 0.0, 1.0], dtype=np.float32),
        ]
        out = compose_visual_vector(text, images, text_weight=1.0, image_weights=[1.0, 2.0])
        expected = np.array([1.0, 1.0, 2.0], dtype=np.float32)
        expected /= float(np.linalg.norm(expected))
        np.testing.assert_allclose(out, expected, atol=1e-5)

    def test_zero_text_weight_uses_images_only(self) -> None:
        text = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        images = [np.array([0.0, 1.0, 0.0], dtype=np.float32)]
        out = compose_visual_vector(text, images, text_weight=0.0, image_weights=[1.0])
        np.testing.assert_allclose(out, np.array([0.0, 1.0, 0.0], dtype=np.float32), atol=1e-5)

    def test_rejects_all_zero_weights(self) -> None:
        with self.assertRaises(ValueError):
            compose_visual_vector(
                np.array([1.0, 0.0], dtype=np.float32),
                [np.array([0.0, 1.0], dtype=np.float32)],
                text_weight=0.0,
                image_weights=[0.0],
            )


class MultiReferenceSearchTests(unittest.TestCase):
    def test_search_accepts_multiple_images(self) -> None:
        class RecordingStore(FakeStore):
            def query(self, vector, using, limit, point_ids=None, universe=None):
                if using == "visual":
                    self.last_visual = vector
                return super().query(vector, using, limit, point_ids, universe)

        engine = object.__new__(SearchEngine)
        engine.config = AppConfig()
        engine.config.llm.enabled = False
        engine.embedder = FakeEmbedder()
        engine.store = RecordingStore()
        engine.search(
            "姬袖",
            expand=False,
            images=[WIDE / "a.png", WIDE / "b.png"],
            image_weights=[1.0, 2.0],
            text_weight=1.0,
            limit=10,
        )
        expected = np.array([1.0, 1.0, 2.0], dtype=np.float32)
        expected /= float(np.linalg.norm(expected))
        np.testing.assert_allclose(engine.store.last_visual, expected, atol=1e-5)


class WarmupTests(unittest.TestCase):
    def test_warmup_uses_embedder_hook_when_llm_disabled(self) -> None:
        engine = object.__new__(SearchEngine)
        engine.config = AppConfig()
        engine.config.llm.enabled = False

        class WarmEmbedder:
            def __init__(self) -> None:
                self.calls = 0

            def warmup(self) -> None:
                self.calls += 1

        engine.embedder = WarmEmbedder()
        messages: list[str] = []
        engine.warmup(status=messages.append)
        self.assertEqual(engine.embedder.calls, 1)
        self.assertTrue(any("预热" in item for item in messages))

    def test_warmup_preloads_query_model(self) -> None:
        engine = object.__new__(SearchEngine)
        engine.config = AppConfig()
        engine.config.llm.enabled = True

        class WarmEmbedder:
            def warmup(self) -> None:
                return

        class FakeLLM:
            def __init__(self) -> None:
                self.preloaded = False

            def preload(self) -> None:
                self.preloaded = True

        engine.embedder = WarmEmbedder()
        engine.llm = FakeLLM()
        engine.warmup()
        self.assertTrue(engine.llm.preloaded)

    def test_warmup_continues_if_ollama_missing(self) -> None:
        from artseek.llm import OllamaUnavailable

        engine = object.__new__(SearchEngine)
        engine.config = AppConfig()

        class WarmEmbedder:
            def warmup(self) -> None:
                return

        class FakeLLM:
            def preload(self) -> None:
                raise OllamaUnavailable("Ollama 未运行")

        engine.embedder = WarmEmbedder()
        engine.llm = FakeLLM()
        messages: list[str] = []
        engine.warmup(status=messages.append)
        self.assertTrue(any("内置词典" in item for item in messages))


if __name__ == "__main__":
    unittest.main()

