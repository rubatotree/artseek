from __future__ import annotations

import unittest

import numpy as np

from artseek.config import AppConfig
from artseek.search import SearchEngine
from artseek.store import Neighbor


class FakeEmbedder:
    def encode_texts(self, texts: list[str]) -> np.ndarray:
        return np.array([[1.0, 0.0, 0.0] for _ in texts], dtype=np.float32)


class FakeStore:
    def query(self, vector: list[float], using: str, limit: int) -> list[Neighbor]:
        del vector, limit
        payloads = {
            "a": {
                "path": "a.png",
                "name": "a.png",
                "width": 100,
                "height": 200,
                "tags": ["wide_sleeves", "sitting"],
            },
            "b": {
                "path": "b.png",
                "name": "b.png",
                "width": 100,
                "height": 200,
                "tags": ["wide_sleeves", "standing"],
            },
            "c": {
                "path": "c.png",
                "name": "c.png",
                "width": 100,
                "height": 200,
                "tags": ["shirt"],
            },
        }
        order = ["a", "b", "c"] if using == "visual" else ["c", "a", "b"]
        return [Neighbor(key, 0.9 - index * 0.1, payloads[key]) for index, key in enumerate(order)]


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


if __name__ == "__main__":
    unittest.main()

