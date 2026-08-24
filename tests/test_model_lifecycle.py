from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from artseek.config import (
    DEFAULT_AUTO_UNLOAD_SECONDS,
    AppConfig,
    WebSearchSettings,
    clamp_auto_unload_seconds,
    load_auto_unload_seconds,
    load_web_search_settings,
    save_auto_unload_seconds,
    save_web_search_settings,
)
from artseek.image_io import canonical_path
from artseek.query import static_expand
from artseek.search import SearchEngine
from artseek.store import Neighbor

LIB = Path("lib")
WIDE = LIB / "wide"


class RecordingEmbedder:
    dimension = 3

    def __init__(self) -> None:
        self.closed = False
        self.text_calls = 0

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        self.text_calls += 1
        return np.array([[1.0, 0.0, 0.0] for _ in texts], dtype=np.float32)

    def encode_images(self, images: list) -> np.ndarray:
        del images
        return np.array([[0.0, 1.0, 0.0]], dtype=np.float32)

    def close(self) -> None:
        self.closed = True


class DummyStore:
    def __init__(self) -> None:
        self.bound_dimension: int | None = None
        self.payloads = {
            "a": {
                "path": str(WIDE / "a.png"),
                "name": "a.png",
                "width": 100,
                "height": 200,
                "root": str(LIB),
                "folder_key": canonical_path(WIDE),
                "tags": ["wide_sleeves"],
            }
        }

    def known_dimension(self) -> int | None:
        return None

    def ensure_collection(self, dimension: int) -> None:
        self.bound_dimension = dimension

    def close(self) -> None:
        return

    def query(
        self,
        vector: list[float],
        using: str,
        limit: int,
        point_ids: list[str] | None = None,
        universe: int | None = None,
    ) -> list[Neighbor]:
        del vector, using, limit, point_ids, universe
        payload = self.payloads["a"]
        return [Neighbor("a", 0.9, payload)]


class DummyLLM:
    def __init__(self) -> None:
        self.unloads = 0
        self.keep_alive: int | None = None

    def expand_query(self, query: str, *, keep_alive: int | None = None):
        self.keep_alive = keep_alive
        return static_expand(query)

    def unload(self) -> None:
        self.unloads += 1

    def preload(self) -> None:
        return


def make_engine(
    tmp: Path,
    *,
    factory=None,
    auto_unload_seconds: int = 600,
) -> tuple[SearchEngine, RecordingEmbedder]:
    embedder = RecordingEmbedder()
    cfg = AppConfig(data_dir=tmp, cache_dir=tmp / "cache")
    cfg.ensure_dirs()
    cfg.llm.enabled = False
    engine = SearchEngine(
        cfg,
        store=DummyStore(),
        embedder_factory=factory or (lambda: embedder),
        llm=DummyLLM(),
        auto_unload_seconds=auto_unload_seconds,
    )
    return engine, embedder


class RuntimeSettingsTests(unittest.TestCase):
    def test_clamp_rejects_negative_and_caps_day(self) -> None:
        self.assertEqual(clamp_auto_unload_seconds(-3), 0)
        self.assertEqual(clamp_auto_unload_seconds(99_999), 86400)
        self.assertEqual(clamp_auto_unload_seconds("nope"), DEFAULT_AUTO_UNLOAD_SECONDS)

    def test_round_trip_runtime_file(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            cfg = AppConfig(data_dir=Path(raw), cache_dir=Path(raw) / "cache")
            cfg.ensure_dirs()
            self.assertEqual(load_auto_unload_seconds(cfg), DEFAULT_AUTO_UNLOAD_SECONDS)
            save_auto_unload_seconds(cfg, 120)
            self.assertEqual(load_auto_unload_seconds(cfg), 120)

            settings = WebSearchSettings(use_llm=False, strict_tags=True, dedupe=False, limit=52)
            save_web_search_settings(cfg, settings)
            self.assertEqual(load_auto_unload_seconds(cfg), 120)
            self.assertEqual(load_web_search_settings(cfg), settings)


class LazySearchEngineTests(unittest.TestCase):
    def test_init_does_not_construct_embedder(self) -> None:
        created = {"count": 0}

        def factory() -> RecordingEmbedder:
            created["count"] += 1
            return RecordingEmbedder()

        with tempfile.TemporaryDirectory() as raw:
            engine, _ = make_engine(Path(raw), factory=factory)
            self.assertIsNone(engine.embedder)
            self.assertEqual(created["count"], 0)
            status = engine.model_status()
            self.assertFalse(status["loaded"])
            self.assertEqual(status["auto_unload_minutes"], 10)

    def test_search_loads_then_unload_releases(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            engine, embedder = make_engine(Path(raw))
            response = engine.search("姬袖", expand=False, limit=5)
            self.assertEqual(embedder.text_calls, 1)
            self.assertIs(engine.embedder, embedder)
            self.assertTrue(engine.model_status()["loaded"])
            self.assertEqual([item.id for item in response.results], ["a"])
            self.assertTrue(engine.unload_model())
            self.assertIsNone(engine.embedder)
            self.assertTrue(embedder.closed)
            self.assertEqual(engine.llm.unloads, 1)

    def test_idle_unload_skips_while_held(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            engine, embedder = make_engine(Path(raw), auto_unload_seconds=10)
            with engine.hold_model():
                self.assertIs(engine.embedder, embedder)
                engine._last_used_at = 0
                engine._idle_unload()
                self.assertIs(engine.embedder, embedder)
                self.assertFalse(embedder.closed)
            engine._last_used_at = 0
            engine._idle_unload()
            self.assertIsNone(engine.embedder)
            self.assertTrue(embedder.closed)

    def test_zero_auto_unload_does_not_arm_timer(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            engine, embedder = make_engine(Path(raw), auto_unload_seconds=0)
            engine.load_model()
            self.assertIs(engine.embedder, embedder)
            self.assertIsNone(engine._idle_timer)
            engine.set_auto_unload_seconds(600)
            self.assertIsNotNone(engine._idle_timer)
            engine.set_auto_unload_seconds(0)
            self.assertIsNone(engine._idle_timer)

    def test_load_binds_collection_dimension(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            engine, _ = make_engine(Path(raw))
            engine.load_model()
            self.assertEqual(engine.store.bound_dimension, 3)


class ReleaseAcceleratorTests(unittest.TestCase):
    def test_release_collects_cuda_cache_when_available(self) -> None:
        from artseek.compute import release_accelerator

        with (
            patch("artseek.compute.gc.collect") as collect,
            patch("artseek.compute.torch.cuda.is_available", return_value=True),
            patch("artseek.compute.torch.cuda.empty_cache") as empty,
            patch("artseek.compute.torch.cuda.ipc_collect") as ipc,
        ):
            release_accelerator()
        collect.assert_called_once()
        empty.assert_called_once()
        ipc.assert_called_once()


if __name__ == "__main__":
    unittest.main()
