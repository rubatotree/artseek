"""Reading the library must stay possible while an index job is queued or preparing.

The index gate used to be `if index_job.running`, and `running` flips to True the
moment a job is submitted, so the whole waterfall and search stayed blocked for
the entire queue + prepare + scan window. Reads are safe (SQLite WAL snapshots and
a locked VectorIndex), so only mutations that would interleave with the scan may
still answer 409.
"""

from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from artseek.config import AppConfig
from artseek.image_io import canonical_path


class _FakeEmbedder:
    dimension = 8

    def encode_images(self, images: list[Any]) -> np.ndarray:
        out = []
        for idx, img in enumerate(images):
            arr = np.asarray(img, dtype=np.float32)
            mean_val = float(arr.mean()) / 255.0 if arr.size else 0.1
            vec = np.array(
                [mean_val + (i + 1) * 0.05 + idx * 0.01 for i in range(self.dimension)],
                dtype=np.float32,
            )
            vec /= max(float(np.linalg.norm(vec)), 1e-12)
            out.append(vec)
        return (
            np.stack(out, axis=0)
            if out
            else np.zeros((0, self.dimension), dtype=np.float32)
        )

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        out = []
        for text in texts:
            seed = sum(ord(ch) for ch in text) % 97
            vec = np.array(
                [(seed + i + 1) * 0.03 for i in range(self.dimension)],
                dtype=np.float32,
            )
            vec /= max(float(np.linalg.norm(vec)), 1e-12)
            out.append(vec)
        return (
            np.stack(out, axis=0)
            if out
            else np.zeros((0, self.dimension), dtype=np.float32)
        )

    def close(self) -> None:
        pass


def _save_test_image(path: Path, size: tuple[int, int], color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path, format="JPEG", quality=90)


class ReadWhileIndexingTests(unittest.TestCase):
    def setUp(self) -> None:
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            self.skipTest("fastapi 未安装")

        from artseek.api.app import create_app

        self._tmp = tempfile.TemporaryDirectory()
        self.base = Path(self._tmp.name)
        self.root = self.base / "pics"
        _save_test_image(self.root / "a.jpg", (320, 480), (30, 90, 190))
        _save_test_image(self.root / "nested" / "b.jpg", (480, 320), (190, 70, 40))

        self.config = AppConfig(
            data_dir=self.base / "data",
            cache_dir=self.base / "cache",
        )
        self.config.llm.enabled = False
        self.config.tagger.enabled = False
        self.config.ensure_dirs()

        self.app = create_app(self.config, embedder_factory=_FakeEmbedder)  # type: ignore[arg-type]
        self.container = self.app.state.container
        self.folder_key = canonical_path(self.root)
        self.nested_key = canonical_path(self.root / "nested")
        self.client = TestClient(self.app).__enter__()

        response = self.client.post(
            "/api/index",
            json={"path": str(self.root), "recursive": True, "auto_enrich": False},
        )
        self.assertEqual(response.status_code, 200)
        # Poll the container instead of /api/index-status on purpose: that route
        # drains the pending provenance refresh into its own thread, which would
        # still be writing when the temporary database is torn down.
        deadline = time.time() + 20.0
        while time.time() < deadline and self.container.index_job.running:
            time.sleep(0.05)
        self.assertFalse(self.container.index_job.running)
        self.assertIsNone(self.container.index_job.error)

    def tearDown(self) -> None:
        self.client.__exit__(None, None, None)
        self._tmp.cleanup()

    def _set_index_phase(self, message: str) -> None:
        self.container.index_job.running = True
        self.container.index_job.message = message

    def _assert_reads_ok(self) -> None:
        scope = self.client.post(
            "/api/scope-images",
            json={"folder_keys": [self.folder_key, self.nested_key], "limit": 10},
        )
        self.assertEqual(scope.status_code, 200)
        self.assertEqual(len(scope.json()["results"]), 2)

        random_page = self.client.post(
            "/api/random-browse",
            json={"folder_keys": [self.folder_key, self.nested_key], "limit": 2, "offset": 0},
        )
        self.assertEqual(random_page.status_code, 200)
        self.assertTrue(random_page.json()["results"])

        search = self.client.post(
            "/api/search",
            json={"query": "blue archive", "expand": False, "limit": 5},
        )
        self.assertEqual(search.status_code, 200)

        local = self.client.post(
            "/api/local-search",
            json={"query": "blue_archive", "limit": 5},
        )
        self.assertEqual(local.status_code, 200)

    def test_reads_work_while_index_is_queued_and_preparing(self) -> None:
        for message in ("正在排队…", "正在准备索引…", "正在索引 1/2 (pics)"):
            with self.subTest(message=message):
                self._set_index_phase(message)
                self._assert_reads_ok()

    def test_mutations_are_still_gated_while_indexing(self) -> None:
        self._set_index_phase("正在索引 1/2 (pics)")
        removed = self.client.post(
            "/api/index-remove",
            json={"folder_key": self.folder_key},
        )
        self.assertEqual(removed.status_code, 409)
        queued = self.client.post("/api/jobs/retag", json={})
        self.assertEqual(queued.status_code, 409)


if __name__ == "__main__":
    unittest.main()
