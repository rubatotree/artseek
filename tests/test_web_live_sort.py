"""`sort=indexed` must order by when a row entered the library, not by file mtime.

A freshly indexed folder often holds old files: their mtime puts them deep in the
"最近更新" order, which is why the live gallery needs its own sort key.
"""

from __future__ import annotations

import os
import tempfile
import time
import unittest
from pathlib import Path

from PIL import Image

from artseek.config import AppConfig
from artseek.image_io import canonical_path


def _save(path: Path, color: tuple[int, int, int], mtime: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (320, 480), color).save(path, format="JPEG", quality=90)
    os.utime(path, (mtime, mtime))


class IndexedSortTests(unittest.TestCase):
    def setUp(self) -> None:
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            self.skipTest("fastapi 未安装")

        from artseek.api.app import create_app
        from tests.test_live_index_visibility import _FakeEmbedder

        self._tmp = tempfile.TemporaryDirectory()
        self.base = Path(self._tmp.name)
        self.root = self.base / "pics"
        self.config = AppConfig(
            data_dir=self.base / "data",
            cache_dir=self.base / "cache",
        )
        self.config.llm.enabled = False
        self.config.tagger.enabled = False
        self.config.ensure_dirs()

        self.app = create_app(self.config, embedder_factory=_FakeEmbedder)  # type: ignore[arg-type]
        self.folder_key = canonical_path(self.root)
        self.client = TestClient(self.app).__enter__()

    def tearDown(self) -> None:
        self.client.__exit__(None, None, None)
        self._tmp.cleanup()

    def _index(self) -> None:
        response = self.client.post(
            "/api/index",
            json={"path": str(self.root), "recursive": True, "auto_enrich": False},
        )
        self.assertEqual(response.status_code, 200)
        container = self.app.state.container
        deadline = time.time() + 30.0
        while time.time() < deadline and container.index_job.running:
            time.sleep(0.05)
        self.assertIsNone(container.index_job.error)

    def _names(self, sort: str) -> list[str]:
        response = self.client.get(
            "/api/folder-images",
            params={"folder_key": self.folder_key, "sort": sort, "limit": 10},
        )
        self.assertEqual(response.status_code, 200)
        return [item["name"] for item in response.json()["results"]]

    def test_indexed_sort_beats_old_file_mtime(self) -> None:
        now = time.time()
        # `old_first.jpg` is indexed first but carries the newest mtime.
        _save(self.root / "old_first.jpg", (200, 60, 60), now)
        self._index()
        # `new_later.jpg` lands afterwards with an ancient mtime.
        _save(self.root / "new_later.jpg", (60, 60, 200), now - 120_000)
        self._index()

        self.assertEqual(self._names("indexed"), ["new_later.jpg", "old_first.jpg"])
        self.assertEqual(self._names("mtime"), ["old_first.jpg", "new_later.jpg"])

    def test_index_status_exposes_a_monotonic_revision(self) -> None:
        container = self.app.state.container
        _save(self.root / "a.jpg", (10, 120, 10), time.time())
        baseline = container.images.max_id()

        self._index()
        # Clear the flag first: /api/index-status would otherwise start a
        # provenance thread that is still reading when the temporary database
        # is torn down (that teardown race is covered elsewhere).
        container.provenance_refresh_pending = False

        response = self.client.get("/api/index-status")
        self.assertEqual(response.status_code, 200)
        self.assertGreater(int(response.json()["revision"]), baseline)


if __name__ == "__main__":
    unittest.main()
