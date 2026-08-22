from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

from artseek.config import AppConfig
from artseek.image_io import canonical_path
from artseek.library import FolderCatalog, build_catalog
from artseek.web.app import create_app


def _payload(path: Path, root: Path, mtime_ns: int) -> dict[str, Any]:
    return {
        "path": str(path),
        "path_key": canonical_path(path),
        "root": str(root),
        "root_key": canonical_path(root),
        "folder_key": canonical_path(path.parent),
        "name": path.name,
        "mtime_ns": mtime_ns,
        "width": 16,
        "height": 12,
        "tags": [],
        "caption": None,
    }


class _Store:
    def __init__(self, payloads: dict[str, dict[str, Any]]) -> None:
        self.payloads = payloads

    def retrieve_payloads(self, point_ids: list[str]) -> dict[str, dict[str, Any]]:
        return {
            point_id: self.payloads[point_id]
            for point_id in point_ids
            if point_id in self.payloads
        }


class _Engine:
    def __init__(self, catalog: FolderCatalog, payloads: dict[str, dict[str, Any]]) -> None:
        self.catalog = catalog
        self.store = _Store(payloads)

    def folder_catalog(self) -> FolderCatalog:
        return self.catalog

    def close(self) -> None:
        return


class RandomBrowseApiTests(unittest.TestCase):
    def setUp(self) -> None:
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            self.skipTest("fastapi 未安装")
        self._temporary_directory = tempfile.TemporaryDirectory()
        root = Path(self._temporary_directory.name) / "art"
        nested = root / "nested"
        nested.mkdir(parents=True)
        payloads = {
            "root": _payload(root / "root.png", root, 1),
            "a": _payload(nested / "a.png", root, 2),
            "b": _payload(nested / "b.png", root, 3),
            "c": _payload(nested / "c.png", root, 4),
        }
        catalog = build_catalog(list(payloads.items()))
        self.nested_key = canonical_path(nested)
        config = AppConfig(data_dir=root / "data", cache_dir=root / "cache")
        self.client = TestClient(create_app(config, engine=_Engine(catalog, payloads)))

    def tearDown(self) -> None:
        self.client.close()
        self._temporary_directory.cleanup()

    def test_random_pages_are_seeded_unique_and_stay_in_selected_scope(self) -> None:
        request = {"folder_keys": [self.nested_key], "seed": 17, "limit": 2, "offset": 0}
        first = self.client.post("/api/random-browse", json=request)
        self.assertEqual(first.status_code, 200)
        first_data = first.json()
        self.assertEqual(first_data["total"], 3)
        self.assertTrue(first_data["has_more"])
        first_ids = [item["id"] for item in first_data["results"]]
        self.assertEqual(len(first_ids), 2)
        self.assertTrue(set(first_ids) <= {"a", "b", "c"})

        repeated = self.client.post("/api/random-browse", json=request).json()
        self.assertEqual([item["id"] for item in repeated["results"]], first_ids)

        second = self.client.post(
            "/api/random-browse",
            json={**request, "offset": 2},
        ).json()
        second_ids = [item["id"] for item in second["results"]]
        self.assertEqual(len(second_ids), 1)
        self.assertFalse(set(first_ids) & set(second_ids))
        self.assertEqual(set(first_ids) | set(second_ids), {"a", "b", "c"})
        self.assertFalse(second["has_more"])

    def test_random_browse_requires_an_explicit_or_saved_scope(self) -> None:
        response = self.client.post("/api/random-browse", json={"folder_keys": []})
        self.assertEqual(response.status_code, 400)
        self.assertIn("勾选", response.json()["detail"])

    def test_homepage_exposes_random_waterfall_action(self) -> None:
        html = self.client.get("/").text
        self.assertIn('id="random-browse"', html)
        self.assertIn("/api/random-browse", html)


if __name__ == "__main__":
    unittest.main()
