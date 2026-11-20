"""Browser safety rating gate applied to every waterfall endpoint."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from artseek.api.app import create_app
from artseek.config import AppConfig, TaggerConfig
from artseek.db.connection import Database
from artseek.db.repos.images import ImageRepo, ImageUpsertRecord
from artseek.image_io import canonical_path


class RatingFilterApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        root = Path(self._temporary_directory.name)
        img_dir = root / "art"
        img_dir.mkdir()
        safe = img_dir / "safe.png"
        r15 = img_dir / "r15.png"
        unrated = img_dir / "unrated.png"
        for path in (safe, r15, unrated):
            Image.new("RGB", (32, 32), (120, 140, 200)).save(path)

        cfg = AppConfig(
            data_dir=root / "data",
            cache_dir=root / "cache",
            tagger=TaggerConfig(enabled=False),
        )
        cfg.ensure_dirs()
        db = Database(cfg.db_path)
        try:
            images = ImageRepo(db)
            folder_key = canonical_path(img_dir)
            row_ids = images.upsert_images_batch(
                [
                    self._record(point_id, path, folder_key, img_dir, mtime_ns, phash)
                    for point_id, path, mtime_ns, phash in (
                        ("safe-id", safe, 100, "0" * 16),
                        ("r15-id", r15, 200, "1" * 16),
                        ("unrated-id", unrated, 300, "2" * 16),
                    )
                ]
            )
            images.update_vision_attributes_batch(
                [
                    {"image_id": row_ids[0], "rating": "safe"},
                    {"image_id": row_ids[1], "rating": "r15"},
                ]
            )
        finally:
            db.close()

        self.folder_key = folder_key
        self.image_dir = img_dir
        self.client = TestClient(create_app(cfg)).__enter__()

    def tearDown(self) -> None:
        self.client.__exit__(None, None, None)
        self._temporary_directory.cleanup()

    @staticmethod
    def _record(
        point_id: str,
        path: Path,
        folder_key: str,
        folder_path: Path,
        mtime_ns: int,
        phash: str,
    ) -> ImageUpsertRecord:
        return ImageUpsertRecord(
            point_id=point_id,
            root_key=folder_key,
            root_path=str(folder_path),
            folder_key=folder_key,
            folder_path=str(folder_path),
            folder_mtime_ns=1,
            path=str(path),
            path_key=canonical_path(path),
            name=path.name,
            ext=".png",
            mtime_ns=mtime_ns,
            size_bytes=32,
            width=32,
            height=32,
            content_hash=point_id,
            phash=phash,
        )

    @staticmethod
    def _ids(payload: dict[str, object]) -> set[str]:
        return {str(item["id"]) for item in payload["results"]}  # type: ignore[index]

    def test_scope_images_default_and_rating_gates(self) -> None:
        base = {"folder_keys": [self.folder_key]}
        everything = self.client.post("/api/scope-images", json=base)
        self.assertEqual(everything.status_code, 200)
        self.assertEqual(self._ids(everything.json()), {"safe-id", "r15-id", "unrated-id"})

        safe_only = self.client.post("/api/scope-images", json={**base, "ratings": ["safe"]})
        self.assertEqual(safe_only.status_code, 200)
        self.assertEqual(self._ids(safe_only.json()), {"safe-id"})
        self.assertEqual(safe_only.json()["total"], 1)

        r15_only = self.client.post(
            "/api/scope-images", json={**base, "ratings": ["r15"]}
        )
        self.assertEqual(r15_only.status_code, 200)
        self.assertEqual(self._ids(r15_only.json()), {"r15-id"})

        up_to_r15 = self.client.post(
            "/api/scope-images", json={**base, "ratings": ["safe", "r15"]}
        )
        self.assertEqual(up_to_r15.status_code, 200)
        self.assertEqual(self._ids(up_to_r15.json()), {"safe-id", "r15-id"})

    def test_folder_random_collection_and_get_scope_paths(self) -> None:
        folder = self.client.get(
            "/api/folder-images",
            params=[("folder_key", self.folder_key), ("ratings", "safe")],
        )
        self.assertEqual(folder.status_code, 200)
        self.assertEqual(self._ids(folder.json()), {"safe-id"})

        random_page = self.client.post(
            "/api/random-browse",
            json={
                "folder_keys": [self.folder_key],
                "ratings": ["safe", "r15"],
                "seed": 5,
                "limit": 10,
            },
        )
        self.assertEqual(random_page.status_code, 200)
        self.assertEqual(self._ids(random_page.json()), {"safe-id", "r15-id"})

        created = self.client.post("/api/collections", json={"name": "安全测试"})
        collection_id = created.json()["collection"]["id"]
        for image_id in ("safe-id", "r15-id", "unrated-id"):
            added = self.client.post(
                f"/api/collections/{collection_id}/items", json={"image_id": image_id}
            )
            self.assertEqual(added.status_code, 200)
        safe_collection = self.client.get(
            f"/api/collections/{collection_id}/images",
            params=[("ratings", "safe")],
        )
        self.assertEqual(safe_collection.status_code, 200)
        self.assertEqual(self._ids(safe_collection.json()), {"safe-id"})
        self.assertEqual(safe_collection.json()["total"], 1)

        get_scope = self.client.get(
            "/api/scope-images",
            params={"folder_keys": json.dumps([self.folder_key]), "ratings": ["safe"]},
        )
        self.assertEqual(get_scope.status_code, 200)
        self.assertEqual(self._ids(get_scope.json()), {"safe-id"})

    def test_author_images_honor_folder_scope(self) -> None:
        other_dir = self.image_dir / "other"
        other_dir.mkdir()
        other = other_dir / "other.png"
        Image.new("RGB", (32, 32), (50, 90, 180)).save(other)
        container = self.client.app.state.container
        container.images.upsert_images_batch(
            [
                self._record(
                    "other-id",
                    other,
                    canonical_path(other_dir),
                    other_dir,
                    400,
                    "3" * 16,
                )
            ]
        )
        author = container.prov_repo.ensure_author("pixiv", "1", name="Scope Author")
        with container.db.write_tx() as conn:
            for point_id in ("safe-id", "other-id"):
                conn.execute(
                    "INSERT INTO image_source ("
                    "point_id, platform, source_id, url, author_key, updated_at"
                    ") VALUES (?, 'pixiv', ?, 'https://example.com/x', ?, 1)",
                    (point_id, point_id, author),
                )

        scoped = self.client.post(
            f"/api/artists/{author}/images",
            json={"folder_keys": [self.folder_key], "limit": 10},
        )
        self.assertEqual(scoped.status_code, 200)
        self.assertEqual(self._ids(scoped.json()), {"safe-id"})

        unscoped = self.client.post(
            f"/api/artists/{author}/images",
            json={"limit": 10},
        )
        self.assertEqual(unscoped.status_code, 200)
        self.assertEqual(self._ids(unscoped.json()), {"safe-id", "other-id"})


if __name__ == "__main__":
    unittest.main()
