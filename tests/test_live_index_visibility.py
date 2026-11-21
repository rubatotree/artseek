"""New rows must be readable before their tags finish.

The gallery shows images the moment the visual vectors are committed, so the
indexing pipeline splits each batch into two stages:

1. file rows + visual vectors (descriptor mirrors visual as a placeholder)
2. tags + descriptor text + descriptor vectors

These tests pin that ordering, and pin the `keep_enrichment` upsert mode that
prevents stage 1 from blanking tags written by stage 2 of a previous run.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
from PIL import Image

from artseek.config import AppConfig
from artseek.db.connection import Database
from artseek.db.repos.images import ImageRepo, ImageUpsertRecord
from artseek.ml.pipeline import IndexingPipeline
from artseek.ml.tagger import PredictedTags
from artseek.vectors.index import VectorIndex


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


class _ProbingTagger:
    """Records the database state at the exact moment tagging starts."""

    model_versions = {"camie_v2": 1}

    def __init__(self, db: Database) -> None:
        self.db = db
        self.observed: dict[str, Any] | None = None

    def predict(self, images: list[Any]) -> list[PredictedTags]:
        conn = self.db.read_conn()
        rows = conn.execute(
            "SELECT id, descriptor_text FROM images WHERE deleted = 0 ORDER BY id"
        ).fetchall()
        tag_count = conn.execute("SELECT COUNT(*) AS c FROM image_tags").fetchone()
        self.observed = {
            "rows": [(int(r["id"]), str(r["descriptor_text"])) for r in rows],
            "tag_count": int(tag_count["c"]) if tag_count else 0,
        }
        return [
            PredictedTags(
                tags=["1girl", "solo", "blue_archive"],
                scores={"1girl": 0.98, "solo": 0.91, "blue_archive": 0.77},
            )
            for _ in images
        ]

    def close(self) -> None:
        pass


def _save_test_image(path: Path, size: tuple[int, int], color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path, format="JPEG", quality=90)


class LiveVisibilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.base = Path(self._tmp.name)
        self.cfg = AppConfig()
        self.cfg.data_dir = self.base / "data"
        self.cfg.ensure_dirs()
        self.db = Database(self.cfg.db_path)
        self.images = ImageRepo(self.db)
        self.vectors = VectorIndex(
            self.cfg.visual_vectors_path,
            self.cfg.descriptor_vectors_path,
        )

    def tearDown(self) -> None:
        self.vectors.release_cuda()
        self.db.close()
        self._tmp.cleanup()

    def _pipeline(self, tagger: Any) -> IndexingPipeline:
        return IndexingPipeline(
            self.cfg,
            self.images,
            self.vectors,
            embedder=_FakeEmbedder(),  # type: ignore[arg-type]
            tagger=tagger,
            cpu_workers=1,
        )

    def _items(self, root: Path, names: list[str]) -> list[tuple[Path, Path, int, int, int]]:
        items = []
        for index, name in enumerate(names):
            path = root / name
            _save_test_image(path, (320, 480), (30 + index * 40, 90, 190))
            stat = path.stat()
            items.append((root, path, stat.st_mtime_ns, stat.st_size, 0))
        return items

    def test_rows_land_before_tags(self) -> None:
        root = self.base / "pics"
        tagger = _ProbingTagger(self.db)
        items = self._items(root, ["a.jpg", "b.jpg"])

        self._pipeline(tagger).process_items(items)

        observed = tagger.observed
        self.assertIsNotNone(observed)
        assert observed is not None
        # Stage A already committed both rows, without tags or descriptor text.
        self.assertEqual(len(observed["rows"]), 2)
        self.assertTrue(all(text == "" for _, text in observed["rows"]))
        self.assertEqual(observed["tag_count"], 0)

        # Stage B patched tags and descriptor text into the same rows.
        conn = self.db.read_conn()
        rows = conn.execute(
            "SELECT id, descriptor_text FROM images WHERE deleted = 0 ORDER BY id"
        ).fetchall()
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(str(r["descriptor_text"]).strip() for r in rows))
        tag_count = conn.execute("SELECT COUNT(*) AS c FROM image_tags").fetchone()
        self.assertGreater(int(tag_count["c"]), 0)

    def test_tag_versions_marked_only_after_stage_b(self) -> None:
        root = self.base / "pics"
        tagger = _ProbingTagger(self.db)
        self._pipeline(tagger).process_items(self._items(root, ["a.jpg"]))

        conn = self.db.read_conn()
        row = conn.execute(
            "SELECT tag_models_json FROM image_task_state LIMIT 1"
        ).fetchone()
        self.assertIsNotNone(row)
        self.assertIn("camie_v2", str(row["tag_models_json"]))

    def test_keep_enrichment_preserves_existing_tags(self) -> None:
        root = self.base / "pics"
        path = root / "a.jpg"
        _save_test_image(path, (320, 480), (60, 90, 190))
        stat = path.stat()
        record = ImageUpsertRecord(
            point_id="pid-1",
            root_key=str(root),
            root_path=str(root),
            folder_key=str(root),
            folder_path=str(root),
            folder_mtime_ns=0,
            path=str(path),
            path_key=str(path),
            name="a.jpg",
            ext=".jpg",
            mtime_ns=stat.st_mtime_ns,
            size_bytes=stat.st_size,
            width=320,
            height=480,
            content_hash="hash-1",
            phash="phash-1",
            descriptor_text="1girl, solo",
            caption=None,
            tags=["1girl", "solo"],
            tag_scores={"1girl": 0.95, "solo": 0.9},
            tag_categories={"1girl": 0, "solo": 0},
        )
        self.images.upsert_images_batch([record])

        refreshed = _refreshed_record(record)
        self.images.upsert_images_batch([refreshed], keep_enrichment=True)

        conn = self.db.read_conn()
        row = conn.execute(
            "SELECT descriptor_text FROM images WHERE point_id = 'pid-1'"
        ).fetchone()
        self.assertEqual(str(row["descriptor_text"]), "1girl, solo")
        tag_count = conn.execute(
            "SELECT COUNT(*) AS c FROM image_tags WHERE image_id = "
            "(SELECT id FROM images WHERE point_id = 'pid-1')"
        ).fetchone()
        self.assertEqual(int(tag_count["c"]), 2)


def _refreshed_record(record: ImageUpsertRecord) -> ImageUpsertRecord:
    """Same file, new mtime — the shape stage A writes on a re-index."""
    return ImageUpsertRecord(
        point_id=record.point_id,
        root_key=record.root_key,
        root_path=record.root_path,
        folder_key=record.folder_key,
        folder_path=record.folder_path,
        folder_mtime_ns=record.folder_mtime_ns,
        path=record.path,
        path_key=record.path_key,
        name=record.name,
        ext=record.ext,
        mtime_ns=record.mtime_ns + 1,
        size_bytes=record.size_bytes,
        width=record.width,
        height=record.height,
        content_hash=record.content_hash,
        phash=record.phash,
        descriptor_text="",
        caption=None,
        tags=[],
        tag_scores={},
        tag_categories={},
    )


if __name__ == "__main__":
    unittest.main()
