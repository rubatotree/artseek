"""Unit and integration tests for M4 indexing pipeline, IncrementalScanner, and JobManager."""

from __future__ import annotations

import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
from PIL import Image

from artseek.config import AppConfig
from artseek.db.connection import Database
from artseek.db.repos.images import ImageRepo
from artseek.db.repos.jobs import JobRepo
from artseek.db.repos.provenance import ProvenanceRepo
from artseek.indexing.jobs import JobManager
from artseek.indexing.scan import IncrementalScanner
from artseek.ml.pipeline import decode_and_prepare_image
from artseek.ml.tagger import PredictedTags
from artseek.services.model_lifecycle import ModelLifecycle
from artseek.services.provenance import ProvenanceAppService
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


class _FakeTagger:
    def predict(self, images: list[Any]) -> list[PredictedTags]:
        return [
            PredictedTags(
                tags=["1girl", "solo", "blue_archive"],
                scores={"1girl": 0.98, "solo": 0.91, "blue_archive": 0.77},
            )
            for _ in images
        ]

    def close(self) -> None:
        pass


class _FakeLLM:
    def caption_image(self, path: Path) -> dict[str, Any]:
        return {
            "subject": f"subject of {path.stem}",
            "outfit": "school uniform",
            "pose": "standing",
            "scene": "classroom",
            "composition": "upper body",
            "colors": "blue and white",
            "keywords": ["halo", "rifle"],
        }


def _save_test_image(
    path: Path,
    size: tuple[int, int] = (800, 1200),
    color: tuple[int, int, int] = (80, 140, 220),
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", size, color)
    pixels = np.asarray(img, dtype=np.uint8).copy()
    xs = np.linspace(0, 120, size[0], dtype=np.uint8)
    pixels[:, :, 0] = np.clip(pixels[:, :, 0] + xs[None, :], 0, 255)
    Image.fromarray(pixels, mode="RGB").save(path, format="JPEG", quality=90)


class V2IndexingPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.base = Path(self._tmp.name)
        self.cfg = AppConfig()
        self.cfg.data_dir = self.base / "data"
        self.cfg.ensure_dirs()
        self.db = Database(self.cfg.db_path)
        self.images = ImageRepo(self.db)
        self.prov_repo = ProvenanceRepo(self.db)
        self.jobs_repo = JobRepo(self.db)
        self.vectors = VectorIndex(
            self.cfg.visual_vectors_path,
            self.cfg.descriptor_vectors_path,
        )

    def tearDown(self) -> None:
        self.vectors.release_cuda()
        self.db.close()
        self._tmp.cleanup()

    def test_decode_records_true_original_dimensions_and_pipeline_persists(self) -> None:
        root = self.base / "pics"
        img_a = root / "sub1" / "alpha.jpg"
        img_b = root / "sub2" / "beta.jpg"
        _save_test_image(img_a, size=(1440, 2160), color=(40, 90, 180))
        _save_test_image(img_b, size=(1920, 1080), color=(180, 70, 50))

        prep = decode_and_prepare_image(root, img_a)
        self.assertEqual(prep.orig_width, 1440)
        self.assertEqual(prep.orig_height, 2160)
        self.assertEqual(len(prep.content_hash), 64)
        self.assertEqual(len(prep.phash), 16)

        scanner = IncrementalScanner(
            self.cfg,
            self.db,
            self.images,
            self.vectors,
            embedder_factory=_FakeEmbedder,  # type: ignore[arg-type]
            tagger_factory=_FakeTagger,  # type: ignore[arg-type]
            cpu_workers=2,
        )
        summary1 = scanner.index_roots([root])
        self.assertEqual(summary1.discovered, 2)
        self.assertEqual(summary1.indexed, 2)
        self.assertEqual(summary1.unchanged, 0)
        self.assertEqual(summary1.deleted, 0)
        self.assertEqual(self.images.count(), 2)
        self.assertEqual(self.vectors.count, 2)

        payloads = self.images.get_payloads_by_ids([0, 1])
        dims = {(p["width"], p["height"]) for p in payloads.values()}
        self.assertEqual(dims, {(1440, 2160), (1920, 1080)})
        for p in payloads.values():
            self.assertIn("blue_archive", p["tags"])

        # Second scan with no filesystem changes should take the delta fast-path (0 indexed)
        summary2 = scanner.index_roots([root])
        self.assertEqual(summary2.discovered, 2)
        self.assertEqual(summary2.indexed, 0)
        self.assertEqual(summary2.unchanged, 2)
        self.assertEqual(summary2.deleted, 0)

        # Delete one file and add a new one -> delta scan detects 1 new + 1 deleted
        img_b.unlink()
        img_c = root / "sub1" / "gamma.jpg"
        _save_test_image(img_c, size=(1000, 1000), color=(50, 200, 100))
        summary3 = scanner.index_roots([root])
        self.assertEqual(summary3.indexed, 1)
        self.assertEqual(summary3.deleted, 1)
        self.assertEqual(self.images.count(), 2)

    def test_overlapping_roots_rejected(self) -> None:
        parent = self.base / "library"
        child = parent / "nested"
        _save_test_image(child / "one.jpg")

        scanner = IncrementalScanner(
            self.cfg,
            self.db,
            self.images,
            self.vectors,
            embedder_factory=_FakeEmbedder,  # type: ignore[arg-type]
            tagger_factory=_FakeTagger,  # type: ignore[arg-type]
        )
        with self.assertRaises(ValueError):
            scanner.index_roots([parent, child])

    def test_job_manager_index_and_caption_backfill(self) -> None:
        root = self.base / "job_pics"
        _save_test_image(root / "p1.jpg", size=(640, 960))
        _save_test_image(root / "p2.jpg", size=(960, 640))

        lifecycle = ModelLifecycle(
            self.cfg,
            embedder_factory=_FakeEmbedder,
            llm=_FakeLLM(),  # type: ignore[arg-type]
            vectors=self.vectors,
        )
        prov_svc = ProvenanceAppService(
            self.cfg,
            self.images,
            self.prov_repo,
            self.vectors,
        )
        jm = JobManager(
            self.cfg,
            self.db,
            self.images,
            self.prov_repo,
            self.jobs_repo,
            self.vectors,
            lifecycle,
            prov_svc,
        )
        try:
            # First index via scanner directly with fake tagger, then run caption backfill job
            scanner = IncrementalScanner(
                self.cfg,
                self.db,
                self.images,
                self.vectors,
                embedder_factory=_FakeEmbedder,  # type: ignore[arg-type]
                tagger_factory=_FakeTagger,  # type: ignore[arg-type]
            )
            scanner.index_roots([root])
            cap_job_id = jm.submit_caption_backfill_job()
            deadline = time.time() + 5.0
            while time.time() < deadline:
                row = self.jobs_repo.get(cap_job_id)
                if row and row["status"] in {"completed", "failed", "stopped"}:
                    break
                time.sleep(0.05)
            job_row = self.jobs_repo.get(cap_job_id)
            self.assertIsNotNone(job_row)
            assert job_row is not None
            self.assertEqual(job_row["status"], "completed")
            self.assertEqual(job_row["summary"]["updated"], 2)

            p0 = self.images.get_payloads_by_ids([0])[0]
            self.assertIsNotNone(p0["caption"])
            self.assertIn("school uniform", p0["descriptor_text"])
        finally:
            jm.close()
            lifecycle.close()


if __name__ == "__main__":
    unittest.main()
