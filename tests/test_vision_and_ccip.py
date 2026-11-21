"""Unit tests for VisionAttributeAnalyzer, HeadFocalDetector, CCIP, VectorIndex CCIP, and jobs."""

from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from artseek.config import AppConfig, EmbeddingConfig, TaggerConfig
from artseek.db.connection import Database
from artseek.db.repos.images import ImageRepo, ImageUpsertRecord
from artseek.db.repos.jobs import JobRepo
from artseek.db.repos.provenance import ProvenanceRepo
from artseek.db.repos.task_state import TaskStateRepo
from artseek.indexing.jobs import JobDependencyError, JobManager
from artseek.ml.ccip import CCIPExtractor
from artseek.ml.tag_postprocess import (
    CAT_CHARACTER,
    CAT_COPYRIGHT,
    CAT_MEDIUM,
    CAT_RATING,
    PredictedTags,
)
from artseek.ml.vision_attributes import (
    VISION_META_VERSION,
    HeadFocalDetector,
    VisionAttributeAnalyzer,
    extract_embedded_ai_metadata,
)
from artseek.services.model_lifecycle import ModelLifecycle
from artseek.services.provenance import ProvenanceAppService
from artseek.vectors.index import VectorIndex


class _FakeORTTensor:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeORTSession:
    def __init__(self, output_fn, input_name: str = "input", output_name: str = "output") -> None:
        self._output_fn = output_fn
        self._input_name = input_name
        self._output_name = output_name
        self.last_shape: tuple[int, ...] | None = None

    def get_inputs(self) -> list[_FakeORTTensor]:
        return [_FakeORTTensor(self._input_name)]

    def get_outputs(self) -> list[_FakeORTTensor]:
        return [_FakeORTTensor(self._output_name)]

    def run(self, _outputs: list[str], feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        arr = feed[self._input_name]
        self.last_shape = tuple(arr.shape)
        return [self._output_fn(arr)]


class _FakeEmbedder:
    dimension = 8

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for i, txt in enumerate(texts):
            out[i, 0] = max(1.0, float(len(txt)))
            out[i, 1] = 1.0
        return out


class TestVisionAttributesAndCCIP(unittest.TestCase):
    def test_embedded_ai_metadata_detection(self) -> None:
        img_plain = Image.new("RGB", (32, 32), (255, 255, 255))
        self.assertEqual(extract_embedded_ai_metadata(img_plain), (False, None))

        img_sd = Image.new("RGB", (32, 32), (255, 255, 255))
        img_sd.info["parameters"] = "1girl, masterpiece\nSteps: 28, Sampler: Euler a, CFG scale: 7"
        self.assertEqual(extract_embedded_ai_metadata(img_sd), (True, "sd_webui"))

        img_nai = Image.new("RGB", (32, 32), (255, 255, 255))
        img_nai.info["Software"] = "NovelAI"
        self.assertEqual(extract_embedded_ai_metadata(img_nai), (True, "novelai"))

        img_comfy = Image.new("RGB", (32, 32), (255, 255, 255))
        img_comfy.info["prompt"] = '{"1": {"class_type": "KSampler"}}'
        self.assertEqual(extract_embedded_ai_metadata(img_comfy), (True, "comfyui"))

    def test_head_detector_and_vision_analyzer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dummy_onnx = root / "model.onnx"
            dummy_onnx.write_bytes(b"fake")

            meta_files = {
                "deepghs/anime_classification": [
                    "3d",
                    "bangumi",
                    "comic",
                    "illustration",
                    "not_painting",
                ],
                "deepghs/anime_rating": ["safe", "r15", "r18"],
                "deepghs/monochrome_detect": ["monochrome", "normal"],
                "deepghs/anime_ai_check": ["ai", "human"],
                "deepghs/anime_completeness": ["polished", "rough", "monochrome"],
                "deepghs/anime_aesthetic": [
                    "masterpiece",
                    "best",
                    "great",
                    "good",
                    "normal",
                    "low",
                    "worst",
                ],
            }
            paths: dict[tuple[str, str], Path] = {}
            for repo_id, labels in meta_files.items():
                mp = root / f"{repo_id.replace('/', '_')}.json"
                mp.write_text(json.dumps({"labels": labels}), encoding="utf-8")
                paths[(repo_id, "meta")] = mp

            def fake_dl(repo_id: str, filename: str) -> Path:
                if filename.endswith(".json"):
                    return paths[(repo_id, "meta")]
                return dummy_onnx

            def head_out(arr: np.ndarray) -> np.ndarray:
                b = arr.shape[0]
                # [B, 5, 2]: cx, cy, w, h, conf in 640x640 letterbox coords
                out = np.zeros((b, 5, 2), dtype=np.float32)
                # Head at top center-right (cx=400, cy=180, w=120, h=160) -> focal_y ~ 0.28
                out[:, 0, 0] = 400.0
                out[:, 1, 0] = 180.0
                out[:, 2, 0] = 120.0
                out[:, 3, 0] = 160.0
                out[:, 4, 0] = 0.92
                return out

            logits_by_size = {
                5: np.array([[0.0, 0.0, 0.0, 5.0, 0.0]], dtype=np.float32),
                3: np.array([[5.0, 0.0, 0.0]], dtype=np.float32),
                2: np.array([[0.0, 5.0]], dtype=np.float32),
                7: np.array([[5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            }
            captured_inputs: list[tuple[int, float, float]] = []

            def _capture_and_return(key: int):
                def _fn(arr: np.ndarray) -> np.ndarray:
                    captured_inputs.append(
                        (int(arr.shape[-1]), float(np.min(arr)), float(np.max(arr)))
                    )
                    return np.repeat(logits_by_size[key], arr.shape[0], axis=0)

                return _fn

            sessions = [
                _FakeORTSession(head_out, input_name="images", output_name="output0"),
                _FakeORTSession(_capture_and_return(5)),
                _FakeORTSession(_capture_and_return(3)),
                _FakeORTSession(_capture_and_return(2)),
                _FakeORTSession(_capture_and_return(2)),
                _FakeORTSession(_capture_and_return(3)),
                _FakeORTSession(_capture_and_return(7)),
            ]
            sess_iter = iter(sessions)

            with (
                patch("artseek.ml.vision_attributes._cached_or_download", side_effect=fake_dl),
                patch(
                    "artseek.ml.vision_attributes.create_ort_session",
                    side_effect=lambda *args, **kwargs: next(sess_iter),
                ),
            ):
                analyzer = VisionAttributeAnalyzer(EmbeddingConfig(device="cpu"))
                img = Image.new("RGB", (640, 640), (200, 220, 255))
                updates = analyzer.analyze_batch([(7, img)])
                analyzer.close()

            self.assertEqual(len(updates), 1)
            u = updates[0]
            self.assertEqual(u["image_id"], 7)
            self.assertEqual(u["media_type"], "illustration")
            self.assertEqual(u["rating"], "safe")
            self.assertEqual(u["is_monochrome"], 0)
            self.assertEqual(u["is_ai"], 0)
            self.assertAlmostEqual(u["focal_x"], 400.0 / 640.0, places=2)
            self.assertAlmostEqual(u["focal_y"], 180.0 / 640.0, places=2)
            self.assertEqual(u["shot_type"], "upper_body")
            self.assertIn("medium:illustration", u["attr_tags"])
            self.assertIn("rating:safe", u["attr_tags"])
            self.assertIn("hand_drawn", u["attr_tags"])
            self.assertIn("polished", u["attr_tags"])
            self.assertGreater(u["aesthetic_score"], 0.80)
            # Verify official [-1, 1] normalization & 384/448 resolutions
            self.assertEqual([c[0] for c in captured_inputs], [384, 384, 384, 384, 384, 448])
            for _, arr_min, arr_max in captured_inputs:
                self.assertGreaterEqual(arr_min, -1.01)
                self.assertLessEqual(arr_max, 1.01)
                self.assertGreater(arr_max, 0.5)

            # Domain guard: 3D/photo medium must not be flagged AI without embedded metadata
            guard_logits = {
                5: np.array([[5.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32),  # 3d
                3: np.array([[5.0, 0.0, 0.0]], dtype=np.float32),
                2: np.array([[5.0, 0.0]], dtype=np.float32),  # ai=1.0 & mono=1.0
                7: np.array([[0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            }
            guard_sessions = [
                _FakeORTSession(head_out, input_name="images", output_name="output0"),
                _FakeORTSession(lambda arr: np.repeat(guard_logits[5], arr.shape[0], axis=0)),
                _FakeORTSession(lambda arr: np.repeat(guard_logits[3], arr.shape[0], axis=0)),
                _FakeORTSession(lambda arr: np.repeat(guard_logits[2], arr.shape[0], axis=0)),
                _FakeORTSession(lambda arr: np.repeat(guard_logits[2], arr.shape[0], axis=0)),
                _FakeORTSession(lambda arr: np.repeat(guard_logits[3], arr.shape[0], axis=0)),
                _FakeORTSession(lambda arr: np.repeat(guard_logits[7], arr.shape[0], axis=0)),
            ]
            g_iter = iter(guard_sessions)
            with (
                patch("artseek.ml.vision_attributes._cached_or_download", side_effect=fake_dl),
                patch(
                    "artseek.ml.vision_attributes.create_ort_session",
                    side_effect=lambda *args, **kwargs: next(g_iter),
                ),
            ):
                g_analyzer = VisionAttributeAnalyzer(EmbeddingConfig(device="cpu"))
                g_updates = g_analyzer.analyze_batch([(8, img)])
                g_analyzer.close()
            self.assertEqual(g_updates[0]["media_type"], "3d")
            self.assertEqual(g_updates[0]["is_ai"], 0)
            self.assertEqual(g_updates[0]["ai_score"], 0.0)

    def test_vision_classifiers_skip_double_softmax_when_already_normalized(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            onnx_p = root / "model.onnx"
            onnx_p.write_bytes(b"fake")
            meta_p = root / "meta.json"
            meta_p.write_text(
                json.dumps({"labels": ["polished", "rough", "monochrome"]}),
                encoding="utf-8",
            )
            from artseek.ml.vision_attributes import _OnnxImageClassifier

            sess = _FakeORTSession(
                lambda arr: np.repeat(
                    np.array([[0.95, 0.04, 0.01]], dtype=np.float32),
                    arr.shape[0],
                    axis=0,
                )
            )
            with (
                patch(
                    "artseek.ml.vision_attributes._cached_or_download",
                    side_effect=lambda r, f: meta_p if f.endswith(".json") else onnx_p,
                ),
                patch("artseek.ml.vision_attributes.create_ort_session", return_value=sess),
            ):
                clf = _OnnxImageClassifier(
                    torch.device("cpu"),
                    "deepghs/anime_completeness",
                    "model.onnx",
                    "meta.json",
                    ("polished", "rough", "monochrome"),
                    224,
                )
                probs = clf.predict_probs([Image.new("RGB", (64, 64))])[0]
            self.assertAlmostEqual(probs["polished"], 0.95, places=3)

    def test_head_detector_handles_transposed_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            onnx_p = Path(tmp) / "head.onnx"
            onnx_p.write_bytes(b"fake")
            sess = _FakeORTSession(
                lambda arr: np.array([[[320.0, 200.0, 100.0, 100.0, 0.88]]], dtype=np.float32),
                input_name="images",
                output_name="output0",
            )
            with (
                patch("artseek.ml.vision_attributes._cached_or_download", return_value=onnx_p),
                patch("artseek.ml.vision_attributes.create_ort_session", return_value=sess),
            ):
                det = HeadFocalDetector(torch.device("cpu"))
                res = det.predict([Image.new("RGB", (640, 640))])[0]
            self.assertEqual(res.head_count, 1)
            self.assertAlmostEqual(res.focal_x, 0.5, places=2)

    def test_vector_index_ccip_search_compact_and_extractor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vis_p = root / "visual.f32.npy"
            desc_p = root / "descriptor.f32.npy"
            ccip_p = root / "ccip.f32.npy"

            idx = VectorIndex(vis_p, desc_p, ccip_path=ccip_p)
            idx.set_rows(
                [0, 1, 2],
                np.eye(3, 4, dtype=np.float32),
                np.eye(3, 4, dtype=np.float32),
            )
            # Populate CCIP only on row 0 and row 2 (leave row 1 unpopulated -> masked out)
            c0 = np.zeros(768, dtype=np.float32)
            c0[0] = 1.0
            c2 = np.zeros(768, dtype=np.float32)
            c2[1] = 1.0
            idx.set_ccip_rows([0, 2], np.stack([c0, c2]))
            idx.flush()
            self.assertTrue(ccip_p.is_file())

            idx2 = VectorIndex(vis_p, desc_p, ccip_path=ccip_p)
            rows, scores = idx2.search(c0, "ccip", k=5)
            self.assertEqual(rows.tolist(), [0, 2])
            self.assertAlmostEqual(float(scores[0]), 1.0, places=4)
            self.assertIsNone(idx2.ccip_vector(1))
            self.assertIsNotNone(idx2.ccip_vector(2))

            # Compact dropping row 1 keeps CCIP row alignment
            idx2.compact([0, 2])
            rows_after, _ = idx2.search(c2, "ccip", k=5)
            self.assertEqual(rows_after.tolist(), [1, 0])

            # CCIPExtractor crop + L2 normalization
            fake_feat = np.zeros((1, 768), dtype=np.float32)
            fake_feat[0, 3] = 5.0
            ccip_sess = _FakeORTSession(lambda arr: fake_feat)
            onnx_p = root / "ccip.onnx"
            onnx_p.write_bytes(b"fake")
            with (
                patch("artseek.ml.ccip._cached_or_download", return_value=onnx_p),
                patch("artseek.ml.ccip.create_ort_session", return_value=ccip_sess),
            ):
                ext = CCIPExtractor(EmbeddingConfig(device="cpu"))
                vec = ext.extract_one(
                    Image.new("RGB", (200, 300), (120, 120, 120)),
                    crop_json='{"body":[0.1,0.1,0.9,0.9]}',
                )
            self.assertEqual(ccip_sess.last_shape, (1, 3, 384, 384))
            self.assertEqual(vec.shape, (768,))
            self.assertAlmostEqual(float(np.linalg.norm(vec)), 1.0, places=5)

    def test_job_manager_retag_vision_and_ccip_skip_completed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            img_dir = root / "imgs"
            img_dir.mkdir()
            p1_path = img_dir / "a.png"
            p2_path = img_dir / "b.png"
            Image.new("RGB", (64, 64), (255, 0, 0)).save(p1_path)
            Image.new("RGB", (64, 64), (0, 255, 0)).save(p2_path)

            cfg = AppConfig(
                data_dir=root / "data",
                cache_dir=root / "cache",
                tagger=TaggerConfig(backend="camie+pixai0.9"),
            )
            cfg.ensure_dirs()
            db = Database(cfg.db_path)
            try:
                images = ImageRepo(db)
                prov = ProvenanceRepo(db)
                jobs_repo = JobRepo(db)
                vectors = VectorIndex(
                    cfg.visual_vectors_path,
                    cfg.descriptor_vectors_path,
                    ccip_path=cfg.ccip_vectors_path,
                )
                records = []
                for idx, (pid, path) in enumerate([("p1", p1_path), ("p2", p2_path)]):
                    records.append(
                        ImageUpsertRecord(
                            point_id=pid,
                            root_key="r1",
                            root_path=str(img_dir),
                            folder_key="r1",
                            folder_path=str(img_dir),
                            folder_mtime_ns=1,
                            path=str(path),
                            path_key=str(path).casefold(),
                            name=path.name,
                            ext=".png",
                            mtime_ns=10 + idx,
                            size_bytes=100,
                            width=64,
                            height=64,
                            content_hash=f"hash_{idx}",
                            phash="0" * 16,
                            descriptor_text="old",
                            tags=["old"],
                            tag_scores={"old": 0.5},
                        )
                    )
                images.upsert_images_batch(
                    [records[0]], tag_model_versions={"camie_v2": 1, "pixai_v09": 1}
                )
                images.upsert_images_batch([records[1]], tag_model_versions={"camie_v2": 1})
                vectors.set_rows(
                    [0, 1],
                    np.eye(2, 8, dtype=np.float32),
                    np.eye(2, 8, dtype=np.float32),
                )

                lifecycle = ModelLifecycle(cfg, embedder_factory=_FakeEmbedder)
                prov_svc = ProvenanceAppService(cfg, images, prov, vectors)
                mgr = JobManager(cfg, db, images, prov, jobs_repo, vectors, lifecycle, prov_svc)

                class _DummyTagger:
                    model_versions = {"camie_v2": 1, "pixai_v09": 1}

                    def predict(self, imgs):
                        return [
                            PredictedTags(
                                tags=["furina", "genshin_impact"],
                                scores={"furina": 0.91, "genshin_impact": 0.88},
                                categories={
                                    "furina": CAT_CHARACTER,
                                    "genshin_impact": CAT_COPYRIGHT,
                                },
                            )
                            for _ in imgs
                        ]

                    def close(self):
                        pass

                class _DummyAnalyzer:
                    def __init__(self, _runtime):
                        pass

                    def analyze_batch(self, items):
                        return [
                            {
                                "image_id": img_id,
                                "media_type": "illustration",
                                "rating": "safe",
                                "is_monochrome": 0,
                                "monochrome_score": 0.05,
                                "is_ai": 0,
                                "ai_source": "human",
                                "ai_score": 0.02,
                                "aesthetic_score": 0.9,
                                "shot_type": "upper_body",
                                "focal_x": 0.5,
                                "focal_y": 0.3,
                                "character_crop_json": '{"body":[0.1,0.1,0.9,0.9]}',
                                "vision_meta_version": 2,
                                "attr_tags": {"medium:illustration": 7, "rating:safe": 6},
                            }
                            for img_id, _ in items
                        ]

                    def close(self):
                        pass

                class _DummyCCIP:
                    def __init__(self, _runtime):
                        pass

                    def extract_batch(self, items):
                        out = np.zeros((len(items), 768), dtype=np.float32)
                        out[:, 0] = 1.0
                        return out

                    def close(self):
                        pass

                with (
                    patch("artseek.indexing.jobs.build_tagger", return_value=_DummyTagger()),
                    patch("artseek.indexing.jobs.VisionAttributeAnalyzer", _DummyAnalyzer),
                    patch("artseek.indexing.jobs.CCIPExtractor", _DummyCCIP),
                    patch("artseek.indexing.jobs.release_accelerator", return_value=None),
                ):
                    def _wait(job_ids: tuple[str, ...]) -> None:
                        deadline = time.monotonic() + 15.0
                        while time.monotonic() < deadline:
                            statuses = [jobs_repo.get(jid)["status"] for jid in job_ids]
                            if all(s in {"completed", "failed", "stopped"} for s in statuses):
                                return
                            time.sleep(0.05)

                    j1 = mgr.submit_retag_job(skip_completed=True)
                    j2 = mgr.submit_vision_enrich_job(skip_completed=True)
                    # CCIP crops from the vision output, so it may only be queued afterwards.
                    _wait((j1, j2))
                    j3 = mgr.submit_ccip_job(skip_completed=True)
                    _wait((j3,))

                mgr.close()
                for jid in (j1, j2, j3):
                    job = jobs_repo.get(jid)
                    self.assertEqual(job["status"], "completed", f"job={job}")
                # Only p2 needed retag; p1 already had camie_v2+pixai_v09!
                self.assertEqual(jobs_repo.get(j1)["summary"]["updated"], 1)
                self.assertEqual(jobs_repo.get(j2)["summary"]["updated"], 2)
                self.assertEqual(jobs_repo.get(j3)["summary"]["updated"], 2)

                summary = TaskStateRepo(db).summary_counts()
                self.assertEqual(summary["vision_done"], 2)
                self.assertEqual(summary["ccip_done"], 2)
                self.assertEqual(summary["camie_v2_done"], 2)
                self.assertEqual(summary["pixai_v09_done"], 2)
                self.assertIsNotNone(vectors.ccip_vector(1))
            finally:
                db.close()

    def test_ccip_gate_and_reset(self) -> None:
        """CCIP refuses to run before vision crops exist; reset clears vectors and flags."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            img_dir = root / "imgs"
            img_dir.mkdir()
            img_path = img_dir / "a.png"
            Image.new("RGB", (64, 64), (10, 20, 30)).save(img_path)

            cfg = AppConfig(
                data_dir=root / "data",
                cache_dir=root / "cache",
                tagger=TaggerConfig(backend="camie+pixai0.9"),
            )
            cfg.ensure_dirs()
            db = Database(cfg.db_path)
            try:
                images = ImageRepo(db)
                prov = ProvenanceRepo(db)
                jobs_repo = JobRepo(db)
                task_state = TaskStateRepo(db)
                vectors = VectorIndex(
                    cfg.visual_vectors_path,
                    cfg.descriptor_vectors_path,
                    ccip_path=cfg.ccip_vectors_path,
                )
                images.upsert_images_batch(
                    [
                        ImageUpsertRecord(
                            point_id="p1",
                            root_key="r1",
                            root_path=str(img_dir),
                            folder_key="r1",
                            folder_path=str(img_dir),
                            folder_mtime_ns=1,
                            path=str(img_path),
                            path_key=str(img_path).casefold(),
                            name=img_path.name,
                            ext=".png",
                            mtime_ns=10,
                            size_bytes=100,
                            width=64,
                            height=64,
                            content_hash="hash_0",
                            phash="0" * 16,
                            descriptor_text="x",
                            tags=["x"],
                            tag_scores={"x": 0.5},
                        )
                    ]
                )
                vectors.set_rows(
                    [0], np.eye(1, 8, dtype=np.float32), np.eye(1, 8, dtype=np.float32)
                )

                class _DummyCCIP:
                    def __init__(self, _runtime):
                        pass

                    def extract_batch(self, items):
                        out = np.zeros((len(items), 768), dtype=np.float32)
                        out[:, 0] = 1.0
                        return out

                    def close(self):
                        pass

                def _wait(job_ids: tuple[str, ...]) -> None:
                    deadline = time.monotonic() + 15.0
                    while time.monotonic() < deadline:
                        statuses = [jobs_repo.get(jid)["status"] for jid in job_ids]
                        if all(s in {"completed", "failed", "stopped"} for s in statuses):
                            return
                        time.sleep(0.05)

                lifecycle = ModelLifecycle(cfg, embedder_factory=_FakeEmbedder)
                prov_svc = ProvenanceAppService(cfg, images, prov, vectors)
                with (
                    patch("artseek.indexing.jobs.CCIPExtractor", _DummyCCIP),
                    patch("artseek.indexing.jobs.release_accelerator", return_value=None),
                ):
                    mgr = JobManager(
                        cfg, db, images, prov, jobs_repo, vectors, lifecycle, prov_svc
                    )
                    try:
                        with self.assertRaises(JobDependencyError):
                            mgr.submit_ccip_job(skip_completed=True)

                        forced = mgr.submit_ccip_job(skip_completed=True, require_vision=False)
                        _wait((forced,))
                        job = jobs_repo.get(forced)
                        self.assertEqual(job["status"], "completed", job)
                        # No crop boxes yet, so the whole-image fallback must be reported.
                        self.assertEqual(job["summary"]["missing_crop"], 1)

                        # The gate reads both the task flag and images.vision_meta_version.
                        with db.write_tx() as conn:
                            conn.execute(
                                "UPDATE images SET vision_meta_version = ?, "
                                "character_crop_json = ? WHERE id = 0",
                                (
                                    VISION_META_VERSION,
                                    '{"body":[0.1,0.1,0.9,0.9]}',
                                ),
                            )
                        task_state.mark_vision_done(
                            [("p1", 0, "hash_0")], version=VISION_META_VERSION
                        )
                        allowed = mgr.submit_ccip_job(skip_completed=False)
                        _wait((allowed,))
                        done = jobs_repo.get(allowed)
                        self.assertEqual(done["status"], "completed", done)
                        self.assertEqual(done["summary"]["missing_crop"], 0)
                    finally:
                        mgr.close()

                # Reset clears the stored vectors and the completion flags, nothing else.
                vectors.set_ccip_rows([0], np.ones((1, 768), dtype=np.float32))
                task_state.mark_ccip_done([("p1", 0, "hash_0")])
                self.assertIsNotNone(vectors.ccip_vector(0))

                self.assertEqual(TaskStateRepo(db).reset_ccip(), 1)
                self.assertEqual(vectors.clear_ccip_rows([0]), 1)
                self.assertIsNone(vectors.ccip_vector(0))
                self.assertEqual(TaskStateRepo(db).summary_counts()["ccip_done"], 0)
                visual = vectors.visual_vectors([0])[0]
                self.assertGreater(float(np.linalg.norm(visual)), 0.0)
            finally:
                db.close()

    def test_index_auto_enriches_vision_and_ccip(self) -> None:
        """Directory indexing automatically queues and completes vision_enrich and ccip."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            img_dir = root / "pics"
            img_dir.mkdir()
            for name in ("a.png", "b.png"):
                Image.new("RGB", (128, 128), (40, 80, 120)).save(img_dir / name)

            cfg = AppConfig(
                data_dir=root / "data",
                cache_dir=root / "cache",
                tagger=TaggerConfig(enabled=True, backend="camie+pixai0.9"),
            )
            cfg.ensure_dirs()
            db = Database(cfg.db_path)
            try:
                images = ImageRepo(db)
                prov = ProvenanceRepo(db)
                jobs_repo = JobRepo(db)
                vectors = VectorIndex(
                    cfg.visual_vectors_path,
                    cfg.descriptor_vectors_path,
                    ccip_path=cfg.ccip_vectors_path,
                )

                class _FakeEmbedder:
                    dimension = 8

                    def encode_images(self, items):
                        out = np.zeros((len(items), self.dimension), dtype=np.float32)
                        out[:, 0] = 1.0
                        return out

                    def encode_texts(self, texts):
                        out = np.zeros((len(texts), self.dimension), dtype=np.float32)
                        out[:, 1] = 1.0
                        return out

                    def close(self):
                        pass

                class _DummyTagger:
                    model_versions = {"camie_v2": 1, "pixai_v09": 1}

                    def predict(self, items):
                        return [
                            PredictedTags(
                                tags=["1girl", "blue_hair"],
                                scores={"1girl": 0.9, "blue_hair": 0.85},
                                categories={"1girl": 0, "blue_hair": 0},
                            )
                            for _ in items
                        ]

                    def close(self):
                        pass

                class _DummyAnalyzer:
                    def __init__(self, _runtime):
                        pass

                    def analyze_batch(self, items):
                        return [
                            {
                                "image_id": img_id,
                                "media_type": "illustration",
                                "rating": "safe",
                                "is_monochrome": 0,
                                "is_ai": 0,
                                "aesthetic_score": 0.85,
                                "focal_x": 0.5,
                                "focal_y": 0.4,
                                "character_crop_json": '{"body":[0.1,0.1,0.9,0.9]}',
                                "vision_meta_version": VISION_META_VERSION,
                                "attr_tags": {
                                    "medium:illustration": CAT_MEDIUM,
                                    "rating:safe": CAT_RATING,
                                },
                            }
                            for img_id, _ in items
                        ]

                    def close(self):
                        pass

                class _DummyCCIP:
                    def __init__(self, _runtime):
                        pass

                    def extract_batch(self, items):
                        out = np.zeros((len(items), 768), dtype=np.float32)
                        out[:, 0] = 1.0
                        return out

                    def close(self):
                        pass

                lifecycle = ModelLifecycle(cfg, embedder_factory=_FakeEmbedder)
                prov_svc = ProvenanceAppService(cfg, images, prov, vectors)
                with (
                    patch("artseek.ml.tagger.build_tagger", return_value=_DummyTagger()),
                    patch("artseek.indexing.jobs.build_tagger", return_value=_DummyTagger()),
                    patch("artseek.indexing.jobs.VisionAttributeAnalyzer", _DummyAnalyzer),
                    patch("artseek.indexing.jobs.CCIPExtractor", _DummyCCIP),
                    patch("artseek.indexing.jobs.release_accelerator", return_value=None),
                ):
                    mgr = JobManager(
                        cfg, db, images, prov, jobs_repo, vectors, lifecycle, prov_svc
                    )
                    try:
                        mgr.submit_index_job([img_dir], recursive=True, captions=False)
                        deadline = time.monotonic() + 15.0
                        while time.monotonic() < deadline:
                            counts = TaskStateRepo(db).summary_counts()
                            if (
                                counts["total"] == 2
                                and counts["vision_done"] == 2
                                and counts["ccip_done"] == 2
                                and not mgr.is_busy()
                            ):
                                break
                            time.sleep(0.05)
                    finally:
                        mgr.close()

                counts = TaskStateRepo(db).summary_counts()
                recent = jobs_repo.list_recent(limit=10)
                self.assertEqual(counts["total"], 2, recent)
                self.assertEqual(counts["camie_v2_done"], 2, recent)
                self.assertEqual(counts["pixai_v09_done"], 2, recent)
                self.assertEqual(counts["vision_done"], 2, recent)
                self.assertEqual(counts["ccip_done"], 2, recent)

                for pid in images.point_ids_for_rows([0, 1]).values():
                    payload = images.get_by_point_id(pid)
                    self.assertIsNotNone(payload)
                    assert payload is not None
                    self.assertIn("1girl", payload["tags"])
                    self.assertIn("medium:illustration", payload["tags"])
                    self.assertIn("rating:safe", payload["tags"])
            finally:
                db.close()


if __name__ == "__main__":
    unittest.main()
