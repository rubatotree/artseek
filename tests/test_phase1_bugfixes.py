"""Regression tests for Phase 1 bug fixes: WD preprocessing, IDF auto-tags, and dictionary."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from artseek.config import AppConfig
from artseek.core.ranking import order_tags_for_descriptor, rank_tags_by_idf
from artseek.core.tag_dictionary import tag_info
from artseek.db.connection import Database
from artseek.db.repos.images import ImageRepo, ImageUpsertRecord
from artseek.db.repos.provenance import ProvenanceRepo
from artseek.image_io import composite_on_white
from artseek.ml.tagger import WDTagger as V2WDTagger
from artseek.models.tagger import WDTagger as V1WDTagger
from artseek.provenance.parser import ParsedSource, SourceRef
from artseek.provenance.scan import scan_payloads
from artseek.services.provenance import ProvenanceAppService
from artseek.vectors.index import VectorIndex


class TaggerPreprocessTests(unittest.TestCase):
    def test_composite_on_white_flattens_transparent_rgba(self) -> None:
        img = Image.new("RGBA", (8, 8), (255, 0, 0, 0))
        img.putpixel((0, 0), (255, 0, 0, 255))
        flat = composite_on_white(img)
        self.assertEqual(flat.mode, "RGB")
        self.assertEqual(flat.getpixel((0, 0)), (255, 0, 0))
        self.assertEqual(flat.getpixel((4, 4)), (255, 255, 255))

    def test_v2_wd_preprocess_uses_white_pad_and_bgr(self) -> None:
        # 10x20 portrait: top half pure red (RGB 255,0,0), bottom half pure blue (RGB 0,0,255)
        img = Image.new("RGB", (10, 20), (255, 0, 0))
        for y in range(10, 20):
            for x in range(10):
                img.putpixel((x, y), (0, 0, 255))

        stub = object.__new__(V2WDTagger)
        stub.device = torch.device("cpu")
        stub.dtype = torch.float32
        stub._input_h = 20
        stub._input_w = 20
        stub._interp_mode = "bicubic"
        stub._mean = (0.5, 0.5, 0.5)
        stub._std = (0.5, 0.5, 0.5)

        batch = stub._preprocess_images_gpu([img])
        self.assertEqual(tuple(batch.shape), (1, 3, 20, 20))
        # Left pad column (x=0) must be white (+1.0 in all BGR channels)
        self.assertTrue(
            torch.allclose(batch[0, :, 10, 0], torch.tensor([1.0, 1.0, 1.0]), atol=1e-3)
        )
        # Center-top (y=2, x=10) was RGB red -> BGR is [-1.0, -1.0, +1.0]
        self.assertTrue(
            torch.allclose(batch[0, :, 2, 10], torch.tensor([-1.0, -1.0, 1.0]), atol=1e-2)
        )
        # Center-bottom (y=17, x=10) was RGB blue -> BGR is [+1.0, -1.0, -1.0]
        self.assertTrue(
            torch.allclose(batch[0, :, 17, 10], torch.tensor([1.0, -1.0, -1.0]), atol=1e-2)
        )

    def test_v1_wd_preprocess_matches_white_pad_and_bgr(self) -> None:
        img = Image.new("RGBA", (10, 20), (255, 0, 0, 255))
        stub = object.__new__(V1WDTagger)
        stub._input_h = 20
        stub._input_w = 20
        stub._mean = (0.5, 0.5, 0.5)
        stub._std = (0.5, 0.5, 0.5)

        tensor = stub._preprocess_one(img)
        self.assertEqual(tuple(tensor.shape), (3, 20, 20))
        self.assertTrue(
            torch.allclose(tensor[:, 10, 0], torch.tensor([1.0, 1.0, 1.0]), atol=1e-3)
        )
        self.assertTrue(
            torch.allclose(tensor[:, 2, 10], torch.tensor([-1.0, -1.0, 1.0]), atol=1e-2)
        )


class AutoTagAndRankingTests(unittest.TestCase):
    def test_order_and_rank_demotes_generic_boilerplate(self) -> None:
        ordered = order_tags_for_descriptor(
            ["1girl", "solo", "wide_sleeves", "looking_at_viewer", "knees_to_chest"]
        )
        self.assertEqual(
            ordered,
            ["wide_sleeves", "knees_to_chest", "1girl", "solo", "looking_at_viewer"],
        )

        tags, _ = rank_tags_by_idf(
            [
                ("1girl", 0.99),
                ("solo", 0.98),
                ("looking_at_viewer", 0.95),
                ("wide_sleeves", 0.72),
                ("hatsune_miku", 0.70),
            ],
            total_docs=1000,
            doc_counts={
                "1girl": 950,
                "solo": 900,
                "looking_at_viewer": 600,
                "wide_sleeves": 30,
                "hatsune_miku": 15,
            },
            limit=3,
        )
        self.assertEqual(tags[:2], ["hatsune_miku", "wide_sleeves"])

    def test_run_auto_tag_prefers_own_idf_tags_over_neighbor_votes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            cfg = AppConfig()
            cfg.data_dir = base / "data"
            cfg.ensure_dirs()
            db = Database(cfg.db_path)
            images = ImageRepo(db)
            prov = ProvenanceRepo(db)
            vectors = VectorIndex(cfg.visual_vectors_path, cfg.descriptor_vectors_path)
            try:
                p_main = base / "main.png"
                p_n1 = base / "n1.png"
                p_n2 = base / "n2.png"
                for p in (p_main, p_n1, p_n2):
                    Image.new("RGB", (32, 32), (10, 20, 30)).save(p)

                row_ids = images.upsert_images_batch(
                    [
                        ImageUpsertRecord(
                            point_id="pid-main",
                            root_key=str(base),
                            root_path=str(base),
                            folder_key=str(base),
                            folder_path=str(base),
                            folder_mtime_ns=1,
                            path=str(p_main),
                            path_key=str(p_main).casefold(),
                            name="main.png",
                            ext=".png",
                            mtime_ns=1,
                            size_bytes=100,
                            width=32,
                            height=32,
                            content_hash="1" * 64,
                            phash="",
                            tags=["1girl", "solo", "wide_sleeves", "hatsune_miku"],
                            tag_scores={
                                "1girl": 0.99,
                                "solo": 0.98,
                                "wide_sleeves": 0.80,
                                "hatsune_miku": 0.75,
                            },
                        ),
                        ImageUpsertRecord(
                            point_id="pid-n1",
                            root_key=str(base),
                            root_path=str(base),
                            folder_key=str(base),
                            folder_path=str(base),
                            folder_mtime_ns=1,
                            path=str(p_n1),
                            path_key=str(p_n1).casefold(),
                            name="n1.png",
                            ext=".png",
                            mtime_ns=1,
                            size_bytes=100,
                            width=32,
                            height=32,
                            content_hash="2" * 64,
                            phash="",
                            tags=["1girl", "solo", "red_eyes"],
                            tag_scores={"1girl": 0.99, "solo": 0.97, "red_eyes": 0.90},
                        ),
                        ImageUpsertRecord(
                            point_id="pid-n2",
                            root_key=str(base),
                            root_path=str(base),
                            folder_key=str(base),
                            folder_path=str(base),
                            folder_mtime_ns=1,
                            path=str(p_n2),
                            path_key=str(p_n2).casefold(),
                            name="n2.png",
                            ext=".png",
                            mtime_ns=1,
                            size_bytes=100,
                            width=32,
                            height=32,
                            content_hash="3" * 64,
                            phash="",
                            tags=["1girl", "solo", "red_eyes"],
                            tag_scores={"1girl": 0.99, "solo": 0.97, "red_eyes": 0.90},
                        ),
                    ]
                )
                vecs = np.array(
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.99, 0.1, 0.0, 0.0],
                        [0.98, 0.1, 0.0, 0.0],
                    ],
                    dtype=np.float32,
                )
                vectors.set_rows(row_ids, vecs, vecs)

                for pid, p, ch in (
                    ("pid-main", p_main, "1" * 64),
                    ("pid-n1", p_n1, "2" * 64),
                    ("pid-n2", p_n2, "3" * 64),
                ):
                    prov.upsert_unresolved(
                        pid,
                        {"path": str(p), "content_hash": ch, "mtime_ns": 1},
                        path_key=str(p).casefold(),
                    )

                svc = ProvenanceAppService(cfg, images, prov, vectors)
                svc.run_auto_tag(limit=10, top_tags=2)
                rec = prov.get("pid-main")
                assert rec is not None
                # Prefer pid-main's own high-IDF tags over neighbors' red_eyes or boilerplate
                self.assertEqual(set(rec["auto_tags"]), {"hatsune_miku", "wide_sleeves"})
            finally:
                db.close()

    def test_scan_skips_completed_ok_and_manual_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = Database(root / "artseek.sqlite3")
            store = ProvenanceRepo(db)
            try:
                img = root / "12345678_p0.png"
                Image.new("RGB", (16, 16), (20, 40, 60)).save(img)
                payload = {
                    "path": str(img),
                    "path_key": str(img).casefold(),
                    "width": 16,
                    "height": 16,
                    "mtime_ns": 1,
                    "content_hash": "a" * 64,
                }
                author = store.ensure_author("pixiv", "1", name="Tester")
                store.upsert_scan(
                    "p1",
                    payload,
                    ParsedSource(
                        SourceRef(
                            "pixiv",
                            "12345678",
                            0,
                            "https://www.pixiv.net/artworks/12345678",
                            "filename",
                            0.9,
                        )
                    ),
                    path_key=payload["path_key"],
                )
                store.set_fetch_result(
                    "p1",
                    {
                        "platform": "pixiv",
                        "source_id": "12345678",
                        "page": 0,
                        "url": "https://www.pixiv.net/artworks/12345678",
                        "status": "ok",
                        "title": "Done",
                    },
                    author_key=author,
                )
                summary = scan_payloads(
                    store, [("p1", payload)], total=1, deep=True, only_new=False
                )
                self.assertEqual(summary["skipped"], 1)
                self.assertEqual(summary["scanned"], 0)
            finally:
                db.close()

    def test_cleaned_translations_fix_mistranslations(self) -> None:
        ark = tag_info("Arknights")
        ba = tag_info("ブルーアーカイブ")
        maid = tag_info("メイド服")
        self.assertIsNotNone(ark)
        self.assertIsNotNone(ba)
        self.assertIsNotNone(maid)
        assert ark is not None and ba is not None and maid is not None
        self.assertEqual(ark.zh, "明日方舟")
        self.assertEqual(ba.zh, "蔚蓝档案")
        self.assertEqual(maid.zh, "女仆装")


if __name__ == "__main__":
    unittest.main()
