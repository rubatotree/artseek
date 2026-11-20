"""End-to-end tests for attribute filters, dictionary, IDF ranking, CCIP, and task APIs."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from artseek.api.app import create_app
from artseek.config import AppConfig, TaggerConfig
from artseek.core.query_plan import QueryPlan, static_expand
from artseek.core.ranking import fuse_and_rank
from artseek.core.scope import ImageScope
from artseek.core.tag_dictionary import expand_tag_aliases
from artseek.db.connection import Database
from artseek.db.repos.images import ImageRepo, ImageUpsertRecord
from artseek.db.repos.task_state import TaskStateRepo
from artseek.db.scope_resolver import ScopeResolver
from artseek.image_io import canonical_path
from artseek.ml.tag_postprocess import (
    CAT_ATTRIBUTE,
    CAT_CHARACTER,
    CAT_COPYRIGHT,
    CAT_MEDIUM,
    CAT_RATING,
    PredictedTags,
)
from artseek.vectors.index import VectorIndex


class _FakeEmbedder:
    dimension = 8

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dimension), dtype=np.float32)
        out[:, 0] = 1.0
        return out

    def encode_images(self, paths: list[Path]) -> np.ndarray:
        out = np.zeros((len(paths), self.dimension), dtype=np.float32)
        out[:, 0] = 1.0
        return out


class TestStageDSearchAndWebAPI(unittest.TestCase):
    def test_dictionary_lexicon_and_idf_ranking(self) -> None:
        self.assertIn("medium:illustration", expand_tag_aliases("插画"))
        self.assertIn("rating:r18", expand_tag_aliases("R18"))
        self.assertIn("genshin_impact", expand_tag_aliases("原神"))
        self.assertIn("hatsune_miku", expand_tag_aliases("初音未来"))

        plan_attr = static_expand("黑白 插画")
        self.assertIn("monochrome", plan_attr.positive_tags)
        self.assertIn("medium:illustration", plan_attr.positive_tags)

        # Rare high-IDF tag hit should outweigh generic low-IDF tag hit in fuse_and_rank.
        plan = QueryPlan(
            original="miku girl",
            semantic_zh="",
            positive_tags=["hatsune_miku", "1girl"],
        )
        payloads = {
            0: {
                "point_id": "p0",
                "tags": ["1girl"],
                "path": "a.png",
                "name": "a.png",
                "width": 1,
                "height": 1,
            },
            1: {
                "point_id": "p1",
                "tags": ["hatsune_miku"],
                "path": "b.png",
                "name": "b.png",
                "width": 1,
                "height": 1,
            },
        }
        items, _ = fuse_and_rank(
            [(0, 0.8), (1, 0.8)],
            [],
            payloads=payloads,
            plan=plan,
            descriptor_weight=0.0,
            keyword_bonus=0.1,
            tag_idf={"hatsune_miku": 4.5, "1girl": 0.3},
            limit=10,
        )
        self.assertEqual([item["id"] for item in items], ["p1", "p0"])

    def test_scope_attribute_filters_and_aesthetic_sort(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Database(Path(tmp) / "test.sqlite3")
            try:
                repo = ImageRepo(db)
                resolver = ScopeResolver(db)
                records = [
                    ImageUpsertRecord(
                        point_id="p_illus_safe",
                        root_key="r",
                        root_path="D:/art",
                        folder_key="r",
                        folder_path="D:/art",
                        folder_mtime_ns=1,
                        path="D:/art/1.png",
                        path_key="d:/art/1.png",
                        name="1.png",
                        ext=".png",
                        mtime_ns=10,
                        size_bytes=10,
                        width=100,
                        height=100,
                        content_hash="h1",
                        phash="0" * 16,
                    ),
                    ImageUpsertRecord(
                        point_id="p_comic_mono_ai",
                        root_key="r",
                        root_path="D:/art",
                        folder_key="r",
                        folder_path="D:/art",
                        folder_mtime_ns=1,
                        path="D:/art/2.png",
                        path_key="d:/art/2.png",
                        name="2.png",
                        ext=".png",
                        mtime_ns=20,
                        size_bytes=10,
                        width=100,
                        height=100,
                        content_hash="h2",
                        phash="1" * 16,
                    ),
                ]
                ids = repo.upsert_images_batch(records)
                repo.update_vision_attributes_batch(
                    [
                        {
                            "image_id": ids[0],
                            "media_type": "illustration",
                            "rating": "safe",
                            "is_monochrome": 0,
                            "is_ai": 0,
                            "aesthetic_score": 0.92,
                            "focal_x": 0.48,
                            "focal_y": 0.25,
                            "attr_tags": {
                                "medium:illustration": CAT_MEDIUM,
                                "rating:safe": CAT_RATING,
                            },
                        },
                        {
                            "image_id": ids[1],
                            "media_type": "comic",
                            "rating": "r18",
                            "is_monochrome": 1,
                            "is_ai": 1,
                            "aesthetic_score": 0.55,
                            "focal_x": 0.5,
                            "focal_y": 0.5,
                            "attr_tags": {
                                "medium:comic": CAT_MEDIUM,
                                "rating:r18": CAT_RATING,
                                "monochrome": CAT_ATTRIBUTE,
                            },
                        },
                    ]
                )
                illus_rows = resolver.rows(
                    ImageScope(media_types=("illustration",), exclude_ai=True)
                )
                self.assertEqual(illus_rows.tolist(), [ids[0]])
                self.assertEqual(
                    resolver.rows(ImageScope(ratings=("r18",), monochrome=True)).tolist(),
                    [ids[1]],
                )
                sorted_ids, total, _, norm_sort, _ = resolver.page_rows(
                    ImageScope(), sort="aesthetic_desc", limit=10
                )
                self.assertEqual(norm_sort, "aesthetic_desc")
                self.assertEqual(total, 2)
                self.assertEqual(sorted_ids, [ids[0], ids[1]])
            finally:
                db.close()

    def test_web_api_character_search_filters_ignore_url_and_single_retag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            img_dir = root / "art"
            img_dir.mkdir()
            p1 = img_dir / "furina_1.png"
            p2 = img_dir / "furina_2.png"
            Image.new("RGB", (128, 128), (100, 120, 255)).save(p1)
            Image.new("RGB", (128, 128), (220, 220, 220)).save(p2)

            cfg = AppConfig(
                data_dir=root / "data",
                cache_dir=root / "cache",
                tagger=TaggerConfig(enabled=False),
            )
            cfg.ensure_dirs()
            db = Database(cfg.db_path)
            images = ImageRepo(db)
            fk = canonical_path(img_dir)
            rec_ids = images.upsert_images_batch(
                [
                    ImageUpsertRecord(
                        point_id="id1",
                        root_key=fk,
                        root_path=str(img_dir),
                        folder_key=fk,
                        folder_path=str(img_dir),
                        folder_mtime_ns=1,
                        path=str(p1),
                        path_key=canonical_path(p1),
                        name=p1.name,
                        ext=".png",
                        mtime_ns=100,
                        size_bytes=100,
                        width=128,
                        height=128,
                        content_hash="ch1",
                        phash="0" * 16,
                        descriptor_text="genshin_impact, furina",
                        tags=["genshin_impact", "furina"],
                        tag_scores={"genshin_impact": 0.9, "furina": 0.95},
                        tag_categories={"genshin_impact": CAT_COPYRIGHT, "furina": CAT_CHARACTER},
                    ),
                    ImageUpsertRecord(
                        point_id="id2",
                        root_key=fk,
                        root_path=str(img_dir),
                        folder_key=fk,
                        folder_path=str(img_dir),
                        folder_mtime_ns=1,
                        path=str(p2),
                        path_key=canonical_path(p2),
                        name=p2.name,
                        ext=".png",
                        mtime_ns=200,
                        size_bytes=100,
                        width=128,
                        height=128,
                        content_hash="ch2",
                        phash="1" * 16,
                        descriptor_text="genshin_impact, furina",
                        tags=["genshin_impact", "furina", "monochrome"],
                        tag_scores={"genshin_impact": 0.88, "furina": 0.9, "monochrome": 0.8},
                    ),
                ]
            )
            images.update_vision_attributes_batch(
                [
                    {
                        "image_id": rec_ids[0],
                        "media_type": "illustration",
                        "rating": "safe",
                        "is_monochrome": 0,
                        "is_ai": 0,
                        "aesthetic_score": 0.9,
                        "focal_x": 0.51,
                        "focal_y": 0.27,
                        "attr_tags": {"medium:illustration": CAT_MEDIUM, "rating:safe": CAT_RATING},
                    },
                    {
                        "image_id": rec_ids[1],
                        "media_type": "comic",
                        "rating": "r15",
                        "is_monochrome": 1,
                        "is_ai": 1,
                        "aesthetic_score": 0.6,
                        "focal_x": 0.4,
                        "focal_y": 0.3,
                        "attr_tags": {
                            "medium:comic": CAT_MEDIUM,
                            "rating:r15": CAT_RATING,
                            "monochrome": CAT_ATTRIBUTE,
                        },
                    },
                ]
            )
            db.close()

            vecs = VectorIndex(
                cfg.visual_vectors_path,
                cfg.descriptor_vectors_path,
                ccip_path=cfg.ccip_vectors_path,
            )
            vecs.set_rows(
                [0, 1],
                np.ones((2, 8), dtype=np.float32),
                np.ones((2, 8), dtype=np.float32),
            )
            ccip_vecs = np.zeros((2, 768), dtype=np.float32)
            ccip_vecs[0, 0] = 1.0
            ccip_vecs[1, 0] = 0.9
            ccip_vecs[1, 1] = 0.2
            vecs.set_ccip_rows([0, 1], ccip_vecs)
            vecs.flush()

            class _FakeSingleTagger:
                model_versions = {"camie_v2": 1, "pixai_v09": 1}

                def predict(self, imgs):
                    return [
                        PredictedTags(
                            tags=["hatsune_miku", "vocaloid"],
                            scores={"hatsune_miku": 0.94, "vocaloid": 0.89},
                            categories={"hatsune_miku": CAT_CHARACTER, "vocaloid": CAT_COPYRIGHT},
                        )
                        for _ in imgs
                    ]

                def close(self):
                    pass

            app = create_app(cfg, embedder_factory=_FakeEmbedder)  # type: ignore[arg-type]
            with TestClient(app) as client:
                # 1. Task stats endpoint
                stats = client.get("/api/jobs/task-stats")
                self.assertEqual(stats.status_code, 200)
                self.assertEqual(stats.json()["counts"]["total"], 2)

                # 2. Local search with projected attribute tag + focal exposure
                loc = client.post(
                    "/api/local-search",
                    json={"query": "插画", "folder_keys": [fk], "exclude_ai": True},
                )
                self.assertEqual(loc.status_code, 200)
                loc_res = loc.json()["results"]
                self.assertEqual([r["id"] for r in loc_res], ["id1"])
                self.assertAlmostEqual(loc_res[0]["focal_y"], 0.27, places=2)
                self.assertEqual(loc_res[0]["media_type"], "illustration")

                # 2b. Filter-only /api/search sorted by aesthetic_desc with no query
                aes = client.post(
                    "/api/search",
                    json={"query": "", "folder_keys": [fk], "sort": "aesthetic_desc"},
                )
                self.assertEqual(aes.status_code, 200)
                self.assertEqual([r["id"] for r in aes.json()["results"]], ["id1", "id2"])

                # 3. CCIP character search from id1 finds id2
                char_res = client.post(
                    "/api/search/character",
                    json={"reference_id": "id1", "folder_keys": [fk]},
                )
                self.assertEqual(char_res.status_code, 200)
                self.assertEqual([r["id"] for r in char_res.json()["results"]], ["id2"])
                self.assertGreater(char_res.json()["results"][0]["ccip_score"], 0.9)

                # 4. Mark URL ignored updates image_task_state url_done=True
                ign = client.post(
                    "/api/images/id1/provenance/ignore-url",
                    json={"ignored": True},
                )
                self.assertEqual(ign.status_code, 200)
                ts_db = Database(cfg.db_path)
                try:
                    ts = TaskStateRepo(ts_db).get("id1")
                    self.assertIsNotNone(ts)
                    self.assertTrue(ts["url_done"])
                finally:
                    ts_db.close()

                # 5. Single-image retag synchronously refreshes tags & preserves media_type
                with patch("artseek.indexing.jobs.build_tagger", return_value=_FakeSingleTagger()):
                    rt = client.post("/api/images/id1/retag", json={"backend": "camie+pixai0.9"})
                self.assertEqual(rt.status_code, 200)
                retagged = rt.json()["image"]
                self.assertIn("hatsune_miku", retagged["tags"])
                self.assertIn("medium:illustration", retagged["tags"])

                # 6. HTML page includes new filter, scheduler, and focal controls
                home = client.get("/")
                self.assertEqual(home.status_code, 200)
                for needle in (
                    "filter-media-type",
                    "filter-rating",
                    "task-tag-backend",
                    "runCharacterSearch",
                    "focalObjectPositionStyle",
                ):
                    self.assertIn(needle, home.text)


if __name__ == "__main__":
    unittest.main()
