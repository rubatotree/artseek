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

                # 6. HTML page includes new filter, queue scheduler, and focal controls
                home = client.get("/")
                self.assertEqual(home.status_code, 200)
                for needle in (
                    "filter-media-type",
                    "filter-rating",
                    "filter-attr-row",
                    "artseek-select",
                    "task-tag-backend",
                    "moveQueuedTask",
                    "图库扫描与标注",
                    "runCharacterSearch",
                    "focalObjectPositionStyle",
                ):
                    self.assertIn(needle, home.text)

    def test_job_queue_reorder_delete_and_auto_fill(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            img_dir = root / "art"
            img_dir.mkdir()
            p1 = img_dir / "a.png"
            Image.new("RGB", (64, 64), (100, 120, 255)).save(p1)

            cfg = AppConfig(
                data_dir=root / "data",
                cache_dir=root / "cache",
                tagger=TaggerConfig(enabled=False),
            )
            cfg.ensure_dirs()
            db = Database(cfg.db_path)
            images = ImageRepo(db)
            fk = canonical_path(img_dir)
            images.upsert_images_batch(
                [
                    ImageUpsertRecord(
                        point_id="p1",
                        root_key=fk,
                        root_path=str(img_dir),
                        folder_key=fk,
                        folder_path=str(img_dir),
                        folder_mtime_ns=1,
                        path=str(p1),
                        path_key=canonical_path(p1),
                        name="a.png",
                        ext=".png",
                        mtime_ns=1,
                        size_bytes=10,
                        width=64,
                        height=64,
                        content_hash="h1",
                        phash="0" * 16,
                    )
                ]
            )
            db.close()

            app = create_app(cfg, embedder_factory=_FakeEmbedder)  # type: ignore[arg-type]
            with TestClient(app) as client:
                c = app.state.container
                # Block worker on _cond briefly while inspecting pure queue ops
                blocker = __import__("threading").Event()

                def _stall_wait() -> None:
                    blocker.wait(timeout=3.0)

                try:
                    c.job_manager._enqueue_job(
                        "index", "stall", {}, "stalling", lambda _jid: _stall_wait()
                    )
                    res_fill = client.post(
                        "/api/jobs/auto-fill", json={"backend": "camie+pixai0.9"}
                    )
                    self.assertEqual(res_fill.status_code, 200)
                    added = res_fill.json()["added"]
                    self.assertEqual(len(added), 3)  # retag, vision_enrich, ccip

                    # Auto-fill again immediately should add 0 duplicates
                    res_fill2 = client.post(
                        "/api/jobs/auto-fill", json={"backend": "camie+pixai0.9"}
                    )
                    self.assertEqual(res_fill2.status_code, 200)
                    self.assertEqual(res_fill2.json()["added"], [])

                    q0 = client.get("/api/jobs/queue").json()
                    q_ids = [j["id"] for j in q0["queued"]]
                    self.assertEqual(q_ids, added)

                    # Move last queued item (ccip) up once
                    mv = client.post(f"/api/jobs/{added[2]}/move", json={"direction": "up"})
                    self.assertEqual(mv.status_code, 200)
                    self.assertTrue(mv.json()["moved"])
                    after_mv = [j["id"] for j in mv.json()["queue"]["queued"]]
                    self.assertEqual(after_mv, [added[0], added[2], added[1]])

                    # Delete queued item
                    dl = client.delete(f"/api/jobs/{added[2]}")
                    self.assertEqual(dl.status_code, 200)
                    after_del = [j["id"] for j in dl.json()["queue"]["queued"]]
                    self.assertEqual(after_del, [added[0], added[1]])

                    # Clear remaining queued jobs via stop so teardown is instant
                    client.post("/api/jobs/stop")
                finally:
                    blocker.set()

    def test_purge_legacy_auto_tags_and_retag_syncs_provenance_auto_tags(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            img_dir = root / "art"
            img_dir.mkdir()
            p_legacy = img_dir / "legacy.png"
            p_new = img_dir / "new.png"
            Image.new("RGB", (64, 64), (10, 200, 220)).save(p_legacy)
            Image.new("RGB", (64, 64), (240, 200, 180)).save(p_new)

            cfg = AppConfig(
                data_dir=root / "data",
                cache_dir=root / "cache",
                tagger=TaggerConfig(enabled=False),
            )
            cfg.ensure_dirs()
            db = Database(cfg.db_path)
            images = ImageRepo(db)
            task_state = TaskStateRepo(db)
            fk = canonical_path(img_dir)
            rec_ids = images.upsert_images_batch(
                [
                    ImageUpsertRecord(
                        point_id="legacy_pid",
                        root_key=fk,
                        root_path=str(img_dir),
                        folder_key=fk,
                        folder_path=str(img_dir),
                        folder_mtime_ns=1,
                        path=str(p_legacy),
                        path_key=canonical_path(p_legacy),
                        name=p_legacy.name,
                        ext=".png",
                        mtime_ns=100,
                        size_bytes=100,
                        width=64,
                        height=64,
                        content_hash="ch_leg",
                        phash="0" * 16,
                        descriptor_text="old_wrong_tag",
                        tags=["old_wrong_tag"],
                        tag_scores={"old_wrong_tag": 0.9},
                        tag_categories={"old_wrong_tag": 0},
                    ),
                    ImageUpsertRecord(
                        point_id="new_pid",
                        root_key=fk,
                        root_path=str(img_dir),
                        folder_key=fk,
                        folder_path=str(img_dir),
                        folder_mtime_ns=1,
                        path=str(p_new),
                        path_key=canonical_path(p_new),
                        name=p_new.name,
                        ext=".png",
                        mtime_ns=200,
                        size_bytes=100,
                        width=64,
                        height=64,
                        content_hash="ch_new",
                        phash="1" * 16,
                        descriptor_text="keep_new_tag",
                        tags=["keep_new_tag"],
                        tag_scores={"keep_new_tag": 0.95},
                        tag_categories={"keep_new_tag": 0},
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
                        "aesthetic_score": 0.8,
                        "focal_x": 0.5,
                        "focal_y": 0.5,
                        "attr_tags": {"medium:illustration": CAT_MEDIUM},
                    }
                ]
            )
            task_state.mark_tag_models_done(
                [("new_pid", rec_ids[1], "ch_new")],
                {"camie_v2": 1, "pixai_v09": 1},
            )

            # Seed provenance records: manual + pixiv + legacy auto tags
            from artseek.db.repos.provenance import ProvenanceRepo

            prov = ProvenanceRepo(db)
            with db.write_tx() as conn:
                conn.execute(
                    """
                    INSERT INTO image_source (
                        point_id, path_key, platform, source_id, url,
                        tags_json, manual_tags_json, auto_tags_json, updated_at
                    ) VALUES (?, ?, 'pixiv', 'legacy', 'https://example.com/legacy',
                              '["keep_pixiv"]', ?, ?, 1)
                    """,
                    (
                        "legacy_pid",
                        canonical_path(p_legacy),
                        '["keep_manual"]',
                        '["old_24nn"]',
                    ),
                )
                conn.execute(
                    """
                    INSERT INTO image_source (
                        point_id, path_key, platform, source_id, url,
                        manual_tags_json, auto_tags_json, updated_at
                    ) VALUES (?, ?, 'pixiv', 'new', 'https://example.com/new', '[]', ?, 1)
                    """,
                    ("new_pid", canonical_path(p_new), '["keep_new_tag"]'),
                )
            prov.set_auto_tags_many([("legacy_pid", ["old_24nn"], {"old_24nn": 0.8})])
            db.close()

            vecs = VectorIndex(cfg.visual_vectors_path, cfg.descriptor_vectors_path)
            vecs.set_rows(
                [0, 1],
                np.ones((2, 8), dtype=np.float32),
                np.ones((2, 8), dtype=np.float32),
            )
            vecs.flush()

            class _FakeTagger:
                model_versions = {"camie_v2": 1, "pixai_v09": 1}

                def predict(self, imgs):
                    return [
                        PredictedTags(
                            tags=["hatsune_miku", "1girl"],
                            scores={"hatsune_miku": 0.96, "1girl": 0.91},
                            categories={"hatsune_miku": CAT_CHARACTER, "1girl": 0},
                        )
                        for _ in imgs
                    ]

                def close(self):
                    pass

            app = create_app(cfg, embedder_factory=_FakeEmbedder)  # type: ignore[arg-type]

            with (
                patch("artseek.indexing.jobs.build_tagger", return_value=_FakeTagger()),
                TestClient(app) as client,
            ):
                home = client.get("/")
                self.assertEqual(home.status_code, 200)
                self.assertIn("清理旧版错误标签", home.text)

                purge = client.post("/api/jobs/purge-legacy-tags")
                self.assertEqual(purge.status_code, 200)
                summary = purge.json()["summary"]
                self.assertEqual(summary["purged_images"], 1)
                self.assertGreaterEqual(summary["deleted_tag_links"], 1)
                self.assertEqual(summary["purged_source_auto_tags"], 1)

                # Legacy image: old_wrong_tag removed, medium:illustration preserved;
                # manual_tags and pixiv source tag preserved; old_24nn auto tag removed;
                # new_pid keep_new_tag preserved untouched.
                conn = app.state.container.db.read_conn()
                leg_tags = {
                    r["tag"]
                    for r in conn.execute(
                        "SELECT t.name AS tag FROM image_tags AS it "
                        "JOIN tags AS t ON t.id = it.tag_id WHERE it.image_id = ?",
                        (rec_ids[0],),
                    ).fetchall()
                }
                self.assertNotIn("old_wrong_tag", leg_tags)
                self.assertIn("medium:illustration", leg_tags)

                new_tags = {
                    r["tag"]
                    for r in conn.execute(
                        "SELECT t.name AS tag FROM image_tags AS it "
                        "JOIN tags AS t ON t.id = it.tag_id WHERE it.image_id = ?",
                        (rec_ids[1],),
                    ).fetchall()
                }
                self.assertIn("keep_new_tag", new_tags)

                prov_leg = conn.execute(
                    "SELECT manual_tags_json, auto_tags_json, auto_tagged_at "
                    "FROM image_source WHERE point_id = 'legacy_pid'"
                ).fetchone()
                self.assertEqual(prov_leg["manual_tags_json"], '["keep_manual"]')
                self.assertEqual(prov_leg["auto_tags_json"], "[]")
                self.assertIsNone(prov_leg["auto_tagged_at"])

                src_leg = {
                    (r["tag"], r["origin"])
                    for r in conn.execute(
                        "SELECT tag, origin FROM source_tags WHERE point_id = 'legacy_pid'"
                    ).fetchall()
                }
                self.assertIn(("keep_pixiv", "source"), src_leg)
                self.assertNotIn(("old_24nn", "auto"), src_leg)

                prov_new = conn.execute(
                    "SELECT auto_tags_json FROM image_source WHERE point_id = 'new_pid'"
                ).fetchone()
                self.assertEqual(prov_new["auto_tags_json"], '["keep_new_tag"]')

                # Single image retag must update both image_tags and image_source.auto_tags
                retag = client.post("/api/images/legacy_pid/retag", json={})
                self.assertEqual(retag.status_code, 200)
                prov_res = client.get("/api/images/legacy_pid/provenance")
                self.assertEqual(prov_res.status_code, 200)
                record = prov_res.json()["record"]
                self.assertIn("hatsune_miku", record["auto_tags"])
                self.assertNotIn("old_24nn", record["auto_tags"])
                self.assertEqual(record["manual_tags"], ["keep_manual"])


if __name__ == "__main__":
    unittest.main()
