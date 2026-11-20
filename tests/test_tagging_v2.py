"""Unit tests for Schema v4, TaskStateRepo, tag postprocessing, and multi-model tagger fusion."""

from __future__ import annotations

import csv
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from artseek.config import EmbeddingConfig, TaggerConfig
from artseek.db.connection import Database
from artseek.db.migrations import LATEST_VERSION, MIGRATIONS
from artseek.db.repos.images import ImageRepo, ImageUpsertRecord
from artseek.db.repos.task_state import TaskStateRepo, sync_task_state_from_index_locked
from artseek.ml.tag_postprocess import (
    CAT_ATTRIBUTE,
    CAT_CHARACTER,
    CAT_COPYRIGHT,
    CAT_GENERAL,
    CAT_MEDIUM,
    CAT_RATING,
    PredictedTags,
    build_high_density_descriptor,
    mcut_threshold,
    merge_predicted_tags,
    prune_overlapping_for_summary,
)
from artseek.ml.tagger import CompositeTagger, PixAITaggerV09


class _FakeORTTensor:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakePixAISession:
    def __init__(self, logits_rows: np.ndarray) -> None:
        self._logits = logits_rows
        self.last_input_shape: tuple[int, ...] | None = None
        self.last_outputs: list[str] | None = None

    def get_inputs(self) -> list[_FakeORTTensor]:
        return [_FakeORTTensor("input")]

    def get_outputs(self) -> list[_FakeORTTensor]:
        return [
            _FakeORTTensor("embedding"),
            _FakeORTTensor("logits"),
            _FakeORTTensor("prediction"),
        ]

    def run(self, outputs: list[str], feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        arr = feed["input"]
        self.last_input_shape = tuple(arr.shape)
        self.last_outputs = list(outputs)
        return [self._logits[: arr.shape[0]]]


def _make_record(
    point_id: str,
    name: str,
    *,
    content_hash: str = "h1",
    descriptor_text: str = "desc",
    tags: list[str] | None = None,
    tag_scores: dict[str, float] | None = None,
    tag_categories: dict[str, int] | None = None,
    mtime_ns: int = 10,
) -> ImageUpsertRecord:
    return ImageUpsertRecord(
        point_id=point_id,
        root_key="root1",
        root_path="D:/art",
        folder_key="root1",
        folder_path="D:/art",
        folder_mtime_ns=1,
        path=f"D:/art/{name}",
        path_key=f"d:/art/{name}",
        name=name,
        ext=".png",
        mtime_ns=mtime_ns,
        size_bytes=123,
        width=100,
        height=100,
        content_hash=content_hash,
        phash="0" * 16,
        descriptor_text=descriptor_text,
        tags=tags or [],
        tag_scores=tag_scores or {},
        tag_categories=tag_categories or {},
    )


class TestSchemaV4AndTaskState(unittest.TestCase):
    def test_v4_migration_backfills_doc_count_and_task_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "test.sqlite3"
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row
            for ver, fn in MIGRATIONS:
                if ver < 4:
                    fn(conn)
                    conn.execute(f"PRAGMA user_version = {ver}")
            conn.execute(
                "INSERT INTO roots(root_key, root_path) VALUES ('rk', 'D:/art')"
            )
            root_id = int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])
            conn.execute(
                "INSERT INTO folders(root_id, folder_key, path, name) "
                "VALUES (?, 'rk', 'D:/art', 'art')",
                (root_id,),
            )
            folder_id = int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])
            conn.execute(
                """
                INSERT INTO images(
                    id, point_id, root_id, folder_id, path, path_key, name,
                    width, height, size_bytes, mtime_ns, content_hash,
                    descriptor_text, caption_json, deleted
                ) VALUES (
                    0, 'p1', ?, ?, 'D:/art/a.png', 'd:/art/a.png', 'a.png',
                    100, 100, 10, 1, 'h1', 'desc', '{"short":"x"}', 0
                )
                """,
                (root_id, folder_id),
            )
            conn.execute("INSERT INTO tags(name, category) VALUES ('blue_eyes', 0)")
            tag_id = int(
                conn.execute("SELECT id FROM tags WHERE name = 'blue_eyes'").fetchone()[0]
            )
            conn.execute(
                "INSERT INTO image_tags(image_id, tag_id, score) VALUES (0, ?, 0.9)",
                (tag_id,),
            )
            conn.execute(
                "INSERT INTO meta(key, value) VALUES ('tagger_model', 'wd-eva02-large-tagger-v3')"
            )
            conn.commit()
            conn.close()

            db = Database(db_path)
            try:
                version = int(db.read_conn().execute("PRAGMA user_version").fetchone()[0])
                self.assertEqual(version, LATEST_VERSION)

                doc_count = db.read_conn().execute(
                    "SELECT doc_count FROM tags WHERE name = 'blue_eyes'"
                ).fetchone()[0]
                self.assertEqual(doc_count, 1)

                ts = TaskStateRepo(db).get("p1")
                self.assertIsNotNone(ts)
                assert ts is not None
                self.assertTrue(ts["index_done"])
                self.assertTrue(ts["descriptor_done"])
                self.assertTrue(ts["caption_done"])
                self.assertEqual(ts["tag_models"], {})
            finally:
                db.close()

    def test_task_state_skip_and_content_hash_invalidation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Database(Path(tmp) / "test.sqlite3")
            try:
                repo = ImageRepo(db)
                tasks = TaskStateRepo(db)

                repo.upsert_images_batch(
                    [
                        _make_record(
                            "p1",
                            "a.png",
                            content_hash="hash_v1",
                            descriptor_text="hatsune miku",
                            tags=["hatsune_miku"],
                            tag_scores={"hatsune_miku": 0.9},
                            tag_categories={"hatsune_miku": CAT_CHARACTER},
                            mtime_ns=10,
                        )
                    ],
                    tag_model_versions={"camie_v2": 1, "pixai_v09": 1},
                )
                repo.upsert_images_batch(
                    [
                        _make_record(
                            "p2",
                            "b.png",
                            content_hash="hash_b",
                            descriptor_text="landscape",
                            tags=["sky"],
                            tag_scores={"sky": 0.8},
                            tag_categories={"sky": CAT_GENERAL},
                            mtime_ns=20,
                        )
                    ],
                    tag_model_versions={"camie_v2": 1},
                )

                pending = tasks.pending_for_tagger(
                    {"camie_v2": 1, "pixai_v09": 1}, skip_completed=True
                )
                self.assertEqual([item[1] for item in pending], ["p2"])

                all_items = tasks.pending_for_tagger(
                    {"camie_v2": 1, "pixai_v09": 1}, skip_completed=False
                )
                self.assertEqual(len(all_items), 2)

                img1 = repo.row_id_for_point("p1")
                assert img1 is not None
                tasks.mark_vision_done([("p1", img1, "hash_v1")], version=1)
                tasks.mark_ccip_done([("p1", img1, "hash_v1")])
                self.assertTrue(tasks.get("p1")["vision_done"])
                self.assertTrue(tasks.get("p1")["ccip_done"])

                with db.write_tx() as conn:
                    sync_task_state_from_index_locked(
                        conn,
                        [("p1", img1, "hash_v2", True)],
                        tag_model_versions={"camie_v2": 1},
                    )
                after = tasks.get("p1")
                assert after is not None
                self.assertEqual(after["content_hash"], "hash_v2")
                self.assertFalse(after["vision_done"])
                self.assertFalse(after["ccip_done"])
                self.assertEqual(after["tag_models"], {"camie_v2": 1})

                summary = tasks.summary_counts()
                self.assertEqual(summary["total"], 2)
                self.assertEqual(summary["index_done"], 2)
            finally:
                db.close()

    def test_update_tags_preserves_vision_attributes_and_doc_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Database(Path(tmp) / "test.sqlite3")
            try:
                repo = ImageRepo(db)
                repo.upsert_images_batch(
                    [
                        _make_record(
                            "p1",
                            "a.png",
                            content_hash="h1",
                            descriptor_text="old",
                            tags=["old_tag"],
                            tag_scores={"old_tag": 0.9},
                        )
                    ]
                )
                img_id = repo.row_id_for_point("p1")
                assert img_id is not None
                repo.update_vision_attributes_batch(
                    [
                        {
                            "image_id": img_id,
                            "media_type": "illustration",
                            "rating": "safe",
                            "is_monochrome": 0,
                            "monochrome_score": 0.05,
                            "is_ai": 0,
                            "ai_source": "human",
                            "ai_score": 0.02,
                            "aesthetic_score": 0.88,
                            "shot_type": "upper_body",
                            "focal_x": 0.52,
                            "focal_y": 0.28,
                            "attr_tags": {
                                "medium:illustration": CAT_MEDIUM,
                                "rating:safe": CAT_RATING,
                                "hand_drawn": CAT_ATTRIBUTE,
                            },
                        }
                    ]
                )
                repo.update_tags_and_descriptor_batch(
                    [
                        (
                            img_id,
                            ["rem_(re:zero)", "re:zero_kara_hajimeru_isekai_seikatsu", "maid"],
                            {
                                "rem_(re:zero)": 0.95,
                                "re:zero_kara_hajimeru_isekai_seikatsu": 0.91,
                                "maid": 0.84,
                            },
                            {
                                "rem_(re:zero)": CAT_CHARACTER,
                                "re:zero_kara_hajimeru_isekai_seikatsu": CAT_COPYRIGHT,
                                "maid": CAT_GENERAL,
                            },
                            "new descriptor",
                        )
                    ]
                )
                payload = repo.get_payloads_by_point_ids(["p1"])["p1"]
                self.assertEqual(payload["media_type"], "illustration")
                self.assertEqual(payload["rating"], "safe")
                self.assertAlmostEqual(payload["focal_y"], 0.28, places=2)
                self.assertIn("medium:illustration", payload["tags"])
                self.assertIn("rem_(re:zero)", payload["tags"])
                self.assertNotIn("old_tag", payload["tags"])
                total_docs, counts = repo.tag_doc_counts()
                self.assertEqual(total_docs, 1)
                self.assertEqual(counts.get("old_tag", 0), 0)
                self.assertEqual(counts.get("rem_(re:zero)"), 1)
            finally:
                db.close()


class TestTagPostprocessingAndFusion(unittest.TestCase):
    def test_mcut_overlap_merge_and_descriptor_packing(self) -> None:
        cut = mcut_threshold(
            [0.92, 0.88, 0.85, 0.32, 0.28], min_threshold=0.25, max_threshold=0.65
        )
        self.assertAlmostEqual(cut, 0.585, places=3)

        pruned = prune_overlapping_for_summary(
            ["serafuku", "school_uniform", "sailor_collar", "blue_sky"]
        )
        self.assertEqual(pruned, ["serafuku", "blue_sky"])

        p_camie = PredictedTags(
            tags=["blue_eyes", "serafuku", "1girl"],
            scores={"blue_eyes": 0.80, "serafuku": 0.72, "1girl": 0.98},
            categories={"blue_eyes": CAT_GENERAL, "serafuku": CAT_GENERAL, "1girl": CAT_GENERAL},
            rating="safe",
        )
        p_pixai = PredictedTags(
            tags=["hatsune_miku", "vocaloid", "serafuku", "school_uniform"],
            scores={
                "hatsune_miku": 0.86,
                "vocaloid": 0.82,
                "serafuku": 0.65,
                "school_uniform": 0.70,
            },
            categories={
                "hatsune_miku": CAT_CHARACTER,
                "vocaloid": CAT_COPYRIGHT,
                "serafuku": CAT_GENERAL,
                "school_uniform": CAT_GENERAL,
            },
        )
        merged = merge_predicted_tags([p_camie, p_pixai], max_tags=20)
        self.assertGreater(merged.scores["serafuku"], 0.80)
        self.assertEqual(merged.categories["hatsune_miku"], CAT_CHARACTER)
        self.assertEqual(merged.rating, "safe")

        desc = build_high_density_descriptor(
            merged.tags,
            merged.categories,
            title="Miku Fanart",
            drop_overlap=True,
        )
        self.assertIn("vocaloid, hatsune miku", desc)
        self.assertIn("serafuku", desc)
        self.assertNotIn("school uniform", desc)
        self.assertIn("Miku Fanart", desc)

    def test_pixai09_and_composite_decode_with_ip_projection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_path = root / "model.onnx"
            model_path.write_bytes(b"fake")
            tags_path = root / "selected_tags.csv"
            with tags_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["tag_id", "name", "category", "count", "ips"])
                writer.writerow(["1", "1girl", "0", "1000", "[]"])
                writer.writerow(["2", "blue_hair", "0", "800", "[]"])
                writer.writerow(["3", "furina_(genshin_impact)", "4", "500", '["genshin_impact"]'])

            fake_session = _FakePixAISession(np.array([[3.0, 2.0, 2.5]], dtype=np.float32))

            def fake_download(repo_id: str, filename: str) -> Path:
                return model_path if filename.endswith(".onnx") else tags_path

            with (
                patch("artseek.ml.tagger._cached_or_download", side_effect=fake_download),
                patch("artseek.ml.tagger.create_ort_session", return_value=fake_session),
            ):
                cfg = TaggerConfig(
                    backend="pixai0.9",
                    mcut_enabled=False,
                    general_threshold=0.35,
                    character_threshold=0.5,
                )
                tagger = PixAITaggerV09(cfg, EmbeddingConfig(device="cpu"))
                img = Image.new("RGB", (64, 64), (255, 255, 255))
                preds = tagger.predict([img])

            self.assertEqual(fake_session.last_input_shape, (1, 3, 448, 448))
            self.assertEqual(fake_session.last_outputs, ["prediction"])
            self.assertEqual(len(preds), 1)
            pred = preds[0]
            self.assertIn("furina_(genshin_impact)", pred.tags)
            self.assertIn("genshin_impact", pred.tags)
            self.assertEqual(pred.categories["genshin_impact"], CAT_COPYRIGHT)
            self.assertEqual(pred.categories["furina_(genshin_impact)"], CAT_CHARACTER)

            dummy_peer = SimpleNamespace(
                model_versions={"camie_v2": 1},
                predict=lambda imgs: [
                    PredictedTags(
                        tags=["blue_hair", "twintails"],
                        scores={"blue_hair": 0.90, "twintails": 0.85},
                        categories={"blue_hair": CAT_GENERAL, "twintails": CAT_GENERAL},
                        rating="safe",
                    )
                    for _ in imgs
                ],
            )
            comp = CompositeTagger([dummy_peer, tagger])
            self.assertEqual(comp.model_versions, {"camie_v2": 1, "pixai_v09": 1})
            comp_preds = comp.predict([img])
            self.assertIn("genshin_impact", comp_preds[0].tags)
            self.assertEqual(comp_preds[0].rating, "safe")


if __name__ == "__main__":
    unittest.main()
