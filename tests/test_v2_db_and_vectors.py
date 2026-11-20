"""Unit tests for ArtSeek v2 Database, Repositories, VectorIndex, and ScopeResolver."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from artseek.core.scope import ImageScope
from artseek.core.sort_keys import natural_sort_text
from artseek.db.connection import Database
from artseek.db.repos.collections import CollectionRepo
from artseek.db.repos.images import ImageRepo
from artseek.db.repos.jobs import JobRepo
from artseek.db.repos.notes import NotesIndexRepo
from artseek.db.repos.provenance import ProvenanceRepo
from artseek.db.repos.query_cache import QueryPlanCacheRepo
from artseek.db.scope_resolver import ScopeResolver
from artseek.image_io import canonical_path, point_id
from artseek.library import natural_key
from artseek.provenance.parser import ParsedSource, SourceRef
from artseek.vectors.index import VectorIndex


def _vec(values: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    return (arr / max(float(np.linalg.norm(arr)), 1e-12)).astype(np.float32)


class V2DataLayerTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.base = Path(self._tmp.name)
        self.db = Database(self.base / "artseek.sqlite3")
        self.images = ImageRepo(self.db)
        self.collections = CollectionRepo(self.db)
        self.prov = ProvenanceRepo(self.db)
        self.notes_idx = NotesIndexRepo(self.db)
        self.qcache = QueryPlanCacheRepo(self.db)
        self.jobs = JobRepo(self.db)
        self.resolver = ScopeResolver(self.db)
        self.vindex = VectorIndex(
            self.base / "visual.f32.npy",
            self.base / "descriptor.f32.npy",
        )

    def tearDown(self) -> None:
        self.vindex.release_cuda()
        self.db.close()
        self._tmp.cleanup()

    def test_natural_sort_text_matches_python_natural_key(self) -> None:
        names = [
            "2.jpg",
            "10.jpg",
            "a2",
            "a10",
            "A2",
            "bjd",
            "133",
            "b 1",
            "a01",
            "a1",
            "x100y2",
            "x100y10",
            "(1-027 白) タグマール.png",
            "00.jpg",
            "",
        ]
        self.assertEqual(
            sorted(names, key=natural_key),
            sorted(names, key=natural_sort_text),
        )

    def _seed_sample_library(self) -> dict[str, tuple[int, str]]:
        root = self.base / "pics"
        sub = root / "sub"
        sub.mkdir(parents=True)
        files = [
            ("img2", root / "2.png", 100, ["wide_sleeves", "sitting"], [1.0, 0.0, 0.0, 0.0]),
            ("img10", sub / "10.png", 200, ["wide_sleeves", "standing"], [0.8, 0.6, 0.0, 0.0]),
            ("img3", sub / "3.png", 300, ["shirt", "smile"], [0.0, 0.0, 1.0, 0.0]),
        ]
        records = []
        ids: dict[str, str] = {}
        vis_rows = []
        desc_rows = []
        for alias, path, mtime_ns, tags, vec in files:
            path.write_bytes(b"png")
            pid = point_id(path)
            ids[alias] = pid
            records.append(
                ImageRepo.build_upsert_record(
                    path,
                    root=root,
                    point_id=pid,
                    width=100,
                    height=100,
                    mtime_ns=mtime_ns,
                    size_bytes=3,
                    content_hash=f"{alias:064s}".replace(" ", "a"),
                    phash="0123456789abcdef",
                    tags=tags,
                    tag_scores={t: 0.9 for t in tags},
                    descriptor_text=", ".join(tags),
                )
            )
            vis_rows.append(_vec(vec))
            desc_rows.append(_vec(list(reversed(vec))))
        row_ids = self.images.upsert_images_batch(records)
        self.vindex.set_rows(row_ids, np.stack(vis_rows), np.stack(desc_rows))
        self.vindex.flush()
        return {
            alias: (row_id, ids[alias])
            for (alias, *_), row_id in zip(files, row_ids, strict=True)
        }

    def test_vector_index_flush_search_votes_and_compact_with_fks(self) -> None:
        seeded = self._seed_sample_library()
        self.assertEqual([seeded["img2"][0], seeded["img10"][0], seeded["img3"][0]], [0, 1, 2])
        self.assertTrue((self.base / "visual.f32.npy").is_file())
        self.assertFalse((self.base / "visual.f32.npy.tmp.npy").exists())

        # Reload from disk and verify search
        reloaded = VectorIndex(
            self.base / "visual.f32.npy",
            self.base / "descriptor.f32.npy",
            active_rows=self.images.active_row_ids(),
        )
        rows, scores = reloaded.search(_vec([1.0, 0.0, 0.0, 0.0]), "visual", 2)
        self.assertEqual(rows.tolist(), [0, 1])
        self.assertGreater(scores[0], scores[1])

        # Subset search
        sub_rows, _ = reloaded.search(
            _vec([1.0, 0.0, 0.0, 0.0]), "visual", 5, rows=np.array([2], dtype=np.int64)
        )
        self.assertEqual(sub_rows.tolist(), [2])

        # Neighbor tag voting
        votes = reloaded.neighbor_tag_votes([0], self.images.row_tags_map(), limit=2, top_tags=2)
        self.assertIn(0, votes)
        self.assertTrue(votes[0][0])

        # Provenance + Collection reference row 1 and row 2 before soft-deleting row 1
        pid2 = seeded["img2"][1]
        pid10 = seeded["img10"][1]
        pid3 = seeded["img3"][1]
        col = self.collections.create("测试")
        self.collections.add_item(col.id, pid2)
        self.collections.add_item(col.id, pid3)
        self.prov.upsert_scan(
            pid3,
            {"width": 100, "height": 100, "mtime_ns": 300},
            ParsedSource(SourceRef("pixiv", "999", 0, "https://www.pixiv.net/artworks/999")),
        )
        self.prov.set_manual(pid3, {"manual_tags": ["thighhighs"]})

        # Delete middle image (row 1 = img10) then compact
        deleted_rows = self.images.soft_delete_by_point_ids([pid10])
        self.assertEqual(deleted_rows, [1])
        reloaded.zero_rows(deleted_rows)
        rows_after_del, _ = reloaded.search(_vec([0.8, 0.6, 0.0, 0.0]), "visual", 5)
        self.assertNotIn(1, rows_after_del.tolist())

        old_order = self.images.compact_ids()
        self.assertEqual(old_order, [0, 2])
        reloaded.compact(old_order)
        self.assertEqual(reloaded.count, 2)
        self.assertEqual(self.images.row_id_for_point(pid2), 0)
        self.assertEqual(self.images.row_id_for_point(pid3), 1)
        # Verify FK tables updated to dense row ID 1 for pid3
        row_src = self.db.read_conn().execute(
            "SELECT image_id FROM image_source WHERE point_id = ?", (pid3,)
        ).fetchone()
        self.assertEqual(int(row_src["image_id"]), 1)
        row_ci = self.db.read_conn().execute(
            "SELECT image_id FROM collection_items WHERE item_id = ?", (pid3,)
        ).fetchone()
        self.assertEqual(int(row_ci["image_id"]), 1)

    def test_scope_resolver_filters_sorts_and_large_folder_lists(self) -> None:
        seeded = self._seed_sample_library()
        root_key = canonical_path(self.base / "pics")
        sub_key = canonical_path(self.base / "pics" / "sub")

        alice = self.prov.ensure_author("pixiv", "10", name="Alice", handle="alice")
        for alias, mtime_ns, tags, author in (
            ("img2", 100, ["thighhighs"], alice),
            ("img10", 200, ["smile"], None),
            ("img3", 300, ["thighhighs", "barefoot"], alice),
        ):
            _, pid = seeded[alias]
            self.prov.upsert_scan(
                pid,
                {"width": 100, "height": 100, "mtime_ns": mtime_ns},
                ParsedSource(SourceRef("pixiv", "100", 0, "https://www.pixiv.net/artworks/100")),
            )
            self.prov.set_fetch_result(
                pid,
                {
                    "platform": "pixiv",
                    "source_id": "100",
                    "page": 0,
                    "url": "https://www.pixiv.net/artworks/100",
                    "status": "ok",
                    "title": f"Work {alias}",
                    "tags": tags,
                },
                author_key=author,
            )

        # Unfiltered returns None
        self.assertIsNone(self.resolver.rows(ImageScope()))

        # Direct vs recursive folder scope
        direct_root = self.resolver.rows(ImageScope(folders=(root_key,), folder_recursive=False))
        self.assertEqual(direct_root.tolist(), [0])
        rec_root = self.resolver.rows(ImageScope(folders=(root_key,), folder_recursive=True))
        self.assertEqual(rec_root.tolist(), [0, 1, 2])

        # >1000 folder keys without hitting SQLite parameter limit
        many_folders = tuple([sub_key] + [f"{sub_key}/missing_{i}" for i in range(1200)])
        many_rows = self.resolver.rows(ImageScope(folders=many_folders))
        self.assertEqual(many_rows.tolist(), [1, 2])

        # Author + "none" + source tag alias expansion ("过膝袜" -> "thighhighs") + WD tags_none
        none_rows = self.resolver.rows(ImageScope(author_keys=("none",)))
        self.assertEqual(none_rows.tolist(), [1])

        alice_socks = self.resolver.rows(
            ImageScope(
                author_keys=("pixiv:10",),
                source_tags_all=("过膝袜",),
                tags_none=("sitting",),
            )
        )
        self.assertEqual(alice_socks.tolist(), [2])

        # Boolean tag expression
        expr_rows = self.resolver.rows(
            ImageScope(tag_expression="#thighhighs and not #barefoot")
        )
        self.assertEqual(expr_rows.tolist(), [0])

        # Natural name sort ("2.png" before "3.png" before "10.png")
        page, total, has_more, norm_sort, _ = self.resolver.page_rows(
            ImageScope(folders=(root_key,), folder_recursive=True),
            sort="name_asc",
            limit=10,
            offset=0,
        )
        self.assertEqual(total, 3)
        self.assertFalse(has_more)
        self.assertEqual(norm_sort, "name_asc")
        self.assertEqual(page, [0, 2, 1])

        # Deterministic random sort
        r1, *_ = self.resolver.page_rows(
            ImageScope(folders=(root_key,), folder_recursive=True), sort="random", seed=77
        )
        r2, *_ = self.resolver.page_rows(
            ImageScope(folders=(root_key,), folder_recursive=True), sort="random", seed=77
        )
        self.assertEqual(r1, r2)

    def test_repos_notes_query_cache_jobs_and_collections(self) -> None:
        self.notes_idx.sync_hash("ab" * 32, [("20260101T000000.md", "构图", 1000)])
        self.assertEqual(self.notes_idx.counts(), (1, 1))
        self.assertEqual(self.notes_idx.list_for_hash("ab" * 32)[0]["title"], "构图")

        self.qcache.put(
            "白袜",
            "qwen3:4b",
            {"semantic_zh": "白袜", "positive_tags": ["white_socks"]},
        )
        self.assertEqual(self.qcache.get("白袜", "qwen3:4b")["positive_tags"], ["white_socks"])
        self.assertIsNone(self.qcache.get("白袜", "qwen3:8b"))

        job = self.jobs.create("job-1", "index", {"paths": ["D:/art"]}, total=10)
        self.assertEqual(job["status"], "queued")
        updated = self.jobs.update("job-1", status="running", done=4, message="indexing")
        self.assertEqual(updated["done"], 4)
        self.assertEqual(self.jobs.mark_interrupted_running(), 1)
        self.assertEqual(self.jobs.get("job-1")["status"], "interrupted")

        c1 = self.collections.create("收藏一")
        self.collections.add_item(c1.id, "artist:1", item_name="Alice")
        self.assertEqual(self.collections.get(c1.id).item_ids, ("artist:1",))
        self.assertIsNotNone(self.collections.undo())
        self.assertEqual(self.collections.get(c1.id).item_ids, ())
        self.assertIsNotNone(self.collections.redo())
        self.assertEqual(self.collections.get(c1.id).item_ids, ("artist:1",))


if __name__ == "__main__":
    unittest.main()
