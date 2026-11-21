"""Comprehensive service-layer unit tests for ArtSeek v2."""

from __future__ import annotations

import hashlib
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from artseek.config import AppConfig
from artseek.core.query_plan import QueryPlan
from artseek.db.connection import Database
from artseek.db.repos.collections import CollectionRepo
from artseek.db.repos.images import ImageRepo
from artseek.db.repos.notes import NotesIndexRepo
from artseek.db.repos.provenance import ProvenanceRepo
from artseek.db.repos.query_cache import QueryPlanCacheRepo
from artseek.db.scope_resolver import ScopeResolver
from artseek.image_io import canonical_path, file_digest, perceptual_hash, point_id
from artseek.notes import NoteStore
from artseek.provenance.scan import parse_payload_path, record_unresolved
from artseek.search_cache import SearchSnapshotStore
from artseek.services import (
    ArtistService,
    CollectionService,
    GalleryService,
    ModelLifecycle,
    NoteService,
    ProvenanceAppService,
    SearchService,
    ServiceError,
    StorageService,
)
from artseek.vectors.index import VectorIndex


def _norm_vec(values: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    return (arr / max(float(np.linalg.norm(arr)), 1e-12)).astype(np.float32)


def _make_img(path: Path, *, size: tuple[int, int], seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = size
    xs = np.linspace(0, 255, w, dtype=np.int32)[None, :]
    ys = np.linspace(0, 255, h, dtype=np.int32)[:, None]
    r = ((xs * ((seed % 5) + 1) + ys * 2) % 256).astype(np.uint8)
    g = ((xs + ys * ((seed % 3) + 1) + seed * 37) % 256).astype(np.uint8)
    b = ((xs * 2 + ys + seed * 53) % 256).astype(np.uint8)
    Image.fromarray(np.stack([r, g, b], axis=-1), mode="RGB").save(path, format="PNG")


class _FakeEmbedder:
    dimension = 8

    def __init__(self) -> None:
        self.calls = 0

    @staticmethod
    def _vec(seed_text: str) -> np.ndarray:
        digest = hashlib.sha256(seed_text.encode("utf-8")).digest()
        return _norm_vec([b / 255.0 - 0.5 for b in digest[:8]])

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        self.calls += 1
        return np.stack([self._vec(f"text:{t}") for t in texts], axis=0)

    def encode_images(self, images: list[Any]) -> np.ndarray:
        self.calls += 1
        return np.stack([self._vec(f"img:{Path(p).name}") for p in images], axis=0)

    def close(self) -> None:
        return


class _SlowLLM:
    def __init__(self, delay: float = 0.05) -> None:
        self.delay = delay
        self.calls = 0
        self.done = threading.Event()

    def expand_query(self, query: str, *, keep_alive: int | None = None) -> QueryPlan:
        del keep_alive
        self.calls += 1
        time.sleep(self.delay)
        self.done.set()
        return QueryPlan(
            original=query,
            semantic_zh=query,
            semantic_en="wide sleeves, long hair",
            positive_tags=["wide_sleeves"],
            used_llm=True,
        )

    def unload(self) -> None:
        return


class V2ServicesTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.base = Path(self._tmp.name)
        self.config = AppConfig(data_dir=self.base / "data", cache_dir=self.base / "cache")
        self.config.llm.enabled = False
        self.config.tagger.enabled = False
        self.config.ensure_dirs()

        self.db = Database(self.config.db_path)
        self.images = ImageRepo(self.db)
        self.collections_repo = CollectionRepo(self.db)
        self.prov_repo = ProvenanceRepo(self.db)
        self.notes_index = NotesIndexRepo(self.db)
        self.query_cache = QueryPlanCacheRepo(self.db)
        self.resolver = ScopeResolver(self.db)
        self.vectors = VectorIndex(
            self.config.visual_vectors_path,
            self.config.descriptor_vectors_path,
            active_rows=self.images.active_row_ids(),
        )
        self.snapshots = SearchSnapshotStore(self.config.search_snapshot_dir)
        self.note_store = NoteStore(self.config.notes_dir)

        self.fake_embedder = _FakeEmbedder()
        self.fake_llm = _SlowLLM(delay=0.05)
        self.lifecycle = ModelLifecycle(
            self.config,
            embedder_factory=lambda: self.fake_embedder,
            llm=self.fake_llm,  # type: ignore[arg-type]
            vectors=self.vectors,
            auto_unload_seconds=0,
        )

        self.gallery = GalleryService(self.images, self.prov_repo, self.resolver)
        self.artists = ArtistService(
            self.images, self.prov_repo, self.resolver, self.collections_repo, self.snapshots
        )
        self.collections = CollectionService(self.collections_repo, self.images, self.prov_repo)
        self.notes = NoteService(self.note_store, self.notes_index, self.images)
        self.provenance = ProvenanceAppService(
            self.config,
            self.images,
            self.prov_repo,
            self.vectors,
            on_author_change=self.artists.invalidate_cache,
        )
        self.search = SearchService(
            self.config,
            self.images,
            self.prov_repo,
            self.vectors,
            self.resolver,
            self.lifecycle,
            self.query_cache,
            self.snapshots,
        )
        self.storage = StorageService(
            self.config,
            self.db,
            self.images,
            self.vectors,
            self.notes,
            self.snapshots,
        )

        self.root_a = self.base / "root_a"
        self.sub_a = self.root_a / "sub"
        self.root_b = self.base / "root_b"
        self.items: dict[str, str] = {}
        self._seed_library()

    def tearDown(self) -> None:
        self.lifecycle.close()
        self.vectors.release_cuda()
        self.db.close()
        self._tmp.cleanup()

    def _seed_library(self) -> None:
        specs = [
            (
                "pixiv_1",
                self.root_a,
                self.root_a / "12345678_p0.png",
                (128, 96),
                11,
                1_700_000_100_000_000_000,
                ["wide_sleeves", "sitting", "thighhighs"],
                [1.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.9, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ),
            (
                "pixiv_2_dup",
                self.root_a,
                self.sub_a / "87654321_p0.png",
                (64, 48),
                11,  # same pattern -> same dHash as pixiv_1
                1_700_000_200_000_000_000,
                ["wide_sleeves", "standing"],
                [0.95, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.85, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ),
            (
                "twitter_1",
                self.root_a,
                self.sub_a / "1800000000000000000_1.png",
                (96, 128),
                22,
                1_700_000_300_000_000_000,
                ["shirt", "smile"],
                [0.0, 0.0, 1.0, 0.2, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.9, 0.4, 0.0, 0.0, 0.0, 0.0],
            ),
            (
                "plain_1",
                self.root_b,
                self.root_b / "2_sketch.png",
                (140, 100),
                33,
                1_700_000_400_000_000_000,
                ["wide_sleeves"],
                [0.2, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.2, 0.0, 0.0, 0.0, 0.9, 0.1, 0.0, 0.0],
            ),
            (
                "plain_2",
                self.root_b,
                self.root_b / "10_sketch.png",
                (140, 100),
                44,
                1_700_000_500_000_000_000,
                [],
                [0.0, 0.0, 0.0, 0.0, 0.1, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.1, 1.0, 0.0, 0.0],
            ),
        ]
        records = []
        vrows = []
        drows = []
        for alias, root, path, size, seed, mtime_ns, tags, vvec, dvec in specs:
            _make_img(path, size=size, seed=seed)
            os.utime(path, ns=(mtime_ns, mtime_ns))
            pid = point_id(path)
            self.items[alias] = pid
            stat = path.stat()
            with Image.open(path) as img:
                w, h = img.size
                ph = perceptual_hash(img)
            records.append(
                ImageRepo.build_upsert_record(
                    path,
                    root=root,
                    point_id=pid,
                    width=w,
                    height=h,
                    mtime_ns=mtime_ns,
                    size_bytes=stat.st_size,
                    content_hash=file_digest(path),
                    phash=ph,
                    tags=tags,
                    tag_scores={t: 0.9 for t in tags},
                    descriptor_text=", ".join(tags),
                )
            )
            vrows.append(_norm_vec(vvec))
            drows.append(_norm_vec(dvec))

        row_ids = self.images.upsert_images_batch(records)
        self.vectors.set_rows(row_ids, np.stack(vrows), np.stack(drows))
        self.vectors.flush()

        self.alice = self.prov_repo.ensure_author("pixiv", "9001", name="Alice", handle="alice_art")
        self.bob = self.prov_repo.ensure_author("twitter", "8002", name="Bob", handle="bob_draw")
        for alias in ("pixiv_1", "pixiv_2_dup", "twitter_1", "plain_1", "plain_2"):
            pid = self.items[alias]
            payload = self.images.get_by_point_id(pid)
            assert payload is not None
            path = Path(payload["path"])
            parsed = parse_payload_path(path, deep=False)
            if parsed is not None:
                self.prov_repo.upsert_scan(pid, payload, parsed)
            else:
                record_unresolved(
                    self.prov_repo, pid, payload, path_key=payload["path_key"], path=path
                )
        self.prov_repo.set_fetch_result(
            self.items["pixiv_1"],
            {
                "platform": "pixiv",
                "source_id": "12345678",
                "page": 0,
                "url": "https://www.pixiv.net/artworks/12345678",
                "status": "ok",
                "title": "Hime Sleeves Study",
                "tags": ["thighhighs", "kimono"],
            },
            author_key=self.alice,
        )
        self.prov_repo.set_fetch_result(
            self.items["pixiv_2_dup"],
            {
                "platform": "pixiv",
                "source_id": "87654321",
                "page": 0,
                "url": "https://www.pixiv.net/artworks/87654321",
                "status": "ok",
                "title": "Standing Sketch",
                "tags": ["wide_sleeves"],
            },
            author_key=self.alice,
        )
        self.prov_repo.set_fetch_result(
            self.items["twitter_1"],
            {
                "platform": "twitter",
                "source_id": "1800000000000000000",
                "page": 1,
                "url": "https://x.com/bob_draw/status/1800000000000000000",
                "status": "ok",
                "title": "Morning Sketch",
                "tags": ["smile"],
            },
            author_key=self.bob,
        )

    def test_gallery_tree_scope_folder_random_and_tags(self) -> None:
        tree = self.gallery.library_tree()
        self.assertEqual(tree["image_count"], 5)
        self.assertEqual(tree["selected"], [])
        self.assertEqual(len(tree["roots"]), 2)

        sub_key = canonical_path(self.sub_a)
        root_b_key = canonical_path(self.root_b)
        updated = self.gallery.update_scope(sort="name", selected=[sub_key, root_b_key])
        self.assertEqual(updated["selected_image_count"], 4)

        # Direct scope vs recursive folder
        scoped = self.gallery.scope_images(None, sort="name", limit=10)
        self.assertEqual(scoped["total"], 4)
        self.assertNotIn(self.items["pixiv_1"], [r["id"] for r in scoped["results"]])

        with self.assertRaises(ServiceError) as ctx:
            self.gallery.scope_images([])
        self.assertEqual(ctx.exception.status_code, 400)

        folder_rec = self.gallery.folder_images(
            canonical_path(self.root_a), sort="count", limit=10
        )
        self.assertEqual(folder_rec["total"], 3)

        # Natural name sort ("2_sketch.png" before "10_sketch.png")
        b_names = self.gallery.folder_images(root_b_key, sort="name", limit=10)
        self.assertEqual(
            [r["id"] for r in b_names["results"]],
            [self.items["plain_1"], self.items["plain_2"]],
        )

        r1 = self.gallery.random_browse(folder_keys=[sub_key, root_b_key], seed=7, limit=10)
        r2 = self.gallery.random_browse(folder_keys=[sub_key, root_b_key], seed=7, limit=10)
        self.assertEqual(
            [r["id"] for r in r1["results"]],
            [r["id"] for r in r2["results"]],
        )

        # Tags
        src_tags = self.gallery.tag_suggestions(kind="source", q="", limit=20)
        self.assertTrue(any(t["tag"] == "thighhighs" for t in src_tags["tags"]))
        img_tags = self.gallery.tag_suggestions(kind="image", q="wide", limit=20)
        self.assertTrue(any(t["tag"] == "wide_sleeves" for t in img_tags["tags"]))
        tag_imgs = self.gallery.tag_images("thighhighs", limit=10)
        self.assertEqual([r["id"] for r in tag_imgs["results"]], [self.items["pixiv_1"]])

    def test_search_filter_only_hybrid_and_nonblocking_llm(self) -> None:
        all_folders = [
            canonical_path(self.root_a),
            canonical_path(self.sub_a),
            canonical_path(self.root_b),
        ]

        # 1. Filter-only never loads SigLIP
        unassigned = self.search.search(query="author:none", folder_keys=all_folders, limit=10)
        self.assertEqual(
            set(r["id"] for r in unassigned["results"]),
            {self.items["plain_1"], self.items["plain_2"]},
        )
        self.assertEqual(self.fake_embedder.calls, 0)

        # 2. Local metadata search
        local = self.search.local_search(query="过膝袜", folder_keys=all_folders, limit=10)
        self.assertTrue(local["metadata_search"])
        self.assertEqual([r["id"] for r in local["results"]], [self.items["pixiv_1"]])

        # Pinyin/romaji aliases resolve to the same tags.
        pinyin = self.search.local_search(query="guoxiwa", folder_keys=all_folders, limit=10)
        self.assertEqual([r["id"] for r in pinyin["results"]], [self.items["pixiv_1"]])
        lexicon = self.search.local_search(query="jixiu", folder_keys=all_folders, limit=10)
        self.assertEqual(
            {r["id"] for r in lexicon["results"]},
            {self.items["pixiv_1"], self.items["pixiv_2_dup"], self.items["plain_1"]},
        )
        # Artist names no longer match images; artists are matched by the mixed search.
        by_artist = self.search.local_search(query="alice", folder_keys=all_folders, limit=10)
        self.assertEqual(by_artist["results"], [])

        # 3. Hybrid search with negative tag + strict_tags + dedupe + snapshot
        hybrid = self.search.search(
            query="姬袖 -站立",
            expand=False,
            strict_tags=True,
            folder_keys=all_folders,
            limit=10,
        )
        ids = [r["id"] for r in hybrid["results"]]
        self.assertIn(self.items["pixiv_1"], ids)
        self.assertNotIn(self.items["pixiv_2_dup"], ids)
        snap = self.search.read_snapshot(hybrid["snapshot_id"])
        self.assertEqual([r["id"] for r in snap["results"]], ids)

        # 4. Non-blocking LLM query expansion
        self.config.llm.enabled = True
        first = self.search.search(
            query="长发少女",
            expand=True,
            wait_for_llm=False,
            folder_keys=all_folders,
            limit=10,
        )
        self.assertEqual(first["plan_revision"], 1)
        self.assertTrue(first["llm_pending"])
        self.assertEqual(first["query_plan"]["used_llm"], False)
        self.assertTrue(self.fake_llm.done.wait(timeout=2.0))
        # Give background thread a brief moment to finish writing query_plan_cache
        for _ in range(40):
            status = self.search.get_query_plan_status("长发少女")
            if not status["pending"] and status["plan_revision"] == 2:
                break
            time.sleep(0.01)
        self.assertEqual(status["plan_revision"], 2)
        self.assertFalse(status["pending"])
        self.assertTrue(status["query_plan"]["used_llm"])

        second = self.search.search(
            query="长发少女",
            expand=True,
            folder_keys=all_folders,
            limit=10,
        )
        self.assertEqual(second["plan_revision"], 2)
        self.assertFalse(second["llm_pending"])
        self.assertTrue(second["query_plan"]["used_llm"])

    def test_collections_artists_notes_provenance_and_storage_migration(self) -> None:
        # Collections: mixed artist+image, saved vs temp sort, seed_sort, anchor conflict, undo
        col = self.collections.create_collection("精选")["collection"]
        cid = col["id"]
        self.collections.add_item(cid, self.items["pixiv_1"])
        self.collections.add_item(cid, self.items["twitter_1"])
        artist_item = f"artist:{self.alice}"
        self.collections.add_item(cid, artist_item)
        self.collections.update_collection(cid, sort="name_asc")
        temp_rand = self.collections.collection_images(cid, sort="random", seed=99)
        self.assertEqual(temp_rand["sort"], "random")
        self.assertEqual(temp_rand["saved_sort"], "name_asc")
        self.assertTrue(any(item.get("kind") == "artist" for item in temp_rand["results"]))

        self.collections.update_collection(cid, sort="manual", seed_sort="mtime_desc")
        self.collections.reorder_item(cid, self.items["pixiv_1"], after_id=artist_item)
        with self.assertRaises(ServiceError) as err409:
            self.collections.reorder_item(cid, self.items["pixiv_1"], before_id="nonexistent")
        self.assertEqual(err409.exception.status_code, 409)
        self.collections.undo()
        self.collections.redo()

        # Artists: gallery + "none" pinned first + merge + legacy lookup
        gal = self.artists.artists_gallery(include_unassigned=True, sort="latest_desc")
        self.assertEqual(gal["authors"][0]["key"], "none")
        # Filtered lookups go through the lazily built artist search index;
        # ProvenanceRepo skips ProvenanceStore.__init__, so this guards against
        # the index cache being missing there.
        self.assertGreaterEqual(gal["total"], 2)
        searched = self.artists.artists_gallery(
            q="alice", include_unassigned=True, sort="latest_desc"
        )
        self.assertEqual([item["name"] for item in searched["authors"]], ["Alice"])
        self.assertEqual(
            [
                item["name"]
                for item in self.artists.artists_gallery(
                    q="ＡＬＩＣＥ", include_unassigned=True, sort="latest_desc"
                )["authors"]
            ],
            ["Alice"],
        )
        # Simplified/variant/romaji input must reach a CJK artist name too.
        self.artists.rename_author(self.alice, "花宮なつか")
        for query in ("花宫なつか", "花宮ナツカ", "natsuka"):
            names = [
                item["name"]
                for item in self.artists.artists_gallery(
                    q=query, include_unassigned=True, sort="latest_desc"
                )["authors"]
            ]
            self.assertIn("花宮なつか", names, query)
        first_page = self.artists.artists_gallery(include_unassigned=True, limit=1, offset=0)
        self.assertEqual(first_page["has_more"], first_page["total"] > 1)
        last_page = self.artists.artists_gallery(
            include_unassigned=True,
            limit=1,
            offset=max(0, first_page["total"] - 1),
        )
        self.assertFalse(last_page["has_more"])
        self.artists.add_identity(self.alice, kind="link", value="https://example.com/alice/")
        self.artists.rename_author(self.alice, "Alice Renamed")
        self.artists.set_cover(self.alice, self.items["pixiv_1"])
        self.artists.merge_author(self.alice, self.bob)
        legacy = self.artists.get_author(self.bob)
        self.assertEqual(legacy["author"]["key"], self.alice)

        # Notes: Markdown vault + dHash sharing + notes_index SQLite sync
        p1 = self.items["pixiv_1"]
        p2 = self.items["pixiv_2_dup"]
        note_doc = self.notes.create_note(p1, title="结构", content="# 结构\n内容")
        self.assertEqual(self.notes_index.counts(), (1, 1))
        shared = self.notes.list_notes(p2)
        self.assertEqual(len(shared["notes"]), 1)
        self.assertEqual(shared["notes"][0]["name"], note_doc["name"])

        # Provenance: auto-tag plain_2 using VectorIndex neighbor_tag_votes + scan
        auto_summary = self.provenance.run_auto_tag(limit=10)
        self.assertGreaterEqual(auto_summary["ok"], 1)
        prov_p2 = self.provenance.get_image_provenance(p2)
        self.assertEqual(prov_p2["record"]["platform"], "pixiv")

        # Storage migration of `index` dir reopens Database & VectorIndex seamlessly
        new_index_dir = self.base / "moved_index"
        migrated = self.storage.migrate_path("index", str(new_index_dir))
        self.assertEqual(Path(migrated["destination"]).resolve(), new_index_dir.resolve())
        self.assertEqual(self.images.count(), 5)
        self.assertEqual(self.vectors.active_count, 5)
        self.assertTrue((new_index_dir / "artseek.sqlite3").is_file())
        self.assertTrue((new_index_dir / "visual.f32.npy").is_file())

    def test_refresh_artist_avatars_replaces_pximg_and_skips_uploaded(self) -> None:
        class _FakeTwitter:
            def fetch_user(self, handle: str) -> dict[str, Any]:
                return {
                    "handle": handle,
                    "avatar_url": f"https://pbs.twimg.com/profile_images/{handle}.jpg",
                }

            def close(self) -> None:
                pass

        # Artist with pximg.net avatar and a linked Twitter handle should be updated.
        px_artist = self.prov_repo.ensure_author(
            "pixiv",
            "705370",
            name="しらたま",
            avatar_url="https://i.pximg.net/user-profile/img/old.jpg",
        )
        tw_member = self.prov_repo.ensure_author(
            "twitter", "shiratamacaron", handle="shiratamacaron"
        )
        self.prov_repo.link_authors(tw_member, px_artist, evidence="pixiv_social")

        # Artist with custom uploaded avatar must NOT be overwritten.
        custom_artist = self.prov_repo.ensure_author(
            "twitter",
            "custom_user",
            handle="custom_user",
            avatar_url="/api/authors/99/avatar?v=abc",
        )

        targets = self.provenance.select_artist_avatar_targets(skip_completed=True)
        keys = {t["artist_key"] for t in targets}
        self.assertIn(px_artist, keys)
        self.assertNotIn(custom_artist, keys)

        summary = self.provenance.run_artist_avatars(
            targets, twitter_client=_FakeTwitter()  # type: ignore[arg-type]
        )
        self.assertGreaterEqual(summary["ok"], 1)
        updated = self.prov_repo.author_summary(px_artist)
        assert updated is not None
        self.assertEqual(
            updated["avatar_url"],
            "https://pbs.twimg.com/profile_images/shiratamacaron.jpg",
        )
        custom = self.prov_repo.author_summary(custom_artist)
        assert custom is not None
        self.assertEqual(custom["avatar_url"], "/api/authors/99/avatar?v=abc")


if __name__ == "__main__":
    unittest.main()
