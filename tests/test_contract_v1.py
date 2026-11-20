"""HTTP contract baseline suite for ArtSeek v1 -> v2 refactoring (M0)."""

from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from artseek.config import AppConfig
from artseek.image_io import canonical_path, file_digest, perceptual_hash, point_id
from artseek.provenance.scan import parse_payload_path, record_unresolved
from artseek.query import static_expand
from artseek.search import SearchEngine
from artseek.store import Neighbor
from artseek.web.app import create_app


class _FakeEmbedder:
    dimension = 8

    @staticmethod
    def _vec(seed_text: str) -> np.ndarray:
        digest = hashlib.sha256(seed_text.encode("utf-8")).digest()
        arr = np.array([b / 255.0 - 0.5 for b in digest[:8]], dtype=np.float32)
        norm = float(np.linalg.norm(arr))
        if norm <= 1e-8:
            arr[0] = 1.0
            norm = 1.0
        return (arr / norm).astype(np.float32)

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vec(f"text:{t}") for t in texts], axis=0)

    def encode_images(self, images: list[Any]) -> np.ndarray:
        vecs = []
        for idx, item in enumerate(images):
            if isinstance(item, (str, Path)):
                vecs.append(self._vec(f"img:{Path(item).name}"))
            else:
                vecs.append(self._vec(f"img_mem:{idx}:{getattr(item, 'size', (0, 0))}"))
        return np.stack(vecs, axis=0)

    def close(self) -> None:
        return


class _FakeLLM:
    def expand_query(self, query: str, *, keep_alive: int | None = None):
        del keep_alive
        return static_expand(query)

    def unload(self) -> None:
        return

    def preload(self, *, keep_alive: int | None = None) -> None:
        del keep_alive
        return


class _MemoryVectorStore:
    """In-memory VectorStore drop-in for HTTP contract tests without Qdrant/ML."""

    def __init__(self) -> None:
        self.payloads: dict[str, dict[str, Any]] = {}
        self.vectors: dict[str, dict[str, np.ndarray]] = {}

    def known_dimension(self) -> int | None:
        return 8

    def ensure_collection(self, dimension: int) -> None:
        del dimension

    def close(self) -> None:
        return

    def count(self) -> int:
        return len(self.payloads)

    def get(self, pid: str) -> dict[str, Any] | None:
        payload = self.payloads.get(pid)
        return dict(payload) if payload is not None else None

    def retrieve_payloads(self, ids: list[str]) -> dict[str, dict[str, Any]]:
        return {pid: dict(self.payloads[pid]) for pid in ids if pid in self.payloads}

    def iter_payloads(self):
        for pid, payload in self.payloads.items():
            yield pid, dict(payload)

    def iter_folder_payloads(self):
        for pid, payload in self.payloads.items():
            yield pid, {
                "path": payload["path"],
                "path_key": payload["path_key"],
                "root": payload["root"],
                "root_key": payload["root_key"],
                "folder_key": payload["folder_key"],
                "mtime_ns": payload["mtime_ns"],
            }

    def invalidate_folder_catalog(self) -> None:
        return

    def visual_vectors(self, ids: list[str]) -> dict[str, list[float]]:
        return {
            pid: self.vectors[pid]["visual"].tolist()
            for pid in ids
            if pid in self.vectors
        }

    def point_ids_for_image_tags(self, tags: list[str]) -> list[str]:
        wanted = {str(t).strip().lower().replace(" ", "_") for t in tags if str(t).strip()}
        if not wanted:
            return []
        hits = []
        for pid, payload in self.payloads.items():
            row_tags = {str(t).strip().lower() for t in payload.get("tags") or []}
            if wanted.issubset(row_tags):
                hits.append((int(payload.get("mtime_ns") or 0), pid))
        hits.sort(key=lambda item: (-item[0], item[1]))
        return [pid for _, pid in hits]

    def query(
        self,
        vector: list[float],
        using: str,
        limit: int,
        point_ids: list[str] | None = None,
        universe: int | None = None,
    ) -> list[Neighbor]:
        del universe
        q = np.asarray(vector, dtype=np.float32)
        q_norm = float(np.linalg.norm(q))
        if q_norm > 0:
            q = q / q_norm
        candidates = (
            list(self.payloads)
            if point_ids is None
            else [p for p in point_ids if p in self.payloads]
        )
        scored: list[tuple[float, str]] = []
        for pid in candidates:
            vec = self.vectors[pid][using]
            scored.append((float(q @ vec), pid))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [
            Neighbor(pid, score, dict(self.payloads[pid]))
            for score, pid in scored[: max(1, limit)]
        ]

    def delete_ids(self, ids: list[str]) -> None:
        for pid in ids:
            self.payloads.pop(pid, None)
            self.vectors.pop(pid, None)


def _norm_vec(values: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    return (arr / max(float(np.linalg.norm(arr)), 1e-12)).astype(np.float32)


def _make_patterned_image(
    path: Path, *, size: tuple[int, int] = (128, 96), seed: int = 1
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = size
    xs = np.linspace(0, 255, w, dtype=np.int32)[None, :]
    ys = np.linspace(0, 255, h, dtype=np.int32)[:, None]
    r = ((xs * ((seed % 5) + 1) + ys * 2) % 256).astype(np.uint8)
    g = ((xs + ys * ((seed % 3) + 1) + seed * 37) % 256).astype(np.uint8)
    b = ((xs * 2 + ys + seed * 53) % 256).astype(np.uint8)
    rgb = np.stack([r, g, b], axis=-1)
    Image.fromarray(rgb, mode="RGB").save(path, format="PNG")


class V1HttpContractBaselineTests(unittest.TestCase):
    def setUp(self) -> None:
        from fastapi.testclient import TestClient

        self._tmp = tempfile.TemporaryDirectory()
        self.base = Path(self._tmp.name)
        self.root_a = self.base / "lib_a"
        self.root_b = self.base / "lib_b"
        self.sub_a = self.root_a / "nested"

        self.config = AppConfig(data_dir=self.base / "data", cache_dir=self.base / "cache")
        self.config.llm.enabled = False
        self.config.tagger.enabled = False
        self.config.ensure_dirs()

        self.store = _MemoryVectorStore()
        self.items: dict[str, str] = {}

        specs = [
            (
                "pixiv_1",
                self.root_a,
                self.root_a / "12345678_p0.png",
                (160, 120),
                11,
                1_700_000_100_000_000_000,
                ["wide_sleeves", "sitting", "thighhighs"],
                [1.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.9, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ),
            (
                "pixiv_2_small",
                self.root_a,
                self.sub_a / "87654321_p0.png",
                (80, 60),
                11,  # same pattern resized -> matches dHash with pixiv_1
                1_700_000_200_000_000_000,
                ["wide_sleeves", "standing"],
                [0.5, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.4, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ),
            (
                "twitter_1",
                self.root_a,
                self.sub_a / "1800000000000000000_1.png",
                (120, 160),
                22,
                1_700_000_300_000_000_000,
                ["shirt", "smile"],
                [0.0, 0.0, 1.0, 0.1, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.9, 0.3, 0.0, 0.0, 0.0, 0.0],
            ),
            (
                "xhs_1",
                self.root_b,
                self.root_b / "软妹穿搭_1_alice_来自小红书.png",
                (140, 140),
                33,
                1_700_000_400_000_000_000,
                ["dress", "lace"],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.2, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.8, 0.5, 0.0, 0.0],
            ),
            (
                "plain_1",
                self.root_b,
                self.root_b / "sketch_01.png",
                (200, 100),
                44,
                1_700_000_500_000_000_000,
                ["wide_sleeves", "looking_back"],
                [0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.3, 0.2, 0.0, 0.0, 0.0, 0.0, 0.9, 0.1],
            ),
        ]

        import os

        for alias, root, path, size, seed, mtime_ns, tags, vvec, dvec in specs:
            _make_patterned_image(path, size=size, seed=seed)
            os.utime(path, ns=(mtime_ns, mtime_ns))
            pid = point_id(path)
            self.items[alias] = pid
            stat = path.stat()
            with Image.open(path) as img:
                phash = perceptual_hash(img)
                w, h = img.size
            self.store.payloads[pid] = {
                "path": str(path.resolve()),
                "path_key": canonical_path(path),
                "name": path.name,
                "extension": path.suffix.lower(),
                "mtime_ns": stat.st_mtime_ns,
                "size_bytes": stat.st_size,
                "width": w,
                "height": h,
                "root": str(root.resolve()),
                "root_key": canonical_path(root),
                "folder_key": canonical_path(path.parent),
                "content_hash": file_digest(path),
                "phash": phash,
                "tags": tags,
                "tag_scores": {t: 0.9 for t in tags},
                "caption": None,
                "descriptor_text": ", ".join(tags),
            }
            self.store.vectors[pid] = {
                "visual": _norm_vec(vvec),
                "descriptor": _norm_vec(dvec),
            }

        self.engine = SearchEngine(
            self.config,
            store=self.store,
            embedder_factory=_FakeEmbedder,
            llm=_FakeLLM(),
        )
        self.app = create_app(self.config, engine=self.engine)
        self.client = TestClient(self.app)

        # Seed provenance for pixiv_1, twitter_1, xhs_1; leave plain_1 unassigned ("none").
        prov = self.app.state.provenance
        self.author_alice = prov.ensure_author("pixiv", "9001", name="Alice", handle="alice_art")
        self.author_bob = prov.ensure_author("twitter", "8002", name="Bob", handle="bob_draw")
        for alias in ("pixiv_1", "pixiv_2_small", "twitter_1", "xhs_1", "plain_1"):
            pid = self.items[alias]
            payload = self.store.payloads[pid]
            path = Path(payload["path"])
            parsed = parse_payload_path(path, deep=False)
            if parsed is not None:
                prov.upsert_scan(pid, payload, parsed)
            else:
                record_unresolved(prov, pid, payload, path_key=payload["path_key"], path=path)
        prov.set_fetch_result(
            self.items["pixiv_2_small"],
            {
                "platform": "pixiv",
                "source_id": "87654321",
                "page": 0,
                "url": "https://www.pixiv.net/artworks/87654321",
                "status": "ok",
                "title": "Hime Sleeves Study Resized",
                "tags": ["wide_sleeves"],
            },
            author_key=self.author_alice,
        )
        prov.set_fetch_result(
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
            author_key=self.author_alice,
        )
        prov.set_fetch_result(
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
            author_key=self.author_bob,
        )

    def tearDown(self) -> None:
        self.client.close()
        self._tmp.cleanup()

    def test_spa_routes_and_media_endpoints(self) -> None:
        pid = self.items["pixiv_1"]
        for route in (
            "/",
            "/search",
            "/folder",
            "/random",
            "/scope",
            "/c/col1",
            f"/author/{self.author_alice}",
            "/tag/thighhighs",
            f"/i/{pid}",
            f"/i/{pid}/notes",
            f"/i/{pid}/notes/20260101T000000.md",
        ):
            res = self.client.get(route)
            self.assertEqual(res.status_code, 200, route)
            self.assertIn("text/html", res.headers["content-type"])

        thumb = self.client.get(f"/thumb/{pid}")
        self.assertEqual(thumb.status_code, 200)
        self.assertEqual(thumb.headers["content-type"], "image/webp")

        orig = self.client.get(f"/image/{pid}")
        self.assertEqual(orig.status_code, 200)

        search_img = self.client.get(f"/search-image/{pid}")
        self.assertEqual(search_img.status_code, 200)
        self.assertEqual(search_img.headers["content-type"], "image/jpeg")

        item = self.client.get(f"/api/item/{pid}")
        self.assertEqual(item.status_code, 200)
        self.assertEqual(item.json()["name"], "12345678_p0.png")

    def test_library_tree_scope_folder_and_random_contracts(self) -> None:
        tree = self.client.get("/api/library-tree")
        self.assertEqual(tree.status_code, 200)
        tbody = tree.json()
        self.assertEqual(tbody["image_count"], 5)
        self.assertEqual(tbody["selected"], [])
        self.assertEqual(len(tbody["roots"]), 2)

        sub_key = canonical_path(self.sub_a)
        root_b_key = canonical_path(self.root_b)
        updated = self.client.put(
            "/api/library-tree",
            json={"sort": "mtime", "selected": [sub_key, root_b_key]},
        )
        self.assertEqual(updated.status_code, 200)
        self.assertEqual(updated.json()["selected_image_count"], 4)

        folder_res = self.client.get(
            "/api/folder-images",
            params={
                "folder_key": canonical_path(self.root_a),
                "sort": "random",
                "seed": 42,
                "limit": 10,
            },
        )
        self.assertEqual(folder_res.status_code, 200)
        fbody = folder_res.json()
        self.assertEqual(fbody["total"], 3)
        self.assertEqual(fbody["seed"], 42)
        folder_res2 = self.client.get(
            "/api/folder-images",
            params={
                "folder_key": canonical_path(self.root_a),
                "sort": "random",
                "seed": 42,
                "limit": 10,
            },
        )
        self.assertEqual(
            [r["id"] for r in fbody["results"]],
            [r["id"] for r in folder_res2.json()["results"]],
        )

        scope_post = self.client.post(
            "/api/scope-images",
            json={"folder_keys": [sub_key], "sort": "name_desc", "limit": 10, "offset": 0},
        )
        self.assertEqual(scope_post.status_code, 200)
        self.assertEqual(scope_post.json()["total"], 2)

        scope_get = self.client.get(
            "/api/scope-images",
            params={"folder_keys": json.dumps([sub_key]), "author_key": self.author_bob},
        )
        self.assertEqual(scope_get.status_code, 200)
        self.assertEqual(
            [r["id"] for r in scope_get.json()["results"]], [self.items["twitter_1"]]
        )

        rand1 = self.client.post(
            "/api/random-browse", json={"folder_keys": [sub_key], "seed": 7, "limit": 10}
        )
        rand2 = self.client.post(
            "/api/random-browse", json={"folder_keys": [sub_key], "seed": 7, "limit": 10}
        )
        self.assertEqual(rand1.status_code, 200)
        self.assertEqual(
            [r["id"] for r in rand1.json()["results"]],
            [r["id"] for r in rand2.json()["results"]],
        )

    def test_search_snapshot_local_search_and_upload(self) -> None:
        all_folders = [
            canonical_path(self.root_a),
            canonical_path(self.sub_a),
            canonical_path(self.root_b),
        ]
        res = self.client.post(
            "/api/search",
            json={
                "query": "姬袖 -站立",
                "expand": False,
                "strict_tags": True,
                "dedupe": False,
                "folder_keys": all_folders,
                "limit": 10,
            },
        )
        self.assertEqual(res.status_code, 200)
        body = res.json()
        ids = [r["id"] for r in body["results"]]
        self.assertNotIn(self.items["pixiv_2_small"], ids)
        self.assertIn(self.items["pixiv_1"], ids)
        snap_id = body["snapshot_id"]
        snap = self.client.get(f"/api/search-snapshots/{snap_id}")
        self.assertEqual(snap.status_code, 200)
        self.assertEqual([r["id"] for r in snap.json()["results"]], ids)

        # Filter-only search with author:none and tag expression
        unassigned = self.client.post(
            "/api/search",
            json={"query": "author:none", "expand": False, "folder_keys": all_folders, "limit": 10},
        )
        self.assertEqual(unassigned.status_code, 200)
        self.assertEqual([r["id"] for r in unassigned.json()["results"]], [self.items["plain_1"]])

        local = self.client.post(
            "/api/local-search",
            json={"query": "thighhighs", "folder_keys": all_folders, "limit": 10},
        )
        self.assertEqual(local.status_code, 200)
        self.assertTrue(local.json()["metadata_search"])
        self.assertEqual([r["id"] for r in local.json()["results"]], [self.items["pixiv_1"]])

        buf = io.BytesIO()
        Image.new("RGB", (32, 32), (10, 20, 30)).save(buf, format="PNG")
        upload = self.client.post(
            "/api/search-upload",
            files=[("files", ("ref.png", buf.getvalue(), "image/png"))],
            data={
                "query": "姬袖",
                "expand": "false",
                "folder_keys": json.dumps(all_folders),
                "limit": "5",
            },
        )
        self.assertEqual(upload.status_code, 200)
        self.assertTrue(upload.json()["results"])

    def test_collections_mixed_sort_and_undo_redo(self) -> None:
        c1 = self.client.post("/api/collections", json={"name": ""}).json()["collection"]
        c2 = self.client.post("/api/collections", json={"name": "精选"}).json()["collection"]

        reordered = self.client.put(
            "/api/collections/order",
            json={"collection_ids": [c2["id"], c1["id"]]},
        )
        self.assertEqual(reordered.status_code, 200)
        stale = self.client.put("/api/collections/order", json={"collection_ids": [c1["id"]]})
        self.assertEqual(stale.status_code, 409)

        p1 = self.items["pixiv_1"]
        p2 = self.items["twitter_1"]
        artist_item = f"artist:{self.author_alice}"
        for item_id in (p1, p2, artist_item):
            add = self.client.post(f"/api/collections/{c2['id']}/items", json={"image_id": item_id})
            self.assertEqual(add.status_code, 200)

        # Saved sort vs temporary sort
        self.client.patch(f"/api/collections/{c2['id']}", json={"sort": "name_asc"})
        temp_rand = self.client.get(
            f"/api/collections/{c2['id']}/images",
            params={"sort": "random", "seed": 99},
        )
        self.assertEqual(temp_rand.status_code, 200)
        tbody = temp_rand.json()
        self.assertEqual(tbody["sort"], "random")
        self.assertEqual(tbody["saved_sort"], "name_asc")
        self.assertTrue(any(item.get("kind") == "artist" for item in tbody["results"]))

        # Anchor reorder + undo/redo
        moved = self.client.put(
            f"/api/collections/{c2['id']}/items/order",
            json={"image_id": p1, "after_id": artist_item},
        )
        self.assertEqual(moved.status_code, 200)
        bad_anchor = self.client.put(
            f"/api/collections/{c2['id']}/items/order",
            json={"image_id": p1, "before_id": "missing"},
        )
        self.assertEqual(bad_anchor.status_code, 409)

        undone = self.client.post("/api/collections/undo")
        self.assertEqual(undone.status_code, 200)
        redone = self.client.post("/api/collections/redo")
        self.assertEqual(redone.status_code, 200)

    def test_tags_provenance_and_artists_lifecycle(self) -> None:
        tags_src = self.client.post("/api/tags", json={"kind": "source", "q": "", "limit": 20})
        self.assertEqual(tags_src.status_code, 200)
        self.assertTrue(any(t["tag"] == "thighhighs" for t in tags_src.json()["tags"]))

        tags_img = self.client.get("/api/tags", params={"kind": "image", "q": "wide"})
        self.assertEqual(tags_img.status_code, 200)
        self.assertTrue(any(t["tag"] == "wide_sleeves" for t in tags_img.json()["tags"]))

        dict_res = self.client.get("/api/tag-dictionary", params={"q": "过膝袜"})
        self.assertEqual(dict_res.status_code, 200)
        self.assertTrue(any(t["tag"] == "thighhighs" for t in dict_res.json()["tags"]))

        tag_imgs = self.client.get("/api/tags/thighhighs/images")
        self.assertEqual(tag_imgs.status_code, 200)
        self.assertEqual([r["id"] for r in tag_imgs.json()["results"]], [self.items["pixiv_1"]])

        # Provenance & artists
        stats = self.client.get("/api/provenance/stats")
        self.assertEqual(stats.status_code, 200)
        self.assertGreaterEqual(stats.json()["total"], 4)

        prov_get = self.client.get(f"/api/images/{self.items['plain_1']}/provenance")
        self.assertEqual(prov_get.status_code, 200)

        gallery = self.client.post(
            "/api/artists/gallery",
            json={"include_unassigned": True, "sort": "latest_desc", "limit": 20},
        )
        self.assertEqual(gallery.status_code, 200)
        authors = gallery.json()["authors"]
        self.assertEqual(authors[0]["key"], "none")

        # Add identity, rename, set cover, merge Bob into Alice
        self.assertEqual(
            self.client.post(
                f"/api/authors/{self.author_alice}/identities",
                json={"kind": "link", "value": "https://example.com/alice/"},
            ).status_code,
            200,
        )
        self.assertEqual(
            self.client.post(
                f"/api/authors/{self.author_alice}/name",
                json={"name": "Alice Renamed"},
            ).status_code,
            200,
        )
        self.assertEqual(
            self.client.post(
                f"/api/authors/{self.author_alice}/cover",
                json={"point_id": self.items["pixiv_1"]},
            ).status_code,
            200,
        )
        merged = self.client.post(
            f"/api/authors/{self.author_alice}/merge",
            json={"source": self.author_bob},
        )
        self.assertEqual(merged.status_code, 200)
        legacy_lookup = self.client.get(f"/api/authors/{self.author_bob}")
        self.assertEqual(legacy_lookup.status_code, 200)
        self.assertEqual(legacy_lookup.json()["author"]["key"], self.author_alice)

        artist_imgs = self.client.post(
            f"/api/artists/{self.author_alice}/images",
            json={"sort": "mtime_desc", "limit": 10},
        )
        self.assertEqual(artist_imgs.status_code, 200)
        self.assertGreaterEqual(artist_imgs.json()["total"], 2)

    def test_notes_dhash_sharing_and_system_routes(self) -> None:
        p1 = self.items["pixiv_1"]
        p2 = self.items["pixiv_2_small"]

        n1 = self.client.post(
            f"/api/images/{p1}/notes", json={"title": "结构", "content": "# 结构\n"}
        ).json()
        n2 = self.client.post(
            f"/api/images/{p1}/notes", json={"title": "配色", "content": "# 配色\n"}
        ).json()

        # p2 is resized copy of p1 -> shares same notes folder via dHash
        p2_notes = self.client.get(f"/api/images/{p2}/notes").json()
        self.assertEqual(len(p2_notes["notes"]), 2)

        ordered = self.client.put(
            f"/api/images/{p1}/note-order",
            json={"names": [n2["name"], n1["name"]]},
        )
        self.assertEqual(ordered.status_code, 200)
        self.assertEqual([n["name"] for n in ordered.json()["notes"]], [n2["name"], n1["name"]])

        asset = self.client.post(
            f"/api/images/{p1}/assets",
            files={"file": ("crop.png", b"pngdata", "image/png")},
        )
        self.assertEqual(asset.status_code, 200)
        fetched = self.client.get(asset.json()["url"])
        self.assertEqual(fetched.status_code, 200)
        self.assertEqual(fetched.content, b"pngdata")

        self.assertEqual(
            self.client.delete(f"/api/images/{p1}/notes/{n1['name']}").status_code, 200
        )
        self.assertEqual(self.client.delete(f"/api/images/{p1}/notes").status_code, 200)

        # Model, web-settings, storage-paths
        self.assertEqual(self.client.get("/api/model").status_code, 200)
        self.assertEqual(
            self.client.post("/api/model", json={"auto_unload_minutes": 5}).status_code, 200
        )
        self.assertEqual(self.client.get("/api/web-settings").status_code, 200)
        self.assertEqual(
            self.client.put(
                "/api/web-settings",
                json={"use_llm": False, "strict_tags": True, "dedupe": False, "limit": 24},
            ).status_code,
            200,
        )
        self.assertEqual(self.client.get("/api/storage-paths").status_code, 200)


if __name__ == "__main__":
    unittest.main()
