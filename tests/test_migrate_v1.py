"""Tests for read-only ArtSeek v1 -> v2 migration (`migrate_v1`)."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from qdrant_client import QdrantClient, models

from artseek.collections import CollectionStore
from artseek.config import AppConfig
from artseek.db.connection import Database
from artseek.db.migrate_v1 import migrate_v1
from artseek.db.repos.collections import CollectionRepo
from artseek.db.repos.provenance import ProvenanceRepo
from artseek.image_io import canonical_path, file_digest, point_id
from artseek.incremental_state import (
    DirectoryState,
    IncrementalIndexState,
    IndexedImageState,
    RootIndexState,
)
from artseek.library import FolderScope, save_scope
from artseek.notes import NoteStore
from artseek.provenance.parser import ParsedSource, SourceRef
from artseek.provenance.store import ProvenanceStore
from artseek.store import COLLECTION, VectorStore
from artseek.vectors.index import VectorIndex


def _rand_unit_vec(rng: np.random.Generator, dim: int = 16) -> np.ndarray:
    v = rng.standard_normal(dim).astype(np.float32)
    return (v / max(float(np.linalg.norm(v)), 1e-12)).astype(np.float32)


class MigrateV1Tests(unittest.TestCase):
    def test_end_to_end_readonly_migration_parity_and_idempotency(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            cfg = AppConfig(data_dir=base / "data", cache_dir=base / "cache")
            cfg.tagger.enabled = True
            cfg.ensure_dirs()

            img_root = base / "pictures"
            sub = img_root / "sub"
            sub.mkdir(parents=True)

            rng = np.random.default_rng(1234)
            dim = 16
            points: list[models.PointStruct] = []
            state_images: dict[str, IndexedImageState] = {}
            point_ids: list[str] = []
            hashes: list[str] = []

            vstore = VectorStore(cfg)
            vstore.ensure_collection(dim)

            for i in range(25):
                target_dir = sub if i % 2 else img_root
                img_path = target_dir / f"12345{i:03d}_p0.png"
                Image.new("RGB", (64, 48), (i * 9 % 255, 40, 80)).save(img_path)
                pid = point_id(img_path)
                point_ids.append(pid)
                digest = file_digest(img_path)
                hashes.append(digest)
                stat = img_path.stat()
                vis = _rand_unit_vec(rng, dim)
                desc = _rand_unit_vec(rng, dim)
                # Exercise legacy tuple-list tag_scores & missing folder_key on row 0
                payload: dict[str, object] = {
                    "path": str(img_path.resolve()),
                    "path_key": canonical_path(img_path),
                    "name": img_path.name,
                    "extension": ".png",
                    "mtime_ns": stat.st_mtime_ns,
                    "size_bytes": stat.st_size,
                    "width": 64,
                    "height": 48,
                    "root": str(img_root.resolve()),
                    "root_key": canonical_path(img_root),
                    "tags": ["1girl", "thighhighs" if i % 3 == 0 else "shirt"],
                    "tag_scores": [("1girl", 0.99)] if i == 0 else {"1girl": 0.95},
                    "caption": None,
                    "descriptor_text": "1girl",
                }
                if i > 0:
                    payload["folder_key"] = canonical_path(target_dir)
                points.append(
                    models.PointStruct(
                        id=pid,
                        vector={"visual": vis.tolist(), "descriptor": desc.tolist()},
                        payload=payload,
                    )
                )
                state_images[canonical_path(img_path)] = IndexedImageState(
                    path_key=canonical_path(img_path),
                    root_key=canonical_path(img_root),
                    point_id=pid,
                    mtime_ns=stat.st_mtime_ns,
                    size_bytes=stat.st_size,
                    folder_key=canonical_path(target_dir),
                    name=img_path.name,
                )

            vstore.upsert(points)

            # Query 20 items from live Qdrant BEFORE closing to compare top-20 overlap
            qdrant_top20: list[tuple[np.ndarray, set[str]]] = []
            for p in points[:20]:
                qvec = np.asarray(p.vector["visual"], dtype=np.float32)  # type: ignore[index]
                res = vstore.client.query_points(
                    COLLECTION, query=qvec.tolist(), using="visual", limit=20
                )
                qdrant_top20.append((qvec, {str(pt.id) for pt in res.points}))
            vstore.close()

            # Incremental state + folder_scope
            state = IncrementalIndexState(cfg.index_state_path)
            state.replace_root(
                RootIndexState(
                    root_key=canonical_path(img_root),
                    root_path=str(img_root.resolve()),
                    recursive=True,
                    last_full_verification_ns=123456,
                    images=state_images,
                    directories={
                        canonical_path(img_root): DirectoryState(canonical_path(img_root), 111),
                        canonical_path(sub): DirectoryState(canonical_path(sub), 222),
                    },
                )
            )
            save_scope(
                cfg,
                FolderScope(
                    sort="mtime",
                    selected=[canonical_path(sub)],
                    known=[canonical_path(img_root), canonical_path(sub)],
                    mtime_after="2025-01-01",
                ),
            )

            # Provenance
            pstore = ProvenanceStore(cfg.provenance_path)
            a1 = pstore.ensure_author("pixiv", "111", name="Alice", handle="alice_h")
            a2 = pstore.ensure_author("twitter", "222", name="Bob", handle="bob_h")
            pstore.merge_authors(a2, a1)
            for idx, pid in enumerate(point_ids[:10]):
                pstore.upsert_scan(
                    pid,
                    {
                        "width": 64,
                        "height": 48,
                        "mtime_ns": 1000 + idx,
                        "content_hash": hashes[idx],
                    },
                    ParsedSource(
                        SourceRef(
                            "pixiv",
                            f"12345{idx:03d}",
                            0,
                            "https://www.pixiv.net/artworks/1",
                        )
                    ),
                )
                pstore.set_fetch_result(
                    pid,
                    {
                        "platform": "pixiv",
                        "source_id": f"12345{idx:03d}",
                        "page": 0,
                        "url": "https://www.pixiv.net/artworks/1",
                        "status": "ok",
                        "tags": ["thighhighs" if idx % 2 == 0 else "skirt"],
                    },
                    author_key=a1,
                )

            # Collections (with mixed artist item + stale ID + undo/redo)
            cstore = CollectionStore(cfg.collections_path)
            col = cstore.create("精选")
            cstore.add_item(col.id, point_ids[0])
            cstore.add_item(col.id, f"artist:{a1}")
            cstore.add_item(col.id, "stale-missing-id")
            cstore.remove_item(col.id, "stale-missing-id")
            cstore.undo()

            # Notes
            nstore = NoteStore(cfg.notes_dir)
            nstore.create_note(
                hashes[0],
                name="img0.png",
                path=str(img_root / "12345000_p0.png"),
                point_id=point_ids[0],
                title="笔记一",
                content="# 笔记一\n内容",
            )

            # Snapshot v1 file hashes to guarantee migrate_v1 never touches v1 files
            v1_files = [
                cfg.qdrant_path / "collection" / COLLECTION / "storage.sqlite",
                cfg.index_state_path,
                cfg.provenance_path,
                cfg.collections_path,
                cfg.folder_scope_path,
                cfg.manifest_path,
            ]
            before_digests = {
                p: hashlib.sha256(p.read_bytes()).hexdigest() for p in v1_files
            }

            report = migrate_v1(cfg)
            self.assertEqual(report.images_v1, 25)
            self.assertEqual(report.images_v2, 25)
            self.assertEqual(report.collections_v2, 1)
            self.assertEqual(report.artists_v2, 1)
            self.assertEqual(report.legacy_v2, report.legacy_v1)
            self.assertEqual(report.image_source_v2, 10)
            self.assertEqual(report.notes_files_v2, 1)

            # Verify v1 files untouched
            for p, expected_digest in before_digests.items():
                self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(), expected_digest)

            # Verify 20 random Qdrant queries match v2 VectorIndex top-20 (>= 95%, expect 100%)
            v2_vec = VectorIndex(cfg.visual_vectors_path, cfg.descriptor_vectors_path)
            db = Database(cfg.db_path)
            try:
                for qvec, expected_pids in qdrant_top20:
                    rows, _ = v2_vec.search(qvec, "visual", 20)
                    rows_db = db.read_conn().execute(
                        "SELECT id, point_id FROM images"
                    ).fetchall()
                    row_map = {int(r["id"]): str(r["point_id"]) for r in rows_db}
                    got_pids = {row_map[int(r)] for r in rows}
                    overlap = len(got_pids & expected_pids) / 20.0
                    self.assertGreaterEqual(overlap, 0.95)

                # Verify migrated repos work seamlessly
                v2_col = CollectionRepo(db).get(col.id)
                self.assertIsNotNone(v2_col)
                assert v2_col is not None
                self.assertEqual(len(v2_col.item_ids), 3)
                v2_prov = ProvenanceRepo(db)
                self.assertEqual(len(v2_prov.point_ids_for_tags(["thighhighs"])), 5)
            finally:
                db.close()

            # Verify idempotency (second run overwrites atomically cleanly)
            report2 = migrate_v1(cfg)
            self.assertEqual(report2.images_v2, 25)

            # Sanity check QdrantClient not corrupted
            qc = QdrantClient(path=str(cfg.qdrant_path))
            self.assertEqual(qc.count(COLLECTION, exact=True).count, 25)
            qc.close()


if __name__ == "__main__":
    unittest.main()
