from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from artseek.config import AppConfig
from artseek.image_io import canonical_path
from artseek.incremental_state import IndexedImageState
from artseek.indexer import ImageIndexer
from artseek.library import build_catalog, save_catalog
from artseek.search import SearchEngine


class StubEmbedder:
    dimension = 3

    def __init__(self) -> None:
        self.image_calls = 0
        self.text_calls = 0

    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        self.image_calls += 1
        return np.ones((len(images), self.dimension), dtype=np.float32)

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        self.text_calls += 1
        return np.ones((len(texts), self.dimension), dtype=np.float32)


class StubStore:
    def __init__(self) -> None:
        self.points: dict[str, dict[str, Any]] = {}
        self.scroll_calls = 0
        self.upsert_calls = 0
        self.delete_calls = 0
        self.folder_payload_calls = 0

    def scroll_root(self, root_key: str) -> dict[str, tuple[str, dict[str, Any]]]:
        self.scroll_calls += 1
        return {
            str(payload["path_key"]): (point_id, payload)
            for point_id, payload in self.points.items()
            if payload.get("root_key") == root_key
        }

    def upsert(self, points: list[Any]) -> None:
        self.upsert_calls += 1
        for point in points:
            self.points[str(point.id)] = dict(point.payload)

    def delete_ids(self, ids: list[str]) -> None:
        self.delete_calls += 1
        for point_id in ids:
            self.points.pop(point_id, None)

    def iter_folder_payloads(self):
        self.folder_payload_calls += 1
        return list(self.points.items())

    def count(self) -> int:
        return len(self.points)


def make_config(directory: Path, *, verification_seconds: int = 24 * 60 * 60) -> AppConfig:
    config = AppConfig(data_dir=directory / "data", cache_dir=directory / "cache")
    config.tagger.enabled = False
    config.llm.enabled = False
    config.indexing.full_verification_interval_seconds = verification_seconds
    config.ensure_dirs()
    return config


def write_image(path: Path, color: tuple[int, int, int] = (10, 20, 30)) -> None:
    Image.new("RGB", (3, 2), color).save(path)


class IncrementalIndexTests(unittest.TestCase):
    def test_unchanged_root_skips_qdrant_scroll_embedding_and_catalog_rebuild(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "images"
            root.mkdir()
            write_image(root / "one.png")
            config = make_config(base)
            store = StubStore()
            embedder = StubEmbedder()
            indexer = ImageIndexer(config, store=store, embedder=embedder)

            first = indexer.index([root])
            self.assertEqual(first.indexed, 1)
            self.assertEqual(store.scroll_calls, 1)
            self.assertEqual(embedder.image_calls, 1)
            self.assertEqual(store.folder_payload_calls, 0)

            second = indexer.index([root])
            self.assertEqual(second.discovered, 1)
            self.assertEqual(second.unchanged, 1)
            self.assertEqual(second.indexed, 0)
            self.assertEqual(store.scroll_calls, 1)
            self.assertEqual(embedder.image_calls, 1)
            self.assertEqual(store.folder_payload_calls, 0)

    def test_delta_add_and_delete_apply_exact_qdrant_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "images"
            nested = root / "nested"
            nested.mkdir(parents=True)
            first_path = nested / "one.png"
            write_image(first_path)
            config = make_config(base)
            store = StubStore()
            embedder = StubEmbedder()
            indexer = ImageIndexer(config, store=store, embedder=embedder)
            indexer.index([root])

            second_path = nested / "two.png"
            write_image(second_path, (40, 50, 60))
            added = indexer.index([root])
            self.assertEqual(added.indexed, 1)
            self.assertEqual(store.scroll_calls, 1)
            self.assertEqual(store.folder_payload_calls, 0)
            self.assertEqual(len(store.points), 2)
            point_paths = {item["path_key"] for item in store.points.values()}
            self.assertIn(canonical_path(second_path), point_paths)

            first_path.unlink()
            deleted = indexer.index([root])
            self.assertEqual(deleted.indexed, 0)
            self.assertEqual(deleted.deleted, 1)
            self.assertEqual(store.scroll_calls, 1)
            self.assertEqual(store.folder_payload_calls, 0)
            self.assertEqual(len(store.points), 1)
            self.assertEqual(
                {item["path_key"] for item in store.points.values()}, {canonical_path(second_path)}
            )

            second_path.unlink()
            nested.rmdir()
            removed_subtree = indexer.index([root])
            self.assertEqual(removed_subtree.deleted, 1)
            self.assertEqual(store.scroll_calls, 1)
            self.assertEqual(store.folder_payload_calls, 0)
            self.assertEqual(store.points, {})

    def test_dirty_snapshot_forces_a_full_qdrant_reconciliation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "images"
            root.mkdir()
            write_image(root / "one.png")
            config = make_config(base)
            store = StubStore()
            indexer = ImageIndexer(config, store=store, embedder=StubEmbedder())
            indexer.index([root])

            indexer.state.mark_dirty(canonical_path(root))
            summary = indexer.index([root])

            self.assertEqual(summary.indexed, 1)
            self.assertEqual(store.scroll_calls, 2)
            self.assertFalse(indexer.state.is_dirty(canonical_path(root)))
            self.assertIsNotNone(indexer.state.load_root(canonical_path(root)))

    def test_qdrant_upsert_is_preceded_by_a_persisted_dirty_marker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "images"
            root.mkdir()
            write_image(root / "one.png")
            config = make_config(base)
            store = StubStore()
            indexer = ImageIndexer(config, store=store, embedder=StubEmbedder())
            indexer.index([root])
            root_key = canonical_path(root)
            observed_states = []
            original_upsert = store.upsert

            def inspect_upsert(points: list[Any]) -> None:
                observed_states.append(indexer.state.load_root(root_key))
                original_upsert(points)

            store.upsert = inspect_upsert  # type: ignore[method-assign]
            write_image(root / "two.png", (40, 50, 60))
            indexer.index([root])

            self.assertEqual(observed_states, [None])
            self.assertIsNotNone(indexer.state.load_root(root_key))

    def test_missing_catalog_is_restored_from_a_valid_sidecar_without_qdrant_scroll(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "images"
            root.mkdir()
            write_image(root / "one.png")
            config = make_config(base)
            store = StubStore()
            indexer = ImageIndexer(config, store=store, embedder=StubEmbedder())
            indexer.index([root])
            config.folder_catalog_path.unlink()

            indexer.index([root])

            self.assertTrue(config.folder_catalog_path.exists())
            self.assertEqual(store.scroll_calls, 1)
            self.assertEqual(store.folder_payload_calls, 0)

    def test_case_only_display_name_change_is_reindexed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "Case.png"
            write_image(path)
            stat = path.stat()
            key = canonical_path(path)
            previous = IndexedImageState(
                path_key=key,
                root_key="root",
                point_id="point",
                mtime_ns=stat.st_mtime_ns,
                size_bytes=stat.st_size,
                folder_key=canonical_path(path.parent),
                name="case.png",
            )

            changed = ImageIndexer._changed_paths({key: path}, {key: previous}, force=False)

            self.assertEqual(changed, {key: path})

    def test_deleted_root_invalidates_snapshot_and_preserves_file_not_found_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "images"
            root.mkdir()
            image = root / "one.png"
            write_image(image)
            config = make_config(base)
            store = StubStore()
            indexer = ImageIndexer(config, store=store, embedder=StubEmbedder())
            indexer.index([root])

            image.unlink()
            root.rmdir()
            root_key = canonical_path(root)
            with self.assertRaises(FileNotFoundError):
                indexer.index([root])

            self.assertIsNone(indexer.state.load_root(root_key))
            self.assertNotIn(root_key, indexer.state.root_keys())

    def test_drop_subtree_removes_images_by_folder_when_path_key_is_outside(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            deleted_folder = canonical_path(root / "deleted")
            images = {
                canonical_path(root / "outside" / "linked.png"): IndexedImageState(
                    path_key=canonical_path(root / "outside" / "linked.png"),
                    root_key=canonical_path(root),
                    point_id="linked",
                    mtime_ns=1,
                    size_bytes=1,
                    folder_key=canonical_path(root / "deleted" / "nested"),
                    name="linked.png",
                )
            }

            ImageIndexer._drop_subtree(images, {}, deleted_folder)

            self.assertEqual(images, {})

    def test_force_and_expired_verification_use_full_reconciliation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "images"
            root.mkdir()
            write_image(root / "one.png")
            config = make_config(base, verification_seconds=0)
            store = StubStore()
            embedder = StubEmbedder()
            indexer = ImageIndexer(config, store=store, embedder=embedder)
            indexer.index([root])
            indexer.index([root])
            self.assertEqual(store.scroll_calls, 2)

            config.indexing.full_verification_interval_seconds = 24 * 60 * 60
            indexer.index([root], force=True)
            self.assertEqual(store.scroll_calls, 3)
            self.assertEqual(embedder.image_calls, 2)

    def test_remove_missing_false_keeps_missing_points_and_invalidates_delta_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "images"
            root.mkdir()
            image = root / "one.png"
            write_image(image)
            config = make_config(base)
            store = StubStore()
            indexer = ImageIndexer(config, store=store, embedder=StubEmbedder())
            indexer.index([root])

            image.unlink()
            summary = indexer.index([root], remove_missing=False)

            self.assertEqual(summary.deleted, 0)
            self.assertEqual(store.scroll_calls, 2)
            self.assertEqual(len(store.points), 1)
            self.assertIsNone(indexer.state.load_root(canonical_path(root)))


class FolderCatalogHandoffTests(unittest.TestCase):
    def test_replacing_memory_catalog_keeps_persisted_catalog(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            config = make_config(base)
            image = base / "images" / "one.png"
            image.parent.mkdir()
            catalog = build_catalog(
                [
                    (
                        "point-1",
                        {
                            "path": str(image),
                            "path_key": canonical_path(image),
                            "root": str(image.parent),
                            "root_key": canonical_path(image.parent),
                            "folder_key": canonical_path(image.parent),
                            "name": image.name,
                            "mtime_ns": 1,
                        },
                    )
                ]
            )
            save_catalog(config.folder_catalog_path, catalog, count=1)
            engine = SearchEngine(config, store=StubStore(), embedder_factory=StubEmbedder)

            engine.replace_folder_catalog(catalog)
            self.assertTrue(config.folder_catalog_path.exists())
            self.assertIs(engine.folder_catalog(), catalog)
