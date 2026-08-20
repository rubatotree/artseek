from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from artseek.config import AppConfig
from artseek.image_io import canonical_path
from artseek.library import (
    FolderScope,
    apply_scope,
    build_catalog,
    day_end_ns,
    day_start_ns,
    folder_key_of,
    load_catalog,
    load_scope,
    save_catalog,
    save_scope,
)


def _payload(path: Path, root: Path, mtime: int = 1) -> dict:
    return {
        "path": str(path),
        "path_key": canonical_path(path),
        "root": str(root),
        "root_key": canonical_path(root),
        "folder_key": canonical_path(path.parent),
        "mtime_ns": mtime,
        "name": path.name,
    }


class LibraryTreeTests(unittest.TestCase):
    def test_nested_tree_and_direct_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nested = root / "clothes" / "sleeves"
            nested.mkdir(parents=True)
            payloads = [
                ("a", _payload(root / "top.png", root, mtime=10)),
                ("b", _payload(nested / "wide.png", root, mtime=30)),
                ("c", _payload(nested / "puff.png", root, mtime=20)),
            ]
            catalog = build_catalog(payloads)
            roots = catalog.tree("name")
            self.assertEqual(len(roots), 1)
            self.assertEqual(roots[0].count, 1)
            self.assertEqual(roots[0].total, 3)
            clothes = roots[0].children[0]
            self.assertEqual(clothes.name, "clothes")
            self.assertEqual(clothes.count, 0)
            self.assertEqual(clothes.total, 2)
            self.assertEqual(clothes.children[0].count, 2)

    def test_new_folders_are_unselected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = root / "old"
            fresh = root / "fresh"
            old.mkdir()
            fresh.mkdir()
            catalog = build_catalog(
                [
                    ("a", _payload(old / "a.png", root)),
                    ("b", _payload(fresh / "b.png", root)),
                ]
            )
            old_key = canonical_path(old)
            fresh_key = canonical_path(fresh)
            scope, new_keys = apply_scope(
                catalog,
                FolderScope(selected=[old_key], known=[old_key, canonical_path(root)]),
            )
            self.assertIn(old_key, scope.selected)
            self.assertNotIn(fresh_key, scope.selected)
            self.assertIn(fresh_key, new_keys)

    def test_first_visit_selects_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = build_catalog([("a", _payload(root / "a.png", root))])
            scope, new_keys = apply_scope(catalog, FolderScope())
            self.assertEqual(scope.selected, [])
            self.assertEqual(new_keys, [])
            self.assertIn(canonical_path(root), scope.known)

    def test_name_sort_mixed_digits_and_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            numeric = root / "133"
            text = root / "bjd"
            numeric.mkdir()
            text.mkdir()
            catalog = build_catalog(
                [
                    ("a", _payload(numeric / "a.png", root)),
                    ("b", _payload(text / "b.png", root)),
                ]
            )
            names = [node.name for node in catalog.tree("name")[0].children]
            self.assertEqual(names, ["133", "bjd"])

    def test_sort_by_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            few = root / "few"
            many = root / "many"
            few.mkdir()
            many.mkdir()
            catalog = build_catalog(
                [
                    ("a", _payload(few / "a.png", root)),
                    ("b", _payload(many / "b.png", root)),
                    ("c", _payload(many / "c.png", root)),
                ]
            )
            children = catalog.tree("count")[0].children
            self.assertEqual([node.name for node in children], ["many", "few"])

    def test_all_folders_selected_skips_id_filter(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = build_catalog([("a", _payload(root / "a.png", root))])
            self.assertIsNone(catalog.point_ids(list(catalog.all_keys())))

    def test_subtree_ids_include_nested(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nested = root / "nested"
            nested.mkdir()
            catalog = build_catalog(
                [
                    ("root-img", _payload(root / "a.png", root)),
                    ("nested-img", _payload(nested / "b.png", root)),
                ]
            )
            self.assertEqual(set(catalog.subtree_ids(canonical_path(root))), {"root-img", "nested-img"})
            self.assertEqual(catalog.subtree_ids(canonical_path(nested)), ["nested-img"])

    def test_point_ids_only_direct_folder(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nested = root / "nested"
            nested.mkdir()
            catalog = build_catalog(
                [
                    ("root-img", _payload(root / "a.png", root)),
                    ("nested-img", _payload(nested / "b.png", root)),
                ]
            )
            ids = catalog.point_ids([canonical_path(root)])
            self.assertEqual(ids, ["root-img"])

    def test_catalog_sidecar_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nested = root / "nested"
            nested.mkdir()
            catalog = build_catalog(
                [
                    ("a", _payload(root / "a.png", root, mtime=11)),
                    ("b", _payload(nested / "b.png", root, mtime=22)),
                ]
            )
            path = root / "folder_catalog.json"
            save_catalog(path, catalog, 2)
            loaded = load_catalog(path, 2)
            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertEqual(loaded.point_ids([canonical_path(nested)]), ["b"])
            self.assertIsNone(load_catalog(path, 3))

    def test_scope_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cfg = AppConfig(data_dir=Path(directory), cache_dir=Path(directory) / "cache")
            save_scope(cfg, FolderScope(sort="mtime", selected=["D:/pics/a"], known=["D:/pics/a"]))
            loaded = load_scope(cfg)
            self.assertEqual(loaded.sort, "mtime")
            self.assertEqual(loaded.selected, [canonical_path("D:/pics/a")])

    def test_time_range_filters_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = build_catalog(
                [
                    ("old", _payload(root / "old.png", root, mtime=10)),
                    ("new", _payload(root / "new.png", root, mtime=50)),
                ]
            )
            keys = list(catalog.all_keys())
            self.assertIsNone(catalog.point_ids(keys))
            self.assertEqual(catalog.point_ids(keys, after_ns=40), ["new"])
            self.assertEqual(catalog.point_ids(keys, before_ns=20), ["old"])
            self.assertEqual(catalog.point_ids(keys, after_ns=40, before_ns=45), [])

    def test_day_bounds_cover_full_local_day(self) -> None:
        start = day_start_ns("2024-05-01")
        end = day_end_ns("2024-05-01")
        self.assertIsNotNone(start)
        self.assertIsNotNone(end)
        assert start is not None and end is not None
        self.assertLess(start, end)
        self.assertEqual(day_start_ns(""), None)

    def test_folder_key_fallback(self) -> None:
        path = Path("D:/art/set/img.png")
        self.assertEqual(
            folder_key_of({"path": str(path)}),
            canonical_path(path.parent),
        )


if __name__ == "__main__":
    unittest.main()
