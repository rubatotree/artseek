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
    index_roots_from_scope,
    load_catalog,
    load_scope,
    root_image_counts,
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
            self.assertEqual(set(loaded.subtree_ids(canonical_path(root))), {"a", "b"})
            self.assertIsNone(load_catalog(path, 3))
            unscoped = load_catalog(path)
            self.assertIsNotNone(unscoped)
            assert unscoped is not None
            self.assertEqual(set(unscoped.subtree_ids(canonical_path(root))), {"a", "b"})

    def test_scope_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cfg = AppConfig(data_dir=Path(directory), cache_dir=Path(directory) / "cache")
            save_scope(cfg, FolderScope(sort="mtime", selected=["D:/pics/a"], known=["D:/pics/a"]))
            loaded = load_scope(cfg)
            self.assertEqual(loaded.sort, "mtime")
            self.assertEqual(loaded.selected, [canonical_path("D:/pics/a")])

    def test_apply_scope_keeps_sort_as_string(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = build_catalog([("a", _payload(root / "a.png", root))])
            scope, _ = apply_scope(catalog, FolderScope(sort="count", selected=[], known=[]))
            self.assertEqual(scope.sort, "count")
            self.assertIsInstance(scope.sort, str)
            cfg = AppConfig(data_dir=root, cache_dir=root / "cache")
            save_scope(cfg, scope)
            self.assertEqual(load_scope(cfg).sort, "count")

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

    def test_folder_images_follow_directory_sort(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            few = root / "few"
            many = root / "many"
            few.mkdir()
            many.mkdir()
            catalog = build_catalog(
                [
                    ("few-a", _payload(few / "a.png", root, mtime=10)),
                    ("many-z", _payload(many / "z.png", root, mtime=30)),
                    ("many-m", _payload(many / "m.png", root, mtime=20)),
                ]
            )
            root_key = canonical_path(root)

            def names(method: str) -> list[str]:
                return [name for _, name, _ in catalog.list_folder_images(root_key, method)]

            self.assertEqual(names("name"), ["a.png", "m.png", "z.png"])
            self.assertEqual(names("name_desc"), ["z.png", "m.png", "a.png"])
            self.assertEqual(names("mtime"), ["z.png", "m.png", "a.png"])
            self.assertEqual(names("mtime_asc"), ["a.png", "m.png", "z.png"])
            self.assertEqual(names("count"), ["m.png", "z.png", "a.png"])
            self.assertEqual(names("count_asc"), ["a.png", "m.png", "z.png"])

    def test_folder_images_stay_inside_clicked_subtree(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clothes = root / "clothes"
            other = root / "other"
            clothes.mkdir()
            other.mkdir()
            catalog = build_catalog(
                [
                    ("c", _payload(clothes / "coat.png", root, mtime=5)),
                    ("o", _payload(other / "sky.png", root, mtime=9)),
                ]
            )
            names = [name for _, name, _ in catalog.list_folder_images(canonical_path(clothes), "name")]
            self.assertEqual(names, ["coat.png"])

    def test_folder_images_unknown_key_is_empty(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = build_catalog([("a", _payload(root / "a.png", root))])
            self.assertEqual(catalog.list_folder_images(str(root / "missing"), "name"), [])

    def test_catalog_roundtrip_keeps_image_names(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = build_catalog([("a", _payload(root / "sky.png", root, mtime=11))])
            path = root / "folder_catalog.json"
            save_catalog(path, catalog, 1)
            loaded = load_catalog(path, 1)
            self.assertIsNotNone(loaded)
            assert loaded is not None
            refs = loaded.list_folder_images(canonical_path(root), "name")
            self.assertEqual(refs, [("a", "sky.png", 11)])

    def test_day_bounds_cover_full_local_day(self) -> None:
        start = day_start_ns("2024-05-01")
        end = day_end_ns("2024-05-01")
        self.assertIsNotNone(start)
        self.assertIsNotNone(end)
        assert start is not None and end is not None
        self.assertLess(start, end)
        self.assertEqual(day_start_ns(""), None)

    def test_root_image_counts_include_nested(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "one"
            second = Path(directory) / "two"
            nested = first / "nested"
            nested.mkdir(parents=True)
            second.mkdir()
            catalog = build_catalog(
                [
                    ("a", _payload(first / "a.png", first)),
                    ("b", _payload(nested / "b.png", first)),
                    ("c", _payload(second / "c.png", second)),
                ]
            )
            counts = dict(root_image_counts(catalog))
            self.assertEqual(counts[str(first)], 2)
            self.assertEqual(counts[str(second)], 1)

    def test_index_roots_from_scope_ignore_children(self) -> None:
        parent = canonical_path(Path("D:/pics/a"))
        child = canonical_path(Path("D:/pics/a/child"))
        other = canonical_path(Path("D:/pics/b"))
        roots = index_roots_from_scope(FolderScope(known=[parent, child, other]))
        self.assertEqual(roots, sorted([parent, other], key=str.casefold))

    def test_folder_key_fallback(self) -> None:
        path = Path("D:/art/set/img.png")
        self.assertEqual(
            folder_key_of({"path": str(path)}),
            canonical_path(path.parent),
        )


if __name__ == "__main__":
    unittest.main()
