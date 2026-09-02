from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from artseek.config import (
    AppConfig,
    apply_runtime_paths,
    load_config,
    migrate_storage_tree,
    save_storage_paths,
)
from artseek.image_io import file_digest, perceptual_hash
from artseek.notes import (
    NoteError,
    NoteStore,
    first_heading_title,
    resolve_content_hash,
    sanitize_note_filename,
    vault_asset_refs,
)


class FilenameTests(unittest.TestCase):
    def test_strips_markdown_suffix_and_forbidden_characters(self) -> None:
        self.assertEqual(sanitize_note_filename("构图观察.md"), "构图观察.md")
        self.assertEqual(sanitize_note_filename("a/b\\c"), "a_b_c.md")
        self.assertEqual(sanitize_note_filename("  "), "笔记.md")
        self.assertEqual(sanitize_note_filename("con"), "_con.md")
        self.assertTrue(sanitize_note_filename(".." ).endswith(".md"))
        self.assertNotIn("..", sanitize_note_filename("../secret").split("."))

    def test_first_heading_title_reads_only_leading_h1(self) -> None:
        self.assertEqual(first_heading_title("# 构图观察\n\n正文"), "构图观察")
        self.assertEqual(first_heading_title("# 服装 ##\n"), "服装")
        self.assertEqual(first_heading_title("#构图\n"), "构图")
        self.assertEqual(first_heading_title("# [链接](https://ex)\n"), "链接")
        self.assertEqual(first_heading_title("## 二级\n# 一级\n"), "")
        self.assertEqual(first_heading_title("\n# 空行之后\n"), "空行之后")
        self.assertEqual(first_heading_title("正文\n# 构图\n"), "")


class ContentHashTests(unittest.TestCase):
    def test_prefers_payload_hash_and_falls_back_to_file(self) -> None:
        digest = "a" * 64
        self.assertEqual(resolve_content_hash({"content_hash": digest.upper()}), digest)
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "image.bin"
            path.write_bytes(b"hello-notes")
            computed = resolve_content_hash({"path": str(path)})
            self.assertEqual(len(computed), 64)
            self.assertEqual(
                resolve_content_hash({"content_hash": computed, "path": str(path)}),
                computed,
            )
        with self.assertRaises(FileNotFoundError):
            resolve_content_hash({"path": "Z:/missing/artseek-note-test.png"})


class NoteStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary_directory.name) / "notes"
        self.store = NoteStore(self.root)
        self.digest = "ab" * 32

    def tearDown(self) -> None:
        self._temporary_directory.cleanup()

    def test_create_list_read_rename_delete(self) -> None:
        created = self.store.create_note(
            self.digest,
            title="构图",
            content="# 观察\n",
            name="girl.png",
            path="D:/pics/girl.png",
            point_id="point-1",
        )
        self.assertEqual(created.name, "构图.md")
        self.assertEqual(created.content, "# 观察\n")
        listed = self.store.list_notes(self.digest, name="girl.png", point_id="point-1")
        self.assertEqual([item.name for item in listed], ["构图.md"])
        second = self.store.create_note(self.digest, title="构图", content="dup")
        self.assertEqual(second.name, "构图-2.md")
        renamed = self.store.write_note(self.digest, "构图.md", title="服装层次", content="袖型")
        self.assertEqual(renamed.name, "服装层次.md")
        self.assertEqual(renamed.content, "袖型")
        with self.assertRaises(FileNotFoundError):
            self.store.read_note(self.digest, "构图.md")
        self.store.delete_note(self.digest, "服装层次.md")
        names = [item.name for item in self.store.list_notes(self.digest)]
        self.assertEqual(names, ["构图-2.md"])
        index = json.loads((self.root / ".artseek-index.json").read_text(encoding="utf-8"))
        self.assertEqual(index["images"][self.digest]["note_count"], 1)
        self.assertEqual(index["images"][self.digest]["name"], "girl.png")

    def test_h1_on_first_line_renames_note_and_skips_conflicts(self) -> None:
        created = self.store.create_note(
            self.digest, title="笔记", content="# 构图观察\n\n草稿\n"
        )
        self.assertEqual(created.name, "构图观察.md")
        draft = self.store.create_note(self.digest, title="草稿", content="旧")
        renamed = self.store.write_note(
            self.digest, draft.name, content="# 服装层次\n继续\n"
        )
        self.assertEqual(renamed.name, "服装层次.md")
        self.assertEqual(renamed.title, "服装层次")
        conflicted = self.store.write_note(
            self.digest, "服装层次.md", content="# 构图观察\n仍保存\n"
        )
        self.assertEqual(conflicted.name, "服装层次.md")
        self.assertEqual(conflicted.content, "# 构图观察\n仍保存\n")
        kept = self.store.write_note(
            self.digest, "构图观察.md", title="手动标题", content="# 构图观察\n"
        )
        self.assertEqual(kept.name, "手动标题.md")
        leading = self.store.create_note(self.digest, title="草稿二", content="x")
        synced = self.store.write_note(
            self.digest, leading.name, content="\n# 镜头\n正文\n"
        )
        self.assertEqual(synced.name, "镜头.md")

    def test_insertion_order_reorder_and_rename_keep_position(self) -> None:
        self.store.create_note(self.digest, title="服装", content="a")
        self.store.create_note(self.digest, title="构图", content="b")
        self.store.create_note(self.digest, title="观察", content="c")
        names = [item.name for item in self.store.list_notes(self.digest)]
        self.assertEqual(names, ["服装.md", "构图.md", "观察.md"])
        reordered = self.store.reorder_notes(self.digest, ["观察.md", "服装.md", "构图.md"])
        self.assertEqual([item.name for item in reordered], ["观察.md", "服装.md", "构图.md"])
        self.assertEqual(
            [item.name for item in self.store.list_notes(self.digest)],
            ["观察.md", "服装.md", "构图.md"],
        )
        renamed = self.store.write_note(self.digest, "服装.md", title="配色")
        self.assertEqual(renamed.name, "配色.md")
        self.assertEqual(
            [item.name for item in self.store.list_notes(self.digest)],
            ["观察.md", "配色.md", "构图.md"],
        )
        with self.assertRaises(NoteError):
            self.store.reorder_notes(self.digest, ["观察.md", "构图.md"])
        with self.assertRaises(NoteError):
            self.store.reorder_notes(self.digest, ["观察.md", "配色.md", "构图.md", "构图.md"])
        meta = json.loads((self.root / self.digest / ".artseek.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["note_order"], ["观察.md", "配色.md", "构图.md"])

    def test_legacy_folder_lists_alphabetically_until_reordered(self) -> None:
        folder = self.store.image_dir(self.digest)
        folder.mkdir(parents=True)
        (folder / "note-b.md").write_text("b", encoding="utf-8")
        (folder / "note-a.md").write_text("a", encoding="utf-8")
        self.assertEqual(
            [item.name for item in self.store.list_notes(self.digest)],
            ["note-a.md", "note-b.md"],
        )
        self.store.reorder_notes(self.digest, ["note-b.md", "note-a.md"])
        self.assertEqual(
            [item.name for item in self.store.list_notes(self.digest)],
            ["note-b.md", "note-a.md"],
        )

    def test_rejects_path_escape_and_oversized_content(self) -> None:
        with self.assertRaises(NoteError):
            self.store.read_note("not-a-hash", "a.md")
        self.store.create_note(self.digest, title="safe", content="ok")
        with self.assertRaises(NoteError):
            self.store.public_file(self.digest, "../secret.txt")
        with self.assertRaises(NoteError):
            self.store.public_file(self.digest, "note.md")
        with self.assertRaises(NoteError):
            self.store.create_note(self.digest, content="x" * (2 * 1024 * 1024 + 8))

    def test_assets_stay_inside_image_folder(self) -> None:
        relative = self.store.save_asset(self.digest, "paste.png", b"\x89PNG", name="girl.png")
        self.assertTrue(relative.startswith("assets/"))
        path = self.store.public_file(self.digest, relative)
        self.assertEqual(path.read_bytes(), b"\x89PNG")
        self.assertTrue(path.is_relative_to(self.root / self.digest / "assets"))

    def test_delete_note_only_removes_unreferenced_vault_images(self) -> None:
        shared = self.store.save_asset(self.digest, "shared.png", b"shared")
        owned = self.store.save_asset(self.digest, "only.png", b"only")
        self.store.create_note(
            self.digest,
            title="甲",
            content=f"![]({shared})\n![]({owned})\n![](https://example.com/x.png)\n",
            point_id="point-1",
        )
        self.store.create_note(
            self.digest,
            title="乙",
            content=f"![]({shared})\n",
            point_id="point-1",
        )
        removed = self.store.delete_note(self.digest, "甲.md")
        self.assertEqual(removed, [owned])
        self.assertFalse((self.root / self.digest / owned).exists())
        self.assertTrue((self.root / self.digest / shared).exists())
        self.assertTrue((self.root / self.digest / "乙.md").exists())

    def test_delete_folder_keeps_notes_until_explicit_and_survives_point_lookup(self) -> None:
        self.store.create_note(self.digest, title="构图", content="ok", point_id="point-9")
        self.assertEqual(self.store.lookup_by_point_id("point-9"), self.digest)
        self.store.delete_folder(self.digest)
        self.assertFalse((self.root / self.digest).exists())
        self.assertIsNone(self.store.lookup_by_point_id("point-9"))

    def test_resized_copy_shares_canonical_folder(self) -> None:
        pattern = Image.new("RGB", (96, 64))
        pixels = pattern.load()
        for y in range(64):
            for x in range(96):
                pixels[x, y] = ((x * 13) % 210, (y * 19) % 190, (x + y) % 170)
        small = self.root.parent / "small.png"
        large = self.root.parent / "large.png"
        pattern.save(small)
        pattern.resize((384, 256), Image.Resampling.LANCZOS).save(large)
        small_hash = file_digest(small)
        large_hash = file_digest(large)
        self.assertNotEqual(small_hash, large_hash)
        created = self.store.create_note(
            small_hash,
            title="构图",
            content="same picture",
            phash=perceptual_hash(small),
            width=96,
            height=64,
            point_id="point-small",
        )
        listed = self.store.list_notes(
            large_hash,
            phash=perceptual_hash(large),
            width=384,
            height=256,
            point_id="point-large",
        )
        self.assertEqual(created.content_hash, small_hash)
        self.assertEqual([item.name for item in listed], ["构图.md"])
        self.assertEqual(self.store.resolve_canonical(
            large_hash,
            phash=perceptual_hash(large),
            width=384,
            height=256,
        ), small_hash)
        self.assertTrue((self.root / small_hash / "构图.md").is_file())
        self.assertFalse((self.root / large_hash).exists())

    def test_vault_asset_refs_ignore_external_urls(self) -> None:
        refs = vault_asset_refs(
            "![](assets/a.png)\n![](/api/note-files/" + self.digest + "/assets/b.webp)\n"
            "![](https://cdn.example/c.png)\n![](D:/Pictures/orig.png)\n",
            self.digest,
        )
        self.assertEqual(refs, {"assets/a.png", "assets/b.webp"})


class StoragePathTests(unittest.TestCase):
    def test_runtime_overrides_win_over_toml(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            data_dir = root / "data"
            cache_dir = root / "cache"
            toml_qdrant = root / "toml-qdrant"
            runtime_qdrant = root / "runtime-qdrant"
            runtime_notes = root / "runtime-notes"
            config_path = root / "config.toml"
            config_path.write_text(
                "\n".join(
                    [
                        f'data_dir = "{data_dir.as_posix()}"',
                        f'cache_dir = "{cache_dir.as_posix()}"',
                        f'qdrant_path = "{toml_qdrant.as_posix()}"',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            cfg = AppConfig(data_dir=data_dir, cache_dir=cache_dir)
            cfg.ensure_dirs()
            save_storage_paths(
                cfg,
                qdrant_path=runtime_qdrant,
                notes_dir=runtime_notes,
                update_qdrant=True,
                update_notes=True,
            )
            loaded = load_config(config_path)
            self.assertEqual(loaded.qdrant_path.resolve(), runtime_qdrant.resolve())
            self.assertEqual(loaded.notes_dir.resolve(), runtime_notes.resolve())
            self.assertTrue(runtime_notes.is_dir())

    def test_empty_runtime_path_restores_default(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            cfg = AppConfig(data_dir=root / "data", cache_dir=root / "cache")
            cfg.ensure_dirs()
            default_qdrant = cfg.qdrant_path
            save_storage_paths(cfg, qdrant_path=root / "other-qdrant", update_qdrant=True)
            self.assertNotEqual(cfg.qdrant_path, default_qdrant)
            save_storage_paths(cfg, qdrant_path=None, update_qdrant=True)
            apply_runtime_paths(cfg)
            self.assertEqual(cfg.qdrant_path, cfg.data_dir / "qdrant")

    def test_migrate_storage_tree_copies_and_skips_lock(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "qdrant"
            dest = root / "moved" / "qdrant"
            (source / "collection").mkdir(parents=True)
            (source / "meta.json").write_text("{}", encoding="utf-8")
            (source / ".lock").write_text("pid", encoding="utf-8")
            path, copied = migrate_storage_tree(source, dest)
            self.assertTrue(copied)
            self.assertEqual(path.resolve(), dest.resolve())
            self.assertTrue((dest / "meta.json").is_file())
            self.assertTrue((dest / "collection").is_dir())
            self.assertFalse((dest / ".lock").exists())
            self.assertTrue((source / "meta.json").is_file())

    def test_migrate_storage_tree_rejects_overlap_and_nonempty(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "notes"
            source.mkdir()
            (source / "keep.md").write_text("x", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "相同"):
                migrate_storage_tree(source, source)
            with self.assertRaisesRegex(ValueError, "内部"):
                migrate_storage_tree(source, source / "nested")
            with self.assertRaisesRegex(ValueError, "上级"):
                migrate_storage_tree(source, root)
            occupied = root / "occupied"
            occupied.mkdir()
            (occupied / "other.md").write_text("y", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "空目录"):
                migrate_storage_tree(source, occupied)
            empty, copied = migrate_storage_tree(root / "missing", root / "fresh")
            self.assertFalse(copied)
            self.assertTrue(empty.is_dir())
