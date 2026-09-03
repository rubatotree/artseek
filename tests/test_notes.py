from __future__ import annotations

import json
import re
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from artseek.config import (
    AppConfig,
    apply_runtime_paths,
    load_config,
    migrate_index_sidecars,
    migrate_storage_tree,
    save_storage_paths,
)
from artseek.image_io import file_digest, perceptual_hash
from artseek.notes import (
    NoteError,
    NoteStore,
    apply_title_to_content,
    display_title,
    first_line_title,
    resolve_content_hash,
    sanitize_note_filename,
    timestamp_note_filename,
    vault_asset_refs,
)

NOTE_TIMESTAMP_NAME = re.compile(r"^\d{8}T\d{6}(?:-\d+)?\.md$")


class FilenameTests(unittest.TestCase):
    def test_strips_markdown_suffix_and_forbidden_characters(self) -> None:
        self.assertEqual(sanitize_note_filename("构图观察.md"), "构图观察.md")
        self.assertEqual(sanitize_note_filename("a/b\\c"), "a_b_c.md")
        self.assertEqual(sanitize_note_filename("  "), "笔记.md")
        self.assertEqual(sanitize_note_filename("con"), "_con.md")
        self.assertTrue(sanitize_note_filename(".." ).endswith(".md"))
        self.assertNotIn("..", sanitize_note_filename("../secret").split("."))

    def test_timestamp_filename_is_utc_and_unique_shape(self) -> None:
        name = timestamp_note_filename(datetime(2026, 9, 2, 14, 30, 52, tzinfo=UTC))
        self.assertEqual(name, "20260902T143052.md")
        self.assertRegex(timestamp_note_filename(), NOTE_TIMESTAMP_NAME)

    def test_first_line_title_uses_any_leading_line(self) -> None:
        self.assertEqual(first_line_title("# 构图观察\n\n正文"), "构图观察")
        self.assertEqual(first_line_title("# 服装 ##\n"), "服装")
        self.assertEqual(first_line_title("#构图\n"), "构图")
        self.assertEqual(first_line_title("# [链接](https://ex)\n"), "链接")
        self.assertEqual(first_line_title("## 二级\n# 一级\n"), "二级")
        self.assertEqual(first_line_title("\n# 空行之后\n"), "空行之后")
        self.assertEqual(first_line_title("正文\n# 构图\n"), "正文")
        self.assertEqual(first_line_title("构图观察\n下一段"), "构图观察")
        self.assertEqual(first_line_title(""), "")
        self.assertEqual(display_title("", "20260902T143052.md"), "笔记")
        self.assertEqual(display_title("", "构图.md"), "构图")
        self.assertEqual(
            apply_title_to_content("# 构图观察\n\n草稿\n", "服装层次"),
            "# 服装层次\n\n草稿\n",
        )
        self.assertEqual(apply_title_to_content("", "构图"), "构图\n")


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
        self.assertRegex(created.name, NOTE_TIMESTAMP_NAME)
        self.assertNotEqual(created.name, "笔记.md")
        self.assertEqual(created.title, "观察")
        self.assertEqual(created.content, "# 观察\n")
        listed = self.store.list_notes(self.digest, name="girl.png", point_id="point-1")
        self.assertEqual([item.name for item in listed], [created.name])
        self.assertEqual(listed[0].title, "观察")
        second = self.store.create_note(self.digest, title="构图", content="dup")
        self.assertRegex(second.name, NOTE_TIMESTAMP_NAME)
        self.assertNotEqual(second.name, created.name)
        self.assertEqual(second.title, "dup")
        updated = self.store.write_note(self.digest, created.name, content="袖型")
        self.assertEqual(updated.name, created.name)
        self.assertEqual(updated.title, "袖型")
        renamed = self.store.write_note(self.digest, created.name, title="服装层次")
        self.assertEqual(renamed.name, created.name)
        self.assertEqual(renamed.title, "服装层次")
        self.assertEqual(self.store.read_note(self.digest, created.name).title, "服装层次")
        self.store.delete_note(self.digest, created.name)
        names = [item.name for item in self.store.list_notes(self.digest)]
        self.assertEqual(names, [second.name])
        index = json.loads((self.root / ".artseek-index.json").read_text(encoding="utf-8"))
        self.assertEqual(index["images"][self.digest]["note_count"], 1)
        self.assertEqual(index["images"][self.digest]["name"], "girl.png")

    def test_empty_notes_use_unique_timestamps_not_default_filename(self) -> None:
        first = self.store.create_note(self.digest)
        second = self.store.create_note(self.digest)
        self.assertRegex(first.name, NOTE_TIMESTAMP_NAME)
        self.assertRegex(second.name, NOTE_TIMESTAMP_NAME)
        self.assertNotEqual(first.name, second.name)
        self.assertEqual(first.title, "笔记")
        self.assertEqual(second.title, "笔记")
        self.assertNotIn(first.name, {"笔记.md", "未命名.md"})
        with patch("artseek.notes.timestamp_note_filename", return_value="20260902T143052.md"):
            collided = self.store.create_note(self.digest)
            again = self.store.create_note(self.digest)
        self.assertEqual(collided.name, "20260902T143052.md")
        self.assertEqual(again.name, "20260902T143052-2.md")
        self.assertEqual(collided.title, "笔记")

    def test_first_line_is_display_title_without_renaming_file(self) -> None:
        created = self.store.create_note(self.digest, content="# 构图观察\n\n草稿\n")
        self.assertRegex(created.name, NOTE_TIMESTAMP_NAME)
        self.assertEqual(created.title, "构图观察")
        draft = self.store.create_note(self.digest, content="旧")
        renamed = self.store.write_note(
            self.digest, draft.name, content="# 服装层次\n继续\n"
        )
        self.assertEqual(renamed.name, draft.name)
        self.assertEqual(renamed.title, "服装层次")
        same_title = self.store.write_note(
            self.digest, renamed.name, content="# 构图观察\n仍保存\n"
        )
        self.assertEqual(same_title.name, draft.name)
        self.assertEqual(same_title.title, "构图观察")
        self.assertEqual(same_title.content, "# 构图观察\n仍保存\n")
        kept = self.store.write_note(
            self.digest, created.name, title="手动标题", content="# 构图观察\n"
        )
        self.assertEqual(kept.name, created.name)
        self.assertEqual(kept.title, "构图观察")
        retitled = self.store.write_note(self.digest, created.name, title="手动标题")
        self.assertEqual(retitled.name, created.name)
        self.assertEqual(retitled.title, "手动标题")
        self.assertTrue(retitled.content.startswith("# 手动标题"))
        leading = self.store.create_note(self.digest, content="x")
        synced = self.store.write_note(
            self.digest, leading.name, content="\n镜头\n正文\n"
        )
        self.assertEqual(synced.name, leading.name)
        self.assertEqual(synced.title, "镜头")
        seeded = self.store.create_note(self.digest, title="服装")
        self.assertEqual(seeded.title, "服装")
        self.assertEqual(seeded.content, "服装\n")

    def test_insertion_order_reorder_and_rename_keep_position(self) -> None:
        clothing = self.store.create_note(self.digest, title="服装")
        composition = self.store.create_note(self.digest, title="构图")
        observation = self.store.create_note(self.digest, title="观察")
        names = [item.name for item in self.store.list_notes(self.digest)]
        self.assertEqual(names, [clothing.name, composition.name, observation.name])
        self.assertEqual(
            [item.title for item in self.store.list_notes(self.digest)],
            ["服装", "构图", "观察"],
        )
        reordered = self.store.reorder_notes(
            self.digest, [observation.name, clothing.name, composition.name]
        )
        self.assertEqual(
            [item.name for item in reordered],
            [observation.name, clothing.name, composition.name],
        )
        self.assertEqual(
            [item.name for item in self.store.list_notes(self.digest)],
            [observation.name, clothing.name, composition.name],
        )
        renamed = self.store.write_note(self.digest, clothing.name, title="配色")
        self.assertEqual(renamed.name, clothing.name)
        self.assertEqual(renamed.title, "配色")
        self.assertEqual(
            [item.name for item in self.store.list_notes(self.digest)],
            [observation.name, clothing.name, composition.name],
        )
        self.assertEqual(
            [item.title for item in self.store.list_notes(self.digest)],
            ["观察", "配色", "构图"],
        )
        with self.assertRaises(NoteError):
            self.store.reorder_notes(self.digest, [observation.name, composition.name])
        with self.assertRaises(NoteError):
            self.store.reorder_notes(
                self.digest,
                [observation.name, clothing.name, composition.name, composition.name],
            )
        meta = json.loads((self.root / self.digest / ".artseek.json").read_text(encoding="utf-8"))
        self.assertEqual(
            meta["note_order"],
            [observation.name, clothing.name, composition.name],
        )

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
        first = self.store.create_note(
            self.digest,
            content=f"甲\n![]({shared})\n![]({owned})\n![](https://example.com/x.png)\n",
            point_id="point-1",
        )
        second = self.store.create_note(
            self.digest,
            content=f"乙\n![]({shared})\n",
            point_id="point-1",
        )
        removed = self.store.delete_note(self.digest, first.name)
        self.assertEqual(removed, [owned])
        self.assertFalse((self.root / self.digest / owned).exists())
        self.assertTrue((self.root / self.digest / shared).exists())
        self.assertTrue((self.root / self.digest / second.name).exists())

    def test_delete_folder_keeps_notes_until_explicit_and_survives_point_lookup(self) -> None:
        self.store.create_note(self.digest, content="构图", point_id="point-9")
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
            content="构图",
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
        self.assertEqual([item.name for item in listed], [created.name])
        self.assertEqual(listed[0].title, "构图")
        self.assertEqual(self.store.resolve_canonical(
            large_hash,
            phash=perceptual_hash(large),
            width=384,
            height=256,
        ), small_hash)
        self.assertTrue((self.root / small_hash / created.name).is_file())
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

    def test_runtime_cache_dir_wins_over_toml(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            data_dir = root / "data"
            cache_dir = root / "cache"
            runtime_cache = root / "runtime-cache"
            config_path = root / "config.toml"
            config_path.write_text(
                "\n".join(
                    [
                        f'data_dir = "{data_dir.as_posix()}"',
                        f'cache_dir = "{cache_dir.as_posix()}"',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            cfg = AppConfig(data_dir=data_dir, cache_dir=cache_dir)
            cfg.ensure_dirs()
            save_storage_paths(cfg, cache_dir=runtime_cache, update_cache=True)
            loaded = load_config(config_path)
            self.assertEqual(loaded.cache_dir.resolve(), runtime_cache.resolve())
            self.assertTrue(runtime_cache.is_dir())

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

    def test_migrate_index_sidecars_copies_cluster_not_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "data"
            dest = root / "index-b"
            source.mkdir()
            (source / "index_state.sqlite3").write_bytes(b"sqlite")
            (source / "index_state.sqlite3-wal").write_bytes(b"wal")
            (source / "folder_catalog.json").write_text("{}", encoding="utf-8")
            (source / "folder_scope.json").write_text("{}", encoding="utf-8")
            (source / "collections.json").write_text("[]", encoding="utf-8")
            (source / "manifest.json").write_text("{}", encoding="utf-8")
            (source / "runtime.json").write_text("{}", encoding="utf-8")
            (source / "qdrant").mkdir()
            (source / "qdrant" / "meta.json").write_text("{}", encoding="utf-8")
            path, copied = migrate_index_sidecars(source, dest)
            self.assertTrue(copied)
            self.assertEqual(path.resolve(), dest.resolve())
            self.assertTrue((dest / "index_state.sqlite3").is_file())
            self.assertTrue((source / "index_state.sqlite3-wal").is_file())
            self.assertTrue((dest / "folder_catalog.json").is_file())
            self.assertTrue((dest / "folder_scope.json").is_file())
            self.assertTrue((dest / "collections.json").is_file())
            self.assertTrue((dest / "manifest.json").is_file())
            self.assertFalse((dest / "runtime.json").exists())
            self.assertFalse((dest / "qdrant").exists())
            self.assertTrue((source / "index_state.sqlite3").is_file())
            with self.assertRaisesRegex(ValueError, "索引文件"):
                migrate_index_sidecars(source, dest)

    def test_runtime_index_dir_wins_over_default(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            data_dir = root / "data"
            cache_dir = root / "cache"
            index_dir = root / "index-b"
            config_path = root / "config.toml"
            config_path.write_text(
                "\n".join(
                    [
                        f'data_dir = "{data_dir.as_posix()}"',
                        f'cache_dir = "{cache_dir.as_posix()}"',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            cfg = AppConfig(data_dir=data_dir, cache_dir=cache_dir)
            cfg.ensure_dirs()
            save_storage_paths(cfg, index_dir=index_dir, update_index=True)
            loaded = load_config(config_path)
            self.assertEqual(loaded.index_dir.resolve(), index_dir.resolve())
            self.assertEqual(loaded.index_state_path, index_dir / "index_state.sqlite3")
            self.assertEqual(loaded.collections_path, index_dir / "collections.json")
            self.assertEqual(loaded.runtime_path, data_dir / "runtime.json")
