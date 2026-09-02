from __future__ import annotations

import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from artseek.config import AppConfig
from artseek.web.app import create_app


class _Store:
    def __init__(self, payloads: dict[str, dict[str, Any]]) -> None:
        self.payloads = payloads

    def get(self, point_id: str) -> dict[str, Any] | None:
        return self.payloads.get(point_id)


class _Engine:
    def __init__(self, payloads: dict[str, dict[str, Any]]) -> None:
        self.store = _Store(payloads)

    def close(self) -> None:
        return

    def hold_model(self):
        return nullcontext()


def _payload(name: str, digest: str, *, missing_file: bool = False) -> dict[str, Any]:
    return {
        "path": "Z:/missing/artseek.png" if missing_file else name,
        "name": name,
        "content_hash": digest,
        "width": 10,
        "height": 10,
        "tags": [],
        "caption": None,
    }


class NotesApiTests(unittest.TestCase):
    def setUp(self) -> None:
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            self.skipTest("fastapi 未安装")
        self._temporary_directory = tempfile.TemporaryDirectory()
        root = Path(self._temporary_directory.name)
        self.digest = "cd" * 32
        self.config = AppConfig(data_dir=root / "data", cache_dir=root / "cache")
        self.engine = _Engine({"img-1": _payload("girl.png", self.digest)})
        self.client = TestClient(create_app(self.config, engine=self.engine))

    def tearDown(self) -> None:
        self.client.close()
        self._temporary_directory.cleanup()

    def test_storage_paths_notes_apply_immediately(self) -> None:
        default = self.client.get("/api/storage-paths").json()
        self.assertEqual(Path(default["notes_dir"]), self.config.notes_dir)
        self.assertFalse(default["qdrant_restart_required"])
        moved = Path(self._temporary_directory.name) / "vault"
        saved = self.client.put("/api/storage-paths", json={"notes_dir": str(moved)}).json()
        self.assertEqual(Path(saved["notes_dir"]).resolve(), moved.resolve())
        self.assertTrue(moved.is_dir())
        qdrant = Path(self._temporary_directory.name) / "alt-qdrant"
        qdrant_saved = self.client.put(
            "/api/storage-paths", json={"qdrant_path": str(qdrant)}
        ).json()
        self.assertTrue(qdrant_saved["qdrant_restart_required"])
        self.assertEqual(Path(qdrant_saved["active_qdrant_path"]), self.config.data_dir / "qdrant")

    def test_storage_migrate_copies_notes_and_qdrant_marker(self) -> None:
        created = self.client.post(
            "/api/images/img-1/notes",
            json={"title": "构图", "content": "keep"},
        ).json()
        self.assertEqual(created["name"], "构图.md")
        original_notes = self.config.notes_dir
        moved_notes = Path(self._temporary_directory.name) / "vault-b"
        migrated = self.client.post(
            "/api/storage-paths/migrate",
            json={"kind": "notes", "destination": str(moved_notes)},
        ).json()
        self.assertTrue(migrated["copied"])
        self.assertEqual(Path(migrated["notes_dir"]).resolve(), moved_notes.resolve())
        self.assertTrue((moved_notes / self.digest / "构图.md").is_file())
        self.assertTrue((original_notes / self.digest / "构图.md").is_file())
        listed = self.client.get("/api/images/img-1/notes").json()
        self.assertEqual([item["name"] for item in listed["notes"]], ["构图.md"])
        marker = self.config.data_dir / "qdrant"
        marker.mkdir(parents=True, exist_ok=True)
        (marker / "meta.json").write_text("{}", encoding="utf-8")
        (marker / ".lock").write_text("pid", encoding="utf-8")
        moved_qdrant = Path(self._temporary_directory.name) / "qdrant-b"
        qdrant = self.client.post(
            "/api/storage-paths/migrate",
            json={"kind": "qdrant", "destination": str(moved_qdrant)},
        ).json()
        self.assertTrue(qdrant["copied"])
        self.assertEqual(Path(qdrant["qdrant_path"]).resolve(), moved_qdrant.resolve())
        self.assertEqual(Path(qdrant["active_qdrant_path"]).resolve(), moved_qdrant.resolve())
        self.assertFalse(qdrant["qdrant_restart_required"])
        self.assertTrue((moved_qdrant / "meta.json").is_file())
        self.assertFalse((moved_qdrant / ".lock").exists())
        same = self.client.post(
            "/api/storage-paths/migrate",
            json={"kind": "notes", "destination": str(moved_notes)},
        )
        self.assertEqual(same.status_code, 400)

    def test_note_crud_and_path_escape(self) -> None:
        listed = self.client.get("/api/images/img-1/notes").json()
        self.assertEqual(listed["image"]["content_hash"], self.digest)
        self.assertEqual(listed["notes"], [])
        created = self.client.post(
            "/api/images/img-1/notes",
            json={"title": "构图", "content": "# hi\n"},
        ).json()
        self.assertEqual(created["name"], "构图.md")
        self.assertEqual(created["content"], "# hi\n")
        read = self.client.get("/api/images/img-1/notes/构图.md").json()
        self.assertEqual(read["content"], "# hi\n")
        renamed = self.client.put(
            "/api/images/img-1/notes/构图.md",
            json={"title": "服装", "content": "袖"},
        ).json()
        self.assertEqual(renamed["name"], "服装.md")
        missing = self.client.get("/api/images/img-1/notes/构图.md")
        self.assertEqual(missing.status_code, 404)
        escaped = self.client.get("/api/note-files/" + self.digest + "/../secret.txt")
        self.assertEqual(escaped.status_code, 400)
        upload = self.client.post(
            "/api/images/img-1/assets",
            files={"file": ("shot.png", b"png-bytes", "image/png")},
        ).json()
        self.assertTrue(upload["path"].startswith("assets/"))
        fetched = self.client.get(upload["url"])
        self.assertEqual(fetched.status_code, 200)
        self.assertEqual(fetched.content, b"png-bytes")
        deleted = self.client.delete("/api/images/img-1/notes/服装.md").json()
        self.assertTrue(deleted["deleted"])
        self.assertEqual(deleted["removed_assets"], [])
        self.assertEqual(self.client.get("/api/images/img-1/notes").json()["notes"], [])

    def test_note_order_api_persists_complete_permutation(self) -> None:
        self.client.post("/api/images/img-1/notes", json={"title": "服装", "content": "a"})
        self.client.post("/api/images/img-1/notes", json={"title": "构图", "content": "b"})
        listed = self.client.get("/api/images/img-1/notes").json()["notes"]
        self.assertEqual([item["name"] for item in listed], ["服装.md", "构图.md"])
        reordered = self.client.put(
            "/api/images/img-1/note-order",
            json={"names": ["构图.md", "服装.md"]},
        )
        self.assertEqual(reordered.status_code, 200)
        self.assertEqual(
            [item["name"] for item in reordered.json()["notes"]],
            ["构图.md", "服装.md"],
        )
        again = self.client.get("/api/images/img-1/notes").json()["notes"]
        self.assertEqual([item["name"] for item in again], ["构图.md", "服装.md"])
        stale = self.client.put("/api/images/img-1/note-order", json={"names": ["构图.md"]})
        self.assertEqual(stale.status_code, 400)

    def test_saving_h1_first_line_renames_note(self) -> None:
        created = self.client.post(
            "/api/images/img-1/notes",
            json={"title": "笔记", "content": ""},
        ).json()
        self.assertEqual(created["name"], "笔记.md")
        saved = self.client.put(
            "/api/images/img-1/notes/笔记.md",
            json={"content": "# 构图\n说明\n"},
        ).json()
        self.assertEqual(saved["name"], "构图.md")
        self.assertEqual(saved["title"], "构图")

    def test_homepage_exposes_notes_and_storage_controls(self) -> None:
        html = self.client.get("/").text
        self.assertIn('id="viewer-notes"', html)
        self.assertIn('id="notes-pane"', html)
        self.assertIn('id="qdrant-path"', html)
        self.assertIn('id="notes-dir"', html)
        self.assertIn('id="qdrant-migrate"', html)
        self.assertIn('id="notes-migrate"', html)
        self.assertIn("/api/storage-paths", html)
        self.assertIn("/api/storage-paths/migrate", html)
        self.assertIn("/static/vendor/vditor", html)
        self.assertIn("function isNotesEditorEvent", html)
        self.assertIn('id="notes-resize"', html)
        self.assertIn('id="notes-delete-all"', html)
        self.assertIn("theme: 'classic'", html)
        self.assertIn("function handleNotesPaste", html)
        self.assertIn("function bindNotesResize", html)
        self.assertIn("function handleNotesWheel", html)
        self.assertIn("function nudgeNotesScroll", html)
        self.assertIn("function bindNoteTabs", html)
        self.assertIn("function listenNoteTabDrag", html)
        self.assertIn("function positionNoteTabGhost", html)
        self.assertIn("notes-tab-hole", html)
        self.assertNotIn("notes-tab-grip", html)
        self.assertIn("/note-order", html)
        self.assertIn(
            "await fetch(\n          `/api/images/${encodeURIComponent(notes.imageId)}/note-order`",
            html,
        )
        self.assertIn("scrollbar-width: thin", html)
        self.assertIn("scrollbar-width: none", html)
        self.assertIn("function tickNoteTabEdgeScroll", html)
        self.assertIn("function firstHeadingTitle", html)
        self.assertIn("function applySavedNote", html)
        self.assertIn("notes.saveLock", html)
        self.assertIn("pane.classList.add('is-open')", html)
        self.assertIn("overscroll-behavior: contain", html)
        self.assertIn('id="notes-edit-toggle"', html)
        self.assertIn("function setNotesEditing", html)
        self.assertIn("imageCaption: true", html)
        self.assertIn("function applyNoteLinkBase", html)
        self.assertIn("function applyNoteImageSrcs", html)
        self.assertIn("markdown: { linkBase: noteLinkBase(), imageCaption: true }", html)
        vditor_js = (
            Path(__file__).resolve().parents[1]
            / "src/artseek/web/static/vendor/vditor/dist/index.min.js"
        )
        if not vditor_js.is_file():
            self.skipTest("Vditor dist 未构建；请初始化子模块并在其中 pnpm build")
        vditor = self.client.get("/static/vendor/vditor/dist/index.min.js")
        self.assertEqual(vditor.status_code, 200)
        self.assertTrue(len(vditor.content) > 1000)

    def test_missing_image_is_404(self) -> None:
        response = self.client.get("/api/images/missing/notes")
        self.assertEqual(response.status_code, 404)

    def test_notes_survive_index_removal_and_folder_delete(self) -> None:
        self.client.post("/api/images/img-1/notes", json={"title": "构图", "content": "keep"})
        self.engine.store.payloads.pop("img-1")
        surviving = self.client.get("/api/images/img-1/notes").json()
        self.assertEqual(surviving["image"]["content_hash"], self.digest)
        self.assertEqual([item["name"] for item in surviving["notes"]], ["构图.md"])
        wiped = self.client.delete("/api/images/img-1/notes").json()
        self.assertTrue(wiped["deleted"])
        gone = self.client.get("/api/images/img-1/notes")
        self.assertEqual(gone.status_code, 404)
