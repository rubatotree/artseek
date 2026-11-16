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
        self.assertEqual(created["title"], "keep")
        note_name = created["name"]
        original_notes = self.config.notes_dir
        moved_notes = Path(self._temporary_directory.name) / "vault-b"
        migrated = self.client.post(
            "/api/storage-paths/migrate",
            json={"kind": "notes", "destination": str(moved_notes)},
        ).json()
        self.assertTrue(migrated["copied"])
        self.assertEqual(Path(migrated["notes_dir"]).resolve(), moved_notes.resolve())
        self.assertTrue((moved_notes / self.digest / note_name).is_file())
        self.assertTrue((original_notes / self.digest / note_name).is_file())
        listed = self.client.get("/api/images/img-1/notes").json()
        self.assertEqual([item["name"] for item in listed["notes"]], [note_name])
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

    def test_storage_migrate_copies_cache_thumbnails(self) -> None:
        original = self.config.cache_dir
        original.mkdir(parents=True, exist_ok=True)
        (original / "thumbnails").mkdir(exist_ok=True)
        marker = original / "thumbnails" / "img-1.webp"
        marker.write_bytes(b"webp")
        moved = Path(self._temporary_directory.name) / "cache-b"
        migrated = self.client.post(
            "/api/storage-paths/migrate",
            json={"kind": "cache", "destination": str(moved)},
        ).json()
        self.assertTrue(migrated["copied"])
        self.assertEqual(Path(migrated["cache_dir"]).resolve(), moved.resolve())
        self.assertTrue((moved / "thumbnails" / "img-1.webp").is_file())
        self.assertTrue(marker.is_file())
        saved = self.client.put("/api/storage-paths", json={"cache_dir": str(moved)}).json()
        self.assertEqual(Path(saved["cache_dir"]).resolve(), moved.resolve())

    def test_storage_migrate_copies_index_sidecars(self) -> None:
        source = self.config.index_dir
        source.mkdir(parents=True, exist_ok=True)
        (source / "index_state.sqlite3").write_bytes(b"sqlite")
        (source / "folder_catalog.json").write_text('{"folders":[]}', encoding="utf-8")
        (source / "folder_scope.json").write_text("{}", encoding="utf-8")
        (source / "collections.json").write_text("[]", encoding="utf-8")
        (source / "manifest.json").write_text("{}", encoding="utf-8")
        (source / "runtime.json").write_text("{}", encoding="utf-8")
        moved = Path(self._temporary_directory.name) / "index-b"
        migrated = self.client.post(
            "/api/storage-paths/migrate",
            json={"kind": "index", "destination": str(moved)},
        ).json()
        self.assertTrue(migrated["copied"])
        self.assertEqual(Path(migrated["index_dir"]).resolve(), moved.resolve())
        self.assertTrue((moved / "index_state.sqlite3").is_file())
        self.assertTrue((moved / "folder_catalog.json").is_file())
        self.assertTrue((moved / "folder_scope.json").is_file())
        self.assertTrue((moved / "collections.json").is_file())
        self.assertTrue((moved / "manifest.json").is_file())
        self.assertFalse((moved / "runtime.json").exists())
        listed = self.client.get("/api/collections").json()
        self.assertIn("collections", listed)

    def test_note_crud_and_path_escape(self) -> None:
        listed = self.client.get("/api/images/img-1/notes").json()
        self.assertEqual(listed["image"]["content_hash"], self.digest)
        self.assertEqual(listed["notes"], [])
        created = self.client.post(
            "/api/images/img-1/notes",
            json={"title": "构图", "content": "# hi\n"},
        ).json()
        note_name = created["name"]
        self.assertRegex(note_name, r"^\d{8}T\d{6}(?:-\d+)?\.md$")
        self.assertEqual(created["title"], "hi")
        self.assertEqual(created["content"], "# hi\n")
        read = self.client.get(f"/api/images/img-1/notes/{note_name}").json()
        self.assertEqual(read["content"], "# hi\n")
        renamed = self.client.put(
            f"/api/images/img-1/notes/{note_name}",
            json={"title": "服装"},
        ).json()
        self.assertEqual(renamed["name"], note_name)
        self.assertEqual(renamed["title"], "服装")
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
        deleted = self.client.delete(f"/api/images/img-1/notes/{note_name}").json()
        self.assertTrue(deleted["deleted"])
        self.assertEqual(deleted["removed_assets"], [])
        self.assertEqual(self.client.get("/api/images/img-1/notes").json()["notes"], [])

    def test_note_order_api_persists_complete_permutation(self) -> None:
        clothing = self.client.post(
            "/api/images/img-1/notes", json={"title": "服装"}
        ).json()
        composition = self.client.post(
            "/api/images/img-1/notes", json={"title": "构图"}
        ).json()
        listed = self.client.get("/api/images/img-1/notes").json()["notes"]
        self.assertEqual(
            [item["name"] for item in listed],
            [clothing["name"], composition["name"]],
        )
        self.assertEqual([item["title"] for item in listed], ["服装", "构图"])
        reordered = self.client.put(
            "/api/images/img-1/note-order",
            json={"names": [composition["name"], clothing["name"]]},
        )
        self.assertEqual(reordered.status_code, 200)
        self.assertEqual(
            [item["name"] for item in reordered.json()["notes"]],
            [composition["name"], clothing["name"]],
        )
        again = self.client.get("/api/images/img-1/notes").json()["notes"]
        self.assertEqual(
            [item["name"] for item in again],
            [composition["name"], clothing["name"]],
        )
        stale = self.client.put(
            "/api/images/img-1/note-order", json={"names": [composition["name"]]}
        )
        self.assertEqual(stale.status_code, 400)

    def test_saving_first_line_updates_title_not_filename(self) -> None:
        created = self.client.post(
            "/api/images/img-1/notes",
            json={"content": ""},
        ).json()
        self.assertRegex(created["name"], r"^\d{8}T\d{6}(?:-\d+)?\.md$")
        self.assertEqual(created["title"], "笔记")
        saved = self.client.put(
            f"/api/images/img-1/notes/{created['name']}",
            json={"content": "# 构图\n说明\n"},
        ).json()
        self.assertEqual(saved["name"], created["name"])
        self.assertEqual(saved["title"], "构图")
        plain = self.client.put(
            f"/api/images/img-1/notes/{created['name']}",
            json={"content": "服装层次\n说明\n"},
        ).json()
        self.assertEqual(plain["name"], created["name"])
        self.assertEqual(plain["title"], "服装层次")

    def test_homepage_exposes_notes_and_storage_controls(self) -> None:
        html = self.client.get("/").text
        self.assertIn('id="viewer-notes"', html)
        self.assertIn('id="notes-pane"', html)
        self.assertIn('id="qdrant-path"', html)
        self.assertIn('id="notes-dir"', html)
        self.assertIn('id="qdrant-migrate"', html)
        self.assertIn('id="notes-migrate"', html)
        self.assertIn('id="cache-dir"', html)
        self.assertIn('id="cache-migrate"', html)
        self.assertIn('id="index-dir"', html)
        self.assertIn('id="index-migrate"', html)
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
        self.assertIn("function firstLineTitle", html)
        self.assertIn("function applyFirstLineTitleToTab", html)
        self.assertIn("JSON.stringify({ content: '' })", html)
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
        self.assertNotIn('id="agent-system-prompt"', html)
        self.assertNotIn('id="agent-first-prompt"', html)
        self.assertNotIn('id="notes-agent"', html)
        self.assertNotIn("function sendNoteAgent", html)
        self.assertNotIn("/api/agents", html)
        self.assertNotIn("/api/agent-settings", html)
        self.assertNotIn("/agent/ws", html)
        self.assertIn("function parseAppRoute", html)
        self.assertIn("function commitRoute", html)
        self.assertIn("function applyAppRoute", html)
        self.assertIn("function handleAppLink", html)
        self.assertIn("function dismissViewer", html)
        self.assertIn("function dismissNotes", html)
        self.assertIn("history.pushState", html)
        self.assertIn("popstate", html)
        self.assertIn(
            "`/i/${encodeURIComponent(route.imageId || '')}${route.gallery || ''}`",
            html,
        )
        self.assertIn("params.set('q', route.query)", html)
        self.assertIn("params.set('ref', ref)", html)
        self.assertIn("params.set('s', route.snapshot)", html)
        self.assertIn("function parseSearchRefs", html)
        self.assertIn("function restoreSearchSnapshot", html)
        self.assertIn("/api/search-snapshots/", html)

    def test_spa_view_routes_serve_the_app_shell(self) -> None:
        for path in (
            "/search",
            "/folder",
            "/random",
            "/c/col-1",
            "/i/img-1",
            "/i/img-1/notes",
            "/i/img-1/notes/20260903T010203.md",
        ):
            with self.subTest(path=path):
                response = self.client.get(path)
                self.assertEqual(response.status_code, 200)
                self.assertIn("ArtSeek", response.text)
                self.assertIn("function parseAppRoute", response.text)
        image = self.client.get("/image/img-1")
        self.assertEqual(image.status_code, 404)
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
        created = self.client.post(
            "/api/images/img-1/notes", json={"content": "构图"}
        ).json()
        self.engine.store.payloads.pop("img-1")
        surviving = self.client.get("/api/images/img-1/notes").json()
        self.assertEqual(surviving["image"]["content_hash"], self.digest)
        self.assertEqual([item["name"] for item in surviving["notes"]], [created["name"]])
        self.assertEqual(surviving["notes"][0]["title"], "构图")
        wiped = self.client.delete("/api/images/img-1/notes").json()
        self.assertTrue(wiped["deleted"])
        gone = self.client.get("/api/images/img-1/notes")
        self.assertEqual(gone.status_code, 404)
