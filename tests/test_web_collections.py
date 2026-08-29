from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from typing import Any
from unittest.mock import patch

from artseek.collections import CollectionStore
from artseek.config import AppConfig
from artseek.web.app import create_app


class _Store:
    def __init__(self, payloads: dict[str, dict[str, Any]]) -> None:
        self.payloads = payloads
        self.retrieve_requests: list[list[str]] = []
        self.raise_missing_collection = False
        self.count_calls = 0

    def get(self, point_id: str) -> dict[str, Any] | None:
        return self.payloads.get(point_id)

    def retrieve_payloads(self, point_ids: list[str]) -> dict[str, dict[str, Any]]:
        self.retrieve_requests.append(point_ids)
        if self.raise_missing_collection:
            raise ValueError("Collection artseek_images not found")
        return {
            point_id: self.payloads[point_id]
            for point_id in point_ids
            if point_id in self.payloads
        }

    def count(self) -> int:
        self.count_calls += 1
        return 0


class _Engine:
    def __init__(self, payloads: dict[str, dict[str, Any]]) -> None:
        self.store = _Store(payloads)
        self.embedder = object()
        self.llm = None
        self.replaced_catalog = None

    def close(self) -> None:
        return

    def hold_model(self):
        return nullcontext(self.embedder)

    def replace_folder_catalog(self, catalog: object) -> None:
        self.replaced_catalog = catalog


class _InlineThread:
    def __init__(self, *, target: Any, args: tuple[Any, ...], **_kwargs: Any) -> None:
        self._target = target
        self._args = args

    def start(self) -> None:
        self._target(*self._args)


class _PartiallyFailingIndexer:
    def __init__(self, _config: AppConfig, *, store: _Store, **_kwargs: Any) -> None:
        self.store = store
        self.folder_catalog = None

    def index(self, *_args: Any, **_kwargs: Any) -> None:
        self.store.payloads["partial"] = _payload("partial.png")
        raise RuntimeError("partial index failure")


def _payload(
    name: str,
    *,
    tags: list[str] | None = None,
    caption: str | None = None,
) -> dict[str, Any]:
    return {
        "path": f"D:/art/{name}",
        "name": name,
        "width": 640,
        "height": 480,
        "tags": tags or [],
        "caption": caption,
    }


class CollectionsApiTests(unittest.TestCase):
    def setUp(self) -> None:
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            self.skipTest("fastapi 未安装")
        self._temporary_directory = tempfile.TemporaryDirectory()
        root = Path(self._temporary_directory.name)
        self.config = AppConfig(data_dir=root / "data", cache_dir=root / "cache")
        self.config.tagger.enabled = False
        payloads = {
            "a": _payload("alpha.png", tags=["blue_hair"], caption="a bright portrait"),
            "b": _payload("beta.png", tags=["landscape"], caption="moonlit hill"),
            "c": _payload("gamma.png", tags=["city"], caption="blue hour city"),
        }
        self.engine = _Engine(payloads)
        self.client = TestClient(create_app(self.config, engine=self.engine))

    def tearDown(self) -> None:
        self.client.close()
        self._temporary_directory.cleanup()

    def create_collection(self, name: str = "灵感") -> dict[str, Any]:
        response = self.client.post("/api/collections", json={"name": name})
        self.assertEqual(response.status_code, 201)
        return response.json()["collection"]

    def test_create_rename_delete_and_list_preserve_order(self) -> None:
        blank = self.client.post("/api/collections", json={"name": ""})
        self.assertEqual(blank.status_code, 201)
        self.assertEqual(blank.json()["collection"]["name"], "")
        blank_collection = blank.json()["collection"]
        blank_rename = self.client.patch(
            f"/api/collections/{blank_collection['id']}", json={"name": ""}
        )
        self.assertEqual(blank_rename.status_code, 200)
        self.assertEqual(blank_rename.json()["collection"]["name"], "")
        first = self.create_collection("灵感")
        second = self.create_collection("待研究")
        listed = self.client.get("/api/collections").json()["collections"]
        self.assertEqual([item["name"] for item in listed], ["", "灵感", "待研究"])

        renamed = self.client.patch(
            f"/api/collections/{first['id']}", json={"name": "角色设计"}
        )
        self.assertEqual(renamed.status_code, 200)
        self.assertEqual(renamed.json()["collection"]["name"], "角色设计")
        self.assertEqual(renamed.json()["history"]["undo_label"], "重命名「灵感」为「角色设计」")
        deleted = self.client.delete(f"/api/collections/{second['id']}")
        self.assertEqual(deleted.status_code, 200)
        self.assertTrue(deleted.json()["deleted"])
        listed_after_delete = self.client.get("/api/collections").json()
        self.assertEqual(len(listed_after_delete["collections"]), 2)
        self.assertTrue(listed_after_delete["history"]["can_undo"])
        self.assertEqual(listed_after_delete["history"]["undo_label"], "删除「待研究」")

        empty_redo = self.client.post("/api/collections/redo")
        self.assertEqual(empty_redo.status_code, 409)
        undone = self.client.post("/api/collections/undo")
        self.assertEqual(undone.status_code, 200)
        self.assertEqual(undone.json()["applied"]["kind"], "delete_collection")
        self.assertEqual(
            [item["id"] for item in undone.json()["collections"]],
            [blank_collection["id"], first["id"], second["id"]],
        )
        redone = self.client.post("/api/collections/redo")
        self.assertEqual(redone.status_code, 200)
        self.assertEqual(len(redone.json()["collections"]), 2)
        self.assertTrue(redone.json()["history"]["can_undo"])
        self.assertEqual(self.client.post("/api/collections/redo").status_code, 409)

    def test_collection_reordering_replaces_pinning_and_rejects_stale_order(self) -> None:
        first = self.create_collection("先创建")
        second = self.create_collection("后创建")
        third = self.create_collection("第三个")

        reordered = self.client.put(
            "/api/collections/order",
            json={"collection_ids": [third["id"], first["id"], second["id"]]},
        )
        self.assertEqual(reordered.status_code, 200)
        self.assertEqual(
            [item["id"] for item in reordered.json()["collections"]],
            [third["id"], first["id"], second["id"]],
        )
        self.assertEqual(reordered.json()["history"]["undo_label"], "调整收藏夹顺序")
        listed = self.client.get("/api/collections").json()["collections"]
        self.assertEqual([item["id"] for item in listed], [third["id"], first["id"], second["id"]])
        stale_order = self.client.put(
            "/api/collections/order", json={"collection_ids": [first["id"], second["id"]]}
        )
        self.assertEqual(stale_order.status_code, 409)
        empty_update = self.client.patch(f"/api/collections/{second['id']}", json={})
        self.assertEqual(empty_update.status_code, 422)
        pinned_update = self.client.patch(
            f"/api/collections/{second['id']}", json={"name": "不再置顶", "pinned": True}
        )
        self.assertEqual(pinned_update.status_code, 422)

    def test_item_membership_changes_expose_undo_and_redo(self) -> None:
        self.assertEqual(self.client.post("/api/collections/undo").status_code, 409)
        collection = self.create_collection()
        endpoint = f"/api/collections/{collection['id']}/items"
        added = self.client.post(endpoint, json={"image_id": "a"}).json()
        self.assertTrue(added["added"])
        self.assertEqual(added["history"]["undo_label"], "收藏「alpha.png」到「灵感」")
        listed = self.client.get("/api/collections").json()
        self.assertEqual(listed["history"]["undo_label"], "收藏「alpha.png」到「灵感」")

        undone_add = self.client.post("/api/collections/undo")
        self.assertEqual(undone_add.status_code, 200)
        self.assertEqual(undone_add.json()["applied"]["kind"], "add_item")
        self.assertEqual(undone_add.json()["collection"]["count"], 0)
        redone_add = self.client.post("/api/collections/redo")
        self.assertEqual(redone_add.json()["collection"]["count"], 1)

        removed = self.client.delete(f"{endpoint}/a")
        self.assertTrue(removed.json()["removed"])
        self.assertEqual(removed.json()["history"]["undo_label"], "从「灵感」移出「alpha.png」")
        restored = self.client.post("/api/collections/undo")
        self.assertEqual(restored.status_code, 200)
        self.assertEqual(restored.json()["applied"]["kind"], "remove_item")
        self.assertEqual(restored.json()["collection"]["count"], 1)
        self.assertEqual(
            [item["id"] for item in self.client.get(
                f"/api/collections/{collection['id']}/images"
            ).json()["results"]],
            ["a"],
        )

    def test_items_are_validated_idempotent_and_return_ordered_filtered_pages(self) -> None:
        collection = self.create_collection()
        endpoint = f"/api/collections/{collection['id']}/items"
        for point_id in ("c", "a", "b"):
            response = self.client.post(endpoint, json={"image_id": point_id})
            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.json()["added"])
        repeated = self.client.post(endpoint, json={"image_id": "a"})
        self.assertEqual(repeated.status_code, 200)
        self.assertFalse(repeated.json()["added"])
        self.assertEqual(repeated.json()["collection"]["count"], 3)

        missing = self.client.post(endpoint, json={"image_id": "missing"})
        self.assertEqual(missing.status_code, 404)
        self.assertIn("不在索引", missing.json()["detail"])

        # A stale point can remain on disk after an index lifecycle change. It
        # must not be returned or silently removed from the collection file.
        CollectionStore(self.config.collections_path).add_item(collection["id"], "missing")
        first_page = self.client.get(
            f"/api/collections/{collection['id']}/images", params={"limit": 2}
        )
        self.assertEqual(first_page.status_code, 200)
        data = first_page.json()
        self.assertEqual([item["id"] for item in data["results"]], ["c", "a"])
        self.assertEqual(data["total"], 3)
        self.assertTrue(data["has_more"])
        self.assertEqual(
            set(data["results"][0]),
            {"id", "score", "path", "name", "width", "height", "tags", "tag_hits", "caption"},
        )
        second_page = self.client.get(
            f"/api/collections/{collection['id']}/images", params={"limit": 2, "offset": 2}
        )
        self.assertEqual([item["id"] for item in second_page.json()["results"]], ["b"])

        query = self.client.get(
            f"/api/collections/{collection['id']}/images", params={"query": "BLUE"}
        ).json()
        self.assertEqual([item["id"] for item in query["results"]], ["c", "a"])

        removed = self.client.delete(f"{endpoint}/a")
        self.assertTrue(removed.json()["removed"])
        repeated_remove = self.client.delete(f"{endpoint}/a")
        self.assertFalse(repeated_remove.json()["removed"])

    def test_homepage_exposes_collection_controls(self) -> None:
        html = self.client.get("/").text
        self.assertIn('viewport-fit=cover', html)
        self.assertIn('inline-size: 100dvw; block-size: 100dvh;', html)
        self.assertIn('id="tab-btn-collections"', html)
        self.assertIn(".tree-panel:not(.is-open) .tree-body { display: block; }", html)
        self.assertIn("window.matchMedia('(max-width: 720px)').matches", html)
        self.assertIn('id="collection-search"', html)
        self.assertIn('id="collection-undo"', html)
        self.assertIn('id="collection-redo"', html)
        self.assertIn("/api/collections/undo", html)
        self.assertIn("/api/collections/redo", html)
        self.assertIn("function undoCollectionHistory", html)
        self.assertIn("function redoCollectionHistory", html)
        self.assertIn("function clearActiveCollectionView", html)
        self.assertIn("function handleCollectionHistoryShortcut", html)
        self.assertIn("function isCollectionNameField", html)
        self.assertIn(
            "document.addEventListener('keydown', handleCollectionHistoryShortcut, true)",
            html,
        )
        self.assertIn("isCollectionUndoShortcut(event)", html)
        self.assertIn("isCollectionRedoShortcut(event)", html)
        self.assertIn("appendUndoHint", html)
        self.assertIn("applyCollectionHistory(data.history)", html)
        self.assertNotIn("function isEditableKeyTarget", html)
        self.assertIn('aria-labelledby="tab-btn-collections"', html)
        self.assertNotIn("pinnedCollections()", html)
        self.assertIn('id="ref-drop-zone"', html)
        self.assertNotIn('id="drag-dock"', html)
        self.assertIn('id="drop-mask"', html)
        self.assertIn('function isFileDrag(event)', html)
        self.assertIn('function clipboardImageFiles(clipboardData)', html)
        self.assertIn('function dedupeClipboardImages(fileList)', html)
        self.assertIn('function handleSearchPaste(event)', html)
        self.assertIn("document.addEventListener('dragover'", html)
        self.assertIn("document.addEventListener('drop'", html)
        self.assertIn("document.addEventListener('paste'", html)
        self.assertIn("addReferenceFiles(event.dataTransfer.files)", html)
        self.assertIn("clipboardImageFiles(event.clipboardData)", html)
        self.assertIn("if (fromFiles.length) return dedupeClipboardImages(fromFiles)", html)
        self.assertIn("addReferenceFiles(files)", html)
        self.assertIn('placeholder="搜索图片，或粘贴参考图..."', html)
        self.assertIn('title="可输入描述，或直接粘贴参考图"', html)
        self.assertIn("event.key === 'Home'", html)
        self.assertIn('/api/collections', html)
        self.assertNotIn('data-collection-action="pin"', html)
        self.assertIn('data-collection-tile', html)
        self.assertIn("function startCollectionReorder", html)
        self.assertIn("function moveCollectionWithKeyboard", html)
        self.assertIn("/api/collections/order", html)
        self.assertIn(
            "const COLLECTIONS_PER_PAGE = COLLECTION_GRID_COLUMNS * COLLECTION_GRID_ROWS;",
            html,
        )
        self.assertIn("grid-template-rows: repeat(6, 44px);", html)
        self.assertIn("grid-template-rows: repeat(6, 44px);", html)
        self.assertIn("function collectionPaginationMarkup", html)
        self.assertIn("data-collection-page", html)
        self.assertIn("function moveCollectionToIndex", html)
        self.assertIn("拖到分页按钮可跨页", html)
        self.assertIn('id="collection-create-tool"', html)
        self.assertIn('id="collection-rename-tool"', html)
        self.assertIn('id="collection-reorder-tool"', html)
        self.assertIn('id="collection-delete-tool"', html)
        self.assertIn("collectionEditMode: ''", html)
        self.assertIn("function setCollectionEditMode(mode)", html)
        self.assertIn("async function createCollectionFromTool()", html)
        self.assertIn("state.collectionEditMode = 'rename';", html)
        self.assertIn("state.collectionEditMode === 'reorder'", html)
        self.assertIn("deleteCollection(collection, { exitMode: true })", html)
        self.assertIn("input.addEventListener('blur'", html)
        self.assertIn("event.button !== 2", html)
        self.assertIn("document.addEventListener('contextmenu'", html)
        self.assertIn("function exitCollectionEditMode()", html)
        self.assertIn(
            ".collection-list.is-editing.is-editing-visible .collection-row.collection-tile",
            html,
        )
        self.assertIn("transform: translateY(-2px) scale(1.01);", html)
        self.assertIn("function queueCollectionListAfterEditExit()", html)
        self.assertIn("}, 190);", html)
        self.assertIn("pendingCreatedCollectionId", html)
        self.assertIn("?discard=true", html)
        self.assertNotIn('>保存</button>', html)
        self.assertIn("applied.kind !== 'rename_collection'", html)
        self.assertIn('M12 3v18M3 12h18', html)
        self.assertIn('M4 7h16M10 11v6M14 11v6', html)
        self.assertIn("${collectionPaginationMarkup('sidebar')}", html)
        self.assertIn(
            ".collection-name {\n      position: absolute; inset: 0;"
            " display: grid; place-items: center;",
            html,
        )
        self.assertIn(
            "position: absolute; right: 6px; bottom: 5px; color: var(--muted); font-size: 9px;",
            html,
        )
        self.assertIn(
            'id="collection-rename-tool" type="button" aria-pressed="false"'
            ' aria-label="重命名收藏夹"',
            html,
        )
        self.assertIn('>↶</button>', html)
        self.assertIn('>↷</button>', html)

        card_start = html.index("function cardHTML")
        card_end = html.index("function renderGallery", card_start)
        card_source = html[card_start:card_end]
        self.assertNotIn('class="card-actions"', card_source)
        self.assertNotIn("collectionPicker", card_source)
        self.assertNotIn("card.addEventListener('drag'", card_source)
        self.assertIn('role="button"', card_source)
        self.assertIn('data-card-action="favorite"', card_source)
        self.assertIn('class="card-title-row"', card_source)
        self.assertIn('class="card-title-favorite"', card_source)
        self.assertIn('aria-controls="card-collection-menu"', card_source)
        self.assertIn('>♡</button>', card_source)
        self.assertIn('data-gallery-drag', card_source)
        self.assertIn('draggable="true"', card_source)
        self.assertIn('draggable="false"', card_source)
        self.assertIn("imageWrap?.addEventListener('dragstart'", html)
        self.assertIn('function bindGalleryReferenceDropTarget()', html)
        self.assertIn("event.dataTransfer.setData(GALLERY_ITEM_MIME, item.id)", html)
        self.assertNotIn("card.addEventListener('pointermove'", html)
        self.assertIn('function startCardMenuGesture(item, trigger, event)', html)
        self.assertIn('function updateCardMenuQuickTarget(clientX, clientY)', html)
        self.assertIn('function updateCardMenuGesture(event)', html)
        self.assertIn('function finishCardMenuGesture(event', html)
        self.assertIn('is-quick-action-target', html)
        self.assertIn('CARD_MENU_DRAG_THRESHOLD_TOUCH = 18', html)
        self.assertIn('const wasDragging = cardMenuGesture.dragging;', html)
        self.assertIn("pageHoverLock: ''", html)
        self.assertIn("cardMenuGesture.pageHoverLock === direction", html)
        self.assertIn("drag.pageHoverLock === direction", html)
        self.assertIn("collectionEditModeCancelUntil", html)
        self.assertIn("cancelCollectionReorder();", html)
        self.assertIn('data-card-menu-page', html)
        self.assertIn("favoriteButton?.addEventListener('pointermove'", html)
        self.assertIn('data-card-action="remove"', card_source)
        self.assertIn("action === 'favorite'", html)
        self.assertIn("action === 'remove'", html)

        self.assertIn('id="viewer-collection-menu"', html)
        self.assertIn('aria-controls="viewer-collection-menu"', html)
        self.assertIn('aria-haspopup="dialog"', html)
        self.assertIn('role="dialog" aria-labelledby="viewer-collection-menu-title"', html)
        self.assertNotIn('role="menu"', html)
        self.assertNotIn('role="menuitemcheckbox"', html)
        self.assertNotIn('id="viewer-collection-picker"', html)
        self.assertIn("function closeViewerCollectionMenu", html)
        self.assertIn("function focusViewerCollectionDialog()", html)
        self.assertIn("function captureViewerCollectionDialogFocus()", html)
        self.assertIn("function restoreViewerCollectionDialogFocus", html)
        self.assertIn("const pickerId = `${namespace}-picker-create`;", html)
        self.assertIn('data-${namespace}-menu-create', html)
        self.assertIn("requestAnimationFrame(focusViewerCollectionDialog);", html)
        self.assertIn("const focusSnapshot = captureViewerCollectionDialogFocus();", html)
        self.assertIn("target.value = focusSnapshot.value;", html)
        self.assertIn("target.setSelectionRange(", html)
        self.assertIn(
            "await refreshMenu(item, { preserveCreateDraft: false });", html
        )
        self.assertIn("renderCollectionMenu(item, 'viewer', refreshViewerCollectionMenu);", html)
        self.assertIn('data-${namespace}-menu-reference', html)
        self.assertIn('id="card-collection-menu"', html)
        self.assertIn(
            'function openCardCollectionMenu(item, trigger, { toggle = true } = {})', html
        )
        self.assertIn('function refreshCardCollectionMenu(item', html)
        self.assertIn('function positionCardCollectionMenu()', html)
        self.assertIn('positionCardCollectionMenu();', html)
        self.assertIn('openCardCollectionMenu(item, actionTarget);', html)
        self.assertIn("position: fixed; z-index: 1300", html)
        self.assertIn("--collection-picker-height: min(474px", html)
        self.assertIn('class="collection-picker-tile-label"', html)
        self.assertIn(
            ".collection-picker-tile-label {\n      position: absolute; inset: 0;"
            " display: grid; place-items: center;",
            html,
        )
        self.assertIn(
            "position: absolute; right: 6px; bottom: 5px; color: #bdb4c1; font-size: 9px;",
            html,
        )
        self.assertIn("if (!$('viewer-collection-menu').hidden) {", html)
        self.assertIn("window.addEventListener('orientationchange', sync);", html)
        self.assertIn(
            "if (!$('viewer-collection-menu').hidden) positionViewerCollectionMenu();", html
        )
        self.assertIn('collectionToggle.hidden = viewingRef || !state.current?.id;', html)
        viewer_menu_start = html.index("function openViewerCollectionMenu")
        viewer_menu_end = html.index("function renderCollectionList", viewer_menu_start)
        viewer_menu_source = html[viewer_menu_start:viewer_menu_end]
        self.assertNotIn("state.mode === 'collection'", viewer_menu_source)
        self.assertIn('function advanceViewerAfterCollectionRemoval', html)
        self.assertIn('advanceViewerAfterCollectionRemoval(preferredId', html)

        gallery_state_start = html.index("function applyGalleryViewerState")
        gallery_state_end = html.index("function applyReferenceViewerState", gallery_state_start)
        gallery_state_source = html[gallery_state_start:gallery_state_end]
        self.assertLess(
            gallery_state_source.index("closeViewerCollectionMenu({ restoreFocus: false });"),
            gallery_state_source.index("state.current = item;"),
        )
        viewer_step_start = html.index("async function stepViewer")
        viewer_step_end = html.index("function toggleViewerZoom", viewer_step_start)
        viewer_step_source = html[viewer_step_start:viewer_step_end]
        self.assertLess(
            viewer_step_source.index("closeViewerCollectionMenu({ restoreFocus: false });"),
            viewer_step_source.index("if (view.flipping) return false;"),
        )

        clear_start = html.index("function clearActiveCollectionView")
        clear_end = html.index("async function refreshOpenCollectionMenus", clear_start)
        clear_source = html[clear_start:clear_end]
        self.assertIn("state.searchSeq += 1;", clear_source)
        self.assertIn("$('load-more').hidden = true;", clear_source)
        delete_start = html.index("async function deleteCollection")
        delete_end = html.index("async function fetchCollectionImages", delete_start)
        delete_source = html[delete_start:delete_end]
        self.assertLess(
            delete_source.index("await parseResponse"),
            delete_source.index("clearActiveCollectionView"),
        )
        self.assertIn(
            "const activeCollectionWasDeleted = state.collectionId === collection.id;",
            delete_source,
        )
        self.assertLess(
            delete_source.index("if (activeCollectionWasDeleted)"),
            delete_source.index("} catch (error)"),
        )
        self.assertIn("await fetchCollectionImages({ reset: true });", delete_source)
        self.assertIn("$('panel').classList.remove('loading');", delete_source)
        self.assertIn("border-radius: 15px 15px 0 0", html)
        self.assertIn(".image-wrap { border-radius: 11px 11px 0 0; }", html)
        self.assertIn(
            ".filename { flex: 1 1 auto; min-width: 0; max-width: 100%;"
            " overflow: hidden; white-space: nowrap; text-overflow: ellipsis;",
            html,
        )

    def test_missing_qdrant_collection_returns_an_empty_page_without_pruning_membership(
        self,
    ) -> None:
        collection = self.create_collection()
        endpoint = f"/api/collections/{collection['id']}/items"
        self.assertEqual(self.client.post(endpoint, json={"image_id": "a"}).status_code, 200)
        self.engine.store.raise_missing_collection = True

        response = self.client.get(f"/api/collections/{collection['id']}/images")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["results"], [])
        persisted = CollectionStore(self.config.collections_path).get(collection["id"])
        self.assertEqual(persisted.item_ids if persisted else (), ("a",))

    def test_unfiltered_pages_use_bounded_batches_without_retaining_payloads(self) -> None:
        collection = self.create_collection("大收藏夹")
        point_ids = [f"large-{index}" for index in range(600)]
        self.engine.store.payloads.update(
            {point_id: _payload(f"{point_id}.png") for point_id in point_ids}
        )
        self.config.collections_path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "collections": [
                        {"id": collection["id"], "name": "大收藏夹", "item_ids": point_ids}
                    ],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        self.engine.store.retrieve_requests.clear()

        first = self.client.get(
            f"/api/collections/{collection['id']}/images", params={"limit": 40}
        )
        self.assertEqual(first.status_code, 200)
        self.assertEqual(len(first.json()["results"]), 40)
        self.assertIsNone(first.json()["total"])
        self.assertFalse(first.json()["total_exact"])
        self.assertEqual(len(self.engine.store.retrieve_requests), 1)
        self.assertLessEqual(len(self.engine.store.retrieve_requests[0]), 256)
        self.assertEqual(self.engine.store.count_calls, 0)

        second = self.client.get(
            f"/api/collections/{collection['id']}/images",
            params={"limit": 40, "offset": 40},
        )
        self.assertEqual(second.status_code, 200)
        self.assertEqual(len(second.json()["results"]), 40)
        self.assertEqual(len(self.engine.store.retrieve_requests), 2)
        self.assertEqual(len(self.engine.store.retrieve_requests[1]), 40)
        self.assertTrue(
            all(len(request) <= 256 for request in self.engine.store.retrieve_requests)
        )
        cursors = self.client.app.state.collection_id_cursors
        cursor = cursors[collection["id"]]
        self.assertLessEqual(len(cursors), 4)
        self.assertLessEqual(len(cursor.live_item_ids), 10_240)
        self.assertFalse(hasattr(cursor, "payloads"))

        too_deep = self.client.get(
            f"/api/collections/{collection['id']}/images", params={"offset": 10_001}
        )
        self.assertEqual(too_deep.status_code, 422)

    def test_id_cursor_lru_and_membership_invalidation(self) -> None:
        collections = [self.create_collection(f"收藏夹 {index}") for index in range(5)]
        for collection in collections:
            endpoint = f"/api/collections/{collection['id']}/items"
            self.assertEqual(self.client.post(endpoint, json={"image_id": "a"}).status_code, 200)
            response = self.client.get(f"/api/collections/{collection['id']}/images")
            self.assertEqual(response.status_code, 200)

        cursors = self.client.app.state.collection_id_cursors
        self.assertLessEqual(len(cursors), 4)
        self.assertNotIn(collections[0]["id"], cursors)
        active = collections[-1]
        self.assertIn(active["id"], cursors)

        endpoint = f"/api/collections/{active['id']}/items"
        self.assertTrue(self.client.post(endpoint, json={"image_id": "b"}).json()["added"])
        self.assertNotIn(active["id"], self.client.app.state.collection_id_cursors)

    def test_deep_offset_with_only_stale_ids_has_no_false_tail(self) -> None:
        collection = self.create_collection("陈旧记录")
        stale_ids = [f"stale-{index}" for index in range(600)]
        self.config.collections_path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "collections": [
                        {"id": collection["id"], "name": "陈旧记录", "item_ids": stale_ids}
                    ],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        self.engine.store.retrieve_requests.clear()

        response = self.client.get(
            f"/api/collections/{collection['id']}/images",
            params={"limit": 40, "offset": 500},
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["results"], [])
        self.assertEqual(data["total"], 0)
        self.assertFalse(data["has_more"])
        self.assertTrue(
            all(len(request) <= 256 for request in self.engine.store.retrieve_requests)
        )

    def test_failed_index_resets_empty_collection_state_after_partial_write(self) -> None:
        self.client.app.state.collection_index_empty = True
        with (
            patch("artseek.web.app.ImageIndexer", _PartiallyFailingIndexer),
            patch("artseek.web.app.threading.Thread", _InlineThread),
        ):
            response = self.client.post(
                "/api/index", json={"path": self._temporary_directory.name}
            )

        self.assertEqual(response.status_code, 200)
        self.assertFalse(self.client.app.state.collection_index_empty)
        self.assertIn("partial", self.engine.store.payloads)
        self.assertEqual(self.client.get("/api/index-status").json()["message"], "索引失败")


if __name__ == "__main__":
    unittest.main()
