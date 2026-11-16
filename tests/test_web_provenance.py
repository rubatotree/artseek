from __future__ import annotations

import json
import tempfile
import time
import unittest
from contextlib import nullcontext
from pathlib import Path
from typing import Any
from unittest.mock import patch

from PIL import Image

from artseek.config import AppConfig
from artseek.image_io import canonical_path
from artseek.provenance.fetch import FetchResult, NotFoundError
from artseek.provenance.parser import ParsedSource, parse_filename
from artseek.search import SearchResponse
from artseek.web.app import create_app


class _Store:
    def __init__(self, payloads: dict[str, dict[str, Any]]) -> None:
        self.payloads = payloads

    def get(self, point_id: str) -> dict[str, Any] | None:
        return self.payloads.get(point_id)

    def retrieve_payloads(self, point_ids: list[str]) -> dict[str, dict[str, Any]]:
        return {
            point_id: self.payloads[point_id]
            for point_id in point_ids
            if point_id in self.payloads
        }

    def iter_payloads(self):
        yield from self.payloads.items()

    def count(self) -> int:
        return len(self.payloads)


class _Engine:
    def __init__(self, payloads: dict[str, dict[str, Any]]) -> None:
        self.store = _Store(payloads)
        self.search_calls: list[dict[str, Any]] = []

    def close(self) -> None:
        return

    def hold_model(self):
        return nullcontext()

    def search(self, text: str = "", **kwargs: Any) -> SearchResponse:
        self.search_calls.append({"text": text, **kwargs})
        return SearchResponse(query_plan=None, results=[], has_more=False)


class _FolderAcc:
    name = "收藏"
    path = "D:/art/收藏"


class _Catalog:
    def __init__(self, point_ids: list[str]) -> None:
        self._ids = point_ids
        self.folders = {canonical_path("d:/art/收藏"): _FolderAcc()}

    def point_ids(self, selected, *, after_ns=None, before_ns=None):
        del selected, after_ns, before_ns
        return list(self._ids)

    def list_folder_images(self, key: str, method: str):
        return [(point_id, f"{point_id}.png", 0) for point_id in self._ids]

    def all_keys(self):
        return list(self.folders)


class _FolderEngine(_Engine):
    def folder_catalog(self):
        return _Catalog(list(self.store.payloads))


class ProvenanceApiTests(unittest.TestCase):
    def setUp(self) -> None:
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            self.skipTest("fastapi 未安装")
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.image = self.root / "12345678_p0.png"
        Image.new("RGB", (1200, 1600), (30, 40, 50)).save(self.image)
        self.payload = {
            "path": str(self.image),
            "path_key": str(self.image).casefold(),
            "name": self.image.name,
            "width": 1200,
            "height": 1600,
            "content_hash": "a" * 64,
            "tags": [],
        }
        self.engine = _FolderEngine({"p1": self.payload})
        self.config = AppConfig(data_dir=self.root / "data", cache_dir=self.root / "cache")
        self.config.tagger.enabled = False
        self.client = TestClient(create_app(self.config, engine=self.engine))
        self.store = self.client.app.state.provenance

    def tearDown(self) -> None:
        self.client.close()
        self._temporary.cleanup()

    def test_viewer_payload_parses_filename_on_demand(self) -> None:
        response = self.client.get("/api/images/p1/provenance")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["record"]["platform"], "pixiv")
        self.assertEqual(data["record"]["source_id"], "12345678")
        self.assertEqual(data["record"]["status"], "pending")
        self.assertEqual(data["page_label"], "p0")
        self.assertEqual(data["reverse_search"]["saucenao"], "https://saucenao.com/")

    def test_unresolved_record_is_reparsed_from_exif_on_view(self) -> None:
        path = self.root / "GnLt47zaMAAjCZD.jpg"
        exif = Image.Exif()
        exif[40092] = (
            "https://x.com/example_artist2/status/1905879515315732000/photo/1"
        ).encode("utf-16-le")
        Image.new("RGB", (8, 8), (20, 30, 40)).save(path, exif=exif, quality=90)
        payload = {
            **self.payload,
            "path": str(path),
            "path_key": str(path).casefold(),
            "name": path.name,
            "width": 8,
            "height": 8,
        }
        self.engine.store.payloads["p1"] = payload
        self.store.upsert_unresolved("p1", payload)
        response = self.client.get("/api/images/p1/provenance")
        self.assertEqual(response.status_code, 200)
        record = response.json()["record"]
        self.assertEqual(record["platform"], "twitter")
        self.assertEqual(record["source_id"], "1905879515315732000")

    def test_search_image_endpoint_returns_bounded_jpeg(self) -> None:
        response = self.client.get("/search-image/p1")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "image/jpeg")
        with Image.open(__import__("io").BytesIO(response.content)) as image:
            self.assertEqual(image.format, "JPEG")
            self.assertLessEqual(max(image.size), 1200)

    def test_manual_fields_and_author_filter_are_persisted(self) -> None:
        put = self.client.put(
            "/api/images/p1/provenance",
            json={"title": "手动标题", "author_name": "某画师", "tags": ["tag1"]},
        )
        self.assertEqual(put.status_code, 200)
        record = put.json()["record"]
        self.assertEqual(record["title"], "手动标题")
        self.assertIn("title", record["manual_fields"])
        author_key = record["author_key"]
        self.assertTrue(author_key.startswith("manual:"))

        listed = self.client.get("/api/authors", params={"q": "某画师"})
        self.assertEqual(listed.status_code, 200)
        self.assertEqual(listed.json()["authors"][0]["key"], author_key)

        folder = self.client.get(
            "/api/folder-images",
            params={"folder_key": "d:/art/收藏", "author_key": author_key},
        )
        self.assertEqual(folder.status_code, 200)
        self.assertEqual([item["id"] for item in folder.json()["results"]], ["p1"])
        source = folder.json()["results"][0]["source"]
        self.assertEqual(source["title"], "手动标题")
        self.assertEqual(source["author_name"], "某画师")
        self.assertEqual(source["author_key"], author_key)

        search = self.client.post(
            "/api/search",
            json={"query": "test", "author_key": author_key, "folder_keys": []},
        )
        self.assertEqual(search.status_code, 200)
        self.assertEqual(self.engine.search_calls[-1]["restrict_ids"], ["p1"])

    @patch("artseek.web.app.PixivClient.fetch_author")
    @patch("artseek.web.app.PixivClient.fetch")
    def test_manual_url_fetches_metadata(self, fetch, fetch_author) -> None:
        fetch.return_value = FetchResult(
            platform="pixiv",
            source_id="12345678",
            page=0,
            url="https://www.pixiv.net/artworks/12345678",
            title="初音ミク",
            author_platform="pixiv",
            author_id="1554775",
            author_name="米山舞",
            author_handle="yoneyamai",
            author_profile_url="https://www.pixiv.net/users/1554775",
            page_count=1,
            width=1200,
            height=1600,
            tags=["オリジナル"],
        )
        fetch_author.return_value = {
            "name": "米山舞",
            "handle": "yoneyamai",
            "profile_url": "https://www.pixiv.net/users/1554775",
            "avatar_url": None,
            "social": {"twitter": {"url": "https://twitter.com/yoneyamai"}},
        }
        response = self.client.put(
            "/api/images/p1/provenance",
            json={"url": "https://www.pixiv.net/artworks/12345678"},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["record"]["title"], "初音ミク")
        self.assertEqual(data["record"]["author_key"], "pixiv:1554775")
        self.assertEqual(data["author"]["image_count"], 1)

    def test_search_filters_parse_author_and_tag_tokens(self) -> None:
        parsed = parse_filename("12345678_p0.png")
        assert parsed is not None
        self.store.upsert_scan("p1", self.payload, ParsedSource(parsed))
        author_key = self.store.ensure_author("pixiv", "1554775", name="米山舞")
        self.store.set_fetch_result(
            "p1",
            {
                "platform": "pixiv",
                "source_id": "12345678",
                "page": 0,
                "url": "https://www.pixiv.net/artworks/12345678",
                "status": "ok",
                "title": "初音ミク",
                "tags": ["オリジナル"],
            },
            author_key=author_key,
        )
        filtered = self.client.post(
            "/api/search",
            json={"query": "author:pixiv:1554775", "folder_keys": []},
        )
        self.assertEqual(filtered.status_code, 200)
        self.assertEqual([item["id"] for item in filtered.json()["results"]], ["p1"])
        self.assertEqual(filtered.json()["applied_filters"]["author_key"], author_key)

        tagged = self.client.post(
            "/api/search",
            json={"query": "tag:オリジナル", "folder_keys": []},
        )
        self.assertEqual(tagged.status_code, 200)
        self.assertEqual(
            self.engine.search_calls[-1]["restrict_ids"], ["p1"]
        )
        self.assertEqual(
            self.engine.search_calls[-1]["text"], "オリジナル"
        )

        missing = self.client.post(
            "/api/search",
            json={"query": "author:没有这个人", "folder_keys": []},
        )
        self.assertEqual(missing.status_code, 400)

    def test_enrich_from_empty_sidecar_starts_a_scan(self) -> None:
        with patch(
            "artseek.web.app.PixivClient.fetch",
            side_effect=NotFoundError("smoke"),
        ):
            response = self.client.post(
                "/api/provenance/enrich",
                json={"platforms": ["pixiv", "twitter"]},
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["kind"], "scan")
            self.client.post("/api/provenance/job/stop")
            deadline = time.time() + 5
            while time.time() < deadline:
                status = self.client.get("/api/provenance/job").json()
                if not status["running"]:
                    break
                time.sleep(0.02)
        self.assertFalse(status["running"])

    def test_scope_and_folder_random_sort_are_seeded(self) -> None:
        folder_key = "d:/art/收藏"
        scope = self.client.get(
            "/api/scope-images",
            params={"folder_keys": json.dumps([folder_key]), "sort": "random", "seed": 7},
        )
        self.assertEqual(scope.status_code, 200)
        self.assertEqual(scope.json()["seed"], 7)
        self.assertEqual([item["id"] for item in scope.json()["results"]], ["p1"])
        repeat = self.client.get(
            "/api/scope-images",
            params={"folder_keys": json.dumps([folder_key]), "sort": "random", "seed": 7},
        )
        self.assertEqual(
            [item["id"] for item in repeat.json()["results"]],
            [item["id"] for item in scope.json()["results"]],
        )
        folder = self.client.get(
            "/api/folder-images",
            params={"folder_key": folder_key, "sort": "random", "seed": 9},
        )
        self.assertEqual(folder.status_code, 200)
        self.assertEqual(folder.json()["seed"], 9)
        self.assertEqual([item["id"] for item in folder.json()["results"]], ["p1"])

    def test_scan_job_updates_records(self) -> None:
        response = self.client.post(
            "/api/provenance/scan",
            json={"deep": False, "only_new": False, "auto_enrich": False},
        )
        self.assertEqual(response.status_code, 200)
        deadline = time.time() + 5
        while time.time() < deadline:
            status = self.client.get("/api/provenance/job").json()
            if not status["running"]:
                break
            time.sleep(0.02)
        self.assertFalse(status["running"])
        self.assertEqual(status["kind"], "scan")
        self.assertEqual(self.store.get("p1")["platform"], "pixiv")
        stats = self.client.get("/api/provenance/stats").json()
        self.assertGreaterEqual(stats["total"], 1)


if __name__ == "__main__":
    unittest.main()
