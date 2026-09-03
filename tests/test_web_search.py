from __future__ import annotations

import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from artseek.config import AppConfig
from artseek.query import QueryPlan
from artseek.search import SearchResponse, SearchResult
from artseek.search_cache import is_snapshot_id
from artseek.web.app import create_app


class _Store:
    def __init__(self, payloads: dict[str, dict[str, Any]]) -> None:
        self.payloads = payloads

    def get(self, point_id: str) -> dict[str, Any] | None:
        return self.payloads.get(point_id)


class _Engine:
    def __init__(self, payloads: dict[str, dict[str, Any]], rows: list[SearchResult]) -> None:
        self.store = _Store(payloads)
        self.rows = rows
        self.calls: list[dict[str, Any]] = []

    def search(self, text: str = "", **kwargs: Any) -> SearchResponse:
        self.calls.append({"text": text, **kwargs})
        offset = int(kwargs.get("offset") or 0)
        limit = int(kwargs.get("limit") or 40)
        page = self.rows[offset : offset + limit]
        return SearchResponse(
            query_plan=QueryPlan(original=text, semantic_zh=text, positive_tags=["white_socks"]),
            results=page,
            has_more=offset + limit < len(self.rows),
            offset=offset,
        )

    def close(self) -> None:
        return

    def hold_model(self):
        return nullcontext()


def _row(point_id: str, name: str) -> SearchResult:
    return SearchResult(
        id=point_id,
        score=90.0,
        path=name,
        name=name,
        width=16,
        height=12,
        tags=["white_socks"],
        tag_hits=["white_socks"],
        caption=None,
    )


class SearchSnapshotApiTests(unittest.TestCase):
    def setUp(self) -> None:
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            self.skipTest("fastapi 未安装")
        self._temporary_directory = tempfile.TemporaryDirectory()
        root = Path(self._temporary_directory.name)
        payloads = {
            "img-1": {
                "path": "ref.png",
                "name": "ref.png",
                "width": 16,
                "height": 12,
                "tags": [],
            },
            "hit-1": {
                "path": "a.png",
                "name": "a.png",
                "width": 16,
                "height": 12,
                "tags": ["white_socks"],
            },
            "hit-2": {
                "path": "b.png",
                "name": "b.png",
                "width": 16,
                "height": 12,
                "tags": [],
            },
        }
        self.engine = _Engine(payloads, [_row("hit-1", "a.png"), _row("hit-2", "b.png")])
        self.config = AppConfig(data_dir=root / "data", cache_dir=root / "cache")
        self.client = TestClient(create_app(self.config, engine=self.engine))

    def tearDown(self) -> None:
        self.client.close()
        self._temporary_directory.cleanup()

    def test_search_writes_snapshot_and_get_restores_without_research(self) -> None:
        first = self.client.post(
            "/api/search",
            json={
                "query": "白袜",
                "expand": False,
                "folder_keys": [],
                "references": [{"id": "img-1", "weight": 1.5}],
                "limit": 1,
            },
        )
        self.assertEqual(first.status_code, 200)
        data = first.json()
        snapshot_id = data["snapshot_id"]
        self.assertTrue(is_snapshot_id(snapshot_id))
        self.assertEqual([item["id"] for item in data["results"]], ["hit-1"])
        self.assertEqual(len(self.engine.calls), 1)

        restored = self.client.get(f"/api/search-snapshots/{snapshot_id}")
        self.assertEqual(restored.status_code, 200)
        body = restored.json()
        self.assertEqual(body["query"], "白袜")
        self.assertEqual(body["references"], [{"id": "img-1", "weight": 1.5}])
        self.assertEqual(body["results"][0]["id"], "hit-1")
        self.assertEqual(body["results"][0]["name"], "a.png")
        self.assertEqual(len(self.engine.calls), 1)

        more = self.client.post(
            "/api/search",
            json={
                "query": "白袜",
                "expand": False,
                "folder_keys": [],
                "references": [{"id": "img-1", "weight": 1.5}],
                "limit": 1,
                "offset": 1,
                "snapshot_id": snapshot_id,
                "exclude_ids": ["hit-1"],
            },
        )
        self.assertEqual(more.status_code, 200)
        self.assertEqual(more.json()["snapshot_id"], snapshot_id)
        appended = self.client.get(f"/api/search-snapshots/{snapshot_id}").json()
        self.assertEqual([item["id"] for item in appended["results"]], ["hit-1", "hit-2"])

    def test_missing_snapshot_is_404(self) -> None:
        response = self.client.get("/api/search-snapshots/" + "ab" * 6)
        self.assertEqual(response.status_code, 404)
        bad = self.client.get("/api/search-snapshots/nope")
        self.assertEqual(bad.status_code, 400)
