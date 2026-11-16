from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import httpx

from artseek.provenance.fetch import (
    BudgetExhausted,
    FxTwitterClient,
    NotFoundError,
    PixivClient,
    ProvenanceService,
    RateLimiter,
)
from artseek.provenance.parser import SourceRef, parse_filename
from artseek.provenance.store import ProvenanceStore


class FakeResponse:
    pass


class PixivClientTests(unittest.TestCase):
    def test_fetch_reads_page_metadata_and_author(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/ajax/illust/12345678":
                return httpx.Response(
                    200,
                    json={
                        "error": False,
                        "body": {
                            "illustId": "12345678",
                            "illustTitle": "初音ミク",
                            "description": "示例说明",
                            "userId": "1554775",
                            "userName": "米山舞",
                            "userAccount": "yoneyamai",
                            "pageCount": 2,
                            "width": 1200,
                            "height": 1600,
                            "createDate": "2015-05-30T15:31:00+00:00",
                            "tags": {"tags": [{"tag": "オリジナル"}, {"tag": "廃墟"}]},
                        },
                    },
                )
            if request.url.path == "/ajax/illust/12345678/pages":
                return httpx.Response(
                    200,
                    json={
                        "error": False,
                        "body": [
                            {"width": 1200, "height": 1600},
                            {"width": 200, "height": 300},
                        ],
                    },
                )
            if request.url.path == "/ajax/user/1554775":
                return httpx.Response(
                    200,
                    json={
                        "error": False,
                        "body": {
                            "name": "米山舞",
                            "pixivId": "yoneyamai",
                            "imageBig": "https://example.test/avatar.jpg",
                            "social": {"twitter": {"url": "https://twitter.com/yoneyamai"}},
                        },
                    },
                )
            return httpx.Response(404)

        client = httpx.Client(transport=httpx.MockTransport(handler))
        pixiv = PixivClient(client)
        ref = SourceRef("pixiv", "12345678", 1, "https://www.pixiv.net/artworks/12345678")
        result = pixiv.fetch(ref, local_size=(200, 300))
        self.assertEqual(result.page, 1)
        self.assertEqual(result.title, "初音ミク")
        self.assertEqual(result.author_id, "1554775")
        self.assertEqual(result.tags, ["オリジナル", "廃墟"])
        self.assertEqual(result.page_count, 2)
        self.assertEqual(result.width, 200)
        self.assertEqual(result.confidence, 0.9)
        profile = pixiv.fetch_author("1554775")
        self.assertEqual(profile["social"]["twitter"]["url"], "https://twitter.com/yoneyamai")

    def test_restricted_pages_404_keeps_illustration_metadata(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/ajax/illust/678901234":
                return httpx.Response(
                    200,
                    json={
                        "error": False,
                        "body": {
                            "illustId": "678901234",
                            "illustTitle": "meta",
                            "userId": "1",
                            "userName": "author",
                            "pageCount": 2,
                            "width": 2480,
                            "height": 3894,
                            "xRestrict": 0,
                            "tags": {"tags": [{"tag": "SFW"}]},
                        },
                    },
                )
            return httpx.Response(404, json={"error": True, "body": []})

        client = httpx.Client(transport=httpx.MockTransport(handler))
        result = PixivClient(client).fetch(
            SourceRef("pixiv", "678901234", 1, "https://www.pixiv.net/artworks/678901234"),
            local_size=(2480, 3894),
        )
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.title, "meta")
        self.assertEqual(result.page_count, 2)
        self.assertEqual(result.tags, ["SFW"])
        self.assertTrue(result.raw["pages_unavailable"])
        self.assertLessEqual(result.confidence, 0.6)

    def test_missing_work_is_not_found(self) -> None:
        client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(404)))
        with self.assertRaises(NotFoundError):
            PixivClient(client).fetch(SourceRef("pixiv", "1", 0, "https://pixiv.net/artworks/1"))


class FxTwitterClientTests(unittest.TestCase):
    def test_empty_tweet_text_does_not_fall_back_to_author_bio(self) -> None:
        tweet = {
            "tweet": {
                "id": "1761931720818122000",
                "text": "",
                "author": {
                    "id": "1",
                    "screen_name": "artist",
                    "name": "Artist",
                    "description": "author bio should not become the title",
                    "url": "https://x.com/artist",
                },
                "media": {"all": [{"type": "photo", "width": 300, "height": 400}]},
            }
        }
        client = httpx.Client(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json=tweet))
        )
        result = FxTwitterClient(client).fetch(
            SourceRef("twitter", "1761931720818122000", 1, "https://x.com/i/status/1761931720818122000")
        )
        self.assertIsNone(result.description)
        self.assertIsNone(result.title)
        self.assertEqual(
            result.raw["author_description"],
            "author bio should not become the title",
        )

    def test_multi_photo_page_is_matched_by_dimensions(self) -> None:
        tweet = {
            "tweet": {
                "id": "1761931720818122000",
                "text": "hello #tag",
                "created_at": "Thu Sep 21 08:34:50 +0000 2023",
                "author": {
                    "id": "1",
                    "screen_name": "artist",
                    "name": "Artist",
                    "description": "see https://www.pixiv.net/users/1554775",
                    "url": "https://x.com/artist",
                    "avatar_url": "https://example.test/avatar.jpg",
                },
                "media": {
                    "all": [
                        {"type": "photo", "width": 100, "height": 200},
                        {"type": "photo", "width": 300, "height": 400},
                    ]
                },
                "possibly_sensitive": False,
            }
        }
        client = httpx.Client(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json=tweet))
        )
        fx = FxTwitterClient(client)
        ref = SourceRef(
            "twitter", "1761931720818122000", None, "https://x.com/i/status/1761931720818122000"
        )
        result = fx.fetch(ref, local_size=(300, 400))
        self.assertEqual(result.page, 2)
        self.assertEqual(result.page_count, 2)
        self.assertEqual(result.tags, ["tag"])
        self.assertEqual(result.author_social["pixiv_user_id"], "1554775")
        self.assertEqual(result.author_handle, "artist")


class ServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.store = ProvenanceStore(Path(self._temporary.name) / "provenance.sqlite3")
        self.payload = {
            "path": "D:/pics/12345678_p1.png",
            "path_key": "d:/pics/12345678_p1.png",
            "width": 200,
            "height": 300,
            "content_hash": "a" * 64,
        }
        ref = parse_filename("12345678_p1.png")
        assert ref is not None
        from artseek.provenance.parser import ParsedSource

        self.store.upsert_scan("p1", self.payload, ParsedSource(ref))

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def test_enrichment_stores_metadata_and_links_authors(self) -> None:
        class FakePixiv:
            def fetch(self, ref: SourceRef, *, local_size=None):
                from artseek.provenance.fetch import FetchResult

                return FetchResult(
                    platform="pixiv",
                    source_id=ref.source_id,
                    page=1,
                    url=ref.url,
                    title="初音ミク",
                    description="desc",
                    author_platform="pixiv",
                    author_id="1554775",
                    author_name="米山舞",
                    author_handle="yoneyamai",
                    author_profile_url="https://www.pixiv.net/users/1554775",
                    page_count=2,
                    width=200,
                    height=300,
                    tags=["オリジナル"],
                    confidence=1.0,
                )

            def fetch_author(self, author_id: str):
                return {
                    "name": "米山舞",
                    "handle": "yoneyamai",
                    "profile_url": "https://www.pixiv.net/users/1554775",
                    "avatar_url": None,
                    "social": {"twitter": {"url": "https://twitter.com/yoneyamai"}},
                }

            def close(self):
                pass

        pixiv = FakePixiv()
        service = ProvenanceService(
            self.store,
            pixiv=pixiv,  # type: ignore[arg-type]
            twitter=FakeTwitter(),  # type: ignore[arg-type]
            pixiv_limiter=RateLimiter(self.store, "pixiv.net", min_interval=0, daily_limit=10),
            twitter_limiter=RateLimiter(
                self.store, "api.fxtwitter.com", min_interval=0, daily_limit=10
            ),
        )
        service.enrich_record(self.store.get("p1"))  # type: ignore[arg-type]
        record = self.store.get("p1")
        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(record["status"], "ok")
        self.assertEqual(record["title"], "初音ミク")
        self.assertEqual(record["page"], 1)
        summary = self.store.author_summary("pixiv:1554775")
        self.assertIsNotNone(summary)
        assert summary is not None
        self.assertEqual(summary["image_count"], 1)
        members = {item["key"] for item in summary["members"]}
        self.assertIn("twitter:yoneyamai", members)


class FakeTwitter:
    def fetch(self, ref: SourceRef, *, local_size=None):
        raise AssertionError("twitter should not be used")

    def close(self):
        pass


class RateLimiterTests(unittest.TestCase):
    def test_daily_budget_stops_before_request(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = ProvenanceStore(Path(temporary) / "provenance.sqlite3")
            limiter = RateLimiter(store, "pixiv.net", min_interval=0, daily_limit=1)
            limiter.record()
            with self.assertRaises(BudgetExhausted):
                limiter.wait()

    def test_zero_daily_limit_is_unlimited(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = ProvenanceStore(Path(temporary) / "provenance.sqlite3")
            limiter = RateLimiter(store, "pixiv.net", min_interval=0, daily_limit=0)
            for _ in range(5):
                limiter.wait()
                limiter.record()
            snapshot = limiter.snapshot()
            self.assertIsNone(snapshot.daily_limit)
            self.assertEqual(snapshot.used_today, 5)

    def test_blocked_domain_waits_until_clock_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = ProvenanceStore(Path(temporary) / "provenance.sqlite3")
            now = [float(__import__("time").time())]
            limiter = RateLimiter(
                store,
                "pixiv.net",
                min_interval=0,
                daily_limit=10,
                clock=lambda: now[0],
                sleep=lambda seconds: now.__setitem__(0, now[0] + seconds),
            )
            store.set_blocked("pixiv.net", 5, "429")
            limiter.wait()
            self.assertGreaterEqual(now[0], float(__import__("time").time()) + 4)


if __name__ == "__main__":
    unittest.main()
