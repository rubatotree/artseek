from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from artseek.provenance.parser import ParsedSource, SourceRef, parse_filename
from artseek.provenance.store import ProvenanceStore, manual_author_key


def _parsed(value: str) -> ParsedSource:
    ref = parse_filename(value)
    assert ref is not None
    return ParsedSource(ref)


class ProvenanceStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.path = Path(self._temporary.name) / "provenance.sqlite3"
        self.store = ProvenanceStore(self.path)
        self.payload = {
            "path": "D:/pics/12345678_p0.png",
            "path_key": "d:/pics/12345678_p0.png",
            "width": 1200,
            "height": 1600,
            "mtime_ns": 1_000,
            "content_hash": "a" * 64,
        }

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def test_scan_inserts_pending_record_and_is_idempotent(self) -> None:
        first = self.store.upsert_scan("p1", self.payload, _parsed("12345678_p0.png"))
        second = self.store.upsert_scan("p1", self.payload, _parsed("12345678_p0.png"))
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        assert first is not None and second is not None
        self.assertEqual(first["platform"], "pixiv")
        self.assertEqual(first["status"], "pending")
        self.assertEqual(first["width"], 1200)
        self.assertEqual(second["source_id"], "12345678")

    def test_embedded_reference_replaces_filename_reference(self) -> None:
        self.store.upsert_scan("p1", self.payload, _parsed("12345678_p0.png"))
        embedded = ParsedSource(
            SourceRef(
                "twitter",
                "1872266114488828000",
                None,
                "https://x.com/i/status/1872266114488828000",
                origin="embedded",
                confidence=0.95,
            )
        )
        record = self.store.upsert_scan("p1", self.payload, embedded, origin="embedded")
        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(record["platform"], "twitter")
        self.assertEqual(record["status"], "pending")

    def test_manual_fields_survive_rescan_and_fetch(self) -> None:
        self.store.upsert_scan("p1", self.payload, _parsed("12345678_p0.png"))
        author_key = self.store.ensure_author("pixiv", "1554775", name="作者")
        updated = self.store.set_manual(
            "p1",
            {"title": "手动标题", "author_key": author_key, "tags": ["手动"]},
        )
        self.assertIsNotNone(updated)
        assert updated is not None
        self.assertEqual(updated["title"], "手动标题")

        self.store.set_fetch_result(
            "p1",
            {
                "platform": "pixiv",
                "source_id": "12345678",
                "page": 0,
                "url": "https://www.pixiv.net/artworks/12345678",
                "status": "ok",
                "title": "自动标题",
                "tags": ["自动"],
                "page_count": 2,
            },
            author_key=self.store.ensure_author("pixiv", "1", name="另一作者"),
        )
        record = self.store.get("p1")
        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(record["title"], "手动标题")
        self.assertEqual(record["author_key"], author_key)
        self.assertEqual(record["tags"], ["手动"])
        self.assertEqual(record["status"], "ok")
        self.assertEqual(record["page_count"], 2)

    def test_clear_manual_restores_automatic_fields(self) -> None:
        self.store.upsert_scan("p1", self.payload, _parsed("12345678_p0.png"))
        self.store.set_manual("p1", {"title": "手动", "tags": ["x"]})
        cleared = self.store.clear_manual("p1")
        self.assertIsNotNone(cleared)
        assert cleared is not None
        self.assertIsNone(cleared["title"])
        self.assertEqual(cleared["tags"], [])
        self.assertEqual(cleared["status"], "pending")

    def test_author_alias_group_and_point_filter(self) -> None:
        pixiv = self.store.ensure_author("pixiv", "1554775", name="米山舞")
        twitter = self.store.ensure_author("twitter", "yoneyamai", name="yoneyamai")
        canonical = self.store.link_authors(twitter, pixiv, evidence="pixiv_social")
        self.assertEqual(canonical, pixiv)
        self.assertEqual(self.store.canonical_author_key(twitter), pixiv)
        self.assertEqual(self.store.author_members(pixiv), sorted([pixiv, twitter]))
        groups = self.store.author_groups([pixiv])
        self.assertEqual(len(groups[pixiv]), 2)
        self.assertEqual(groups[pixiv], groups[twitter])

        self.store.upsert_scan("p1", self.payload, _parsed("12345678_p0.png"))
        self.store.set_fetch_result(
            "p1",
            {
                "platform": "pixiv",
                "source_id": "12345678",
                "page": 0,
                "url": "https://www.pixiv.net/artworks/12345678",
                "status": "ok",
            },
            author_key=canonical,
        )
        self.assertEqual(self.store.point_ids_for_author(twitter), ["p1"])
        summary = self.store.author_summary(twitter)
        self.assertIsNotNone(summary)
        assert summary is not None
        self.assertEqual(summary["key"], pixiv)
        self.assertEqual(summary["image_count"], 1)
        self.assertEqual(len(summary["members"]), 2)

    def test_tag_images_are_newest_first_by_file_mtime(self) -> None:
        older = dict(self.payload, mtime_ns=1_000)
        newer = dict(
            self.payload,
            path="D:/pics/50640084_p0.png",
            path_key="d:/pics/50640084_p0.png",
            mtime_ns=3_000,
            content_hash="b" * 64,
        )
        self.store.upsert_scan("p1", older, _parsed("12345678_p0.png"))
        self.store.upsert_scan("p2", newer, _parsed("50640084_p0.png"))
        for point_id in ("p1", "p2"):
            self.store.set_fetch_result(
                point_id,
                {
                    "platform": "pixiv",
                    "source_id": point_id,
                    "page": 0,
                    "url": f"https://www.pixiv.net/artworks/{point_id}",
                    "status": "ok",
                    "tags": ["long_hair"],
                },
            )
        self.assertEqual(self.store.point_ids_for_tags(["long_hair"]), ["p2", "p1"])

    def test_author_images_are_newest_first_by_file_mtime(self) -> None:
        author = self.store.ensure_author("pixiv", "1554775", name="作者")
        older = dict(self.payload, mtime_ns=1_000)
        newer = dict(
            self.payload,
            path="D:/pics/50640084_p0.png",
            path_key="d:/pics/50640084_p0.png",
            mtime_ns=3_000,
            content_hash="b" * 64,
        )
        self.store.upsert_scan("p1", older, _parsed("12345678_p0.png"))
        self.store.upsert_scan("p2", newer, _parsed("50640084_p0.png"))
        for point_id in ("p1", "p2"):
            self.store.set_fetch_result(
                point_id,
                {
                    "platform": "pixiv",
                    "source_id": point_id,
                    "page": 0,
                    "url": f"https://www.pixiv.net/artworks/{point_id}",
                    "status": "ok",
                },
                author_key=author,
            )
        self.assertEqual(self.store.point_ids_for_author(author), ["p2", "p1"])

    def test_author_search_is_fuzzy_and_ranked(self) -> None:
        key = self.store.ensure_author(
            "twitter",
            "yoneyamai",
            name="米山舞",
            handle="yoneyamai",
        )
        by_handle = self.store.list_authors("yoneyama")
        self.assertEqual(by_handle[0]["key"], key)
        self.assertGreater(by_handle[0]["score"], 0.4)
        by_name = self.store.list_authors("米山")
        self.assertEqual(by_name[0]["key"], key)
        self.assertGreaterEqual(by_name[0]["score"], 0.85)

    def test_manual_author_key_without_handle(self) -> None:
        key = manual_author_key("某画师")
        self.assertTrue(key.startswith("manual:"))
        self.assertEqual(key, manual_author_key("  某画师 "))
        self.assertEqual(manual_author_key("x", "@Handle"), "twitter:handle")

    def test_pending_records_and_status(self) -> None:
        self.store.upsert_scan("p1", self.payload, _parsed("12345678_p0.png"))
        self.store.upsert_scan(
            "p2",
            {**self.payload, "content_hash": "b" * 64},
            _parsed("1761931720818122000_1.jpg"),
        )
        self.assertEqual(len(self.store.pending_records(limit=1)), 1)
        self.store.set_status("p1", "ok")
        pending = self.store.pending_records(platforms=("pixiv",))
        self.assertEqual(pending, [])
        self.store.set_status("p1", "error", "boom")
        pending = self.store.pending_records(platforms=("pixiv",))
        self.assertEqual(pending[0]["error"], "boom")

    def test_pending_records_are_ordered_by_file_mtime_desc(self) -> None:
        self.store.upsert_scan(
            "old",
            {**self.payload, "mtime_ns": 100, "content_hash": "b" * 64},
            _parsed("12345678_p0.png"),
        )
        self.store.upsert_scan(
            "new",
            {**self.payload, "mtime_ns": 300, "content_hash": "c" * 64},
            _parsed("23456789_p1.jpg"),
        )
        self.assertEqual(
            [item["point_id"] for item in self.store.pending_records()],
            ["new", "old"],
        )

    def test_fetch_state_tracks_daily_counts_and_blocks(self) -> None:
        self.store.note_request("pixiv.net")
        self.store.note_request("pixiv.net")
        state = self.store.domain_state("pixiv.net")
        self.assertEqual(state["day_count"], 2)
        until = self.store.set_blocked("pixiv.net", 60, "429")
        self.assertGreaterEqual(until, int(__import__("time").time()))
        self.assertEqual(self.store.domain_state("pixiv.net")["last_error"], "429")
        self.store.set_long_remaining("saucenao.com", 0)
        self.assertEqual(self.store.domain_state("saucenao.com")["long_remaining"], 0)

    def test_legacy_twitter_bio_description_is_repaired(self) -> None:
        self.store.upsert_scan(
            "p1",
            self.payload,
            _parsed("1761931720818122000_1.jpg"),
        )
        bio = "author bio that was wrongly stored as tweet text"
        self.store.set_fetch_result(
            "p1",
            {
                "platform": "twitter",
                "source_id": "1761931720818122000",
                "page": 1,
                "url": "https://x.com/i/status/1761931720818122000",
                "status": "ok",
                "description": bio,
                "raw": {"author_description": bio},
            },
        )
        connection = sqlite3.connect(self.path)
        connection.execute("UPDATE meta SET value = '1' WHERE key = 'schema_version'")
        connection.commit()
        connection.close()
        reopened = ProvenanceStore(self.path)
        record = reopened.get("p1")
        self.assertIsNotNone(record)
        assert record is not None
        self.assertIsNone(record["description"])

    def test_upgrade_requeues_legacy_pixiv_not_found(self) -> None:
        self.store.upsert_scan("p1", self.payload, _parsed("12345678_p0.png"))
        self.store.set_status("p1", "not_found", "pages 404")
        connection = sqlite3.connect(self.path)
        connection.execute("UPDATE meta SET value = '3' WHERE key = 'schema_version'")
        connection.commit()
        connection.close()
        reopened = ProvenanceStore(self.path)
        record = reopened.get("p1")
        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(record["status"], "pending")
        self.assertIsNone(record["error"])

    def test_stale_detection_uses_content_hash(self) -> None:
        self.store.upsert_scan("p1", self.payload, _parsed("12345678_p0.png"))
        self.assertEqual(self.store.stale_points({"p1": "c" * 64}), ["p1"])
        self.assertEqual(self.store.get("p1")["status"], "stale")
        self.assertEqual(self.store.stale_points({"p1": "a" * 64}), [])

    def test_stats_reports_platforms_and_authors(self) -> None:
        self.store.upsert_scan("p1", self.payload, _parsed("12345678_p0.png"))
        self.store.upsert_scan(
            "p2",
            {**self.payload, "content_hash": "b" * 64},
            _parsed("1761931720818122000_1.jpg"),
        )
        self.store.set_status("p1", "ok")
        stats = self.store.stats()
        self.assertEqual(stats["total"], 2)
        self.assertEqual(stats["by_platform"]["pixiv"]["ok"], 1)
        self.assertEqual(stats["by_platform"]["twitter"]["pending"], 1)


if __name__ == "__main__":
    unittest.main()
