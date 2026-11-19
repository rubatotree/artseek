from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from artseek.provenance.parser import (
    ParsedSource,
    SourceRef,
    parse_filename,
    parse_xiaohongshu_filename,
)
from artseek.provenance.store import ProvenanceStore


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

    def test_auto_tag_queue_ignores_existing_source_or_manual_tags(self) -> None:
        self.store.upsert_scan("p1", self.payload, _parsed("12345678_p0.png"))
        self.store.set_fetch_result(
            "p1",
            {
                "platform": "pixiv",
                "source_id": "12345678",
                "page": 0,
                "url": "https://www.pixiv.net/artworks/12345678",
                "status": "ok",
                "tags": ["来源标签"],
            },
        )
        self.store.set_manual("p1", {"manual_tags": ["手动标签"]})
        self.assertEqual(self.store.point_ids_without_auto_tags(), ["p1"])
        self.store.set_auto_tags("p1", ["自动标签"], {"自动标签": 0.88})
        self.assertEqual(self.store.point_ids_without_auto_tags(), [])

    def test_set_auto_tags_many_updates_multiple_records(self) -> None:
        self.store.upsert_scan("p1", self.payload, _parsed("12345678_p0.png"))
        second = {
            **self.payload,
            "path": "D:/pics/50640084_p0.png",
            "path_key": "d:/pics/50640084_p0.png",
        }
        self.store.upsert_scan("p2", second, _parsed("50640084_p0.png"))
        self.store.set_auto_tags_many(
            [
                ("p1", ["自动标签"], {"自动标签": 0.88}),
                ("p2", [], {}),
            ]
        )
        self.assertEqual(self.store.point_ids_without_auto_tags(), [])
        first = self.store.get("p1")
        second_record = self.store.get("p2")
        assert first is not None and second_record is not None
        self.assertEqual(first["auto_tags"], ["自动标签"])
        self.assertEqual(first["auto_tag_scores"]["自动标签"], 0.88)
        self.assertEqual(second_record["auto_tags"], [])

    def test_manual_and_auto_tags_are_searchable(self) -> None:
        self.store.upsert_scan("p1", self.payload, _parsed("12345678_p0.png"))
        self.store.set_manual("p1", {"manual_tags": ["手动标签"]})
        self.store.set_auto_tags("p1", ["自动标签"], {"自动标签": 0.88})
        record = self.store.get("p1")
        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(record["manual_tags"], ["手动标签"])
        self.assertEqual(record["auto_tags"], ["自动标签"])
        self.assertEqual(record["auto_tag_scores"]["自动标签"], 0.88)
        self.assertEqual(self.store.point_ids_for_tags(["手动标签"]), ["p1"])
        self.assertEqual(self.store.point_ids_for_tags(["自动标签"]), ["p1"])

    def test_clear_manual_restores_automatic_fields(self) -> None:
        self.store.upsert_scan("p1", self.payload, _parsed("12345678_p0.png"))
        self.store.set_manual(
            "p1", {"title": "手动", "tags": ["来源"], "manual_tags": ["x"]}
        )
        cleared = self.store.clear_manual("p1")
        self.assertIsNotNone(cleared)
        assert cleared is not None
        self.assertIsNone(cleared["title"])
        self.assertEqual(cleared["tags"], ["来源"])
        self.assertEqual(cleared["manual_tags"], [])
        self.assertEqual(cleared["status"], "pending")

    def test_author_alias_group_and_point_filter(self) -> None:
        pixiv = self.store.ensure_author("pixiv", "1554775", name="米山舞")
        twitter = self.store.ensure_author("twitter", "yoneyamai", name="yoneyamai")
        canonical = self.store.link_authors(twitter, pixiv, evidence="pixiv_social")
        self.assertEqual(canonical, pixiv)
        self.assertEqual(self.store.canonical_author_key(twitter), pixiv)
        self.assertEqual(
            self.store.author_members(pixiv),
            ["handle:yoneyamai", "pixiv_id:1554775"],
        )
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

    def test_conflicting_identity_merges_artists_and_keeps_images(self) -> None:
        pixiv = self.store.ensure_author("pixiv", "100", name="Pixiv Name")
        twitter = self.store.ensure_author(
            "twitter", "200", name="X Name", handle="shared_handle"
        )
        self.assertNotEqual(pixiv, twitter)
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
            author_key=pixiv,
        )
        self.assertEqual(self.store.author_summary(pixiv)["image_count"], 1)
        merged = self.store.ensure_author("pixiv", "100", handle="shared_handle")
        self.assertEqual(merged, pixiv)
        self.assertEqual(self.store.canonical_author_key(twitter), pixiv)
        summary = self.store.author_summary(pixiv)
        self.assertIsNotNone(summary)
        assert summary is not None
        self.assertEqual(summary["image_count"], 1)
        self.assertEqual(summary["pixiv_ids"], ["100"])
        self.assertEqual(summary["x_handles"], ["shared_handle"])
        self.assertEqual(self.store.point_ids_for_author(twitter), ["p1"])

    def test_upgrade_merges_legacy_shared_handle_and_keeps_legacy_keys(self) -> None:
        path = Path(self._temporary.name) / "legacy.sqlite3"
        connection = sqlite3.connect(path)
        connection.executescript(
            """
            CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE image_source (
                point_id TEXT PRIMARY KEY,
                content_hash TEXT,
                author_key TEXT,
                platform TEXT NOT NULL DEFAULT 'pixiv',
                status TEXT NOT NULL DEFAULT 'ok',
                updated_at INTEGER NOT NULL
            );
            CREATE TABLE author (
                key TEXT PRIMARY KEY,
                platform TEXT NOT NULL,
                platform_id TEXT NOT NULL,
                name TEXT,
                handle TEXT,
                profile_url TEXT,
                avatar_url TEXT,
                social_json TEXT,
                profile_fetched INTEGER NOT NULL DEFAULT 0,
                updated_at INTEGER NOT NULL,
                UNIQUE(platform, platform_id)
            );
            CREATE TABLE author_alias (
                alias_key TEXT PRIMARY KEY,
                canonical_key TEXT NOT NULL,
                evidence TEXT,
                confidence REAL NOT NULL DEFAULT 0.5,
                updated_at INTEGER NOT NULL
            );
            INSERT INTO meta VALUES ('schema_version', '4');
            INSERT INTO author VALUES
                ('pixiv:100', 'pixiv', '100', 'Pixiv Name', 'shared', NULL, NULL, '{}', 1, 1),
                ('twitter:shared', 'twitter', 'shared', 'X Name', 'shared', NULL, NULL, '{}', 0, 1);
            INSERT INTO image_source (
                point_id, content_hash, author_key, platform, status, updated_at
            )
            VALUES ('p1', 'a', 'pixiv:100', 'pixiv', 'ok', 1),
                   ('p2', 'b', 'twitter:shared', 'twitter', 'ok', 2);
            """
        )
        connection.commit()
        connection.close()
        upgraded = ProvenanceStore(path)
        pixiv_key = upgraded.canonical_author_key("pixiv:100")
        self.assertEqual(upgraded.canonical_author_key("twitter:shared"), pixiv_key)
        summary = upgraded.author_summary(pixiv_key)
        self.assertIsNotNone(summary)
        assert summary is not None
        self.assertEqual(summary["name"], "Pixiv Name")
        self.assertEqual(summary["image_count"], 2)
        self.assertEqual(
            sorted(upgraded.point_ids_for_author(pixiv_key)), ["p1", "p2"]
        )

    def test_manual_author_is_reused_and_numeric_keyed(self) -> None:
        key = self.store.ensure_manual_author("某画师")
        self.assertTrue(key.isdigit())
        self.assertEqual(key, self.store.ensure_manual_author("  某画师 "))
        summary = self.store.author_summary(key)
        self.assertIsNotNone(summary)
        assert summary is not None
        self.assertEqual(summary["name"], "某画师")
        self.assertEqual(summary["identities"][0]["kind"], "name")

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

    def test_social_links_are_normalized_and_conflicts_surface(self) -> None:
        first = self.store.ensure_author("pixiv", "1", name="Alice")
        second = self.store.ensure_author("pixiv", "2", name="Bob")
        self.store.add_link(first, "HTTPS://Example.COM/Artist/", label="主页")
        summary = self.store.author_summary(first)
        assert summary is not None
        self.assertEqual(summary["links"][0]["identity"], "https://example.com/Artist")
        self.assertEqual(summary["links"][0]["display"], "主页")
        with self.assertRaises(ValueError):
            self.store.add_link(second, "https://example.com/Artist")
        with self.assertRaises(ValueError):
            self.store.add_link(first, "ftp://example.com/a")

    def test_xiaohongshu_name_groups_artists_and_profile_is_attachable(self) -> None:
        key = self.store.ensure_xhs_author("李岫荇")
        self.assertEqual(key, self.store.ensure_xhs_author(" 李岫荇 "))
        self.store.upsert_unresolved(
            "p1",
            self.payload,
            path_key="p1",
            hint=parse_xiaohongshu_filename("衣架兔！！_1_李岫荇_来自小红书网页版.jpg"),
            author_key=key,
        )
        record = self.store.get("p1")
        assert record is not None
        self.assertEqual(record["platform"], "xiaohongshu")
        self.assertEqual(record["status"], "unresolved")
        self.assertEqual(record["title"], "衣架兔！！")
        self.assertEqual(record["page"], 1)
        self.assertEqual(record["author_key"], key)

        uid = "5f1c2d3e000000000101abcd"
        attached = self.store.attach_identity(
            key, "xiaohongshu", f"https://www.xiaohongshu.com/user/profile/{uid}"
        )
        self.assertEqual(attached, key)
        self.assertEqual(self.store.identity_owner("xiaohongshu", uid), key)
        summary = self.store.author_summary(key)
        assert summary is not None
        self.assertEqual(summary["xiaohongshu_ids"], [uid])
        self.assertEqual(summary["members"][0]["platform"], "xiaohongshu")
        with self.assertRaises(ValueError):
            self.store.attach_identity(key, "xiaohongshu", "not-a-link")

        self.store.remove_identity(key, "xiaohongshu", uid)
        summary = self.store.author_summary(key)
        assert summary is not None
        self.assertEqual(summary["xiaohongshu_ids"], [])

    def test_renamed_xiaohongshu_author_merges_without_losing_images(self) -> None:
        old = self.store.ensure_xhs_author("旧昵称")
        new = self.store.ensure_xhs_author("新昵称")
        self.assertNotEqual(old, new)
        self.store.upsert_unresolved(
            "p1",
            self.payload,
            path_key="p1",
            hint=parse_xiaohongshu_filename("标题_1_旧昵称_来自小红书网页版.jpg"),
            author_key=old,
        )
        merged = self.store.merge_authors(new, old)
        self.assertEqual(merged, old)
        self.assertEqual(self.store.canonical_author_key(new), old)
        summary = self.store.author_summary(old)
        assert summary is not None
        self.assertEqual(summary["image_count"], 1)
        self.assertEqual(self.store.get("p1")["author_key"], old)

    def test_unassigned_author_is_a_pseudo_author(self) -> None:
        author = self.store.ensure_author("pixiv", "1", name="Alice")
        self.store.upsert_scan("p1", self.payload, _parsed("12345678_p0.png"))
        self.store.upsert_scan(
            "p2",
            {**self.payload, "content_hash": "b" * 64},
            _parsed("1761931720818122000_1.jpg"),
        )
        self.store.set_fetch_result(
            "p2",
            {
                "platform": "twitter",
                "source_id": "1761931720818122000",
                "page": 1,
                "url": "https://x.com/i/status/1761931720818122000",
                "status": "ok",
                "tags": [],
            },
            author_key=author,
        )
        self.assertEqual(self.store.point_ids_for_author("none"), ["p1"])
        self.assertEqual(
            self.store.author_counts_for_points(["p1", "p2"]),
            {"none": 1, author: 1},
        )
        summary = self.store.author_summary("none")
        assert summary is not None
        self.assertEqual(summary["name"], "无画师")
        self.assertEqual(summary["image_count"], 1)

    def test_artist_stats_orders_and_filters_scopes(self) -> None:
        alice = self.store.ensure_author("pixiv", "1", name="Alice")
        bob = self.store.ensure_author("pixiv", "2", name="Bob")
        for point_id, mtime, author in [
            ("p1", 1_000, alice),
            ("p2", 4_000, alice),
            ("p3", 3_000, bob),
        ]:
            payload = {
                **self.payload,
                "content_hash": f"{point_id}" * 20,
                "mtime_ns": mtime,
            }
            self.store.upsert_scan(point_id, payload, _parsed(f"1234567{point_id[-1]}_p0.png"))
            self.store.set_fetch_result(
                point_id,
                {
                    "platform": "pixiv",
                    "source_id": f"1234567{point_id[-1]}",
                    "page": 0,
                    "url": f"https://www.pixiv.net/artworks/1234567{point_id[-1]}",
                    "status": "ok",
                    "tags": [],
                },
                author_key=author,
            )
        self.store.upsert_scan(
            "p4",
            {**self.payload, "content_hash": "d" * 64, "mtime_ns": 5_000},
            _parsed("99999999_p0.png"),
        )
        latest = self.store.artist_stats(
            None, sort="latest_desc", include_unassigned=True, limit=10
        )
        self.assertEqual([item["key"] for item in latest["authors"]], ["none", alice, bob])
        by_count = self.store.artist_stats(
            None, sort="count_desc", include_unassigned=False, limit=10
        )
        self.assertEqual([item["key"] for item in by_count["authors"]], [alice, bob])
        searched = self.store.artist_stats(None, query="无画师", include_unassigned=True, limit=10)
        self.assertEqual([item["key"] for item in searched["authors"]], ["none"])
        scoped = self.store.artist_stats(["p1", "p3"], sort="name_asc", limit=10)
        self.assertEqual([item["key"] for item in scoped["authors"]], [alice, bob])

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
