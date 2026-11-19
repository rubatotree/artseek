from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image
from PIL.PngImagePlugin import PngInfo

from artseek.provenance.parser import ParsedSource, parse_filename
from artseek.provenance.scan import scan_payloads
from artseek.provenance.store import ProvenanceStore


class ProvenanceScanTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.store = ProvenanceStore(self.root / "provenance.sqlite3")
        self.pixiv = self.root / "12345678_p0.png"
        Image.new("RGB", (12, 16), (20, 30, 40)).save(self.pixiv)
        self.hashfile = self.root / "F1jQ358aIAATjmn.png"
        xmp = (
            '<x:xmpmeta xmlns:x="adobe:ns:meta/">'
            '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
            '<rdf:Description xmlns:exif="http://ns.adobe.com/exif/1.0/">'
            "<exif:UserComment><rdf:Alt><rdf:li>"
            "https://x.com/example_artist/status/1872266114488828000"
            "</rdf:li></rdf:Alt></exif:UserComment>"
            "</rdf:Description></rdf:RDF></x:xmpmeta>"
        )
        metadata = PngInfo()
        metadata.add_text("XML:com.adobe.xmp", xmp)
        Image.new("RGB", (30, 40), (10, 90, 40)).save(self.hashfile, pnginfo=metadata)

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def _payload(self, path: Path, digest: str) -> dict:
        return {
            "path": str(path),
            "path_key": str(path).casefold(),
            "width": 12,
            "height": 16,
            "mtime_ns": 1,
            "content_hash": digest,
        }

    def test_filename_and_embedded_metadata_are_scanned(self) -> None:
        payloads = {
            "p1": self._payload(self.pixiv, "a" * 64),
            "p2": self._payload(self.hashfile, "b" * 64),
        }
        summary = scan_payloads(self.store, payloads.items(), total=2)
        self.assertEqual(summary["identified"], 2)
        self.assertEqual(summary["unresolved"], 0)
        pixiv = self.store.get("p1")
        embedded = self.store.get("p2")
        self.assertEqual(pixiv["platform"], "pixiv")
        self.assertEqual(embedded["platform"], "twitter")
        self.assertEqual(embedded["source_id"], "1872266114488828000")
        self.assertEqual(summary["authors_backfilled"], 1)
        author = self.store.author_summary(str(embedded["author_key"]))
        self.assertIsNotNone(author)
        assert author is not None
        self.assertEqual(author["name"], "example_artist")

    def test_unresolved_row_is_visible(self) -> None:
        unknown = self.root / "unknown-name.png"
        Image.new("RGB", (8, 8), (0, 0, 0)).save(unknown)
        summary = scan_payloads(
            self.store, {"p3": self._payload(unknown, "c" * 64)}.items(), total=1
        )
        self.assertEqual(summary["unresolved"], 1)
        record = self.store.get("p3")
        self.assertEqual(record["status"], "unresolved")
        self.assertEqual(record["platform"], "other")

    def test_xiaohongshu_names_keep_title_index_and_author_group(self) -> None:
        first = self.root / "衣架兔！！_1_李岫荇_来自小红书网页版.jpg"
        second = self.root / "衣架兔！！_2_李岫荇_来自小红书网页版.jpg"
        Image.new("RGB", (10, 12), (1, 2, 3)).save(first)
        Image.new("RGB", (10, 12), (4, 5, 6)).save(second)
        payloads = {
            "x1": self._payload(first, "e" * 64),
            "x2": self._payload(second, "f" * 64),
        }
        summary = scan_payloads(self.store, payloads.items(), total=2)
        self.assertEqual(summary["unresolved"], 2)
        self.assertEqual(summary["identified"], 0)
        record = self.store.get("x1")
        self.assertEqual(record["platform"], "xiaohongshu")
        self.assertEqual(record["status"], "unresolved")
        self.assertEqual(record["title"], "衣架兔！！")
        self.assertEqual(record["page"], 1)
        self.assertEqual(record["source_id"], "")
        author = self.store.author_summary(str(record["author_key"]))
        self.assertIsNotNone(author)
        assert author is not None
        self.assertEqual(author["name"], "李岫荇")
        self.assertEqual(author["image_count"], 2)
        self.assertEqual(
            self.store.get("x2")["author_key"], record["author_key"]
        )

        again = scan_payloads(self.store, payloads.items(), total=2)
        self.assertEqual(again["unresolved"], 2)
        self.assertEqual(self.store.count_authors(), 1)

    def test_content_hash_replacement_drops_stale_metadata(self) -> None:
        payload = self._payload(self.pixiv, "a" * 64)
        self.store.upsert_scan("p1", payload, ParsedSource(parse_filename(self.pixiv)))
        self.store.set_fetch_result(
            "p1",
            {
                "platform": "pixiv",
                "source_id": "12345678",
                "page": 0,
                "url": "https://www.pixiv.net/artworks/12345678",
                "status": "ok",
                "title": "旧图",
            },
        )
        replaced = self._payload(self.pixiv, "d" * 64)
        scan_payloads(self.store, {"p1": replaced}.items(), total=1)
        record = self.store.get("p1")
        self.assertEqual(record["content_hash"], "d" * 64)
        self.assertIsNone(record["title"])
        self.assertEqual(record["status"], "pending")

    def test_only_new_revisits_moved_same_content(self) -> None:
        payload = self._payload(self.pixiv, "a" * 64)
        scan_payloads(self.store, {"p1": payload}.items(), total=1)
        moved = self.root / "renamed" / self.pixiv.name
        moved.parent.mkdir()
        self.pixiv.rename(moved)
        moved_payload = self._payload(moved, "a" * 64)
        summary = scan_payloads(
            self.store, {"p1": moved_payload}.items(), total=1, only_new=True
        )
        self.assertEqual(summary["scanned"], 1)
        record = self.store.get("p1")
        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(record["path_key"], moved_payload["path_key"])

    def test_only_new_skips_unchanged_points(self) -> None:
        payload = self._payload(self.pixiv, "a" * 64)
        scan_payloads(self.store, {"p1": payload}.items(), total=1)
        summary = scan_payloads(
            self.store, {"p1": payload}.items(), total=1, only_new=True
        )
        self.assertEqual(summary["skipped"], 1)
        self.assertEqual(summary["scanned"], 0)


if __name__ == "__main__":
    unittest.main()
