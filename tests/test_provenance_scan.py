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
