from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image
from PIL.PngImagePlugin import PngInfo

from artseek.provenance.parser import (
    extract_embedded_refs,
    parse_filename,
    parse_source_text,
    parse_url,
    parsed_sources,
    strip_duplicate_suffix,
)


class FilenameParsingTests(unittest.TestCase):
    def test_pixiv_names_cover_download_variants(self) -> None:
        cases = {
            "12345678_p0.png": ("12345678", 0),
            "23456789_p1.jpg": ("23456789", 1),
            "234567890-00000000000000000000000000000000_p0.jpg": ("234567890", 0),
            "345678901_p0_master1200.jpg": ("345678901", 0),
            "85113323_p4.png": ("85113323", 4),
        }
        for name, (source_id, page) in cases.items():
            with self.subTest(name=name):
                ref = parse_filename(name)
                self.assertIsNotNone(ref)
                assert ref is not None
                self.assertEqual(ref.platform, "pixiv")
                self.assertEqual(ref.source_id, source_id)
                self.assertEqual(ref.page, page)

    def test_duplicate_suffix_is_not_part_of_the_id(self) -> None:
        self.assertEqual(strip_duplicate_suffix("456789012_p0 (1)"), "456789012_p0")
        ref = parse_filename("456789012_p0 (1).jpg")
        self.assertIsNotNone(ref)
        assert ref is not None
        self.assertEqual(ref.source_id, "456789012")

    def test_twitter_names_use_one_based_photo_index(self) -> None:
        ref = parse_filename("1761931720818122000_4.jpg")
        self.assertIsNotNone(ref)
        assert ref is not None
        self.assertEqual(ref.platform, "twitter")
        self.assertEqual(ref.source_id, "1761931720818122000")
        self.assertEqual(ref.page, 4)

    def test_eighteen_digit_status_ids_are_accepted(self) -> None:
        ref = parse_filename("859403023310770177_1.jpg")
        self.assertIsNotNone(ref)
        assert ref is not None
        self.assertEqual(ref.platform, "twitter")
        self.assertEqual(ref.source_id, "859403023310770177")

    def test_old_pixiv_id_naming_is_low_confidence(self) -> None:
        ref = parse_filename("pixiv ID 567890123.jpg")
        self.assertIsNotNone(ref)
        assert ref is not None
        self.assertEqual(ref.platform, "pixiv")
        self.assertEqual(ref.source_id, "567890123")
        self.assertLess(ref.confidence, 0.9)

    def test_xiaohongshu_and_arbitrary_names_are_not_guessed(self) -> None:
        self.assertIsNone(parse_filename("少女与枪_1_阿斯莫代_来自小红书网页版.jpg"))
        self.assertIsNone(parse_filename("F1jQ358aIAATjmn.jpg"))


class UrlParsingTests(unittest.TestCase):
    def test_pixiv_url_forms(self) -> None:
        for value in (
            "https://www.pixiv.net/artworks/12345678",
            "https://www.pixiv.net/en/artworks/12345678",
            "https://www.pixiv.net/i/12345678",
            "https://www.pixiv.net/member_illust.php?mode=medium&illust_id=12345678",
        ):
            with self.subTest(value=value):
                ref = parse_url(value)
                self.assertIsNotNone(ref)
                assert ref is not None
                self.assertEqual((ref.platform, ref.source_id), ("pixiv", "12345678"))

    def test_twitter_photo_index_and_host_aliases(self) -> None:
        for value in (
            "https://x.com/user/status/1234567890123456789/photo/2",
            "https://twitter.com/user/status/1234567890123456789/photo/2",
            "https://mobile.twitter.com/user/status/1234567890123456789/photo/2",
            "https://vxtwitter.com/user/status/1234567890123456789/photo/2",
            "https://fxtwitter.com/user/status/1234567890123456789/photo/2",
        ):
            with self.subTest(value=value):
                ref = parse_url(value)
                self.assertIsNotNone(ref)
                assert ref is not None
                self.assertEqual(ref.platform, "twitter")
                self.assertEqual(ref.source_id, "1234567890123456789")
                self.assertEqual(ref.page, 2)

    def test_twitter_status_without_photo_has_unknown_page(self) -> None:
        ref = parse_url("https://x.com/example_artist/status/1872266114488828000")
        self.assertIsNotNone(ref)
        assert ref is not None
        self.assertIsNone(ref.page)

    def test_arbitrary_http_url_is_kept_as_other(self) -> None:
        ref = parse_source_text("https://example.com/art/123")
        self.assertIsNotNone(ref)
        assert ref is not None
        self.assertEqual(ref.platform, "other")
        self.assertEqual(ref.url, "https://example.com/art/123")

    def test_manual_bare_ids(self) -> None:
        pixiv = parse_source_text("12345678_p2")
        twitter = parse_source_text("1761931720818122000_3")
        self.assertIsNotNone(pixiv)
        self.assertIsNotNone(twitter)
        assert pixiv is not None and twitter is not None
        self.assertEqual((pixiv.platform, pixiv.page), ("pixiv", 2))
        self.assertEqual((twitter.platform, twitter.page), ("twitter", 3))


class EmbeddedMetadataTests(unittest.TestCase):
    def test_xmp_user_comment_url_is_extracted(self) -> None:
        xmp = (
            '<x:xmpmeta xmlns:x="adobe:ns:meta/">'
            '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
            '<rdf:Description xmlns:exif="http://ns.adobe.com/exif/1.0/">'
            "<exif:UserComment><rdf:Alt>"
            '<rdf:li xml:lang="x-default">'
            "https://x.com/example_artist/status/1872266114488828000/photo/1"
            "</rdf:li></rdf:Alt></exif:UserComment>"
            "</rdf:Description></rdf:RDF></x:xmpmeta>"
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "F1jQ358aIAATjmn.png"
            metadata = PngInfo()
            metadata.add_text("XML:com.adobe.xmp", xmp)
            Image.new("RGB", (8, 8), (20, 30, 40)).save(path, pnginfo=metadata)
            refs = extract_embedded_refs(path)
            self.assertEqual(len(refs), 1)
            self.assertEqual(refs[0].source_id, "1872266114488828000")
            self.assertEqual(refs[0].page, 1)
            parsed = parsed_sources(path)
            self.assertIsNotNone(parsed)
            assert parsed is not None
            self.assertEqual(parsed.primary.origin, "embedded")

    def test_xpcomment_utf16le_url_is_extracted(self) -> None:
        url = "https://x.com/example_artist2/status/1905879515315732000/photo/1"
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "GnLt47zaMAAjCZD.jpg"
            exif = Image.Exif()
            exif[40092] = url.encode("utf-16-le")
            Image.new("RGB", (8, 8), (20, 30, 40)).save(path, exif=exif, quality=90)
            refs = extract_embedded_refs(path)
            self.assertEqual(len(refs), 1)
            self.assertEqual(refs[0].source_id, "1905879515315732000")
            self.assertEqual(refs[0].page, 1)

    def test_embedded_url_wins_over_filename_but_conflict_is_kept(self) -> None:
        xmp = (
            '<x:xmpmeta xmlns:x="adobe:ns:meta/">'
            '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
            '<rdf:Description xmlns:exif="http://ns.adobe.com/exif/1.0/">'
            "<exif:UserComment><rdf:Alt>"
            '<rdf:li xml:lang="x-default">https://x.com/a/status/1872266114488828000</rdf:li>'
            "</rdf:Alt></exif:UserComment>"
            "</rdf:Description></rdf:RDF></x:xmpmeta>"
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "12345678_p0.png"
            metadata = PngInfo()
            metadata.add_text("XML:com.adobe.xmp", xmp)
            Image.new("RGB", (8, 8), (20, 30, 40)).save(path, pnginfo=metadata)
            parsed = parsed_sources(path)
            self.assertIsNotNone(parsed)
            assert parsed is not None
            self.assertEqual(parsed.primary.platform, "twitter")
            self.assertEqual(len(parsed.candidates), 1)
            self.assertEqual(parsed.candidates[0].platform, "pixiv")


if __name__ == "__main__":
    unittest.main()
