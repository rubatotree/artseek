from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from artseek.image_io import (
    PHASH_MAX_DISTANCE,
    canonical_path,
    downscale_image,
    file_digest,
    hamming_distance,
    load_rgb,
    make_search_image,
    make_thumbnail,
    perceptual_hash,
    point_id,
    scan_images,
)


class ImageIOTests(unittest.TestCase):
    def test_scan_and_stable_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nested = root / "nested"
            nested.mkdir()
            image_path = nested / "sample.png"
            Image.new("RGBA", (12, 8), (255, 100, 120, 128)).save(image_path)
            (root / "ignore.txt").write_text("not an image", encoding="utf-8")

            results = list(scan_images([root]))
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0][1], image_path)
            self.assertEqual(point_id(image_path), point_id(Path(canonical_path(image_path))))
            self.assertEqual(load_rgb(image_path).mode, "RGB")
            self.assertEqual(file_digest(image_path), file_digest(image_path))
            self.assertEqual(len(file_digest(image_path)), 64)

    def test_search_image_is_bounded_jpeg(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            destination = root / "search.jpg"
            Image.new("RGB", (2400, 1200), (20, 80, 160)).save(source)
            make_search_image(source, destination, size=1200)
            with Image.open(destination) as image:
                self.assertEqual(image.format, "JPEG")
                self.assertLessEqual(max(image.size), 1200)
                self.assertEqual(image.mode, "RGB")

    def test_perceptual_hash_survives_resize_not_a_different_picture(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = Image.new("RGB", (80, 60))
            pixels = original.load()
            for y in range(60):
                for x in range(80):
                    pixels[x, y] = ((x * 17 + y * 9) % 200, (x * 5) % 180, (y * 11) % 160)
            source = root / "orig.png"
            scaled = root / "scaled.png"
            other = root / "other.png"
            original.save(source)
            original.resize((320, 240), Image.Resampling.LANCZOS).save(scaled)
            Image.new("RGB", (80, 60), (20, 180, 40)).save(other)
            left = perceptual_hash(source)
            right = perceptual_hash(scaled)
            alien = perceptual_hash(other)
            self.assertEqual(len(left), 16)
            self.assertLessEqual(hamming_distance(left, right), PHASH_MAX_DISTANCE)
            self.assertGreater(hamming_distance(left, alien), PHASH_MAX_DISTANCE)

    def test_downscale_image_progressive_linear_light_and_alpha(self) -> None:
        rgb = Image.new("RGB", (1200, 800), (240, 240, 240))
        scaled = downscale_image(rgb, 128)
        self.assertEqual(scaled.mode, "RGB")
        self.assertEqual(scaled.size, (128, 85))

        # No-op when image is already within bounds
        self.assertIs(downscale_image(scaled, 128), scaled)

        # Premultiplied alpha should avoid black fringing on transparent edges
        rgba = Image.new("RGBA", (600, 600), (0, 0, 0, 0))
        opaque_patch = Image.new("RGBA", (300, 600), (255, 255, 255, 255))
        rgba.paste(opaque_patch, (0, 0))
        scaled_rgba = downscale_image(rgba, 64)
        self.assertEqual(scaled_rgba.mode, "RGBA")
        self.assertEqual(scaled_rgba.size, (64, 64))
        # Boundary pixel should remain white-ish rather than darkened by black transparent pixels
        mid_pixel = scaled_rgba.getpixel((31, 32))
        self.assertGreater(mid_pixel[0], 200)
        self.assertGreater(mid_pixel[1], 200)
        self.assertGreater(mid_pixel[2], 200)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            src = root / "src.png"
            dst = root / "thumb.webp"
            rgb.save(src)
            make_thumbnail(src, dst, size=128)
            with Image.open(dst) as thumb:
                self.assertEqual(thumb.size, (128, 85))

        with self.assertRaises(ValueError):
            downscale_image(rgb, 0)


if __name__ == "__main__":
    unittest.main()

