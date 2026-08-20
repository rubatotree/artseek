from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from artseek.image_io import canonical_path, load_rgb, point_id, scan_images


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


if __name__ == "__main__":
    unittest.main()

