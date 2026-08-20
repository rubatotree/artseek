from __future__ import annotations

import unittest

from artseek.compute import resolve_device, resolve_dtype


class ComputeTests(unittest.TestCase):
    def test_cpu_override(self) -> None:
        device = resolve_device("cpu")
        self.assertEqual(device.type, "cpu")
        self.assertEqual(resolve_dtype("auto", device), __import__("torch").float32)

    def test_explicit_float16(self) -> None:
        device = resolve_device("cpu")
        self.assertEqual(resolve_dtype("float16", device), __import__("torch").float16)


if __name__ == "__main__":
    unittest.main()
