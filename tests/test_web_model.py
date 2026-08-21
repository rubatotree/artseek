from __future__ import annotations

import unittest
from contextlib import nullcontext

from artseek.config import AppConfig
from artseek.web.app import create_app


class FakeEngine:
    def __init__(self) -> None:
        self.loaded = False
        self.auto_unload_seconds = 600
        self.busy = False
        self.config = AppConfig()
        self.config.embedding.model = "fake-embedder"

    def model_status(self) -> dict:
        minutes, extra = divmod(max(0, self.auto_unload_seconds), 60)
        if extra:
            minutes += 1
        return {
            "loaded": self.loaded,
            "loading": False,
            "busy": self.busy,
            "model": self.config.embedding.model,
            "auto_unload_seconds": self.auto_unload_seconds,
            "auto_unload_minutes": minutes,
            "idle_remaining_seconds": 500 if self.loaded else None,
        }

    def load_model(self) -> None:
        self.loaded = True

    def unload_model(self, *, force: bool = False) -> bool:
        if self.busy and not force:
            return False
        self.loaded = False
        return True

    def set_auto_unload_seconds(self, seconds: int) -> None:
        self.auto_unload_seconds = seconds

    def hold_model(self):
        return nullcontext()

    def close(self) -> None:
        return


class ModelApiTests(unittest.TestCase):
    def setUp(self) -> None:
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            self.skipTest("fastapi 未安装")
        self.engine = FakeEngine()
        self.client = TestClient(create_app(AppConfig(), engine=self.engine))

    def test_starts_unloaded(self) -> None:
        data = self.client.get("/api/model").json()
        self.assertFalse(data["loaded"])
        self.assertEqual(data["auto_unload_minutes"], 10)

    def test_homepage_exposes_model_controls(self) -> None:
        html = self.client.get("/").text
        self.assertIn('id="model-loaded"', html)
        self.assertIn('id="auto-unload"', html)
        self.assertIn("图文向量模型", html)

    def test_toggle_load_and_unload(self) -> None:
        loaded = self.client.post("/api/model", json={"loaded": True}).json()
        self.assertTrue(loaded["loaded"])
        self.assertTrue(self.engine.loaded)
        unloaded = self.client.post("/api/model", json={"loaded": False}).json()
        self.assertFalse(unloaded["loaded"])

    def test_busy_unload_is_deferred(self) -> None:
        self.engine.loaded = True
        self.engine.busy = True
        data = self.client.post("/api/model", json={"loaded": False}).json()
        self.assertTrue(data["loaded"])
        self.assertIn("检索或索引", data["message"])

    def test_auto_unload_minutes_round_trip(self) -> None:
        data = self.client.post("/api/model", json={"auto_unload_minutes": 15}).json()
        self.assertEqual(self.engine.auto_unload_seconds, 900)
        self.assertEqual(data["auto_unload_minutes"], 15)


if __name__ == "__main__":
    unittest.main()
