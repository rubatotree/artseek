from __future__ import annotations

import json
import unittest
from unittest.mock import patch
from urllib.error import URLError

from artseek.config import LLMConfig
from artseek.llm import OllamaClient, OllamaUnavailable


class ModelInstalledTests(unittest.TestCase):
    def test_matches_exact_and_quantized_names(self) -> None:
        self.assertTrue(OllamaClient.model_installed("qwen3:4b", ["qwen3:4b"]))
        self.assertTrue(OllamaClient.model_installed("qwen3:4b", ["qwen3:4b-q4_K_M"]))
        self.assertFalse(OllamaClient.model_installed("qwen3:4b", ["qwen3:8b"]))
        self.assertFalse(OllamaClient.model_installed("qwen3:4b", []))


class PreloadTests(unittest.TestCase):
    def test_disabled_client_does_not_touch_network(self) -> None:
        client = OllamaClient(LLMConfig(enabled=False))
        with patch("artseek.llm.urllib.request.urlopen") as urlopen:
            client.preload()
        urlopen.assert_not_called()

    def test_preload_loads_installed_query_model(self) -> None:
        client = OllamaClient(LLMConfig(enabled=True, query_model="qwen3:4b"))
        tags = json.dumps({"models": [{"name": "qwen3:4b"}]}).encode("utf-8")
        loaded = json.dumps({"done": True}).encode("utf-8")
        responses = [tags, loaded]

        def fake_urlopen(request, timeout=None):
            del timeout
            payload = responses.pop(0)

            class Handle:
                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    return False

                def read(self):
                    return payload

            return Handle()

        with patch("artseek.llm.urllib.request.urlopen", side_effect=fake_urlopen):
            client.preload()
        self.assertEqual(responses, [])

    def test_preload_errors_when_model_missing(self) -> None:
        client = OllamaClient(LLMConfig(enabled=True, query_model="qwen3:4b"))
        tags = json.dumps({"models": [{"name": "qwen3:8b"}]}).encode("utf-8")

        def fake_urlopen(request, timeout=None):
            del request, timeout

            class Handle:
                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    return False

                def read(self):
                    return tags

            return Handle()

        with patch("artseek.llm.urllib.request.urlopen", side_effect=fake_urlopen):
            with self.assertRaises(OllamaUnavailable):
                client.preload()

    def test_preload_errors_when_ollama_down(self) -> None:
        client = OllamaClient(LLMConfig(enabled=True))
        with patch("artseek.llm.urllib.request.urlopen", side_effect=URLError("down")):
            with self.assertRaises(OllamaUnavailable):
                client.preload()

    def test_unload_sends_keep_alive_zero(self) -> None:
        client = OllamaClient(LLMConfig(enabled=True, query_model="qwen3:4b", vision_model="qwen3-vl:4b"))
        seen: list[dict] = []

        def fake_urlopen(request, timeout=None):
            del timeout
            seen.append(json.loads(request.data.decode("utf-8")))

            class Handle:
                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    return False

                def read(self):
                    return json.dumps({"done": True}).encode("utf-8")

            return Handle()

        with patch("artseek.llm.urllib.request.urlopen", side_effect=fake_urlopen):
            client.unload()
        self.assertEqual(
            seen,
            [
                {"model": "qwen3:4b", "keep_alive": 0},
                {"model": "qwen3-vl:4b", "keep_alive": 0},
            ],
        )

    def test_disabled_unload_does_not_touch_network(self) -> None:
        client = OllamaClient(LLMConfig(enabled=False))
        with patch("artseek.llm.urllib.request.urlopen") as urlopen:
            client.unload()
        urlopen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
