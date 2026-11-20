from __future__ import annotations

import json
import os
import unittest
from unittest.mock import patch

import httpx

from artseek.config import LLMConfig
from artseek.ml.llm import OllamaClient, OllamaUnavailable, is_localhost_url


class ModelInstalledTests(unittest.TestCase):
    def test_matches_exact_and_quantized_names(self) -> None:
        self.assertTrue(OllamaClient.model_installed("qwen3:4b", ["qwen3:4b"]))
        self.assertTrue(OllamaClient.model_installed("qwen3:4b", ["qwen3:4b-q4_K_M"]))
        self.assertFalse(OllamaClient.model_installed("qwen3:4b", ["qwen3:8b"]))
        self.assertFalse(OllamaClient.model_installed("qwen3:4b", []))


class ProxyBypassTests(unittest.TestCase):
    def test_localhost_disables_env_proxy_while_remote_keeps_it(self) -> None:
        self.assertTrue(is_localhost_url("http://127.0.0.1:11434"))
        self.assertTrue(is_localhost_url("http://localhost:11434"))
        self.assertFalse(is_localhost_url("http://192.168.1.50:11434"))

        with patch.dict(
            os.environ,
            {
                "HTTP_PROXY": "http://127.0.0.1:7897",
                "HTTPS_PROXY": "http://127.0.0.1:7897",
                "ALL_PROXY": "socks5://127.0.0.1:7897",
            },
        ):
            local = OllamaClient(LLMConfig(base_url="http://127.0.0.1:11434"))
            with local._make_client(timeout=2.0) as c_local:
                self.assertFalse(c_local.trust_env)
                self.assertEqual(c_local._mounts, {})

            remote = OllamaClient(LLMConfig(base_url="http://192.168.1.50:11434"))
            with remote._make_client(timeout=2.0) as c_remote:
                self.assertTrue(c_remote.trust_env)
                self.assertTrue(bool(c_remote._mounts))


class PreloadTests(unittest.TestCase):
    def test_disabled_client_does_not_touch_network(self) -> None:
        called = False

        def handler(_request: httpx.Request) -> httpx.Response:
            nonlocal called
            called = True
            return httpx.Response(200, json={})

        client = OllamaClient(
            LLMConfig(enabled=False), transport=httpx.MockTransport(handler)
        )
        client.preload()
        self.assertFalse(called)

    def test_preload_loads_installed_query_model(self) -> None:
        paths: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            paths.append(request.url.path)
            if request.url.path == "/api/tags":
                return httpx.Response(200, json={"models": [{"name": "qwen3:4b"}]})
            return httpx.Response(200, json={"done": True})

        client = OllamaClient(
            LLMConfig(enabled=True, query_model="qwen3:4b"),
            transport=httpx.MockTransport(handler),
        )
        client.preload()
        self.assertEqual(paths, ["/api/tags", "/api/generate"])

    def test_preload_errors_when_model_missing(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"models": [{"name": "qwen3:8b"}]})

        client = OllamaClient(
            LLMConfig(enabled=True, query_model="qwen3:4b"),
            transport=httpx.MockTransport(handler),
        )
        with self.assertRaises(OllamaUnavailable):
            client.preload()

    def test_preload_errors_when_ollama_down(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("down", request=request)

        client = OllamaClient(
            LLMConfig(enabled=True), transport=httpx.MockTransport(handler)
        )
        with self.assertRaises(OllamaUnavailable):
            client.preload()

    def test_unload_sends_keep_alive_zero(self) -> None:
        seen: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(json.loads(request.content.decode("utf-8")))
            return httpx.Response(200, json={"done": True})

        client = OllamaClient(
            LLMConfig(enabled=True, query_model="qwen3:4b", vision_model="qwen3-vl:4b"),
            transport=httpx.MockTransport(handler),
        )
        client.unload()
        self.assertEqual(
            seen,
            [
                {"model": "qwen3:4b", "keep_alive": 0},
                {"model": "qwen3-vl:4b", "keep_alive": 0},
            ],
        )

    def test_disabled_unload_does_not_touch_network(self) -> None:
        called = False

        def handler(_request: httpx.Request) -> httpx.Response:
            nonlocal called
            called = True
            return httpx.Response(200, json={})

        client = OllamaClient(
            LLMConfig(enabled=False), transport=httpx.MockTransport(handler)
        )
        client.unload()
        self.assertFalse(called)


if __name__ == "__main__":
    unittest.main()
