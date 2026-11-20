"""VLM caption generator wrapper over proxy-bypassing OllamaClient."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import LLMConfig
from .llm import CAPTION_SCHEMA, OllamaClient, OllamaUnavailable, flatten_caption
from .tag_postprocess import build_high_density_descriptor


class Captioner:
    """Thin service wrapper around `OllamaClient.caption_image` for post-index backfills."""

    def __init__(self, config: LLMConfig, client: OllamaClient | None = None) -> None:
        self.config = config
        self.client = client or OllamaClient(config)

    def caption_image(self, path: Path) -> dict[str, Any]:
        return self.client.caption_image(path)

    @staticmethod
    def build_descriptor_text(
        tags: list[str],
        caption: dict[str, Any] | None,
        *,
        categories: dict[str, int] | None = None,
        title: str | None = None,
    ) -> str:
        return build_high_density_descriptor(
            tags,
            categories,
            caption=caption,
            title=title,
        )


__all__ = [
    "CAPTION_SCHEMA",
    "Captioner",
    "OllamaUnavailable",
    "flatten_caption",
]
