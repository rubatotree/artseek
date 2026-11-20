"""VLM caption generator wrapper over proxy-bypassing OllamaClient."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import LLMConfig
from ..core.ranking import order_tags_for_descriptor
from .llm import CAPTION_SCHEMA, OllamaClient, OllamaUnavailable, flatten_caption


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
    ) -> str:
        ordered = order_tags_for_descriptor(tags)
        tag_text = ", ".join(tag.replace("_", " ") for tag in ordered)
        parts = [flatten_caption(caption), tag_text]
        return "。".join(filter(None, parts))


__all__ = [
    "CAPTION_SCHEMA",
    "Captioner",
    "OllamaUnavailable",
    "flatten_caption",
]
