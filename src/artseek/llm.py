"""Compatibility re-export of OllamaClient."""

from .ml.llm import (
    CAPTION_SCHEMA,
    QUERY_SCHEMA,
    OllamaClient,
    OllamaUnavailable,
    flatten_caption,
    is_localhost_url,
)

__all__ = [
    "CAPTION_SCHEMA",
    "QUERY_SCHEMA",
    "OllamaClient",
    "OllamaUnavailable",
    "flatten_caption",
    "is_localhost_url",
]
