"""Shared image pre-shrinking for GPU-saturated indexing and query embedding."""

from __future__ import annotations

import math

from PIL import Image

PRE_SHRINK_SHORT_EDGE = 512

# `Image.draft` only reduces decoding work for these formats.
_DRAFT_FORMATS = {"JPEG", "MPO", "WEBP"}


def draft_target(image: Image.Image, short_edge: int) -> tuple[int, int] | None:
    """Return a JPEG/WebP draft size that never undershoots `short_edge`."""
    width, height = image.size
    short = min(width, height)
    if short <= short_edge:
        return None
    scale = short_edge / float(short)
    return (max(1, math.ceil(width * scale)), max(1, math.ceil(height * scale)))


def shrink_to_short_edge(
    image: Image.Image, short_edge: int = PRE_SHRINK_SHORT_EDGE
) -> Image.Image:
    """Downscale preserving aspect ratio; never upscales."""
    width, height = image.size
    short = min(width, height)
    if short <= short_edge:
        return image
    scale = short_edge / float(short)
    target = (
        max(1, int(round(width * scale))),
        max(1, int(round(height * scale))),
    )
    return image.resize(target, Image.Resampling.BICUBIC)
