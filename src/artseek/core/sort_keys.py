"""Deterministic natural-sort key shared by indexing and SQL ordering."""

from __future__ import annotations

import re

_DIGITS = re.compile(r"(\d+)")
_PAD = 20


def natural_sort_text(name: str) -> str:
    """Return a text key whose binary order matches `library.natural_key`."""
    parts = _DIGITS.split(name.casefold())
    return "\x01".join(
        ("0" + str(int(part)).zfill(_PAD)) if part.isdigit() else ("1" + part)
        for part in parts
        if part
    )
