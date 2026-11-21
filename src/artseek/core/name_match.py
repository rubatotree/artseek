"""Folded comparison keys for personal names, handles and artist aliases.

One :class:`NameKey` is built per searchable value (artist name, identity
display, handle, pixiv id). Matching then compares two prepared keys, which
makes the search tolerant to the ways people actually type a name:

* simplified/traditional/variant characters — ``宫`` / ``宮``, ``長髮`` / ``长发``;
* katakana vs hiragana — ``ナツカ`` / ``なつか``;
* pinyin and romaji — ``huagong``, ``natsuka`` for ``花宮なつか``;
* spacing, punctuation and emoji — ``ぴおてぐ（旧:potg）``.

Scores are in ``[0, 1]`` and are meant for ranking, not for thresholds alone:
``0`` means "no relation", ``>= 0.85`` means "very likely the one".
"""

from __future__ import annotations

import difflib
import unicodedata
from dataclasses import dataclass
from functools import lru_cache

from .transliterate import fold_key, lookup_keys

_EXACT = 1.0
_COMPACT_CONTAINS = 0.86
_KEY_EXACT = 0.9
_KEY_PREFIX = 0.82
_KEY_CONTAINS = 0.68
_DIGIT_KEY = 0.88
_FUZZY_FLOOR = 0.6
_FUZZY_MIN_LENGTH = 3


@dataclass(frozen=True, slots=True)
class NameKey:
    """Prepared comparison form of one name-like value."""

    text: str
    compact: str
    keys: frozenset[str]

    @property
    def empty(self) -> bool:
        return not self.compact and not self.keys


def _display_text(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return " ".join(text.lstrip("@").split())


def build(value: object) -> NameKey:
    """Prepare one value for comparison."""
    text = _display_text(value)
    if not text:
        return NameKey("", "", frozenset())
    return NameKey(text, fold_key(text), frozenset(lookup_keys(text)))


@lru_cache(maxsize=32768)
def build_cached(value: str) -> NameKey:
    """``build`` with a process-wide cache; use for repeated name values."""
    return build(value)


def build_all(values: object) -> tuple[NameKey, ...]:
    """Prepare several values, dropping empties and duplicates."""
    if isinstance(values, str):
        values = (values,)
    keys: list[NameKey] = []
    seen: set[tuple[str, frozenset[str]]] = set()
    for value in values or ():  # type: ignore[union-attr]
        key = build(value)
        if key.empty:
            continue
        marker = (key.compact, key.keys)
        if marker in seen:
            continue
        seen.add(marker)
        keys.append(key)
    return tuple(keys)


def _is_digits(value: str) -> bool:
    return value.isdigit()


def score(needle: NameKey, candidate: NameKey) -> float:
    """Rank one prepared candidate against a prepared query; ``0`` = no match."""
    if needle.empty or candidate.empty:
        return 0.0
    if candidate.compact == needle.compact:
        return _EXACT
    best = 0.0
    if needle.compact and needle.compact in candidate.compact:
        best = _COMPACT_CONTAINS + 0.14 * len(needle.compact) / len(candidate.compact)
    elif _is_digits(needle.compact) and any(
        needle.compact in key for key in candidate.keys
    ):
        # Numeric queries (pixiv ids) are specific; a substring hit is enough.
        best = _DIGIT_KEY
    for key in candidate.keys:
        for wanted in needle.keys:
            if not wanted:
                continue
            if key == wanted:
                matched = _KEY_EXACT
            elif key.startswith(wanted):
                matched = _KEY_PREFIX
            elif wanted in key:
                matched = _KEY_CONTAINS + 0.1 * len(wanted) / len(key)
            else:
                continue
            if matched > best:
                best = matched
    return max(best, _fuzzy_score(needle, candidate, best))


def _fuzzy_score(needle: NameKey, candidate: NameKey, best: float) -> float:
    """Typo tolerance for latin queries, skipped once a solid hit exists."""
    if best >= _FUZZY_FLOOR or len(needle.compact) < _FUZZY_MIN_LENGTH:
        return 0.0
    limit = len(needle.compact)
    score_value = 0.0
    for text in (candidate.compact, *candidate.keys):
        if not text or abs(len(text) - limit) > max(4, limit // 2):
            continue
        ratio = difflib.SequenceMatcher(None, needle.compact, text).ratio()
        if ratio >= 0.62:
            score_value = max(score_value, ratio * 0.8)
    return score_value


def best_score(needle: NameKey, candidates: object) -> float:
    """Highest score across several prepared candidate values."""
    best = 0.0
    for candidate in candidates or ():  # type: ignore[union-attr]
        value = score(needle, candidate)  # type: ignore[arg-type]
        if value > best:
            best = value
            if best >= _EXACT:
                break
    return best


def search_keys(needle: NameKey) -> tuple[str, ...]:
    """Folded keys worth scanning in a candidate index, longest first."""
    values = {needle.compact, *needle.keys}
    values.discard("")
    return tuple(sorted(values, key=lambda item: (-len(item), item)))
