"""Script folding and pinyin/romaji lookup keys for fuzzy CJK search.

Three layers are provided, each cheaper than the next:

* ``fold_text`` normalizes a string to a comparison form: NFKC, casefold,
  katakana→hiragana and simplified/traditional/variant folding (宫/宮, 髪/髮).
* ``fold_key`` additionally drops punctuation, emoji and spaces so that
  ``"ぴおてぐ（旧:potg）"`` and ``"ぴおてぐ 旧potg"`` compare equal.
* ``lookup_keys``/``transliteration_keys`` add latin keys so a query typed as
  pinyin (huagong) or romaji (natsuka) still reaches a CJK name.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from itertools import product

from .cjk_variants import GROUPS

_HAN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_KANA_RE = re.compile(r"[\u3040-\u30ff]")
_HIRA_RE = re.compile(r"[\u3041-\u3096\u309d-\u309f]")
_HAN_RUN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+")
_KANA_RUN = re.compile(r"[\u3040-\u309f\u30a0-\u30fa\u30fc-\u30ff]+")
_SCRIPT_RUN = re.compile(
    r"[\u3040-\u309f\u30a0-\u30fa\u30fc-\u30ff]+"
    r"|[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+"
    r"|[0-9a-z]+"
)
_KEY_KEEP = re.compile(
    r"[0-9a-z\u3040-\u309f\u30a0-\u30fa\u30fc-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+"
)

_KANA_TO_ROMAJI: dict[str, str] = {
    "あ": "a", "い": "i", "う": "u", "え": "e", "お": "o",
    "か": "ka", "き": "ki", "く": "ku", "け": "ke", "こ": "ko",
    "さ": "sa", "し": "shi", "す": "su", "せ": "se", "そ": "so",
    "た": "ta", "ち": "chi", "つ": "tsu", "て": "te", "と": "to",
    "な": "na", "に": "ni", "ぬ": "nu", "ね": "ne", "の": "no",
    "は": "ha", "ひ": "hi", "ふ": "fu", "へ": "he", "ほ": "ho",
    "ま": "ma", "み": "mi", "む": "mu", "め": "me", "も": "mo",
    "や": "ya", "ゆ": "yu", "よ": "yo",
    "ら": "ra", "り": "ri", "る": "ru", "れ": "re", "ろ": "ro",
    "わ": "wa", "を": "o", "ん": "n", "ゔ": "vu",
    "が": "ga", "ぎ": "gi", "ぐ": "gu", "げ": "ge", "ご": "go",
    "ざ": "za", "じ": "ji", "ず": "zu", "ぜ": "ze", "ぞ": "zo",
    "だ": "da", "ぢ": "ji", "づ": "zu", "で": "de", "ど": "do",
    "ば": "ba", "び": "bi", "ぶ": "bu", "べ": "be", "ぼ": "bo",
    "ぱ": "pa", "ぴ": "pi", "ぷ": "pu", "ぺ": "pe", "ぽ": "po",
    "ぁ": "a", "ぃ": "i", "ぅ": "u", "ぇ": "e", "ぉ": "o",
    "ゃ": "ya", "ゅ": "yu", "ょ": "yo", "ゎ": "wa",
}

_KANA_DIGRAPHS: dict[str, str] = {
    "きゃ": "kya", "きゅ": "kyu", "きょ": "kyo",
    "しゃ": "sha", "しゅ": "shu", "しょ": "sho",
    "ちゃ": "cha", "ちゅ": "chu", "ちょ": "cho",
    "にゃ": "nya", "にゅ": "nyu", "にょ": "nyo",
    "ひゃ": "hya", "ひゅ": "hyu", "ひょ": "hyo",
    "みゃ": "mya", "みゅ": "myu", "みょ": "myo",
    "りゃ": "rya", "りゅ": "ryu", "りょ": "ryo",
    "ぎゃ": "gya", "ぎゅ": "gyu", "ぎょ": "gyo",
    "じゃ": "ja", "じゅ": "ju", "じょ": "jo",
    "ぢゃ": "ja", "ぢゅ": "ju", "ぢょ": "jo",
    "びゃ": "bya", "びゅ": "byu", "びょ": "byo",
    "ぴゃ": "pya", "ぴゅ": "pyu", "ぴょ": "pyo",
    "ふぁ": "fa", "ふぃ": "fi", "ふぇ": "fe", "ふぉ": "fo",
    "うぃ": "wi", "うぇ": "we", "うぉ": "wo",
    "てぃ": "ti", "でぃ": "di", "とぅ": "tu", "どぅ": "du",
    "ゔぁ": "va", "ゔぃ": "vi", "ゔぇ": "ve", "ゔぉ": "vo",
}


def _build_variant_translation() -> dict[int, str]:
    table: dict[int, str] = {}
    for group in GROUPS:
        canonical = group[0]
        for char in group[1:]:
            table[ord(char)] = canonical
    return table


_VARIANT_TRANSLATION = _build_variant_translation()


def fold_text(value: str) -> str:
    """Return the comparison form of a name/tag: case, kana and variant folded."""
    return _fold_text_cached(str(value or ""))


@lru_cache(maxsize=32768)
def _fold_text_cached(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = _to_hiragana(text).casefold()
    return text.translate(_VARIANT_TRANSLATION)


def fold_key(value: str) -> str:
    """Return ``fold_text`` with spaces, punctuation and emoji removed."""
    return "".join(_KEY_KEEP.findall(fold_text(value)))


def normalize_latin(value: str) -> str:
    """Lowercase, strip tones/punctuation and ü/v ambiguity for lookup keys."""
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = text.replace("u\u0308", "v").replace("U\u0308", "v")
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.casefold().replace("u:", "v")
    return re.sub(r"[^0-9a-z]+", "", text)


def _to_hiragana(value: str) -> str:
    out: list[str] = []
    for ch in str(value or ""):
        code = ord(ch)
        if 0x30A1 <= code <= 0x30F6:
            out.append(chr(code - 0x60))
        elif ch == "ヴ":
            out.append("ゔ")
        else:
            out.append(ch)
    return "".join(out)


def _kana_to_romaji(value: str) -> str:
    text = _to_hiragana(value)
    out: list[str] = []
    pending_geminate = False
    index = 0
    while index < len(text):
        ch = text[index]
        if ch == "っ":
            pending_geminate = True
            index += 1
            continue
        if ch == "ー":
            if out:
                for vowel in reversed(out[-1]):
                    if vowel in "aeiou":
                        out.append(vowel)
                        break
            index += 1
            continue
        pair = text[index : index + 2]
        romaji = _KANA_DIGRAPHS.get(pair)
        if romaji is not None:
            index += 2
        else:
            romaji = _KANA_TO_ROMAJI.get(ch)
            if romaji is None:
                if not _HIRA_RE.match(ch):
                    out.append(ch)
                index += 1
                continue
            index += 1
        if pending_geminate:
            if romaji.startswith("ch"):
                romaji = "t" + romaji
            elif romaji and romaji[0] in "kgsztdhbpmyrl":
                romaji = romaji[0] + romaji
            pending_geminate = False
        out.append(romaji)
    return "".join(out)


def _loose_romaji(value: str) -> str:
    text = _kana_to_romaji(value)
    text = re.sub(r"([aeiou])\1+", r"\1", text)
    text = text.replace("ou", "o").replace("oo", "o")
    text = re.sub(r"([aeiou])\1+", r"\1", text)
    return text


def romaji_keys(value: str) -> set[str]:
    text = fold_text(value)
    if not _KANA_RE.search(text):
        return set()
    keys = {
        normalize_latin(_kana_to_romaji(text)),
        normalize_latin(_loose_romaji(text)),
    }
    return {key for key in keys if len(key) >= 2}


@lru_cache(maxsize=16384)
def _pinyin_options(value: str) -> tuple[tuple[str, ...], ...]:
    """All readings of one Han run: per character, tone and ü folded."""
    try:
        from pypinyin import Style, pinyin
    except ImportError:
        return ()
    options: list[tuple[str, ...]] = []
    for item in pinyin(value, style=Style.NORMAL, heteronym=True):
        values = [item] if isinstance(item, str) else list(item)
        clean = [normalize_latin(entry) for entry in values]
        clean = [entry for entry in clean if entry]
        if clean:
            options.append(tuple(dict.fromkeys(clean)))
    return tuple(options)


def _pinyin_reading(value: str) -> str:
    """Most common pinyin reading of one Han run."""
    return "".join(option[0] for option in _pinyin_options(value))


def pinyin_keys(value: str) -> frozenset[str]:
    text = fold_text(value)
    if not _HAN_RE.search(text) or _KANA_RE.search(text):
        return frozenset()
    options = _pinyin_options(text)
    if not options:
        return frozenset()
    combinations = 1
    for option in options:
        combinations *= len(option)
    if combinations > 32:
        options = tuple((option[0],) for option in options)
    keys: set[str] = set()
    for combination in product(*options):
        joined = "".join(combination)
        keys.add(joined)
        keys.add(joined.replace("v", "u"))
    return frozenset(key for key in keys if len(key) >= 2)


def _script_reading(value: str) -> str:
    """Romanize a folded string run by run, keeping the original order."""

    def replace(match: re.Match[str]) -> str:
        chunk = match.group()
        if _KANA_RUN.fullmatch(chunk):
            return _kana_to_romaji(chunk)
        if _HAN_RUN.fullmatch(chunk):
            return _pinyin_reading(chunk)
        return chunk

    # Drop separators first: "旧:potg" must not trip the legacy "u:" → "v" rule.
    romanized = re.sub(r"[^0-9A-Za-z]+", "", _SCRIPT_RUN.sub(replace, value))
    return normalize_latin(romanized)


def transliteration_keys(value: str) -> set[str]:
    """Latin keys (pinyin and/or romaji) for one Chinese/Japanese field.

    Mixed fields keep both halves: ``花宮なつか`` yields ``natsuka`` from the
    kana, ``huagong`` from the kanji, and the combined ``huagongnatsuka``.
    """
    text = fold_text(str(value or "")).strip()
    if not text:
        return set()
    keys: set[str] = set()
    for run in _KANA_RUN.findall(text):
        keys.add(normalize_latin(_kana_to_romaji(run)))
        keys.add(normalize_latin(_loose_romaji(run)))
    for run in _HAN_RUN.findall(text):
        keys.update(pinyin_keys(run))
    keys.add(_script_reading(text))
    keys.discard("")
    return {key for key in keys if len(key) >= 2}


@lru_cache(maxsize=8192)
def query_lookup_key(value: str) -> str:
    return fold_key(value)


@lru_cache(maxsize=16384)
def lookup_keys(value: str) -> frozenset[str]:
    """Latin/folded keys used to compare one query against alias keys."""
    text = str(value or "")
    folded = fold_key(text)
    keys = {normalize_latin(text), folded}
    keys.update(transliteration_keys(text))
    if "v" in folded:
        keys.add(folded.replace("v", "u"))
    return frozenset(key for key in keys if len(key) >= 2)
