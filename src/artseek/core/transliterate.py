"""Pinyin/romaji lookup keys for fuzzy Chinese and Japanese search."""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from itertools import product

_HAN_RE = re.compile(r"[\u3400-\u9fff]")
_KANA_RE = re.compile(r"[\u3040-\u30ff]")
_HIRA_RE = re.compile(r"[\u3041-\u3096\u309d-\u309f]")

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
}


def normalize_latin(value: str) -> str:
    """Lowercase, strip tones/punctuation and ü/v ambiguity for lookup keys."""
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.casefold().replace("ü", "v").replace("u:", "v")
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
    if not _KANA_RE.search(str(value or "")):
        return set()
    keys = {
        normalize_latin(_kana_to_romaji(value)),
        normalize_latin(_loose_romaji(value)),
    }
    return {key for key in keys if len(key) >= 2}


def pinyin_keys(value: str) -> set[str]:
    text = str(value or "")
    if not _HAN_RE.search(text) or _KANA_RE.search(text):
        return set()
    try:
        from pypinyin import Style, pinyin
    except ImportError:
        return set()
    syllables = pinyin(text, style=Style.NORMAL, heteronym=True)
    options: list[list[str]] = []
    for item in syllables:
        values = [item] if isinstance(item, str) else list(item)
        clean = [normalize_latin(value) for value in values]
        clean = [value for value in clean if value]
        if clean:
            options.append(clean)
    if not options:
        return set()
    combinations = 1
    for option in options:
        combinations *= len(option)
    if combinations > 32:
        options = [[option[0]] for option in options]
    keys: set[str] = set()
    for combination in product(*options):
        joined = "".join(combination)
        keys.add(joined)
        keys.add(joined.replace("v", "u"))
    return {key for key in keys if len(key) >= 2}


def transliteration_keys(value: str) -> set[str]:
    """Return normalized pinyin/romaji keys for one Chinese/Japanese field."""
    text = str(value or "").strip()
    if not text:
        return set()
    if _KANA_RE.search(text):
        return romaji_keys(text)
    if _HAN_RE.search(text):
        return pinyin_keys(text)
    return set()


@lru_cache(maxsize=4096)
def query_lookup_key(value: str) -> str:
    return normalize_latin(value)


def lookup_keys(value: str) -> set[str]:
    """Latin/romaji/pinyin keys used to compare one query against alias keys."""
    keys = {normalize_latin(value)}
    keys.update(transliteration_keys(value))
    return {key for key in keys if len(key) >= 2}
