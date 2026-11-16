from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TagEntry:
    tag: str
    zh: str
    aliases: tuple[str, ...] = ()


# Curated seed entries. The dictionary is deliberately additive: unknown WD tags
# still work, but these entries make common Chinese/alias lookup and display
# useful without an online translation service.
_TAG_ENTRIES: tuple[TagEntry, ...] = (
    TagEntry("long_hair", "长发", ("长发", "长头发", "ロングヘア", "long hair")),
    TagEntry("short_hair", "短发", ("短发", "ショートヘア", "short hair")),
    TagEntry("blue_eyes", "蓝眼", ("蓝眼", "蓝色眼睛", "青い目", "blue eyes")),
    TagEntry("red_eyes", "红眼", ("红眼", "红色眼睛", "red eyes")),
    TagEntry("green_eyes", "绿眼", ("绿眼", "绿色眼睛", "green eyes")),
    TagEntry("blonde_hair", "金发", ("金发", "blonde hair")),
    TagEntry("black_hair", "黑发", ("黑发", "black hair")),
    TagEntry("white_hair", "白发", ("白发", "银发", "white hair")),
    TagEntry("twintails", "双马尾", ("双马尾", "ツインテール", "twintails")),
    TagEntry("ponytail", "马尾", ("马尾", "ponytail")),
    TagEntry("cat_ears", "猫耳", ("猫耳", "猫耳朵", "猫耳", "cat ears")),
    TagEntry("animal_ears", "兽耳", ("兽耳", "animal ears")),
    TagEntry("school_uniform", "校服", ("校服", "制服", "学校制服", "school uniform")),
    TagEntry("serafuku", "水手服", ("水手服", "セーラー服", "sailor uniform", "serafuku")),
    TagEntry("maid", "女仆", ("女仆", "女仆装", "maid")),
    TagEntry("dress", "连衣裙", ("连衣裙", "裙子", "dress")),
    TagEntry("kimono", "和服", ("和服", "着物", "kimono")),
    TagEntry("swimsuit", "泳装", ("泳装", "泳衣", "水着", "swimsuit")),
    TagEntry("lolita_fashion", "洛丽塔", ("洛丽塔", "lolita", "lolita fashion")),
    TagEntry("hair_ornament", "发饰", ("发饰", "头饰", "hair ornament")),
    TagEntry("ribbon", "蝴蝶结", ("蝴蝶结", "ribbon")),
    TagEntry("gloves", "手套", ("手套", "gloves")),
    TagEntry("thighhighs", "过膝袜", ("过膝袜", "大腿袜", "thighhighs")),
    TagEntry("sitting", "坐姿", ("坐姿", "坐着", "sitting")),
    TagEntry("standing", "站姿", ("站姿", "standing")),
    TagEntry("looking_at_viewer", "看向观众", ("看向观众", "看镜头", "looking at viewer")),
    TagEntry("smile", "微笑", ("微笑", "笑容", "笑顔", "smile")),
    TagEntry("blush", "脸红", ("脸红", "blush")),
    TagEntry("landscape", "风景", ("风景", "景色", "landscape", "scenery")),
    TagEntry("city", "城市", ("城市", "city")),
    TagEntry("sky", "天空", ("天空", "sky")),
    TagEntry("flower", "花", ("花", "花朵", "flower")),
    TagEntry("1girl", "单人女性", ("单人女性", "一个女孩", "女孩", "少女", "女の子", "1girl")),
    TagEntry("1boy", "单人男性", ("单人男性", "一个男孩", "1boy")),
    TagEntry("solo", "单人", ("单人", "solo")),
    TagEntry("original", "原创", ("原创", "创作", "創作", "オリジナル", "original")),
    TagEntry("soles", "脚底", ("足裏", "脚底")),
    TagEntry("navel", "肚脐", ("おへそ", "肚脐")),
    TagEntry(
        "blue_archive",
        "蔚蓝档案",
        ("蔚蓝档案", "ブルアカ", "ブルーアーカイブ", "BlueArchive", "blue archive"),
    ),
    TagEntry("skeb", "Skeb 约稿", ("skeb", "Skeb", "约稿")),
    TagEntry("barefoot", "赤足", ("赤足", "裸足", "barefoot")),
    TagEntry("thighhighs", "过膝袜", ("过膝袜", "大腿袜", "thighhighs")),
)

_ENTRIES = {entry.tag: entry for entry in _TAG_ENTRIES}


def _load_generated_translations() -> list[TagEntry]:
    path = Path(__file__).with_name("tag_translations.json")
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    curated_fields = set()
    for entry in _TAG_ENTRIES:
        curated_fields.add(entry.tag.casefold())
        curated_fields.add(entry.zh.casefold())
        curated_fields.update(alias.casefold() for alias in entry.aliases)
    entries: list[TagEntry] = []
    for tag, zh in raw.items():
        text = str(tag or "").strip()
        display = str(zh or "").strip()
        if text and display and text.casefold() not in curated_fields:
            entries.append(TagEntry(text, display, (text,)))
    return entries


_GENERATED_ENTRIES = _load_generated_translations()
_GENERATED_BY_TAG = {entry.tag.casefold(): entry for entry in _GENERATED_ENTRIES}


def expand_tag_aliases(value: str) -> set[str]:
    needle = str(value or "").strip().casefold()
    if not needle:
        return set()
    for entry in _TAG_ENTRIES:
        fields = {entry.tag.casefold(), entry.zh.casefold()}
        fields.update(alias.casefold() for alias in entry.aliases)
        if needle in fields:
            return {entry.tag, entry.zh, *entry.aliases}
    generated = _GENERATED_BY_TAG.get(needle)
    if generated is not None:
        return {generated.tag, generated.zh}
    for entry in _GENERATED_ENTRIES:
        if entry.zh.casefold() == needle:
            return {entry.tag, entry.zh}
    return {str(value).strip().lower().replace(" ", "_")}


def tag_info(tag: str) -> TagEntry | None:
    text = str(tag or "").strip().casefold()
    direct = _ENTRIES.get(text.replace(" ", "_"))
    if direct is not None:
        return direct
    for entry in _TAG_ENTRIES:
        fields = {entry.tag.casefold(), entry.zh.casefold()}
        fields.update(alias.casefold() for alias in entry.aliases)
        if text in fields:
            return entry
    return _GENERATED_BY_TAG.get(text)


def search_tags(query: str, *, limit: int = 40) -> list[TagEntry]:
    needle = str(query or "").strip().casefold()
    entries = [*_TAG_ENTRIES, *_GENERATED_ENTRIES]
    if not needle:
        return entries[: max(1, int(limit))]
    result: list[TagEntry] = []
    for entry in entries:
        fields = (entry.tag, entry.zh, *entry.aliases)
        if any(needle in value.casefold() for value in fields):
            result.append(entry)
    return result[: max(1, int(limit))]
