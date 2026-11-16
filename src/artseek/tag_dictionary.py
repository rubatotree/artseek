from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TagEntry:
    tag: str
    zh: str
    aliases: tuple[str, ...] = ()


# Curated seed entries. The dictionary is deliberately additive: unknown WD tags
# still work, but these entries make common Chinese/alias lookup and display
# useful without an online translation service.
_TAG_ENTRIES: tuple[TagEntry, ...] = (
    TagEntry("long_hair", "长发", ("长发", "长头发", "long hair")),
    TagEntry("short_hair", "短发", ("短发", "short hair")),
    TagEntry("blue_eyes", "蓝眼", ("蓝眼", "蓝色眼睛", "blue eyes")),
    TagEntry("red_eyes", "红眼", ("红眼", "红色眼睛", "red eyes")),
    TagEntry("green_eyes", "绿眼", ("绿眼", "绿色眼睛", "green eyes")),
    TagEntry("blonde_hair", "金发", ("金发", "blonde hair")),
    TagEntry("black_hair", "黑发", ("黑发", "black hair")),
    TagEntry("white_hair", "白发", ("白发", "银发", "white hair")),
    TagEntry("twintails", "双马尾", ("双马尾", "twintails")),
    TagEntry("ponytail", "马尾", ("马尾", "ponytail")),
    TagEntry("cat_ears", "猫耳", ("猫耳", "猫耳朵", "cat ears")),
    TagEntry("animal_ears", "兽耳", ("兽耳", "animal ears")),
    TagEntry("school_uniform", "校服", ("校服", "制服", "school uniform")),
    TagEntry("serafuku", "水手服", ("水手服", "sailor uniform", "serafuku")),
    TagEntry("maid", "女仆", ("女仆", "女仆装", "maid")),
    TagEntry("dress", "连衣裙", ("连衣裙", "裙子", "dress")),
    TagEntry("kimono", "和服", ("和服", "kimono")),
    TagEntry("swimsuit", "泳装", ("泳装", "泳衣", "swimsuit")),
    TagEntry("lolita_fashion", "洛丽塔", ("洛丽塔", "lolita", "lolita fashion")),
    TagEntry("hair_ornament", "发饰", ("发饰", "头饰", "hair ornament")),
    TagEntry("ribbon", "蝴蝶结", ("蝴蝶结", "ribbon")),
    TagEntry("gloves", "手套", ("手套", "gloves")),
    TagEntry("thighhighs", "过膝袜", ("过膝袜", "大腿袜", "thighhighs")),
    TagEntry("sitting", "坐姿", ("坐姿", "坐着", "sitting")),
    TagEntry("standing", "站姿", ("站姿", "standing")),
    TagEntry("looking_at_viewer", "看向观众", ("看向观众", "看镜头", "looking at viewer")),
    TagEntry("smile", "微笑", ("微笑", "笑容", "smile")),
    TagEntry("blush", "脸红", ("脸红", "blush")),
    TagEntry("landscape", "风景", ("风景", "景色", "landscape", "scenery")),
    TagEntry("city", "城市", ("城市", "city")),
    TagEntry("sky", "天空", ("天空", "sky")),
    TagEntry("flower", "花", ("花", "花朵", "flower")),
    TagEntry("1girl", "单人女性", ("单人女性", "一个女孩", "1girl")),
    TagEntry("1boy", "单人男性", ("单人男性", "一个男孩", "1boy")),
    TagEntry("solo", "单人", ("单人", "solo")),
    TagEntry("original", "原创", ("原创", "original")),
)

_ENTRIES = {entry.tag: entry for entry in _TAG_ENTRIES}


def tag_info(tag: str) -> TagEntry | None:
    return _ENTRIES.get(str(tag or "").strip().lower().replace(" ", "_"))


def search_tags(query: str, *, limit: int = 40) -> list[TagEntry]:
    needle = str(query or "").strip().casefold()
    if not needle:
        return list(_TAG_ENTRIES)[: max(1, int(limit))]
    result: list[TagEntry] = []
    for entry in _TAG_ENTRIES:
        fields = (entry.tag, entry.zh, *entry.aliases)
        if any(needle in value.casefold() for value in fields):
            result.append(entry)
    return result[: max(1, int(limit))]
