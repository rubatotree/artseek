"""Curated and generated Danbooru/illustration tag dictionary lookup."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .transliterate import lookup_keys, transliteration_keys


@dataclass(frozen=True)
class TagEntry:
    tag: str
    zh: str
    aliases: tuple[str, ...] = ()


_TAG_ENTRIES: tuple[TagEntry, ...] = (
    TagEntry("long_hair", "长发", ("长发", "长头发", "ロングヘア", "long hair")),
    TagEntry("short_hair", "短发", ("短发", "ショートヘア", "short hair")),
    TagEntry(
        "blue_eyes",
        "蓝眼",
        ("蓝眼", "蓝色眼睛", "青い目", "あおいめ", "aoi me", "blue eyes"),
    ),
    TagEntry("red_eyes", "红眼", ("红眼", "红色眼睛", "red eyes")),
    TagEntry("green_eyes", "绿眼", ("绿眼", "绿色眼睛", "green eyes")),
    TagEntry("blonde_hair", "金发", ("金发", "blonde hair")),
    TagEntry("black_hair", "黑发", ("黑发", "black hair")),
    TagEntry("white_hair", "白发", ("白发", "银发", "white hair")),
    TagEntry(
        "blue_hair",
        "蓝发",
        ("蓝发", "蓝色头发", "蓝头发", "青い髪", "あおいかみ", "aoi kami", "blue hair"),
    ),
    TagEntry("red_hair", "红发", ("红发", "红色头发", "赤发", "red hair")),
    TagEntry("green_hair", "绿发", ("绿发", "绿色头发", "green hair")),
    TagEntry("purple_hair", "紫发", ("紫发", "紫色头发", "purple hair")),
    TagEntry("pink_hair", "粉发", ("粉发", "粉色头发", "粉红色头发", "pink hair")),
    TagEntry("brown_hair", "棕发", ("棕发", "棕色头发", "茶色头发", "brown hair")),
    TagEntry("twintails", "双马尾", ("双马尾", "ツインテール", "twintails")),
    TagEntry("ponytail", "马尾", ("马尾", "ponytail")),
    TagEntry("cat_ears", "猫耳", ("猫耳", "猫耳朵", "猫耳", "cat ears")),
    TagEntry("animal_ears", "兽耳", ("兽耳", "animal ears")),
    TagEntry("school_uniform", "校服", ("校服", "制服", "学校制服", "school uniform")),
    TagEntry(
        "serafuku",
        "水手服",
        ("水手服", "セーラー服", "せーらーふく", "seeraafuku", "sailor uniform", "serafuku"),
    ),
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
    # Mediums & attributes projected from vision analyzers
    TagEntry(
        "medium:illustration",
        "插画媒介",
        ("插画", "cg", "板绘", "数字插画", "illustration", "medium:illustration"),
    ),
    TagEntry("medium:comic", "漫画媒介", ("漫画", "黑白漫", "manga", "comic", "medium:comic")),
    TagEntry(
        "medium:bangumi",
        "动画截图",
        ("动画截图", "番剧截图", "动画", "bangumi", "anime screenshot", "medium:bangumi"),
    ),
    TagEntry("medium:3d", "3D渲染", ("3d", "3D", "mmd", "blender", "medium:3d")),
    TagEntry("rating:safe", "全年龄", ("全年龄", "清水", "一般向", "safe", "rating:safe")),
    TagEntry("rating:r15", "R15", ("r15", "R15", "擦边", "sensitive", "rating:r15")),
    TagEntry("rating:r18", "R18", ("r18", "R18", "nsfw", "explicit", "rating:r18")),
    TagEntry("monochrome", "黑白线稿", ("黑白", "单色", "线稿", "素描", "monochrome", "lineart")),
    TagEntry("ai_generated", "AI生成", ("ai生成", "ai绘图", "ai_generated", "ai")),
    TagEntry("hand_drawn", "手绘非AI", ("手绘", "人类绘制", "非ai", "hand_drawn")),
    TagEntry("polished", "精绘完成", ("精绘", "完成度高", "精良", "polished")),
    TagEntry("rough", "草稿速涂", ("草稿", "速涂", "涂鸦", "sketch", "rough")),
    # Popular IPs, characters & poses
    TagEntry("genshin_impact", "原神", ("原神", "genshin", "genshin_impact")),
    TagEntry(
        "honkai:_star_rail",
        "崩坏：星穹铁道",
        ("星穹铁道", "崩坏星穹铁道", "星铁", "hsr", "honkai:_star_rail"),
    ),
    TagEntry("zenless_zone_zero", "绝区零", ("绝区零", "zzz", "zenless_zone_zero")),
    TagEntry("arknights", "明日方舟", ("明日方舟", "方舟", "arknights")),
    TagEntry("touhou", "东方Project", ("东方", "东方project", "touhou")),
    TagEntry("fate/grand_order", "FGO", ("fgo", "fate", "fate/grand_order")),
    TagEntry("hatsune_miku", "初音未来", ("初音未来", "初音", "miku", "hatsune_miku")),
    TagEntry("rem_(re:zero)", "雷姆", ("雷姆", "rem", "rem_(re:zero)")),
    TagEntry("back_view", "背影", ("背影", "背面", "from_behind", "back_view")),
    TagEntry("looking_back", "回头", ("回头", "回眸", "looking_back")),
    TagEntry("side_profile", "侧脸", ("侧脸", "侧面", "profile", "side_profile")),
    TagEntry("hugging", "拥抱", ("拥抱", "抱抱", "hug", "hugging")),
    TagEntry("jumping", "跳跃", ("跳跃", "跃起", "jumping")),
)

_ENTRIES = {entry.tag: entry for entry in _TAG_ENTRIES}


def _read_translation_file(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _load_generated_translations() -> list[TagEntry]:
    pkg_dir = Path(__file__).resolve().parents[1]
    repo_root = Path(__file__).resolve().parents[3]
    raw: dict[str, object] = {}
    raw.update(_read_translation_file(pkg_dir / "tag_translations.json"))
    raw.update(
        _read_translation_file(repo_root / ".local-memory" / "tag_translations.local.json")
    )
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

_FUZZY_INDEX: list[tuple[TagEntry, tuple[str, ...]]] | None = None


def _fuzzy_index() -> list[tuple[TagEntry, tuple[str, ...]]]:
    global _FUZZY_INDEX
    if _FUZZY_INDEX is None:
        indexed: list[tuple[TagEntry, tuple[str, ...]]] = []
        for entry in (*_TAG_ENTRIES, *_GENERATED_ENTRIES):
            keys: set[str] = set()
            for value in (entry.tag, entry.zh, *entry.aliases):
                keys.update(transliteration_keys(value))
            if keys:
                indexed.append((entry, tuple(sorted(keys))))
        _FUZZY_INDEX = indexed
    return _FUZZY_INDEX


def fuzzy_tag_entries(value: str, *, limit: int = 8) -> list[TagEntry]:
    """Find entries by pinyin/romaji lookup keys, best match first."""
    needles = lookup_keys(value)
    if not needles:
        return []
    scored: list[tuple[int, int, TagEntry]] = []
    for entry, keys in _fuzzy_index():
        best = 0
        best_length = 10_000
        for key in keys:
            for needle in needles:
                if key == needle:
                    score = 3
                elif key.startswith(needle):
                    score = 2
                elif needle in key:
                    score = 1
                else:
                    continue
                if score > best or (score == best and len(key) < best_length):
                    best = score
                    best_length = len(key)
        if best:
            scored.append((best, best_length, entry))
    scored.sort(key=lambda item: (-item[0], item[1], item[2].tag))
    result: list[TagEntry] = []
    seen: set[str] = set()
    for _, _, entry in scored:
        if entry.tag in seen:
            continue
        seen.add(entry.tag)
        result.append(entry)
        if len(result) >= max(1, int(limit)):
            break
    return result


def _fuzzy_fields(value: str, *, limit: int = 6) -> set[str] | None:
    matches = fuzzy_tag_entries(value, limit=limit)
    if not matches:
        return None
    fields: set[str] = set()
    for entry in matches:
        fields.add(entry.tag)
        fields.add(entry.zh)
        fields.update(entry.aliases)
    return fields



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
    fuzzy = _fuzzy_fields(value)
    if fuzzy:
        return fuzzy
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
    generated = _GENERATED_BY_TAG.get(text)
    if generated is not None:
        return generated
    fuzzy = fuzzy_tag_entries(tag, limit=1)
    return fuzzy[0] if fuzzy else None


def search_tags(query: str, *, limit: int = 40) -> list[TagEntry]:
    needle = str(query or "").strip().casefold()
    entries = [*_TAG_ENTRIES, *_GENERATED_ENTRIES]
    if not needle:
        return entries[: max(1, int(limit))]
    result: list[TagEntry] = []
    seen: set[str] = set()
    fuzzy_matches = fuzzy_tag_entries(query, limit=limit)
    fuzzy_tags = {entry.tag for entry in fuzzy_matches}
    for entry in (*fuzzy_matches, *entries):
        if entry.tag in seen:
            continue
        fields = (entry.tag, entry.zh, *entry.aliases)
        include = entry.tag in fuzzy_tags or any(
            needle in value.casefold() for value in fields
        )
        if include:
            seen.add(entry.tag)
            result.append(entry)
        if len(result) >= max(1, int(limit)):
            break
    return result

