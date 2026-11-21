"""Pure natural-language query parsing and dictionary expansion."""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field
from typing import Any

from .tag_dictionary import fuzzy_tag_entries
from .transliterate import fold_text, lookup_keys


def normalize_tag(value: str) -> str:
    return re.sub(r"\s+", "_", value.strip().lower()).strip("_,")


def normalize_query(value: str) -> str:
    """Canonical key for persistent query plan caching."""
    return re.sub(r"\s+", " ", str(value or "").strip()).casefold()


ILLUSTRATION_LEXICON: dict[str, tuple[str, ...]] = {
    # 服装轮廓
    "洛丽塔": ("lolita_fashion", "frills", "lace"),
    "lolita": ("lolita_fashion", "frills", "lace"),
    "哥特洛丽塔": ("gothic_lolita", "black_dress", "lace"),
    "甜系洛丽塔": ("sweet_lolita", "pastel_colors", "frills"),
    "水手服": ("sailor_collar", "school_uniform"),
    "セーラー服": ("sailor_collar", "school_uniform"),
    "女仆装": ("maid", "maid_headdress", "apron"),
    "メイド服": ("maid", "maid_headdress", "apron"),
    "护士服": ("nurse", "nurse_cap"),
    "睡衣": ("pajamas", "sleepwear"),
    "睡裙": ("nightgown", "sleepwear"),
    "居家服": ("loungewear", "casual"),
    "连衣裙": ("dress",),
    "吊带裙": ("sundress", "spaghetti_straps"),
    "娃娃裙": ("babydoll", "short_dress"),
    "背带裙": ("jumper_skirt",),
    "罩衫": ("smock", "loose_clothes"),
    "衬衫": ("shirt",),
    "短上衣": ("crop_top", "midriff"),
    "露脐装": ("crop_top", "midriff"),
    "超大上衣": ("oversized_clothes",),
    "大码卫衣": ("oversized_hoodie",),
    "连帽衫": ("hoodie",),
    "针织衫": ("sweater", "knit"),
    "开衫": ("cardigan",),
    "高领": ("turtleneck",),
    "短裙": ("short_skirt",),
    "百褶裙": ("pleated_skirt",),
    "蓬蓬裙": ("puffy_skirt",),
    "纱裙": ("tulle_skirt",),
    "南瓜裤": ("bloomers",),
    "灯笼裤": ("pumpkin_shorts",),
    "短裤": ("shorts",),
    "热裤": ("short_shorts",),
    "背带裤": ("overalls",),
    "泳装": ("swimsuit",),
    "泳衣": ("swimsuit",),
    "比基尼": ("bikini",),
    "连体泳衣": ("one-piece_swimsuit",),
    "死库水": ("school_swimsuit",),
    "スクール水着": ("school_swimsuit",),
    "运动服": ("sportswear",),
    "体操服": ("gym_uniform",),
    "ジャージ": ("track_suit",),
    # 领口、袖子、布料与装饰
    "露肩": ("off_shoulder", "bare_shoulders"),
    "一字肩": ("off_shoulder",),
    "挖肩": ("shoulder_cutout",),
    "吊带": ("spaghetti_straps",),
    "无袖": ("sleeveless",),
    "泡泡袖": ("puffy_sleeves",),
    "姬袖": ("wide_sleeves", "detached_sleeves"),
    "喇叭袖": ("wide_sleeves",),
    "灯笼袖": ("puffy_sleeves", "long_sleeves"),
    "超长袖": ("sleeves_past_wrists",),
    "萌袖": ("sleeves_past_wrists",),
    "袖套": ("detached_sleeves",),
    "荷叶边": ("frills",),
    "褶边": ("frills",),
    "蕾丝": ("lace",),
    "薄纱": ("sheer_fabric",),
    "透明布料": ("see-through", "sheer_fabric"),
    "镂空": ("cutout",),
    "绑带": ("lace-up", "straps"),
    "蝴蝶结": ("bow",),
    "大蝴蝶结": ("large_bow",),
    "胸前蝴蝶结": ("bowtie",),
    "腰带": ("belt",),
    "束腰": ("corset",),
    "围裙": ("apron",),
    "彼得潘领": ("peter_pan_collar",),
    "娃娃领": ("peter_pan_collar",),
    "水手领": ("sailor_collar",),
    "大翻领": ("wide_collar",),
    # 腿部、鞋袜、配饰
    "过膝袜": ("thighhighs",),
    "长筒袜": ("thighhighs",),
    "膝袜": ("kneehighs",),
    "短袜": ("socks",),
    "泡泡袜": ("loose_socks",),
    "堆堆袜": ("leg_warmers",),
    "腿套": ("leg_warmers",),
    "连裤袜": ("pantyhose",),
    "网袜": ("fishnet_legwear",),
    "吊袜带": ("garter_straps",),
    "腿环": ("thigh_strap",),
    "赤脚": ("barefoot",),
    "裸足": ("barefoot",),
    "厚底鞋": ("platform_footwear",),
    "玛丽珍": ("mary_janes",),
    "乐福鞋": ("loafers",),
    "芭蕾鞋": ("ballet_flats",),
    "凉鞋": ("sandals",),
    "创可贴": ("bandage",),
    "发带": ("hairband",),
    "发箍": ("headband",),
    "发夹": ("hairclip",),
    "发饰": ("hair_ornament",),
    "头纱": ("veil",),
    "猫耳": ("cat_ears",),
    "脐钉": ("navel_piercing",),
    # 动作与身体朝向
    "站立": ("standing",),
    "坐着": ("sitting",),
    "跪坐": ("kneeling", "seiza"),
    "跪着": ("kneeling",),
    "蹲着": ("squatting",),
    "蜷缩": ("curled_up",),
    "抱膝": ("knees_to_chest", "hugging_own_legs"),
    "盘腿": ("crossed_legs", "sitting"),
    "翘腿": ("crossed_legs",),
    "侧躺": ("lying_on_side",),
    "仰躺": ("on_back", "lying"),
    "趴着": ("on_stomach", "lying"),
    "躺着": ("lying",),
    "伸懒腰": ("stretching",),
    "抬手": ("arm_up",),
    "双手举起": ("arms_up",),
    "手放脑后": ("hands_behind_head",),
    "托腮": ("hand_on_own_cheek",),
    "捂嘴": ("hand_over_mouth",),
    "摸头发": ("hand_in_hair",),
    "拉袖子": ("sleeve_pull",),
    "提裙子": ("skirt_hold",),
    "回头": ("looking_back",),
    "低头": ("looking_down",),
    "仰头": ("looking_up",),
    "侧脸": ("profile", "side_profile"),
    "背对镜头": ("from_behind",),
    "走路": ("walking",),
    "奔跑": ("running",),
    "跳跃": ("jumping",),
    "踮脚": ("tiptoes",),
    # 构图与镜头
    "全身": ("full_body",),
    "半身": ("upper_body",),
    "胸像": ("portrait", "upper_body"),
    "特写": ("close-up",),
    "脸部特写": ("face_focus", "close-up"),
    "脚部特写": ("feet_focus",),
    "腿部特写": ("leg_focus",),
    "俯视": ("from_above",),
    "仰视": ("from_below",),
    "平视": ("straight-on",),
    "广角": ("wide_angle",),
    "鱼眼": ("fisheye",),
    "荷兰角": ("dutch_angle",),
    "对称构图": ("symmetrical_composition",),
    "留白": ("simple_background", "negative_space"),
    "动态透视": ("dynamic_pose", "foreshortening"),
    "大透视": ("foreshortening",),
    # 风格、光色与情绪
    "梦幻可爱": ("yume_kawaii", "pastel_colors", "dreamy"),
    "水色": ("watercolor", "pastel_colors"),
    "水彩": ("watercolor",),
    "病态可爱": ("yami_kawaii", "pastel_colors"),
    "量产型": ("jirai_kei", "frills", "ribbon"),
    "地雷系": ("jirai_kei",),
    "亚文化": ("subculture_fashion",),
    "赛博": ("cyberpunk", "neon_lights"),
    "柔和": ("soft_colors", "soft_lighting"),
    "低饱和": ("muted_colors",),
    "粉彩": ("pastel_colors",),
    "单色": ("monochrome",),
    "黑白": ("monochrome",),
    "线稿": ("monochrome", "lineart"),
    "草稿": ("rough", "sketch"),
    "速涂": ("rough",),
    "精绘": ("polished",),
    "插画": ("medium:illustration",),
    "漫画": ("medium:comic",),
    "动画截图": ("medium:bangumi",),
    "番剧截图": ("medium:bangumi",),
    "3d渲染": ("medium:3d",),
    "实拍": ("medium:not_painting",),
    "照片": ("medium:not_painting",),
    "全年龄": ("rating:safe",),
    "r15": ("rating:r15",),
    "r18": ("rating:r18",),
    "ai生成": ("ai_generated",),
    "手绘": ("hand_drawn",),
    "背影": ("back_view", "from_behind"),
    "拥抱": ("hugging",),
    "逆光": ("backlighting",),
    "轮廓光": ("rim_light",),
    "室内": ("indoors",),
    "卧室": ("bedroom",),
    "床上": ("on_bed",),
    "夏日": ("summer",),
    "冬日": ("winter",),
}

_LEXICON_FUZZY: dict[str, tuple[str, ...]] | None = None


def _lexicon_fuzzy_tags(value: str, *, limit: int = 12) -> list[str]:
    global _LEXICON_FUZZY
    needles = lookup_keys(value)
    if not needles:
        return []
    if _LEXICON_FUZZY is None:
        index: dict[str, set[str]] = {}
        for alias, tags in ILLUSTRATION_LEXICON.items():
            for key in lookup_keys(alias):
                index.setdefault(key, set()).update(tags)
        _LEXICON_FUZZY = {
            key: tuple(sorted(tags)) for key, tags in index.items()
        }
    scored: dict[str, int] = {}
    for key, tags in _LEXICON_FUZZY.items():
        best = 0
        for needle in needles:
            if key == needle:
                best = max(best, 3)
            elif key.startswith(needle):
                best = max(best, 2)
            elif needle in key:
                best = max(best, 1)
        if not best:
            continue
        for tag in tags:
            scored[tag] = max(scored.get(tag, 0), best)
    ranked = sorted(scored.items(), key=lambda item: (-item[1], item[0]))
    return [tag for tag, _ in ranked[: max(1, int(limit))]]


@dataclass(slots=True)
class QueryPlan:
    original: str
    semantic_zh: str
    semantic_en: str = ""
    positive_tags: list[str] = field(default_factory=list)
    negative_tags: list[str] = field(default_factory=list)
    used_llm: bool = False

    @property
    def embedding_text(self) -> str:
        parts = [self.semantic_zh, self.semantic_en]
        if self.positive_tags:
            parts.append(", ".join(tag.replace("_", " ") for tag in self.positive_tags))
        return ". ".join(part for part in parts if part)

    def to_dict(self) -> dict[str, Any]:
        return {
            "original": self.original,
            "semantic_zh": self.semantic_zh,
            "semantic_en": self.semantic_en,
            "positive_tags": list(self.positive_tags),
            "negative_tags": list(self.negative_tags),
            "used_llm": bool(self.used_llm),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QueryPlan:
        return cls(
            original=str(data.get("original") or ""),
            semantic_zh=str(data.get("semantic_zh") or ""),
            semantic_en=str(data.get("semantic_en") or ""),
            positive_tags=_unique(list(data.get("positive_tags") or [])),
            negative_tags=_unique(list(data.get("negative_tags") or [])),
            used_llm=bool(data.get("used_llm", False)),
        )


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(filter(None, (normalize_tag(v) for v in values))))


def static_expand(query: str) -> QueryPlan:
    positive: list[str] = []
    negative: list[str] = []
    lowered = fold_text(query)

    try:
        tokens = shlex.split(query)
    except ValueError:
        tokens = query.split()

    for token in tokens:
        if token.startswith("-") and len(token) > 1:
            term = token[1:]
            mapped = ILLUSTRATION_LEXICON.get(term, (term,))
            negative.extend(mapped)
            negative.extend(entry.tag for entry in fuzzy_tag_entries(term))
            negative.extend(_lexicon_fuzzy_tags(term))
        elif re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_ -]{1,50}", token):
            positive.append(token)

    negative_match = re.search(
        r"(?:不要|排除|不含|没有|无)\s*([A-Za-z0-9][A-Za-z0-9_ -]*)",
        query,
    ) or re.search(
        r"(?:不要|排除|不含|没有)\s*([^\s，,。;；]+)",
        query,
    )
    negative_text = negative_match.group(1).strip() if negative_match else ""
    if negative_text:
        negative.extend(entry.tag for entry in fuzzy_tag_entries(negative_text))
        negative.extend(_lexicon_fuzzy_tags(negative_text))

    for alias, tags in ILLUSTRATION_LEXICON.items():
        alias_key = fold_text(alias)
        if alias_key in lowered:
            negative_context = re.search(
                rf"(?:(?:不要|排除|不含|没有|无)\s*|(?:^|\s)-){re.escape(alias_key)}",
                lowered,
            )
            (negative if negative_context else positive).extend(tags)

    positive_text = re.sub(r"(?:^|\s)-\S+", " ", query)
    positive_text = re.sub(
        r"(?:不要|排除|不含|没有|无)\s*[A-Za-z0-9][A-Za-z0-9_ -]*", " ", positive_text
    )
    positive_text = re.sub(r"(?:不要|排除|不含|没有)\s*[^\s，,。;；]+", " ", positive_text)
    if positive_text.strip():
        positive.extend(entry.tag for entry in fuzzy_tag_entries(positive_text))
        positive.extend(_lexicon_fuzzy_tags(positive_text))

    return QueryPlan(
        original=query,
        semantic_zh=query,
        positive_tags=_unique(positive),
        negative_tags=_unique(negative),
    )


def merge_llm_plan(base: QueryPlan, data: dict[str, Any]) -> QueryPlan:
    return QueryPlan(
        original=base.original,
        semantic_zh=str(data.get("semantic_zh") or base.semantic_zh),
        semantic_en=str(data.get("semantic_en") or ""),
        positive_tags=_unique(base.positive_tags + list(data.get("positive_tags") or [])),
        negative_tags=_unique(base.negative_tags + list(data.get("negative_tags") or [])),
        used_llm=True,
    )
