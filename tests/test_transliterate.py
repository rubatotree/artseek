"""Fuzzy pinyin/romaji lookup tests for the tag and query dictionaries."""

from __future__ import annotations

import unittest

from artseek.core.name_match import build, score
from artseek.core.query_plan import static_expand
from artseek.core.tag_dictionary import expand_tag_aliases, search_tags
from artseek.core.transliterate import (
    fold_key,
    fold_text,
    lookup_keys,
    transliteration_keys,
)


class TransliterationSearchTests(unittest.TestCase):
    def test_chinese_pinyin_resolves_tag_aliases(self) -> None:
        self.assertIn("blue_hair", expand_tag_aliases("lanse"))
        self.assertIn("blue_hair", expand_tag_aliases("lansetoufa"))
        self.assertIn("thighhighs", expand_tag_aliases("guoxiwa"))
        self.assertIn("blue_archive", expand_tag_aliases("weilandangan"))

    def test_japanese_kana_romaji_resolves_tag_aliases(self) -> None:
        self.assertIn("long_hair", expand_tag_aliases("ronguhea"))
        self.assertIn("serafuku", expand_tag_aliases("seeraafuku"))
        self.assertIn("blue_hair", expand_tag_aliases("aoikami"))

    def test_static_plan_adds_fuzzy_lexicon_tags(self) -> None:
        self.assertIn("thighhighs", static_expand("guoxiwa").positive_tags)
        self.assertIn("wide_sleeves", static_expand("jixiu").positive_tags)
        self.assertIn("knees_to_chest", static_expand("baoxi").positive_tags)
        self.assertIn("blue_hair", static_expand("lanse toufa").positive_tags)

    def test_negative_pinyin_goes_to_negative_tags(self) -> None:
        plan = static_expand("不要 lanse")
        self.assertIn("blue_hair", plan.negative_tags)
        self.assertNotIn("blue_hair", plan.positive_tags)

    def test_tag_autocomplete_uses_pinyin(self) -> None:
        tags = [entry.tag for entry in search_tags("guoxiwa", limit=10)]
        self.assertIn("thighhighs", tags)


class ScriptFoldTests(unittest.TestCase):
    def test_simplified_traditional_and_variant_chars_fold_together(self) -> None:
        self.assertEqual(fold_key("宮"), fold_key("宫"))
        self.assertEqual(fold_key("長髮"), fold_key("长发"))
        self.assertEqual(fold_key("長髪"), fold_key("长发"))
        self.assertEqual(fold_key("澤"), fold_key("沢"))
        self.assertEqual(fold_key("髙橋"), fold_key("高桥"))

    def test_katakana_folds_to_hiragana(self) -> None:
        self.assertEqual(fold_text("ナツカ"), fold_text("なつか"))
        self.assertEqual(fold_text("ガオパペット"), fold_text("がおぱぺっと"))

    def test_fold_key_drops_punctuation_and_emoji(self) -> None:
        self.assertEqual(fold_key("ぴおてぐ（旧:potg）"), fold_key("ぴおてぐ 旧:potg"))
        self.assertEqual(fold_key("うろつき🦐"), "うろつき")

    def test_mixed_kanji_kana_name_keeps_both_romanizations(self) -> None:
        keys = transliteration_keys("花宮なつか")
        self.assertIn("natsuka", keys)
        self.assertIn("huagong", keys)
        self.assertIn("huagongnatsuka", keys)

    def test_query_keys_reach_variant_spellings(self) -> None:
        name_keys = transliteration_keys("花宮なつか")
        for query in ("花宫ナツカ", "natsuka", "huagong"):
            self.assertTrue(lookup_keys(query) & name_keys, query)

    def test_name_match_scores_variant_and_romaji_queries(self) -> None:
        target = build("花宮なつか")
        for query in ("花宫なつか", "花宮ナツカ", "natsuka", "huagong"):
            self.assertGreaterEqual(score(build(query), target), 0.85, query)
        self.assertEqual(score(build("宮坂みゆ"), target), 0.0)

    def test_traditional_tag_input_hits_simplified_aliases(self) -> None:
        self.assertIn("long_hair", expand_tag_aliases("長髮"))
        self.assertIn("blue_hair", expand_tag_aliases("藍色頭髮"))
        self.assertIn("thighhighs", static_expand("過膝襪").positive_tags)
        self.assertIn("blue_hair", static_expand("不要 藍髮").negative_tags)
        self.assertIn("long_hair", [entry.tag for entry in search_tags("長髮", limit=5)])


if __name__ == "__main__":
    unittest.main()
