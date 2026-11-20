"""Fuzzy pinyin/romaji lookup tests for the tag and query dictionaries."""

from __future__ import annotations

import unittest

from artseek.core.query_plan import static_expand
from artseek.core.tag_dictionary import expand_tag_aliases, search_tags


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


if __name__ == "__main__":
    unittest.main()
