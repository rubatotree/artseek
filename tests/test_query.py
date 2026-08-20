from __future__ import annotations

import unittest

from artseek.query import merge_llm_plan, static_expand


class QueryExpansionTests(unittest.TestCase):
    def test_clothing_pose_and_composition(self) -> None:
        plan = static_expand("低饱和的姬袖少女，抱膝坐着，俯视")
        self.assertIn("wide_sleeves", plan.positive_tags)
        self.assertIn("knees_to_chest", plan.positive_tags)
        self.assertIn("from_above", plan.positive_tags)
        self.assertIn("muted_colors", plan.positive_tags)

    def test_negative_terms(self) -> None:
        plan = static_expand("水色连衣裙，不要过膝袜 -站立")
        self.assertIn("thighhighs", plan.negative_tags)
        self.assertIn("standing", plan.negative_tags)
        self.assertIn("dress", plan.positive_tags)

    def test_llm_merge_is_deduplicated(self) -> None:
        base = static_expand("泡泡袖")
        merged = merge_llm_plan(
            base,
            {
                "semantic_zh": "泡泡袖服装",
                "semantic_en": "clothing with puff sleeves",
                "positive_tags": ["puffy sleeves", "lace"],
                "negative_tags": [],
            },
        )
        self.assertEqual(merged.positive_tags.count("puffy_sleeves"), 1)
        self.assertTrue(merged.used_llm)


if __name__ == "__main__":
    unittest.main()

