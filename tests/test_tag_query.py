from __future__ import annotations

import unittest

from artseek.tag_query import (
    TagQueryError,
    evaluate_tag_expression_with_universe,
    parse_tag_expression,
)


class TagQueryTests(unittest.TestCase):
    def evaluate(self, expression: str, tags: dict[str, set[str]]) -> set[str]:
        universe = set().union(*tags.values()) if tags else set()
        node = parse_tag_expression(expression)
        return evaluate_tag_expression_with_universe(
            node, lambda atom: tags.get(atom, set()), universe
        )

    def test_boolean_operators_and_precedence(self) -> None:
        tags = {
            "a": {"1", "2"},
            "b": {"1"},
            "c": {"2"},
            "d": {"1", "2", "3"},
        }
        self.assertEqual(self.evaluate("(#a+#b*#c)", tags), {"1", "2"})
        self.assertEqual(self.evaluate("#a*#b-#c", tags), {"1"})
        self.assertEqual(self.evaluate("#a|#b", tags), {"1", "2"})
        self.assertEqual(self.evaluate("not #a", tags), {"3"})

    def test_word_operators_and_quoted_tags(self) -> None:
        tags = {"long hair": {"1"}, "blue eyes": {"2"}}
        self.assertEqual(self.evaluate('#"long hair" and #"blue eyes"', tags), set())
        self.assertEqual(self.evaluate('#"long hair" or #"blue eyes"', tags), {"1", "2"})

    def test_invalid_expression(self) -> None:
        with self.assertRaises(TagQueryError):
            parse_tag_expression("#a+")


if __name__ == "__main__":
    unittest.main()
