from __future__ import annotations

import unittest

from artseek.web.app import FolderScopeRequest


class FolderScopeRequestTests(unittest.TestCase):
    def test_null_sort_and_selected_items_are_coerced(self) -> None:
        model = FolderScopeRequest.model_validate(
            {
                "sort": None,
                "selected": [None, "D:/pics/a", "", {"key": "D:/pics/b"}],
                "mtime_after": None,
                "mtime_before": "",
            }
        )
        self.assertEqual(model.sort, "name")
        self.assertEqual(model.selected, ["D:/pics/a", "D:/pics/b"])
        self.assertIsNone(model.mtime_after)
        self.assertIsNone(model.mtime_before)

    def test_unknown_sort_falls_back(self) -> None:
        model = FolderScopeRequest.model_validate({"sort": "nope", "selected": []})
        self.assertEqual(model.sort, "name")
