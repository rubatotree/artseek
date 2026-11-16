from __future__ import annotations

import unittest

import numpy as np

from artseek.store import VectorStore


class NeighborTagVotesTests(unittest.TestCase):
    def test_excludes_target_and_ranks_neighbor_tags(self) -> None:
        point_ids = ["p0", "p1", "p2"]
        matrix = np.asarray([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        tags = [["self"], ["same"], ["orthogonal"]]
        votes = VectorStore.neighbor_tag_votes_from_data(
            (point_ids, matrix, tags), ["p0"], limit=2, top_tags=5
        )
        self.assertEqual(votes["p0"][0][0], "same")
        self.assertNotIn("self", votes["p0"][0])
        self.assertIn("orthogonal", votes["p0"][0])
        self.assertEqual(votes["p0"][1]["same"], 1.0)

    def test_unknown_target_is_skipped(self) -> None:
        data = (["p0"], np.asarray([[1.0, 0.0]], dtype=np.float32), [["tag"]])
        self.assertEqual(
            VectorStore.neighbor_tag_votes_from_data(data, ["missing"]), {}
        )


if __name__ == "__main__":
    unittest.main()
