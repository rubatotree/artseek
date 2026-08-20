from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import torch

from artseek.models.embedder import SiglipEmbedder, pooled_features


class PooledFeaturesTests(unittest.TestCase):
    def test_tensor_passthrough(self) -> None:
        features = torch.tensor([[3.0, 4.0]])
        self.assertIs(pooled_features(features), features)

    def test_pooler_output_object(self) -> None:
        pooled = torch.tensor([[1.0, 0.0, 0.0]])
        output = SimpleNamespace(pooler_output=pooled, last_hidden_state=torch.zeros(1, 4, 3))
        self.assertTrue(torch.equal(pooled_features(output), pooled))

    def test_tuple_uses_pooler_slot(self) -> None:
        last = torch.zeros(1, 4, 2)
        pooled = torch.tensor([[0.0, 5.0]])
        self.assertTrue(torch.equal(pooled_features((last, pooled)), pooled))

    def test_rejects_unknown_payload(self) -> None:
        with self.assertRaises(TypeError):
            pooled_features({"embedding": torch.ones(2)})


class NormalizeTests(unittest.TestCase):
    def test_normalize_tensor(self) -> None:
        out = SiglipEmbedder._normalize(torch.tensor([[3.0, 4.0]]))
        np.testing.assert_allclose(out, np.array([[0.6, 0.8]], dtype=np.float32))

    def test_normalize_transformers_v5_output(self) -> None:
        output = SimpleNamespace(
            pooler_output=torch.tensor([[0.0, 2.0]]),
            last_hidden_state=torch.zeros(1, 8, 2),
        )
        out = SiglipEmbedder._normalize(output)
        np.testing.assert_allclose(out, np.array([[0.0, 1.0]], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
