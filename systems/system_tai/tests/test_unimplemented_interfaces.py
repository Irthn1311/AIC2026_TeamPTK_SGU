from __future__ import annotations

import unittest
from pathlib import Path

from system_tai.evaluation.kis_fixture import KISFixtureEvaluator


class ExplicitFailureTests(unittest.TestCase):
    def test_evaluator_is_explicitly_unimplemented(self) -> None:
        with self.assertRaises(NotImplementedError):
            KISFixtureEvaluator().evaluate(Path("predictions.jsonl"), Path("ground_truth.jsonl"))


if __name__ == "__main__":
    unittest.main()
