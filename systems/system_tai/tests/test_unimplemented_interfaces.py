from __future__ import annotations

import unittest
from pathlib import Path

from system_tai.evaluation.kis_fixture import KISFixtureEvaluator


class EvaluatorFileInterfaceTests(unittest.TestCase):
    def test_evaluator_raises_file_not_found_on_missing_inputs(self) -> None:
        with self.assertRaises(FileNotFoundError):
            KISFixtureEvaluator().evaluate(
                Path("non_existent_predictions.jsonl"),
                Path("non_existent_ground_truth.jsonl"),
            )


if __name__ == "__main__":
    unittest.main()
