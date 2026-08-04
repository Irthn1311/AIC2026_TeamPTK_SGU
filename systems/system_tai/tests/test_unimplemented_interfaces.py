from __future__ import annotations

import unittest
from pathlib import Path

from system_tai.checkpointing.exporter import CheckpointExporter
from system_tai.evaluation.kis_fixture import KISFixtureEvaluator
from system_tai.ranking.kis_ranker import KISRanker
from system_tai.retrieval.candidates import CandidateConstructor
from system_tai.retrieval.vector_search import VectorSearch
from system_tai.validation.checkpoint_validator import CheckpointValidator


class ExplicitFailureTests(unittest.TestCase):
    def test_vector_search_is_explicitly_unimplemented(self) -> None:
        with self.assertRaises(NotImplementedError):
            VectorSearch().search([], object(), top_k=1)

    def test_candidate_constructor_is_explicitly_unimplemented(self) -> None:
        with self.assertRaises(NotImplementedError):
            CandidateConstructor().build("Q001", (), ())

    def test_ranker_is_explicitly_unimplemented(self) -> None:
        with self.assertRaises(NotImplementedError):
            KISRanker().rank(())

    def test_exporter_is_explicitly_unimplemented(self) -> None:
        with self.assertRaises(NotImplementedError):
            CheckpointExporter().export((), Path("predictions.jsonl"))

    def test_validator_is_explicitly_unimplemented(self) -> None:
        with self.assertRaises(NotImplementedError):
            CheckpointValidator().validate(
                Path("predictions.jsonl"),
                object(),
                query_set_path=Path("queries.jsonl"),
            )

    def test_evaluator_is_explicitly_unimplemented(self) -> None:
        with self.assertRaises(NotImplementedError):
            KISFixtureEvaluator().evaluate(
                Path("predictions.jsonl"), Path("ground_truth.jsonl")
            )


if __name__ == "__main__":
    unittest.main()
