from collections.abc import Sequence

import numpy as np

from system_tai.preliminary.matching import NormalizedAliasAnswerMatcher
from system_tai.preliminary.schemas import QAPrediction
from system_tai.preliminary.validation import validate_ranked_top100

from .answer_candidates import (
    AnswerCandidateProvider,
    BaselineQuestionCandidateProvider,
)
from .answer_scoring import CosineEvidenceAnswerScorer, EvidenceAnswerScorer
from .models import QAEvidenceCandidate, QAQuery, QAResult
from .question_types import QuestionType, classify_question_type


class QABaselineEngine:
    def __init__(
        self,
        candidate_provider: AnswerCandidateProvider | None = None,
        scorer: EvidenceAnswerScorer | None = None,
    ) -> None:
        self.candidate_provider = candidate_provider or BaselineQuestionCandidateProvider()
        self.scorer = scorer or CosineEvidenceAnswerScorer()
        self.matcher = NormalizedAliasAnswerMatcher(strip_punctuation=True)

    def answer(
        self,
        query: QAQuery,
        evidence_candidates: Sequence[QAEvidenceCandidate],
        image_embeddings: dict[tuple[str, int], np.ndarray] | None = None,
        prompt_embeddings: dict[str, np.ndarray] | None = None,
    ) -> QAResult:
        qtype = query.question_type or classify_question_type(
            query.question, query.question_en
        )
        confidence_level = "EXPERIMENTAL" if qtype == QuestionType.YES_NO else "BASELINE"

        if qtype == QuestionType.UNSUPPORTED:
            return QAResult(
                query_id=query.query_id,
                question_type=qtype,
                predictions=[],
                unsupported_reason="Open-ended or unsupported question pattern in B1",
                diagnostics={"confidence_level": "UNSUPPORTED"},
            )

        if not evidence_candidates:
            return QAResult(
                query_id=query.query_id,
                question_type=qtype,
                predictions=[],
                unsupported_reason="No evidence candidates provided",
                diagnostics={"confidence_level": confidence_level},
            )

        hypotheses = self.candidate_provider.get_candidates(qtype)
        if not hypotheses:
            return QAResult(
                query_id=query.query_id,
                question_type=qtype,
                predictions=[],
                unsupported_reason="No answer hypotheses available for question type",
                diagnostics={"confidence_level": confidence_level},
            )

        # Validate candidate query_ids and ranks
        ranks_seen = set()
        for cand in evidence_candidates:
            if cand.query_id != query.query_id:
                raise ValueError(
                    f"Evidence candidate query_id mismatch: expected {query.query_id}, "
                    f"got {cand.query_id}"
                )
            if cand.rank in ranks_seen:
                raise ValueError(f"Duplicate evidence candidate rank found: {cand.rank}")
            ranks_seen.add(cand.rank)

        # Process candidates
        predictions: list[QAPrediction] = []
        seen_keys: set[tuple[str, int, str]] = set()
        scores_by_rank: dict[int, float] = {}

        for cand in evidence_candidates:
            img_emb = None
            if image_embeddings is not None:
                img_emb = image_embeddings.get((cand.video_id, cand.frame_id))

            scored_hyps = self.scorer.score_answers(
                cand, hypotheses, img_emb, prompt_embeddings
            )
            if not scored_hyps:
                continue

            best_hyp, best_score = scored_hyps[0]
            scores_by_rank[cand.rank] = float(best_score)
            canonical = best_hyp.canonical_answer
            norm_ans = self.matcher.normalize(canonical)
            dedup_key = (cand.video_id, cand.frame_id, norm_ans)
            if dedup_key in seen_keys:
                continue
            seen_keys.add(dedup_key)

            pred = QAPrediction(
                query_id=query.query_id,
                rank=cand.rank,
                video_id=cand.video_id,
                frame_id=cand.frame_id,
                answer=canonical,
            )
            predictions.append(pred)

        # Sort predictions by evidence rank ascending
        predictions.sort(key=lambda p: p.rank)

        errors = validate_ranked_top100(
            predictions, "qa", expected_query_id=query.query_id
        )
        if errors:
            msg = "; ".join(e.message for e in errors)
            raise ValueError(f"P0-A QA validation failed: {msg}")

        return QAResult(
            query_id=query.query_id,
            question_type=qtype,
            predictions=predictions,
            diagnostics={
                "candidate_count": len(evidence_candidates),
                "returned_count": len(predictions),
                "confidence_level": confidence_level,
                "scores_by_rank": scores_by_rank,
            },
        )
