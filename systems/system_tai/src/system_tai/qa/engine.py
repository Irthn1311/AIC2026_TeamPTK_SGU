from collections.abc import Mapping, Sequence

import numpy as np

from system_tai.preliminary.matching import NormalizedAliasAnswerMatcher
from system_tai.preliminary.schemas import QAPrediction

from .answer_candidates import (
    AnswerCandidateProvider,
    BaselineQuestionCandidateProvider,
)
from .answer_scoring import CosineEvidenceAnswerScorer, EvidenceAnswerScorer
from .models import QAEvidenceCandidate, QAQuery, QAResult
from .question_types import QuestionType, classify_question_type
from .top100_constructor import construct_ranked_qa_top100


class QABaselineEngine:
    def __init__(
        self,
        candidate_provider: AnswerCandidateProvider | None = None,
        scorer: EvidenceAnswerScorer | None = None,
        expand_temporal: bool = False,
        allow_unsupported_provider_fallback: bool = False,
        secondary_temporal_micro_budget: bool = False,
        primary_11_12_micro_coverage: bool = False,
        tier3_primary_first: bool = False,
    ) -> None:
        self.candidate_provider = candidate_provider or BaselineQuestionCandidateProvider()
        self.scorer = scorer or CosineEvidenceAnswerScorer()
        self.matcher = NormalizedAliasAnswerMatcher(strip_punctuation=True)
        self.expand_temporal = expand_temporal
        self.allow_unsupported_provider_fallback = allow_unsupported_provider_fallback
        self.secondary_temporal_micro_budget = secondary_temporal_micro_budget
        self.primary_11_12_micro_coverage = primary_11_12_micro_coverage
        self.tier3_primary_first = tier3_primary_first

    def answer(
        self,
        query: QAQuery,
        evidence_candidates: Sequence[QAEvidenceCandidate],
        image_embeddings: dict[tuple[str, int], np.ndarray] | None = None,
        prompt_embeddings: dict[str, np.ndarray] | None = None,
        output_top_k: int | None = None,
        expand_temporal: bool | None = None,
    ) -> QAResult:
        qtype = query.question_type or classify_question_type(
            query.question, query.question_en
        )

        if qtype == QuestionType.UNSUPPORTED:
            if not self.allow_unsupported_provider_fallback:
                return QAResult(
                    query_id=query.query_id,
                    question_type=qtype,
                    predictions=[],
                    unsupported_reason="Open-ended or unsupported question pattern in B1",
                    diagnostics={"confidence_level": "UNSUPPORTED"},
                )
            confidence_level = "FALLBACK"
        elif qtype == QuestionType.YES_NO:
            confidence_level = "EXPERIMENTAL"
        else:
            confidence_level = "BASELINE"

        if not evidence_candidates:
            return QAResult(
                query_id=query.query_id,
                question_type=qtype,
                predictions=[],
                unsupported_reason="No evidence candidates provided",
                diagnostics={
                    "confidence_level": (
                        "UNSUPPORTED"
                        if qtype == QuestionType.UNSUPPORTED
                        else confidence_level
                    )
                },
            )

        query_text = query.question_en or query.question
        if qtype == QuestionType.UNSUPPORTED:
            try:
                query_aware = getattr(
                    self.candidate_provider,
                    "get_candidates_for_query",
                    None,
                )
                if callable(query_aware):
                    hypotheses = tuple(query_aware(qtype, query_text))
                else:
                    hypotheses = tuple(self.candidate_provider.get_candidates(qtype))
            except Exception:
                hypotheses = ()
        else:
            query_aware = getattr(
                self.candidate_provider,
                "get_candidates_for_query",
                None,
            )
            if callable(query_aware):
                hypotheses = tuple(query_aware(qtype, query_text))
            else:
                hypotheses = tuple(self.candidate_provider.get_candidates(qtype))

        if not hypotheses:
            return QAResult(
                query_id=query.query_id,
                question_type=qtype,
                predictions=[],
                unsupported_reason="No answer hypotheses available for question type",
                diagnostics={
                    "confidence_level": (
                        "UNSUPPORTED"
                        if qtype == QuestionType.UNSUPPORTED
                        else confidence_level
                    )
                },
            )

        if len(evidence_candidates) > 100:
            raise ValueError("Cannot exceed 100 predictions: candidate count exceeds 100")

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

        # Process candidates & score hypotheses
        scored_candidates: list[dict] = []
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
            cand_prov = getattr(cand, "provenance", None)
            scored_candidates.append(
                {
                    "video_id": cand.video_id,
                    "frame_id": cand.frame_id,
                    "answers": [hyp.canonical_answer for hyp, _ in scored_hyps[:3]],
                    "scores": [score for _, score in scored_hyps[:3]],
                    "evidence_rank": cand.rank,
                    "video_nomination_rank": (
                        cand_prov.get("video_nomination_rank")
                        if isinstance(cand_prov, (dict, Mapping))
                        else None
                    ),
                    "local_anchor_rank": (
                        cand_prov.get("local_anchor_rank")
                        if isinstance(cand_prov, (dict, Mapping))
                        else None
                    ),
                }
            )

        target_k = 100 if output_top_k is None else output_top_k
        use_expansion = self.expand_temporal if expand_temporal is None else expand_temporal

        # Construct metric-aware ranked Top-100 list with temporal diversity
        predictions: list[QAPrediction] = construct_ranked_qa_top100(
            query_id=query.query_id,
            scored_candidates=scored_candidates,
            output_top_k=target_k,
            expand_temporal=use_expansion,
            secondary_temporal_micro_budget=self.secondary_temporal_micro_budget,
            primary_11_12_micro_coverage=self.primary_11_12_micro_coverage,
            tier3_primary_first=self.tier3_primary_first,
        )

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
