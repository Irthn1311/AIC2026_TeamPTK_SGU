"""Evidence-grounded closed-set Q&A baseline core for Preliminary P0-B1."""

from .answer_candidates import (
    AnswerCandidateProvider,
    BaselineQuestionCandidateProvider,
)
from .answer_scoring import CosineEvidenceAnswerScorer, EvidenceAnswerScorer
from .engine import QABaselineEngine, QAResult
from .models import AnswerHypothesis, QAEvidenceCandidate, QAQuery
from .question_types import QuestionType, classify_question_type

__all__ = [
    "AnswerCandidateProvider",
    "AnswerHypothesis",
    "BaselineQuestionCandidateProvider",
    "CosineEvidenceAnswerScorer",
    "EvidenceAnswerScorer",
    "QABaselineEngine",
    "QAEvidenceCandidate",
    "QAQuery",
    "QAResult",
    "QuestionType",
    "classify_question_type",
]
