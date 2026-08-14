from .answer_candidates import AnswerCandidateProvider, BaselineQuestionCandidateProvider
from .answer_scoring import CosineEvidenceAnswerScorer, EvidenceAnswerScorer
from .engine import QABaselineEngine
from .grounding import (
    QA_MULTI_SEED_TEMPORAL_REFINEMENT_V1,
    QA_VIDEO_CONDITIONED_EVIDENCE_V1,
    QAVideoConditionedEvidenceConfig,
)
from .models import AnswerHypothesis, QAEvidenceCandidate, QAQuery, QAResult
from .object_provider import (
    ObjectAnswerProviderConfig,
    ObjectEntityAnswerProvider,
    normalize_object_label,
)
from .ocr_provider import (
    OCRAnswerProvider,
    OCRAnswerProviderConfig,
    OCRBackendUnavailableError,
    OCRDetection,
    OCRInferenceError,
    OCRObservation,
    TesseractCLIBackend,
    normalize_ocr_text,
)
from .question_types import (
    QuestionClassification,
    QuestionType,
    classify_ocr_question,
    classify_question,
    classify_question_legacy,
    classify_question_type,
)
from .runtime import QAPipelineTimings, QARuntimePipeline
from .visual_ontology import (
    VisualAnswerOntology,
    VisualOntologyAnswerCandidateProvider,
    VisualOntologyConfig,
    VisualOntologyError,
    load_visual_answer_ontology,
)

__all__ = [
    "AnswerCandidateProvider",
    "AnswerHypothesis",
    "BaselineQuestionCandidateProvider",
    "CosineEvidenceAnswerScorer",
    "EvidenceAnswerScorer",
    "QABaselineEngine",
    "QA_VIDEO_CONDITIONED_EVIDENCE_V1",
    "QA_MULTI_SEED_TEMPORAL_REFINEMENT_V1",
    "QAVideoConditionedEvidenceConfig",
    "QAEvidenceCandidate",
    "ObjectAnswerProviderConfig",
    "ObjectEntityAnswerProvider",
    "OCRAnswerProvider",
    "OCRAnswerProviderConfig",
    "OCRBackendUnavailableError",
    "OCRDetection",
    "OCRInferenceError",
    "OCRObservation",
    "QAPipelineTimings",
    "QAQuery",
    "QAResult",
    "QARuntimePipeline",
    "QuestionClassification",
    "QuestionType",
    "TesseractCLIBackend",
    "VisualAnswerOntology",
    "VisualOntologyAnswerCandidateProvider",
    "VisualOntologyConfig",
    "VisualOntologyError",
    "classify_question",
    "classify_question_legacy",
    "classify_question_type",
    "classify_ocr_question",
    "normalize_object_label",
    "normalize_ocr_text",
    "load_visual_answer_ontology",
]
