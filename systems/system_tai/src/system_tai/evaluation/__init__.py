"""Ground-truth KIS benchmark evaluation interfaces."""

from .annotation import (
    AnnotationCandidate,
    build_annotation_candidates,
    write_draft_annotation_review,
)
from .benchmark_schema import (
    AnnotationStatus,
    BenchmarkLanguage,
    BenchmarkQuery,
    KISBenchmark,
    RelevantFrame,
    VariantType,
    load_benchmark,
)
from .benchmark_validator import BenchmarkValidationResult, BenchmarkValidator
from .kis_benchmark import (
    KISBenchmarkEvaluator,
    KISBenchmarkReport,
    NoVerifiedQueriesResult,
)
from .kis_fixture import KISFixtureEvaluator
from .reports import BenchmarkReportPaths, write_benchmark_reports

__all__ = [
    "AnnotationCandidate",
    "AnnotationStatus",
    "BenchmarkLanguage",
    "BenchmarkQuery",
    "BenchmarkReportPaths",
    "BenchmarkValidationResult",
    "BenchmarkValidator",
    "KISBenchmark",
    "KISBenchmarkEvaluator",
    "KISBenchmarkReport",
    "NoVerifiedQueriesResult",
    "KISFixtureEvaluator",
    "RelevantFrame",
    "VariantType",
    "build_annotation_candidates",
    "load_benchmark",
    "write_benchmark_reports",
    "write_draft_annotation_review",
]
