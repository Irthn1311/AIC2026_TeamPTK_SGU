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
from .fusion_benchmark import (
    FusionBenchmarkEvaluator,
    FusionBenchmarkReport,
    NoComparableFusionGroupsError,
    select_comparable_fusion_groups,
)
from .fusion_reports import FusionReportPaths, write_fusion_reports
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
    "FusionBenchmarkEvaluator",
    "FusionBenchmarkReport",
    "FusionReportPaths",
    "KISBenchmark",
    "KISBenchmarkEvaluator",
    "KISBenchmarkReport",
    "NoVerifiedQueriesResult",
    "NoComparableFusionGroupsError",
    "KISFixtureEvaluator",
    "RelevantFrame",
    "VariantType",
    "build_annotation_candidates",
    "load_benchmark",
    "select_comparable_fusion_groups",
    "write_benchmark_reports",
    "write_draft_annotation_review",
    "write_fusion_reports",
]
