"""Data models and immutable outcome contracts for live multilingual translation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

OriginProvider = Literal[
    "google-cloud",
    "vinai",
    "vinai-translate",
    "legacy-unverified",
    "legacy-revalidated",
]

SEMANTIC_ERROR_CODES: frozenset[str] = frozenset(
    {
        "spatial_inversion",
        "spatial_swap",
        "spatial_dropped",
        "direction_dropped",
        "count_dropped",
        "count_entity_swap",
        "cardinal_ordinal_mismatch",
        "comparator_dropped",
        "comparator_mismatch",
        "comparator_polarity_inversion",
        "temporal_inversion",
        "scene_dropped",
        "negation_dropped",
        "negation_introduced",
        "negation_entity_misplaced",
        "untranslated_vietnamese",
        "degenerate_output",
        "color_mismatch",
        "color_dropped",
        "perspective_dropped",
    }
)

PROVIDER_ERROR_CODES: frozenset[str] = frozenset(
    {
        "missing_project_id",
        "google_timeout",
        "google_quota_exceeded",
        "google_auth_error",
        "google_unavailable",
        "google_bad_argument",
        "google_api_error",
        "google_empty_response",
        "google_cardinality_mismatch",
        "vinai_unsupported_device",
        "vinai_snapshot_mismatch",
        "vinai_snapshot_not_found",
        "vinai_missing_manifest",
        "vinai_manifest_corrupted",
        "vinai_manifest_path_traversal",
        "vinai_missing_artifact",
        "vinai_cardinality_mismatch",
        "manifest_revision_mismatch",
        "artifact_checksum_mismatch",
        "vinai_load_error",
        "vinai_generation_error",
        "vinai_validation_failed",
        "cache_integrity_failure",
        "all_providers_failed",
    }
)

FALLBACK_REASON_ALLOWLIST: frozenset[str] = frozenset(
    PROVIDER_ERROR_CODES | {f"semantic_error_{code}" for code in SEMANTIC_ERROR_CODES}
)


@dataclass(frozen=True)
class SemanticValidationIssue:
    """Diagnostic issue produced by SemanticSanityValidator."""

    severity: Literal["ERROR", "WARNING"]
    code: str
    message: str


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of semantic sanity validation."""

    is_valid: bool
    issues: tuple[SemanticValidationIssue, ...]
    extracted_entities: dict[str, Any]


@dataclass(frozen=True)
class TranslationMetadata:
    """Immutable audit telemetry for a translation request."""

    origin_provider: str
    served_from_cache: bool
    call_timestamp_utc: str
    latency_ms: float
    fallback_triggered: bool
    fallback_reason_code: str | None
    validation_passed: bool
    validation_issues: tuple[SemanticValidationIssue, ...]
    google_api_surface: str = "v3"
    google_client_version: str = "unavailable"
    google_project_location_redacted: str | None = None
    vinai_model_name: str | None = None
    vinai_revision: str | None = None
    output_sha256: str = ""
    validator_version: str = "sem_sanity_v1"
    policy_version: str = "live_p2_v1"
    normalization_version: str = "norm_v1"


@dataclass(frozen=True)
class TranslationOutcome:
    """Full outcome containing translated text and immutable audit telemetry."""

    translated_text: str
    metadata: TranslationMetadata
