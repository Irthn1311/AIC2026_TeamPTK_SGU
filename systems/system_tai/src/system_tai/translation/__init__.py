"""Translation providers and token budget guards for multilingual query ingestion."""

from __future__ import annotations

from .cache import (
    PerKeyAtomicTranslationCache,
    compute_cache_key,
    normalize_source_text,
)
from .composite import LiveFallbackTranslationProvider
from .google_cloud import (
    GoogleCloudTranslationError,
    GoogleCloudTranslationProvider,
)
from .models import (
    SemanticValidationIssue,
    TranslationMetadata,
    TranslationOutcome,
    ValidationResult,
)
from .provider import (
    TokenBudgetGuard,
    TranslationError,
    TranslationProvider,
    VinAITranslateProvider,
)
from .sidecar_provider import (
    ImmutableSidecarTranslationProvider,
    canonical_sidecar_sha256,
)
from .validator import SemanticSanityValidator

__all__ = [
    "GoogleCloudTranslationError",
    "GoogleCloudTranslationProvider",
    "ImmutableSidecarTranslationProvider",
    "LiveFallbackTranslationProvider",
    "PerKeyAtomicTranslationCache",
    "SemanticSanityValidator",
    "SemanticValidationIssue",
    "TokenBudgetGuard",
    "TranslationError",
    "TranslationMetadata",
    "TranslationOutcome",
    "TranslationProvider",
    "ValidationResult",
    "VinAITranslateProvider",
    "canonical_sidecar_sha256",
    "compute_cache_key",
    "normalize_source_text",
]
