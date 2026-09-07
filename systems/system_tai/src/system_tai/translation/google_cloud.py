"""Official Google Cloud Translation v3 Provider with lazy import and ADC authentication."""

from __future__ import annotations

import importlib.metadata
import logging
import math
import os
from typing import Any

from .models import FALLBACK_REASON_ALLOWLIST
from .provider import TranslationError

logger = logging.getLogger(__name__)

STATIC_ERROR_MESSAGES: dict[str, str] = {
    "missing_project_id": "Google Cloud project_id is required before making API calls.",
    "google_timeout": "Google Cloud Translation API request timed out.",
    "google_quota_exceeded": "Google Cloud Translation API quota exceeded.",
    "google_auth_error": "Google Cloud Translation API authentication failed.",
    "google_unavailable": "Google Cloud Translation API is currently unavailable.",
    "google_bad_argument": "Google Cloud Translation API received invalid arguments.",
    "google_api_error": "Google Cloud Translation API request failed.",
    "google_empty_response": "Google Cloud Translation API returned empty response.",
    "google_cardinality_mismatch": "Google Cloud Translation API response cardinality mismatch.",
}


class GoogleCloudTranslationError(TranslationError):
    """Raised when Google Cloud Translation fails, with categorical reason code."""

    def __init__(self, message: str, *, reason_code: str = "google_api_error") -> None:
        super().__init__(message)
        self.reason_code = (
            reason_code if reason_code in FALLBACK_REASON_ALLOWLIST else "google_api_error"
        )


def _get_client_library_version() -> str:
    try:
        return importlib.metadata.version("google-cloud-translate")
    except Exception:
        return "unavailable"


class GoogleCloudTranslationProvider:
    """Official Google Cloud Translation Provider backed exclusively by v3 API.

    Guarantees:
    - Exclusively uses v3 API (TranslationServiceClient). Zero web scraping.
    - Lazy SDK import: code imports cleanly even if google-cloud-translate is missing.
    - Zero credential leakage: project ID is redacted; secrets/tokens/paths are never serialized.
    - Finite positive timeout (rejects NaN, Inf, zero, negative numbers).
    - Exception chaining suppression: prevents raw tracebacks from leaking via __cause__.
    - Deduplicated batch query optimization.
    """

    API_SURFACE: str = "google-cloud-translate-v3"

    def __init__(
        self,
        *,
        project_id: str | None = None,
        location: str = "global",
        timeout_seconds: float = 10.0,
        client: Any | None = None,
        enable_network: bool = True,
    ) -> None:
        try:
            t = float(timeout_seconds)
        except (ValueError, TypeError):
            raise ValueError(
                f"timeout_seconds must be a finite positive number, got {timeout_seconds!r}"
            )
        if math.isnan(t) or math.isinf(t) or t <= 0:
            raise ValueError(f"timeout_seconds must be a finite positive number, got {t}")
        self.timeout_seconds = t

        resolved_project = (
            project_id or os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCP_PROJECT")
        )
        self.project_id = str(resolved_project).strip() if resolved_project else None
        self.location = str(location or "global").strip()
        self.enable_network = enable_network
        self._client = client
        self.client_library_version = _get_client_library_version()

    @property
    def provider_name(self) -> str:
        return "google-cloud"

    @property
    def device(self) -> str:
        return "cloud"

    @property
    def redacted_parent(self) -> str:
        return f"projects/***/locations/{self.location}"

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client

        if not self.enable_network:
            raise GoogleCloudTranslationError(
                "Network is disabled on GoogleCloudTranslationProvider",
                reason_code="google_unavailable",
            )

        try:
            from google.cloud import translate_v3
        except ImportError:
            raise GoogleCloudTranslationError(
                "google-cloud-translate package is required to use GoogleCloudTranslationProvider. "
                "Install with: pip install '.[live]' or pip install google-cloud-translate",
                reason_code="google_unavailable",
            ) from None

        try:
            self._client = translate_v3.TranslationServiceClient()
            return self._client
        except Exception as exc:
            code = self._classify_exception(exc)
            msg = STATIC_ERROR_MESSAGES.get(code, "Failed to initialize TranslationServiceClient.")
            raise GoogleCloudTranslationError(
                f"Failed to initialize TranslationServiceClient [{code}]: {msg}",
                reason_code=code,
            ) from None

    def _classify_exception(self, exc: Exception) -> str:
        """Categorize exception into a standard error code without leaking secrets."""
        err_type = type(exc).__name__
        err_msg = str(exc).lower()

        if "timeout" in err_msg or "deadline" in err_msg or "timed out" in err_msg:
            return "google_timeout"
        if "quota" in err_msg or "429" in err_msg or "resourceexhausted" in err_msg:
            return "google_quota_exceeded"
        if (
            "unauthenticated" in err_msg
            or "permissiondenied" in err_msg
            or "401" in err_msg
            or "403" in err_msg
            or "credentials" in err_msg
            or "forbidden" in err_msg
        ):
            return "google_auth_error"
        if "invalid argument" in err_msg or "bad argument" in err_msg or "400" in err_msg:
            return "google_bad_argument"
        if "unavailable" in err_msg or "503" in err_msg or "connection" in err_msg:
            return "google_unavailable"

        if "DeadlineExceeded" in err_type:
            return "google_timeout"
        if "ResourceExhausted" in err_type:
            return "google_quota_exceeded"
        if "PermissionDenied" in err_type or "Unauthenticated" in err_type:
            return "google_auth_error"
        if "InvalidArgument" in err_type:
            return "google_bad_argument"
        if "ServiceUnavailable" in err_type:
            return "google_unavailable"

        return "google_api_error"

    def translate(self, text: str) -> str:
        """Translate a single Vietnamese query string to English."""
        return self.translate_many((text,))[0]

    def translate_many(
        self,
        texts: tuple[str, ...] | list[str],
    ) -> tuple[str, ...]:
        """Translate a batch of Vietnamese queries to English using Google v3 API."""
        if not texts:
            raise GoogleCloudTranslationError(
                "Cannot translate empty batch",
                reason_code="google_api_error",
            )

        # Validate non-string or whitespace items
        cleaned_inputs: list[str] = []
        for item in texts:
            if not isinstance(item, str):
                raise GoogleCloudTranslationError(
                    f"Invalid input text type: expected str, got {type(item).__name__}",
                    reason_code="google_bad_argument",
                )
            cleaned = item.strip()
            if not cleaned:
                raise GoogleCloudTranslationError(
                    "Cannot translate empty or whitespace-only query",
                    reason_code="google_bad_argument",
                )
            cleaned_inputs.append(cleaned)

        # Fail-closed check: project_id is strictly required before API call
        if not self.project_id:
            raise GoogleCloudTranslationError(
                STATIC_ERROR_MESSAGES["missing_project_id"],
                reason_code="missing_project_id",
            )

        client = self._get_client()
        parent = f"projects/{self.project_id}/locations/{self.location}"

        # Deduplicate inputs while preserving order mapping
        unique_texts: list[str] = []
        text_to_idx: dict[str, int] = {}
        input_mapping: list[int] = []

        for txt in cleaned_inputs:
            if txt not in text_to_idx:
                text_to_idx[txt] = len(unique_texts)
                unique_texts.append(txt)
            input_mapping.append(text_to_idx[txt])

        request_payload = {
            "parent": parent,
            "contents": unique_texts,
            "mime_type": "text/plain",
            "source_language_code": "vi",
            "target_language_code": "en",
        }

        try:
            # Build retry policy if google.api_core is available
            retry_kw: dict[str, Any] = {}
            try:
                from google.api_core import retry as google_retry

                def _is_transient(e: Any) -> bool:
                    code = self._classify_exception(e)
                    return code in {"google_timeout", "google_quota_exceeded", "google_unavailable"}

                retry_kw["retry"] = google_retry.Retry(
                    predicate=_is_transient,
                    deadline=self.timeout_seconds,
                )
            except Exception:
                pass

            response = client.translate_text(
                request=request_payload,
                timeout=self.timeout_seconds,
                **retry_kw,
            )
        except Exception as exc:
            code = self._classify_exception(exc)
            msg = STATIC_ERROR_MESSAGES.get(code, "Google Cloud Translation API request failed.")
            raise GoogleCloudTranslationError(
                f"Google Cloud Translation v3 API call failed [{code}]: {msg}",
                reason_code=code,
            ) from None

        raw_translations = getattr(response, "translations", None)
        if raw_translations is None or len(raw_translations) != len(unique_texts):
            raise GoogleCloudTranslationError(
                STATIC_ERROR_MESSAGES["google_cardinality_mismatch"],
                reason_code="google_cardinality_mismatch",
            )

        unique_results: list[str] = []
        for tr in raw_translations:
            text_val = getattr(tr, "translated_text", "")
            if not text_val or not str(text_val).strip():
                raise GoogleCloudTranslationError(
                    STATIC_ERROR_MESSAGES["google_empty_response"],
                    reason_code="google_empty_response",
                )
            unique_results.append(str(text_val).strip())

        # Map back to original ordering
        ordered_results = tuple(unique_results[idx] for idx in input_mapping)
        return ordered_results
