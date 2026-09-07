"""Orchestrating live translation provider with single-flight locking and fail-closed fallback."""

from __future__ import annotations

import hashlib
import logging
import re
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future
from datetime import UTC, datetime
from typing import Any

from .cache import PerKeyAtomicTranslationCache, normalize_source_text
from .google_cloud import GoogleCloudTranslationError, GoogleCloudTranslationProvider
from .models import TranslationMetadata, TranslationOutcome
from .provider import TranslationError
from .validator import SemanticSanityValidator

logger = logging.getLogger(__name__)


class SingleFlightGroup:
    """Coordinates concurrent requests so identical keys execute translation exactly once."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._in_flight: dict[str, Future[TranslationOutcome]] = {}

    def do(
        self,
        key: str,
        fn: Callable[[], TranslationOutcome],
    ) -> TranslationOutcome:
        with self._lock:
            if key in self._in_flight:
                fut = self._in_flight[key]
                wait = True
            else:
                fut = Future()
                self._in_flight[key] = fut
                wait = False

        if wait:
            return fut.result()

        try:
            res = fn()
            fut.set_result(res)
            return res
        except BaseException as exc:
            fut.set_exception(exc)
            raise
        finally:
            with self._lock:
                self._in_flight.pop(key, None)


class LiveFallbackTranslationProvider:
    """Production live translator: Google Cloud v3 primary with hardened VinAI fallback.

    Architecture:
    1. Single-flight coordination: concurrent identical queries make exactly one provider call.
    2. Cache lookups with active policy revalidation (cache hits never bypass validator).
    3. Item-level fine-grained partial batch fallback.
    4. Two-tier semantic sanity validation (ERROR forces fallback; WARNING logged).
    5. Strict fail-closed: raises TranslationError when both providers fail or are invalid.
    6. Zero secret leakage: no raw credentials, tokens, or exceptions logged or chained.
    """

    def __init__(
        self,
        *,
        google_provider: GoogleCloudTranslationProvider | None = None,
        vinai_provider: Any | None = None,
        validator: SemanticSanityValidator | None = None,
        cache: PerKeyAtomicTranslationCache | None = None,
    ) -> None:
        self.google_provider = google_provider
        self.vinai_provider = vinai_provider
        self.validator = validator or SemanticSanityValidator()
        self.cache = cache
        self._flight_group = SingleFlightGroup()

    @property
    def provider_name(self) -> str:
        return "composite-fallback:google-v3+vinai-v2"

    @property
    def device(self) -> str:
        return getattr(self.vinai_provider, "device", "cloud")

    def translate(self, text: str) -> str:
        """Translate a single query returning English string, with single-flight protection."""
        outcome = self.translate_with_metadata(text)
        return outcome.translated_text

    def translate_with_metadata(self, text: str) -> TranslationOutcome:
        """Translate a single query with single-flight deduplication across concurrent threads."""
        cleaned = normalize_source_text(text)
        if not cleaned:
            raise TranslationError("Cannot translate empty or whitespace-only query")

        outcomes = self.translate_many_with_metadata([cleaned])
        return outcomes[0]

    def translate_many(
        self,
        texts: tuple[str, ...] | list[str],
    ) -> tuple[str, ...]:
        """Translate a batch of queries, returning tuple of translated English texts."""
        outcomes = self.translate_many_with_metadata(texts)
        return tuple(outcome.translated_text for outcome in outcomes)

    def translate_many_with_metadata(
        self,
        texts: tuple[str, ...] | list[str],
    ) -> tuple[TranslationOutcome, ...]:
        """Translate a batch preserving cardinality and order, with item-level fallback."""
        if not texts:
            raise TranslationError("Cannot translate empty batch")

        cleaned_inputs: list[str] = []
        for item in texts:
            if not isinstance(item, str):
                raise TranslationError(
                    f"Invalid input text type: expected str, got {type(item).__name__}"
                )
            cleaned = normalize_source_text(item)
            if not cleaned:
                raise TranslationError("Cannot translate empty or whitespace-only query")
            cleaned_inputs.append(cleaned)

        # Deduplicate inputs to avoid redundant API queries while tracking order
        unique_texts: list[str] = []
        text_to_idx: dict[str, int] = {}
        input_mapping: list[int] = []

        for txt in cleaned_inputs:
            if txt not in text_to_idx:
                text_to_idx[txt] = len(unique_texts)
                unique_texts.append(txt)
            input_mapping.append(text_to_idx[txt])

        unique_outcomes: list[TranslationOutcome | None] = [None] * len(unique_texts)

        # 1. Cache Lookup with Active Policy Revalidation
        uncached_indices: list[int] = []
        for idx, q_text in enumerate(unique_texts):
            if self.cache is not None:
                cached = self.cache.get(q_text)
                if cached is not None:
                    # Revalidate via composite's active validator
                    val_res = self.validator.validate(q_text, cached.translated_text)
                    if (
                        val_res.is_valid
                        and cached.metadata.validator_version == self.validator.version
                    ):
                        unique_outcomes[idx] = cached
                        continue
                    logger.warning("Cache hit failed active composite revalidation; bypassing")
            uncached_indices.append(idx)

        # If all items were resolved from cache, return immediately
        if not uncached_indices:
            return tuple(unique_outcomes[idx] for idx in input_mapping)  # type: ignore[misc]

        # 2. Coordinate through SingleFlightGroup across concurrent threads
        waiter_futures: dict[int, Future[TranslationOutcome]] = {}
        leader_indices: list[int] = []

        with self._flight_group._lock:
            for orig_idx in uncached_indices:
                q_str = unique_texts[orig_idx]
                if q_str in self._flight_group._in_flight:
                    waiter_futures[orig_idx] = self._flight_group._in_flight[q_str]
                else:
                    fut: Future[TranslationOutcome] = Future()
                    self._flight_group._in_flight[q_str] = fut
                    leader_indices.append(orig_idx)

        try:
            if leader_indices:
                self._execute_leader_batch(
                    leader_indices=leader_indices,
                    unique_texts=unique_texts,
                    unique_outcomes=unique_outcomes,
                )
        finally:
            # Clean up leader futures and resolve any unhandled ones with failure
            with self._flight_group._lock:
                for orig_idx in leader_indices:
                    q_str = unique_texts[orig_idx]
                    fut = self._flight_group._in_flight.pop(q_str, None)
                    if fut and not fut.done():
                        fut.set_exception(
                            TranslationError(
                                "Leader translation failed unexpectedly [all_providers_failed]"
                            )
                        )

        # Wait on any queries where this thread was a waiter
        for orig_idx, fut in waiter_futures.items():
            outcome = fut.result()
            unique_outcomes[orig_idx] = outcome

        return tuple(unique_outcomes[idx] for idx in input_mapping)  # type: ignore[misc]

    def _execute_leader_batch(
        self,
        *,
        leader_indices: list[int],
        unique_texts: list[str],
        unique_outcomes: list[TranslationOutcome | None],
    ) -> None:
        """Execute translation for queries where this thread is the registered leader."""
        fallback_indices: list[int] = []
        fallback_reasons: dict[int, str] = {}

        # Step 2a: Primary Provider (Google Cloud v3)
        if self.google_provider is not None:
            google_needed_texts = [unique_texts[i] for i in leader_indices]
            t0 = time.perf_counter()
            try:
                raw_google_results = self.google_provider.translate_many(google_needed_texts)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0 / max(len(google_needed_texts), 1)

                if len(raw_google_results) != len(google_needed_texts):
                    raise GoogleCloudTranslationError(
                        "Google Cloud response cardinality mismatch",
                        reason_code="google_cardinality_mismatch",
                    )

                for sub_i, orig_idx in enumerate(leader_indices):
                    cand_text = raw_google_results[sub_i]
                    q_source = unique_texts[orig_idx]
                    val_res = self.validator.validate(q_source, cand_text)

                    if not val_res.is_valid:
                        first_error = next(
                            (iss for iss in val_res.issues if iss.severity == "ERROR"), None
                        )
                        err_code = (
                            f"semantic_error_{first_error.code}"
                            if first_error
                            else "semantic_validation_failed"
                        )
                        fallback_indices.append(orig_idx)
                        fallback_reasons[orig_idx] = err_code
                        logger.warning("Google translation failed validation [%s]", err_code)
                    else:
                        meta = TranslationMetadata(
                            origin_provider="google-cloud",
                            served_from_cache=False,
                            call_timestamp_utc=datetime.now(UTC).isoformat(),
                            latency_ms=elapsed_ms,
                            fallback_triggered=False,
                            fallback_reason_code=None,
                            validation_passed=True,
                            validation_issues=val_res.issues,
                            google_api_surface="v3",
                            google_client_version=self.google_provider.client_library_version,
                            google_project_location_redacted=self.google_provider.redacted_parent,
                            output_sha256=hashlib.sha256(cand_text.encode("utf-8")).hexdigest(),
                            validator_version=self.validator.version,
                        )
                        outcome = TranslationOutcome(translated_text=cand_text, metadata=meta)
                        unique_outcomes[orig_idx] = outcome

                        # Set single-flight future
                        fut = self._flight_group._in_flight.get(q_source)
                        if fut and not fut.done():
                            fut.set_result(outcome)

                        if self.cache is not None:
                            try:
                                self.cache.put(
                                    q_source,
                                    cand_text,
                                    origin_provider="google-cloud",
                                    metadata=meta,
                                )
                            except Exception:
                                logger.warning("Failed to cache translation")

            except GoogleCloudTranslationError as exc:
                logger.warning("Google Cloud Translation API failed [%s]", exc.reason_code)
                for orig_idx in leader_indices:
                    fallback_indices.append(orig_idx)
                    fallback_reasons[orig_idx] = exc.reason_code
            except Exception:
                logger.warning("Unexpected error communicating with Google Cloud Translation API")
                for orig_idx in leader_indices:
                    fallback_indices.append(orig_idx)
                    fallback_reasons[orig_idx] = "google_api_error"
        else:
            for orig_idx in leader_indices:
                fallback_indices.append(orig_idx)
                fallback_reasons[orig_idx] = "google_unavailable"

        # Step 2b: Fallback Provider (VinAI Translate v2)
        if fallback_indices:
            if self.vinai_provider is None:
                first_reason = fallback_reasons[fallback_indices[0]]
                err = TranslationError(
                    f"Google Cloud translation failed [{first_reason}] and no VinAI fallback "
                    "provider is configured. Fail-closed: refusing to return unvalidated query."
                )
                for orig_idx in fallback_indices:
                    fut = self._flight_group._in_flight.get(unique_texts[orig_idx])
                    if fut and not fut.done():
                        fut.set_exception(err)
                raise err

            vinai_queries = [unique_texts[i] for i in fallback_indices]
            t0 = time.perf_counter()
            try:
                raw_vinai = self.vinai_provider.translate_many(vinai_queries)
                vinai_latency = (time.perf_counter() - t0) * 1000.0 / max(len(vinai_queries), 1)
            except TranslationError:
                sanitized_err = TranslationError(
                    "VinAI translation execution failed [vinai_generation_error]"
                )
                for orig_idx in fallback_indices:
                    fut = self._flight_group._in_flight.get(unique_texts[orig_idx])
                    if fut and not fut.done():
                        fut.set_exception(sanitized_err)
                raise sanitized_err from None
            except Exception:
                sanitized_err = TranslationError(
                    "Both Google and VinAI failed: VinAI execution error [vinai_generation_error]"
                )
                for orig_idx in fallback_indices:
                    fut = self._flight_group._in_flight.get(unique_texts[orig_idx])
                    if fut and not fut.done():
                        fut.set_exception(sanitized_err)
                raise sanitized_err from None

            if raw_vinai is None or len(raw_vinai) != len(fallback_indices):
                card_err = TranslationError(
                    f"VinAI translation cardinality mismatch: expected {len(fallback_indices)}, "
                    f"got {len(raw_vinai or [])}"
                )
                for orig_idx in fallback_indices:
                    fut = self._flight_group._in_flight.get(unique_texts[orig_idx])
                    if fut and not fut.done():
                        fut.set_exception(card_err)
                raise card_err

            raw_model_name = getattr(
                self.vinai_provider, "model_name", "vinai/vinai-translate-vi2en-v2"
            )
            if isinstance(raw_model_name, str) and raw_model_name in {
                "vinai/vinai-translate-vi2en-v2",
                "vinai-translate",
            }:
                vinai_model_name = raw_model_name
            else:
                vinai_model_name = "vinai/vinai-translate-vi2en-v2"

            raw_revision = getattr(
                self.vinai_provider, "revision", "ae7baa85da07dbe8e23ac26a9f5ef560c17e2138"
            )
            if (
                isinstance(raw_revision, str)
                and len(raw_revision) == 40
                and re.match(r"^[0-9a-fA-F]{40}$", raw_revision)
            ):
                vinai_revision = raw_revision.lower()
            else:
                vinai_revision = "ae7baa85da07dbe8e23ac26a9f5ef560c17e2138"

            for sub_i, orig_idx in enumerate(fallback_indices):
                cand_text = raw_vinai[sub_i]
                q_source = unique_texts[orig_idx]
                val_res = self.validator.validate(q_source, cand_text)

                if not val_res.is_valid:
                    first_err = next(
                        (iss for iss in val_res.issues if iss.severity == "ERROR"), None
                    )
                    v_code = (
                        f"semantic_error_{first_err.code}"
                        if first_err
                        else "vinai_validation_failed"
                    )
                    sem_err = TranslationError(
                        f"Both Google and VinAI translations failed semantic validation [{v_code}]"
                    )
                    fut = self._flight_group._in_flight.get(q_source)
                    if fut and not fut.done():
                        fut.set_exception(sem_err)
                    raise sem_err

                fb_code = fallback_reasons.get(orig_idx, "google_api_error")
                meta = TranslationMetadata(
                    origin_provider="vinai",
                    served_from_cache=False,
                    call_timestamp_utc=datetime.now(UTC).isoformat(),
                    latency_ms=vinai_latency,
                    fallback_triggered=True,
                    fallback_reason_code=fb_code,
                    validation_passed=True,
                    validation_issues=val_res.issues,
                    vinai_model_name=vinai_model_name,
                    vinai_revision=vinai_revision,
                    output_sha256=hashlib.sha256(cand_text.encode("utf-8")).hexdigest(),
                    validator_version=self.validator.version,
                )
                outcome = TranslationOutcome(translated_text=cand_text, metadata=meta)
                unique_outcomes[orig_idx] = outcome

                fut = self._flight_group._in_flight.get(q_source)
                if fut and not fut.done():
                    fut.set_result(outcome)

                if self.cache is not None:
                    try:
                        self.cache.put(
                            q_source,
                            cand_text,
                            origin_provider="vinai",
                            metadata=meta,
                        )
                    except Exception:
                        logger.warning("Failed to cache translation")
