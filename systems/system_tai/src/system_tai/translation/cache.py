"""Validator-backed per-key atomic translation cache with cross-instance synchronization."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import re
import threading
import unicodedata
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .models import (
    FALLBACK_REASON_ALLOWLIST,
    OriginProvider,
    TranslationMetadata,
    TranslationOutcome,
)
from .validator import SemanticSanityValidator

logger = logging.getLogger(__name__)

SCHEMA_VERSION: str = "v1"
POLICY_VERSION: str = "live_p2_v1"
NORMALIZATION_VERSION: str = "norm_v1"

ALLOWED_ORIGIN_PROVIDERS: frozenset[str] = frozenset(
    {
        "google-cloud",
        "vinai",
        "vinai-translate",
        "legacy-unverified",
        "legacy-revalidated",
    }
)

ALLOWED_REGIONS: frozenset[str] = frozenset(
    {
        "global",
        "us-central1",
        "asia-southeast1",
        "europe-west1",
    }
)

# Module-level registry ensuring any instance targeting the same directory shares the same RLock
_REGISTRY_LOCK = threading.Lock()
_DIR_LOCKS: dict[str, threading.RLock] = {}


def _get_dir_lock(resolved_path: str) -> threading.RLock:
    with _REGISTRY_LOCK:
        if resolved_path not in _DIR_LOCKS:
            _DIR_LOCKS[resolved_path] = threading.RLock()
        return _DIR_LOCKS[resolved_path]


def normalize_source_text(text: str) -> str:
    """Canonical Unicode NFC and whitespace normalization."""
    if not text:
        return ""
    nfc = unicodedata.normalize("NFC", str(text))
    return " ".join(nfc.split()).strip()


def compute_cache_key(
    *,
    source_text_normalized: str,
    source_lang: str = "vi",
    target_lang: str = "en",
    schema_version: str = SCHEMA_VERSION,
    policy_version: str = POLICY_VERSION,
    validator_version: str = SemanticSanityValidator.version,
    normalization_version: str = NORMALIZATION_VERSION,
    google_api_surface: str = "v3",
    vinai_revision: str = "ae7baa85da07dbe8e23ac26a9f5ef560c17e2138",
) -> str:
    """Generate SHA256 cache key using canonical JSON encoding to avoid delimiter collisions."""
    key_fields = {
        "google_api_surface": str(google_api_surface),
        "normalization_version": str(normalization_version),
        "policy_version": str(policy_version),
        "schema_version": str(schema_version),
        "source_lang": str(source_lang),
        "source_text_normalized": str(source_text_normalized),
        "target_lang": str(target_lang),
        "validator_version": str(validator_version),
        "vinai_revision": str(vinai_revision),
    }
    canonical = json.dumps(
        key_fields,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _resolve_hmac_key(key_input: bytes | str | None = None) -> bytes | None:
    """Resolve HMAC key from parameter or SYSTEM_TAI_CACHE_HMAC_KEY environment variable.

    Must provide at least 32 bytes of key material when configured.
    """
    raw = key_input or os.environ.get("SYSTEM_TAI_CACHE_HMAC_KEY")
    if raw is None:
        return None
    if isinstance(raw, str):
        raw_str = raw.strip()
        # Attempt hex decoding
        if len(raw_str) >= 64:
            try:
                candidate = bytes.fromhex(raw_str)
                if len(candidate) >= 32:
                    return candidate
            except ValueError:
                pass
        # Attempt base64 decoding
        try:
            candidate = base64.b64decode(raw_str)
            if len(candidate) >= 32:
                return candidate
        except Exception:
            pass
        key_bytes = raw_str.encode("utf-8")
    else:
        key_bytes = raw

    if len(key_bytes) < 32:
        raise ValueError(
            f"HMAC key must provide at least 32 bytes of key material, got {len(key_bytes)}"
        )
    return key_bytes


def _compute_payload_signature(
    payload_without_sig: dict[str, Any],
    hmac_key: bytes | None = None,
) -> str:
    """Compute payload signature: HMAC-SHA256 if key configured, or SHA256 checksum-only."""
    canonical = json.dumps(
        payload_without_sig,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if hmac_key is not None:
        return hmac.new(hmac_key, canonical, hashlib.sha256).hexdigest()
    return hashlib.sha256(canonical).hexdigest()


class PerKeyAtomicTranslationCache:
    """Thread-safe, per-key atomic disk translation cache.

    Guarantees:
    - Per-key file isolation: no multi-thread read-modify-write overwrite race.
    - Explicit integrity mode: 'hmac-required' or 'checksum-only' (corruption detection).
    - Zero secret leakage: scrubs and sanitizes all telemetry metadata against strict allowlists.
    - Cache hit revalidation: entries are re-validated with active policy before serving.
    - Provenance enforcement: legacy-unverified entries are never served as trusted cache hits.
    """

    def __init__(
        self,
        cache_dir: Path | str,
        *,
        validator: SemanticSanityValidator | None = None,
        google_api_surface: str = "v3",
        vinai_revision: str = "ae7baa85da07dbe8e23ac26a9f5ef560c17e2138",
        hmac_key: bytes | str | None = None,
        require_hmac: bool = False,
        hmac_mode: str | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir).resolve()
        self.entries_dir = self.cache_dir / "entries"
        self.entries_dir.mkdir(parents=True, exist_ok=True)
        self._dir_lock = _get_dir_lock(str(self.cache_dir))
        self.validator = validator or SemanticSanityValidator()
        self.google_api_surface = google_api_surface
        self.vinai_revision = vinai_revision

        if require_hmac and hmac_mode == "checksum-only":
            raise ValueError(
                "Conflicting cache integrity options: require_hmac=True cannot be combined "
                "with hmac_mode='checksum-only'"
            )
        if hmac_mode not in (None, "checksum-only", "hmac-required"):
            raise ValueError("Invalid hmac_mode. Must be 'checksum-only' or 'hmac-required'.")

        resolved_key = _resolve_hmac_key(hmac_key)
        if require_hmac or hmac_mode == "hmac-required":
            if resolved_key is None:
                raise ValueError(
                    "HMAC key required but not configured (SYSTEM_TAI_CACHE_HMAC_KEY missing)"
                )
            self.hmac_key = resolved_key
            self.integrity_mode = "hmac-required"
        elif hmac_mode == "checksum-only":
            self.hmac_key = None
            self.integrity_mode = "checksum-only"
        else:
            self.hmac_key = resolved_key
            self.integrity_mode = "hmac-required" if self.hmac_key is not None else "checksum-only"

    def get(self, source_text: str) -> TranslationOutcome | None:
        """Retrieve, verify integrity, and re-validate cached translation.

        Returns None on miss, corruption, tamper, mismatch, or validation failure.
        """
        norm_source = normalize_source_text(source_text)
        if not norm_source:
            return None

        cache_key = compute_cache_key(
            source_text_normalized=norm_source,
            validator_version=self.validator.version,
            google_api_surface=self.google_api_surface,
            vinai_revision=self.vinai_revision,
        )
        entry_path = self.entries_dir / f"{cache_key}.json"

        with self._dir_lock:
            if not entry_path.exists():
                return None

            try:
                raw_data = json.loads(entry_path.read_text(encoding="utf-8"))
                if not isinstance(raw_data, dict):
                    return None

                # 1. Integrity check (HMAC or checksum-only depending on configured mode)
                stored_mode = raw_data.get("integrity_mode")
                if stored_mode != self.integrity_mode:
                    logger.warning(
                        "Cache entry integrity mode mismatch for %.16s",
                        cache_key,
                    )
                    return None

                stored_sig = raw_data.get("integrity_signature")
                if not stored_sig:
                    logger.warning(
                        "Cache entry %.16s lacks integrity signature; rejecting", cache_key
                    )
                    return None

                payload_to_verify = {
                    k: v for k, v in raw_data.items() if k != "integrity_signature"
                }
                expected_sig = _compute_payload_signature(payload_to_verify, self.hmac_key)
                if not hmac.compare_digest(str(stored_sig), expected_sig):
                    logger.warning("Tampered or invalid signature for cache entry %.16s", cache_key)
                    return None

                # 2. Cross-entry substitution check
                stored_key = raw_data.get("cache_key")
                if stored_key != cache_key:
                    logger.warning("Cache entry key substitution detected for %.16s", cache_key)
                    return None

                stored_source = raw_data.get("source_text", "")
                if normalize_source_text(stored_source) != norm_source:
                    logger.warning("Cache entry source text mismatch for %.16s", cache_key)
                    return None

                # 3. Provider and version checks
                origin_provider = raw_data.get("origin_provider")
                if origin_provider not in ALLOWED_ORIGIN_PROVIDERS:
                    logger.warning("Disallowed origin provider in cache")
                    return None

                # Disallow unverified legacy provenance from being served as trusted hit
                if origin_provider == "legacy-unverified":
                    logger.debug(
                        "Cache entry %.16s has unverified legacy provenance; bypassing", cache_key
                    )
                    return None

                if (
                    raw_data.get("schema_version") != SCHEMA_VERSION
                    or raw_data.get("policy_version") != POLICY_VERSION
                    or raw_data.get("validator_version") != self.validator.version
                    or raw_data.get("normalization_version") != NORMALIZATION_VERSION
                    or raw_data.get("google_api_surface") != self.google_api_surface
                    or raw_data.get("vinai_revision") != self.vinai_revision
                ):
                    return None

                translated_text = raw_data.get("translated_text", "")

                # 4. Strict Revalidation: cache hit MUST NEVER bypass validator
                val_res = self.validator.validate(source_text, translated_text)
                if not val_res.is_valid:
                    logger.warning(
                        "Cached entry %.16s failed semantic revalidation; evicting", cache_key
                    )
                    try:
                        entry_path.unlink(missing_ok=True)
                    except Exception:
                        pass
                    return None

                # 5. Faithful provenance preservation
                fallback_triggered = bool(raw_data.get("fallback_triggered", False))
                fallback_reason_code = raw_data.get("fallback_reason_code")
                if fallback_reason_code and fallback_reason_code not in FALLBACK_REASON_ALLOWLIST:
                    fallback_reason_code = None

                client_ver = raw_data.get("google_client_version", "unavailable")
                is_valid_ver = (
                    re.match(r"^\d+\.\d+\.\d+$", str(client_ver)) or client_ver == "unavailable"
                )
                if not is_valid_ver:
                    client_ver = "unavailable"

                vinai_name = raw_data.get("vinai_model_name")
                if vinai_name and vinai_name not in {
                    "vinai/vinai-translate-vi2en-v2",
                    "vinai-translate",
                }:
                    vinai_name = "vinai/vinai-translate-vi2en-v2"

                meta = TranslationMetadata(
                    origin_provider=origin_provider,
                    served_from_cache=True,
                    call_timestamp_utc=raw_data.get(
                        "created_timestamp_utc",
                        datetime.now(UTC).isoformat(),
                    ),
                    latency_ms=0.0,
                    fallback_triggered=fallback_triggered,
                    fallback_reason_code=fallback_reason_code,
                    validation_passed=True,
                    validation_issues=val_res.issues,
                    google_api_surface=raw_data.get("google_api_surface", self.google_api_surface),
                    google_client_version=client_ver,
                    google_project_location_redacted=raw_data.get(
                        "google_project_location_redacted"
                    ),
                    vinai_model_name=vinai_name,
                    vinai_revision=raw_data.get("vinai_revision", self.vinai_revision),
                    output_sha256=hashlib.sha256(translated_text.encode("utf-8")).hexdigest(),
                    validator_version=self.validator.version,
                    policy_version=POLICY_VERSION,
                    normalization_version=NORMALIZATION_VERSION,
                )
                return TranslationOutcome(translated_text=translated_text, metadata=meta)

            except Exception:
                logger.warning("Error reading cache entry for key %.16s", cache_key)
                return None

    def put(
        self,
        source_text: str,
        translated_text: str,
        *,
        origin_provider: OriginProvider,
        metadata: TranslationMetadata | None = None,
        fallback_reason_code: str | None = None,
        google_client_version: str | None = None,
        vinai_model_name: str | None = None,
        google_project_location_redacted: str | None = None,
    ) -> str:
        """Store translation atomically in its own JSON file with integrity signature."""
        norm_source = normalize_source_text(source_text)
        if not norm_source:
            raise ValueError("Cannot cache empty source text")

        if origin_provider not in ALLOWED_ORIGIN_PROVIDERS:
            raise ValueError("Disallowed origin provider: provider not in allowlist")

        # Extract candidates from explicit kwargs or metadata
        reason_input = (
            fallback_reason_code
            if fallback_reason_code is not None
            else (metadata.fallback_reason_code if metadata else None)
        )
        client_ver_input = (
            google_client_version
            if google_client_version is not None
            else (metadata.google_client_version if metadata else None)
        )
        vinai_name_input = (
            vinai_model_name
            if vinai_model_name is not None
            else (metadata.vinai_model_name if metadata else None)
        )
        loc_input = (
            google_project_location_redacted
            if google_project_location_redacted is not None
            else (metadata.google_project_location_redacted if metadata else None)
        )

        sanitized_reason = None
        if reason_input is not None:
            cand_reason = str(reason_input).strip().lower()
            if cand_reason not in FALLBACK_REASON_ALLOWLIST:
                raise ValueError("Invalid fallback_reason_code: code not in allowlist")
            sanitized_reason = cand_reason

        sanitized_client_version = "unavailable"
        if client_ver_input is not None:
            cand_ver = str(client_ver_input).strip()
            if not (re.match(r"^\d+\.\d+\.\d+$", cand_ver) or cand_ver == "unavailable"):
                raise ValueError("Invalid google_client_version: must be semver or 'unavailable'")
            sanitized_client_version = cand_ver

        sanitized_vinai_name = None
        if vinai_name_input is not None:
            cand_model = str(vinai_name_input).strip()
            if cand_model not in {"vinai/vinai-translate-vi2en-v2", "vinai-translate"}:
                raise ValueError("Invalid vinai_model_name: arbitrary paths forbidden")
            sanitized_vinai_name = cand_model

        redacted_location = None
        if loc_input is not None:
            cand_loc = str(loc_input).strip()
            match = re.match(r"^projects/[^/]+/locations/([a-zA-Z0-9_-]+)$", cand_loc)
            if not match or match.group(1) not in ALLOWED_REGIONS:
                raise ValueError("Invalid google_project_location_redacted: region not approved")
            redacted_location = f"projects/***/locations/{match.group(1)}"

        cache_key = compute_cache_key(
            source_text_normalized=norm_source,
            validator_version=self.validator.version,
            google_api_surface=self.google_api_surface,
            vinai_revision=self.vinai_revision,
        )
        entry_path = self.entries_dir / f"{cache_key}.json"
        tmp_path = self.entries_dir / f"{cache_key}.tmp.{uuid.uuid4().hex}"

        payload: dict[str, Any] = {
            "cache_key": cache_key,
            "created_timestamp_utc": datetime.now(UTC).isoformat(),
            "fallback_reason_code": sanitized_reason,
            "fallback_triggered": metadata.fallback_triggered if metadata else False,
            "google_api_surface": self.google_api_surface,
            "google_client_version": sanitized_client_version,
            "google_project_location_redacted": redacted_location,
            "integrity_mode": self.integrity_mode,
            "normalization_version": NORMALIZATION_VERSION,
            "origin_provider": origin_provider,
            "policy_version": POLICY_VERSION,
            "schema_version": SCHEMA_VERSION,
            "source_text": source_text,
            "source_text_normalized": norm_source,
            "translated_text": translated_text,
            "validator_version": self.validator.version,
            "vinai_model_name": sanitized_vinai_name,
            "vinai_revision": self.vinai_revision,
        }

        signature = _compute_payload_signature(payload, self.hmac_key)
        payload["integrity_signature"] = signature

        with self._dir_lock:
            try:
                tmp_path.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                os.replace(tmp_path, entry_path)
            except Exception:
                if tmp_path.exists():
                    tmp_path.unlink(missing_ok=True)
                raise

        return cache_key

    def migrate_legacy_cache(
        self,
        legacy_dict_or_path: dict[str, str] | Path | str,
        *,
        validator: SemanticSanityValidator | None = None,
    ) -> int:
        """Import legacy cache entries, revalidating and tagging as 'legacy-revalidated'."""
        val = validator or self.validator
        imported_count = 0

        raw_pairs: dict[str, str] = {}
        if isinstance(legacy_dict_or_path, (str, Path)):
            p = Path(legacy_dict_or_path)
            if p.exists():
                try:
                    loaded = json.loads(p.read_text(encoding="utf-8"))
                    if isinstance(loaded, dict):
                        raw_pairs = {str(k): str(v) for k, v in loaded.items()}
                except Exception as exc:
                    logger.warning("Failed to read legacy cache file: %s", type(exc).__name__)
        elif isinstance(legacy_dict_or_path, dict):
            raw_pairs = {str(k): str(v) for k, v in legacy_dict_or_path.items()}

        for src, tgt in raw_pairs.items():
            cleaned_src = normalize_source_text(src)
            cleaned_tgt = " ".join((tgt or "").split()).strip()
            if not cleaned_src or not cleaned_tgt:
                continue

            val_res = val.validate(cleaned_src, cleaned_tgt)
            if not val_res.is_valid:
                logger.debug("Skipping invalid legacy pair during migration")
                continue

            self.put(
                cleaned_src,
                cleaned_tgt,
                origin_provider="legacy-revalidated",
            )
            imported_count += 1

        return imported_count
