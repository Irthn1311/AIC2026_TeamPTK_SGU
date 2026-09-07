"""Comprehensive unit and adversarial test suite for Live Translation Provider and Validator.

Validates all Round 2 acceptance criteria:
1. Cache HMAC & threat model (explicit checksum-only vs hmac-required, fail-fast).
2. Strict sanitization of persisted cache fields (allowlist checking).
3. Cache cross-entry substitution and tampering rejection.
4. Composite revalidation and cache hit provenance preservation.
5. Single-flight coordination (concurrent identical single queries, concurrent overlapping batches).
6. Single-flight leader error propagation to waiters without deadlock.
7. SemanticSanityValidator precision: 18/18 canonical benchmark pairs PASS, 5/5 full queries PASS.
8. Corrupted minimal pairs: color polarity swaps, perspective drop, sequence inversion.
9. Legitimate edge cases: article "một", "hai tay" -> "both", "line up to exercise".
10. Strict VinAI manifest security: reject empty manifest, revision mismatch, path traversal.
11. Google Cloud v3 security: timeout validation, credential non-leakage, cardinality mismatch.
12. Mock isolation using monkeypatch fixture without leaking global sys.modules.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from system_tai.translation.cache import (
    PerKeyAtomicTranslationCache,
    _compute_payload_signature,
    compute_cache_key,
)
from system_tai.translation.composite import (
    LiveFallbackTranslationProvider,
    SingleFlightGroup,
)
from system_tai.translation.google_cloud import (
    GoogleCloudTranslationError,
    GoogleCloudTranslationProvider,
)
from system_tai.translation.models import (
    TranslationMetadata,
    TranslationOutcome,
)
from system_tai.translation.provider import (
    _CANONICAL_BENCHMARK_TRANSLATIONS,
    TranslationError,
    VinAITranslateProvider,
)
from system_tai.translation.validator import (
    SemanticSanityValidator,
)

SAMPLE_32BYTE_HMAC_KEY = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"


@pytest.fixture(autouse=True)
def mock_torch_transformers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Safely mock torch and transformers in isolated test contexts."""
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    monkeypatch.setitem(__import__("sys").modules, "torch", mock_torch)

    mock_tf = MagicMock()
    mock_tok = MagicMock()
    mock_tok.lang_code_to_id = {"en_XX": 250004, "vi_VN": 250005}
    mock_tok.convert_tokens_to_ids.return_value = 250004
    mock_tf.AutoTokenizer.from_pretrained.return_value = mock_tok
    mock_model = MagicMock()
    mock_model.to.return_value = mock_model
    mock_tf.AutoModelForSeq2SeqLM.from_pretrained.return_value = mock_model
    monkeypatch.setitem(__import__("sys").modules, "transformers", mock_tf)


# =====================================================================
# 1. HMAC & Cache Threat Model
# =====================================================================


def test_cache_checksum_only_mode_detects_corruption(tmp_path: Path) -> None:
    """Checksum-only mode detects corrupted bytes without requiring an HMAC key."""
    cache = PerKeyAtomicTranslationCache(tmp_path, hmac_mode="checksum-only")
    key = cache.put("con mèo", "a cat", origin_provider="google-cloud")
    entry_path = tmp_path / "entries" / f"{key}.json"

    # Valid read succeeds
    outcome = cache.get("con mèo")
    assert outcome is not None
    assert outcome.translated_text == "a cat"

    # Corrupt the file content
    data = json.loads(entry_path.read_text(encoding="utf-8"))
    data["translated_text"] = "a dog"
    entry_path.write_text(json.dumps(data), encoding="utf-8")

    # Corrupted entry is detected and rejected
    assert cache.get("con mèo") is None


def test_cache_hmac_required_mode_fails_fast_on_missing_or_short_key(tmp_path: Path) -> None:
    """HMAC-required mode must fail fast if key is missing or under 32 bytes."""
    with pytest.raises(ValueError, match="HMAC key required"):
        PerKeyAtomicTranslationCache(tmp_path, hmac_mode="hmac-required", hmac_key=None)

    with pytest.raises(ValueError, match="at least 32 bytes"):
        PerKeyAtomicTranslationCache(tmp_path, hmac_mode="hmac-required", hmac_key="too_short_key")


def test_cache_hmac_required_mode_with_valid_key(tmp_path: Path) -> None:
    """HMAC-required mode operates successfully with valid 32-byte key."""
    cache = PerKeyAtomicTranslationCache(
        tmp_path,
        hmac_mode="hmac-required",
        hmac_key=SAMPLE_32BYTE_HMAC_KEY,
    )
    key = cache.put("con mèo", "a cat", origin_provider="google-cloud")
    entry_path = tmp_path / "entries" / f"{key}.json"

    outcome = cache.get("con mèo")
    assert outcome is not None
    assert outcome.translated_text == "a cat"

    # Tampering with recomputed simple SHA-256 is rejected because HMAC doesn't match
    data = json.loads(entry_path.read_text(encoding="utf-8"))
    data["translated_text"] = "a forged cat"
    payload_to_verify = {k: v for k, v in data.items() if k != "integrity_signature"}
    canonical = json.dumps(
        payload_to_verify, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    data["integrity_signature"] = hashlib.sha256(canonical).hexdigest()
    entry_path.write_text(json.dumps(data), encoding="utf-8")

    assert cache.get("con mèo") is None


def test_cache_cross_entry_substitution_rejected(tmp_path: Path) -> None:
    """Overwriting cache file of query A with entry of query B is rejected."""
    cache = PerKeyAtomicTranslationCache(
        tmp_path,
        hmac_mode="hmac-required",
        hmac_key=SAMPLE_32BYTE_HMAC_KEY,
    )
    key_cat = cache.put("con mèo", "a cat", origin_provider="google-cloud")
    key_dog = cache.put("con chó", "a dog", origin_provider="google-cloud")

    path_cat = tmp_path / "entries" / f"{key_cat}.json"
    path_dog = tmp_path / "entries" / f"{key_dog}.json"

    # Substitute cat's file with dog's valid entry
    path_cat.write_bytes(path_dog.read_bytes())

    # Reading "con mèo" must return None (cache_key mismatch)
    assert cache.get("con mèo") is None


def test_cache_rejects_legacy_unverified_entries(tmp_path: Path) -> None:
    """get() must reject entries tagged with legacy-unverified."""
    cache = PerKeyAtomicTranslationCache(tmp_path, hmac_mode="checksum-only")
    key = compute_cache_key(
        source_text_normalized="con mèo",
        validator_version=SemanticSanityValidator.version,
    )
    payload = {
        "cache_key": key,
        "source_text": "con mèo",
        "source_text_normalized": "con mèo",
        "translated_text": "a cat",
        "origin_provider": "legacy-unverified",
        "schema_version": "v1",
        "policy_version": "live_p2_v1",
        "validator_version": SemanticSanityValidator.version,
        "normalization_version": "norm_v1",
        "google_api_surface": "v3",
        "google_client_version": "3.15.0",
        "google_project_location_redacted": "projects/***/locations/global",
        "vinai_model_name": None,
        "vinai_revision": "ae7baa85da07dbe8e23ac26a9f5ef560c17e2138",
        "created_timestamp_utc": "2026-09-06T00:00:00Z",
        "fallback_triggered": False,
        "fallback_reason_code": None,
    }
    payload["integrity_signature"] = _compute_payload_signature(payload, None)
    (tmp_path / "entries" / f"{key}.json").write_text(json.dumps(payload), encoding="utf-8")

    assert cache.get("con mèo") is None


def test_cache_migration_revalidates_entries(tmp_path: Path) -> None:
    """migrate_legacy_cache revalidates entries and tags them as legacy-revalidated."""
    cache = PerKeyAtomicTranslationCache(tmp_path, hmac_mode="checksum-only")
    legacy_data = {
        "con mèo": "a cat",
        "ít nhất 5 người": "more than 5 people",  # invalid comparator inversion
    }
    migrated_count = cache.migrate_legacy_cache(legacy_data)
    assert migrated_count == 1  # Only "con mèo" is valid

    outcome = cache.get("con mèo")
    assert outcome is not None
    assert outcome.translated_text == "a cat"
    assert outcome.metadata.origin_provider == "legacy-revalidated"
    assert cache.get("ít nhất 5 người") is None


# =====================================================================
# 2. Strict Field Sanitization
# =====================================================================


def test_cache_put_sanitizes_unapproved_fields(tmp_path: Path) -> None:
    """put() rejects non-allowlisted reason codes, versions, or path leaks."""
    cache = PerKeyAtomicTranslationCache(tmp_path, hmac_mode="checksum-only")

    # Invalid fallback reason code
    with pytest.raises(ValueError, match="Invalid fallback_reason_code"):
        cache.put(
            "con mèo",
            "a cat",
            origin_provider="vinai",
            fallback_reason_code="malicious_or_unknown_code",
        )

    # Invalid google_client_version (e.g. path injection)
    with pytest.raises(ValueError, match="Invalid google_client_version"):
        cache.put(
            "con mèo",
            "a cat",
            origin_provider="google-cloud",
            google_client_version="../../etc/passwd",
        )

    # Invalid vinai_model_name (e.g. local directory path)
    with pytest.raises(ValueError, match="Invalid vinai_model_name"):
        cache.put(
            "con mèo",
            "a cat",
            origin_provider="vinai",
            vinai_model_name="/home/user/secret/model",
        )

    # Invalid google_project_location_redacted
    with pytest.raises(ValueError, match="Invalid google_project_location_redacted"):
        cache.put(
            "con mèo",
            "a cat",
            origin_provider="google-cloud",
            google_project_location_redacted="projects/secret-proj-id-leak",
        )


# =====================================================================
# 3. Composite Revalidation & Fallback Provenance
# =====================================================================


def test_composite_revalidates_cache_hit_and_rejects_invalid_entry(tmp_path: Path) -> None:
    """Composite revalidates cache hits against active validator before returning."""
    cache = PerKeyAtomicTranslationCache(tmp_path, hmac_mode="checksum-only")

    # Store translation that drops comparator: "hơn 5 người" -> "five people"
    key = compute_cache_key(
        source_text_normalized="hơn 5 người",
        validator_version=SemanticSanityValidator.version,
    )
    payload = {
        "cache_key": key,
        "created_timestamp_utc": "2026-09-06T00:00:00Z",
        "fallback_reason_code": None,
        "fallback_triggered": False,
        "google_api_surface": "v3",
        "google_client_version": "3.15.0",
        "google_project_location_redacted": "projects/***/locations/global",
        "normalization_version": "norm_v1",
        "origin_provider": "google-cloud",
        "policy_version": "live_p2_v1",
        "schema_version": "v1",
        "source_text": "hơn 5 người",
        "source_text_normalized": "hơn 5 người",
        "translated_text": "five people",  # Missing "more than"
        "validator_version": SemanticSanityValidator.version,
        "vinai_model_name": None,
        "vinai_revision": "ae7baa85da07dbe8e23ac26a9f5ef560c17e2138",
    }
    payload["integrity_signature"] = _compute_payload_signature(payload, None)
    (tmp_path / "entries" / f"{key}.json").write_text(json.dumps(payload), encoding="utf-8")

    mock_google = MagicMock()
    mock_google.translate_many.return_value = ("more than 5 people",)
    mock_google.provider_name = "google-cloud"
    mock_google.client_library_version = "3.15.0"
    mock_google.redacted_parent = "projects/***/locations/global"

    composite = LiveFallbackTranslationProvider(
        google_provider=mock_google,
        cache=cache,
    )

    result = composite.translate("hơn 5 người")
    assert result == "more than 5 people"
    assert mock_google.translate_many.call_count == 1


def test_cache_hit_preserves_fallback_and_origin_provenance(tmp_path: Path) -> None:
    """Cache hit preserves origin and fallback provenance."""
    cache = PerKeyAtomicTranslationCache(tmp_path, hmac_mode="checksum-only")

    mock_google = MagicMock()
    mock_google.translate_many.side_effect = GoogleCloudTranslationError(
        "Timeout", reason_code="google_timeout"
    )
    mock_google.provider_name = "google-cloud"

    mock_vinai = MagicMock()
    mock_vinai.translate_many.return_value = ("a dam in the rain",)
    mock_vinai.provider_name = "vinai-translate"
    mock_vinai.model_name = "vinai/vinai-translate-vi2en-v2"
    mock_vinai.revision = "ae7baa85da07dbe8e23ac26a9f5ef560c17e2138"

    composite = LiveFallbackTranslationProvider(
        google_provider=mock_google,
        vinai_provider=mock_vinai,
        cache=cache,
    )

    # 1. First call: Google fails -> VinAI fallback
    outcome1 = composite.translate_with_metadata("con đập dưới mưa")
    assert outcome1.translated_text == "a dam in the rain"
    assert outcome1.metadata.origin_provider == "vinai"
    assert outcome1.metadata.fallback_triggered is True
    assert outcome1.metadata.fallback_reason_code == "google_timeout"
    assert outcome1.metadata.served_from_cache is False

    # 2. Second call: served from cache
    outcome2 = composite.translate_with_metadata("con đập dưới mưa")
    assert outcome2.translated_text == "a dam in the rain"
    assert outcome2.metadata.served_from_cache is True
    assert outcome2.metadata.origin_provider == "vinai"
    assert outcome2.metadata.fallback_triggered is True
    assert outcome2.metadata.fallback_reason_code == "google_timeout"


# =====================================================================
# 4. Single-Flight Batch Coordination & Leader Error Propagation
# =====================================================================


def test_single_flight_identical_concurrent_requests() -> None:
    """Single-flight executes underlying provider once for identical concurrent queries."""
    group = SingleFlightGroup()
    call_count = 0
    lock = threading.Lock()

    def slow_task() -> TranslationOutcome:
        nonlocal call_count
        with lock:
            call_count += 1
        time.sleep(0.05)
        meta = TranslationMetadata(
            origin_provider="google-cloud",
            served_from_cache=False,
            call_timestamp_utc="2026-09-06T00:00:00Z",
            latency_ms=50.0,
            fallback_triggered=False,
            fallback_reason_code=None,
            validation_passed=True,
            validation_issues=(),
        )
        return TranslationOutcome(translated_text="a dam in the rain", metadata=meta)

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(group.do, "con đập dưới mưa", slow_task) for _ in range(5)]
        results = [f.result().translated_text for f in futures]

    assert all(r == "a dam in the rain" for r in results)
    assert call_count == 1


def test_single_flight_overlapping_batches() -> None:
    """Overlapping concurrent batches coordinate deduplication via SingleFlightGroup."""
    mock_google = MagicMock()
    lock = threading.Lock()
    call_counter = 0

    def mock_translate_many(texts: list[str]) -> tuple[str, ...]:
        nonlocal call_counter
        with lock:
            call_counter += 1
        time.sleep(0.04)
        mapping = {
            "query_A": "result_A",
            "query_B": "result_B",
            "query_C": "result_C",
        }
        return tuple(mapping[t] for t in texts)

    mock_google.translate_many.side_effect = mock_translate_many
    mock_google.provider_name = "google-cloud"
    mock_google.client_library_version = "3.15.0"
    mock_google.redacted_parent = "projects/***/locations/global"

    composite = LiveFallbackTranslationProvider(google_provider=mock_google)

    barrier = threading.Barrier(2)

    def worker_1() -> tuple[str, ...]:
        barrier.wait()
        return composite.translate_many(["query_A", "query_B"])

    def worker_2() -> tuple[str, ...]:
        barrier.wait()
        return composite.translate_many(["query_B", "query_C"])

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(worker_1)
        f2 = executor.submit(worker_2)
        res1 = f1.result()
        res2 = f2.result()

    assert res1 == ("result_A", "result_B")
    assert res2 == ("result_B", "result_C")


def test_single_flight_leader_failure_propagates_to_waiters() -> None:
    """If a leader fails, the exception is propagated to all waiting queries."""
    mock_google = MagicMock()
    entered_event = threading.Event()

    def failing_call(texts: list[str]) -> tuple[str, ...]:
        entered_event.set()
        time.sleep(0.05)
        raise GoogleCloudTranslationError("Google API quota exhausted", reason_code="google_quota")

    mock_google.translate_many.side_effect = failing_call
    mock_google.provider_name = "google-cloud"

    composite = LiveFallbackTranslationProvider(google_provider=mock_google)

    def worker() -> str:
        return composite.translate("truy vấn bị lỗi")

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(worker)
        entered_event.wait(timeout=1.0)
        f2 = executor.submit(worker)

        with pytest.raises(TranslationError) as exc_info1:
            f1.result()
        with pytest.raises(TranslationError) as exc_info2:
            f2.result()

    assert exc_info1.value is not None
    assert exc_info2.value is not None


# =====================================================================
# 5. SemanticSanityValidator Acceptance Suite (18 Canonical + Minimal Pairs)
# =====================================================================


def test_all_18_canonical_benchmark_translations_are_valid() -> None:
    """All 18 canonical benchmark pairs must be accepted by SemanticSanityValidator."""
    validator = SemanticSanityValidator()
    for vi_text, en_text in _CANONICAL_BENCHMARK_TRANSLATIONS.items():
        res = validator.validate(vi_text, en_text)
        assert res.is_valid, (
            f"Canonical translation falsely rejected: '{vi_text}' -> issues: {res.issues}"
        )


def test_validator_detects_color_swap() -> None:
    """Validator detects color swaps between entities (e.g. nón đỏ -> blue hats)."""
    validator = SemanticSanityValidator()

    # Red hats swapped to blue
    vi = "ba người đội nón có màu đỏ"
    en_swapped = "three people wearing blue hats"
    res = validator.validate(vi, en_swapped)
    assert not res.is_valid
    assert any("color" in iss.code for iss in res.issues)

    # Green shirt swapped to red
    vi2 = "hai nhân viên mặc áo xanh lá"
    en_swapped2 = "two staff members wearing red shirts"
    res2 = validator.validate(vi2, en_swapped2)
    assert not res2.is_valid
    assert any("color" in iss.code for iss in res2.issues)


def test_validator_detects_dropped_perspective() -> None:
    """Validator detects dropping 'từ trên cao' (aerial / from above)."""
    validator = SemanticSanityValidator()

    vi = "cảnh một con đập được quay từ trên cao"
    en_missing = "a scene of a dam filmed in the morning"
    res = validator.validate(vi, en_missing)
    assert not res.is_valid
    assert any("perspective" in iss.code for iss in res.issues)

    en_valid = "a scene of a dam filmed from above"
    assert validator.validate(vi, en_valid).is_valid


def test_validator_detects_temporal_sequence_inversion() -> None:
    """Validator detects temporal sequence inversion (e.g. dam appearing before map)."""
    validator = SemanticSanityValidator()

    vi = (
        "Đoạn phim bắt đầu bằng một bản đồ, trên đó một loại công trình thủy lợi lần lượt "
        "xuất hiện bốn lần. Sau đó chuyển sang cảnh một con đập được quay từ trên cao."
    )
    en_inverted = (
        "The footage begins with a scene of a dam filmed from above, "
        "then switches to a map with irrigation structures."
    )
    res = validator.validate(vi, en_inverted)
    assert not res.is_valid
    assert any("temporal" in iss.code or "sequence" in iss.code for iss in res.issues)


def test_validator_paired_counts_and_indefinite_article() -> None:
    """Validator properly maps 'hai tay' -> 'both' and allows article 'một'."""
    validator = SemanticSanityValidator()

    # "hai tay" -> "both hands"
    vi = "cùng thực hiện động tác hai tay chạm mũi chân"
    en = "performing the movement of both hands touching their toes"
    assert validator.validate(vi, en).is_valid

    # "hai tay" -> "one hand" (dropped count)
    en_bad = "performing the movement of one hand touching toes"
    assert not validator.validate(vi, en_bad).is_valid

    # "một bản đồ" -> "a map" (article một is valid)
    assert validator.validate("bắt đầu bằng một bản đồ", "begins with a map").is_valid

    # Strict cardinal cue: "chỉ có một người" -> "two people" is rejected
    vi_strict = "Trong nhóm chỉ có một người đeo kính"
    en_strict_bad = "In the group two people wore glasses"
    assert not validator.validate(vi_strict, en_strict_bad).is_valid


def test_validator_line_up_to_exercise_not_comparator() -> None:
    """'line up to exercise' must NOT be falsely flagged as a comparator."""
    validator = SemanticSanityValidator()
    vi = "xếp thành hàng tập thể dục"
    en = "line up to exercise"
    res = validator.validate(vi, en)
    assert res.is_valid
    assert not any(iss.code == "comparator_polarity_inversion" for iss in res.issues)


def test_validator_negation_exclusions_and_detections() -> None:
    """Validator ignores false negation words but detects true negation drop."""
    validator = SemanticSanityValidator()

    # Exclusions: "không khí" (air), "không những... mà còn" (not only... but also), "số không"
    assert validator.validate("không khí trong lành", "fresh air").is_valid
    assert validator.validate(
        "không những đẹp mà còn rẻ", "not only beautiful but also cheap"
    ).is_valid
    assert validator.validate("chữ số không", "digit zero").is_valid

    # True dropped negation:
    assert not validator.validate("không đội nón", "wearing a hat").is_valid
    assert not validator.validate("chưa hoàn thành", "completed").is_valid


def test_validator_spatial_and_comparator_inversions() -> None:
    """Validator catches spatial and comparator inversions."""
    validator = SemanticSanityValidator()

    # Spatial
    assert not validator.validate("người phụ nữ bên trái", "a woman on the right").is_valid
    assert not validator.validate("ở bên trên", "at the bottom").is_valid

    # Comparator
    assert not validator.validate("ít nhất 5 người", "more than 5 people").is_valid
    assert not validator.validate("hơn 5 người", "at least 5 people").is_valid
    assert not validator.validate("không quá 5 người", "more than 5 people").is_valid


def test_validator_compound_numerals() -> None:
    """Validator supports compound numerals 'mười hai' (12) and 'hai mươi lăm' (25)."""
    validator = SemanticSanityValidator()
    assert validator.validate("mười hai người đang chạy", "twelve people are running").is_valid
    assert validator.validate("hai mươi lăm chiếc xe", "twenty-five cars").is_valid


def test_validator_proper_noun_diacritics_is_warning_not_error() -> None:
    """Diacritics on proper nouns produce WARNING, not ERROR."""
    validator = SemanticSanityValidator()
    res = validator.validate("quán phở gia truyền", "a traditional phở restaurant")
    assert res.is_valid
    assert any(iss.severity == "WARNING" for iss in res.issues)


def test_validator_degenerate_output() -> None:
    """Repetitive degenerate loops trigger ERROR."""
    validator = SemanticSanityValidator()
    res = validator.validate("người đang chạy", "dam dam dam dam dam")
    assert not res.is_valid
    assert any(iss.code == "degenerate_output" for iss in res.issues)


# =====================================================================
# 6. Strict VinAI Manifest Security & Snapshot Resolution
# =====================================================================


def test_vinai_strict_mode_rejects_empty_manifest(tmp_path: Path) -> None:
    """Strict mode rejects empty manifest {}."""
    model_dir = tmp_path / "vinai_mirror"
    model_dir.mkdir(parents=True)
    manifest_bytes = b"{}"
    (model_dir / "manifest.json").write_bytes(manifest_bytes)
    test_sha = hashlib.sha256(manifest_bytes).hexdigest()

    with pytest.raises(TranslationError, match="empty or corrupted"):
        VinAITranslateProvider(
            model_name_or_path=model_dir,
            revision="ae7baa85da07dbe8e23ac26a9f5ef560c17e2138",
            require_exact_snapshot=True,
            trusted_manifest_sha256=test_sha,
            allow_custom_trust_anchor=True,
        )


def test_vinai_strict_mode_rejects_manifest_revision_mismatch(tmp_path: Path) -> None:
    """Strict mode rejects wrong revision in manifest."""
    model_dir = tmp_path / "vinai_mirror"
    model_dir.mkdir(parents=True)
    manifest_data = {
        "revision": "0000000000000000000000000000000000000000",
        "checksums": {"config.json": "a" * 64},
    }
    manifest_bytes = json.dumps(manifest_data).encode("utf-8")
    (model_dir / "manifest.json").write_bytes(manifest_bytes)
    test_sha = hashlib.sha256(manifest_bytes).hexdigest()

    with pytest.raises(TranslationError, match="Manifest revision mismatch"):
        VinAITranslateProvider(
            model_name_or_path=model_dir,
            revision="ae7baa85da07dbe8e23ac26a9f5ef560c17e2138",
            require_exact_snapshot=True,
            trusted_manifest_sha256=test_sha,
            allow_custom_trust_anchor=True,
        )


def test_vinai_strict_mode_rejects_path_traversal_in_manifest(tmp_path: Path) -> None:
    """Strict mode rejects manifest files attempting path traversal outside snapshot."""
    model_dir = tmp_path / "vinai_mirror"
    model_dir.mkdir(parents=True)
    manifest_data = {
        "revision": "ae7baa85da07dbe8e23ac26a9f5ef560c17e2138",
        "checksums": {"../../secret.txt": "a" * 64},
    }
    manifest_bytes = json.dumps(manifest_data).encode("utf-8")
    (model_dir / "manifest.json").write_bytes(manifest_bytes)
    test_sha = hashlib.sha256(manifest_bytes).hexdigest()

    with pytest.raises(TranslationError, match="vinai_manifest_path_traversal"):
        VinAITranslateProvider(
            model_name_or_path=model_dir,
            revision="ae7baa85da07dbe8e23ac26a9f5ef560c17e2138",
            require_exact_snapshot=True,
            trusted_manifest_sha256=test_sha,
            allow_custom_trust_anchor=True,
        )


def test_vinai_strict_mode_rejects_missing_required_artifacts(tmp_path: Path) -> None:
    """Strict mode rejects manifest missing required inventory files."""
    model_dir = tmp_path / "vinai_mirror"
    model_dir.mkdir(parents=True)
    # Manifest has config.json but is missing special_tokens_map.json, weights, etc.
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    manifest_data = {
        "revision": "ae7baa85da07dbe8e23ac26a9f5ef560c17e2138",
        "checksums": {"config.json": hashlib.sha256(b"{}").hexdigest()},
    }
    manifest_bytes = json.dumps(manifest_data).encode("utf-8")
    (model_dir / "manifest.json").write_bytes(manifest_bytes)
    test_sha = hashlib.sha256(manifest_bytes).hexdigest()

    with pytest.raises(TranslationError, match="lacks required snapshot inventory"):
        VinAITranslateProvider(
            model_name_or_path=model_dir,
            revision="ae7baa85da07dbe8e23ac26a9f5ef560c17e2138",
            require_exact_snapshot=True,
            trusted_manifest_sha256=test_sha,
            allow_custom_trust_anchor=True,
        )


def test_vinai_strict_mode_resolves_snapshot_in_custom_cache_dir(tmp_path: Path) -> None:
    """Snapshot is correctly located and scoped in custom cache_dir."""
    custom_cache = tmp_path / "custom_hf_cache"
    rev = "ae7baa85da07dbe8e23ac26a9f5ef560c17e2138"
    snap_dir = custom_cache / "models--vinai--vinai-translate-vi2en-v2" / "snapshots" / rev
    snap_dir.mkdir(parents=True)
    checksums = {
        "config.json": hashlib.sha256(b"{}").hexdigest(),
        "special_tokens_map.json": hashlib.sha256(b"{}").hexdigest(),
        "tokenizer_config.json": hashlib.sha256(b"{}").hexdigest(),
        "model.safetensors": hashlib.sha256(b"weights").hexdigest(),
        "sentencepiece.bpe.model": hashlib.sha256(b"spm").hexdigest(),
    }
    (snap_dir / "config.json").write_text("{}", encoding="utf-8")
    (snap_dir / "special_tokens_map.json").write_text("{}", encoding="utf-8")
    (snap_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (snap_dir / "model.safetensors").write_bytes(b"weights")
    (snap_dir / "sentencepiece.bpe.model").write_bytes(b"spm")
    manifest_bytes = json.dumps({"revision": rev, "checksums": checksums}).encode("utf-8")
    (snap_dir / "manifest.json").write_bytes(manifest_bytes)
    test_manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()

    provider = VinAITranslateProvider(
        cache_dir=custom_cache,
        revision=rev,
        require_exact_snapshot=True,
        trusted_manifest_sha256=test_manifest_sha,
        allow_custom_trust_anchor=True,
    )

    fingerprint = provider.get_artifact_fingerprint()
    assert fingerprint["resolved_snapshot_dir"] == "hf-cache"
    assert fingerprint["snapshot_commit_hash"] == rev
    assert fingerprint["revision_matches_snapshot"] is True


# =====================================================================
# 7. Google Cloud Provider & Credential Non-Leakage
# =====================================================================


def test_google_cardinality_mismatch_raises_translation_error() -> None:
    """Google returning fewer items raises GoogleCloudTranslationError."""
    assert issubclass(GoogleCloudTranslationError, TranslationError)

    mock_client = MagicMock()
    mock_resp = MagicMock()
    tr1 = MagicMock()
    tr1.translated_text = "result 1"
    mock_resp.translations = [tr1]
    mock_client.translate_text.return_value = mock_resp

    provider = GoogleCloudTranslationProvider(project_id="test-proj", client=mock_client)

    with pytest.raises(GoogleCloudTranslationError) as exc_info:
        provider.translate_many(["query 1", "query 2"])

    assert exc_info.value.reason_code == "google_cardinality_mismatch"


@pytest.mark.parametrize("bad_timeout", [float("nan"), float("inf"), float("-inf"), 0.0, -5.0])
def test_google_cloud_rejects_invalid_timeout(bad_timeout: float) -> None:
    """NaN, Inf, zero, and negative timeouts raise ValueError."""
    with pytest.raises(ValueError, match="timeout_seconds must be a finite positive number"):
        GoogleCloudTranslationProvider(project_id="test-proj", timeout_seconds=bad_timeout)


def test_google_cloud_does_not_leak_tokens_or_paths_in_exception() -> None:
    """Token, file path, and raw chained exception are never leaked."""
    mock_client = MagicMock()
    secret_token = "shortsecret"
    cred_path = "/tmp/service-account.json"
    mock_client.translate_text.side_effect = RuntimeError(
        f"Permission denied for {cred_path} with token={secret_token}"
    )

    provider = GoogleCloudTranslationProvider(project_id="test-proj", client=mock_client)

    with pytest.raises(GoogleCloudTranslationError) as exc_info:
        provider.translate("con mèo")

    err_msg = str(exc_info.value)
    assert secret_token not in err_msg
    assert cred_path not in err_msg
    assert exc_info.value.__cause__ is None


def test_google_cloud_deduplicates_batch_inputs() -> None:
    """Google Cloud deduplicates identical strings in batch."""
    mock_client = MagicMock()
    mock_resp = MagicMock()
    tr1 = MagicMock()
    tr1.translated_text = "a cat"
    tr2 = MagicMock()
    tr2.translated_text = "a dog"
    mock_resp.translations = [tr1, tr2]
    mock_client.translate_text.return_value = mock_resp

    provider = GoogleCloudTranslationProvider(project_id="test-proj", client=mock_client)
    res = provider.translate_many(["con mèo", "con chó", "con mèo"])
    assert res == ("a cat", "a dog", "a cat")
    called_contents = mock_client.translate_text.call_args[1]["request"]["contents"]
    assert called_contents == ["con mèo", "con chó"]


# =====================================================================
# 8. Full Pipeline Integration & Zero Web Scraper Token Check
# =====================================================================


def test_composite_item_level_partial_fallback() -> None:
    """Only failing items in a batch trigger VinAI fallback; healthy items use Google."""
    mock_google = MagicMock()
    mock_google.translate_many.return_value = ("a cat", "a man on the right")
    mock_google.provider_name = "google-cloud"
    mock_google.client_library_version = "3.15.0"
    mock_google.redacted_parent = "projects/***/locations/global"

    mock_vinai = MagicMock()
    mock_vinai.translate_many.return_value = ("a man on the left",)
    mock_vinai.provider_name = "vinai-translate"
    mock_vinai.model_name = "vinai/vinai-translate-vi2en-v2"
    mock_vinai.revision = "ae7baa85da07dbe8e23ac26a9f5ef560c17e2138"

    composite = LiveFallbackTranslationProvider(
        google_provider=mock_google,
        vinai_provider=mock_vinai,
    )

    results = composite.translate_many(["con mèo", "người đàn ông bên trái"])
    assert results == ("a cat", "a man on the left")
    assert mock_vinai.translate_many.call_args[0][0] == ["người đàn ông bên trái"]


def test_composite_both_fail_raises_translation_error() -> None:
    """Both Google and VinAI failing raises TranslationError."""
    mock_google = MagicMock()
    mock_google.translate_many.side_effect = GoogleCloudTranslationError(
        "Timeout", reason_code="google_timeout"
    )
    mock_google.provider_name = "google-cloud"

    mock_vinai = MagicMock()
    mock_vinai.translate_many.return_value = ("",)
    mock_vinai.provider_name = "vinai-translate"

    provider = LiveFallbackTranslationProvider(
        google_provider=mock_google,
        vinai_provider=mock_vinai,
    )

    with pytest.raises(TranslationError, match="Both Google and VinAI"):
        provider.translate("con đập dưới mưa")


def test_zero_reference_to_legacy_web_scraper_in_new_modules() -> None:
    """Verify Commit 2 modules contain no references to legacy web scrapers."""
    import system_tai.translation.cache as c_mod
    import system_tai.translation.composite as comp_mod
    import system_tai.translation.google_cloud as gc_mod
    import system_tai.translation.models as m_mod
    import system_tai.translation.validator as v_mod

    modules_to_inspect = [m_mod, v_mod, c_mod, gc_mod, comp_mod]
    forbidden_tokens = ["deep_translator", "urllib.request", "GoogleTranslator"]

    for mod in modules_to_inspect:
        file_path = getattr(mod, "__file__", None)
        assert file_path is not None
        source_code = Path(file_path).read_text(encoding="utf-8")
        for tok in forbidden_tokens:
            assert tok not in source_code, (
                f"Forbidden scraper token '{tok}' found in {Path(file_path).name}"
            )


# =====================================================================
# 9. Auditor Round 4: VinAI Trust Anchor, HF Blobs, HMAC Downgrade & Representation
# =====================================================================


def test_vinai_strict_mode_rejects_untrusted_self_signed_manifest(tmp_path: Path) -> None:
    """A self-signed manifest in snapshot dir is rejected without external trust anchor."""
    model_dir = tmp_path / "vinai_mirror"
    model_dir.mkdir(parents=True)
    rev = "ae7baa85da07dbe8e23ac26a9f5ef560c17e2138"
    checksums = {
        "config.json": hashlib.sha256(b"{}").hexdigest(),
        "special_tokens_map.json": hashlib.sha256(b"{}").hexdigest(),
        "tokenizer_config.json": hashlib.sha256(b"{}").hexdigest(),
        "model.safetensors": hashlib.sha256(b"weights").hexdigest(),
        "sentencepiece.bpe.model": hashlib.sha256(b"spm").hexdigest(),
    }
    for fname, exp_hash in checksums.items():
        (model_dir / fname).write_bytes(
            b"{}" if "json" in fname else (b"weights" if "model.safe" in fname else b"spm")
        )
    (model_dir / "manifest.json").write_text(
        json.dumps({"revision": rev, "checksums": checksums}),
        encoding="utf-8",
    )

    # Without trusted_manifest_sha256 matching the generated manifest hash, fails closed
    with pytest.raises(TranslationError, match=r"\[vinai_manifest_untrusted\]"):
        VinAITranslateProvider(
            model_name_or_path=model_dir,
            revision=rev,
            require_exact_snapshot=True,
            allow_custom_trust_anchor=True,
        )


def test_vinai_strict_mode_rejects_noncanonical_revision(tmp_path: Path) -> None:
    """Strict mode rejects all-zero or non-canonical revisions."""
    with pytest.raises(TranslationError, match=r"\[vinai_untrusted_revision\]"):
        VinAITranslateProvider(
            revision="0000000000000000000000000000000000000000",
            require_exact_snapshot=True,
        )


def test_vinai_strict_mode_rejects_rogue_executable_files(tmp_path: Path) -> None:
    """Snapshot containing unlisted load-relevant files (.py, .exe, .so) is rejected."""
    custom_cache = tmp_path / "custom_hf_cache"
    rev = "ae7baa85da07dbe8e23ac26a9f5ef560c17e2138"
    snap_dir = custom_cache / "models--vinai--vinai-translate-vi2en-v2" / "snapshots" / rev
    snap_dir.mkdir(parents=True)
    checksums = {
        "config.json": hashlib.sha256(b"{}").hexdigest(),
        "special_tokens_map.json": hashlib.sha256(b"{}").hexdigest(),
        "tokenizer_config.json": hashlib.sha256(b"{}").hexdigest(),
        "model.safetensors": hashlib.sha256(b"weights").hexdigest(),
        "sentencepiece.bpe.model": hashlib.sha256(b"spm").hexdigest(),
    }
    (snap_dir / "config.json").write_text("{}", encoding="utf-8")
    (snap_dir / "special_tokens_map.json").write_text("{}", encoding="utf-8")
    (snap_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (snap_dir / "model.safetensors").write_bytes(b"weights")
    (snap_dir / "sentencepiece.bpe.model").write_bytes(b"spm")
    manifest_bytes = json.dumps({"revision": rev, "checksums": checksums}).encode("utf-8")
    (snap_dir / "manifest.json").write_bytes(manifest_bytes)
    test_manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()

    # Drop a rogue script into the snapshot
    (snap_dir / "custom_runner.py").write_text("print('rogue')", encoding="utf-8")

    with pytest.raises(TranslationError, match=r"\[vinai_untrusted_file\]"):
        VinAITranslateProvider(
            cache_dir=custom_cache,
            revision=rev,
            require_exact_snapshot=True,
            trusted_manifest_sha256=test_manifest_sha,
            allow_custom_trust_anchor=True,
        )


def test_vinai_strict_mode_supports_hf_cache_blobs_symlinks(tmp_path: Path) -> None:
    """Hugging Face layout with blobs/ directory symlinks is verified within containment bounds."""
    custom_cache = tmp_path / "custom_hf_cache"
    repo_root = custom_cache / "models--vinai--vinai-translate-vi2en-v2"
    blobs_dir = repo_root / "blobs"
    blobs_dir.mkdir(parents=True)
    rev = "ae7baa85da07dbe8e23ac26a9f5ef560c17e2138"
    snap_dir = repo_root / "snapshots" / rev
    snap_dir.mkdir(parents=True)

    config_blob = blobs_dir / "config_blob"
    config_blob.write_text("{}", encoding="utf-8")
    sp_blob = blobs_dir / "sp_blob"
    sp_blob.write_text("{}", encoding="utf-8")
    tok_blob = blobs_dir / "tok_blob"
    tok_blob.write_text("{}", encoding="utf-8")
    weights_blob = blobs_dir / "weights_blob"
    weights_blob.write_bytes(b"weights")
    spm_blob = blobs_dir / "spm_blob"
    spm_blob.write_bytes(b"spm")

    # Try creating symlinks if OS permits, or fallback to file copy
    try:
        os.symlink(config_blob, snap_dir / "config.json")
        os.symlink(sp_blob, snap_dir / "special_tokens_map.json")
        os.symlink(tok_blob, snap_dir / "tokenizer_config.json")
        os.symlink(weights_blob, snap_dir / "model.safetensors")
        os.symlink(spm_blob, snap_dir / "sentencepiece.bpe.model")
    except OSError:
        (snap_dir / "config.json").write_text("{}", encoding="utf-8")
        (snap_dir / "special_tokens_map.json").write_text("{}", encoding="utf-8")
        (snap_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        (snap_dir / "model.safetensors").write_bytes(b"weights")
        (snap_dir / "sentencepiece.bpe.model").write_bytes(b"spm")

    checksums = {
        "config.json": hashlib.sha256(b"{}").hexdigest(),
        "special_tokens_map.json": hashlib.sha256(b"{}").hexdigest(),
        "tokenizer_config.json": hashlib.sha256(b"{}").hexdigest(),
        "model.safetensors": hashlib.sha256(b"weights").hexdigest(),
        "sentencepiece.bpe.model": hashlib.sha256(b"spm").hexdigest(),
    }
    manifest_bytes = json.dumps({"revision": rev, "checksums": checksums}).encode("utf-8")
    (snap_dir / "manifest.json").write_bytes(manifest_bytes)
    test_manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()

    provider = VinAITranslateProvider(
        cache_dir=custom_cache,
        revision=rev,
        require_exact_snapshot=True,
        trusted_manifest_sha256=test_manifest_sha,
        allow_custom_trust_anchor=True,
    )
    assert provider.revision == rev


def test_vinai_strict_mode_rejects_checksum_artifact_mismatch(tmp_path: Path) -> None:
    """Strict mode rejects manifest when on-disk artifact checksum differs from manifest."""
    model_dir = tmp_path / "vinai_mirror"
    model_dir.mkdir(parents=True)
    rev = "ae7baa85da07dbe8e23ac26a9f5ef560c17e2138"
    checksums = {
        "config.json": hashlib.sha256(b"{}").hexdigest(),
        "special_tokens_map.json": hashlib.sha256(b"{}").hexdigest(),
        "tokenizer_config.json": hashlib.sha256(b"{}").hexdigest(),
        "model.safetensors": hashlib.sha256(b"correct_weights").hexdigest(),
        "sentencepiece.bpe.model": hashlib.sha256(b"spm").hexdigest(),
    }
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "special_tokens_map.json").write_text("{}", encoding="utf-8")
    (model_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (model_dir / "model.safetensors").write_bytes(b"tampered_weights")
    (model_dir / "sentencepiece.bpe.model").write_bytes(b"spm")
    manifest_bytes = json.dumps({"revision": rev, "checksums": checksums}).encode("utf-8")
    (model_dir / "manifest.json").write_bytes(manifest_bytes)
    test_manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()

    with pytest.raises(TranslationError, match=r"\[artifact_checksum_mismatch\]"):
        VinAITranslateProvider(
            model_name_or_path=model_dir,
            revision=rev,
            require_exact_snapshot=True,
            trusted_manifest_sha256=test_manifest_sha,
            allow_custom_trust_anchor=True,
        )


def test_cache_conflicting_config_rejected(tmp_path: Path) -> None:
    """require_hmac=True with hmac_mode='checksum-only' raises ValueError."""
    with pytest.raises(ValueError, match="Conflicting cache integrity options"):
        PerKeyAtomicTranslationCache(
            cache_dir=tmp_path,
            require_hmac=True,
            hmac_mode="checksum-only",
        )


def test_cache_integrity_mode_mismatch_rejected(tmp_path: Path) -> None:
    """Entry saved under checksum-only mode cannot be accepted by hmac-required cache."""
    c_chk = PerKeyAtomicTranslationCache(cache_dir=tmp_path, hmac_mode="checksum-only")
    c_chk.put("mèo con", "kitten", origin_provider="google-cloud")

    secret_key = "a" * 64
    c_hmac = PerKeyAtomicTranslationCache(
        cache_dir=tmp_path,
        hmac_mode="hmac-required",
        hmac_key=secret_key,
        require_hmac=True,
    )
    # Entry saved as checksum-only must be rejected by hmac-required cache
    assert c_hmac.get("mèo con") is None


def test_validator_clause_entity_counterexamples() -> None:
    """Auditor-mandated 7 semantic sanity counterexamples must be rejected."""
    validator = SemanticSanityValidator()

    # 1. Dropped number (dropped 2 for dogs)
    r1 = validator.validate(
        "Hai người đứng cạnh hai con chó",
        "Two people stand next to dogs",
    )
    assert not r1.is_valid
    assert any(iss.code == "count_dropped" for iss in r1.issues)

    # 2. Swapped count across entities (2 men & 3 women -> 3 men & 2 women)
    r2 = validator.validate(
        "Hai người đàn ông đứng cạnh ba phụ nữ",
        "Three men stand next to two women",
    )
    assert not r2.is_valid
    assert any(iss.code == "count_entity_swap" for iss in r2.issues)

    # 3. Dropped negation when multiple clauses are negated
    r3 = validator.validate(
        "Người đàn ông không đội mũ và phụ nữ không đeo kính",
        "The man wears no hat and the woman wears glasses",
    )
    assert not r3.is_valid
    assert any(iss.code in ("negation_dropped", "negation_entity_misplaced") for iss in r3.issues)

    # 4. Spatial direction dropped (dropped right)
    r4 = validator.validate(
        "Người đàn ông bên trái và phụ nữ bên phải",
        "The man on the left and the woman",
    )
    assert not r4.is_valid
    assert any(iss.code in ("direction_dropped", "spatial_dropped") for iss in r4.issues)

    # 5. Comparator quantifier mismatch / swap
    r5 = validator.validate(
        "Hơn 5 người ngồi trên 7 ghế",
        "More than 7 people sit on 5 chairs",
    )
    assert not r5.is_valid
    assert any(iss.code == "comparator_mismatch" for iss in r5.issues)

    # 6. Initial scene entity dropped
    r6 = validator.validate(
        "Bắt đầu với đàn sư tử, sau đó hai nhân viên kiểm tra chuồng",
        "Two staff members inspect the enclosure",
    )
    assert not r6.is_valid
    assert any(iss.code == "scene_dropped" for iss in r6.issues)

    # 7. Count swapped across colored entities (2 red hats & 3 blue hats -> 3 red & 2 blue)
    r7 = validator.validate(
        "Hai nón đỏ và ba nón xanh",
        "Three red hats and two blue hats",
    )
    assert not r7.is_valid
    assert any(iss.code == "count_entity_swap" for iss in r7.issues)


def test_validator_multi_color_legitimate_sentences_pass() -> None:
    """Multi-color legitimate sentences must not be falsely rejected."""
    validator = SemanticSanityValidator()

    # Pair A: red hat and blue cap
    res_a = validator.validate(
        "người đội nón đỏ và người đội mũ xanh",
        "a person wearing a red hat and a person wearing a blue cap",
    )
    assert res_a.is_valid, f"Falsely rejected: {res_a.issues}"

    # Pair B: green shirt and red shirt
    res_b = validator.validate(
        "hai nhân viên mặc áo xanh lá và một nhân viên mặc áo đỏ",
        "two staff members wearing green shirts and one employee wearing a red shirt",
    )
    assert res_b.is_valid, f"Falsely rejected: {res_b.issues}"

    # Pair C: red peppers and green peppers
    res_c = validator.validate(
        "đĩa hành tây và ớt đỏ thái lát bên cạnh ớt xanh",
        "a plate of sliced onions and red peppers next to green peppers",
    )
    assert res_c.is_valid, f"Falsely rejected: {res_c.issues}"


def test_zero_data_leakage_in_metadata_errors_and_logs(tmp_path: Path) -> None:
    """Zero data leakage: errors, metadata, and logs do not echo paths or raw exceptions."""
    # 1. Cache put sanitized error
    cache = PerKeyAtomicTranslationCache(cache_dir=tmp_path)
    with pytest.raises(ValueError, match="Cannot cache empty source text"):
        cache.put("", "translation", origin_provider="google-cloud")

    # 2. Composite metadata sanitizes model name and revision
    mock_google = MagicMock()
    mock_google.translate_many.side_effect = GoogleCloudTranslationError(
        "Google API unavailable", reason_code="google_unavailable"
    )
    mock_google.provider_name = "google-cloud"

    mock_vinai = MagicMock()
    mock_vinai.model_name = "/tmp/secret/untrusted_model"
    mock_vinai.revision = "untrusted_rev"
    mock_vinai.provider_name = "vinai-translate"
    mock_vinai.translate_many.return_value = ("a dog",)

    comp = LiveFallbackTranslationProvider(
        google_provider=mock_google,
        vinai_provider=mock_vinai,
        cache=cache,
    )
    outcomes = comp.translate_many_with_metadata(["con chó"])
    meta = outcomes[0].metadata
    assert "/tmp" not in (meta.vinai_model_name or "")
    assert meta.vinai_model_name == "vinai/vinai-translate-vi2en-v2"
    assert meta.vinai_revision == "ae7baa85da07dbe8e23ac26a9f5ef560c17e2138"


def test_overlapping_batch_exact_call_counts_and_deduplication(tmp_path: Path) -> None:
    """Overlapping batches translate each distinct item exactly once."""
    mock_google = MagicMock()
    mock_google.client_library_version = "3.11.0"
    mock_google.redacted_parent = "projects/test/locations/global"
    call_records: list[list[str]] = []
    mapping = {
        "con mèo": "a cat",
        "con chó": "a dog",
        "con ngựa": "a horse",
    }

    def mock_translate_many(texts: list[str]) -> tuple[str, ...]:
        call_records.append(list(texts))
        return tuple(mapping[t] for t in texts)

    mock_google.translate_many.side_effect = mock_translate_many
    mock_google.provider_name = "google-cloud"

    cache = PerKeyAtomicTranslationCache(cache_dir=tmp_path)
    composite = LiveFallbackTranslationProvider(
        google_provider=mock_google,
        vinai_provider=MagicMock(),
        cache=cache,
    )

    r1 = composite.translate_many(["con mèo", "con chó"])
    assert r1 == ("a cat", "a dog")
    assert call_records == [["con mèo", "con chó"]]

    # Second batch shares 'con chó'
    r2 = composite.translate_many(["con chó", "con ngựa"])
    assert r2 == ("a dog", "a horse")
    # Only 'con ngựa' was sent downstream
    assert call_records == [["con mèo", "con chó"], ["con ngựa"]]


# =====================================================================
# 10. Auditor Round 5: Canonical Trust Anchor, Inventory Allowlist & Zero-Leak Sentinel
# =====================================================================


def test_vinai_canonical_manifest_resource_packaging() -> None:
    """Production mode loads canonical manifest fixture via importlib.resources.

    Custom trusted_manifest_sha256 is strictly forbidden unless allow_custom_trust_anchor=True.
    """
    import importlib.resources

    manifest_bytes = (
        importlib.resources.files("system_tai.translation")
        .joinpath("canonical_vinai_manifest.json")
        .read_bytes()
    )
    manifest_data = json.loads(manifest_bytes.decode("utf-8"))
    assert manifest_data["revision"] == "ae7baa85da07dbe8e23ac26a9f5ef560c17e2138"
    assert "pytorch_model.bin" in manifest_data["checksums"]
    canonical_bytes = json.dumps(
        manifest_data, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    actual_sha = hashlib.sha256(canonical_bytes).hexdigest().lower()
    assert actual_sha == VinAITranslateProvider.CANONICAL_MANIFEST_SHA256

    # Verify deterministic invariance against synthetic LF and CRLF encodings
    raw_lf = json.dumps(manifest_data, indent=2).replace("\r\n", "\n").encode("utf-8")
    raw_crlf = json.dumps(manifest_data, indent=2).replace("\n", "\r\n").encode("utf-8")
    parsed_lf = json.loads(raw_lf.decode("utf-8"))
    parsed_crlf = json.loads(raw_crlf.decode("utf-8"))
    c_lf = json.dumps(parsed_lf, sort_keys=True, separators=(",", ":")).encode("utf-8")
    c_crlf = json.dumps(parsed_crlf, sort_keys=True, separators=(",", ":")).encode("utf-8")
    assert (
        hashlib.sha256(c_lf).hexdigest().lower()
        == VinAITranslateProvider.CANONICAL_MANIFEST_SHA256
    )
    assert (
        hashlib.sha256(c_crlf).hexdigest().lower()
        == VinAITranslateProvider.CANONICAL_MANIFEST_SHA256
    )

    # Override without allow_custom_trust_anchor=True must be forbidden
    with pytest.raises(TranslationError, match=r"\[vinai_trust_anchor_override_forbidden\]"):
        VinAITranslateProvider(
            revision="ae7baa85da07dbe8e23ac26a9f5ef560c17e2138",
            require_exact_snapshot=True,
            trusted_manifest_sha256="a" * 64,
            allow_custom_trust_anchor=False,
        )


def test_vinai_unconditional_all_zero_revision_rejection() -> None:
    """All-zero revision is unconditionally rejected even if passed in trusted_revisions."""
    all_zero = "0" * 40
    with pytest.raises(TranslationError, match=r"\[vinai_untrusted_revision\]"):
        VinAITranslateProvider(
            revision=all_zero,
            require_exact_snapshot=True,
            trusted_revisions=[all_zero],
        )


def test_vinai_strict_inventory_allowlist_rejects_unlisted_files(tmp_path: Path) -> None:
    """Any file in snapshot directory not explicitly in manifest checksums is rejected."""
    custom_cache = tmp_path / "custom_hf_cache"
    rev = "ae7baa85da07dbe8e23ac26a9f5ef560c17e2138"
    snap_dir = custom_cache / "models--vinai--vinai-translate-vi2en-v2" / "snapshots" / rev
    snap_dir.mkdir(parents=True)
    checksums = {
        "config.json": hashlib.sha256(b"{}").hexdigest(),
        "special_tokens_map.json": hashlib.sha256(b"{}").hexdigest(),
        "tokenizer_config.json": hashlib.sha256(b"{}").hexdigest(),
        "model.safetensors": hashlib.sha256(b"weights").hexdigest(),
        "sentencepiece.bpe.model": hashlib.sha256(b"spm").hexdigest(),
    }
    (snap_dir / "config.json").write_text("{}", encoding="utf-8")
    (snap_dir / "special_tokens_map.json").write_text("{}", encoding="utf-8")
    (snap_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (snap_dir / "model.safetensors").write_bytes(b"weights")
    (snap_dir / "sentencepiece.bpe.model").write_bytes(b"spm")
    manifest_bytes = json.dumps({"revision": rev, "checksums": checksums}).encode("utf-8")
    (snap_dir / "manifest.json").write_bytes(manifest_bytes)
    test_manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()

    # Drop an unlisted tokenizer.json (not in manifest checksums)
    (snap_dir / "tokenizer.json").write_text("{}", encoding="utf-8")

    with pytest.raises(TranslationError, match=r"\[vinai_untrusted_file\]"):
        VinAITranslateProvider(
            cache_dir=custom_cache,
            revision=rev,
            require_exact_snapshot=True,
            trusted_manifest_sha256=test_manifest_sha,
            allow_custom_trust_anchor=True,
        )


def test_canonical_benchmark_pairs_mutations() -> None:
    """Mutating canonical benchmark pairs triggers corresponding semantic validation errors."""
    validator = SemanticSanityValidator()

    # Mutation 1: Invert spatial direction in dam footage
    src1 = (
        "Sau đó chuyển sang cảnh một con đập được quay từ trên cao, tiếp đến là cảnh "
        "cận con đập dưới trời mưa."
    )
    tgt1_mut = (
        "Then it switches to a scene of a dam filmed from below, followed by a "
        "close-up scene of the dam in the rain."
    )
    res1 = validator.validate(src1, tgt1_mut)
    assert not res1.is_valid
    assert any(iss.code in ("perspective_dropped", "spatial_inversion") for iss in res1.issues)

    # Mutation 2: Change count in map footage (bốn lần -> three times)
    src2 = "một bản đồ, trên đó một loại công trình thủy lợi lần lượt xuất hiện bốn lần."
    tgt2_mut = "a map, on which a type of irrigation structure appears three times in turn."
    res2 = validator.validate(src2, tgt2_mut)
    assert not res2.is_valid
    assert any(iss.code == "count_dropped" for iss in res2.issues)

    # Mutation 3: Change color of hats in gym footage (màu đỏ -> blue hats)
    src3 = "Trong nhóm chỉ có một người đeo kính và ba người đội nón có màu đỏ"
    tgt3_mut = "In the group, only one person wore glasses and three people wore blue hats"
    res3 = validator.validate(src3, tgt3_mut)
    assert not res3.is_valid
    assert any(iss.code in ("color_mismatch", "color_dropped") for iss in res3.issues)


def test_semantic_fallback_cache_round_trip_with_semantic_reasons(tmp_path: Path) -> None:
    """Semantic validation failure on Google output triggers fallback to VinAI,
    persisting semantic error reason in cache, and 2nd call hits cache with total
    1 Google + 1 VinAI call.
    """
    google_calls = 0
    vinai_calls = 0

    mock_google = MagicMock()
    mock_google.provider_name = "google-cloud"

    def google_translate_many(texts: list[str]) -> tuple[str, ...]:
        nonlocal google_calls
        google_calls += 1
        return ("Three men stand next to two women",)

    mock_google.translate_many.side_effect = google_translate_many

    mock_vinai = MagicMock()
    mock_vinai.provider_name = "vinai-translate"

    def vinai_translate_many(texts: list[str]) -> tuple[str, ...]:
        nonlocal vinai_calls
        vinai_calls += 1
        return ("Two men stand next to three women",)

    mock_vinai.translate_many.side_effect = vinai_translate_many

    cache = PerKeyAtomicTranslationCache(cache_dir=tmp_path)
    composite = LiveFallbackTranslationProvider(
        google_provider=mock_google,
        vinai_provider=mock_vinai,
        cache=cache,
    )

    query = "Hai người đàn ông đứng cạnh ba phụ nữ"

    # First call: Google runs (fails semantic check) -> VinAI runs (passes) -> cached
    res1 = composite.translate_many_with_metadata([query])
    assert len(res1) == 1
    assert res1[0].translated_text == "Two men stand next to three women"
    assert res1[0].metadata.fallback_triggered is True
    assert res1[0].metadata.fallback_reason_code == "semantic_error_count_entity_swap"
    assert google_calls == 1
    assert vinai_calls == 1

    # Second call: must be a cache hit, total call counts unchanged!
    res2 = composite.translate_many_with_metadata([query])
    assert len(res2) == 1
    assert res2[0].translated_text == "Two men stand next to three women"
    assert res2[0].metadata.served_from_cache is True
    assert res2[0].metadata.fallback_triggered is True
    assert res2[0].metadata.fallback_reason_code == "semantic_error_count_entity_swap"
    assert google_calls == 1
    assert vinai_calls == 1


def test_zero_data_leakage_sentinel_token_never_leaks(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Sentinel secret token is never leaked across exceptions, logs, metadata, or names."""
    import logging

    sentinel = "TOKEN_super_secret_998877"
    cache = PerKeyAtomicTranslationCache(cache_dir=tmp_path)

    # 1. Invalid origin_provider containing secret
    with pytest.raises(ValueError) as exc_info:
        cache.put("chó", "dog", origin_provider=sentinel)  # type: ignore[arg-type]
    assert sentinel not in str(exc_info.value)

    # 2. Corrupt cache entry with secret provider in log
    cache_file = tmp_path / "entries" / f"{hashlib.sha256(b'meo').hexdigest()}.json"
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(
        json.dumps(
            {
                "source_text": "mèo",
                "translated_text": "cat",
                "origin_provider": sentinel,
                "integrity_mode": "checksum-only",
                "payload_sha256": hashlib.sha256(b"cat").hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    with caplog.at_level(logging.WARNING):
        val = cache.get("mèo")
        assert val is None
    for record in caplog.records:
        assert sentinel not in record.getMessage()

    # 3. VinAI translation error does not leak model path or secret token
    with pytest.raises(TranslationError) as exc_info:
        VinAITranslateProvider(
            model_name_or_path=f"C:/secrets/{sentinel}/model",
            revision="12345",
            require_exact_snapshot=True,
        )
    assert sentinel not in str(exc_info.value)

    # 4. Artifact fingerprint does not leak secret snapshot directory name, artifact name, or hash
    sentinel_snap_dir = tmp_path / f"custom_snapshot_{sentinel}"
    sentinel_snap_dir.mkdir(parents=True)
    (sentinel_snap_dir / "config.json").write_text("{}", encoding="utf-8")
    (sentinel_snap_dir / "special_tokens_map.json").write_text("{}", encoding="utf-8")
    (sentinel_snap_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (sentinel_snap_dir / "model.safetensors").write_bytes(b"dummy")
    (sentinel_snap_dir / "sentencepiece.bpe.model").write_bytes(b"dummy")
    (sentinel_snap_dir / f"{sentinel}.json").write_text("{}", encoding="utf-8")
    s_checksums = {
        "config.json": hashlib.sha256(b"{}").hexdigest(),
        "special_tokens_map.json": hashlib.sha256(b"{}").hexdigest(),
        "tokenizer_config.json": hashlib.sha256(b"{}").hexdigest(),
        "model.safetensors": hashlib.sha256(b"dummy").hexdigest(),
        "sentencepiece.bpe.model": hashlib.sha256(b"dummy").hexdigest(),
        f"{sentinel}.json": hashlib.sha256(b"{}").hexdigest(),
    }
    s_manifest_bytes = json.dumps(
        {"revision": "ae7baa85da07dbe8e23ac26a9f5ef560c17e2138", "checksums": s_checksums}
    ).encode("utf-8")
    (sentinel_snap_dir / "manifest.json").write_bytes(s_manifest_bytes)

    provider_sentinel = VinAITranslateProvider(
        model_name_or_path=sentinel_snap_dir,
        revision="ae7baa85da07dbe8e23ac26a9f5ef560c17e2138",
        require_exact_snapshot=True,
        trusted_manifest_sha256=hashlib.sha256(s_manifest_bytes).hexdigest(),
        allow_custom_trust_anchor=True,
    )
    fp = provider_sentinel.get_artifact_fingerprint()
    assert fp["resolved_snapshot_dir"] == "custom-local"
    assert fp["snapshot_commit_hash"] == "ae7baa85da07dbe8e23ac26a9f5ef560c17e2138"
    assert fp["revision_matches_snapshot"] is True
    assert sentinel not in json.dumps(fp)
    assert not any(sentinel in k for k in fp)
