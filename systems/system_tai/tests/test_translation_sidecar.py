"""Unit tests for Immutable Translation Sidecar and Zero Network Isolation."""

import socket
import pytest
from pathlib import Path

from system_tai.translation.sidecar_provider import (
    ImmutableSidecarTranslationProvider,
    canonical_sidecar_sha256,
)
from system_tai.retrieval.semantic_query import (
    compile_vietnamese_semantic_query,
    SemanticQueryConfig,
)
from system_tai.translation.provider import TokenBudgetGuard, TranslationError

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SIDECAR_DIR = REPO_ROOT / "scratch" / "benchmarks" / "translation_ablation"
PATH_NEW = SIDECAR_DIR / "translation_p1_focus_v2_new.json"
PATH_OLD = SIDECAR_DIR / "translation_p1_focus_v1_old_candidate.json"

EXPECTED_SHA_NEW = "545bd4a37c57af53713a1d9f382241ef729c287a1817a5671fdc923115b0be2a"
EXPECTED_SHA_OLD = "022a6c1db48d5fe00a223ec9f637aa1d64eea5d55c06e901caa42e04ff0e3367"


def test_canonical_content_hash_integrity():
    assert PATH_NEW.exists(), f"Missing sidecar {PATH_NEW}"
    assert PATH_OLD.exists(), f"Missing sidecar {PATH_OLD}"

    sha_new = canonical_sidecar_sha256(PATH_NEW)
    sha_old = canonical_sidecar_sha256(PATH_OLD)

    assert sha_new == EXPECTED_SHA_NEW
    assert sha_old == EXPECTED_SHA_OLD

    prov_new = ImmutableSidecarTranslationProvider(PATH_NEW, EXPECTED_SHA_NEW)
    prov_old = ImmutableSidecarTranslationProvider(PATH_OLD, EXPECTED_SHA_OLD)

    assert prov_new.sidecar_id == "translation-p1-focus-v2-new"
    assert prov_old.sidecar_id == "translation-p1-focus-v1-old-candidate"


def test_sidecar_fail_fast_on_tampered_hash_or_missing_file():
    with pytest.raises(ValueError, match="canonical content SHA256 mismatch"):
        ImmutableSidecarTranslationProvider(PATH_NEW, "0000000000000000000000000000000000000000000000000000000000000000")

    with pytest.raises(FileNotFoundError):
        ImmutableSidecarTranslationProvider("non_existent_sidecar.json")


def test_zero_network_isolation(monkeypatch):
    def blocked_socket(*args, **kwargs):
        raise RuntimeError("NETWORK CALL ATTEMPTED! Zero-network isolation violated!")

    monkeypatch.setattr(socket, "socket", blocked_socket)

    prov_new = ImmutableSidecarTranslationProvider(PATH_NEW, EXPECTED_SHA_NEW)
    sample_text = "Cảnh quay một nhóm hơn 5 người xếp thành hàng tập thể dục, cùng thực hiện động tác hai tay chạm mũi chân. Trong nhóm chỉ có một người đeo kính và ba người đội nón có màu đỏ."
    
    en = prov_new.translate(sample_text, query_id="query-p1-1-kis")
    assert "exercise" in en.lower()


def test_sidecars_differ_strictly_only_on_p1_5():
    prov_new = ImmutableSidecarTranslationProvider(PATH_NEW, EXPECTED_SHA_NEW)
    prov_old = ImmutableSidecarTranslationProvider(PATH_OLD, EXPECTED_SHA_OLD)

    # 4 Negative Controls: P1-1, P1-2, P1-4, P1-6 are 100% identical
    for qid in ("query-p1-1-kis", "query-p1-2-kis", "query-p1-4-kis", "query-p1-6-kis"):
        assert prov_new.expected_semantic_hash(qid) == prov_old.expected_semantic_hash(qid)
        assert prov_new.expected_variant_count(qid) == prov_old.expected_variant_count(qid)

    # Treatment Query P1-5 differs
    assert prov_new.expected_semantic_hash("query-p1-5-kis") == "243b0f915c63"
    assert prov_old.expected_semantic_hash("query-p1-5-kis") == "99f24deaaf56"


def test_sidecar_translation_miss_raises_fail_fast():
    prov = ImmutableSidecarTranslationProvider(PATH_NEW, EXPECTED_SHA_NEW)
    with pytest.raises(TranslationError, match="Sidecar translation miss"):
        prov.translate("Câu tiếng Việt chưa từng xuất hiện trong benchmark")
