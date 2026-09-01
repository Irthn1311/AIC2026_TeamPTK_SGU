"""Unit tests for Immutable Translation Sidecar and Zero Network Isolation."""

import json
import hashlib
import re
import socket
import tempfile
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
from system_tai.translation.provider import TranslationError

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SIDECAR_DIR = REPO_ROOT / "scratch" / "benchmarks" / "translation_ablation"
PATH_NEW = SIDECAR_DIR / "translation_p1_focus_v2_new.json"
PATH_OLD = SIDECAR_DIR / "translation_p1_focus_v1_old_candidate.json"

EXPECTED_SHA_NEW = "545bd4a37c57af53713a1d9f382241ef729c287a1817a5671fdc923115b0be2a"
EXPECTED_SHA_OLD = "022a6c1db48d5fe00a223ec9f637aa1d64eea5d55c06e901caa42e04ff0e3367"

FOCUS_QUERIES = {
    "query-p1-1-kis": "Cảnh quay một nhóm hơn 5 người xếp thành hàng tập thể dục, cùng thực hiện động tác hai tay chạm mũi chân. Trong nhóm chỉ có một người đeo kính và ba người đội nón có màu đỏ.",
    "query-p1-2-kis": "Đoạn phim bắt đầu bằng một bản đồ, trên đó một loại công trình thủy lợi lần lượt xuất hiện bốn lần. Sau đó chuyển sang cảnh một con đập được quay từ trên cao, tiếp đến là cảnh cận con đập dưới trời mưa.",
    "query-p1-4-kis": "Một đàn sư tử đang nghỉ ngơi và leo trèo trên các bục gỗ trong khu nuôi dưỡng, phía trước có bảng thông tin của London Zoo phục vụ công tác theo dõi và bảo tồn động vật.. Sau đó có cảnh hai nhân viên mặc áo xanh lá đang cân và ghi nhận số liệu của một con vật trong khuôn viên sở thú.",
    "query-p1-5-kis": "Đoạn clip bắt đầu bằng việc đậu hà lan được bỏ vào với mực đang được xào trên chảo, bên cạnh là đĩa hành tây và ớt đỏ thái lát chuẩn bị cho vào món ăn. Đoạn clip kết thúc với khung quay chậm (slow motion) cảnh lắc chảo trên bếp lửa.",
    "query-p1-6-kis": "Mẩu tin bắt đầu với hình ảnh một người đàn ông mặc vest xanh đậm, sơ mi trắng và cà vạt, đang ngồi trên một chiếc ghế lớn. Ông cầm bằng hai tay một khối đá quý thô khá lớn, đưa lên gần mặt để quan sát. Bên phải là một phụ nữ mặc trang phục công sở màu đen và khăn trùm đầu màu hồng tím, đang đứng cạnh và mỉm cười. Tiếp theo có hình ảnh toàn cảnh từ trên cao của một mỏ đá quý lộ thiên quy mô lớn với hố khai thác sâu nhiều tầng và hệ thống đường vận chuyển bao quanh.",
}


class MockBpeTokenBudgetGuard:
    """Mock guard providing exact sentence splitting behavior when CLIP package is mocked or offline."""
    _BOUNDARY_RE = re.compile(r"(?<=[.!?;:,])\s+")

    def count_tokens(self, text: str) -> int:
        return len(text.split()) + 2

    def split_for_clip(self, text: str) -> tuple[str, ...]:
        cleaned = " ".join(text.split())
        if len(cleaned.split()) <= 60:
            return (cleaned,)
        clauses = tuple(c.strip() for c in self._BOUNDARY_RE.split(cleaned) if c.strip())
        if len(clauses) > 1 and len(cleaned.split()) > 60:
            return clauses
        return (cleaned,)


def compute_semantic_hash_from_compiled(compiled) -> tuple[str, int]:
    meta = compiled.to_metadata()
    units = meta.get("units", [])
    semantic_payload = [
        {
            "variant_id": seg.get("variant_id"),
            "text": seg.get("text"),
            "weight": seg.get("weight"),
        }
        for unit in units
        for seg in unit.get("segments", [])
    ]
    raw_json = json.dumps(
        semantic_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(raw_json.encode("utf-8")).hexdigest()[:12], len(semantic_payload)


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


def test_mock_segmentation_compile_all_5_queries_against_golden_hashes_both_sidecars():
    prov_new = ImmutableSidecarTranslationProvider(PATH_NEW, EXPECTED_SHA_NEW)
    prov_old = ImmutableSidecarTranslationProvider(PATH_OLD, EXPECTED_SHA_OLD)
    guard = MockBpeTokenBudgetGuard()
    cfg = SemanticQueryConfig()

    for qid, qvi in FOCUS_QUERIES.items():
        comp_new = compile_vietnamese_semantic_query(
            query_id=qid,
            query_vi=qvi,
            provider=prov_new,
            token_budget_guard=guard,
            config=cfg,
        )
        h_new, cnt_new = compute_semantic_hash_from_compiled(comp_new)
        assert h_new == prov_new.expected_semantic_hash(qid)
        assert cnt_new == prov_new.expected_variant_count(qid)

        comp_old = compile_vietnamese_semantic_query(
            query_id=qid,
            query_vi=qvi,
            provider=prov_old,
            token_budget_guard=guard,
            config=cfg,
        )
        h_old, cnt_old = compute_semantic_hash_from_compiled(comp_old)
        assert h_old == prov_old.expected_semantic_hash(qid)
        assert cnt_old == prov_old.expected_variant_count(qid)

        if qid != "query-p1-5-kis":
            assert h_new == h_old, f"Negative control {qid} drifted: {h_new} != {h_old}"
            assert cnt_new == cnt_old
        else:
            assert h_new == "243b0f915c63"
            assert h_old == "99f24deaaf56"


def test_real_clip_tokenizer_golden_compilation_if_available():
    pytest.importorskip("clip")
    from system_tai.translation.provider import TokenBudgetGuard

    guard = TokenBudgetGuard(max_tokens=75)
    prov_new = ImmutableSidecarTranslationProvider(PATH_NEW, EXPECTED_SHA_NEW)
    prov_old = ImmutableSidecarTranslationProvider(PATH_OLD, EXPECTED_SHA_OLD)
    cfg = SemanticQueryConfig()

    for qid, qvi in FOCUS_QUERIES.items():
        comp_new = compile_vietnamese_semantic_query(
            query_id=qid,
            query_vi=qvi,
            provider=prov_new,
            token_budget_guard=guard,
            config=cfg,
        )
        h_new, cnt_new = compute_semantic_hash_from_compiled(comp_new)
        assert h_new == prov_new.expected_semantic_hash(qid)
        assert cnt_new == prov_new.expected_variant_count(qid)

        comp_old = compile_vietnamese_semantic_query(
            query_id=qid,
            query_vi=qvi,
            provider=prov_old,
            token_budget_guard=guard,
            config=cfg,
        )
        h_old, cnt_old = compute_semantic_hash_from_compiled(comp_old)
        assert h_old == prov_old.expected_semantic_hash(qid)
        assert cnt_old == prov_old.expected_variant_count(qid)

        if qid != "query-p1-5-kis":
            assert h_new == h_old, f"Negative control {qid} drifted: {h_new} != {h_old}"
            assert cnt_new == cnt_old
        else:
            assert h_new == "243b0f915c63"
            assert h_old == "99f24deaaf56"


def test_global_translation_collision_fail_fast():
    with tempfile.TemporaryDirectory() as td:
        bad_sidecar = {
            "$schema_version": "1.0.0",
            "sidecar_id": "test-collision",
            "target_queries_count": 2,
            "queries": {
                "q1": {
                    "query_vi": "con mèo",
                    "units": [
                        {"vi_text": "con mèo", "en_text": "a cat"}
                    ]
                },
                "q2": {
                    "query_vi": "con mèo",
                    "units": [
                        {"vi_text": "con mèo", "en_text": "a feline"}
                    ]
                }
            }
        }
        bad_file = Path(td) / "bad_sidecar.json"
        bad_file.write_text(json.dumps(bad_sidecar), encoding="utf-8")

        with pytest.raises(ValueError, match="Translation collision detected"):
            ImmutableSidecarTranslationProvider(bad_file)
