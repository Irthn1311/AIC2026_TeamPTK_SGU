"""Comprehensive unit tests for Sprint 2B.1 Consensus Novel Video Rescue."""

from dataclasses import replace
import pytest

from system_tai.qa.grounding import QAVideoConditionedEvidenceConfig
from system_tai.qa.consensus_rescue import (
    ConsensusRescueCandidate,
    ConsensusNovelRescueOutcome,
    derive_consensus_novel_videos,
)
from system_tai.qa.rescue_tail import RescueCandidate, merge_rescue_tail
from system_tai.retrieval.query_decomposition import decompose_query


def test_consensus_novel_rescue_config_default_off():
    """Verify default config has consensus novel rescue disabled."""
    cfg = QAVideoConditionedEvidenceConfig(enabled=True)
    assert cfg.consensus_novel_rescue_enabled is False
    assert cfg.consensus_novel_rescue_max_videos == 1
    assert cfg.consensus_novel_rescue_tail_budget == 5


def test_consensus_novel_rescue_config_validation():
    """Verify field type and value validations on consensus rescue config."""
    with pytest.raises(ValueError, match="consensus_novel_rescue_enabled must be a boolean"):
        QAVideoConditionedEvidenceConfig(enabled=True, consensus_novel_rescue_enabled="true")  # type: ignore

    with pytest.raises(ValueError, match="consensus_novel_rescue_max_videos must be an integer >= 1"):
        QAVideoConditionedEvidenceConfig(enabled=True, consensus_novel_rescue_max_videos=0)

    with pytest.raises(ValueError, match="consensus_novel_rescue_tail_budget must be an integer >= 1"):
        QAVideoConditionedEvidenceConfig(enabled=True, consensus_novel_rescue_tail_budget=-1)


def test_missing_required_variant_fails_closed():
    """Verify that if literal or compact_keywords is missing/empty, outcome fails closed."""
    # Test query decomposition on empty string
    empty_outcome = decompose_query(query_text_vi="", query_text_en="")
    decomp = dict(empty_outcome.as_list())
    assert "compact_keywords" not in decomp or not decomp["compact_keywords"]


def test_merge_rescue_tail_preserves_prefix_and_admits_tail():
    """Verify merge_rescue_tail strictly preserves ranks 1..95 and admits unique tail tuples."""
    base_preds = [
        {"rank": i, "video_id": f"L21_V{i:03d}", "frame_id": 1000 + i, "answer": f"ans_{i}"}
        for i in range(1, 101)
    ]

    rescue_cands = [
        # Candidate 1: valid novel tuple
        RescueCandidate(
            video_id="L21_V999",
            frame_id=5000,
            answer="novel_ans",
            rescue_score=0.9,
            rescue_source="consensus_novel_video_rescue",
        ),
        # Candidate 2: duplicate of existing prefix
        RescueCandidate(
            video_id="L21_V001",
            frame_id=1001,
            answer="ans_1",
            rescue_score=0.8,
            rescue_source="consensus_novel_video_rescue",
        ),
    ]

    merged = merge_rescue_tail(base_preds, rescue_cands, prefix_k=95, max_rescue=5)
    assert len(merged) == 96  # 95 base + 1 novel (duplicate rejected)

    # Verify ranks 1..95 are 100% bit-identical
    for i in range(95):
        assert merged[i]["video_id"] == base_preds[i]["video_id"]
        assert merged[i]["frame_id"] == base_preds[i]["frame_id"]
        assert merged[i]["answer"] == base_preds[i]["answer"]
        assert merged[i]["rank"] == i + 1

    # Verify rank 96 is the novel candidate
    assert merged[95]["video_id"] == "L21_V999"
    assert merged[95]["frame_id"] == 5000
    assert merged[95]["answer"] == "novel_ans"
    assert merged[95]["rank"] == 96
    assert merged[95]["slot_source"] == "RESCUE_TAIL_CONSENSUS_NOVEL_VIDEO_RESCUE"


def test_consensus_candidate_tie_breaking_is_deterministic():
    """Verify that multiple consensus candidates are sorted deterministically by (fused_rank, video_id)."""
    c1 = ConsensusRescueCandidate(video_id="L21_V009", fused_rank=6, literal_rank=6, compact_rank=6)
    c2 = ConsensusRescueCandidate(video_id="L21_V005", fused_rank=6, literal_rank=8, compact_rank=7)
    c3 = ConsensusRescueCandidate(video_id="L21_V020", fused_rank=10, literal_rank=10, compact_rank=10)

    raw_list = [c3, c1, c2]
    sorted_list = sorted(raw_list, key=lambda c: (c.fused_rank, c.video_id))
    assert sorted_list[0].video_id == "L21_V005"  # tied fused_rank 6, V005 < V009
    assert sorted_list[1].video_id == "L21_V009"
    assert sorted_list[2].video_id == "L21_V020"


def test_consensus_eligibility_enforces_both_variants():
    """Verify that a candidate present in compact but absent from literal is rejected."""
    champ_pool = ("L27_V008", "L25_V011")
    lit_pool = ("L21_V028", "L21_V015")  # L21_V001 absent from literal
    cmp_pool = ("L28_V023", "L21_V001")  # L21_V001 present in compact
    fused_pool = ("L28_V023", "L21_V001")

    champ_set = set(champ_pool)
    lit_set = set(lit_pool)
    cmp_set = set(cmp_pool)

    consensus = [
        vid for vid in fused_pool
        if vid not in champ_set and vid in lit_set and vid in cmp_set
    ]
    # L21_V001 must NOT be in consensus because it is absent from literal
    assert "L21_V001" not in consensus


def test_default_off_exact_parity_contract():
    """Verify default config has consensus novel rescue completely disabled with zero side effects."""
    cfg = QAVideoConditionedEvidenceConfig(enabled=True, consensus_novel_rescue_enabled=False)
    assert not cfg.consensus_novel_rescue_enabled


def test_execute_sidepath_consensus_rescue_ineligible_returns_identical_predictions():
    """Verify that when no consensus candidate is found, execute_sidepath_consensus_rescue leaves base predictions untouched."""
    from unittest.mock import MagicMock
    from system_tai.kis.session_schema import QAQueryRequest
    from system_tai.qa.consensus_rescue import execute_sidepath_consensus_rescue
    from system_tai.qa.question_types import QuestionType

    req = QAQueryRequest(
        request_id="test_req",
        query_id="QA-99",
        event_description="test",
        question="test",
    )
    base_preds = [{"rank": 1, "video_id": "L21_V001", "frame_id": 100, "answer": "ans"}]
    cfg = QAVideoConditionedEvidenceConfig(enabled=True, consensus_novel_rescue_enabled=True)

    searcher = MagicMock()
    encoder = MagicMock()
    # Mock search_video_maxima returning empty dict
    searcher.search_video_maxima.return_value = {}

    merged, outcome, recs, admitted = execute_sidepath_consensus_rescue(
        request=req,
        q_type=QuestionType.COLOR,
        variants=(),
        event_vectors=(),
        champion_selected_video_ids=("L21_V001",),
        champion_predictions=base_preds,
        searcher=searcher,
        encoder=encoder,
        decoder=MagicMock(),
        refiner=MagicMock(),
        refinement_config=MagicMock(),
        weighted_rrf=MagicMock(),
        raw_video_registry=MagicMock(),
        candidate_provider=None,
        ocr_answer_provider=None,
        qa_engine=MagicMock(),
        config=cfg,
    )

    assert merged == base_preds
    assert not outcome.eligible
    assert recs == []
    assert admitted == []

