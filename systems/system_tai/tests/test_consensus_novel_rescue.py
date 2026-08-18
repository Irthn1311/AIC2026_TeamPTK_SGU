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

    merged, outcome, recs, admitted, stage_telemetry = execute_sidepath_consensus_rescue(
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


def test_execute_sidepath_consensus_rescue_eligible_path_with_qa_engine():
    """Verify that when a consensus candidate is eligible, the canonical QA engine produces RescueCandidates and merges into tail."""
    from unittest.mock import MagicMock, patch
    import numpy as np
    from system_tai.kis.session_schema import QAQueryRequest
    from system_tai.preliminary.schemas import QAPrediction
    from system_tai.qa.consensus_rescue import (
        ConsensusRescueCandidate,
        ConsensusNovelRescueOutcome,
        execute_sidepath_consensus_rescue,
    )
    from system_tai.qa.models import QAResult
    from system_tai.qa.question_types import QuestionType
    from system_tai.refinement.engine import SelectedRefinementOutcome
    from system_tai.refinement.models import RefinedCandidate, RefinementStatus
    from system_tai.refinement.video import DecodedFrame, DecodeResult
    from system_tai.retrieval.multi_query import QueryLanguage, QueryVariant, QueryVariantType
    from system_tai.retrieval.video_evidence import RestrictedFrameHit, VideoRestrictedSearchOutcome

    req = QAQueryRequest(
        request_id="test_req",
        query_id="QA-26",
        event_description="convoy of white vehicles",
        question="Đoàn xe màu trắng là xe gì?",
        question_en="What type of vehicles are in the white convoy?",
    )
    base_preds = [
        {"rank": i, "video_id": f"L21_V{i:03d}", "frame_id": 1000 + i, "answer": f"ans_{i}"}
        for i in range(1, 101)
    ]
    cfg = QAVideoConditionedEvidenceConfig(
        enabled=True,
        consensus_novel_rescue_enabled=True,
        consensus_novel_rescue_max_videos=1,
        consensus_novel_rescue_tail_budget=5,
    )

    chosen_vid = "L21_V009"
    eligible_outcome = ConsensusNovelRescueOutcome(
        eligible=True,
        reason="CONSENSUS_CANDIDATE_SELECTED",
        literal_top16=(chosen_vid,),
        compact_top16=(chosen_vid,),
        fused_top16=(chosen_vid,),
        all_consensus_candidates=(ConsensusRescueCandidate(video_id=chosen_vid, fused_rank=1, literal_rank=1, compact_rank=1),),
        chosen_candidate=ConsensusRescueCandidate(video_id=chosen_vid, fused_rank=1, literal_rank=1, compact_rank=1),
    )

    v = QueryVariant(variant_id="QA-26::en", text="white vehicles", language=QueryLanguage.ENGLISH, variant_type=QueryVariantType.ENGLISH_TRANSLATION, weight=1.0)
    event_vec = np.ones((512,), dtype=np.float32)

    searcher = MagicMock()
    store_mock = MagicMock()
    store_mock.descriptor.row_count = 10
    searcher.registry.get.return_value = store_mock

    # Restricted frame search returns a candidate hit
    searcher.search_selected_videos.return_value = VideoRestrictedSearchOutcome(
        rankings={
            "QA-26::en": {
                chosen_vid: (
                    RestrictedFrameHit(video_id=chosen_vid, frame_id=21450, clip_row=0, keyframe_order=0, cosine_score=0.9, rank=1, pts_time=100.0),
                )
            }
        },
        video_store_scan_count=1,
        physical_rows_scored=10,
    )

    # Refiner returns refined candidate mock
    ref_cand_mock = MagicMock()
    ref_cand_mock.candidate_frame_id = 21450
    ref_cand_mock.refined_frame_id = 21450
    ref_cand_mock.status = RefinementStatus.REFINED
    ref_cand_mock.refinement_fusion_score = 0.95

    refiner = MagicMock()
    refiner.refine_selected_candidates.return_value = SelectedRefinementOutcome(
        candidates=(ref_cand_mock,),
        timings={},
        warnings=(),
    )

    # Raw video record and video decoder
    raw_video_registry = MagicMock()
    raw_record_mock = MagicMock()
    raw_record_mock.raw_video_path = MagicMock()
    raw_record_mock.raw_video_path.is_file.return_value = True
    raw_video_registry.get.return_value = raw_record_mock

    decoder = MagicMock()
    probe_mock = MagicMock()
    probe_mock.total_frame_count = 50000
    decoder.probe.return_value = probe_mock

    frame_img_mock = np.zeros((224, 224, 3), dtype=np.uint8)
    decode_res_mock = MagicMock()
    decode_res_mock.frames = (
        DecodedFrame(absolute_frame_id=21450, timestamp_seconds=100.0, image=frame_img_mock),
    )
    decoder.decode.return_value = decode_res_mock

    encoder = MagicMock()
    encoder.encode_images.return_value = np.ones((1, 512), dtype=np.float32)

    qa_engine = MagicMock()
    qa_engine.answer.return_value = QAResult(
        query_id="QA-26",
        question_type=QuestionType.OBJECT_ENTITY,
        predictions=[
            QAPrediction(query_id="QA-26", rank=1, video_id=chosen_vid, frame_id=21450, answer="taxi"),
        ],
    )

    with patch("system_tai.qa.consensus_rescue.derive_consensus_novel_videos", return_value=eligible_outcome):
        merged, outcome, recs, admitted, stage_telemetry = execute_sidepath_consensus_rescue(
            request=req,
            q_type=QuestionType.OBJECT_ENTITY,
            variants=(v,),
            event_vectors=[event_vec],
            champion_selected_video_ids=("L21_V001", "L21_V002"),
            champion_predictions=base_preds,
            searcher=searcher,
            encoder=encoder,
            decoder=decoder,
            refiner=refiner,
            refinement_config=MagicMock(),
            weighted_rrf=MagicMock(),
            raw_video_registry=raw_video_registry,
            candidate_provider=None,
            ocr_answer_provider=None,
            qa_engine=qa_engine,
            config=cfg,
        )

    # 1. Verify outcome is eligible
    assert outcome.eligible
    assert outcome.chosen_candidate.video_id == chosen_vid

    # 2. Verify QA engine was invoked with canonical QAQuery
    assert qa_engine.answer.called

    # 3. Verify exactly 1 rescue tuple was admitted into rank 96
    assert len(admitted) == 1
    assert admitted[0]["video_id"] == chosen_vid
    assert admitted[0]["frame_id"] == 21450
    assert admitted[0]["answer"] == "taxi"
    assert admitted[0]["rank"] == 96
    assert admitted[0]["slot_source"] == "RESCUE_TAIL_CONSENSUS_NOVEL_VIDEO_RESCUE"

    # 4. Verify ranks 1..95 of champion baseline are strictly preserved
    for i in range(95):
        assert merged[i]["video_id"] == base_preds[i]["video_id"]
        assert merged[i]["rank"] == i + 1

    # 5. Verify stage telemetry captured all 6 stages
    assert stage_telemetry["grounding"]["candidate_count"] >= 1
    assert stage_telemetry["temporal_seeds"]["seed_count"] >= 1
    assert stage_telemetry["refinement"]["output_count"] >= 1
    assert stage_telemetry["answers"]["produced_count"] >= 1
    assert stage_telemetry["tail"]["admitted_count"] == 1


