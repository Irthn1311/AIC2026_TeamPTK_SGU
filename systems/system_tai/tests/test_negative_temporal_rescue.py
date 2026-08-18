# ==============================================================================================================
# Unit & Contract Tests for Sprint 2C.1 Bounded Negative Temporal Tail Rescue
# ==============================================================================================================

import pytest
from unittest.mock import MagicMock
import numpy as np

from system_tai.kis.session_schema import QAQueryRequest
from system_tai.preliminary.schemas import QAPrediction
from system_tai.qa.grounding import QAVideoConditionedEvidenceConfig
from system_tai.qa.models import QAResult
from system_tai.qa.negative_temporal_rescue import execute_sidepath_negative_temporal_rescue
from system_tai.qa.question_types import QuestionType
from system_tai.refinement.models import RefinedCandidate, RefinementStatus
from system_tai.refinement.video import DecodedFrame, DecodeResult


def test_negative_temporal_rescue_config_default_off():
    cfg = QAVideoConditionedEvidenceConfig(enabled=True)
    assert not cfg.bounded_negative_temporal_rescue_enabled
    assert cfg.bounded_negative_temporal_rescue_offsets == (-100, -200, -300)
    assert cfg.bounded_negative_temporal_rescue_tail_budget == 5


def test_negative_temporal_rescue_config_validation():
    with pytest.raises(ValueError, match="bounded_negative_temporal_rescue_enabled must be a boolean"):
        QAVideoConditionedEvidenceConfig(enabled=True, bounded_negative_temporal_rescue_enabled="true")  # type: ignore

    with pytest.raises(ValueError, match="bounded_negative_temporal_rescue_offsets must be a non-empty tuple of integers"):
        QAVideoConditionedEvidenceConfig(enabled=True, bounded_negative_temporal_rescue_offsets=())  # type: ignore

    with pytest.raises(ValueError, match="bounded_negative_temporal_rescue_offsets must be a non-empty tuple of integers"):
        QAVideoConditionedEvidenceConfig(enabled=True, bounded_negative_temporal_rescue_offsets=(100, "200"))  # type: ignore

    with pytest.raises(ValueError, match="bounded_negative_temporal_rescue_tail_budget must be an integer >= 1"):
        QAVideoConditionedEvidenceConfig(enabled=True, bounded_negative_temporal_rescue_tail_budget=0)


def test_negative_temporal_rescue_drop_out_of_bounds_no_clamp():
    """Verify that offset calculations drop out-of-bounds frame IDs and NEVER clamp to 0."""
    req = QAQueryRequest(
        request_id="test_req",
        query_id="QA-23",
        event_description="test",
        question="test",
    )
    base_preds = [{"rank": i, "video_id": "L21_V008", "frame_id": 1000 + i, "answer": f"ans_{i}"} for i in range(1, 101)]
    cfg = QAVideoConditionedEvidenceConfig(
        enabled=True,
        bounded_negative_temporal_rescue_enabled=True,
        bounded_negative_temporal_rescue_offsets=(-100, -200, -300),
    )

    # Source anchor at frame 150. Offsets: 150-100=50 (valid), 150-200=-50 (DROP, not clamp to 0), 150-300=-150 (DROP)
    ref_cand_mock = MagicMock()
    ref_cand_mock.video_id = "L21_V008"
    ref_cand_mock.candidate_frame_id = 150
    ref_cand_mock.refined_frame_id = 150
    ref_cand_mock.status = RefinementStatus.REFINED
    ref_cand_mock.original_candidate_rank = 1

    raw_video_reg = MagicMock()
    raw_rec = MagicMock()
    raw_rec.raw_video_path.is_file.return_value = True
    raw_video_reg.get.return_value = raw_rec

    decoder = MagicMock()
    probe_mock = MagicMock()
    probe_mock.total_frame_count = 50000
    decoder.probe.return_value = probe_mock

    # Decodes only the single valid frame 50
    decode_mock = MagicMock()
    decode_mock.frames = (
        DecodedFrame(absolute_frame_id=50, timestamp_seconds=2.0, image=np.zeros((224, 224, 3), dtype=np.uint8)),
    )
    decoder.decode.return_value = decode_mock

    ocr_provider = MagicMock()
    ocr_provider.answer.return_value = (
        QAResult(
            query_id="QA-23",
            question_type=QuestionType.OCR,
            predictions=[QAPrediction(query_id="QA-23", rank=1, video_id="L21_V008", frame_id=50, answer="dior")],
        ),
        {},
    )

    merged, recs, admitted, stage_telemetry = execute_sidepath_negative_temporal_rescue(
        request=req,
        q_type=QuestionType.OCR,
        champion_selected_video_ids=("L21_V008",),
        champion_refined_candidates=(ref_cand_mock,),
        champion_predictions=base_preds,
        raw_video_registry=raw_video_reg,
        decoder=decoder,
        encoder=MagicMock(),
        qa_engine=MagicMock(),
        ocr_answer_provider=ocr_provider,
        ocr_provider_supported=True,
        config=cfg,
    )

    # Assert only frame 50 was kept, and no 0 frame was artificially created
    assert stage_telemetry["valid_offset_frames"] == [50]
    assert len(admitted) == 1
    assert admitted[0]["frame_id"] == 50
    assert admitted[0]["answer"] == "dior"
    assert admitted[0]["rank"] == 96


def test_negative_temporal_rescue_ocr_eligible_path():
    """Verify that for an OCR-supported query (like QA-23), OCR answer provider is invoked."""
    req = QAQueryRequest(
        request_id="test_req",
        query_id="QA-23",
        event_description="woman pushing bike with bag",
        question="Trên chiếc túi xách có chữ gì?",
        question_en="What letters are on the handbag?",
    )
    base_preds = [{"rank": i, "video_id": f"L21_V{i:03d}", "frame_id": 1000 + i, "answer": f"ans_{i}"} for i in range(1, 101)]
    cfg = QAVideoConditionedEvidenceConfig(
        enabled=True,
        bounded_negative_temporal_rescue_enabled=True,
        bounded_negative_temporal_rescue_offsets=(-100, -200, -300),
    )

    # QA-23 anchor at 29213 -> offsets: 29113, 29013, 28913
    ref_cand_mock = MagicMock()
    ref_cand_mock.video_id = "L21_V008"
    ref_cand_mock.candidate_frame_id = 29213
    ref_cand_mock.refined_frame_id = 29213
    ref_cand_mock.status = RefinementStatus.REFINED
    ref_cand_mock.original_candidate_rank = 1

    raw_video_reg = MagicMock()
    raw_rec = MagicMock()
    raw_rec.raw_video_path.is_file.return_value = True
    raw_video_reg.get.return_value = raw_rec

    decoder = MagicMock()
    probe_mock = MagicMock()
    probe_mock.total_frame_count = 50000
    decoder.probe.return_value = probe_mock

    decode_mock = MagicMock()
    decode_mock.frames = (
        DecodedFrame(absolute_frame_id=29113, timestamp_seconds=100.0, image=np.zeros((224, 224, 3), dtype=np.uint8)),
        DecodedFrame(absolute_frame_id=29013, timestamp_seconds=100.0, image=np.zeros((224, 224, 3), dtype=np.uint8)),
        DecodedFrame(absolute_frame_id=28913, timestamp_seconds=100.0, image=np.zeros((224, 224, 3), dtype=np.uint8)),
    )
    decoder.decode.return_value = decode_mock

    ocr_provider = MagicMock()
    ocr_provider.answer.return_value = (
        QAResult(
            query_id="QA-23",
            question_type=QuestionType.OCR,
            predictions=[
                QAPrediction(query_id="QA-23", rank=1, video_id="L21_V008", frame_id=29013, answer="dior"),
            ],
        ),
        {},
    )

    merged, recs, admitted, stage_telemetry = execute_sidepath_negative_temporal_rescue(
        request=req,
        q_type=QuestionType.OCR,
        champion_selected_video_ids=("L21_V008",),
        champion_refined_candidates=(ref_cand_mock,),
        champion_predictions=base_preds,
        raw_video_registry=raw_video_reg,
        decoder=decoder,
        encoder=MagicMock(),
        qa_engine=MagicMock(),
        ocr_answer_provider=ocr_provider,
        ocr_provider_supported=True,
        config=cfg,
    )

    # Verify OCR route was used
    assert stage_telemetry["answer_route_used"] == "OCR"
    assert ocr_provider.answer.called

    # Verify exactly 1 rescue candidate admitted at rank 96
    assert len(admitted) == 1
    assert admitted[0]["video_id"] == "L21_V008"
    assert admitted[0]["frame_id"] == 29013
    assert admitted[0]["answer"] == "dior"
    assert admitted[0]["rank"] == 96
    assert admitted[0]["slot_source"] == "RESCUE_TAIL_BOUNDED_NEGATIVE_TEMPORAL_RESCUE"

    # Verify ranks 1..95 are 100% untouched
    for i in range(95):
        assert merged[i]["video_id"] == base_preds[i]["video_id"]
        assert merged[i]["rank"] == i + 1


def test_negative_temporal_rescue_default_off_parity():
    """Verify that when disabled, execute_sidepath_negative_temporal_rescue returns predictions untouched."""
    req = QAQueryRequest(
        request_id="test_req",
        query_id="QA-23",
        event_description="test",
        question="test",
    )
    base_preds = [{"rank": i, "video_id": f"L21_V{i:03d}", "frame_id": 1000 + i, "answer": f"ans_{i}"} for i in range(1, 101)]
    cfg = QAVideoConditionedEvidenceConfig(
        enabled=True,
        bounded_negative_temporal_rescue_enabled=False,
    )

    merged, recs, admitted, stage_telemetry = execute_sidepath_negative_temporal_rescue(
        request=req,
        q_type=QuestionType.OCR,
        champion_selected_video_ids=("L21_V008",),
        champion_refined_candidates=(),
        champion_predictions=base_preds,
        raw_video_registry=MagicMock(),
        decoder=MagicMock(),
        encoder=MagicMock(),
        qa_engine=MagicMock(),
        ocr_answer_provider=None,
        ocr_provider_supported=False,
        config=cfg,
    )

    assert merged == base_preds
    assert recs == []
    assert admitted == []
    assert not stage_telemetry["enabled"]
