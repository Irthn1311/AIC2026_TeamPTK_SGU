from pathlib import Path
from unittest.mock import MagicMock

import pytest

from system_tai.kis.session_engine import OperationalKISRuntime
from system_tai.kis.session_schema import QueryRequest, SessionConfig
from system_tai.refinement.engine import QueryRefinementOutcome
from system_tai.retrieval.multi_query import KISResult


@pytest.fixture
def fake_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config = SessionConfig(
        input_root=tmp_path,
        output_root=tmp_path / "out",
        device="cpu",
        rrf_constant=60.0,
    )
    
    class FakeValidation:
        valid = True
        identifiers = {}
        missing_raw_video = []
        missing_npy = []
        malformed_metadata = []

    class FakeValidator:
        def validate(self, *args, **kwargs):
            return FakeValidation()

    registry = MagicMock()
    registry.embedding_dimension = 512
    registry.total_rows = 1000
    shared_encoder = MagicMock()
    shared_encoder.dimension = 512
    shared_encoder.identifiers = {"model": "ViT-B/32", "device": "cpu"}
    
    manifest = MagicMock()
    manifest.fingerprint = "fake_fingerprint"

    # Patch writers and payloads
    import system_tai.kis.session_engine as eng
    monkeypatch.setattr(eng, "_write_json", lambda *a, **k: tmp_path / "out" / "mock.json")
    monkeypatch.setattr(eng, "_write_internal_csv", lambda *a, **k: tmp_path / "out" / "mock.csv")
    monkeypatch.setattr(eng, "_write_refined_csv", lambda *a, **k: tmp_path / "out" / "mock.csv")
    monkeypatch.setattr(eng, "_validation_payload", lambda *a, **k: {})
    runtime = OperationalKISRuntime(
        config=config,
        manifest_path=tmp_path / "manifest.json",
        manifest=manifest,
        registry=registry,
        raw_video_registry=MagicMock(),
        shared_encoder=shared_encoder,
        decoder=MagicMock(),
        validator=FakeValidator(),
        clock=lambda: 0.0,
        bootstrap_timings={}
    )

    # Mock retrieval
    import numpy as np
    runtime.shared_encoder.encode_texts.side_effect = lambda texts: np.ones((len(texts), 512))
    runtime.weighted_rrf = MagicMock()
    from system_tai.common.schemas import CandidateFrame
    mock_candidate = CandidateFrame(
        video_id="v1",
        frame_id=1,
        clip_row=1,
        keyframe_order=1,
        score=1.0,
        rank=1,
        source="mock",
        diagnostic_metadata={}
    )
    runtime.weighted_rrf.fuse_rankings.return_value = KISResult("q1", [mock_candidate])

    # Mock refiner
    runtime.refiner = MagicMock()
    
    return runtime


def test_telemetry_propagation_retrieval_only(fake_runtime):
    req = QueryRequest(request_id="r1", query_id="q1", query_vi="vi", refine_top_n=0)
    resp = fake_runtime.handle_query(req)

    t = resp["timings"]
    assert t["video_probe_seconds"] == 0.0
    assert t["video_open_seconds"] == 0.0
    assert t["coarse_decode_seconds"] == 0.0
    assert t["coarse_encode_seconds"] == 0.0
    assert t["coarse_score_seconds"] == 0.0
    assert t["coarse_fusion_seconds"] == 0.0
    assert t["fine_decode_seconds"] == 0.0
    assert t["fine_encode_seconds"] == 0.0
    assert t["fine_score_seconds"] == 0.0
    assert t["fine_fusion_seconds"] == 0.0
    assert t["candidate_total_seconds"] == 0.0


def test_telemetry_propagation_refinement(fake_runtime):
    fake_outcome = QueryRefinementOutcome(
        query_id="q1",
        result=fake_runtime.weighted_rrf.fuse_rankings.return_value,
        candidates=(),
        timings={
            "refined_candidate_count": 3,
            "decoded_frame_count": 1200,
            "encoded_image_count": 100,
            "coarse_requested_frame_count": 60,
            "coarse_decoded_frame_count": 900,
            "fine_requested_frame_count": 180,
            "fine_decoded_frame_count": 300,
            "coarse_sparse_request_count": 0,
            "coarse_sparse_success_count": 0,
            "coarse_sparse_fallback_count": 0,
            "video_probe_seconds": 0.1,
            "video_open_seconds": 0.2,
            "coarse_decode_seconds": 1.0,
            "coarse_encode_seconds": 2.0,
            "coarse_score_seconds": 0.01,
            "coarse_fusion_seconds": 0.02,
            "fine_decode_seconds": 3.0,
            "fine_encode_seconds": 4.0,
            "fine_score_seconds": 0.03,
            "fine_fusion_seconds": 0.04,
            "candidate_total_seconds": 10.4,
        },
        warnings=(),
    )
    fake_runtime.refiner.refine_query.return_value = fake_outcome
    req = QueryRequest(request_id="r1", query_id="q1", query_vi="vi", refine_top_n=3)
    resp = fake_runtime.handle_query(req)

    t = resp["timings"]
    assert t["video_probe_seconds"] == 0.1
    assert t["video_open_seconds"] == 0.2
    assert t["coarse_decode_seconds"] == 1.0
    assert t["coarse_encode_seconds"] == 2.0
    assert t["coarse_score_seconds"] == 0.01
    assert t["coarse_fusion_seconds"] == 0.02
    assert t["fine_decode_seconds"] == 3.0
    assert t["fine_encode_seconds"] == 4.0
    assert t["fine_score_seconds"] == 0.03
    assert t["fine_fusion_seconds"] == 0.04
    assert t["candidate_total_seconds"] == 10.4
