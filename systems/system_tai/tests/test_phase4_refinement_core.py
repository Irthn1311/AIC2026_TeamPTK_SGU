from __future__ import annotations

from pathlib import Path, PurePosixPath, PureWindowsPath
from types import SimpleNamespace

import numpy as np
import pytest

from system_tai.kis.benchmark import resolve_device
from system_tai.refinement.engine import (
    ExactFrameRefiner,
    build_frame_window,
    coarse_frame_ids,
    fine_frame_ids,
    fuse_local_frame_rankings,
)
from system_tai.refinement.models import (
    CandidateFailurePolicy,
    MissingRawVideoPolicy,
    Phase3Candidate,
    QueryRefinementError,
    RefinementConfig,
    RefinementQuery,
    RefinementStatus,
)
from system_tai.refinement.video import (
    DecodedFrame,
    DecodeRequest,
    DecodeResult,
    RawVideoRecord,
    RawVideoRegistry,
    VideoProbe,
)
from system_tai.retrieval.multi_query import (
    QueryLanguage,
    QueryVariant,
    QueryVariantType,
)


def variant(variant_id: str = "vi", text: str = "target", weight: float = 1.0):
    return QueryVariant(
        variant_id,
        text,
        QueryLanguage.VIETNAMESE if variant_id == "vi" else QueryLanguage.ENGLISH,
        QueryVariantType.VIETNAMESE_DIRECT
        if variant_id == "vi"
        else QueryVariantType.ENGLISH_TRANSLATION,
        weight,
    )


def candidate(rank: int = 1, frame_id: int = 50, video_id: str = "L21_V001"):
    return Phase3Candidate(
        "Q1",
        rank,
        video_id,
        frame_id,
        0.5,
        {"fusion_score": 0.5, "clip_row_diagnostic": 99, "keyframe_order_diagnostic": 7},
    )


class FakeEncoder:
    dimension = 2
    identifiers = {"model": "fake", "device": "cpu"}

    def __init__(self, target: int = 55):
        self.target = target
        self.text_calls = 0
        self.image_batches: list[int] = []

    def encode_texts(self, texts):
        self.text_calls += 1
        rows = [[1.0, 0.0] if "target" in text else [0.0, 1.0] for text in texts]
        return np.asarray(rows, dtype=np.float32)

    def encode_images(self, images, *, batch_size):
        self.image_batches.extend(
            min(batch_size, len(images) - start) for start in range(0, len(images), batch_size)
        )
        rows = []
        for image in images:
            frame_id = int(image)
            rows.append([1.0 / (1.0 + abs(frame_id - self.target)), frame_id / 100.0])
        matrix = np.asarray(rows, dtype=np.float32)
        return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


class FakeDecoder:
    backend_identifier = "fake-absolute"

    def __init__(self, *, fail: bool = False):
        self.fail = fail
        self.probe_calls = 0
        self.decode_requests: list[DecodeRequest] = []

    def probe(self, record):
        self.probe_calls += 1
        return VideoProbe(
            record.video_id, record.raw_video_path, self.backend_identifier, 10, 100, 8, 8, 10
        )

    def decode(self, request):
        if self.fail:
            raise RuntimeError("controlled decode failure")
        self.decode_requests.append(request)
        frames = tuple(
            DecodedFrame(frame_id, frame_id / 10, frame_id) for frame_id in request.frame_ids
        )
        return DecodeResult(
            frames,
            request.frame_ids[-1] - request.frame_ids[0] + 1,
            0.01,
            0.02,
            self.backend_identifier,
            (),
        )


def raw_registry(tmp_path: Path, *, present: bool = True):
    path = tmp_path / "L21_V001.mp4"
    if present:
        path.touch()
    return RawVideoRegistry((RawVideoRecord("L21_V001", path if present else None),))


def test_window_bounds_and_sampling() -> None:
    assert build_frame_window(
        2, fps=10, total_frame_count=100, before_seconds=1, after_seconds=1
    ) == (0, 12)
    assert build_frame_window(
        98, fps=10, total_frame_count=100, before_seconds=1, after_seconds=1
    ) == (88, 99)
    assert coarse_frame_ids(40, 60, stride=7, candidate_frame_id=50) == (40, 47, 50, 54)
    assert fine_frame_ids((50, 52), window_start=48, window_end=54, radius=2, stride=2) == (
        48,
        50,
        52,
        54,
    )


def test_decode_request_guard_and_absolute_contract(tmp_path: Path) -> None:
    probe = FakeDecoder().probe(RawVideoRecord("L21_V001", tmp_path / "x.mp4"))
    with pytest.raises(RuntimeError, match="max_decoded_frames"):
        DecodeRequest(probe, (0, 20), 10)
    result = FakeDecoder().decode(DecodeRequest(probe, (4, 10), 10))
    assert [frame.absolute_frame_id for frame in result.frames] == [4, 10]


def test_local_weighted_rrf_and_tie_breaking() -> None:
    frames = (8, 3)
    image = np.asarray([[1, 0], [1, 0]], dtype=np.float32)
    text = np.asarray([[1, 0], [0, 1]], dtype=np.float32)
    fused = fuse_local_frame_rankings(
        frames,
        image,
        (variant("vi", "target", 2), variant("en", "other", 1)),
        text,
        rrf_constant=60,
    )
    assert fused[0].absolute_frame_id == 3
    assert fused[0].fusion_score == pytest.approx(3 / 61)
    assert all("cosine_score" in item for item in fused[0].per_variant_provenance)


def test_successful_coarse_to_fine_preserves_absolute_frame(tmp_path: Path) -> None:
    decoder = FakeDecoder()
    encoder = FakeEncoder(target=55)
    refiner = ExactFrameRefiner(raw_videos=raw_registry(tmp_path), decoder=decoder, encoder=encoder)
    outcome = refiner.refine_query(
        RefinementQuery("Q1", (variant(),), (candidate(),)),
        RefinementConfig(
            top_candidates_to_refine=1,
            window_before_seconds=1,
            window_after_seconds=1,
            coarse_stride_frames=5,
            coarse_top_n=1,
            fine_radius_frames=3,
            fine_stride_frames=1,
            image_batch_size=2,
            max_decoded_frames_per_candidate=30,
            output_top_k=1,
        ),
    )
    record = outcome.candidates[0]
    assert record.refined_frame_id == 55
    assert 40 <= record.refined_frame_id <= 60
    assert outcome.result.ranked_candidates[0].frame_id == 55
    assert outcome.result.ranked_candidates[0].frame_id != 99
    assert record.status is RefinementStatus.REFINED
    assert encoder.text_calls == 1
    assert decoder.probe_calls == 1
    assert max(encoder.image_batches) <= 2


@pytest.mark.parametrize(
    ("policy", "expected_frame", "expected_status"),
    [
        (MissingRawVideoPolicy.KEEP_ORIGINAL, 50, RefinementStatus.KEEP_ORIGINAL),
        (MissingRawVideoPolicy.SKIP_CANDIDATE, None, RefinementStatus.SKIPPED),
    ],
)
def test_missing_raw_policies(tmp_path, policy, expected_frame, expected_status) -> None:
    outcome = ExactFrameRefiner(
        raw_videos=raw_registry(tmp_path, present=False),
        decoder=FakeDecoder(),
        encoder=FakeEncoder(),
    ).refine_query(
        RefinementQuery("Q1", (variant(),), (candidate(),)),
        RefinementConfig(
            top_candidates_to_refine=1, output_top_k=1, missing_raw_video_policy=policy
        ),
    )
    assert outcome.candidates[0].refined_frame_id == expected_frame
    assert outcome.candidates[0].status is expected_status


def test_missing_raw_fail_query(tmp_path: Path) -> None:
    with pytest.raises(QueryRefinementError, match="raw video missing"):
        ExactFrameRefiner(
            raw_videos=raw_registry(tmp_path, present=False),
            decoder=FakeDecoder(),
            encoder=FakeEncoder(),
        ).refine_query(
            RefinementQuery("Q1", (variant(),), (candidate(),)),
            RefinementConfig(missing_raw_video_policy=MissingRawVideoPolicy.FAIL_QUERY),
        )


@pytest.mark.parametrize(
    ("policy", "expected_status"),
    [
        (CandidateFailurePolicy.KEEP_ORIGINAL, RefinementStatus.KEEP_ORIGINAL),
        (CandidateFailurePolicy.SKIP_CANDIDATE, RefinementStatus.SKIPPED),
    ],
)
def test_decode_failure_policies(tmp_path, policy, expected_status) -> None:
    outcome = ExactFrameRefiner(
        raw_videos=raw_registry(tmp_path), decoder=FakeDecoder(fail=True), encoder=FakeEncoder()
    ).refine_query(
        RefinementQuery("Q1", (variant(),), (candidate(),)),
        RefinementConfig(candidate_failure_policy=policy),
    )
    assert outcome.candidates[0].status is expected_status
    assert outcome.candidates[0].failure_reason


def test_decode_failure_fail_query_policy(tmp_path: Path) -> None:
    with pytest.raises(QueryRefinementError, match="controlled decode failure"):
        ExactFrameRefiner(
            raw_videos=raw_registry(tmp_path),
            decoder=FakeDecoder(fail=True),
            encoder=FakeEncoder(),
        ).refine_query(
            RefinementQuery("Q1", (variant(),), (candidate(),)),
            RefinementConfig(candidate_failure_policy=CandidateFailurePolicy.FAIL_QUERY),
        )


def test_original_order_dedup_and_contiguous_ranks() -> None:
    from system_tai.refinement.engine import QueryRefinementOutcome

    del QueryRefinementOutcome
    base = candidate()
    refiner = ExactFrameRefiner.__new__(ExactFrameRefiner)
    unchanged = ExactFrameRefiner._unchanged_candidate
    records = (
        unchanged(base, status=RefinementStatus.KEEP_ORIGINAL, warning="x"),
        unchanged(candidate(2, 50), status=RefinementStatus.KEEP_ORIGINAL, warning="x"),
        unchanged(candidate(3, 70), status=RefinementStatus.KEEP_ORIGINAL, warning="x"),
    )
    result, warnings = refiner._build_final_result("Q1", records, 100)
    assert [item.frame_id for item in result.ranked_candidates] == [50, 70]
    assert [item.rank for item in result.ranked_candidates] == [1, 2]
    assert any("duplicate" in warning for warning in warnings)


def test_final_output_is_capped_at_one_hundred() -> None:
    unchanged = ExactFrameRefiner._unchanged_candidate
    records = tuple(
        unchanged(
            candidate(rank, frame_id=rank),
            status=RefinementStatus.NOT_REFINED,
            warning="outside refinement range",
        )
        for rank in range(1, 102)
    )
    result, _warnings = ExactFrameRefiner._build_final_result("Q1", records, 100)
    assert len(result.ranked_candidates) == 100
    assert result.ranked_candidates[-1].rank == 100


def test_config_validation_and_cross_platform_path_types() -> None:
    with pytest.raises(ValueError):
        RefinementConfig(top_candidates_to_refine=0)
    with pytest.raises(ValueError):
        RefinementConfig(max_decoded_frames_per_candidate=0)
    assert PureWindowsPath(r"C:\BTC\L21_V001.mp4").stem == "L21_V001"
    assert PurePosixPath("/btc/L21_V001.mp4").stem == "L21_V001"


def test_phase4_cpu_and_fake_cuda_device_selection() -> None:
    unavailable = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    available = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
    assert resolve_device("cpu", torch_module=unavailable) == "cpu"
    assert resolve_device("auto", torch_module=available) == "cuda"
    assert resolve_device("cuda", torch_module=available) == "cuda"
    with pytest.raises(RuntimeError, match="unavailable"):
        resolve_device("cuda", torch_module=unavailable)
