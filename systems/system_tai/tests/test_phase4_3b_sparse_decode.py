from pathlib import Path
from typing import Any

import numpy as np
import pytest

from system_tai.refinement.models import (
    Phase3Candidate,
    RefinementConfig,
    RefinementQuery,
)
from system_tai.refinement.video import (
    CoarseDecodeStrategy,
    DecodeRequest,
    OpenCVVideoDecoder,
    RawVideoError,
    RawVideoRecord,
    RawVideoRegistry,
    VideoProbe,
)
from system_tai.retrieval.multi_query import QueryLanguage, QueryVariant, QueryVariantType
from tests.test_phase4_2_session import setup_runtime


class FakeVideoCapture:
    def __init__(
        self,
        total_frames: int = 200,
        fail_seek_at: int | None = None,
        fail_read_at: int | None = None,
        pos_frames_offset: float = 0.0,
    ):
        self._total = total_frames
        self._pos = 0
        self._opened = True
        self.fail_seek_at = fail_seek_at
        self.fail_read_at = fail_read_at
        self.pos_frames_offset = pos_frames_offset
        self.set_calls: list[int] = []
        self.read_calls = 0

    def isOpened(self) -> bool:
        return self._opened

    def set(self, propId: int, value: float) -> bool:
        # CAP_PROP_POS_FRAMES = 1
        if propId == 1:
            val = int(value)
            self.set_calls.append(val)
            if self.fail_seek_at is not None and val == self.fail_seek_at:
                return False
            self._pos = val
            return True
        return False

    def get(self, propId: int) -> float:
        # CAP_PROP_POS_FRAMES = 1
        if propId == 1:
            return float(self._pos) + self.pos_frames_offset
        # CAP_PROP_FRAME_COUNT = 7, CAP_PROP_FPS = 5
        if propId == 7:
            return float(self._total)
        if propId == 5:
            return 30.0
        # WIDTH/HEIGHT = 3/4
        if propId == 3:
            return 1920.0
        if propId == 4:
            return 1080.0
        return 0.0

    def read(self) -> tuple[bool, Any]:
        self.read_calls += 1
        if self.fail_read_at is not None and self._pos == self.fail_read_at:
            self.fail_read_at = None
            return False, None
        if self._pos >= self._total:
            return False, None
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        self._pos += 1
        return True, img

    def release(self) -> None:
        self._opened = False


class FakeCV2Module:
    CAP_PROP_POS_FRAMES = 1
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4
    CAP_PROP_FPS = 5
    CAP_PROP_FRAME_COUNT = 7

    def __init__(self, **kwargs: Any):
        self.kwargs = kwargs

    def VideoCapture(self, path: str) -> Any:
        kwargs = self.kwargs.copy()
        if kwargs.get("fail_on_nth_capture", 0) > 0:
            self.kwargs["fail_on_nth_capture"] -= 1
            if "fail_read_at" in kwargs:
                del kwargs["fail_read_at"]
            if "fail_seek_at" in kwargs:
                del kwargs["fail_seek_at"]
        else:
            if "fail_read_at" in self.kwargs:
                del self.kwargs["fail_read_at"]
            if "fail_seek_at" in self.kwargs:
                del self.kwargs["fail_seek_at"]
        kwargs.pop("fail_on_nth_capture", None)
        return FakeVideoCapture(**kwargs)


def test_sparse_success(tmp_path: Path) -> None:
    """1. SPARSE SUCCESS"""
    decoder = OpenCVVideoDecoder(cv2_module=FakeCV2Module())
    probe = VideoProbe(
        "v1", tmp_path / "v1.mp4", decoder.backend_identifier, 30.0, 200, 1920, 1080, 200 / 30.0
    )
    req = DecodeRequest(probe, (100, 115, 130, 145), 500)

    res = decoder.decode_sparse_verified(req, fallback_to_sequential=False)

    assert res.decode_strategy == "sparse_verified"
    assert res.decoded_frame_count == 4
    assert len(res.frames) == 4
    assert tuple(f.absolute_frame_id for f in res.frames) == (100, 115, 130, 145)

    # Verify exact seeks
    # To inspect the instance used, we can hack the module.


def _get_tracked_decoder(**kwargs: Any) -> tuple[OpenCVVideoDecoder, list[FakeVideoCapture]]:
    caps = []

    class TrackingFakeCV2(FakeCV2Module):
        def VideoCapture(self, path: str) -> Any:
            cap = super().VideoCapture(path)
            caps.append(cap)
            return cap

    return OpenCVVideoDecoder(cv2_module=TrackingFakeCV2(**kwargs)), caps


def test_sparse_success_tracking(tmp_path: Path) -> None:
    decoder, caps = _get_tracked_decoder()
    probe = VideoProbe(
        "v1", tmp_path / "v1.mp4", decoder.backend_identifier, 30.0, 200, 1920, 1080, 200 / 30.0
    )
    req = DecodeRequest(probe, (100, 115, 130, 145), 500)

    res = decoder.decode_sparse_verified(req, fallback_to_sequential=False)

    assert res.decode_strategy == "sparse_verified"
    assert res.decoded_frame_count == 4

    assert len(caps) == 1
    cap = caps[0]
    assert cap.set_calls == [100, 115, 130, 145]
    assert cap.read_calls == 4
    assert not cap.isOpened()  # Released


def test_position_verification(tmp_path: Path) -> None:
    """2. POSITION VERIFICATION"""
    decoder, _ = _get_tracked_decoder(pos_frames_offset=-1.0)  # observer position will disagree
    probe = VideoProbe(
        "v1", tmp_path / "v1.mp4", decoder.backend_identifier, 30.0, 200, 1920, 1080, 200 / 30.0
    )
    req = DecodeRequest(probe, (115,), 500)

    with pytest.raises(
        RawVideoError, match="decoder position disagrees with requested absolute frame"
    ):
        decoder.decode_sparse_verified(req, fallback_to_sequential=False)


def test_seek_failure(tmp_path: Path) -> None:
    """3. SEEK FAILURE"""
    decoder, _ = _get_tracked_decoder(fail_seek_at=115)
    probe = VideoProbe(
        "v1", tmp_path / "v1.mp4", decoder.backend_identifier, 30.0, 200, 1920, 1080, 200 / 30.0
    )
    req = DecodeRequest(probe, (100, 115), 500)

    with pytest.raises(RawVideoError, match="raw-video seek failed at frame 115"):
        decoder.decode_sparse_verified(req, fallback_to_sequential=False)


def test_read_failure(tmp_path: Path) -> None:
    """4. READ FAILURE"""
    decoder, _ = _get_tracked_decoder(fail_read_at=115)
    probe = VideoProbe(
        "v1", tmp_path / "v1.mp4", decoder.backend_identifier, 30.0, 200, 1920, 1080, 200 / 30.0
    )
    req = DecodeRequest(probe, (100, 115), 500)

    with pytest.raises(RawVideoError, match="raw-video decode failed at absolute frame 115"):
        decoder.decode_sparse_verified(req, fallback_to_sequential=False)


def test_fallback_to_sequential(tmp_path: Path) -> None:
    """5. FALLBACK TO SEQUENTIAL"""
    decoder, caps = _get_tracked_decoder(fail_read_at=115)
    probe = VideoProbe(
        "v1", tmp_path / "v1.mp4", decoder.backend_identifier, 30.0, 200, 1920, 1080, 200 / 30.0
    )
    req = DecodeRequest(probe, (100, 115), 500)

    res = decoder.decode_sparse_verified(req, fallback_to_sequential=True)

    assert res.decode_strategy == "sparse_verified_fallback_sequential"
    # fallback sequential reads from 100 to 115 = 16 frames.
    # initial sparse decoded 1 frame (100) before failing at 115.
    # total physical reads = 2 + 16 = 18.
    assert res.decoded_frame_count == 18
    assert len(res.frames) == 2
    assert res.frames[0].absolute_frame_id == 100
    assert res.frames[1].absolute_frame_id == 115
    assert any("fallback used" in w for w in res.warnings)

    assert len(caps) == 2  # 1 for sparse, 1 for sequential


def test_sequential_backward_compatibility(tmp_path: Path) -> None:
    """6. SEQUENTIAL BACKWARD COMPATIBILITY"""
    decoder, _ = _get_tracked_decoder()
    probe = VideoProbe(
        "v1", tmp_path / "v1.mp4", decoder.backend_identifier, 30.0, 200, 1920, 1080, 200 / 30.0
    )
    req = DecodeRequest(probe, (100, 115), 500)

    res = decoder.decode(req)
    assert res.decode_strategy == "sequential_bounded"
    assert res.decoded_frame_count == 16  # 100 to 115
    assert tuple(f.absolute_frame_id for f in res.frames) == (100, 115)


def test_coarse_strategy_only_and_refinement_semantic_equality(tmp_path: Path) -> None:
    """7. COARSE STRATEGY ONLY and 8. REFINEMENT SEMANTIC EQUALITY"""
    runtime_seq, _ = setup_runtime(tmp_path / "seq")
    runtime_sparse, _ = setup_runtime(tmp_path / "spa")

    # We patch the decoder in both runtimes to use our tracking fake cv2
    class TrackingFakeCV2Local(FakeCV2Module):
        def __init__(self, **kwargs: Any):
            super().__init__(**kwargs)
            self.sparse_calls = 0
            self.seq_calls = 0
            self.caps: list[FakeVideoCapture] = []

        def VideoCapture(self, path: str) -> Any:
            cap = FakeVideoCapture(**self.kwargs)
            self.caps.append(cap)
            return cap

    cv2_seq = TrackingFakeCV2Local()
    cv2_spa = TrackingFakeCV2Local()

    runtime_seq.refiner.decoder = OpenCVVideoDecoder(cv2_module=cv2_seq)
    runtime_sparse.refiner.decoder = OpenCVVideoDecoder(cv2_module=cv2_spa)

    # override decode and decode_sparse_verified to count calls
    original_seq_decode = runtime_seq.refiner.decoder.decode

    def counted_seq_decode(req: DecodeRequest) -> Any:
        cv2_seq.seq_calls += 1
        return original_seq_decode(req)

    runtime_seq.refiner.decoder.decode = counted_seq_decode

    original_spa_decode = runtime_sparse.refiner.decoder.decode

    def counted_spa_decode(req: DecodeRequest) -> Any:
        cv2_spa.seq_calls += 1
        return original_spa_decode(req)

    runtime_sparse.refiner.decoder.decode = counted_spa_decode

    original_spa_sparse = runtime_sparse.refiner.decoder.decode_sparse_verified

    def counted_spa_sparse(req: DecodeRequest, **kwargs: Any) -> Any:
        cv2_spa.sparse_calls += 1
        return original_spa_sparse(req, **kwargs)

    runtime_sparse.refiner.decoder.decode_sparse_verified = counted_spa_sparse

    query = RefinementQuery(
        query_id="q1",
        variants=(
            QueryVariant(
                "vi",
                "nhiều người",
                QueryLanguage.VIETNAMESE,
                QueryVariantType.VIETNAMESE_DIRECT,
                1.0,
            ),
        ),
        candidates=(Phase3Candidate("q1", 1, "L21_V001", 100, 1.0, {}),),
    )

    config_seq = RefinementConfig(coarse_decode_strategy=CoarseDecodeStrategy.SEQUENTIAL)
    config_spa = RefinementConfig(coarse_decode_strategy=CoarseDecodeStrategy.SPARSE_VERIFIED)

    # We need to setup a mock raw video file because decoder probe checks it
    vid_file = tmp_path / "seq" / "L21_V001.mp4"
    vid_file.parent.mkdir(parents=True, exist_ok=True)
    vid_file.touch()

    vid_file2 = tmp_path / "spa" / "L21_V001.mp4"
    vid_file2.parent.mkdir(parents=True, exist_ok=True)
    vid_file2.touch()

    runtime_seq.refiner.raw_videos = RawVideoRegistry([RawVideoRecord("L21_V001", vid_file, ())])
    runtime_sparse.refiner.raw_videos = RawVideoRegistry(
        [RawVideoRecord("L21_V001", vid_file2, ())]
    )

    out_seq = runtime_seq.refiner.refine_query(query, config_seq)
    out_spa = runtime_sparse.refiner.refine_query(query, config_spa)

    # verify call counts
    assert cv2_seq.seq_calls == 2  # 1 coarse, 1 fine
    assert cv2_seq.sparse_calls == 0

    assert cv2_spa.seq_calls == 1  # 1 fine
    assert cv2_spa.sparse_calls == 1  # 1 coarse

    # verify identical result
    c_seq = out_seq.candidates[0]
    c_spa = out_spa.candidates[0]
    assert c_seq.refined_frame_id == c_spa.refined_frame_id
    assert c_seq.status == c_spa.status

    if out_seq.result.ranked_candidates:
        r_seq = out_seq.result.ranked_candidates[0]
        r_spa = out_spa.result.ranked_candidates[0]
        assert r_seq.frame_id == r_spa.frame_id
        assert r_seq.score == r_spa.score
        assert r_seq.source == r_spa.source


def test_failure_fallback_semantic_equality(tmp_path: Path) -> None:
    """9. FAILURE FALLBACK SEMANTIC EQUALITY"""
    runtime_seq, _ = setup_runtime(tmp_path / "seq")
    runtime_spa, _ = setup_runtime(tmp_path / "spa")

    cv2_seq = FakeCV2Module()
    cv2_spa = FakeCV2Module(
        fail_read_at=100, fail_on_nth_capture=1
    )  # force sparse to fail and fallback

    runtime_seq.refiner.decoder = OpenCVVideoDecoder(cv2_module=cv2_seq)
    runtime_spa.refiner.decoder = OpenCVVideoDecoder(cv2_module=cv2_spa)

    query = RefinementQuery(
        query_id="q1",
        variants=(
            QueryVariant(
                "vi",
                "nhiều người",
                QueryLanguage.VIETNAMESE,
                QueryVariantType.VIETNAMESE_DIRECT,
                1.0,
            ),
        ),
        candidates=(Phase3Candidate("q1", 1, "L21_V001", 100, 1.0, {}),),
    )

    config_seq = RefinementConfig(coarse_decode_strategy=CoarseDecodeStrategy.SEQUENTIAL)
    config_spa = RefinementConfig(coarse_decode_strategy=CoarseDecodeStrategy.SPARSE_VERIFIED)

    vid_file = tmp_path / "seq" / "L21_V001.mp4"
    vid_file.parent.mkdir(parents=True, exist_ok=True)
    vid_file.touch()

    vid_file2 = tmp_path / "spa" / "L21_V001.mp4"
    vid_file2.parent.mkdir(parents=True, exist_ok=True)
    vid_file2.touch()

    runtime_seq.refiner.raw_videos = RawVideoRegistry([RawVideoRecord("L21_V001", vid_file, ())])
    runtime_spa.refiner.raw_videos = RawVideoRegistry([RawVideoRecord("L21_V001", vid_file2, ())])

    out_seq = runtime_seq.refiner.refine_query(query, config_seq)
    out_spa = runtime_spa.refiner.refine_query(query, config_spa)

    # Both outcomes should be semantically identical
    assert out_spa.timings["coarse_sparse_fallback_count"] == 1

    c_seq = out_seq.candidates[0]
    c_spa = out_spa.candidates[0]
    assert c_seq.refined_frame_id == c_spa.refined_frame_id
    assert c_seq.status == c_spa.status

    if out_seq.result.ranked_candidates:
        r_seq = out_seq.result.ranked_candidates[0]
        r_spa = out_spa.result.ranked_candidates[0]
        assert r_seq.frame_id == r_spa.frame_id
        assert r_seq.score == r_spa.score
        assert r_seq.source == r_spa.source
