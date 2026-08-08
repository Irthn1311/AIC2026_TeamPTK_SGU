"""Unit and integration test suite for PRELIMINARY P0-OPT1.

Request-Scoped Raw-Frame CLIP Image Embedding Reuse and Equivalence.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from system_tai.refinement.engine import (
    ExactFrameRefiner,
    _encode_frames_with_cache,
)
from system_tai.refinement.models import (
    Phase3Candidate,
    RefinementConfig,
    RefinementQuery,
    RefinementStatus,
)
from system_tai.refinement.video import (
    DecodedFrame,
    DecodeResult,
    RawVideoRecord,
    VideoProbe,
)
from system_tai.retrieval.multi_query import QueryLanguage, QueryVariant, QueryVariantType


class MockEncoder:
    def __init__(self, dim: int = 16) -> None:
        self.dim = dim
        self.encode_images_calls: list[list[Any]] = []
        self.encode_texts_calls: list[list[str]] = []

    def encode_images(self, images: list[Any], batch_size: int = 32) -> np.ndarray:
        self.encode_images_calls.append(list(images))
        rows = []
        for img in images:
            seed = abs(hash(str(img))) % 10000 + 1
            rng = np.random.RandomState(seed)
            vec = rng.randn(self.dim).astype(np.float32)
            norm = np.linalg.norm(vec)
            rows.append(vec / norm)
        return np.vstack(rows).astype(np.float32)

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        self.encode_texts_calls.append(list(texts))
        rows = []
        for txt in texts:
            seed = abs(hash(txt)) % 10000 + 1
            rng = np.random.RandomState(seed)
            vec = rng.randn(self.dim).astype(np.float32)
            norm = np.linalg.norm(vec)
            rows.append(vec / norm)
        return np.vstack(rows).astype(np.float32)


class MockDecoder:
    def __init__(self) -> None:
        self.probe_calls: list[str] = []
        self.decode_calls: list[list[int]] = []

    def probe(self, record: RawVideoRecord) -> VideoProbe:
        self.probe_calls.append(record.video_id)
        return VideoProbe(
            video_id=record.video_id,
            raw_video_path=record.raw_video_path or Path("mock.mp4"),
            fps=30.0,
            total_frame_count=1000,
            duration_seconds=33.33,
            width=640,
            height=480,
            decoder_backend="mock",
        )

    def decode(self, req: Any) -> DecodeResult:
        self.decode_calls.append(list(req.frame_ids))
        frames = [
            DecodedFrame(
                absolute_frame_id=fid,
                image=f"img::{req.probe.video_id}::{fid}",
                timestamp_seconds=fid / req.probe.fps,
            )
            for fid in req.frame_ids
        ]
        return DecodeResult(
            frames=tuple(frames),
            decoded_frame_count=len(frames),
            video_open_seconds=0.001,
            decode_seconds=0.005,
            decoder_backend="mock",
            warnings=(),
        )


class MockRawVideoRegistry:
    def __init__(self, video_ids: list[str]) -> None:
        self.video_ids = set(video_ids)

    def get(self, video_id: str) -> RawVideoRecord:
        if video_id in self.video_ids:
            return RawVideoRecord(video_id=video_id, raw_video_path=Path(f"/path/{video_id}.mp4"))
        return RawVideoRecord(video_id=video_id, raw_video_path=None)


def test_cache_none_legacy_path() -> None:
    encoder = MockEncoder()
    frames = [
        DecodedFrame(absolute_frame_id=10, image="img10", timestamp_seconds=0.3),
        DecodedFrame(absolute_frame_id=20, image="img20", timestamp_seconds=0.6),
    ]

    res = _encode_frames_with_cache(
        video_id="V1",
        frames=frames,
        encoder=encoder,  # type: ignore[arg-type]
        batch_size=32,
        frame_embedding_cache=None,
    )

    assert len(encoder.encode_images_calls) == 1
    assert encoder.encode_images_calls[0] == ["img10", "img20"]
    assert res.shape == (2, 16)


def test_same_video_same_frame_hit() -> None:
    encoder = MockEncoder()
    cache: dict[tuple[str, int], np.ndarray] = {}
    frames = [DecodedFrame(absolute_frame_id=100, image="img100", timestamp_seconds=3.3)]

    res1 = _encode_frames_with_cache(
        video_id="V1",
        frames=frames,
        encoder=encoder,  # type: ignore[arg-type]
        batch_size=32,
        frame_embedding_cache=cache,
    )
    assert len(encoder.encode_images_calls) == 1
    assert ("V1", 100) in cache

    # Second call for SAME video & SAME frame
    res2 = _encode_frames_with_cache(
        video_id="V1",
        frames=frames,
        encoder=encoder,  # type: ignore[arg-type]
        batch_size=32,
        frame_embedding_cache=cache,
    )
    # Encoder count must remain 1 (cache hit!)
    assert len(encoder.encode_images_calls) == 1
    np.testing.assert_array_equal(res1, res2)


def test_different_video_no_collision() -> None:
    encoder = MockEncoder()
    cache: dict[tuple[str, int], np.ndarray] = {}
    f1 = [DecodedFrame(absolute_frame_id=100, image="img_v1_100", timestamp_seconds=3.3)]
    f2 = [DecodedFrame(absolute_frame_id=100, image="img_v2_100", timestamp_seconds=3.3)]

    _encode_frames_with_cache(
        video_id="V1",
        frames=f1,
        encoder=encoder,  # type: ignore[arg-type]
        batch_size=32,
        frame_embedding_cache=cache,
    )
    assert len(encoder.encode_images_calls) == 1

    _encode_frames_with_cache(
        video_id="V2",
        frames=f2,
        encoder=encoder,  # type: ignore[arg-type]
        batch_size=32,
        frame_embedding_cache=cache,
    )
    # Must encode V2 because video_id is different!
    assert len(encoder.encode_images_calls) == 2
    assert ("V1", 100) in cache
    assert ("V2", 100) in cache


def test_same_call_duplicate_dedup() -> None:
    encoder = MockEncoder()
    cache: dict[tuple[str, int], np.ndarray] = {}
    frames = [
        DecodedFrame(absolute_frame_id=100, image="img100", timestamp_seconds=3.3),
        DecodedFrame(absolute_frame_id=110, image="img110", timestamp_seconds=3.6),
        DecodedFrame(absolute_frame_id=100, image="img100", timestamp_seconds=3.3),
        DecodedFrame(absolute_frame_id=120, image="img120", timestamp_seconds=4.0),
    ]

    res = _encode_frames_with_cache(
        video_id="V1",
        frames=frames,
        encoder=encoder,  # type: ignore[arg-type]
        batch_size=32,
        frame_embedding_cache=cache,
    )

    # Encoder received unique missing images [100, 110, 120]
    assert len(encoder.encode_images_calls) == 1
    assert encoder.encode_images_calls[0] == ["img100", "img110", "img120"]
    assert res.shape == (4, 16)
    # Row 0 and Row 2 correspond to frame 100 and must be equal
    np.testing.assert_array_equal(res[0], res[2])


def test_partial_cache_hit() -> None:
    encoder = MockEncoder()
    cache: dict[tuple[str, int], np.ndarray] = {}
    cache[("V1", 100)] = np.ones((16,), dtype=np.float32)
    cache[("V1", 120)] = np.ones((16,), dtype=np.float32) * 2.0

    frames = [
        DecodedFrame(absolute_frame_id=100, image="img100", timestamp_seconds=3.3),
        DecodedFrame(absolute_frame_id=110, image="img110", timestamp_seconds=3.6),
        DecodedFrame(absolute_frame_id=120, image="img120", timestamp_seconds=4.0),
        DecodedFrame(absolute_frame_id=130, image="img130", timestamp_seconds=4.3),
    ]

    res = _encode_frames_with_cache(
        video_id="V1",
        frames=frames,
        encoder=encoder,  # type: ignore[arg-type]
        batch_size=32,
        frame_embedding_cache=cache,
    )

    # Only img110 and img130 encoded
    assert len(encoder.encode_images_calls) == 1
    assert encoder.encode_images_calls[0] == ["img110", "img130"]
    assert res.shape == (4, 16)
    assert res[0][0] == 1.0
    assert res[2][0] == 2.0


def test_all_hit_no_encoder_call() -> None:
    encoder = MockEncoder()
    cache: dict[tuple[str, int], np.ndarray] = {}
    cache[("V1", 100)] = np.ones((16,), dtype=np.float32)
    frames = [DecodedFrame(absolute_frame_id=100, image="img100", timestamp_seconds=3.3)]

    res = _encode_frames_with_cache(
        video_id="V1",
        frames=frames,
        encoder=encoder,  # type: ignore[arg-type]
        batch_size=32,
        frame_embedding_cache=cache,
    )

    assert len(encoder.encode_images_calls) == 0
    assert res.shape == (1, 16)


def test_coarse_to_fine_reuse() -> None:
    encoder = MockEncoder()
    decoder = MockDecoder()
    raw_videos = MockRawVideoRegistry(["V1"])
    refiner = ExactFrameRefiner(
        raw_videos=raw_videos,  # type: ignore[arg-type]
        decoder=decoder,  # type: ignore[arg-type]
        encoder=encoder,  # type: ignore[arg-type]
    )

    v1 = QueryVariant(
        variant_id="var1",
        text="query text",
        language=QueryLanguage.ENGLISH,
        variant_type=QueryVariantType.ENGLISH_TRANSLATION,
        weight=1.0,
    )
    query = RefinementQuery(
        query_id="Q1",
        variants=(v1,),
        candidates=(
            Phase3Candidate(
                query_id="Q1",
                rank=1,
                video_id="V1",
                frame_id=100,
                retrieval_score=0.9,
                retrieval_provenance={},
            ),
        ),
    )
    config = RefinementConfig(
        window_before_seconds=0.3,
        window_after_seconds=0.3,
        coarse_stride_frames=5,
        fine_radius_frames=2,
        fine_stride_frames=1,
        coarse_top_n=1,
        max_decoded_frames_per_candidate=100,
    )

    cache: dict[tuple[str, int], np.ndarray] = {}
    outcome = refiner.refine_query(query, config, frame_embedding_cache=cache)

    assert outcome.candidates[0].status in {
        RefinementStatus.REFINED,
        RefinementStatus.KEEP_ORIGINAL,
    }
    # Both coarse and fine stages executed
    assert len(encoder.encode_images_calls) >= 2
    # Verify every cached entry is in video V1
    assert all(k[0] == "V1" for k in cache.keys())


def test_cross_event_semantic_independence() -> None:
    encoder = MockEncoder(dim=4)
    decoder = MockDecoder()
    raw_videos = MockRawVideoRegistry(["V1"])
    refiner = ExactFrameRefiner(
        raw_videos=raw_videos,  # type: ignore[arg-type]
        decoder=decoder,  # type: ignore[arg-type]
        encoder=encoder,  # type: ignore[arg-type]
    )

    # Populate cache for all frames 80..120 with background vector [0.5, 0.5, 0.5, 0.5]
    bg_vec = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    bg_vec /= np.linalg.norm(bg_vec)
    cache: dict[tuple[str, int], np.ndarray] = {
        ("V1", fid): bg_vec.copy() for fid in range(80, 121)
    }

    # Frame 95 vector strongly aligned with text A
    v95 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    # Frame 105 vector strongly aligned with text B
    v105 = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    cache[("V1", 95)] = v95
    cache[("V1", 105)] = v105

    # Text A aligns with frame 95
    text_emb_A = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    # Text B aligns with frame 105
    text_emb_B = np.array([[0.0, 0.0, 1.0, 0.0]], dtype=np.float32)

    varA = QueryVariant(
        "vA", "text A", QueryLanguage.ENGLISH, QueryVariantType.ENGLISH_TRANSLATION, 1.0
    )
    varB = QueryVariant(
        "vB", "text B", QueryLanguage.ENGLISH, QueryVariantType.ENGLISH_TRANSLATION, 1.0
    )

    qA = RefinementQuery("QA", (varA,), (Phase3Candidate("QA", 1, "V1", 100, 0.9, {}),))
    qB = RefinementQuery("QB", (varB,), (Phase3Candidate("QB", 1, "V1", 100, 0.9, {}),))

    config = RefinementConfig(
        window_before_seconds=0.3,
        window_after_seconds=0.3,
        coarse_stride_frames=1,
        fine_radius_frames=1,
        coarse_top_n=1,
        max_decoded_frames_per_candidate=100,
    )

    outcomeA = refiner.refine_query(
        qA, config, precomputed_text_embeddings=text_emb_A, frame_embedding_cache=cache
    )
    outcomeB = refiner.refine_query(
        qB, config, precomputed_text_embeddings=text_emb_B, frame_embedding_cache=cache
    )

    winnerA = outcomeA.candidates[0].refined_frame_id
    winnerB = outcomeB.candidates[0].refined_frame_id

    # Text A chose 95, Text B chose 105 despite sharing image embedding cache!
    assert winnerA == 95
    assert winnerB == 105
    assert winnerA != winnerB


def test_refiner_equivalence_cache_off_vs_on() -> None:
    encoder = MockEncoder(dim=16)
    decoder = MockDecoder()
    raw_videos = MockRawVideoRegistry(["V1"])

    refiner_off = ExactFrameRefiner(
        raw_videos=raw_videos,  # type: ignore[arg-type]
        decoder=decoder,  # type: ignore[arg-type]
        encoder=encoder,  # type: ignore[arg-type]
    )
    refiner_on = ExactFrameRefiner(
        raw_videos=raw_videos,  # type: ignore[arg-type]
        decoder=decoder,  # type: ignore[arg-type]
        encoder=encoder,  # type: ignore[arg-type]
    )

    var = QueryVariant(
        "v1", "test", QueryLanguage.ENGLISH, QueryVariantType.ENGLISH_TRANSLATION, 1.0
    )
    query = RefinementQuery(
        query_id="Q_EQ",
        variants=(var,),
        candidates=(
            Phase3Candidate("Q_EQ", 1, "V1", 100, 0.9, {}),
            Phase3Candidate("Q_EQ", 2, "V1", 110, 0.8, {}),
        ),
    )
    config = RefinementConfig(
        window_before_seconds=0.3,
        window_after_seconds=0.3,
        coarse_stride_frames=5,
        fine_radius_frames=2,
        coarse_top_n=1,
        max_decoded_frames_per_candidate=100,
    )

    outcome_off = refiner_off.refine_query(query, config, frame_embedding_cache=None)
    calls_off = len(encoder.encode_images_calls)

    encoder.encode_images_calls.clear()
    cache_on: dict[tuple[str, int], np.ndarray] = {}
    outcome_on = refiner_on.refine_query(query, config, frame_embedding_cache=cache_on)
    calls_on = len(encoder.encode_images_calls)

    # Compare semantic outputs (must be 100% identical)
    assert outcome_off.result.query_id == outcome_on.result.query_id
    assert [c.frame_id for c in outcome_off.result.ranked_candidates] == [
        c.frame_id for c in outcome_on.result.ranked_candidates
    ]
    assert len(outcome_off.candidates) == len(outcome_on.candidates)
    for c_off, c_on in zip(outcome_off.candidates, outcome_on.candidates):
        assert c_off.refined_frame_id == c_on.refined_frame_id
        assert c_off.status == c_on.status
        assert c_off.coarse_frame_ids == c_on.coarse_frame_ids
        assert c_off.fine_frame_ids == c_on.fine_frame_ids
        assert c_off.refinement_fusion_score == pytest.approx(c_on.refinement_fusion_score)

    # Cache ON must require fewer or equal physical image encode calls
    assert calls_on <= calls_off


def test_no_runtime_persistent_cache() -> None:
    encoder = MockEncoder()
    decoder = MockDecoder()
    raw_videos = MockRawVideoRegistry(["V1"])
    refiner = ExactFrameRefiner(
        raw_videos=raw_videos,  # type: ignore[arg-type]
        decoder=decoder,  # type: ignore[arg-type]
        encoder=encoder,  # type: ignore[arg-type]
    )

    assert not hasattr(refiner, "frame_embedding_cache")
    assert not hasattr(refiner, "_frame_embedding_cache")
    assert not hasattr(encoder, "frame_embedding_cache")


def test_failure_atomicity_on_encode_error() -> None:
    class FailingEncoder(MockEncoder):
        def encode_images(self, images: list[Any], batch_size: int = 32) -> np.ndarray:
            raise RuntimeError("CUDA out of memory")

    encoder = FailingEncoder()
    cache: dict[tuple[str, int], np.ndarray] = {("V1", 50): np.zeros((16,), dtype=np.float32)}
    frames = [
        DecodedFrame(absolute_frame_id=50, image="img50", timestamp_seconds=1.6),
        DecodedFrame(absolute_frame_id=60, image="img60", timestamp_seconds=2.0),
    ]

    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        _encode_frames_with_cache(
            video_id="V1",
            frames=frames,
            encoder=encoder,  # type: ignore[arg-type]
            batch_size=32,
            frame_embedding_cache=cache,
        )

    # Cached entry for frame 50 remains intact, frame 60 was NOT partially committed!
    assert ("V1", 50) in cache
    assert ("V1", 60) not in cache


def test_empty_frame_legacy_rejection() -> None:
    class EmptyRejectingEncoder(MockEncoder):
        def encode_images(self, images: list[Any], batch_size: int = 32) -> np.ndarray:
            if not images:
                raise ValueError("cannot encode empty image sequence")
            return super().encode_images(images, batch_size=batch_size)

    encoder = EmptyRejectingEncoder()

    # A. frame_embedding_cache=None + empty frames -> raises ValueError
    with pytest.raises(ValueError, match="cannot encode empty image sequence"):
        _encode_frames_with_cache(
            video_id="V1",
            frames=[],
            encoder=encoder,  # type: ignore[arg-type]
            batch_size=32,
            frame_embedding_cache=None,
        )

    # B. frame_embedding_cache={} + empty frames -> raises ValueError & cache remains empty
    cache: dict[tuple[str, int], np.ndarray] = {}
    with pytest.raises(ValueError, match="cannot encode empty image sequence"):
        _encode_frames_with_cache(
            video_id="V1",
            frames=[],
            encoder=encoder,  # type: ignore[arg-type]
            batch_size=32,
            frame_embedding_cache=cache,
        )
    assert len(cache) == 0
