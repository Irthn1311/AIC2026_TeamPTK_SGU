"""Synthetic acceptance tests for automatic selected-video timeline scouting."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from system_tai.kis.session import build_parser, session_config_from_args
from system_tai.refinement.engine import (
    ExactFrameRefiner,
    timeline_sparse_frame_ids,
)
from system_tai.refinement.models import (
    Phase3Candidate,
    RefinementConfig,
    SelectedVideoTimelineScoutConfig,
)
from system_tai.refinement.video import (
    DecodedFrame,
    DecodeResult,
    RawVideoRecord,
    RawVideoRegistry,
    SparseDecodeRequest,
    VideoProbe,
)
from system_tai.retrieval.multi_query import (
    QueryLanguage,
    QueryVariant,
    QueryVariantType,
)


class _SyntheticEncoder:
    dimension = 2
    identifiers = {"model": "synthetic", "device": "cpu"}

    def __init__(self, peak_frame: int) -> None:
        self.peak_frame = peak_frame
        self.encoded_images = 0

    def encode_images(self, images, *, batch_size):
        del batch_size
        self.encoded_images += len(images)
        return np.asarray(
            [
                [
                    1.0,
                    abs(int(frame_id) - self.peak_frame) / 100.0,
                ]
                for frame_id in images
            ],
            dtype=np.float32,
        )


class _SyntheticDecoder:
    backend_identifier = "synthetic-absolute-sparse"

    def __init__(self, total_frames: int = 101) -> None:
        self.total_frames = total_frames
        self.sparse_requests: list[SparseDecodeRequest] = []

    def probe(self, record):
        return VideoProbe(
            record.video_id,
            record.raw_video_path,
            self.backend_identifier,
            10.0,
            self.total_frames,
            16,
            9,
            self.total_frames / 10.0,
        )

    def decode_sparse_verified(self, request, *, fallback_to_sequential):
        assert fallback_to_sequential is False
        self.sparse_requests.append(request)
        frames = tuple(
            DecodedFrame(frame_id, frame_id / request.probe.fps, frame_id)
            for frame_id in request.frame_ids
        )
        return DecodeResult(
            frames,
            len(frames),
            0.0,
            0.0,
            self.backend_identifier,
            (),
            "sparse_verified",
        )


def _slot(rank: int, video_id: str, frame_id: int) -> Phase3Candidate:
    return Phase3Candidate("Q", rank, video_id, frame_id, 1.0 / rank, {})


def test_sparse_timeline_sampling_covers_tail_without_known_target() -> None:
    probe = VideoProbe("synthetic", Path("synthetic.mp4"), "fake", 10, 101, 8, 8, 10.1)
    frame_ids = timeline_sparse_frame_ids(
        probe,
        sample_stride_seconds=2.0,
        max_samples=6,
    )

    assert frame_ids == (0, 20, 40, 60, 80, 100)
    assert len(frame_ids) == 6
    SparseDecodeRequest(probe, frame_ids, max_decoded_frames=6)


def test_scout_uses_retrieval_video_order_and_finds_late_synthetic_region(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "video-alpha.mp4"
    video_path.touch()
    decoder = _SyntheticDecoder()
    encoder = _SyntheticEncoder(peak_frame=80)
    refiner = ExactFrameRefiner(
        raw_videos=RawVideoRegistry((RawVideoRecord("video-alpha", video_path),)),
        decoder=decoder,
        encoder=encoder,
    )
    variant = QueryVariant(
        "scene",
        "synthetic target action",
        QueryLanguage.ENGLISH,
        QueryVariantType.ENGLISH_TRANSLATION,
        1.0,
    )
    outcome = refiner.scout_selected_video_timelines(
        query_id="Q",
        variants=(variant,),
        ranked_video_ids=("video-alpha",),
        rank_slots=(_slot(1, "video-alpha", 10), _slot(2, "video-alpha", 20)),
        config=SelectedVideoTimelineScoutConfig(
            enabled=True,
            max_videos=1,
            sample_stride_seconds=2.0,
            max_samples_per_video=6,
            max_regions_per_video=2,
            minimum_region_gap_seconds=1.0,
        ),
        refinement_config=RefinementConfig(),
        precomputed_text_embeddings=np.asarray([[1.0, 0.0]], dtype=np.float32),
        frame_embedding_cache={},
    )

    assert outcome.candidates[0].video_id == "video-alpha"
    assert outcome.candidates[0].frame_id == 80
    assert outcome.candidates[0].rank == 1
    assert outcome.trace["selection_source"] == "system_video_first_nomination"
    assert outcome.trace["hard_coded_target"] is False
    assert outcome.trace["videos"][0]["first_sample_frame_id"] == 0
    assert outcome.trace["videos"][0]["last_sample_frame_id"] == 100
    assert len(decoder.sparse_requests) == 1
    assert encoder.encoded_images == 6
    assert "ground_truth" not in str(outcome.trace).casefold()


def test_timeline_cli_is_opt_in_and_requires_video_first() -> None:
    parser = build_parser()
    enabled = parser.parse_args(
        [
            "--enable-kis-semantic-video-first",
            "--enable-kis-selected-video-timeline-scout",
            "--default-refine-top-n",
            "3",
        ]
    )
    config = session_config_from_args(enabled)
    assert config.selected_video_timeline_scout_config.enabled is True

    disabled = session_config_from_args(parser.parse_args([]))
    assert disabled.selected_video_timeline_scout_config.enabled is False
