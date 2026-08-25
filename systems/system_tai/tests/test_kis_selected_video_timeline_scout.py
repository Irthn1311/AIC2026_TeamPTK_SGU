"""Synthetic acceptance tests for automatic selected-video timeline scouting."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from system_tai.kis.session import build_parser, session_config_from_args
from system_tai.refinement.engine import (
    ExactFrameRefiner,
    LocalFrameFusion,
    build_visual_verification_shortlist,
    fuse_temporal_neighborhood_rankings,
    timeline_sparse_frame_ids,
)
from system_tai.refinement.models import (
    Phase3Candidate,
    RefinementConfig,
    SelectedVideoTimelineScoutConfig,
    SelectedVideoVisualVerifierConfig,
    VisualVerifierExecutionMode,
    VisualVerifierFailurePolicy,
)
from system_tai.refinement.video import (
    DecodedFrame,
    DecodeResult,
    RawVideoRecord,
    RawVideoRegistry,
    SparseDecodeRequest,
    VideoProbe,
)
from system_tai.refinement.visual_verifier import (
    VisualPredicateRequirement,
    VisualPredicateScore,
    VisualVerificationError,
    VisualVerificationFailure,
    VisualVerificationResult,
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
        query_vi="hành động mục tiêu tổng hợp",
        query_en="synthetic target action",
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
        visual_verifier_config=SelectedVideoVisualVerifierConfig(),
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


class _TailVerifier:
    identifiers = {"provider": "fake-structured-verifier"}

    def __init__(self, target_frame: int) -> None:
        self.target_frame = target_frame
        self.inputs = ()

    def verify(self, *, query_vi, query_en, candidates):
        assert query_vi == "mô tả tiếng Việt"
        assert query_en == "English description"
        self.inputs = tuple(candidates)
        return tuple(
            VisualVerificationResult(
                video_id=item.video_id,
                absolute_frame_id=item.absolute_frame_id,
                match_score=(1.0 if item.absolute_frame_id == self.target_frame else 0.1),
                requirement_coverage=(
                    1.0 if item.absolute_frame_id == self.target_frame else 0.2
                ),
                all_visible_requirements_satisfied=(
                    item.absolute_frame_id == self.target_frame
                ),
                predicates=(
                    VisualPredicateScore(
                        "synthetic conjunction",
                        1.0 if item.absolute_frame_id == self.target_frame else 0.1,
                        True,
                        "synthetic evidence",
                    ),
                ),
                summary="synthetic",
            )
            for item in candidates
        )


class _FailingVerifier:
    identifiers = {"provider": "failing"}

    def verify(self, **kwargs):
        del kwargs
        raise RuntimeError("synthetic verifier failure")


class _NonPromotableVerifier:
    identifiers = {"provider": "non-promotable"}
    last_failures = ()
    last_recovered_retries = ()
    last_predicate_contract = (
        VisualPredicateRequirement(
            "scene_conjunction_1",
            "exact query-derived scene conjunction",
        ),
    )

    def verify(self, *, query_vi, query_en, candidates):
        del query_vi, query_en
        return tuple(
            VisualVerificationResult(
                video_id=item.video_id,
                absolute_frame_id=item.absolute_frame_id,
                match_score=0.9,
                requirement_coverage=0.5,
                all_visible_requirements_satisfied=False,
                predicates=(
                    VisualPredicateScore(
                        "exact query-derived scene conjunction",
                        0.0,
                        True,
                        "image 1: visible but incomplete",
                        satisfied=False,
                        predicate_id="scene_conjunction_1",
                    ),
                ),
                summary="incomplete",
                contract_validated=True,
            )
            for item in candidates
        )


class _AllCandidateFailuresVerifier:
    identifiers = {"provider": "all-candidate-failures"}
    last_recovered_retries = ()
    last_predicate_contract = (
        VisualPredicateRequirement(
            "scene_conjunction_1",
            "exact query-derived scene conjunction",
        ),
    )

    def __init__(self) -> None:
        self.last_failures = ()

    def verify(self, *, query_vi, query_en, candidates):
        del query_vi, query_en
        self.last_failures = tuple(
            VisualVerificationFailure(
                item.video_id,
                item.absolute_frame_id,
                "primary schema mismatch",
                "retry schema mismatch",
            )
            for item in candidates
        )
        return ()


class _PartiallyFailingVerifier:
    identifiers = {"provider": "partial"}
    last_recovered_retries = ()

    def __init__(self) -> None:
        self.last_failures = ()
        self.target_frame = None

    def verify(self, *, query_vi, query_en, candidates):
        del query_vi, query_en
        failed = candidates[1]
        target = candidates[-1]
        self.target_frame = target.absolute_frame_id
        self.last_failures = (
            VisualVerificationFailure(
                failed.video_id,
                failed.absolute_frame_id,
                "primary",
                "retry",
            ),
        )
        return tuple(
            VisualVerificationResult(
                video_id=item.video_id,
                absolute_frame_id=item.absolute_frame_id,
                match_score=(1.0 if item is target else 0.1),
                requirement_coverage=(1.0 if item is target else 0.1),
                all_visible_requirements_satisfied=(item is target),
                predicates=(
                    VisualPredicateScore(
                        "synthetic conjunction",
                        1.0 if item is target else 0.1,
                        item is target,
                        "synthetic evidence",
                    ),
                ),
                summary="synthetic",
            )
            for item in candidates
            if item is not failed
        )


def test_coverage_shortlist_uses_temporal_midpoint_not_semantic_bin_edge() -> None:
    # Early frames dominate globally and frame 100 is the strongest semantic item
    # inside the final coverage bin, but frame 90 is nearer its midpoint.
    frame_order = (0, 10, 20, 30, 40, 50, 60, 70, 100, 90)
    ranked = tuple(
        LocalFrameFusion(frame_id, 1.0 / (index + 1), 1, index + 1, ())
        for index, frame_id in enumerate(frame_order)
    )

    shortlist = build_visual_verification_shortlist(
        ranked,
        total_frame_count=101,
        shortlist_size=6,
        coverage_bins=5,
    )

    assert 90 in {item.absolute_frame_id for item in shortlist}
    assert len(shortlist) == 6


def test_shortlist_preserves_semantic_leaders_and_late_midpoint_hypothesis() -> None:
    # This reproduces the geometry of a 6,932-frame timeline sampled every 25
    # frames. The final-bin semantic winner is near 6,375, while independent
    # midpoint coverage must retain the distinct late hypothesis near 6,650.
    frame_ids = tuple(range(0, 6932, 25)) + (6931,)
    semantic_order = (6375,) + tuple(
        frame_id for frame_id in frame_ids if frame_id != 6375
    )
    ranked = tuple(
        LocalFrameFusion(frame_id, 1.0 / (index + 1), 1, index + 1, ())
        for index, frame_id in enumerate(semantic_order)
    )

    shortlist = build_visual_verification_shortlist(
        ranked,
        total_frame_count=6932,
        shortlist_size=16,
        coverage_bins=12,
    )
    selected = {item.absolute_frame_id for item in shortlist}

    assert 6375 in selected
    assert 6650 in selected
    assert len(shortlist) == 16


def test_temporal_neighborhood_fuses_variant_ranks_without_raw_score_addition() -> None:
    def item(
        frame_id: int,
        first_rank: int,
        second_rank: int,
    ) -> LocalFrameFusion:
        provenance = (
            {
                "variant_id": "action",
                "variant_type": "english_translation",
                "language": "en",
                "weight": 1.0,
                "rank": first_rank,
                "cosine_score": 999.0,
            },
            {
                "variant_id": "attributes",
                "variant_type": "english_translation",
                "language": "en",
                "weight": 1.0,
                "rank": second_rank,
                "cosine_score": -999.0,
            },
        )
        return LocalFrameFusion(frame_id, 0.0, 2, min(first_rank, second_rank), provenance)

    ranked = (
        item(0, 2, 2),
        item(80, 1, 100),
        item(90, 100, 100),
        item(100, 100, 1),
    )

    fused = fuse_temporal_neighborhood_rankings(
        ranked,
        fps=1.0,
        evidence_window_seconds=10.0,
        rrf_constant=60.0,
    )

    assert fused[0].absolute_frame_id == 90
    assert fused[0].fusion_score == pytest.approx(2.0 / 61.0)
    assert {
        item["evidence_frame_id"] for item in fused[0].per_variant_provenance
    } == {80, 100}
    assert fused[0].fusion_score != pytest.approx(0.0)


def test_temporal_neighborhood_zero_window_preserves_exact_frame_ranking() -> None:
    ranked = (
        LocalFrameFusion(10, 0.2, 1, 1, ()),
        LocalFrameFusion(20, 0.1, 1, 2, ()),
    )

    assert fuse_temporal_neighborhood_rankings(
        ranked,
        fps=25.0,
        evidence_window_seconds=0.0,
        rrf_constant=60.0,
    ) == ranked


def test_visual_verifier_promotes_tail_without_target_video_or_frame_config(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "video-alpha.mp4"
    video_path.touch()
    decoder = _SyntheticDecoder()
    encoder = _SyntheticEncoder(peak_frame=0)
    verifier = _TailVerifier(target_frame=80)
    refiner = ExactFrameRefiner(
        raw_videos=RawVideoRegistry((RawVideoRecord("video-alpha", video_path),)),
        decoder=decoder,
        encoder=encoder,
        visual_verifier=verifier,
    )
    variant = QueryVariant(
        "scene",
        "English description",
        QueryLanguage.ENGLISH,
        QueryVariantType.ENGLISH_TRANSLATION,
        1.0,
    )

    outcome = refiner.scout_selected_video_timelines(
        query_id="Q",
        query_vi="mô tả tiếng Việt",
        query_en="English description",
        variants=(variant,),
        ranked_video_ids=("video-alpha",),
        rank_slots=(_slot(1, "video-alpha", 10),),
        config=SelectedVideoTimelineScoutConfig(
            enabled=True,
            max_videos=1,
            sample_stride_seconds=2.0,
            max_samples_per_video=6,
            max_regions_per_video=1,
            minimum_region_gap_seconds=1.0,
        ),
        visual_verifier_config=SelectedVideoVisualVerifierConfig(
            enabled=True,
            shortlist_per_video=4,
            coverage_bins=3,
            neighbor_sample_radius=1,
        ),
        refinement_config=RefinementConfig(),
        precomputed_text_embeddings=np.asarray([[1.0, 0.0]], dtype=np.float32),
        frame_embedding_cache={},
    )

    assert outcome.candidates[0].video_id == "video-alpha"
    assert outcome.candidates[0].rank == 1
    assert outcome.candidates[0].frame_id == 80
    assert outcome.trace["hard_coded_target"] is False
    assert outcome.trace["visual_verifier_enabled"] is True
    assert outcome.timings["timeline_visual_verified_candidate_count"] == 4
    assert 80 in {item.absolute_frame_id for item in verifier.inputs}


def test_visual_verifier_failure_falls_back_to_clip_with_explicit_warning(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "video-alpha.mp4"
    video_path.touch()
    refiner = ExactFrameRefiner(
        raw_videos=RawVideoRegistry((RawVideoRecord("video-alpha", video_path),)),
        decoder=_SyntheticDecoder(),
        encoder=_SyntheticEncoder(peak_frame=0),
        visual_verifier=_FailingVerifier(),
    )
    variant = QueryVariant(
        "scene",
        "English description",
        QueryLanguage.ENGLISH,
        QueryVariantType.ENGLISH_TRANSLATION,
        1.0,
    )

    outcome = refiner.scout_selected_video_timelines(
        query_id="Q",
        query_vi="mô tả tiếng Việt",
        query_en="English description",
        variants=(variant,),
        ranked_video_ids=("video-alpha",),
        rank_slots=(_slot(1, "video-alpha", 10),),
        config=SelectedVideoTimelineScoutConfig(
            enabled=True,
            max_videos=1,
            sample_stride_seconds=2.0,
            max_samples_per_video=6,
            max_regions_per_video=1,
            minimum_region_gap_seconds=1.0,
        ),
        visual_verifier_config=SelectedVideoVisualVerifierConfig(
            enabled=True,
            shortlist_per_video=4,
            coverage_bins=3,
        ),
        refinement_config=RefinementConfig(),
        precomputed_text_embeddings=np.asarray([[1.0, 0.0]], dtype=np.float32),
        frame_embedding_cache={},
    )

    assert outcome.candidates[0].frame_id == 0
    assert any("fallback to CLIP" in warning for warning in outcome.warnings)
    assert outcome.trace["videos"][0]["visual_verification"]["status"] == "FALLBACK_CLIP"


def test_non_promotable_shortlist_preserves_complete_pre_verifier_ranking(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "video-alpha.mp4"
    video_path.touch()
    refiner = ExactFrameRefiner(
        raw_videos=RawVideoRegistry((RawVideoRecord("video-alpha", video_path),)),
        decoder=_SyntheticDecoder(),
        encoder=_SyntheticEncoder(peak_frame=0),
        visual_verifier=_NonPromotableVerifier(),
    )
    variant = QueryVariant(
        "scene",
        "English description",
        QueryLanguage.ENGLISH,
        QueryVariantType.ENGLISH_TRANSLATION,
        1.0,
    )

    outcome = refiner.scout_selected_video_timelines(
        query_id="Q",
        query_vi="mô tả tiếng Việt",
        query_en="English description",
        variants=(variant,),
        ranked_video_ids=("video-alpha",),
        rank_slots=(
            _slot(1, "video-alpha", 10),
            _slot(2, "video-alpha", 20),
            _slot(3, "video-alpha", 30),
        ),
        config=SelectedVideoTimelineScoutConfig(
            enabled=True,
            max_videos=1,
            sample_stride_seconds=2.0,
            max_samples_per_video=6,
            max_regions_per_video=3,
            minimum_region_gap_seconds=1.0,
        ),
        visual_verifier_config=SelectedVideoVisualVerifierConfig(
            enabled=True,
            shortlist_per_video=2,
            coverage_bins=1,
        ),
        refinement_config=RefinementConfig(),
        precomputed_text_embeddings=np.asarray([[1.0, 0.0]], dtype=np.float32),
        frame_embedding_cache={},
    )

    assert tuple(item.frame_id for item in outcome.candidates) == (0, 20, 40)
    video_trace = outcome.trace["videos"][0]
    assert video_trace["selected_region_frame_ids"] == [0, 20, 40]
    assert len(video_trace["visual_verification"]["shortlist_frame_ids"]) == 2
    assert video_trace["visual_verification"]["strictly_promotable_candidate_count"] == 0


def test_all_candidate_failures_preserve_fixed_contract_fallback_trace(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "video-alpha.mp4"
    video_path.touch()
    verifier = _AllCandidateFailuresVerifier()
    refiner = ExactFrameRefiner(
        raw_videos=RawVideoRegistry((RawVideoRecord("video-alpha", video_path),)),
        decoder=_SyntheticDecoder(),
        encoder=_SyntheticEncoder(peak_frame=0),
        visual_verifier=verifier,
    )
    variant = QueryVariant(
        "scene",
        "English description",
        QueryLanguage.ENGLISH,
        QueryVariantType.ENGLISH_TRANSLATION,
        1.0,
    )

    outcome = refiner.scout_selected_video_timelines(
        query_id="Q",
        query_vi="mô tả tiếng Việt",
        query_en="English description",
        variants=(variant,),
        ranked_video_ids=("video-alpha",),
        rank_slots=(_slot(1, "video-alpha", 10),),
        config=SelectedVideoTimelineScoutConfig(
            enabled=True,
            max_videos=1,
            sample_stride_seconds=2.0,
            max_samples_per_video=6,
            max_regions_per_video=1,
            minimum_region_gap_seconds=1.0,
        ),
        visual_verifier_config=SelectedVideoVisualVerifierConfig(
            enabled=True,
            shortlist_per_video=4,
            coverage_bins=3,
        ),
        refinement_config=RefinementConfig(),
        precomputed_text_embeddings=np.asarray([[1.0, 0.0]], dtype=np.float32),
        frame_embedding_cache={},
    )

    trace = outcome.trace["videos"][0]["visual_verification"]
    assert trace["status"] == "FALLBACK_CLIP"
    assert trace["successful_candidate_count"] == 0
    assert trace["failed_candidate_count"] == 4
    assert trace["strictly_promotable_candidate_count"] == 0
    assert trace["predicate_contract"] == [
        {
            "id": "scene_conjunction_1",
            "requirement": "exact query-derived scene conjunction",
            "comparison": None,
            "expected_value": None,
        }
    ]
    assert len(trace["failures"]) == 4


def test_visual_verifier_candidate_failure_keeps_successes_and_later_candidate(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "video-alpha.mp4"
    video_path.touch()
    verifier = _PartiallyFailingVerifier()
    refiner = ExactFrameRefiner(
        raw_videos=RawVideoRegistry((RawVideoRecord("video-alpha", video_path),)),
        decoder=_SyntheticDecoder(),
        encoder=_SyntheticEncoder(peak_frame=0),
        visual_verifier=verifier,
    )
    variant = QueryVariant(
        "scene",
        "English description",
        QueryLanguage.ENGLISH,
        QueryVariantType.ENGLISH_TRANSLATION,
        1.0,
    )

    outcome = refiner.scout_selected_video_timelines(
        query_id="Q",
        query_vi="mô tả tiếng Việt",
        query_en="English description",
        variants=(variant,),
        ranked_video_ids=("video-alpha",),
        rank_slots=(_slot(1, "video-alpha", 10),),
        config=SelectedVideoTimelineScoutConfig(
            enabled=True,
            max_videos=1,
            sample_stride_seconds=2.0,
            max_samples_per_video=6,
            max_regions_per_video=1,
            minimum_region_gap_seconds=1.0,
        ),
        visual_verifier_config=SelectedVideoVisualVerifierConfig(
            enabled=True,
            shortlist_per_video=4,
            coverage_bins=3,
        ),
        refinement_config=RefinementConfig(),
        precomputed_text_embeddings=np.asarray([[1.0, 0.0]], dtype=np.float32),
        frame_embedding_cache={},
    )

    assert outcome.candidates[0].frame_id == verifier.target_frame
    verification = outcome.trace["videos"][0]["visual_verification"]
    assert verification["status"] == "PARTIAL_SUCCESS"
    assert verification["successful_candidate_count"] == 3
    assert verification["failed_candidate_count"] == 1
    assert any("candidate-local fallback" in item for item in outcome.warnings)


def test_visual_verifier_cpu_auto_profile_bounds_candidates_and_neighbor_images(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "video-alpha.mp4"
    video_path.touch()
    verifier = _TailVerifier(target_frame=100)
    refiner = ExactFrameRefiner(
        raw_videos=RawVideoRegistry((RawVideoRecord("video-alpha", video_path),)),
        decoder=_SyntheticDecoder(total_frames=201),
        encoder=_SyntheticEncoder(peak_frame=0),
        visual_verifier=verifier,
    )
    variant = QueryVariant(
        "scene",
        "English description",
        QueryLanguage.ENGLISH,
        QueryVariantType.ENGLISH_TRANSLATION,
        1.0,
    )

    outcome = refiner.scout_selected_video_timelines(
        query_id="Q",
        query_vi="mô tả tiếng Việt",
        query_en="English description",
        variants=(variant,),
        ranked_video_ids=("video-alpha",),
        rank_slots=(_slot(1, "video-alpha", 10),),
        config=SelectedVideoTimelineScoutConfig(
            enabled=True,
            max_videos=1,
            sample_stride_seconds=1.0,
            max_samples_per_video=21,
            max_regions_per_video=1,
            minimum_region_gap_seconds=1.0,
        ),
        visual_verifier_config=SelectedVideoVisualVerifierConfig(
            enabled=True,
            shortlist_per_video=16,
            coverage_bins=12,
            neighbor_sample_radius=1,
            max_new_tokens=384,
            device="cpu",
        ),
        refinement_config=RefinementConfig(),
        precomputed_text_embeddings=np.asarray([[1.0, 0.0]], dtype=np.float32),
        frame_embedding_cache={},
    )

    assert len(verifier.inputs) == 6
    assert all(len(item.images) == 1 for item in verifier.inputs)
    verification = outcome.trace["videos"][0]["visual_verification"]
    assert verification["execution"]["cpu_fast_profile_applied"] is True
    assert verification["execution"]["effective"]["shortlist_per_video"] == 6
    assert outcome.timings["timeline_visual_verified_candidate_count"] == 6
    assert "CPU-fast profile applied" in outcome.warnings[0]


def test_visual_verifier_fail_query_policy_propagates(tmp_path: Path) -> None:
    video_path = tmp_path / "video-alpha.mp4"
    video_path.touch()
    refiner = ExactFrameRefiner(
        raw_videos=RawVideoRegistry((RawVideoRecord("video-alpha", video_path),)),
        decoder=_SyntheticDecoder(),
        encoder=_SyntheticEncoder(peak_frame=0),
        visual_verifier=_FailingVerifier(),
    )
    variant = QueryVariant(
        "scene",
        "English description",
        QueryLanguage.ENGLISH,
        QueryVariantType.ENGLISH_TRANSLATION,
        1.0,
    )

    with pytest.raises(VisualVerificationError, match="synthetic verifier failure"):
        refiner.scout_selected_video_timelines(
            query_id="Q",
            query_vi="mô tả tiếng Việt",
            query_en="English description",
            variants=(variant,),
            ranked_video_ids=("video-alpha",),
            rank_slots=(_slot(1, "video-alpha", 10),),
            config=SelectedVideoTimelineScoutConfig(
                enabled=True,
                max_videos=1,
                sample_stride_seconds=2.0,
                max_samples_per_video=6,
                max_regions_per_video=1,
                minimum_region_gap_seconds=1.0,
            ),
            visual_verifier_config=SelectedVideoVisualVerifierConfig(
                enabled=True,
                shortlist_per_video=4,
                coverage_bins=3,
                failure_policy=VisualVerifierFailurePolicy.FAIL_QUERY,
            ),
            refinement_config=RefinementConfig(),
            precomputed_text_embeddings=np.asarray([[1.0, 0.0]], dtype=np.float32),
            frame_embedding_cache={},
        )


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

    verified = parser.parse_args(
        [
            "--enable-kis-semantic-video-first",
            "--enable-kis-selected-video-timeline-scout",
            "--enable-kis-visual-predicate-verifier",
            "--default-refine-top-n",
            "3",
        ]
    )
    verified_config = session_config_from_args(verified)
    assert verified_config.selected_video_visual_verifier_config.enabled is True

    full = parser.parse_args(
        [
            "--enable-kis-semantic-video-first",
            "--enable-kis-selected-video-timeline-scout",
            "--enable-kis-visual-predicate-verifier",
            "--kis-visual-verifier-execution-mode",
            "full",
            "--kis-visual-verifier-temporal-evidence-window-seconds",
            "8",
            "--default-refine-top-n",
            "3",
        ]
    )
    full_config = session_config_from_args(full)
    assert (
        full_config.selected_video_visual_verifier_config.execution_mode
        is VisualVerifierExecutionMode.FULL
    )
    assert (
        full_config.selected_video_visual_verifier_config
        .temporal_evidence_window_seconds
        == 8.0
    )
