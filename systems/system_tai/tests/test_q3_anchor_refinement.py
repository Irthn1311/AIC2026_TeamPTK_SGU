from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from system_tai.common.schemas import CandidateFrame, KISResult
from system_tai.refinement.engine import ExactFrameRefiner
from system_tai.refinement.models import (
    Phase3Candidate,
    Q3AnchorRefinementConfig,
    RefinementConfig,
    RefinementStatus,
)
from system_tai.refinement.q3_anchor import (
    integrate_q3_anchor_refinements,
    select_q3_anchor_candidates,
)
from system_tai.refinement.video import (
    DecodedFrame,
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
from system_tai.retrieval.video_restricted import VIDEO_CONDITIONED_KEYFRAME_DIVERSITY


def _phase3(
    rank: int,
    video_id: str,
    frame_id: int,
    cosine: float | None,
) -> Phase3Candidate:
    provenance = {"fusion_score": 1.0 / (60 + rank), "candidate_source": "weighted_rrf"}
    if cosine is not None:
        provenance.update(
            {
                "q3_policy": VIDEO_CONDITIONED_KEYFRAME_DIVERSITY,
                "q3_restricted_cosine_score": cosine,
                "q3_restricted_rank": rank,
                "candidate_source": "video_conditioned_keyframe_diversity",
            }
        )
    return Phase3Candidate("Q", rank, video_id, frame_id, 0.5, provenance)


def _baseline(candidates: tuple[Phase3Candidate, ...]) -> KISResult:
    return KISResult(
        "Q",
        tuple(
            CandidateFrame(
                video_id=item.video_id,
                frame_id=item.frame_id,
                clip_row=item.rank,
                keyframe_order=item.rank,
                score=item.retrieval_score,
                rank=item.rank,
                source="raw_video_exact_frame_refinement",
                diagnostic_metadata={"original_candidate_rank": item.rank},
            )
            for item in candidates
        ),
    )


def test_q3_anchor_selection_is_bounded_deterministic_and_video_diverse() -> None:
    candidates = (
        _phase3(1, "P", 1, None),
        _phase3(4, "A", 40, 0.99),
        _phase3(5, "A", 50, 0.98),
        _phase3(6, "B", 60, 0.97),
        _phase3(7, "C", 70, 0.96),
        _phase3(8, "D", 80, None),
        _phase3(9, "B", 90, 0.95),
    )
    first = select_q3_anchor_candidates(
        candidates,
        protected_prefix_rank=3,
        max_extra_q3_anchors=5,
    )
    second = select_q3_anchor_candidates(
        candidates,
        protected_prefix_rank=3,
        max_extra_q3_anchors=5,
    )

    assert first == second
    assert [item.rank for item in first.eligible] == [4, 5, 6, 7, 9]
    assert [item.rank for item in first.selected] == [4, 6, 7, 5, 9]
    assert len(first.selected) <= Q3AnchorRefinementConfig().max_extra_q3_anchors
    assert all(item.rank > 3 for item in first.selected)
    assert all(item.retrieval_provenance["q3_policy"] for item in first.selected)

    over_budget = candidates + tuple(
        _phase3(rank, f"V{rank}", rank * 10, 0.9 - rank / 1000)
        for rank in range(10, 14)
    )
    capped = select_q3_anchor_candidates(
        over_budget,
        protected_prefix_rank=3,
        max_extra_q3_anchors=6,
    )
    assert len(capped.eligible) > 6
    assert len(capped.selected) == 6


def test_same_slot_integration_preserves_top3_sequences_and_collision_rows() -> None:
    candidates = tuple(
        _phase3(rank, "A" if rank != 3 else "B", rank * 10, None)
        for rank in range(1, 6)
    )
    baseline = _baseline(candidates)
    unchanged = ExactFrameRefiner._unchanged_candidate
    refined_rank4 = replace(
        unchanged(candidates[3], status=RefinementStatus.KEEP_ORIGINAL, warning="x"),
        status=RefinementStatus.REFINED,
        refined_frame_id=44,
    )
    collision_rank5 = replace(
        unchanged(candidates[4], status=RefinementStatus.KEEP_ORIGINAL, warning="x"),
        status=RefinementStatus.REFINED,
        refined_frame_id=10,
    )

    integrated = integrate_q3_anchor_refinements(
        baseline,
        (refined_rank4, collision_rank5),
    )

    assert integrated.result.ranked_candidates[:3] == baseline.ranked_candidates[:3]
    assert integrated.result.ranked_candidates[3].frame_id == 44
    assert integrated.result.ranked_candidates[4] == baseline.ranked_candidates[4]
    assert [item.rank for item in integrated.result.ranked_candidates] == [1, 2, 3, 4, 5]
    assert [item.video_id for item in integrated.result.ranked_candidates] == [
        "A",
        "A",
        "B",
        "A",
        "A",
    ]
    assert integrated.refined_count == 1
    assert integrated.collision_skip_count == 1
    assert len(
        {(item.video_id, item.frame_id) for item in integrated.result.ranked_candidates}
    ) == 5


class _Encoder:
    dimension = 2
    identifiers = {"model": "fake", "device": "cpu"}

    def __init__(self) -> None:
        self.encoded_images = 0

    def encode_images(self, images, *, batch_size):
        del batch_size
        self.encoded_images += len(images)
        return np.asarray([[1.0, int(image) / 100.0] for image in images], dtype=np.float32)


class _Decoder:
    backend_identifier = "fake-absolute"

    def __init__(self) -> None:
        self.probe_calls = 0
        self.decode_requests = []

    def probe(self, record):
        self.probe_calls += 1
        return VideoProbe(record.video_id, record.raw_video_path, "fake", 10, 200, 8, 8, 20)

    def decode(self, request):
        self.decode_requests.append(request)
        frames = tuple(
            DecodedFrame(frame_id, frame_id / 10, frame_id)
            for frame_id in request.frame_ids
        )
        return DecodeResult(frames, len(frames), 0.0, 0.0, "fake", ())


def test_grouped_selected_refinement_merges_windows_and_reuses_request_cache(
    tmp_path: Path,
) -> None:
    raw_path = tmp_path / "A.mp4"
    raw_path.touch()
    decoder = _Decoder()
    encoder = _Encoder()
    refiner = ExactFrameRefiner(
        raw_videos=RawVideoRegistry((RawVideoRecord("A", raw_path),)),
        decoder=decoder,
        encoder=encoder,
    )
    candidates = (_phase3(4, "A", 50, 0.9), _phase3(5, "A", 55, 0.8))
    cache = {("A", 50): np.asarray([1.0, 0.5], dtype=np.float32)}
    variants = (
        QueryVariant(
            "en",
            "target",
            QueryLanguage.ENGLISH,
            QueryVariantType.ENGLISH_TRANSLATION,
            1.0,
        ),
    )
    outcome = refiner.refine_selected_candidates(
        query_id="Q",
        variants=variants,
        candidates=candidates,
        config=RefinementConfig(
            top_candidates_to_refine=3,
            window_before_seconds=1,
            window_after_seconds=1,
            coarse_stride_frames=5,
            coarse_top_n=1,
            fine_radius_frames=2,
            fine_stride_frames=1,
            max_decoded_frames_per_candidate=30,
        ),
        precomputed_text_embeddings=np.asarray([[1.0, 0.0]], dtype=np.float32),
        frame_embedding_cache=cache,
    )

    assert len(outcome.candidates) == 2
    assert all(item.status is RefinementStatus.REFINED for item in outcome.candidates)
    assert decoder.probe_calls == 1
    assert len(decoder.decode_requests) == 2
    assert outcome.timings["merged_temporal_region_count"] == 1
    assert outcome.timings["frame_embedding_cache_hit_count"] >= 1
    assert encoder.encoded_images == outcome.timings["frame_embedding_cache_miss_count"]
    assert outcome.timings["unique_q3_coarse_frame_count"] < sum(
        item.coarse_sample_count for item in outcome.candidates
    )
    assert "ground_truth" not in json.dumps(outcome.timings).casefold()


def test_missing_raw_selected_anchor_keeps_original(tmp_path: Path) -> None:
    refiner = ExactFrameRefiner(
        raw_videos=RawVideoRegistry((RawVideoRecord("A", None),)),
        decoder=_Decoder(),
        encoder=_Encoder(),
    )
    candidate = _phase3(4, "A", 50, 0.9)
    variant = QueryVariant(
        "en", "target", QueryLanguage.ENGLISH, QueryVariantType.ENGLISH_TRANSLATION, 1.0
    )
    outcome = refiner.refine_selected_candidates(
        query_id="Q",
        variants=(variant,),
        candidates=(candidate,),
        config=RefinementConfig(),
        precomputed_text_embeddings=np.asarray([[1.0, 0.0]], dtype=np.float32),
        frame_embedding_cache={},
    )

    assert outcome.candidates[0].status is RefinementStatus.KEEP_ORIGINAL
    assert outcome.candidates[0].refined_frame_id == candidate.frame_id
