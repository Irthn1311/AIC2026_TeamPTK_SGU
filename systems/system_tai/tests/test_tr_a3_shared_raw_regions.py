from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from system_tai.kis.session_schema import TRAKEQueryRequest
from system_tai.refinement.engine import (
    ExactFrameRefiner,
    SelectedRefinementOutcome,
    SharedRefinementBatchOutcome,
    SharedRefinementGroup,
)
from system_tai.refinement.models import (
    Phase3Candidate,
    RefinementConfig,
    RefinementQuery,
    RefinementStatus,
    SharedRawRegionRefinementConfig,
)
from system_tai.refinement.video import (
    DecodedFrame,
    DecodeResult,
    RawVideoError,
    RawVideoRecord,
    RawVideoRegistry,
    VideoProbe,
)
from system_tai.retrieval.multi_query import (
    QueryLanguage,
    QueryVariant,
    QueryVariantType,
)
from tests.test_preliminary_p0c2_trake_runtime import make_test_pipeline


class _Encoder:
    dimension = 2
    identifiers = {"model": "fake", "device": "cpu"}

    def __init__(self) -> None:
        self.image_ids: list[int] = []

    def encode_images(self, images, *, batch_size):
        del batch_size
        self.image_ids.extend(int(image) for image in images)
        return np.asarray(
            [[1.0, int(image) / 200.0] for image in images],
            dtype=np.float32,
        )

    def encode_texts(self, texts):
        return np.asarray(
            [[1.0, 0.0] if "early" in text else [0.0, 1.0] for text in texts],
            dtype=np.float32,
        )


class _Decoder:
    backend_identifier = "fake-sequential"

    def __init__(self, *, fail_video: str | None = None) -> None:
        self.fail_video = fail_video
        self.requests = []
        self.probe_calls: list[str] = []

    def probe(self, record):
        self.probe_calls.append(record.video_id)
        return VideoProbe(
            record.video_id,
            record.raw_video_path,
            self.backend_identifier,
            10.0,
            220,
            8,
            8,
            22.0,
        )

    def decode(self, request):
        self.requests.append(request)
        if request.probe.video_id == self.fail_video:
            raise RawVideoError("synthetic decode failure")
        frames = tuple(
            DecodedFrame(frame_id, frame_id / request.probe.fps, frame_id)
            for frame_id in request.frame_ids
        )
        return DecodeResult(
            frames=frames,
            decoded_frame_count=request.frame_ids[-1] - request.frame_ids[0] + 1,
            video_open_seconds=0.0,
            decode_seconds=0.0,
            decoder_backend=self.backend_identifier,
            warnings=(),
        )


def _variant(group: int) -> QueryVariant:
    return QueryVariant(
        variant_id=f"v{group}",
        text="early" if group == 0 else "late",
        language=QueryLanguage.ENGLISH,
        variant_type=QueryVariantType.ENGLISH_TRANSLATION,
        weight=1.0,
    )


def _candidate(group: int, rank: int, video_id: str, frame_id: int) -> Phase3Candidate:
    return Phase3Candidate(
        query_id=f"Q::e{group}",
        rank=rank,
        video_id=video_id,
        frame_id=frame_id,
        retrieval_score=0.5,
        retrieval_provenance={"event_index": group},
    )


def _config(*, max_frames: int = 30) -> RefinementConfig:
    return RefinementConfig(
        top_candidates_to_refine=3,
        output_top_k=3,
        window_before_seconds=1.0,
        window_after_seconds=1.0,
        coarse_stride_frames=5,
        coarse_top_n=1,
        fine_radius_frames=2,
        fine_stride_frames=1,
        image_batch_size=16,
        max_decoded_frames_per_candidate=max_frames,
    )


def _refiner(tmp_path: Path, decoder: _Decoder, encoder: _Encoder) -> ExactFrameRefiner:
    records = []
    for video_id in ("A", "B"):
        raw = tmp_path / f"{video_id}.mp4"
        raw.touch(exist_ok=True)
        records.append(RawVideoRecord(video_id, raw))
    return ExactFrameRefiner(
        raw_videos=RawVideoRegistry(tuple(records)),
        decoder=decoder,
        encoder=encoder,
    )


def _groups(candidates_by_group, encoder: _Encoder) -> tuple[SharedRefinementGroup, ...]:
    groups = []
    for group_index, candidates in enumerate(candidates_by_group):
        variant = _variant(group_index)
        groups.append(
            SharedRefinementGroup(
                query_id=f"Q::e{group_index}",
                variants=(variant,),
                candidates=tuple(candidates),
                text_embeddings=encoder.encode_texts([variant.text]),
            )
        )
    return tuple(groups)


def _assert_semantic_candidate_equal(left, right) -> None:
    assert left.video_id == right.video_id
    assert left.candidate_frame_id == right.candidate_frame_id
    assert left.refined_frame_id == right.refined_frame_id
    assert left.status is right.status
    assert left.refinement_fusion_score == pytest.approx(right.refinement_fusion_score)
    assert left.coarse_frame_ids == right.coarse_frame_ids
    assert left.fine_frame_ids == right.fine_frame_ids
    assert left.per_variant_provenance == right.per_variant_provenance


def test_default_is_disabled_and_legacy_semantics_match_shared(tmp_path: Path) -> None:
    assert SharedRawRegionRefinementConfig().enabled is False
    candidate_groups = (
        (_candidate(0, 1, "A", 50),),
        (_candidate(1, 1, "A", 55),),
    )
    legacy_encoder = _Encoder()
    legacy_decoder = _Decoder()
    legacy_refiner = _refiner(tmp_path, legacy_decoder, legacy_encoder)
    legacy = []
    for group in _groups(candidate_groups, legacy_encoder):
        legacy.append(
            legacy_refiner.refine_query(
                RefinementQuery(group.query_id, group.variants, group.candidates),
                _config(),
                precomputed_text_embeddings=group.text_embeddings,
                frame_embedding_cache={},
            )
        )

    shared_encoder = _Encoder()
    shared_decoder = _Decoder()
    shared = _refiner(tmp_path, shared_decoder, shared_encoder).refine_shared_candidate_groups(
        _groups(candidate_groups, shared_encoder),
        config=_config(),
    )

    for legacy_group, shared_group in zip(legacy, shared.groups):
        _assert_semantic_candidate_equal(
            legacy_group.candidates[0],
            shared_group.candidates[0],
        )
    assert len(legacy_decoder.requests) == 4
    assert len(shared_decoder.requests) < len(legacy_decoder.requests)
    assert shared.timings["raw_decode_request_count_before_estimate"] == 4
    assert shared.timings["raw_decode_request_count_actual"] == len(
        shared_decoder.requests
    )


def test_shared_coarse_and_fine_cache_encode_each_unique_frame_once(tmp_path: Path) -> None:
    encoder = _Encoder()
    decoder = _Decoder()
    shared_variant = _variant(0)
    shared_text = encoder.encode_texts([shared_variant.text])
    groups = (
        SharedRefinementGroup(
            "Q::e0",
            (shared_variant,),
            (_candidate(0, 1, "A", 50), _candidate(0, 2, "A", 55)),
            shared_text,
        ),
        SharedRefinementGroup(
            "Q::e1",
            (shared_variant,),
            (_candidate(1, 1, "A", 50),),
            shared_text,
        ),
    )
    outcome = _refiner(tmp_path, decoder, encoder).refine_shared_candidate_groups(
        groups,
        config=_config(),
    )

    assert len(encoder.image_ids) == len(set(encoder.image_ids))
    assert outcome.timings["coarse_requested_frame_count"] > outcome.timings[
        "coarse_unique_requested_frame_count"
    ]
    assert outcome.timings["fine_requested_frame_count"] > outcome.timings[
        "fine_unique_requested_frame_count"
    ]
    assert outcome.timings["frame_cache_hit_count"] > 0
    assert outcome.timings["frame_embedding_cache_hit_count"] > 0


def test_nonoverlapping_regions_and_different_videos_never_merge(tmp_path: Path) -> None:
    encoder = _Encoder()
    decoder = _Decoder()
    groups = _groups(
        (
            (
                _candidate(0, 1, "A", 20),
                _candidate(0, 2, "A", 150),
                _candidate(0, 3, "B", 20),
            ),
        ),
        encoder,
    )
    outcome = _refiner(tmp_path, decoder, encoder).refine_shared_candidate_groups(
        groups,
        config=_config(),
    )

    assert len(outcome.groups[0].candidates) == 3
    assert len(decoder.requests) == 6
    assert all(
        len({request.probe.video_id}) == 1
        for request in decoder.requests
    )


def test_max_decode_guard_splits_touching_regions(tmp_path: Path) -> None:
    encoder = _Encoder()
    decoder = _Decoder()
    groups = _groups(
        (
            (_candidate(0, 1, "A", 50),),
            (_candidate(1, 1, "A", 70),),
        ),
        encoder,
    )
    _refiner(tmp_path, decoder, encoder).refine_shared_candidate_groups(
        groups,
        config=_config(max_frames=21),
    )

    assert all(
        request.frame_ids[-1] - request.frame_ids[0] + 1 <= 21
        for request in decoder.requests
    )
    assert len(decoder.requests) >= 4


def test_decode_failure_matches_legacy_fallback_status(tmp_path: Path) -> None:
    candidate_groups = ((_candidate(0, 1, "A", 50),),)
    legacy_encoder = _Encoder()
    legacy_refiner = _refiner(tmp_path, _Decoder(fail_video="A"), legacy_encoder)
    group = _groups(candidate_groups, legacy_encoder)[0]
    legacy = legacy_refiner.refine_query(
        RefinementQuery(group.query_id, group.variants, group.candidates),
        _config(),
        precomputed_text_embeddings=group.text_embeddings,
        frame_embedding_cache={},
    )

    shared_encoder = _Encoder()
    shared = _refiner(
        tmp_path,
        _Decoder(fail_video="A"),
        shared_encoder,
    ).refine_shared_candidate_groups(
        _groups(candidate_groups, shared_encoder),
        config=_config(),
    )

    assert legacy.candidates[0].status is RefinementStatus.KEEP_ORIGINAL
    assert shared.groups[0].candidates[0].status is legacy.candidates[0].status
    assert shared.groups[0].candidates[0].refined_frame_id == 50
    assert shared.groups[0].candidates[0].failure_reason == legacy.candidates[0].failure_reason


def test_trake_opt_in_preserves_final_path_and_disabled_uses_legacy() -> None:
    event_candidates = [
        [{"rank": 1, "video_id": "V1", "frame_id": 100}],
        [{"rank": 1, "video_id": "V1", "frame_id": 200}],
    ]
    proposals = {(0, "V1", 100): 101, (1, "V1", 200): 201}
    request = TRAKEQueryRequest(
        request_id="req-tr-a3",
        query_id="TR-A3",
        events=({"description": "event1"}, {"description": "event2"}),
        refine_top_n=1,
    )
    legacy_pipeline, _, _, _, legacy_refiner, _ = make_test_pipeline(
        cands_per_event=event_candidates,
        proposal_map=proposals,
    )
    legacy_result, _, legacy_diag = legacy_pipeline.process_trake_query(
        request,
        refinement_config=RefinementConfig(),
    )
    assert len(legacy_refiner.refine_query_calls) == 2
    assert legacy_diag["shared_raw_region_refinement"][
        "shared_raw_region_refinement_enabled"
    ] is False

    shared_pipeline, _, _, _, shared_refiner, _ = make_test_pipeline(
        cands_per_event=event_candidates,
        proposal_map=proposals,
    )
    shared_call_count = 0

    def _shared(groups, *, config, frame_embedding_cache):
        nonlocal shared_call_count
        shared_call_count += 1
        outcomes = []
        for group in groups:
            outcome = shared_refiner.refine_query(
                RefinementQuery(group.query_id, group.variants, group.candidates),
                config,
                precomputed_text_embeddings=group.text_embeddings,
                frame_embedding_cache=frame_embedding_cache,
            )
            outcomes.append(
                SelectedRefinementOutcome(outcome.candidates, outcome.warnings, {})
            )
        return SharedRefinementBatchOutcome(
            groups=tuple(outcomes),
            timings={"shared_raw_region_refinement_enabled": True},
        )

    shared_refiner.refine_shared_candidate_groups = _shared
    shared_result, _, shared_diag = shared_pipeline.process_trake_query(
        request,
        refinement_config=RefinementConfig(),
        shared_raw_region_config=SharedRawRegionRefinementConfig(enabled=True),
    )

    assert shared_call_count == 1
    assert shared_result.predictions == legacy_result.predictions
    assert shared_diag["path_diagnostics"] == legacy_diag["path_diagnostics"]
    assert shared_diag["shared_raw_region_refinement"][
        "shared_raw_region_refinement_enabled"
    ] is True
