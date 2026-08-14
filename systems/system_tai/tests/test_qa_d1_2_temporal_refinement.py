from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from system_tai.common.schemas import CandidateFrame, KISResult
from system_tai.kis.session_schema import QAQueryRequest
from system_tai.preliminary.schemas import QAPrediction
from system_tai.qa.grounding import (
    KEYFRAME_ANCHOR,
    QA_MULTI_SEED_TEMPORAL_REFINEMENT_V1,
    TEMPORAL_REFINED,
    QAVideoConditionedEvidenceConfig,
    select_temporal_seed_anchors,
)
from system_tai.qa.models import QAResult
from system_tai.qa.question_types import QuestionType
from system_tai.qa.runtime import QARuntimePipeline
from system_tai.refinement.engine import (
    QueryRefinementOutcome,
    SelectedRefinementOutcome,
)
from system_tai.refinement.models import (
    RefinedCandidate,
    RefinementConfig,
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
from system_tai.retrieval.multi_query import WeightedRRFRetriever
from system_tai.retrieval.video_evidence import (
    FullCorpusVideoMaximaOutcome,
    RestrictedFrameHit,
    VideoMaximumHit,
    VideoRestrictedSearchOutcome,
)


def _candidate(
    video_id: str,
    frame_id: int,
    *,
    rank: int,
    nomination_rank: int,
    local_rank: int,
) -> CandidateFrame:
    return CandidateFrame(
        video_id=video_id,
        frame_id=frame_id,
        clip_row=rank - 1,
        keyframe_order=rank,
        score=1.0 - rank / 100,
        rank=rank,
        source="qa",
        diagnostic_metadata={
            "video_nomination_rank": nomination_rank,
            "local_anchor_rank": local_rank,
            "localization_score": 1.0 - rank / 100,
            "source_localization_variant_ids": ["q::en"],
        },
    )


def test_temporal_config_is_opt_in_and_validates_bounded_dependencies() -> None:
    default = QAVideoConditionedEvidenceConfig()
    assert default.temporal_refinement_enabled is False
    assert default.keyframe_evidence_anchors_per_video == 1
    assert default.temporal_seed_anchors_per_video == 3
    assert default.temporal_refinement_video_cap == 32
    assert default.temporal_refinement_total_seed_cap == 96

    invalid = (
        {"keyframe_evidence_anchors_per_video": 2},
        {"temporal_refinement_enabled": True},
        {
            "enabled": True,
            "preserve_keyframe_evidence": True,
            "anchors_per_video": 2,
            "temporal_refinement_enabled": True,
        },
        {
            "enabled": True,
            "selected_video_cap": 2,
            "preserve_keyframe_evidence": True,
            "keyframe_evidence_video_cap": 2,
            "temporal_refinement_enabled": True,
            "temporal_refinement_video_cap": 3,
        },
        {
            "enabled": True,
            "selected_video_cap": 2,
            "anchors_per_video": 3,
            "preserve_keyframe_evidence": True,
            "keyframe_evidence_video_cap": 2,
            "temporal_refinement_enabled": True,
            "temporal_refinement_video_cap": 2,
            "temporal_refinement_total_seed_cap": 7,
        },
        {
            "enabled": True,
            "selected_video_cap": 2,
            "anchors_per_video": 2,
            "preserve_keyframe_evidence": True,
            "keyframe_evidence_video_cap": 2,
            "keyframe_evidence_anchors_per_video": 3,
        },
        {
            "enabled": True,
            "selected_video_cap": 40,
            "anchors_per_video": 3,
            "preserve_keyframe_evidence": True,
            "keyframe_evidence_video_cap": 40,
            "keyframe_evidence_anchors_per_video": 3,
        },
        {
            "enabled": True,
            "selected_video_cap": 2,
            "anchors_per_video": 3,
            "preserve_keyframe_evidence": True,
            "keyframe_evidence_video_cap": 2,
            "keyframe_evidence_anchors_per_video": 2,
            "temporal_refinement_enabled": True,
            "temporal_refinement_video_cap": 2,
            "temporal_refinement_total_seed_cap": 3,
        },
    )
    for kwargs in invalid:
        with pytest.raises(ValueError):
            QAVideoConditionedEvidenceConfig(**kwargs)


def test_seed_selector_is_deterministic_bounded_bidirectional_and_gt_free() -> None:
    candidates = (
        _candidate("V2", 900, rank=1, nomination_rank=2, local_rank=1),
        _candidate("V1", 100, rank=2, nomination_rank=1, local_rank=1),
        _candidate("V2", 1100, rank=3, nomination_rank=2, local_rank=2),
        _candidate("V1", 80, rank=4, nomination_rank=1, local_rank=2),
        _candidate("V2", 1000, rank=5, nomination_rank=2, local_rank=3),
        _candidate("V1", 120, rank=6, nomination_rank=1, local_rank=3),
        _candidate("V1", 130, rank=7, nomination_rank=1, local_rank=4),
    )
    selected = select_temporal_seed_anchors(
        candidates,
        anchors_per_video=3,
        video_cap=2,
        total_seed_cap=5,
    )
    assert [(item.video_id, item.frame_id) for item in selected] == [
        ("V1", 100),
        ("V2", 900),
        ("V1", 80),
        ("V2", 1100),
        ("V1", 120),
    ]
    assert select_temporal_seed_anchors(
        candidates,
        anchors_per_video=3,
        video_cap=2,
        total_seed_cap=5,
    ) == selected
    assert all("target" not in str(dict(item.diagnostic_metadata)) for item in selected)


class _Encoder:
    dimension = 4

    def __init__(self) -> None:
        self.text_calls: list[list[str]] = []
        self.image_calls: list[int] = []

    def encode_texts(self, texts):
        self.text_calls.append(list(texts))
        return np.asarray([[1.0, 0.0, 0.0, 0.0] for _ in texts], dtype=np.float32)

    def encode_images(self, images, *, batch_size=None):
        self.image_calls.append(len(images))
        return np.asarray([[1.0, 0.0, 0.0, 0.0] for _ in images], dtype=np.float32)


class _LegacyRetriever:
    def search_vector(self, **kwargs):
        raise AssertionError("legacy retrieval must not run")


class _Registry:
    def get(self, _video):
        return SimpleNamespace(descriptor=SimpleNamespace(row_count=3))


class _Searcher:
    registry = _Registry()

    def search_video_maxima(self, *, query_ids, query_vectors):
        return FullCorpusVideoMaximaOutcome(
            rankings={
                query_id: tuple(
                    VideoMaximumHit(
                        query_id=query_id,
                        video_id=video_id,
                        frame_id=rank * 100,
                        clip_row=rank - 1,
                        keyframe_order=rank,
                        cosine_score=1.0 - rank / 10,
                        rank=rank,
                    )
                    for rank, video_id in enumerate(("V1", "V2"), start=1)
                )
                for query_id in query_ids
            },
            physical_rows_scored=6,
            video_store_scan_count=2,
        )

    def search_selected_videos(
        self, *, video_ids, query_ids, query_vectors, per_query_result_cap
    ):
        starts = {"V1": 100, "V2": 200}
        return VideoRestrictedSearchOutcome(
            rankings={
                query_id: {
                    video_id: tuple(
                        RestrictedFrameHit(
                            video_id=video_id,
                            frame_id=starts[video_id] + offset,
                            clip_row=rank - 1,
                            keyframe_order=rank,
                            pts_time=(starts[video_id] + offset) / 30,
                            cosine_score=1.0 - rank / 10,
                            rank=rank,
                        )
                        for rank, offset in enumerate((0, -20, 20), start=1)
                    )
                    for video_id in video_ids
                }
                for query_id in query_ids
            },
            physical_rows_scored=6,
            video_store_scan_count=2,
        )


class _Decoder:
    backend_identifier = "fake"

    def __init__(self) -> None:
        self.decode_calls: list[tuple[str, tuple[int, ...]]] = []

    def probe(self, record):
        return VideoProbe(
            video_id=record.video_id,
            raw_video_path=record.raw_video_path,
            decoder_backend="fake",
            fps=30.0,
            total_frame_count=1000,
            width=10,
            height=10,
            duration_seconds=1000 / 30,
        )

    def decode(self, request: DecodeRequest):
        self.decode_calls.append((request.probe.video_id, request.frame_ids))
        return DecodeResult(
            frames=tuple(
                DecodedFrame(
                    absolute_frame_id=frame_id,
                    timestamp_seconds=frame_id / 30,
                    image=np.zeros((2, 2, 3), dtype=np.uint8),
                )
                for frame_id in request.frame_ids
            ),
            decoded_frame_count=len(request.frame_ids),
            video_open_seconds=0.0,
            decode_seconds=0.0,
            decoder_backend="fake",
            warnings=(),
        )


def _refined(candidate, *, success: bool, duplicate_frame: int | None = None):
    refined_frame = (
        duplicate_frame if duplicate_frame is not None else candidate.frame_id + 1
    )
    return RefinedCandidate(
        query_id=candidate.query_id,
        original_candidate_rank=candidate.rank,
        video_id=candidate.video_id,
        candidate_frame_id=candidate.frame_id,
        refined_frame_id=refined_frame if success else None,
        candidate_timestamp_seconds=candidate.frame_id / 30,
        refined_timestamp_seconds=refined_frame / 30 if success else None,
        fps=30.0,
        total_frame_count=1000,
        window_start_frame=(
            min(candidate.frame_id - 5, refined_frame) if success else None
        ),
        window_end_frame=(
            max(candidate.frame_id + 5, refined_frame) if success else None
        ),
        coarse_frame_ids=(candidate.frame_id,) if success else (),
        fine_frame_ids=(refined_frame,) if success else (),
        coarse_sample_count=1 if success else 0,
        fine_sample_count=1 if success else 0,
        decoded_frame_count=2 if success else 0,
        encoded_image_count=2 if success else 0,
        refinement_fusion_score=0.99 if success else None,
        variant_hit_count=1 if success else 0,
        best_individual_rank=1 if success else None,
        per_variant_provenance=(),
        decoder_backend="fake",
        raw_video_path=Path(f"{candidate.video_id}.mp4"),
        status=RefinementStatus.REFINED if success else RefinementStatus.FAILED,
        warnings=(),
        failure_reason=None if success else "fixture failure",
        original_retrieval_provenance=candidate.retrieval_provenance,
        timings={},
    )


class _SelectedRefiner:
    def __init__(self, outcomes=(True, False, True)) -> None:
        self.outcomes = outcomes
        self.selected_calls = []
        self.legacy_calls = 0

    def refine_query(self, *args, **kwargs):
        self.legacy_calls += 1
        raise AssertionError("legacy refine_query must not run in QA-D1.2")

    def refine_selected_candidates(self, **kwargs):
        self.selected_calls.append(kwargs)
        records = tuple(
            _refined(candidate, success=self.outcomes[index])
            for index, candidate in enumerate(kwargs["candidates"])
        )
        return SelectedRefinementOutcome(
            candidates=records,
            warnings=(),
            timings={
                "merged_temporal_region_count": 2,
                "decoded_frame_count": 12,
                "encoded_image_count": 8,
                "frame_embedding_cache_hit_count": 4,
                "frame_embedding_cache_miss_count": 8,
            },
        )


class _Engine:
    def __init__(self) -> None:
        self.evidence = ()

    def answer(self, query, evidence_candidates, image_embeddings=None, prompt_embeddings=None):
        self.evidence = tuple(evidence_candidates)
        first = self.evidence[0]
        return QAResult(
            query_id=query.query_id,
            question_type=QuestionType.COLOR,
            predictions=[
                QAPrediction(query.query_id, 1, first.video_id, first.frame_id, "red")
            ],
            diagnostics={"confidence_level": "TEST", "scores_by_rank": {1: 1.0}},
        )


def _pipeline(
    tmp_path: Path,
    *,
    refiner=None,
    temporal=True,
    engine=None,
    keyframe_anchors=1,
):
    videos = []
    for video_id in ("V1", "V2"):
        path = tmp_path / f"{video_id}.mp4"
        path.touch()
        videos.append(RawVideoRecord(video_id, path))
    encoder = _Encoder()
    decoder = _Decoder()
    legacy = _LegacyRetriever()
    resolved_refiner = refiner or _SelectedRefiner()
    resolved_engine = engine or _Engine()
    pipeline = QARuntimePipeline(
        exact_retriever=legacy,
        weighted_rrf=WeightedRRFRetriever(legacy),
        refiner=resolved_refiner,
        raw_video_registry=RawVideoRegistry(videos),
        decoder=decoder,
        shared_encoder=encoder,
        video_restricted_searcher=_Searcher(),
        video_conditioned_evidence_config=QAVideoConditionedEvidenceConfig(
            enabled=True,
            selected_video_cap=2,
            anchors_per_video=3,
            preserve_keyframe_evidence=True,
            keyframe_evidence_video_cap=2,
            keyframe_evidence_anchors_per_video=keyframe_anchors,
            temporal_refinement_enabled=temporal,
            temporal_seed_anchors_per_video=3,
            temporal_refinement_video_cap=2,
            temporal_refinement_total_seed_cap=3,
        ),
        qa_engine=resolved_engine,
    )
    return pipeline, encoder, resolved_refiner, decoder, resolved_engine


def _request(question="Chiếc xe có màu gì?"):
    return QAQueryRequest(
        request_id="d12",
        query_id="QA-D1.2",
        event_description="Một chiếc xe dừng lại.",
        event_description_en="A car stops.",
        question=question,
        question_en=None,
        include_vi_variant=False,
        output_top_k=10,
        refine_top_n=1,
    )


def test_runtime_uses_selected_set_once_keeps_fallback_and_unattempted_seeds(
    tmp_path: Path,
) -> None:
    pipeline, encoder, refiner, decoder, engine = _pipeline(tmp_path)
    result, _timings, diagnostics = pipeline.process_qa_query(
        _request(),
        RefinementConfig(
            top_candidates_to_refine=1,
            window_before_seconds=2,
            window_after_seconds=3,
            coarse_stride_frames=10,
            fine_radius_frames=4,
        ),
    )

    assert len(result.predictions) == 1
    assert refiner.legacy_calls == 0
    assert len(refiner.selected_calls) == 1
    call = refiner.selected_calls[0]
    assert [candidate.video_id for candidate in call["candidates"]] == [
        "V1",
        "V2",
        "V1",
    ]
    assert call["config"].window_before_seconds == 2
    assert call["config"].window_after_seconds == 3
    assert [(item.video_id, item.frame_id) for item in engine.evidence] == [
        ("V1", 101),
        ("V2", 200),
        ("V1", 81),
        ("V2", 180),
        ("V1", 120),
        ("V2", 220),
    ]
    assert [item.source_status for item in engine.evidence] == [
        TEMPORAL_REFINED,
        KEYFRAME_ANCHOR,
        TEMPORAL_REFINED,
        KEYFRAME_ANCHOR,
        KEYFRAME_ANCHOR,
        KEYFRAME_ANCHOR,
    ]
    assert diagnostics["qa_temporal_refinement_policy"] == (
        QA_MULTI_SEED_TEMPORAL_REFINEMENT_V1
    )
    assert diagnostics["temporal_seed_candidate_count"] == 6
    assert diagnostics["temporal_refinement_video_count"] == 2
    assert diagnostics["temporal_refinement_seed_count"] == 3
    assert diagnostics["temporal_refinement_success_count"] == 2
    assert diagnostics["temporal_refinement_failure_count"] == 1
    assert diagnostics["temporal_refinement_fallback_count"] == 4
    assert diagnostics["temporal_refined_evidence_count"] == 2
    assert diagnostics["temporal_merged_region_count"] == 2
    assert diagnostics["temporal_decoded_frame_count"] == 12
    assert diagnostics["temporal_encoded_image_count"] == 8
    assert diagnostics["temporal_embedding_cache_hit_count"] == 4
    assert diagnostics["temporal_embedding_cache_miss_count"] == 8
    assert encoder.text_calls[0] == ["A car stops."]
    assert diagnostics["qa_localization_policy"] == "EN_ONLY"
    assert diagnostics["localization_variant_count"] == 1
    assert diagnostics["localization_variant_languages"] == ["en"]
    assert diagnostics["answer_routing_question_language"] == "vi"
    assert all(item.provenance["candidate_frame_id"] >= 0 for item in engine.evidence)
    assert decoder.decode_calls


def test_duplicate_refined_identity_deduplicates_deterministically(tmp_path: Path) -> None:
    class _DuplicateRefiner(_SelectedRefiner):
        def refine_selected_candidates(self, **kwargs):
            self.selected_calls.append(kwargs)
            records = tuple(
                _refined(candidate, success=True, duplicate_frame=150)
                for candidate in kwargs["candidates"]
            )
            return SelectedRefinementOutcome(records, (), {})

    pipeline, _encoder, _refiner, _decoder, engine = _pipeline(
        tmp_path, refiner=_DuplicateRefiner()
    )
    pipeline.process_qa_query(_request())
    assert sum(
        item.video_id == "V1" and item.frame_id == 150 for item in engine.evidence
    ) == 1
    identities = [(item.video_id, item.frame_id) for item in engine.evidence]
    assert len(identities) == len(set(identities))


def test_unsupported_query_skips_multi_seed_refinement_and_image_decode(
    tmp_path: Path,
) -> None:
    pipeline, encoder, refiner, decoder, _engine = _pipeline(tmp_path)
    pipeline.candidate_provider = SimpleNamespace(get_candidates=lambda _question_type: ())
    result, _timings, diagnostics = pipeline.process_qa_query(
        _request("Sau đó chuyện gì xảy ra?")
    )
    assert result.unsupported_reason == "UNSUPPORTED_NO_PROVIDER"
    assert diagnostics["temporal_seed_candidate_count"] == 6
    assert diagnostics["temporal_refinement_seed_count"] == 0
    assert diagnostics["temporal_refinement_fallback_count"] == 6
    assert refiner.selected_calls == []
    assert decoder.decode_calls == []
    assert encoder.image_calls == []


def test_feature_off_uses_legacy_refine_query_path(tmp_path: Path) -> None:
    class _LegacyRefiner(_SelectedRefiner):
        def refine_query(self, query, config, precomputed_text_embeddings=None):
            self.legacy_calls += 1
            records = tuple(
                _refined(candidate, success=candidate.rank == 1)
                for candidate in query.candidates
            )
            return QueryRefinementOutcome(
                query.query_id,
                KISResult(query.query_id, ()),
                records,
                (),
                {},
            )

        def refine_selected_candidates(self, **kwargs):
            raise AssertionError("selected-set path must stay off")

    pipeline, _encoder, refiner, _decoder, _engine = _pipeline(
        tmp_path, refiner=_LegacyRefiner(), temporal=False
    )
    _result, _timings, diagnostics = pipeline.process_qa_query(_request())
    assert refiner.legacy_calls == 1
    assert diagnostics["qa_temporal_refinement_policy"] == "DISABLED"
    assert diagnostics["keyframe_evidence_count"] == 2


def test_keyframe_only_multi_anchor_retains_three_per_video_without_temporal_refine(
    tmp_path: Path,
) -> None:
    class _LegacyRefiner(_SelectedRefiner):
        def refine_query(self, query, config, precomputed_text_embeddings=None):
            self.legacy_calls += 1
            records = tuple(
                _refined(candidate, success=candidate.rank == 1)
                for candidate in query.candidates
            )
            return QueryRefinementOutcome(
                query.query_id,
                KISResult(query.query_id, ()),
                records,
                (),
                {},
            )

        def refine_selected_candidates(self, **kwargs):
            raise AssertionError("selected-set temporal refinement must stay off")

    pipeline, _encoder, refiner, _decoder, engine = _pipeline(
        tmp_path,
        refiner=_LegacyRefiner(),
        temporal=False,
        keyframe_anchors=3,
    )
    _result, _timings, diagnostics = pipeline.process_qa_query(_request())

    assert refiner.legacy_calls == 1
    assert refiner.selected_calls == []
    assert diagnostics["qa_temporal_refinement_policy"] == "DISABLED"
    assert diagnostics["temporal_seed_candidate_count"] == 0
    assert diagnostics["keyframe_evidence_count"] == 6
    assert [(item.video_id, item.provenance["local_anchor_rank"]) for item in engine.evidence] == [
        ("V1", 1),
        ("V2", 1),
        ("V1", 2),
        ("V2", 2),
        ("V1", 3),
        ("V2", 3),
    ]
