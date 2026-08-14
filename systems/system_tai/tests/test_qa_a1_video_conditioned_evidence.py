from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from system_tai.common.schemas import (
    CandidateFrame,
    FrameMappingRecord,
    KISResult,
    VideoFeatureStore,
)
from system_tai.features.btc_clip_store import (
    FeatureStoreRegistry,
    LoadedVideoFeatureStore,
)
from system_tai.kis.session_schema import QAQueryRequest, SessionConfig
from system_tai.preliminary.schemas import QAPrediction
from system_tai.qa.grounding import (
    QA_VIDEO_CONDITIONED_EVIDENCE_V1,
    QAVideoConditionedEvidenceConfig,
    build_qa_grounding_result,
    nominate_qa_videos,
    select_primary_keyframe_anchors,
)
from system_tai.qa.models import QAResult
from system_tai.qa.ocr_provider import (
    OCRAnswerProvider,
    OCRAnswerProviderConfig,
    OCRDetection,
)
from system_tai.qa.question_types import QuestionType
from system_tai.qa.runtime import QARuntimePipeline
from system_tai.refinement.engine import QueryRefinementOutcome
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
from system_tai.retrieval.multi_query import (
    QueryLanguage,
    QueryVariant,
    QueryVariantType,
    WeightedRRFRetriever,
)
from system_tai.retrieval.video_evidence import (
    FullCorpusVideoMaximaOutcome,
    RestrictedFrameHit,
    VideoMaximumHit,
    VideoRestrictedFeatureSearcher,
    VideoRestrictedSearchOutcome,
)


def _variant(identifier: str, text: str, language: QueryLanguage) -> QueryVariant:
    return QueryVariant(
        variant_id=identifier,
        text=text,
        language=language,
        variant_type=(
            QueryVariantType.VIETNAMESE_DIRECT
            if language is QueryLanguage.VIETNAMESE
            else QueryVariantType.ENGLISH_TRANSLATION
        ),
        weight=1.0,
    )


def _maximum(query_id: str, video_id: str, rank: int, score: float) -> VideoMaximumHit:
    return VideoMaximumHit(
        query_id=query_id,
        video_id=video_id,
        frame_id=rank * 10,
        clip_row=rank - 1,
        keyframe_order=rank,
        cosine_score=score,
        rank=rank,
    )


def _restricted(video_id: str, frame_id: int, rank: int, score: float) -> RestrictedFrameHit:
    return RestrictedFrameHit(
        video_id=video_id,
        frame_id=frame_id,
        clip_row=rank - 1,
        keyframe_order=rank,
        pts_time=frame_id / 30.0,
        cosine_score=score,
        rank=rank,
    )


def test_config_default_disabled_and_strict_validation() -> None:
    config = QAVideoConditionedEvidenceConfig()
    assert config.enabled is False
    assert config.preserve_keyframe_evidence is False
    assert config.keyframe_evidence_video_cap == 32
    assert config.keyframe_evidence_anchors_per_video == 1
    assert SessionConfig().qa_video_conditioned_evidence_config == config
    for kwargs in (
        {"enabled": 1},
        {"selected_video_cap": 0},
        {"anchors_per_video": 0},
        {"video_rrf_constant": float("nan")},
        {"video_rrf_constant": 0.0},
        {"keyframe_evidence_video_cap": 0},
        {"preserve_keyframe_evidence": True},
        {
            "enabled": True,
            "selected_video_cap": 2,
            "preserve_keyframe_evidence": True,
            "keyframe_evidence_video_cap": 3,
        },
    ):
        with pytest.raises(ValueError):
            QAVideoConditionedEvidenceConfig(**kwargs)


def test_video_rrf_is_deterministic_and_cap_is_enforced() -> None:
    vi = _variant("q::vi", "su kien", QueryLanguage.VIETNAMESE)
    en = _variant("q::en", "event", QueryLanguage.ENGLISH)
    maxima = FullCorpusVideoMaximaOutcome(
        rankings={
            vi.variant_id: (
                _maximum(vi.variant_id, "V2", 1, 0.9),
                _maximum(vi.variant_id, "V1", 2, 0.8),
                _maximum(vi.variant_id, "V3", 3, 0.7),
            ),
            en.variant_id: (
                _maximum(en.variant_id, "V1", 1, 0.9),
                _maximum(en.variant_id, "V2", 2, 0.8),
                _maximum(en.variant_id, "V3", 3, 0.7),
            ),
        },
        physical_rows_scored=30,
        video_store_scan_count=3,
    )
    config = QAVideoConditionedEvidenceConfig(enabled=True, selected_video_cap=2)
    first = nominate_qa_videos(variants=(vi, en), maxima=maxima, config=config)
    second = nominate_qa_videos(variants=(vi, en), maxima=maxima, config=config)

    assert first == second
    assert [item.video_id for item in first] == ["V1", "V2"]
    assert [item.nomination_rank for item in first] == [1, 2]
    assert first[0].video_rrf_score == pytest.approx(
        1 / 61 + 1 / 62
    )


def test_anchor_bounds_cross_video_order_unique_frames_and_contiguous_ranks() -> None:
    variant = _variant("q::vi", "su kien", QueryLanguage.VIETNAMESE)
    maxima = FullCorpusVideoMaximaOutcome(
        rankings={
            variant.variant_id: (
                _maximum(variant.variant_id, "V1", 1, 0.9),
                _maximum(variant.variant_id, "V2", 2, 0.8),
            )
        },
        physical_rows_scored=8,
        video_store_scan_count=2,
    )
    config = QAVideoConditionedEvidenceConfig(
        enabled=True,
        selected_video_cap=2,
        anchors_per_video=2,
    )
    nominations = nominate_qa_videos(
        variants=(variant,),
        maxima=maxima,
        config=config,
    )
    restricted = VideoRestrictedSearchOutcome(
        rankings={
            variant.variant_id: {
                "V1": (
                    _restricted("V1", 100, 1, 0.95),
                    _restricted("V1", 110, 2, 0.90),
                    _restricted("V1", 110, 3, 0.89),
                ),
                "V2": (
                    _restricted("V2", 200, 1, 0.94),
                    _restricted("V2", 210, 2, 0.88),
                    _restricted("V2", 220, 3, 0.80),
                ),
            }
        },
        physical_rows_scored=8,
        video_store_scan_count=2,
    )
    result = build_qa_grounding_result(
        query_id="q",
        variants=(variant,),
        nominations=nominations,
        restricted=restricted,
        weighted_rrf=WeightedRRFRetriever(object()),
        config=config,
        output_top_k=4,
    )

    assert [(item.video_id, item.frame_id) for item in result.ranked_candidates] == [
        ("V1", 100),
        ("V2", 200),
        ("V1", 110),
        ("V2", 210),
    ]
    assert [item.rank for item in result.ranked_candidates] == [1, 2, 3, 4]
    assert len({(item.video_id, item.frame_id) for item in result.ranked_candidates}) == 4
    assert result.ranked_candidates[0].score == pytest.approx(0.95)
    assert [
        (item.diagnostic_metadata or {})["local_anchor_rank"]
        for item in result.ranked_candidates
    ] == [1, 1, 2, 2]
    assert all(item.source == QA_VIDEO_CONDITIONED_EVIDENCE_V1 for item in result.ranked_candidates)

    primary = select_primary_keyframe_anchors(
        result.ranked_candidates,
        video_cap=1,
    )
    assert [(item.video_id, item.frame_id) for item in primary] == [("V1", 100)]


class _FakeEncoder:
    dimension = 4
    identifiers = {"device": "cpu", "model": "fake"}

    def __init__(self) -> None:
        self.text_calls: list[list[str]] = []
        self.image_calls: list[int] = []

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        self.text_calls.append(list(texts))
        result = np.zeros((len(texts), 4), dtype=np.float32)
        result[:, 0] = 1.0
        return result

    def encode_images(self, images: list[np.ndarray]) -> np.ndarray:
        self.image_calls.append(len(images))
        result = np.zeros((len(images), 4), dtype=np.float32)
        result[:, 0] = 1.0
        return result


class _LegacyRetriever:
    def __init__(self) -> None:
        self.calls = 0

    def search_vector(self, *, query_id: str, query_vector: np.ndarray, top_k: int):
        self.calls += 1
        return KISResult(
            query_id=query_id,
            ranked_candidates=(
                CandidateFrame("V1", 100, 0, 1, 0.9, 1, "legacy"),
            ),
        )


class _FakeRegistry:
    embedding_dimension = 4
    total_rows = 3

    def get(self, video_id: str):
        if video_id != "V1":
            raise KeyError(video_id)
        return SimpleNamespace(descriptor=SimpleNamespace(row_count=3))


class _FakeVideoSearcher:
    def __init__(self) -> None:
        self.registry = _FakeRegistry()
        self.maxima_calls: list[tuple[str, ...]] = []
        self.restricted_calls: list[tuple[str, ...]] = []

    def search_video_maxima(self, *, query_ids, query_vectors):
        self.maxima_calls.append(tuple(query_ids))
        return FullCorpusVideoMaximaOutcome(
            rankings={
                query_id: (_maximum(query_id, "V1", 1, 0.9),)
                for query_id in query_ids
            },
            physical_rows_scored=3,
            video_store_scan_count=1,
        )

    def search_selected_videos(
        self, *, video_ids, query_ids, query_vectors, per_query_result_cap
    ):
        self.restricted_calls.append(tuple(video_ids))
        return VideoRestrictedSearchOutcome(
            rankings={
                query_id: {
                    "V1": (
                        _restricted("V1", 100, 1, 0.9),
                        _restricted("V1", 110, 2, 0.8),
                    )
                }
                for query_id in query_ids
            },
            physical_rows_scored=3,
            video_store_scan_count=1,
        )


class _EchoRefiner:
    def __init__(self) -> None:
        self.calls = 0
        self.last_query = None

    def refine_query(self, query, config, precomputed_text_embeddings=None):
        self.calls += 1
        self.last_query = query
        input_candidate = query.candidates[0]
        refined = RefinedCandidate(
            query_id=query.query_id,
            original_candidate_rank=1,
            video_id=input_candidate.video_id,
            candidate_frame_id=input_candidate.frame_id,
            refined_frame_id=input_candidate.frame_id,
            candidate_timestamp_seconds=input_candidate.frame_id / 30.0,
            refined_timestamp_seconds=input_candidate.frame_id / 30.0,
            fps=30.0,
            total_frame_count=1000,
            window_start_frame=max(0, input_candidate.frame_id - 10),
            window_end_frame=input_candidate.frame_id + 10,
            coarse_frame_ids=(input_candidate.frame_id,),
            fine_frame_ids=(input_candidate.frame_id,),
            coarse_sample_count=1,
            fine_sample_count=1,
            decoded_frame_count=2,
            encoded_image_count=2,
            refinement_fusion_score=0.99,
            variant_hit_count=1,
            best_individual_rank=1,
            per_variant_provenance=(),
            decoder_backend="fake",
            raw_video_path=Path("V1.mp4"),
            status=RefinementStatus.REFINED,
            warnings=(),
            failure_reason=None,
            original_retrieval_provenance=input_candidate.retrieval_provenance,
            timings={},
        )
        result = KISResult(
            query_id=query.query_id,
            ranked_candidates=(
                CandidateFrame(
                    video_id="V1",
                    frame_id=input_candidate.frame_id,
                    clip_row=0,
                    keyframe_order=1,
                    score=0.99,
                    rank=1,
                    source="refined",
                ),
            ),
        )
        return QueryRefinementOutcome(query.query_id, result, (refined,), (), {})


class _FakeDecoder:
    backend_identifier = "fake"

    def __init__(self) -> None:
        self.decode_calls: list[tuple[str, int]] = []

    def probe(self, record: RawVideoRecord) -> VideoProbe:
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

    def decode(self, request: DecodeRequest) -> DecodeResult:
        frame_id = request.frame_ids[0]
        self.decode_calls.append((request.probe.video_id, frame_id))
        return DecodeResult(
            frames=(
                DecodedFrame(
                    absolute_frame_id=frame_id,
                    timestamp_seconds=frame_id / 30.0,
                    image=np.zeros((2, 2, 3), dtype=np.uint8),
                ),
            ),
            decoded_frame_count=1,
            video_open_seconds=0.0,
            decode_seconds=0.0,
            decoder_backend="fake",
            warnings=(),
        )


class _RecordingQAEngine:
    def __init__(self) -> None:
        self.evidence = ()

    def answer(self, query, evidence_candidates, image_embeddings=None, prompt_embeddings=None):
        self.evidence = tuple(evidence_candidates)
        candidate = self.evidence[0]
        return QAResult(
            query_id=query.query_id,
            question_type=QuestionType.COLOR,
            predictions=[
                QAPrediction(
                    query_id=query.query_id,
                    rank=candidate.rank,
                    video_id=candidate.video_id,
                    frame_id=candidate.frame_id,
                    answer="red",
                )
            ],
            diagnostics={"confidence_level": "BASELINE", "scores_by_rank": {1: 1.0}},
        )


class _NoProvider:
    def get_candidates(self, question_type: QuestionType):
        return ()


def _pipeline(tmp_path: Path, *, qa_engine=None, ocr_answer_provider=None):
    video_path = tmp_path / "V1.mp4"
    video_path.touch()
    encoder = _FakeEncoder()
    legacy = _LegacyRetriever()
    searcher = _FakeVideoSearcher()
    refiner = _EchoRefiner()
    pipeline = QARuntimePipeline(
        exact_retriever=legacy,
        weighted_rrf=WeightedRRFRetriever(legacy),
        refiner=refiner,
        raw_video_registry=RawVideoRegistry([RawVideoRecord("V1", video_path)]),
        decoder=_FakeDecoder(),
        shared_encoder=encoder,
        video_restricted_searcher=searcher,
        video_conditioned_evidence_config=QAVideoConditionedEvidenceConfig(
            enabled=True,
            selected_video_cap=1,
            anchors_per_video=1,
        ),
        qa_engine=qa_engine,
        ocr_answer_provider=ocr_answer_provider,
    )
    return pipeline, encoder, legacy, searcher, refiner


class _MultiRegistry:
    embedding_dimension = 4
    total_rows = 6

    def get(self, video_id: str):
        if video_id not in {"V1", "V2", "V3"}:
            raise KeyError(video_id)
        return SimpleNamespace(descriptor=SimpleNamespace(row_count=2))


class _MultiVideoSearcher:
    registry = _MultiRegistry()

    def search_video_maxima(self, *, query_ids, query_vectors):
        return FullCorpusVideoMaximaOutcome(
            rankings={
                query_id: tuple(
                    _maximum(query_id, video_id, rank, 1.0 - rank / 10)
                    for rank, video_id in enumerate(("V1", "V2", "V3"), start=1)
                )
                for query_id in query_ids
            },
            physical_rows_scored=6,
            video_store_scan_count=3,
        )

    def search_selected_videos(
        self, *, video_ids, query_ids, query_vectors, per_query_result_cap
    ):
        starts = {"V1": 100, "V2": 200, "V3": 300}
        return VideoRestrictedSearchOutcome(
            rankings={
                query_id: {
                    video_id: (
                        _restricted(video_id, starts[video_id], 1, 0.9),
                        _restricted(video_id, starts[video_id] + 10, 2, 0.8),
                    )
                    for video_id in video_ids
                }
                for query_id in query_ids
            },
            physical_rows_scored=6,
            video_store_scan_count=3,
        )


class _BudgetRefiner:
    def __init__(self, *, first_status: RefinementStatus) -> None:
        self.first_status = first_status
        self.configs = []

    def refine_query(self, query, config, precomputed_text_embeddings=None):
        self.configs.append(config)
        records = []
        for candidate in query.candidates:
            selected = candidate.rank == 1
            status = self.first_status if selected else RefinementStatus.NOT_REFINED
            succeeded = status is RefinementStatus.REFINED
            records.append(
                RefinedCandidate(
                    query_id=query.query_id,
                    original_candidate_rank=candidate.rank,
                    video_id=candidate.video_id,
                    candidate_frame_id=candidate.frame_id,
                    refined_frame_id=candidate.frame_id + 1 if succeeded else None,
                    candidate_timestamp_seconds=candidate.frame_id / 30.0,
                    refined_timestamp_seconds=(
                        (candidate.frame_id + 1) / 30.0 if succeeded else None
                    ),
                    fps=30.0 if selected else None,
                    total_frame_count=1000 if selected else None,
                    window_start_frame=candidate.frame_id if succeeded else None,
                    window_end_frame=candidate.frame_id + 1 if succeeded else None,
                    coarse_frame_ids=(candidate.frame_id,) if succeeded else (),
                    fine_frame_ids=(candidate.frame_id + 1,) if succeeded else (),
                    coarse_sample_count=1 if succeeded else 0,
                    fine_sample_count=1 if succeeded else 0,
                    decoded_frame_count=1 if succeeded else 0,
                    encoded_image_count=1 if succeeded else 0,
                    refinement_fusion_score=0.99 if succeeded else None,
                    variant_hit_count=1 if succeeded else 0,
                    best_individual_rank=1 if succeeded else None,
                    per_variant_provenance=(),
                    decoder_backend="fake" if selected else None,
                    raw_video_path=Path(f"{candidate.video_id}.mp4") if selected else None,
                    status=status,
                    warnings=(),
                    failure_reason=(
                        "fixture failure"
                        if status is RefinementStatus.FAILED
                        else None
                    ),
                    original_retrieval_provenance=candidate.retrieval_provenance,
                    timings={},
                )
            )
        return QueryRefinementOutcome(
            query.query_id,
            KISResult(query.query_id, ()),
            tuple(records),
            (),
            {},
        )


def _keyframe_bank_pipeline(
    tmp_path: Path,
    *,
    first_status: RefinementStatus = RefinementStatus.REFINED,
    qa_engine=None,
):
    videos = []
    for video_id in ("V1", "V2", "V3"):
        video_path = tmp_path / f"{video_id}.mp4"
        video_path.touch()
        videos.append(RawVideoRecord(video_id, video_path))
    encoder = _FakeEncoder()
    refiner = _BudgetRefiner(first_status=first_status)
    decoder = _FakeDecoder()
    legacy = _LegacyRetriever()
    pipeline = QARuntimePipeline(
        exact_retriever=legacy,
        weighted_rrf=WeightedRRFRetriever(legacy),
        refiner=refiner,
        raw_video_registry=RawVideoRegistry(videos),
        decoder=decoder,
        shared_encoder=encoder,
        video_restricted_searcher=_MultiVideoSearcher(),
        video_conditioned_evidence_config=QAVideoConditionedEvidenceConfig(
            enabled=True,
            selected_video_cap=3,
            anchors_per_video=2,
            preserve_keyframe_evidence=True,
            keyframe_evidence_video_cap=3,
        ),
        qa_engine=qa_engine,
    )
    return pipeline, encoder, refiner, decoder


def test_keyframe_bank_preserves_diverse_videos_and_refinement_is_upgrade(
    tmp_path: Path,
) -> None:
    engine = _RecordingQAEngine()
    pipeline, _encoder, refiner, decoder = _keyframe_bank_pipeline(
        tmp_path,
        qa_engine=engine,
    )
    _result, _timings, diagnostics = pipeline.process_qa_query(
        QAQueryRequest(
            "bank",
            "bank",
            "Một chiếc xe dừng.",
            "Chiếc xe có màu gì?",
            event_description_en="A car stops.",
            include_vi_variant=False,
            output_top_k=10,
            refine_top_n=1,
        )
    )

    assert refiner.configs[0].top_candidates_to_refine == 1
    assert [(item.video_id, item.frame_id) for item in engine.evidence] == [
        ("V1", 101),
        ("V2", 200),
        ("V3", 300),
    ]
    assert [item.source_status for item in engine.evidence] == [
        "RAW_REFINED",
        "KEYFRAME_ANCHOR",
        "KEYFRAME_ANCHOR",
    ]
    assert diagnostics["keyframe_evidence_count"] == 3
    assert diagnostics["raw_refined_evidence_count"] == 1
    assert diagnostics["generic_evidence_bank_count"] == 3
    assert diagnostics["provider_evidence_count"] == 3
    assert diagnostics["refinement_selected_count"] == 1
    assert diagnostics["refinement_success_count"] == 1
    assert decoder.decode_calls == [("V1", 101), ("V2", 200), ("V3", 300)]
    assert len({(item.video_id, item.frame_id) for item in engine.evidence}) == 3


def test_keyframe_bank_failed_refinement_falls_back_without_extra_decode(
    tmp_path: Path,
) -> None:
    engine = _RecordingQAEngine()
    pipeline, _encoder, _refiner, _decoder = _keyframe_bank_pipeline(
        tmp_path,
        first_status=RefinementStatus.FAILED,
        qa_engine=engine,
    )
    _result, _timings, diagnostics = pipeline.process_qa_query(
        QAQueryRequest(
            "fallback",
            "fallback",
            "Một chiếc xe dừng.",
            "Chiếc xe có màu gì?",
            output_top_k=10,
            refine_top_n=1,
        )
    )
    assert [(item.video_id, item.frame_id) for item in engine.evidence] == [
        ("V1", 100),
        ("V2", 200),
        ("V3", 300),
    ]
    assert all(item.source_status == "KEYFRAME_ANCHOR" for item in engine.evidence)
    assert diagnostics["refinement_success_count"] == 0
    assert diagnostics["generic_evidence_bank_candidates"][0][
        "fallback_to_keyframe"
    ] is True


def test_keyframe_bank_unsupported_query_does_not_decode_images(tmp_path: Path) -> None:
    pipeline, encoder, refiner, decoder = _keyframe_bank_pipeline(tmp_path)
    pipeline.candidate_provider = _NoProvider()
    result, _timings, diagnostics = pipeline.process_qa_query(
        QAQueryRequest(
            "unsupported-bank",
            "unsupported-bank",
            "Một người bước vào.",
            "Sau đó chuyện gì xảy ra?",
            output_top_k=10,
            refine_top_n=1,
        )
    )
    assert result.unsupported_reason == "UNSUPPORTED_NO_PROVIDER"
    assert diagnostics["generic_evidence_bank_count"] == 3
    assert diagnostics["provider_evidence_count"] == 0
    assert refiner.configs == []
    assert decoder.decode_calls == []
    assert encoder.image_calls == []


class _RuntimeOCRBackend:
    identifiers = {
        "backend": "fake_runtime_ocr",
        "device": "cpu",
        "model_download": False,
    }

    def recognize(self, image: np.ndarray) -> tuple[OCRDetection, ...]:
        assert image.shape == (2, 2, 3)
        return (OCRDetection("ĐỒNG BẰNG SÔNG HỒNG", 96.0),)


def test_qa_a3_runtime_routes_ocr_to_bounded_decoded_evidence(tmp_path: Path) -> None:
    ocr_provider = OCRAnswerProvider(
        backend=_RuntimeOCRBackend(),
        config=OCRAnswerProviderConfig(enabled=True, evidence_frame_budget=1),
        clock=lambda: 0.0,
    )
    pipeline, encoder, legacy, searcher, refiner = _pipeline(
        tmp_path,
        ocr_answer_provider=ocr_provider,
    )
    result, timings, diagnostics = pipeline.process_qa_query(
        QAQueryRequest(
            request_id="qa-a3",
            query_id="qa-a3",
            event_description="Một bài giảng địa lý",
            question="Dòng chữ trên màn hình là gì?",
            output_top_k=10,
            refine_top_n=1,
        )
    )

    assert result.question_type is QuestionType.OCR
    assert [(item.video_id, item.frame_id, item.answer) for item in result.predictions] == [
        ("V1", 100, "ĐỒNG BẰNG SÔNG HỒNG")
    ]
    assert diagnostics["ocr_provider_enabled"] is True
    assert diagnostics["ocr_frames_requested"] == 1
    assert diagnostics["ocr_frames_processed"] == 1
    assert diagnostics["decoded_frame_count"] == 1
    assert diagnostics["encoded_image_count"] == 0
    assert timings.ocr_decode_seconds >= 0.0
    assert diagnostics["ocr_decode_seconds"] == timings.ocr_decode_seconds
    assert legacy.calls == 0
    assert len(searcher.maxima_calls) == 1
    assert refiner.calls == 1
    assert encoder.image_calls == []


def test_qa_a1_passes_batched_numpy_encoder_output_to_real_video_search(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "V1.mp4"
    video_path.touch()
    mappings = tuple(
        FrameMappingRecord(
            clip_row=index,
            keyframe_order=index + 1,
            frame_id=frame_id,
            pts_time=frame_id / 30.0,
            fps=30.0,
        )
        for index, frame_id in enumerate((100, 110, 120))
    )
    store = LoadedVideoFeatureStore(
        descriptor=VideoFeatureStore(
            video_id="V1",
            mapping_csv_path=tmp_path / "V1.csv",
            clip_npy_path=tmp_path / "V1.npy",
            row_count=3,
            embedding_dimension=4,
            normalized=True,
        ),
        matrix=np.asarray(
            [[1.0, 0.0, 0.0, 0.0], [0.8, 0.2, 0.0, 0.0], [0.6, 0.4, 0.0, 0.0]],
            dtype=np.float32,
        ),
        mappings=mappings,
    )
    registry = FeatureStoreRegistry([store])
    encoder = _FakeEncoder()
    legacy = _LegacyRetriever()
    pipeline = QARuntimePipeline(
        exact_retriever=legacy,
        weighted_rrf=WeightedRRFRetriever(legacy),
        refiner=_EchoRefiner(),
        raw_video_registry=RawVideoRegistry([RawVideoRecord("V1", video_path)]),
        decoder=_FakeDecoder(),
        shared_encoder=encoder,
        video_restricted_searcher=VideoRestrictedFeatureSearcher(
            registry,
            chunk_size=2,
        ),
        video_conditioned_evidence_config=QAVideoConditionedEvidenceConfig(
            enabled=True,
            selected_video_cap=1,
            anchors_per_video=1,
        ),
    )

    result, _timings, diagnostics = pipeline.process_qa_query(
        QAQueryRequest(
            request_id="numpy-matrix",
            query_id="qa-numpy-matrix",
            event_description="A person enters a shop",
            question="What happens after that?",
            output_top_k=10,
            refine_top_n=1,
        )
    )

    assert isinstance(encoder.encode_texts(["contract probe"]), np.ndarray)
    assert result.predictions == []
    assert diagnostics["grounding_candidate_count"] == 1
    assert diagnostics["selected_video_ids"] == ["V1"]
    assert legacy.calls == 0


def test_enabled_unsupported_query_runs_grounding_without_inventing_answer(
    tmp_path: Path,
) -> None:
    pipeline, encoder, legacy, searcher, refiner = _pipeline(tmp_path)
    request = QAQueryRequest(
        request_id="r1",
        query_id="q1",
        event_description="A person enters a shop",
        event_description_en="A person enters a shop",
        question="What happens after that?",
        question_en="What happens after that?",
        output_top_k=10,
        refine_top_n=1,
    )
    result, _timings, diagnostics = pipeline.process_qa_query(
        request,
        RefinementConfig(),
    )

    assert result.predictions == []
    assert result.unsupported_reason == "UNSUPPORTED_NO_PROVIDER"
    assert diagnostics["question_capability_reason"] == "QUESTION_PATTERN_UNSUPPORTED"
    assert diagnostics["grounding_candidate_count"] == 1
    assert diagnostics["decoded_frame_count"] == 1
    assert diagnostics["encoded_image_count"] == 0
    assert legacy.calls == 0
    assert len(searcher.maxima_calls) == 1
    assert len(searcher.restricted_calls) == 1
    assert refiner.calls == 1
    assert encoder.text_calls[0] == [
        "A person enters a shop",
        "A person enters a shop",
    ]
    serialized = json.dumps(diagnostics)
    assert "target_video" not in serialized
    assert "ground_truth" not in serialized
    assert "accepted_answer" not in serialized


def test_supported_answer_engine_receives_refined_grounding_evidence(
    tmp_path: Path,
) -> None:
    engine = _RecordingQAEngine()
    pipeline, encoder, legacy, _searcher, _refiner = _pipeline(
        tmp_path,
        qa_engine=engine,
    )
    request = QAQueryRequest(
        request_id="r2",
        query_id="q2",
        event_description="A car stops near a shop",
        question="What color is the car?",
        output_top_k=10,
        refine_top_n=1,
    )
    result, _timings, diagnostics = pipeline.process_qa_query(request)

    assert len(result.predictions) == 1
    assert len(engine.evidence) == 1
    evidence = engine.evidence[0]
    assert evidence.frame_id == 100
    assert evidence.provenance["video_nomination_rank"] == 1
    assert evidence.provenance["local_anchor_rank"] == 1
    assert diagnostics["question_supported_by_current_provider"] is True
    assert legacy.calls == 0
    assert encoder.text_calls[0] == ["A car stops near a shop"]
    assert "What color is the car?" not in encoder.text_calls[0]


def test_qa_d1_en_only_changes_localization_only_and_records_diagnostics(
    tmp_path: Path,
) -> None:
    engine = _RecordingQAEngine()
    pipeline, encoder, legacy, _searcher, _refiner = _pipeline(
        tmp_path,
        qa_engine=engine,
    )
    request = QAQueryRequest(
        request_id="qa-d1",
        query_id="qa-d1",
        event_description="Một chiếc xe dừng lại.",
        question="Chiếc xe có màu gì?",
        event_description_en="A car stops.",
        question_en=None,
        include_vi_variant=False,
        output_top_k=10,
        refine_top_n=1,
    )
    result, _timings, diagnostics = pipeline.process_qa_query(request)

    assert len(result.predictions) == 1
    assert encoder.text_calls[0] == ["A car stops."]
    assert diagnostics["question_type"] == "COLOR"
    assert diagnostics["qa_localization_policy"] == "EN_ONLY"
    assert diagnostics["include_vi_variant"] is False
    assert diagnostics["localization_variant_count"] == 1
    assert diagnostics["localization_variant_languages"] == ["en"]
    assert diagnostics["localization_variant_types"] == ["english_translation"]
    assert diagnostics["localization_text_provenance"] == [
        "explicit_event_description_en"
    ]
    assert diagnostics["answer_routing_question_language"] == "vi"
    assert legacy.calls == 0


def test_supported_pattern_without_provider_is_distinguished(tmp_path: Path) -> None:
    pipeline, _encoder, _legacy, _searcher, _refiner = _pipeline(tmp_path)
    pipeline.candidate_provider = _NoProvider()
    result, _timings, diagnostics = pipeline.process_qa_query(
        QAQueryRequest("r-provider", "q-provider", "A car stops", "What color is it?")
    )
    assert result.predictions == []
    assert result.unsupported_reason == "UNSUPPORTED_NO_PROVIDER"
    assert diagnostics["question_capability_reason"] == "SUPPORTED_PATTERN_NO_PROVIDER"


def test_disabled_unsupported_path_preserves_legacy_early_return(tmp_path: Path) -> None:
    encoder = _FakeEncoder()
    retriever = _LegacyRetriever()
    pipeline = QARuntimePipeline(
        exact_retriever=retriever,
        weighted_rrf=WeightedRRFRetriever(retriever),
        refiner=_EchoRefiner(),
        raw_video_registry=RawVideoRegistry(
            [RawVideoRecord("V1", tmp_path / "missing.mp4")]
        ),
        decoder=_FakeDecoder(),
        shared_encoder=encoder,
    )
    result, _timings, diagnostics = pipeline.process_qa_query(
        QAQueryRequest("r3", "q3", "event only", "What happened next?")
    )
    assert result.predictions == []
    assert diagnostics["qa_grounding_enabled"] is False
    assert retriever.calls == 0
    assert encoder.text_calls == []


def test_disabled_supported_path_uses_legacy_global_retrieval(tmp_path: Path) -> None:
    video_path = tmp_path / "V1.mp4"
    video_path.touch()
    encoder = _FakeEncoder()
    retriever = _LegacyRetriever()
    engine = _RecordingQAEngine()
    pipeline = QARuntimePipeline(
        exact_retriever=retriever,
        weighted_rrf=WeightedRRFRetriever(retriever),
        refiner=_EchoRefiner(),
        raw_video_registry=RawVideoRegistry([RawVideoRecord("V1", video_path)]),
        decoder=_FakeDecoder(),
        shared_encoder=encoder,
        qa_engine=engine,
    )
    result, _timings, diagnostics = pipeline.process_qa_query(
        QAQueryRequest("r4", "q4", "A car stops", "What color is the car?")
    )
    assert retriever.calls == 1
    assert result.predictions[0].frame_id == 100
    assert diagnostics["qa_grounding_enabled"] is False
    assert "grounding_candidates" not in diagnostics


def test_l21_runner_flag_is_dev_qa_only() -> None:
    runner_path = Path(__file__).parents[1] / "scripts" / "l21_150_run_baseline.py"
    spec = importlib.util.spec_from_file_location("qa_a1_l21_runner", runner_path)
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runner
    spec.loader.exec_module(runner)

    args = runner.build_parser().parse_args(
        [
            "--benchmark",
            "benchmark.json",
            "--reuse-manifest",
            "manifest.json",
            "--output-dir",
            "out",
            "--qa-video-conditioned-evidence",
            "--qa-ocr-evidence",
        ]
    )
    assert args.qa_video_conditioned_evidence is True
    assert args.qa_ocr_evidence is True
    enabled = QAVideoConditionedEvidenceConfig(enabled=True)
    with pytest.raises(ValueError, match="QA-A1 is restricted"):
        runner.run_l21_150_baseline(
            object(),
            object(),
            Path("out"),
            experiment_id="qa-a1",
            split="holdout",
            task="qa",
            top_k=100,
            refine_top_n=3,
            resume=False,
            fail_fast=True,
            benchmark_sha256="0" * 64,
            manifest_sha256=None,
            gt_policy="proposed",
            qa_video_conditioned_evidence_config=enabled,
        )
    with pytest.raises(ValueError, match="QA-A3 requires QA DEV"):
        runner.run_l21_150_baseline(
            object(),
            object(),
            Path("out"),
            experiment_id="qa-a3",
            split="holdout",
            task="qa",
            top_k=100,
            refine_top_n=3,
            resume=False,
            fail_fast=True,
            benchmark_sha256="0" * 64,
            manifest_sha256=None,
            gt_policy="proposed",
            qa_ocr_answer_provider_config=OCRAnswerProviderConfig(enabled=True),
        )
    with pytest.raises(ValueError, match="QA-A1 is restricted"):
        runner.run_l21_150_baseline(
            object(),
            object(),
            Path("out"),
            experiment_id="qa-a1",
            split="dev",
            task="kis",
            top_k=100,
            refine_top_n=3,
            resume=False,
            fail_fast=True,
            benchmark_sha256="0" * 64,
            manifest_sha256=None,
            gt_policy="proposed",
            qa_video_conditioned_evidence_config=enabled,
        )
