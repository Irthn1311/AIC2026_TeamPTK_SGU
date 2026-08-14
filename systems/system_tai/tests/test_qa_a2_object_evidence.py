from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from system_tai.common.schemas import CandidateFrame, FrameMappingRecord, KISResult
from system_tai.evidence.object_artifacts import (
    BTC_OBJECT_ARTIFACT_SCHEMA,
    ObjectArtifactError,
    ObjectArtifactIndex,
)
from system_tai.kis.session_schema import QAQueryRequest, SessionConfig
from system_tai.qa.grounding import QAVideoConditionedEvidenceConfig
from system_tai.qa.models import QAEvidenceCandidate
from system_tai.qa.object_provider import (
    GLOBAL_SUPPORT_RANKING,
    QUERY_CONDITIONED_FRAME_RANKING,
    ObjectAnswerProviderConfig,
    ObjectEntityAnswerProvider,
    normalize_object_label,
)
from system_tai.qa.question_types import QuestionType, classify_question
from system_tai.qa.runtime import (
    LEGACY_PHASE_P0,
    QA_A2_CAPABILITY_AWARE,
    QARuntimePipeline,
    classify_runtime_question,
)
from system_tai.refinement.engine import QueryRefinementOutcome
from system_tai.refinement.models import RefinedCandidate, RefinementStatus
from system_tai.refinement.video import RawVideoRecord, RawVideoRegistry
from system_tai.retrieval.multi_query import WeightedRRFRetriever
from system_tai.retrieval.video_evidence import (
    FullCorpusVideoMaximaOutcome,
    RestrictedFrameHit,
    VideoMaximumHit,
    VideoRestrictedSearchOutcome,
)


def _mapping(order: int, frame_id: int, row: int = 0) -> FrameMappingRecord:
    return FrameMappingRecord(row, order, frame_id, frame_id / 30.0, 30.0)


def _write_object(
    root: Path,
    video_id: str,
    order: int,
    detections: list[tuple[str, float, int]],
) -> Path:
    video_dir = root / video_id
    video_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "detection_scores": [str(score) for _label, score, _class_id in detections],
        "detection_boxes": [
            ["0.1", "0.2", "0.5", "0.8"] for _item in detections
        ],
        "detection_class_entities": [label for label, _score, _class_id in detections],
        "detection_class_labels": [
            str(class_id) for _label, _score, class_id in detections
        ],
    }
    path = video_dir / f"{order:04d}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _index(tmp_path: Path) -> ObjectArtifactIndex:
    root = tmp_path / "objects"
    root.mkdir()
    _write_object(root, "V1", 1, [("Traffic light", 0.8, 1), ("Car", 0.9, 2)])
    _write_object(root, "V1", 2, [("car", 0.7, 2)])
    _write_object(root, "V1", 3, [("Person", 0.99, 3), ("CAR", 0.6, 2)])
    return ObjectArtifactIndex(
        object_root=root,
        mappings_by_video={
            "V1": (_mapping(1, 100, 0), _mapping(2, 100, 1), _mapping(3, 200, 2))
        },
    )


def _refined(rank: int, candidate_frame: int, refined_frame: int) -> RefinedCandidate:
    return RefinedCandidate(
        query_id="q",
        original_candidate_rank=rank,
        video_id="V1",
        candidate_frame_id=candidate_frame,
        refined_frame_id=refined_frame,
        candidate_timestamp_seconds=candidate_frame / 30.0,
        refined_timestamp_seconds=refined_frame / 30.0,
        fps=30.0,
        total_frame_count=1000,
        window_start_frame=max(0, candidate_frame - 20),
        window_end_frame=candidate_frame + 20,
        coarse_frame_ids=(candidate_frame,),
        fine_frame_ids=(refined_frame,),
        coarse_sample_count=1,
        fine_sample_count=1,
        decoded_frame_count=1,
        encoded_image_count=1,
        refinement_fusion_score=0.9,
        variant_hit_count=1,
        best_individual_rank=1,
        per_variant_provenance=(),
        decoder_backend="fake",
        raw_video_path=Path("V1.mp4"),
        status=RefinementStatus.REFINED,
        warnings=(),
        failure_reason=None,
        original_retrieval_provenance={"fusion_score": 0.9},
        timings={},
    )


def _evidence(
    rank: int,
    frame_id: int,
    *,
    candidate_frame_id: int | None = None,
) -> QAEvidenceCandidate:
    return QAEvidenceCandidate(
        "q",
        rank,
        "V1",
        frame_id,
        0.9,
        provenance={
            "candidate_frame_id": (
                frame_id if candidate_frame_id is None else candidate_frame_id
            )
        },
    )


class _ObjectTextEncoder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def encode_texts(self, texts):
        resolved = tuple(texts)
        self.calls.append(resolved)
        vectors = []
        for text in resolved:
            if text == "Who is visible?" or text == "a photo of person":
                vectors.append((1.0, 0.0))
            elif text == "a photo of car":
                vectors.append((0.0, 1.0))
            elif text == "a photo of traffic light":
                vectors.append((0.0, 0.5))
            else:
                raise AssertionError(f"unexpected text: {text}")
        return np.asarray(vectors, dtype=np.float32)


def test_object_index_maps_json_ordinal_to_original_frame_and_is_deterministic(
    tmp_path: Path,
) -> None:
    index = _index(tmp_path)
    first = index.lookup("V1", 100)
    second = index.lookup("V1", 100)
    assert first == second
    assert first is not None
    assert first.object_source_frame_id == 100
    assert [item.label for item in first.detections] == [
        "Car",
        "Traffic light",
        "car",
    ]
    assert {item.source_keyframe_order for item in first.detections} == {1, 2}
    assert all(
        item.source_keyframe_order != first.object_source_frame_id
        for item in first.detections
    )
    assert index.schema_identity == BTC_OBJECT_ARTIFACT_SCHEMA


def test_object_index_rejects_ambiguous_numeric_json_stem(tmp_path: Path) -> None:
    root = tmp_path / "objects"
    root.mkdir()
    _write_object(root, "V1", 1, [("Car", 0.9, 1)])
    (root / "V1" / "1.json").write_text(
        (root / "V1" / "0001.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    index = ObjectArtifactIndex(
        object_root=root,
        mappings_by_video={"V1": (_mapping(1, 100),)},
    )
    with pytest.raises(ObjectArtifactError, match="ambiguous object artifacts"):
        index.lookup("V1", 100)


def test_object_label_normalization_and_artifact_backed_aggregation(tmp_path: Path) -> None:
    index = _index(tmp_path)
    provider = ObjectEntityAnswerProvider(
        index=index,
        config=ObjectAnswerProviderConfig(enabled=True),
    )
    assert normalize_object_label("  Traffic\u00a0  Light ") == "traffic light"
    result, telemetry = provider.answer(
        query_id="q",
        question_type=QuestionType.OBJECT_ENTITY,
        evidence=(_evidence(1, 100), _evidence(2, 200)),
        output_top_k=10,
    )
    assert [prediction.answer for prediction in result.predictions] == [
        "car",
        "traffic light",
        "person",
    ]
    assert result.predictions[0].frame_id == 100
    assert telemetry["object_detection_count"] == 5
    assert telemetry["unique_object_label_count"] == 3
    assert telemetry["top_object_candidates"][0]["supporting_evidence_count"] == 2
    serialized = json.dumps(telemetry)
    assert "target_video" not in serialized
    assert "ground_truth" not in serialized
    assert "accepted_answer" not in serialized


def test_query_conditioned_frame_ranking_preserves_frame_answer_tuples(
    tmp_path: Path,
) -> None:
    encoder = _ObjectTextEncoder()
    provider = ObjectEntityAnswerProvider(
        index=_index(tmp_path),
        config=ObjectAnswerProviderConfig(
            enabled=True,
            ranking_policy=QUERY_CONDITIONED_FRAME_RANKING,
        ),
        text_encoder=encoder,
    )
    result, telemetry = provider.answer(
        query_id="q",
        question_type=QuestionType.OBJECT_ENTITY,
        evidence=(_evidence(1, 100), _evidence(2, 200)),
        output_top_k=4,
        question_text="Who is visible?",
    )
    assert [
        (prediction.rank, prediction.video_id, prediction.frame_id, prediction.answer)
        for prediction in result.predictions
    ] == [
        (1, "V1", 100, "car"),
        (2, "V1", 200, "person"),
        (3, "V1", 100, "traffic light"),
        (4, "V1", 200, "car"),
    ]
    assert encoder.calls == [
        (
            "Who is visible?",
            "a photo of car",
            "a photo of person",
            "a photo of traffic light",
        )
    ]
    assert telemetry["object_answer_ranking_policy"] == (
        QUERY_CONDITIONED_FRAME_RANKING
    )
    assert telemetry["top_object_prediction_candidates"][1]["frame_id"] == 200
    assert telemetry["top_object_prediction_candidates"][1]["frame_label_rank"] == 1
    serialized = json.dumps(telemetry)
    assert "ground_truth" not in serialized
    assert "accepted_answer" not in serialized


def test_query_conditioned_frame_ranking_requires_encoder_and_question(
    tmp_path: Path,
) -> None:
    config = ObjectAnswerProviderConfig(
        enabled=True,
        ranking_policy=QUERY_CONDITIONED_FRAME_RANKING,
    )
    missing_encoder_root = tmp_path / "missing-encoder"
    missing_encoder_root.mkdir()
    with pytest.raises(ValueError, match="requires a shared text encoder"):
        ObjectEntityAnswerProvider(index=_index(missing_encoder_root), config=config)

    missing_question_root = tmp_path / "missing-question"
    missing_question_root.mkdir()
    provider = ObjectEntityAnswerProvider(
        index=_index(missing_question_root),
        config=config,
        text_encoder=_ObjectTextEncoder(),
    )
    with pytest.raises(ValueError, match="requires non-empty question_text"):
        provider.answer(
            query_id="q",
            question_type=QuestionType.OBJECT_ENTITY,
            evidence=(_evidence(1, 100),),
            output_top_k=4,
        )


def test_global_support_ranking_remains_the_default() -> None:
    config = ObjectAnswerProviderConfig()
    assert config.ranking_policy == GLOBAL_SUPPORT_RANKING


def test_anchor_fallback_is_explicit_and_preserves_authoritative_source_frame(
    tmp_path: Path,
) -> None:
    provider = ObjectEntityAnswerProvider(
        index=_index(tmp_path),
        config=ObjectAnswerProviderConfig(enabled=True),
    )
    result, telemetry = provider.answer(
        query_id="q",
        question_type=QuestionType.OBJECT_ENTITY,
        evidence=(_evidence(1, 105, candidate_frame_id=100),),
        output_top_k=5,
    )
    assert result.predictions[0].frame_id == 100
    assert telemetry["exact_object_frame_hit_count"] == 0
    assert telemetry["nearest_object_frame_fallback_count"] == 0
    assert telemetry["candidate_anchor_object_fallback_count"] == 1
    assert telemetry["object_evidence"][0]["frame_distance"] == 5
    assert telemetry["object_evidence"][0]["lookup_kind"] == (
        "AUTHORITATIVE_CANDIDATE_ANCHOR"
    )


@pytest.mark.parametrize(
    ("question", "expected", "reason_prefix"),
    [
        ("Đây là vật gì?", QuestionType.OBJECT_ENTITY, "OBJECT_ENTITY_PATTERN"),
        ("What is he holding?", QuestionType.OBJECT_ENTITY, "OBJECT_ENTITY_PATTERN"),
        ("How many people are visible?", QuestionType.OBJECT_COUNT, "OBJECT_COUNT_PATTERN"),
        ("Bao nhiêu tiền?", QuestionType.UNSUPPORTED, "OCR_PATTERN_PROVIDER_MISSING"),
        ("Biển số xe ghi gì?", QuestionType.UNSUPPORTED, "OCR_PATTERN_PROVIDER_MISSING"),
        ("Người đó đang làm gì?", QuestionType.UNSUPPORTED, "NO_SUPPORTED"),
        ("Đây là cảnh ở đâu?", QuestionType.UNSUPPORTED, "NO_SUPPORTED"),
        ("What color is the car?", QuestionType.COLOR, "LEGACY_COLOR_PATTERN"),
        ("How many times?", QuestionType.COUNT, "LEGACY_COUNT_PATTERN"),
        (
            "Cây trồng đang được máy thu hoạch trong cảnh là gì?",
            QuestionType.OBJECT_ENTITY,
            "OBJECT_ENTITY_PATTERN",
        ),
        (
            "Đoàn xe màu trắng chủ yếu là loại xe gì?",
            QuestionType.OBJECT_ENTITY,
            "OBJECT_ENTITY_PATTERN",
        ),
        (
            "Khán giả đang dùng thiết bị gì để quay?",
            QuestionType.OBJECT_ENTITY,
            "OBJECT_ENTITY_PATTERN",
        ),
        ("Có mấy người trong cảnh?", QuestionType.COUNT, "LEGACY_COUNT_PATTERN"),
        ("Máy thu hoạch đang chạy.", QuestionType.UNSUPPORTED, "NO_SUPPORTED"),
    ],
)
def test_question_classifier_is_conservative(
    question: str,
    expected: QuestionType,
    reason_prefix: str,
) -> None:
    classification = classify_question(question)
    assert classification.question_type is expected
    assert classification.reason.startswith(reason_prefix)


def test_object_count_has_no_provider_and_config_requires_qa_a1(tmp_path: Path) -> None:
    provider = ObjectEntityAnswerProvider(
        index=_index(tmp_path),
        config=ObjectAnswerProviderConfig(enabled=True),
    )
    assert provider.supports(QuestionType.OBJECT_ENTITY)
    assert not provider.supports(QuestionType.OBJECT_COUNT)
    with pytest.raises(ValueError, match="requires QA video-conditioned evidence"):
        SessionConfig(qa_object_answer_provider_config=ObjectAnswerProviderConfig(enabled=True))


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Có bao nhiêu người?", QuestionType.COUNT),
        ("Có bao nhiêu tiền?", QuestionType.COUNT),
        ("How many people are visible?", QuestionType.COUNT),
        ("How much does it cost?", QuestionType.COUNT),
        ("What object is he holding?", QuestionType.UNSUPPORTED),
        ("What color is the car?", QuestionType.COLOR),
        ("Có phải chiếc xe đang dừng không?", QuestionType.YES_NO),
        ("The car is on the left or right?", QuestionType.DIRECTION),
    ],
)
def test_runtime_qa_a2_off_reproduces_frozen_parent_classifier(
    question: str,
    expected: QuestionType,
) -> None:
    classification, policy = classify_runtime_question(
        question,
        None,
        qa_a2_enabled=False,
    )
    assert classification.question_type is expected
    assert policy == LEGACY_PHASE_P0


def test_runtime_qa_a2_enabled_uses_capability_classifier() -> None:
    object_count, object_policy = classify_runtime_question(
        "Có bao nhiêu người?",
        None,
        qa_a2_enabled=True,
    )
    monetary, monetary_policy = classify_runtime_question(
        "Có bao nhiêu tiền?",
        None,
        qa_a2_enabled=True,
    )
    assert object_count.question_type is QuestionType.OBJECT_COUNT
    assert object_policy == QA_A2_CAPABILITY_AWARE
    assert monetary.question_type is QuestionType.UNSUPPORTED
    assert monetary.reason.startswith("OCR_PATTERN_PROVIDER_MISSING")
    assert monetary_policy == QA_A2_CAPABILITY_AWARE


def test_runtime_qa_a2_enabled_object_count_and_ocr_fail_closed(tmp_path: Path) -> None:
    encoder = _Encoder()
    provider = ObjectEntityAnswerProvider(
        index=_index(tmp_path),
        config=ObjectAnswerProviderConfig(enabled=True),
    )
    retriever = SimpleNamespace()
    pipeline = QARuntimePipeline(
        exact_retriever=retriever,
        weighted_rrf=WeightedRRFRetriever(retriever),
        refiner=SimpleNamespace(),
        raw_video_registry=RawVideoRegistry([RawVideoRecord("V1", None)]),
        decoder=SimpleNamespace(),
        shared_encoder=encoder,
        object_answer_provider=provider,
    )
    count_result, _timings, count_diagnostics = pipeline.process_qa_query(
        QAQueryRequest("count", "q-count", "event", "Có bao nhiêu người?")
    )
    ocr_result, _timings, ocr_diagnostics = pipeline.process_qa_query(
        QAQueryRequest("ocr", "q-ocr", "event", "Có bao nhiêu tiền?")
    )
    assert count_result.question_type is QuestionType.OBJECT_COUNT
    assert count_result.unsupported_reason == "UNSUPPORTED_OBJECT_COUNT_PROVIDER_MISSING"
    assert count_diagnostics["question_classifier_policy"] == QA_A2_CAPABILITY_AWARE
    assert ocr_result.question_type is QuestionType.UNSUPPORTED
    assert ocr_result.unsupported_reason == "UNSUPPORTED_OCR_PROVIDER_MISSING"
    assert ocr_diagnostics["question_classifier_policy"] == QA_A2_CAPABILITY_AWARE


def test_disabled_qa_a1_does_not_activate_object_provider(tmp_path: Path) -> None:
    encoder = _Encoder()
    provider = ObjectEntityAnswerProvider(
        index=_index(tmp_path),
        config=ObjectAnswerProviderConfig(enabled=True),
    )
    retriever = SimpleNamespace()
    pipeline = QARuntimePipeline(
        exact_retriever=retriever,
        weighted_rrf=WeightedRRFRetriever(retriever),
        refiner=SimpleNamespace(),
        raw_video_registry=RawVideoRegistry([RawVideoRecord("V1", None)]),
        decoder=SimpleNamespace(),
        shared_encoder=encoder,
        object_answer_provider=provider,
    )
    result, _timings, diagnostics = pipeline.process_qa_query(
        QAQueryRequest("request-off", "q-off", "event", "What object is he holding?")
    )
    assert result.predictions == []
    assert result.unsupported_reason == "UNSUPPORTED_NO_PROVIDER"
    assert diagnostics["qa_grounding_enabled"] is False
    assert diagnostics["object_artifact_lookup_count"] == 0


def test_qa_a1_enabled_qa_a2_disabled_keeps_legacy_classification() -> None:
    classification, policy = classify_runtime_question(
        "Có bao nhiêu người?",
        None,
        qa_a2_enabled=False,
    )
    qa_a1_config = QAVideoConditionedEvidenceConfig(enabled=True)
    assert qa_a1_config.enabled is True
    assert classification.question_type is QuestionType.COUNT
    assert policy == LEGACY_PHASE_P0


class _Encoder:
    identifiers = {"device": "cpu", "model": "fake"}

    def __init__(self) -> None:
        self.image_calls = 0

    def encode_texts(self, texts):
        result = np.zeros((len(texts), 4), dtype=np.float32)
        result[:, 0] = 1.0
        return result

    def encode_images(self, images):
        self.image_calls += 1
        raise AssertionError("artifact-backed object provider must not encode images")


class _Searcher:
    class _Registry:
        total_rows = 1

        def get(self, _video):
            return SimpleNamespace(descriptor=SimpleNamespace(row_count=1))

    registry = _Registry()

    def search_video_maxima(self, *, query_ids, query_vectors):
        return FullCorpusVideoMaximaOutcome(
            rankings={
                query_id: (
                    VideoMaximumHit(query_id, "V1", 100, 0, 1, 0.9, 1),
                )
                for query_id in query_ids
            },
            physical_rows_scored=1,
            video_store_scan_count=1,
        )

    def search_selected_videos(
        self, *, video_ids, query_ids, query_vectors, per_query_result_cap
    ):
        return VideoRestrictedSearchOutcome(
            rankings={
                query_id: {
                    "V1": (RestrictedFrameHit("V1", 100, 0, 1, 100 / 30, 0.9, 1),)
                }
                for query_id in query_ids
            },
            physical_rows_scored=1,
            video_store_scan_count=1,
        )


class _Refiner:
    def refine_query(self, query, config, precomputed_text_embeddings=None):
        candidate = query.candidates[0]
        refined = _refined(1, candidate.frame_id, candidate.frame_id)
        result = KISResult(
            query.query_id,
            (CandidateFrame("V1", 100, 0, 1, 0.9, 1, "refined"),),
        )
        return QueryRefinementOutcome(query.query_id, result, (refined,), (), {})


class _UnrefinedAnchorRefiner:
    def refine_query(self, query, config, precomputed_text_embeddings=None):
        candidate = query.candidates[0]
        record = RefinedCandidate(
            query_id=query.query_id,
            original_candidate_rank=1,
            video_id=candidate.video_id,
            candidate_frame_id=candidate.frame_id,
            refined_frame_id=None,
            candidate_timestamp_seconds=100 / 30,
            refined_timestamp_seconds=None,
            fps=None,
            total_frame_count=None,
            window_start_frame=None,
            window_end_frame=None,
            coarse_frame_ids=(),
            fine_frame_ids=(),
            coarse_sample_count=0,
            fine_sample_count=0,
            decoded_frame_count=0,
            encoded_image_count=0,
            refinement_fusion_score=None,
            variant_hit_count=0,
            best_individual_rank=None,
            per_variant_provenance=(),
            decoder_backend=None,
            raw_video_path=None,
            status=RefinementStatus.NOT_REFINED,
            warnings=("candidate outside top_candidates_to_refine",),
            failure_reason=None,
            original_retrieval_provenance=candidate.retrieval_provenance,
            timings={},
        )
        return QueryRefinementOutcome(
            query.query_id,
            KISResult(query.query_id, ()),
            (record,),
            record.warnings,
            {},
        )


def test_qa_a1_routes_object_entity_to_artifact_provider_without_image_encoding(
    tmp_path: Path,
) -> None:
    video = tmp_path / "V1.mp4"
    video.touch()
    index = _index(tmp_path)
    encoder = _Encoder()
    unused_retriever = SimpleNamespace()
    pipeline = QARuntimePipeline(
        exact_retriever=unused_retriever,
        weighted_rrf=WeightedRRFRetriever(unused_retriever),
        refiner=_Refiner(),
        raw_video_registry=RawVideoRegistry([RawVideoRecord("V1", video)]),
        decoder=SimpleNamespace(),
        shared_encoder=encoder,
        video_restricted_searcher=_Searcher(),
        video_conditioned_evidence_config=QAVideoConditionedEvidenceConfig(
            enabled=True,
            selected_video_cap=1,
            anchors_per_video=1,
        ),
        object_answer_provider=ObjectEntityAnswerProvider(
            index=index,
            config=ObjectAnswerProviderConfig(enabled=True),
        ),
    )
    result, _timings, diagnostics = pipeline.process_qa_query(
        QAQueryRequest(
            "request",
            "q",
            "A person holds an object",
            "What object is he holding?",
            output_top_k=10,
            refine_top_n=1,
        )
    )
    assert result.predictions[0].answer == "car"
    assert result.predictions[0].frame_id == 100
    assert encoder.image_calls == 0
    assert diagnostics["object_provider_enabled"] is True
    assert diagnostics["question_classification_reason"].startswith(
        "OBJECT_ENTITY_PATTERN"
    )


def test_keyframe_bank_routes_unrefined_anchor_to_object_provider(
    tmp_path: Path,
) -> None:
    video = tmp_path / "V1.mp4"
    video.touch()
    encoder = _Encoder()
    unused_retriever = SimpleNamespace()
    pipeline = QARuntimePipeline(
        exact_retriever=unused_retriever,
        weighted_rrf=WeightedRRFRetriever(unused_retriever),
        refiner=_UnrefinedAnchorRefiner(),
        raw_video_registry=RawVideoRegistry([RawVideoRecord("V1", video)]),
        decoder=SimpleNamespace(),
        shared_encoder=encoder,
        video_restricted_searcher=_Searcher(),
        video_conditioned_evidence_config=QAVideoConditionedEvidenceConfig(
            enabled=True,
            selected_video_cap=1,
            anchors_per_video=1,
            preserve_keyframe_evidence=True,
            keyframe_evidence_video_cap=1,
        ),
        object_answer_provider=ObjectEntityAnswerProvider(
            index=_index(tmp_path),
            config=ObjectAnswerProviderConfig(enabled=True),
        ),
    )
    result, _timings, diagnostics = pipeline.process_qa_query(
        QAQueryRequest(
            "keyframe-object",
            "q",
            "A person holds an object",
            "What object is he holding?",
            output_top_k=10,
            refine_top_n=1,
        )
    )
    assert result.predictions[0].answer == "car"
    assert result.predictions[0].frame_id == 100
    assert encoder.image_calls == 0
    assert diagnostics["refinement_success_count"] == 0
    assert diagnostics["keyframe_evidence_count"] == 1
    assert diagnostics["provider_evidence_count"] == 1
    assert diagnostics["usable_evidence_candidates"][0]["evidence_source"] == (
        "KEYFRAME_ANCHOR"
    )


def test_l21_object_flag_is_isolated_to_qa_a1_dev() -> None:
    runner_path = Path(__file__).parents[1] / "scripts" / "l21_150_run_baseline.py"
    spec = importlib.util.spec_from_file_location("qa_a2_l21_runner", runner_path)
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
            "--qa-object-evidence",
        ]
    )
    assert args.qa_object_evidence is True
    assert args.qa_object_ranking_policy == GLOBAL_SUPPORT_RANKING
    query_conditioned = runner.build_parser().parse_args(
        [
            "--benchmark",
            "benchmark.json",
            "--reuse-manifest",
            "manifest.json",
            "--output-dir",
            "out",
            "--qa-video-conditioned-evidence",
            "--qa-object-evidence",
            "--qa-object-ranking-policy",
            QUERY_CONDITIONED_FRAME_RANKING,
        ]
    )
    assert query_conditioned.qa_object_ranking_policy == (
        QUERY_CONDITIONED_FRAME_RANKING
    )
    enabled = ObjectAnswerProviderConfig(enabled=True)
    with pytest.raises(ValueError, match="QA-A2 requires QA DEV"):
        runner.run_l21_150_baseline(
            object(),
            object(),
            Path("out"),
            experiment_id="qa-a2",
            split="dev",
            task="qa",
            top_k=100,
            refine_top_n=3,
            resume=False,
            fail_fast=True,
            benchmark_sha256="0" * 64,
            manifest_sha256=None,
            gt_policy="proposed",
            qa_object_answer_provider_config=enabled,
        )


def test_l21_qa_a2_provenance_scope_has_precedence(tmp_path: Path) -> None:
    runner_path = Path(__file__).parents[1] / "scripts" / "l21_150_run_baseline.py"
    spec = importlib.util.spec_from_file_location("qa_a2_scope_runner", runner_path)
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runner
    spec.loader.exec_module(runner)
    grounding = QAVideoConditionedEvidenceConfig(
        enabled=True,
        selected_video_cap=32,
        preserve_keyframe_evidence=True,
        keyframe_evidence_video_cap=32,
    )
    object_config = ObjectAnswerProviderConfig(enabled=True)
    runtime = SimpleNamespace(
        config=SimpleNamespace(
            device="cpu",
            qa_video_conditioned_evidence_config=grounding,
            qa_object_answer_provider_config=object_config,
        ),
        manifest=SimpleNamespace(fingerprint="fixture", schema_version=2),
        shared_encoder=SimpleNamespace(identifiers={"model": "fake"}),
        object_artifact_index=SimpleNamespace(
            schema_identity=BTC_OBJECT_ARTIFACT_SCHEMA,
            source_root_identity="objects-aic25-b1/objects",
        ),
    )
    report = runner.run_l21_150_baseline(
        SimpleNamespace(queries=(), benchmark_id="empty"),
        runtime,
        tmp_path,
        experiment_id="qa-a2-scope",
        split="dev",
        task="qa",
        top_k=100,
        refine_top_n=1,
        resume=False,
        fail_fast=True,
        benchmark_sha256="0" * 64,
        manifest_sha256=None,
        gt_policy="proposed",
        qa_video_conditioned_evidence_config=grounding,
        qa_object_answer_provider_config=object_config,
    )
    assert report["production_algorithm_modified_scope"] == (
        runner.QA_ARTIFACT_BACKED_OBJECT_EVIDENCE
    )
    assert report["qa_grounding_policy"] == runner.QA_VIDEO_CONDITIONED_EVIDENCE_V1
    assert report["qa_object_provider_enabled"] is True
    assert report["qa_video_conditioned_evidence_config"] == {
        "selected_video_cap": 32,
        "anchors_per_video": 5,
        "video_rrf_constant": 60.0,
            "preserve_keyframe_evidence": True,
            "keyframe_evidence_video_cap": 32,
            "keyframe_evidence_anchors_per_video": 1,
            "temporal_refinement_enabled": False,
            "temporal_seed_anchors_per_video": 3,
            "temporal_refinement_video_cap": 32,
            "temporal_refinement_total_seed_cap": 96,
        }
    assert report["qa_keyframe_evidence_bank"] == {
        "policy": runner.QA_KEYFRAME_EVIDENCE_BANK_V1,
        "enabled": True,
            "selection": "ONE_PRIMARY_LOCAL_ANCHOR_PER_NOMINATED_VIDEO",
            "video_cap": 32,
            "anchors_per_video": 1,
            "raw_refinement_budget": 1,
        "refinement_is_upgrade_not_admission_gate": True,
    }
