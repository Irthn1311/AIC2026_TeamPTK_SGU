from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from system_tai.checkpointing.exporter import CheckpointExporter
from system_tai.common.schemas import CandidateFrame, KISResult
from system_tai.kis.session_engine import OperationalKISRuntime
from system_tai.kis.session_schema import (
    DuplicateRequestIdError,
    QAQueryRequest,
    QueryRequest,
    SessionConfig,
    TRAKEQueryRequest,
)
from system_tai.preliminary.evaluation import evaluate_ranked_query
from system_tai.preliminary.matching import NormalizedAliasAnswerMatcher
from system_tai.preliminary.runtime_bridge import (
    RuntimeTop100MismatchError,
    audit_runtime_top100_artifact,
    kis_result_to_top100_query,
    qa_predictions_to_top100_query,
    trake_predictions_to_top100_query,
)
from system_tai.preliminary.schemas import (
    KISGroundTruth,
    KISPrediction,
    QAGroundTruth,
    QAPrediction,
    TRAKEGroundTruth,
    TRAKEPrediction,
)
from system_tai.preliminary.scoring import (
    score_kis_prediction,
    score_qa_prediction,
    score_trake_prediction,
)
from system_tai.preliminary.top100 import (
    RankedTop100Dataset,
    RankedTop100Query,
    load_top100_jsonl,
    write_top100_jsonl,
)
from system_tai.qa.models import QAResult
from system_tai.qa.question_types import QuestionType
from system_tai.qa.runtime import QAPipelineTimings
from system_tai.refinement.engine import QueryRefinementOutcome
from system_tai.trake.models import TRAKEResult
from system_tai.trake.runtime import TRAKERuntimeTimings


def _candidate(frame_id: int, *, rank: int = 1) -> CandidateFrame:
    return CandidateFrame(
        video_id="V001",
        frame_id=frame_id,
        clip_row=900 + rank,
        keyframe_order=700 + rank,
        score=0.9 - rank / 1000,
        rank=rank,
        source="synthetic",
        diagnostic_metadata={"private": "must-not-leak"},
    )


def _kis_result(query_id: str, frame_id: int = 42) -> KISResult:
    return KISResult(query_id, (_candidate(frame_id),))


class _FakeEncoder:
    dimension = 4
    identifiers = {"device": "cpu", "model": "fake"}

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        result = np.zeros((len(texts), self.dimension), dtype=np.float32)
        if texts:
            result[:, 0] = 1.0
        return result


class _FakeVideoStore:
    def contains_frame(self, frame_id: int) -> bool:
        return frame_id >= 0


class _FakeRegistry:
    embedding_dimension = 4
    total_rows = 1
    stores: tuple[Any, ...] = ()

    def get(self, video_id: str) -> _FakeVideoStore:
        return _FakeVideoStore()


class _FakeRetriever:
    def search_vector(
        self, query_id: str, query_vector: np.ndarray, top_k: int
    ) -> KISResult:
        return _kis_result(query_id)


class _FakeRRF:
    def __init__(self, frame_id: int = 42) -> None:
        self.frame_id = frame_id

    def fuse_rankings(
        self,
        *,
        query_id: str,
        variants: tuple[Any, ...],
        rankings: dict[str, KISResult],
        output_top_k: int,
        rrf_constant: float,
    ) -> KISResult:
        return _kis_result(query_id, self.frame_id)


class _FakeRefiner:
    def __init__(self, frame_id: int = 77) -> None:
        self.frame_id = frame_id

    def refine_query(
        self,
        query: Any,
        config: Any,
        *,
        precomputed_text_embeddings: np.ndarray | None = None,
    ) -> QueryRefinementOutcome:
        timings = {
            "refined_candidate_count": 1,
            "decoded_frame_count": 1,
            "encoded_image_count": 1,
            "coarse_requested_frame_count": 1,
            "coarse_decoded_frame_count": 1,
            "fine_requested_frame_count": 1,
            "fine_decoded_frame_count": 1,
            "coarse_sparse_request_count": 0,
            "coarse_sparse_success_count": 0,
            "coarse_sparse_fallback_count": 0,
            "video_probe_seconds": 0.0,
            "video_open_seconds": 0.0,
            "coarse_decode_seconds": 0.0,
            "coarse_encode_seconds": 0.0,
            "coarse_score_seconds": 0.0,
            "coarse_fusion_seconds": 0.0,
            "fine_decode_seconds": 0.0,
            "fine_encode_seconds": 0.0,
            "fine_score_seconds": 0.0,
            "fine_fusion_seconds": 0.0,
            "candidate_total_seconds": 0.0,
        }
        return QueryRefinementOutcome(
            query_id=query.query_id,
            result=_kis_result(query.query_id, self.frame_id),
            candidates=(),
            warnings=(),
            timings=timings,
        )


class _FakeQAPipeline:
    def __init__(self, *, empty: bool = False) -> None:
        self.empty = empty

    def process_qa_query(
        self,
        request: QAQueryRequest,
        *,
        refinement_config: Any,
        rrf_constant: float,
    ) -> tuple[QAResult, QAPipelineTimings, dict[str, Any]]:
        predictions = []
        question_type = QuestionType.UNSUPPORTED if self.empty else QuestionType.COLOR
        if not self.empty:
            predictions.append(
                QAPrediction(request.query_id, 1, "V001", 55, "  xanh dương  ")
            )
        return (
            QAResult(request.query_id, question_type, predictions, warnings=[]),
            QAPipelineTimings(),
            {"query_id": request.query_id, "evidence": [], "warnings": []},
        )


class _FakeTrakePipeline:
    def __init__(self, *, wrong_event_count: bool = False) -> None:
        self.wrong_event_count = wrong_event_count

    def process_trake_query(
        self,
        request: TRAKEQueryRequest,
        *,
        refinement_config: Any,
        rrf_constant: float,
    ) -> tuple[TRAKEResult, TRAKERuntimeTimings, dict[str, Any]]:
        frames = (
            (205,)
            if self.wrong_event_count
            else tuple(205 - 100 * i for i in range(len(request.events)))
        )
        prediction = TRAKEPrediction(request.query_id, 1, "V001", frames)
        result = TRAKEResult(
            request.query_id,
            len(request.events),
            (prediction,),
            diagnostics={"refinement_requested": False, "warnings": ()},
        )
        diagnostics = {
            "event_candidate_pools": tuple(() for _ in request.events),
            "c1_diagnostics": {},
            "refinement_node_records": [],
            "path_diagnostics": {},
            "flattened_variants": [],
        }
        return result, TRAKERuntimeTimings(), diagnostics


def _runtime(tmp_path: Path) -> OperationalKISRuntime:
    runtime = OperationalKISRuntime(
        config=SessionConfig(output_root=tmp_path / "out"),
        manifest_path=tmp_path / "manifest.json",
        manifest=SimpleNamespace(fingerprint="fixture", schema_version=1, videos={}),
        registry=_FakeRegistry(),
        raw_video_registry=SimpleNamespace(records=()),
        shared_encoder=_FakeEncoder(),
        decoder=SimpleNamespace(backend_identifier="fake"),
    )
    runtime.exact_retriever = _FakeRetriever()
    runtime.weighted_rrf = _FakeRRF()
    runtime.refiner = _FakeRefiner()
    runtime.qa_pipeline = _FakeQAPipeline()
    runtime.trake_pipeline = _FakeTrakePipeline()
    return runtime


def _write_dataset(path: Path, query: RankedTop100Query) -> None:
    write_top100_jsonl(RankedTop100Dataset(query.task_type, (query,)), path)


def test_bridge_conversions_preserve_exact_runtime_fields() -> None:
    kis = kis_result_to_top100_query(_kis_result("K1", 12345))
    assert kis.predictions == (KISPrediction("K1", 1, "V001", 12345),)

    qa_predictions = (
        QAPrediction("Q1", 20, "V020", 20, "  xanh dương  "),
        QAPrediction("Q1", 1, "V001", 1, "đỏ"),
    )
    qa = qa_predictions_to_top100_query(query_id="Q1", predictions=qa_predictions)
    assert qa.predictions == qa_predictions

    trake_predictions = (TRAKEPrediction("T1", 5, "V001", (205, 105)),)
    trake = trake_predictions_to_top100_query(
        query_id="T1",
        predictions=trake_predictions,
        expected_event_count=2,
    )
    assert trake.predictions == trake_predictions


@pytest.mark.parametrize(
    "query",
    [
        RankedTop100Query("kis", "K", (KISPrediction("K", 1, "V", 1),)),
        RankedTop100Query("qa", "Q", (QAPrediction("Q", 1, "V", 2, "có"),)),
        RankedTop100Query(
            "trake", "T", (TRAKEPrediction("T", 1, "V", (205, 105)),)
        ),
    ],
)
def test_nonempty_artifact_roundtrip_is_exact(
    tmp_path: Path, query: RankedTop100Query
) -> None:
    path = tmp_path / f"{query.task_type}.jsonl"
    _write_dataset(path, query)
    event_count = 2 if query.task_type == "trake" else None
    audit = audit_runtime_top100_artifact(
        query,
        path,
        expected_trake_event_count=event_count,
    )
    assert audit.roundtrip_status == "EXACT"
    assert audit.loaded_dataset == RankedTop100Dataset(query.task_type, (query,))


@pytest.mark.parametrize("task_type", ["kis", "qa", "trake"])
@pytest.mark.parametrize("payload", [b"", b"\n \t\n"])
def test_zero_prediction_artifact_is_explicitly_unrepresentable(
    tmp_path: Path, task_type: str, payload: bytes
) -> None:
    query = RankedTop100Query(task_type, "EMPTY", ())  # type: ignore[arg-type]
    path = tmp_path / f"{task_type}.jsonl"
    path.write_bytes(payload)
    event_count = 2 if task_type == "trake" else None
    audit = audit_runtime_top100_artifact(
        query,
        path,
        expected_trake_event_count=event_count,
    )
    assert audit.roundtrip_status == "EMPTY_QUERY_UNREPRESENTABLE"
    assert audit.prediction_count == 0 and audit.loaded_dataset is None


def test_missing_artifact_and_bridge_task_query_mismatches_fail_closed(
    tmp_path: Path,
) -> None:
    query = RankedTop100Query("kis", "K", (KISPrediction("K", 1, "V", 1),))
    with pytest.raises(RuntimeTop100MismatchError, match="artifact_path"):
        audit_runtime_top100_artifact(query, tmp_path / "missing.jsonl")
    with pytest.raises(RuntimeTop100MismatchError, match="QAPrediction"):
        qa_predictions_to_top100_query(
            query_id="Q",
            predictions=(KISPrediction("Q", 1, "V", 1),),  # type: ignore[arg-type]
        )
    with pytest.raises(RuntimeTop100MismatchError, match="query_id mismatch"):
        qa_predictions_to_top100_query(
            query_id="Q",
            predictions=(QAPrediction("OTHER", 1, "V", 1, "yes"),),
        )


def test_trake_expected_event_count_mismatch_is_rejected() -> None:
    with pytest.raises(RuntimeTop100MismatchError, match="event-count mismatch"):
        trake_predictions_to_top100_query(
            query_id="T",
            predictions=(TRAKEPrediction("T", 1, "V", (1, 2)),),
            expected_event_count=3,
        )


@pytest.mark.parametrize(
    ("query", "tampered"),
    [
        (
            RankedTop100Query("kis", "K", (KISPrediction("K", 1, "V", 10),)),
            {"query_id": "K", "rank": 1, "video_id": "V", "frame_id": 11},
        ),
        (
            RankedTop100Query("qa", "Q", (QAPrediction("Q", 1, "V", 10, "đỏ"),)),
            {
                "query_id": "Q",
                "rank": 1,
                "video_id": "V",
                "frame_id": 10,
                "answer": "xanh",
            },
        ),
        (
            RankedTop100Query(
                "trake", "T", (TRAKEPrediction("T", 1, "V", (205, 105)),)
            ),
            {"query_id": "T", "rank": 1, "video_id": "V", "frame_ids": [105, 205]},
        ),
        (
            RankedTop100Query("kis", "K", (KISPrediction("K", 1, "V", 10),)),
            {"query_id": "K", "rank": 2, "video_id": "V", "frame_id": 10},
        ),
        (
            RankedTop100Query("kis", "K", (KISPrediction("K", 1, "V", 10),)),
            {"query_id": "OTHER", "rank": 1, "video_id": "V", "frame_id": 10},
        ),
    ],
)
def test_artifact_tampering_is_detected(
    tmp_path: Path,
    query: RankedTop100Query,
    tampered: dict[str, Any],
) -> None:
    path = tmp_path / "tampered.jsonl"
    path.write_text(json.dumps(tampered, ensure_ascii=False) + "\n", encoding="utf-8")
    event_count = 2 if query.task_type == "trake" else None
    with pytest.raises(RuntimeTop100MismatchError):
        audit_runtime_top100_artifact(
            query,
            path,
            expected_trake_event_count=event_count,
        )


def test_extra_record_and_physical_reordering_are_detected(tmp_path: Path) -> None:
    predictions = (
        KISPrediction("K", 20, "V20", 20),
        KISPrediction("K", 1, "V01", 1),
        KISPrediction("K", 5, "V05", 5),
    )
    query = RankedTop100Query("kis", "K", predictions)
    valid_path = tmp_path / "valid.jsonl"
    _write_dataset(valid_path, query)
    assert audit_runtime_top100_artifact(query, valid_path).roundtrip_status == "EXACT"

    reordered = RankedTop100Query("kis", "K", tuple(sorted(predictions, key=lambda p: p.rank)))
    reordered_path = tmp_path / "reordered.jsonl"
    _write_dataset(reordered_path, reordered)
    with pytest.raises(RuntimeTop100MismatchError):
        audit_runtime_top100_artifact(query, reordered_path)

    extra_path = tmp_path / "extra.jsonl"
    extra_path.write_bytes(
        valid_path.read_bytes()
        + b'{"query_id":"K","rank":99,"video_id":"VX","frame_id":99}\n'
    )
    with pytest.raises(RuntimeTop100MismatchError):
        audit_runtime_top100_artifact(query, extra_path)


@pytest.mark.parametrize(
    "payload",
    [
        b"\xff",
        b'\xef\xbb\xbf{"query_id":"K","rank":1,"video_id":"V","frame_id":1}\n',
        b'{"query_id":"K","rank":1,"video_id":"V","frame_id":1,"x":2}\n',
        b'{"query_id":"K","rank":1,"video_id":"V"}\n',
        b'{"query_id":"K","rank":true,"video_id":"V","frame_id":1}\n',
        b"not-json\n",
    ],
)
def test_nonempty_audit_reuses_strict_p0d_parser(tmp_path: Path, payload: bytes) -> None:
    query = RankedTop100Query("kis", "K", (KISPrediction("K", 1, "V", 1),))
    path = tmp_path / "invalid.jsonl"
    path.write_bytes(payload)
    with pytest.raises(RuntimeTop100MismatchError, match="strict P0-D artifact load failed"):
        audit_runtime_top100_artifact(query, path)


def test_qa_unicode_and_whitespace_are_exact(tmp_path: Path) -> None:
    prediction = QAPrediction("Q", 1, "V", 1, "  xanh dương  ")
    query = qa_predictions_to_top100_query(query_id="Q", predictions=(prediction,))
    exact_path = tmp_path / "exact.jsonl"
    _write_dataset(exact_path, query)
    assert audit_runtime_top100_artifact(query, exact_path).roundtrip_status == "EXACT"

    trimmed_path = tmp_path / "trimmed.jsonl"
    trimmed_path.write_text(
        '{"query_id":"Q","rank":1,"video_id":"V","frame_id":1,'
        '"answer":"xanh dương"}\n',
        encoding="utf-8",
    )
    with pytest.raises(RuntimeTop100MismatchError):
        audit_runtime_top100_artifact(query, trimmed_path)


def test_bridge_predictions_are_directly_evaluator_ready() -> None:
    kis = RankedTop100Query(
        "kis",
        "K",
        (KISPrediction("K", 1, "V", 105),),
    )
    kis_report = evaluate_ranked_query(
        "K",
        "kis",
        kis.predictions,
        KISGroundTruth("K", "V", 100, 110),
        score_kis_prediction,
    )

    qa = qa_predictions_to_top100_query(
        query_id="Q",
        predictions=(QAPrediction("Q", 1, "V", 105, "đỏ"),),
    )
    matcher = NormalizedAliasAnswerMatcher(strip_punctuation=True)
    qa_report = evaluate_ranked_query(
        "Q",
        "qa",
        qa.predictions,
        QAGroundTruth("Q", "V", 100, 110, ("đỏ", "red")),
        lambda prediction, ground_truth: score_qa_prediction(
            prediction, ground_truth, matcher
        ),
    )

    trake = trake_predictions_to_top100_query(
        query_id="T",
        predictions=(TRAKEPrediction("T", 1, "V", (105, 205)),),
        expected_event_count=2,
    )
    trake_report = evaluate_ranked_query(
        "T",
        "trake",
        trake.predictions,
        TRAKEGroundTruth("T", "V", ((100, 110), (200, 210))),
        score_trake_prediction,
    )

    assert (kis_report.r_at_1, kis_report.r_at_5) == (1.0, 1.0)
    assert (qa_report.r_at_1, qa_report.r_at_5) == (1.0, 1.0)
    assert (trake_report.r_at_1, trake_report.r_at_5) == (1.0, 1.0)


def test_real_kis_handle_audits_base_artifact_without_response_change(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path)
    response = runtime.handle_query(QueryRequest("rk", "K", "query", refine_top_n=0))
    assert response["status"] == "SUCCESS"
    assert set(response) == {
        "type",
        "request_id",
        "query_id",
        "status",
        "retrieval_valid",
        "refinement_requested",
        "refinement_valid",
        "result_count",
        "refined_count",
        "artifacts",
        "timings",
    }
    assert set(response["artifacts"]) == {
        "top100_jsonl",
        "top100_csv",
        "candidates_json",
        "validation_report",
    }
    artifact = runtime.output_root / response["artifacts"]["top100_jsonl"]
    loaded = load_top100_jsonl(artifact, task_type="kis", expected_query_ids=("K",))
    assert loaded.predictions_for("K") == (KISPrediction("K", 1, "V001", 42),)


def test_kis_refinement_audits_refined_not_base_artifact(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    response = runtime.handle_query(QueryRequest("rr", "K", "query", refine_top_n=1))
    base = load_top100_jsonl(
        runtime.output_root / response["artifacts"]["top100_jsonl"], task_type="kis"
    )
    refined = load_top100_jsonl(
        runtime.output_root / response["artifacts"]["refined_top100_jsonl"],
        task_type="kis",
    )
    assert base.predictions_for("K")[0].frame_id == 42  # type: ignore[union-attr]
    assert refined.predictions_for("K")[0].frame_id == 77  # type: ignore[union-attr]


def test_qa_runtime_nonempty_and_zero_prediction_contracts(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    nonempty = runtime.handle_qa_query(QAQueryRequest("rq1", "Q1", "event", "Màu gì?"))
    assert nonempty["status"] == "SUCCESS" and nonempty["prediction_count"] == 1
    assert set(nonempty["artifacts"]) == {
        "qa_predictions_jsonl",
        "qa_evidence_json",
        "qa_request_manifest",
        "qa_timings",
    }
    assert set(nonempty) == {
        "type",
        "request_id",
        "query_id",
        "status",
        "question_type",
        "prediction_count",
        "predictions",
        "warnings",
        "timings",
        "artifacts",
    }

    runtime.qa_pipeline = _FakeQAPipeline(empty=True)
    empty = runtime.handle_qa_query(QAQueryRequest("rq2", "Q2", "event", "What?"))
    artifact = runtime.output_root / empty["artifacts"]["qa_predictions_jsonl"]
    assert empty["status"] == "SUCCESS" and empty["prediction_count"] == 0
    assert not [line for line in artifact.read_text(encoding="utf-8").splitlines() if line.strip()]
    empty_query = qa_predictions_to_top100_query(query_id="Q2", predictions=())
    assert (
        audit_runtime_top100_artifact(empty_query, artifact).roundtrip_status
        == "EMPTY_QUERY_UNREPRESENTABLE"
    )


def test_trake_runtime_exact_five_artifacts_and_event_count_gate(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    request = TRAKEQueryRequest(
        "rt",
        "T",
        ({"description": "event 1"}, {"description": "event 2"}),
        refine_top_n=0,
    )
    response = runtime.handle_trake_query(request)
    assert response["status"] == "SUCCESS" and response["event_count"] == 2
    assert set(response["artifacts"]) == {
        "trake_predictions_jsonl",
        "trake_event_candidates_json",
        "trake_refinement_json",
        "trake_request_manifest",
        "trake_timings",
    }
    assert set(response) == {
        "type",
        "request_id",
        "query_id",
        "status",
        "event_count",
        "prediction_count",
        "predictions",
        "warnings",
        "timings",
        "artifacts",
    }
    artifact = runtime.output_root / response["artifacts"]["trake_predictions_jsonl"]
    loaded = load_top100_jsonl(
        artifact,
        task_type="trake",
        expected_query_ids=("T",),
        expected_trake_event_counts={"T": 2},
    )
    assert loaded.predictions_for("T")[0].frame_ids == (205, 105)  # type: ignore[union-attr]

    bad_runtime = _runtime(tmp_path / "bad")
    bad_runtime.trake_pipeline = _FakeTrakePipeline(wrong_event_count=True)
    with pytest.raises(RuntimeTop100MismatchError, match="event-count mismatch"):
        bad_runtime.handle_trake_query(request)


def test_one_runtime_handles_all_tasks_and_preserves_global_request_namespace(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path)
    kis = runtime.handle_query(QueryRequest("r-kis", "K", "query", refine_top_n=0))
    qa = runtime.handle_qa_query(QAQueryRequest("r-qa", "Q", "event", "Màu gì?"))
    trake = runtime.handle_trake_query(
        TRAKEQueryRequest(
            "r-trake",
            "T",
            ({"description": "e1"}, {"description": "e2"}),
            refine_top_n=0,
        )
    )
    assert (kis["status"], qa["status"], trake["status"]) == (
        "SUCCESS",
        "SUCCESS",
        "SUCCESS",
    )
    with pytest.raises(DuplicateRequestIdError):
        runtime.handle_qa_query(QAQueryRequest("r-kis", "Q2", "event", "Màu gì?"))


class _TamperingExporter(CheckpointExporter):
    def export(self, results: Any, destination: Path, **kwargs: Any) -> Any:
        summary = super().export(results, destination, **kwargs)
        lines = destination.read_text(encoding="utf-8").splitlines()
        record = json.loads(lines[0])
        record["frame_id"] += 1
        destination.write_text(json.dumps(record) + "\n", encoding="utf-8")
        return summary


def test_runtime_artifact_mismatch_fails_closed_at_handle_boundary(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    runtime.exporter = _TamperingExporter()
    with pytest.raises(RuntimeTop100MismatchError, match="differs from canonical"):
        runtime.handle_query(QueryRequest("bad", "K", "query", refine_top_n=0))
