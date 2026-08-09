from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from system_tai.preliminary import (
    TASK_RECORD_FIELDS,
    KISGroundTruth,
    KISPrediction,
    QAPrediction,
    RankedTop100Dataset,
    RankedTop100Query,
    Top100FormatError,
    TRAKEPrediction,
    evaluate_ranked_query,
    load_top100_jsonl,
    prediction_to_record,
    record_to_prediction,
    score_kis_prediction,
    validate_top100_dataset,
    validate_top100_jsonl,
    validate_top100_query,
    write_top100_jsonl,
)


def _query(task: str, query_id: str, predictions: tuple = ()) -> RankedTop100Query:
    return RankedTop100Query(task, query_id, predictions)  # type: ignore[arg-type]


def _dataset(task: str, *queries: RankedTop100Query) -> RankedTop100Dataset:
    return RankedTop100Dataset(task, tuple(queries))  # type: ignore[arg-type]


def _kis_predictions(query_id: str, count: int) -> tuple[KISPrediction, ...]:
    return tuple(
        KISPrediction(query_id, rank, f"L01_V{rank:03d}", rank * 10)
        for rank in range(1, count + 1)
    )


def _roundtrip(tmp_path: Path, dataset: RankedTop100Dataset) -> RankedTop100Dataset:
    path = tmp_path / f"{dataset.task_type}.jsonl"
    write_top100_jsonl(dataset, path)
    return load_top100_jsonl(path, task_type=dataset.task_type)


def test_kis_single_and_multiple_query_roundtrip_exact(tmp_path: Path) -> None:
    q1 = _query("kis", "Q1", (KISPrediction("Q1", 1, "L01_V001", 101),))
    q2 = _query("kis", "Q2", (KISPrediction("Q2", 3, "L02_V001", 202),))
    dataset = _dataset("kis", q1, q2)
    assert _roundtrip(tmp_path, dataset) == dataset


def test_kis_max_100_is_per_query_and_101_is_rejected() -> None:
    q1 = _query("kis", "Q1", _kis_predictions("Q1", 100))
    q2 = _query("kis", "Q2", _kis_predictions("Q2", 100))
    assert len(_dataset("kis", q1, q2).queries) == 2
    with pytest.raises(ValueError, match="Cannot exceed 100"):
        _query("kis", "Q3", _kis_predictions("Q3", 101))


def test_non_contiguous_ranks_and_physical_order_are_preserved(tmp_path: Path) -> None:
    predictions = (
        KISPrediction("Q1", 20, "L01_V003", 300),
        KISPrediction("Q1", 1, "L01_V001", 100),
        KISPrediction("Q1", 5, "L01_V002", 200),
    )
    dataset = _dataset("kis", _query("kis", "Q1", predictions))
    loaded = _roundtrip(tmp_path, dataset)
    assert tuple(item.rank for item in loaded.predictions_for("Q1")) == (20, 1, 5)
    assert loaded == dataset


def test_duplicate_rank_and_kis_semantic_key_are_rejected() -> None:
    with pytest.raises(ValueError, match="Duplicate rank"):
        _query(
            "kis",
            "Q1",
            (
                KISPrediction("Q1", 1, "L01_V001", 10),
                KISPrediction("Q1", 1, "L01_V002", 20),
            ),
        )
    with pytest.raises(ValueError, match="Duplicate KIS key"):
        _query(
            "kis",
            "Q1",
            (
                KISPrediction("Q1", 1, "L01_V001", 10),
                KISPrediction("Q1", 2, "L01_V001", 10),
            ),
        )


def test_query_identity_task_type_and_tuple_contracts() -> None:
    with pytest.raises(ValueError, match="query_id mismatch"):
        _query("kis", "Q1", (KISPrediction("Q2", 1, "L01_V001", 10),))
    with pytest.raises(TypeError, match="KISPrediction"):
        _query("kis", "Q1", (QAPrediction("Q1", 1, "L01_V001", 10, "yes"),))
    with pytest.raises(TypeError, match="predictions must be a tuple"):
        RankedTop100Query("kis", "Q1", [])  # type: ignore[arg-type]


def test_arbitrary_absolute_raw_video_frame_is_allowed_without_registry() -> None:
    prediction = KISPrediction("Q1", 1, "L99_V999", 987_654_321)
    query = _query("kis", "Q1", (prediction,))
    assert query.predictions == (prediction,)


def test_qa_unicode_whitespace_and_exact_shape_roundtrip(tmp_path: Path) -> None:
    answer = "  xanh dương  "
    prediction = QAPrediction("QA1", 5, "L05_V005", 850, answer)
    dataset = _dataset("qa", _query("qa", "QA1", (prediction,)))
    path = tmp_path / "qa.jsonl"
    write_top100_jsonl(dataset, path)
    assert path.read_bytes().decode("utf-8") == (
        '{"query_id":"QA1","rank":5,"video_id":"L05_V005",'
        '"frame_id":850,"answer":"  xanh dương  "}\n'
    )
    assert load_top100_jsonl(path, task_type="qa") == dataset
    assert TASK_RECORD_FIELDS["qa"] == (
        "query_id",
        "rank",
        "video_id",
        "frame_id",
        "answer",
    )


def test_qa_empty_duplicate_normalized_and_distinct_answer_semantics() -> None:
    with pytest.raises(ValueError, match="answer must be a non-empty string"):
        QAPrediction("QA1", 1, "L05_V005", 850, "   ")
    with pytest.raises(ValueError, match="Duplicate QA key"):
        _query(
            "qa",
            "QA1",
            (
                QAPrediction("QA1", 1, "L05_V005", 850, "  A RED Bus!!! "),
                QAPrediction("QA1", 2, "L05_V005", 850, "a red bus"),
            ),
        )
    query = _query(
        "qa",
        "QA1",
        (
            QAPrediction("QA1", 1, "L05_V005", 850, "red"),
            QAPrediction("QA1", 2, "L05_V005", 850, "blue"),
        ),
    )
    assert len(query.predictions) == 2


def test_qa_multiple_query_roundtrip(tmp_path: Path) -> None:
    dataset = _dataset(
        "qa",
        _query("qa", "QA2", (QAPrediction("QA2", 8, "L01_V001", 8, "có"),)),
        _query("qa", "QA1", (QAPrediction("QA1", 2, "L01_V002", 2, "không"),)),
    )
    loaded = _roundtrip(tmp_path, dataset)
    assert tuple(query.query_id for query in loaded.queries) == ("QA2", "QA1")
    assert loaded == dataset


def test_trake_roundtrip_and_authoritative_frame_order(tmp_path: Path) -> None:
    prediction = TRAKEPrediction("T1", 1, "L10_V010", (205, 105))
    dataset = _dataset("trake", _query("trake", "T1", (prediction,)))
    loaded = _roundtrip(tmp_path, dataset)
    assert loaded == dataset
    assert loaded.predictions_for("T1")[0].frame_ids == (205, 105)  # type: ignore[union-attr]


def test_trake_ordered_duplicate_semantics() -> None:
    query = _query(
        "trake",
        "T1",
        (
            TRAKEPrediction("T1", 1, "L10_V010", (100, 200)),
            TRAKEPrediction("T1", 2, "L10_V010", (200, 100)),
        ),
    )
    assert len(query.predictions) == 2
    with pytest.raises(ValueError, match="Duplicate TRAKE key"):
        _query(
            "trake",
            "T1",
            (
                TRAKEPrediction("T1", 1, "L10_V010", (100, 200)),
                TRAKEPrediction("T1", 2, "L10_V010", (100, 200)),
            ),
        )


def test_trake_expected_event_count_validation() -> None:
    query = _query(
        "trake",
        "T1",
        (TRAKEPrediction("T1", 1, "L10_V010", (100, 200)),),
    )
    validate_top100_query(query, expected_trake_event_count=2)
    with pytest.raises(ValueError, match="event-count mismatch"):
        validate_top100_query(query, expected_trake_event_count=3)
    with pytest.raises(ValueError, match="only for task 'trake'"):
        validate_top100_query(
            _query("kis", "Q1", (KISPrediction("Q1", 1, "L01_V001", 1),)),
            expected_trake_event_count=1,
        )


def test_trake_empty_and_bool_frame_ids_rejected() -> None:
    with pytest.raises(ValueError, match="non-empty tuple"):
        TRAKEPrediction("T1", 1, "L10_V010", ())
    with pytest.raises(TypeError, match="must be an integer"):
        TRAKEPrediction("T1", 1, "L10_V010", (100, True))


def test_dataset_task_mixing_duplicate_queries_and_order() -> None:
    kis = _query("kis", "Q2", (KISPrediction("Q2", 1, "L01_V001", 1),))
    qa = _query("qa", "Q1", (QAPrediction("Q1", 1, "L01_V001", 1, "yes"),))
    with pytest.raises(ValueError, match="does not match dataset task"):
        _dataset("kis", kis, qa)
    with pytest.raises(ValueError, match="duplicate dataset query_id"):
        _dataset("kis", kis, kis)
    dataset = _dataset(
        "kis",
        kis,
        _query("kis", "Q1", (KISPrediction("Q1", 1, "L01_V002", 2),)),
    )
    assert tuple(query.query_id for query in dataset.queries) == ("Q2", "Q1")


def test_expected_query_membership_and_duplicate_expectations() -> None:
    dataset = _dataset(
        "kis",
        _query("kis", "Q2", (KISPrediction("Q2", 1, "L01_V001", 1),)),
        _query("kis", "Q1", (KISPrediction("Q1", 1, "L01_V002", 2),)),
    )
    validate_top100_dataset(dataset, expected_query_ids=("Q1", "Q2"))
    with pytest.raises(ValueError, match="missing=.*Q3"):
        validate_top100_dataset(dataset, expected_query_ids=("Q1", "Q2", "Q3"))
    with pytest.raises(ValueError, match="unexpected=.*Q2"):
        validate_top100_dataset(dataset, expected_query_ids=("Q1",))
    with pytest.raises(ValueError, match="contains duplicates"):
        validate_top100_dataset(dataset, expected_query_ids=("Q1", "Q1"))


def test_empty_dataset_and_empty_query_are_in_memory_only(tmp_path: Path) -> None:
    empty_dataset = _dataset("kis")
    with pytest.raises(ValueError, match="empty Top-100 dataset"):
        write_top100_jsonl(empty_dataset, tmp_path / "empty.jsonl")
    empty_query_dataset = _dataset("kis", _query("kis", "Q1"))
    with pytest.raises(ValueError, match="zero predictions"):
        write_top100_jsonl(empty_query_dataset, tmp_path / "empty-query.jsonl")


def test_deterministic_bytes_and_failure_atomicity(tmp_path: Path) -> None:
    dataset = _dataset(
        "kis",
        _query(
            "kis",
            "Q1",
            (
                KISPrediction("Q1", 5, "L01_V002", 20),
                KISPrediction("Q1", 1, "L01_V001", 10),
            ),
        ),
    )
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    write_top100_jsonl(dataset, first)
    write_top100_jsonl(dataset, second)
    assert first.read_bytes() == second.read_bytes()
    assert hashlib.sha256(first.read_bytes()).digest() == hashlib.sha256(
        second.read_bytes()
    ).digest()
    assert first.read_bytes().endswith(b"\n") and not first.read_bytes().endswith(b"\n\n")

    destination = tmp_path / "existing.jsonl"
    destination.write_bytes(b"preexisting\n")
    before = hashlib.sha256(destination.read_bytes()).digest()
    with pytest.raises(ValueError, match="query ID set mismatch"):
        write_top100_jsonl(dataset, destination, expected_query_ids=("MISSING",))
    assert hashlib.sha256(destination.read_bytes()).digest() == before


@pytest.mark.parametrize(
    ("task", "record"),
    [
        ("kis", []),
        ("kis", {"query_id": "Q", "rank": 1, "video_id": "V"}),
        ("kis", {"query_id": "Q", "rank": 1, "video_id": "V", "frame_id": 1, "x": 2}),
        ("kis", {"query_id": "Q", "rank": 1.0, "video_id": "V", "frame_id": 1}),
        ("kis", {"query_id": "Q", "rank": True, "video_id": "V", "frame_id": 1}),
        ("kis", {"query_id": "Q", "rank": 1, "video_id": "V", "frame_id": 1.0}),
        ("kis", {"query_id": "Q", "rank": 1, "video_id": "V", "frame_id": True}),
        ("kis", {"query_id": "Q", "rank": 1, "video_id": "V", "frame_id": -1}),
        ("kis", {"query_id": "", "rank": 1, "video_id": "V", "frame_id": 1}),
        ("kis", {"query_id": "Q", "rank": 1, "video_id": "", "frame_id": 1}),
        ("qa", {"query_id": "Q", "rank": 1, "video_id": "V", "frame_id": 1}),
        ("qa", {"query_id": "Q", "rank": 1, "video_id": "V", "frame_id": 1, "answer": 2}),
        ("trake", {"query_id": "Q", "rank": 1, "video_id": "V", "frame_id": 1}),
        ("trake", {"query_id": "Q", "rank": 1, "video_id": "V", "frame_ids": (1, 2)}),
        ("trake", {"query_id": "Q", "rank": 1, "video_id": "V", "frame_ids": []}),
        ("trake", {"query_id": "Q", "rank": 1, "video_id": "V", "frame_ids": [1, True]}),
    ],
)
def test_strict_record_parser_rejects_invalid_shapes_and_types(task: str, record: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        record_to_prediction(record, task_type=task)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "payload",
    [
        "not-json\n",
        "[]\n",
        '{"query_id":"Q","rank":1,"video_id":"V"}\n',
        '{"query_id":"Q","rank":1,"video_id":"V","frame_id":1,"extra":2}\n',
    ],
)
def test_jsonl_loader_rejects_invalid_json_and_records(tmp_path: Path, payload: str) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(Top100FormatError):
        load_top100_jsonl(path, task_type="kis")
    report = validate_top100_jsonl(path, task_type="kis")
    assert report.valid is False and len(report.errors) == 1


def test_jsonl_utf8_bom_invalid_utf8_and_blank_line_policy(tmp_path: Path) -> None:
    bom = tmp_path / "bom.jsonl"
    bom.write_bytes(b"\xef\xbb\xbf{}\n")
    with pytest.raises(Top100FormatError, match="BOM"):
        load_top100_jsonl(bom, task_type="kis")
    invalid = tmp_path / "invalid-utf8.jsonl"
    invalid.write_bytes(b"\xff")
    with pytest.raises(Top100FormatError, match="UTF-8"):
        load_top100_jsonl(invalid, task_type="kis")

    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n \t\n", encoding="utf-8")
    with pytest.raises(Top100FormatError, match="no prediction records"):
        load_top100_jsonl(empty, task_type="kis")

    valid = tmp_path / "blank-lines.jsonl"
    valid.write_text(
        '\n {"query_id":"Q","rank":1,"video_id":"V","frame_id":1}\n\t\n',
        encoding="utf-8",
    )
    assert len(load_top100_jsonl(valid, task_type="kis").queries) == 1


def test_serializer_exact_shapes_and_type_consistency() -> None:
    kis = KISPrediction("Q", 1, "V", 1)
    qa = QAPrediction("Q", 1, "V", 1, "yes")
    trake = TRAKEPrediction("Q", 1, "V", (2, 1))
    assert tuple(prediction_to_record(kis, task_type="kis")) == TASK_RECORD_FIELDS["kis"]
    assert tuple(prediction_to_record(qa, task_type="qa")) == TASK_RECORD_FIELDS["qa"]
    trake_record = prediction_to_record(trake, task_type="trake")
    assert tuple(trake_record) == TASK_RECORD_FIELDS["trake"]
    assert trake_record["frame_ids"] == [2, 1]
    with pytest.raises(TypeError, match="requires QAPrediction"):
        prediction_to_record(kis, task_type="qa")


def test_loader_groups_by_first_occurrence_without_sorting(tmp_path: Path) -> None:
    path = tmp_path / "grouping.jsonl"
    records = [
        {"query_id": "Q2", "rank": 5, "video_id": "V2", "frame_id": 20},
        {"query_id": "Q1", "rank": 3, "video_id": "V1", "frame_id": 10},
        {"query_id": "Q2", "rank": 1, "video_id": "V3", "frame_id": 30},
    ]
    path.write_text(
        "\n".join(json.dumps(record, separators=(",", ":")) for record in records) + "\n",
        encoding="utf-8",
    )
    loaded = load_top100_jsonl(path, task_type="kis")
    assert tuple(query.query_id for query in loaded.queries) == ("Q2", "Q1")
    assert tuple(pred.rank for pred in loaded.predictions_for("Q2")) == (5, 1)


def test_evaluator_consumes_boundary_predictions_without_translation() -> None:
    dataset = _dataset(
        "kis",
        _query(
            "kis",
            "Q1",
            (
                KISPrediction("Q1", 1, "V", 999),
                KISPrediction("Q1", 5, "V", 105),
            ),
        ),
    )
    predictions = dataset.predictions_for("Q1")
    report = evaluate_ranked_query(
        "Q1",
        "kis",
        predictions,  # type: ignore[arg-type]
        KISGroundTruth("Q1", "V", 100, 110),
        score_kis_prediction,
    )
    assert report.r_at_1 == 0.0
    assert report.r_at_5 == 1.0


def test_expected_trake_event_count_map_and_unknown_key() -> None:
    dataset = _dataset(
        "trake",
        _query("trake", "T1", (TRAKEPrediction("T1", 1, "V", (1, 2)),)),
    )
    validate_top100_dataset(dataset, expected_trake_event_counts={"T1": 2})
    with pytest.raises(ValueError, match="event-count mismatch"):
        validate_top100_dataset(dataset, expected_trake_event_counts={"T1": 3})
    with pytest.raises(ValueError, match="unknown query IDs"):
        validate_top100_dataset(dataset, expected_trake_event_counts={"OTHER": 2})


def test_validation_report_pass_and_export_summary(tmp_path: Path) -> None:
    dataset = _dataset(
        "kis",
        _query("kis", "Q1", (KISPrediction("Q1", 1, "V", 123),)),
    )
    path = tmp_path / "checkpoint.jsonl"
    summary = write_top100_jsonl(dataset, path)
    assert summary.destination == path
    assert summary.task_type == "kis"
    assert summary.query_count == 1 and summary.record_count == 1
    assert validate_top100_jsonl(path, task_type="kis").valid is True
