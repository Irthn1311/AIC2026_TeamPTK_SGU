from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from system_tai.preliminary.evaluation import OFFICIAL_K
from system_tai.preliminary.schemas import (
    KISGroundTruth,
    KISPrediction,
    QAGroundTruth,
    QAPrediction,
    TRAKEGroundTruth,
    TRAKEPrediction,
)
from system_tai.preliminary.top100 import RankedTop100Query
from system_tai.quality import (
    DeltaClassification,
    KISQualityQuery,
    LabelOrigin,
    QAQualityQuery,
    QualityBenchmarkFormatError,
    QualityTaskType,
    TRAKEQualityQuery,
    compare_quality_reports,
    evaluate_quality_benchmark,
    load_quality_benchmark_json,
    parse_quality_benchmark_payload,
    write_quality_comparison_json,
    write_quality_report_csv,
    write_quality_report_json,
)


def _common(
    task_type: str,
    query_id: str,
    *,
    status: str = "verified",
    origin: str = "human_raw_video",
    difficulty: str = "medium",
    tags: list[str] | None = None,
    source_reference: str = "raw video V manually reviewed at original frames",
) -> dict[str, object]:
    return {
        "task_type": task_type,
        "query_id": query_id,
        "annotation_status": status,
        "label_origin": origin,
        "difficulty": difficulty,
        "tags": tags if tags is not None else ["motion", "city"],
        "annotation_notes": "synthetic unit-test record" if origin == "synthetic" else "",
        "source_reference": source_reference,
    }


def _kis_payload(
    query_id: str = "kis-1",
    *,
    status: str = "verified",
    origin: str = "human_raw_video",
    ground_truth: object = ...,
) -> dict[str, object]:
    item = {
        **_common("kis", query_id, status=status, origin=origin),
        "query_vi": "một xe màu xanh",
        "query_en": "a blue car",
        "query_en_expansion": None,
    }
    item["ground_truth"] = (
        {"video_id": "V", "start_frame_id": 10, "end_frame_id": 20}
        if ground_truth is ...
        else ground_truth
    )
    return item


def _qa_payload(query_id: str = "qa-1") -> dict[str, object]:
    return {
        **_common("qa", query_id, difficulty="easy", tags=["color", "city"]),
        "event_description": "một chiếc xe dừng lại",
        "question": "Xe màu gì?",
        "event_description_en": "a car stops",
        "question_en": "What color is the car?",
        "ground_truth": {
            "video_id": "V",
            "start_frame_id": 30,
            "end_frame_id": 40,
            "accepted_answers": ["xanh", "blue"],
        },
    }


def _trake_payload(query_id: str = "trake-1") -> dict[str, object]:
    return {
        **_common("trake", query_id, difficulty="hard", tags=["sequence", "city"]),
        "events": [
            {"description": "sự kiện hai", "description_en": "event two"},
            {"description": "sự kiện một", "description_en": "event one"},
        ],
        "ground_truth": {
            "video_id": "V",
            "event_intervals": [[50, 60], [70, 80]],
        },
    }


def _payload(*queries: dict[str, object], benchmark_id: str = "quality-fixture") -> dict:
    return {
        "schema_version": 1,
        "benchmark_id": benchmark_id,
        "description": "synthetic mechanics only",
        "queries": list(queries),
    }


def _three_task_benchmark(*, include_draft: bool = False, include_synthetic: bool = False):
    queries = [_kis_payload(), _qa_payload(), _trake_payload()]
    if include_draft:
        draft = _kis_payload("draft", status="draft", origin="unlabeled", ground_truth=None)
        draft["source_reference"] = ""
        queries.append(draft)
    if include_synthetic:
        queries.append(_kis_payload("synthetic", origin="synthetic"))
    return parse_quality_benchmark_payload(_payload(*queries))


def _predictions(
    *,
    kis_hit: bool = True,
    trake_frames: tuple[int, int] = (55, 999),
    include_synthetic: bool = False,
) -> tuple[RankedTop100Query, ...]:
    values = [
        RankedTop100Query(
            "kis",
            "kis-1",
            (KISPrediction("kis-1", 1, "V", 15 if kis_hit else 999),),
        ),
        RankedTop100Query(
            "qa",
            "qa-1",
            (QAPrediction("qa-1", 1, "V", 35, "BLUE!"),),
        ),
        RankedTop100Query(
            "trake",
            "trake-1",
            (TRAKEPrediction("trake-1", 1, "V", trake_frames),),
        ),
    ]
    if include_synthetic:
        values.append(
            RankedTop100Query(
                "kis",
                "synthetic",
                (KISPrediction("synthetic", 1, "V", 15),),
            )
        )
    return tuple(values)


def test_schema_1_valid_kis_verified_human_raw_video() -> None:
    query = parse_quality_benchmark_payload(_payload(_kis_payload())).queries[0]
    assert type(query) is KISQualityQuery
    assert type(query.ground_truth) is KISGroundTruth
    assert query.ground_truth.start_frame_id == 10


def test_schema_2_valid_qa_verified_human_raw_video() -> None:
    query = parse_quality_benchmark_payload(_payload(_qa_payload())).queries[0]
    assert type(query) is QAQualityQuery
    assert type(query.ground_truth) is QAGroundTruth
    assert query.ground_truth.accepted_answers == ("xanh", "blue")


def test_schema_3_valid_trake_verified_human_raw_video() -> None:
    query = parse_quality_benchmark_payload(_payload(_trake_payload())).queries[0]
    assert type(query) is TRAKEQualityQuery
    assert type(query.ground_truth) is TRAKEGroundTruth
    assert query.ground_truth.event_intervals == ((50, 60), (70, 80))


def test_schema_4_draft_query_with_null_gt_is_accepted() -> None:
    query = _kis_payload(status="draft", origin="unlabeled", ground_truth=None)
    query["source_reference"] = ""
    parsed = parse_quality_benchmark_payload(_payload(query))
    assert parsed.queries[0].ground_truth is None


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda item: item.update(ground_truth=None), "requires ground_truth"),
        (lambda item: item.update(label_origin="unlabeled"), "cannot use label_origin"),
        (lambda item: item.update(source_reference=""), "source_reference"),
    ],
)
def test_schema_5_to_7_verified_contract_rejections(mutation, message: str) -> None:
    query = _kis_payload()
    mutation(query)
    with pytest.raises(QualityBenchmarkFormatError, match=message):
        parse_quality_benchmark_payload(_payload(query))


def test_schema_8_synthetic_verified_parses() -> None:
    query = parse_quality_benchmark_payload(
        _payload(_kis_payload(origin="synthetic"))
    ).queries[0]
    assert query.label_origin is LabelOrigin.SYNTHETIC


def test_schema_9_duplicate_query_id_rejected() -> None:
    with pytest.raises(QualityBenchmarkFormatError, match="globally unique"):
        parse_quality_benchmark_payload(_payload(_kis_payload(), _qa_payload("kis-1")))


def test_schema_10_duplicate_tags_rejected() -> None:
    query = _kis_payload()
    query["tags"] = ["same", "same"]
    with pytest.raises(QualityBenchmarkFormatError, match="tags must be unique"):
        parse_quality_benchmark_payload(_payload(query))


@pytest.mark.parametrize("location", ["top", "query", "ground_truth"])
def test_schema_11_unknown_fields_rejected(location: str) -> None:
    query = _kis_payload()
    payload = _payload(query)
    if location == "top":
        payload["unknown"] = True
    elif location == "query":
        query["unknown"] = True
    else:
        query["ground_truth"]["unknown"] = True
    with pytest.raises(QualityBenchmarkFormatError, match="unknown"):
        parse_quality_benchmark_payload(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("task_type", "unknown"),
        ("annotation_status", "accepted"),
        ("label_origin", "retrieval_inspection"),
        ("difficulty", "impossible"),
    ],
)
def test_schema_invalid_enum_values_are_rejected(field: str, value: str) -> None:
    query = _kis_payload()
    query[field] = value
    with pytest.raises(QualityBenchmarkFormatError, match="invalid"):
        parse_quality_benchmark_payload(_payload(query))


def test_schema_missing_required_field_is_rejected() -> None:
    query = _qa_payload()
    del query["question_en"]
    with pytest.raises(QualityBenchmarkFormatError, match="missing"):
        parse_quality_benchmark_payload(_payload(query))


def test_schema_12_bom_rejected(tmp_path: Path) -> None:
    path = tmp_path / "benchmark.json"
    path.write_bytes(b"\xef\xbb\xbf{}")
    with pytest.raises(QualityBenchmarkFormatError, match="BOM"):
        load_quality_benchmark_json(path)


def test_schema_13_invalid_utf8_rejected(tmp_path: Path) -> None:
    path = tmp_path / "benchmark.json"
    path.write_bytes(b"\xff")
    with pytest.raises(QualityBenchmarkFormatError, match="UTF-8"):
        load_quality_benchmark_json(path)


@pytest.mark.parametrize("field", ["start_frame_id", "end_frame_id"])
def test_schema_14_bool_frame_ids_rejected(field: str) -> None:
    query = _kis_payload()
    query["ground_truth"][field] = True
    with pytest.raises(QualityBenchmarkFormatError, match="must be an integer"):
        parse_quality_benchmark_payload(_payload(query))


def test_schema_15_trake_event_count_mismatch_rejected() -> None:
    query = _trake_payload()
    query["ground_truth"]["event_intervals"] = [[50, 60]]
    with pytest.raises(QualityBenchmarkFormatError, match="equal length"):
        parse_quality_benchmark_payload(_payload(query))


def test_schema_16_gt_query_id_is_constructed_from_enclosing_query() -> None:
    benchmark = _three_task_benchmark()
    assert [query.ground_truth.query_id for query in benchmark.queries] == [
        "kis-1",
        "qa-1",
        "trake-1",
    ]


def test_schema_17_event_order_is_preserved() -> None:
    query = parse_quality_benchmark_payload(_payload(_trake_payload())).queries[0]
    assert [event.description for event in query.events] == [
        "sự kiện hai",
        "sự kiện một",
    ]
    assert query.ground_truth.event_intervals == ((50, 60), (70, 80))


def test_schema_18_tag_physical_order_is_preserved() -> None:
    query = _kis_payload()
    query["tags"] = ["z-last", "a-first"]
    parsed = parse_quality_benchmark_payload(_payload(query))
    assert parsed.queries[0].tags == ("z-last", "a-first")


def test_json_loader_preserves_unicode(tmp_path: Path) -> None:
    path = tmp_path / "quality.json"
    path.write_text(
        json.dumps(_payload(_kis_payload()), ensure_ascii=False),
        encoding="utf-8",
    )
    benchmark = load_quality_benchmark_json(path)
    assert benchmark.queries[0].query_vi == "một xe màu xanh"


def test_eval_1_to_3_reuses_p0_metrics_for_all_tasks() -> None:
    report = evaluate_quality_benchmark(_three_task_benchmark(), _predictions())
    by_id = {item.query_id: item.evaluation for item in report.query_reports}
    assert tuple(getattr(by_id["kis-1"], f"r_at_{k}") for k in OFFICIAL_K) == (
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    )
    assert by_id["qa-1"].final_score == 1.0
    assert by_id["trake-1"].per_rank_r_scores == (0.5,)
    assert by_id["trake-1"].final_score == 0.5


def test_eval_4_explicit_zero_prediction_query_scores_zero() -> None:
    benchmark = parse_quality_benchmark_payload(_payload(_kis_payload()))
    predictions = (RankedTop100Query("kis", "kis-1", ()),)
    report = evaluate_quality_benchmark(benchmark, predictions)
    assert report.query_reports[0].evaluation.prediction_count == 0
    assert report.query_reports[0].evaluation.final_score == 0.0


def test_eval_5_draft_is_excluded_and_counted() -> None:
    benchmark = _three_task_benchmark(include_draft=True)
    report = evaluate_quality_benchmark(benchmark, _predictions())
    assert report.scored_query_count == 3
    assert report.skipped_draft_count == 1
    assert "draft" not in {item.query_id for item in report.query_reports}


def test_eval_6_and_7_synthetic_is_opt_in() -> None:
    benchmark = _three_task_benchmark(include_synthetic=True)
    default = evaluate_quality_benchmark(benchmark, _predictions())
    assert default.skipped_synthetic_count == 1
    assert default.scored_query_count == 3
    included = evaluate_quality_benchmark(
        benchmark,
        _predictions(include_synthetic=True),
        include_synthetic=True,
    )
    assert included.skipped_synthetic_count == 0
    assert included.scored_query_count == 4


def test_eval_8_missing_scored_prediction_rejected() -> None:
    with pytest.raises(ValueError, match="missing"):
        evaluate_quality_benchmark(_three_task_benchmark(), _predictions()[:-1])


def test_eval_9_unexpected_prediction_rejected() -> None:
    unexpected = RankedTop100Query(
        "kis", "unexpected", (KISPrediction("unexpected", 1, "V", 1),)
    )
    with pytest.raises(ValueError, match="unexpected"):
        evaluate_quality_benchmark(
            _three_task_benchmark(),
            (*_predictions(), unexpected),
        )


def test_eval_10_task_mismatch_rejected() -> None:
    wrong = RankedTop100Query(
        "qa", "kis-1", (QAPrediction("kis-1", 1, "V", 15, "blue"),)
    )
    with pytest.raises(ValueError, match="task mismatch"):
        evaluate_quality_benchmark(
            _three_task_benchmark(),
            (wrong, *_predictions()[1:]),
        )


def test_eval_11_duplicate_prediction_query_rejected() -> None:
    values = _predictions()
    with pytest.raises(ValueError, match="duplicate prediction"):
        evaluate_quality_benchmark(_three_task_benchmark(), (*values, values[0]))


def test_eval_12_task_summaries_are_correct() -> None:
    report = evaluate_quality_benchmark(_three_task_benchmark(), _predictions())
    means = {item.task_type: item.mean_final_score for item in report.task_summaries}
    assert means == {
        QualityTaskType.KIS: 1.0,
        QualityTaskType.QA: 1.0,
        QualityTaskType.TRAKE: 0.5,
    }


def test_eval_13_and_14_query_macro_and_task_macro_are_distinct() -> None:
    second = _kis_payload("kis-2")
    benchmark = parse_quality_benchmark_payload(
        _payload(_kis_payload(), second, _qa_payload(), _trake_payload())
    )
    predictions = (
        *_predictions(),
        RankedTop100Query("kis", "kis-2", ()),
    )
    report = evaluate_quality_benchmark(benchmark, predictions)
    assert report.overall_query_macro_score == pytest.approx(0.625)
    assert report.task_macro_score == pytest.approx(2.0 / 3.0)


def test_eval_15_and_16_breakdowns_are_deterministic_and_correct() -> None:
    report = evaluate_quality_benchmark(_three_task_benchmark(), _predictions())
    dimensions = [(item.dimension, item.value) for item in report.breakdowns]
    assert dimensions == [
        ("difficulty", "easy"),
        ("difficulty", "medium"),
        ("difficulty", "hard"),
        ("tag", "city"),
        ("tag", "color"),
        ("tag", "motion"),
        ("tag", "sequence"),
    ]
    city = next(item for item in report.breakdowns if item.value == "city")
    assert city.query_count == 3
    assert city.mean_final_score == pytest.approx(5.0 / 6.0)


def _reports_for_comparison():
    benchmark = _three_task_benchmark()
    baseline = evaluate_quality_benchmark(
        benchmark,
        _predictions(kis_hit=False, trake_frames=(55, 75)),
    )
    candidate = evaluate_quality_benchmark(
        benchmark,
        _predictions(kis_hit=True, trake_frames=(55, 999)),
    )
    return baseline, candidate


def test_compare_1_same_report_is_all_tied() -> None:
    baseline, _ = _reports_for_comparison()
    comparison = compare_quality_reports(baseline, baseline)
    assert comparison.improved_count == 0
    assert comparison.tied_count == 3
    assert comparison.regressed_count == 0


def test_compare_2_to_6_mixed_deltas_and_summaries_are_correct() -> None:
    baseline, candidate = _reports_for_comparison()
    comparison = compare_quality_reports(
        baseline,
        candidate,
        baseline_label="frozen",
        candidate_label="experiment",
    )
    assert (comparison.improved_count, comparison.tied_count, comparison.regressed_count) == (
        1,
        1,
        1,
    )
    deltas = {item.query_id: item for item in comparison.query_deltas}
    assert deltas["kis-1"].classification is DeltaClassification.IMPROVED
    assert deltas["qa-1"].classification is DeltaClassification.TIED
    assert deltas["trake-1"].classification is DeltaClassification.REGRESSED
    assert comparison.overall_delta == pytest.approx(1.0 / 6.0)
    task_deltas = {item.task_type: item.delta for item in comparison.task_deltas}
    assert task_deltas[QualityTaskType.KIS] == 1.0
    assert task_deltas[QualityTaskType.QA] == 0.0
    assert task_deltas[QualityTaskType.TRAKE] == -0.5


def test_compare_7_benchmark_id_mismatch_rejected() -> None:
    baseline, candidate = _reports_for_comparison()
    with pytest.raises(ValueError, match="benchmark_id"):
        compare_quality_reports(baseline, replace(candidate, benchmark_id="other"))


def test_compare_8_query_set_mismatch_rejected() -> None:
    baseline, candidate = _reports_for_comparison()
    with pytest.raises(ValueError, match="query ID set"):
        compare_quality_reports(
            baseline,
            replace(candidate, query_reports=candidate.query_reports[:-1]),
        )


def test_compare_9_task_mismatch_rejected() -> None:
    baseline, candidate = _reports_for_comparison()
    first = replace(candidate.query_reports[0], task_type=QualityTaskType.QA)
    with pytest.raises(ValueError, match="task mismatch"):
        compare_quality_reports(
            baseline,
            replace(candidate, query_reports=(first, *candidate.query_reports[1:])),
        )


def test_report_1_to_6_are_deterministic_utf8_complete_and_timestamp_free(
    tmp_path: Path,
) -> None:
    baseline, candidate = _reports_for_comparison()
    comparison = compare_quality_reports(baseline, candidate)
    json_a = write_quality_report_json(baseline, tmp_path / "a.json")
    json_b = write_quality_report_json(baseline, tmp_path / "b.json")
    csv_a = write_quality_report_csv(baseline, tmp_path / "a.csv")
    csv_b = write_quality_report_csv(baseline, tmp_path / "b.csv")
    comp_a = write_quality_comparison_json(comparison, tmp_path / "ca.json")
    comp_b = write_quality_comparison_json(comparison, tmp_path / "cb.json")
    assert json_a.read_bytes() == json_b.read_bytes()
    assert csv_a.read_bytes() == csv_b.read_bytes()
    assert comp_a.read_bytes() == comp_b.read_bytes()
    for path in (json_a, csv_a, comp_a):
        payload = path.read_bytes()
        assert payload.endswith(b"\n")
        assert b"timestamp" not in payload.lower()
    text = json_a.read_text(encoding="utf-8")
    assert "frozen" not in text
    assert all(f'"r_at_{k}"' in text for k in OFFICIAL_K)
    header = csv_a.read_text(encoding="utf-8").splitlines()[0]
    assert all(f"r_at_{k}" in header for k in OFFICIAL_K)
    assert "city|motion" not in csv_a.read_text(encoding="utf-8")
    assert "motion|city" in csv_a.read_text(encoding="utf-8")


def test_quality_package_does_not_import_historical_phase25_package() -> None:
    quality_root = Path(__file__).parents[1] / "src" / "system_tai" / "quality"
    sources = "\n".join(path.read_text(encoding="utf-8") for path in quality_root.glob("*.py"))
    assert "system_tai.evaluation" not in sources
