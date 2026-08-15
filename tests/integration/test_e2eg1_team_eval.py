from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from triage_eg.e2e1.pipeline import PredictionResult
from triage_eg.e2eg1 import (
    VARIANTS,
    combine_prediction_variants,
    compare_variants,
    decisions,
    evaluate_finalized,
    formal_report_lines,
    post_inference_diagnostics,
    run_prediction_variant,
    runtime_summary,
)


def _queries() -> list[dict]:
    return [
        {"query_id": "K1", "task": "KIS", "query": "person", "language": "en"},
        {
            "query_id": "Q1",
            "task": "QA",
            "query": "red car",
            "question": "what color",
            "language": "en",
        },
        {
            "query_id": "T1",
            "task": "TRAKE",
            "query": "events",
            "event_count": 2,
            "event_descriptions": [
                {"event_id": "E1", "description": "first"},
                {"event_id": "E2", "description": "second"},
            ],
            "language": "en",
        },
    ]


def _gt() -> list[dict]:
    return [
        {
            "query_id": "K1",
            "task": "KIS",
            "correct_video": "L01_V001",
            "acceptable_intervals": [[9, 11]],
        },
        {
            "query_id": "Q1",
            "task": "QA",
            "correct_video": "L01_V001",
            "acceptable_intervals": [[9, 11]],
            "accepted_answers": ["red"],
        },
        {
            "query_id": "T1",
            "task": "TRAKE",
            "correct_video": "L01_V001",
            "event_intervals": [[9, 11], [19, 21]],
        },
    ]


class FakePipeline:
    def __init__(self) -> None:
        self.calls = []

    def runtime_diagnostics(self) -> dict:
        return {
            "m1_call_count": 0,
            "m1_cache_hits": 0,
            "raw_decoded_frames": 0,
            "raw_clip_encode_count": 0,
            "refined_alternative_count": 0,
            "refined_duplicate_dropped_count": 0,
            "refined_order_invalid_dropped_count": 0,
            "qa_machine_ids_filtered": 0,
        }

    def predict_queries(self, queries: list[dict], variant: str) -> list[PredictionResult]:
        self.calls.append(variant)
        results = []
        for query in queries:
            if query["task"] == "KIS":
                rows = ({"query_id": "K1", "rank": 1, "video_id": "L01_V001", "frame_id": 10},)
            elif query["task"] == "QA":
                rows = (
                    {
                        "query_id": "Q1",
                        "rank": 1,
                        "video_id": "L01_V001",
                        "frame_id": 10,
                        "answer": "red",
                    },
                )
            else:
                rows = (
                    {
                        "query_id": "T1",
                        "rank": 1,
                        "video_id": "L01_V001",
                        "frame_ids": [10, 20],
                    },
                )
            results.append(
                PredictionResult(
                    {"query_id": query["query_id"], "task": query["task"]}, rows, (), 0.01
                )
            )
        return results


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _run(tmp_path: Path, benchmark_id: str = "DEV_CROSS_60"):
    inference = tmp_path / "inference"
    _write_jsonl(inference / "queries.jsonl", _queries())
    pipeline = FakePipeline()
    runs = [
        run_prediction_variant(pipeline, inference, benchmark_id, variant, tmp_path / "output")
        for variant in VARIANTS
    ]
    return pipeline, combine_prediction_variants(*runs)


def test_all_three_variants_finalize_and_pass_shared_validator(tmp_path: Path) -> None:
    pipeline, run = _run(tmp_path)
    assert pipeline.calls == list(VARIANTS)
    assert set(run["variants"]) == set(VARIANTS)
    assert all(value["validation"]["status"] == "PASS" for value in run["variants"].values())
    assert all(value["sha256"] for value in run["variants"].values())
    assert {path.name for path in (tmp_path / "output/predictions").iterdir()} == {
        "dev_cross_60_g0.jsonl",
        "dev_cross_60_g1.jsonl",
        "dev_cross_60_g2.jsonl",
    }


def test_gt_is_loaded_only_after_all_hashes_exist(tmp_path: Path) -> None:
    _, run = _run(tmp_path)
    benchmark = tmp_path / "benchmark"
    _write_jsonl(benchmark / "gt.jsonl", _gt())
    run["variants"]["G2_SAFE_M1"]["sha256"] = ""
    with pytest.raises(RuntimeError, match="PREDICTIONS_NOT_FINALIZED"):
        evaluate_finalized(run, benchmark, "DEV_CROSS_60", tmp_path / "output")


def test_shared_evaluator_reports_cross_variants_separately(tmp_path: Path) -> None:
    _, run = _run(tmp_path)
    benchmark = tmp_path / "benchmark"
    _write_jsonl(benchmark / "gt.jsonl", _gt())
    evaluation = evaluate_finalized(run, benchmark, "DEV_CROSS_60", tmp_path / "output")
    assert set(evaluation) == set(VARIANTS)
    assert all(value["summary"]["final_score"] == 1.0 for value in evaluation.values())
    assert (tmp_path / "output/evaluation/cross_g0_summary.json").is_file()
    assert (tmp_path / "output/evaluation/cross_g2_slices.json").is_file()
    decision = decisions(
        {"DEV_CROSS_60": evaluation},
        compare_variants({"DEV_CROSS_60": evaluation}),
    )
    assert decision["g1_coverage_decision"] == "DROP"
    assert decision["selected_grounding_policy"] == "G0_COARSE"
    assert decision["selected_system_variant"] == "G0_E2E1_COARSE"


def test_cross_and_l21_comparisons_remain_distinct(tmp_path: Path) -> None:
    evaluations = {}
    for benchmark_id in ("DEV_CROSS_60", "DEV_L21_150"):
        sub = tmp_path / benchmark_id
        _, run = _run(sub, benchmark_id)
        benchmark = sub / "benchmark"
        _write_jsonl(benchmark / "gt.jsonl", _gt())
        evaluations[benchmark_id] = evaluate_finalized(run, benchmark, benchmark_id, sub / "output")
    comparison = compare_variants(evaluations)
    assert set(comparison) == {"DEV_CROSS_60", "DEV_L21_150"}
    assert comparison["DEV_CROSS_60"]["GROUNDING_POLICY_DELTA"]["attribution"] == (
        "G1_COVERAGE_ALLOCATION_ONLY"
    )
    assert comparison["DEV_CROSS_60"]["QA_HYGIENE_DELTA"]["policy"].startswith("COMMON_")
    assert (tmp_path / "DEV_L21_150/output/evaluation/l21_g1_summary.json").is_file()


def test_g0_g1_g2_prediction_contracts_are_identical_when_hypotheses_match(
    tmp_path: Path,
) -> None:
    _, run = _run(tmp_path)
    values = [run["variants"][variant]["predictions"] for variant in VARIANTS]
    assert values[0] == values[1] == values[2]


def test_post_gt_trake_safety_diagnostic_accepts_one_interval_per_event(
    tmp_path: Path,
) -> None:
    _, run = _run(tmp_path)
    benchmark = tmp_path / "benchmark"
    _write_jsonl(benchmark / "gt.jsonl", _gt())
    evaluation = evaluate_finalized(run, benchmark, "DEV_CROSS_60", tmp_path / "output")
    results = run["variants"]["G2_SAFE_M1"]["results"]
    for index, result in enumerate(results):
        if result.query_plan["task"] != "TRAKE":
            continue
        results[index] = PredictionResult(
            result.query_plan,
            result.predictions,
            (
                {
                    "diagnostic_type": "trake_dual_hypothesis",
                    "query_id": "T1",
                    "hypothesis_kind": "M1_REFINED_ALTERNATIVE",
                    "video_id": "L01_V001",
                    "source_coarse_frame_ids": [10, 20],
                    "refined_frame_ids": [50, 60],
                    "emitted": True,
                },
            ),
            result.latency_seconds,
        )
    post = post_inference_diagnostics(
        run, evaluation, benchmark, "DEV_CROSS_60", tmp_path / "output"
    )
    safety = post["m1_safety"]
    assert safety["trake_source_chain_count"] == 1
    assert safety["trake_source_event_hits_preserved_vs_destructive"] == 2


def test_runtime_and_formal_report_contracts_are_complete(tmp_path: Path) -> None:
    pipeline, run = _run(tmp_path)
    pipeline.runtime = SimpleNamespace(runtime_manifest=lambda: {"devices": {"clip": "cpu"}})
    for value in run["variants"].values():
        for row in value["predictions"]:
            if row.get("query_id") == "Q1":
                row["answer"] = "2024"
    benchmark = tmp_path / "benchmark"
    _write_jsonl(benchmark / "gt.jsonl", _gt())
    evaluation = evaluate_finalized(run, benchmark, "DEV_CROSS_60", tmp_path / "output")
    post = post_inference_diagnostics(
        run, evaluation, benchmark, "DEV_CROSS_60", tmp_path / "output"
    )
    evaluations = {"DEV_CROSS_60": evaluation}
    comparison = compare_variants(evaluations)
    decision = decisions(evaluations, comparison)
    runtime = runtime_summary({"DEV_CROSS_60": run}, pipeline, 0.25)
    assert runtime["qa_opaque_machine_id_output_count"] == 0
    assert runtime["qa_opaque_machine_id_output_samples"] == []
    lines = formal_report_lines(
        git_commit="abc123",
        evaluations=evaluations,
        comparison=comparison,
        post={"DEV_CROSS_60": post},
        runtime=runtime,
        decision=decision,
        zip_path="/kaggle/working/triage_eg_e2eg1_v01_bundle.zip",
    )
    assert "STARTUP_SECONDS=0.25" in lines
    assert "RAW_DECODED_FRAMES=0" in lines
    assert "RAW_CLIP_ENCODE_COUNT=0" in lines
    assert "SELECTED_GROUNDING_POLICY=G0_COARSE" in lines
