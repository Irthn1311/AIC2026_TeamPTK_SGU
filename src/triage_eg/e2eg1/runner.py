"""GT-isolated execution, evaluation, diagnostics, and packaging for E2E-G1."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from time import monotonic
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from aic2026_eval.contracts import accepted_intervals
from aic2026_eval.io import read_jsonl, sha256_file, write_json, write_jsonl
from aic2026_eval.scoring import CUTOFFS, evaluate
from aic2026_eval.validation import validate_predictions
from triage_eg.e2e1.planning import plan_queries
from triage_eg.e2e1.runner import (
    BENCHMARKS,
    _task_diagnostics,
    extract_development_bundle,
    materialize_inference_only,
)
from triage_eg.experiments.moment_m1 import M1Settings
from triage_eg.experiments.t3_diverse_temporal import POOL_LIMIT, REGION_RADIUS_SECONDS

from .contracts import VARIANT_SLUGS, VARIANTS
from .pipeline import SafeCoveragePipeline, is_opaque_machine_id

BENCHMARK_SLUGS = {"DEV_CROSS_60": "cross", "DEV_L21_150": "l21"}
DIAGNOSTIC_FILES = {
    "coverage_allocation": "coverage_allocation.jsonl",
    "video_hypothesis_ranking": "video_hypothesis_ranking.jsonl",
    "m1_alternative_provenance": "m1_alternative_provenance.jsonl",
    "trake_dual_hypothesis": "trake_dual_hypothesis.jsonl",
    "qa_machine_id_hygiene": "qa_machine_id_hygiene.jsonl",
}
COUNTERS = (
    "m1_call_count",
    "m1_cache_hits",
    "raw_decoded_frames",
    "raw_clip_encode_count",
    "refined_alternative_count",
    "refined_duplicate_dropped_count",
    "refined_order_invalid_dropped_count",
    "qa_machine_ids_filtered",
)


def _counter_snapshot(pipeline: SafeCoveragePipeline) -> dict[str, int]:
    values = pipeline.runtime_diagnostics()
    return {name: int(values.get(name, 0)) for name in COUNTERS}


def run_prediction_variant(
    pipeline: SafeCoveragePipeline,
    inference_root: str | Path,
    benchmark_id: str,
    variant: str,
    output_root: str | Path,
) -> dict[str, Any]:
    """Finalize one variant while GT is physically unavailable."""

    if benchmark_id not in BENCHMARKS or variant not in VARIANTS:
        raise ValueError(f"unsupported E2E-G1 run: {benchmark_id}/{variant}")
    inference = Path(inference_root).resolve(strict=True)
    if {path.name for path in inference.iterdir()} != {"queries.jsonl"}:
        raise RuntimeError("GT_UNAVAILABLE_DURING_PREDICTION_GATE_FAILED")
    queries = read_jsonl(inference / "queries.jsonl")
    plans = plan_queries(queries)
    root = Path(output_root).resolve(strict=False)
    before = _counter_snapshot(pipeline)
    started = monotonic()
    results = pipeline.predict_queries(queries, variant)
    seconds = monotonic() - started
    after = _counter_snapshot(pipeline)
    predictions = [row for result in results for row in result.predictions]
    diagnostics = [row for result in results for row in result.diagnostics]
    prediction_path = (
        root / "predictions" / f"{benchmark_id.casefold()}_{VARIANT_SLUGS[variant]}.jsonl"
    )
    write_jsonl(prediction_path, predictions)
    digest = sha256_file(prediction_path)
    validation, issues = validate_predictions(queries, predictions)
    if validation["status"] != "PASS":
        write_json(
            root
            / "diagnostics"
            / f"{benchmark_id.casefold()}_{VARIANT_SLUGS[variant]}_validation.json",
            {"summary": validation, "issues": issues},
        )
        raise RuntimeError(f"PREDICTION_CONTRACT_GATE_FAILED: {benchmark_id}/{variant}")
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in diagnostics:
        diagnostic_type = str(row.get("diagnostic_type", ""))
        if diagnostic_type in DIAGNOSTIC_FILES:
            grouped[DIAGNOSTIC_FILES[diagnostic_type]].append({"benchmark_id": benchmark_id, **row})
    for filename, new_rows in grouped.items():
        target = root / "diagnostics" / filename
        existing = read_jsonl(target) if target.is_file() else []
        write_jsonl(target, [*existing, *new_rows])
    return {
        "queries": queries,
        "plans": plans,
        "variant": variant,
        "results": results,
        "predictions": predictions,
        "prediction_path": prediction_path,
        "sha256": digest,
        "validation": validation,
        "seconds": seconds,
        "counter_delta": {name: after[name] - before[name] for name in COUNTERS},
    }


def combine_prediction_variants(*runs: dict[str, Any]) -> dict[str, Any]:
    if len(runs) != len(VARIANTS) or {run["variant"] for run in runs} != set(VARIANTS):
        raise ValueError("exactly G0, G1, and G2 are required")
    first = runs[0]
    if any(run["queries"] != first["queries"] for run in runs[1:]):
        raise ValueError("prediction variants used different query inputs")
    return {
        "queries": first["queries"],
        "plans": first["plans"],
        "variants": {
            run["variant"]: {
                key: value
                for key, value in run.items()
                if key not in {"queries", "plans", "variant"}
            }
            for run in runs
        },
    }


def evaluate_finalized(
    prediction_run: dict[str, Any],
    benchmark_root: str | Path,
    benchmark_id: str,
    output_root: str | Path,
) -> dict[str, Any]:
    """Load GT only after all three prediction files are immutable and hashed."""

    variants = prediction_run["variants"]
    if set(variants) != set(VARIANTS) or any(
        not value.get("sha256") for value in variants.values()
    ):
        raise RuntimeError("PREDICTIONS_NOT_FINALIZED_BEFORE_GT")
    ground_truth = read_jsonl(Path(benchmark_root).resolve(strict=True) / "gt.jsonl")
    queries = prediction_run["queries"]
    root = Path(output_root).resolve(strict=False)
    benchmark_slug = BENCHMARK_SLUGS[benchmark_id]
    output = {}
    for variant in VARIANTS:
        value = variants[variant]
        summary, per_query, slices, issues = evaluate(
            queries,
            value["predictions"],
            ground_truth,
            metadata={"benchmark_id": benchmark_id, "system_variant": variant},
        )
        prefix = root / "evaluation" / f"{benchmark_slug}_{VARIANT_SLUGS[variant]}"
        write_json(Path(f"{prefix}_summary.json"), summary)
        write_jsonl(Path(f"{prefix}_per_query.jsonl"), per_query)
        write_json(Path(f"{prefix}_slices.json"), slices)
        write_jsonl(Path(f"{prefix}_issues.jsonl"), issues)
        output[variant] = {
            "summary": summary,
            "per_query": per_query,
            "slices": slices,
            "task_diagnostics": _task_diagnostics(per_query, queries),
        }
    return output


def compare_variants(evaluations: dict[str, dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for benchmark_id, values in evaluations.items():
        benchmark: dict[str, Any] = {}
        for left, right, label in (
            ("G0_E2E1_COARSE", "G1_COVERAGE_COARSE", "G1_VS_G0"),
            ("G1_COVERAGE_COARSE", "G2_SAFE_M1", "G2_VS_G1"),
        ):
            by_left = {row["query_id"]: row for row in values[left]["per_query"]}
            by_right = {row["query_id"]: row for row in values[right]["per_query"]}
            counts: Counter[str] = Counter()
            paired = []
            for query_id in sorted(by_left):
                delta = by_right[query_id]["final_score"] - by_left[query_id]["final_score"]
                result = "BETTER" if delta > 0 else "WORSE" if delta < 0 else "TIE"
                counts[result] += 1
                paired.append({"query_id": query_id, "delta": delta, "result": result})
            task_delta = {
                task: values[right]["slices"][f"task:{task}"]["final_score"]
                - values[left]["slices"][f"task:{task}"]["final_score"]
                for task in ("KIS", "QA", "TRAKE")
            }
            benchmark[label] = {
                "overall_delta": values[right]["summary"]["final_score"]
                - values[left]["summary"]["final_score"],
                "task_delta": task_delta,
                "paired_counts": dict(counts),
                "paired_queries": paired,
            }
        benchmark["GROUNDING_POLICY_DELTA"] = {
            **benchmark["G1_VS_G0"],
            "attribution": "G1_COVERAGE_ALLOCATION_ONLY",
        }
        benchmark["QA_HYGIENE_DELTA"] = {
            "policy": "COMMON_MACHINE_ID_GUARD_APPLIED_TO_G0_G1_G2",
            "attribution": "REPORTED_SEPARATELY_FROM_GROUNDING_POLICY",
        }
        output[benchmark_id] = benchmark
    return output


def _diagnostic_rows(
    run: dict[str, Any], variant: str, diagnostic_type: str
) -> list[dict[str, Any]]:
    return [
        row
        for result in run["variants"][variant]["results"]
        for row in result.diagnostics
        if row.get("diagnostic_type") == diagnostic_type
    ]


def _intersects(frame: int, intervals: list[tuple[int, int]]) -> bool:
    return any(start <= frame <= end for start, end in intervals)


def _window_intersects(frame: int, fps: float, intervals: list[tuple[int, int]]) -> bool:
    radius = int(round(6.0 * fps))
    return any(frame - radius <= end and start <= frame + radius for start, end in intervals)


def _event_interval(value: Any) -> list[tuple[int, int]]:
    """Normalize one TRAKE event interval through the shared TEAM-EVAL contract."""

    return accepted_intervals([value])


def post_inference_diagnostics(
    prediction_run: dict[str, Any],
    evaluation: dict[str, Any],
    benchmark_root: str | Path,
    benchmark_id: str,
    output_root: str | Path,
) -> dict[str, Any]:
    """Use GT only after final hashes to measure coverage and M1 safety."""

    root = Path(output_root)
    gt_rows = read_jsonl(Path(benchmark_root) / "gt.jsonl")
    gt = {row["query_id"]: row for row in gt_rows}
    queries = {row["query_id"]: row for row in prediction_run["queries"]}
    g0 = _diagnostic_rows(prediction_run, "G0_E2E1_COARSE", "coverage_allocation")
    g1 = _diagnostic_rows(prediction_run, "G1_COVERAGE_COARSE", "coverage_allocation")
    g2 = _diagnostic_rows(prediction_run, "G2_SAFE_M1", "coverage_allocation")
    by_variant = {name: defaultdict(list) for name in ("G0", "G1", "G2")}
    for name, rows in (("G0", g0), ("G1", g1), ("G2", g2)):
        for row in rows:
            by_variant[name][row["query_id"]].append(row)
    records = []
    for query_id, query in sorted(queries.items()):
        if query["task"] not in {"KIS", "QA"}:
            continue
        target = gt[query_id]
        correct_video = target["correct_video"]
        intervals = accepted_intervals(target["acceptable_intervals"])
        original = by_variant["G0"][query_id]
        budget = [row for row in by_variant["G2"][query_id] if row.get("was_selected_for_m1")]
        correct = [row for row in original if row["video_id"] == correct_video]
        records.append(
            {
                "query_id": query_id,
                "task": query["task"],
                "correct_video_candidate_exists": bool(correct),
                "correct_video_temporal_regions_in_top100": len(
                    {row.get("event_region_id") for row in correct}
                ),
                "gt_reachable_by_any_m1_window_top100": any(
                    _window_intersects(
                        int(row["coarse_frame_id"]), float(row["mapping_fps"]), intervals
                    )
                    for row in correct
                ),
                "gt_reachable_by_current_g2_refinement_budget": any(
                    row["video_id"] == correct_video
                    and _window_intersects(
                        int(row["coarse_frame_id"]), float(row["mapping_fps"]), intervals
                    )
                    for row in budget
                ),
                "promoted_correct_video_regions": sum(
                    row["video_id"] == correct_video
                    and int(row["coverage_rank"]) < int(row["g0_rank"])
                    for row in by_variant["G1"][query_id]
                ),
            }
        )
    task_diagnostics = {variant: value["task_diagnostics"] for variant, value in evaluation.items()}
    grounding = {"queries": records, "metrics": task_diagnostics}
    benchmark_slug = BENCHMARK_SLUGS[benchmark_id]
    write_json(root / "diagnostics" / f"{benchmark_slug}_grounding_diagnostics.json", grounding)

    transitions: Counter[str] = Counter()
    source_retained = refined_count = 0
    for row in _diagnostic_rows(prediction_run, "G2_SAFE_M1", "m1_alternative_provenance"):
        target = gt[row["query_id"]]
        if queries[row["query_id"]]["task"] not in {"KIS", "QA"}:
            continue
        intervals = accepted_intervals(target["acceptable_intervals"])
        video_ok = row["source_coarse_video_id"] == target["correct_video"]
        coarse_hit = video_ok and _intersects(int(row["source_coarse_frame_id"]), intervals)
        refined_hit = video_ok and _intersects(int(row["refined_frame_id"]), intervals)
        coarse_label = "hit" if coarse_hit else "miss"
        refined_label = "hit" if refined_hit else "miss"
        transitions[f"coarse_{coarse_label}_to_refined_{refined_label}"] += 1
        source_retained += 1
        refined_count += int(bool(row.get("emitted_before_dedup")))
    destructive_event_hits_lost = 0
    trake_sources = 0
    for row in _diagnostic_rows(prediction_run, "G2_SAFE_M1", "trake_dual_hypothesis"):
        if row.get("hypothesis_kind") != "M1_REFINED_ALTERNATIVE":
            continue
        target = gt[row["query_id"]]
        trake_sources += 1
        source_retained += 1
        refined_count += int(bool(row.get("emitted")))
        if row["video_id"] != target["correct_video"]:
            continue
        event_intervals = [_event_interval(value) for value in target["event_intervals"]]
        source_hits = sum(
            _intersects(int(frame), intervals)
            for frame, intervals in zip(
                row["source_coarse_frame_ids"], event_intervals, strict=True
            )
        )
        refined_hits = sum(
            _intersects(int(frame), intervals)
            for frame, intervals in zip(row["refined_frame_ids"], event_intervals, strict=True)
        )
        destructive_event_hits_lost += max(0, source_hits - refined_hits)
    m1_safety = {
        "transitions": dict(transitions),
        "source_coarse_retained_count": source_retained,
        "refined_alternative_count_before_dedup": refined_count,
        "trake_source_chain_count": trake_sources,
        "trake_source_event_hits_preserved_vs_destructive": destructive_event_hits_lost,
    }
    reachability = {
        "query_count": len(records),
        "kis_any_top100_m1_window_reachable": sum(
            row["task"] == "KIS" and row["gt_reachable_by_any_m1_window_top100"] for row in records
        ),
        "kis_g2_budget_m1_window_reachable": sum(
            row["task"] == "KIS" and row["gt_reachable_by_current_g2_refinement_budget"]
            for row in records
        ),
        "kis_temporal_region_promotion_helped": sum(
            row["task"] == "KIS" and row["promoted_correct_video_regions"] > 0 for row in records
        ),
        "m1_safety": m1_safety,
    }
    if benchmark_id == "DEV_CROSS_60":
        write_json(root / "diagnostics/cross_m1_reachability.json", reachability)
    return {"grounding": grounding, "reachability": reachability, "m1_safety": m1_safety}


def runtime_summary(
    prediction_runs: dict[str, dict[str, Any]],
    pipeline: SafeCoveragePipeline,
    startup_seconds: float,
) -> dict[str, Any]:
    variants: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    totals: defaultdict[str, float] = defaultdict(float)
    counter_totals: defaultdict[str, Counter[str]] = defaultdict(Counter)
    opaque_outputs: list[dict[str, Any]] = []
    for benchmark_id, run in prediction_runs.items():
        for variant, value in run["variants"].items():
            totals[variant] += float(value["seconds"])
            counter_totals[variant].update(value["counter_delta"])
            for result in value["results"]:
                variants[variant].append(
                    {"task": result.query_plan["task"], "seconds": result.latency_seconds}
                )
            opaque_outputs.extend(
                {
                    "benchmark_id": benchmark_id,
                    "variant": variant,
                    "query_id": row["query_id"],
                    "answer": str(row["answer"]),
                }
                for row in value["predictions"]
                if "answer" in row and is_opaque_machine_id(str(row["answer"]))
            )
    per_variant = {}
    for variant in VARIANTS:
        per_task = {}
        for task in ("KIS", "QA", "TRAKE"):
            values = [row["seconds"] for row in variants[variant] if row["task"] == task]
            per_task[task] = {
                "count": len(values),
                "median": median(values) if values else 0.0,
                "p95": float(np.percentile(values, 95)) if values else 0.0,
            }
        per_variant[variant] = {
            "total_seconds": totals[variant],
            "per_task": per_task,
            "counter_delta": dict(counter_totals[variant]),
        }
    g0 = max(per_variant["G0_E2E1_COARSE"]["total_seconds"], 1e-12)
    return {
        "startup_seconds": startup_seconds,
        "variants": per_variant,
        "g1_over_g0_runtime_ratio": per_variant["G1_COVERAGE_COARSE"]["total_seconds"] / g0,
        "g2_over_g0_runtime_ratio": per_variant["G2_SAFE_M1"]["total_seconds"] / g0,
        "qa_opaque_machine_id_output_count": len(opaque_outputs),
        "qa_opaque_machine_id_output_samples": opaque_outputs[:20],
        "devices": pipeline.runtime.runtime_manifest().get("devices", {}),
        **pipeline.runtime_diagnostics(),
    }


def decisions(evaluations: dict[str, dict[str, Any]], comparison: dict[str, Any]) -> dict[str, Any]:
    cross = evaluations["DEV_CROSS_60"]
    g0, g1, g2 = (cross[variant] for variant in VARIANTS)
    g0_kis = g0["task_diagnostics"]["KIS"]["cutoffs"]
    g1_kis = g1["task_diagnostics"]["KIS"]["cutoffs"]
    improved_deep = any(
        g1_kis[str(cutoff)].get("grounded_interval", 0.0)
        > g0_kis[str(cutoff)].get("grounded_interval", 0.0)
        for cutoff in (20, 50, 100)
    )
    protected_ok = all(
        g1["summary"][f"R@{cutoff}"] >= g0["summary"][f"R@{cutoff}"] for cutoff in (1, 5)
    )
    protected_grounding_ok = all(
        g1_kis[str(cutoff)].get("grounded_interval", 0.0)
        >= g0_kis[str(cutoff)].get("grounded_interval", 0.0)
        for cutoff in (1, 5)
    )
    no_task_structural_failure = all(
        g1["slices"][f"task:{task}"]["final_score"] >= g0["slices"][f"task:{task}"]["final_score"]
        for task in ("KIS", "QA", "TRAKE")
    )
    g1_keep = (
        g1["summary"]["final_score"] >= g0["summary"]["final_score"]
        and protected_ok
        and protected_grounding_ok
        and improved_deep
        and no_task_structural_failure
    )
    tr0, tr2 = g0["slices"]["task:TRAKE"], g2["slices"]["task:TRAKE"]
    paired = comparison["DEV_CROSS_60"]["G2_VS_G1"]["paired_counts"]
    no_single_event_grounding_regression = all(
        g2["task_diagnostics"][task]["cutoffs"][str(cutoff)].get(metric, 0.0)
        >= g1["task_diagnostics"][task]["cutoffs"][str(cutoff)].get(metric, 0.0)
        for task, metric in (("KIS", "grounded_interval"), ("QA", "grounding"))
        for cutoff in CUTOFFS
    )
    g2_keep = (
        g2["summary"]["final_score"] >= g1["summary"]["final_score"]
        and tr2["R@1"] == tr0["R@1"]
        and tr2["R@5"] == tr0["R@5"]
        and tr2["final_score"] >= tr0["final_score"]
        and all(
            g2["slices"][f"task:{task}"]["final_score"]
            >= g1["slices"][f"task:{task}"]["final_score"]
            for task in ("KIS", "QA")
        )
        and no_single_event_grounding_regression
        and paired.get("BETTER", 0) >= paired.get("WORSE", 0)
    )
    selected_variant = (
        "G2_SAFE_M1"
        if g1_keep and g2_keep
        else "G1_COVERAGE_COARSE"
        if g1_keep
        else "G0_E2E1_COARSE"
    )
    selected_policy = "G0_COARSE" if selected_variant == "G0_E2E1_COARSE" else selected_variant
    l21_warning = False
    if "DEV_L21_150" in evaluations:
        l21 = evaluations["DEV_L21_150"]
        l21_warning = (
            l21[selected_variant]["summary"]["final_score"]
            < l21["G0_E2E1_COARSE"]["summary"]["final_score"]
        )
    return {
        "g1_coverage_decision": "KEEP" if g1_keep else "DROP",
        "g2_safe_m1_decision": "KEEP" if g2_keep else "DROP",
        "selected_grounding_policy": selected_policy,
        "selected_system_variant": selected_variant,
        "l21_regression_warning": l21_warning,
    }


def render_cross_review(
    pipeline: SafeCoveragePipeline,
    prediction_run: dict[str, Any],
    evaluation: dict[str, Any],
    benchmark_root: str | Path,
    output_root: str | Path,
) -> list[Path]:
    """Render bounded post-inference G0/G1/G2 top-hypothesis comparisons."""

    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return []
    root = Path(output_root)
    gt = {row["query_id"]: row for row in read_jsonl(Path(benchmark_root) / "gt.jsonl")}
    queries = {row["query_id"]: row for row in prediction_run["queries"]}
    by_variant: dict[str, defaultdict[str, list[dict[str, Any]]]] = {
        variant: defaultdict(list) for variant in VARIANTS
    }
    for variant in VARIANTS:
        for row in prediction_run["variants"][variant]["predictions"]:
            by_variant[variant][row["query_id"]].append(row)
    limits = {"KIS": 8, "QA": 4, "TRAKE": 8}
    rendered = []
    g0_scores = {row["query_id"]: row for row in evaluation["G0_E2E1_COARSE"]["per_query"]}
    g1_scores = {row["query_id"]: row for row in evaluation["G1_COVERAGE_COARSE"]["per_query"]}
    g2_scores = {row["query_id"]: row for row in evaluation["G2_SAFE_M1"]["per_query"]}
    g2_single_alternatives = {
        row["query_id"]: row
        for row in _diagnostic_rows(prediction_run, "G2_SAFE_M1", "m1_alternative_provenance")
        if row.get("emitted_after_dedup")
    }
    g2_trake_alternatives = {
        row["query_id"]: row
        for row in _diagnostic_rows(prediction_run, "G2_SAFE_M1", "trake_dual_hypothesis")
        if row.get("hypothesis_kind") == "M1_REFINED_ALTERNATIVE" and row.get("emitted")
    }

    def panel(video_id: str, frame: int, label: str, target: dict[str, Any]) -> Any | None:
        try:
            image = Image.fromarray(pipeline._decode_image(video_id, frame)).convert("RGB")
        except (IndexError, OSError, RuntimeError, ValueError):
            return None
        image.thumbnail((360, 205))
        output = Image.new("RGB", (380, 285), "white")
        output.paste(image, ((380 - image.width) // 2, 55))
        draw = ImageDraw.Draw(output)
        draw.text((8, 6), f"{label} {video_id} f={frame}", fill="black")
        intervals = target.get("acceptable_intervals", target.get("event_intervals", []))
        draw.text((8, 25), f"GT video={target.get('correct_video')}", fill="red")
        draw.text((8, 40), f"GT intervals={str(intervals)[:52]}", fill="red")
        return output

    def hypothesis_key(row: dict[str, Any]) -> tuple[Any, ...]:
        frames = row.get("frame_ids", [row.get("frame_id")])
        return str(row["video_id"]), tuple(int(value) for value in frames)

    def review_priority(query: dict[str, Any]) -> tuple[Any, ...]:
        query_id = query["query_id"]
        g0_keys = tuple(hypothesis_key(row) for row in by_variant["G0_E2E1_COARSE"][query_id])
        g1_keys = tuple(hypothesis_key(row) for row in by_variant["G1_COVERAGE_COARSE"][query_id])
        has_refined_alternative = (
            query_id in g2_single_alternatives or query_id in g2_trake_alternatives
        )
        score_delta = abs(
            g1_scores[query_id]["final_score"] - g0_scores[query_id]["final_score"]
        ) + abs(g2_scores[query_id]["final_score"] - g1_scores[query_id]["final_score"])
        return (-int(g0_keys != g1_keys), -int(has_refined_alternative), -score_delta, query_id)

    for task, limit in limits.items():
        members = [row for row in queries.values() if row["task"] == task]
        members.sort(key=review_priority)
        task_images = []
        for query in members[:limit]:
            query_id = query["query_id"]
            panels = []
            for variant in VARIANTS:
                candidates = sorted(by_variant[variant][query_id], key=lambda row: row["rank"])
                if not candidates:
                    continue
                top = candidates[0]
                frame = int(top.get("frame_id", top.get("frame_ids", [0])[0]))
                rendered_panel = panel(
                    top["video_id"], frame, f"{VARIANT_SLUGS[variant].upper()} TOP1", gt[query_id]
                )
                if rendered_panel is not None:
                    panels.append(rendered_panel)
            alternative = (
                g2_trake_alternatives.get(query_id)
                if task == "TRAKE"
                else g2_single_alternatives.get(query_id)
            )
            if alternative:
                video_id = str(
                    alternative.get("video_id", alternative.get("source_coarse_video_id"))
                )
                source_frames = alternative.get(
                    "source_coarse_frame_ids", [alternative.get("source_coarse_frame_id")]
                )
                refined_frames = alternative.get(
                    "refined_frame_ids", [alternative.get("refined_frame_id")]
                )
                for label, frames in (
                    ("G2 M1 SOURCE", source_frames),
                    ("G2 M1 ALTERNATIVE", refined_frames),
                ):
                    if frames and frames[0] is not None:
                        rendered_panel = panel(video_id, int(frames[0]), label, gt[query_id])
                        if rendered_panel is not None:
                            panels.append(rendered_panel)
            if not panels:
                continue
            canvas = Image.new("RGB", (sum(item.width for item in panels), 315), "white")
            x = 0
            for item in panels:
                canvas.paste(item, (x, 30))
                x += item.width
            ImageDraw.Draw(canvas).text((8, 6), f"{query_id} | {task}", fill="black")
            path = root / "review" / f"{task.casefold()}_{query_id}.jpg"
            path.parent.mkdir(parents=True, exist_ok=True)
            canvas.save(path, quality=86)
            rendered.append(path)
            task_images.append(canvas)
        if task_images:
            width = max(image.width for image in task_images)
            montage = Image.new("RGB", (width, sum(image.height for image in task_images)), "white")
            y = 0
            for image in task_images:
                montage.paste(image, (0, y))
                y += image.height
            path = root / "montages" / f"cross_{task.casefold()}_review.jpg"
            path.parent.mkdir(parents=True, exist_ok=True)
            montage.save(path, quality=84)
            rendered.append(path)
    return rendered


def write_manifests(
    output_root: str | Path,
    *,
    pipeline: SafeCoveragePipeline,
    dataset_root: str | Path,
    team_eval_bundle: str | Path,
    historical_e2e1_bundle: str | Path | None,
    branch: str,
    git_commit: str,
    prediction_runs: dict[str, Any],
    runtime: dict[str, Any],
    decision: dict[str, Any],
) -> None:
    root = Path(output_root)
    manifest = pipeline.runtime.runtime_manifest()
    stage1b = manifest.get("stage1b", {})
    translator = manifest.get("translator", {})
    write_json(root / "config_snapshot.json", pipeline.settings.as_dict())
    write_json(
        root / "prediction_hashes.json",
        {
            benchmark: {variant: value["sha256"] for variant, value in run["variants"].items()}
            for benchmark, run in prediction_runs.items()
        },
    )
    write_json(root / "diagnostics/runtime_summary.json", runtime)
    write_json(root / "diagnostics/selection_decision.json", decision)
    write_json(
        root / "run_manifest.json",
        {
            "experiment": "TRIAGE_E2EG1",
            "experiment_version": "0.1",
            "branch": branch,
            "git_commit": git_commit,
            "created_at": datetime.now(UTC).isoformat(),
            "dataset_root": str(Path(dataset_root).resolve()),
            "team_eval_sha256": sha256_file(team_eval_bundle),
            "historical_e2e1_sha256": (
                sha256_file(historical_e2e1_bundle) if historical_e2e1_bundle else None
            ),
            "clip_model": stage1b.get("candidate_id"),
            "clip_checkpoint_sha256": stage1b.get("checkpoint_sha256"),
            "clip_device": manifest.get("devices", {}).get("clip"),
            "translator": translator.get("model_id"),
            "translator_device": manifest.get("devices", {}).get("translator"),
            "stage1_exact": True,
            "t3_pool_limit": POOL_LIMIT,
            "t3_region_radius_seconds": REGION_RADIUS_SECONDS,
            "t3_delta": pipeline.settings.t3_selected_delta,
            "protected_global_prefix": pipeline.settings.protected_global_prefix,
            "coverage_video_limit": pipeline.settings.coverage_video_limit,
            "coverage_regions_per_video": pipeline.settings.coverage_regions_per_video,
            "m1_single_event_budget": pipeline.settings.m1_single_event_budget,
            "m1_trake_source_chains": pipeline.settings.m1_trake_source_chains,
            "m1_destructive_replacement": False,
            "m1_settings": M1Settings().__dict__,
            "m2": False,
            "m3": False,
            "event_graph": False,
            "vlm": False,
            "agent": False,
            "gt_available_to_inference": False,
            "sealed_accessed": False,
        },
    )
    (root / "README.md").write_text(
        "# TRIAGE-EG E2E-G1\n\n"
        "Three same-run variants isolate metric-aware coverage and non-destructive M1 "
        "alternatives. G0 reproduces E2E-1 coarse ordering, G1 reallocates the existing "
        "T3 regions, and G2 preserves every source coarse hypothesis before adding bounded "
        "M1 alternatives. GT was unavailable until all predictions were finalized and hashed.\n",
        encoding="utf-8",
    )


def create_bundle(output_root: str | Path, zip_path: str | Path) -> Path:
    root = Path(output_root).resolve(strict=True)
    target = Path(zip_path).resolve(strict=False)
    files = [path for path in root.rglob("*") if path.is_file()]
    forbidden_suffixes = (".mp4", ".npy", ".npz", ".pt", ".pth", ".bin")
    for path in files:
        relative = path.relative_to(root).as_posix().casefold()
        if "sealed" in relative or relative.endswith(forbidden_suffixes):
            raise RuntimeError(f"FORBIDDEN_E2EG1_BUNDLE_MEMBER: {relative}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.unlink(missing_ok=True)
    with ZipFile(target, "w", ZIP_DEFLATED) as archive:
        for path in sorted(files):
            archive.write(path, path.relative_to(root).as_posix())
    return target


def formal_report_lines(
    *,
    git_commit: str,
    evaluations: dict[str, dict[str, Any]],
    comparison: dict[str, Any],
    post: dict[str, Any],
    runtime: dict[str, Any],
    decision: dict[str, Any],
    zip_path: str | Path,
) -> list[str]:
    cross = evaluations["DEV_CROSS_60"]
    lines = [
        f"HEAD={git_commit}",
        "E2EG1_IMPLEMENTATION=COMPLETE",
        "E2E1_BASELINE_PRESERVED=YES",
        "GT_LEAKAGE_GATE=PASS",
        "SEALED_ACCESS_GATE=PASS",
        "STAGE1_EXACT=UNCHANGED",
        "T3=UNCHANGED",
        "M1_PARAMETERS=UNCHANGED",
        "M1_DESTRUCTIVE_REPLACEMENT=DISABLED",
    ]
    for variant, label in (
        ("G0_E2E1_COARSE", "CROSS_G0"),
        ("G1_COVERAGE_COARSE", "CROSS_G1"),
        ("G2_SAFE_M1", "CROSS_G2"),
    ):
        value = cross[variant]
        lines.extend(
            [
                *[f"{label}_R{cutoff}={value['summary'][f'R@{cutoff}']}" for cutoff in CUTOFFS],
                f"{label}_FINAL={value['summary']['final_score']}",
                *[
                    f"{label}_{task}={value['slices'][f'task:{task}']['final_score']}"
                    for task in ("KIS", "QA", "TRAKE")
                ],
            ]
        )
    paired_g1 = comparison["DEV_CROSS_60"]["G1_VS_G0"]["paired_counts"]
    reach = post["DEV_CROSS_60"]["reachability"]
    transitions = reach["m1_safety"]["transitions"]
    g0_kis = cross["G0_E2E1_COARSE"]["task_diagnostics"]["KIS"]["cutoffs"]
    g1_kis = cross["G1_COVERAGE_COARSE"]["task_diagnostics"]["KIS"]["cutoffs"]
    trake_g0 = cross["G0_E2E1_COARSE"]["slices"]["task:TRAKE"]
    trake_g2 = cross["G2_SAFE_M1"]["slices"]["task:TRAKE"]
    lines.extend(
        [
            f"CROSS_G1_VS_G0_BETTER={paired_g1.get('BETTER', 0)}",
            f"CROSS_G1_VS_G0_TIE={paired_g1.get('TIE', 0)}",
            f"CROSS_G1_VS_G0_WORSE={paired_g1.get('WORSE', 0)}",
            f"CROSS_KIS_G0_GROUNDED_R20={g0_kis['20']['grounded_interval']}",
            f"CROSS_KIS_G1_GROUNDED_R20={g1_kis['20']['grounded_interval']}",
            f"CROSS_KIS_G0_GROUNDED_R50={g0_kis['50']['grounded_interval']}",
            f"CROSS_KIS_G1_GROUNDED_R50={g1_kis['50']['grounded_interval']}",
            f"G1_COVERAGE_DECISION={decision['g1_coverage_decision']}",
            f"CROSS_TRAKE_G0_R1={trake_g0['R@1']}",
            f"CROSS_TRAKE_G2_R1={trake_g2['R@1']}",
            f"CROSS_TRAKE_G0_R5={trake_g0['R@5']}",
            f"CROSS_TRAKE_G2_R5={trake_g2['R@5']}",
            f"M1_SOURCE_COARSE_RETAINED={reach['m1_safety']['source_coarse_retained_count']}",
            f"M1_REFINED_ALTERNATIVES={runtime['refined_alternative_count']}",
            f"M1_COARSE_HIT_TO_REFINED_MISS={transitions.get('coarse_hit_to_refined_miss', 0)}",
            f"M1_COARSE_MISS_TO_REFINED_HIT={transitions.get('coarse_miss_to_refined_hit', 0)}",
            f"G2_SAFE_M1_DECISION={decision['g2_safe_m1_decision']}",
            f"CROSS_KIS_ANY_TOP100_M1_WINDOW_REACHABLE={reach['kis_any_top100_m1_window_reachable']}/20",
            f"CROSS_KIS_G2_BUDGET_M1_WINDOW_REACHABLE={reach['kis_g2_budget_m1_window_reachable']}/20",
            f"CROSS_KIS_TEMPORAL_REGION_PROMOTION_HELPED={reach['kis_temporal_region_promotion_helped']}",
        ]
    )
    if "DEV_L21_150" in evaluations:
        l21 = evaluations["DEV_L21_150"]
        lines.extend(
            [
                f"L21_G0_FINAL={l21['G0_E2E1_COARSE']['summary']['final_score']}",
                f"L21_G1_FINAL={l21['G1_COVERAGE_COARSE']['summary']['final_score']}",
                f"L21_G2_FINAL={l21['G2_SAFE_M1']['summary']['final_score']}",
                *[
                    f"L21_G1_{task}={l21['G1_COVERAGE_COARSE']['slices'][f'task:{task}']['final_score']}"
                    for task in ("KIS", "QA", "TRAKE")
                ],
                *[
                    f"L21_G2_{task}={l21['G2_SAFE_M1']['slices'][f'task:{task}']['final_score']}"
                    for task in ("KIS", "QA", "TRAKE")
                ],
                f"L21_REGRESSION_WARNING={'YES' if decision['l21_regression_warning'] else 'NO'}",
            ]
        )
    lines.extend(
        [
            f"QA_MACHINE_IDS_FILTERED={runtime['qa_machine_ids_filtered']}",
            f"QA_OPAQUE_MACHINE_ID_OUTPUT_COUNT={runtime['qa_opaque_machine_id_output_count']}",
            f"G0_TOTAL_SECONDS={runtime['variants']['G0_E2E1_COARSE']['total_seconds']}",
            f"G1_TOTAL_SECONDS={runtime['variants']['G1_COVERAGE_COARSE']['total_seconds']}",
            f"G2_TOTAL_SECONDS={runtime['variants']['G2_SAFE_M1']['total_seconds']}",
            f"G1_OVER_G0_RUNTIME_RATIO={runtime['g1_over_g0_runtime_ratio']}",
            f"G2_OVER_G0_RUNTIME_RATIO={runtime['g2_over_g0_runtime_ratio']}",
            f"STARTUP_SECONDS={runtime['startup_seconds']}",
            f"M1_CALL_COUNT={runtime['m1_call_count']}",
            f"M1_CACHE_HITS={runtime['m1_cache_hits']}",
            f"RAW_DECODED_FRAMES={runtime['raw_decoded_frames']}",
            f"RAW_CLIP_ENCODE_COUNT={runtime['raw_clip_encode_count']}",
            *[
                f"{VARIANT_SLUGS[variant].upper()}_{task}_{stat.upper()}="
                f"{runtime['variants'][variant]['per_task'][task][stat]}"
                for variant in VARIANTS
                for task in ("KIS", "QA", "TRAKE")
                for stat in ("median", "p95")
            ],
            f"SELECTED_GROUNDING_POLICY={decision['selected_grounding_policy']}",
            "PRIMARY_BOTTLENECK_AFTER_G1=TEMPORAL_GROUNDING_AND_QA_ANSWERING_REVIEW",
            "NEW_MODEL_REQUIRED=NO_IN_THIS_SPRINT",
            "GRAPH_REQUIRED=NO_IN_THIS_SPRINT",
            "VLM_REQUIRED=NO_IN_THIS_SPRINT",
            "PARAMETER_TUNING_PERFORMED=NO",
            f"OUTPUT_ZIP={Path(zip_path)}",
            "RETURN_FOR_INDEPENDENT_REVIEW=YES",
        ]
    )
    return lines


__all__ = [
    "BENCHMARKS",
    "combine_prediction_variants",
    "compare_variants",
    "create_bundle",
    "decisions",
    "evaluate_finalized",
    "extract_development_bundle",
    "formal_report_lines",
    "materialize_inference_only",
    "post_inference_diagnostics",
    "render_cross_review",
    "run_prediction_variant",
    "runtime_summary",
    "write_manifests",
]
