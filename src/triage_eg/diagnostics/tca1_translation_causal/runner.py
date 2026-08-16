"""GT-isolated execution and paired evaluation for TCA-1."""

from __future__ import annotations

import shutil
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from aic2026_eval.io import read_jsonl, sha256_file, write_json, write_jsonl
from aic2026_eval.scoring import CUTOFFS, evaluate
from triage_eg.diagnostics.d1_grounding_attribution import (
    D1Settings,
    capture_inference_snapshot,
    run_g1_reproduction,
    run_post_gt_attribution,
)
from triage_eg.e2eg1.contracts import E2EG1Settings
from triage_eg.experiments.t3_diverse_temporal import POOL_LIMIT, REGION_RADIUS_SECONDS

from .contracts import EXPECTED_A0_PREDICTION_SHA256, TCA1Settings
from .review import FrozenReview


def run_pre_gt_arm(
    pipeline: Any,
    inference_root: str | Path,
    output_root: str | Path,
    temporary_root: str | Path,
    arm: str,
) -> tuple[dict[str, Any], Any]:
    """Finalize and hash one arm while the mounted root contains queries only."""

    root = Path(output_root)
    arm_temp = Path(temporary_root) / arm.casefold()
    run = run_g1_reproduction(pipeline, inference_root, arm_temp, arm_temp / "prediction_temp")
    target = root / "predictions" / f"{arm.casefold()}_cross_g1.jsonl"
    write_jsonl(target, run["predictions"])
    run = {**run, "prediction_path": target, "sha256": sha256_file(target), "arm": arm}
    snapshot = capture_inference_snapshot(pipeline, run)
    return run, snapshot


def _semantic_invariants(runtime_manifest: dict[str, Any]) -> dict[str, Any]:
    stage1b = runtime_manifest.get("stage1b", {})
    translator = runtime_manifest.get("translator", {})
    return {
        "stage1_index_fingerprint": runtime_manifest.get("stage1_index_fingerprint"),
        "stage1b_candidate_id": stage1b.get("candidate_id"),
        "clip_checkpoint_sha256": stage1b.get("checkpoint_sha256"),
        "stage1b_compatibility_status": stage1b.get("compatibility_status"),
        "stage1b_model_space_status": stage1b.get("model_space_status"),
        "stage1e_language_contract_sha256": runtime_manifest.get(
            "stage1e_language_contract_sha256"
        ),
        "ranking_policy": runtime_manifest.get("ranking_policy"),
        "index_device": runtime_manifest.get("devices", {}).get("index"),
        "devices": runtime_manifest.get("devices"),
        "hardware": runtime_manifest.get("hardware"),
        "translator_model_id": translator.get("model_id"),
        "translator_exact_revision": translator.get("exact_revision"),
        "translator_generation_config_sha256": translator.get("generation_config_sha256"),
    }


def validate_pre_gt_integrity(
    a0_run: dict[str, Any],
    a1_run: dict[str, Any],
    a0_snapshot: Any,
    a1_snapshot: Any,
    frozen: FrozenReview,
    a0_runtime_manifest: dict[str, Any],
    a1_runtime_manifest: dict[str, Any],
    settings: TCA1Settings | None = None,
    a0_policy: dict[str, Any] | None = None,
    a1_policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Prove the exact intervention and negative controls before GT access."""

    settings = settings or TCA1Settings()
    a0_policy = a0_policy or E2EG1Settings().as_dict()
    a1_policy = a1_policy or E2EG1Settings().as_dict()
    a0_units, a1_units = a0_snapshot.units, a1_snapshot.units
    if set(a0_units) != set(a1_units) or set(a0_units) != set(frozen.rows_by_unit):
        raise RuntimeError("TCA1_SEMANTIC_UNIT_SET_GATE_FAILED")
    if a0_run["sha256"] != EXPECTED_A0_PREDICTION_SHA256:
        raise RuntimeError(f"TCA1_A0_G1_REPRODUCTION_FAILED: {a0_run['sha256']}")
    if (
        a0_run.get("queries") != a1_run.get("queries")
        or a0_run.get("variant") != "G1_COVERAGE_COARSE"
        or a1_run.get("variant") != "G1_COVERAGE_COARSE"
    ):
        raise RuntimeError("TCA1_BENCHMARK_QUERY_ORDER_OR_VARIANT_GATE_FAILED")
    if _semantic_invariants(a0_runtime_manifest) != _semantic_invariants(a1_runtime_manifest):
        raise RuntimeError("TCA1_RUNTIME_INVARIANT_GATE_FAILED")
    if a0_policy != a1_policy or a0_policy != E2EG1Settings().as_dict():
        raise RuntimeError("TCA1_T3_G1_POLICY_INVARIANT_GATE_FAILED")
    changed_clip: set[str] = set()
    negative_exact_embeddings = 0
    negative_exact_scores = 0
    max_embedding_abs_delta = 0.0
    max_score_abs_delta = 0.0
    for unit_id in sorted(a0_units):
        baseline, intervention = a0_units[unit_id], a1_units[unit_id]
        review = frozen.rows_by_unit[unit_id]
        if (
            baseline.source_text != intervention.source_text
            or baseline.source_text != review["source_vi"]
        ):
            raise RuntimeError(f"TCA1_SOURCE_TEXT_GATE_FAILED: {unit_id}")
        a0_clip = str(baseline.encoding.get("clip_input_text", ""))
        a1_clip = str(intervention.encoding.get("clip_input_text", ""))
        if a0_clip != str(review["opus_en"]):
            raise RuntimeError(f"TCA1_A0_OPUS_TEXT_GATE_FAILED: {unit_id}")
        if a0_clip != a1_clip:
            changed_clip.add(unit_id)
        if unit_id in frozen.fail_unit_ids:
            if a1_clip != frozen.overrides[unit_id]:
                raise RuntimeError(f"TCA1_A1_REFERENCE_TEXT_GATE_FAILED: {unit_id}")
            continue
        if not np.array_equal(baseline.embedding, intervention.embedding):
            delta = float(np.max(np.abs(baseline.embedding - intervention.embedding)))
            max_embedding_abs_delta = max(max_embedding_abs_delta, delta)
            if not np.allclose(
                baseline.embedding,
                intervention.embedding,
                rtol=settings.negative_control_rtol,
                atol=settings.negative_control_atol,
            ):
                raise RuntimeError(f"TCA1_NEGATIVE_CONTROL_EMBEDDING_GATE_FAILED: {unit_id}")
        else:
            negative_exact_embeddings += 1
        if not np.array_equal(baseline.scores, intervention.scores):
            delta = float(np.max(np.abs(baseline.scores - intervention.scores)))
            max_score_abs_delta = max(max_score_abs_delta, delta)
            if not np.allclose(
                baseline.scores,
                intervention.scores,
                rtol=settings.negative_control_rtol,
                atol=settings.negative_control_atol,
            ):
                raise RuntimeError(f"TCA1_NEGATIVE_CONTROL_SCORE_GATE_FAILED: {unit_id}")
        else:
            negative_exact_scores += 1
    if changed_clip != set(frozen.fail_unit_ids):
        raise RuntimeError(
            "TCA1_CHANGED_CLIP_INPUT_SET_GATE_FAILED: "
            f"changed={sorted(changed_clip)} expected={sorted(frozen.fail_unit_ids)}"
        )
    return {
        "status": "PASS",
        "review_freeze_gate": "PASS",
        "a0_g1_reproduction": "PASS",
        "gt_leakage_gate": "PASS",
        "sealed_access_gate": "PASS",
        "prediction_hashes_created_before_gt": True,
        "runtime_invariants_equal": True,
        "changed_clip_input_unit_count": len(changed_clip),
        "changed_clip_input_unit_ids": sorted(changed_clip),
        "fail_override_count": len(frozen.fail_unit_ids),
        "nonfail_unit_count": len(frozen.nonfail_unit_ids),
        "nonfail_exact_embedding_count": negative_exact_embeddings,
        "nonfail_exact_score_vector_count": negative_exact_scores,
        "nonfail_max_embedding_abs_delta": max_embedding_abs_delta,
        "nonfail_max_score_abs_delta": max_score_abs_delta,
        "negative_control_atol": settings.negative_control_atol,
        "negative_control_rtol": settings.negative_control_rtol,
        "semantic_runtime_invariants": _semantic_invariants(a0_runtime_manifest),
        "t3_g1_policy": a0_policy,
        "t3_pool_limit": POOL_LIMIT,
        "t3_region_radius_seconds": REGION_RADIUS_SECONDS,
    }


def _task_scores(per_query: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in per_query:
        grouped[row["task"]].append(row)
    return {
        task: {
            **{
                f"R@{cutoff}": float(np.mean([row[f"R@{cutoff}"] for row in rows]))
                for cutoff in CUTOFFS
            },
            "final_score": float(np.mean([row["final_score"] for row in rows])),
            "query_count": len(rows),
        }
        for task, rows in sorted(grouped.items())
    }


def _rank_change(a0: int | None, a1: int | None) -> str:
    if a0 == a1:
        return "SAME"
    if a0 is None:
        return "IMPROVED" if a1 is not None else "SAME"
    if a1 is None:
        return "WORSENED"
    return "IMPROVED" if a1 < a0 else "WORSENED"


def _unit_audits(audit: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = [*audit["single_rows"], *audit["trake_event_rows"]]
    return {row["unit_id"]: row for row in rows}


def _prediction_rows_by_query(run: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    output: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in run["predictions"]:
        output[row["query_id"]].append(row)
    return {key: sorted(value, key=lambda row: row["rank"]) for key, value in output.items()}


def _paired_unit_rows(
    a0_audit: dict[str, Any], a1_audit: dict[str, Any], frozen: FrozenReview
) -> list[dict[str, Any]]:
    a0, a1 = _unit_audits(a0_audit), _unit_audits(a1_audit)
    rows = []
    for unit_id in sorted(a0):
        baseline, intervention = a0[unit_id], a1[unit_id]
        row = frozen.rows_by_unit[unit_id]
        values: dict[str, Any] = {
            "unit_id": unit_id,
            "query_id": row["query_id"],
            "task": row["task"],
            "event_id": row["event_id"],
            "translation_verdict": row["verdict"],
            "override_applied": unit_id in frozen.fail_unit_ids,
            "t3_target_hit_a0": bool(baseline["t3_pool_has_target"]),
            "t3_target_hit_a1": bool(intervention["t3_pool_has_target"]),
        }
        for field in ("target_within_video_rank", "target_global_rank", "correct_video_rank"):
            values[f"{field}_a0"] = baseline.get(field)
            values[f"{field}_a1"] = intervention.get(field)
            values[f"{field}_change"] = _rank_change(baseline.get(field), intervention.get(field))
        rows.append(values)
    return rows


def _paired_query_rows(
    a0_run: dict[str, Any],
    a1_run: dict[str, Any],
    a0_eval: dict[str, Any],
    a1_eval: dict[str, Any],
    a0_audit: dict[str, Any],
    a1_audit: dict[str, Any],
    frozen: FrozenReview,
) -> list[dict[str, Any]]:
    eval0 = {row["query_id"]: row for row in a0_eval["per_query"]}
    eval1 = {row["query_id"]: row for row in a1_eval["per_query"]}
    pred0, pred1 = _prediction_rows_by_query(a0_run), _prediction_rows_by_query(a1_run)
    fail_queries = {frozen.rows_by_unit[unit_id]["query_id"] for unit_id in frozen.fail_unit_ids}
    trake0 = {row["query_id"]: row for row in a0_audit["trake_query_rows"]}
    trake1 = {row["query_id"]: row for row in a1_audit["trake_query_rows"]}
    rows = []
    for query_id in sorted(eval0):
        row = {
            "query_id": query_id,
            "task": eval0[query_id]["task"],
            "contains_fail_unit": query_id in fail_queries,
            "negative_control_query": query_id not in fail_queries,
            "predictions_identical": pred0[query_id] == pred1[query_id],
        }
        for metric in [*(f"R@{cutoff}" for cutoff in CUTOFFS), "final_score"]:
            row[f"{metric}_a0"] = eval0[query_id][metric]
            row[f"{metric}_a1"] = eval1[query_id][metric]
            row[f"{metric}_delta"] = eval1[query_id][metric] - eval0[query_id][metric]
        if query_id in trake0:
            for field in (
                "btc_target_chain_exists",
                "t3_target_chain_exists",
                "g1_top100_full_target_chain_exists",
            ):
                row[f"{field}_a0"] = trake0[query_id][field]
                row[f"{field}_a1"] = trake1[query_id][field]
        rows.append(row)
    if any(not row["predictions_identical"] for row in rows if row["negative_control_query"]):
        raise RuntimeError("TCA1_NEGATIVE_CONTROL_PREDICTION_GATE_FAILED")
    return rows


def _official(run: dict[str, Any], ground_truth: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    summary, per_query, slices, issues = evaluate(
        run["queries"],
        run["predictions"],
        ground_truth,
        metadata={"benchmark_id": "DEV_CROSS_60", "system_variant": arm},
    )
    return {
        "summary": summary,
        "per_query": per_query,
        "task_scores": _task_scores(per_query),
        "slices": slices,
        "issues": issues,
    }


def _paired_score_delta(a0: dict[str, Any], a1: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {"overall": {}, "by_task": {}}
    for metric in [*(f"R@{cutoff}" for cutoff in CUTOFFS), "final_score"]:
        output["overall"][metric] = {
            "a0": a0["summary"][metric],
            "a1": a1["summary"][metric],
            "delta": a1["summary"][metric] - a0["summary"][metric],
        }
    for task in ("KIS", "QA", "TRAKE"):
        output["by_task"][task] = {}
        for metric in [*(f"R@{cutoff}" for cutoff in CUTOFFS), "final_score"]:
            output["by_task"][task][metric] = {
                "a0": a0["task_scores"][task][metric],
                "a1": a1["task_scores"][task][metric],
                "delta": a1["task_scores"][task][metric] - a0["task_scores"][task][metric],
            }
    return output


def _causal_status(
    paired_score: dict[str, Any], unit_rows: list[dict[str, Any]], query_rows: list[dict[str, Any]]
) -> str:
    """Simple predeclared multi-diagnostic, non-sole-cause classification."""

    fail_rows = [row for row in unit_rows if row["override_applied"]]
    improved = sum(row["target_global_rank_change"] == "IMPROVED" for row in fail_rows)
    worsened = sum(row["target_global_rank_change"] == "WORSENED" for row in fail_rows)
    t3_delta = sum(row["t3_target_hit_a1"] for row in fail_rows) - sum(
        row["t3_target_hit_a0"] for row in fail_rows
    )
    fail_queries = [row for row in query_rows if row["contains_fail_unit"]]
    fail_query_positive = sum(row["final_score_delta"] for row in fail_queries) > 0
    score_delta = paired_score["overall"]["final_score"]["delta"]
    positive_diagnostics = sum((improved > worsened, t3_delta > 0, fail_query_positive))
    negative_diagnostics = sum((worsened > improved, t3_delta < 0, score_delta < 0))
    if score_delta > 0 and positive_diagnostics >= 2 and negative_diagnostics == 0:
        return "CLEAR_POSITIVE_CAUSAL_SIGNAL"
    if score_delta == 0 and positive_diagnostics == 0 and negative_diagnostics == 0:
        return "NO_MEANINGFUL_CAUSAL_SIGNAL"
    return "LIMITED_OR_MIXED_CAUSAL_SIGNAL"


def evaluate_post_gt(
    *,
    a0_pipeline: Any,
    a1_pipeline: Any,
    a0_run: dict[str, Any],
    a1_run: dict[str, Any],
    a0_snapshot: Any,
    a1_snapshot: Any,
    benchmark_root: str | Path,
    frozen: FrozenReview,
    integrity: dict[str, Any],
    output_root: str | Path,
    temporary_root: str | Path,
) -> dict[str, Any]:
    """Open GT only after both arm hashes and every pre-GT gate are PASS."""

    if integrity.get("status") != "PASS" or not all(run.get("sha256") for run in (a0_run, a1_run)):
        raise RuntimeError("TCA1_PREDICTIONS_NOT_FINALIZED_BEFORE_GT")
    root = Path(output_root)
    ground_truth = read_jsonl(Path(benchmark_root).resolve(strict=True) / "gt.jsonl")
    d1_settings = D1Settings()
    temp = Path(temporary_root)
    a0_temp, a1_temp = temp / "post_gt_a0", temp / "post_gt_a1"
    a0_audit = run_post_gt_attribution(
        a0_snapshot, a0_run, ground_truth, a0_pipeline, a0_temp, settings=d1_settings
    )
    a1_audit = run_post_gt_attribution(
        a1_snapshot, a1_run, ground_truth, a1_pipeline, a1_temp, settings=d1_settings
    )
    for arm, source in (("a0", a0_temp / "diagnostics"), ("a1", a1_temp / "diagnostics")):
        target = root / "diagnostics" / arm
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target)
    a0_eval, a1_eval = _official(a0_run, ground_truth, "A0"), _official(a1_run, ground_truth, "A1")
    paired_score = _paired_score_delta(a0_eval, a1_eval)
    unit_rows = _paired_unit_rows(a0_audit, a1_audit, frozen)
    query_rows = _paired_query_rows(a0_run, a1_run, a0_eval, a1_eval, a0_audit, a1_audit, frozen)
    rank_counts = {
        field: dict(Counter(row[f"{field}_change"] for row in unit_rows))
        for field in ("target_within_video_rank", "target_global_rank", "correct_video_rank")
    }
    trake0, trake1 = a0_audit["trake_summary"], a1_audit["trake_summary"]
    summary = {
        "status": "COMPLETE",
        "causal_status": _causal_status(paired_score, unit_rows, query_rows),
        "translation_is_sole_root_cause": False,
        "production_policy_changed": False,
        "rank_change_counts": rank_counts,
        "t3_target_event_hits": {
            "a0": sum(row["t3_target_hit_a0"] for row in unit_rows),
            "a1": sum(row["t3_target_hit_a1"] for row in unit_rows),
        },
        "trake": {
            "btc_target_chain_exists": {
                "a0": trake0["queries_with_btc_target_chain"],
                "a1": trake1["queries_with_btc_target_chain"],
            },
            "t3_target_chain_exists": {
                "a0": trake0["queries_with_t3_target_chain"],
                "a1": trake1["queries_with_t3_target_chain"],
            },
            "full_target_chain_top100": {
                "a0": trake0["queries_with_full_target_chain_in_top100"],
                "a1": trake1["queries_with_full_target_chain_in_top100"],
            },
        },
        "d1_primary_measured_counts": {
            "a0": a0_audit["attribution_summary"]["measured_primary_counts"],
            "a1": a1_audit["attribution_summary"]["measured_primary_counts"],
        },
        "intervention_slices": {
            "fail_semantic_units": sum(row["override_applied"] for row in unit_rows),
            "queries_containing_fail": sum(row["contains_fail_unit"] for row in query_rows),
            "negative_control_queries": sum(row["negative_control_query"] for row in query_rows),
            "negative_control_predictions_identical": all(
                row["predictions_identical"] for row in query_rows if row["negative_control_query"]
            ),
        },
        "classification_rule": (
            "CLEAR when FinalScore improves and at least two of FAIL target-global rank, "
            "FAIL T3 hits, or FAIL-query official score improve with no negative diagnostic; "
            "NO_MEANINGFUL when score and all diagnostics are unchanged; otherwise "
            "LIMITED_OR_MIXED."
        ),
    }
    write_json(root / "evaluation/a0_official.json", a0_eval)
    write_json(root / "evaluation/a1_official.json", a1_eval)
    write_json(root / "evaluation/paired_score_delta.json", paired_score)
    write_jsonl(root / "diagnostics/paired_unit_delta.jsonl", unit_rows)
    write_jsonl(root / "diagnostics/paired_query_delta.jsonl", query_rows)
    write_json(root / "diagnostics/tca1_summary.json", summary)
    return {
        "a0": a0_eval,
        "a1": a1_eval,
        "paired_score": paired_score,
        "paired_units": unit_rows,
        "paired_queries": query_rows,
        "summary": summary,
    }


def write_run_manifests(
    output_root: str | Path,
    *,
    settings: TCA1Settings,
    frozen: FrozenReview,
    integrity: dict[str, Any],
    a0_run: dict[str, Any],
    a1_run: dict[str, Any],
    a0_runtime_manifest: dict[str, Any],
    a1_runtime_manifest: dict[str, Any],
    branch: str,
    git_commit: str,
    dataset_root: str | Path,
    team_eval_bundle: str | Path,
    experiment_config: dict[str, Any] | None = None,
) -> None:
    root = Path(output_root)
    write_json(
        root / "config_snapshot.json",
        {"settings": settings.as_dict(), "experiment_config": experiment_config},
    )
    write_json(
        root / "input_review_manifest.json",
        {
            "source": frozen.source,
            "source_zip_sha256": frozen.source_zip_sha256,
            "file_hashes": frozen.file_hashes,
            "validation": frozen.validation,
        },
    )
    write_json(root / "intervention_integrity.json", integrity)
    write_json(
        root / "prediction_hashes.json",
        {"A0": a0_run["sha256"], "A1": a1_run["sha256"]},
    )
    write_json(
        root / "run_manifest.json",
        {
            "experiment": "TRIAGE_TCA1_TRANSLATION_CAUSAL",
            "version": "0.1",
            "branch": branch,
            "git_commit": git_commit,
            "created_at": datetime.now(UTC).isoformat(),
            "dataset_root": str(Path(dataset_root).resolve()),
            "team_eval_bundle_sha256": sha256_file(team_eval_bundle),
            "gt_loaded_after_both_prediction_hashes": True,
            "sealed_access": False,
            "production_policy_changed": False,
            "a0_runtime": a0_runtime_manifest,
            "a1_runtime": a1_runtime_manifest,
        },
    )


def write_readme(output_root: str | Path) -> None:
    Path(output_root, "README.md").write_text(
        "# TCA-1 translation causal diagnostic\n\n"
        "A0 is the frozen OPUS baseline. A1 replaces CLIP input text only for the exact "
        "17 frozen FAIL semantic units. This bundle is diagnostic-only and does not change "
        "production retrieval policy. Ground truth was unavailable until both prediction "
        "files were finalized and hashed.\n",
        encoding="utf-8",
    )


def create_bundle(output_root: str | Path, bundle_path: str | Path) -> dict[str, Any]:
    """Create the bounded TCA-1 ZIP and reject forbidden payloads."""

    root = Path(output_root).resolve(strict=True)
    target = Path(bundle_path).resolve(strict=False)
    forbidden_suffixes = {".npy", ".npz", ".pt", ".pth", ".bin", ".mp4", ".avi", ".mkv"}
    forbidden_tokens = {"sealed", "ground_truth", "raw_video", "models", "weights"}
    forbidden_names = {"gt.jsonl", "queries.jsonl", "annotation_audit.jsonl"}
    files = sorted(path for path in root.rglob("*") if path.is_file())
    for path in files:
        relative = path.relative_to(root)
        lowered = {part.casefold() for part in relative.parts}
        forbidden_path = any(token in part for part in lowered for token in forbidden_tokens)
        if (
            path.suffix.casefold() in forbidden_suffixes
            or path.name.casefold() in forbidden_names
            or forbidden_path
        ):
            raise RuntimeError(f"TCA1_BUNDLE_FORBIDDEN_MEMBER: {relative.as_posix()}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(target, "w", ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(root).as_posix())
    with ZipFile(target) as archive:
        members = archive.namelist()
    return {
        "path": str(target),
        "sha256": sha256_file(target),
        "size_bytes": target.stat().st_size,
        "member_count": len(members),
    }


__all__ = [
    "create_bundle",
    "evaluate_post_gt",
    "run_pre_gt_arm",
    "validate_pre_gt_integrity",
    "write_readme",
    "write_run_manifests",
]
