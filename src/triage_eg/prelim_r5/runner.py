"""Pre-GT freeze, separated benchmark evaluation, decision, and R5 artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

from aic2026_eval.scoring import evaluate, score_prediction
from aic2026_eval.validation import validate_predictions

from .fusion import R5Settings, candidate_key

ARMS = ("TRUE_BCF1", "SAFE_R4_LIVE_WINNER", "SAFE_R5_QE", "SAFE_R5_GATED")
BENCHMARKS = ("cross", "l21")


def _flat_candidate_key(value: tuple[Any, ...] | list[Any]) -> tuple[Any, ...]:
    values = tuple(value)
    if len(values) == 2 and isinstance(values[1], tuple | list):
        return values[0], *values[1]
    return values


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n" for row in rows
        ),
        encoding="utf-8",
        newline="\n",
    )


def finalize_pre_gt_predictions(
    output_root: str | Path,
    queries: dict[str, list[dict[str, Any]]],
    predictions: dict[str, dict[str, list[dict[str, Any]]]],
    *,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Validate and hash every arm before any GT object is accepted by evaluation."""

    root = Path(output_root)
    if set(queries) != set(BENCHMARKS) or set(predictions) != set(BENCHMARKS):
        raise RuntimeError("R5_BENCHMARKS_MUST_REMAIN_SEPARATE_CROSS_AND_L21")
    if "sealed" in json.dumps(config, default=str).casefold():
        raise RuntimeError("R5_SEALED_FINAL_REFERENCE_FORBIDDEN")
    hashes, validations = {}, {}
    for benchmark in BENCHMARKS:
        if set(predictions[benchmark]) != set(ARMS):
            raise RuntimeError(f"R5_PRE_GT_ARM_SET_MISMATCH:{benchmark}")
        hashes[benchmark], validations[benchmark] = {}, {}
        for arm in ARMS:
            rows = predictions[benchmark][arm]
            summary, issues = validate_predictions(queries[benchmark], rows)
            if summary["status"] != "PASS" or issues:
                raise RuntimeError(f"R5_PRE_GT_VALIDATION_FAILED:{benchmark}:{arm}:{issues}")
            path = root / "predictions" / f"{benchmark}_{arm}.jsonl"
            _write_jsonl(path, rows)
            hashes[benchmark][arm] = _sha256(path)
            validations[benchmark][arm] = summary
    if any(not digest for arms in hashes.values() for digest in arms.values()):
        raise RuntimeError("R5_PRE_GT_HASH_MISSING")
    payload = {
        "status": "PASS",
        "benchmarks_separate": True,
        "prediction_hashes": hashes,
        "validation": validations,
        "config": config,
        "gt_opened": False,
        "sealed_final_access": False,
        "all_predictions_finalized_before_gt": True,
    }
    _write_json(root / "pre_gt_prediction_hashes.json", payload)
    return payload


def evaluate_frozen_arms(
    queries: dict[str, list[dict[str, Any]]],
    predictions: dict[str, dict[str, list[dict[str, Any]]]],
    ground_truth: dict[str, list[dict[str, Any]]],
    pre_gt: dict[str, Any],
) -> dict[str, Any]:
    """Open the evaluation boundary only after all separate arm hashes exist."""

    if (
        pre_gt.get("status") != "PASS"
        or pre_gt.get("all_predictions_finalized_before_gt") is not True
        or pre_gt.get("sealed_final_access") is not False
    ):
        raise RuntimeError("R5_GT_BOUNDARY_OPENED_BEFORE_PRE_GT_FREEZE")
    if set(ground_truth) != set(BENCHMARKS):
        raise RuntimeError("R5_GT_MUST_CONTAIN_CROSS_AND_L21_SEPARATELY")
    output = {}
    for benchmark in BENCHMARKS:
        output[benchmark] = {}
        for arm in ARMS:
            summary, per_query, slices, issues = evaluate(
                queries[benchmark],
                predictions[benchmark][arm],
                ground_truth[benchmark],
                metadata={"benchmark": benchmark, "system_variant": arm},
            )
            output[benchmark][arm] = {
                "summary": summary,
                "per_query": per_query,
                "slices": slices,
                "issues": issues,
            }
    return output


def _query_delta(
    evaluations: dict[str, Any], benchmark: str, candidate: str, baseline: str
) -> tuple[int, int, int]:
    left = {row["query_id"]: row for row in evaluations[benchmark][candidate]["per_query"]}
    right = {row["query_id"]: row for row in evaluations[benchmark][baseline]["per_query"]}
    wins = sum(left[key]["final_score"] > right[key]["final_score"] for key in left)
    losses = sum(left[key]["final_score"] < right[key]["final_score"] for key in left)
    return wins, losses, len(left) - wins - losses


def select_production_policy(
    evaluations: dict[str, Any],
    diagnostics: dict[str, Any],
    *,
    settings: R5Settings | None = None,
) -> dict[str, Any]:
    """Apply thresholds frozen before GT and retain SAFE_R4 unless all gates justify R5."""

    settings = settings or R5Settings()
    scores = {
        benchmark: {arm: evaluations[benchmark][arm]["summary"] for arm in ARMS}
        for benchmark in BENCHMARKS
    }
    live_cross = float(scores["cross"]["SAFE_R4_LIVE_WINNER"]["final_score"])
    qe_cross = float(scores["cross"]["SAFE_R5_QE"]["final_score"])
    gated_cross = float(scores["cross"]["SAFE_R5_GATED"]["final_score"])
    live_l21 = float(scores["l21"]["SAFE_R4_LIVE_WINNER"]["final_score"])
    qe_l21 = float(scores["l21"]["SAFE_R5_QE"]["final_score"])
    gated_l21 = float(scores["l21"]["SAFE_R5_GATED"]["final_score"])
    qe_wins, qe_losses, qe_ties = _query_delta(
        evaluations, "cross", "SAFE_R5_QE", "SAFE_R4_LIVE_WINNER"
    )
    gated_live_wins, gated_live_losses, _ = _query_delta(
        evaluations, "cross", "SAFE_R5_GATED", "SAFE_R4_LIVE_WINNER"
    )
    gated_qe_wins, gated_qe_losses, _ = _query_delta(
        evaluations, "cross", "SAFE_R5_GATED", "SAFE_R5_QE"
    )
    structural = bool(diagnostics.get("all_structural_gates_pass"))
    coverage_improved = bool(diagnostics.get("coverage_improved"))
    qe_conditions = {
        "cross_materially_improves_or_flat_with_coverage": (
            qe_cross - live_cross >= settings.cross_material_delta
            or (qe_cross - live_cross >= -settings.cross_flat_tolerance and coverage_improved)
        ),
        "l21_not_catastrophic": qe_l21 - live_l21 >= settings.l21_catastrophic_delta,
        "structural_gates_pass": structural,
    }
    gated_conditions = {
        "beats_live_meaningfully": gated_cross - live_cross >= settings.gated_meaningful_delta,
        "beats_qe_meaningfully": gated_cross - qe_cross >= settings.gated_meaningful_delta,
        "gain_not_isolated": gated_live_wins >= settings.gated_min_query_wins
        and gated_qe_wins >= settings.gated_min_query_wins,
        "l21_not_catastrophic": gated_l21 - live_l21 >= settings.l21_catastrophic_delta,
        "structural_gates_pass": structural,
        "override_audit_sane": bool(diagnostics.get("override_audit_sane")),
    }
    if all(gated_conditions.values()):
        policy = "PRODUCTION_SAFE_R5_GATED"
    elif all(qe_conditions.values()):
        policy = "PRODUCTION_SAFE_R5_QE"
    else:
        policy = "PRODUCTION_SAFE_R4_LIVE_WINNER"
    return {
        "production_policy": policy,
        "qe_conditions": qe_conditions,
        "gated_conditions": gated_conditions,
        "frozen_thresholds": settings.__dict__,
        "deltas": {
            "cross_qe_minus_live": qe_cross - live_cross,
            "cross_gated_minus_live": gated_cross - live_cross,
            "cross_gated_minus_qe": gated_cross - qe_cross,
            "l21_qe_minus_live": qe_l21 - live_l21,
            "l21_gated_minus_live": gated_l21 - live_l21,
        },
        "paired_counts": {
            "qe_vs_live": {"wins": qe_wins, "losses": qe_losses, "ties": qe_ties},
            "gated_vs_live": {
                "wins": gated_live_wins,
                "losses": gated_live_losses,
            },
            "gated_vs_qe": {"wins": gated_qe_wins, "losses": gated_qe_losses},
        },
        "post_gt_tuning": False,
    }


def _top_overlap(task: str, left: list[dict[str, Any]], right: list[dict[str, Any]], k: int) -> int:
    return len(
        {candidate_key(task, row) for row in left[:k]}
        & {candidate_key(task, row) for row in right[:k]}
    )


def build_candidate_comparison(
    queries: dict[str, list[dict[str, Any]]],
    predictions: dict[str, dict[str, list[dict[str, Any]]]],
    ground_truth: dict[str, list[dict[str, Any]]],
    view_provenance: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    provenance = {}
    for row in view_provenance:
        provenance.setdefault(
            (row["query_id"], _flat_candidate_key(row["candidate_key"])), []
        ).append(row)
    output = []
    for benchmark in BENCHMARKS:
        gt = {row["query_id"]: row for row in ground_truth[benchmark]}
        grouped = {
            arm: {
                query["query_id"]: sorted(
                    [
                        row
                        for row in predictions[benchmark][arm]
                        if row["query_id"] == query["query_id"]
                    ],
                    key=lambda row: int(row["rank"]),
                )
                for query in queries[benchmark]
            }
            for arm in ARMS
        }
        for query in queries[benchmark]:
            query_id, task = query["query_id"], query["task"]
            first_correct_by_arm = {}
            correct_key = None
            for arm in ARMS:
                correct = []
                for row in grouped[arm][query_id]:
                    score, _ = score_prediction(query, row, gt[query_id])
                    if score > 0:
                        correct.append((int(row["rank"]), candidate_key(task, row)))
                first = min(correct, default=(None, None), key=lambda item: item[0] or 101)
                first_correct_by_arm[arm] = first[0]
                if arm == "SAFE_R5_QE":
                    correct_key = first[1]
            sources = provenance.get((query_id, _flat_candidate_key(correct_key or ())), [])
            live_keys = {
                _flat_candidate_key(candidate_key(task, row))
                for row in grouped["SAFE_R4_LIVE_WINNER"][query_id]
            }
            new_by_view = {}
            for source in (
                row
                for row in view_provenance
                if row["query_id"] == query_id
                and _flat_candidate_key(row["candidate_key"]) not in live_keys
            ):
                for view in source.get("view_ranks", {}):
                    new_by_view[view] = new_by_view.get(view, 0) + 1
            strong_asr = [
                source
                for source in view_provenance
                if source["query_id"] == query_id and source.get("tier") == "TIER_A_DIRECT"
            ]
            output.append(
                {
                    "benchmark": benchmark,
                    "query_id": query_id,
                    "task": task,
                    "bcf1_top1": list(candidate_key(task, grouped["TRUE_BCF1"][query_id][0])),
                    "safe_r4_top1": list(
                        candidate_key(task, grouped["SAFE_R4_LIVE_WINNER"][query_id][0])
                    ),
                    "safe_r5_qe_top1": list(
                        candidate_key(task, grouped["SAFE_R5_QE"][query_id][0])
                    ),
                    "safe_r5_gated_top1": list(
                        candidate_key(task, grouped["SAFE_R5_GATED"][query_id][0])
                    ),
                    "qe_vs_live_top5_overlap": _top_overlap(
                        task,
                        grouped["SAFE_R5_QE"][query_id],
                        grouped["SAFE_R4_LIVE_WINNER"][query_id],
                        5,
                    ),
                    "qe_vs_live_top20_overlap": _top_overlap(
                        task,
                        grouped["SAFE_R5_QE"][query_id],
                        grouped["SAFE_R4_LIVE_WINNER"][query_id],
                        20,
                    ),
                    "qe_unique_videos_top20": len(
                        {row["video_id"] for row in grouped["SAFE_R5_QE"][query_id][:20]}
                    ),
                    "correct_hit_by_arm": {
                        arm: rank is not None for arm, rank in first_correct_by_arm.items()
                    },
                    "first_correct_rank_by_arm": first_correct_by_arm,
                    "first_correct_qe_rank": first_correct_by_arm["SAFE_R5_QE"],
                    "first_correct_view_sources": sources,
                    "new_candidates_by_view": new_by_view,
                    "strong_asr_best_rank": min(
                        (int(row["rank"]) for row in strong_asr), default=None
                    ),
                }
            )
    return output


def write_r5_artifacts(
    output_root: str | Path,
    *,
    evaluations: dict[str, Any],
    decision: dict[str, Any],
    pre_gt: dict[str, Any],
    comparison: list[dict[str, Any]],
    query_view_diagnostics: list[dict[str, Any]],
    head_override_audit: list[dict[str, Any]],
    qa_evidence: list[dict[str, Any]],
    run_provenance: dict[str, Any],
    bundle_path: str | Path,
) -> dict[str, Any]:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    cross_scores = {arm: evaluations["cross"][arm]["summary"] for arm in ARMS}
    l21_scores = {arm: evaluations["l21"][arm]["summary"] for arm in ARMS}
    _write_json(root / "cross60_scores.json", cross_scores)
    _write_json(root / "l21_scores.json", l21_scores)
    _write_jsonl(root / "r5_query_view_diagnostics.jsonl", query_view_diagnostics)
    _write_jsonl(root / "r5_head_override_audit.jsonl", head_override_audit)
    _write_jsonl(root / "r5_qa_deterministic_evidence.jsonl", qa_evidence)
    _write_json(root / "r5_candidate_comparison.json", comparison)
    with (root / "r5_candidate_comparison.csv").open("w", encoding="utf-8", newline="") as stream:
        fieldnames = [
            "benchmark",
            "query_id",
            "task",
            "bcf1_top1",
            "safe_r4_top1",
            "safe_r5_qe_top1",
            "safe_r5_gated_top1",
            "qe_vs_live_top5_overlap",
            "qe_vs_live_top20_overlap",
            "qe_unique_videos_top20",
            "first_correct_qe_rank",
            "first_correct_view_sources",
            "correct_hit_by_arm",
            "first_correct_rank_by_arm",
            "new_candidates_by_view",
            "strong_asr_best_rank",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(comparison)
    _write_json(root / "production_policy.json", decision)
    view_rescues: dict[str, int] = {}
    for row in comparison:
        live_rank = row["first_correct_rank_by_arm"]["SAFE_R4_LIVE_WINNER"]
        qe_rank = row["first_correct_rank_by_arm"]["SAFE_R5_QE"]
        if qe_rank is None or live_rank is not None:
            continue
        views = {
            view
            for source in row["first_correct_view_sources"]
            for view in source.get("view_ranks", {})
        }
        for view in views:
            view_rescues[view] = view_rescues.get(view, 0) + 1
    evaluation_rows = {
        arm: {row["query_id"]: row["final_score"] for row in evaluations["cross"][arm]["per_query"]}
        for arm in ("SAFE_R5_QE", "SAFE_R5_GATED")
    }
    override_ids = {
        row["query_id"]
        for row in head_override_audit
        if row.get("override") is True and row.get("benchmark", "cross") == "cross"
    }
    override_wins = sum(
        evaluation_rows["SAFE_R5_GATED"].get(query_id, 0)
        > evaluation_rows["SAFE_R5_QE"].get(query_id, 0)
        for query_id in override_ids
    )
    override_losses = sum(
        evaluation_rows["SAFE_R5_GATED"].get(query_id, 0)
        < evaluation_rows["SAFE_R5_QE"].get(query_id, 0)
        for query_id in override_ids
    )
    aggregate = {
        "unique_correct_hits_rescued_by_view": view_rescues,
        "head_override_count": len(override_ids),
        "head_override_wins": override_wins,
        "head_override_losses": override_losses,
        "qa_deterministic_extraction_success_count": sum(
            row.get("decision") == "DETERMINISTIC_EVIDENCE_FIRST_WITH_BCF1_TAIL"
            for row in qa_evidence
        ),
    }
    _write_json(root / "r5_aggregate_diagnostics.json", aggregate)
    _write_json(
        root / "run_provenance.json",
        {**run_provenance, "pre_gt": pre_gt, "aggregate_diagnostics": aggregate},
    )
    report = [
        "# Prelim R5 Final Decision",
        "",
        f"`{decision['production_policy']}`",
        "",
        "Cross60 and L21 were evaluated and reported separately. SEALED_FINAL_30 was not opened.",
        "All arm prediction files were finalized and hashed before GT was read.",
        "No model/index/corpus preprocessing, post-GT tuning, submission upload, "
        "or automatic deployment occurred.",
        "",
        "## DEV_CROSS_60 scores (primary)",
        "",
        "```json",
        json.dumps(cross_scores, indent=2, sort_keys=True),
        "```",
        "",
        "## DEV_L21_150 scores (regression only)",
        "",
        "```json",
        json.dumps(l21_scores, indent=2, sort_keys=True),
        "```",
        "",
        "## Query-view and override diagnostics",
        "",
        "```json",
        json.dumps(aggregate, indent=2, sort_keys=True),
        "```",
        "",
        "## Decision details",
        "",
        "```json",
        json.dumps(decision, indent=2, sort_keys=True),
        "```",
    ]
    (root / "R5_FINAL_DECISION.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    target = Path(bundle_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            relative = path.relative_to(root).as_posix()
            if "sealed" in relative.casefold() or relative.endswith("gt.jsonl"):
                raise RuntimeError(f"R5_FORBIDDEN_BUNDLE_MEMBER:{relative}")
            archive.write(path, relative)
    return {
        "path": str(target.resolve()),
        "sha256": _sha256(target),
        "size_bytes": target.stat().st_size,
        "production_policy": decision["production_policy"],
    }


__all__ = [
    "ARMS",
    "BENCHMARKS",
    "build_candidate_comparison",
    "evaluate_frozen_arms",
    "finalize_pre_gt_predictions",
    "select_production_policy",
    "write_r5_artifacts",
]
