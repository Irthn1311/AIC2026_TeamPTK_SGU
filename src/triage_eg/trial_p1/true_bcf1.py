"""True frozen BCF-1 Trial P1 recomputation with causal QA contracts."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

from aic2026_eval.io import sha256_file, write_json, write_jsonl
from aic2026_eval.validation import validate_predictions
from triage_eg.diagnostics.bcf1_protected_late_fusion import BCF1Settings
from triage_eg.diagnostics.bcf1_protected_late_fusion.fusion import fuse_predictions
from triage_eg.e2e1.qa import garbage_reason
from triage_eg.submission import create_submission_zip, validate_submission_zip

MODES = ("TRIAL_BCF1_SAFE", "TRIAL_TRIAGEEG_PREP")


def _run_arm(pipeline: Any, queries: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    results = pipeline.predict_queries(queries, "G1_COVERAGE_COARSE")
    predictions = [row for result in results for row in result.predictions]
    diagnostics = [row for result in results for row in result.diagnostics]
    validation, issues = validate_predictions(queries, predictions)
    counts = Counter(str(row["query_id"]) for row in predictions)
    if (
        validation["status"] != "PASS"
        or issues
        or any(counts[q["query_id"]] != 100 for q in queries)
    ):
        raise RuntimeError(f"TRUE_BCF1_{arm}_TOP100_GATE_FAILED: {validation}/{issues}/{counts}")
    return {"arm": arm, "predictions": predictions, "diagnostics": diagnostics}


def _qa_gates(
    compiled: list[dict[str, Any]],
    arm_runs: list[dict[str, Any]],
    fused: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    expected = {plan["query_id"]: plan["answer_type"] for plan in compiled if plan["task"] == "QA"}
    causal_rows, garbage_rows = [], []
    for run in arm_runs:
        for row in run["diagnostics"]:
            query_id = str(row.get("query_id", ""))
            if query_id not in expected or "compiled_answer_type" not in row:
                continue
            causal_rows.append(
                {
                    "arm": run["arm"],
                    "query_id": query_id,
                    "expected_answer_type": expected[query_id],
                    "executed_answer_type": row.get("compiled_answer_type"),
                    "intent": row.get("intent"),
                    "intent_source": row.get("intent_source", "COMPILED_ANSWER_TYPE"),
                    "answer_policy": row.get("answer_policy"),
                    "compiled_routing": row.get("compiled_routing", []),
                    "asr_status": row.get("asr_status"),
                    "evidence_sufficient": row.get("evidence_sufficient"),
                }
            )
    observed = {(row["arm"], row["query_id"], row["executed_answer_type"]) for row in causal_rows}
    missing = [
        (arm["arm"], query_id, kind)
        for arm in arm_runs
        for query_id, kind in expected.items()
        if (arm["arm"], query_id, kind) not in observed
    ]
    if missing or all(row.get("intent") == "GENERIC_VISUAL" for row in causal_rows):
        raise RuntimeError(f"QA_COMPILER_NOT_CAUSAL: {missing}")
    for row in fused:
        if row["query_id"] not in expected:
            continue
        reason = garbage_reason(str(row.get("answer", "")), expected[row["query_id"]])
        garbage_rows.append(
            {
                "query_id": row["query_id"],
                "rank": row["rank"],
                "answer": row.get("answer"),
                "answer_type": expected[row["query_id"]],
                "rejection_reason": reason,
                "status": "PASS" if reason is None else "FAIL",
            }
        )
    failures = [row for row in garbage_rows if row["status"] == "FAIL"]
    if failures:
        raise RuntimeError(f"QA_GARBAGE_ANSWER_GATE: {failures[:10]}")
    return causal_rows, garbage_rows


def run_true_bcf1(
    a0_pipeline: Any,
    s1_pipeline: Any,
    compiled: list[dict[str, Any]],
    output_root: str | Path,
    submission_zip: str | Path,
    *,
    mode: str = "TRIAL_BCF1_SAFE",
) -> dict[str, Any]:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    root = Path(output_root).resolve(strict=False)
    root.mkdir(parents=True, exist_ok=True)
    queries = [plan["team_query"] for plan in compiled]
    if any("P0_COARSE" in str(query) for query in queries):
        raise RuntimeError("LEGACY_P0_COARSE_LABEL_FORBIDDEN")
    a0 = _run_arm(a0_pipeline, queries, "A0_OPENAI_CLIP_G1")
    s1 = _run_arm(s1_pipeline, queries, "S1_SIGLIP2_G1")
    fused, provenance = fuse_predictions(
        queries, a0["predictions"], s1["predictions"], settings=BCF1Settings()
    )
    qa_causal, qa_garbage = _qa_gates(compiled, [a0, s1], fused)
    paths = {
        "a0": root / "trial_p1_A0_predictions.jsonl",
        "s1": root / "trial_p1_S1_predictions.jsonl",
        "f1": root / "trial_p1_BCF1_F1_predictions.jsonl",
    }
    write_jsonl(paths["a0"], a0["predictions"])
    write_jsonl(paths["s1"], s1["predictions"])
    write_jsonl(paths["f1"], fused)
    write_jsonl(root / "trial_p1_BCF1_F1_provenance.jsonl", provenance)
    write_jsonl(root / "qa_causal_routing_diagnostics.jsonl", qa_causal)
    write_jsonl(root / "qa_garbage_filter_diagnostics.jsonl", qa_garbage)
    write_jsonl(
        root / "long_kis_variant_diagnostics.jsonl",
        (
            {
                "query_id": plan["query_id"],
                "canonical_raw_query_executed_first": True,
                "canonical_bcf1_semantics_mutated": False,
                "augmentation_label": "QUERY_VARIANT_AUGMENTED_NOT_EXECUTED_IN_FROZEN_ARM",
                "selected_variants": plan["retrieval_variants"],
                "selection": plan["variant_selection_diagnostics"],
            }
            for plan in compiled
            if plan["task"] == "KIS"
        ),
    )
    submission = create_submission_zip(queries, fused, Path(submission_zip))
    submission_validation = validate_submission_zip(submission, queries)
    result = {
        "status": "PASS",
        "mode": mode,
        "policy": BCF1Settings().policy,
        "query_count": len(queries),
        "prediction_count_per_arm": len(a0["predictions"]),
        "a0_sha256": sha256_file(paths["a0"]),
        "s1_sha256": sha256_file(paths["s1"]),
        "f1_sha256": sha256_file(paths["f1"]),
        "qa_compiler_causal_gate": "PASS",
        "qa_garbage_answer_gate": "PASS",
        "submission_validation": submission_validation,
        "submission_zip": str(submission),
        "gt_opened": False,
        "asr_v12_touched": False,
        "automatic_production_promotion": False,
    }
    write_json(root / "official_submission_validator_report.json", submission_validation)
    write_json(root / "trial_p1_true_bcf1_summary.json", result)
    return result


def write_report_and_bundle(
    output_root: str | Path,
    bundle_zip: str | Path,
    result: dict[str, Any],
    *,
    provenance: dict[str, Any],
) -> Path:
    root = Path(output_root).resolve(strict=True)
    report = "\n".join(
        [
            "# TRIAGE-EG Trial P1 True BCF-1 Repair Report",
            "",
            f"- Status: {result['status']}",
            f"- Mode: {result['mode']}",
            f"- Frozen policy: {result['policy']}",
            f"- Queries: {result['query_count']}",
            f"- A0 SHA-256: `{result['a0_sha256']}`",
            f"- S1 SHA-256: `{result['s1_sha256']}`",
            f"- F1 SHA-256: `{result['f1_sha256']}`",
            f"- QA compiler causal gate: {result['qa_compiler_causal_gate']}",
            f"- QA garbage answer gate: {result['qa_garbage_answer_gate']}",
            f"- Submission validator: {result['submission_validation']['status']}",
            "- GT opened: NO",
            "- ASR v1.2 touched: NO",
            "- Production promoted: NO",
            "",
            "## Provenance",
            "",
            "```json",
            json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True),
            "```",
        ]
    )
    (root / "TRIAL_P1_REPAIR_REPORT.md").write_text(report + "\n", encoding="utf-8")
    target = Path(bundle_zip).resolve(strict=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(target, "w", ZIP_DEFLATED) as archive:
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            archive.write(path, path.relative_to(root).as_posix())
        archive.write(Path(result["submission_zip"]), Path(result["submission_zip"]).name)
    return target


__all__ = ["MODES", "run_true_bcf1", "write_report_and_bundle"]
