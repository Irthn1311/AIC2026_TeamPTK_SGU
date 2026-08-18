"""GT-isolated BCF-1 reproduction, L21 run, evaluation, and packaging."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from aic2026_eval.io import read_jsonl, sha256_file, write_json
from aic2026_eval.scoring import evaluate
from aic2026_eval.validation import validate_predictions
from triage_eg.diagnostics.sca1_siglip2_complementarity import validate_siglip2_index
from triage_eg.e2e1.runner import run_prediction_variant
from triage_eg.e2eg1.pipeline import is_opaque_machine_id

from .attribution import fusion_diagnostics, paired_evaluation
from .contracts import (
    A0_CROSS_SHA256,
    EXPERIMENT,
    F1_CROSS_SHA256,
    INDEX_DTYPE,
    INDEX_FINGERPRINT,
    INDEX_ROWS,
    INDEX_SHAPE,
    INDEX_ZIP_SHA256,
    NORM_SHA256,
    POLICY,
    S1_CROSS_SHA256,
    VECTOR_SHA256,
    BCF1Settings,
)
from .fusion import fuse_predictions
from .io import write_jsonl_lf
from .preparation import BCF1Preparation


def validate_frozen_index(
    index_root: str | Path,
    *,
    stage1_root: str | Path,
    index_zip: str | Path | None = None,
) -> dict[str, Any]:
    validation = validate_siglip2_index(index_root, stage1_root=stage1_root)
    manifest = validation["manifest"]
    expected = {
        "index_fingerprint": INDEX_FINGERPRINT,
        "vector_sha256": VECTOR_SHA256,
        "norm_sha256": NORM_SHA256,
        "rows": INDEX_ROWS,
        "shape": INDEX_SHAPE,
        "dtype": INDEX_DTYPE,
    }
    mismatches = {
        key: {"actual": manifest.get(key), "expected": value}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"BCF1_FROZEN_INDEX_PROVENANCE_MISMATCH: {mismatches}")
    zip_gate = "NOT_AVAILABLE_AFTER_KAGGLE_EXPANSION"
    zip_path = None
    if index_zip is not None:
        zip_path = Path(index_zip).expanduser().resolve(strict=True)
        digest = sha256_file(zip_path)
        if digest != INDEX_ZIP_SHA256:
            raise RuntimeError(f"BCF1_INDEX_ZIP_SHA256_MISMATCH: {digest}")
        zip_gate = "PASS"
    return {
        "status": "PASS",
        "root": str(Path(index_root).resolve()),
        "index_zip": str(zip_path) if zip_path else None,
        "index_zip_expected_sha256": INDEX_ZIP_SHA256,
        "index_zip_sha256_gate": zip_gate,
        "manifest": {key: manifest[key] for key in expected},
        "index_rebuilt": False,
    }


def reproduce_cross(
    preparation: BCF1Preparation,
    queries: list[dict[str, Any]],
    output_root: str | Path,
    *,
    settings: BCF1Settings | None = None,
) -> dict[str, Any]:
    settings = settings or BCF1Settings()
    root = Path(output_root)
    a0_path = write_jsonl_lf(root / "predictions/cross_a0.jsonl", preparation.a0_cross)
    s1_path = write_jsonl_lf(root / "predictions/cross_s1.jsonl", preparation.s1_cross)
    if sha256_file(a0_path) != A0_CROSS_SHA256 or sha256_file(s1_path) != S1_CROSS_SHA256:
        raise RuntimeError("BCF1_FROZEN_CROSS_ARM_HASH_MISMATCH")
    f1, provenance = fuse_predictions(
        queries, preparation.a0_cross, preparation.s1_cross, settings=settings
    )
    f1_path = write_jsonl_lf(root / "predictions/cross_f1.jsonl", f1)
    f1_hash = sha256_file(f1_path)
    if f1_hash != F1_CROSS_SHA256 or f1 != preparation.frozen_f1_cross:
        raise RuntimeError(f"BCF1_CROSS_F1_REPRODUCTION_FAILED: {f1_hash}")
    write_jsonl_lf(root / "diagnostics/fusion_provenance_cross.jsonl", provenance)
    diagnostics = fusion_diagnostics(
        queries, preparation.a0_cross, preparation.s1_cross, f1, provenance
    )
    return {
        "benchmark_id": "DEV_CROSS_60",
        "queries": queries,
        "A0": {"predictions": preparation.a0_cross, "path": a0_path, "sha256": A0_CROSS_SHA256},
        "S1": {"predictions": preparation.s1_cross, "path": s1_path, "sha256": S1_CROSS_SHA256},
        "F1": {"predictions": f1, "path": f1_path, "sha256": f1_hash},
        "provenance": provenance,
        "diagnostics": diagnostics,
        "cross_reproduction_gate": "PASS",
    }


def run_l21_arm(
    pipeline: Any,
    inference_root: str | Path,
    output_root: str | Path,
    temporary_root: str | Path,
    arm: str,
) -> dict[str, Any]:
    if arm not in {"A0", "S1"}:
        raise ValueError("BCF-1 L21 arm must be A0 or S1")
    inference = Path(inference_root).resolve(strict=True)
    if {path.name for path in inference.iterdir()} != {"queries.jsonl"}:
        raise RuntimeError("BCF1_GT_UNAVAILABLE_DURING_L21_PREDICTION")
    run = run_prediction_variant(
        pipeline,
        inference,
        "DEV_L21_150",
        "G1_COVERAGE_COARSE",
        Path(temporary_root) / f"{arm.casefold()}_prediction_work",
    )
    target = write_jsonl_lf(
        Path(output_root) / f"predictions/l21_{arm.casefold()}.jsonl",
        run["predictions"],
    )
    digest = sha256_file(target)
    validation, issues = validate_predictions(run["queries"], run["predictions"])
    opaque = [
        row
        for row in run["predictions"]
        if "answer" in row and is_opaque_machine_id(str(row["answer"]))
    ]
    if not digest or validation["status"] != "PASS" or issues or opaque:
        raise RuntimeError(f"BCF1_L21_{arm}_PREDICTION_GATE_FAILED")
    return {
        **run,
        "prediction_path": target,
        "sha256": digest,
        "validation": validation,
        "qa_opaque_machine_id_output_count": 0,
        "arm": arm,
    }


def fuse_l21(
    a0_run: dict[str, Any],
    s1_run: dict[str, Any],
    output_root: str | Path,
    *,
    settings: BCF1Settings | None = None,
) -> dict[str, Any]:
    if a0_run.get("queries") != s1_run.get("queries") or not all(
        run.get("sha256") for run in (a0_run, s1_run)
    ):
        raise RuntimeError("BCF1_L21_ARMS_NOT_FINALIZED_BEFORE_FUSION")
    f1, provenance = fuse_predictions(
        a0_run["queries"], a0_run["predictions"], s1_run["predictions"], settings=settings
    )
    root = Path(output_root)
    path = write_jsonl_lf(root / "predictions/l21_f1.jsonl", f1)
    digest = sha256_file(path)
    write_jsonl_lf(root / "diagnostics/fusion_provenance_l21.jsonl", provenance)
    diagnostics = fusion_diagnostics(
        a0_run["queries"], a0_run["predictions"], s1_run["predictions"], f1, provenance
    )
    return {
        "benchmark_id": "DEV_L21_150",
        "queries": a0_run["queries"],
        "A0": a0_run,
        "S1": s1_run,
        "F1": {"predictions": f1, "prediction_path": path, "sha256": digest},
        "provenance": provenance,
        "diagnostics": diagnostics,
    }


def validate_all_hashes_before_gt(
    cross: dict[str, Any], l21: dict[str, Any], index: dict[str, Any]
) -> dict[str, Any]:
    hashes = {
        benchmark: {arm: run[arm].get("sha256") for arm in ("A0", "S1", "F1")}
        for benchmark, run in (("cross", cross), ("l21", l21))
    }
    if (
        cross.get("cross_reproduction_gate") != "PASS"
        or index.get("status") != "PASS"
        or index.get("index_rebuilt") is not False
        or any(not digest for values in hashes.values() for digest in values.values())
    ):
        raise RuntimeError("BCF1_GT_OPENED_BEFORE_ALL_HASHES_FINALIZED")
    return {
        "status": "PASS",
        "prediction_hashes": hashes,
        "cross_f1_reproduction_gate": "PASS",
        "l21_all_hashes_finalized_before_gt": "PASS",
        "gt_unavailable_to_fusion": "PASS",
        "siglip2_index_reused_without_rebuild": "PASS",
        "production_policy_changed": False,
    }


def _evaluate_arm(
    queries: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    gt: list[dict[str, Any]],
    *,
    benchmark_id: str,
    arm: str,
) -> dict[str, Any]:
    summary, per_query, slices, issues = evaluate(
        queries,
        predictions,
        gt,
        metadata={"benchmark_id": benchmark_id, "system_variant": arm},
    )
    return {"summary": summary, "per_query": per_query, "slices": slices, "issues": issues}


def evaluate_post_gt(
    cross: dict[str, Any],
    l21: dict[str, Any],
    integrity: dict[str, Any],
    *,
    cross_gt_path: str | Path,
    l21_gt_path: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    if integrity.get("status") != "PASS" or any(
        not run[arm].get("sha256") for run in (cross, l21) for arm in ("A0", "S1", "F1")
    ):
        raise RuntimeError("BCF1_POST_GT_CALLED_BEFORE_ALL_HASHES")
    root = Path(output_root)
    output = {}
    for benchmark, run, gt_path in (
        ("cross", cross, cross_gt_path),
        ("l21", l21, l21_gt_path),
    ):
        gt = read_jsonl(Path(gt_path).resolve(strict=True))
        values = {
            arm: _evaluate_arm(
                run["queries"],
                run[arm]["predictions"],
                gt,
                benchmark_id=run["benchmark_id"],
                arm=arm,
            )
            for arm in ("A0", "S1", "F1")
        }
        for arm, value in values.items():
            write_json(root / f"evaluation/{benchmark}_{arm.casefold()}.json", value)
        paired = paired_evaluation(values, benchmark_id=run["benchmark_id"])
        write_json(root / f"evaluation/{benchmark}_paired_delta.json", paired)
        output[benchmark] = {"arms": values, "paired": paired}
    return output


def promotion_decision(integrity: dict[str, Any], evaluations: dict[str, Any]) -> dict[str, Any]:
    l21 = evaluations["l21"]["arms"]
    a0, f1 = l21["A0"], l21["F1"]
    task_deltas = evaluations["l21"]["paired"]["task_delta_f1_minus_a0"]
    conditions = {
        "all_integrity_gates_pass": integrity.get("status") == "PASS",
        "cross_f1_reproduction_hash_pass": (integrity.get("cross_f1_reproduction_gate") == "PASS"),
        "l21_f1_final_not_below_a0": (f1["summary"]["final_score"] >= a0["summary"]["final_score"]),
        "l21_r1_equal_a0": f1["summary"]["R@1"] == a0["summary"]["R@1"],
        "l21_r5_equal_a0": f1["summary"]["R@5"] == a0["summary"]["R@5"],
        "no_l21_task_final_regression_over_0_02": all(
            delta >= -0.02 for delta in task_deltas.values()
        ),
    }
    classification = "KEEP_FOR_MOCK" if all(conditions.values()) else "DROP_OR_HOLD"
    return {
        "classification": classification,
        "conditions": conditions,
        "task_final_deltas": task_deltas,
        "automatic_production_promotion": False,
        "production_policy_changed": False,
    }


REQUIRED_BUNDLE_MEMBERS = frozenset(
    {
        "README.md",
        "FORMAL_REPORT.md",
        "run_manifest.json",
        "config_snapshot.json",
        "prediction_hashes.json",
        "predictions/cross_a0.jsonl",
        "predictions/cross_s1.jsonl",
        "predictions/cross_f1.jsonl",
        "predictions/l21_a0.jsonl",
        "predictions/l21_s1.jsonl",
        "predictions/l21_f1.jsonl",
        "diagnostics/fusion_provenance_cross.jsonl",
        "diagnostics/fusion_provenance_l21.jsonl",
        "diagnostics/fusion_summary.json",
        "evaluation/cross_a0.json",
        "evaluation/cross_s1.json",
        "evaluation/cross_f1.json",
        "evaluation/cross_paired_delta.json",
        "evaluation/l21_a0.json",
        "evaluation/l21_s1.json",
        "evaluation/l21_f1.json",
        "evaluation/l21_paired_delta.json",
        "tests/test_summary.json",
    }
)


def write_manifests(
    output_root: str | Path,
    *,
    settings: BCF1Settings,
    preparation: BCF1Preparation,
    index_validation: dict[str, Any],
    integrity: dict[str, Any],
    cross: dict[str, Any],
    l21: dict[str, Any],
    evaluations: dict[str, Any],
    decision: dict[str, Any],
    post_gt_design_sanity: dict[str, Any],
    test_summary: dict[str, Any],
    config_snapshot: dict[str, Any],
    git_commit: str,
    source_ref: str,
) -> None:
    root = Path(output_root)
    write_json(root / "tests/test_summary.json", test_summary)
    write_json(root / "config_snapshot.json", {"settings": settings.as_dict(), **config_snapshot})
    write_json(
        root / "prediction_hashes.json",
        integrity["prediction_hashes"],
    )
    fusion_summary = {
        "cross": cross["diagnostics"],
        "l21": l21["diagnostics"],
        "promotion": decision,
    }
    write_json(root / "diagnostics/fusion_summary.json", fusion_summary)
    write_json(
        root / "run_manifest.json",
        {
            "experiment": EXPERIMENT,
            "version": "0.1",
            "created_at": datetime.now(UTC).isoformat(),
            "git_commit": git_commit,
            "source_ref": source_ref,
            "policy": POLICY,
            "sca1_anchor_is_ancestor": True,
            "preparation_validation": preparation.validation,
            "index_validation": index_validation,
            "integrity": integrity,
            "post_gt_design_sanity_opened_after_all_prediction_hashes": True,
            "post_gt_design_sanity_f1_sha256": post_gt_design_sanity["f1_prediction_sha256"],
            "promotion": decision,
            "sealed_access": False,
            "production_policy_changed": False,
            "automatic_production_promotion": False,
        },
    )
    (root / "README.md").write_text(
        "# BCF-1 protected late rank fusion\n\n"
        "F1 copies A0 Top5 and applies equal RRF60 only to the remaining final A0/S1 "
        "Top100 candidates. Cross is a frozen reproduction gate; DEV_L21_150 is the "
        "fresh safety run. No model/index rebuild, raw-score fusion, parameter sweep, "
        "or automatic production promotion occurred.\n",
        encoding="utf-8",
    )
    report_lines = [
        "# BCF-1 formal report",
        "",
        f"- HEAD: `{git_commit}`",
        f"- Policy: `{POLICY}`",
        f"- Classification: `{decision['classification']}`",
        "- Production policy changed: `false`",
        "- Automatic promotion: `false`",
        "",
        "| Benchmark | Arm | R@1 | R@5 | R@20 | R@50 | R@100 | Final |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for benchmark in ("cross", "l21"):
        for arm in ("A0", "S1", "F1"):
            summary = evaluations[benchmark]["arms"][arm]["summary"]
            report_lines.append(
                "| {benchmark} | {arm} | {r1} | {r5} | {r20} | {r50} | {r100} | {final} |".format(
                    benchmark=benchmark.upper(),
                    arm=arm,
                    r1=summary["R@1"],
                    r5=summary["R@5"],
                    r20=summary["R@20"],
                    r50=summary["R@50"],
                    r100=summary["R@100"],
                    final=summary["final_score"],
                )
            )
    report_lines.extend(
        [
            "",
            "All prediction hashes were finalized before GT was opened. The frozen "
            "Cross F1 hash and exact SigLIP2 index provenance gates passed. This "
            "experiment is diagnostic-only and does not promote a production policy.",
            "",
        ]
    )
    (root / "FORMAL_REPORT.md").write_text("\n".join(report_lines), encoding="utf-8")


def create_bundle(output_root: str | Path, bundle_path: str | Path) -> dict[str, Any]:
    root = Path(output_root).resolve(strict=True)
    target = Path(bundle_path).resolve(strict=False)
    files = sorted(path for path in root.rglob("*") if path.is_file())
    names = {path.relative_to(root).as_posix() for path in files}
    missing = sorted(REQUIRED_BUNDLE_MEMBERS - names)
    if missing:
        raise RuntimeError(f"BCF1_BUNDLE_REQUIRED_MEMBERS_MISSING: {missing}")
    forbidden_suffixes = {
        ".npy",
        ".npz",
        ".safetensors",
        ".bin",
        ".pt",
        ".pth",
        ".jpg",
        ".jpeg",
        ".png",
        ".mp4",
    }
    for path in files:
        name = path.relative_to(root).as_posix().casefold()
        if (
            path.suffix.casefold() in forbidden_suffixes
            or path.name.casefold() in {"gt.jsonl", "queries.jsonl"}
            or "sealed" in name
            or "siglip2_vectors" in name
        ):
            raise RuntimeError(f"BCF1_BUNDLE_FORBIDDEN_MEMBER: {name}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(target, "w", ZIP_DEFLATED, allowZip64=True) as archive:
        for path in files:
            info = ZipInfo(path.relative_to(root).as_posix(), (1980, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())
    return {
        "path": str(target),
        "sha256": sha256_file(target),
        "size_bytes": target.stat().st_size,
        "member_count": len(files),
    }


__all__ = [
    "REQUIRED_BUNDLE_MEMBERS",
    "create_bundle",
    "evaluate_post_gt",
    "fuse_l21",
    "promotion_decision",
    "reproduce_cross",
    "run_l21_arm",
    "validate_all_hashes_before_gt",
    "validate_frozen_index",
    "write_manifests",
]
