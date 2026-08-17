"""GT-isolated A0/S1 execution, evaluation, manifests, and packaging."""

from __future__ import annotations

import shutil
from collections import defaultdict
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
from triage_eg.e2eg1.pipeline import SafeCoveragePipeline

from .assets import validate_offline_asset
from .attribution import oracle_union_diagnostics, paired_unit_deltas, summarize_paired
from .contracts import (
    EXPECTED_A0_PREDICTION_SHA256,
    EXPECTED_OPENAI_CLIP_ID,
    EXPECTED_OPENAI_CLIP_SHA256,
    EXPECTED_ROWS,
    EXPECTED_STAGE1_FINGERPRINT,
    EXPECTED_TRANSLATOR_ID,
    EXPECTED_TRANSLATOR_REVISION,
    MODEL_ID,
    SEMANTIC_UNIT_COUNT,
    SCA1Settings,
)
from .index import validate_siglip2_index
from .pipeline import Siglip2GroundingPipeline
from .preparation import PreparationFreeze


def run_pre_gt_arm(
    pipeline: Any,
    inference_root: str | Path,
    output_root: str | Path,
    temporary_root: str | Path,
    arm: str,
) -> tuple[dict[str, Any], Any]:
    if arm not in {"A0", "S1"}:
        raise ValueError("SCA-1 arm must be A0 or S1")
    temporary = Path(temporary_root) / arm.casefold()
    run = run_g1_reproduction(pipeline, inference_root, temporary, temporary / "prediction_work")
    target = Path(output_root) / "predictions" / f"{arm.casefold()}_cross_g1.jsonl"
    write_jsonl(target, run["predictions"])
    run = {**run, "prediction_path": target, "sha256": sha256_file(target), "arm": arm}
    snapshot = capture_inference_snapshot(pipeline, run)
    return run, snapshot


def _runtime_invariants(manifest: dict[str, Any]) -> dict[str, Any]:
    stage1b, translator = manifest.get("stage1b", {}), manifest.get("translator", {})
    return {
        "stage1_index_fingerprint": manifest.get("stage1_index_fingerprint"),
        "openai_clip_candidate_id": stage1b.get("candidate_id"),
        "openai_clip_checkpoint_sha256": stage1b.get("checkpoint_sha256"),
        "openai_clip_compatibility": stage1b.get("compatibility_status"),
        "openai_clip_model_space": stage1b.get("model_space_status"),
        "translator_model_id": translator.get("model_id"),
        "translator_revision": translator.get("exact_revision"),
        "translator_generation_config_sha256": translator.get("generation_config_sha256"),
        "stage1e_language_contract_sha256": manifest.get("stage1e_language_contract_sha256"),
        "ranking_policy": manifest.get("ranking_policy"),
    }


def validate_pre_gt_integrity(
    *,
    a0_run: dict[str, Any],
    s1_run: dict[str, Any],
    a0_snapshot: Any,
    s1_snapshot: Any,
    a0_pipeline: SafeCoveragePipeline,
    s1_pipeline: Siglip2GroundingPipeline,
    preparation: PreparationFreeze,
    siglip2_asset_root: str | Path,
    siglip2_index_root: str | Path,
    stage1_root: str | Path,
    settings: SCA1Settings | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    settings = settings or SCA1Settings()
    if preparation.validation.get("status") != "PASS":
        raise RuntimeError("SCA1_PREPARATION_FREEZE_GATE_FAILED")
    if a0_run.get("sha256") != EXPECTED_A0_PREDICTION_SHA256:
        raise RuntimeError(f"SCA1_A0_REPRODUCTION_FAILED: {a0_run.get('sha256')}")
    if (
        a0_run.get("queries") != s1_run.get("queries")
        or a0_run.get("variant") != "G1_COVERAGE_COARSE"
        or s1_run.get("variant") != "G1_COVERAGE_COARSE"
    ):
        raise RuntimeError("SCA1_QUERY_ORDER_OR_VARIANT_GATE_FAILED")
    a0_manifest = a0_pipeline.runtime.runtime_manifest()
    s1_manifest = s1_pipeline.runtime.runtime_manifest()
    if _runtime_invariants(a0_manifest) != _runtime_invariants(s1_manifest):
        raise RuntimeError("SCA1_FROZEN_RUNTIME_INVARIANTS_CHANGED")
    frozen = _runtime_invariants(a0_manifest)
    if (
        frozen["stage1_index_fingerprint"] != EXPECTED_STAGE1_FINGERPRINT
        or frozen["openai_clip_candidate_id"] != EXPECTED_OPENAI_CLIP_ID
        or frozen["openai_clip_checkpoint_sha256"] != EXPECTED_OPENAI_CLIP_SHA256
        or frozen["translator_model_id"] != EXPECTED_TRANSLATOR_ID
        or frozen["translator_revision"] != EXPECTED_TRANSLATOR_REVISION
    ):
        raise RuntimeError("SCA1_FROZEN_PRODUCTION_BASELINE_MISMATCH")
    if a0_pipeline.settings.as_dict() != s1_pipeline.settings.as_dict() or (
        a0_pipeline.settings.as_dict() != E2EG1Settings().as_dict()
    ):
        raise RuntimeError("SCA1_T3_G1_POLICY_INVARIANT_GATE_FAILED")
    asset = validate_offline_asset(siglip2_asset_root)
    index = validate_siglip2_index(siglip2_index_root, stage1_root=stage1_root)
    manifest = index["manifest"]
    if (
        manifest.get("rows") != EXPECTED_ROWS
        or manifest.get("stage1_index_fingerprint") != EXPECTED_STAGE1_FINGERPRINT
    ):
        raise RuntimeError("SCA1_SIGLIP2_INDEX_CATALOG_GATE_FAILED")
    if set(a0_snapshot.units) != set(s1_snapshot.units) or len(a0_snapshot.units) != (
        SEMANTIC_UNIT_COUNT
    ):
        raise RuntimeError("SCA1_SEMANTIC_UNIT_SET_GATE_FAILED")
    text_rows = build_text_identity_rows(a0_snapshot, s1_snapshot)
    diagnostics = s1_pipeline.runtime_diagnostics()
    if (
        diagnostics.get("sca1_qa_frame_encoder_id") != EXPECTED_OPENAI_CLIP_ID
        or diagnostics.get("sca1_qa_answer_text_encoder_id") != EXPECTED_OPENAI_CLIP_ID
        or "_frame_embedding" in type(s1_pipeline).__dict__
        or "_answer_embeddings" in type(s1_pipeline).__dict__
    ):
        raise RuntimeError("SCA1_QA_ANSWER_SPACE_ISOLATION_GATE_FAILED")
    integrity = {
        "status": "PASS",
        "preparation_freeze_gate": "PASS",
        "a0_reproduction_gate": "PASS",
        "gt_unavailable_during_a0_prediction": "PASS",
        "gt_unavailable_during_s1_prediction": "PASS",
        "a0_hash_finalized_before_gt": "PASS",
        "s1_hash_finalized_before_gt": "PASS",
        "sealed_access": False,
        "text_identity_gate": "PASS",
        "text_identity_count": len(text_rows),
        "same_btc_catalog_rows": manifest["rows"],
        "same_stage1_catalog_fingerprint": manifest["stage1_index_fingerprint"],
        "qa_answer_space_isolation": "PASS",
        "a0_runtime_invariants": frozen,
        "siglip2_asset": asset,
        "siglip2_index": {
            key: manifest[key]
            for key in (
                "index_fingerprint",
                "rows",
                "shape",
                "dtype",
                "vector_sha256",
                "norm_sha256",
                "catalog_row_fingerprint",
                "build_seconds",
            )
        },
        "t3_g1_policy": a0_pipeline.settings.as_dict(),
        "production_policy_changed": False,
    }
    return integrity, text_rows


def build_text_identity_rows(a0_snapshot: Any, s1_snapshot: Any) -> list[dict[str, Any]]:
    """Prove byte identity without consuming GT, ranks, or outcomes."""

    if set(a0_snapshot.units) != set(s1_snapshot.units) or len(a0_snapshot.units) != (
        SEMANTIC_UNIT_COUNT
    ):
        raise RuntimeError("SCA1_SEMANTIC_UNIT_SET_GATE_FAILED")
    text_rows = []
    for unit_id in sorted(a0_snapshot.units):
        a0, s1 = a0_snapshot.units[unit_id], s1_snapshot.units[unit_id]
        a0_text = str(a0.encoding.get("clip_input_text", ""))
        s1_text = str(s1.encoding.get("sca1_clip_input_text", ""))
        a0_hash = _text_sha256(a0_text)
        s1_hash = _text_sha256(s1_text)
        if (
            not a0_text
            or a0_text != s1_text
            or a0_hash != s1_hash
            or a0.source_text != s1.source_text
            or a0.source_language != s1.source_language
        ):
            raise RuntimeError(f"SCA1_TEXT_IDENTITY_GATE_FAILED: {unit_id}")
        text_rows.append(
            {
                "unit_id": unit_id,
                "query_id": a0.query_id,
                "task": a0.task,
                "event_id": a0.event_id,
                "source_language": a0.source_language,
                "source_text": a0.source_text,
                "clip_input_text": a0_text,
                "a0_text_sha256": a0_hash,
                "s1_text_sha256": s1_hash,
                "a0_encoder_id": EXPECTED_OPENAI_CLIP_ID,
                "s1_encoder_id": MODEL_ID,
                "identity": True,
            }
        )
    return text_rows


def _text_sha256(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _task_scores(per_query: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in per_query:
        grouped[row["task"]].append(row)
    return {
        task: {
            "query_count": len(rows),
            **{
                f"R@{cutoff}": float(np.mean([row[f"R@{cutoff}"] for row in rows]))
                for cutoff in CUTOFFS
            },
            "final_score": float(np.mean([row["final_score"] for row in rows])),
        }
        for task, rows in sorted(grouped.items())
    }


def _official(run: dict[str, Any], gt: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    summary, per_query, slices, issues = evaluate(
        run["queries"],
        run["predictions"],
        gt,
        metadata={"benchmark_id": "DEV_CROSS_60", "system_variant": arm},
    )
    return {
        "summary": summary,
        "per_query": per_query,
        "task_scores": _task_scores(per_query),
        "slices": slices,
        "issues": issues,
    }


def _paired_score(a0: dict[str, Any], s1: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {"overall": {}, "by_task": {}}
    metrics = [*(f"R@{cutoff}" for cutoff in CUTOFFS), "final_score"]
    for metric in metrics:
        output["overall"][metric] = {
            "a0": a0["summary"][metric],
            "s1": s1["summary"][metric],
            "delta": s1["summary"][metric] - a0["summary"][metric],
        }
    for task in ("KIS", "QA", "TRAKE"):
        output["by_task"][task] = {}
        for metric in metrics:
            output["by_task"][task][metric] = {
                "a0": a0["task_scores"][task][metric],
                "s1": s1["task_scores"][task][metric],
                "delta": s1["task_scores"][task][metric] - a0["task_scores"][task][metric],
            }
    return output


def evaluate_post_gt(
    *,
    a0_pipeline: SafeCoveragePipeline,
    s1_pipeline: Siglip2GroundingPipeline,
    a0_run: dict[str, Any],
    s1_run: dict[str, Any],
    a0_snapshot: Any,
    s1_snapshot: Any,
    integrity: dict[str, Any],
    benchmark_root: str | Path,
    output_root: str | Path,
    temporary_root: str | Path,
) -> dict[str, Any]:
    if integrity.get("status") != "PASS" or not all(run.get("sha256") for run in (a0_run, s1_run)):
        raise RuntimeError("SCA1_ORACLE_OR_GT_CALLED_BEFORE_FINALIZED_HASHES")
    root, temporary = Path(output_root), Path(temporary_root)
    gt = read_jsonl(Path(benchmark_root).resolve(strict=True) / "gt.jsonl")
    d1_settings = D1Settings()
    a0_temp, s1_temp = temporary / "a0", temporary / "s1"
    a0_audit = run_post_gt_attribution(
        a0_snapshot, a0_run, gt, a0_pipeline, a0_temp, settings=d1_settings
    )
    s1_audit = run_post_gt_attribution(
        s1_snapshot, s1_run, gt, s1_pipeline, s1_temp, settings=d1_settings
    )
    for arm, source in (("a0", a0_temp / "diagnostics"), ("s1", s1_temp / "diagnostics")):
        target = root / "diagnostics" / arm
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target)
    a0_eval, s1_eval = _official(a0_run, gt, "A0"), _official(s1_run, gt, "S1")
    paired_score = _paired_score(a0_eval, s1_eval)
    units = paired_unit_deltas(a0_audit, s1_audit)
    oracle = oracle_union_diagnostics(a0_audit, s1_audit, units)
    summary = summarize_paired(units, oracle)
    a0_per_query = {row["query_id"]: row for row in a0_eval["per_query"]}
    s1_per_query = {row["query_id"]: row for row in s1_eval["per_query"]}
    oracle_queries = {row["query_id"]: row for row in oracle["trake_queries"]}
    paired_queries = []
    for query_id in sorted(a0_per_query):
        left, right = a0_per_query[query_id], s1_per_query[query_id]
        row = {"query_id": query_id, "task": left["task"]}
        for metric in [*(f"R@{cutoff}" for cutoff in CUTOFFS), "final_score"]:
            row[f"{metric}_a0"] = left[metric]
            row[f"{metric}_s1"] = right[metric]
            row[f"{metric}_delta"] = right[metric] - left[metric]
        if query_id in oracle_queries:
            row.update(oracle_queries[query_id])
        paired_queries.append(row)
    write_json(root / "evaluation/a0_official.json", a0_eval)
    write_json(root / "evaluation/s1_official.json", s1_eval)
    write_json(root / "evaluation/paired_score_delta.json", paired_score)
    write_jsonl(root / "diagnostics/paired_unit_delta.jsonl", units)
    write_jsonl(root / "diagnostics/paired_query_delta.jsonl", paired_queries)
    write_json(root / "diagnostics/oracle_union.json", oracle)
    write_json(root / "diagnostics/complementarity_summary.json", summary)
    return {
        "a0": a0_eval,
        "s1": s1_eval,
        "paired_score": paired_score,
        "paired_units": units,
        "paired_queries": paired_queries,
        "oracle": oracle,
        "summary": summary,
    }


REQUIRED_BUNDLE_MEMBERS = frozenset(
    {
        "README.md",
        "run_manifest.json",
        "config_snapshot.json",
        "asset_manifest.json",
        "index_manifest.json",
        "prediction_hashes.json",
        "predictions/a0_cross_g1.jsonl",
        "predictions/s1_cross_g1.jsonl",
        "text_identity.jsonl",
        "diagnostics/paired_unit_delta.jsonl",
        "diagnostics/paired_query_delta.jsonl",
        "diagnostics/complementarity_summary.json",
        "diagnostics/oracle_union.json",
        "evaluation/a0_official.json",
        "evaluation/s1_official.json",
        "evaluation/paired_score_delta.json",
        "tests/test_summary.json",
    }
)


def write_manifests(
    output_root: str | Path,
    *,
    settings: SCA1Settings,
    preparation: PreparationFreeze,
    integrity: dict[str, Any],
    a0_run: dict[str, Any],
    s1_run: dict[str, Any],
    asset_root: str | Path,
    index_root: str | Path,
    git_commit: str,
    branch: str,
    test_summary: dict[str, Any],
    experiment_config: dict[str, Any],
) -> None:
    root = Path(output_root)
    asset = Path(asset_root) / "manifests/asset_manifest.json"
    index = Path(index_root) / "index_manifest.json"
    shutil.copy2(asset, root / "asset_manifest.json")
    shutil.copy2(index, root / "index_manifest.json")
    write_json(root / "config_snapshot.json", {"settings": settings.as_dict(), **experiment_config})
    write_json(root / "tests/test_summary.json", test_summary)
    write_json(
        root / "prediction_hashes.json",
        {"A0": a0_run["sha256"], "S1": s1_run["sha256"]},
    )
    write_json(
        root / "run_manifest.json",
        {
            "experiment": "TRIAGE_SCA1_SIGLIP2_COMPLEMENTARITY",
            "version": "0.1",
            "created_at": datetime.now(UTC).isoformat(),
            "git_commit": git_commit,
            "branch": branch,
            "tca1_anchor_is_ancestor": True,
            "preparation_freeze_sha256": preparation.zip_sha256,
            "asset_manifest_sha256": sha256_file(asset),
            "index_manifest_sha256": sha256_file(index),
            "gt_loaded_after_both_prediction_hashes": True,
            "sealed_access": False,
            "production_policy_changed": False,
            "fusion_implemented": False,
            "integrity": integrity,
        },
    )
    (root / "README.md").write_text(
        "# SCA-1 SigLIP2 complementarity diagnostic\n\n"
        "A0 reproduces frozen OpenAI CLIP G1. S1 changes grounding only to the pinned "
        "SigLIP2 space over the exact same OPUS-English strings and BTC rows. U1 is a "
        "post-GT oracle diagnostic, not a prediction arm. Production policy is unchanged.\n",
        encoding="utf-8",
    )


def create_bundle(output_root: str | Path, bundle_path: str | Path) -> dict[str, Any]:
    root, target = Path(output_root).resolve(strict=True), Path(bundle_path).resolve(strict=False)
    files = sorted(path for path in root.rglob("*") if path.is_file())
    relative = {path.relative_to(root).as_posix() for path in files}
    missing = sorted(REQUIRED_BUNDLE_MEMBERS - relative)
    if (
        missing
        or not any(name.startswith("diagnostics/a0/") for name in relative)
        or not any(name.startswith("diagnostics/s1/") for name in relative)
    ):
        raise RuntimeError(f"SCA1_BUNDLE_REQUIRED_MEMBERS_MISSING: {missing}")
    forbidden_suffixes = {
        ".npy",
        ".npz",
        ".safetensors",
        ".bin",
        ".pt",
        ".pth",
        ".mp4",
        ".avi",
        ".mkv",
        ".jpg",
        ".jpeg",
        ".png",
    }
    forbidden_names = {"gt.jsonl", "queries.jsonl", "annotation_audit.jsonl"}
    for path in files:
        name = path.relative_to(root).as_posix().casefold()
        if (
            path.suffix.casefold() in forbidden_suffixes
            or path.name.casefold() in forbidden_names
            or "sealed" in name
            or "raw_video" in name
        ):
            raise RuntimeError(f"SCA1_BUNDLE_FORBIDDEN_MEMBER: {name}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(target, "w", ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(root).as_posix())
    return {
        "path": str(target),
        "sha256": sha256_file(target),
        "size_bytes": target.stat().st_size,
        "member_count": len(files),
    }


__all__ = [
    "REQUIRED_BUNDLE_MEMBERS",
    "build_text_identity_rows",
    "create_bundle",
    "evaluate_post_gt",
    "run_pre_gt_arm",
    "validate_pre_gt_integrity",
    "write_manifests",
]
