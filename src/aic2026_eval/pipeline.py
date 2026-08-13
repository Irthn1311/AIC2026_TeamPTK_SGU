"""Bounded orchestration for TEAM-EVAL bootstrap and dense-render passes."""

from __future__ import annotations

import shutil
import subprocess
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .census import build_corpus_inventory, build_usage_census
from .contracts import contract_document
from .holdout import select_heldout_candidates
from .io import read_jsonl, write_json, write_jsonl
from .mapping import audit_l21_bootstrap
from .render import render_dense_requests, render_overview_atlas
from .report import create_bundle


def _commit(repository_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or "UNKNOWN"


def _reset_output(root: Path) -> None:
    if not root.name.startswith("aic2026_team_eval"):
        raise ValueError("output directory name must start with aic2026_team_eval")
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)


def run_bootstrap(
    *,
    dataset_root: str | Path,
    repository_root: str | Path,
    output_root: str | Path,
    l21_bootstrap_zip: str | Path | None = None,
    manual_exclude_path: str | Path | None = None,
    seed: int = 20260821,
    build_commit: str | None = None,
) -> dict[str, Any]:
    dataset = Path(dataset_root).expanduser().resolve(strict=True)
    repository = Path(repository_root).expanduser().resolve(strict=True)
    output = Path(output_root)
    _reset_output(output)
    inventory, corpus_summary, inventory_issues = build_corpus_inventory(dataset)
    census, usage_summary = build_usage_census(
        inventory,
        repository,
        manual_exclude_path=manual_exclude_path,
    )
    heldout, selection_report = select_heldout_candidates(
        inventory,
        census,
        seed=seed,
    )
    atlas_manifest, atlas_index, atlas_issues = render_overview_atlas(heldout, output)
    l21_rows, l21_summary = audit_l21_bootstrap(
        l21_bootstrap_zip,
        inventory,
        temporary_root=output / "_l21_tmp",
    )
    issues = inventory_issues + atlas_issues
    write_json(output / "corpus_summary.json", corpus_summary)
    write_jsonl(output / "corpus_inventory.jsonl", inventory)
    write_json(output / "video_usage_summary.json", usage_summary)
    write_jsonl(output / "video_usage_census.jsonl", census)
    write_jsonl(output / "heldout_candidate_manifest.jsonl", heldout)
    write_json(output / "heldout_selection_report.json", selection_report)
    write_jsonl(output / "atlas_manifest.jsonl", atlas_manifest)
    write_json(output / "atlas_index.json", atlas_index)
    if l21_summary["status"] != "SKIPPED_NO_INPUT":
        write_jsonl(output / "l21_mapping_audit.jsonl", l21_rows)
        write_json(output / "l21_mapping_summary.json", l21_summary)
    write_json(output / "evaluation_contract.json", contract_document())
    write_jsonl(output / "issues.jsonl", issues)
    commit = build_commit or _commit(repository)
    run_manifest = {
        "sprint": "TEAM-EVAL_E0_E1",
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": commit,
        "dataset_root": str(dataset),
        "seed": seed,
        "semantic_query_generation_performed": False,
        "blind_benchmark_complete": False,
        "source_pool_only": True,
        "l21_bootstrap_zip": str(l21_bootstrap_zip) if l21_bootstrap_zip else None,
        "statuses": {
            "CORPUS_INVENTORY": corpus_summary["status"],
            "CONTAMINATION_CENSUS": usage_summary["status"],
            "L21_MAPPING_AUDIT": l21_summary["status"],
            "ATLAS_STATUS": atlas_index["status"],
        },
    }
    write_json(output / "run_manifest.json", run_manifest)
    (output / "README.md").write_text(
        "# AIC2026 TEAM-EVAL E0/E1 source pool\n\n"
        "Team-neutral inventory, contamination census, deterministic held-out candidate "
        "roles, and BTC overview evidence. This is not a completed blind benchmark and "
        "contains no semantic query or GT authoring. Raw videos and model assets are excluded.\n\n"
        "Validate a system export with `python -m aic2026_eval.validate_predictions "
        "--queries queries.jsonl --predictions predictions.jsonl --inventory "
        "corpus_inventory.jsonl`. Score adjudicated GT with `python -m "
        "aic2026_eval.evaluate_predictions --help`. Internal QA alias scoring is "
        "deterministic but is not claimed to reproduce official BTC semantic matching.\n",
        encoding="utf-8",
    )
    shutil.rmtree(output / "_l21_tmp", ignore_errors=True)
    zip_path = create_bundle(
        output,
        "/kaggle/working/aic2026_team_eval_e01_bundle.zip"
        if str(output).startswith("/kaggle/working")
        else output.parent / "aic2026_team_eval_e01_bundle.zip",
    )
    return {
        "TEAM_EVAL_INFRA_STATUS": (
            "READY"
            if corpus_summary["status"] == usage_summary["status"] == "PASS"
            and atlas_index["status"] == "READY"
            and selection_report["blind_sealed_video_overlap"] == 0
            else "BLOCKED"
        ),
        "CORPUS_INVENTORY": corpus_summary["status"],
        "CONTAMINATION_CENSUS": usage_summary["status"],
        "L21_MAPPING_AUDIT": (
            "SKIPPED" if l21_summary["status"] == "SKIPPED_NO_INPUT" else l21_summary["status"]
        ),
        "HELDOUT_CANDIDATES": len(heldout),
        "BLIND_CANDIDATE_VIDEOS": selection_report["blind_candidate_count"],
        "SEALED_CANDIDATE_VIDEOS": selection_report["sealed_candidate_count"],
        "BLIND_SEALED_VIDEO_OVERLAP": selection_report["blind_sealed_video_overlap"],
        "ATLAS_STATUS": atlas_index["status"],
        "DENSE_RENDERER_STATUS": "READY",
        "PREDICTION_VALIDATOR": "READY",
        "SHARED_EVALUATOR": "READY",
        "READY_FOR_AI_SEMANTIC_SELECTION": ("YES" if atlas_index["status"] == "READY" else "NO"),
        "corpus_summary": corpus_summary,
        "usage_summary": usage_summary,
        "selection_report": selection_report,
        "l21_mapping_summary": l21_summary,
        "atlas_index": atlas_index,
        "issues": issues,
        "zip_path": str(zip_path),
    }


def run_dense(
    *,
    dataset_root: str | Path,
    repository_root: str | Path,
    anchor_requests_path: str | Path,
    output_root: str | Path,
    build_commit: str | None = None,
) -> dict[str, Any]:
    dataset = Path(dataset_root).expanduser().resolve(strict=True)
    repository = Path(repository_root).expanduser().resolve(strict=True)
    output = Path(output_root)
    _reset_output(output)
    requests = read_jsonl(anchor_requests_path)
    requested_ids = {row.get("video_id") for row in requests}
    inventory, _, inventory_issues = build_corpus_inventory(dataset, video_ids=requested_ids)
    manifest, render_issues = render_dense_requests(requests, inventory, output)
    issues = inventory_issues + render_issues
    write_jsonl(output / "anchor_requests.jsonl", requests)
    write_jsonl(output / "dense_manifest.jsonl", manifest)
    write_jsonl(output / "issues.jsonl", issues)
    write_json(
        output / "run_manifest.json",
        {
            "sprint": "TEAM-EVAL_E1_DENSE_RENDER",
            "created_at": datetime.now(UTC).isoformat(),
            "git_commit": build_commit or _commit(repository),
            "dataset_root": str(dataset),
            "anchor_count": len(requests),
            "rendered_anchor_count": len(manifest),
            "frame_identity_policy": "EXACT_ACTUAL_RAW_FRAME_IDX",
            "semantic_judgment_performed": False,
            "mode_counts": dict(Counter(row["mode"] for row in manifest)),
        },
    )
    (output / "README.md").write_text(
        "# AIC2026 TEAM-EVAL dense raw evidence\n\n"
        "Every tile is labelled with exact `actual_frame_idx`. This bundle contains no "
        "semantic judgment, raw video, model, feature dump, or runtime cache.\n",
        encoding="utf-8",
    )
    zip_path = create_bundle(
        output,
        "/kaggle/working/aic2026_team_eval_dense_bundle.zip"
        if str(output).startswith("/kaggle/working")
        else output.parent / "aic2026_team_eval_dense_bundle.zip",
        dense=True,
    )
    return {
        "DENSE_RENDERER_STATUS": "READY" if len(manifest) == len(requests) else "FAIL",
        "anchor_count": len(requests),
        "rendered_anchor_count": len(manifest),
        "issues": issues,
        "zip_path": str(zip_path),
    }


__all__ = ["run_bootstrap", "run_dense"]
