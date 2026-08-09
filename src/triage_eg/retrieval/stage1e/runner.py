"""Stage 1E evaluation-ingest and language-path freeze runner."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from triage_eg.retrieval.stage1b.writers import write_json, write_jsonl
from triage_eg.retrieval.stage1d.review import REVIEW_FIELDS

from .contracts import (
    AI_REVIEW_STATUS,
    CLIP_CANDIDATE,
    EVALUATION_MODE,
    EXPECTED_JUDGMENTS,
    EXPECTED_PAIRS,
    HUMAN_REVIEW_STATUS,
    JUDGE_MODEL,
    JUDGE_PROVIDER,
    LANGUAGE_BRIDGE_INTERNAL_GATE,
    LANGUAGE_BRIDGE_QUALITY_STATUS,
    LANGUAGE_PATH_STATUS,
    PAIR_METRIC_FIELDS,
    STAGE1E_VERSION,
    STAGE2_READINESS,
    TRANSLATOR_MODEL_ID,
    TRANSLATOR_REVISION,
)
from .evaluation import validate_and_score_ai_review, validate_supplied_ai_metrics


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_inventory(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    }


def _fingerprint(inventory: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(inventory.items()):
        digest.update(f"{name}\0{value}\n".encode())
    return digest.hexdigest()


def _write_review_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(REVIEW_FIELDS))
        writer.writeheader()
        writer.writerows(rows)


def _write_pair_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(PAIR_METRIC_FIELDS))
        writer.writeheader()
        writer.writerows(rows)


def _metrics_markdown(metrics: dict[str, Any]) -> str:
    lines = [
        "# Stage 1E AI-Judged Blinded Review Metrics",
        "",
        f"- Evaluation mode: `{EVALUATION_MODE}`",
        f"- AI review: `{AI_REVIEW_STATUS}` ({EXPECTED_JUDGMENTS}/{EXPECTED_JUDGMENTS})",
        f"- Human review: `{HUMAN_REVIEW_STATUS}`",
        "",
        "| Arm | Relevant@1 | Relevant@5 | Graded@5 | Uncertain |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm, values in metrics["per_arm"].items():
        lines.append(
            f"| {arm} | {values['ai_relevance_rate_top1']:.6f} | "
            f"{values['ai_relevance_rate_top5']:.6f} | "
            f"{values['ai_graded_relevance_top5']:.6f} | "
            f"{values['uncertain_count']} |"
        )
    lines.extend(
        [
            "",
            "These are AI-judged internal qualitative metrics, not human labels or "
            "competition Recall@K.",
        ]
    )
    return "\n".join(lines) + "\n"


def _report(summary: dict[str, Any], metrics: dict[str, Any]) -> str:
    comparison = metrics["pair_comparison"]
    metric_rows = []
    for arm in ("EN_DIRECT", "VI_DIRECT", "VI_TRANSLATED_EN"):
        values = metrics["per_arm"][arm]
        metric_rows.append(
            f"| {arm} | {values['ai_relevance_rate_top1']:.6f} | "
            f"{values['ai_relevance_rate_top5']:.6f} | "
            f"{values['ai_graded_relevance_top5']:.6f} |"
        )
    metric_table = "\n".join(metric_rows)
    return f"""# Stage 1E AI Evaluation Gate and Language Path Freeze

## Status

- Stage 1E execution: `COMPLETE`
- AI review status: `{AI_REVIEW_STATUS}`
- Human review status: `{HUMAN_REVIEW_STATUS}`
- Language bridge internal gate: `{LANGUAGE_BRIDGE_INTERNAL_GATE}`
- Language path status: `{LANGUAGE_PATH_STATUS}`
- Stage 2 readiness: `{STAGE2_READINESS}`

## Evaluation provenance

- Mode: `{EVALUATION_MODE}`
- Judge: `{JUDGE_PROVIDER} / {JUDGE_MODEL}`
- Judgments: `{EXPECTED_JUDGMENTS}/{EXPECTED_JUDGMENTS}`
- Blinded during judgment: true
- Unblinded only for scoring: true
- Score strings restored from frozen identity within one ULP: 
  {metrics["identity_validation"]["score_strings_canonicalized_within_one_ulp"]}

## Recomputed evidence

| Arm | AI Relevant@1 | AI Relevant@5 | AI Graded@5 |
|---|---:|---:|---:|
{metric_table}

Translated versus VI direct by pair, graded Top-5: 
{comparison["translated_better_than_vi_direct_pairs_by_graded_top5"]} better, 
{comparison["translated_tied_vi_direct_pairs_by_graded_top5"]} tied, 
{comparison["translated_worse_than_vi_direct_pairs_by_graded_top5"]} worse.

## Frozen operational paths

- English: direct verified OpenAI CLIP ViT-B/32 text encoding.
- Vietnamese: OPUS-MT vi→en at exact revision `{TRANSLATOR_REVISION}`, then the
  same verified OpenAI CLIP encoder and frozen Stage 1A exact BTC index.

## Carried failure modes

- `difficult_01`: `SEMANTIC_RETRIEVAL_FAILURE_AFTER_TRANSLATION`.
- `obj_01`: English and translated paths are both weak; translation alone does
  not repair every CLIP semantic retrieval failure.

## Non-claims

- AI judgments are not human judgments.
- This does not prove the translator is globally optimal.
- This does not prove final competition retrieval quality.
- No retrieval, translation, encoding, ranking, or index artifact was regenerated.
"""


def run_stage1e_language_path_freeze(
    stage1d_root: str | Path,
    ai_review_root: str | Path,
    output_root: str | Path,
    *,
    build_git_commit: str | None = None,
) -> dict[str, Any]:
    """Ingest frozen AI judgments, score them, and emit the Stage 2 path contract."""

    frozen = Path(stage1d_root).expanduser().resolve(strict=True)
    ai_root = Path(ai_review_root).expanduser().resolve(strict=True)
    output = Path(output_root).expanduser().resolve(strict=False)
    if output.exists():
        raise FileExistsError(f"Stage 1E output already exists: {output}")
    required = {
        name: ai_root / name
        for name in (
            "review_template_ai_judged.csv",
            "ai_pair_metrics.csv",
            "ai_review_metrics.json",
            "ai_review_metrics.md",
        )
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"AI review artifacts missing: {missing}")
    supplied_report = required["ai_review_metrics.md"].read_text(encoding="utf-8")
    if EVALUATION_MODE not in supplied_report or JUDGE_MODEL not in supplied_report:
        raise ValueError("AI_REVIEW_PROVENANCE_MISMATCH: supplied Markdown report")
    frozen_summary = json.loads((frozen / "stage1d_summary.json").read_text(encoding="utf-8"))
    if frozen_summary.get("execution_status") != "COMPLETE_WITH_WARNINGS":
        raise ValueError("STAGE1D_FROZEN_INPUT_INVALID: execution status")
    before = _tree_inventory(frozen)
    metrics, normalized, pair_metrics = validate_and_score_ai_review(
        frozen, required["review_template_ai_judged.csv"]
    )
    validate_supplied_ai_metrics(
        metrics,
        pair_metrics,
        required["ai_review_metrics.json"],
        required["ai_pair_metrics.csv"],
    )
    translated = metrics["per_arm"]["VI_TRANSLATED_EN"]["ai_graded_relevance_top5"]
    direct = metrics["per_arm"]["VI_DIRECT"]["ai_graded_relevance_top5"]
    pair_comparison = metrics["pair_comparison"]
    if not (
        translated > direct
        and pair_comparison["translated_better_than_vi_direct_pairs_by_graded_top5"]
        > EXPECTED_PAIRS // 2
        and pair_comparison["translated_worse_than_vi_direct_pairs_by_graded_top5"] == 0
    ):
        raise ValueError("LANGUAGE_BRIDGE_INTERNAL_GATE_REJECTED")
    encoder = frozen_summary["stage1b_encoder"]
    translator = frozen_summary["translator"]
    if (
        encoder.get("candidate_id") != CLIP_CANDIDATE
        or encoder.get("compatibility_status") != "VERIFIED"
        or encoder.get("model_space_status") != "MODEL_SPACE_VERIFIED"
        or translator.get("model_id") != TRANSLATOR_MODEL_ID
        or translator.get("exact_revision") != TRANSLATOR_REVISION
    ):
        raise ValueError("FROZEN_MODEL_PROVENANCE_MISMATCH")

    started = datetime.now(UTC).isoformat()
    staging = output.with_name(f".{output.name}.building")
    if staging.exists():
        shutil.rmtree(staging)
    try:
        ai_output = staging / "ai_evaluation"
        _write_review_csv(ai_output / "review_template_ai_judged_canonical.csv", normalized)
        _write_pair_metrics(ai_output / "ai_pair_metrics.csv", pair_metrics)
        write_json(ai_output / "ai_review_metrics.json", metrics)
        (ai_output / "ai_review_metrics.md").write_text(
            _metrics_markdown(metrics), encoding="utf-8"
        )
        source_hashes = {name: _sha256(path) for name, path in required.items()}
        evaluation_contract = {
            "evaluation_mode": EVALUATION_MODE,
            "judge": {"provider": JUDGE_PROVIDER, "model": JUDGE_MODEL},
            "blinded_during_judgment": True,
            "unblinded_only_for_scoring": True,
            "judgments_expected": EXPECTED_JUDGMENTS,
            "judgments_completed": EXPECTED_JUDGMENTS,
            "label_space": ["RELEVANT", "PARTIAL", "IRRELEVANT", "UNCERTAIN"],
            "human_review_performed": False,
            "ai_review_status": AI_REVIEW_STATUS,
            "human_review_status": HUMAN_REVIEW_STATUS,
            "identity_validation": metrics["identity_validation"],
            "source_artifact_sha256": source_hashes,
        }
        write_json(ai_output / "ai_evaluation_contract.json", evaluation_contract)
        language_contract = {
            "stage1e_version": STAGE1E_VERSION,
            "english_path": {"mode": "DIRECT", "text_encoder": CLIP_CANDIDATE},
            "vietnamese_path": {
                "mode": "TRANSLATE_TO_ENGLISH_THEN_CLIP",
                "translator": {
                    "model_id": TRANSLATOR_MODEL_ID,
                    "exact_revision": TRANSLATOR_REVISION,
                },
                "text_encoder": CLIP_CANDIDATE,
            },
            "stage1_index_fingerprint": frozen_summary["stage1_index_fingerprint"],
            "clip_compatibility": encoder["compatibility_status"],
            "model_space_status": encoder["model_space_status"],
            "decision_basis": "AI_JUDGED_STAGE1D_TRANSLATION_ABLATION",
            "ai_review_status": AI_REVIEW_STATUS,
            "human_review_status": HUMAN_REVIEW_STATUS,
            "language_bridge_internal_gate": LANGUAGE_BRIDGE_INTERNAL_GATE,
            "language_path_status": LANGUAGE_PATH_STATUS,
        }
        write_json(staging / "language_path_contract.json", language_contract)
        issues = [
            {
                "severity": "WARNING",
                "code": "AI_REVIEW_SCORE_SERIALIZATION_CANONICALIZED",
                "message": (
                    "AI CSV score strings were restored from frozen identities within one ULP"
                ),
                "evidence": metrics["identity_validation"],
            },
            {
                "severity": "WARNING",
                "code": "SEMANTIC_RETRIEVAL_FAILURE_AFTER_TRANSLATION",
                "pair_id": "difficult_01",
                "message": "Translation did not rescue Top-5 semantic retrieval",
                "carry_forward_to": "STAGE2_EVALUATION",
            },
            {
                "severity": "WARNING",
                "code": "LANGUAGE_BRIDGE_INSUFFICIENT_FOR_SEMANTIC_FAILURE",
                "pair_id": "obj_01",
                "message": "English and translated paths were both weak for the red-car intent",
                "carry_forward_to": "STAGE2_EVALUATION",
            },
        ]
        write_jsonl(staging / "issues.jsonl", issues)
        summary = {
            "stage1e_version": STAGE1E_VERSION,
            "stage1e_execution": "COMPLETE",
            "evaluation_mode": EVALUATION_MODE,
            "ai_review_status": AI_REVIEW_STATUS,
            "ai_judgments": f"{EXPECTED_JUDGMENTS}/{EXPECTED_JUDGMENTS}",
            "human_review_status": HUMAN_REVIEW_STATUS,
            "language_bridge_internal_gate": LANGUAGE_BRIDGE_INTERNAL_GATE,
            "language_bridge_quality_status": LANGUAGE_BRIDGE_QUALITY_STATUS,
            "language_path_status": LANGUAGE_PATH_STATUS,
            "vi_operational_path": "VI_TO_EN_OPUS_MT_THEN_CLIP",
            "stage2_readiness": STAGE2_READINESS,
            "ai_metrics": metrics,
            "language_path_contract": language_contract,
            "stage1d_frozen_artifact_fingerprint": _fingerprint(before),
            "issues": {"total": len(issues), "by_severity": {"WARNING": len(issues)}},
        }
        write_json(staging / "stage1e_summary.json", summary)
        (staging / "stage1e_report.md").write_text(_report(summary, metrics), encoding="utf-8")
        manifest = {
            "stage1e_version": STAGE1E_VERSION,
            "status": "COMPLETE",
            "build_git_commit": build_git_commit,
            "started_at": started,
            "completed_at": datetime.now(UTC).isoformat(),
            "stage1d_root": str(frozen),
            "ai_review_root": str(ai_root),
            "output_root": str(output),
            "stage1d_frozen_artifact_fingerprint": _fingerprint(before),
            "source_artifact_sha256": source_hashes,
            "retrieval_invoked": False,
            "encoder_invoked": False,
            "translator_invoked": False,
            "model_downloaded": False,
            "stage1d_artifacts_modified": False,
        }
        write_json(staging / "run_manifest.json", manifest)
        after = _tree_inventory(frozen)
        if before != after:
            raise RuntimeError("FROZEN_STAGE1D_ARTIFACTS_CHANGED")
        staging.parent.mkdir(parents=True, exist_ok=True)
        staging.replace(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return summary


__all__ = ["run_stage1e_language_path_freeze"]
