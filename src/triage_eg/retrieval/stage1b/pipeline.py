"""Stage 1B orchestration without Stage 0 or Stage 1A rebuilds."""

from __future__ import annotations

import json
import os
import shutil
from collections import Counter
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from triage_eg.data.dataset_survey import KAGGLE_INPUT_ROOT, _is_within
from triage_eg.retrieval.stage1.builder import resolve_git_commit
from triage_eg.retrieval.stage1.stage0_loader import load_stage0_bundle
from triage_eg.retrieval.stage1b.assets import AdapterFactory, issue, preflight_candidate
from triage_eg.retrieval.stage1b.contracts import STAGE1B_VERSION, CandidateContract, Stage1BConfig
from triage_eg.retrieval.stage1b.evidence import discover_encoder_evidence
from triage_eg.retrieval.stage1b.probe import ALIGNMENT_BASIS, probe_candidate
from triage_eg.retrieval.stage1b.registry import load_candidate_registry
from triage_eg.retrieval.stage1b.sampling import select_probe_samples
from triage_eg.retrieval.stage1b.smoke import run_text_smoke
from triage_eg.retrieval.stage1b.writers import write_json, write_jsonl


@dataclass(frozen=True)
class Stage1BResult:
    output_root: Path
    summary: dict[str, Any]
    selected_contract: dict[str, Any]
    reused: bool


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _validate(config: Stage1BConfig) -> tuple[Path, Path, Path, dict, dict]:
    dataset = config.dataset_root.expanduser().resolve(strict=True)
    stage1 = config.stage1_root.expanduser().resolve(strict=True)
    output = config.output_root.expanduser().resolve(strict=False)
    if config.strict_root and not _is_within(dataset, KAGGLE_INPUT_ROOT):
        raise ValueError("Strict dataset root must be below /kaggle/input")
    if _is_within(output, dataset) or _is_within(output, stage1):
        raise ValueError("Stage 1B output must not write into dataset or Stage 1A")
    if output == Path(output.anchor) or len(output.parts) < 3:
        raise ValueError("Stage 1B output root is too broad")
    summary = _json(stage1 / "stage1_summary.json")
    manifest = _json(stage1 / "index/index_manifest.json")
    if summary.get("index_status") != "COMPLETE" or manifest.get("status") != "COMPLETE":
        raise ValueError("Stage 1A index must be COMPLETE")
    if summary.get("next_stage_readiness", {}).get("corpus_index") not in {
        "READY",
        "READY_WITH_TIE_WARNINGS",
    }:
        raise ValueError("Stage 1A corpus index is not ready")
    for name in ("clip_vectors.f16.npy", "vector_norms.f32.npy", "frame_n.npy"):
        if not (stage1 / "index" / name).is_file():
            raise FileNotFoundError(f"Missing Stage 1A index artifact: {name}")
    return dataset, stage1, output, summary, manifest


def _contract_dict(candidate: CandidateContract, provenance: dict[str, Any]) -> dict[str, Any]:
    value = asdict(candidate)
    value["contract_version"] = STAGE1B_VERSION
    value["checkpoint_fingerprint"] = provenance.get("checkpoint_fingerprint")
    checkpoint_sha = _candidate_checkpoint_sha(candidate, provenance)
    if checkpoint_sha:
        value["checkpoint_sha256"] = checkpoint_sha
    elif not value.get("checkpoint_sha256"):
        value.pop("checkpoint_sha256")
    value["asset_source"] = provenance.get("asset_source", "UNKNOWN")
    value["asset_provenance"] = {
        key: provenance.get(key)
        for key in (
            "source_repository",
            "source_commit",
            "module_file",
            "module_origin_valid",
            "checkpoint_size_bytes",
            "checkpoint_sha256",
            "declared_hash_match",
        )
        if provenance.get(key) is not None
    }
    value["notes"] = list(candidate.notes)
    return value


def _issues_summary(issues: list[dict]) -> dict[str, Any]:
    return {
        "total": len(issues),
        "by_code": dict(sorted(Counter(item["code"] for item in issues).items())),
        "by_severity": dict(sorted(Counter(item["severity"] for item in issues).items())),
    }


def _empty_alignment(alignment_basis: str, *, diagnostic_only: bool) -> dict[str, Any]:
    return {
        "alignment_basis": alignment_basis,
        "diagnostic_only": diagnostic_only,
        "top1_count": 0,
        "top5_count": 0,
        "top20_count": 0,
        "top1_rate": 0.0,
        "top5_rate": 0.0,
        "top20_rate": 0.0,
        "mean_rank_within_returned_topk": None,
        "max_rank_within_returned_topk": None,
        "missing_from_returned_topk_count": 0,
    }


def _empty_candidate_summary(
    candidate: CandidateContract,
    samples_requested: int,
    *,
    samples_failed: int,
    dimension_match: bool,
    decision: str,
    reasons: list[str],
) -> dict[str, Any]:
    literal = _empty_alignment("LITERAL_GLOBAL_ROW_DIAGNOSTIC", diagnostic_only=True)
    equivalent = _empty_alignment(ALIGNMENT_BASIS, diagnostic_only=False)
    return {
        "candidate_id": candidate.candidate_id,
        "samples_requested": samples_requested,
        "samples_completed": 0,
        "samples_failed": samples_failed,
        "cosine": {
            key: None for key in ("min", "max", "mean", "median", "p05", "p95", "std")
        },
        "literal_target_alignment": literal,
        "stored_vector_equivalence_alignment": equivalent,
        "gate_alignment_basis": ALIGNMENT_BASIS,
        "retrieval_alignment": {
            "alignment_basis": ALIGNMENT_BASIS,
            "target_top1_count": 0,
            "target_top5_count": 0,
            "target_top20_count": 0,
            "target_top1_rate": 0.0,
            "target_top5_rate": 0.0,
            "target_top20_rate": 0.0,
            "mean_target_rank": None,
            "max_target_rank": None,
        },
        "dimension_match": dimension_match,
        "finite_all": False,
        "decision": decision,
        "decision_reasons": reasons,
    }


def _candidate_checkpoint_sha(
    candidate: CandidateContract, provenance: dict[str, Any]
) -> str | None:
    return (
        provenance.get("checkpoint_sha256")
        or provenance.get("checkpoint_fingerprint")
        or candidate.checkpoint_sha256
    )


def _official_candidate_report(
    candidate: CandidateContract | None,
    provenance: dict[str, Any],
    runtime: dict[str, Any],
    metrics: dict[str, Any] | None,
    smoke_status: str,
    *,
    selected: bool,
) -> str:
    decision = candidate.compatibility_status if candidate else "NOT_EVALUATED"
    label = f"[{decision}]"
    candidate_id = candidate.candidate_id if candidate else "not evaluated"
    return (
        "\n\n# Official OpenAI CLIP Candidate\n\n"
        f"{label} Candidate: {candidate_id}\n"
        f"\nSelected: {str(selected).lower()}\n"
        "\n# Asset Provenance\n\n"
        f"[INFERRED] {json.dumps(provenance, ensure_ascii=False, default=str)}\n"
        "\n# Module Origin\n\n"
        f"{label} {provenance.get('module_file', 'UNKNOWN')}\n"
        "\n# Checkpoint Integrity\n\n"
        f"{label} SHA-256: {provenance.get('checkpoint_sha256', 'UNKNOWN')}\n"
        "\n# Image Preprocessing Source\n\n"
        f"{label} {runtime.get('preprocess_source', 'UNKNOWN')}\n"
        "\n# Empirical Compatibility Metrics\n\n"
        f"{label} {json.dumps(metrics, ensure_ascii=False, default=str)}\n"
        "\n# Retrieval Alignment\n\n"
        f"{label} {json.dumps(metrics.get('retrieval_alignment') if metrics else None)}\n"
        "\n# Compatibility Decision\n\n"
        f"{label} {decision}\n"
        "\n# Text Smoke Status\n\n"
        f"{label} {smoke_status}\n"
        "\n# Offline Runtime Guarantee\n\n"
        "[VERIFIED] Only an absolute local checkpoint path is accepted; the package "
        "download helper is blocked during model load.\n"
        "\n# Non-Claims\n\n"
        "[UNKNOWN] BTC original implementation is not proven. Vietnamese retrieval "
        "quality is not proven.\n"
    )


def _evaluated_candidates_report(
    candidates: list[CandidateContract],
    provenances: dict[str, dict[str, Any]],
    runtime_manifests: dict[str, dict[str, Any]],
    summaries: list[dict[str, Any]],
    selected: CandidateContract | None,
) -> str:
    summaries_by_id = {item["candidate_id"]: item for item in summaries}
    sections = ["\n\n## Evaluated Candidates\n"]
    enabled = [candidate for candidate in candidates if candidate.enabled]
    if not enabled:
        sections.append("\n- None\n")
    for candidate in enabled:
        provenance = provenances.get(candidate.candidate_id, {})
        runtime = runtime_manifests.get(candidate.candidate_id, {})
        metrics = summaries_by_id.get(candidate.candidate_id, {})
        image_execution = runtime.get("image_execution", {})
        text_execution = runtime.get("text_execution", {})
        equivalence_alignment = json.dumps(
            metrics.get("stored_vector_equivalence_alignment"), ensure_ascii=False
        )
        sections.extend(
            [
                f"\n### {candidate.candidate_id}\n",
                f"\n- Implementation: {candidate.implementation}\n",
                f"- Architecture: {candidate.architecture}\n",
                f"- Pretrained: {candidate.pretrained}\n",
                f"- Candidate decision: {candidate.compatibility_status}\n",
                "- Decision reasons: "
                f"{json.dumps(metrics.get('decision_reasons', []), ensure_ascii=False)}\n",
                f"- Selected: {str(selected == candidate).lower()}\n",
                f"- Checkpoint path: {candidate.checkpoint_path or 'UNKNOWN'}\n",
                f"- Checkpoint size: {provenance.get('checkpoint_size_bytes', 'UNKNOWN')}\n",
                "- Checkpoint SHA-256: "
                f"{_candidate_checkpoint_sha(candidate, provenance) or 'UNKNOWN'}\n",
                f"- Declared hash match: {provenance.get('declared_hash_match', 'UNKNOWN')}\n",
                f"- Source repository: {provenance.get('source_repository', 'UNKNOWN')}\n",
                f"- Source commit: {provenance.get('source_commit', 'UNKNOWN')}\n",
                f"- Module origin: {provenance.get('module_file', 'UNKNOWN')}\n",
                f"- Module origin valid: {provenance.get('module_origin_valid', 'UNKNOWN')}\n",
                f"- Required API present: {provenance.get('required_api_present', 'UNKNOWN')}\n",
                f"- Device: {provenance.get('selected_device', 'UNKNOWN')}\n",
                f"- Model dtype: {runtime.get('model_parameter_dtype', 'UNKNOWN')}\n",
                f"- Image output dtype: {image_execution.get('output_dtype', 'UNKNOWN')}\n",
                f"- Text output dtype: {text_execution.get('output_dtype', 'UNKNOWN')}\n",
                f"- Preprocessing source: {runtime.get('preprocess_source', 'UNKNOWN')}\n",
                f"- Tokenizer: {candidate.tokenizer or 'UNKNOWN'}\n",
                f"- Runtime: {json.dumps(runtime, ensure_ascii=False, default=str)}\n",
                "- Samples: "
                f"{metrics.get('samples_completed', 0)}/{metrics.get('samples_requested', 0)} "
                f"completed; {metrics.get('samples_failed', 0)} failed\n",
                f"- Cosine: {json.dumps(metrics.get('cosine'), ensure_ascii=False)}\n",
                "- Literal target-row alignment: "
                f"{json.dumps(metrics.get('literal_target_alignment'), ensure_ascii=False)}\n",
                "- Stored-vector equivalence alignment: "
                f"{equivalence_alignment}\n",
                f"- Gate alignment basis: {metrics.get('gate_alignment_basis', ALIGNMENT_BASIS)}\n",
            ]
        )
    sections.extend(
        [
            "\n## Selected Candidate\n",
            f"\n- Candidate: {selected.candidate_id if selected else 'None'}\n",
        ]
    )
    return "".join(sections)


def _default_smoke_queries() -> list[dict[str, str]]:
    return [
        {
            "query_id": "q_en_001",
            "text": "a person cooking in a kitchen",
            "language": "en",
            "type": "action",
        },
        {
            "query_id": "q_vi_001",
            "text": "một người đang nấu ăn trong bếp",
            "language": "vi",
            "type": "action",
        },
        {
            "query_id": "q_en_002",
            "text": "a red car driving on a road",
            "language": "en",
            "type": "object",
        },
        {
            "query_id": "q_vi_002",
            "text": "một chiếc ô tô màu đỏ đang chạy trên đường",
            "language": "vi",
            "type": "object",
        },
        {
            "query_id": "q_en_003",
            "text": "a crowded outdoor scene",
            "language": "en",
            "type": "scene",
        },
        {
            "query_id": "q_vi_003",
            "text": "một cảnh ngoài trời đông người",
            "language": "vi",
            "type": "difficult",
        },
    ]


def _load_smoke_queries(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return _default_smoke_queries()
    records = [
        json.loads(line)
        for line in path.resolve(strict=True).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    required = {"query_id", "text", "language", "type"}
    if not records or any(
        not isinstance(item, dict) or not required <= set(item) for item in records
    ):
        raise ValueError("Invalid Stage 1B smoke query JSONL")
    return records


def _publish(staging: Path, output: Path, overwrite: bool) -> None:
    if output.exists():
        if not overwrite:
            raise FileExistsError("Stage 1B output exists; use overwrite or reuse-results")
        backup = output.with_name(f".{output.name}.previous")
        if backup.exists():
            raise FileExistsError(f"Stale Stage 1B backup exists: {backup}")
        os.replace(output, backup)
        try:
            os.replace(staging, output)
        except Exception:
            os.replace(backup, output)
            raise
        shutil.rmtree(backup)
    else:
        os.replace(staging, output)


def run_stage1b(
    config: Stage1BConfig,
    *,
    adapter_factory: AdapterFactory | None = None,
) -> Stage1BResult:
    started_at = datetime.now(UTC).isoformat()
    dataset, stage1, output, stage1_summary, stage1_manifest = _validate(config)
    if config.reuse_results:
        summary_path = output / "stage1b_summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError("No complete Stage 1B result exists for reuse")
        summary = _json(summary_path)
        if summary.get("status") != "COMPLETE":
            raise ValueError("Existing Stage 1B result is not COMPLETE")
        return Stage1BResult(
            output, summary, _json(output / "encoder/selected_encoder_contract.json"), True
        )
    if output.exists() and not config.overwrite:
        raise FileExistsError("Stage 1B output exists; use overwrite or reuse-results")
    bundle = load_stage0_bundle(
        config.stage0_root,
        require_full=stage1_manifest.get("vector_count") == 177_321,
    )
    if bundle.summary["mapping_rows"] != stage1_manifest["vector_count"]:
        raise ValueError("Stage 0 and Stage 1A row counts differ")
    candidates, gate, _, config_fingerprint = load_candidate_registry(
        config.candidate_config, config.candidate_ids
    )
    commit, commit_source = resolve_git_commit(config.repo_root, config.build_git_commit)
    staging = output.with_name(f".{output.name}.building")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    for name in ("evidence", "probe", "encoder", "smoke", "logs"):
        (staging / name).mkdir()
    issues: list[dict] = []
    evidence, evidence_summary = discover_encoder_evidence(config.repo_root, dataset)
    if not evidence_summary["authoritative_metadata_found"]:
        issues.append(
            issue(
                "WARNING",
                "AUTHORITATIVE_ENCODER_METADATA_NOT_FOUND",
                None,
                None,
                "No authoritative encoder metadata was found in bounded discovery",
            )
        )
    samples, sample_issues = select_probe_samples(stage1, dataset, config.sample_size, config.seed)
    issues.extend(sample_issues)
    if sample_issues:
        issues.append(
            issue(
                "ERROR",
                "PROBE_SAMPLE_INCOMPLETE",
                None,
                None,
                f"{len(sample_issues)} probe keyframes are unavailable",
            )
        )
    candidate_results: list[dict] = []
    candidate_summaries: list[dict] = []
    finalized_candidates: list[CandidateContract] = []
    provenances: dict[str, dict[str, Any]] = {}
    embeddings: dict[str, np.ndarray] = {}
    encoders: dict[str, Any] = {}
    runtime_manifests: dict[str, dict[str, Any]] = {}
    executed_candidates = 0
    for candidate in candidates:
        if not candidate.enabled:
            finalized_candidates.append(candidate)
            continue
        provenance, preflight_issues = preflight_candidate(candidate, config.repo_root, dataset)
        provenances[candidate.candidate_id] = provenance
        issues.extend(preflight_issues)
        blocking_preflight = [item for item in preflight_issues if item["severity"] == "ERROR"]
        if blocking_preflight:
            decision = (
                "REJECTED"
                if any(
                    item["code"] == "ENCODER_OUTPUT_DIMENSION_MISMATCH"
                    for item in blocking_preflight
                )
                else "BLOCKED"
            )
            blocked = replace(candidate, compatibility_status=decision)
            finalized_candidates.append(blocked)
            candidate_summaries.append(
                _empty_candidate_summary(
                    candidate,
                    len(samples),
                    samples_failed=0,
                    dimension_match=candidate.output_dimension == 512,
                    decision=decision,
                    reasons=[item["code"] for item in blocking_preflight],
                )
            )
            continue
        try:
            executed_candidates += 1
            final, results, summary, matrix, encoder = probe_candidate(
                candidate, samples, stage1, gate, provenance, adapter_factory
            )
        except (
            FileNotFoundError,
            ImportError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as error:
            error_message = str(error)
            prefix = error_message.split(":", 1)[0]
            code = (
                prefix
                if prefix.startswith(
                    (
                        "ENCODER_",
                        "IMAGE_",
                        "OPENAI_CLIP_",
                        "CHECKPOINT_",
                        "ASSET_MANIFEST_",
                        "NETWORK_",
                    )
                )
                else "ENCODER_CHECKPOINT_LOAD_FAILED"
            )
            rejected_codes = {
                "ENCODER_OUTPUT_DIMENSION_MISMATCH",
                "ENCODER_OUTPUT_NON_FINITE",
                "ENCODER_OUTPUT_ZERO_NORM",
            }
            decision = "REJECTED" if code in rejected_codes else "BLOCKED"
            final = replace(candidate, compatibility_status=decision)
            issues.append(
                issue("ERROR", code, candidate, Path(candidate.checkpoint_path or ""), str(error))
            )
            summary = _empty_candidate_summary(
                candidate,
                len(samples),
                samples_failed=len(samples),
                dimension_match=False,
                decision=decision,
                reasons=[code],
            )
            results, matrix, encoder = [], None, None
        finalized_candidates.append(final)
        candidate_results.extend(results)
        candidate_summaries.append(summary)
        if matrix is not None:
            embeddings[candidate.candidate_id] = matrix
            encoders[candidate.candidate_id] = encoder
            runtime_manifest = getattr(encoder, "runtime_manifest", None)
            if callable(runtime_manifest):
                runtime_manifests[candidate.candidate_id] = runtime_manifest()
        for result_item in results:
            if "LITERAL_TARGET_ROW_TIE_DISPLACEMENT" in result_item["issue_codes"]:
                tie_issue = issue(
                    "INFO",
                    "LITERAL_TARGET_ROW_TIE_DISPLACEMENT",
                    candidate,
                    None,
                    "Literal target row was displaced by an exact stored-vector equivalent",
                    target_global_row=result_item["global_row"],
                    returned_top1_global_row=result_item["top1_global_row_when_searched"],
                    equivalent_vector_rank=result_item["equivalent_vector_rank"],
                    literal_target_row_rank=result_item["exact_target_row_rank"],
                    exact_stored_vector_equal=True,
                    alignment_basis=ALIGNMENT_BASIS,
                )
                tie_issue["global_row"] = result_item["global_row"]
                tie_issue["video_id"] = result_item["video_id"]
                issues.append(tie_issue)
        for reason in summary["decision_reasons"]:
            if reason in {
                "PAIRWISE_COSINE_BELOW_GATE",
                "STORED_VECTOR_EQUIVALENCE_ALIGNMENT_BELOW_GATE",
                "ENCODER_PROVENANCE_INCOMPLETE",
            }:
                issues.append(
                    issue(
                        "WARNING",
                        reason,
                        candidate,
                        None,
                        alignment_basis=ALIGNMENT_BASIS,
                    )
                )
            elif reason == "ENCODER_CONTRACT_NOT_REPRODUCIBLE":
                issues.append(issue("ERROR", reason, candidate, None))
    verified = [item for item in finalized_candidates if item.compatibility_status == "VERIFIED"]
    matrix_records = []
    for left_index, left in enumerate(sorted(embeddings)):
        for right in sorted(embeddings)[left_index + 1 :]:
            similarity = np.sum(embeddings[left] * embeddings[right], axis=1) / (
                np.linalg.norm(embeddings[left], axis=1) * np.linalg.norm(embeddings[right], axis=1)
            )
            matrix_records.append(
                {"left": left, "right": right, "mean_cosine": float(np.mean(similarity))}
            )
    selected: CandidateContract | None = None
    if len(verified) == 1:
        selected = verified[0]
    elif len(verified) > 1:
        pairs = [
            item
            for item in matrix_records
            if item["left"] in {v.candidate_id for v in verified}
            and item["right"] in {v.candidate_id for v in verified}
        ]
        if pairs and all(
            item["mean_cosine"] >= gate.implementation_equivalence_cosine_min for item in pairs
        ):
            selected = sorted(
                verified, key=lambda item: (item.runtime_priority, item.candidate_id)
            )[0]
        else:
            issues.append(issue("ERROR", "ENCODER_CANDIDATE_AMBIGUOUS", None, None))
            finalized_candidates = [
                replace(item, compatibility_status="UNVERIFIED") if item in verified else item
                for item in finalized_candidates
            ]
            verified_ids = {item.candidate_id for item in verified}
            for item in candidate_summaries:
                if item["candidate_id"] in verified_ids:
                    item["decision"] = "UNVERIFIED"
                    item["decision_reasons"] = ["ENCODER_CANDIDATE_AMBIGUOUS"]
            verified = []
    selected_contract = (
        {
            **_contract_dict(selected, provenances[selected.candidate_id]),
            "selected_candidate_id": selected.candidate_id,
        }
        if selected
        else {
            "contract_version": STAGE1B_VERSION,
            "compatibility_status": "BLOCKED",
            "selected_candidate_id": None,
            "reason": (
                "No uniquely reproducible candidate passed the project-defined empirical gate"
            ),
        }
    )
    smoke_queries = _load_smoke_queries(config.smoke_queries)
    smoke_results: list[dict] = []
    smoke_status = "NOT_RUN"
    if selected and config.run_text_smoke:
        try:
            smoke_results, smoke_status = run_text_smoke(
                selected, encoders[selected.candidate_id], smoke_queries, stage1, staging
            )
        except (
            FileNotFoundError,
            ImportError,
            OSError,
            PermissionError,
            AttributeError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as error:
            smoke_status = "FAIL"
            text_code = str(error).split(":", 1)[0]
            if text_code in {
                "TEXT_EMBEDDING_INVALID",
                "TEXT_TOKENIZATION_FAILED",
                "TEXT_CONTEXT_LENGTH_EXCEEDED",
            }:
                issues.append(issue("ERROR", text_code, selected, None, str(error)))
            issues.append(issue("ERROR", "TEXT_SEARCH_SMOKE_FAILED", selected, None, str(error)))
    elif not selected:
        issues.append(issue("INFO", "TEXT_ENCODER_BLOCKED", None, None))
    if selected:
        runtime_manifest = getattr(encoders[selected.candidate_id], "runtime_manifest", None)
        if callable(runtime_manifest):
            runtime_manifests[selected.candidate_id] = runtime_manifest()
    counts = Counter(item.compatibility_status for item in finalized_candidates if item.enabled)
    candidate_summaries_by_id = {item["candidate_id"]: item for item in candidate_summaries}
    evaluated_candidates = [
        {
            "candidate_id": item.candidate_id,
            "candidate_decision": item.compatibility_status,
            "decision_reasons": candidate_summaries_by_id.get(item.candidate_id, {}).get(
                "decision_reasons", []
            ),
            "selected": item == selected,
        }
        for item in finalized_candidates
        if item.enabled
    ]
    selected_summary = {
        "candidate_id": selected.candidate_id if selected else None,
        "implementation": selected.implementation if selected else None,
        "architecture": selected.architecture if selected else None,
        "pretrained": selected.pretrained if selected else None,
        "checkpoint_fingerprint": provenances.get(selected.candidate_id, {}).get(
            "checkpoint_fingerprint"
        )
        if selected
        else None,
        "compatibility_status": "VERIFIED" if selected else "BLOCKED",
    }
    summary = {
        "status": "COMPLETE",
        "evaluation_status": "COMPLETE",
        "stage1b_version": STAGE1B_VERSION,
        "dataset_version": "aic25-b1",
        "stage0_root": str(Path(config.stage0_root).resolve(strict=False)),
        "stage1_root": str(stage1),
        "stage1_index_fingerprint": stage1_summary["index_fingerprint"],
        "stage1_build_commit": stage1_manifest["build_git_commit"],
        "compatibility_gate": {
            "name": "PROJECT_DEFINED_EMPIRICAL_GATE",
            "alignment_basis": ALIGNMENT_BASIS,
            "thresholds": asdict(gate),
        },
        "evidence": {
            "authoritative_metadata_found": evidence_summary["authoritative_metadata_found"],
            "records": len(evidence),
        },
        "probe": {
            "sample_size": len(samples),
            "candidates_configured": len(candidates),
            "candidates_executed": executed_candidates,
            "candidates_blocked": counts["BLOCKED"],
            "candidates_rejected": counts["REJECTED"],
            "candidates_unverified": counts["UNVERIFIED"],
            "candidates_verified": counts["VERIFIED"],
        },
        "selected_encoder": selected_summary,
        "selected_candidate": (
            {
                "candidate_id": selected.candidate_id,
                "candidate_decision": selected.compatibility_status,
            }
            if selected
            else None
        ),
        "evaluated_candidates": evaluated_candidates,
        "text_search_available": bool(selected and smoke_status in {"PASS", "PASS_WITH_WARNINGS"}),
        "text_smoke_status": smoke_status,
        "readiness": {
            "encoder_readiness": "VERIFIED" if selected else "BLOCKED",
            "encoder_compatibility": "VERIFIED" if selected else "BLOCKED",
            "text_retrieval": "READY_FOR_QUALITATIVE_TESTING"
            if selected and smoke_status in {"PASS", "PASS_WITH_WARNINGS"}
            else "BLOCKED",
        },
        "non_claims": [
            "Compatibility does not prove retrieval quality",
            "Compatibility does not prove Vietnamese query quality",
            "Smoke testing is not Recall@K evaluation",
        ],
        "issues": _issues_summary(issues),
    }
    run_manifest = {
        "stage1b_version": STAGE1B_VERSION,
        "status": "COMPLETE",
        "build_git_commit": commit,
        "build_git_commit_source": commit_source,
        "config_fingerprint": config_fingerprint,
        "compatibility_gate": asdict(gate),
        "gate_alignment_basis": ALIGNMENT_BASIS,
        "stage1_index_fingerprint": stage1_summary["index_fingerprint"],
        "started_at": started_at,
        "completed_at": datetime.now(UTC).isoformat(),
        "no_stage0_rerun": True,
        "no_stage1_rebuild": True,
        "no_model_download": True,
    }
    write_jsonl(staging / "evidence/evidence_inventory.jsonl", evidence)
    write_json(staging / "evidence/evidence_summary.json", evidence_summary)
    write_jsonl(staging / "probe/probe_samples.jsonl", samples)
    write_jsonl(staging / "probe/candidate_results.jsonl", candidate_results)
    write_jsonl(staging / "probe/candidate_summaries.jsonl", candidate_summaries)
    write_json(
        staging / "probe/compatibility_matrix.json",
        {
            "pairs": matrix_records,
            "equivalence_threshold": gate.implementation_equivalence_cosine_min,
        },
    )
    write_jsonl(
        staging / "encoder/candidate_contracts.jsonl",
        [
            _contract_dict(item, provenances.get(item.candidate_id, {}))
            for item in finalized_candidates
        ],
    )
    write_json(staging / "encoder/selected_encoder_contract.json", selected_contract)
    write_json(
        staging / "encoder/runtime_adapter_manifest.json",
        {
            "selected_candidate_id": selected.candidate_id if selected else None,
            "adapter": selected.implementation if selected else None,
            "model_space_status": "MODEL_SPACE_VERIFIED" if selected else "BLOCKED",
            "selection": "CANONICAL_ADAPTER_SELECTED" if selected else None,
            "selection_criteria": (
                "lowest_configured_runtime_priority_then_candidate_id" if selected else None
            ),
            "runtime": runtime_manifests.get(selected.candidate_id, {}) if selected else {},
            "asset_provenance": (provenances.get(selected.candidate_id, {}) if selected else {}),
            "evaluated_candidates": [
                {
                    **item,
                    "runtime": runtime_manifests.get(item["candidate_id"], {}),
                    "asset_provenance": provenances.get(item["candidate_id"], {}),
                }
                for item in evaluated_candidates
            ],
        },
    )
    write_jsonl(staging / "smoke/smoke_queries.jsonl", smoke_queries)
    write_jsonl(staging / "smoke/smoke_results.jsonl", smoke_results)
    write_jsonl(staging / "issues.jsonl", issues)
    write_json(staging / "run_manifest.json", run_manifest)
    write_json(staging / "stage1b_summary.json", summary)
    blocker_lines = [
        "- "
        f"{item['code']}: candidate={item['candidate_id'] or 'NONE'}, "
        f"path={item['path'] or 'NONE'}"
        for item in issues
        if item["severity"] == "ERROR"
    ]
    decisions = {item["decision"] for item in candidate_summaries}
    if selected:
        next_step = "Candidate passed the compatibility gate; text smoke may proceed."
    elif "REJECTED" in decisions:
        next_step = (
            "Candidate executed successfully but did not satisfy the configured compatibility gate."
        )
    elif "UNVERIFIED" in decisions:
        next_step = (
            "Candidate evidence is insufficient for verification; inspect listed evidence gaps."
        )
    else:
        next_step = "Provide or fix the required local asset or dependency, then rerun Stage 1B."
    enabled_candidates = [item for item in finalized_candidates if item.enabled]
    report_candidate = selected or next(
        (item for item in enabled_candidates if item.implementation == "openai_clip"),
        enabled_candidates[0] if enabled_candidates else None,
    )
    report_provenance = (
        provenances.get(report_candidate.candidate_id, {}) if report_candidate else {}
    )
    report_runtime = (
        runtime_manifests.get(report_candidate.candidate_id, {}) if report_candidate else {}
    )
    report_metrics = (
        candidate_summaries_by_id.get(report_candidate.candidate_id) if report_candidate else None
    )
    (staging / "stage1b_report.md").write_text(
        "# Stage 1B CLIP Encoder Compatibility Validation\n\n"
        f"- Encoder compatibility: {summary['readiness']['encoder_compatibility']}\n"
        f"- Text retrieval: {summary['readiness']['text_retrieval']}\n"
        f"- Candidates verified: {counts['VERIFIED']}\n"
        "- Decision policy: PROJECT_DEFINED_EMPIRICAL_GATE.\n"
        f"- Gate alignment basis: {ALIGNMENT_BASIS}.\n"
        "- Compatibility is not retrieval-quality or Vietnamese-quality evidence.\n"
        "- Smoke testing is not Recall@K evaluation.\n"
        "- Stage 0 and Stage 1A were reused without rebuild.\n"
        + "\n## Blocking issues\n\n"
        + ("\n".join(blocker_lines) if blocker_lines else "- None")
        + _evaluated_candidates_report(
            finalized_candidates,
            provenances,
            runtime_manifests,
            candidate_summaries,
            selected,
        )
        + _official_candidate_report(
            report_candidate,
            report_provenance,
            report_runtime,
            report_metrics,
            smoke_status,
            selected=report_candidate == selected and selected is not None,
        )
        + f"\n\n## Next step\n\n{next_step}\n",
        encoding="utf-8",
    )
    for encoder in encoders.values():
        close = getattr(encoder, "close", None)
        if callable(close):
            close()
    _publish(staging, output, config.overwrite)
    return Stage1BResult(output, summary, selected_contract, False)
