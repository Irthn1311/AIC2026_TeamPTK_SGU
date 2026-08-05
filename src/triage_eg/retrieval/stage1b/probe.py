"""Image-side empirical compatibility probe and project-defined gate."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from triage_eg.retrieval.stage1.contracts import SearchConfig
from triage_eg.retrieval.stage1.search import load_search_backend
from triage_eg.retrieval.stage1b.assets import AdapterFactory, load_multimodal_encoder
from triage_eg.retrieval.stage1b.contracts import CandidateContract, CompatibilityGate


def validate_embedding_matrix(values: Any, rows: int, dimension: int = 512) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.shape != (rows, dimension):
        raise ValueError("ENCODER_OUTPUT_DIMENSION_MISMATCH")
    if not np.isfinite(matrix).all():
        raise ValueError("ENCODER_OUTPUT_NON_FINITE")
    if np.any(np.linalg.norm(matrix, axis=1) == 0):
        raise ValueError("ENCODER_OUTPUT_ZERO_NORM")
    return matrix


def _stats(values: np.ndarray) -> dict[str, float | None]:
    if len(values) == 0:
        return {key: None for key in ("min", "max", "mean", "median", "p05", "p95", "std")}
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p05": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
        "std": float(np.std(values)),
    }


def decide_candidate(
    candidate: CandidateContract,
    summary: dict[str, Any],
    gate: CompatibilityGate,
    provenance_complete: bool,
) -> tuple[str, list[str]]:
    reasons = []
    completed = summary["samples_completed"]
    if completed < gate.minimum_completed_samples:
        reasons.append("INSUFFICIENT_COMPLETED_SAMPLES")
    if gate.require_dimension_512 and not summary["dimension_match"]:
        return "REJECTED", ["ENCODER_OUTPUT_DIMENSION_MISMATCH"]
    if gate.require_all_finite and not summary["finite_all"]:
        return "REJECTED", ["ENCODER_OUTPUT_NON_FINITE"]
    cosine = summary["cosine"]
    alignment = summary["retrieval_alignment"]
    if completed and (
        cosine["mean"] < gate.pairwise_cosine_mean_min
        or cosine["min"] < gate.pairwise_cosine_min_min
    ):
        reasons.append("PAIRWISE_COSINE_BELOW_GATE")
    if completed and (
        alignment["target_top1_rate"] < gate.target_top1_rate_min
        or alignment["target_top5_rate"] < gate.target_top5_rate_min
    ):
        reasons.append("TARGET_RETRIEVAL_ALIGNMENT_BELOW_GATE")
    if not provenance_complete or not candidate.reproducible():
        if provenance_complete:
            return "REJECTED", ["ENCODER_CONTRACT_NOT_REPRODUCIBLE"]
        reasons.append("ENCODER_PROVENANCE_INCOMPLETE")
    if not reasons:
        return "VERIFIED", ["PROJECT_DEFINED_EMPIRICAL_GATE"]
    if "INSUFFICIENT_COMPLETED_SAMPLES" in reasons or "ENCODER_PROVENANCE_INCOMPLETE" in reasons:
        return "UNVERIFIED", reasons
    return "REJECTED", reasons


def probe_candidate(
    candidate: CandidateContract,
    samples: list[dict],
    stage1_root: Path,
    gate: CompatibilityGate,
    provenance: dict[str, Any],
    adapter_factory: AdapterFactory | None = None,
) -> tuple[CandidateContract, list[dict], dict, np.ndarray | None, Any | None]:
    encoder = load_multimodal_encoder(candidate, adapter_factory)
    paths = [Path(item["keyframe_path"]) for item in samples]
    try:
        encoded = encoder.encode_images(paths)
    except (FileNotFoundError, OSError) as error:
        raise ValueError(f"IMAGE_LOAD_FAILED: {error}") from error
    except ValueError as error:
        if str(error).startswith(("IMAGE_", "ENCODER_")):
            raise
        raise ValueError(f"IMAGE_PREPROCESS_FAILED: {error}") from error
    values = validate_embedding_matrix(encoded, len(samples), 512)
    raw_norms = np.linalg.norm(values, axis=1)
    normalized = values / raw_norms[:, None] if candidate.image_embedding_normalization else values
    backend, _ = load_search_backend(
        SearchConfig(stage1_root, f"stage1b_{candidate.candidate_id}", top_k=20)
    )
    target_rows = np.asarray([item["global_row"] for item in samples], dtype=np.int64)
    stored = backend.vectors_at(target_rows)
    stored_norms = np.linalg.norm(stored, axis=1)
    cosine = np.sum(normalized * stored, axis=1) / (
        np.linalg.norm(normalized, axis=1) * stored_norms
    )
    scores, rows = backend.search(normalized, 20)
    results, target_ranks = [], []
    for index, sample in enumerate(samples):
        positions = np.flatnonzero(rows[index] == sample["global_row"])
        rank = int(positions[0] + 1) if len(positions) else None
        if rank is not None:
            target_ranks.append(rank)
        results.append(
            {
                "candidate_id": candidate.candidate_id,
                "global_row": sample["global_row"],
                "video_id": sample["video_id"],
                "n": sample["n"],
                "encode_status": "SUCCESS",
                "candidate_dimension": int(values.shape[1]),
                "candidate_norm_before_normalization": float(raw_norms[index]),
                "candidate_norm_after_normalization": float(np.linalg.norm(normalized[index])),
                "stored_norm": float(stored_norms[index]),
                "cosine_to_stored": float(cosine[index]),
                "top1_global_row_when_searched": int(rows[index, 0]),
                "matched_row_rank": rank,
                "matched_row_in_top1": rank == 1,
                "matched_row_in_top5": rank is not None and rank <= 5,
                "matched_row_in_top20": rank is not None,
                "issue_codes": [],
            }
        )
    completed = len(results)
    top1 = sum(item["matched_row_in_top1"] for item in results)
    top5 = sum(item["matched_row_in_top5"] for item in results)
    top20 = sum(item["matched_row_in_top20"] for item in results)
    summary = {
        "candidate_id": candidate.candidate_id,
        "samples_requested": len(samples),
        "samples_completed": completed,
        "samples_failed": len(samples) - completed,
        "cosine": _stats(cosine),
        "retrieval_alignment": {
            "target_top1_count": top1,
            "target_top5_count": top5,
            "target_top20_count": top20,
            "target_top1_rate": top1 / completed if completed else 0.0,
            "target_top5_rate": top5 / completed if completed else 0.0,
            "target_top20_rate": top20 / completed if completed else 0.0,
            "mean_target_rank": float(np.mean(target_ranks)) if target_ranks else None,
            "max_target_rank": max(target_ranks) if target_ranks else None,
        },
        "dimension_match": values.shape[1] == 512,
        "finite_all": bool(np.isfinite(values).all()),
    }
    decision, reasons = decide_candidate(
        candidate,
        summary,
        gate,
        bool(provenance.get("checkpoint_fingerprint")),
    )
    summary.update(decision=decision, decision_reasons=reasons)
    evidence_source = "EMPIRICAL_PROBE" if decision == "VERIFIED" else candidate.evidence_source
    return (
        replace(candidate, compatibility_status=decision, evidence_source=evidence_source),
        results,
        summary,
        normalized,
        encoder,
    )
