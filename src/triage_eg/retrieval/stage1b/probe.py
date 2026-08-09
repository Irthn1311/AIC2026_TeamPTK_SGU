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

ALIGNMENT_BASIS = "EXACT_STORED_VECTOR_EQUIVALENCE_CLASS"
LITERAL_ALIGNMENT_BASIS = "LITERAL_GLOBAL_ROW_DIAGNOSTIC"


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


def _rank_summary(
    ranks: list[int | None],
    *,
    alignment_basis: str,
    diagnostic_only: bool = False,
) -> dict[str, Any]:
    completed = len(ranks)
    observed = [rank for rank in ranks if rank is not None]
    top1 = sum(rank == 1 for rank in ranks)
    top5 = sum(rank is not None and rank <= 5 for rank in ranks)
    top20 = sum(rank is not None and rank <= 20 for rank in ranks)
    return {
        "alignment_basis": alignment_basis,
        "diagnostic_only": diagnostic_only,
        "top1_count": top1,
        "top5_count": top5,
        "top20_count": top20,
        "top1_rate": top1 / completed if completed else 0.0,
        "top5_rate": top5 / completed if completed else 0.0,
        "top20_rate": top20 / completed if completed else 0.0,
        "mean_rank_within_returned_topk": float(np.mean(observed)) if observed else None,
        "max_rank_within_returned_topk": max(observed) if observed else None,
        "missing_from_returned_topk_count": completed - len(observed),
    }


def exact_stored_vector_alignment(
    search_rows: np.ndarray,
    target_rows: np.ndarray,
    returned_stored_vectors: np.ndarray,
    target_stored_vectors: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Compare literal rows and exact stored-vector classes within bounded Top-K results."""
    rows = np.asarray(search_rows, dtype=np.int64)
    targets = np.asarray(target_rows, dtype=np.int64)
    returned = np.asarray(returned_stored_vectors)
    stored = np.asarray(target_stored_vectors)
    if rows.ndim != 2 or targets.shape != (len(rows),):
        raise ValueError("Stage 1B alignment rows have invalid shapes")
    if returned.shape[:2] != rows.shape or returned.ndim != 3:
        raise ValueError("Stage 1B returned stored vectors have invalid shape")
    if stored.shape != (len(rows), returned.shape[2]):
        raise ValueError("Stage 1B target stored vectors have invalid shape")

    # Both operands come from backend.vectors_at. Its float16-to-float32 cast preserves
    # exact stored values, so equality here remains exact stored-index equality.
    equivalent_mask = np.all(returned == stored[:, None, :], axis=2)
    diagnostics: list[dict[str, Any]] = []
    literal_ranks: list[int | None] = []
    equivalent_ranks: list[int | None] = []
    for index, target_row in enumerate(targets):
        literal_positions = np.flatnonzero(rows[index] == target_row)
        equivalent_positions = np.flatnonzero(equivalent_mask[index])
        literal_rank = int(literal_positions[0] + 1) if len(literal_positions) else None
        equivalent_rank = (
            int(equivalent_positions[0] + 1) if len(equivalent_positions) else None
        )
        literal_ranks.append(literal_rank)
        equivalent_ranks.append(equivalent_rank)
        tie_displacement = literal_rank != 1 and equivalent_rank == 1
        diagnostics.append(
            {
                "alignment_basis": ALIGNMENT_BASIS,
                "exact_target_row_rank": literal_rank,
                "exact_target_row_in_top1": literal_rank == 1,
                "exact_target_row_in_top5": literal_rank is not None and literal_rank <= 5,
                "exact_target_row_in_top20": literal_rank is not None and literal_rank <= 20,
                "equivalent_vector_rank": equivalent_rank,
                "equivalent_vector_in_top1": equivalent_rank == 1,
                "equivalent_vector_in_top5": (
                    equivalent_rank is not None and equivalent_rank <= 5
                ),
                "equivalent_vector_in_top20": (
                    equivalent_rank is not None and equivalent_rank <= 20
                ),
                "equivalent_vector_count_in_returned_topk": int(
                    np.count_nonzero(equivalent_mask[index])
                ),
                "literal_target_row_tie_displacement": tie_displacement,
            }
        )
    return (
        diagnostics,
        _rank_summary(
            literal_ranks,
            alignment_basis=LITERAL_ALIGNMENT_BASIS,
            diagnostic_only=True,
        ),
        _rank_summary(equivalent_ranks, alignment_basis=ALIGNMENT_BASIS),
    )


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
    alignment = summary.get(
        "stored_vector_equivalence_alignment", summary.get("retrieval_alignment", {})
    )
    top1_rate = alignment.get("top1_rate", alignment.get("target_top1_rate", 0.0))
    top5_rate = alignment.get("top5_rate", alignment.get("target_top5_rate", 0.0))
    if completed and (
        cosine["mean"] < gate.pairwise_cosine_mean_min
        or cosine["min"] < gate.pairwise_cosine_min_min
    ):
        reasons.append("PAIRWISE_COSINE_BELOW_GATE")
    if completed and (
        top1_rate < gate.target_top1_rate_min or top5_rate < gate.target_top5_rate_min
    ):
        reasons.append("STORED_VECTOR_EQUIVALENCE_ALIGNMENT_BELOW_GATE")
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
    encoder = load_multimodal_encoder(candidate, adapter_factory, provenance)
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
    adapter_metrics = getattr(encoder, "last_image_metrics", {})
    raw_norms = np.asarray(
        adapter_metrics.get("raw_norms", np.linalg.norm(values, axis=1)),
        dtype=np.float32,
    )
    if adapter_metrics.get("normalized_output"):
        normalized = values
    else:
        value_norms = np.linalg.norm(values, axis=1)
        normalized = (
            values / value_norms[:, None] if candidate.image_embedding_normalization else values
        )
    normalized_norms = np.asarray(
        adapter_metrics.get("normalized_norms", np.linalg.norm(normalized, axis=1)),
        dtype=np.float32,
    )
    latencies = list(adapter_metrics.get("latency_seconds", [None] * len(samples)))
    backend, _ = load_search_backend(
        SearchConfig(stage1_root, f"stage1b_{candidate.candidate_id}", top_k=20)
    )
    target_rows = np.asarray([item["global_row"] for item in samples], dtype=np.int64)
    stored = backend.vectors_at(target_rows)
    stored_norms = np.linalg.norm(stored, axis=1)
    cosine = np.sum(normalized * stored, axis=1) / (
        np.linalg.norm(normalized, axis=1) * stored_norms
    )
    _, rows = backend.search(normalized, 20)
    returned_stored = backend.vectors_at(rows.reshape(-1)).reshape(
        rows.shape[0], rows.shape[1], stored.shape[1]
    )
    alignment_diagnostics, literal_alignment, equivalence_alignment = (
        exact_stored_vector_alignment(rows, target_rows, returned_stored, stored)
    )
    results = []
    for index, sample in enumerate(samples):
        alignment = alignment_diagnostics[index]
        diagnostic_codes = (
            ["LITERAL_TARGET_ROW_TIE_DISPLACEMENT"]
            if alignment["literal_target_row_tie_displacement"]
            else []
        )
        results.append(
            {
                "candidate_id": candidate.candidate_id,
                "global_row": sample["global_row"],
                "video_id": sample["video_id"],
                "n": sample["n"],
                "original_frame_idx": sample["original_frame_idx"],
                "encode_status": "SUCCESS",
                "candidate_dimension": int(values.shape[1]),
                "candidate_norm_before_normalization": float(raw_norms[index]),
                "candidate_norm_after_normalization": float(normalized_norms[index]),
                "stored_norm": float(stored_norms[index]),
                "cosine_to_stored": float(cosine[index]),
                "top1_global_row_when_searched": int(rows[index, 0]),
                **alignment,
                # Backward-compatible literal-row diagnostics. New consumers should
                # use the explicitly named exact_target_row_* fields above.
                "matched_row_rank": alignment["exact_target_row_rank"],
                "matched_row_in_top1": alignment["exact_target_row_in_top1"],
                "matched_row_in_top5": alignment["exact_target_row_in_top5"],
                "matched_row_in_top20": alignment["exact_target_row_in_top20"],
                "legacy_matched_row_fields_diagnostic_only": True,
                "encode_latency_seconds": latencies[index],
                "issue_codes": diagnostic_codes,
            }
        )
    completed = len(results)
    summary = {
        "candidate_id": candidate.candidate_id,
        "samples_requested": len(samples),
        "samples_completed": completed,
        "samples_failed": len(samples) - completed,
        "cosine": _stats(cosine),
        "literal_target_alignment": literal_alignment,
        "stored_vector_equivalence_alignment": equivalence_alignment,
        "gate_alignment_basis": ALIGNMENT_BASIS,
        "retrieval_alignment": {
            "alignment_basis": ALIGNMENT_BASIS,
            "target_top1_count": equivalence_alignment["top1_count"],
            "target_top5_count": equivalence_alignment["top5_count"],
            "target_top20_count": equivalence_alignment["top20_count"],
            "target_top1_rate": equivalence_alignment["top1_rate"],
            "target_top5_rate": equivalence_alignment["top5_rate"],
            "target_top20_rate": equivalence_alignment["top20_rate"],
            "mean_target_rank": equivalence_alignment["mean_rank_within_returned_topk"],
            "max_target_rank": equivalence_alignment["max_rank_within_returned_topk"],
        },
        "dimension_match": values.shape[1] == 512,
        "finite_all": bool(np.isfinite(values).all()),
    }
    decision, reasons = decide_candidate(
        candidate,
        summary,
        gate,
        bool(provenance.get("reproducible", provenance.get("checkpoint_fingerprint"))),
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
