"""Realistic GPU G1.1 benchmarks and consumer-specific promotion decisions."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from statistics import mean, median
from tempfile import TemporaryDirectory
from time import monotonic
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from triage_eg.experiments.mb1_v021.signals import (
    COARSE_SAMPLES_PER_SECOND,
    hard_cut_mask,
    scan_coarse_video,
)
from triage_eg.experiments.mb1_v022.signals import final_continuity_audit
from triage_eg.experiments.moment_m1 import (
    M1Settings,
    clipped_local_window,
    coarse_frame_indices,
    dense_frame_indices,
)
from triage_eg.retrieval.numpy_index import NumPyMemmapExactIndex

from .decoder import DecodedFrame, OpenCVRawVideoDecoder, create_raw_video_decoder

G11_BUNDLE_FILES = (
    "gpu_preflight.json",
    "mb1_decoder_benchmark.json",
    "m1_decoder_benchmark.json",
    "nvdec_mb1_parity.json",
    "nvdec_neural_parity.json",
    "clip_embedding_parity.json",
    "clip_retrieval_parity.json",
    "clip_batch_benchmark.json",
    "translator_gpu_sanity.json",
    "gpu_promotion_policy.json",
    "performance_summary.json",
    "run_manifest.json",
    "issues.jsonl",
)


def load_frozen_query_suite(
    stage1c_query_suite: str | Path,
    rt2_benchmark: str | Path,
    *,
    maximum: int = 100,
) -> list[dict[str, str]]:
    """Combine existing frozen texts without creating labels or changing text."""
    rows: list[dict[str, str]] = []
    seen: set[str] = set()

    def append(query_id: str, text: str, language: str, source: str) -> None:
        normalized = str(text).strip()
        if not normalized or normalized in seen or len(rows) >= maximum:
            return
        seen.add(normalized)
        rows.append(
            {
                "query_id": query_id,
                "text": normalized,
                "language": language,
                "source": source,
            }
        )

    with Path(stage1c_query_suite).open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            append(row["query_id"], row["text"], row["language"], "stage1c")
    with Path(rt2_benchmark).open(encoding="utf-8") as stream:
        for line in stream:
            query = json.loads(line)
            for event in query["events"]:
                append(
                    f"{query['query_id']}_{event['event_id']}",
                    event["text"],
                    query.get("language", "en"),
                    "rt2",
                )
    if len(rows) < 32:
        raise ValueError(f"G11_QUERY_SUITE_TOO_SMALL: {len(rows)}")
    return rows


def embedding_parity(cpu: np.ndarray, gpu: np.ndarray) -> dict[str, Any]:
    cpu32, gpu32 = np.asarray(cpu, dtype=np.float32), np.asarray(gpu, dtype=np.float32)
    if cpu32.shape != gpu32.shape or cpu32.ndim != 2:
        return {"status": "DROP", "reason": "SHAPE_MISMATCH"}
    cpu_norms, gpu_norms = np.linalg.norm(cpu32, axis=1), np.linalg.norm(gpu32, axis=1)
    finite = bool(np.isfinite(cpu32).all() and np.isfinite(gpu32).all())
    normalized = bool(
        np.allclose(cpu_norms, 1.0, rtol=0.0, atol=1e-4)
        and np.allclose(gpu_norms, 1.0, rtol=0.0, atol=1e-4)
    )
    cosine = np.sum(cpu32 * gpu32, axis=1) / np.maximum(cpu_norms * gpu_norms, 1e-12)
    result = {
        "query_count": int(cpu32.shape[0]),
        "shape": list(cpu32.shape),
        "all_finite": finite,
        "normalized_output_contract": normalized,
        "cosine_min": float(np.min(cosine)),
        "cosine_mean": float(np.mean(cosine)),
        "cosine_median": float(np.median(cosine)),
        "max_abs_difference": float(np.max(np.abs(cpu32 - gpu32))),
    }
    strong = finite and normalized and result["cosine_min"] >= 0.999
    result["status"] = "PASS" if strong else "CONDITIONAL" if finite and normalized else "DROP"
    return result


def in_memory_path_parity(adapter: Any, images: Sequence[np.ndarray]) -> dict[str, Any]:
    """Audit the new RGB-array path against official path preprocessing, then clean up."""
    selected = list(images[: min(8, len(images))])
    if not selected:
        return {"status": "DROP", "reason": "NO_IMAGES"}
    in_memory = adapter.encode_rgb_arrays(selected)
    with TemporaryDirectory(prefix="triage_eg_g11_path_parity_") as temporary:
        paths = []
        for index, image in enumerate(selected):
            path = Path(temporary) / f"frame_{index:03d}.png"
            with adapter._image.fromarray(np.asarray(image), mode="RGB") as value:
                value.save(path, format="PNG")
            paths.append(path)
        path_based = adapter.encode_images(paths)
    result = embedding_parity(path_based, in_memory)
    result["temporary_path_images_cleaned"] = True
    result["operational_path_remains_in_memory"] = True
    return result


def _exact_rankings(
    queries: np.ndarray,
    index_vectors: np.ndarray,
    *,
    index_norms: np.ndarray | None = None,
    top_k: int = 100,
    chunk_rows: int = 16_384,
) -> list[list[int]]:
    vectors = np.asarray(index_vectors)
    norms = (
        np.asarray(index_norms)
        if index_norms is not None
        else np.linalg.norm(np.asarray(vectors, dtype=np.float32), axis=1)
    )
    backend = NumPyMemmapExactIndex(
        vectors,
        norms,
        metric="cosine",
        chunk_rows=chunk_rows,
    )
    _, rows = backend.search(np.asarray(queries, dtype=np.float32), top_k)
    return [[int(value) for value in row] for row in rows]


def retrieval_agreement(
    cpu_queries: np.ndarray,
    gpu_queries: np.ndarray,
    index_vectors: np.ndarray,
    *,
    index_norms: np.ndarray | None = None,
    top_k: int = 100,
) -> dict[str, Any]:
    """Competition-aligned set/rank agreement; exact Top50 order is diagnostic only."""
    cpu = _exact_rankings(
        cpu_queries,
        index_vectors,
        index_norms=index_norms,
        top_k=top_k,
    )
    gpu = _exact_rankings(
        gpu_queries,
        index_vectors,
        index_norms=index_norms,
        top_k=top_k,
    )
    cutoffs = (1, 5, 20, 50, 100)
    per_query = []
    displacements: list[int] = []
    top1_diagnostics = []
    vectors = np.asarray(index_vectors)
    norms = (
        np.asarray(index_norms)
        if index_norms is not None
        else np.linalg.norm(np.asarray(vectors, dtype=np.float32), axis=1)
    )
    for query_index, (cpu_row, gpu_row) in enumerate(zip(cpu, gpu, strict=True)):
        gpu_positions = {row: index for index, row in enumerate(gpu_row)}
        common = set(cpu_row) & set(gpu_row)
        row_displacements = [
            abs(index - gpu_positions[row])
            for index, row in enumerate(cpu_row)
            if row in common
        ]
        displacements.extend(row_displacements)
        per_query.append(
            {
                "top1_exact": cpu_row[0] == gpu_row[0],
                "ordered_top50_exact": cpu_row[:50] == gpu_row[:50],
                "overlap": {
                    str(k): len(set(cpu_row[:k]) & set(gpu_row[:k]))
                    / min(k, len(cpu_row))
                    for k in cutoffs
                },
                "common_candidate_count": len(common),
                "mean_rank_displacement": mean(row_displacements)
                if row_displacements
                else None,
                "max_rank_displacement": max(row_displacements)
                if row_displacements
                else None,
            }
        )
        if cpu_row[0] != gpu_row[0]:
            cpu_candidates = cpu_row[:2]
            gpu_candidates = gpu_row[:2]

            def score(query: np.ndarray, row: int) -> float:
                vector = np.asarray(vectors[row], dtype=np.float32)
                denominator = float(norms[row]) * float(np.linalg.norm(query))
                return float(np.dot(vector, query) / max(denominator, 1e-12))

            cpu_scores = [score(cpu_queries[query_index], row) for row in cpu_candidates]
            gpu_scores = [score(gpu_queries[query_index], row) for row in gpu_candidates]
            cpu_margin = cpu_scores[0] - cpu_scores[1]
            gpu_margin = gpu_scores[0] - gpu_scores[1]
            top1_diagnostics.append(
                {
                    "query_index": query_index,
                    "cpu_top1_row": cpu_row[0],
                    "gpu_top1_row": gpu_row[0],
                    "cpu_top1_margin": cpu_margin,
                    "gpu_top1_margin": gpu_margin,
                    "near_tie_numerical_effect": max(cpu_margin, gpu_margin) <= 1e-4,
                }
            )
    overlaps = {
        str(k): {
            "mean": mean(row["overlap"][str(k)] for row in per_query),
            "median": median(row["overlap"][str(k)] for row in per_query),
            "minimum": min(row["overlap"][str(k)] for row in per_query),
        }
        for k in cutoffs
    }
    top1_changes = sum(not row["top1_exact"] for row in per_query)
    result = {
        "query_count": len(per_query),
        "top1_exact_matches": len(per_query) - top1_changes,
        "top1_changes": top1_changes,
        "top1_change_diagnostics": top1_diagnostics,
        "top1_change_rate": top1_changes / max(1, len(per_query)),
        "overlap": overlaps,
        "mean_rank_displacement": mean(displacements) if displacements else None,
        "median_rank_displacement": median(displacements) if displacements else None,
        "maximum_rank_displacement": max(displacements) if displacements else None,
        "exact_top50_order_matches": sum(
            row["ordered_top50_exact"] for row in per_query
        ),
        "exact_top50_order_is_hard_gate": False,
        "per_query": per_query,
        "ranking_backend": "CANONICAL_EXACT_NUMPY_STABLE",
    }
    high_set_agreement = (
        overlaps.get("20", {}).get("mean", 0.0) >= 0.95
        and overlaps.get("50", {}).get("mean", 0.0) >= 0.97
    )
    rare_top1_changes = result["top1_change_rate"] <= 0.10 and all(
        row["near_tie_numerical_effect"] for row in top1_diagnostics
    )
    result["status"] = (
        "PASS"
        if high_set_agreement and rare_top1_changes
        else "CONDITIONAL"
        if overlaps.get("20", {}).get("mean", 0.0) >= 0.90
        else "DROP"
    )
    return result


def _coarse_parity(cpu: Any, gpu: Any) -> dict[str, Any]:
    identities = bool(np.array_equal(cpu.frame_indices, gpu.frame_indices))
    differences = {
        "pixel": float(np.max(np.abs(cpu.pixel_differences - gpu.pixel_differences))),
        "histogram": float(
            np.max(np.abs(cpu.histogram_differences - gpu.histogram_differences))
        ),
        "spatial": float(
            np.max(np.abs(cpu.spatial_concentrations - gpu.spatial_concentrations))
        ),
    }
    cpu_cuts = hard_cut_mask(cpu.pixel_percentiles, cpu.histogram_percentiles)
    gpu_cuts = hard_cut_mask(gpu.pixel_percentiles, gpu.histogram_percentiles)
    return {
        "frame_identity": identities,
        "sample_count": len(cpu.frame_indices),
        "max_abs_signal_difference": differences,
        "hard_cut_decisions_equal": bool(np.array_equal(cpu_cuts, gpu_cuts)),
        "hard_cut_count_cpu": int(cpu_cuts.sum()),
        "hard_cut_count_nvdec": int(gpu_cuts.sum()),
    }


def benchmark_mb1_workload(
    video_paths: Sequence[Path], *, nvdec_available: bool
) -> tuple[dict[str, Any], dict[str, Any]]:
    records, parity_rows = [], []
    for path in video_paths:
        cpu = OpenCVRawVideoDecoder(path.stem, path)
        started = monotonic()
        cpu_series = scan_coarse_video(cpu)
        cpu_ms = (monotonic() - started) * 1000
        records.append(
            {
                "video_id": path.stem,
                "backend": "opencv",
                "wall_ms": cpu_ms,
                "sample_count": len(cpu_series.frame_indices),
                "samples_per_second": COARSE_SAMPLES_PER_SECOND,
                "stride_frames": cpu_series.stride_frames,
                "decoder": cpu.runtime_manifest(),
            }
        )
        info = cpu.info
        midpoint = info.total_frames // 2
        radius = min(int(round(info.fps * 3)), midpoint, info.total_frames - 1 - midpoint)
        audit_start, audit_end = midpoint - radius, midpoint + radius
        cpu_continuity = final_continuity_audit(cpu, audit_start, audit_end)
        cpu.close()
        if not nvdec_available:
            continue
        gpu = create_raw_video_decoder(path.stem, path, backend="nvdec")
        started = monotonic()
        gpu_series = scan_coarse_video(gpu)
        gpu_ms = (monotonic() - started) * 1000
        records.append(
            {
                "video_id": path.stem,
                "backend": "nvdec",
                "wall_ms": gpu_ms,
                "sample_count": len(gpu_series.frame_indices),
                "samples_per_second": COARSE_SAMPLES_PER_SECOND,
                "stride_frames": gpu_series.stride_frames,
                "decoder": gpu.runtime_manifest(),
            }
        )
        gpu_continuity = final_continuity_audit(gpu, audit_start, audit_end)
        parity = _coarse_parity(cpu_series, gpu_series)
        parity.update(
            {
                "video_id": path.stem,
                "speedup": cpu_ms / gpu_ms if gpu_ms > 0 else None,
                "orb_decisions_equal": (
                    cpu_continuity.abrupt_frames == gpu_continuity.abrupt_frames
                    and cpu_continuity.soft_frames == gpu_continuity.soft_frames
                ),
                "continuity_quality_cpu": cpu_continuity.continuity_quality,
                "continuity_quality_nvdec": gpu_continuity.continuity_quality,
                "orb_continuity_mean_cpu": cpu_continuity.orb_continuity_mean,
                "orb_continuity_mean_nvdec": gpu_continuity.orb_continuity_mean,
            }
        )
        parity_rows.append(parity)
        gpu.close()
    promoted = bool(parity_rows) and all(
        row["frame_identity"]
        and row["hard_cut_decisions_equal"]
        and row["orb_decisions_equal"]
        and row["speedup"] >= 1.5
        for row in parity_rows
    )
    return {
        "workload": "FROZEN_MB1_COARSE_SCAN",
        "samples_per_second": COARSE_SAMPLES_PER_SECOND,
        "records": records,
    }, {
        "status": "KEEP" if promoted else "NOT_PROMOTED" if nvdec_available else "UNAVAILABLE",
        "threshold_tuning_performed": False,
        "rows": parity_rows,
    }


def _decode_timed(decoder: Any, indices: list[int]) -> tuple[list[DecodedFrame], float]:
    started = monotonic()
    frames = decoder.decode_indices(indices)
    return frames, (monotonic() - started) * 1000


def benchmark_m1_local_workload(
    video_paths: Sequence[Path], *, nvdec_available: bool
) -> tuple[dict[str, Any], dict[str, Any], list[np.ndarray], list[np.ndarray]]:
    settings = M1Settings()
    records, parity_rows = [], []
    cpu_images: list[np.ndarray] = []
    gpu_images: list[np.ndarray] = []
    for path in video_paths:
        probe = OpenCVRawVideoDecoder(path.stem, path)
        anchor = probe.info.total_frames // 2
        start, end = clipped_local_window(
            anchor,
            fps=probe.info.fps,
            total_frames=probe.info.total_frames,
            seconds=settings.local_window_seconds,
        )
        coarse = coarse_frame_indices(start, end, anchor, settings.coarse_stride_frames)
        peak = anchor
        dense = dense_frame_indices(start, end, peak, settings.dense_radius_frames)
        cpu_coarse, cpu_coarse_ms = _decode_timed(probe, coarse)
        cpu_dense, cpu_dense_ms = _decode_timed(probe, dense)
        records.append(
            {
                "video_id": path.stem,
                "backend": "opencv",
                "anchor": anchor,
                "window_start": start,
                "window_end": end,
                "window_seconds_each_side": settings.local_window_seconds,
                "coarse_stride_frames": settings.coarse_stride_frames,
                "dense_radius_frames": settings.dense_radius_frames,
                "coarse_count": len(coarse),
                "dense_count": len(dense),
                "coarse_decode_ms": cpu_coarse_ms,
                "dense_decode_ms": cpu_dense_ms,
                "combined_decode_ms": cpu_coarse_ms + cpu_dense_ms,
            }
        )
        probe.close()
        if not nvdec_available:
            cpu_images.extend(frame.image for frame in cpu_dense[:8])
            continue
        gpu = create_raw_video_decoder(path.stem, path, backend="nvdec")
        gpu_coarse, gpu_coarse_ms = _decode_timed(gpu, coarse)
        gpu_dense, gpu_dense_ms = _decode_timed(gpu, dense)
        gpu.close()
        records.append(
            {
                "video_id": path.stem,
                "backend": "nvdec",
                "anchor": anchor,
                "window_start": start,
                "window_end": end,
                "coarse_stride_frames": settings.coarse_stride_frames,
                "dense_radius_frames": settings.dense_radius_frames,
                "coarse_count": len(coarse),
                "dense_count": len(dense),
                "coarse_decode_ms": gpu_coarse_ms,
                "dense_decode_ms": gpu_dense_ms,
                "combined_decode_ms": gpu_coarse_ms + gpu_dense_ms,
            }
        )
        cpu_all, gpu_all = cpu_coarse + cpu_dense, gpu_coarse + gpu_dense
        rgb_differences = [
            np.abs(left.image.astype(np.int16) - right.image.astype(np.int16))
            for left, right in zip(cpu_all, gpu_all, strict=True)
        ]
        parity_rows.append(
            {
                "video_id": path.stem,
                "frame_identity": [x.actual_frame_idx for x in cpu_all]
                == [x.actual_frame_idx for x in gpu_all],
                "max_abs_rgb_difference": max(int(x.max()) for x in rgb_differences),
                "mean_abs_rgb_difference": mean(float(x.mean()) for x in rgb_differences),
                "coarse_speedup": cpu_coarse_ms / gpu_coarse_ms,
                "dense_speedup": cpu_dense_ms / gpu_dense_ms,
                "combined_speedup": (cpu_coarse_ms + cpu_dense_ms)
                / (gpu_coarse_ms + gpu_dense_ms),
                "all_requested_frames_local": min(coarse + dense) >= start
                and max(coarse + dense) <= end,
            }
        )
        cpu_images.extend(frame.image for frame in cpu_dense[:8])
        gpu_images.extend(frame.image for frame in gpu_dense[:8])
    return {
        "workload": "FROZEN_M1_LOCAL_PLUS_MINUS_6S_COARSE12_DENSE15",
        "records": records,
    }, {"rows": parity_rows}, cpu_images, gpu_images


def benchmark_clip_batches(
    cpu_adapter: Any,
    gpu_adapter: Any,
    images: Sequence[np.ndarray],
    *,
    batch_sizes: Sequence[int] = (1, 8, 16, 32, 64),
) -> dict[str, Any]:
    if not images:
        return {
            "warmup_performed": False,
            "rows": [],
            "issues": [{"severity": "WARNING", "code": "NO_IMAGES_FOR_CLIP_BATCH_SWEEP"}],
        }
    rows, issues = [], []
    original_cpu_contract = cpu_adapter.contract
    original_gpu_contract = gpu_adapter.contract
    try:
        for batch_size in batch_sizes:
            values = [images[index % len(images)] for index in range(batch_size)]
            devices = []
            for device, adapter in (("cpu", cpu_adapter), ("cuda:0", gpu_adapter)):
                adapter.contract = replace(adapter.contract, batch_size=batch_size)
                try:
                    adapter.encode_rgb_arrays(values[: min(len(values), batch_size)])
                    if device.startswith("cuda"):
                        adapter._torch.cuda.synchronize()
                        adapter._torch.cuda.reset_peak_memory_stats()
                    started = monotonic()
                    adapter.encode_rgb_arrays(values)
                    if device.startswith("cuda"):
                        adapter._torch.cuda.synchronize()
                    elapsed = monotonic() - started
                    devices.append(
                        {
                            "device": device,
                            "batch_size": batch_size,
                            "latency_ms": elapsed * 1000,
                            "images_per_second": batch_size / elapsed,
                            "peak_gpu_memory_bytes": (
                                int(adapter._torch.cuda.max_memory_allocated())
                                if device.startswith("cuda")
                                else None
                            ),
                        }
                    )
                except RuntimeError as error:
                    if "out of memory" not in str(error).lower():
                        raise
                    issues.append(
                        {
                            "severity": "WARNING",
                            "code": "CLIP_BATCH_OOM",
                            "batch_size": batch_size,
                        }
                    )
                    if device.startswith("cuda"):
                        adapter._torch.cuda.empty_cache()
            row = {"batch_size": batch_size, "devices": devices}
            by_device = {item["device"]: item for item in devices}
            if {"cpu", "cuda:0"} <= set(by_device):
                row["gpu_speedup"] = (
                    by_device["cuda:0"]["images_per_second"]
                    / by_device["cpu"]["images_per_second"]
                )
            rows.append(row)
    finally:
        cpu_adapter.contract = original_cpu_contract
        gpu_adapter.contract = original_gpu_contract
    return {"warmup_performed": True, "rows": rows, "issues": issues}


def build_promotion_policy(
    *,
    clip_status: str,
    translator_status: str,
    nvdec_mb1_status: str,
    nvdec_neural_status: str,
    build_commit: str,
    evidence_run_id: str,
) -> dict[str, Any]:
    clip_promoted = clip_status == "KEEP"
    translator_promoted = translator_status == "KEEP"
    mb1_promoted = nvdec_mb1_status == "KEEP"
    neural_promoted = nvdec_neural_status == "KEEP"
    return {
        "policy_version": "GPU_G1.1_PROPOSED",
        "translator_gpu_promoted": translator_promoted,
        "clip_gpu_promoted": clip_promoted,
        "nvdec_mb1_promoted": mb1_promoted,
        "nvdec_neural_promoted": neural_promoted,
        "default_mb1_video_backend": "nvdec" if mb1_promoted else "opencv",
        "default_neural_video_backend": "nvdec" if neural_promoted else "opencv",
        "default_clip_device": "cuda:0" if clip_promoted else "cpu",
        "default_translator_device": "cuda:0" if translator_promoted else "cpu",
        "stage1_backend": "exact_numpy_cpu",
        "cpu_fallback_ready": True,
        "evidence_run_id": evidence_run_id,
        "build_commit": build_commit,
        "production_auto_mutated": False,
        "requires_user_approval_to_freeze": True,
    }


def consumer_specific_nvdec_verdicts(
    mb1_parity: dict[str, Any],
    neural_embedding: dict[str, Any],
    neural_retrieval: dict[str, Any],
    *,
    nvdec_available: bool,
) -> dict[str, str]:
    if not nvdec_available:
        return {"NVDEC_MB1": "UNAVAILABLE", "NVDEC_NEURAL": "UNAVAILABLE"}
    mb1_status = str(mb1_parity.get("status", "NOT_PROMOTED"))
    neural_rows = neural_embedding.get("rows", [])
    identity = bool(neural_rows) and all(row.get("frame_identity") for row in neural_rows)
    embedding_status = neural_embedding.get("status")
    retrieval_status = neural_retrieval.get("status")
    useful_speed = bool(neural_rows) and median(
        float(row.get("combined_speedup", 0.0)) for row in neural_rows
    ) >= 1.5
    if identity and embedding_status == "PASS" and retrieval_status == "PASS":
        neural_status = "KEEP" if useful_speed else "OPTIONAL"
    elif identity and embedding_status in {"PASS", "CONDITIONAL"}:
        neural_status = "OPTIONAL"
    else:
        neural_status = "DROP"
    return {
        "NVDEC_MB1": "KEEP" if mb1_status == "KEEP" else "NOT_PROMOTED",
        "NVDEC_NEURAL": neural_status,
    }


def clip_gpu_verdict(
    embedding: dict[str, Any], retrieval: dict[str, Any]
) -> str:
    if embedding.get("status") == "DROP" or retrieval.get("status") == "DROP":
        return "DROP"
    if embedding.get("status") == "PASS" and retrieval.get("status") == "PASS":
        return "KEEP"
    return "CONDITIONAL"


def write_g11_bundle(output_root: Path, artifacts: dict[str, Any]) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    missing = set(G11_BUNDLE_FILES) - set(artifacts)
    extra = set(artifacts) - set(G11_BUNDLE_FILES)
    if missing or extra:
        raise ValueError(
            f"G11_BUNDLE_CONTRACT_INVALID missing={sorted(missing)} "
            f"extra={sorted(extra)}"
        )
    for name in G11_BUNDLE_FILES:
        path = output_root / name
        value = artifacts[name]
        if name.endswith(".jsonl"):
            path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in value),
                encoding="utf-8",
            )
        else:
            path.write_text(
                json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
    readme = output_root / "README.md"
    readme.write_text(
        "# TRIAGE-EG GPU G1.1\n\nCompact realistic promotion evidence only. "
        "No runtime cache, model, vector dump, image, or raw video is included.\n",
        encoding="utf-8",
    )
    zip_path = output_root.parent / "triage_eg_gpu_g11_bundle.zip"
    with ZipFile(zip_path, "w", ZIP_DEFLATED) as archive:
        for name in (*G11_BUNDLE_FILES, "README.md"):
            archive.write(output_root / name, name)
    return zip_path


__all__ = [
    "G11_BUNDLE_FILES",
    "benchmark_clip_batches",
    "benchmark_m1_local_workload",
    "benchmark_mb1_workload",
    "build_promotion_policy",
    "consumer_specific_nvdec_verdicts",
    "clip_gpu_verdict",
    "embedding_parity",
    "in_memory_path_parity",
    "load_frozen_query_suite",
    "retrieval_agreement",
    "write_g11_bundle",
]
