"""Bounded GPU G1 parity, benchmark, decision, and bundle helpers."""

from __future__ import annotations

import json
from pathlib import Path
from time import monotonic
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from triage_eg.experiments.mb1_v021.signals import (
    _small_frame_features,
    spatial_activity_concentration,
)

from .decoder import OpenCVRawVideoDecoder, create_raw_video_decoder, sampled_frame_indices


def representative_videos(root: str | Path, *, limit: int = 4) -> list[Path]:
    """Select a deterministic bounded spread by file size from real raw videos."""
    paths = sorted(
        path for path in Path(root).rglob("*") if path.suffix.lower() in {".mp4", ".avi", ".mkv"}
    )
    if not paths:
        raise FileNotFoundError(f"NO_RAW_VIDEOS_FOUND: {root}")
    ranked = sorted(paths, key=lambda path: (path.stat().st_size, path.as_posix()))
    count = min(max(1, limit), len(ranked))
    positions = np.linspace(0, len(ranked) - 1, count, dtype=int)
    return [ranked[int(index)] for index in positions]


def audit_indices(total_frames: int) -> list[int]:
    if total_frames <= 0:
        raise ValueError("video contains no frames")
    final = total_frames - 1
    middle = final // 2
    candidates = [0, 1, 2, middle - 1, middle, middle + 1, final - 2, final - 1, final]
    return sorted({max(0, min(final, value)) for value in candidates})


def compare_mb1_signals(cpu_frames: list[Any], gpu_frames: list[Any]) -> dict[str, Any]:
    """Compare unchanged CPU MB1 signals after two decoder paths."""

    def values(frames: list[Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pixel = np.zeros(len(frames), dtype=np.float64)
        histogram = np.zeros(len(frames), dtype=np.float64)
        concentration = np.zeros(len(frames), dtype=np.float64)
        previous_gray = previous_histogram = None
        for index, frame in enumerate(frames):
            gray, current_histogram = _small_frame_features(frame.image)
            if previous_gray is not None and previous_histogram is not None:
                pixel[index] = float(
                    np.mean(np.abs(gray.astype(np.float32) - previous_gray.astype(np.float32)))
                    / 255.0
                )
                histogram[index] = float(0.5 * np.abs(current_histogram - previous_histogram).sum())
                concentration[index] = spatial_activity_concentration(gray, previous_gray)
            previous_gray, previous_histogram = gray, current_histogram
        return pixel, histogram, concentration

    cpu_values, gpu_values = values(cpu_frames), values(gpu_frames)
    names = ("pixel_difference", "histogram_difference", "spatial_concentration")
    differences = {
        name: float(np.max(np.abs(left - right))) if left.size else 0.0
        for name, left, right in zip(names, cpu_values, gpu_values, strict=True)
    }
    exact = all(value == 0.0 for value in differences.values())
    equivalent = (
        differences["pixel_difference"] <= 2.0 / 255.0
        and differences["histogram_difference"] <= 0.01
        and differences["spatial_concentration"] <= 0.01
    )
    return {
        "status": "PASS" if equivalent else "FAIL",
        "exact": exact,
        "numerically_equivalent": equivalent,
        "max_abs_differences": differences,
        "tolerances": {
            "pixel_difference": 2.0 / 255.0,
            "histogram_difference": 0.01,
            "spatial_concentration": 0.01,
        },
        "signal_math_device": "cpu",
    }


def benchmark_decoder_paths(
    video_paths: list[Path], *, nvdec_available: bool, coarse_samples_per_second: float = 1.0
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compare legacy short-range seeks, sequential OpenCV, and optional NVDEC."""
    records: list[dict[str, Any]] = []
    parity: list[dict[str, Any]] = []
    for path in video_paths:
        video_id = path.stem
        probe = OpenCVRawVideoDecoder(video_id, path)
        info = probe.info
        probe.close()
        stride = max(1, int(round(info.fps / coarse_samples_per_second)))
        indices = sampled_frame_indices(info.total_frames, stride=stride)

        legacy = OpenCVRawVideoDecoder(video_id, path)
        started = monotonic()
        retained = 0
        for offset in range(0, len(indices), 4):
            retained += len(legacy.decode_indices(indices[offset : offset + 4]))
        wall_ms = (monotonic() - started) * 1000
        records.append(
            {
                "video_id": video_id,
                "workload": "coarse_legacy_seek",
                "backend": "opencv",
                "wall_ms": wall_ms,
                "retained": retained,
                **legacy.runtime_manifest(),
            }
        )
        legacy.close()

        sequential = OpenCVRawVideoDecoder(video_id, path)
        started = monotonic()
        identities = [
            frame.actual_frame_idx for frame in sequential.iter_sampled_frames(stride=stride)
        ]
        wall_ms = (monotonic() - started) * 1000
        records.append(
            {
                "video_id": video_id,
                "workload": "coarse_sequential",
                "backend": "opencv",
                "wall_ms": wall_ms,
                "retained": len(identities),
                **sequential.runtime_manifest(),
            }
        )
        if identities != indices:
            raise RuntimeError("OPENCV_SEQUENTIAL_IDENTITY_MISMATCH")
        sequential.close()

        wanted = audit_indices(info.total_frames)
        cpu = OpenCVRawVideoDecoder(video_id, path)
        cpu_started = monotonic()
        cpu_frames = cpu.decode_indices(wanted)
        cpu_wall_ms = (monotonic() - cpu_started) * 1000
        records.append(
            {
                "video_id": video_id,
                "workload": "local_indexed",
                "backend": "opencv",
                "wall_ms": cpu_wall_ms,
                **cpu.runtime_manifest(),
            }
        )
        cpu.close()
        if nvdec_available:
            gpu = create_raw_video_decoder(video_id, path, backend="nvdec")
            gpu_coarse_started = monotonic()
            gpu_coarse_ids = [
                frame.actual_frame_idx for frame in gpu.iter_sampled_frames(stride=stride)
            ]
            gpu_coarse_wall_ms = (monotonic() - gpu_coarse_started) * 1000
            records.append(
                {
                    "video_id": video_id,
                    "workload": "coarse_sequential",
                    "backend": "nvdec",
                    "wall_ms": gpu_coarse_wall_ms,
                    "retained": len(gpu_coarse_ids),
                    **gpu.runtime_manifest(),
                }
            )
            if gpu_coarse_ids != indices:
                parity.append(
                    {
                        "video_id": video_id,
                        "status": "FAIL",
                        "workload": "coarse_sequential",
                        "reason": "FRAME_IDENTITY_MISMATCH",
                    }
                )
            gpu.close()

            gpu = create_raw_video_decoder(video_id, path, backend="nvdec")
            gpu_started = monotonic()
            gpu_frames = gpu.decode_indices(wanted)
            gpu_wall_ms = (monotonic() - gpu_started) * 1000
            records.append(
                {
                    "video_id": video_id,
                    "workload": "local_indexed",
                    "backend": "nvdec",
                    "wall_ms": gpu_wall_ms,
                    **gpu.runtime_manifest(),
                }
            )
            comparisons = []
            for left, right in zip(cpu_frames, gpu_frames, strict=True):
                difference = np.abs(left.image.astype(np.int16) - right.image.astype(np.int16))
                comparisons.append(
                    {
                        "frame_idx": left.actual_frame_idx,
                        "identity_match": left.actual_frame_idx == right.actual_frame_idx,
                        "max_abs_pixel_difference": int(difference.max()),
                        "mean_abs_pixel_difference": float(difference.mean()),
                    }
                )
            identity_pass = all(item["identity_match"] for item in comparisons)
            parity.append(
                {
                    "video_id": video_id,
                    "status": "PASS" if identity_pass else "FAIL",
                    "frames": comparisons,
                    "signals": compare_mb1_signals(cpu_frames, gpu_frames),
                }
            )
            gpu.close()
    return {"records": records}, {
        "records": parity,
        "identity_status": "PASS"
        if parity and all(item["status"] == "PASS" for item in parity)
        else ("UNAVAILABLE" if not parity else "FAIL"),
    }


def vector_parity(cpu: np.ndarray, gpu: np.ndarray, *, top_k: int = 10) -> dict[str, Any]:
    if cpu.shape != gpu.shape:
        return {"status": "FAIL", "reason": "SHAPE_MISMATCH"}
    cpu32, gpu32 = np.asarray(cpu, dtype=np.float32), np.asarray(gpu, dtype=np.float32)
    cosine = np.sum(cpu32 * gpu32, axis=1) / np.maximum(
        np.linalg.norm(cpu32, axis=1) * np.linalg.norm(gpu32, axis=1), 1e-12
    )
    cpu_ranking = np.argsort(-cpu32 @ cpu32.T, axis=1, kind="stable")[:, :top_k]
    gpu_ranking = np.argsort(-gpu32 @ gpu32.T, axis=1, kind="stable")[:, :top_k]
    ranking_equal = bool(np.array_equal(cpu_ranking, gpu_ranking))
    return {
        "status": "PASS" if ranking_equal and np.all(cosine >= 0.999) else "FAIL",
        "shape": list(cpu32.shape),
        "finite": bool(np.isfinite(cpu32).all() and np.isfinite(gpu32).all()),
        "cosine_min": float(cosine.min()) if len(cosine) else None,
        "cosine_mean": float(cosine.mean()) if len(cosine) else None,
        "max_abs_difference": float(np.max(np.abs(cpu32 - gpu32))) if cpu32.size else 0.0,
        "top_k_ranking_equal": ranking_equal,
    }


def retrieval_parity(
    cpu_queries: np.ndarray,
    gpu_queries: np.ndarray,
    index_vectors: np.ndarray,
    *,
    top_k: int = 50,
    chunk_rows: int = 16_384,
) -> dict[str, Any]:
    """Compare stable exact rankings without changing the canonical NumPy search path."""
    if cpu_queries.shape != gpu_queries.shape or cpu_queries.shape[1] != index_vectors.shape[1]:
        return {"status": "FAIL", "reason": "SHAPE_MISMATCH"}

    def rank(queries: np.ndarray) -> list[list[int]]:
        output: list[list[int]] = []
        for query in np.asarray(queries, dtype=np.float32):
            scores = np.empty(index_vectors.shape[0], dtype=np.float32)
            for start in range(0, index_vectors.shape[0], chunk_rows):
                stop = min(index_vectors.shape[0], start + chunk_rows)
                scores[start:stop] = np.asarray(index_vectors[start:stop], dtype=np.float32) @ query
            ordered = np.argsort(-scores, kind="stable")[:top_k]
            output.append([int(value) for value in ordered])
        return output

    cpu_rankings, gpu_rankings = rank(cpu_queries), rank(gpu_queries)
    exact = [left == right for left, right in zip(cpu_rankings, gpu_rankings, strict=True)]
    overlaps = [
        len(set(left) & set(right)) / max(1, len(left))
        for left, right in zip(cpu_rankings, gpu_rankings, strict=True)
    ]
    return {
        "status": "PASS" if all(exact) else "FAIL",
        "top_k": top_k,
        "query_count": len(exact),
        "exact_ranking_matches": sum(exact),
        "top_k_overlap_min": min(overlaps) if overlaps else None,
        "stable_tie_policy": "numpy_argsort_stable_global_row",
    }


def build_g1_decision(
    decoder_benchmark: dict[str, Any],
    decoder_parity: dict[str, Any],
    clip_parity: dict[str, Any],
    translator_parity: dict[str, Any],
    *,
    cuda_available: bool,
    nvdec_available: bool,
) -> dict[str, Any]:
    """Apply the frozen G1 acceptance policy to measured audit evidence."""
    records = decoder_benchmark.get("records", [])

    def median_ms(workload: str, backend: str) -> float | None:
        values = [
            float(item["wall_ms"])
            for item in records
            if item.get("workload") == workload
            and item.get("backend") == backend
            and float(item.get("wall_ms", 0)) > 0
        ]
        return float(np.median(values)) if values else None

    sequential_cpu = median_ms("coarse_sequential", "opencv")
    sequential_gpu = median_ms("coarse_sequential", "nvdec")
    local_cpu = median_ms("local_indexed", "opencv")
    local_gpu = median_ms("local_indexed", "nvdec")
    speedups = {
        "coarse_sequential_nvdec_vs_opencv": (
            sequential_cpu / sequential_gpu
            if sequential_cpu is not None and sequential_gpu is not None
            else None
        ),
        "local_indexed_nvdec_vs_opencv": (
            local_cpu / local_gpu if local_cpu is not None and local_gpu is not None else None
        ),
    }
    useful_nvdec = any(value is not None and value >= 1.5 for value in speedups.values())
    identity_pass = decoder_parity.get("identity_status") == "PASS"
    signal_pass = all(
        item.get("signals", {}).get("status") == "PASS"
        for item in decoder_parity.get("records", [])
        if "signals" in item
    )
    if not nvdec_available:
        nvdec_status = "UNAVAILABLE"
    elif identity_pass and signal_pass and useful_nvdec:
        nvdec_status = "KEEP"
    elif identity_pass and signal_pass:
        nvdec_status = "OPTIONAL"
    else:
        nvdec_status = "DROP"
    clip_status = (
        "UNAVAILABLE"
        if not cuda_available
        else ("KEEP" if clip_parity.get("status") == "PASS" else "DROP")
    )
    translator_status = (
        "UNAVAILABLE"
        if not cuda_available
        else ("KEEP" if translator_parity.get("status") == "PASS" else "DROP")
    )
    blocking = clip_status == "DROP" or translator_status == "DROP" or nvdec_status == "DROP"
    partial = not cuda_available or not nvdec_available
    return {
        "GPU_G1_STATUS": "FAIL" if blocking else ("PARTIAL" if partial else "PASS"),
        "CPU_FALLBACK_STATUS": "READY",
        "OPTIMIZED_OPENCV_STATUS": "KEEP",
        "NVDEC_STATUS": nvdec_status,
        "CLIP_GPU_STATUS": clip_status,
        "TRANSLATOR_GPU_STATUS": translator_status,
        "M1_IN_MEMORY_CLIP_STATUS": "KEEP",
        "DEFAULT_VIDEO_BACKEND": ("AUTO_NVDEC" if nvdec_status == "KEEP" else "AUTO_OPENCV"),
        "MAIN_RUNTIME_GPU_AWARE": "YES" if not blocking else "NO",
        "speedups": speedups,
        "nvdec_promotion_threshold": 1.5,
    }


def write_g1_bundle(output_root: Path, artifacts: dict[str, Any]) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    for name, value in artifacts.items():
        if name.endswith(".jsonl"):
            lines = "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in value)
            (output_root / name).write_text(lines, encoding="utf-8")
        else:
            (output_root / name).write_text(
                json.dumps(value, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
    readme = output_root / "README.md"
    if not readme.exists():
        readme.write_text(
            "# TRIAGE-EG GPU G1 audit\n\n"
            "Compact parity and performance evidence; no models or raw videos included.\n",
            encoding="utf-8",
        )
    zip_path = output_root.parent / "triage_eg_gpu_g1_bundle.zip"
    if zip_path.exists():
        zip_path.unlink()
    with ZipFile(zip_path, "w", ZIP_DEFLATED) as archive:
        for path in sorted(output_root.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(output_root).as_posix())
    return zip_path


__all__ = [
    "audit_indices",
    "benchmark_decoder_paths",
    "build_g1_decision",
    "compare_mb1_signals",
    "representative_videos",
    "retrieval_parity",
    "vector_parity",
    "write_g1_bundle",
]
