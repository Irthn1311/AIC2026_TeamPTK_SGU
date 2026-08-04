"""Calibrate BTC map-keyframe coordinates against decoded original-video frames."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True, slots=True)
class CalibrationMappingRow:
    keyframe_order: int
    actual_frame_id: int
    physical_row: int
    pts_time: float
    fps: float


def load_mapping_rows(path: Path) -> tuple[CalibrationMappingRow, ...]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"mapping CSV not found: {path}")
    rows: list[CalibrationMappingRow] = []
    seen_orders: set[int] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"n", "pts_time", "fps", "frame_idx"}
        missing = sorted(required - set(reader.fieldnames or ()))
        if missing:
            raise ValueError(f"mapping CSV missing columns: {', '.join(missing)}")
        for line_number, raw in enumerate(reader, start=2):
            if not any((value or "").strip() for value in raw.values()):
                continue
            try:
                order = int((raw.get("n") or "").strip())
                frame_idx = int((raw.get("frame_idx") or "").strip())
                pts_time = float((raw.get("pts_time") or "").strip())
                fps = float((raw.get("fps") or "").strip())
            except ValueError as exc:
                raise ValueError(f"invalid mapping value at line {line_number}") from exc
            if order < 0 or frame_idx < 0 or pts_time < 0 or fps <= 0:
                raise ValueError(f"invalid non-positive mapping value at line {line_number}")
            if not np.isfinite(pts_time) or not np.isfinite(fps):
                raise ValueError(f"non-finite mapping value at line {line_number}")
            if order in seen_orders:
                raise ValueError(f"duplicate keyframe order at line {line_number}: {order}")
            seen_orders.add(order)
            rows.append(
                CalibrationMappingRow(
                    keyframe_order=order,
                    actual_frame_id=frame_idx,
                    physical_row=len(rows),
                    pts_time=pts_time,
                    fps=fps,
                )
            )
    if not rows:
        raise ValueError("mapping CSV contains no records")
    return tuple(rows)


def select_mapping_rows(
    rows: Sequence[CalibrationMappingRow],
    *,
    sample_count: int,
    explicit_orders: Sequence[int] | None = None,
) -> tuple[CalibrationMappingRow, ...]:
    if explicit_orders:
        by_order = {row.keyframe_order: row for row in rows}
        missing = [order for order in explicit_orders if order not in by_order]
        if missing:
            raise ValueError(f"unknown keyframe orders: {missing}")
        if len(set(explicit_orders)) != len(explicit_orders):
            raise ValueError("explicit keyframe orders must be unique")
        return tuple(by_order[order] for order in explicit_orders)
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    count = min(sample_count, len(rows))
    if count == 1:
        indices = [0]
    else:
        indices = [round(index * (len(rows) - 1) / (count - 1)) for index in range(count)]
    return tuple(rows[index] for index in indices)


def compare_images(
    reference: NDArray[np.number], candidate: NDArray[np.number]
) -> dict[str, float]:
    if reference.shape != candidate.shape:
        raise ValueError(
            f"image shape mismatch after alignment: reference={reference.shape}, "
            f"candidate={candidate.shape}"
        )
    if reference.size == 0:
        raise ValueError("images must not be empty")
    reference_float = reference.astype(np.float64, copy=False)
    candidate_float = candidate.astype(np.float64, copy=False)
    difference = reference_float - candidate_float
    mae = float(np.mean(np.abs(difference)))
    rmse = float(np.sqrt(np.mean(np.square(difference))))
    dynamic_range = 255.0
    reference_flat = reference_float.reshape(-1)
    candidate_flat = candidate_float.reshape(-1)
    denominator = float(np.linalg.norm(reference_flat) * np.linalg.norm(candidate_flat))
    cosine = (
        float(np.dot(reference_flat, candidate_flat) / denominator)
        if denominator > 0
        else float(reference_flat.size > 0 and np.array_equal(reference_flat, candidate_flat))
    )
    return {
        "normalized_mae": mae / dynamic_range,
        "normalized_rmse": rmse / dynamic_range,
        "pixel_cosine": cosine,
        "exact_pixel_fraction": float(np.mean(reference_float == candidate_float)),
        "similarity": 1.0 - mae / dynamic_range,
    }


def evaluate_frame_offsets(
    rows: Sequence[CalibrationMappingRow],
    references: dict[int, NDArray[np.number]],
    decode_frame: Callable[[int], NDArray[np.number]],
    *,
    total_frames: int,
    offsets: Sequence[int] = (-1, 0, 1),
    align_candidate: Callable[[NDArray[np.number], NDArray[np.number]], NDArray[np.number]]
    | None = None,
    superiority_margin: float = 1e-4,
    consistency_ratio: float = 0.8,
) -> dict[str, Any]:
    if total_frames <= 0:
        raise ValueError("total_frames must be positive")
    if 0 not in offsets:
        raise ValueError("offset candidates must include zero")
    if len(set(offsets)) != len(offsets):
        raise ValueError("offset candidates must be unique")
    if superiority_margin < 0:
        raise ValueError("superiority_margin must be non-negative")
    if not 0 < consistency_ratio <= 1:
        raise ValueError("consistency_ratio must be in (0, 1]")

    samples: list[dict[str, Any]] = []
    for row in rows:
        try:
            reference = references[row.keyframe_order]
        except KeyError as exc:
            raise KeyError(f"missing keyframe image for order {row.keyframe_order}") from exc
        comparisons: list[dict[str, Any]] = []
        for offset in offsets:
            decoded_frame_id = row.actual_frame_id + offset
            if not 0 <= decoded_frame_id < total_frames:
                comparisons.append(
                    {
                        "offset": offset,
                        "decoded_frame_id": decoded_frame_id,
                        "valid": False,
                        "reason": "outside decoded video bounds",
                    }
                )
                continue
            candidate = decode_frame(decoded_frame_id)
            if align_candidate is not None:
                candidate = align_candidate(reference, candidate)
            scores = compare_images(reference, candidate)
            comparisons.append(
                {
                    "offset": offset,
                    "decoded_frame_id": decoded_frame_id,
                    "valid": True,
                    "scores": scores,
                }
            )
        valid = [item for item in comparisons if item["valid"]]
        if not valid:
            raise ValueError(f"no valid offset for frame {row.actual_frame_id}")
        best = sorted(
            valid,
            key=lambda item: (
                -item["scores"]["similarity"],
                item["offset"] != 0,
                abs(item["offset"]),
                item["offset"],
            ),
        )[0]
        samples.append(
            {
                "keyframe_order": row.keyframe_order,
                "mapped_actual_frame_id": row.actual_frame_id,
                "physical_row": row.physical_row,
                "comparisons": comparisons,
                "best_offset": best["offset"],
            }
        )

    aggregate: dict[str, dict[str, float | int]] = {}
    for offset in offsets:
        similarities = [
            comparison["scores"]["similarity"]
            for sample in samples
            for comparison in sample["comparisons"]
            if comparison["valid"] and comparison["offset"] == offset
        ]
        aggregate[str(offset)] = {
            "valid_sample_count": len(similarities),
            "mean_similarity": float(np.mean(similarities)) if similarities else float("nan"),
            "median_similarity": (float(np.median(similarities)) if similarities else float("nan")),
        }

    superior_nonzero: list[dict[str, Any]] = []
    for offset in offsets:
        if offset == 0:
            continue
        paired: list[tuple[float, float]] = []
        for sample in samples:
            by_offset = {
                item["offset"]: item["scores"]["similarity"]
                for item in sample["comparisons"]
                if item["valid"]
            }
            if 0 in by_offset and offset in by_offset:
                paired.append((by_offset[0], by_offset[offset]))
        if not paired:
            continue
        wins = sum(candidate > zero + superiority_margin for zero, candidate in paired)
        ratio = wins / len(paired)
        improvement = float(np.mean([candidate - zero for zero, candidate in paired]))
        if ratio >= consistency_ratio and improvement > superiority_margin:
            superior_nonzero.append(
                {
                    "offset": offset,
                    "win_ratio_vs_zero": ratio,
                    "mean_similarity_improvement_vs_zero": improvement,
                }
            )

    zero_best_ratio = sum(sample["best_offset"] == 0 for sample in samples) / len(samples)
    if superior_nonzero:
        status = "FAILED"
        reason = "a consistent non-zero offset is superior to the preserved frame_idx"
    elif zero_best_ratio >= consistency_ratio:
        status = "PASSED"
        reason = "zero offset is the deterministic best match for the required ratio"
    else:
        status = "INCONCLUSIVE"
        reason = "no consistent superior offset was established"

    return {
        "status": status,
        "reason": reason,
        "frame_policy": {
            "mapped_value": "frame_idx preserved exactly",
            "working_interpretation": "zero_based",
            "automatic_offset_correction_applied": False,
        },
        "sample_count": len(samples),
        "offset_candidates": list(offsets),
        "zero_best_ratio": zero_best_ratio,
        "superior_nonzero_offsets": superior_nonzero,
        "aggregate_by_offset": aggregate,
        "samples": samples,
    }


def _trailing_number(path: Path) -> int | None:
    match = re.search(r"(\d+)$", path.stem)
    return int(match.group(1)) if match else None


def _resolve_keyframe_paths(source: Path, orders: Sequence[int]) -> dict[int, Path]:
    source = Path(source)
    if source.is_file():
        if len(orders) != 1:
            raise ValueError("a single keyframe file requires exactly one sampled order")
        return {orders[0]: source}
    if not source.is_dir():
        raise FileNotFoundError(f"keyframe path not found: {source}")
    by_order: dict[int, list[Path]] = {}
    for path in source.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        order = _trailing_number(path)
        if order is not None:
            by_order.setdefault(order, []).append(path)
    resolved: dict[int, Path] = {}
    for order in orders:
        matches = sorted(by_order.get(order, []), key=lambda path: str(path).lower())
        if not matches:
            raise FileNotFoundError(f"no keyframe image found for order {order}")
        if len(matches) > 1:
            raise ValueError(f"ambiguous keyframe images for order {order}: {matches}")
        resolved[order] = matches[0]
    return resolved


def _load_cv2() -> Any:
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is required for real video calibration; install/import cv2 in Kaggle"
        ) from exc
    return cv2


class OpenCVFrameDecoder:
    def __init__(self, video_path: Path) -> None:
        self.cv2 = _load_cv2()
        self.video_path = Path(video_path)
        if not self.video_path.is_file():
            raise FileNotFoundError(f"original video not found: {self.video_path}")
        self.capture = self.cv2.VideoCapture(str(self.video_path))
        if not self.capture.isOpened():
            raise RuntimeError(f"cannot open original video: {self.video_path}")
        self.total_frames = int(round(self.capture.get(self.cv2.CAP_PROP_FRAME_COUNT)))
        if self.total_frames <= 0:
            self.close()
            raise RuntimeError(f"decoder reported invalid frame count: {self.total_frames}")

    def decode(self, frame_id: int) -> NDArray[np.number]:
        if not 0 <= frame_id < self.total_frames:
            raise ValueError(f"decoded frame outside bounds: {frame_id}")
        if not self.capture.set(self.cv2.CAP_PROP_POS_FRAMES, frame_id):
            raise RuntimeError(f"decoder seek failed for frame {frame_id}")
        ok, frame = self.capture.read()
        if not ok or frame is None:
            raise RuntimeError(f"decoder read failed for frame {frame_id}")
        return frame

    def close(self) -> None:
        if hasattr(self, "capture"):
            self.capture.release()

    def __enter__(self) -> OpenCVFrameDecoder:
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        self.close()


def calibrate_case(
    *,
    video_id: str,
    video_path: Path,
    mapping_csv: Path,
    keyframes: Path,
    sample_count: int,
    keyframe_orders: Sequence[int] | None,
    offsets: Sequence[int],
    superiority_margin: float,
    consistency_ratio: float,
) -> dict[str, Any]:
    rows = load_mapping_rows(mapping_csv)
    selected = select_mapping_rows(rows, sample_count=sample_count, explicit_orders=keyframe_orders)
    paths = _resolve_keyframe_paths(keyframes, [row.keyframe_order for row in selected])
    cv2 = _load_cv2()
    references: dict[int, NDArray[np.number]] = {}
    for order, path in paths.items():
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"cannot decode keyframe image: {path}")
        references[order] = image

    def align(reference: NDArray[np.number], candidate: NDArray[np.number]) -> NDArray[np.number]:
        if candidate.shape == reference.shape:
            return candidate
        height, width = reference.shape[:2]
        return cv2.resize(candidate, (width, height), interpolation=cv2.INTER_AREA)

    with OpenCVFrameDecoder(video_path) as decoder:
        report = evaluate_frame_offsets(
            selected,
            references,
            decoder.decode,
            total_frames=decoder.total_frames,
            offsets=offsets,
            align_candidate=align,
            superiority_margin=superiority_margin,
            consistency_ratio=consistency_ratio,
        )
    report.update(
        {
            "video_id": video_id,
            "inputs": {
                "video_path": str(Path(video_path).resolve(strict=False)),
                "mapping_csv": str(Path(mapping_csv).resolve(strict=False)),
                "keyframes": str(Path(keyframes).resolve(strict=False)),
            },
            "keyframe_files": {
                str(order): str(path.resolve(strict=False)) for order, path in paths.items()
            },
            "evidence_scope": "REAL_ARTIFACT_EXECUTION",
        }
    )
    return report


def _load_batch_cases(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"batch manifest not found: {path}")
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    cases = payload.get("cases") if isinstance(payload, dict) else None
    if not isinstance(cases, list) or not cases:
        raise ValueError("batch manifest must contain a non-empty cases list")
    return cases


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-path", type=Path)
    parser.add_argument("--mapping-csv", type=Path)
    parser.add_argument("--keyframes", type=Path)
    parser.add_argument("--video-id")
    parser.add_argument("--batch-manifest", type=Path)
    parser.add_argument("--sample-count", type=int, default=9)
    parser.add_argument("--keyframe-orders", type=int, nargs="+")
    parser.add_argument("--offset-candidates", type=int, nargs="+", default=[-1, 0, 1])
    parser.add_argument("--superiority-margin", type=float, default=1e-4)
    parser.add_argument("--consistency-ratio", type=float, default=0.8)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _case_from_args(args: argparse.Namespace) -> dict[str, Any]:
    required = {
        "video_id": args.video_id,
        "video_path": args.video_path,
        "mapping_csv": args.mapping_csv,
        "keyframes": args.keyframes,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(f"single-case execution missing arguments: {', '.join(missing)}")
    return required


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cases = (
            _load_batch_cases(args.batch_manifest)
            if args.batch_manifest
            else [_case_from_args(args)]
        )
        reports: list[dict[str, Any]] = []
        for case in cases:
            reports.append(
                calibrate_case(
                    video_id=str(case["video_id"]),
                    video_path=Path(case["video_path"]),
                    mapping_csv=Path(case["mapping_csv"]),
                    keyframes=Path(case["keyframes"]),
                    sample_count=int(case.get("sample_count", args.sample_count)),
                    keyframe_orders=case.get("keyframe_orders", args.keyframe_orders),
                    offsets=case.get("offset_candidates", args.offset_candidates),
                    superiority_margin=float(
                        case.get("superiority_margin", args.superiority_margin)
                    ),
                    consistency_ratio=float(case.get("consistency_ratio", args.consistency_ratio)),
                )
            )
        statuses = [report["status"] for report in reports]
        overall = (
            "FAILED"
            if "FAILED" in statuses
            else "INCONCLUSIVE"
            if "INCONCLUSIVE" in statuses
            else "PASSED"
        )
        output = {
            "status": overall,
            "case_count": len(reports),
            "minimum_recommended_real_video_count": 3,
            "multi_video_requirement_met": len(reports) >= 3,
            "cases": reports,
            "automatic_offset_correction_applied": False,
        }
        exit_code = 1 if overall == "FAILED" else 0
    except (OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        output = {
            "status": "ERROR",
            "error": str(exc),
            "automatic_offset_correction_applied": False,
        }
        exit_code = 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, ensure_ascii=False, separators=(",", ":")))
    return exit_code


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
