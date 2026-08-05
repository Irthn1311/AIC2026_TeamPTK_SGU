"""Calibrate BTC map-keyframe coordinates against decoded original-video frames."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from decimal import ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP, Decimal, InvalidOperation
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
    pts_time: Decimal | float | str
    fps: Decimal | float | str


def load_mapping_rows(path: Path) -> tuple[CalibrationMappingRow, ...]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"mapping CSV not found: {path}")
    rows: list[CalibrationMappingRow] = []
    seen_orders: set[int] = set()
    seen_frames: set[int] = set()
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
                pts_time = Decimal((raw.get("pts_time") or "").strip())
                fps = Decimal((raw.get("fps") or "").strip())
            except (InvalidOperation, ValueError) as exc:
                raise ValueError(f"invalid mapping value at line {line_number}") from exc
            if order < 0 or frame_idx < 0 or pts_time < 0 or fps <= 0:
                raise ValueError(f"invalid non-positive mapping value at line {line_number}")
            if not pts_time.is_finite() or not fps.is_finite():
                raise ValueError(f"non-finite mapping value at line {line_number}")
            if order in seen_orders:
                raise ValueError(f"duplicate keyframe order at line {line_number}: {order}")
            if frame_idx in seen_frames:
                raise ValueError(
                    f"ambiguous duplicate frame_idx at line {line_number}: {frame_idx}"
                )
            seen_orders.add(order)
            seen_frames.add(frame_idx)
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


def _as_decimal(value: Decimal | float | str, field: str) -> Decimal:
    try:
        decimal_value = value if isinstance(value, Decimal) else Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError(f"invalid {field}: {value!r}") from exc
    if not decimal_value.is_finite():
        raise ValueError(f"non-finite {field}: {value!r}")
    return decimal_value


def timestamp_rounding_diagnostic(row: CalibrationMappingRow) -> dict[str, Any]:
    pts_time = _as_decimal(row.pts_time, "pts_time")
    fps = _as_decimal(row.fps, "fps")
    if pts_time < 0 or fps <= 0:
        raise ValueError("pts_time must be non-negative and fps must be positive")
    decimal_exact_product = pts_time * fps
    decimal_floor = int(decimal_exact_product.to_integral_value(rounding=ROUND_FLOOR))
    decimal_round_half_up = int(
        decimal_exact_product.to_integral_value(rounding=ROUND_HALF_UP)
    )
    decimal_ceil = int(decimal_exact_product.to_integral_value(rounding=ROUND_CEILING))
    binary_float_product = float(pts_time) * float(fps)
    if not math.isfinite(binary_float_product):
        raise ValueError("binary float timestamp product must be finite")
    binary_float_truncation = int(binary_float_product)
    return {
        "decimal_exact_product": format(decimal_exact_product, "f"),
        "decimal_floor": decimal_floor,
        "decimal_round_half_up": decimal_round_half_up,
        "decimal_ceil": decimal_ceil,
        "binary_float_product": repr(binary_float_product),
        "binary_float_truncation": binary_float_truncation,
        "binary_float_floor": math.floor(binary_float_product),
        "frame_idx_minus_decimal_floor": row.actual_frame_id - decimal_floor,
        "frame_idx_minus_binary_float_truncation": (
            row.actual_frame_id - binary_float_truncation
        ),
        "decimal_nearest_minus_frame_idx": decimal_round_half_up - row.actual_frame_id,
        "predicted_visual_offset": decimal_round_half_up - row.actual_frame_id,
        "numeric_generation_rule_is_diagnostic": True,
        "numeric_rule_modifies_mapping_validity": False,
    }


def validate_mapping_coordinates(
    rows: Sequence[CalibrationMappingRow], *, total_frames: int
) -> dict[str, Any]:
    if total_frames <= 0:
        raise ValueError("total_frames must be positive")
    if not rows:
        raise ValueError("mapping rows must not be empty")
    invalid = [
        {
            "keyframe_order": row.keyframe_order,
            "actual_frame_id": row.actual_frame_id,
            "expected_bounds": [0, total_frames - 1],
        }
        for row in rows
        if not 0 <= row.actual_frame_id < total_frames
    ]
    return {
        "status": "MAPPING_POLICY_PASSED" if not invalid else "MAPPING_POLICY_FAILED",
        "actual_frame_id_source": "frame_idx_exactly",
        "working_coordinate": "zero_based",
        "frame_bounds": [0, total_frames - 1],
        "mapping_row_count": len(rows),
        "actual_frame_range": [
            min(row.actual_frame_id for row in rows),
            max(row.actual_frame_id for row in rows),
        ],
        "out_of_bounds_rows": invalid,
        "numeric_generation_rule_required": False,
        "automatic_offset_correction_applied": False,
    }


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


def classify_visual_decision(
    valid_comparisons: Sequence[dict[str, Any]], *, superiority_margin: float
) -> dict[str, Any]:
    if not valid_comparisons:
        raise ValueError("at least one valid visual comparison is required")
    if superiority_margin < 0:
        raise ValueError("superiority_margin must be non-negative")
    ordered = sorted(
        valid_comparisons,
        key=lambda item: (
            -float(item["scores"]["similarity"]),
            item["offset"] != 0,
            abs(int(item["offset"])),
            int(item["offset"]),
        ),
    )
    best = ordered[0]
    best_similarity = float(best["scores"]["similarity"])
    second_best_similarity = (
        float(ordered[1]["scores"]["similarity"]) if len(ordered) > 1 else None
    )
    best_minus_second_margin = (
        best_similarity - second_best_similarity
        if second_best_similarity is not None
        else None
    )
    ambiguous = (
        best_minus_second_margin is not None
        and best_minus_second_margin <= superiority_margin
    )
    tied_offsets = [
        int(item["offset"])
        for item in ordered
        if best_similarity - float(item["scores"]["similarity"])
        <= superiority_margin
    ]
    return {
        "best": best,
        "visual_decision_status": "AMBIGUOUS" if ambiguous else "DECISIVE",
        "best_similarity": best_similarity,
        "second_best_similarity": second_best_similarity,
        "best_minus_second_margin": best_minus_second_margin,
        "tied_offset_candidates": tied_offsets,
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
    sequential_decode_frame: Callable[[int], NDArray[np.number]] | None = None,
    decoder_agreement_tolerance: float = 1e-6,
    superiority_margin: float = 1e-4,
    consistency_ratio: float = 0.8,
) -> dict[str, Any]:
    if 0 not in offsets:
        raise ValueError("offset candidates must include zero")
    if len(set(offsets)) != len(offsets):
        raise ValueError("offset candidates must be unique")
    if decoder_agreement_tolerance < 0:
        raise ValueError("decoder_agreement_tolerance must be non-negative")
    if superiority_margin < 0:
        raise ValueError("superiority_margin must be non-negative")
    if not 0 < consistency_ratio <= 1:
        raise ValueError("consistency_ratio must be in (0, 1]")

    mapping_validation = validate_mapping_coordinates(rows, total_frames=total_frames)
    if mapping_validation["status"] == "MAPPING_POLICY_FAILED":
        return {
            "status": "MAPPING_POLICY_FAILED",
            "reason": "one or more frame_idx values are outside zero-based video bounds",
            "mapping_coordinate_validation": mapping_validation,
            "timestamp_rounding_diagnostics": {"status": "NOT_RUN"},
            "visual_artifact_agreement": {"status": "NOT_RUN"},
            "decoder_agreement": {"status": "NOT_RUN"},
            "frame_policy": {
                "mapped_value": "frame_idx preserved exactly",
                "automatic_offset_correction_applied": False,
            },
            "samples": [],
        }

    samples: list[dict[str, Any]] = []
    decoder_disagreements: list[dict[str, Any]] = []
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
            decoder_scores: dict[str, float] | None = None
            if sequential_decode_frame is not None:
                sequential_candidate = sequential_decode_frame(decoded_frame_id)
                if align_candidate is not None:
                    sequential_candidate = align_candidate(reference, sequential_candidate)
                decoder_scores = compare_images(candidate, sequential_candidate)
                if decoder_scores["normalized_mae"] > decoder_agreement_tolerance:
                    decoder_disagreements.append(
                        {
                            "keyframe_order": row.keyframe_order,
                            "decoded_frame_id": decoded_frame_id,
                            "offset": offset,
                            "scores": decoder_scores,
                        }
                    )
            comparisons.append(
                {
                    "offset": offset,
                    "decoded_frame_id": decoded_frame_id,
                    "valid": True,
                    "scores": scores,
                    "random_vs_sequential_scores": decoder_scores,
                }
            )
        valid = [item for item in comparisons if item["valid"]]
        if not valid:
            raise ValueError(f"no valid offset for frame {row.actual_frame_id}")
        decision = classify_visual_decision(
            valid, superiority_margin=superiority_margin
        )
        best = decision["best"]
        rounding = timestamp_rounding_diagnostic(row)
        visual_best_offset = int(best["offset"])
        predicted_visual_offset = int(rounding["predicted_visual_offset"])
        is_decisive = decision["visual_decision_status"] == "DECISIVE"
        prediction_match = visual_best_offset == predicted_visual_offset
        samples.append(
            {
                "keyframe_order": row.keyframe_order,
                "mapped_actual_frame_id": row.actual_frame_id,
                "physical_row": row.physical_row,
                "timestamp_rounding": rounding,
                "comparisons": comparisons,
                "visual_best_offset": visual_best_offset,
                "predicted_visual_offset": predicted_visual_offset,
                "visual_decision_status": decision["visual_decision_status"],
                "best_similarity": decision["best_similarity"],
                "second_best_similarity": decision["second_best_similarity"],
                "best_minus_second_margin": decision["best_minus_second_margin"],
                "tied_offset_candidates": decision["tied_offset_candidates"],
                "visual_best_matches_round_half_up_prediction": (
                    prediction_match if is_decisive else None
                ),
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

    visual_best_counts = {
        str(offset): sum(sample["visual_best_offset"] == offset for sample in samples)
        for offset in offsets
    }
    decisive_samples = [
        sample for sample in samples if sample["visual_decision_status"] == "DECISIVE"
    ]
    ambiguous_samples = [
        sample for sample in samples if sample["visual_decision_status"] == "AMBIGUOUS"
    ]
    explained_decisive_samples = [
        sample
        for sample in decisive_samples
        if sample["visual_best_matches_round_half_up_prediction"] is True
    ]
    contradictory_decisive_samples = [
        sample
        for sample in decisive_samples
        if sample["visual_best_matches_round_half_up_prediction"] is False
    ]
    decisive_count = len(decisive_samples)
    explained_decisive_count = len(explained_decisive_samples)
    explained_decisive_ratio = (
        explained_decisive_count / decisive_count if decisive_count else 1.0
    )
    systematic_unexplained_offsets: list[dict[str, Any]] = []
    for offset in offsets:
        if offset == 0:
            continue
        unexplained = [
            sample
            for sample in contradictory_decisive_samples
            if sample["visual_best_offset"] == offset
        ]
        ratio = len(unexplained) / decisive_count if decisive_count else 0.0
        if ratio >= consistency_ratio:
            systematic_unexplained_offsets.append(
                {
                    "offset": offset,
                    "sample_count": len(unexplained),
                    "ratio": ratio,
                }
            )

    if decoder_disagreements:
        status = "MAPPING_POLICY_FAILED"
        reason = "random and sequential decoding disagree materially"
    elif systematic_unexplained_offsets:
        status = "MAPPING_POLICY_FAILED"
        reason = "a systematic visual offset is not explained by Decimal-nearest rounding"
    elif not contradictory_decisive_samples:
        status = "VISUAL_ALIGNMENT_EXPLAINED"
        reason = (
            "every decisive visual offset equals the Decimal-nearest prediction; "
            "remaining samples are ambiguous"
        )
    else:
        status = "VISUAL_ALIGNMENT_INCONCLUSIVE"
        reason = "visual offsets are not systematic but are not fully explained"

    return {
        "status": status,
        "reason": reason,
        "mapping_coordinate_validation": mapping_validation,
        "timestamp_rounding_diagnostics": {
            "status": "DIAGNOSTIC_ONLY",
            "mapping_validity_dependency": False,
            "numeric_models": [
                "decimal_exact_floor",
                "binary_float_truncation",
                "decimal_round_half_up_visual_prediction",
            ],
            "visual_prediction": "decimal_round_half_up(pts_time * fps) - frame_idx",
        },
        "visual_artifact_agreement": {
            "status": (
                status
                if status in {"VISUAL_ALIGNMENT_EXPLAINED", "VISUAL_ALIGNMENT_INCONCLUSIVE"}
                else "MAPPING_POLICY_FAILED"
            ),
            "decisive_sample_count": decisive_count,
            "ambiguous_sample_count": len(ambiguous_samples),
            "explained_decisive_sample_count": explained_decisive_count,
            "explained_decisive_ratio": explained_decisive_ratio,
            "contradictory_decisive_sample_count": len(
                contradictory_decisive_samples
            ),
            "visual_best_offset_distribution": visual_best_counts,
            "systematic_unexplained_offsets": systematic_unexplained_offsets,
            "superiority_margin": superiority_margin,
        },
        "decoder_agreement": {
            "status": (
                "NOT_COMPARED"
                if sequential_decode_frame is None
                else "DISAGREEMENT"
                if decoder_disagreements
                else "AGREEMENT"
            ),
            "normalized_mae_tolerance": decoder_agreement_tolerance,
            "material_disagreement_count": len(decoder_disagreements),
            "material_disagreements": decoder_disagreements,
        },
        "frame_policy": {
            "mapped_value": "frame_idx preserved exactly",
            "working_interpretation": "zero_based",
            "keyframe_visual_frame_id": "decimal_round_half_up(pts_time * fps)",
            "visual_diagnostic_modifies_actual_frame_id": False,
            "automatic_offset_correction_applied": False,
        },
        "sample_count": len(samples),
        "decisive_sample_count": decisive_count,
        "ambiguous_sample_count": len(ambiguous_samples),
        "explained_decisive_sample_count": explained_decisive_count,
        "explained_decisive_ratio": explained_decisive_ratio,
        "contradictory_decisive_sample_count": len(contradictory_decisive_samples),
        "offset_candidates": list(offsets),
        "zero_best_ratio": visual_best_counts.get("0", 0) / len(samples),
        "systematic_unexplained_offsets": systematic_unexplained_offsets,
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


def decode_frames_sequentially(
    video_path: Path, frame_ids: Sequence[int]
) -> dict[int, NDArray[np.number]]:
    cv2 = _load_cv2()
    requested = sorted(set(frame_ids))
    if not requested:
        return {}
    if requested[0] < 0:
        raise ValueError("sequential decoder frame IDs must be non-negative")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open original video for sequential decode: {video_path}")
    decoded: dict[int, NDArray[np.number]] = {}
    targets = set(requested)
    try:
        for frame_id in range(requested[-1] + 1):
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError(f"sequential decoder read failed at frame {frame_id}")
            if frame_id in targets:
                decoded[frame_id] = frame
    finally:
        capture.release()
    missing = sorted(targets - set(decoded))
    if missing:
        raise RuntimeError(f"sequential decoder did not produce requested frames: {missing}")
    return decoded


def calibrate_case(
    *,
    video_id: str,
    video_path: Path,
    mapping_csv: Path,
    keyframes: Path,
    sample_count: int,
    keyframe_orders: Sequence[int] | None,
    offsets: Sequence[int],
    decoder_agreement_tolerance: float,
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
        mapping_validation = validate_mapping_coordinates(
            rows, total_frames=decoder.total_frames
        )
        if mapping_validation["status"] == "MAPPING_POLICY_FAILED":
            report = {
                "status": "MAPPING_POLICY_FAILED",
                "reason": "one or more frame_idx values are outside video bounds",
                "mapping_coordinate_validation": mapping_validation,
                "timestamp_rounding_diagnostics": {"status": "NOT_RUN"},
                "visual_artifact_agreement": {"status": "NOT_RUN"},
                "decoder_agreement": {"status": "NOT_RUN"},
                "samples": [],
                "frame_policy": {
                    "mapped_value": "frame_idx preserved exactly",
                    "automatic_offset_correction_applied": False,
                },
            }
        else:
            target_frames = [
                row.actual_frame_id + offset
                for row in selected
                for offset in offsets
                if 0 <= row.actual_frame_id + offset < decoder.total_frames
            ]
            sequential_frames = decode_frames_sequentially(video_path, target_frames)
            report = evaluate_frame_offsets(
                selected,
                references,
                decoder.decode,
                total_frames=decoder.total_frames,
                offsets=offsets,
                align_candidate=align,
                sequential_decode_frame=sequential_frames.__getitem__,
                decoder_agreement_tolerance=decoder_agreement_tolerance,
                superiority_margin=superiority_margin,
                consistency_ratio=consistency_ratio,
            )
            report["mapping_coordinate_validation"] = mapping_validation
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
    parser.add_argument("--decoder-agreement-tolerance", type=float, default=1e-6)
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
            video_id = str(case.get("video_id", "UNKNOWN"))
            try:
                reports.append(
                    calibrate_case(
                        video_id=video_id,
                        video_path=Path(case["video_path"]),
                        mapping_csv=Path(case["mapping_csv"]),
                        keyframes=Path(case["keyframes"]),
                        sample_count=int(case.get("sample_count", args.sample_count)),
                        keyframe_orders=case.get("keyframe_orders", args.keyframe_orders),
                        offsets=case.get("offset_candidates", args.offset_candidates),
                        decoder_agreement_tolerance=float(
                            case.get(
                                "decoder_agreement_tolerance",
                                args.decoder_agreement_tolerance,
                            )
                        ),
                        superiority_margin=float(
                            case.get("superiority_margin", args.superiority_margin)
                        ),
                        consistency_ratio=float(
                            case.get("consistency_ratio", args.consistency_ratio)
                        ),
                    )
                )
            except ValueError as exc:
                reports.append(
                    {
                        "video_id": video_id,
                        "status": "MAPPING_POLICY_FAILED",
                        "reason": str(exc),
                        "automatic_offset_correction_applied": False,
                    }
                )
            except (OSError, RuntimeError, KeyError) as exc:
                reports.append(
                    {
                        "video_id": video_id,
                        "status": "ERROR",
                        "reason": str(exc),
                        "automatic_offset_correction_applied": False,
                    }
                )
        statuses = [report["status"] for report in reports]
        overall = (
            "ERROR"
            if "ERROR" in statuses
            else "MAPPING_POLICY_FAILED"
            if "MAPPING_POLICY_FAILED" in statuses
            else "VISUAL_ALIGNMENT_INCONCLUSIVE"
            if "VISUAL_ALIGNMENT_INCONCLUSIVE" in statuses
            else "VISUAL_ALIGNMENT_EXPLAINED"
        )
        output = {
            "status": overall,
            "case_count": len(reports),
            "minimum_recommended_real_video_count": 3,
            "multi_video_requirement_met": len(reports) >= 3,
            "cases": reports,
            "automatic_offset_correction_applied": False,
        }
        exit_code = 1 if overall in {"ERROR", "MAPPING_POLICY_FAILED"} else 0
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
