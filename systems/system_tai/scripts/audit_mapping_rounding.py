"""Audit BTC mapping timestamp/frame relations with Decimal arithmetic."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from decimal import ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path
from typing import Any

REQUIRED_COLUMNS = {"n", "pts_time", "fps", "frame_idx"}


def _decimal(raw_value: str, field: str, line_number: int) -> Decimal:
    raw = raw_value.strip()
    try:
        value = Decimal(raw)
    except InvalidOperation as exc:
        raise ValueError(f"invalid {field} at line {line_number}: {raw!r}") from exc
    if not value.is_finite():
        raise ValueError(f"non-finite {field} at line {line_number}: {raw!r}")
    return value


def calculate_rounding_diagnostic(
    *,
    keyframe_order: int,
    pts_time: str | Decimal,
    fps: str | Decimal,
    frame_idx: int,
    physical_row: int,
) -> dict[str, Any]:
    pts_decimal = pts_time if isinstance(pts_time, Decimal) else Decimal(str(pts_time))
    fps_decimal = fps if isinstance(fps, Decimal) else Decimal(str(fps))
    if not pts_decimal.is_finite() or pts_decimal < 0:
        raise ValueError("pts_time must be finite and non-negative")
    if not fps_decimal.is_finite() or fps_decimal <= 0:
        raise ValueError("fps must be finite and positive")
    if keyframe_order < 0 or frame_idx < 0 or physical_row < 0:
        raise ValueError("mapping indexes must be non-negative")

    decimal_exact_product = pts_decimal * fps_decimal
    decimal_floor = int(decimal_exact_product.to_integral_value(rounding=ROUND_FLOOR))
    decimal_round_half_up = int(
        decimal_exact_product.to_integral_value(rounding=ROUND_HALF_UP)
    )
    decimal_ceil = int(decimal_exact_product.to_integral_value(rounding=ROUND_CEILING))

    binary_float_product = float(pts_decimal) * float(fps_decimal)
    if not math.isfinite(binary_float_product):
        raise ValueError("binary float timestamp product must be finite")
    binary_float_truncation = int(binary_float_product)
    binary_float_floor = math.floor(binary_float_product)

    frame_idx_minus_decimal_floor = frame_idx - decimal_floor
    frame_idx_minus_binary_float_truncation = frame_idx - binary_float_truncation
    decimal_nearest_minus_frame_idx = decimal_round_half_up - frame_idx
    matches_decimal_floor = frame_idx == decimal_floor
    matches_binary_float_truncation = frame_idx == binary_float_truncation
    matches_decimal_nearest = frame_idx == decimal_round_half_up
    numeric_rule_unresolved = not (
        matches_decimal_floor
        or matches_binary_float_truncation
        or matches_decimal_nearest
    )
    return {
        "physical_row": physical_row,
        "keyframe_order": keyframe_order,
        "pts_time": format(pts_decimal, "f"),
        "fps": format(fps_decimal, "f"),
        "decimal_exact_product": format(decimal_exact_product, "f"),
        "decimal_floor": decimal_floor,
        "decimal_round_half_up": decimal_round_half_up,
        "decimal_ceil": decimal_ceil,
        "binary_float_product": repr(binary_float_product),
        "binary_float_truncation": binary_float_truncation,
        "binary_float_floor": binary_float_floor,
        "frame_idx": frame_idx,
        "frame_idx_minus_decimal_floor": frame_idx_minus_decimal_floor,
        "frame_idx_minus_binary_float_truncation": (
            frame_idx_minus_binary_float_truncation
        ),
        "decimal_nearest_minus_frame_idx": decimal_nearest_minus_frame_idx,
        "matches_decimal_floor": matches_decimal_floor,
        "matches_binary_float_truncation": matches_binary_float_truncation,
        "matches_decimal_nearest": matches_decimal_nearest,
        "numeric_rule_unresolved": numeric_rule_unresolved,
    }


def load_rounding_diagnostics(mapping_csv: Path) -> tuple[dict[str, Any], ...]:
    mapping_csv = Path(mapping_csv)
    if not mapping_csv.is_file():
        raise FileNotFoundError(f"mapping CSV not found: {mapping_csv}")
    diagnostics: list[dict[str, Any]] = []
    seen_orders: set[int] = set()
    with mapping_csv.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = sorted(REQUIRED_COLUMNS - set(reader.fieldnames or ()))
        if missing:
            raise ValueError(f"mapping CSV missing columns: {', '.join(missing)}")
        for line_number, row in enumerate(reader, start=2):
            if not any((value or "").strip() for value in row.values()):
                continue
            try:
                keyframe_order = int((row.get("n") or "").strip())
                frame_idx = int((row.get("frame_idx") or "").strip())
            except ValueError as exc:
                raise ValueError(f"invalid integer mapping value at line {line_number}") from exc
            if keyframe_order in seen_orders:
                raise ValueError(
                    f"duplicate keyframe order at line {line_number}: {keyframe_order}"
                )
            seen_orders.add(keyframe_order)
            diagnostics.append(
                calculate_rounding_diagnostic(
                    keyframe_order=keyframe_order,
                    pts_time=_decimal(row.get("pts_time") or "", "pts_time", line_number),
                    fps=_decimal(row.get("fps") or "", "fps", line_number),
                    frame_idx=frame_idx,
                    physical_row=len(diagnostics),
                )
            )
    if not diagnostics:
        raise ValueError("mapping CSV contains no rows")
    return tuple(diagnostics)


def _distribution(rows: tuple[dict[str, Any], ...], field: str) -> dict[str, int]:
    counts = Counter(int(row[field]) for row in rows)
    return {str(value): counts[value] for value in sorted(counts)}


def summarize_rounding(
    rows: tuple[dict[str, Any], ...], *, video_id: str
) -> dict[str, Any]:
    if not video_id.strip():
        raise ValueError("video_id must not be empty")
    total = len(rows)
    decimal_floor_count = sum(bool(row["matches_decimal_floor"]) for row in rows)
    binary_truncation_count = sum(
        bool(row["matches_binary_float_truncation"]) for row in rows
    )
    decimal_nearest_count = sum(bool(row["matches_decimal_nearest"]) for row in rows)
    unresolved_rows = [row for row in rows if row["numeric_rule_unresolved"]]
    if decimal_floor_count == total:
        observed_rule = "DECIMAL_FLOOR_OBSERVED"
    elif binary_truncation_count == total:
        observed_rule = "BINARY_FLOAT_TRUNCATION_OBSERVED"
    elif decimal_floor_count / total >= 0.9:
        observed_rule = "DECIMAL_FLOOR_MOSTLY_OBSERVED"
    else:
        observed_rule = "NUMERIC_RULE_UNRESOLVED"
    return {
        "status": observed_rule,
        "audit_valid": True,
        "video_id": video_id,
        "arithmetic_models": [
            "decimal_exact",
            "binary_float_truncation",
            "decimal_round_half_up_visual_diagnostic",
        ],
        "total_rows": total,
        "frame_idx_equals_decimal_floor": {
            "count": decimal_floor_count,
            "ratio": decimal_floor_count / total,
        },
        "frame_idx_equals_binary_float_truncation": {
            "count": binary_truncation_count,
            "ratio": binary_truncation_count / total,
        },
        "frame_idx_equals_decimal_nearest": {
            "count": decimal_nearest_count,
            "ratio": decimal_nearest_count / total,
        },
        "frame_idx_minus_decimal_floor_distribution": _distribution(
            rows, "frame_idx_minus_decimal_floor"
        ),
        "frame_idx_minus_binary_float_truncation_distribution": _distribution(
            rows, "frame_idx_minus_binary_float_truncation"
        ),
        "decimal_nearest_minus_frame_idx_distribution": _distribution(
            rows, "decimal_nearest_minus_frame_idx"
        ),
        "numeric_rule_unresolved_row_count": len(unresolved_rows),
        "numeric_rule_unresolved_rows": unresolved_rows,
        "observed_rule_summary": observed_rule,
        "coordinate_policy": {
            "actual_frame_id_source": "frame_idx_exactly",
            "rounding_diagnostics_modify_shared_frame_id": False,
            "numeric_generation_rule_is_diagnostic": True,
            "numeric_rule_resolution_required_for_mapping_validity": False,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping-csv", type=Path, required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        rows = load_rounding_diagnostics(args.mapping_csv)
        report = summarize_rounding(rows, video_id=args.video_id)
        if args.output.suffix.lower() == ".csv":
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            report["row_diagnostics_csv"] = str(args.output)
        elif args.output.suffix.lower() == ".json":
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
        else:
            raise ValueError("output extension must be .json or .csv")
        exit_code = 0
    except (OSError, ValueError) as exc:
        report = {"status": "ERROR", "video_id": args.video_id, "error": str(exc)}
        exit_code = 1
    print(json.dumps(report, ensure_ascii=False, separators=(",", ":")))
    return exit_code


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
