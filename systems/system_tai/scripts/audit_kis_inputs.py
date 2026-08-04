"""Audit BTC video catalog, frame mapping, and CLIP features without retrieval."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from system_tai.data.frame_mapping import FrameMappingLoader
from system_tai.data.video_catalog import BenchmarkVideoCatalog
from system_tai.features.btc_clip_store import BTCClipFeatureStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-catalog", type=Path, required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--mapping-csv", type=Path, required=True)
    parser.add_argument("--clip-npy", type=Path, required=True)
    parser.add_argument("--expected-dimension", type=int)
    parser.add_argument("--strict-video-path-check", action="store_true")
    parser.add_argument("--duration-tolerance-seconds", type=float, default=1.0)
    parser.add_argument("--fps-tolerance", type=float, default=1e-6)
    parser.add_argument("--normalization-tolerance", type=float, default=1e-3)
    parser.add_argument("--mapping-version", default="audit-unversioned")
    parser.add_argument("--encoder-id", default="btc-visual-feature-artifact")
    parser.add_argument("--output", type=Path)
    return parser


def _frame_example(record: Any) -> dict[str, Any]:
    return {
        "video_id": record.video_id,
        "physical_row": record.physical_row,
        "keyframe_order": record.keyframe_order,
        "clip_row": record.clip_row,
        "pts_time": record.pts_time,
        "fps": record.fps,
        "actual_frame_id": record.actual_frame_id,
        "keyframe_filename": record.keyframe_filename,
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    report: dict[str, Any] = {
        "valid": False,
        "inputs": {
            "video_catalog": str(args.video_catalog),
            "video_id": args.video_id,
            "mapping_csv": str(args.mapping_csv),
            "clip_npy": str(args.clip_npy),
            "expected_dimension": args.expected_dimension,
            "strict_video_path_check": args.strict_video_path_check,
        },
        "catalog": {"valid": False},
        "mapping": {"valid": False},
        "features": {"valid": False},
        "errors": [],
    }

    try:
        catalog = BenchmarkVideoCatalog(
            strict_paths=args.strict_video_path_check,
            duration_tolerance_seconds=args.duration_tolerance_seconds,
        )
        catalog.load(args.video_catalog)
        video = catalog.get(args.video_id)
        lower_bound, upper_bound = catalog.frame_bounds(args.video_id)
        report["catalog"] = {
            "valid": True,
            "record_count": len(catalog.records),
            "selected_video": {
                "video_id": video.video_id,
                "video_path": str(video.video_path),
                "fps": video.fps,
                "duration_seconds": video.duration_seconds,
                "total_frames": video.total_frames,
                "frame_index_base": video.frame_index_base.value,
                "actual_frame_bounds": [lower_bound, upper_bound],
                "codec": video.codec,
                "resolution": (
                    [video.width, video.height] if video.width is not None else None
                ),
            },
        }
    except (OSError, ValueError, KeyError) as exc:
        report["errors"].append(
            {"stage": "catalog", "type": type(exc).__name__, "message": str(exc)}
        )
        return report

    try:
        mapping_loader = FrameMappingLoader(fps_tolerance=args.fps_tolerance)
        records = mapping_loader.load(
            args.mapping_csv,
            catalog,
            mapping_version=args.mapping_version,
            video_id=args.video_id,
            use_physical_clip_rows=True,
        )
        relation_n_minus_one = all(
            record.keyframe_order is not None
            and record.clip_row == record.keyframe_order - 1
            for record in records
        )
        relation_physical_row = all(
            record.clip_row == record.physical_row for record in records
        )
        frame_time_errors = [
            abs(record.actual_frame_id - record.pts_time * record.fps) for record in records
        ]
        report["mapping"] = {
            "valid": True,
            "feature_row_alignment_validated": False,
            "row_count": len(records),
            "first": _frame_example(records[0]),
            "last": _frame_example(records[-1]),
            "actual_frame_min": min(record.actual_frame_id for record in records),
            "actual_frame_max": max(record.actual_frame_id for record in records),
            "observed_relations": {
                "clip_row_equals_physical_zero_based_row": relation_physical_row,
                "clip_row_equals_keyframe_order_minus_one": relation_n_minus_one,
                "frame_idx_vs_pts_time_times_fps": {
                    "max_absolute_error_frames": max(frame_time_errors),
                    "mean_absolute_error_frames": sum(frame_time_errors)
                    / len(frame_time_errors),
                },
                "warning": (
                    "Observed relations apply only to this audited artifact and are not "
                    "dataset-wide invariants."
                ),
            },
        }
    except (OSError, ValueError, KeyError) as exc:
        report["errors"].append(
            {"stage": "mapping", "type": type(exc).__name__, "message": str(exc)}
        )
        return report

    try:
        store = BTCClipFeatureStore(
            normalization_tolerance=args.normalization_tolerance
        )
        stats = store.load(
            args.clip_npy,
            records,
            encoder_id=args.encoder_id,
            expected_dimension=args.expected_dimension,
        )
        report["features"] = {
            "valid": True,
            "shape": [stats.row_count, stats.dimension],
            "dtype": stats.dtype,
            "contains_nan": stats.contains_nan,
            "contains_infinity": stats.contains_infinity,
            "value_statistics": {
                "min": stats.value_min,
                "max": stats.value_max,
                "mean": stats.value_mean,
            },
            "norm_statistics": {
                "min": stats.norm_min,
                "max": stats.norm_max,
                "mean": stats.norm_mean,
                "appears_l2_normalized": stats.appears_l2_normalized,
                "tolerance": stats.normalization_tolerance,
            },
            "row_count_agreement": stats.row_count == len(records),
            "mapping_coverage": {
                "mapped_rows": len(store.feature_records),
                "feature_rows": stats.row_count,
                "ratio": len(store.feature_records) / stats.row_count,
            },
            "first_row_actual_frame_id": store.frame_for_row(0).actual_frame_id,
            "last_row_actual_frame_id": store.frame_for_row(
                stats.row_count - 1
            ).actual_frame_id,
        }
        report["mapping"]["feature_row_alignment_validated"] = True
    except (OSError, ValueError, KeyError) as exc:
        report["errors"].append(
            {"stage": "features", "type": type(exc).__name__, "message": str(exc)}
        )
        return report

    report["valid"] = True
    return report


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = audit(args)
    serialized = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
