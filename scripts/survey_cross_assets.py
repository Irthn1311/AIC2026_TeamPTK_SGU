"""Run TRIAGE-EG Targeted Cross-Asset Survey v0.2."""

from __future__ import annotations

import argparse
import sys

from triage_eg.data.cross_asset_survey import (
    DEFAULT_OUTPUT_ROOT,
    CrossAssetLimits,
    resolve_dataset_root,
    result_json,
    survey_cross_assets,
    write_outputs,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--video-ids", help="Comma-separated preferred IDs")
    parser.add_argument("--max-videos", type=int, default=5)
    parser.add_argument("--max-object-json-total", type=int, default=25)
    parser.add_argument("--max-object-json-bytes", type=int, default=1_048_576)
    parser.add_argument("--max-mapping-rows", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--strict-root", action="store_true")
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--fail-on-id-set-mismatch", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    requested = (
        [value.strip() for value in args.video_ids.split(",") if value.strip()]
        if args.video_ids
        else None
    )
    try:
        result = survey_cross_assets(
            resolve_dataset_root(args.dataset_root),
            limits=CrossAssetLimits(
                max_videos=args.max_videos,
                max_object_json_total=args.max_object_json_total,
                max_object_json_bytes=args.max_object_json_bytes,
                max_mapping_rows=args.max_mapping_rows,
                seed=args.seed,
            ),
            video_ids=requested,
            strict_root=args.strict_root,
        )
        if args.no_write:
            print(result_json(result))
        else:
            paths = write_outputs(result, args.output_root)
            for name, path in paths.items():
                print(f"{name}: {path}")
    except (FileNotFoundError, OSError, TypeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    if args.fail_on_id_set_mismatch and not result.summary["id_set_comparison"]["all_equal"]:
        return 3
    print("This is a targeted cross-asset contract survey, not a complete Data Audit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
