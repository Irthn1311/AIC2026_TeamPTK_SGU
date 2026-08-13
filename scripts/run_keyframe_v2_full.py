from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from _bootstrap import PROJECT_ROOT
from src.preprocessing.keyframe_v2.pipeline import run_keyframe_v2


def natural_video_key(path: Path) -> tuple[str, int]:
    stem = path.stem
    try:
        return stem.rsplit("_V", 1)[0], int(stem.rsplit("_V", 1)[1])
    except Exception:
        return stem, 0


def is_complete(video_dir: Path) -> bool:
    keyframes = video_dir / "keyframes"
    return (
        (video_dir / "final_keyframes.csv").exists()
        and (video_dir / "keyframe_v2_map.parquet").exists()
        and keyframes.is_dir()
        and any(keyframes.glob("*.jpg"))
    )


def build_global_map(output_root: Path) -> pd.DataFrame:
    rows = []
    video_dirs = [p for p in output_root.iterdir() if p.is_dir() and (p / "final_keyframes.csv").exists()]
    for video_dir in sorted(video_dirs, key=lambda p: natural_video_key(p)):
        final_path = video_dir / "final_keyframes.csv"
        if not final_path.exists():
            continue
        df = pd.read_csv(final_path)
        if df.empty:
            continue
        for _, row in df.sort_values("keyframe_v2_idx").iterrows():
            rows.append(
                {
                    "global_v2_id": len(rows),
                    "video_id": str(row["video_id"]),
                    "keyframe_v2_idx": int(row["keyframe_v2_idx"]),
                    "actual_frame_id": int(row["actual_frame_id"]),
                    "timestamp_sec": float(row["timestamp_sec"]),
                    "image_path": str(row["image_path"]),
                    "shot_id": int(row["shot_id"]),
                    "quality_score": float(row["quality_score"]),
                    "representative_score": float(row["representative_score"]),
                    "temporal_score": float(row["temporal_score"]),
                    "final_score": float(row["final_score"]),
                }
            )
    global_df = pd.DataFrame(rows)
    index_dir = output_root / "indexes"
    index_dir.mkdir(parents=True, exist_ok=True)
    global_df.to_csv(index_dir / "keyframe_v2_global_map.csv", index=False, encoding="utf-8-sig")
    global_df.to_parquet(index_dir / "keyframe_v2_global_map.parquet", index=False)
    return global_df


def collect_videos(video_roots: list[Path]) -> list[Path]:
    videos: dict[str, Path] = {}
    for video_root in video_roots:
        for path in sorted(video_root.glob("*.mp4"), key=natural_video_key):
            if path.name.startswith("."):
                continue
            videos.setdefault(path.stem, path)
    return sorted(videos.values(), key=natural_video_key)


def write_summaries(output_root: Path, started: float, errors: list[dict]) -> tuple[pd.DataFrame, dict]:
    summary_dir = output_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    totals = {
        "Total videos": 0,
        "Total duration": 0.0,
        "Total shots": 0,
        "Total candidates": 0,
        "Total final keyframes": 0,
        "Total cross-shot removed": 0,
        "Validation PASS": 0,
        "Validation FAIL": 0,
    }

    for video_dir in sorted([p for p in output_root.iterdir() if p.is_dir() and (p / "summary.json").exists()], key=lambda p: natural_video_key(p)):
        summary_path = video_dir / "summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        validation_path = video_dir / "final_frame_validation.csv"
        validation = pd.read_csv(validation_path) if validation_path.exists() else pd.DataFrame()
        validation_pass = int((validation.get("validation_status", pd.Series(dtype=str)) == "ok").sum()) if not validation.empty else 0
        validation_fail = int((validation.get("validation_status", pd.Series(dtype=str)) != "ok").sum()) if not validation.empty else 0
        duration = float(summary["video"]["duration"])
        final_keyframes = int(summary["keyframes"]["final_keyframes"])
        row = {
            "video_id": video_dir.name,
            "duration_sec": duration,
            "total_frames": int(summary["video"]["total_original_frames"]),
            "shots": int(summary["shot"]["total_shots"]),
            "candidates": int(summary["candidates"]["candidate_frames"]),
            "selected_before_dedup": int(summary["keyframes"]["selected_before_dedup"]),
            "cross_shot_removed": int(summary["keyframes"]["cross_shot_removed"]),
            "final_keyframes": final_keyframes,
            "keyframes_per_minute": round(final_keyframes / max(duration / 60.0, 1e-6), 4),
            "frame_validation_pass": validation_pass,
            "frame_validation_fail": validation_fail,
            "shot_runtime_sec": float(summary["performance"].get("shot_detection", 0.0)),
            "clip_runtime_sec": float(summary["performance"].get("candidate_decoding_clip_quality_selection", 0.0)),
            "total_runtime_sec": float(summary["performance"].get("total", 0.0)),
            "status": "validation_warning" if validation_fail else "ok",
        }
        rows.append(row)
        totals["Total videos"] += 1
        totals["Total duration"] += duration
        totals["Total shots"] += row["shots"]
        totals["Total candidates"] += row["candidates"]
        totals["Total final keyframes"] += final_keyframes
        totals["Total cross-shot removed"] += row["cross_shot_removed"]
        totals["Validation PASS"] += validation_pass
        totals["Validation FAIL"] += validation_fail

    df = pd.DataFrame(rows)
    df.to_csv(summary_dir / "full_run_summary.csv", index=False, encoding="utf-8-sig")
    total_runtime = time.time() - started
    total_json = {
        **totals,
        "Average keyframes/video": round(totals["Total final keyframes"] / max(1, totals["Total videos"]), 4),
        "Average keyframes/minute": round(totals["Total final keyframes"] / max(totals["Total duration"] / 60.0, 1e-6), 4),
        "Total runtime": round(total_runtime, 3),
        "Failed/skipped videos": errors,
    }
    (summary_dir / "full_run_summary.json").write_text(json.dumps(total_json, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(errors).to_csv(summary_dir / "errors.csv", index=False, encoding="utf-8-sig")
    return df, total_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Keyframe V2 over video dataset with resume support.")
    parser.add_argument(
        "--video-root",
        action="append",
        default=None,
        help="Directory containing mp4 videos. Can be repeated for multi-folder Kaggle datasets.",
    )
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "keyframe_v2.yaml"))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "outputs" / "keyframe_v2_full"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--debug", action="store_true", help="Generate debug contact sheets for every video.")
    parser.add_argument("--validate-btc-mapping", action="store_true", help="Run BTC mapping validation for every video.")
    parser.add_argument("--limit", type=int, help="Optional development limit. Do not use for final full run.")
    parser.add_argument("--video-id", action="append", default=[], help="Only process this video id. Can be repeated.")
    parser.add_argument("--no-aggregate", action="store_true", help="Only process per-video outputs; skip global map/summary writes.")
    parser.add_argument("--aggregate-only", action="store_true", help="Only rebuild global map and summaries from existing per-video outputs.")
    args = parser.parse_args()

    started = time.time()
    video_roots = [Path(path) for path in (args.video_root or [PROJECT_ROOT / "datasets_L21" / "Videos_L21_a" / "video"])]
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    if args.aggregate_only:
        global_df = build_global_map(output_root)
        summary_df, total_summary = write_summaries(output_root, started, [])
        print(f"Global V2 map rows: {len(global_df)}")
        print(f"Summary videos: {len(summary_df)}")
        print(json.dumps(total_summary, indent=2, ensure_ascii=False))
        if len(global_df) == 0:
            raise SystemExit("Keyframe V2 aggregate produced 0 keyframes; see per-video outputs.")
        return

    videos = collect_videos(video_roots)
    requested = {str(video_id).strip() for video_id in args.video_id if str(video_id).strip()}
    if requested:
        videos = [path for path in videos if path.stem in requested]
    if args.limit:
        videos = videos[: args.limit]

    errors: list[dict] = []
    for idx, video_path in enumerate(videos, start=1):
        video_id = video_path.stem
        video_dir = output_root / video_id
        if not args.force and is_complete(video_dir):
            print(f"[{idx}/{len(videos)}] SKIP complete: {video_id}")
            continue
        print(f"[{idx}/{len(videos)}] RUN Keyframe V2: {video_id}")
        try:
            run_keyframe_v2(
                video_path=video_path,
                config_path=args.config,
                output_root=output_root,
                validate_btc_mapping=args.validate_btc_mapping,
                debug=args.debug,
            )
        except Exception as exc:
            errors.append({"video_id": video_id, "stage": "keyframe_v2", "exception": repr(exc), "status": "failed"})
            print(f"[{idx}/{len(videos)}] FAIL {video_id}: {exc}")
            continue

    if args.no_aggregate:
        if errors:
            raise SystemExit(f"Keyframe V2 shard failed for {len(errors)} video(s): {errors}")
        print(json.dumps({"mode": "per_video_only", "videos": len(videos), "errors": errors}, indent=2, ensure_ascii=False))
        return

    global_df = build_global_map(output_root)
    summary_df, total_summary = write_summaries(output_root, started, errors)
    print(f"Global V2 map rows: {len(global_df)}")
    print(f"Summary videos: {len(summary_df)}")
    print(json.dumps(total_summary, indent=2, ensure_ascii=False))
    if videos and len(global_df) == 0:
        raise SystemExit("Keyframe V2 produced 0 keyframes; see summary/errors.csv for failed videos.")


if __name__ == "__main__":
    main()
