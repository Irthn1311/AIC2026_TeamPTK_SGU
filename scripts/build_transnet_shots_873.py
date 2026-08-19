"""
Build TransNetV2 Shot Detection Index for 873 Videos (AI Challenge 2026)
===========================================================================
Extracts shot boundaries for all discovered videos using TransNetV2 without
video resampling or hard-coded FPS assumptions.

Output Layout:
artifacts/
└── event_graph/
    └── shots/
        ├── per_video/
        │   ├── L21_V001.parquet
        │   └── ...
        ├── all_shots.parquet
        ├── video_metadata.parquet
        ├── fps_audit.json
        ├── validation_report.json
        └── failed_videos.json

Usage:
    python scripts/build_transnet_shots_873.py \\
        --video-root /path/to/videos \\
        --output-root artifacts/event_graph/shots \\
        --device cuda \\
        --threshold 0.5 \\
        --resume
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from _bootstrap import PROJECT_ROOT
from src.preprocessing.keyframe_v2.shot_detector import Shot, detect_shots
from src.preprocessing.keyframe_v2.video_metadata import VideoMetadata, probe_video


def natural_video_key(path: Path | str) -> tuple[str, int]:
    stem = Path(path).stem
    try:
        if "_V" in stem:
            prefix, number = stem.rsplit("_V", 1)
            return prefix, int(number)
        return stem, 0
    except Exception:
        return stem, 0


def discover_video_paths(video_roots: list[str | Path], limit: int | None = None) -> list[Path]:
    videos: dict[str, Path] = {}
    for root_item in video_roots:
        root_str = str(root_item).strip()
        if not root_str:
            continue
        candidate = Path(root_str).expanduser()
        if candidate.is_dir():
            matched_roots = [candidate]
        elif any(char in root_str for char in "*?["):
            matched_roots = [Path(p) for p in sorted(glob.glob(root_str)) if Path(p).is_dir()]
        else:
            matched_roots = [PROJECT_ROOT / root_str] if not candidate.is_absolute() else []

        for vroot in matched_roots:
            if not vroot.is_dir():
                continue
            for path in sorted(vroot.rglob("*.mp4"), key=natural_video_key):
                if path.name.startswith("."):
                    continue
                videos.setdefault(path.stem, path)

    discovered = sorted(videos.values(), key=natural_video_key)
    if limit is not None and limit > 0:
        discovered = discovered[:limit]
    return discovered


def validate_video_shots(df: pd.DataFrame, meta: VideoMetadata | None = None) -> tuple[bool, list[str]]:
    errors: list[str] = []
    if df.empty:
        return False, ["DataFrame is empty"]

    required_cols = [
        "video_id",
        "shot_id",
        "start_frame",
        "end_frame",
        "start_sec",
        "end_sec",
        "num_frames",
        "duration_sec",
        "source_fps",
        "frame_count",
        "detector_backend",
    ]
    for col in required_cols:
        if col not in df.columns:
            errors.append(f"Missing required column: {col}")
    if errors:
        return False, errors

    # Check NaNs in critical columns
    critical_cols = ["video_id", "shot_id", "start_frame", "end_frame", "start_sec", "end_sec", "source_fps", "frame_count"]
    for col in critical_cols:
        if df[col].isnull().any():
            errors.append(f"NaN values found in critical column: {col}")

    # Check FPS
    source_fps = float(df["source_fps"].iloc[0])
    if math.isnan(source_fps) or source_fps <= 0:
        errors.append(f"Invalid FPS: {source_fps}")

    frame_count = int(df["frame_count"].iloc[0]) if meta is None else meta.total_frames

    # Check Shot range & order
    prev_end_frame = -1
    prev_start_sec = -0.001
    for idx, row in df.iterrows():
        sf = int(row["start_frame"])
        ef = int(row["end_frame"])
        s_sec = float(row["start_sec"])
        e_sec = float(row["end_sec"])
        dur = float(row["duration_sec"])

        # Check A: Frame range
        if sf < 0:
            errors.append(f"Shot {idx}: start_frame {sf} < 0")
        if ef < sf:
            errors.append(f"Shot {idx}: end_frame {ef} < start_frame {sf}")
        if frame_count > 0 and ef >= frame_count:
            # Allow minor frame count tolerance if off by 1-2 frames from metadata estimate
            if ef >= frame_count + 5:
                errors.append(f"Shot {idx}: end_frame {ef} >= frame_count {frame_count}")

        # Check B: Timestamp range
        if s_sec < 0:
            errors.append(f"Shot {idx}: start_sec {s_sec} < 0")
        if e_sec < s_sec:
            errors.append(f"Shot {idx}: end_sec {e_sec} < start_sec {s_sec}")

        # Check C: Chronological order
        if sf <= prev_end_frame and idx > 0:
            errors.append(f"Shot {idx}: start_frame {sf} overlaps previous end_frame {prev_end_frame}")
        if s_sec < prev_start_sec:
            errors.append(f"Shot {idx}: start_sec {s_sec} < previous start_sec {prev_start_sec}")

        # Check D: Duration
        if dur < 0:
            errors.append(f"Shot {idx}: duration_sec {dur} < 0")

        prev_end_frame = ef
        prev_start_sec = s_sec

    is_valid = len(errors) == 0
    return is_valid, errors


def shots_to_dataframe(shots: list[Shot], meta: VideoMetadata, video_path: Path) -> pd.DataFrame:
    rows = []
    for s in shots:
        rows.append({
            "video_id": meta.video_id,
            "shot_id": int(s.shot_id),
            "start_frame": int(s.start_frame),
            "end_frame": int(s.end_frame),
            "start_sec": float(s.start_timestamp),
            "end_sec": float(s.end_timestamp),
            "num_frames": int(s.num_frames),
            "duration_sec": float(s.duration_sec),
            "source_fps": float(meta.reported_fps),
            "frame_count": int(meta.total_frames),
            "detector_backend": str(s.detector_backend),
            "confidence": float(s.confidence) if s.confidence is not None and not math.isnan(s.confidence) else np.nan,
            "video_path": str(video_path),
        })
    return pd.DataFrame(rows)


def load_transnetv2_model(device: str):
    try:
        from transnetv2_pytorch import TransNetV2

        print(f"[TRANSNETV2] Loading model on device: {device}...")
        model = TransNetV2(device=device)
        model.eval()
        print("[TRANSNETV2] Model loaded successfully.")
        return model
    except Exception as exc:
        print(f"[TRANSNETV2][WARN] Could not load TransNetV2 model: {exc}")
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract TransNetV2 shot boundaries for 873 videos.")
    parser.add_argument("--video-root", action="append", required=True, help="Path or glob to video directories (can specify multiple times).")
    parser.add_argument("--output-root", default="artifacts/event_graph/shots", help="Output directory for shot artifacts.")
    parser.add_argument("--device", default="cuda", help="Device for TransNetV2 model (cuda, cpu, auto).")
    parser.add_argument("--threshold", type=float, default=0.5, help="TransNetV2 scene cut threshold.")
    parser.add_argument("--resume", action="store_true", help="Skip videos with existing valid per_video parquets.")
    parser.add_argument("--force", action="store_true", help="Force re-processing even if per_video parquets exist.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of videos to process for testing.")
    parser.add_argument("--require-transnetv2", action="store_true", help="Raise error if TransNetV2 is unavailable instead of falling back to HistDiff.")
    args = parser.parse_args()

    started_time = time.time()
    output_root = Path(args.output_root).resolve()
    per_video_dir = output_root / "per_video"
    per_video_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(" 🎬 STARTING TRANSNETV2 SHOT DETECTION PIPELINE")
    print("=" * 80)

    video_paths = discover_video_paths(args.video_root, args.limit)
    print(f"Discovered {len(video_paths)} videos for shot detection.")
    if not video_paths:
        print("[ERROR] No .mp4 videos found matching the provided --video-root.")
        return 1

    # Load TransNetV2 model once if possible
    model = load_transnetv2_model(args.device)
    if model is None and args.require_transnetv2:
        raise RuntimeError("TransNetV2 model is required but failed to load.")

    processed_count = 0
    skipped_count = 0
    failed_count = 0
    transnet_backend_count = 0
    histdiff_backend_count = 0

    all_per_video_dfs: list[pd.DataFrame] = []
    video_meta_records: list[dict] = []
    fps_counter: Counter[str] = Counter()
    failed_videos: list[dict] = []
    validation_failures: list[dict] = []

    for idx, video_path in enumerate(video_paths, start=1):
        video_id = video_path.stem
        per_video_parquet = per_video_dir / f"{video_id}.parquet"
        v_start = time.time()

        # Check resume condition
        if args.resume and not args.force and per_video_parquet.is_file():
            try:
                df_existing = pd.read_parquet(per_video_parquet)
                is_valid, val_errs = validate_video_shots(df_existing)
                if is_valid:
                    source_fps = float(df_existing["source_fps"].iloc[0])
                    frame_count = int(df_existing["frame_count"].iloc[0])
                    duration_sec = float(df_existing["end_sec"].iloc[-1]) if not df_existing.empty else 0.0
                    backend = str(df_existing["detector_backend"].iloc[0])

                    fps_str = f"{source_fps:.3f}"
                    fps_counter[fps_str] += 1
                    skipped_count += 1
                    if "transnet" in backend.lower():
                        transnet_backend_count += 1
                    else:
                        histdiff_backend_count += 1

                    all_per_video_dfs.append(df_existing)
                    video_meta_records.append({
                        "video_id": video_id,
                        "video_path": str(video_path),
                        "fps": source_fps,
                        "frame_count": frame_count,
                        "duration_sec": duration_sec,
                        "num_shots": len(df_existing),
                        "backend": backend,
                        "status": "SKIPPED",
                        "error_message": "",
                    })
                    print(f"[{idx:03d}/{len(video_paths):03d}] {video_id} | FPS: {source_fps:.3f} | Frames: {frame_count} | SKIPPED (Resume valid) | Shots: {len(df_existing)}")
                    continue
            except Exception as exc:
                print(f"[{idx:03d}/{len(video_paths):03d}] {video_id} | Resume read failed: {exc}, re-processing...")

        # Process Video
        try:
            meta = probe_video(video_path)
            fps_str = f"{meta.reported_fps:.3f}"
            fps_counter[fps_str] += 1

            cfg = {
                "require_transnetv2": args.require_transnetv2,
                "use_histdiff_only": False if model is not None else True,
                "backend": "transnetv2" if model is not None else "histdiff",
                "transnetv2_device": args.device,
                "transnetv2_threshold": args.threshold,
                "transnetv2_frame_reader": "opencv",
                "model": model,
            }

            shots, warnings = detect_shots(video_path, meta, cfg)
            df_shots = shots_to_dataframe(shots, meta, video_path)

            # Validate generated shots
            is_valid, val_errs = validate_video_shots(df_shots, meta)
            elapsed_v = round(time.time() - v_start, 2)
            backend = df_shots["detector_backend"].iloc[0] if not df_shots.empty else "unknown"

            if "transnet" in backend.lower():
                transnet_backend_count += 1
            else:
                histdiff_backend_count += 1

            if not is_valid:
                print(f"[{idx:03d}/{len(video_paths):03d}] {video_id} | FPS: {meta.reported_fps:.3f} | Frames: {meta.total_frames} | {backend.upper()}: PASS | Shots: {len(shots)} | Time: {elapsed_v}s | Validation: FAIL ({val_errs})")
                validation_failures.append({"video_id": video_id, "errors": val_errs})
            else:
                print(f"[{idx:03d}/{len(video_paths):03d}] {video_id} | FPS: {meta.reported_fps:.3f} | Frames: {meta.total_frames} | {backend.upper()}: PASS | Shots: {len(shots)} | Time: {elapsed_v}s | Validation: PASS")

            # Save per-video parquet
            df_shots.to_parquet(per_video_parquet, index=False)
            all_per_video_dfs.append(df_shots)
            processed_count += 1

            video_meta_records.append({
                "video_id": video_id,
                "video_path": str(video_path),
                "fps": meta.reported_fps,
                "frame_count": meta.total_frames,
                "duration_sec": meta.duration_sec,
                "num_shots": len(df_shots),
                "backend": backend,
                "status": "PASS" if is_valid else "FAIL",
                "error_message": "; ".join(val_errs) if val_errs else "",
            })

        except Exception as exc:
            elapsed_v = round(time.time() - v_start, 2)
            failed_count += 1
            err_msg = str(exc)
            print(f"[{idx:03d}/{len(video_paths):03d}] {video_id} | FAILED in {elapsed_v}s: {err_msg}")
            failed_videos.append({"video_id": video_id, "video_path": str(video_path), "error": err_msg})
            video_meta_records.append({
                "video_id": video_id,
                "video_path": str(video_path),
                "fps": 0.0,
                "frame_count": 0,
                "duration_sec": 0.0,
                "num_shots": 0,
                "backend": "none",
                "status": "FAIL",
                "error_message": err_msg,
            })

    # Save Concatenated All Shots
    if all_per_video_dfs:
        all_shots_df = pd.concat(all_per_video_dfs, ignore_index=True)
        all_shots_path = output_root / "all_shots.parquet"
        all_shots_df.to_parquet(all_shots_path, index=False)
        total_shots = len(all_shots_df)
        shots_per_video = [len(df) for df in all_per_video_dfs]
        mean_shots = float(np.mean(shots_per_video)) if shots_per_video else 0.0
        median_shots = float(np.median(shots_per_video)) if shots_per_video else 0.0
    else:
        all_shots_df = pd.DataFrame()
        total_shots = 0
        mean_shots = 0.0
        median_shots = 0.0

    # Save Video Metadata Parquet
    df_video_meta = pd.DataFrame(video_meta_records)
    df_video_meta.to_parquet(output_root / "video_metadata.parquet", index=False)

    # Save FPS Audit JSON
    fps_audit_data = {
        "total_videos": len(video_paths),
        "fps_distribution": dict(fps_counter.most_common()),
    }
    with open(output_root / "fps_audit.json", "w", encoding="utf-8") as f:
        json.dump(fps_audit_data, f, indent=2, ensure_ascii=False)

    # Save Failed Videos JSON
    with open(output_root / "failed_videos.json", "w", encoding="utf-8") as f:
        json.dump(failed_videos, f, indent=2, ensure_ascii=False)

    # Save Validation Report JSON
    val_report = {
        "total_videos_discovered": len(video_paths),
        "videos_processed": processed_count,
        "videos_skipped": skipped_count,
        "videos_failed": failed_count,
        "transnetv2_count": transnet_backend_count,
        "histdiff_count": histdiff_backend_count,
        "total_shots": total_shots,
        "mean_shots_per_video": round(mean_shots, 2),
        "median_shots_per_video": round(median_shots, 2),
        "validation_pass_count": len(video_paths) - len(validation_failures) - failed_count,
        "validation_fail_count": len(validation_failures),
        "validation_failures": validation_failures,
    }
    with open(output_root / "validation_report.json", "w", encoding="utf-8") as f:
        json.dump(val_report, f, indent=2, ensure_ascii=False)

    elapsed_total = round(time.time() - started_time, 2)

    # Print Summary Report
    print("\n" + "=" * 80)
    print(" SHOT DETECTION SUMMARY")
    print("=" * 80)
    print(f" Total videos discovered : {len(video_paths)}")
    print(f" Videos processed        : {processed_count}")
    print(f" Videos skipped          : {skipped_count}")
    print(f" Videos failed           : {failed_count}")
    print()
    print(f" TransNetV2 videos       : {transnet_backend_count}")
    print(f" HistDiff fallback videos: {histdiff_backend_count}")
    print()
    print(f" Total shots extracted   : {total_shots}")
    print(f" Mean shots/video        : {mean_shots:.2f}")
    print(f" Median shots/video      : {median_shots:.2f}")
    print()
    print(" FPS distribution:")
    for fps_k, count in fps_counter.most_common():
        print(f"   {fps_k} FPS: {count} videos")
    print()
    print(f" Validation PASS         : {val_report['validation_pass_count']}")
    print(f" Validation FAIL         : {val_report['validation_fail_count']}")
    print(f" Total Pipeline Time     : {elapsed_total}s")
    print()
    print(" Output Artifacts:")
    print(f"   - {output_root / 'all_shots.parquet'}")
    print(f"   - {output_root / 'video_metadata.parquet'}")
    print(f"   - {output_root / 'fps_audit.json'}")
    print(f"   - {output_root / 'validation_report.json'}")
    print(f"   - {output_root / 'failed_videos.json'}")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
