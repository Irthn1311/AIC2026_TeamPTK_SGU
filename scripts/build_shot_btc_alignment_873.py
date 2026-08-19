"""
Stage 2 — Shot ↔ BTC Keyframe Temporal Alignment Pipeline (AI Challenge 2026)
=============================================================================
Aligns TransNetV2 shots with BTC Keyframes using canonical timestamp_sec
coordinates without FPS conversion, and performs time-frame consistency audit.

Output Layout:
artifacts/
└── event_graph/
    └── alignment/
        ├── per_video/
        │   ├── L21_V001.parquet
        │   └── ...
        ├── shot_keyframe_alignment.parquet
        ├── unassigned_keyframes.parquet
        ├── shots_without_keyframes.parquet
        ├── alignment_disagreements.parquet
        └── alignment_report.json

Usage:
    python scripts/build_shot_btc_alignment_873.py \
        --shots artifacts/event_graph/shots/all_shots.parquet \
        --btc-map artifacts/keyframe_btc_full/indexes/keyframe_btc_global_map.parquet \
        --output-root artifacts/event_graph/alignment \
        --num-workers 4 \
        --resume
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from _bootstrap import PROJECT_ROOT


EPS = 1e-3


def normalize_btc_map(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "video_id" not in df.columns:
        raise ValueError("BTC map missing video_id column")

    # frame_idx
    if "frame_idx" in df.columns:
        df["keyframe_frame_idx"] = df["frame_idx"].astype(int)
    elif "actual_frame_id" in df.columns:
        df["keyframe_frame_idx"] = df["actual_frame_id"].astype(int)
    else:
        raise ValueError("BTC map missing frame_idx or actual_frame_id")

    # timestamp_sec
    if "timestamp_sec" in df.columns:
        df["keyframe_timestamp_sec"] = df["timestamp_sec"].astype(float)
    elif "pts_time" in df.columns:
        df["keyframe_timestamp_sec"] = df["pts_time"].astype(float)
    elif "timestamp_ms" in df.columns:
        df["keyframe_timestamp_sec"] = (df["timestamp_ms"] / 1000.0).astype(float)
    else:
        raise ValueError("BTC map missing timestamp_sec, pts_time, or timestamp_ms")

    # keyframe_id
    if "keyframe_id" in df.columns:
        df["keyframe_id"] = df["keyframe_id"].astype(str)
    elif "global_v2_id" in df.columns:
        df["keyframe_id"] = df["global_v2_id"].astype(str)
    elif "global_id" in df.columns:
        df["keyframe_id"] = df["global_id"].astype(str)
    elif "keyframe_name" in df.columns:
        df["keyframe_id"] = df["keyframe_name"].astype(str)
    else:
        df["keyframe_id"] = df.index.astype(str)

    return df


def align_video_keyframes(
    video_id: str,
    shots_df: pd.DataFrame,
    keyframes_df: pd.DataFrame,
    eps: float = EPS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    aligned_rows = []
    unassigned_rows = []
    disagreement_rows = []

    # Sort shots and keyframes
    shots_df = shots_df.sort_values("start_frame").reset_index(drop=True)
    keyframes_df = keyframes_df.sort_values("keyframe_timestamp_sec").reset_index(drop=True)

    assigned_shots_set: set[int] = set()
    assigned_keyframe_set: set[str] = set()

    for _, k_row in keyframes_df.iterrows():
        k_id = str(k_row["keyframe_id"])
        k_fid = int(k_row["keyframe_frame_idx"])
        k_ts = float(k_row["keyframe_timestamp_sec"])

        # Candidate shots by Time (with EPS margin)
        time_candidates = []
        for _, s_row in shots_df.iterrows():
            s_id = int(s_row["shot_id"])
            sf_sec = float(s_row["start_sec"])
            ef_sec = float(s_row["end_sec"])
            if sf_sec - eps <= k_ts <= ef_sec + eps:
                time_candidates.append(s_row)

        # Candidate shots by Frame
        frame_candidates = []
        for _, s_row in shots_df.iterrows():
            sf = int(s_row["start_frame"])
            ef = int(s_row["end_frame"])
            if sf <= k_fid <= ef:
                frame_candidates.append(s_row)

        has_time = len(time_candidates) > 0
        has_frame = len(frame_candidates) > 0

        best_shot = None
        if has_time:
            # 1. Prefer shots strictly containing timestamp
            strict_candidates = [
                s for s in time_candidates if float(s["start_sec"]) <= k_ts <= float(s["end_sec"])
            ]
            eval_list = strict_candidates if strict_candidates else time_candidates

            # 2. Pick shot with minimum distance to center
            best_shot = min(
                eval_list,
                key=lambda s: abs(k_ts - ((float(s["start_sec"]) + float(s["end_sec"])) / 2.0)),
            )

        # Check TIME ↔ FRAME Agreement
        hit_by_time = has_time
        hit_by_frame = False
        if best_shot is not None:
            hit_by_frame = int(best_shot["shot_id"]) in [int(s["shot_id"]) for s in frame_candidates]

        if hit_by_time and hit_by_frame:
            status = "AGREE"
        elif hit_by_time or hit_by_frame:
            status = "DISAGREE"
        else:
            status = "UNALIGNED"

        if best_shot is not None:
            s_id = int(best_shot["shot_id"])
            sf = int(best_shot["start_frame"])
            ef = int(best_shot["end_frame"])
            sf_sec = float(best_shot["start_sec"])
            ef_sec = float(best_shot["end_sec"])
            fps = float(best_shot.get("source_fps", 0.0))

            duration = max(ef_sec - sf_sec, 1e-6)
            rel_pos = (k_ts - sf_sec) / duration
            center_dist = abs(k_ts - ((sf_sec + ef_sec) / 2.0))

            record = {
                "video_id": video_id,
                "shot_id": s_id,
                "shot_start_frame": sf,
                "shot_end_frame": ef,
                "shot_start_sec": sf_sec,
                "shot_end_sec": ef_sec,
                "source_fps": fps,
                "keyframe_id": k_id,
                "keyframe_frame_idx": k_fid,
                "keyframe_timestamp_sec": k_ts,
                "relative_position": float(rel_pos),
                "temporal_distance_to_center_sec": float(center_dist),
                "hit_by_time": bool(hit_by_time),
                "hit_by_frame": bool(hit_by_frame),
                "alignment_status": status,
            }
            aligned_rows.append(record)
            assigned_shots_set.add(s_id)
            assigned_keyframe_set.add(k_id)

            if not hit_by_frame:
                disagreement_rows.append(record)
        else:
            # Unassigned Keyframe
            unassigned_record = {
                "video_id": video_id,
                "keyframe_id": k_id,
                "keyframe_frame_idx": k_fid,
                "keyframe_timestamp_sec": k_ts,
                "hit_by_time": bool(hit_by_time),
                "hit_by_frame": bool(hit_by_frame),
                "reason": "no_matching_shot_by_time",
            }
            unassigned_rows.append(unassigned_record)
            if hit_by_frame:
                disagreement_rows.append({
                    "video_id": video_id,
                    "shot_id": int(frame_candidates[0]["shot_id"]),
                    "shot_start_frame": int(frame_candidates[0]["start_frame"]),
                    "shot_end_frame": int(frame_candidates[0]["end_frame"]),
                    "shot_start_sec": float(frame_candidates[0]["start_sec"]),
                    "shot_end_sec": float(frame_candidates[0]["end_sec"]),
                    "source_fps": float(frame_candidates[0].get("source_fps", 0.0)),
                    "keyframe_id": k_id,
                    "keyframe_frame_idx": k_fid,
                    "keyframe_timestamp_sec": k_ts,
                    "relative_position": float((k_ts - float(frame_candidates[0]["start_sec"])) / max(float(frame_candidates[0]["end_sec"]) - float(frame_candidates[0]["start_sec"]), 1e-6)),
                    "temporal_distance_to_center_sec": float(abs(k_ts - (float(frame_candidates[0]["start_sec"]) + float(frame_candidates[0]["end_sec"])) / 2.0)),
                    "hit_by_time": False,
                    "hit_by_frame": True,
                    "alignment_status": "DISAGREE",
                })

    # Shots without Keyframes
    shots_without_kf_rows = []
    for _, s_row in shots_df.iterrows():
        s_id = int(s_row["shot_id"])
        if s_id not in assigned_shots_set:
            shots_without_kf_rows.append({
                "video_id": video_id,
                "shot_id": s_id,
                "shot_start_frame": int(s_row["start_frame"]),
                "shot_end_frame": int(s_row["end_frame"]),
                "shot_start_sec": float(s_row["start_sec"]),
                "shot_end_sec": float(s_row["end_sec"]),
                "source_fps": float(s_row.get("source_fps", 0.0)),
            })

    df_aligned = pd.DataFrame(aligned_rows)
    df_unassigned = pd.DataFrame(unassigned_rows)
    df_shots_no_kf = pd.DataFrame(shots_without_kf_rows)
    df_disagreements = pd.DataFrame(disagreement_rows)

    stats = {
        "video_id": video_id,
        "total_shots": len(shots_df),
        "total_keyframes": len(keyframes_df),
        "aligned_keyframes": len(df_aligned),
        "unassigned_keyframes": len(df_unassigned),
        "shots_without_keyframes": len(df_shots_no_kf),
        "time_frame_agree": int((df_aligned["alignment_status"] == "AGREE").sum()) if not df_aligned.empty else 0,
        "time_frame_disagree": len(df_disagreements),
    }

    return df_aligned, df_unassigned, df_shots_no_kf, df_disagreements, stats


def process_single_video_worker(args_tuple) -> tuple[str, dict, dict]:
    video_id, shots_json, keyframes_json, per_video_dir_str, force = args_tuple
    per_video_dir = Path(per_video_dir_str)
    per_video_parquet = per_video_dir / f"{video_id}.parquet"

    shots_df = pd.read_json(shots_json)
    keyframes_df = pd.read_json(keyframes_json)

    if per_video_parquet.is_file() and not force:
        try:
            df_aligned = pd.read_parquet(per_video_parquet)
            # Reconstruct basic stats
            assigned_shots = set(df_aligned["shot_id"].unique()) if not df_aligned.empty else set()
            total_shots = len(shots_df)
            no_kf_count = total_shots - len(assigned_shots)
            agree_count = int((df_aligned["alignment_status"] == "AGREE").sum()) if not df_aligned.empty else 0
            disagree_count = int((df_aligned["alignment_status"] == "DISAGREE").sum()) if not df_aligned.empty else 0
            unaligned_count = len(keyframes_df) - len(df_aligned)

            stats = {
                "video_id": video_id,
                "total_shots": total_shots,
                "total_keyframes": len(keyframes_df),
                "aligned_keyframes": len(df_aligned),
                "unassigned_keyframes": max(0, unaligned_count),
                "shots_without_keyframes": max(0, no_kf_count),
                "time_frame_agree": agree_count,
                "time_frame_disagree": disagree_count,
                "status": "SKIPPED",
            }
            return video_id, stats, {"parquet_path": str(per_video_parquet)}
        except Exception:
            pass

    df_aligned, df_unassigned, df_shots_no_kf, df_disagreements, stats = align_video_keyframes(
        video_id, shots_df, keyframes_df
    )

    df_aligned.to_parquet(per_video_parquet, index=False)
    stats["status"] = "PASS"

    # Write temporary outputs for aggregator
    temp_dir = per_video_dir.parent / "_temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    if not df_unassigned.empty:
        df_unassigned.to_parquet(temp_dir / f"unassigned_{video_id}.parquet", index=False)
    if not df_shots_no_kf.empty:
        df_shots_no_kf.to_parquet(temp_dir / f"no_kf_{video_id}.parquet", index=False)
    if not df_disagreements.empty:
        df_disagreements.to_parquet(temp_dir / f"disagree_{video_id}.parquet", index=False)

    return video_id, stats, {"parquet_path": str(per_video_parquet)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 2 — Shot ↔ BTC Keyframe Temporal Alignment.")
    parser.add_argument("--shots", default="artifacts/event_graph/shots/all_shots.parquet", help="Path to all_shots.parquet.")
    parser.add_argument("--btc-map", default="artifacts/keyframe_btc_full/indexes/keyframe_btc_global_map.parquet", help="Path to BTC keyframe global map.")
    parser.add_argument("--output-root", default="artifacts/event_graph/alignment", help="Output directory for alignment artifacts.")
    parser.add_argument("--num-workers", type=int, default=max(1, (os.cpu_count() or 4)), help="Number of parallel CPU worker threads/processes.")
    parser.add_argument("--resume", action="store_true", help="Resume execution, skipping existing per_video alignment parquets.")
    parser.add_argument("--force", action="store_true", help="Force re-aligning all videos.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of videos to process for testing.")
    args = parser.parse_args()

    started_time = time.time()
    output_root = Path(args.output_root).resolve()
    per_video_dir = output_root / "per_video"
    temp_dir = output_root / "_temp"
    per_video_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    shots_path = Path(args.shots).resolve()
    btc_map_input = Path(args.btc_map).resolve()
    if btc_map_input.is_dir():
        candidates = [
            btc_map_input / "artifacts" / "keyframe_btc_full" / "indexes" / "keyframe_btc_global_map.parquet",
            btc_map_input / "artifacts" / "keyframe_btc_full" / "indexes" / "keyframe_btc_global_map.csv",
            btc_map_input / "keyframe_btc_full" / "indexes" / "keyframe_btc_global_map.parquet",
            btc_map_input / "keyframe_btc_full" / "indexes" / "keyframe_btc_global_map.csv",
            btc_map_input / "indexes" / "keyframe_btc_global_map.parquet",
            btc_map_input / "indexes" / "keyframe_btc_global_map.csv",
        ]
        btc_map_path = None
        for c in candidates:
            if c.is_file():
                btc_map_path = c
                break
        if btc_map_path is None:
            matches = sorted(btc_map_input.glob("**/keyframe_btc_global_map.*"))
            if matches:
                btc_map_path = matches[0]
        if btc_map_path is None:
            raise FileNotFoundError(f"Could not locate keyframe_btc_global_map inside directory: {btc_map_input}")
    else:
        btc_map_path = btc_map_input

    print("=" * 80)
    print(" 🎯 STAGE 2: SHOT ↔ BTC KEYFRAME TEMPORAL ALIGNMENT PIPELINE")
    print("=" * 80)
    print(f" Shots Path     : {shots_path}")
    print(f" BTC Map Path   : {btc_map_path}")
    print(f" Output Root    : {output_root}")
    print(f" CPU Workers    : {args.num_workers}")
    print("=" * 80)

    if not shots_path.is_file():
        raise FileNotFoundError(f"Shots file not found: {shots_path}")
    if not btc_map_path.is_file():
        raise FileNotFoundError(f"BTC global map file not found: {btc_map_path}")

    print("[1/4] Loading input datasets...")
    all_shots_df = pd.read_parquet(shots_path)
    raw_btc_df = pd.read_parquet(btc_map_path) if btc_map_path.suffix == ".parquet" else pd.read_csv(btc_map_path)
    all_btc_df = normalize_btc_map(raw_btc_df)

    shot_videos = set(all_shots_df["video_id"].unique())
    btc_videos = set(all_btc_df["video_id"].unique())
    common_videos = sorted(shot_videos.intersection(btc_videos))

    print(f" Found {len(shot_videos)} videos in shots data, {len(btc_videos)} in BTC keyframes.")
    print(f" Common videos to align: {len(common_videos)}")

    if args.limit and args.limit > 0:
        common_videos = common_videos[: args.limit]
        print(f" [LIMIT] Processing limited to first {len(common_videos)} videos.")

    # Prepare worker arguments
    print(f"[2/4] Dispatching alignment across {args.num_workers} CPU workers...")
    worker_tasks = []
    for vid in common_videos:
        s_df = all_shots_df[all_shots_df["video_id"] == vid]
        k_df = all_btc_df[all_btc_df["video_id"] == vid]
        worker_tasks.append((
            vid,
            s_df.to_json(),
            k_df.to_json(),
            str(per_video_dir),
            args.force if not args.resume else False,
        ))

    video_stats_list = []
    processed_count = 0
    skipped_count = 0

    if args.num_workers > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {executor.submit(process_single_video_worker, task): task[0] for task in worker_tasks}
            for idx, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                vid, stats, res_info = future.result()
                video_stats_list.append(stats)
                if stats.get("status") == "SKIPPED":
                    skipped_count += 1
                else:
                    processed_count += 1
                if idx % 50 == 0 or idx == len(worker_tasks):
                    print(f" [{idx:03d}/{len(worker_tasks):03d}] Aligned {vid} | Status: {stats.get('status')} | Aligned KF: {stats['aligned_keyframes']}/{stats['total_keyframes']}")
    else:
        for idx, task in enumerate(worker_tasks, start=1):
            vid, stats, res_info = process_single_video_worker(task)
            video_stats_list.append(stats)
            if stats.get("status") == "SKIPPED":
                skipped_count += 1
            else:
                processed_count += 1
            if idx % 50 == 0 or idx == len(worker_tasks):
                print(f" [{idx:03d}/{len(worker_tasks):03d}] Aligned {vid} | Status: {stats.get('status')} | Aligned KF: {stats['aligned_keyframes']}/{stats['total_keyframes']}")

    print("[3/4] Aggregating alignment results into global parquets...")
    # Gather per-video aligned parquets
    aligned_parquets = sorted(per_video_dir.glob("*.parquet"))
    aligned_dfs = [pd.read_parquet(p) for p in aligned_parquets if p.stat().st_size > 0]
    global_aligned_df = pd.concat(aligned_dfs, ignore_index=True) if aligned_dfs else pd.DataFrame()
    global_aligned_df.to_parquet(output_root / "shot_keyframe_alignment.parquet", index=False)

    # Gather unassigned keyframes
    unassigned_files = sorted(temp_dir.glob("unassigned_*.parquet"))
    unassigned_dfs = [pd.read_parquet(p) for p in unassigned_files if p.stat().st_size > 0]
    global_unassigned_df = pd.concat(unassigned_dfs, ignore_index=True) if unassigned_dfs else pd.DataFrame(columns=["video_id", "keyframe_id", "keyframe_frame_idx", "keyframe_timestamp_sec", "hit_by_time", "hit_by_frame", "reason"])
    global_unassigned_df.to_parquet(output_root / "unassigned_keyframes.parquet", index=False)

    # Gather shots without keyframes
    no_kf_files = sorted(temp_dir.glob("no_kf_*.parquet"))
    no_kf_dfs = [pd.read_parquet(p) for p in no_kf_files if p.stat().st_size > 0]
    global_no_kf_df = pd.concat(no_kf_dfs, ignore_index=True) if no_kf_dfs else pd.DataFrame(columns=["video_id", "shot_id", "shot_start_frame", "shot_end_frame", "shot_start_sec", "shot_end_sec", "source_fps"])
    global_no_kf_df.to_parquet(output_root / "shots_without_keyframes.parquet", index=False)

    # Gather disagreements
    disagree_files = sorted(temp_dir.glob("disagree_*.parquet"))
    disagree_dfs = [pd.read_parquet(p) for p in disagree_files if p.stat().st_size > 0]
    global_disagree_df = pd.concat(disagree_dfs, ignore_index=True) if disagree_dfs else pd.DataFrame()
    global_disagree_df.to_parquet(output_root / "alignment_disagreements.parquet", index=False)

    print("[4/4] Performing consistency audit & writing report...")
    total_shots = len(all_shots_df[all_shots_df["video_id"].isin(common_videos)])
    total_keyframes = len(all_btc_df[all_btc_df["video_id"].isin(common_videos)])
    aligned_keyframes = len(global_aligned_df)
    unaligned_keyframes = len(global_unassigned_df)
    shots_without_kf = len(global_no_kf_df)

    time_frame_agree = int((global_aligned_df["alignment_status"] == "AGREE").sum()) if not global_aligned_df.empty else 0
    time_frame_disagree = len(global_disagree_df)
    total_eval = time_frame_agree + time_frame_disagree
    agreement_rate = float(time_frame_agree / max(1, total_eval))

    # Duplicate assignments check
    dup_count = int(global_aligned_df.duplicated(subset=["video_id", "keyframe_id"]).sum()) if not global_aligned_df.empty else 0

    disagreement_videos = sorted(global_disagree_df["video_id"].unique().tolist()) if not global_disagree_df.empty else []

    # Calculate keyframe per shot metrics
    if not global_aligned_df.empty:
        kf_per_shot_counts = global_aligned_df.groupby(["video_id", "shot_id"]).size()
        mean_kf_per_shot = float(kf_per_shot_counts.mean())
        median_kf_per_shot = float(kf_per_shot_counts.median())
        max_kf_per_shot = int(kf_per_shot_counts.max())
    else:
        mean_kf_per_shot = 0.0
        median_kf_per_shot = 0.0
        max_kf_per_shot = 0

    report_data = {
        "num_videos": len(common_videos),
        "total_shots": total_shots,
        "total_keyframes": total_keyframes,
        "aligned_keyframes": aligned_keyframes,
        "unaligned_keyframes": unaligned_keyframes,
        "shots_without_keyframes": shots_without_kf,
        "time_frame_agree": time_frame_agree,
        "time_frame_disagree": time_frame_disagree,
        "agreement_rate": round(agreement_rate, 4),
        "duplicate_assignments": dup_count,
        "videos_with_disagreement": disagreement_videos,
        "mean_keyframes_per_shot": round(mean_kf_per_shot, 3),
        "median_keyframes_per_shot": round(median_kf_per_shot, 3),
        "max_keyframes_per_shot": max_kf_per_shot,
    }

    report_path = output_root / "alignment_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)

    elapsed_total = round(time.time() - started_time, 2)

    print("\n" + "=" * 80)
    print(" SHOT ↔ BTC KEYFRAME ALIGNMENT SUMMARY")
    print("=" * 80)
    print(f" Total Videos Processed : {len(common_videos)} (Processed: {processed_count}, Skipped: {skipped_count})")
    print(f" Total Shots            : {total_shots}")
    print(f" Total BTC Keyframes    : {total_keyframes}")
    print(f" Aligned Keyframes      : {aligned_keyframes} ({aligned_keyframes/max(1, total_keyframes)*100:.2f}%)")
    print(f" Unaligned Keyframes    : {unaligned_keyframes}")
    print(f" Shots w/o Keyframes    : {shots_without_kf}")
    print()
    print(f" TIME ↔ FRAME Agreement: {time_frame_agree} AGREE | {time_frame_disagree} DISAGREE ({agreement_rate*100:.2f}% Agreement Rate)")
    print(f" Duplicate Assignments  : {dup_count}")
    print(f" Disagreement Videos    : {len(disagreement_videos)}")
    print()
    print(f" Keyframes/Shot         : Mean {mean_kf_per_shot:.2f} | Median {median_kf_per_shot:.2f} | Max {max_kf_per_shot}")
    print(f" Total Runtime          : {elapsed_total}s")
    print()
    print(" Output Artifacts Created:")
    print(f"   - {output_root / 'shot_keyframe_alignment.parquet'}")
    print(f"   - {output_root / 'unassigned_keyframes.parquet'}")
    print(f"   - {output_root / 'shots_without_keyframes.parquet'}")
    print(f"   - {output_root / 'alignment_disagreements.parquet'}")
    print(f"   - {report_path}")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
