#!/usr/bin/env python3
"""Offline Temporal-Distance Diagnostic for 29 KIS Video-Only Queries.

Evaluates the exact distance from Top100 candidate frames to GT interval:
  Bucket A: <= 30 frames   (<= 1.2s - Near Miss, High ROI)
  Bucket B: 31..90 frames  (1.2s..3.6s - Refinement Radius Extension)
  Bucket C: 91..300 frames (3.6s..12.0s - Anchor Sparsity)
  Bucket D: > 300 frames   (> 12.0s - Distant Scene Keyframe)
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
GT_PATH = REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "l21_150_diagnostic" / "kis_dev_gt.json"
OUTPUT_ROOT = Path("/kaggle/working/output/kis_full38/requests") if Path("/kaggle/working").exists() else REPO_ROOT / "scratch" / "kis_full38" / "requests"


def distance_to_interval(frame: int, start: int, end: int) -> int:
    if start <= frame <= end:
        return 0
    if frame < start:
        return start - frame
    return frame - end


def audit_temporal_gaps() -> None:
    print("=" * 145, flush=True)
    print("OFFLINE TEMPORAL-DISTANCE DIAGNOSTIC: 29 KIS VIDEO-ONLY QUERIES", flush=True)
    print("=" * 145, flush=True)

    gt_data = json.loads(GT_PATH.read_text(encoding="utf-8"))
    gt_map = {q["query_id"]: q for q in gt_data["queries"]}

    bucket_a = []  # <= 30
    bucket_b = []  # 31..90
    bucket_c = []  # 91..300
    bucket_d = []  # > 300

    results = []

    for qid, gt in gt_map.items():
        target_vid = gt["video_id"]
        start_f = gt["start_frame"]
        end_f = gt["end_frame"]

        # Search for top100 file in output root
        query_dir = OUTPUT_ROOT / f"kis-{qid}"
        top100_jsonl = query_dir / "refined_top100.jsonl"
        if not top100_jsonl.exists():
            top100_jsonl = query_dir / "top100.jsonl"

        if not top100_jsonl.exists():
            continue

        preds = [json.loads(l) for l in top100_jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]

        # Filter target video candidates
        target_candidates = [p for p in preds if p["video_id"] == target_vid]
        if not target_candidates:
            continue  # ABSENT query

        # Check if already strict hit
        is_strict_hit = any(start_f <= p["frame_id"] <= end_f for p in target_candidates)
        if is_strict_hit:
            continue  # Skip strict hits, focus on VIDEO_ONLY

        first_vid_rank = target_candidates[0]["rank"]
        vid_count = len(target_candidates)

        # Find candidate with minimum distance to GT interval
        best_cand = min(target_candidates, key=lambda p: distance_to_interval(p["frame_id"], start_f, end_f))
        min_dist = distance_to_interval(best_cand["frame_id"], start_f, end_f)
        nearest_frame = best_cand["frame_id"]
        nearest_rank = best_cand["rank"]

        # Assign bucket
        if min_dist <= 30:
            bucket_str = "Bucket A (<=30f)"
            bucket_a.append(qid)
        elif min_dist <= 90:
            bucket_str = "Bucket B (31-90f)"
            bucket_b.append(qid)
        elif min_dist <= 300:
            bucket_str = "Bucket C (91-300f)"
            bucket_c.append(qid)
        else:
            bucket_str = "Bucket D (>300f)"
            bucket_d.append(qid)

        results.append({
            "qid": qid,
            "target_video": target_vid,
            "gt_interval": f"[{start_f}..{end_f}]",
            "first_vid_rank": first_vid_rank,
            "nearest_frame": nearest_frame,
            "nearest_rank": nearest_rank,
            "distance": min_dist,
            "vid_count": vid_count,
            "bucket": bucket_str,
        })

    print(f"{'QID':<8} | {'Target Video':<12} | {'GT Interval':<16} | {'1st Vid Rank':<12} | {'Nearest Frame':<13} | {'Nearest Rank':<12} | {'Distance (f)':<12} | {'Vid Rows':<8} | {'Bucket'}", flush=True)
    print("-" * 145, flush=True)
    for r in sorted(results, key=lambda x: x["distance"]):
        print(
            f"{r['qid']:<8} | {r['target_video']:<12} | {r['gt_interval']:<16} | @{r['first_vid_rank']:<11} | "
            f"{r['nearest_frame']:<13} | @{r['nearest_rank']:<11} | {r['distance']:<12} | {r['vid_count']:<8} | {r['bucket']}",
            flush=True,
        )

    print("\n" + "=" * 145, flush=True)
    print("DISTRIBUTION SUMMARY OF 29 VIDEO-ONLY QUERIES", flush=True)
    print("=" * 145, flush=True)
    total_analyzed = len(results)
    print(f"• Total Video-Only Queries Analyzed : {total_analyzed} / 29", flush=True)
    print(f"• Bucket A (<= 30 frames / <= 1.2s) : {len(bucket_a):2d} ({len(bucket_a)/total_analyzed*100:5.1f}%) -> {bucket_a}", flush=True)
    print(f"• Bucket B (31..90 frames / 1.2-3.6s): {len(bucket_b):2d} ({len(bucket_b)/total_analyzed*100:5.1f}%) -> {bucket_b}", flush=True)
    print(f"• Bucket C (91..300 frames / 3.6-12s): {len(bucket_c):2d} ({len(bucket_c)/total_analyzed*100:5.1f}%) -> {bucket_c}", flush=True)
    print(f"• Bucket D (> 300 frames / > 12.0s) : {len(bucket_d):2d} ({len(bucket_d)/total_analyzed*100:5.1f}%) -> {bucket_d}", flush=True)
    print("=" * 145, flush=True)


if __name__ == "__main__":
    audit_temporal_gaps()
