#!/usr/bin/env python3
"""Offline Temporal-Distance Diagnostic for 29 KIS Video-Only Queries.

Evaluates the exact distance from Top100 candidate frames to GT interval (assuming ~30 fps):
  Bucket A: <= 30 frames   (<= 1.0s - Near Miss, High ROI)
  Bucket B: 31..90 frames  (1.0s..3.0s - Refinement Radius Extension)
  Bucket C: 91..300 frames (3.0s..10.0s - Anchor Sparsity)
  Bucket D: > 300 frames   (> 10.0s - Distant Scene Keyframe)
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SYSTEM_TAI_SRC = REPO_ROOT / "systems" / "system_tai" / "src"
if str(SYSTEM_TAI_SRC) not in sys.path:
    sys.path.insert(0, str(SYSTEM_TAI_SRC))

GT_PATH = REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "l21_150_diagnostic" / "kis_dev_gt.json"

POSSIBLE_OUTPUT_ROOTS = [
    Path("/kaggle/working/output/kis_full38/requests"),
    Path("/kaggle/working/output/kis_full38"),
    REPO_ROOT / "scratch" / "kis_full38" / "requests",
    REPO_ROOT / "scratch" / "kis_full38",
]


def safe_request_directory_name(request_id: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", request_id).strip("._-") or "request"
    digest = hashlib.sha256(request_id.encode("utf-8")).hexdigest()[:8]
    return f"{normalized}-{digest}"


def distance_to_interval(frame: int, start: int, end: int) -> int:
    if start <= frame <= end:
        return 0
    if frame < start:
        return start - frame
    return frame - end


def find_top100_file(output_root: Path, qid: str) -> Path | None:
    # 1. Direct safe_request_directory_name
    candidates = [
        output_root / safe_request_directory_name(f"kis-{qid}") / "refined_top100.jsonl",
        output_root / safe_request_directory_name(f"kis-{qid}") / "top100.jsonl",
        output_root / f"kis-{qid}" / "refined_top100.jsonl",
        output_root / f"kis-{qid}" / "top100.jsonl",
    ]
    for c in candidates:
        if c.exists():
            return c

    # 2. Pattern glob search
    for pattern in [f"*{qid}*/refined_top100.jsonl", f"*{qid}*/top100.jsonl"]:
        matches = list(output_root.glob(pattern))
        if matches:
            return matches[0]

    return None


def audit_temporal_gaps() -> None:
    print("=" * 145, flush=True)
    print("OFFLINE TEMPORAL-DISTANCE DIAGNOSTIC: 29 KIS VIDEO-ONLY QUERIES (~30 fps)", flush=True)
    print("=" * 145, flush=True)

    gt_data = json.loads(GT_PATH.read_text(encoding="utf-8"))
    gt_map = {q["query_id"]: q for q in gt_data["queries"]}

    output_root = None
    for r in POSSIBLE_OUTPUT_ROOTS:
        if r.exists():
            output_root = r
            break

    if output_root is None:
        print("❌ Error: Output root directory not found. Please ensure Full-38 output exists in /kaggle/working/output/kis_full38.")
        return

    print(f"• Ingested Output Directory: {output_root}", flush=True)

    bucket_a = []  # <= 30
    bucket_b = []  # 31..90
    bucket_c = []  # 91..300
    bucket_d = []  # > 300

    results = []

    for qid, gt in gt_map.items():
        target_vid = gt["video_id"]
        start_f = gt["start_frame"]
        end_f = gt["end_frame"]

        top100_file = find_top100_file(output_root, qid)
        if top100_file is None:
            continue

        preds = [json.loads(l) for l in top100_file.read_text(encoding="utf-8").splitlines() if l.strip()]

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
            bucket_str = "Bucket A (<=30f, <=1.0s)"
            bucket_a.append(qid)
        elif min_dist <= 90:
            bucket_str = "Bucket B (31-90f, 1.0-3.0s)"
            bucket_b.append(qid)
        elif min_dist <= 300:
            bucket_str = "Bucket C (91-300f, 3.0-10.0s)"
            bucket_c.append(qid)
        else:
            bucket_str = "Bucket D (>300f, >10.0s)"
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
    print("DISTRIBUTION SUMMARY OF 29 VIDEO-ONLY QUERIES (~30 fps)", flush=True)
    print("=" * 145, flush=True)
    total_analyzed = len(results)
    if total_analyzed == 0:
        print("• Total Video-Only Queries Analyzed : 0 / 29 (No output files found)", flush=True)
    else:
        print(f"• Total Video-Only Queries Analyzed : {total_analyzed} / 29", flush=True)
        print(f"• Bucket A (<= 30 frames / <= 1.0s) : {len(bucket_a):2d} ({len(bucket_a)/total_analyzed*100:5.1f}%) -> {bucket_a}", flush=True)
        print(f"• Bucket B (31..90 frames / 1.0-3.0s): {len(bucket_b):2d} ({len(bucket_b)/total_analyzed*100:5.1f}%) -> {bucket_b}", flush=True)
        print(f"• Bucket C (91..300 frames / 3-10.0s): {len(bucket_c):2d} ({len(bucket_c)/total_analyzed*100:5.1f}%) -> {bucket_c}", flush=True)
        print(f"• Bucket D (> 300 frames / > 10.0s) : {len(bucket_d):2d} ({len(bucket_d)/total_analyzed*100:5.1f}%) -> {bucket_d}", flush=True)
    print("=" * 145, flush=True)


if __name__ == "__main__":
    audit_temporal_gaps()
