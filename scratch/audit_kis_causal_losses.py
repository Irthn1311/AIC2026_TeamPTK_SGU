#!/usr/bin/env python3
"""Offline Causal Loss Audit for 14 KIS Bucket A+B Near-Miss Queries.

Traces each miss through:
  Anchor -> Coarse Window -> Fine Window -> Scorer Selection -> Materialized Top100

Causal Loss Categories:
  - ANCHOR_PATH_MISS: Target video retrieved, but rank > RefineTopN (not selected for raw refinement)
  - COARSE_WINDOW_MISS: Target video anchor refined, but GT interval fell outside coarse window [anchor ± 150f]
  - FINE_WINDOW_MISS: Coarse best chosen, but GT interval fell outside fine search [coarse_best ± 30f]
  - SCORER_SELECTION_MISS: GT frame fell inside fine search window, but scorer selected a non-GT frame
  - STRIDE_SPARSITY_MISS: Coarse stride (15f) stepped over narrow GT interval
  - MATERIALIZATION_MISS: GT frame refined, but pushed out of Top100 during fusion/dedup
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
GT_PATH = REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "l21_150_diagnostic" / "kis_dev_gt.json"

POSSIBLE_OUTPUT_ROOTS = [
    Path("/kaggle/working/output/kis_full38/requests"),
    Path("/kaggle/working/output/kis_full38"),
    REPO_ROOT / "scratch" / "kis_full38" / "requests",
    REPO_ROOT / "scratch" / "kis_full38",
]

TARGET_QUERIES_A_B = [
    "KIS-26", "KIS-45", "KIS-18", "KIS-03", "KIS-47", "KIS-07", "KIS-27", "KIS-50",  # Bucket A
    "KIS-32", "KIS-42", "KIS-21", "KIS-08", "KIS-11", "KIS-12",                      # Bucket B
]


def safe_request_directory_name(request_id: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", request_id).strip("._-") or "request"
    digest = hashlib.sha256(request_id.encode("utf-8")).hexdigest()[:8]
    return f"{normalized}-{digest}"


def find_query_dir(output_root: Path, qid: str) -> Path | None:
    for cand in [
        output_root / safe_request_directory_name(f"kis-{qid}"),
        output_root / f"kis-{qid}",
    ]:
        if cand.exists() and cand.is_dir():
            return cand

    matches = list(output_root.glob(f"*{qid}*"))
    if matches and matches[0].is_dir():
        return matches[0]

    return None


def audit_causal_losses() -> None:
    print("=" * 150, flush=True)
    print("OFFLINE CAUSAL LOSS AUDIT: 14 KIS BUCKET A+B NEAR-MISS QUERIES", flush=True)
    print("=" * 150, flush=True)

    gt_data = json.loads(GT_PATH.read_text(encoding="utf-8"))
    gt_map = {q["query_id"]: q for q in gt_data["queries"]}

    output_root = None
    for r in POSSIBLE_OUTPUT_ROOTS:
        if r.exists():
            output_root = r
            break

    if output_root is None:
        print("❌ Error: Output root directory not found in /kaggle/working/output/kis_full38.")
        return

    print(f"• Ingested Output Directory: {output_root}", flush=True)

    causal_results = []
    category_counts: dict[str, int] = {}

    for qid in TARGET_QUERIES_A_B:
        gt = gt_map.get(qid)
        if not gt:
            continue

        target_vid = gt["video_id"]
        gt_start = gt["start_frame"]
        gt_end = gt["end_frame"]
        gt_len = gt_end - gt_start + 1

        q_dir = find_query_dir(output_root, qid)
        if not q_dir:
            continue

        # Load refinement trace or candidates
        trace_file = q_dir / "refinement_trace.json"
        cand_file = q_dir / "refinement_candidates.json"
        top100_file = q_dir / "refined_top100.jsonl"
        if not top100_file.exists():
            top100_file = q_dir / "top100.jsonl"

        top_preds = []
        if top100_file.exists():
            top_preds = [json.loads(l) for l in top100_file.read_text(encoding="utf-8").splitlines() if l.strip()]

        ref_candidates = []
        if cand_file.exists():
            try:
                ref_candidates = json.loads(cand_file.read_text(encoding="utf-8"))
            except Exception:
                pass
        elif trace_file.exists():
            try:
                trace_data = json.loads(trace_file.read_text(encoding="utf-8"))
                ref_candidates = trace_data.get("candidates", [])
            except Exception:
                pass

        # Identify target video candidates
        target_ref_cands = [c for c in ref_candidates if c.get("video_id") == target_vid]
        target_top_preds = [p for p in top_preds if p.get("video_id") == target_vid]

        first_top_rank = target_top_preds[0]["rank"] if target_top_preds else None
        nearest_frame = target_top_preds[0]["frame_id"] if target_top_preds else None
        dist_to_gt = None
        if nearest_frame is not None:
            if gt_start <= nearest_frame <= gt_end:
                dist_to_gt = 0
            elif nearest_frame < gt_start:
                dist_to_gt = gt_start - nearest_frame
            else:
                dist_to_gt = nearest_frame - gt_end

        # Causal Analysis
        causal_loss = "UNKNOWN"
        details = ""

        if not target_ref_cands:
            causal_loss = "ANCHOR_PATH_MISS"
            details = f"Target video {target_vid} not in initial candidate pool"
        else:
            # Check refined anchors
            refined_cands = [c for c in target_ref_cands if c.get("status") in ("REFINED", "RefinementStatus.REFINED", "refined")]
            if not refined_cands:
                orig_ranks = [c.get("original_candidate_rank") for c in target_ref_cands]
                causal_loss = "ANCHOR_PATH_MISS"
                details = f"Retrieved at ranks {orig_ranks[:3]}, but outside RefineTopN (rank > 3 -> skipped refinement)"
            else:
                primary_cand = refined_cands[0]
                anchor_f = primary_cand.get("candidate_frame_id")
                w_start = primary_cand.get("window_start_frame", 0)
                w_end = primary_cand.get("window_end_frame", 0)
                coarse_f_ids = primary_cand.get("coarse_frame_ids", [])
                fine_f_ids = primary_cand.get("fine_frame_ids", [])
                chosen_f = primary_cand.get("refined_frame_id")

                # Check coarse window coverage
                gt_in_coarse_window = not (gt_end < w_start or gt_start > w_end)
                gt_in_fine_window = False
                if fine_f_ids:
                    gt_in_fine_window = not (gt_end < min(fine_f_ids) or gt_start > max(fine_f_ids))

                if not gt_in_coarse_window:
                    causal_loss = "COARSE_WINDOW_MISS"
                    details = f"Anchor {anchor_f} coarse window [{w_start}..{w_end}] did not reach GT [{gt_start}..{gt_end}]"
                elif not gt_in_fine_window:
                    causal_loss = "FINE_WINDOW_MISS"
                    details = f"Coarse step selected frame outside fine radius [{min(fine_f_ids) if fine_f_ids else 0}..{max(fine_f_ids) if fine_f_ids else 0}]"
                else:
                    # GT was in fine window
                    gt_evaluated = any(gt_start <= f <= gt_end for f in fine_f_ids)
                    if gt_evaluated:
                        causal_loss = "SCORER_SELECTION_MISS"
                        details = f"GT frame was evaluated in fine search, but scorer preferred frame {chosen_f}"
                    else:
                        causal_loss = "STRIDE_SPARSITY_MISS"
                        details = f"Coarse/fine stride stepped over {gt_len}-frame GT window"

        category_counts[causal_loss] = category_counts.get(causal_loss, 0) + 1
        bucket_label = "Bucket A (<=30f)" if qid in TARGET_QUERIES_A_B[:8] else "Bucket B (31-90f)"

        causal_results.append({
            "qid": qid,
            "bucket": bucket_label,
            "target_video": target_vid,
            "gt_interval": f"[{gt_start}..{gt_end}]",
            "distance": dist_to_gt,
            "causal_loss": causal_loss,
            "details": details,
        })

    print(f"{'QID':<8} | {'Bucket':<18} | {'Target Vid':<11} | {'GT Interval':<16} | {'Dist':<6} | {'First Causal Loss':<24} | {'Details'}", flush=True)
    print("-" * 150, flush=True)
    for r in causal_results:
        print(
            f"{r['qid']:<8} | {r['bucket']:<18} | {r['target_video']:<11} | {r['gt_interval']:<16} | "
            f"{r['distance']:<6} | {r['causal_loss']:<24} | {r['details']}",
            flush=True,
        )

    print("\n" + "=" * 150, flush=True)
    print("CAUSAL LOSS DISTRIBUTION SUMMARY (14 Near-Miss Queries)", flush=True)
    print("=" * 150, flush=True)
    for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"• {cat:<26}: {count:2d} / 14 ({count/14*100:5.1f}%)", flush=True)
    print("=" * 150, flush=True)


if __name__ == "__main__":
    audit_causal_losses()
