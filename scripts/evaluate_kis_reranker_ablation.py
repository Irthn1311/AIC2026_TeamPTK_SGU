"""
KIS Reranker V1 Ablation Evaluation Script
==========================================
Ablation Study:
- Stage 0: Baseline (Top-50 Raw Dynamic Fusion)
- Stage 1: Grouping Only (Temporal Candidate Grouping)
- Stage 2: Grouping + Text Cross-Encoder Reranker (BAAI/bge-reranker-base)

Calculates:
- Hit@1, 5, 10, 20, 30, 50 & MRR across ±1s, ±3s, ±5s
- Per-query rank progression: baseline_rank -> grouped_rank -> reranked_rank
- Detailed audit on target queries (rank 29, 35, 37, 48)
- Outputs metrics JSON and comparative CSVs strictly on Drive E
"""

from __future__ import annotations

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from statistics import mean, median

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Enforce offline mode and Drive E cache
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HOME"] = str(PROJECT_ROOT / ".cache" / "huggingface")
os.environ["TORCH_HOME"] = str(PROJECT_ROOT / ".cache" / "torch")
os.environ["TRANSFORMERS_CACHE"] = str(PROJECT_ROOT / ".cache" / "huggingface" / "hub")

import pandas as pd
from backend.retrieval_service import RetrievalService
from src.retrieval.kis_reranker import KISRerankerV1
from scripts.evaluate_kis import load_gt, VideoMetadata, format_time, first_hit_rank

TOLERANCES_SEC = (1.0, 3.0, 5.0)
HIT_KS = (1, 5, 10, 20, 30, 50)


def compute_metrics_for_stage(query_records: List[Dict[str, Any]], tolerance: float) -> Dict[str, Any]:
    n = len(query_records)
    if n == 0:
        return {"num_queries": 0, "mrr": 0.0, **{f"hit_at_{k}": 0.0 for k in HIT_KS}}

    tol_int = int(tolerance)
    out: Dict[str, Any] = {"num_queries": n}
    
    # Hit@K
    for k in HIT_KS:
        hits = sum(1 for q in query_records if q.get(f"rank_{tol_int}s") is not None and q[f"rank_{tol_int}s"] <= k)
        out[f"hit_at_{k}"] = round(hits / n, 6)

    # MRR
    reciprocal_ranks = [
        0.0 if q.get(f"rank_{tol_int}s") is None else 1.0 / float(q[f"rank_{tol_int}s"])
        for q in query_records
    ]
    out["mrr"] = round(sum(reciprocal_ranks) / n, 6)
    return out


def run_ablation(args: argparse.Namespace) -> None:
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    gt_path = args.gt if args.gt.is_absolute() else PROJECT_ROOT / args.gt

    print("=" * 100)
    print(" 🚀 KIS RERANKING V1 ABLATION EVALUATION")
    print(f" GT File:    {gt_path}")
    print(f" Output Dir: {output_dir}")
    print(f" Top-K Raw:  {args.top_k} | Alpha Fusion: {args.fusion_alpha}")
    print("=" * 100)

    # 1. Initialize Retrieval Service & Reranker
    service = RetrievalService.get_instance()
    service.initialize()

    reranker = KISRerankerV1.get_instance()
    reranker.fusion_alpha = args.fusion_alpha
    reranker.initialize()

    gt_payload, queries = load_gt(gt_path, args.limit)
    fps_lookup = VideoMetadata()

    progression_rows: List[Dict[str, Any]] = []

    stage0_records: List[Dict[str, Any]] = []
    stage1_records: List[Dict[str, Any]] = []
    stage2_records: List[Dict[str, Any]] = []

    print(f"\n[EVAL] Running 3-stage evaluation on {len(queries)} queries...\n")

    for idx, q in enumerate(queries, start=1):
        query_id = str(q.get("query_id", f"query_{idx:03d}"))
        query_text = str(q.get("query", "")).strip()
        gt = q.get("gt") or {}
        metadata = q.get("metadata") or {}
        gt_video_id = str(gt.get("video_id", "")).strip()
        gt_frame = int(gt.get("semantic_frame"))
        gt_fps = fps_lookup.fps(gt_video_id)
        gt_time = gt_frame / gt_fps
        difficulty = str(metadata.get("difficulty", "unknown") or "unknown")
        modality = str(metadata.get("primary_modality", "unknown") or "unknown")

        # Stage 0: Raw Candidates from Dynamic Fusion (no early aggressive dedup dropping)
        search_res = service.search(
            query=query_text,
            top_k=args.top_k,
            fusion_mode=args.fusion_mode,
            temporal_dedup=False,
        )
        raw_candidates = search_res.get("results", [])[: args.top_k]

        # Standardize raw candidate predictions for evaluation
        preds_stage0 = []
        for r_idx, c in enumerate(raw_candidates, start=1):
            ts = float(c.get("timestamp_seconds", 0.0))
            vid = str(c.get("video_id", ""))
            f_idx = int(c.get("frame_idx", 0))
            is_same_vid = (vid == gt_video_id)
            err_sec = abs(ts - gt_time) if is_same_vid else None
            preds_stage0.append({
                "rank": r_idx,
                "video_id": vid,
                "frame_id": f_idx,
                "timestamp_sec": ts,
                "temporal_error_sec": err_sec,
                "score": float(c.get("score", 0.0)),
                "hit_1s": is_same_vid and err_sec <= 1.0,
                "hit_3s": is_same_vid and err_sec <= 3.0,
                "hit_5s": is_same_vid and err_sec <= 5.0,
            })

        # Stage 1: Grouping Only
        grouped_segments = reranker.group_only(raw_candidates)
        preds_stage1_seg = reranker.segments_to_predictions(grouped_segments)
        preds_stage1 = []
        for r_idx, p in enumerate(preds_stage1_seg, start=1):
            ts = float(p.get("timestamp_seconds", p.get("timestamp_sec", 0.0)))
            vid = str(p.get("video_id", ""))
            f_idx = int(p.get("frame_idx", 0))
            is_same_vid = (vid == gt_video_id)
            err_sec = abs(ts - gt_time) if is_same_vid else None
            preds_stage1.append({
                "rank": r_idx,
                "video_id": vid,
                "frame_id": f_idx,
                "timestamp_sec": ts,
                "temporal_error_sec": err_sec,
                "score": float(p.get("score", 0.0)),
                "hit_1s": is_same_vid and err_sec <= 1.0,
                "hit_3s": is_same_vid and err_sec <= 3.0,
                "hit_5s": is_same_vid and err_sec <= 5.0,
            })

        # Stage 2: Grouping + Text Cross-Encoder Reranker
        reranked_segments = reranker.rerank(query=query_text, raw_candidates=raw_candidates, alpha=args.fusion_alpha)
        preds_stage2_seg = reranker.segments_to_predictions(reranked_segments)
        preds_stage2 = []
        for r_idx, p in enumerate(preds_stage2_seg, start=1):
            ts = float(p.get("timestamp_seconds", p.get("timestamp_sec", 0.0)))
            vid = str(p.get("video_id", ""))
            f_idx = int(p.get("frame_idx", 0))
            is_same_vid = (vid == gt_video_id)
            err_sec = abs(ts - gt_time) if is_same_vid else None
            preds_stage2.append({
                "rank": r_idx,
                "video_id": vid,
                "frame_id": f_idx,
                "timestamp_sec": ts,
                "temporal_error_sec": err_sec,
                "score": float(p.get("score", 0.0)),
                "hit_1s": is_same_vid and err_sec <= 1.0,
                "hit_3s": is_same_vid and err_sec <= 3.0,
                "hit_5s": is_same_vid and err_sec <= 5.0,
            })

        # Calculate ranks per tolerance
        r0_1s = first_hit_rank(preds_stage0, tolerance=1.0, k=len(preds_stage0))
        r0_3s = first_hit_rank(preds_stage0, tolerance=3.0, k=len(preds_stage0))
        r0_5s = first_hit_rank(preds_stage0, tolerance=5.0, k=len(preds_stage0))

        r1_1s = first_hit_rank(preds_stage1, tolerance=1.0, k=len(preds_stage1))
        r1_3s = first_hit_rank(preds_stage1, tolerance=3.0, k=len(preds_stage1))
        r1_5s = first_hit_rank(preds_stage1, tolerance=5.0, k=len(preds_stage1))

        r2_1s = first_hit_rank(preds_stage2, tolerance=1.0, k=len(preds_stage2))
        r2_3s = first_hit_rank(preds_stage2, tolerance=3.0, k=len(preds_stage2))
        r2_5s = first_hit_rank(preds_stage2, tolerance=5.0, k=len(preds_stage2))

        stage0_records.append({"query_id": query_id, "rank_1s": r0_1s, "rank_3s": r0_3s, "rank_5s": r0_5s, "modality": modality, "difficulty": difficulty, "video_id": gt_video_id})
        stage1_records.append({"query_id": query_id, "rank_1s": r1_1s, "rank_3s": r1_3s, "rank_5s": r1_5s, "modality": modality, "difficulty": difficulty, "video_id": gt_video_id})
        stage2_records.append({"query_id": query_id, "rank_1s": r2_1s, "rank_3s": r2_3s, "rank_5s": r2_5s, "modality": modality, "difficulty": difficulty, "video_id": gt_video_id})

        # Progression record
        r0_disp = f"#{r0_3s}" if r0_3s is not None else "MISS"
        r1_disp = f"#{r1_3s}" if r1_3s is not None else "MISS"
        r2_disp = f"#{r2_3s}" if r2_3s is not None else "MISS"
        
        # Rank delta
        rank_gain = 0
        if r0_3s is not None and r2_3s is not None:
            rank_gain = r0_3s - r2_3s

        progression_rows.append({
            "query_id": query_id,
            "query": query_text,
            "gt_video_id": gt_video_id,
            "gt_time_sec": round(gt_time, 2),
            "difficulty": difficulty,
            "modality": modality,
            "num_raw_candidates": len(raw_candidates),
            "num_grouped_segments": len(grouped_segments),
            "stage0_baseline_rank_3s": r0_3s,
            "stage1_grouped_rank_3s": r1_3s,
            "stage2_reranked_rank_3s": r2_3s,
            "rank_gain_3s": rank_gain,
            "stage0_rank_1s": r0_1s,
            "stage1_rank_1s": r1_1s,
            "stage2_rank_1s": r2_1s,
            "stage0_rank_5s": r0_5s,
            "stage1_rank_5s": r1_5s,
            "stage2_rank_5s": r2_5s,
        })

        if idx % 10 == 0 or idx == len(queries):
            print(f"  Processed {idx:02d}/{len(queries):02d} queries... ({query_id}: {r0_disp} -> {r1_disp} -> {r2_disp})")

    # 2. Compute Metrics across Tolerances for all 3 Stages
    summary: Dict[str, Any] = {
        "parameters": {
            "top_k": args.top_k,
            "fusion_alpha": args.fusion_alpha,
            "fusion_mode": args.fusion_mode,
            "num_queries": len(queries),
        },
        "stages": {},
    }

    for stage_name, records in [
        ("Stage 0: Baseline (Raw Dynamic Fusion)", stage0_records),
        ("Stage 1: Grouping Only", stage1_records),
        ("Stage 2: Grouping + Reranker V1", stage2_records),
    ]:
        stage_dict = {}
        for tol in TOLERANCES_SEC:
            stage_dict[f"tolerance_{int(tol)}s"] = compute_metrics_for_stage(records, tol)
        summary["stages"][stage_name] = stage_dict

    # 3. Print Comprehensive Comparison Table
    print("\n" + "=" * 105)
    print(" 📊 KIS RERANKING V1 ABLATION RESULTS (BEFORE / AFTER COMPARISON)")
    print("=" * 105)

    header = f"  {'Ablation Stage':<38} | {'Hit@1':<6} | {'Hit@5':<6} | {'Hit@10':<6} | {'Hit@20':<6} | {'Hit@30':<6} | {'Hit@50':<6} | {'MRR':<7}"

    for tol in TOLERANCES_SEC:
        tol_key = f"tolerance_{int(tol)}s"
        print(f"\n--- Temporal Tolerance: ±{int(tol)}s ---")
        print(header)
        print("  " + "-" * 100)
        for stage_name, sdata in summary["stages"].items():
            m = sdata[tol_key]
            print(
                f"  {stage_name:<38} | "
                f"{m['hit_at_1']:6.1%} | "
                f"{m['hit_at_5']:6.1%} | "
                f"{m['hit_at_10']:6.1%} | "
                f"{m['hit_at_20']:6.1%} | "
                f"{m['hit_at_30']:6.1%} | "
                f"{m['hit_at_50']:6.1%} | "
                f"{m['mrr']:.4f}"
            )

    # 4. Target Query Deep-Dive (Rank 29, 35, 37, 48)
    print("\n" + "=" * 105)
    print(" 🎯 TARGET QUERIES PROGRESSION AUDIT (Previously Miss@20 / Rank 21-50)")
    print("=" * 105)
    
    target_qids = ["KIS_L21_V001_001", "KIS_L21_V007_004", "KIS_L21_V001_008", "KIS_L21_V003_005"]
    target_rows = [r for r in progression_rows if r["query_id"] in target_qids]

    for tr in target_rows:
        qid = tr["query_id"]
        qtext = tr["query"]
        r0 = tr["stage0_baseline_rank_3s"]
        r1 = tr["stage1_grouped_rank_3s"]
        r2 = tr["stage2_reranked_rank_3s"]
        diff = tr["difficulty"]
        mod = tr["modality"]
        
        status_sym = "🚀 IMPROVED" if (r2 is not None and (r0 is None or r2 < r0)) else "⚖️ STABLE"
        print(f"• Query: [{diff:<6}] [{mod:<6}] {qid}")
        print(f"  Text: \"{qtext}\"")
        print(f"  Rank Progression (±3s): Baseline #{r0} -> Grouped #{r1} -> Reranked #{r2}  [{status_sym}]")
        print()

    # 5. Overall Rank Gainers & Preserved Recall Summary
    gainers = [r for r in progression_rows if r["rank_gain_3s"] > 0]
    preserved = [r for r in progression_rows if r["stage0_baseline_rank_3s"] is not None and r["stage2_reranked_rank_3s"] is not None]
    
    print("=" * 105)
    print(" 📈 RECALL & PRECISION VERIFICATION SUMMARY")
    print("=" * 105)
    print(f"• Total Queries Evaluated:            {len(progression_rows)}")
    print(f"• Queries Improved in Rank (+gain):   {len(gainers)} queries")
    print(f"• Baseline Recall (Hit@50 @ ±3s):     {summary['stages']['Stage 0: Baseline (Raw Dynamic Fusion)']['tolerance_3s']['hit_at_50']:.1%}")
    print(f"• Reranked Recall (Hit@50 @ ±3s):     {summary['stages']['Stage 2: Grouping + Reranker V1']['tolerance_3s']['hit_at_50']:.1%}")
    print(f"• Hit@1 Gain:                         {summary['stages']['Stage 0: Baseline (Raw Dynamic Fusion)']['tolerance_3s']['hit_at_1']:.1%} -> {summary['stages']['Stage 2: Grouping + Reranker V1']['tolerance_3s']['hit_at_1']:.1%}")
    print(f"• Hit@5 Gain:                         {summary['stages']['Stage 0: Baseline (Raw Dynamic Fusion)']['tolerance_3s']['hit_at_5']:.1%} -> {summary['stages']['Stage 2: Grouping + Reranker V1']['tolerance_3s']['hit_at_5']:.1%}")
    print(f"• Hit@10 Gain:                        {summary['stages']['Stage 0: Baseline (Raw Dynamic Fusion)']['tolerance_3s']['hit_at_10']:.1%} -> {summary['stages']['Stage 2: Grouping + Reranker V1']['tolerance_3s']['hit_at_10']:.1%}")
    print(f"• MRR Gain:                           {summary['stages']['Stage 0: Baseline (Raw Dynamic Fusion)']['tolerance_3s']['mrr']:.4f} -> {summary['stages']['Stage 2: Grouping + Reranker V1']['tolerance_3s']['mrr']:.4f}")
    print("=" * 105 + "\n")

    # 6. Save Artifacts strictly to Drive E
    out_json = output_dir / "ablation_summary.json"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    
    df_progression = pd.DataFrame(progression_rows)
    out_csv = output_dir / "ablation_rank_progression.csv"
    df_progression.to_csv(out_csv, index=False, encoding="utf-8")

    print(f"[SAVE] Summary JSON: {out_json}")
    print(f"[SAVE] Progression CSV: {out_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KIS Reranker V1 Ablation Study")
    parser.add_argument("--gt", type=Path, default=Path("JsonTest/gt_kis.json"), help="Ground truth JSON")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/evaluation/kis_reranker"), help="Output directory")
    parser.add_argument("--top-k", type=int, default=50, help="Candidate pool size")
    parser.add_argument("--fusion-mode", type=str, default="dynamic", help="Fusion mode for Stage 0")
    parser.add_argument("--fusion-alpha", type=float, default=0.55, help="Weight for original fusion score vs rerank score")
    parser.add_argument("--limit", type=int, default=None, help="Optional query limit")
    return parser.parse_args()


if __name__ == "__main__":
    run_ablation(parse_args())
