"""
KIS Reranker V1.1 Evaluation & Alpha/Normalization Sweep
========================================================
- Frozen Raw Top-50 (temporal_dedup=False)
- TemporalCandidateGrouper
- Raw Cross-Encoder Logits (BAAI/bge-reranker-base)
- Query-level Normalization Sweep (min_max, z_score, rank)
- Alpha Sweep (alpha = 0.0 [Grouping], 0.25, 0.50, 0.75, 1.0 [Reranker Only])
- 1 Rank Slot per Segment (expand_members=False)
- Strict Metrics: Hit@1/5/10/20/30/50, MRR @ ±1s, ±3s, ±5s
- Reorder Audit: Improved / Regressed / Unchanged, Mean/Median Delta
- Top 10 Improvements & Regressions
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

import numpy as np
import pandas as pd
from backend.retrieval_service import RetrievalService
from src.retrieval.kis_reranker import KISRerankerV1
from scripts.evaluate_kis import load_gt, VideoMetadata, first_hit_rank

TOLERANCES_SEC = (1.0, 3.0, 5.0)
HIT_KS = (1, 5, 10, 20, 30, 50)


def compute_metrics(records: List[Dict[str, Any]], tolerance: float) -> Dict[str, Any]:
    n = len(records)
    if n == 0:
        return {"num_queries": 0, "mrr": 0.0, **{f"hit_at_{k}": 0.0 for k in HIT_KS}}

    tol_int = int(tolerance)
    out: Dict[str, Any] = {"num_queries": n}
    for k in HIT_KS:
        hits = sum(1 for q in records if q.get(f"rank_{tol_int}s") is not None and q[f"rank_{tol_int}s"] <= k)
        out[f"hit_at_{k}"] = round(hits / n, 6)

    reciprocal_ranks = [
        0.0 if q.get(f"rank_{tol_int}s") is None else 1.0 / float(q[f"rank_{tol_int}s"])
        for q in records
    ]
    out["mrr"] = round(sum(reciprocal_ranks) / n, 6)
    return out


def evaluate_predictions(preds: List[Dict[str, Any]], gt_video_id: str, gt_time: float) -> Dict[str, Optional[float]]:
    for r_idx, p in enumerate(preds, start=1):
        ts = float(p.get("timestamp_seconds", p.get("timestamp_sec", 0.0)))
        vid = str(p.get("video_id", ""))
        err_sec = abs(ts - gt_time) if vid == gt_video_id else None
        p["rank"] = r_idx
        p["video_id"] = vid
        p["timestamp_sec"] = ts
        p["temporal_error_sec"] = err_sec
        p["hit_1s"] = vid == gt_video_id and err_sec is not None and err_sec <= 1.0
        p["hit_3s"] = vid == gt_video_id and err_sec is not None and err_sec <= 3.0
        p["hit_5s"] = vid == gt_video_id and err_sec is not None and err_sec <= 5.0

    return {
        "rank_1s": first_hit_rank(preds, tolerance=1.0, k=len(preds)),
        "rank_3s": first_hit_rank(preds, tolerance=3.0, k=len(preds)),
        "rank_5s": first_hit_rank(preds, tolerance=5.0, k=len(preds)),
    }


def main():
    parser = argparse.ArgumentParser(description="KIS Reranker V1.1 Sweeps & Evaluation")
    parser.add_argument("--gt", type=Path, default=Path("JsonTest/gt_kis.json"), help="Ground truth JSON")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/evaluation/kis_reranker"), help="Output directory")
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    gt_path = args.gt if args.gt.is_absolute() else PROJECT_ROOT / args.gt

    print("=" * 100)
    print(" 🚀 KIS RERANKER V1.1 SWEEP & EVALUATION (DRIVE E OFFLINE)")
    print(f" GT: {gt_path}")
    print(f" Output: {out_dir}")
    print("=" * 100)

    # 1. Initialize
    service = RetrievalService.get_instance()
    service.initialize()

    reranker = KISRerankerV1.get_instance()
    reranker.initialize()

    gt_payload, queries = load_gt(gt_path, None)
    fps_lookup = VideoMetadata()

    # Pre-fetch all frozen raw Top-50 candidates
    print(f"\n[1/3] Fetching Frozen Raw Top-50 Candidates for {len(queries)} queries...")
    query_data = []
    for idx, q in enumerate(queries, start=1):
        qid = str(q.get("query_id", f"query_{idx:03d}"))
        qtext = str(q.get("query", "")).strip()
        gt = q.get("gt") or {}
        metadata = q.get("metadata") or {}
        gt_vid = str(gt.get("video_id", "")).strip()
        gt_frame = int(gt.get("semantic_frame"))
        gt_fps = fps_lookup.fps(gt_vid)
        gt_time = gt_frame / gt_fps
        difficulty = str(metadata.get("difficulty", "unknown") or "unknown")
        modality = str(metadata.get("primary_modality", "unknown") or "unknown")

        search_res = service.search(query=qtext, top_k=50, fusion_mode="dynamic", temporal_dedup=False)
        raw_cands = search_res.get("results", [])[:50]

        query_data.append({
            "query_id": qid,
            "query": qtext,
            "gt_video_id": gt_vid,
            "gt_frame": gt_frame,
            "gt_time": gt_time,
            "difficulty": difficulty,
            "modality": modality,
            "raw_candidates": raw_cands,
        })
        if idx % 15 == 0 or idx == len(queries):
            print(f"  Loaded raw candidates {idx:02d}/{len(queries):02d}")

    # Configurations to test
    # 1. Stage 0: Raw Top-50
    # 2. Stage 1: Grouping Only (alpha=0.0)
    # 3. Sweep: norm_mode in ['min_max', 'z_score', 'rank'] x alpha in [0.25, 0.50, 0.75, 1.0]
    sweep_configs = [
        {"name": "Stage 0: Raw Top-50 Baseline", "type": "raw"},
        {"name": "Stage 1: Grouping Only (alpha=0.0)", "type": "group_only"},
    ]

    norm_modes = ["min_max", "z_score", "rank"]
    alphas = [0.25, 0.50, 0.75, 1.0]

    for nm in norm_modes:
        for a in alphas:
            sweep_configs.append({
                "name": f"Reranker V1.1 ({nm}, alpha={a:.2f})",
                "type": "rerank",
                "norm_mode": nm,
                "alpha": a,
            })

    print(f"\n[2/3] Running Sweep across {len(sweep_configs)} Configurations...")
    config_results = {}
    config_query_ranks = {}

    for cfg_idx, cfg in enumerate(sweep_configs, start=1):
        cfg_name = cfg["name"]
        records = []
        q_ranks = {}

        for item in query_data:
            qid = item["query_id"]
            qtext = item["query"]
            gt_vid = item["gt_video_id"]
            gt_time = item["gt_time"]
            raw_cands = item["raw_candidates"]

            if cfg["type"] == "raw":
                preds = [dict(c) for c in raw_cands]
            elif cfg["type"] == "group_only":
                grouped = reranker.group_only(raw_cands)
                preds = reranker.segments_to_predictions(grouped, expand_members=False)
            elif cfg["type"] == "rerank":
                reranked = reranker.rerank(
                    query=qtext,
                    raw_candidates=raw_cands,
                    alpha=cfg["alpha"],
                    norm_mode=cfg["norm_mode"],
                )
                preds = reranker.segments_to_predictions(reranked, expand_members=False)

            ranks = evaluate_predictions(preds, gt_vid, gt_time)
            records.append({
                "query_id": qid,
                "modality": item["modality"],
                "difficulty": item["difficulty"],
                **ranks,
            })
            q_ranks[qid] = ranks

        # Compute metrics across tolerances
        cfg_metrics = {}
        for tol in TOLERANCES_SEC:
            cfg_metrics[f"tolerance_{int(tol)}s"] = compute_metrics(records, tol)

        config_results[cfg_name] = cfg_metrics
        config_query_ranks[cfg_name] = q_ranks

        m3 = cfg_metrics["tolerance_3s"]
        print(f"  [{cfg_idx:02d}/{len(sweep_configs):02d}] {cfg_name:<42} | H@1={m3['hit_at_1']:5.1%} H@5={m3['hit_at_5']:5.1%} H@10={m3['hit_at_10']:5.1%} H@20={m3['hit_at_20']:5.1%} H@50={m3['hit_at_50']:5.1%} MRR={m3['mrr']:.4f}")

    # 3. Print Comprehensive Comparison Tables
    print("\n" + "=" * 115)
    print(" 📊 KIS RERANKER V1.1 SWEEP RESULTS (BEFORE / AFTER COMPARISON)")
    print("=" * 115)

    header = f"  {'Configuration':<42} | {'Hit@1':<6} | {'Hit@5':<6} | {'Hit@10':<6} | {'Hit@20':<6} | {'Hit@30':<6} | {'Hit@50':<6} | {'MRR':<7}"

    for tol in TOLERANCES_SEC:
        tol_key = f"tolerance_{int(tol)}s"
        print(f"\n--- Temporal Tolerance: ±{int(tol)}s ---")
        print(header)
        print("  " + "-" * 110)
        for cfg_name, sdata in config_results.items():
            m = sdata[tol_key]
            print(
                f"  {cfg_name:<42} | "
                f"{m['hit_at_1']:6.1%} | "
                f"{m['hit_at_5']:6.1%} | "
                f"{m['hit_at_10']:6.1%} | "
                f"{m['hit_at_20']:6.1%} | "
                f"{m['hit_at_30']:6.1%} | "
                f"{m['hit_at_50']:6.1%} | "
                f"{m['mrr']:.4f}"
            )

    # 4. Detailed Comparison & Reorder Audit against Grouping Only & Raw Baseline
    stage0_name = "Stage 0: Raw Top-50 Baseline"
    stage1_name = "Stage 1: Grouping Only (alpha=0.0)"
    
    # Pick top configuration by score
    # Score = Hit@1 + Hit@5 + MRR at ±3s
    best_cfg_name = None
    best_score = -1.0
    for cfg_name, sdata in config_results.items():
        if cfg_name in (stage0_name, stage1_name):
            continue
        m = sdata["tolerance_3s"]
        score = m["hit_at_1"] * 2.0 + m["hit_at_5"] * 1.5 + m["hit_at_10"] + m["mrr"] * 3.0
        if score > best_score:
            best_score = score
            best_cfg_name = cfg_name

    print("\n" + "=" * 115)
    print(f" 🏆 BEST RERANKER V1.1 CONFIGURATION: {best_cfg_name}")
    print("=" * 115)

    s0_ranks = config_query_ranks[stage0_name]
    s1_ranks = config_query_ranks[stage1_name]
    best_ranks = config_query_ranks[best_cfg_name]

    audit_rows = []
    improved_vs_s0 = []
    regressed_vs_s0 = []
    unchanged_vs_s0 = []

    improved_vs_s1 = []
    regressed_vs_s1 = []
    unchanged_vs_s1 = []

    deltas_vs_s0 = []
    deltas_vs_s1 = []

    for item in query_data:
        qid = item["query_id"]
        qtext = item["query"]
        r0 = s0_ranks[qid]["rank_3s"]
        r1 = s1_ranks[qid]["rank_3s"]
        r2 = best_ranks[qid]["rank_3s"]

        # Delta vs Stage 1
        d_s1 = 0
        if r1 is not None and r2 is not None:
            d_s1 = r1 - r2
            deltas_vs_s1.append(d_s1)
            if r2 < r1:
                improved_vs_s1.append((qid, qtext, r1, r2, d_s1))
            elif r2 > r1:
                regressed_vs_s1.append((qid, qtext, r1, r2, d_s1))
            else:
                unchanged_vs_s1.append(qid)
        elif r1 is None and r2 is not None:
            improved_vs_s1.append((qid, qtext, None, r2, 50 - r2))
        elif r1 is not None and r2 is None:
            regressed_vs_s1.append((qid, qtext, r1, None, -50))
        else:
            unchanged_vs_s1.append(qid)

        # Delta vs Stage 0
        d_s0 = 0
        if r0 is not None and r2 is not None:
            d_s0 = r0 - r2
            deltas_vs_s0.append(d_s0)
            if r2 < r0:
                improved_vs_s0.append((qid, qtext, r0, r2, d_s0))
            elif r2 > r0:
                regressed_vs_s0.append((qid, qtext, r0, r2, d_s0))
            else:
                unchanged_vs_s0.append(qid)
        elif r0 is None and r2 is not None:
            improved_vs_s0.append((qid, qtext, None, r2, 50 - r2))
        elif r0 is not None and r2 is None:
            regressed_vs_s0.append((qid, qtext, r0, None, -50))
        else:
            unchanged_vs_s0.append(qid)

        audit_rows.append({
            "query_id": qid,
            "query": qtext,
            "modality": item["modality"],
            "difficulty": item["difficulty"],
            "stage0_rank_3s": r0,
            "stage1_rank_3s": r1,
            "best_reranked_rank_3s": r2,
            "delta_vs_stage1": d_s1,
            "delta_vs_stage0": d_s0,
            "stage0_rank_1s": s0_ranks[qid]["rank_1s"],
            "stage1_rank_1s": s1_ranks[qid]["rank_1s"],
            "best_reranked_rank_1s": best_ranks[qid]["rank_1s"],
            "stage0_rank_5s": s0_ranks[qid]["rank_5s"],
            "stage1_rank_5s": s1_ranks[qid]["rank_5s"],
            "best_reranked_rank_5s": best_ranks[qid]["rank_5s"],
        })

    # Sort improvements and regressions
    improved_vs_s1.sort(key=lambda x: x[4], reverse=True)
    regressed_vs_s1.sort(key=lambda x: x[4])

    print("\n--- Reorder Statistics: Best Reranker vs Stage 1 (Grouping Only) ---")
    print(f"• Total Queries Evaluated:    {len(query_data)}")
    print(f"• Queries Reordered (S1->S2): {len(improved_vs_s1) + len(regressed_vs_s1)} / {len(query_data)} ({(len(improved_vs_s1) + len(regressed_vs_s1))/len(query_data):.1%})")
    print(f"• Improved (+rank gain):      {len(improved_vs_s1)}")
    print(f"• Regressed (-rank drop):     {len(regressed_vs_s1)}")
    print(f"• Unchanged:                  {len(unchanged_vs_s1)}")
    if deltas_vs_s1:
        print(f"• Mean Rank Delta (S1->S2):   {mean(deltas_vs_s1):+.2f} ranks")
        print(f"• Median Rank Delta (S1->S2): {median(deltas_vs_s1):+.2f} ranks")

    print("\n--- Reorder Statistics: Best Reranker vs Stage 0 (Raw Dynamic Fusion) ---")
    print(f"• Improved vs Raw (+gain):    {len(improved_vs_s0)}")
    print(f"• Regressed vs Raw (-drop):   {len(regressed_vs_s0)}")
    print(f"• Unchanged:                  {len(unchanged_vs_s0)}")
    if deltas_vs_s0:
        print(f"• Mean Rank Delta (S0->S2):   {mean(deltas_vs_s0):+.2f} ranks")
        print(f"• Median Rank Delta (S0->S2): {median(deltas_vs_s0):+.2f} ranks")

    # 5. Top 10 Improvements and Top 10 Regressions
    print("\n" + "=" * 115)
    print(" 🌟 TOP IMPROVED QUERIES (Stage 1 -> Best Reranker)")
    print("=" * 115)
    for idx, (qid, qtext, r1, r2, d) in enumerate(improved_vs_s1[:10], start=1):
        print(f"{idx:02d}. [{d:+2d} ranks] {qid}: #{r1} -> #{r2}")
        print(f"    \"{qtext}\"")

    print("\n" + "=" * 115)
    print(" ⚠️ TOP REGRESSED QUERIES (Stage 1 -> Best Reranker)")
    print("=" * 115)
    for idx, (qid, qtext, r1, r2, d) in enumerate(regressed_vs_s1[:10], start=1):
        print(f"{idx:02d}. [{d:+2d} ranks] {qid}: #{r1} -> #{r2}")
        print(f"    \"{qtext}\"")

    # 6. Save Artifacts strictly on Drive E
    out_summary = {
        "best_configuration": best_cfg_name,
        "configs": config_results,
    }
    out_json = out_dir / "v1_1_ablation_summary.json"
    out_json.write_text(json.dumps(out_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    df_prog = pd.DataFrame(audit_rows)
    out_csv = out_dir / "v1_1_rank_progression.csv"
    df_prog.to_csv(out_csv, index=False, encoding="utf-8")

    print(f"\n[SAVE] Sweep Summary: {out_json}")
    print(f"[SAVE] Progression:   {out_csv}")


if __name__ == "__main__":
    main()
