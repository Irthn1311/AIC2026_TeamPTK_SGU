"""
Production Pipeline Runner & Benchmark Evaluator for AI Challenge 2026
======================================================================
Runs the frozen KIS V1.2 Production Pipeline (BGE Text Cross-Encoder + Stage 1 OpenCLIP Direct)
across all queries in a specified dataset file.

Usage:
  E:\\conda_envs\\aic2026\\python.exe scripts/run_production_evaluation.py --input outputs/frozen_raw_top50.json
  E:\\conda_envs\\aic2026\\python.exe scripts/run_production_evaluation.py --input JsonTest/gt_kis_l30.json
"""

import sys
import json
import time
import argparse
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.kis_reranker import KISRerankerV1


def eval_predictions(items_with_preds, tol=3.0):
    n = len(items_with_preds)
    if n == 0:
        return {}
    hits = {k: 0 for k in [1, 5, 10, 20, 30, 50]}
    rr_sum = 0.0
    ranks = []

    for item in items_with_preds:
        preds = item["preds"]
        gt_vid = item.get("gt_vid", "")
        gt_time = float(item.get("gt_time", 0.0))

        for p in preds:
            vid = str(p.get("video_id", ""))
            ts = float(p.get("timestamp_seconds", p.get("timestamp_sec", 0.0)))
            p["hit"] = (vid == gt_vid) and (abs(ts - gt_time) <= tol)

        hit_idx = None
        for i, p in enumerate(preds, start=1):
            if p["hit"]:
                hit_idx = i
                break
        ranks.append(hit_idx)
        if hit_idx is not None:
            rr_sum += 1.0 / hit_idx
            for k in hits:
                if hit_idx <= k:
                    hits[k] += 1

    return {
        "count": n,
        "H@1": round(hits[1] / n * 100, 1),
        "H@5": round(hits[5] / n * 100, 1),
        "H@10": round(hits[10] / n * 100, 1),
        "H@20": round(hits[20] / n * 100, 1),
        "H@30": round(hits[30] / n * 100, 1),
        "H@50": round(hits[50] / n * 100, 1),
        "MRR": round(rr_sum / n, 4),
        "ranks": ranks,
    }


def main():
    parser = argparse.ArgumentParser(description="Run Production KIS V1.2 Evaluation")
    parser.add_argument("--input", type=str, default="outputs/frozen_raw_top50.json", help="Path to input test queries dataset JSON")
    parser.add_argument("--output", type=str, default="outputs/evaluation/production_v12_results.json", help="Path to output evaluation JSON")
    args = parser.parse_args()

    input_path = PROJECT_ROOT / args.input
    if not input_path.exists():
        print(f"Error: Input file {input_path} does not exist.")
        sys.exit(1)

    print("=" * 100)
    print(f"RUNNING FROZEN KIS V1.2 PRODUCTION PIPELINE ON DATASET: {input_path.name}")
    print("=" * 100)

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} test queries from {input_path}")

    # Initialize Production Reranker
    reranker = KISRerankerV1.get_instance()
    reranker.initialize()

    print(f"Active Production Alpha Routing: {reranker.DEFAULT_ALPHA_MAP}")
    print(f"Active Production Visual Beta: {reranker.siglip2_beta} (Stage 1 OpenCLIP Direct)")
    print(f"Active Production SigLIP2 Enabled: {reranker.enable_siglip2}")

    eval_items_base = []
    eval_items_v12 = []

    t_start = time.perf_counter()
    for idx, item in enumerate(data, start=1):
        qid = item.get("query_id", f"Q_{idx}")
        q_text = item.get("query", "")
        modality = item.get("modality", "visual")
        cands = item.get("raw_cands", item.get("candidates", []))
        gt_vid = item.get("gt_vid", "")
        gt_time = item.get("gt_time", 0.0)

        # Baseline (raw candidates)
        base_preds = cands[:50]

        # Production V1.2 Rerank
        segs = reranker.rerank(query=q_text, raw_candidates=cands, modality=modality)
        v12_preds = reranker.segments_to_predictions(segs, modality=modality, top_k=50)

        eval_items_base.append({"query_id": qid, "modality": modality, "gt_vid": gt_vid, "gt_time": gt_time, "preds": base_preds})
        eval_items_v12.append({"query_id": qid, "modality": modality, "gt_vid": gt_vid, "gt_time": gt_time, "preds": v12_preds})

    t_end = time.perf_counter()
    avg_latency_ms = round((t_end - t_start) / len(data) * 1000, 2)

    print("\n" + "=" * 100)
    print(f"EVALUATION METRICS SUMMARY (Total Queries: {len(data)}) | Average Latency: {avg_latency_ms} ms/query")
    print("=" * 100)

    for tol in [1.0, 3.0, 5.0]:
        res_base = eval_predictions(eval_items_base, tol=tol)
        res_v12 = eval_predictions(eval_items_v12, tol=tol)

        print(f"\n--- Tolerance ±{int(tol)}s ---")
        print(f"Baseline (Raw Stage 1) : Hit@1={res_base['H@1']}%, Hit@5={res_base['H@5']}%, Hit@10={res_base['H@10']}%, MRR={res_base['MRR']}")
        print(f"Production V1.2        : Hit@1={res_v12['H@1']}%, Hit@5={res_v12['H@5']}%, Hit@10={res_v12['H@10']}%, MRR={res_v12['MRR']}")

        d_h1 = round(res_v12["H@1"] - res_base["H@1"], 1)
        d_mrr = round(res_v12["MRR"] - res_base["MRR"], 4)
        print(f"Delta                  : Hit@1={d_h1:+.1f}%, MRR={d_mrr:+.4f}")

    # Modality Breakdown
    print("\n" + "=" * 100)
    print("MODALITY BREAKDOWN AT TOLERANCE ±3s")
    print("=" * 100)
    print(f"{'Modality':<12} | {'N':<4} | {'Base Hit@1':<12} | {'V1.2 Hit@1':<12} | {'Base MRR':<10} | {'V1.2 MRR':<10} | {'Status'}")
    print("-" * 100)

    modalities = ["ocr", "asr", "mixed", "object", "visual"]
    summary_dict = {}

    for mod in modalities:
        m_base = [item for item in eval_items_base if item["modality"] == mod]
        m_v12 = [item for item in eval_items_v12 if item["modality"] == mod]
        if not m_base:
            continue
        r_base = eval_predictions(m_base, tol=3.0)
        r_v12 = eval_predictions(m_v12, tol=3.0)

        h1_diff = r_v12["H@1"] - r_base["H@1"]
        mrr_diff = r_v12["MRR"] - r_base["MRR"]

        status = "IMPROVED 🚀" if (h1_diff > 0 or mrr_diff > 0) else ("RETAINED ✅" if (h1_diff == 0 and mrr_diff == 0) else "REGRESSED ⚠️")

        print(f"{mod:<12} | {len(m_base):<4} | {r_base['H@1']:<11}% | {r_v12['H@1']:<11}% | {r_base['MRR']:<10} | {r_v12['MRR']:<10} | {status}")
        summary_dict[mod] = {"count": len(m_base), "base": r_base, "v12": r_v12}

    out_path = PROJECT_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, ensure_ascii=False, indent=2)
    print(f"\nSaved production evaluation summary to {out_path}")


if __name__ == "__main__":
    main()
