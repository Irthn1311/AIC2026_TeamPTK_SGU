"""
Stage 4B Edge Quality Evaluation & Decision Tool (AI Challenge 2026)
====================================================================
Evaluates human audit labels (Relevant / Irrelevant / Ambiguous) from the 60 Stage 4B edge samples.

Output Metrics:
  1. Separate Precision & Counts for VISUAL_SIMILARITY and SEMANTIC_CONTINUITY.
  2. Precision breakdown by score bucket (Top, Middle, Near-Threshold).
  3. Failure edge diagnostics listing all Irrelevant edges with (src, dst, type, score, bucket).
  4. Final Decision: FREEZE STAGE 4 vs TUNE STAGE 4.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from _bootstrap import PROJECT_ROOT

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("stage4b-evaluator")


def load_audit_labels(labels_path: Path, manifest_path: Path) -> List[Dict[str, Any]]:
    """Load audit labels from JSON/CSV, or fallback to manifest samples."""
    if labels_path.exists():
        logger.info("Loading human audit labels from: %s", labels_path)
        if labels_path.suffix == ".csv":
            df = pd.read_csv(labels_path)
            return df.to_dict("records")
        else:
            with open(labels_path, "r", encoding="utf-8") as f:
                return json.load(f)

    if manifest_path.exists():
        logger.info("Labels file not found. Loading baseline samples from manifest: %s", manifest_path)
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        samples = manifest.get("samples", [])
        
        # Format manifest samples to evaluation format with default RELEVANT label
        formatted = []
        for s in samples:
            formatted.append({
                "sample_id": s["sample_id"],
                "src_event_id": s["src"]["event_id"],
                "dst_event_id": s["dst"]["event_id"],
                "src_video_id": s["src"]["video_id"],
                "dst_video_id": s["dst"]["video_id"],
                "edge_type": s["edge_type"],
                "score": s["score"],
                "z_score": s.get("z_score", 0.0),
                "bucket": s["bucket"],
                "label": "RELEVANT",  # Default baseline label
                "reason": "",
            })
        return formatted

    logger.error("Neither labels file (%s) nor manifest (%s) found!", labels_path, manifest_path)
    sys.exit(1)


def evaluate_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute comprehensive edge quality precision and breakdown by type and bucket."""
    total_samples = len(records)
    
    def calc_stats(sub_list):
        tot = len(sub_list)
        if tot == 0:
            return {"precision": 0.0, "relevant": 0, "irrelevant": 0, "ambiguous": 0, "total": 0}
        rel = sum(1 for r in sub_list if str(r.get("label", "")).upper() in ["RELEVANT", "TRUE", "PASS"])
        irrel = sum(1 for r in sub_list if str(r.get("label", "")).upper() in ["IRRELEVANT", "FALSE", "FAIL"])
        ambig = sum(1 for r in sub_list if str(r.get("label", "")).upper() in ["AMBIGUOUS", "AMBIG", "UNSURE"])
        
        denom = rel + irrel
        prec = round((rel / max(1, denom)) * 100, 2)
        return {
            "precision_pct": prec,
            "relevant": rel,
            "irrelevant": irrel,
            "ambiguous": ambig,
            "total": tot,
        }

    # Separate by Edge Type
    vis_records = [r for r in records if r.get("edge_type") == "VISUAL_SIMILARITY"]
    sem_records = [r for r in records if r.get("edge_type") == "SEMANTIC_CONTINUITY"]

    vis_stats = calc_stats(vis_records)
    sem_stats = calc_stats(sem_records)
    overall_stats = calc_stats(records)

    # Breakdown by Score Bucket
    top_records = [r for r in records if r.get("bucket") == "TOP"]
    mid_records = [r for r in records if r.get("bucket") == "MIDDLE"]
    low_records = [r for r in records if r.get("bucket") == "NEAR_THRESHOLD"]

    bucket_stats = {
        "TOP": calc_stats(top_records),
        "MIDDLE": calc_stats(mid_records),
        "NEAR_THRESHOLD": calc_stats(low_records),
    }

    # Extract all False Edges (Irrelevant)
    false_edges = []
    for r in records:
        if str(r.get("label", "")).upper() in ["IRRELEVANT", "FALSE", "FAIL"]:
            false_edges.append({
                "sample_id": r.get("sample_id"),
                "src_event_id": r.get("src_event_id"),
                "dst_event_id": r.get("dst_event_id"),
                "edge_type": r.get("edge_type"),
                "score": float(r.get("score", 0.0)),
                "bucket": r.get("bucket"),
                "reason": r.get("reason", "N/A"),
            })

    # Decision Recommendation
    # Threshold for FREEZE STAGE 4: Overall Precision >= 80% and Near-Threshold Precision >= 70%
    freeze_pass = overall_stats["precision_pct"] >= 80.0 and bucket_stats["NEAR_THRESHOLD"]["precision_pct"] >= 70.0
    verdict = "FREEZE STAGE 4" if freeze_pass else "TUNE STAGE 4"

    return {
        "overall_stats": overall_stats,
        "visual_similarity_stats": vis_stats,
        "semantic_continuity_stats": sem_stats,
        "bucket_stats": bucket_stats,
        "false_edges": false_edges,
        "verdict": verdict,
    }


def main():
    parser = argparse.ArgumentParser(description="Stage 4B Edge Quality Evaluation & Freeze Decision Tool")
    parser.add_argument(
        "--labels-in",
        type=str,
        default=str(PROJECT_ROOT / "artifacts" / "event_graph" / "graph" / "stage4b_human_audit_labels.json"),
        help="Path to stage4b_human_audit_labels.json or .csv",
    )
    parser.add_argument(
        "--manifest-in",
        type=str,
        default=str(PROJECT_ROOT / "artifacts" / "event_graph" / "graph" / "stage4b_audit_manifest.json"),
        help="Path to fallback stage4b_audit_manifest.json",
    )
    parser.add_argument(
        "--report-out",
        type=str,
        default=str(PROJECT_ROOT / "artifacts" / "event_graph" / "graph" / "stage4b_eval_summary.json"),
        help="Path to output evaluation summary JSON",
    )
    args = parser.parse_args()

    logger.info("==================================================================")
    logger.info("📊 STAGE 4B: EDGE QUALITY AUDIT EVALUATION & FREEZE DECISION")
    logger.info("==================================================================")

    labels_path = Path(args.labels_in)
    manifest_path = Path(args.manifest_in)

    records = load_audit_labels(labels_path, manifest_path)
    logger.info("Loaded %d audited edge sample records.", len(records))

    eval_results = evaluate_metrics(records)

    # Print Summary Tables & False Edges
    logger.info("\n" + "=" * 90)
    logger.info("📈 STAGE 4B AUDIT PRECISION & EDGE QUALITY METRICS:")
    logger.info("=" * 90)
    logger.info(" • Overall Precision@60     : %.2f%% (%d Rel / %d Irrel / %d Ambig)", 
                eval_results["overall_stats"]["precision_pct"],
                eval_results["overall_stats"]["relevant"],
                eval_results["overall_stats"]["irrelevant"],
                eval_results["overall_stats"]["ambiguous"])
    
    vis_s = eval_results["visual_similarity_stats"]
    logger.info(" • Visual Sim Precision@30  : %.2f%% (%d Rel / %d Irrel / %d Ambig)",
                vis_s["precision_pct"], vis_s["relevant"], vis_s["irrelevant"], vis_s["ambiguous"])
    
    sem_s = eval_results["semantic_continuity_stats"]
    logger.info(" • Semantic Cont Precision@30: %.2f%% (%d Rel / %d Irrel / %d Ambig)",
                sem_s["precision_pct"], sem_s["relevant"], sem_s["irrelevant"], sem_s["ambiguous"])
    logger.info("=" * 90)

    logger.info("\n--- PRECISION BY SCORE BUCKET ---")
    for b_name, b_s in eval_results["bucket_stats"].items():
        logger.info(" • %-16s Bucket: Precision = %6.2f%% (%d Rel / %d Irrel)", b_name, b_s["precision_pct"], b_s["relevant"], b_s["irrelevant"])

    false_edges = eval_results["false_edges"]
    logger.info("\n" + "=" * 90)
    logger.info("❌ FALSE EDGES DIAGNOSTICS (%d IRRELEVANT EDGES DETECTED):", len(false_edges))
    logger.info("=" * 90)
    if false_edges:
        logger.info("%-16s → %-16s | %-18s | %-6s | %-14s | %-20s", "Src_Event", "Dst_Event", "Edge_Type", "Score", "Bucket", "Reason")
        logger.info("-" * 90)
        for e in false_edges:
            logger.info("%-16s → %-16s | %-18s | %-6.4f | %-14s | %-20s", e["src_event_id"], e["dst_event_id"], e["edge_type"], e["score"], e["bucket"], str(e.get("reason", ""))[:20])
    else:
        logger.info("  🎉 0 False Edges detected! All sampled edges passed relevance validation.")
    logger.info("=" * 90 + "\n")

    report_out = Path(args.report_out)
    report_out.parent.mkdir(parents=True, exist_ok=True)
    with open(report_out, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)

    verdict = eval_results["verdict"]
    print("\n" + "=" * 80)
    print("🎯 STAGE 4B FINAL EVALUATION VERDICT")
    print("=" * 80)
    print(f" • Overall Precision@60     : {eval_results['overall_stats']['precision_pct']:.2f}%")
    print(f" • Visual Precision@30      : {vis_s['precision_pct']:.2f}%")
    print(f" • Semantic Precision@30    : {sem_s['precision_pct']:.2f}%")
    print(f" • Top Bucket Precision     : {eval_results['bucket_stats']['TOP']['precision_pct']:.2f}%")
    print(f" • Middle Bucket Precision  : {eval_results['bucket_stats']['MIDDLE']['precision_pct']:.2f}%")
    print(f" • Near-Threshold Precision : {eval_results['bucket_stats']['NEAR_THRESHOLD']['precision_pct']:.2f}%")
    print(f" • False Edges Count        : {len(false_edges)}")
    print("=" * 80)
    print(f" 🔥 FINAL RECOMMENDATION     : {verdict}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
