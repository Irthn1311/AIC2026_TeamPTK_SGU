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


def load_audit_labels(labels_path: Path, manifest_path: Path) -> Tuple[List[Dict[str, Any]], bool]:
    """
    Load audit labels from JSON/CSV. Returns (records, labels_file_exists).
    Does NOT default missing labels to RELEVANT.
    """
    if labels_path.exists():
        logger.info("Loading human audit labels from: %s", labels_path)
        if labels_path.suffix == ".csv":
            df = pd.read_csv(labels_path)
            return df.to_dict("records"), True
        else:
            with open(labels_path, "r", encoding="utf-8") as f:
                return json.load(f), True

    if manifest_path.exists():
        logger.info("Labels file NOT found (%s). Loading sample metadata from manifest: %s", labels_path, manifest_path)
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        samples = manifest.get("samples", [])
        
        # Load metadata with UNLABELED status
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
                "label": "UNLABELED",  # Strictly UNLABELED
                "reason": "",
            })
        return formatted, False

    logger.error("Neither labels file (%s) nor manifest (%s) found!", labels_path, manifest_path)
    sys.exit(1)


def evaluate_metrics(records: List[Dict[str, Any]], labels_file_exists: bool) -> Dict[str, Any]:
    """
    Compute edge quality precision ONLY on samples with explicit human labels.
    Returns AUDIT_INCOMPLETE if labels file is missing or coverage < 100%.
    """
    total_sampled = len(records)
    
    # Identify labeled vs unlabeled
    valid_labels = ["RELEVANT", "IRRELEVANT", "AMBIGUOUS", "TRUE", "FALSE", "PASS", "FAIL", "AMBIG", "UNSURE"]
    
    labeled_records = []
    unlabeled_records = []
    for r in records:
        lbl_str = str(r.get("label", "")).strip().upper()
        if lbl_str in valid_labels:
            labeled_records.append(r)
        else:
            unlabeled_records.append(r)

    labeled_count = len(labeled_records)
    unlabeled_count = len(unlabeled_records)
    coverage_pct = round((labeled_count / max(1, total_sampled)) * 100, 2)

    # Sanity check: If labels file missing OR incomplete coverage => AUDIT_INCOMPLETE
    is_complete = labels_file_exists and (labeled_count >= total_sampled) and (total_sampled > 0)

    if not is_complete:
        verdict = "AUDIT_INCOMPLETE"
        logger.warning("Audit Incomplete: %d / %d samples labeled (Coverage: %.2f%%). FREEZE STAGE 4 BLOCKED.", labeled_count, total_sampled, coverage_pct)
        return {
            "is_complete": False,
            "verdict": verdict,
            "freeze_allowed": False,
            "total_sampled": total_sampled,
            "labeled_count": labeled_count,
            "unlabeled_count": unlabeled_count,
            "label_coverage_pct": coverage_pct,
            "overall_stats": {"precision_pct": 0.0, "relevant": 0, "irrelevant": 0, "ambiguous": 0, "total": labeled_count},
            "visual_similarity_stats": {"precision_pct": 0.0, "relevant": 0, "irrelevant": 0, "ambiguous": 0, "total": 0},
            "semantic_continuity_stats": {"precision_pct": 0.0, "relevant": 0, "irrelevant": 0, "ambiguous": 0, "total": 0},
            "bucket_stats": {},
            "false_edges": [],
        }

    # Complete coverage calculation
    def calc_stats(sub_list):
        tot = len(sub_list)
        if tot == 0:
            return {"precision_pct": 0.0, "relevant": 0, "irrelevant": 0, "ambiguous": 0, "total": 0}
        rel = sum(1 for r in sub_list if str(r.get("label", "")).upper() in ["RELEVANT", "TRUE", "PASS"])
        irrel = sum(1 for r in sub_list if str(r.get("label", "")).upper() in ["IRRELEVANT", "FALSE", "FAIL"])
        ambig = sum(1 for r in sub_list if str(r.get("label", "")).upper() in ["AMBIGUOUS", "AMBIG", "UNSURE"])
        
        denom = rel + irrel
        prec = round((rel / max(1, denom)) * 100, 2) if denom > 0 else 0.0
        return {
            "precision_pct": prec,
            "relevant": rel,
            "irrelevant": irrel,
            "ambiguous": ambig,
            "total": tot,
        }

    vis_records = [r for r in labeled_records if r.get("edge_type") == "VISUAL_SIMILARITY"]
    sem_records = [r for r in labeled_records if r.get("edge_type") == "SEMANTIC_CONTINUITY"]

    vis_stats = calc_stats(vis_records)
    sem_stats = calc_stats(sem_records)
    overall_stats = calc_stats(labeled_records)

    top_records = [r for r in labeled_records if r.get("bucket") == "TOP"]
    mid_records = [r for r in labeled_records if r.get("bucket") == "MIDDLE"]
    low_records = [r for r in labeled_records if r.get("bucket") == "NEAR_THRESHOLD"]

    bucket_stats = {
        "TOP": calc_stats(top_records),
        "MIDDLE": calc_stats(mid_records),
        "NEAR_THRESHOLD": calc_stats(low_records),
    }

    false_edges = []
    ambiguous_edges = []
    for r in labeled_records:
        lbl = str(r.get("label", "")).upper()
        if lbl in ["IRRELEVANT", "FALSE", "FAIL"]:
            false_edges.append({
                "sample_id": r.get("sample_id"),
                "src_event_id": r.get("src_event_id"),
                "dst_event_id": r.get("dst_event_id"),
                "edge_type": r.get("edge_type"),
                "score": float(r.get("score", 0.0)),
                "bucket": r.get("bucket"),
                "reason": r.get("reason", "N/A"),
            })
        elif lbl in ["AMBIGUOUS", "AMBIG", "UNSURE"]:
            ambiguous_edges.append({
                "sample_id": r.get("sample_id"),
                "src_event_id": r.get("src_event_id"),
                "dst_event_id": r.get("dst_event_id"),
                "edge_type": r.get("edge_type"),
                "score": float(r.get("score", 0.0)),
                "bucket": r.get("bucket"),
            })

    # Threshold for FREEZE STAGE 4: Overall Precision >= 80% and Near-Threshold Precision >= 70%
    freeze_pass = overall_stats["precision_pct"] >= 80.0 and bucket_stats["NEAR_THRESHOLD"]["precision_pct"] >= 70.0
    verdict = "FREEZE STAGE 4" if freeze_pass else "TUNE STAGE 4"

    # Derive tuning recommendations if needed
    tuning_recommendations = []
    if vis_stats["precision_pct"] < 80.0:
        tuning_recommendations.append("Visual Precision < 80%: Increase --visual-threshold to 0.75 or reduce --top-k-visual to 5")
    if sem_stats["precision_pct"] < 80.0:
        tuning_recommendations.append("Semantic Precision < 80%: Increase --semantic-min-z to 2.0 or --semantic-threshold to 0.80")
    if bucket_stats["NEAR_THRESHOLD"]["precision_pct"] < 70.0:
        tuning_recommendations.append("Near-Threshold Precision < 70%: Tighten lower bound thresholds for similarity edge creation")

    return {
        "is_complete": True,
        "verdict": verdict,
        "freeze_allowed": freeze_pass,
        "total_sampled": total_sampled,
        "labeled_count": labeled_count,
        "unlabeled_count": unlabeled_count,
        "label_coverage_pct": coverage_pct,
        "overall_stats": overall_stats,
        "visual_similarity_stats": vis_stats,
        "semantic_continuity_stats": sem_stats,
        "bucket_stats": bucket_stats,
        "false_edges": false_edges,
        "ambiguous_edges": ambiguous_edges,
        "tuning_recommendations": tuning_recommendations,
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

    records, labels_file_exists = load_audit_labels(labels_path, manifest_path)
    eval_results = evaluate_metrics(records, labels_file_exists)

    # Sanity Assertion Check
    if not labels_file_exists or not eval_results["is_complete"]:
        assert eval_results["freeze_allowed"] is False, "Sanity Check Failed: Freeze must NOT be allowed when labels are missing!"
        assert eval_results["verdict"] == "AUDIT_INCOMPLETE", "Sanity Check Failed: Verdict must be AUDIT_INCOMPLETE!"

    logger.info("\n" + "=" * 90)
    logger.info("📈 STAGE 4B AUDIT PROGRESS & LABEL COVERAGE:")
    logger.info("=" * 90)
    logger.info(" • Labels File Exists       : %s", labels_file_exists)
    logger.info(" • Total Samples Needed    : %d", eval_results["total_sampled"])
    logger.info(" • Labeled Samples Count   : %d", eval_results["labeled_count"])
    logger.info(" • Unlabeled Samples Count : %d", eval_results["unlabeled_count"])
    logger.info(" • Label Coverage           : %.2f%%", eval_results["label_coverage_pct"])
    logger.info("=" * 90)

    if eval_results["is_complete"]:
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
        ambiguous_edges = eval_results["ambiguous_edges"]

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
        logger.info("=" * 90)

        logger.info("\n" + "=" * 90)
        logger.info("❓ AMBIGUOUS EDGES DIAGNOSTICS (%d AMBIGUOUS EDGES DETECTED):", len(ambiguous_edges))
        logger.info("=" * 90)
        if ambiguous_edges:
            logger.info("%-16s → %-16s | %-18s | %-6s | %-14s", "Src_Event", "Dst_Event", "Edge_Type", "Score", "Bucket")
            logger.info("-" * 90)
            for e in ambiguous_edges:
                logger.info("%-16s → %-16s | %-18s | %-6.4f | %-14s", e["src_event_id"], e["dst_event_id"], e["edge_type"], e["score"], e["bucket"])
        else:
            logger.info("  🎉 0 Ambiguous Edges detected!")
        logger.info("=" * 90 + "\n")

    report_out = Path(args.report_out)
    report_out.parent.mkdir(parents=True, exist_ok=True)
    with open(report_out, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)

    verdict = eval_results["verdict"]
    print("\n" + "=" * 80)
    print("🎯 STAGE 4B FINAL EVALUATION VERDICT")
    print("=" * 80)
    print(f" • Label File Exists        : {labels_file_exists}")
    print(f" • Label Coverage           : {eval_results['label_coverage_pct']:.2f}% ({eval_results['labeled_count']}/{eval_results['total_sampled']})")
    if eval_results["is_complete"]:
        print(f" • Overall Precision@60     : {eval_results['overall_stats']['precision_pct']:.2f}%")
        print(f" • Visual Precision@30      : {eval_results['visual_similarity_stats']['precision_pct']:.2f}%")
        print(f" • Semantic Precision@30    : {eval_results['semantic_continuity_stats']['precision_pct']:.2f}%")
        print(f" • Top Bucket Precision     : {eval_results['bucket_stats']['TOP']['precision_pct']:.2f}%")
        print(f" • Middle Bucket Precision  : {eval_results['bucket_stats']['MIDDLE']['precision_pct']:.2f}%")
        print(f" • Near-Threshold Precision : {eval_results['bucket_stats']['NEAR_THRESHOLD']['precision_pct']:.2f}%")
        print(f" • False Edges Count        : {len(eval_results['false_edges'])}")
        print(f" • Ambiguous Edges Count    : {len(eval_results['ambiguous_edges'])}")
        if eval_results.get("tuning_recommendations"):
            print("\n 💡 TUNING RECOMMENDATIONS:")
            for rec in eval_results["tuning_recommendations"]:
                print(f"   👉 {rec}")
    else:
        print(" ⚠️  Precision Calculation   : BLOCKED (Requires 100% human ground truth labels)")
    print("=" * 80)
    print(f" 🔥 FINAL RECOMMENDATION     : {verdict}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
