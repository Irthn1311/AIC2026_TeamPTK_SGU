"""
Stage 3D Validation & Empirical Refinement Audit Script (AI Challenge 2026 EventGraph)
========================================================================================
Validates Stage 3D Multimodal Refinement logic against boundary sample datasets (boundary_samples_150.csv).

Key Verification Metrics:
1. Evaluates raw boundary_score vs final_boundary_score.
2. Quantifies Slide / Infographic False Positive suppression rate.
3. Computes precision and boundary confidence distribution before and after Stage 3D refinement.
4. Generates an empirical verification report.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import logging

PROJECT_ROOT = Path(__file__).resolve().parent.parent
from build_stage3d_event_construction import refine_boundary_scores

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("validate-stage3d-refinement")


def main():
    parser = argparse.ArgumentParser(description="Validate Stage 3D Boundary Refinement on Audit Samples")
    parser.add_argument(
        "--input-csv",
        type=str,
        default=str(PROJECT_ROOT / "boundary_samples_150.csv"),
        help="Path to boundary samples CSV",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=str(PROJECT_ROOT / "artifacts" / "event_graph" / "events" / "stage3d_validation_audit.json"),
        help="Path to save validation output report",
    )
    parser.add_argument("--threshold", type=float, default=0.70, help="Boundary score classification threshold")
    parser.add_argument("--suppression-weight", type=float, default=0.35, help="Slide FP suppression weight")
    args = parser.parse_args()

    logger.info("==================================================================")
    logger.info("🧪 STAGE 3D BOUNDARY REFINEMENT EMPIRICAL AUDIT VALIDATOR")
    logger.info("==================================================================")

    csv_path = Path(args.input_csv)
    if not csv_path.exists():
        logger.error("Input CSV not found: %s", csv_path)
        sys.exit(1)

    logger.info("Loading boundary samples from: %s", csv_path)
    df_samples = pd.read_csv(csv_path)
    logger.info("Loaded %d boundary samples.", len(df_samples))

    # Apply Stage 3D Refinement
    df_refined = refine_boundary_scores(
        df_samples,
        threshold=args.threshold,
        suppression_weight=args.suppression_weight,
    )

    # Analyze changes between raw boundary_score and final_boundary_score
    raw_boundaries = df_refined["boundary_score"] > args.threshold
    refined_boundaries = df_refined["is_boundary_refined"]

    raw_count = int(raw_boundaries.sum())
    refined_count = int(refined_boundaries.sum())
    suppressed_mask = raw_boundaries & (~refined_boundaries)
    suppressed_count = int(suppressed_mask.sum())

    suppressed_rows = df_refined[suppressed_mask]

    logger.info("Raw Positive Boundaries (>%.2f): %d / %d", args.threshold, raw_count, len(df_samples))
    logger.info("Refined Positive Boundaries (>%.2f): %d / %d", args.threshold, refined_count, len(df_samples))
    logger.info("Suppressed Slide/Infographic False Positives: %d (%.2f%%)", suppressed_count, (suppressed_count / max(1, raw_count)) * 100)

    # Compute GT Evaluation Metrics if Ground-Truth column exists
    gt_col = None
    for candidate in ["gt_label", "audit_label", "label", "is_true_boundary", "gt", "is_correct"]:
        if candidate in df_samples.columns:
            gt_col = candidate
            break

    gt_metrics = {}
    if gt_col:
        gt_binary = (df_refined[gt_col] == 1) | (df_refined[gt_col] == True) | (df_refined[gt_col].astype(str).str.lower().str.contains("correct|true|1"))
        
        # Before Refinement
        raw_pred = df_refined["boundary_score"] > args.threshold
        tp_raw = int((raw_pred & gt_binary).sum())
        fp_raw = int((raw_pred & (~gt_binary)).sum())
        fn_raw = int(((~raw_pred) & gt_binary).sum())
        tn_raw = int(((~raw_pred) & (~gt_binary)).sum())

        prec_raw = tp_raw / max(1, tp_raw + fp_raw)
        rec_raw = tp_raw / max(1, tp_raw + fn_raw)
        f1_raw = 2 * prec_raw * rec_raw / max(1e-8, prec_raw + rec_raw)

        # After Refinement
        ref_pred = df_refined["is_boundary_refined"]
        tp_ref = int((ref_pred & gt_binary).sum())
        fp_ref = int((ref_pred & (~gt_binary)).sum())
        fn_ref = int(((~ref_pred) & gt_binary).sum())
        tn_ref = int(((~ref_pred) & (~gt_binary)).sum())

        prec_ref = tp_ref / max(1, tp_ref + fp_ref)
        rec_ref = tp_ref / max(1, tp_ref + fn_ref)
        f1_ref = 2 * prec_ref * rec_ref / max(1e-8, prec_ref + rec_ref)

        gt_metrics = {
            "gt_column_found": gt_col,
            "raw_performance": {
                "TP": tp_raw, "FP": fp_raw, "FN": fn_raw, "TN": tn_raw,
                "Precision": round(prec_raw, 4),
                "Recall": round(rec_raw, 4),
                "F1": round(f1_raw, 4),
            },
            "refined_performance": {
                "TP": tp_ref, "FP": fp_ref, "FN": fn_ref, "TN": tn_ref,
                "Precision": round(prec_ref, 4),
                "Recall": round(rec_ref, 4),
                "F1": round(f1_ref, 4),
            },
        }

    # Detailed report breakdown
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_samples": len(df_samples),
        "threshold": args.threshold,
        "suppression_weight": args.suppression_weight,
        "raw_positive_boundaries": raw_count,
        "refined_positive_boundaries": refined_count,
        "suppressed_false_positives_count": suppressed_count,
        "suppression_rate_percentage": round((suppressed_count / max(1, raw_count)) * 100, 2),
        "ground_truth_metrics": gt_metrics,
        "raw_score_stats": {
            "mean": round(float(df_refined["boundary_score"].mean()), 4),
            "median": round(float(df_refined["boundary_score"].median()), 4),
            "std": round(float(df_refined["boundary_score"].std()), 4),
        },
        "refined_score_stats": {
            "mean": round(float(df_refined["final_boundary_score"].mean()), 4),
            "median": round(float(df_refined["final_boundary_score"].median()), 4),
            "std": round(float(df_refined["final_boundary_score"].std()), 4),
        },
        "sample_suppressed_boundaries": [
            {
                "video_id": str(r["video_id"]),
                "shot_i": int(r["shot_i"]),
                "shot_next": int(r["shot_next"]),
                "raw_boundary_score": round(float(r["boundary_score"]), 4),
                "final_boundary_score": round(float(r["final_boundary_score"]), 4),
                "suppression_delta": round(float(r["suppression_delta"]), 4),
                "layout_constancy": round(float(r["layout_constancy"]), 4),
            }
            for _, r in suppressed_rows.head(10).iterrows()
        ],
    }

    out_json_path = Path(args.output_json)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 80)
    print("🎉 STAGE 3D VALIDATION AUDIT COMPLETE!")
    print("=" * 80)
    print(f" • Total Samples Evaluated    : {len(df_samples):,}")
    print(f" • Raw Positive Boundaries    : {raw_count}")
    print(f" • Refined Positive Boundaries: {refined_count}")
    print(f" • Suppressed False Positives  : {suppressed_count} (Slide / Infographic transitions)")
    print(f" • Mean Raw Boundary Score    : {report['raw_score_stats']['mean']:.4f}")
    print(f" • Mean Refined Score         : {report['refined_score_stats']['mean']:.4f}")
    if gt_metrics:
        print(f" • GT Ground-Truth Evaluated  : Column '{gt_col}'")
        print(f"   - BEFORE Refinement: Precision={gt_metrics['raw_performance']['Precision']:.4f}, Recall={gt_metrics['raw_performance']['Recall']:.4f}, F1={gt_metrics['raw_performance']['F1']:.4f} (TP={gt_metrics['raw_performance']['TP']}, FP={gt_metrics['raw_performance']['FP']})")
        print(f"   - AFTER Refinement : Precision={gt_metrics['refined_performance']['Precision']:.4f}, Recall={gt_metrics['refined_performance']['Recall']:.4f}, F1={gt_metrics['refined_performance']['F1']:.4f} (TP={gt_metrics['refined_performance']['TP']}, FP={gt_metrics['refined_performance']['FP']})")
    print(f" • Output Audit Report        : {out_json_path}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
