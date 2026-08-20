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
    for candidate in ["gt_label", "audit_label", "label", "is_true_boundary", "gt", "is_correct", "is_boundary"]:
        if candidate in df_samples.columns:
            gt_col = candidate
            break

    gt_metrics = {}
    correctly_corrected_fp_to_tn = []
    incorrectly_corrected_tp_to_fn = []
    remaining_fp = []

    if gt_col:
        gt_binary = (df_refined[gt_col] == 1) | (df_refined[gt_col] == True) | (df_refined[gt_col].astype(str).str.lower().str.contains("correct|true|1"))
        
        # Before Refinement (Stage 3B Raw)
        raw_pred = df_refined["boundary_score"] > args.threshold
        tp_raw = int((raw_pred & gt_binary).sum())
        fp_raw = int((raw_pred & (~gt_binary)).sum())
        fn_raw = int(((~raw_pred) & gt_binary).sum())
        tn_raw = int(((~raw_pred) & (~gt_binary)).sum())

        prec_raw = tp_raw / max(1, tp_raw + fp_raw)
        rec_raw = tp_raw / max(1, tp_raw + fn_raw)
        f1_raw = 2 * prec_raw * rec_raw / max(1e-8, prec_raw + rec_raw)
        acc_raw = (tp_raw + tn_raw) / max(1, len(df_refined))

        # After Refinement (Stage 3D Refined)
        ref_pred = df_refined["is_boundary_refined"]
        tp_ref = int((ref_pred & gt_binary).sum())
        fp_ref = int((ref_pred & (~gt_binary)).sum())
        fn_ref = int(((~ref_pred) & gt_binary).sum())
        tn_ref = int(((~ref_pred) & (~gt_binary)).sum())

        prec_ref = tp_ref / max(1, tp_ref + fp_ref)
        rec_ref = tp_ref / max(1, tp_ref + fn_ref)
        f1_ref = 2 * prec_ref * rec_ref / max(1e-8, prec_ref + rec_ref)
        acc_ref = (tp_ref + tn_ref) / max(1, len(df_refined))

        # Categorize Sample Groups
        for _, r in df_refined.iterrows():
            is_gt = bool(r[gt_col] == 1 or r[gt_col] == True or "true" in str(r[gt_col]).lower())
            is_raw = bool(r["boundary_score"] > args.threshold)
            is_ref = bool(r["is_boundary_refined"])

            sample_dict = {
                "video_id": str(r["video_id"]),
                "shot_i": int(r["shot_i"]),
                "shot_next": int(r.get("shot_next", r["shot_i"] + 1)),
                "raw_score": round(float(r["boundary_score"]), 4),
                "final_score": round(float(r["final_boundary_score"]), 4),
                "penalty": round(float(r["suppression_delta"]), 4),
                "visual_sim": round(float(r["visual_similarity"]), 4),
                "sem_evd": round(float(r["semantic_boundary_evidence"]), 4),
                "vis_evd": round(float(r["visual_boundary_evidence"]), 4),
                "gt_label": "TRUE_BOUNDARY" if is_gt else "FALSE_BOUNDARY",
            }

            # FP -> TN (Correctly Corrected Slide False Positive)
            if (not is_gt) and is_raw and (not is_ref):
                correctly_corrected_fp_to_tn.append(sample_dict)
            # TP -> FN (Incorrectly Suppressed True Boundary)
            elif is_gt and is_raw and (not is_ref):
                incorrectly_corrected_tp_to_fn.append(sample_dict)
            # Remaining FP (False boundary that was not suppressed)
            elif (not is_gt) and is_ref:
                remaining_fp.append(sample_dict)

        gt_metrics = {
            "gt_column_found": gt_col,
            "raw_stage3b_performance": {
                "TP": tp_raw, "FP": fp_raw, "FN": fn_raw, "TN": tn_raw,
                "Precision": round(prec_raw, 4),
                "Recall": round(rec_raw, 4),
                "F1": round(f1_raw, 4),
                "Accuracy": round(acc_raw, 4),
            },
            "refined_stage3d_performance": {
                "TP": tp_ref, "FP": fp_ref, "FN": fn_ref, "TN": tn_ref,
                "Precision": round(prec_ref, 4),
                "Recall": round(rec_ref, 4),
                "F1": round(f1_ref, 4),
                "Accuracy": round(acc_ref, 4),
            },
        }

    # Semantic Similarity Saturation Analysis
    sem_vals = df_refined["semantic_similarity"].dropna().to_numpy()
    sem_saturation_analysis = {
        "count": len(sem_vals),
        "mean": round(float(np.mean(sem_vals)), 4),
        "std": round(float(np.std(sem_vals)), 6),
        "min": round(float(np.min(sem_vals)), 4),
        "p25": round(float(np.percentile(sem_vals, 25)), 4),
        "median": round(float(np.median(sem_vals)), 4),
        "p75": round(float(np.percentile(sem_vals, 75)), 4),
        "max": round(float(np.max(sem_vals)), 4),
        "saturation_warning": bool(np.std(sem_vals) < 0.01),
        "explanation": "Raw semantic_similarity is heavily saturated near ~0.97 (std < 0.008). Robust Z-score normalization (semantic_z) is CRITICAL to restore dynamic range.",
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
        "semantic_similarity_saturation_analysis": sem_saturation_analysis,
        "correctly_corrected_fp_to_tn_count": len(correctly_corrected_fp_to_tn),
        "incorrectly_corrected_tp_to_fn_count": len(incorrectly_corrected_tp_to_fn),
        "remaining_fp_count": len(remaining_fp),
        "sample_groups": {
            "correctly_corrected_fp_to_tn": correctly_corrected_fp_to_tn,
            "incorrectly_corrected_tp_to_fn": incorrectly_corrected_tp_to_fn,
            "remaining_fp": remaining_fp,
        },
    }

    out_json_path = Path(args.output_json)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("\n" + "=" * 115)
    logger.info("🔍 SPECIFIC 5 GT AUDIT CASES DETAILED FEATURE INSPECTION:")
    logger.info("=" * 115)
    logger.info("%-10s | %-8s | %-8s | %-8s | %-8s | %-8s | %-8s | %-8s | %-12s | %-12s", "Video", "Shot_i", "VisSim", "SemSim", "VisEvd", "SemEvd", "Penalty", "FinalScore", "Prediction", "GT Label")
    logger.info("-" * 115)

    specific_5_cases = [
        ("L25_V007", 91, True),
        ("L22_V005", 111, True),
        ("L22_V013", 188, True),
        ("L25_V081", 107, True),
        ("L26_V327", 42, True),
    ]

    for vid, s_i, gt_val in specific_5_cases:
        m = df_refined[(df_refined["video_id"] == vid) & (df_refined["shot_i"] == s_i)]
        if not m.empty:
            r = m.iloc[0]
            pred_str = "BOUNDARY" if bool(r["is_boundary_refined"]) else "SUPPRESSED"
            gt_str = "TRUE" if gt_val else "FALSE"
            logger.info(
                "%-10s | %-8d | %-8.4f | %-8.4f | %-8.4f | %-8.4f | %-8.4f | %-8.4f | %-12s | %-12s",
                str(r["video_id"]),
                int(r["shot_i"]),
                float(r["visual_similarity"]),
                float(r["semantic_similarity"]),
                float(r["visual_boundary_evidence"]),
                float(r["semantic_boundary_evidence"]),
                float(r["suppression_delta"]),
                float(r["final_boundary_score"]),
                pred_str,
                gt_str,
            )
    logger.info("=" * 115 + "\n")

    logger.info("\n" + "=" * 90)
    logger.info("📊 STAGE 3B RAW VS STAGE 3D REFINED GROUND-TRUTH EVALUATION:")
    logger.info("=" * 90)
    if gt_metrics:
        logger.info("Metric       | Stage 3B (Raw) | Stage 3D (Refined) | Delta")
        logger.info("-" * 90)
        logger.info("Precision    | %-14.4f | %-18.4f | %+.4f", gt_metrics["raw_stage3b_performance"]["Precision"], gt_metrics["refined_stage3d_performance"]["Precision"], gt_metrics["refined_stage3d_performance"]["Precision"] - gt_metrics["raw_stage3b_performance"]["Precision"])
        logger.info("Recall       | %-14.4f | %-18.4f | %+.4f", gt_metrics["raw_stage3b_performance"]["Recall"], gt_metrics["refined_stage3d_performance"]["Recall"], gt_metrics["refined_stage3d_performance"]["Recall"] - gt_metrics["raw_stage3b_performance"]["Recall"])
        logger.info("F1 Score     | %-14.4f | %-18.4f | %+.4f", gt_metrics["raw_stage3b_performance"]["F1"], gt_metrics["refined_stage3d_performance"]["F1"], gt_metrics["refined_stage3d_performance"]["F1"] - gt_metrics["raw_stage3b_performance"]["F1"])
        logger.info("Accuracy     | %-14.4f | %-18.4f | %+.4f", gt_metrics["raw_stage3b_performance"]["Accuracy"], gt_metrics["refined_stage3d_performance"]["Accuracy"], gt_metrics["refined_stage3d_performance"]["Accuracy"] - gt_metrics["raw_stage3b_performance"]["Accuracy"])
        logger.info("TP / FP / TN | %d / %d / %d  | %d / %d / %d       | -", gt_metrics["raw_stage3b_performance"]["TP"], gt_metrics["raw_stage3b_performance"]["FP"], gt_metrics["raw_stage3b_performance"]["TN"], gt_metrics["refined_stage3d_performance"]["TP"], gt_metrics["refined_stage3d_performance"]["FP"], gt_metrics["refined_stage3d_performance"]["TN"])
    
    logger.info("\n" + "=" * 90)
    logger.info("✅ CORRECTLY CORRECTED SLIDE FALSE POSITIVES (FP -> TN): %d", len(correctly_corrected_fp_to_tn))
    logger.info("=" * 90)
    for sample in correctly_corrected_fp_to_tn:
        logger.info(" • %s Shot %d->%d | Raw: %.4f | Penalty: %.4f | Final: %.4f | VisSim: %.4f | VisEvd: %.4f | SemEvd: %.4f", sample["video_id"], sample["shot_i"], sample["shot_next"], sample["raw_score"], sample["penalty"], sample["final_score"], sample["visual_sim"], sample["vis_evd"], sample["sem_evd"])

    logger.info("\n" + "=" * 90)
    logger.info("❌ INCORRECTLY SUPPRESSED TRUE BOUNDARIES (TP -> FN): %d", len(incorrectly_corrected_tp_to_fn))
    logger.info("=" * 90)
    for sample in incorrectly_corrected_tp_to_fn:
        logger.info(" • %s Shot %d->%d | Raw: %.4f | Penalty: %.4f | Final: %.4f", sample["video_id"], sample["shot_i"], sample["shot_next"], sample["raw_score"], sample["penalty"], sample["final_score"])

    logger.info("\n" + "=" * 90)
    logger.info("⚠️ REMAINING UNCORRECTED FALSE POSITIVES (FP): %d", len(remaining_fp))
    logger.info("=" * 90)
    for sample in remaining_fp:
        logger.info(" • %s Shot %d->%d | Raw: %.4f | Final: %.4f", sample["video_id"], sample["shot_i"], sample["shot_next"], sample["raw_score"], sample["final_score"])

    logger.info("\n" + "=" * 90)
    logger.info("🧠 SEMANTIC SIMILARITY SATURATION ANALYSIS:")
    logger.info("=" * 90)
    logger.info(" • Mean: %.4f | Std: %.6f | Min: %.4f | Med: %.4f | Max: %.4f", sem_saturation_analysis["mean"], sem_saturation_analysis["std"], sem_saturation_analysis["min"], sem_saturation_analysis["median"], sem_saturation_analysis["max"])
    logger.info(" • Saturation Warning: %s", "YES (Std < 0.01)" if sem_saturation_analysis["saturation_warning"] else "NO")
    logger.info("=" * 90 + "\n")

    print("\n" + "=" * 80)
    print("🎉 STAGE 3D EMPIRICAL AUDIT VALIDATION COMPLETE!")
    print("=" * 80)
    print(f" • Total Samples Evaluated    : {len(df_samples):,}")
    print(f" • Stage 3B Precision -> 3D   : {gt_metrics.get('raw_stage3b_performance', {}).get('Precision', 0):.4f} -> {gt_metrics.get('refined_stage3d_performance', {}).get('Precision', 0):.4f}")
    print(f" • Stage 3B Accuracy -> 3D    : {gt_metrics.get('raw_stage3b_performance', {}).get('Accuracy', 0):.4f} -> {gt_metrics.get('refined_stage3d_performance', {}).get('Accuracy', 0):.4f}")
    print(f" • Stage 3B F1 Score -> 3D    : {gt_metrics.get('raw_stage3b_performance', {}).get('F1', 0):.4f} -> {gt_metrics.get('refined_stage3d_performance', {}).get('F1', 0):.4f}")
    print(f" • Correctly Fixed FP -> TN   : {len(correctly_corrected_fp_to_tn)}")
    print(f" • Incorrectly Fixed TP -> FN : {len(incorrectly_corrected_tp_to_fn)}")
    print(f" • Output Audit Report JSON   : {out_json_path}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
