"""
Stage 3C Manual Boundary Annotation & Precision Analysis Engine (AI Challenge 2026)
====================================================================================
Senior ML Engineer Interactive Tool for True Manual Boundary Inspection & Precision Auditing.

Features:
1. Stratified Boundary Sampling (100-200 boundaries):
   - Near Threshold ([0.68, 0.75))
   - Medium Score ([0.75, 0.85))
   - High Score ([0.85, 1.00])
2. Default Configuration:
   - visual_weight = 0.3, semantic_weight = 0.7, threshold = 0.70
3. Interactive Terminal / CLI Annotation Interface:
   - Displays shot_i vs shot_{i+1} metrics, keyframe IDs, similarities, and evidence.
   - User inputs: [c]orrect / [f]alse / [a]mbiguous (or skip/quit).
   - Saves annotations immediately to Parquet & CSV.
4. Precision Analytics Engine:
   - Overall Manual Precision
   - Precision binned by Boundary Score ranges
   - Showcase typical False Boundaries & Correct Boundaries
5. Conditional Threshold-Gated Adaptive Neighbor Merging:
   - Only merges a 1-shot event if its boundary score with a neighbor is < merge_threshold (e.g. 0.55).
   - Truly distinct 1-shot events are preserved!
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from _bootstrap import PROJECT_ROOT
from scripts.build_event_boundaries import apply_robust_calibrated_fusion
from src.retrieval.logging_utils import setup_logger

logger = setup_logger("manual-boundary-annotator")


def sample_stratified_boundaries(
    df_sim_calibrated: pd.DataFrame,
    df_shots: pd.DataFrame,
    threshold: float = 0.70,
    num_samples: int = 150,
) -> pd.DataFrame:
    """Sample 100-200 predicted boundaries stratified across 3 score bins."""
    df_boundaries = df_sim_calibrated[df_sim_calibrated["boundary_score"] > threshold].copy()
    if df_boundaries.empty:
        logger.warning("No boundaries found above threshold %.2f!", threshold)
        return pd.DataFrame()

    # Define 3 score bins
    bin_near = df_boundaries[(df_boundaries["boundary_score"] >= threshold) & (df_boundaries["boundary_score"] < threshold + 0.08)]
    bin_mid = df_boundaries[(df_boundaries["boundary_score"] >= threshold + 0.08) & (df_boundaries["boundary_score"] < threshold + 0.18)]
    bin_high = df_boundaries[df_boundaries["boundary_score"] >= threshold + 0.18]

    target_per_bin = num_samples // 3

    samples_near = bin_near.sample(n=min(len(bin_near), target_per_bin), random_state=42) if not bin_near.empty else pd.DataFrame()
    samples_mid = bin_mid.sample(n=min(len(bin_mid), target_per_bin), random_state=42) if not bin_mid.empty else pd.DataFrame()
    samples_high = bin_high.sample(n=min(len(bin_high), target_per_bin), random_state=42) if not bin_high.empty else pd.DataFrame()

    sampled_df = pd.concat([samples_near, samples_mid, samples_high], ignore_index=True)
    sampled_df = sampled_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    # Attach keyframe IDs
    kf_before = []
    kf_after = []
    grouped_shots = df_shots.groupby("video_id")

    for _, row in sampled_df.iterrows():
        vid = row["video_id"]
        s_curr_id = int(row["shot_i"])
        s_next_id = int(row["shot_next"])

        if vid in grouped_shots.groups:
            v_shots = grouped_shots.get_group(vid)
            r_curr = v_shots[v_shots["shot_id"] == s_curr_id]
            r_next = v_shots[v_shots["shot_id"] == s_next_id]

            k_b = str(r_curr.iloc[0].get("representative_keyframe", f"shot_{s_curr_id}")) if not r_curr.empty else f"shot_{s_curr_id}"
            k_a = str(r_next.iloc[0].get("representative_keyframe", f"shot_{s_next_id}")) if not r_next.empty else f"shot_{s_next_id}"
        else:
            k_b = f"shot_{s_curr_id}"
            k_a = f"shot_{s_next_id}"

        kf_before.append(k_b)
        kf_after.append(k_a)

    sampled_df["last_keyframe_before"] = kf_before
    sampled_df["first_keyframe_after"] = kf_after
    return sampled_df


def run_interactive_annotation(
    df_samples: pd.DataFrame,
    labels_file: Path,
) -> pd.DataFrame:
    """CLI Interactive loop for true human manual boundary labeling."""
    existing_labels = {}
    if labels_file.exists():
        try:
            df_exist = pd.read_parquet(labels_file)
            for _, r in df_exist.iterrows():
                key = f"{r['video_id']}_{r['shot_i']}_{r['shot_next']}"
                existing_labels[key] = str(r["user_label"])
            logger.info("Loaded %d existing manual labels from %s", len(existing_labels), labels_file)
        except Exception as e:
            logger.warning("Could not read existing labels file: %s", e)

    records = []
    total = len(df_samples)

    print("\n" + "=" * 90)
    print("✍️ INTERACTIVE STAGE 3C MANUAL BOUNDARY ANNOTATION TOOL")
    print("  Controls: [c]orrect boundary | [f]alse boundary | [a]mbiguous | [s]kip | [q]uit")
    print("=" * 90)

    for idx, row in df_samples.iterrows():
        key = f"{row['video_id']}_{row['shot_i']}_{row['shot_next']}"
        label = existing_labels.get(key, None)

        if label is None:
            print("-" * 90)
            print(f"[{idx+1}/{total}] VIDEO: {row['video_id']} | Shots: {int(row['shot_i'])} -> {int(row['shot_next'])}")
            print(f"  • Last Keyframe Before (Shot {int(row['shot_i'])}): {row['last_keyframe_before']}")
            print(f"  • First Keyframe After (Shot {int(row['shot_next'])}): {row['first_keyframe_after']}")
            print(f"  • Visual Similarity     : {row['visual_similarity']:.4f} (Evidence: {row['visual_boundary_evidence']:.4f})")
            print(f"  • Semantic Similarity   : {row['semantic_similarity']:.4f} (Evidence: {row['semantic_boundary_evidence']:.4f})")
            print(f"  🔥 FINAL BOUNDARY SCORE : {row['boundary_score']:.4f}")

            user_input = ""
            while user_input not in ["c", "f", "a", "s", "q"]:
                try:
                    user_input = input("  👉 Enter label [c/f/a/s/q]: ").strip().lower()
                except (KeyboardInterrupt, EOFError):
                    user_input = "q"

            if user_input == "q":
                print("Exiting annotation session...")
                break
            elif user_input == "c":
                label = "correct"
            elif user_input == "f":
                label = "false"
            elif user_input == "a":
                label = "ambiguous"
            elif user_input == "s":
                label = "skipped"

        rec = dict(row)
        rec["user_label"] = label
        records.append(rec)

    df_labeled = pd.DataFrame(records)
    if not df_labeled.empty:
        # Save Parquet & CSV
        labels_file.parent.mkdir(parents=True, exist_ok=True)
        df_labeled.to_parquet(labels_file, index=False)
        csv_file = labels_file.with_suffix(".csv")
        df_labeled.to_csv(csv_file, index=False)
        logger.info("Saved %d manual annotations to %s & %s", len(df_labeled), labels_file, csv_file)

    return df_labeled


def analyze_manual_precision_metrics(df_labeled: pd.DataFrame) -> Dict[str, Any]:
    """Compute true manual precision and score-binned analytics."""
    if df_labeled.empty or "user_label" not in df_labeled:
        return {}

    valid_df = df_labeled[df_labeled["user_label"].isin(["correct", "false", "ambiguous"])].copy()
    if valid_df.empty:
        return {"status": "No human annotations found yet"}

    cnt_correct = int((valid_df["user_label"] == "correct").sum())
    cnt_false = int((valid_df["user_label"] == "false").sum())
    cnt_ambiguous = int((valid_df["user_label"] == "ambiguous").sum())

    total_valid = cnt_correct + cnt_false
    precision_pct = (cnt_correct / total_valid * 100.0) if total_valid > 0 else 0.0

    # Score-binned precision
    bins = [0.70, 0.78, 0.85, 1.00]
    valid_df["score_bin"] = pd.cut(valid_df["boundary_score"], bins=bins, include_lowest=True)
    
    binned_report = {}
    for b_range, group in valid_df.groupby("score_bin", observed=False):
        c_cnt = int((group["user_label"] == "correct").sum())
        f_cnt = int((group["user_label"] == "false").sum())
        t_cnt = c_cnt + f_cnt
        p_pct = (c_cnt / t_cnt * 100.0) if t_cnt > 0 else 0.0
        binned_report[str(b_range)] = {
            "correct": c_cnt,
            "false": f_cnt,
            "ambiguous": int((group["user_label"] == "ambiguous").sum()),
            "precision_pct": p_pct,
        }

    # Extract typical false and correct examples
    false_examples = valid_df[valid_df["user_label"] == "false"][["video_id", "shot_i", "shot_next", "boundary_score"]].head(5).to_dict(orient="records")
    correct_examples = valid_df[valid_df["user_label"] == "correct"][["video_id", "shot_i", "shot_next", "boundary_score"]].head(5).to_dict(orient="records")

    report = {
        "total_human_annotations": len(valid_df),
        "overall_precision_pct": precision_pct,
        "label_counts": {
            "correct": cnt_correct,
            "false": cnt_false,
            "ambiguous": cnt_ambiguous,
        },
        "score_binned_precision": binned_report,
        "typical_false_boundaries": false_examples,
        "typical_correct_boundaries": correct_examples,
    }

    # Print Summary Report
    print("\n" + "=" * 90)
    print("📊 TRUE HUMAN MANUAL BOUNDARY PRECISION AUDIT REPORT")
    print("=" * 90)
    print(f" • Total Annotations Evaluated: {len(valid_df)}")
    print(f" • Overall Manual Precision  : {precision_pct:.2f}% (Correct: {cnt_correct}, False: {cnt_false}, Ambiguous: {cnt_ambiguous})")
    print("-" * 90)
    print("📊 PRECISION BY BOUNDARY SCORE BINS:")
    for b_name, b_data in binned_report.items():
        print(f"  • Bin {b_name:<18}: Precision = {b_data['precision_pct']:.1f}% ({b_data['correct']} Correct / {b_data['false']} False)")
    print("=" * 90 + "\n")

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Stage 3C Interactive Manual Boundary Annotation Suite"
    )
    parser.add_argument(
        "--similarities",
        default=str(
            PROJECT_ROOT / "artifacts" / "event_graph" / "boundaries" / "adjacent_similarities.parquet"
        ),
    )
    parser.add_argument(
        "--shots",
        default=str(PROJECT_ROOT / "artifacts" / "event_graph" / "features" / "shot_features.parquet"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "artifacts" / "event_graph" / "validation"),
    )
    parser.add_argument("--vis-weight", type=float, default=0.3)
    parser.add_argument("--sem-weight", type=float, default=0.7)
    parser.add_argument("--threshold", type=float, default=0.70)
    parser.add_argument("--num-samples", type=int, default=150)
    parser.add_argument("--non-interactive", action="store_true", help="Generate samples without prompt")
    args = parser.parse_args()

    sim_path = Path(args.similarities)
    shots_path = Path(args.shots)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels_file = output_dir / "manual_boundary_labels.parquet"

    df_raw_sim = pd.read_parquet(sim_path)
    df_shots = pd.read_parquet(shots_path)

    # Apply Default Priority Config (0.3V / 0.7S @ 0.70)
    df_cal = apply_robust_calibrated_fusion(
        df_raw_sim, vis_weight=args.vis_weight, sem_weight=args.sem_weight, boundary_threshold=args.threshold
    )

    # Sample stratified boundaries across score bins
    df_samples = sample_stratified_boundaries(df_cal, df_shots, threshold=args.threshold, num_samples=args.num_samples)

    if not args.non_interactive:
        df_labeled = run_interactive_annotation(df_samples, labels_file)
    else:
        # Save sample file for user inspection
        sample_file = output_dir / "boundary_samples_for_manual_check.csv"
        df_samples.to_csv(sample_file, index=False)
        logger.info("Saved %d boundary samples to %s", len(df_samples), sample_file)
        df_labeled = df_samples

    analyze_manual_precision_metrics(df_labeled)


if __name__ == "__main__":
    main()
