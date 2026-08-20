"""
Stage 3C: Robust Calibrated Threshold & Weight Sweep Diagnostic (AI Challenge 2026)
===================================================================================
Senior ML Engineer Diagnostic Suite evaluating Robust Calibrated Boundary Evidence Fusion.

Key Features:
1. Reuses precomputed adjacent_similarities.parquet & shot_features.parquet (Zero embedding recomputation).
2. Calculates Robust Z-Score & Sigmoidal Boundary Evidence for Visual and Semantic modalities.
3. Sweeps Thresholds: [0.50, 0.55, 0.60, 0.65, 0.70, 0.75].
4. Sweeps Fusion Weights: (0.3 vis / 0.7 sem), (0.5 vis / 0.5 sem), (0.7 vis / 0.3 sem).
5. Reports comprehensive metrics for each (weights, threshold) combination:
   - Boundaries, Total Events, Events/Video
   - Shots/Event (Mean & Median)
   - % 1-shot events & % 2-shot events
   - P90 and P95 event length
6. Exports CSV/Parquet comparison artifacts:
   - artifacts/event_graph/events/calibrated_threshold_sweep_comparison.csv
   - artifacts/event_graph/events/calibrated_threshold_sweep_comparison.parquet
7. Proposes top 2-3 candidate (weights, threshold) configurations for visual inspection.
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

logger = setup_logger("calibrated-sweep-diagnostic")


def compute_distribution_metrics(series: pd.Series) -> Dict[str, float]:
    """Compute detailed distribution metrics."""
    if series.empty:
        return {}
    return {
        "mean": float(series.mean()),
        "std": float(series.std()),
        "min": float(series.min()),
        "max": float(series.max()),
        "q25": float(series.quantile(0.25)),
        "q50_median": float(series.median()),
        "q75": float(series.quantile(0.75)),
        "q90": float(series.quantile(0.90)),
        "q95": float(series.quantile(0.95)),
    }


def evaluate_calibrated_configuration(
    df_sim_calibrated: pd.DataFrame,
    df_shots: pd.DataFrame,
    threshold: float,
    vis_weight: float,
    sem_weight: float,
) -> Dict[str, Any]:
    """Evaluate event segmentation metrics for a specific (vis_weight, sem_weight, threshold) config."""
    grouped_sim = df_sim_calibrated.groupby("video_id")
    grouped_shots = df_shots.groupby("video_id")

    total_boundaries = 0
    total_events = 0
    event_lengths: List[int] = []
    events_per_video_list: List[int] = []

    for video_id, shots_v in grouped_shots:
        shots_sorted = shots_v.sort_values("start_sec").reset_index(drop=True)
        num_shots = len(shots_sorted)
        if num_shots == 0:
            continue

        if video_id in grouped_sim.groups:
            sim_v = grouped_sim.get_group(video_id).sort_values("shot_i").reset_index(drop=True)
            scores = sim_v["boundary_score"].to_numpy()
        else:
            scores = np.array([])

        # Boundary condition: boundary_score > threshold
        boundaries = scores > threshold
        v_boundaries_cnt = int(np.sum(boundaries))
        total_boundaries += v_boundaries_cnt

        v_events_cnt = 0
        curr_len = 1
        for is_b in boundaries:
            if is_b:
                event_lengths.append(curr_len)
                v_events_cnt += 1
                curr_len = 1
            else:
                curr_len += 1

        if curr_len > 0:
            event_lengths.append(curr_len)
            v_events_cnt += 1

        events_per_video_list.append(v_events_cnt)
        total_events += v_events_cnt

    lens_series = pd.Series(event_lengths) if event_lengths else pd.Series([1])
    v_events_series = pd.Series(events_per_video_list) if events_per_video_list else pd.Series([1])

    cnt_1_shot = int((lens_series == 1).sum())
    cnt_2_shot = int((lens_series == 2).sum())
    pct_1_shot = (cnt_1_shot / len(lens_series)) * 100.0 if len(lens_series) > 0 else 0.0
    pct_2_shot = (cnt_2_shot / len(lens_series)) * 100.0 if len(lens_series) > 0 else 0.0

    return {
        "vis_weight": vis_weight,
        "sem_weight": sem_weight,
        "weights_label": f"{vis_weight:.1f}V / {sem_weight:.1f}S",
        "threshold": threshold,
        "total_boundaries": total_boundaries,
        "total_events": total_events,
        "events_per_video_mean": float(v_events_series.mean()),
        "events_per_video_median": float(v_events_series.median()),
        "shots_per_event_mean": float(lens_series.mean()),
        "shots_per_event_median": float(lens_series.median()),
        "pct_1_shot_events": float(pct_1_shot),
        "pct_2_shot_events": float(pct_2_shot),
        "p90_event_length": float(lens_series.quantile(0.90)),
        "p95_event_length": float(lens_series.quantile(0.95)),
    }


def run_calibrated_sweep(
    sim_path: Path,
    shots_path: Path,
    output_dir: Path,
    thresholds: List[float],
    weight_pairs: List[Tuple[float, float]],
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """Execute Stage 3C Calibrated Fusion Sweep over candidate thresholds and weight pairs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = output_dir / "calibrated_threshold_sweep_comparison.csv"
    out_parquet = output_dir / "calibrated_threshold_sweep_comparison.parquet"

    logger.info("==================================================================")
    logger.info("⚡ STAGE 3C: ROBUST CALIBRATED FUSION SWEEP & DIAGNOSTICS")
    logger.info("==================================================================")
    logger.info("Loading precomputed similarities from: %s", sim_path)
    df_sim = pd.read_parquet(sim_path)
    logger.info("Loading shot features from: %s", shots_path)
    df_shots = pd.read_parquet(shots_path)

    logger.info("Loaded %d adjacent pairs across %d shots", len(df_sim), len(df_shots))

    # Apply base robust calibration
    df_calibrated_base = apply_robust_calibrated_fusion(df_sim, vis_weight=0.5, sem_weight=0.5)

    # Print distribution diagnostic table of all generated fields
    diagnostics_fields = [
        "visual_similarity", "semantic_similarity",
        "visual_z", "semantic_z",
        "visual_boundary_evidence", "semantic_boundary_evidence",
        "boundary_score",
    ]
    diag_stats = {}
    for col in diagnostics_fields:
        if col in df_calibrated_base.columns:
            diag_stats[col] = compute_distribution_metrics(df_calibrated_base[col])

    print("\n" + "=" * 105)
    print("📊 ROBUST CALIBRATION DIAGNOSTICS DISTRIBUTION TABLE")
    print("=" * 105)
    print(f"{'Field Name':<28} | {'Mean ± Std':<20} | {'Min / Max':<18} | {'Q25 / Q50 (Med) / Q75':<24}")
    print("-" * 105)
    for col, st in diag_stats.items():
        m_std = f"{st['mean']:.4f} ± {st['std']:.4f}"
        m_minmax = f"{st['min']:.4f} / {st['max']:.4f}"
        m_qs = f"{st['q25']:.4f} / {st['q50_median']:.4f} / {st['q75']:.4f}"
        print(f"{col:<28} | {m_std:<20} | {m_minmax:<18} | {m_qs:<24}")
    print("=" * 105 + "\n")

    # Perform Multi-weight and Multi-threshold sweep
    sweep_results = []
    for w_v, w_s in weight_pairs:
        logger.info("Evaluating Fusion Weight Combination: Vis=%.1f, Sem=%.1f ...", w_v, w_s)
        df_cal_pair = apply_robust_calibrated_fusion(df_sim, vis_weight=w_v, sem_weight=w_s)

        for thresh in thresholds:
            res = evaluate_calibrated_configuration(df_cal_pair, df_shots, thresh, w_v, w_s)
            sweep_results.append(res)

    df_sweep = pd.DataFrame(sweep_results)

    logger.info("Saving calibrated sweep CSV to: %s", out_csv)
    df_sweep.to_csv(out_csv, index=False)
    logger.info("Saving calibrated sweep Parquet to: %s", out_parquet)
    df_sweep.to_parquet(out_parquet, index=False)

    # Print Full Comparison Table
    print("=" * 125)
    print("🎯 STAGE 3C ROBUST CALIBRATED FUSION SWEEP COMPARISON TABLE")
    print("=" * 125)
    header = (
        f"{'Weights':<10} | {'Thresh':<7} | {'Boundaries':<10} | {'Events':<8} | {'Evts/Vid':<8} | "
        f"{'Shots/Evt (M/Med)':<18} | {'% 1-Shot':<10} | {'% 2-Shot':<10} | {'P90 / P95':<12}"
    )
    print(header)
    print("-" * 125)
    for _, r in df_sweep.iterrows():
        row_str = (
            f"{r['weights_label']:<10} | {r['threshold']:<7.2f} | {int(r['total_boundaries']):<10,d} | {int(r['total_events']):<8,d} | "
            f"{r['events_per_video_mean']:<8.2f} | {r['shots_per_event_mean']:.2f} / {r['shots_per_event_median']:.1f}         | "
            f"{r['pct_1_shot_events']:<10.2f}% | {r['pct_2_shot_events']:<10.2f}% | "
            f"{r['p90_event_length']:.0f} / {r['p95_event_length']:.0f}"
        )
        print(row_str)
    print("=" * 125 + "\n")

    # Select top 3 candidate configurations:
    # Target: 10 - 25 events/video (Shots/Evt mean ~ 3-8), minimize % 1-shot events
    df_candidates = df_sweep[
        (df_sweep["events_per_video_mean"] >= 8.0) & (df_sweep["events_per_video_mean"] <= 35.0)
    ].sort_values(by="pct_1_shot_events")

    if df_candidates.empty:
        df_candidates = df_sweep.sort_values(by="pct_1_shot_events")

    top_candidates = df_candidates.head(3)

    print("🏆 CANDIDATE CONFIGURATION RECOMMENDATIONS FOR VISUAL INSPECTION:")
    for c_idx, (_, c_row) in enumerate(top_candidates.iterrows(), start=1):
        print(
            f"  Candidate #{c_idx}: Weights = ({c_row['weights_label']}) | "
            f"Threshold = {c_row['threshold']:.2f} | "
            f"Avg Events/Vid: {c_row['events_per_video_mean']:.1f} | "
            f"Avg Shots/Evt: {c_row['shots_per_event_mean']:.2f} | "
            f"% 1-Shot Events: {c_row['pct_1_shot_events']:.1f}%"
        )
    print("=" * 125 + "\n")

    return df_sweep, diag_stats


def main():
    parser = argparse.ArgumentParser(
        description="Stage 3C: Robust Calibrated Fusion & Threshold Sweep Diagnostic Suite"
    )
    parser.add_argument(
        "--similarities",
        default=str(
            PROJECT_ROOT / "artifacts" / "event_graph" / "boundaries" / "adjacent_similarities.parquet"
        ),
        help="Path to adjacent_similarities.parquet",
    )
    parser.add_argument(
        "--shots",
        default=str(PROJECT_ROOT / "artifacts" / "event_graph" / "features" / "shot_features.parquet"),
        help="Path to shot_features.parquet",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "artifacts" / "event_graph" / "events"),
        help="Output directory for comparison CSV/Parquet",
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.50, 0.55, 0.60, 0.65, 0.70, 0.75],
        help="Candidate thresholds to evaluate",
    )
    args = parser.parse_args()

    weight_pairs = [(0.3, 0.7), (0.5, 0.5), (0.7, 0.3)]

    run_calibrated_sweep(
        sim_path=Path(args.similarities),
        shots_path=Path(args.shots),
        output_dir=Path(args.output_dir),
        thresholds=args.thresholds,
        weight_pairs=weight_pairs,
    )


if __name__ == "__main__":
    main()
