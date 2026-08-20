"""
Stage 3C: Threshold Sweep & Similarity Distribution Diagnostic (AI Challenge 2026)
===================================================================================
Senior ML Engineer Diagnostic Suite to solve Event Over-Segmentation.

Key Features:
1. Reuses pre-computed adjacent_similarities.parquet (Zero embedding re-computation).
2. Computes independent distribution stats for visual, semantic, and combined similarity.
3. Sweeps candidate thresholds: [0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.50].
4. Evaluates key metrics per threshold:
   - Total boundaries & total events
   - Events per video
   - Shots per event (mean & median)
   - % 1-shot events & % 2-shot events
   - P90 and P95 event length
5. Outputs comparison CSV/Parquet artifacts:
   - artifacts/event_graph/events/threshold_sweep_comparison.csv
   - artifacts/event_graph/events/threshold_sweep_comparison.parquet
6. Proposes 1-2 top candidate thresholds for visual verification across 10 sample videos.
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
from src.retrieval.logging_utils import setup_logger

logger = setup_logger("threshold-sweep-diagnostic")


def compute_distribution_metrics(series: pd.Series) -> Dict[str, float]:
    """Compute detailed distribution metrics (mean, std, min, max, percentiles)."""
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


def analyze_similarity_modalities(df_sim: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Separately analyze visual, semantic, and combined similarity distributions."""
    logger.info("Computing modality-specific similarity distribution metrics...")
    modalities = {
        "visual_similarity": df_sim.get("visual_similarity", pd.Series(dtype=float)),
        "semantic_similarity": df_sim.get("semantic_similarity", pd.Series(dtype=float)),
        "combined_similarity": df_sim.get("fused_similarity", pd.Series(dtype=float)),
    }
    
    results = {}
    for mod_name, series in modalities.items():
        results[mod_name] = compute_distribution_metrics(series)
    return results


def evaluate_single_threshold(
    df_sim: pd.DataFrame,
    df_shots: pd.DataFrame,
    threshold: float,
    min_event_shots: int = 1,
) -> Dict[str, Any]:
    """Evaluate event segmentation metrics for a single boundary threshold."""
    # Build fast lookup for adjacent similarities per video
    # Adjacent pairs per video: N shots -> N-1 pairs
    grouped_sim = df_sim.groupby("video_id")
    grouped_shots = df_shots.groupby("video_id")
    
    total_videos = df_shots["video_id"].nunique()
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
            fused_sims = sim_v["fused_similarity"].to_numpy()
        else:
            fused_sims = np.array([])

        # Flag boundaries
        boundaries = fused_sims < threshold
        v_boundaries_cnt = int(np.sum(boundaries))
        total_boundaries += v_boundaries_cnt

        # Group shots into events
        v_events_cnt = 0
        curr_len = 1
        for is_b in boundaries:
            if is_b:
                if curr_len < min_event_shots and event_lengths:
                    # Merge short noise shot into previous event length if needed
                    event_lengths[-1] += curr_len
                else:
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


def run_threshold_sweep(
    sim_path: Path,
    shots_path: Path,
    output_dir: Path,
    thresholds: List[float],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Execute Stage 3C Threshold Sweep over candidate thresholds."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = output_dir / "threshold_sweep_comparison.csv"
    out_parquet = output_dir / "threshold_sweep_comparison.parquet"

    logger.info("==================================================================")
    logger.info("🔍 STAGE 3C: THRESHOLD SWEEP & DIAGNOSTIC SUITE")
    logger.info("==================================================================")
    logger.info("Loading precomputed adjacent similarities from: %s", sim_path)
    df_sim = pd.read_parquet(sim_path)
    logger.info("Loading shot features from: %s", shots_path)
    df_shots = pd.read_parquet(shots_path)

    logger.info("Loaded %d adjacent pairs and %d shots", len(df_sim), len(df_shots))

    # 1. Modality similarity distribution analysis
    modality_stats = analyze_similarity_modalities(df_sim)

    # 2. Sweep across candidate thresholds
    logger.info("Sweeping thresholds: %s", thresholds)
    sweep_results = []
    start_time = time.time()

    for thresh in thresholds:
        logger.info("Evaluating threshold: %.2f ...", thresh)
        res = evaluate_single_threshold(df_sim, df_shots, thresh)
        sweep_results.append(res)

    df_sweep = pd.DataFrame(sweep_results)

    # Save outputs
    logger.info("Saving sweep comparison CSV to: %s", out_csv)
    df_sweep.to_csv(out_csv, index=False)
    logger.info("Saving sweep comparison Parquet to: %s", out_parquet)
    df_sweep.to_parquet(out_parquet, index=False)

    # 3. Propose top candidate thresholds based on optimization objective:
    # Objective: Minimize % 1-shot events while targeting ~15-30 events/video
    df_sorted_candidates = df_sweep.sort_values(by=["pct_1_shot_events", "events_per_video_mean"])
    candidate_thresholds = df_sorted_candidates["threshold"].tolist()[:2]

    # Print Report
    print("\n" + "=" * 90)
    print("📊 MODALITY SIMILARITY DISTRIBUTION STATS (VISUAL vs SEMANTIC vs COMBINED)")
    print("=" * 90)
    print(f"{'Modality':<22} | {'Mean ± Std':<18} | {'Min / Max':<15} | {'Q25 / Q50 (Med) / Q75':<22}")
    print("-" * 90)
    for mod_name, stats in modality_stats.items():
        m_std = f"{stats['mean']:.4f} ± {stats['std']:.4f}"
        m_minmax = f"{stats['min']:.4f} / {stats['max']:.4f}"
        m_qs = f"{stats['q25']:.4f} / {stats['q50_median']:.4f} / {stats['q75']:.4f}"
        print(f"{mod_name:<22} | {m_std:<18} | {m_minmax:<15} | {m_qs:<22}")
    print("=" * 90 + "\n")

    print("=" * 110)
    print("🎯 STAGE 3C THRESHOLD SWEEP COMPARISON TABLE")
    print("=" * 110)
    header = (
        f"{'Thresh':<7} | {'Boundaries':<10} | {'Events':<8} | {'Evts/Vid':<8} | "
        f"{'Shots/Evt (M/Med)':<18} | {'% 1-Shot':<10} | {'% 2-Shot':<10} | {'P90 / P95':<12}"
    )
    print(header)
    print("-" * 110)
    for _, r in df_sweep.iterrows():
        row_str = (
            f"{r['threshold']:<7.2f} | {int(r['total_boundaries']):<10,d} | {int(r['total_events']):<8,d} | "
            f"{r['events_per_video_mean']:<8.2f} | {r['shots_per_event_mean']:.2f} / {r['shots_per_event_median']:.1f}         | "
            f"{r['pct_1_shot_events']:<10.2f}% | {r['pct_2_shot_events']:<10.2f}% | "
            f"{r['p90_event_length']:.0f} / {r['p95_event_length']:.0f}"
        )
        print(row_str)
    print("=" * 110 + "\n")

    print("🏆 CANDIDATE THRESHOLD RECOMMENDATION FOR VISUAL INSPECTION:")
    for c_idx, cand in enumerate(candidate_thresholds, start=1):
        c_row = df_sweep[df_sweep["threshold"] == cand].iloc[0]
        print(
            f"  Candidate #{c_idx}: Threshold = {cand:.2f} | "
            f"Avg Events/Vid: {c_row['events_per_video_mean']:.1f} | "
            f"Avg Shots/Evt: {c_row['shots_per_event_mean']:.2f} | "
            f"% 1-Shot Events: {c_row['pct_1_shot_events']:.1f}%"
        )
    print("=" * 110 + "\n")

    return df_sweep, modality_stats


def main():
    parser = argparse.ArgumentParser(
        description="Stage 3C: Threshold Sweep & Similarity Diagnostic Suite"
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
        default=[0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.50],
        help="Candidate thresholds to evaluate",
    )
    args = parser.parse_args()

    run_threshold_sweep(
        sim_path=Path(args.similarities),
        shots_path=Path(args.shots),
        output_dir=Path(args.output_dir),
        thresholds=args.thresholds,
    )


if __name__ == "__main__":
    main()
