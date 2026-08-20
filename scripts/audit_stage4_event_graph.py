"""
Stage 4 EventGraph Pre-Freeze Audit & Integrity Diagnostic Tool (AI Challenge 2026)
=====================================================================================
Audits Stage 4 EventGraph artifacts prior to final FREEZE:

Audit Scope:
  1. Verify Stage 3D Input Integrity (ensures no Stage 3D regression).
  2. Audit Semantic Continuity (inspects random 20 semantic edges + Top/Bottom 5 with node metadata).
  3. Check Duplicate Bidirectional Edges (symmetric A->B & B->A analysis and deduplication rate).
  4. Validate Keyframe Schema Uniformity (ensures standardized keyframe ID string format).
  5. Final Pre-Freeze Summary Report (nodes, temporal edges, visual unique edges, semantic unique edges,
     duplicate rate, similarity distributions, intra/cross-video ratio).
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
logger = logging.getLogger("stage4-audit")


def audit_stage3d_input(df_events: pd.DataFrame) -> Dict[str, Any]:
    """Verify Stage 3D input integrity."""
    total_events = len(df_events)
    num_videos = df_events["video_id"].nunique()
    
    # Check for expected event schema
    required_cols = ["event_id", "video_id", "start_shot", "end_shot", "start_sec", "end_sec", "shot_ids"]
    missing_cols = [c for c in required_cols if c not in df_events.columns]
    
    status = "OK" if not missing_cols and total_events > 0 else "WARNING"
    
    return {
        "status": status,
        "total_events": total_events,
        "total_videos": num_videos,
        "missing_columns": missing_cols,
        "mean_events_per_video": round(total_events / max(1, num_videos), 2),
    }


def audit_keyframe_schema(df_nodes: pd.DataFrame) -> Dict[str, Any]:
    """Validate representative keyframe schema uniformity."""
    non_standard_count = 0
    total_nodes = len(df_nodes)

    sample_keyframes = []
    for _, r in df_nodes.iterrows():
        rk_list = r.get("representative_keyframes", [])
        if isinstance(rk_list, (list, np.ndarray)) and len(rk_list) > 0:
            rk_val = str(rk_list[0])
        else:
            rk_val = str(rk_list)
            
        sample_keyframes.append(rk_val)
        if rk_val.isdigit() or not rk_val.strip():
            non_standard_count += 1

    return {
        "total_nodes": total_nodes,
        "standardized_count": total_nodes - non_standard_count,
        "non_standard_count": non_standard_count,
        "standardization_percentage": round(((total_nodes - non_standard_count) / max(1, total_nodes)) * 100, 2),
        "sample_keyframes": sample_keyframes[:5],
    }


def audit_duplicate_edges(df_edges: pd.DataFrame) -> Dict[str, Any]:
    """Check duplicate bidirectional edges."""
    total_edges = len(df_edges)
    sim_edges = df_edges[df_edges["edge_type"].isin(["VISUAL_SIMILARITY", "SEMANTIC_CONTINUITY"])]
    
    seen_pairs = set()
    duplicate_count = 0
    
    for _, r in sim_edges.iterrows():
        e_type = r["edge_type"]
        src = str(r["src_event_id"])
        dst = str(r["dst_event_id"])
        pair_key = (e_type, min(src, dst), max(src, dst))
        if pair_key in seen_pairs:
            duplicate_count += 1
        else:
            seen_pairs.add(pair_key)

    unique_sim_edges = len(sim_edges) - duplicate_count
    duplicate_rate = round((duplicate_count / max(1, len(sim_edges))) * 100, 2) if len(sim_edges) > 0 else 0.0

    return {
        "total_edges": total_edges,
        "total_similarity_edges": len(sim_edges),
        "unique_similarity_edges": unique_sim_edges,
        "bidirectional_duplicate_count": duplicate_count,
        "duplicate_rate_percentage": duplicate_rate,
    }


def audit_semantic_continuity_edges(df_nodes: pd.DataFrame, df_edges: pd.DataFrame) -> Dict[str, Any]:
    """Audit semantic continuity edges with full metadata inspection."""
    sem_edges = df_edges[df_edges["edge_type"] == "SEMANTIC_CONTINUITY"].copy()
    if sem_edges.empty:
        return {"count": 0, "status": "NO_SEMANTIC_EDGES"}

    node_map = df_nodes.set_index("event_id").to_dict("index")
    
    scores = sem_edges["score"].to_numpy()
    mean_s = float(np.mean(scores))
    std_s = float(np.std(scores))
    min_s = float(np.min(scores))
    max_s = float(np.max(scores))

    # Select 20 random samples + Top 5 + Bottom 5
    seed = 42
    sample_indices = np.random.RandomState(seed).choice(len(sem_edges), min(20, len(sem_edges)), replace=False)
    random_20 = sem_edges.iloc[sample_indices].to_dict("records")
    
    top_5 = sem_edges.sort_values(by="score", ascending=False).head(5).to_dict("records")
    bottom_5 = sem_edges.sort_values(by="score", ascending=True).head(5).to_dict("records")

    def enrich_edge(e):
        src_meta = node_map.get(e["src_event_id"], {})
        dst_meta = node_map.get(e["dst_event_id"], {})
        return {
            "src_event": e["src_event_id"],
            "dst_event": e["dst_event_id"],
            "score": e["score"],
            "src_video": e["src_video_id"],
            "dst_video": e["dst_video_id"],
            "src_shots": src_meta.get("num_shots", 0),
            "dst_shots": dst_meta.get("num_shots", 0),
            "src_duration": src_meta.get("duration_sec", 0.0),
            "dst_duration": dst_meta.get("duration_sec", 0.0),
            "src_keyframe": str(src_meta.get("representative_keyframes", [""])[0]),
            "dst_keyframe": str(dst_meta.get("representative_keyframes", [""])[0]),
        }

    enriched_random_20 = [enrich_edge(e) for e in random_20]
    enriched_top_5 = [enrich_edge(e) for e in top_5]
    enriched_bottom_5 = [enrich_edge(e) for e in bottom_5]

    return {
        "count": len(sem_edges),
        "mean_score": round(mean_s, 4),
        "std_score": round(std_s, 4),
        "min_score": round(min_s, 4),
        "max_score": round(max_s, 4),
        "saturation_warning": bool(std_s < 0.01 and mean_s > 0.95),
        "top_5_high_scores": enriched_top_5,
        "bottom_5_low_scores": enriched_bottom_5,
        "random_20_sample_edges": enriched_random_20,
    }


def main():
    parser = argparse.ArgumentParser(description="Stage 4: EventGraph Pre-Freeze Audit & Verification")
    parser.add_argument(
        "--nodes-in",
        type=str,
        default=str(PROJECT_ROOT / "artifacts" / "event_graph" / "graph" / "event_nodes.parquet"),
        help="Path to event_nodes.parquet",
    )
    parser.add_argument(
        "--edges-in",
        type=str,
        default=str(PROJECT_ROOT / "artifacts" / "event_graph" / "graph" / "event_edges.parquet"),
        help="Path to event_edges.parquet",
    )
    parser.add_argument(
        "--events-in",
        type=str,
        default=str(PROJECT_ROOT / "artifacts" / "event_graph" / "events" / "all_events.parquet"),
        help="Path to all_events.parquet",
    )
    parser.add_argument(
        "--report-out",
        type=str,
        default=str(PROJECT_ROOT / "artifacts" / "event_graph" / "graph" / "stage4_audit_report.json"),
        help="Path to output audit report JSON",
    )
    args = parser.parse_args()

    logger.info("==================================================================")
    logger.info("🔍 STAGE 4: EVENTGRAPH PRE-FREEZE AUDIT & VERIFICATION")
    logger.info("==================================================================")

    nodes_path = Path(args.nodes_in)
    edges_path = Path(args.edges_in)
    events_path = Path(args.events_in)

    if not nodes_path.exists() or not edges_path.exists():
        logger.error("Node or Edge parquet file not found!")
        sys.exit(1)

    df_nodes = pd.read_parquet(nodes_path)
    df_edges = pd.read_parquet(edges_path)
    df_events = pd.read_parquet(events_path) if events_path.exists() else df_nodes

    # 1. Audit Stage 3D Input
    s3d_audit = audit_stage3d_input(df_events)
    
    # 2. Audit Keyframe Schema
    kf_audit = audit_keyframe_schema(df_nodes)

    # 3. Audit Duplicate Edges
    dup_audit = audit_duplicate_edges(df_edges)

    # 4. Audit Semantic Continuity Edges
    sem_audit = audit_semantic_continuity_edges(df_nodes, df_edges)

    # Breakdown by edge type
    temporal_count = int((df_edges["edge_type"] == "TEMPORAL").sum())
    vis_count = int((df_edges["edge_type"] == "VISUAL_SIMILARITY").sum())
    sem_count = int((df_edges["edge_type"] == "SEMANTIC_CONTINUITY").sum())

    intra_count = int((df_edges["src_video_id"] == df_edges["dst_video_id"]).sum())
    cross_count = len(df_edges) - intra_count

    # Degree stats
    connected_nodes = set(df_edges["src_event_id"]).union(set(df_edges["dst_event_id"]))
    isolated_nodes = len(set(df_nodes["event_id"]) - connected_nodes)

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "audit_pass_status": "PASS" if kf_audit["non_standard_count"] == 0 and s3d_audit["status"] == "OK" else "WARN",
        "stage3d_input_verification": s3d_audit,
        "keyframe_schema_audit": kf_audit,
        "duplicate_edges_audit": dup_audit,
        "semantic_continuity_audit": sem_audit,
        "final_summary": {
            "total_nodes": len(df_nodes),
            "temporal_edges": temporal_count,
            "visual_similarity_edges": vis_count,
            "semantic_continuity_edges": sem_count,
            "total_edges": len(df_edges),
            "intra_video_edges": intra_count,
            "cross_video_edges": cross_count,
            "intra_cross_ratio": round(intra_count / max(1, cross_count), 4),
            "isolated_nodes": isolated_nodes,
            "duplicate_rate_pct": dup_audit["duplicate_rate_percentage"],
        },
    }

    out_json = Path(args.report_out)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("\n" + "=" * 110)
    logger.info("📋 20 RANDOM SEMANTIC CONTINUITY EDGES SAMPLE INSPECTION:")
    logger.info("=" * 110)
    logger.info("%-16s → %-16s | %-6s | %-10s → %-10s | %-22s → %-22s", "Src_Event", "Dst_Event", "Score", "Src_Vid", "Dst_Vid", "Src_Keyframe", "Dst_Keyframe")
    logger.info("-" * 110)
    for e in sem_audit.get("random_20_sample_edges", []):
        logger.info(
            "%-16s → %-16s | %-6.4f | %-10s → %-10s | %-22s → %-22s",
            e["src_event"], e["dst_event"], e["score"], e["src_video"], e["dst_video"], e["src_keyframe"][:22], e["dst_keyframe"][:22]
        )
    logger.info("=" * 110 + "\n")

    logger.info("\n" + "=" * 90)
    logger.info("📊 STAGE 4 PRE-FREEZE AUDIT SUMMARY:")
    logger.info("=" * 90)
    logger.info(" • Overall Audit Status      : %s", report["audit_pass_status"])
    logger.info(" • Total Nodes (Events)      : %d", len(df_nodes))
    logger.info(" • Temporal Edges            : %d", temporal_count)
    logger.info(" • Visual Sim Edges          : %d", vis_count)
    logger.info(" • Semantic Cont. Edges      : %d", sem_count)
    logger.info(" • Total Edges               : %d", len(df_edges))
    logger.info(" • Duplicate Edge Rate       : %.2f%% (%d duplicates)", dup_audit["duplicate_rate_percentage"], dup_audit["bidirectional_duplicate_count"])
    logger.info(" • Intra / Cross Ratio       : %d Intra / %d Cross (Ratio: %.4f)", intra_count, cross_count, report["final_summary"]["intra_cross_ratio"])
    logger.info(" • Keyframe Standardization : %.2f%% (%d non-standard)", kf_audit["standardization_percentage"], kf_audit["non_standard_count"])
    logger.info("=" * 90 + "\n")

    print("\n" + "=" * 80)
    print("🎉 STAGE 4 PRE-FREEZE AUDIT COMPLETE!")
    print("=" * 80)
    print(f" • Audit Status             : {report['audit_pass_status']}")
    print(f" • Total Nodes              : {len(df_nodes):,}")
    print(f" • Temporal Edges           : {temporal_count:,}")
    print(f" • Visual Sim Edges         : {vis_count:,}")
    print(f" • Semantic Cont. Edges     : {sem_count:,}")
    print(f" • Duplicate Edge Rate      : {dup_audit['duplicate_rate_percentage']:.2f}%")
    print(f" • Keyframe Schema Standard : {kf_audit['standardization_percentage']:.2f}%")
    print(f" • Output Audit Report JSON  : {out_json}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
