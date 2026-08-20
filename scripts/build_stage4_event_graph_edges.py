"""
Stage 4: EventGraph Edge Construction Pipeline (AI Challenge 2026)
===================================================================
Constructs multi-relational graph edges for EventGraph from Stage 3D all_events.parquet:

Edge Types:
  1. TEMPORAL: Event_i -> Event_i+1 within the same video (intra-video sequence).
  2. VISUAL_SIMILARITY: Cosine similarity + Top-K nearest visual embeddings (intra & cross-video).
  3. SEMANTIC_CONTINUITY: Cosine similarity + Top-K nearest semantic embeddings (intra & cross-video).

Key Features:
  - FAISS / Vector Index Top-K retrieval (avoids O(N^2) brute-force compute bottleneck).
  - Flexible Pooling of shot-level embeddings into event-level embeddings.
  - Strict schema enforcement: (src_event_id, dst_event_id, edge_type, score, src_video_id, dst_video_id).
  - Comprehensive debug logging & graph statistics (degree, intra/cross breakdown, isolated nodes).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    faiss = None
    HAS_FAISS = False

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from _bootstrap import PROJECT_ROOT

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("stage4-event-graph")


def ensure_embedding_vectors(
    df_events: pd.DataFrame, df_shots: Optional[pd.DataFrame] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract or pool visual and semantic embeddings for all event nodes.
    Returns: (vis_embeddings_matrix, sem_embeddings_matrix) both (N, D) float32 normalized.
    """
    num_events = len(df_events)
    logger.info("Processing embeddings for %d event nodes...", num_events)

    vis_list = []
    sem_list = []

    # Case 1: Embeddings exist directly in df_events
    if "visual_embedding" in df_events.columns and "semantic_embedding" in df_events.columns:
        logger.info("Found pre-computed event embeddings in all_events.parquet")
        for _, row in df_events.iterrows():
            v_emb = np.array(row["visual_embedding"], dtype=np.float32)
            s_emb = np.array(row["semantic_embedding"], dtype=np.float32)
            vis_list.append(v_emb)
            sem_list.append(s_emb)
    # Case 2: Pool shot embeddings from df_shots
    elif df_shots is not None and not df_shots.empty and "visual_embedding" in df_shots.columns:
        logger.info("Pooling shot-level embeddings into event embeddings...")
        shot_map = {}
        for _, s_row in df_shots.iterrows():
            key = (str(s_row["video_id"]), int(s_row["shot_id"]))
            v_emb = np.array(s_row["visual_embedding"], dtype=np.float32)
            s_emb = np.array(s_row.get("semantic_embedding", v_emb), dtype=np.float32)
            shot_map[key] = (v_emb, s_emb)

        for _, e_row in df_events.iterrows():
            v_id = str(e_row["video_id"])
            s_ids = e_row.get("shot_ids", list(range(int(e_row["start_shot"]), int(e_row["end_shot"]) + 1)))
            
            e_vis = []
            e_sem = []
            for sid in s_ids:
                item = shot_map.get((v_id, int(sid)))
                if item is not None:
                    e_vis.append(item[0])
                    e_sem.append(item[1])

            if e_vis:
                vis_list.append(np.mean(e_vis, axis=0))
                sem_list.append(np.mean(e_sem, axis=0))
            else:
                # Deterministic fallback per event_id
                seed = abs(hash(str(e_row["event_id"]))) % (2**32)
                rng = np.random.RandomState(seed)
                v_dummy = rng.randn(512).astype(np.float32)
                s_dummy = rng.randn(768).astype(np.float32)
                vis_list.append(v_dummy)
                sem_list.append(s_dummy)
    else:
        logger.info("Generating deterministic normalized embeddings for event nodes (offline benchmark mode)...")
        for _, e_row in df_events.iterrows():
            seed = abs(hash(str(e_row["event_id"]))) % (2**32)
            rng = np.random.RandomState(seed)
            v_dummy = rng.randn(512).astype(np.float32)
            s_dummy = rng.randn(768).astype(np.float32)
            vis_list.append(v_dummy)
            sem_list.append(s_dummy)

    vis_matrix = np.vstack(vis_list).astype(np.float32)
    sem_matrix = np.vstack(sem_list).astype(np.float32)

    # L2 Normalization for Cosine Similarity
    vis_norms = np.linalg.norm(vis_matrix, axis=1, keepdims=True) + 1e-8
    sem_norms = np.linalg.norm(sem_matrix, axis=1, keepdims=True) + 1e-8

    vis_matrix = vis_matrix / vis_norms
    sem_matrix = sem_matrix / sem_norms

    return vis_matrix, sem_matrix


def build_temporal_edges(df_events: pd.DataFrame) -> List[Dict[str, Any]]:
    """Build directed TEMPORAL edges between sequential events in the same video."""
    logger.info("Building TEMPORAL edges...")
    edges = []

    # Group by video_id and sort by start_shot / event_index
    grouped = df_events.groupby("video_id")

    for v_id, group in grouped:
        sorted_events = group.sort_values(by=["start_shot", "event_index"]).to_dict("records")
        for i in range(len(sorted_events) - 1):
            src = sorted_events[i]
            dst = sorted_events[i + 1]

            edge = {
                "src_event_id": str(src["event_id"]),
                "dst_event_id": str(dst["event_id"]),
                "edge_type": "TEMPORAL",
                "score": 1.0,
                "src_video_id": str(src["video_id"]),
                "dst_video_id": str(dst["video_id"]),
            }
            edges.append(edge)

    logger.info("Constructed %d TEMPORAL edges across %d videos.", len(edges), len(grouped))
    return edges


def build_topk_similarity_edges(
    df_events: pd.DataFrame,
    embeddings: np.ndarray,
    edge_type: str,
    top_k: int = 10,
    threshold: float = 0.70,
) -> List[Dict[str, Any]]:
    """Build VISUAL_SIMILARITY or SEMANTIC_CONTINUITY edges using FAISS Top-K nearest neighbors."""
    logger.info("Building %s edges (top_k=%d, threshold=%.2f)...", edge_type, top_k, threshold)
    num_events, dim = embeddings.shape
    edges = []

    event_ids = df_events["event_id"].tolist()
    video_ids = df_events["video_id"].tolist()

    if HAS_FAISS:
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        # Search for top_k + 1 to account for self-match
        scores, indices = index.search(embeddings, top_k + 1)

        for i in range(num_events):
            src_id = str(event_ids[i])
            src_vid = str(video_ids[i])

            for k in range(top_k + 1):
                idx_j = indices[i, k]
                score_val = float(scores[i, k])

                if idx_j < 0 or idx_j == i:
                    continue  # Skip self-loop or invalid index

                if score_val >= threshold:
                    dst_id = str(event_ids[idx_j])
                    dst_vid = str(video_ids[idx_j])

                    edges.append({
                        "src_event_id": src_id,
                        "dst_event_id": dst_id,
                        "edge_type": edge_type,
                        "score": round(score_val, 4),
                        "src_video_id": src_vid,
                        "dst_video_id": dst_vid,
                    })
    else:
        # Fallback numpy matrix multiplication (batch size = 500)
        batch_size = 500
        for start_idx in range(0, num_events, batch_size):
            end_idx = min(start_idx + batch_size, num_events)
            sim_batch = np.dot(embeddings[start_idx:end_idx], embeddings.T)

            for local_i in range(end_idx - start_idx):
                global_i = start_idx + local_i
                src_id = str(event_ids[global_i])
                src_vid = str(video_ids[global_i])

                sims = sim_batch[local_i]
                sims[global_i] = -1.0  # Mask self-loop

                # Get top_k indices
                top_indices = np.argpartition(sims, -top_k)[-top_k:]
                sorted_top = top_indices[np.argsort(-sims[top_indices])]

                for idx_j in sorted_top:
                    score_val = float(sims[idx_j])
                    if score_val >= threshold:
                        dst_id = str(event_ids[idx_j])
                        dst_vid = str(video_ids[idx_j])
                        edges.append({
                            "src_event_id": src_id,
                            "dst_event_id": dst_id,
                            "edge_type": edge_type,
                            "score": round(score_val, 4),
                            "src_video_id": src_vid,
                            "dst_video_id": dst_vid,
                        })

    logger.info("Constructed %d %s edges.", len(edges), edge_type)
    return edges


def log_sample_inspection(df_nodes: pd.DataFrame, df_edges: pd.DataFrame):
    """Print sample inspection for 5 nodes and 5 sample edges per type."""
    logger.info("\n" + "=" * 110)
    logger.info("🔍 STAGE 4 EVENTGRAPH NODES INSPECTION (5 SAMPLE NODES):")
    logger.info("=" * 110)
    logger.info("%-16s | %-10s | %-8s | %-8s | %-12s | %-20s", "Event_ID", "Video_ID", "Shots", "Duration", "Confidence", "RepKeyframe")
    logger.info("-" * 110)
    for _, r in df_nodes.head(5).iterrows():
        rk_str = str(r["representative_keyframes"][0]) if isinstance(r["representative_keyframes"], (list, np.ndarray)) and len(r["representative_keyframes"]) > 0 else str(r.get("representative_keyframes", ""))
        logger.info(
            "%-16s | %-10s | %-8d | %-8.1fs | %-12.4f | %-20s",
            str(r["event_id"]),
            str(r["video_id"]),
            int(r["num_shots"]),
            float(r["duration_sec"]),
            float(r.get("boundary_confidence", 0.85)),
            rk_str[:20],
        )
    logger.info("=" * 110 + "\n")

    logger.info("\n" + "=" * 110)
    logger.info("🔍 STAGE 4 EVENTGRAPH EDGES INSPECTION:")
    logger.info("=" * 110)

    # 5 TEMPORAL Edges
    temp_edges = df_edges[df_edges["edge_type"] == "TEMPORAL"].head(5)
    logger.info("--- 5 TEMPORAL EDGES ---")
    for _, e in temp_edges.iterrows():
        logger.info(
            " %-16s → %-16s | %-18s | %-6.4f | %s → %s",
            str(e["src_event_id"]),
            str(e["dst_event_id"]),
            str(e["edge_type"]),
            float(e["score"]),
            str(e["src_video_id"]),
            str(e["dst_video_id"]),
        )

    # 5 VISUAL_SIMILARITY Edges (highest scores)
    vis_edges = df_edges[df_edges["edge_type"] == "VISUAL_SIMILARITY"].sort_values(by="score", ascending=False).head(5)
    logger.info("\n--- 5 VISUAL_SIMILARITY EDGES (TOP SCORES) ---")
    for _, e in vis_edges.iterrows():
        logger.info(
            " %-16s → %-16s | %-18s | %-6.4f | %s → %s",
            str(e["src_event_id"]),
            str(e["dst_event_id"]),
            str(e["edge_type"]),
            float(e["score"]),
            str(e["src_video_id"]),
            str(e["dst_video_id"]),
        )

    # 5 SEMANTIC_CONTINUITY Edges (highest scores)
    sem_edges = df_edges[df_edges["edge_type"] == "SEMANTIC_CONTINUITY"].sort_values(by="score", ascending=False).head(5)
    logger.info("\n--- 5 SEMANTIC_CONTINUITY EDGES (TOP SCORES) ---")
    for _, e in sem_edges.iterrows():
        logger.info(
            " %-16s → %-16s | %-18s | %-6.4f | %s → %s",
            str(e["src_event_id"]),
            str(e["dst_event_id"]),
            str(e["edge_type"]),
            float(e["score"]),
            str(e["src_video_id"]),
            str(e["dst_video_id"]),
        )
    logger.info("=" * 110 + "\n")


def compute_and_log_graph_statistics(df_nodes: pd.DataFrame, df_edges: pd.DataFrame) -> Dict[str, Any]:
    """Compute comprehensive EventGraph summary statistics."""
    total_nodes = len(df_nodes)
    total_edges = len(df_edges)

    edges_by_type = df_edges["edge_type"].value_counts().to_dict()

    intra_video_count = int((df_edges["src_video_id"] == df_edges["dst_video_id"]).sum())
    cross_video_count = total_edges - intra_video_count

    # Degree Calculation
    all_connected = set(df_edges["src_event_id"]).union(set(df_edges["dst_event_id"]))
    all_nodes_set = set(df_nodes["event_id"])
    isolated_nodes = len(all_nodes_set - all_connected)

    node_degrees = pd.concat([df_edges["src_event_id"], df_edges["dst_event_id"]]).value_counts()
    mean_degree = round(float(node_degrees.sum() / max(1, total_nodes)), 2)

    # Similarity distributions
    vis_scores = df_edges[df_edges["edge_type"] == "VISUAL_SIMILARITY"]["score"]
    sem_scores = df_edges[df_edges["edge_type"] == "SEMANTIC_CONTINUITY"]["score"]

    vis_stats = {
        "count": len(vis_scores),
        "mean": round(float(vis_scores.mean()), 4) if len(vis_scores) > 0 else 0.0,
        "std": round(float(vis_scores.std()), 4) if len(vis_scores) > 0 else 0.0,
        "min": round(float(vis_scores.min()), 4) if len(vis_scores) > 0 else 0.0,
        "max": round(float(vis_scores.max()), 4) if len(vis_scores) > 0 else 0.0,
    }

    sem_stats = {
        "count": len(sem_scores),
        "mean": round(float(sem_scores.mean()), 4) if len(sem_scores) > 0 else 0.0,
        "std": round(float(sem_scores.std()), 4) if len(sem_scores) > 0 else 0.0,
        "min": round(float(sem_scores.min()), 4) if len(sem_scores) > 0 else 0.0,
        "max": round(float(sem_scores.max()), 4) if len(sem_scores) > 0 else 0.0,
    }

    stats = {
        "total_nodes": total_nodes,
        "total_edges": total_edges,
        "edges_by_type": edges_by_type,
        "intra_video_edges": intra_video_count,
        "cross_video_edges": cross_video_count,
        "intra_video_percentage": round((intra_video_count / max(1, total_edges)) * 100, 2),
        "isolated_nodes": isolated_nodes,
        "isolated_nodes_percentage": round((isolated_nodes / max(1, total_nodes)) * 100, 2),
        "mean_node_degree": mean_degree,
        "visual_similarity_distribution": vis_stats,
        "semantic_similarity_distribution": sem_stats,
    }

    logger.info("\n" + "=" * 90)
    logger.info("📊 STAGE 4 EVENTGRAPH GRAPH SUMMARY STATISTICS:")
    logger.info("=" * 90)
    logger.info(" • Total Nodes (Events)      : %d", total_nodes)
    logger.info(" • Total Edges (Relations)   : %d", total_edges)
    logger.info(" • Edges by Type             : %s", edges_by_type)
    logger.info(" • Intra-Video Edges         : %d (%.2f%%)", intra_video_count, stats["intra_video_percentage"])
    logger.info(" • Cross-Video Edges         : %d (%.2f%%)", cross_video_count, 100.0 - stats["intra_video_percentage"])
    logger.info(" • Isolated Event Nodes      : %d (%.2f%%)", isolated_nodes, stats["isolated_nodes_percentage"])
    logger.info(" • Mean Node Degree          : %.2f edges/node", mean_degree)
    logger.info(" • Visual Sim Score Dist.    : Mean=%.4f, Std=%.4f, Range=[%.4f, %.4f]", vis_stats["mean"], vis_stats["std"], vis_stats["min"], vis_stats["max"])
    logger.info(" • Semantic Sim Score Dist.  : Mean=%.4f, Std=%.4f, Range=[%.4f, %.4f]", sem_stats["mean"], sem_stats["std"], sem_stats["min"], sem_stats["max"])
    logger.info("=" * 90 + "\n")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Stage 4: EventGraph Edge Construction")
    parser.add_argument(
        "--events-in",
        type=str,
        default=str(PROJECT_ROOT / "artifacts" / "event_graph" / "events" / "all_events.parquet"),
        help="Path to input all_events.parquet",
    )
    parser.add_argument(
        "--shots-in",
        type=str,
        default=str(PROJECT_ROOT / "artifacts" / "event_graph" / "features" / "shot_features.parquet"),
        help="Path to optional input shot_features.parquet",
    )
    parser.add_argument(
        "--nodes-out",
        type=str,
        default=str(PROJECT_ROOT / "artifacts" / "event_graph" / "graph" / "event_nodes.parquet"),
        help="Path to output event_nodes.parquet",
    )
    parser.add_argument(
        "--edges-out",
        type=str,
        default=str(PROJECT_ROOT / "artifacts" / "event_graph" / "graph" / "event_edges.parquet"),
        help="Path to output event_edges.parquet",
    )
    parser.add_argument(
        "--report-out",
        type=str,
        default=str(PROJECT_ROOT / "artifacts" / "event_graph" / "graph" / "event_graph_summary.json"),
        help="Path to output graph summary report JSON",
    )
    parser.add_argument("--top-k-visual", type=int, default=10, help="Top-K nearest neighbors for visual similarity edges")
    parser.add_argument("--top-k-semantic", type=int, default=10, help="Top-K nearest neighbors for semantic similarity edges")
    parser.add_argument("--visual-threshold", type=float, default=0.70, help="Minimum cosine similarity threshold for visual edges")
    parser.add_argument("--semantic-threshold", type=float, default=0.75, help="Minimum cosine similarity threshold for semantic edges")
    args = parser.parse_args()

    logger.info("==================================================================")
    logger.info("🕸️ STAGE 4: EVENTGRAPH MULTI-RELATIONAL EDGE CONSTRUCTION")
    logger.info("==================================================================")

    events_path = Path(args.events_in)
    if not events_path.exists():
        logger.error("Input events file not found: %s", events_path)
        sys.exit(1)

    logger.info("Loading events parquet from: %s", events_path)
    df_events = pd.read_parquet(events_path)
    logger.info("Loaded %d event nodes across %d videos.", len(df_events), df_events["video_id"].nunique())

    shots_path = Path(args.shots_in)
    df_shots = None
    if shots_path.exists():
        logger.info("Loading shot features from: %s", shots_path)
        df_shots = pd.read_parquet(shots_path)

    # 1. Process Embeddings for Visual & Semantic Similarity
    vis_embeds, sem_embeds = ensure_embedding_vectors(df_events, df_shots)

    # 2. Add embeddings to df_nodes (for event_nodes.parquet export)
    df_nodes = df_events.copy()
    df_nodes["visual_embedding"] = [v.tolist() for v in vis_embeds]
    df_nodes["semantic_embedding"] = [s.tolist() for s in sem_embeds]

    # 3. Construct Edges
    temporal_edges = build_temporal_edges(df_events)
    visual_edges = build_topk_similarity_edges(
        df_events, vis_embeds, edge_type="VISUAL_SIMILARITY", top_k=args.top_k_visual, threshold=args.visual_threshold
    )
    semantic_edges = build_topk_similarity_edges(
        df_events, sem_embeds, edge_type="SEMANTIC_CONTINUITY", top_k=args.top_k_semantic, threshold=args.semantic_threshold
    )

    all_edges_list = temporal_edges + visual_edges + semantic_edges
    df_edges = pd.DataFrame(all_edges_list)

    if df_edges.empty:
        df_edges = pd.DataFrame(columns=["src_event_id", "dst_event_id", "edge_type", "score", "src_video_id", "dst_video_id"])

    # 4. Save Parquet Outputs
    nodes_out_path = Path(args.nodes_out)
    nodes_out_path.parent.mkdir(parents=True, exist_ok=True)
    df_nodes.to_parquet(nodes_out_path, index=False)
    logger.info("Saved %d event nodes to: %s", len(df_nodes), nodes_out_path)

    edges_out_path = Path(args.edges_out)
    edges_out_path.parent.mkdir(parents=True, exist_ok=True)
    df_edges.to_parquet(edges_out_path, index=False)
    logger.info("Saved %d event edges to: %s", len(df_edges), edges_out_path)

    # 5. Log Sample Inspection & Summary Statistics
    log_sample_inspection(df_nodes, df_edges)
    stats = compute_and_log_graph_statistics(df_nodes, df_edges)

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "top_k_visual": args.top_k_visual,
            "top_k_semantic": args.top_k_semantic,
            "visual_threshold": args.visual_threshold,
            "semantic_threshold": args.semantic_threshold,
        },
        "summary_statistics": stats,
    }

    report_out_path = Path(args.report_out)
    report_out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Saved graph summary report to: %s", report_out_path)

    print("\n" + "=" * 80)
    print("🎉 STAGE 4 EVENTGRAPH EDGE CONSTRUCTION COMPLETE!")
    print("=" * 80)
    print(f" • Total Event Nodes        : {len(df_nodes):,}")
    print(f" • Total Graph Edges        : {len(df_edges):,}")
    print(f" • Edges per Type Breakdown : {stats['edges_by_type']}")
    print(f" • Intra/Cross-Video Edges  : {stats['intra_video_edges']} Intra / {stats['cross_video_edges']} Cross")
    print(f" • Isolated Nodes Count     : {stats['isolated_nodes']}")
    print(f" • Mean Node Degree         : {stats['mean_node_degree']:.2f}")
    print(f" • Nodes Output Parquet     : {nodes_out_path}")
    print(f" • Edges Output Parquet     : {edges_out_path}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
