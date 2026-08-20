"""
Stage 5: Multi-Hop EventGraph Retrieval & Traversal Demonstration Script
========================================================================
Demonstrates:
  1. Direct Multimodal Retrieval (1st-hop seeds)
  2. Multi-Hop Graph Traversal (Hop-1 & Hop-2 expansion over Temporal, Visual, and Semantic edges)
  3. Graph Reranking & Path Trajectory Analysis
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.event_graph_service import EventGraphService

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("stage5-search")


def run_stage5_graph_search(
    nodes_path: Path,
    edges_path: Path,
    sample_queries: List[str],
    max_hops: int = 2,
    graph_weight: float = 0.35,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """Runs Stage 5 Multi-Hop EventGraph Retrieval for sample queries."""
    graph_svc = EventGraphService(nodes_path=nodes_path, edges_path=edges_path)
    graph_svc.initialize()

    all_results = []

    for q_idx, query in enumerate(sample_queries, 1):
        logger.info("\n" + "=" * 70)
        logger.info("🔍 QUERY [%d/%d]: '%s'", q_idx, len(sample_queries), query)
        logger.info("=" * 70)

        # 1. Simulate 1st-hop direct retrieval seeds (or pick top node matches)
        # Match nodes whose action/text contains query terms or random seed pool
        matching_seeds = []
        q_lower = query.lower()

        for ev_id, node in graph_svc.node_map.items():
            act = str(node.get("action_description", "")).lower()
            text = str(node.get("event_text", "")).lower()
            summary = str(node.get("summary", "")).lower()

            match_score = 0.0
            if q_lower in act or q_lower in text or q_lower in summary:
                match_score = 0.90
            elif any(w in act or w in text for w in q_lower.split() if len(w) > 3):
                match_score = 0.65

            if match_score > 0:
                matching_seeds.append({
                    "event_id": ev_id,
                    "fused_score": match_score,
                    "video_id": node.get("video_id", ""),
                })

        # If no direct matches, fallback to picking top 5 nodes for demonstration
        if not matching_seeds:
            sample_ids = list(graph_svc.node_map.keys())[:5]
            matching_seeds = [{"event_id": eid, "fused_score": 0.85 - (i * 0.1), "video_id": graph_svc.node_map[eid].get("video_id", "")} for i, eid in enumerate(sample_ids)]

        logger.info("Found %d direct seed events at Hop-0.", len(matching_seeds))

        # 2. Multi-Hop Graph Reranking
        t0 = time.time()
        reranked_events = graph_svc.rerank_events_with_graph(
            direct_candidates=matching_seeds,
            max_hops=max_hops,
            graph_weight=graph_weight,
            top_k=top_k,
        )
        elapsed_ms = (time.time() - t0) * 1000

        logger.info("Completed Multi-Hop Graph Reranking in %.2f ms (%d events returned).", elapsed_ms, len(reranked_events))

        print(f"\n--- TOP {top_k} MULTI-HOP GRAPH RETRIEVAL RESULTS ---")
        for rank, item in enumerate(reranked_events, 1):
            path_str = " -> ".join(item.get("edge_type_path", [])) or "DIRECT_SEED"
            print(f" #{rank:02d} | Event: {item['event_id']} (Vid: {item.get('video_id', '')}) | Final Score: {item['final_graph_score']:.4f}")
            print(f"       Direct Score: {item['direct_multimodal_score']:.4f} | Graph Boost: {item['graph_propagation_score']:.4f}")
            print(f"       Hop: {item.get('hop_distance', 0)} | Traversal Path: [{path_str}] | Origin: {item.get('expanded_from', '')}")
            print(f"       Time: {float(item.get('start_sec', 0.0)):.1f}s - {float(item.get('end_sec', 0.0)):.1f}s | Shots: {item.get('num_shots', 1)}")
            print("-" * 65)

        all_results.append({
            "query": query,
            "latency_ms": elapsed_ms,
            "top_results": reranked_events,
        })

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Stage 5: Multi-Hop EventGraph Retrieval Search Demo")
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
        "--max-hops",
        type=int,
        default=2,
        help="Maximum graph traversal hop depth",
    )
    parser.add_argument(
        "--graph-weight",
        type=float,
        default=0.35,
        help="Weight assigned to graph propagation score (0.0 to 1.0)",
    )
    args = parser.parse_args()

    sample_queries = [
        "người đi bộ qua đường ở ngã tư",
        "xe ô tô màu đỏ di chuyển rẽ trái",
        "cảnh quay toàn cảnh thành phố ban đêm",
    ]

    run_stage5_graph_search(
        nodes_path=Path(args.nodes_in),
        edges_path=Path(args.edges_in),
        sample_queries=sample_queries,
        max_hops=args.max_hops,
        graph_weight=args.graph_weight,
    )


if __name__ == "__main__":
    main()
