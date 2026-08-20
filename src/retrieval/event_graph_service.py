"""
Stage 5: Multi-Hop EventGraph Retrieval & Traversal Service
============================================================
Provides fast in-memory graph traversal and multi-hop propagation reranking across:
  1. TEMPORAL edges (chronological intra-video sequence)
  2. VISUAL_SIMILARITY edges (cross-video visual similarity)
  3. SEMANTIC_CONTINUITY edges (cross-video semantic narrative similarity)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("aic.event_graph_service")


class EventGraphService:
    """
    In-memory EventGraph Traversal & Graph-Augmented Retrieval Engine.
    """

    _instance: Optional[EventGraphService] = None

    @classmethod
    def get_instance(
        cls,
        nodes_path: Optional[Union[str, Path]] = None,
        edges_path: Optional[Union[str, Path]] = None,
    ) -> EventGraphService:
        if cls._instance is None:
            cls._instance = cls(nodes_path=nodes_path, edges_path=edges_path)
        return cls._instance

    def __init__(
        self,
        nodes_path: Optional[Union[str, Path]] = None,
        edges_path: Optional[Union[str, Path]] = None,
    ):
        self.nodes_path = Path(nodes_path) if nodes_path else None
        self.edges_path = Path(edges_path) if edges_path else None

        if self.nodes_path is None or not self.nodes_path.exists():
            # Fallback path discovery
            cand_paths = [
                PROJECT_ROOT / "artifacts" / "event_graph" / "graph" / "event_nodes.parquet",
                PROJECT_ROOT / "outputs" / "event_graph" / "event_nodes.parquet",
            ]
            for cp in cand_paths:
                if cp.exists():
                    self.nodes_path = cp
                    break

        if self.edges_path is None or not self.edges_path.exists():
            cand_paths = [
                PROJECT_ROOT / "artifacts" / "event_graph" / "graph" / "event_edges.parquet",
                PROJECT_ROOT / "outputs" / "event_graph" / "event_edges.parquet",
            ]
            for cp in cand_paths:
                if cp.exists():
                    self.edges_path = cp
                    break

        self.df_nodes: Optional[pd.DataFrame] = None
        self.df_edges: Optional[pd.DataFrame] = None

        # Adjacency maps
        self.node_map: Dict[str, Dict[str, Any]] = {}
        self.outgoing_edges: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.incoming_edges: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.video_to_events: Dict[str, List[str]] = defaultdict(list)

        self.initialized = False

    def initialize(self) -> None:
        """Loads event_nodes.parquet and event_edges.parquet into memory graph structures."""
        if self.initialized:
            return

        t0 = time.time()
        logger.info("Initializing Stage 5 EventGraphService...")

        if self.nodes_path is None or not self.nodes_path.exists():
            raise FileNotFoundError(f"Event nodes parquet not found at: {self.nodes_path}")
        if self.edges_path is None or not self.edges_path.exists():
            raise FileNotFoundError(f"Event edges parquet not found at: {self.edges_path}")

        self.df_nodes = pd.read_parquet(self.nodes_path)
        self.df_edges = pd.read_parquet(self.edges_path)

        # 1. Build Node Map
        for idx, row in self.df_nodes.iterrows():
            record = row.to_dict()
            ev_id = str(record.get("event_id", ""))
            vid = str(record.get("video_id", ""))
            self.node_map[ev_id] = record
            if vid:
                self.video_to_events[vid].append(ev_id)

        # 2. Build Adjacency Graphs
        for idx, row in self.df_edges.iterrows():
            rec = row.to_dict()
            src = str(rec["src_event_id"])
            dst = str(rec["dst_event_id"])
            etype = str(rec["edge_type"])
            sc = float(rec["score"])
            z_sc = float(rec.get("z_score", 0.0))

            edge_obj = {
                "src_event_id": src,
                "dst_event_id": dst,
                "edge_type": etype,
                "score": sc,
                "z_score": z_sc,
                "src_video_id": str(rec.get("src_video_id", "")),
                "dst_video_id": str(rec.get("dst_video_id", "")),
            }

            self.outgoing_edges[src].append(edge_obj)
            self.incoming_edges[dst].append(edge_obj)

            # Undirected handling for similarity relations
            if etype in ["VISUAL_SIMILARITY", "SEMANTIC_CONTINUITY"]:
                reverse_edge = dict(edge_obj)
                reverse_edge["src_event_id"] = dst
                reverse_edge["dst_event_id"] = src
                reverse_edge["src_video_id"] = str(rec.get("dst_video_id", ""))
                reverse_edge["dst_video_id"] = str(rec.get("src_video_id", ""))
                self.outgoing_edges[dst].append(reverse_edge)
                self.incoming_edges[src].append(reverse_edge)

        self.initialized = True
        logger.info(
            "EventGraphService initialized in %.2fs (%d nodes, %d directed edges).",
            time.time() - t0,
            len(self.node_map),
            len(self.df_edges),
        )

    def get_node(self, event_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve node metadata by event_id."""
        if not self.initialized:
            self.initialize()
        return self.node_map.get(event_id)

    def get_neighbors(
        self, event_id: str, edge_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve 1-hop outgoing neighbors for a given event_id."""
        if not self.initialized:
            self.initialize()
        edges = self.outgoing_edges.get(event_id, [])
        if edge_types:
            return [e for e in edges if e["edge_type"] in edge_types]
        return edges

    def traverse_multihop(
        self,
        seed_scores: Dict[str, float],
        max_hops: int = 2,
        decay: float = 0.80,
        edge_types: Optional[List[str]] = None,
        temporal_boost: float = 1.20,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Executes Multi-Hop Graph Traversal starting from initial seed event scores.
        
        Args:
            seed_scores: Map of event_id -> initial 1st-hop direct search score
            max_hops: Maximum traversal depth (1 to 3)
            decay: Per-hop attenuation factor
            edge_types: Filter edge types (e.g. ['TEMPORAL', 'SEMANTIC_CONTINUITY'])
            temporal_boost: Multiplier boost for consecutive TEMPORAL edges in same video

        Returns:
            Dict mapping event_id -> {
                'graph_score': float,
                'hop_distance': int,
                'expanded_from': str,
                'edge_type_path': List[str]
            }
        """
        if not self.initialized:
            self.initialize()

        traversal_results: Dict[str, Dict[str, Any]] = {}
        queue: deque[Tuple[str, float, int, str, List[str]]] = deque()

        # Initialize queue with seeds at Hop 0
        for ev_id, init_sc in seed_scores.items():
            if init_sc > 0:
                traversal_results[ev_id] = {
                    "graph_score": float(init_sc),
                    "hop_distance": 0,
                    "expanded_from": ev_id,
                    "edge_type_path": [],
                }
                queue.append((ev_id, float(init_sc), 0, ev_id, []))

        while queue:
            curr_id, curr_sc, curr_hop, origin_id, path_types = queue.popleft()

            if curr_hop >= max_hops:
                continue

            neighbors = self.get_neighbors(curr_id, edge_types=edge_types)
            for e in neighbors:
                dst_id = e["dst_event_id"]
                e_sc = float(e["score"])
                e_type = str(e["edge_type"])

                # Temporal intra-video boost
                multiplier = temporal_boost if e_type == "TEMPORAL" else 1.0
                next_sc = curr_sc * e_sc * decay * multiplier

                next_hop = curr_hop + 1
                next_path = path_types + [e_type]

                # Update or insert if higher score achieved
                existing = traversal_results.get(dst_id)
                if existing is None or next_sc > existing["graph_score"]:
                    traversal_results[dst_id] = {
                        "graph_score": round(next_sc, 4),
                        "hop_distance": next_hop,
                        "expanded_from": origin_id,
                        "edge_type_path": next_path,
                    }
                    queue.append((dst_id, next_sc, next_hop, origin_id, next_path))

        return traversal_results

    def rerank_events_with_graph(
        self,
        direct_candidates: List[Dict[str, Any]],
        max_hops: int = 2,
        graph_weight: float = 0.35,
        decay: float = 0.80,
        top_k: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Fuses direct multimodal retrieval scores with Multi-Hop EventGraph propagation scores.

        Args:
            direct_candidates: List of candidate event dicts containing 'event_id' and 'fused_score'
            max_hops: Depth of graph traversal
            graph_weight: Weight assigned to graph propagation (0.0 to 1.0)
            decay: Per-hop attenuation factor
            top_k: Number of final reranked events to return

        Returns:
            List of reranked event dicts enriched with graph trajectory metadata.
        """
        if not self.initialized:
            self.initialize()

        if not direct_candidates:
            return []

        # 1. Extract seed scores
        seed_scores = {str(c["event_id"]): float(c.get("fused_score", 0.0)) for c in direct_candidates}

        # 2. Execute Multi-Hop Graph Propagation
        graph_traversal = self.traverse_multihop(
            seed_scores=seed_scores,
            max_hops=max_hops,
            decay=decay,
        )

        # 3. Normalize Graph Scores
        graph_score_vals = [g["graph_score"] for g in graph_traversal.values()]
        max_g = max(graph_score_vals) if graph_score_vals else 1.0
        min_g = min(graph_score_vals) if graph_score_vals else 0.0
        denom_g = max_g - min_g if max_g > min_g else 1.0

        # 4. Merge candidates
        all_event_ids = set(seed_scores.keys()) | set(graph_traversal.keys())
        reranked: List[Dict[str, Any]] = []

        direct_map = {str(c["event_id"]): c for c in direct_candidates}

        for ev_id in all_event_ids:
            direct_rec = direct_map.get(ev_id)
            node_rec = self.node_map.get(ev_id, {})

            direct_sc = float(direct_rec.get("fused_score", 0.0)) if direct_rec else 0.0
            
            gt_info = graph_traversal.get(ev_id, {})
            raw_g_sc = float(gt_info.get("graph_score", 0.0))
            norm_g_sc = (raw_g_sc - min_g) / denom_g if raw_g_sc > 0 else 0.0

            # Final Fused Score
            final_sc = ((1.0 - graph_weight) * direct_sc) + (graph_weight * norm_g_sc)

            # Combine metadata
            merged_item = dict(node_rec) if node_rec else (dict(direct_rec) if direct_rec else {"event_id": ev_id})
            if direct_rec:
                merged_item.update(direct_rec)

            merged_item["final_graph_score"] = round(final_sc, 4)
            merged_item["direct_multimodal_score"] = round(direct_sc, 4)
            merged_item["graph_propagation_score"] = round(norm_g_sc, 4)
            merged_item["hop_distance"] = gt_info.get("hop_distance", 0 if direct_rec else 99)
            merged_item["expanded_from"] = gt_info.get("expanded_from", ev_id)
            merged_item["edge_type_path"] = gt_info.get("edge_type_path", [])

            reranked.append(merged_item)

        reranked.sort(key=lambda x: x["final_graph_score"], reverse=True)
        return reranked[:top_k]


def main():
    """Quick sanity CLI test for EventGraphService."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    
    graph_svc = EventGraphService.get_instance()
    graph_svc.initialize()

    # Sample test query seeds
    sample_nodes = list(graph_svc.node_map.keys())[:3]
    print(f"\nTesting Graph Traversal from sample seeds: {sample_nodes}")

    seed_scores = {ev_id: 0.95 - (i * 0.1) for i, ev_id in enumerate(sample_nodes)}
    results = graph_svc.traverse_multihop(seed_scores, max_hops=2)

    print(f"Propagated Graph traversal expanded to {len(results)} total events:")
    for ev_id, info in list(results.items())[:10]:
        print(f" • Event: {ev_id} | Hop: {info['hop_distance']} | Score: {info['graph_score']:.4f} | Path: {' -> '.join(info['edge_type_path']) if info['edge_type_path'] else 'SEED'}")


if __name__ == "__main__":
    main()
