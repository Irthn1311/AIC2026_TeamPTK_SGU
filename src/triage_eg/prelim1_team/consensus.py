"""Unsupervised exact, near-frame, and embedding consensus for team candidates."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


def _frame(row: dict[str, Any]) -> int | None:
    value = row.get("frame_id")
    if value in {None, ""}:
        return None
    return int(value)


def _rank(row: dict[str, Any]) -> int:
    return int(row.get("candidate_rank", row.get("rank", 999999)))


def _normalized(vector: np.ndarray | None) -> np.ndarray | None:
    if vector is None:
        return None
    value = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(value))
    if not np.isfinite(value).all() or not np.isfinite(norm) or norm <= 0:
        return None
    return value / norm


def _components(edges: list[set[int]]) -> list[list[int]]:
    seen: set[int] = set()
    output = []
    for start in range(len(edges)):
        if start in seen:
            continue
        stack, component = [start], []
        seen.add(start)
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in edges[current]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        output.append(sorted(component))
    return output


def consensus_rows(
    members: dict[str, list[dict[str, Any]]],
    *,
    embeddings: dict[str, dict[str, np.ndarray]] | None = None,
    near_frame_tolerance: int = 96,
    cosine_threshold: float = 0.92,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """Return deterministic Top-K medoids without training a classifier."""

    if not 2 <= len(members) <= 5:
        raise ValueError("TEAM_CONSENSUS_REQUIRES_2_TO_5_MEMBERS")
    if near_frame_tolerance < 0 or not 0.0 < cosine_threshold <= 1.0 or top_k <= 0:
        raise ValueError("TEAM_CONSENSUS_INVALID_SETTINGS")
    embeddings = embeddings or {}
    by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for member_id, rows in sorted(members.items()):
        seen: set[tuple[str, str, int | None]] = set()
        for raw in rows:
            row = dict(raw)
            query_id = str(row["query_id"])
            identity = query_id, str(row["video_id"]), _frame(row)
            if identity in seen:
                continue
            seen.add(identity)
            key = str(row.get("embedding_key_a0") or "")
            vector = _normalized(embeddings.get(member_id, {}).get(key)) if key else None
            by_query[query_id].append({**row, "member_id": member_id, "_vector": vector})

    output: list[dict[str, Any]] = []
    for query_id, nodes in sorted(by_query.items()):
        exact_members: dict[tuple[str, int | None], set[str]] = defaultdict(set)
        video_members: dict[str, set[str]] = defaultdict(set)
        for node in nodes:
            identity = str(node["video_id"]), _frame(node)
            exact_members[identity].add(str(node["member_id"]))
            video_members[str(node["video_id"])].add(str(node["member_id"]))

        edges = [set() for _ in nodes]
        similarity = np.eye(len(nodes), dtype=np.float32)
        for left in range(len(nodes)):
            left_vector = nodes[left]["_vector"]
            if left_vector is None:
                continue
            for right in range(left + 1, len(nodes)):
                right_vector = nodes[right]["_vector"]
                if right_vector is None or left_vector.shape != right_vector.shape:
                    continue
                score = float(left_vector @ right_vector)
                similarity[left, right] = similarity[right, left] = score
                if score >= cosine_threshold:
                    edges[left].add(right)
                    edges[right].add(left)
        component_by_node: dict[int, list[int]] = {}
        for component in _components(edges):
            for index in component:
                component_by_node[index] = component

        candidates = []
        for index, node in enumerate(nodes):
            video, frame = str(node["video_id"]), _frame(node)
            near_members = {
                str(other["member_id"])
                for other in nodes
                if str(other["video_id"]) == video
                and (
                    frame is None
                    or _frame(other) is None
                    or abs(int(_frame(other)) - frame) <= near_frame_tolerance
                )
            }
            component = component_by_node[index]
            embedding_members = {str(nodes[value]["member_id"]) for value in component}
            component_pairs = [
                float(similarity[left, right])
                for offset, left in enumerate(component)
                for right in component[offset + 1 :]
                if nodes[left]["_vector"] is not None and nodes[right]["_vector"] is not None
            ]
            compactness = float(np.mean(component_pairs)) if component_pairs else 0.0
            component_ranks = [_rank(nodes[value]) for value in component]
            exact_count = len(exact_members[(video, frame)])
            candidates.append(
                {
                    "node_index": index,
                    "distinct_member_support": max(
                        exact_count, len(near_members), len(embedding_members)
                    ),
                    "exact_vote_count": exact_count,
                    "same_video_vote_count": len(video_members[video]),
                    "near_frame_vote_count": len(near_members),
                    "embedding_cluster_member_count": len(embedding_members),
                    "embedding_cluster_compactness": compactness,
                    "average_member_rank": float(np.mean(component_ranks)),
                    "cluster_members": sorted(embedding_members),
                }
            )
        candidates.sort(
            key=lambda row: (
                -int(row["distinct_member_support"]),
                -int(row["same_video_vote_count"]),
                -float(row["embedding_cluster_compactness"]),
                float(row["average_member_rank"]),
                str(nodes[int(row["node_index"])]["video_id"]),
                _frame(nodes[int(row["node_index"])]) or -1,
                str(nodes[int(row["node_index"])]["member_id"]),
            )
        )
        selected, seen = [], set()
        for summary in candidates:
            node = nodes[int(summary["node_index"])]
            identity = str(node["video_id"]), _frame(node)
            if identity in seen:
                continue
            seen.add(identity)
            selected.append((node, summary))
            if len(selected) == top_k:
                break
        for rank, (node, summary) in enumerate(selected, 1):
            output.append(
                {
                    "query_id": query_id,
                    "consensus_rank": rank,
                    "video_id": str(node["video_id"]),
                    "frame_id": _frame(node),
                    "medoid_member_id": str(node["member_id"]),
                    "medoid_member_rank": _rank(node),
                    **{key: value for key, value in summary.items() if key != "node_index"},
                    "automatic_submission": False,
                }
            )
    return output


__all__ = ["consensus_rows"]
