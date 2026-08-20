"""Executable per-event graph whose revised state feeds the T3 solver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from triage_eg.experiments.t3_diverse_temporal import (
    EventCandidate as T3Candidate,
)
from triage_eg.experiments.t3_diverse_temporal import (
    enumerate_feasible_paths,
    select_coverage_aware,
)

from .events import CompiledEvent


@dataclass(frozen=True)
class Candidate:
    event_index: int
    video_id: str
    frame_id: int
    rank: int
    source: str
    evidence: dict[str, Any]


class ExecutableEventGraph:
    def __init__(self, query_id: str, events: tuple[CompiledEvent, ...]) -> None:
        self.query_id = query_id
        self.events = events
        self.by_event: dict[int, list[Candidate]] = {event.event_index: [] for event in events}
        self.revision_count = 0
        self.edges: list[dict[str, Any]] = []
        self.revision: dict[str, Any] | None = None

    def add(self, candidate: Candidate) -> None:
        if candidate.event_index not in self.by_event or not candidate.evidence:
            raise ValueError("graph candidate requires valid event and provenance evidence")
        key = (candidate.video_id, candidate.frame_id, candidate.source)
        if key not in {
            (item.video_id, item.frame_id, item.source)
            for item in self.by_event[candidate.event_index]
        }:
            self.by_event[candidate.event_index].append(candidate)
            self.edges.append(
                {
                    "type": "SUPPORTS",
                    "event_index": candidate.event_index,
                    "video_id": candidate.video_id,
                    "frame_id": candidate.frame_id,
                    "provenance": candidate.evidence,
                }
            )
            for adjacent_index in (candidate.event_index - 1, candidate.event_index + 1):
                for adjacent in self.by_event.get(adjacent_index, []):
                    earlier, later = (
                        (adjacent, candidate)
                        if adjacent.event_index < candidate.event_index
                        else (candidate, adjacent)
                    )
                    if earlier.video_id == later.video_id and earlier.frame_id < later.frame_id:
                        self.edges.append(
                            {
                                "type": "PRECEDES",
                                "from_event_index": earlier.event_index,
                                "to_event_index": later.event_index,
                                "video_id": earlier.video_id,
                                "from_frame_id": earlier.frame_id,
                                "to_frame_id": later.frame_id,
                            }
                        )

    def missing(self) -> list[int]:
        return [index for index, values in self.by_event.items() if not values]

    def revise_once(self, action: str, event_index: int, added: list[Candidate]) -> None:
        if self.revision_count:
            raise RuntimeError("GRAPH_REVISION_LIMIT_EXCEEDED")
        if action not in {"EXPLOIT", "EXPLORE"} or not added:
            raise RuntimeError("GRAPH_REVISION_MUST_ADD_REAL_EVIDENCE")
        occupied = {
            (candidate.video_id, candidate.frame_id)
            for values in self.by_event.values()
            for candidate in values
        }
        novel_coordinates = {
            (candidate.video_id, candidate.frame_id)
            for candidate in added
            if (candidate.video_id, candidate.frame_id) not in occupied
        }
        if not novel_coordinates:
            raise RuntimeError("GRAPH_REVISION_ADDED_NO_NEW_COORDINATE")
        before = self.diagnostics()
        for candidate in added:
            if candidate.event_index != event_index:
                raise ValueError("revision candidate event mismatch")
            self.add(candidate)
        after = self.diagnostics()
        actual_nodes_added = after["node_count"] - before["node_count"]
        if actual_nodes_added <= 0:
            raise RuntimeError("GRAPH_REVISION_ADDED_NO_NEW_CANDIDATE")
        self.revision_count = 1
        self.revision = {
            "action": action,
            "event_index": event_index,
            "nodes_before": before["node_count"],
            "nodes_after": after["node_count"],
            "edges_before": before["edge_count"],
            "edges_after": after["edge_count"],
            "missing_before": before["missing_events"],
            "missing_after": after["missing_events"],
            "evidence_added": len(novel_coordinates),
            "candidates_added": actual_nodes_added,
            "novel_coordinate_count": len(novel_coordinates),
        }

    def diagnostics(self) -> dict[str, Any]:
        candidates = sum(len(values) for values in self.by_event.values())
        return {
            "query_id": self.query_id,
            "query_event_count": len(self.events),
            "node_count": 1 + len(self.events) + candidates,
            "edge_count": len(self.edges),
            "edge_types": sorted({edge["type"] for edge in self.edges}),
            "missing_events": self.missing(),
            "revision_count": self.revision_count,
            "revision": self.revision,
        }


def build_graph_chains(graph: ExecutableEventGraph, *, limit: int = 100) -> list[dict[str, Any]]:
    """Adapt graph state into the frozen T3 feasible-path and coverage selector."""

    return solve_event_candidates(graph.query_id, graph.events, graph.by_event, limit=limit)


def solve_event_candidates(
    query_id: str,
    events: tuple[CompiledEvent, ...],
    by_event: dict[int, list[Candidate]],
    *,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Ground-truth-free T3 adapter shared by graph-free M0 and graph-driven M1."""

    if len(events) < 2 or any(not by_event.get(event.event_index) for event in events):
        return []
    videos = sorted(
        set.intersection(
            *[{candidate.video_id for candidate in by_event[event.event_index]} for event in events]
        )
    )
    chains: list[dict[str, Any]] = []
    for video_id in videos:
        pools = []
        frame_maps = []
        for event in events:
            values = sorted(
                (
                    candidate
                    for candidate in by_event[event.event_index]
                    if candidate.video_id == video_id
                ),
                key=lambda item: (item.rank, item.frame_id, item.source),
            )[:10]
            frame_maps.append({index: value for index, value in enumerate(values)})
            pools.append(
                tuple(
                    T3Candidate(
                        event_id=f"{query_id}:{event.event_index}",
                        event_region_id=f"{query_id}:{event.event_index}:{value.source}:{value.frame_id}",
                        catalog_position=index,
                        original_frame_idx=value.frame_id,
                        similarity=1.0 / (60 + value.rank),
                    )
                    for index, value in enumerate(values)
                )
            )
        # T3 expects temporal positions; sort the shared video pool by actual frame.
        unified = sorted(
            {candidate.frame_id for mapping in frame_maps for candidate in mapping.values()}
        )
        position = {frame: index for index, frame in enumerate(unified)}
        adjusted = tuple(
            tuple(
                T3Candidate(
                    item.event_id,
                    item.event_region_id,
                    position[item.original_frame_idx],
                    item.original_frame_idx,
                    item.similarity,
                )
                for item in pool
            )
            for pool in pools
        )
        feasible, _ = enumerate_feasible_paths(adjusted)
        selected = select_coverage_aware(feasible, 0.05)
        for path in selected:
            frames = tuple(unified[index] for index in path.positions)
            chains.append(
                {
                    "query_id": query_id,
                    "video_id": video_id,
                    "frame_ids": list(frames),
                    "graph_score": path.score,
                    "source": "event_graph_t3",
                }
            )
    chains.sort(key=lambda row: (-row["graph_score"], row["video_id"], row["frame_ids"]))
    return [{**row, "rank": rank} for rank, row in enumerate(chains[:limit], 1)]
