"""Deterministic cross-level held-out source-pool selection."""

from __future__ import annotations

import hashlib
import math
from collections import Counter, deque
from typing import Any


def _key(seed: int, video_id: str) -> str:
    return hashlib.sha256(f"{seed}:{video_id}".encode()).hexdigest()


def select_heldout_candidates(
    inventory: list[dict[str, Any]],
    census: list[dict[str, Any]],
    *,
    count: int = 36,
    blind_count: int = 24,
    seed: int = 20260821,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not 0 < blind_count < count:
        raise ValueError("blind_count must be positive and smaller than count")
    tier_by_id = {row["video_id"]: row["usage_tier"] for row in census}
    technically_valid = [
        row
        for row in inventory
        if row["top_level"] != "L21"
        and 22 <= int(row["top_level"][1:]) <= 30
        and row["valid_frame_metadata"]
        and row["mapping_available"]
        and row["keyframes_available"]
    ]
    t0 = [row for row in technically_valid if tier_by_id.get(row["video_id"]) == "T0_UNREFERENCED"]
    t1 = [row for row in technically_valid if tier_by_id.get(row["video_id"]) == "T1_INFRA_ONLY"]
    if len(t0) + len(t1) < count:
        raise RuntimeError(
            f"HELDOUT_FAIL_CLOSED_T0_T1_SHORTAGE: required={count} available={len(t0) + len(t1)}"
        )
    pool = t0 if len(t0) >= count else t0 + t1
    levels = sorted({row["top_level"] for row in pool})
    max_per_level = max(1, math.ceil(count / max(1, len(levels))) + 1)
    queues: dict[str, deque[dict[str, Any]]] = {}
    for level in levels:
        values = []
        for tier in ("T0_UNREFERENCED", "T1_INFRA_ONLY"):
            source_queues = {}
            for source_group in sorted(
                {
                    row["source_group"]
                    for row in pool
                    if row["top_level"] == level and tier_by_id[row["video_id"]] == tier
                }
            ):
                source_values = [
                    row
                    for row in pool
                    if row["top_level"] == level
                    and row["source_group"] == source_group
                    and tier_by_id[row["video_id"]] == tier
                ]
                source_values.sort(key=lambda row: _key(seed, row["video_id"]))
                source_queues[source_group] = deque(source_values)
            while any(source_queues.values()):
                for source_group in sorted(source_queues):
                    if source_queues[source_group]:
                        values.append(source_queues[source_group].popleft())
        queues[level] = deque(values)
    selected, level_counts = [], Counter()
    while len(selected) < count:
        progressed = False
        for level in levels:
            if len(selected) >= count:
                break
            if level_counts[level] >= max_per_level or not queues[level]:
                continue
            selected.append(queues[level].popleft())
            level_counts[level] += 1
            progressed = True
        if not progressed:
            raise RuntimeError("HELDOUT_BALANCE_CONSTRAINT_SHORTAGE")
    role_order = sorted(selected, key=lambda row: _key(seed + 1, row["video_id"]))
    blind_ids = {row["video_id"] for row in role_order[:blind_count]}
    output = []
    for selection_index, row in enumerate(selected, 1):
        role = "BLIND_CANDIDATE_POOL" if row["video_id"] in blind_ids else "SEALED_CANDIDATE_POOL"
        output.append(
            {
                "selection_index": selection_index,
                "video_id": row["video_id"],
                "top_level": row["top_level"],
                "source_group": row["source_group"],
                "role": role,
                "usage_tier": tier_by_id[row["video_id"]],
                "video_path": row["video_path"],
                "mapping_path": row["mapping_path"],
                "keyframe_directory": row["keyframe_directory"],
                "fps": row["fps"],
                "total_frames": row["total_frames"],
                "role_preassigned_before_visual_inspection": True,
            }
        )
    blind = {row["video_id"] for row in output if row["role"].startswith("BLIND")}
    sealed = {row["video_id"] for row in output if row["role"].startswith("SEALED")}
    tier_counts = Counter(row["usage_tier"] for row in output)
    source_group_counts = Counter(row["source_group"] for row in output)
    return output, {
        "status": "PASS",
        "seed": seed,
        "candidate_count": len(output),
        "blind_candidate_count": len(blind),
        "sealed_candidate_count": len(sealed),
        "blind_sealed_video_overlap": len(blind & sealed),
        "l21_eligible": False,
        "top_level_coverage": sorted(level_counts),
        "top_level_count": len(level_counts),
        "by_top_level": dict(sorted(level_counts.items())),
        "by_source_group": dict(sorted(source_group_counts.items())),
        "max_per_top_level": max_per_level,
        "selected_by_usage_tier": dict(sorted(tier_counts.items())),
        "t0_available": len(t0),
        "t1_available": len(t1),
        "t1_fallback_used": tier_counts.get("T1_INFRA_ONLY", 0) > 0,
        "t1_fallback_reason": (
            f"T0 shortage: required={count}, available={len(t0)}"
            if tier_counts.get("T1_INFRA_ONLY", 0) > 0
            else None
        ),
        "t2_t3_used": False,
        "roles_frozen_before_visual_review": True,
    }


__all__ = ["select_heldout_candidates"]
