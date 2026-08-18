"""Rank-only fixed RRF60 fusion and B0 prefix protection."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any

from .contracts import FS1Settings


def default_key(row: dict[str, Any]) -> tuple[Any, ...]:
    if "frame_ids" in row:
        return row.get("video_id"), tuple(row.get("frame_ids", ()))
    return row.get("video_id"), row.get("frame_id"), row.get("answer")


def reciprocal_rank_fusion(
    lists: Sequence[Sequence[dict[str, Any]]],
    *,
    key: Callable[[dict[str, Any]], Any] = default_key,
    k: int = 60,
) -> list[dict[str, Any]]:
    if k != 60:
        raise ValueError("FS1 only permits RRF k=60")
    scores: dict[Any, float] = defaultdict(float)
    first: dict[Any, dict[str, Any]] = {}
    provenance: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for source_index, rows in enumerate(lists):
        seen = set()
        for rank, row in enumerate(rows, 1):
            identity = key(row)
            if identity in seen:
                continue
            seen.add(identity)
            scores[identity] += 1.0 / (k + rank)
            first.setdefault(identity, dict(row))
            provenance[identity].append({"source": source_index, "rank": rank})
    order = sorted(scores, key=lambda identity: (-scores[identity], repr(identity)))
    return [
        {
            **first[identity],
            "fs1_rrf_score": scores[identity],
            "fs1_source_ranks": provenance[identity],
        }
        for identity in order
    ]


def fuse_tail(
    task: str,
    b0: Sequence[dict[str, Any]],
    evidence_lists: Sequence[Sequence[dict[str, Any]]],
    *,
    settings: FS1Settings | None = None,
) -> list[dict[str, Any]]:
    settings = settings or FS1Settings()
    task = task.upper()
    protected = list(b0[: settings.protected_prefix]) if task in {"KIS", "TRAKE"} else []
    blocked = {default_key(row) for row in protected}
    sources = [
        list(b0[settings.protected_prefix :] if protected else b0),
        *map(list, evidence_lists),
    ]
    tail = [
        row
        for row in reciprocal_rank_fusion(sources, k=settings.rrf_k)
        if default_key(row) not in blocked
    ]
    output = [dict(row) for row in protected] + tail
    return [{**row, "rank": rank} for rank, row in enumerate(output[: settings.max_predictions], 1)]


def assert_protected_prefix(
    task: str, b0: Sequence[dict[str, Any]], candidate: Sequence[dict[str, Any]], count: int = 5
) -> None:
    if task.upper() in {"KIS", "TRAKE"} and [default_key(x) for x in b0[:count]] != [
        default_key(x) for x in candidate[:count]
    ]:
        raise RuntimeError("FS1_B0_TOP5_NOT_PRESERVED")
