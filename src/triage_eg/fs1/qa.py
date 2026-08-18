"""Qwen response boundary; candidate IDs always remain deterministic inputs."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GroundingCandidate:
    video_id: str
    frame_id: int
    evidence_rank: int
    evidence: dict[str, Any]


def canonical_answer(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(value)).strip().split())


def parse_qwen_output(raw: str, candidate: GroundingCandidate) -> dict[str, Any] | None:
    try:
        match = re.search(r"\{.*\}", raw, re.S)
        value = json.loads(match.group(0) if match else raw)
        answer = canonical_answer(value["answer"])
        sufficient = value["evidence_sufficient"]
        if not answer or not isinstance(sufficient, bool):
            return None
        return {
            "video_id": candidate.video_id,
            "frame_id": candidate.frame_id,
            "answer": answer,
            "evidence_sufficient": sufficient,
            "evidence_rank": candidate.evidence_rank,
        }
    except (KeyError, TypeError, ValueError, json.JSONDecodeError, AttributeError):
        return None


def bounded_grounding_candidates(
    rows: list[GroundingCandidate], budget: int = 20
) -> list[GroundingCandidate]:
    if budget != 20:
        raise ValueError("FS1 Qwen budget is frozen at 20")
    output, seen = [], set()
    for row in sorted(rows, key=lambda item: (item.evidence_rank, item.video_id, item.frame_id)):
        key = (row.video_id, row.frame_id)
        if key not in seen:
            seen.add(key)
            output.append(row)
        if len(output) == budget:
            break
    return output
