# ==============================================================================================================
# Generic QA Rescue Tail Allocator (QA_RESCUE_TAIL_V1)
# ==============================================================================================================

from __future__ import annotations

import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


def _normalize_answer(answer: str) -> str:
    if not answer:
        return ""
    t = unicodedata.normalize("NFKC", str(answer)).casefold()
    return "".join(c for c in t if c.isalnum() or c.isspace()).strip()


@dataclass(frozen=True, slots=True)
class RescueCandidate:
    """Generic structured container for any Round-3 rescue candidate producer."""

    video_id: str
    frame_id: int
    answer: str
    rescue_score: float
    rescue_source: str  # e.g., 'query_expansion', 'multi_crop', 'ocr_provider', 'vlm_verifier'
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.video_id, str) or not self.video_id.strip():
            raise ValueError("video_id must be a non-empty string")
        if not isinstance(self.frame_id, int) or self.frame_id < 0:
            raise ValueError("frame_id must be a non-negative integer")
        if not isinstance(self.answer, str) or not self.answer.strip():
            raise ValueError("answer must be a non-empty string")


def merge_rescue_tail(
    champion_predictions: Sequence[dict[str, Any]],
    rescue_candidates: Sequence[RescueCandidate],
    *,
    prefix_k: int = 95,
    max_rescue: int = 5,
) -> list[dict[str, Any]]:
    """
    Generic pure post-hoc tail allocator for Top-100 submission lists.

    Invariants:
    1. Preserves champion_predictions[:prefix_k] strictly in their exact original order.
    2. Rejects any rescue candidate whose (video_id, frame_id, normalized_answer) tuple
       is already present in champion_predictions.
    3. Rejects duplicate rescue tuples within rescue_candidates.
    4. Emits at most max_rescue valid rescue tuples into ranks 96..100.
    5. Re-indexes rank numbers to be continuous (1..N).
    6. Safe for edge cases: empty predictions, <95 baseline predictions, malformed items.
    """
    if prefix_k < 0:
        raise ValueError("prefix_k must be non-negative")
    if max_rescue < 0:
        raise ValueError("max_rescue must be non-negative")

    # Step 1: Slice champion prefix
    base_prefix = list(champion_predictions[:prefix_k])

    # Build existing key set from all champion predictions to ensure global uniqueness
    existing_keys: set[tuple[str, int, str]] = set()
    for item in champion_predictions:
        vid = str(item.get("video_id", "")).strip()
        fid = int(item.get("frame_id", -1))
        ans = _normalize_answer(str(item.get("answer", "")))
        if vid and fid >= 0 and ans:
            existing_keys.add((vid, fid, ans))

    # Step 2: Filter and admit unique rescue candidates
    admitted_rescue: list[dict[str, Any]] = []
    for cand in rescue_candidates:
        if len(admitted_rescue) >= max_rescue:
            break
        if not isinstance(cand, RescueCandidate):
            continue

        tuple_key = (cand.video_id, cand.frame_id, _normalize_answer(cand.answer))
        if tuple_key in existing_keys:
            continue

        existing_keys.add(tuple_key)
        rescue_item: dict[str, Any] = {
            "video_id": cand.video_id,
            "frame_id": cand.frame_id,
            "answer": cand.answer,
            "score": cand.rescue_score,
            "slot_source": f"RESCUE_TAIL_{cand.rescue_source.upper()}",
            "provenance": dict(cand.provenance),
        }
        admitted_rescue.append(rescue_item)

    # Step 3: Combine base prefix with admitted rescue candidates
    combined: list[dict[str, Any]] = []
    for idx, item in enumerate(base_prefix + admitted_rescue, start=1):
        clean_item = dict(item)
        clean_item["rank"] = idx
        combined.append(clean_item)

    return combined
