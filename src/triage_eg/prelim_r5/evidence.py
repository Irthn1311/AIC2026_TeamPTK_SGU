"""GT-free ASR/E5 multi-view evidence fusion over frozen external artifacts."""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from .fusion import R5Settings, candidate_key
from .views import VIEW_NAMES

_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)


def _fold(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(value)).casefold()
    return " ".join(
        "".join(character for character in token if not unicodedata.combining(character))
        for token in _TOKEN.findall(normalized)
    )


def _high_phrases(view: dict[str, Any]) -> list[str]:
    phrases = []
    for source in view.get("source_spans", []):
        compact = _fold(str(source))
        if len(compact.split()) >= 2 and compact not in phrases:
            phrases.append(compact)
    return phrases


def _row_rank(row: dict[str, Any], branch: str) -> int:
    keys = ("asr_rank", "rank") if branch == "LEXICAL" else ("rank", "asr_rank")
    return next(int(row[key]) for key in keys if row.get(key) is not None)


def _representatives(task: str, branches: list[list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for rows in branches:
        for row in sorted(rows, key=lambda value: int(value["rank"])):
            video_id = str(row["video_id"])
            if video_id in output:
                continue
            if task == "TRAKE":
                frames = tuple(int(value) for value in row.get("frame_ids", []))
                if not frames or any(
                    left >= right for left, right in zip(frames, frames[1:], strict=False)
                ):
                    continue
            output[video_id] = dict(row)
    return output


def fuse_asr_multiview(
    query: dict[str, Any],
    views: list[dict[str, Any]],
    lexical_by_view: dict[str, list[dict[str, Any]]],
    e5_by_view: dict[str, list[dict[str, Any]]],
    *,
    a0_multiview: list[dict[str, Any]],
    s1_multiview: list[dict[str, Any]],
    fallback_rows: list[dict[str, Any]],
    safe_r4_rows: list[dict[str, Any]],
    ocr_rows: list[dict[str, Any]] | None = None,
    settings: R5Settings | None = None,
) -> dict[str, Any]:
    """Fuse ASR views by ranks and retain only canonical existing coordinates."""

    settings = settings or R5Settings()
    query_id, task = str(query["query_id"]), str(query["task"]).upper()
    view_rows = {row["view"]: row for row in views if row.get("event_index") is None}
    if task == "TRAKE":
        view_rows = {}
        for name in VIEW_NAMES:
            selected = sorted(
                (row for row in views if row["view"] == name),
                key=lambda row: int(row["event_index"]),
            )
            view_rows[name] = {
                "view": name,
                "source_spans": [span for row in selected for span in row["source_spans"]],
            }
    if set(view_rows) != set(VIEW_NAMES):
        raise RuntimeError(f"R5_ASR_VIEW_SET_MISMATCH:{query_id}:{sorted(view_rows)}")
    source_text = " | ".join(
        _fold(str(row.get("source_text", ""))) for row in views if row.get("source_text")
    )
    high_phrases = {
        phrase
        for row in view_rows.values()
        for phrase in _high_phrases(row)
        if phrase in source_text
    }

    representatives = _representatives(
        task,
        [a0_multiview, s1_multiview, safe_r4_rows, fallback_rows],
    )
    visual_videos = {str(row["video_id"]) for row in [*a0_multiview[:20], *s1_multiview[:20]]}
    evidence: dict[str, dict[str, Any]] = {}
    for view_name in VIEW_NAMES:
        for branch, rows in (
            ("LEXICAL", lexical_by_view.get(view_name, [])),
            ("E5", e5_by_view.get(view_name, [])),
        ):
            seen = set()
            for row in rows[: settings.max_predictions]:
                video_id = str(row["video_id"])
                if video_id in seen or video_id not in representatives:
                    continue
                seen.add(video_id)
                rank = _row_rank(row, branch)
                item = evidence.setdefault(
                    video_id,
                    {
                        "score": 0.0,
                        "best_rank": rank,
                        "view_branches": {},
                        "spans": [],
                        "high_phrases": set(),
                    },
                )
                item["score"] += 1.0 / (settings.rrf_k + rank)
                item["best_rank"] = min(int(item["best_rank"]), rank)
                item["view_branches"].setdefault(view_name, {})[branch] = rank
                item["spans"].append({**row, "view": view_name, "branch": branch})
                text = _fold(str(row.get("text", "")))
                item["high_phrases"].update(phrase for phrase in high_phrases if phrase in text)

    ocr_by_video: dict[str, list[dict[str, Any]]] = {}
    for row in ocr_rows or []:
        ocr_by_video.setdefault(str(row.get("video_id", "")), []).append(row)
    ordered = sorted(
        evidence.items(),
        key=lambda item: (-float(item[1]["score"]), int(item[1]["best_rank"]), item[0]),
    )
    rows, provenance, gated = [], [], []
    for rank, (video_id, item) in enumerate(ordered, 1):
        representative = {**representatives[video_id], "query_id": query_id, "rank": rank}
        branch_pairs = item["view_branches"]
        agreement = any({"LEXICAL", "E5"}.issubset(values) for values in branch_pairs.values())
        high_count = len(item["high_phrases"])
        visual = video_id in visual_videos
        ocr_high = any(
            any(phrase in _fold(str(row.get("text", ""))) for phrase in item["high_phrases"])
            for row in ocr_by_video.get(video_id, [])
        )
        tier = (
            "TIER_A_DIRECT"
            if high_count and agreement
            else "TIER_B_CORROBORATED"
            if high_count and visual
            else "TIER_C_WEAK"
        )
        rows.append({**representative, "system_variant": "R5_ASR_E5_MULTIVIEW"})
        record = {
            "query_id": query_id,
            "task": task,
            "video_id": video_id,
            "rank": rank,
            "candidate_key": list(candidate_key(task, representative)),
            "tier": tier,
            "rrf_k": settings.rrf_k,
            "rrf_score": item["score"],
            "best_view_rank": item["best_rank"],
            "views_agreeing": len(branch_pairs),
            "view_ranks": branch_pairs,
            "high_anchor_match_count": high_count,
            "matched_high_phrases": sorted(item["high_phrases"]),
            "lexical_e5_agreement": agreement,
            "visual_support": visual,
            "ocr_exact_high_with_visual": ocr_high and visual,
            "generic_view_only": set(branch_pairs) == {"ORIGINAL_VI"} and not high_count,
            "source_spans": item["spans"][:10],
            "gt_used": False,
        }
        provenance.append(record)
        gated.append({**record, "representative": representative})
    strong = next(
        (
            {
                **row,
                "qualified": True,
                "representative": representatives[row["video_id"]],
            }
            for row in provenance
            if row["tier"] == "TIER_A_DIRECT" and row["visual_support"]
        ),
        None,
    )
    return {
        "rows": rows[: settings.max_predictions],
        "provenance": provenance,
        "gated_candidates": gated,
        "strong": strong,
    }


def qa_evidence_from_asr(
    asr_result: dict[str, Any],
    context_rows: list[dict[str, Any]],
    *,
    ocr_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Map bounded text evidence to already retrieved canonical context frames."""

    frames: dict[str, int] = {}
    for row in context_rows[:20]:
        if row.get("frame_id") is not None:
            frames.setdefault(str(row["video_id"]), int(row["frame_id"]))
    output = []
    for item in asr_result.get("provenance", []):
        video_id = str(item["video_id"])
        if video_id not in frames:
            continue
        for span in item.get("source_spans", []):
            output.append(
                {
                    **span,
                    "source": "ASR_EXTERNAL_V3_VALIDATED",
                    "frame_id": frames[video_id],
                }
            )
    for row in ocr_rows or []:
        video_id = str(row.get("video_id", ""))
        if video_id in frames:
            output.append({**row, "source": "EXTERNAL_OCR", "frame_id": frames[video_id]})
    return output


__all__ = ["fuse_asr_multiview", "qa_evidence_from_asr"]
