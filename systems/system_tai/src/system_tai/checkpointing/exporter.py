"""UTF-8 JSONL exporter for the proposed shared KIS checkpoint boundary."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from system_tai.common.schemas import KISResult


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class ExportSummary:
    destination: Path
    query_count: int
    record_count: int
    include_internal: bool


class CheckpointExporter:
    def export(
        self,
        results: KISResult | Sequence[KISResult],
        destination: Path,
        *,
        include_internal: bool = False,
    ) -> ExportSummary:
        resolved_results = (results,) if isinstance(results, KISResult) else tuple(results)
        if not resolved_results:
            raise ValueError("at least one KIS result is required")
        seen_query_ids: set[str] = set()
        lines: list[str] = []
        for result in resolved_results:
            if result.query_id in seen_query_ids:
                raise ValueError(f"duplicate query result: {result.query_id}")
            seen_query_ids.add(result.query_id)
            if len(result.ranked_candidates) > 100:
                raise ValueError(f"query {result.query_id} exceeds maximum 100 results")
            seen_pairs: set[tuple[str, int]] = set()
            for expected_rank, candidate in enumerate(result.ranked_candidates, start=1):
                if candidate.rank != expected_rank:
                    raise ValueError(f"query {result.query_id} ranks must be contiguous from one")
                pair = (candidate.video_id, candidate.frame_id)
                if pair in seen_pairs:
                    raise ValueError(
                        f"duplicate query/video/frame tuple: {result.query_id}, {pair}"
                    )
                seen_pairs.add(pair)
                record: dict[str, Any] = {
                    "query_id": result.query_id,
                    "rank": candidate.rank,
                    "video_id": candidate.video_id,
                    "frame_id": candidate.frame_id,
                }
                if include_internal:
                    record["_internal"] = {
                        "score": candidate.score,
                        "clip_row": candidate.clip_row,
                        "keyframe_order": candidate.keyframe_order,
                        "source": candidate.source,
                        "diagnostic_metadata": _json_value(candidate.diagnostic_metadata or {}),
                    }
                lines.append(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return ExportSummary(
            destination=path,
            query_count=len(resolved_results),
            record_count=len(lines),
            include_internal=include_internal,
        )
