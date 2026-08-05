"""Optional deterministic temporal suppression after exact KIS ranking."""

from __future__ import annotations

from dataclasses import dataclass, replace

from system_tai.common.schemas import KISResult


@dataclass(frozen=True, slots=True)
class TemporalSuppressionConfig:
    enabled: bool = False
    minimum_frame_gap: int = 0
    maximum_candidates_per_video: int | None = None

    def __post_init__(self) -> None:
        if self.minimum_frame_gap < 0:
            raise ValueError("minimum_frame_gap must be non-negative")
        if self.maximum_candidates_per_video is not None and self.maximum_candidates_per_video <= 0:
            raise ValueError("maximum_candidates_per_video must be positive")


@dataclass(frozen=True, slots=True)
class TemporalSuppressionReport:
    enabled: bool
    input_count: int
    output_count: int
    removed_count: int


class KISRanker:
    def apply(
        self,
        result: KISResult,
        config: TemporalSuppressionConfig | None = None,
    ) -> tuple[KISResult, TemporalSuppressionReport]:
        resolved = config or TemporalSuppressionConfig()
        input_count = len(result.ranked_candidates)
        if not resolved.enabled:
            return result, TemporalSuppressionReport(
                enabled=False,
                input_count=input_count,
                output_count=input_count,
                removed_count=0,
            )

        selected = []
        selected_frames: dict[str, list[int]] = {}
        selected_counts: dict[str, int] = {}
        for candidate in result.ranked_candidates:
            video_id = candidate.video_id
            if (
                resolved.maximum_candidates_per_video is not None
                and selected_counts.get(video_id, 0) >= resolved.maximum_candidates_per_video
            ):
                continue
            if any(
                abs(candidate.frame_id - existing) < resolved.minimum_frame_gap
                for existing in selected_frames.get(video_id, [])
            ):
                continue
            selected.append(candidate)
            selected_frames.setdefault(video_id, []).append(candidate.frame_id)
            selected_counts[video_id] = selected_counts.get(video_id, 0) + 1

        reranked = tuple(
            replace(candidate, rank=rank) for rank, candidate in enumerate(selected, start=1)
        )
        output = KISResult(query_id=result.query_id, ranked_candidates=reranked)
        return output, TemporalSuppressionReport(
            enabled=True,
            input_count=input_count,
            output_count=len(reranked),
            removed_count=input_count - len(reranked),
        )
