"""Frame-level result grouping and local duplicate suppression."""

from collections import defaultdict

from triage_eg.common.schemas import CandidateFrame


def deduplicate_nearby_frames(
    candidates: list[CandidateFrame], max_frame_distance: int
) -> list[CandidateFrame]:
    """Keep the highest-scoring candidate in nearby same-video regions."""

    if max_frame_distance < 0:
        raise ValueError("max_frame_distance must be non-negative")
    kept: list[CandidateFrame] = []
    for candidate in sorted(candidates, key=lambda item: (-item.score, item.rank)):
        overlaps = any(
            existing.video_id == candidate.video_id
            and abs(existing.frame_id - candidate.frame_id) <= max_frame_distance
            for existing in kept
        )
        if not overlaps:
            kept.append(candidate)
    return sorted(kept, key=lambda item: (-item.score, item.rank))


def group_candidates_by_video(
    candidates: list[CandidateFrame],
) -> dict[str, list[CandidateFrame]]:
    """Group candidates by video while preserving incoming rank order."""

    grouped: defaultdict[str, list[CandidateFrame]] = defaultdict(list)
    for candidate in sorted(candidates, key=lambda item: item.rank):
        grouped[candidate.video_id].append(candidate)
    return dict(grouped)
