"""Simple frame-to-video ranking baselines."""

from triage_eg.common.schemas import CandidateFrame, CandidateVideo
from triage_eg.retrieval.grouping import group_candidates_by_video


def rank_videos(
    candidates: list[CandidateFrame], *, strategy: str = "best_frame", top_k: int = 3
) -> list[CandidateVideo]:
    """Rank videos using best-frame score or the mean of their top-k frames."""

    if strategy not in {"best_frame", "top_k_mean"}:
        raise ValueError("strategy must be 'best_frame' or 'top_k_mean'")
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")
    videos: list[CandidateVideo] = []
    for video_id, group in group_candidates_by_video(candidates).items():
        best = tuple(sorted(group, key=lambda item: -item.score)[:top_k])
        score = (
            best[0].score if strategy == "best_frame" else sum(x.score for x in best) / len(best)
        )
        videos.append(
            CandidateVideo(
                video_id=video_id,
                score=score,
                best_frames=best,
                matched_event_count=len(best),
                source_branches=tuple(sorted({item.source_branch for item in group})),
            )
        )
    return sorted(videos, key=lambda item: (-item.score, item.video_id))
