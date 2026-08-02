"""Text-to-frame retrieval orchestration."""

from triage_eg.common.schemas import CandidateFrame, FrameRecord
from triage_eg.features.interfaces import MultimodalEncoder
from triage_eg.retrieval.interfaces import VectorIndex


class RetrievalEngine:
    """Map vector-index results back to stable frame contracts."""

    def __init__(
        self,
        encoder: MultimodalEncoder,
        index: VectorIndex,
        frames: list[FrameRecord],
        source_branch: str = "numpy_baseline",
    ) -> None:
        self._encoder = encoder
        self._index = index
        self._frames = {frame.frame_uid: frame for frame in frames}
        self._source_branch = source_branch

    def search(self, text_query: str, top_k: int) -> list[CandidateFrame]:
        """Encode one query and return globally ranked candidate frames."""

        if not text_query.strip():
            raise ValueError("text_query must not be empty")
        scores, ids = self._index.search(self._encoder.encode_text([text_query]), top_k)
        candidates: list[CandidateFrame] = []
        for rank, (score, frame_uid) in enumerate(zip(scores[0], ids[0], strict=True), start=1):
            uid = str(frame_uid)
            if uid not in self._frames:
                raise KeyError(f"Index returned unknown frame UID: {uid}")
            frame = self._frames[uid]
            candidates.append(
                CandidateFrame(
                    video_id=frame.video_id,
                    frame_id=frame.actual_frame_id,
                    timestamp_ms=frame.timestamp_ms,
                    frame_uid=frame.frame_uid,
                    score=float(score),
                    rank=rank,
                    source_branch=self._source_branch,
                )
            )
        return candidates
