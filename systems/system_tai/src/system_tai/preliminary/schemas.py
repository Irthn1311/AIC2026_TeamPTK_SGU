from dataclasses import dataclass


def _require_text(val: str, name: str) -> None:
    if not isinstance(val, str) or not val.strip():
        raise ValueError(f"{name} must be a non-empty string")


@dataclass(frozen=True, slots=True)
class KISPrediction:
    query_id: str
    rank: int
    video_id: str
    frame_id: int

    def __post_init__(self) -> None:
        _require_text(self.query_id, "query_id")
        _require_text(self.video_id, "video_id")
        if self.rank < 1:
            raise ValueError("rank must be >= 1")
        if self.frame_id < 0:
            raise ValueError("frame_id must be >= 0")


@dataclass(frozen=True, slots=True)
class QAPrediction:
    query_id: str
    rank: int
    video_id: str
    frame_id: int
    answer: str

    def __post_init__(self) -> None:
        _require_text(self.query_id, "query_id")
        _require_text(self.video_id, "video_id")
        _require_text(self.answer, "answer")
        if self.rank < 1:
            raise ValueError("rank must be >= 1")
        if self.frame_id < 0:
            raise ValueError("frame_id must be >= 0")


@dataclass(frozen=True, slots=True)
class TRAKEPrediction:
    query_id: str
    rank: int
    video_id: str
    frame_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        _require_text(self.query_id, "query_id")
        _require_text(self.video_id, "video_id")
        if self.rank < 1:
            raise ValueError("rank must be >= 1")
        if not self.frame_ids or not isinstance(self.frame_ids, tuple):
            raise ValueError("frame_ids must be a non-empty tuple")
        for fid in self.frame_ids:
            if not isinstance(fid, int) or fid < 0:
                raise ValueError("all frame_ids must be non-negative integers")


@dataclass(frozen=True, slots=True)
class KISGroundTruth:
    query_id: str
    video_id: str
    start_frame_id: int
    end_frame_id: int

    def __post_init__(self) -> None:
        _require_text(self.query_id, "query_id")
        _require_text(self.video_id, "video_id")
        if self.start_frame_id < 0 or self.end_frame_id < 0:
            raise ValueError("frame IDs must be >= 0")
        if self.start_frame_id > self.end_frame_id:
            raise ValueError("start_frame_id must be <= end_frame_id")


@dataclass(frozen=True, slots=True)
class QAGroundTruth:
    query_id: str
    video_id: str
    start_frame_id: int
    end_frame_id: int
    accepted_answers: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_text(self.query_id, "query_id")
        _require_text(self.video_id, "video_id")
        if self.start_frame_id < 0 or self.end_frame_id < 0:
            raise ValueError("frame IDs must be >= 0")
        if self.start_frame_id > self.end_frame_id:
            raise ValueError("start_frame_id must be <= end_frame_id")
        if not self.accepted_answers or not isinstance(self.accepted_answers, tuple):
            raise ValueError("accepted_answers must be a non-empty tuple")
        for ans in self.accepted_answers:
            _require_text(ans, "accepted_answer")


@dataclass(frozen=True, slots=True)
class TRAKEGroundTruth:
    query_id: str
    video_id: str
    event_intervals: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        _require_text(self.query_id, "query_id")
        _require_text(self.video_id, "video_id")
        if not self.event_intervals or not isinstance(self.event_intervals, tuple):
            raise ValueError("event_intervals must be a non-empty tuple")
        for interval in self.event_intervals:
            if len(interval) != 2:
                raise ValueError("Each interval must have length 2")
            if interval[0] < 0 or interval[1] < 0:
                raise ValueError("Interval frame IDs must be >= 0")
            if interval[0] > interval[1]:
                raise ValueError("start_frame_id must be <= end_frame_id")
