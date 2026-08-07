from dataclasses import dataclass
from typing import Any


def _require_text(val: str, name: str) -> None:
    if not isinstance(val, str) or not val.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _require_positive_int(val: Any, name: str) -> None:
    if type(val) is not int:
        raise TypeError(f"{name} must be an integer, got {type(val).__name__}")
    if val < 1:
        raise ValueError(f"{name} must be >= 1, got {val}")


def _require_nonnegative_int(val: Any, name: str) -> None:
    if type(val) is not int:
        raise TypeError(f"{name} must be an integer, got {type(val).__name__}")
    if val < 0:
        raise ValueError(f"{name} must be >= 0, got {val}")


@dataclass(frozen=True, slots=True)
class KISPrediction:
    query_id: str
    rank: int
    video_id: str
    frame_id: int

    def __post_init__(self) -> None:
        _require_text(self.query_id, "query_id")
        _require_text(self.video_id, "video_id")
        _require_positive_int(self.rank, "rank")
        _require_nonnegative_int(self.frame_id, "frame_id")


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
        _require_positive_int(self.rank, "rank")
        _require_nonnegative_int(self.frame_id, "frame_id")


@dataclass(frozen=True, slots=True)
class TRAKEPrediction:
    query_id: str
    rank: int
    video_id: str
    frame_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        _require_text(self.query_id, "query_id")
        _require_text(self.video_id, "video_id")
        _require_positive_int(self.rank, "rank")
        if not isinstance(self.frame_ids, tuple) or len(self.frame_ids) == 0:
            raise ValueError("frame_ids must be a non-empty tuple")
        for fid in self.frame_ids:
            _require_nonnegative_int(fid, "frame_id")


@dataclass(frozen=True, slots=True)
class KISGroundTruth:
    query_id: str
    video_id: str
    start_frame_id: int
    end_frame_id: int

    def __post_init__(self) -> None:
        _require_text(self.query_id, "query_id")
        _require_text(self.video_id, "video_id")
        _require_nonnegative_int(self.start_frame_id, "start_frame_id")
        _require_nonnegative_int(self.end_frame_id, "end_frame_id")
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
        _require_nonnegative_int(self.start_frame_id, "start_frame_id")
        _require_nonnegative_int(self.end_frame_id, "end_frame_id")
        if self.start_frame_id > self.end_frame_id:
            raise ValueError("start_frame_id must be <= end_frame_id")
        if not isinstance(self.accepted_answers, tuple) or len(self.accepted_answers) == 0:
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
        if not isinstance(self.event_intervals, tuple) or len(self.event_intervals) == 0:
            raise ValueError("event_intervals must be a non-empty tuple")
        for interval in self.event_intervals:
            if not isinstance(interval, tuple) or len(interval) != 2:
                raise ValueError("Each interval must be a 2-element tuple")
            _require_nonnegative_int(interval[0], "start_frame_id")
            _require_nonnegative_int(interval[1], "end_frame_id")
            if interval[0] > interval[1]:
                raise ValueError("start_frame_id must be <= end_frame_id")
