from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FrameMapper:
    """Converts between internal zero-based frame indexes and BTC frame ids."""

    btc_convention: str
    fps: float

    def __post_init__(self) -> None:
        if self.btc_convention not in {"0-based", "1-based"}:
            raise ValueError("btc_convention must be '0-based' or '1-based'")
        if self.fps <= 0:
            raise ValueError("fps must be positive")

    def internal_to_btc_frame_id(self, internal_frame_index: int) -> int:
        if internal_frame_index < 0:
            raise ValueError("internal_frame_index must be non-negative")
        if self.btc_convention == "1-based":
            return int(internal_frame_index) + 1
        return int(internal_frame_index)

    def btc_to_internal_frame_id(self, btc_frame_id: int) -> int:
        if self.btc_convention == "1-based":
            internal = int(btc_frame_id) - 1
        else:
            internal = int(btc_frame_id)
        if internal < 0:
            raise ValueError("converted internal frame id is negative")
        return internal

    def frame_to_timestamp(self, internal_frame_index: int) -> float:
        if internal_frame_index < 0:
            raise ValueError("internal_frame_index must be non-negative")
        return float(internal_frame_index) / float(self.fps)
