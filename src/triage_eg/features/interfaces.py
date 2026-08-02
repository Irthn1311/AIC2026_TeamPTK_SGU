"""Replaceable multimodal encoder interface."""

from typing import Protocol

import numpy as np

from triage_eg.common.schemas import FrameRecord


class MultimodalEncoder(Protocol):
    """Contract for aligned frame and text encoders."""

    @property
    def dimension(self) -> int:
        """Return output vector dimension."""
        ...

    @property
    def model_name(self) -> str:
        """Return stable model family name."""
        ...

    @property
    def model_version(self) -> str:
        """Return exact model or weight version."""
        ...

    def encode_frames(self, frames: list[FrameRecord]) -> np.ndarray:
        """Encode frame records as a two-dimensional float array."""
        ...

    def encode_text(self, texts: list[str]) -> np.ndarray:
        """Encode query texts in the same vector space."""
        ...
