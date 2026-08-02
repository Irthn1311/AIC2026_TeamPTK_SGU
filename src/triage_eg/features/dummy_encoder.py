"""Deterministic feature generator for contract and integration testing."""

from __future__ import annotations

from hashlib import sha256

import numpy as np

from triage_eg.common.schemas import FrameRecord


class DeterministicDummyEncoder:
    """Generate stable normalized vectors from identifiers.

    This is not a semantic model. It only demonstrates the contract between
    feature extraction and retrieval, without images, weights, a GPU, or network access.
    """

    def __init__(self, dimension: int = 32, model_version: str = "0.1") -> None:
        if dimension <= 0:
            raise ValueError("dimension must be greater than zero")
        self._dimension = dimension
        self._model_version = model_version

    @property
    def dimension(self) -> int:
        """Return configured vector dimension."""

        return self._dimension

    @property
    def model_name(self) -> str:
        """Return the dummy model name."""

        return "deterministic_dummy"

    @property
    def model_version(self) -> str:
        """Return the dummy implementation version."""

        return self._model_version

    def _encode(self, values: list[str], namespace: str) -> np.ndarray:
        vectors = np.empty((len(values), self.dimension), dtype=np.float32)
        for row, value in enumerate(values):
            seed_material = f"{namespace}|{value}|{self.model_version}".encode()
            chunks = bytearray()
            counter = 0
            while len(chunks) < self.dimension * 4:
                chunks.extend(sha256(seed_material + counter.to_bytes(4, "big")).digest())
                counter += 1
            integers = np.frombuffer(bytes(chunks[: self.dimension * 4]), dtype=np.uint32)
            vector = integers.astype(np.float64) / np.iinfo(np.uint32).max
            vector = vector * 2.0 - 1.0
            norm = np.linalg.norm(vector)
            vectors[row] = (vector / norm).astype(np.float32)
        return vectors

    def encode_frames(self, frames: list[FrameRecord]) -> np.ndarray:
        """Encode stable frame UIDs; image files are intentionally not read."""

        return self._encode([frame.frame_uid for frame in frames], "frame")

    def encode_text(self, texts: list[str]) -> np.ndarray:
        """Encode stable text strings; values have no semantic meaning."""

        return self._encode(texts, "text")
