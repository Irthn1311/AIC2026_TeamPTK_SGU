"""Safe BTC CLIP NPY loading and explicit feature-row frame mapping."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from system_tai.common.schemas import FeatureRecord, FrameRecord


@dataclass(frozen=True, slots=True)
class ClipFeatureStats:
    row_count: int
    dimension: int
    dtype: str
    value_min: float
    value_max: float
    value_mean: float
    norm_min: float
    norm_max: float
    norm_mean: float
    contains_nan: bool
    contains_infinity: bool
    appears_l2_normalized: bool
    normalization_tolerance: float


class BTCClipFeatureStore:
    def __init__(self, *, normalization_tolerance: float = 1e-3) -> None:
        if normalization_tolerance < 0:
            raise ValueError("normalization_tolerance must be non-negative")
        self.normalization_tolerance = normalization_tolerance
        self._source_matrix: NDArray[np.number] | None = None
        self._working_matrix: NDArray[np.float32] | None = None
        self._row_to_frame: dict[int, FrameRecord] = {}
        self._feature_records: tuple[FeatureRecord, ...] = ()
        self._stats: ClipFeatureStats | None = None

    @property
    def source_matrix(self) -> NDArray[np.number]:
        if self._source_matrix is None:
            raise RuntimeError("BTCClipFeatureStore has not been loaded")
        return self._source_matrix

    @property
    def working_matrix(self) -> NDArray[np.float32]:
        if self._working_matrix is None:
            raise RuntimeError("BTCClipFeatureStore has not been loaded")
        return self._working_matrix

    @property
    def feature_records(self) -> tuple[FeatureRecord, ...]:
        return self._feature_records

    @property
    def stats(self) -> ClipFeatureStats:
        if self._stats is None:
            raise RuntimeError("BTCClipFeatureStore has not been loaded")
        return self._stats

    def load(
        self,
        npy_path: Path,
        frame_records: Sequence[FrameRecord],
        *,
        encoder_id: str,
        expected_dimension: int | None = None,
        normalize_working_copy: bool = False,
    ) -> ClipFeatureStats:
        npy_path = Path(npy_path)
        if not npy_path.is_file():
            raise FileNotFoundError(f"CLIP NPY not found: {npy_path}")
        if not encoder_id.strip():
            raise ValueError("encoder_id must not be empty")
        if expected_dimension is not None and expected_dimension <= 0:
            raise ValueError("expected_dimension must be positive when provided")

        matrix = np.load(npy_path, allow_pickle=False)
        if not isinstance(matrix, np.ndarray):
            raise ValueError("CLIP NPY must contain a NumPy array")
        if matrix.ndim != 2:
            raise ValueError(f"CLIP feature matrix must be two-dimensional, got {matrix.ndim}D")
        row_count, dimension = matrix.shape
        if row_count <= 0 or dimension <= 0:
            raise ValueError(f"CLIP feature matrix must be non-empty, got shape {matrix.shape}")
        if not np.issubdtype(matrix.dtype, np.number) or np.issubdtype(
            matrix.dtype, np.complexfloating
        ):
            raise ValueError(f"CLIP feature matrix must have a real numeric dtype: {matrix.dtype}")
        if expected_dimension is not None and dimension != expected_dimension:
            raise ValueError(
                f"CLIP dimension mismatch: observed={dimension}, expected={expected_dimension}"
            )

        contains_nan = bool(np.isnan(matrix).any())
        contains_infinity = bool(np.isinf(matrix).any())
        if contains_nan or contains_infinity:
            raise ValueError(
                "CLIP feature matrix contains "
                f"NaN={contains_nan}, Infinity={contains_infinity}"
            )
        if len(frame_records) != row_count:
            raise ValueError(
                f"CLIP/mapping row-count mismatch: features={row_count}, "
                f"mapping={len(frame_records)}"
            )

        row_to_frame: dict[int, FrameRecord] = {}
        for record in frame_records:
            if record.clip_row in row_to_frame:
                raise ValueError(f"duplicate mapping for clip_row {record.clip_row}")
            if not 0 <= record.clip_row < row_count:
                raise ValueError(
                    f"clip_row {record.clip_row} is outside feature rows [0, {row_count - 1}]"
                )
            row_to_frame[record.clip_row] = record
        missing_rows = sorted(set(range(row_count)) - set(row_to_frame))
        if missing_rows:
            raise ValueError(f"feature rows missing FrameRecord mappings: {missing_rows}")

        source_for_stats = matrix.astype(np.float64, copy=False)
        norms = np.linalg.norm(source_for_stats, axis=1)
        appears_normalized = bool(
            np.all(np.abs(norms - 1.0) <= self.normalization_tolerance)
        )
        working = matrix.astype(np.float32, copy=True)
        if normalize_working_copy:
            working_norms = np.linalg.norm(working, axis=1, keepdims=True)
            if bool(np.any(working_norms == 0)):
                raise ValueError("cannot normalize CLIP feature rows with zero norm")
            working /= working_norms

        stats = ClipFeatureStats(
            row_count=row_count,
            dimension=dimension,
            dtype=str(matrix.dtype),
            value_min=float(np.min(source_for_stats)),
            value_max=float(np.max(source_for_stats)),
            value_mean=float(np.mean(source_for_stats)),
            norm_min=float(np.min(norms)),
            norm_max=float(np.max(norms)),
            norm_mean=float(np.mean(norms)),
            contains_nan=contains_nan,
            contains_infinity=contains_infinity,
            appears_l2_normalized=appears_normalized,
            normalization_tolerance=self.normalization_tolerance,
        )
        feature_records = tuple(
            FeatureRecord(
                video_id=row_to_frame[row].video_id,
                actual_frame_id=row_to_frame[row].actual_frame_id,
                clip_row=row,
                encoder_id=encoder_id,
                dimension=dimension,
            )
            for row in range(row_count)
        )

        self._source_matrix = matrix
        self._working_matrix = working
        self._row_to_frame = row_to_frame
        self._feature_records = feature_records
        self._stats = stats
        return stats

    def frame_for_row(self, clip_row: int) -> FrameRecord:
        if self._stats is None:
            raise RuntimeError("BTCClipFeatureStore has not been loaded")
        try:
            return self._row_to_frame[clip_row]
        except KeyError as exc:
            raise KeyError(f"unknown clip_row: {clip_row}") from exc
