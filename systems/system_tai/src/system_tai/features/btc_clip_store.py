"""Safe BTC CLIP NPY loading and explicit feature-row frame mapping."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from system_tai.common.schemas import (
    FeatureRecord,
    FrameMappingRecord,
    FrameRecord,
    VideoFeatureStore,
)


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
                f"CLIP feature matrix contains NaN={contains_nan}, Infinity={contains_infinity}"
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
        appears_normalized = bool(np.all(np.abs(norms - 1.0) <= self.normalization_tolerance))
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


@dataclass(frozen=True, slots=True)
class LoadedVideoFeatureStore:
    descriptor: VideoFeatureStore
    matrix: NDArray[np.number]
    mappings: tuple[FrameMappingRecord, ...]

    def frame_for_row(self, clip_row: int) -> FrameMappingRecord:
        if not 0 <= clip_row < len(self.mappings):
            raise KeyError(f"unknown clip_row for {self.descriptor.video_id}: {clip_row}")
        return self.mappings[clip_row]

    def contains_frame(self, frame_id: int) -> bool:
        return any(record.frame_id == frame_id for record in self.mappings)


class VideoFeatureStoreLoader:
    REQUIRED_MAPPING_COLUMNS = {"n", "pts_time", "fps", "frame_idx"}

    def __init__(
        self,
        *,
        expected_dimension: int = 512,
        memory_map: bool = True,
        validation_chunk_size: int = 8192,
        normalization_tolerance: float = 1e-3,
    ) -> None:
        if expected_dimension <= 0:
            raise ValueError("expected_dimension must be positive")
        if validation_chunk_size <= 0:
            raise ValueError("validation_chunk_size must be positive")
        if normalization_tolerance < 0:
            raise ValueError("normalization_tolerance must be non-negative")
        self.expected_dimension = expected_dimension
        self.memory_map = memory_map
        self.validation_chunk_size = validation_chunk_size
        self.normalization_tolerance = normalization_tolerance

    def load(
        self, *, video_id: str, mapping_csv_path: Path, clip_npy_path: Path
    ) -> LoadedVideoFeatureStore:
        if not video_id.strip():
            raise ValueError("video_id must not be empty")
        mapping_path = Path(mapping_csv_path)
        npy_path = Path(clip_npy_path)
        mappings = self._load_mapping(mapping_path)
        if not npy_path.is_file():
            raise FileNotFoundError(f"CLIP NPY not found: {npy_path}")
        matrix = np.load(
            npy_path,
            allow_pickle=False,
            mmap_mode="r" if self.memory_map else None,
        )
        if not isinstance(matrix, np.ndarray) or matrix.ndim != 2:
            raise ValueError("CLIP feature matrix must be a two-dimensional NumPy array")
        if not np.issubdtype(matrix.dtype, np.number) or np.issubdtype(
            matrix.dtype, np.complexfloating
        ):
            raise ValueError(f"CLIP feature matrix must have a real numeric dtype: {matrix.dtype}")
        row_count, dimension = matrix.shape
        if dimension != self.expected_dimension:
            raise ValueError(
                f"CLIP dimension mismatch: observed={dimension}, expected={self.expected_dimension}"
            )
        if row_count != len(mappings):
            raise ValueError(
                f"mapping/NPY row-count mismatch: mapping={len(mappings)}, features={row_count}"
            )
        normalized = self._validate_matrix(matrix)
        matrix.setflags(write=False)
        descriptor = VideoFeatureStore(
            video_id=video_id,
            mapping_csv_path=mapping_path.resolve(strict=False),
            clip_npy_path=npy_path.resolve(strict=False),
            row_count=row_count,
            embedding_dimension=dimension,
            normalized=normalized,
        )
        return LoadedVideoFeatureStore(
            descriptor=descriptor,
            matrix=matrix,
            mappings=mappings,
        )

    def _load_mapping(self, path: Path) -> tuple[FrameMappingRecord, ...]:
        if not path.is_file():
            raise FileNotFoundError(f"mapping CSV not found: {path}")
        records: list[FrameMappingRecord] = []
        seen_frames: set[int] = set()
        previous_order: int | None = None
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            missing = sorted(self.REQUIRED_MAPPING_COLUMNS - set(reader.fieldnames or ()))
            if missing:
                raise ValueError(f"mapping CSV missing columns: {', '.join(missing)}")
            for line_number, row in enumerate(reader, start=2):
                if not any((value or "").strip() for value in row.values()):
                    continue
                try:
                    keyframe_order = int((row.get("n") or "").strip())
                    frame_id = int((row.get("frame_idx") or "").strip())
                    pts_time = float((row.get("pts_time") or "").strip())
                    fps = float((row.get("fps") or "").strip())
                except ValueError as exc:
                    raise ValueError(f"invalid mapping value at line {line_number}") from exc
                if previous_order is not None and keyframe_order <= previous_order:
                    raise ValueError(
                        "mapping keyframe order must be strictly increasing: "
                        f"line={line_number}, previous={previous_order}, current={keyframe_order}"
                    )
                if frame_id in seen_frames:
                    raise ValueError(
                        f"ambiguous duplicate frame_idx at line {line_number}: {frame_id}"
                    )
                if not math.isfinite(pts_time) or not math.isfinite(fps):
                    raise ValueError(f"non-finite mapping value at line {line_number}")
                record = FrameMappingRecord(
                    clip_row=len(records),
                    keyframe_order=keyframe_order,
                    frame_id=frame_id,
                    pts_time=pts_time,
                    fps=fps,
                )
                records.append(record)
                seen_frames.add(frame_id)
                previous_order = keyframe_order
        if not records:
            raise ValueError("mapping CSV contains no records")
        return tuple(records)

    def _validate_matrix(self, matrix: NDArray[np.number]) -> bool:
        normalized = True
        for start in range(0, matrix.shape[0], self.validation_chunk_size):
            chunk = np.asarray(matrix[start : start + self.validation_chunk_size], dtype=np.float32)
            if not np.isfinite(chunk).all():
                raise ValueError("CLIP feature matrix contains NaN or Infinity")
            norms = np.linalg.norm(chunk, axis=1)
            if np.any(norms == 0):
                raise ValueError("CLIP feature matrix contains a zero-norm row")
            if np.any(np.abs(norms - 1.0) > self.normalization_tolerance):
                normalized = False
        return normalized


class FeatureStoreRegistry:
    def __init__(self, stores: Sequence[LoadedVideoFeatureStore]) -> None:
        if not stores:
            raise ValueError("feature registry requires at least one video")
        by_video: dict[str, LoadedVideoFeatureStore] = {}
        dimensions: set[int] = set()
        for store in stores:
            video_id = store.descriptor.video_id
            if video_id in by_video:
                raise ValueError(f"duplicate video_id in feature registry: {video_id}")
            by_video[video_id] = store
            dimensions.add(store.descriptor.embedding_dimension)
        if len(dimensions) != 1:
            raise ValueError(f"inconsistent feature dimensions: {sorted(dimensions)}")
        self._stores = tuple(sorted(stores, key=lambda item: item.descriptor.video_id))
        self._by_video = by_video
        self.embedding_dimension = dimensions.pop()

    @classmethod
    def from_manifest(
        cls,
        manifest_path: Path,
        *,
        expected_dimension: int = 512,
        memory_map: bool = True,
        validation_chunk_size: int = 8192,
    ) -> FeatureStoreRegistry:
        path = Path(manifest_path)
        if not path.is_file():
            raise FileNotFoundError(f"feature manifest not found: {path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid feature manifest JSON: {exc}") from exc
        videos = payload.get("videos") if isinstance(payload, dict) else None
        if not isinstance(videos, list) or not videos:
            raise ValueError("feature manifest must contain a non-empty videos list")
        loader = VideoFeatureStoreLoader(
            expected_dimension=expected_dimension,
            memory_map=memory_map,
            validation_chunk_size=validation_chunk_size,
        )
        stores: list[LoadedVideoFeatureStore] = []
        for index, item in enumerate(videos):
            if not isinstance(item, dict):
                raise ValueError(f"manifest video entry {index} must be an object")
            try:
                video_id = str(item["video_id"])
                mapping_value = item.get("mapping_csv_path", item.get("mapping_csv"))
                npy_value = item.get("clip_npy_path", item.get("clip_npy"))
                if mapping_value is None or npy_value is None:
                    raise KeyError("mapping_csv_path/clip_npy_path")
            except KeyError as exc:
                raise ValueError(f"manifest video entry {index} is missing fields") from exc
            mapping_path = cls._resolve_manifest_path(path.parent, mapping_value)
            npy_path = cls._resolve_manifest_path(path.parent, npy_value)
            stores.append(
                loader.load(
                    video_id=video_id,
                    mapping_csv_path=mapping_path,
                    clip_npy_path=npy_path,
                )
            )
        return cls(stores)

    @staticmethod
    def _resolve_manifest_path(base: Path, value: Any) -> Path:
        candidate = Path(str(value))
        return candidate if candidate.is_absolute() else base / candidate

    @property
    def stores(self) -> tuple[LoadedVideoFeatureStore, ...]:
        return self._stores

    @property
    def total_rows(self) -> int:
        return sum(store.descriptor.row_count for store in self._stores)

    def get(self, video_id: str) -> LoadedVideoFeatureStore:
        try:
            return self._by_video[video_id]
        except KeyError as exc:
            raise KeyError(f"unknown video_id in feature registry: {video_id}") from exc

    def contains(self, video_id: str, frame_id: int) -> bool:
        store = self._by_video.get(video_id)
        return store is not None and store.contains_frame(frame_id)
