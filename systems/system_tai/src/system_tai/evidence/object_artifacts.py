"""BTC keyframe object-artifact loading with original-frame coordinates."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from system_tai.common.schemas import FrameMappingRecord

BTC_OBJECT_ARTIFACT_SCHEMA = "btc_faster_rcnn_openimages_v4_keyframe_json_v1"
_REQUIRED_FIELDS = (
    "detection_scores",
    "detection_boxes",
    "detection_class_entities",
    "detection_class_labels",
)


class ObjectArtifactError(ValueError):
    """Object artifact is missing, ambiguous, or malformed."""


@dataclass(frozen=True, slots=True)
class ObjectDetection:
    label: str
    confidence: float
    bbox: tuple[float, float, float, float]
    class_id: int
    source_artifact: str
    source_keyframe_order: int
    source_detection_index: int


@dataclass(frozen=True, slots=True)
class ObjectFrameEvidence:
    video_id: str
    requested_frame_id: int
    object_source_frame_id: int
    frame_distance: int
    lookup_kind: str
    detections: tuple[ObjectDetection, ...]


def resolve_object_artifact_root(dataset_root: Path) -> Path:
    """Resolve the one bounded BTC object family below a known dataset root."""

    root = Path(dataset_root)
    if not root.is_dir():
        raise FileNotFoundError(f"dataset root not found: {root}")
    candidates: set[Path] = set()
    direct = root / "objects-aic25-b1" / "objects"
    if direct.is_dir():
        candidates.add(direct.resolve(strict=False))
    for child in root.iterdir():
        if child.is_dir() and child.name.casefold().startswith("objects-"):
            nested = child / "objects"
            if nested.is_dir():
                candidates.add(nested.resolve(strict=False))
    ordered = sorted(candidates, key=lambda path: path.as_posix().casefold())
    if not ordered:
        raise FileNotFoundError(
            "BTC object artifact root not found under bounded dataset root; expected "
            "objects-aic25-b1/objects"
        )
    if len(ordered) != 1:
        raise ObjectArtifactError(
            "ambiguous BTC object artifact roots: "
            + ", ".join(path.as_posix() for path in ordered)
        )
    return ordered[0]


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ObjectArtifactError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _finite_float(value: Any, *, field: str, index: int) -> float:
    if type(value) is bool:
        raise ObjectArtifactError(f"{field}[{index}] must be numeric")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ObjectArtifactError(f"{field}[{index}] must be numeric") from exc
    if not math.isfinite(parsed):
        raise ObjectArtifactError(f"{field}[{index}] must be finite")
    return parsed


def _integer(value: Any, *, field: str, index: int) -> int:
    parsed = _finite_float(value, field=field, index=index)
    if not parsed.is_integer():
        raise ObjectArtifactError(f"{field}[{index}] must be integer-valued")
    return int(parsed)


class ObjectArtifactIndex:
    """Lazy per-video index from keyframe ordinal JSON to BTC ``frame_idx``."""

    def __init__(
        self,
        *,
        object_root: Path,
        mappings_by_video: Mapping[str, Sequence[FrameMappingRecord]],
        source_root_identity: str = "objects-aic25-b1/objects",
    ) -> None:
        root = Path(object_root)
        if not root.is_dir():
            raise FileNotFoundError(f"object artifact root not found: {root}")
        if not mappings_by_video:
            raise ValueError("mappings_by_video must not be empty")
        normalized: dict[str, tuple[FrameMappingRecord, ...]] = {}
        for video_id, records in mappings_by_video.items():
            if not video_id.strip() or not records:
                raise ValueError("every mapped video requires a non-empty ID and records")
            orders: set[int] = set()
            for record in records:
                if record.keyframe_order in orders:
                    raise ObjectArtifactError(
                        f"duplicate keyframe order for {video_id}: {record.keyframe_order}"
                    )
                orders.add(record.keyframe_order)
            normalized[video_id] = tuple(records)
        self.object_root = root.resolve(strict=False)
        self.source_root_identity = source_root_identity
        self._mappings = MappingProxyType(normalized)
        self._video_files: dict[str, Mapping[int, Path]] = {}
        self._frame_to_orders: dict[str, Mapping[int, tuple[int, ...]]] = {}
        self._loaded_artifacts: dict[tuple[str, int], tuple[ObjectDetection, ...]] = {}

    @property
    def schema_identity(self) -> str:
        return BTC_OBJECT_ARTIFACT_SCHEMA

    def contains_video(self, video_id: str) -> bool:
        return video_id in self._mappings

    def lookup(self, video_id: str, frame_id: int) -> ObjectFrameEvidence | None:
        if type(frame_id) is not int or frame_id < 0:
            raise ValueError("frame_id must be a non-negative integer")
        if video_id not in self._mappings:
            return None
        self._ensure_video_index(video_id)
        orders = self._frame_to_orders[video_id].get(frame_id, ())
        detections: list[ObjectDetection] = []
        for order in orders:
            artifact_path = self._video_files[video_id].get(order)
            if artifact_path is not None:
                detections.extend(self._load_artifact(video_id, order, artifact_path))
        ordered = tuple(
            sorted(
                detections,
                key=lambda item: (
                    -item.confidence,
                    item.label.casefold(),
                    item.bbox,
                    item.source_keyframe_order,
                    item.source_detection_index,
                ),
            )
        )
        if not ordered:
            return None
        return ObjectFrameEvidence(
            video_id=video_id,
            requested_frame_id=frame_id,
            object_source_frame_id=frame_id,
            frame_distance=0,
            lookup_kind="EXACT_KEYFRAME",
            detections=ordered,
        )

    def _ensure_video_index(self, video_id: str) -> None:
        if video_id in self._video_files:
            return
        video_dir = self.object_root / video_id
        files: dict[int, Path] = {}
        if video_dir.is_dir():
            for path in sorted(video_dir.iterdir(), key=lambda item: item.name.casefold()):
                if not path.is_file() or path.suffix.casefold() != ".json":
                    continue
                try:
                    order = int(path.stem)
                except ValueError:
                    continue
                if order < 0:
                    continue
                if order in files:
                    raise ObjectArtifactError(
                        f"ambiguous object artifacts for {video_id} keyframe {order}: "
                        f"{files[order].name}, {path.name}"
                    )
                files[order] = path
        frame_to_orders: dict[int, list[int]] = {}
        for record in self._mappings[video_id]:
            frame_to_orders.setdefault(record.frame_id, []).append(record.keyframe_order)
        self._video_files[video_id] = MappingProxyType(files)
        self._frame_to_orders[video_id] = MappingProxyType(
            {
                frame_id: tuple(sorted(orders))
                for frame_id, orders in frame_to_orders.items()
            }
        )

    def _load_artifact(
        self, video_id: str, order: int, path: Path
    ) -> tuple[ObjectDetection, ...]:
        cache_key = (video_id, order)
        cached = self._loaded_artifacts.get(cache_key)
        if cached is not None:
            return cached
        raw = path.read_bytes()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ObjectArtifactError(f"object JSON is not UTF-8: {path}") from exc
        if text.startswith("\ufeff"):
            raise ObjectArtifactError(f"object JSON must not contain a BOM: {path}")
        try:
            payload = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
        except json.JSONDecodeError as exc:
            raise ObjectArtifactError(f"invalid object JSON {path}: {exc}") from exc
        if type(payload) is not dict:
            raise ObjectArtifactError(f"object JSON must contain an object: {path}")
        missing = [field for field in _REQUIRED_FIELDS if field not in payload]
        if missing:
            raise ObjectArtifactError(
                f"object JSON missing required fields {missing}: {path}"
            )
        arrays = [payload[field] for field in _REQUIRED_FIELDS]
        if any(type(value) is not list for value in arrays):
            raise ObjectArtifactError(f"object JSON detection fields must be arrays: {path}")
        lengths = {len(value) for value in arrays}
        if len(lengths) != 1:
            raise ObjectArtifactError(f"object JSON detection arrays differ in length: {path}")

        source_artifact = f"{video_id}/{path.name}"
        detections: list[ObjectDetection] = []
        scores = payload["detection_scores"]
        boxes = payload["detection_boxes"]
        entities = payload["detection_class_entities"]
        class_labels = payload["detection_class_labels"]
        for index, (score_value, box_value, label_value, class_value) in enumerate(
            zip(scores, boxes, entities, class_labels)
        ):
            if not isinstance(label_value, str) or not label_value.strip():
                raise ObjectArtifactError(
                    f"detection_class_entities[{index}] must be non-empty text"
                )
            score = _finite_float(score_value, field="detection_scores", index=index)
            if not 0.0 <= score <= 1.0:
                raise ObjectArtifactError(f"detection_scores[{index}] must be in [0, 1]")
            if type(box_value) is not list or len(box_value) != 4:
                raise ObjectArtifactError(
                    f"detection_boxes[{index}] must have four coordinates"
                )
            bbox = tuple(
                _finite_float(value, field=f"detection_boxes[{index}]", index=coord)
                for coord, value in enumerate(box_value)
            )
            if any(value < 0.0 or value > 1.0 for value in bbox):
                raise ObjectArtifactError(
                    f"detection_boxes[{index}] coordinates must be normalized to [0, 1]"
                )
            if bbox[0] > bbox[2] or bbox[1] > bbox[3]:
                raise ObjectArtifactError(f"detection_boxes[{index}] has invalid bounds")
            detections.append(
                ObjectDetection(
                    label=" ".join(label_value.split()),
                    confidence=score,
                    bbox=(bbox[0], bbox[1], bbox[2], bbox[3]),
                    class_id=_integer(
                        class_value,
                        field="detection_class_labels",
                        index=index,
                    ),
                    source_artifact=source_artifact,
                    source_keyframe_order=order,
                    source_detection_index=index,
                )
            )
        result = tuple(detections)
        self._loaded_artifacts[cache_key] = result
        return result
