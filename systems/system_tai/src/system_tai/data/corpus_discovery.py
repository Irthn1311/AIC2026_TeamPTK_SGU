"""Bounded discovery and deterministic manifests for BTC KIS artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

VIDEO_ID_PATTERN = re.compile(r"^[A-Za-z]\d{2}_V\d{3}$")
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".webm"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MANIFEST_SCHEMA_VERSION = 1
DISCOVERY_VERSION = "system_tai_btc_corpus_v1"


class CorpusDiscoveryError(RuntimeError):
    def __init__(self, message: str, *, issues: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.issues = issues


@dataclass(frozen=True, slots=True)
class DiscoveredVideo:
    video_id: str
    mapping_csv_path: Path
    clip_npy_path: Path
    keyframe_directory: Path
    raw_video_path: Path | None
    row_count: int
    embedding_dimension: int
    mapping_size_bytes: int
    clip_size_bytes: int
    keyframe_image_count: int


@dataclass(frozen=True, slots=True)
class CorpusManifest:
    input_root: Path
    dataset_root: Path
    fingerprint: str
    videos: tuple[DiscoveredVideo, ...]
    schema_version: int = MANIFEST_SCHEMA_VERSION
    discovery_version: str = DISCOVERY_VERSION

    @property
    def total_rows(self) -> int:
        return sum(video.row_count for video in self.videos)

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "discovery_version": self.discovery_version,
            "manifest_fingerprint": self.fingerprint,
            "input_root": str(self.input_root),
            "dataset_root": str(self.dataset_root),
            "video_count": len(self.videos),
            "feature_row_count": self.total_rows,
            "videos": [_video_payload(video) for video in self.videos],
        }

    def write(self, path: Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_payload(), ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        return destination


def _video_payload(video: DiscoveredVideo) -> dict[str, Any]:
    return {
        "video_id": video.video_id,
        "mapping_csv_path": str(video.mapping_csv_path),
        "clip_npy_path": str(video.clip_npy_path),
        "keyframe_directory": str(video.keyframe_directory),
        "raw_video_path": str(video.raw_video_path) if video.raw_video_path else None,
        "row_count": video.row_count,
        "embedding_dimension": video.embedding_dimension,
        "mapping_size_bytes": video.mapping_size_bytes,
        "clip_size_bytes": video.clip_size_bytes,
        "keyframe_image_count": video.keyframe_image_count,
    }


def _fingerprint(videos: tuple[DiscoveredVideo, ...]) -> str:
    canonical = json.dumps(
        [_video_payload(video) for video in videos],
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _bounded_directories(root: Path, *, max_depth: int) -> tuple[Path, ...]:
    found: list[Path] = []
    queue: list[tuple[Path, int]] = [(root, 0)]
    while queue:
        current, depth = queue.pop(0)
        found.append(current)
        if depth >= max_depth:
            continue
        try:
            children = sorted(
                (path for path in current.iterdir() if path.is_dir()),
                key=lambda path: str(path).casefold(),
            )
        except OSError:
            continue
        queue.extend((child, depth + 1) for child in children)
    return tuple(found)


def _family_kind(path: Path) -> str | None:
    name = path.name.casefold()
    if "map" in name and "keyframe" in name:
        return "mapping"
    if "clip" in name and ("feature" in name or "vit" in name):
        return "clip"
    if "keyframe" in name:
        return "keyframes"
    if name.startswith("video") or name.startswith("videos"):
        return "videos"
    return None


def resolve_dataset_root(input_root: Path, *, max_depth: int = 4) -> Path:
    root = Path(input_root)
    if not root.is_dir():
        raise CorpusDiscoveryError(f"input root is not a directory: {root}")
    candidates: list[Path] = []
    for directory in _bounded_directories(root, max_depth=max_depth):
        try:
            direct_kinds = {
                kind
                for child in directory.iterdir()
                if child.is_dir() and (kind := _family_kind(child)) is not None
            }
        except OSError:
            continue
        if {"mapping", "clip", "keyframes"}.issubset(direct_kinds):
            candidates.append(directory.resolve(strict=False))
    unique = tuple(sorted(set(candidates), key=lambda path: str(path).casefold()))
    if not unique:
        raise CorpusDiscoveryError("no dataset root contains mapping, CLIP, and keyframe families")
    if len(unique) > 1:
        raise CorpusDiscoveryError(
            "multiple dataset roots contain complete artifact families",
            issues=tuple(str(path) for path in unique),
        )
    return unique[0]


def _family_roots(dataset_root: Path) -> dict[str, tuple[Path, ...]]:
    families: dict[str, list[Path]] = {
        "mapping": [],
        "clip": [],
        "keyframes": [],
        "videos": [],
    }
    for child in dataset_root.iterdir():
        if child.is_dir() and (kind := _family_kind(child)) is not None:
            families[kind].append(child)
    return {
        kind: tuple(sorted(paths, key=lambda path: str(path).casefold()))
        for kind, paths in families.items()
    }


def _file_index(roots: tuple[Path, ...], suffixes: set[str]) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for root in roots:
        for path in root.rglob("*"):
            if (
                path.is_file()
                and path.suffix.casefold() in suffixes
                and VIDEO_ID_PATTERN.fullmatch(path.stem)
            ):
                index.setdefault(path.stem, []).append(path.resolve(strict=False))
    for paths in index.values():
        paths.sort(key=lambda path: str(path).casefold())
    return index


def _keyframe_index(roots: tuple[Path, ...]) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for root in roots:
        for path in root.rglob("*"):
            if path.is_dir() and VIDEO_ID_PATTERN.fullmatch(path.name):
                index.setdefault(path.name, []).append(path.resolve(strict=False))
    for paths in index.values():
        paths.sort(key=lambda path: str(path).casefold())
    return index


def _mapping_row_count(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"n", "pts_time", "fps", "frame_idx"}
        missing = sorted(required - set(reader.fieldnames or ()))
        if missing:
            raise CorpusDiscoveryError(
                f"mapping CSV missing columns for {path.name}: {', '.join(missing)}"
            )
        return sum(1 for row in reader if any((value or "").strip() for value in row.values()))


def _npy_shape(path: Path) -> tuple[int, int]:
    matrix = np.load(path, allow_pickle=False, mmap_mode="r")
    if not isinstance(matrix, np.ndarray) or matrix.ndim != 2:
        raise CorpusDiscoveryError(f"CLIP NPY must be a 2D array: {path}")
    return int(matrix.shape[0]), int(matrix.shape[1])


def discover_corpus(
    input_root: Path,
    *,
    expected_dimension: int | None = 512,
    max_root_depth: int = 4,
) -> CorpusManifest:
    dataset_root = resolve_dataset_root(input_root, max_depth=max_root_depth)
    families = _family_roots(dataset_root)
    mappings = _file_index(families["mapping"], {".csv"})
    clips = _file_index(families["clip"], {".npy"})
    keyframes = _keyframe_index(families["keyframes"])
    raw_videos = _file_index(families["videos"], VIDEO_EXTENSIONS)
    video_ids = sorted(set(mappings) | set(clips) | set(keyframes), key=str.casefold)
    issues: list[str] = []
    videos: list[DiscoveredVideo] = []
    for video_id in video_ids:
        required = {
            "mapping CSV": mappings.get(video_id, []),
            "CLIP NPY": clips.get(video_id, []),
            "keyframe directory": keyframes.get(video_id, []),
        }
        invalid = False
        for label, matches in required.items():
            if len(matches) != 1:
                issues.append(
                    f"{video_id}: expected one {label}, found {len(matches)}"
                    + (f" ({', '.join(str(path) for path in matches)})" if matches else "")
                )
                invalid = True
        raw_matches = raw_videos.get(video_id, [])
        if len(raw_matches) > 1:
            issues.append(
                f"{video_id}: expected at most one raw video, found {len(raw_matches)}"
            )
            invalid = True
        if invalid:
            continue
        mapping = required["mapping CSV"][0]
        clip = required["CLIP NPY"][0]
        keyframe_dir = required["keyframe directory"][0]
        mapping_rows = _mapping_row_count(mapping)
        feature_rows, dimension = _npy_shape(clip)
        if mapping_rows != feature_rows:
            issues.append(
                f"{video_id}: mapping/CLIP row mismatch "
                f"mapping={mapping_rows}, features={feature_rows}"
            )
            continue
        if expected_dimension is not None and dimension != expected_dimension:
            issues.append(
                f"{video_id}: CLIP dimension mismatch observed={dimension}, "
                f"expected={expected_dimension}"
            )
            continue
        image_count = sum(
            1
            for path in keyframe_dir.rglob("*")
            if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS
        )
        if image_count == 0:
            issues.append(f"{video_id}: keyframe directory contains no supported images")
            continue
        videos.append(
            DiscoveredVideo(
                video_id=video_id,
                mapping_csv_path=mapping,
                clip_npy_path=clip,
                keyframe_directory=keyframe_dir,
                raw_video_path=raw_matches[0] if raw_matches else None,
                row_count=mapping_rows,
                embedding_dimension=dimension,
                mapping_size_bytes=mapping.stat().st_size,
                clip_size_bytes=clip.stat().st_size,
                keyframe_image_count=image_count,
            )
        )
    if issues:
        raise CorpusDiscoveryError(
            "corpus discovery found incomplete or ambiguous artifacts",
            issues=tuple(issues),
        )
    if not videos:
        raise CorpusDiscoveryError("corpus discovery found no complete videos")
    ordered = tuple(sorted(videos, key=lambda video: video.video_id.casefold()))
    return CorpusManifest(
        input_root=Path(input_root).resolve(strict=False),
        dataset_root=dataset_root,
        fingerprint=_fingerprint(ordered),
        videos=ordered,
    )


def load_corpus_manifest(path: Path, *, validate_sources: bool = True) -> CorpusManifest:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"feature manifest not found: {source}")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid feature manifest JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("feature manifest root must be an object")
    if payload.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported feature manifest schema_version")
    if payload.get("discovery_version") != DISCOVERY_VERSION:
        raise ValueError("unsupported feature manifest discovery_version")
    raw_videos = payload.get("videos")
    if not isinstance(raw_videos, list) or not raw_videos:
        raise ValueError("feature manifest videos must be a non-empty list")
    videos: list[DiscoveredVideo] = []
    for index, item in enumerate(raw_videos):
        if not isinstance(item, dict):
            raise ValueError(f"manifest video entry {index} must be an object")
        try:
            raw_video = item.get("raw_video_path")
            video = DiscoveredVideo(
                video_id=str(item["video_id"]),
                mapping_csv_path=Path(str(item["mapping_csv_path"])),
                clip_npy_path=Path(str(item["clip_npy_path"])),
                keyframe_directory=Path(str(item["keyframe_directory"])),
                raw_video_path=Path(str(raw_video)) if raw_video else None,
                row_count=int(item["row_count"]),
                embedding_dimension=int(item["embedding_dimension"]),
                mapping_size_bytes=int(item["mapping_size_bytes"]),
                clip_size_bytes=int(item["clip_size_bytes"]),
                keyframe_image_count=int(item["keyframe_image_count"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid manifest video entry {index}") from exc
        if validate_sources:
            for label, artifact in (
                ("mapping CSV", video.mapping_csv_path),
                ("CLIP NPY", video.clip_npy_path),
                ("keyframe directory", video.keyframe_directory),
            ):
                if not artifact.exists():
                    raise FileNotFoundError(
                        f"reused manifest {label} is missing for {video.video_id}: {artifact}"
                    )
            if video.raw_video_path is not None and not video.raw_video_path.is_file():
                raise FileNotFoundError(
                    f"reused manifest raw video is missing for {video.video_id}: "
                    f"{video.raw_video_path}"
                )
        videos.append(video)
    ordered = tuple(sorted(videos, key=lambda video: video.video_id.casefold()))
    fingerprint = _fingerprint(ordered)
    if payload.get("manifest_fingerprint") != fingerprint:
        raise ValueError("feature manifest fingerprint mismatch")
    if len({video.video_id for video in ordered}) != len(ordered):
        raise ValueError("feature manifest contains duplicate video_id")
    return CorpusManifest(
        input_root=Path(str(payload.get("input_root", ""))),
        dataset_root=Path(str(payload.get("dataset_root", ""))),
        fingerprint=fingerprint,
        videos=ordered,
    )
