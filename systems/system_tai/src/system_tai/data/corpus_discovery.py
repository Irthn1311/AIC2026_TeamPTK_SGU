"""One-pass bounded BTC corpus discovery and portable manifests."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

import numpy as np

VIDEO_ID_PATTERN = re.compile(r"^[A-Za-z]\d{2}_V\d{3}$")
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".webm"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MANIFEST_SCHEMA_VERSION = 1
PORTABLE_MANIFEST_SCHEMA_VERSION = 2
DISCOVERY_VERSION = "system_tai_btc_corpus_v1"
PORTABLE_DISCOVERY_VERSION = "system_tai_btc_corpus_v2_portable"
PORTABLE_PATH_MODE = "dataset_root_relative_posix"
DATASET_IDENTITY_ALGORITHM = "sha256-relative-artifact-metadata-v1"


class CorpusDiscoveryError(RuntimeError):
    def __init__(self, message: str, *, issues: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.issues = issues


class DiscoveryValidation(StrEnum):
    STRICT = "strict"
    FAST = "fast"


@dataclass(frozen=True, slots=True)
class DiscoveryMetrics:
    dataset_root_resolution_seconds: float = 0.0
    family_index_seconds: float = 0.0
    mapping_validation_seconds: float = 0.0
    clip_shape_validation_seconds: float = 0.0
    keyframe_stats_seconds: float = 0.0
    raw_video_index_seconds: float = 0.0
    manifest_fingerprint_seconds: float = 0.0
    manifest_write_seconds: float = 0.0
    total_discovery_seconds: float = 0.0
    filesystem_directories_visited: int = 0
    filesystem_files_visited: int = 0
    keyframe_images_seen: int = 0
    mapping_files_validated: int = 0
    clip_files_validated: int = 0
    raw_video_files_seen: int = 0
    family_root_traversals: Mapping[str, int] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {
            "dataset_root_resolution_seconds": self.dataset_root_resolution_seconds,
            "family_index_seconds": self.family_index_seconds,
            "mapping_validation_seconds": self.mapping_validation_seconds,
            "clip_shape_validation_seconds": self.clip_shape_validation_seconds,
            "keyframe_stats_seconds": self.keyframe_stats_seconds,
            "raw_video_index_seconds": self.raw_video_index_seconds,
            "manifest_fingerprint_seconds": self.manifest_fingerprint_seconds,
            "manifest_write_seconds": self.manifest_write_seconds,
            "total_discovery_seconds": self.total_discovery_seconds,
            "filesystem_directories_visited": self.filesystem_directories_visited,
            "filesystem_files_visited": self.filesystem_files_visited,
            "keyframe_images_seen": self.keyframe_images_seen,
            "mapping_files_validated": self.mapping_files_validated,
            "clip_files_validated": self.clip_files_validated,
            "raw_video_files_seen": self.raw_video_files_seen,
            "family_root_traversals": dict(self.family_root_traversals),
        }

    @classmethod
    def from_payload(cls, payload: Any) -> DiscoveryMetrics:
        if not isinstance(payload, dict):
            return cls()
        names = {item.name for item in cls.__dataclass_fields__.values()}
        values = {name: payload[name] for name in names if name in payload}
        try:
            return cls(**values)
        except (TypeError, ValueError):
            return cls()


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
    portable: bool = False
    dataset_identity: str | None = None
    validation_mode: DiscoveryValidation = DiscoveryValidation.STRICT
    discovery_metrics: DiscoveryMetrics = field(default_factory=DiscoveryMetrics)

    @property
    def total_rows(self) -> int:
        return sum(video.row_count for video in self.videos)

    def to_payload(self, *, portable: bool | None = None) -> dict[str, Any]:
        use_portable = self.portable if portable is None else portable
        if use_portable:
            videos = [_portable_video_payload(video, self.dataset_root) for video in self.videos]
            fingerprint = _hash_payload(videos)
            discovery_metrics = self.discovery_metrics.to_payload()
            discovery_metrics["family_root_traversals"] = {
                _relative_posix(Path(root), self.dataset_root): count
                for root, count in self.discovery_metrics.family_root_traversals.items()
            }
            return {
                "schema_version": PORTABLE_MANIFEST_SCHEMA_VERSION,
                "discovery_version": PORTABLE_DISCOVERY_VERSION,
                "path_mode": PORTABLE_PATH_MODE,
                "manifest_fingerprint": fingerprint,
                "dataset_identity": {
                    "algorithm": DATASET_IDENTITY_ALGORITHM,
                    "fingerprint": fingerprint,
                },
                "dataset_root_hint": self.dataset_root.name,
                "discovery_validation": self.validation_mode.value,
                "video_count": len(self.videos),
                "feature_row_count": self.total_rows,
                "discovery_metrics": discovery_metrics,
                "videos": videos,
            }
        videos = [_video_payload(video) for video in self.videos]
        return {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "discovery_version": DISCOVERY_VERSION,
            "manifest_fingerprint": _hash_payload(videos),
            "input_root": str(self.input_root),
            "dataset_root": str(self.dataset_root),
            "discovery_validation": self.validation_mode.value,
            "video_count": len(self.videos),
            "feature_row_count": self.total_rows,
            "discovery_metrics": self.discovery_metrics.to_payload(),
            "videos": videos,
        }

    def write(self, path: Path, *, portable: bool | None = None) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(
                self.to_payload(portable=portable),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return destination


@dataclass(frozen=True, slots=True)
class WalkDirectory:
    path: Path
    directory_names: tuple[str, ...]
    file_names: tuple[str, ...]


class TreeWalker(Protocol):
    def walk(self, root: Path) -> Iterator[WalkDirectory]: ...


class DeterministicTreeWalker:
    """Yield every directory and file once, ordered identically on each run."""

    def walk(self, root: Path) -> Iterator[WalkDirectory]:
        for current, directory_names, file_names in os.walk(root):
            directory_names.sort(key=str.casefold)
            file_names.sort(key=str.casefold)
            yield WalkDirectory(
                path=Path(current),
                directory_names=tuple(directory_names),
                file_names=tuple(file_names),
            )


@dataclass(frozen=True, slots=True)
class _FamilyIndex:
    mappings: Mapping[str, tuple[Path, ...]]
    clips: Mapping[str, tuple[Path, ...]]
    keyframes: Mapping[str, tuple[Path, ...]]
    keyframe_image_counts: Mapping[Path, int]
    raw_videos: Mapping[str, tuple[Path, ...]]
    directories_visited: int
    files_visited: int
    keyframe_images_seen: int
    raw_video_files_seen: int
    keyframe_stats_seconds: float
    raw_video_index_seconds: float
    root_traversals: Mapping[str, int]


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


def _relative_posix(path: Path, dataset_root: Path) -> str:
    resolved = path.resolve(strict=False)
    root = dataset_root.resolve(strict=False)
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(f"portable artifact path is outside dataset root: {path}") from exc


def _portable_video_payload(video: DiscoveredVideo, dataset_root: Path) -> dict[str, Any]:
    return {
        "video_id": video.video_id,
        "mapping_csv_path": _relative_posix(video.mapping_csv_path, dataset_root),
        "clip_npy_path": _relative_posix(video.clip_npy_path, dataset_root),
        "keyframe_directory": _relative_posix(video.keyframe_directory, dataset_root),
        "raw_video_path": (
            _relative_posix(video.raw_video_path, dataset_root)
            if video.raw_video_path is not None
            else None
        ),
        "row_count": video.row_count,
        "embedding_dimension": video.embedding_dimension,
        "mapping_size_bytes": video.mapping_size_bytes,
        "clip_size_bytes": video.clip_size_bytes,
        "keyframe_image_count": video.keyframe_image_count,
    }


def _hash_payload(payload: Any) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _fingerprint(videos: tuple[DiscoveredVideo, ...]) -> str:
    """Retain the public Phase 3 absolute-manifest fingerprint helper."""

    return _hash_payload([_video_payload(video) for video in videos])


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


def _frozen_index(index: Mapping[str, list[Path]]) -> dict[str, tuple[Path, ...]]:
    return {
        key: tuple(sorted(paths, key=lambda path: str(path).casefold()))
        for key, paths in index.items()
    }


def _nearest_video_directory(current: Path, root: Path) -> Path | None:
    try:
        parts = current.relative_to(root).parts
    except ValueError:
        return None
    for length in range(len(parts), 0, -1):
        if VIDEO_ID_PATTERN.fullmatch(parts[length - 1]):
            return root.joinpath(*parts[:length]).resolve(strict=False)
    return None


def _index_families(
    families: Mapping[str, tuple[Path, ...]],
    *,
    walker: TreeWalker,
    clock: Callable[[], float],
) -> _FamilyIndex:
    mappings: dict[str, list[Path]] = {}
    clips: dict[str, list[Path]] = {}
    keyframes: dict[str, list[Path]] = {}
    keyframe_counts: dict[Path, int] = {}
    raw_videos: dict[str, list[Path]] = {}
    directories_visited = 0
    files_visited = 0
    keyframe_images_seen = 0
    raw_video_files_seen = 0
    keyframe_seconds = 0.0
    raw_seconds = 0.0
    root_traversals: dict[str, int] = {}
    for kind in ("mapping", "clip", "keyframes", "videos"):
        for root in families.get(kind, ()):
            root_key = str(root.resolve(strict=False))
            root_traversals[root_key] = root_traversals.get(root_key, 0) + 1
            for entry in walker.walk(root):
                directories_visited += 1
                files_visited += len(entry.file_names)
                if kind == "keyframes":
                    keyframe_start = clock()
                    if VIDEO_ID_PATTERN.fullmatch(entry.path.name):
                        directory = entry.path.resolve(strict=False)
                        keyframes.setdefault(entry.path.name, []).append(directory)
                        keyframe_counts.setdefault(directory, 0)
                    owner = _nearest_video_directory(entry.path, root)
                    if owner is not None:
                        count = sum(
                            Path(name).suffix.casefold() in IMAGE_EXTENSIONS
                            for name in entry.file_names
                        )
                        keyframe_counts[owner] = keyframe_counts.get(owner, 0) + count
                        keyframe_images_seen += count
                    keyframe_seconds += clock() - keyframe_start
                for name in entry.file_names:
                    path = entry.path / name
                    stem = path.stem
                    if not VIDEO_ID_PATTERN.fullmatch(stem):
                        continue
                    suffix = path.suffix.casefold()
                    resolved = path.resolve(strict=False)
                    if kind == "mapping" and suffix == ".csv":
                        mappings.setdefault(stem, []).append(resolved)
                    elif kind == "clip" and suffix == ".npy":
                        clips.setdefault(stem, []).append(resolved)
                    elif kind == "videos" and suffix in VIDEO_EXTENSIONS:
                        raw_start = clock()
                        raw_videos.setdefault(stem, []).append(resolved)
                        raw_video_files_seen += 1
                        raw_seconds += clock() - raw_start
    return _FamilyIndex(
        mappings=_frozen_index(mappings),
        clips=_frozen_index(clips),
        keyframes=_frozen_index(keyframes),
        keyframe_image_counts=dict(keyframe_counts),
        raw_videos=_frozen_index(raw_videos),
        directories_visited=directories_visited,
        files_visited=files_visited,
        keyframe_images_seen=keyframe_images_seen,
        raw_video_files_seen=raw_video_files_seen,
        keyframe_stats_seconds=keyframe_seconds,
        raw_video_index_seconds=raw_seconds,
        root_traversals=dict(root_traversals),
    )


def _mapping_row_count(path: Path, *, validate_columns: bool) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        if validate_columns:
            reader = csv.DictReader(stream)
            required = {"n", "pts_time", "fps", "frame_idx"}
            missing = sorted(required - set(reader.fieldnames or ()))
            if missing:
                raise CorpusDiscoveryError(
                    f"mapping CSV missing columns for {path.name}: {', '.join(missing)}"
                )
            return sum(1 for row in reader if any((value or "").strip() for value in row.values()))
        reader = csv.reader(stream)
        try:
            next(reader)
        except StopIteration as exc:
            raise CorpusDiscoveryError(f"mapping CSV is empty: {path}") from exc
        return sum(1 for row in reader if any(value.strip() for value in row))


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
    validation_mode: DiscoveryValidation | str = DiscoveryValidation.STRICT,
    portable: bool = False,
    walker: TreeWalker | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> CorpusManifest:
    mode = DiscoveryValidation(validation_mode)
    discovery_start = clock()
    root_start = clock()
    dataset_root = resolve_dataset_root(input_root, max_depth=max_root_depth)
    root_seconds = clock() - root_start
    families = _family_roots(dataset_root)
    index_start = clock()
    index = _index_families(
        families,
        walker=walker or DeterministicTreeWalker(),
        clock=clock,
    )
    index_seconds = clock() - index_start
    video_ids = sorted(
        set(index.mappings) | set(index.clips) | set(index.keyframes),
        key=str.casefold,
    )
    issues: list[str] = []
    videos: list[DiscoveredVideo] = []
    mapping_seconds = 0.0
    clip_seconds = 0.0
    mapping_files_validated = 0
    clip_files_validated = 0
    for video_id in video_ids:
        required = {
            "mapping CSV": index.mappings.get(video_id, ()),
            "CLIP NPY": index.clips.get(video_id, ()),
            "keyframe directory": index.keyframes.get(video_id, ()),
        }
        invalid = False
        for label, matches in required.items():
            if len(matches) != 1:
                issues.append(
                    f"{video_id}: expected one {label}, found {len(matches)}"
                    + (f" ({', '.join(str(path) for path in matches)})" if matches else "")
                )
                invalid = True
        raw_matches = index.raw_videos.get(video_id, ())
        if len(raw_matches) > 1:
            issues.append(f"{video_id}: expected at most one raw video, found {len(raw_matches)}")
            invalid = True
        if invalid:
            continue
        mapping = required["mapping CSV"][0]
        clip = required["CLIP NPY"][0]
        keyframe_dir = required["keyframe directory"][0]
        mapping_start = clock()
        try:
            mapping_rows = _mapping_row_count(
                mapping,
                validate_columns=True,
            )
            mapping_files_validated += 1
        except (CorpusDiscoveryError, OSError) as exc:
            issues.append(f"{video_id}: mapping validation failed: {exc}")
            mapping_seconds += clock() - mapping_start
            continue
        mapping_seconds += clock() - mapping_start
        clip_start = clock()
        try:
            feature_rows, dimension = _npy_shape(clip)
            clip_files_validated += 1
        except (CorpusDiscoveryError, OSError, ValueError) as exc:
            issues.append(f"{video_id}: CLIP validation failed: {exc}")
            clip_seconds += clock() - clip_start
            continue
        clip_seconds += clock() - clip_start
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
        image_count = index.keyframe_image_counts.get(keyframe_dir, 0)
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
    fingerprint_start = clock()
    fingerprint = (
        _hash_payload([_portable_video_payload(video, dataset_root) for video in ordered])
        if portable
        else _fingerprint(ordered)
    )
    fingerprint_seconds = clock() - fingerprint_start
    metrics = DiscoveryMetrics(
        dataset_root_resolution_seconds=root_seconds,
        family_index_seconds=index_seconds,
        mapping_validation_seconds=mapping_seconds,
        clip_shape_validation_seconds=clip_seconds,
        keyframe_stats_seconds=index.keyframe_stats_seconds,
        raw_video_index_seconds=index.raw_video_index_seconds,
        manifest_fingerprint_seconds=fingerprint_seconds,
        total_discovery_seconds=clock() - discovery_start,
        filesystem_directories_visited=index.directories_visited,
        filesystem_files_visited=index.files_visited,
        keyframe_images_seen=index.keyframe_images_seen,
        mapping_files_validated=mapping_files_validated,
        clip_files_validated=clip_files_validated,
        raw_video_files_seen=index.raw_video_files_seen,
        family_root_traversals=index.root_traversals,
    )
    return CorpusManifest(
        input_root=Path(input_root).resolve(strict=False),
        dataset_root=dataset_root,
        fingerprint=fingerprint,
        videos=ordered,
        schema_version=(PORTABLE_MANIFEST_SCHEMA_VERSION if portable else MANIFEST_SCHEMA_VERSION),
        discovery_version=(PORTABLE_DISCOVERY_VERSION if portable else DISCOVERY_VERSION),
        portable=portable,
        dataset_identity=fingerprint if portable else None,
        validation_mode=mode,
        discovery_metrics=metrics,
    )


def _portable_path(dataset_root: Path, value: Any) -> Path:
    text = str(value)
    pure = PurePosixPath(text)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise ValueError(f"invalid portable artifact path: {text}")
    path = dataset_root.joinpath(*pure.parts).resolve(strict=False)
    if not path.is_relative_to(dataset_root.resolve(strict=False)):
        raise ValueError(f"portable artifact escapes dataset root: {text}")
    return path


def _parse_video(
    item: Any,
    *,
    index: int,
    dataset_root: Path | None,
) -> DiscoveredVideo:
    if not isinstance(item, dict):
        raise ValueError(f"manifest video entry {index} must be an object")
    try:
        resolve_path = (
            (lambda value: _portable_path(dataset_root, value))
            if dataset_root is not None
            else (lambda value: Path(str(value)))
        )
        raw_value = item.get("raw_video_path")
        return DiscoveredVideo(
            video_id=str(item["video_id"]),
            mapping_csv_path=resolve_path(item["mapping_csv_path"]),
            clip_npy_path=resolve_path(item["clip_npy_path"]),
            keyframe_directory=resolve_path(item["keyframe_directory"]),
            raw_video_path=resolve_path(raw_value) if raw_value else None,
            row_count=int(item["row_count"]),
            embedding_dimension=int(item["embedding_dimension"]),
            mapping_size_bytes=int(item["mapping_size_bytes"]),
            clip_size_bytes=int(item["clip_size_bytes"]),
            keyframe_image_count=int(item["keyframe_image_count"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid manifest video entry {index}") from exc


def _validate_manifest_sources(videos: Sequence[DiscoveredVideo]) -> None:
    for video in videos:
        for label, artifact in (
            ("mapping CSV", video.mapping_csv_path),
            ("CLIP NPY", video.clip_npy_path),
            ("keyframe directory", video.keyframe_directory),
        ):
            if not artifact.exists():
                raise FileNotFoundError(
                    f"reused manifest {label} is missing for {video.video_id}: {artifact}"
                )
        if video.mapping_csv_path.stat().st_size != video.mapping_size_bytes:
            raise ValueError(f"dataset identity mismatch for {video.video_id} mapping size")
        if video.clip_npy_path.stat().st_size != video.clip_size_bytes:
            raise ValueError(f"dataset identity mismatch for {video.video_id} CLIP size")
        if video.raw_video_path is not None and not video.raw_video_path.is_file():
            raise FileNotFoundError(
                f"reused manifest raw video is missing for {video.video_id}: {video.raw_video_path}"
            )


def load_corpus_manifest(
    path: Path,
    *,
    input_root: Path | None = None,
    validate_sources: bool = True,
    max_root_depth: int = 4,
) -> CorpusManifest:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"feature manifest not found: {source}")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid feature manifest JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("feature manifest root must be an object")
    schema_version = payload.get("schema_version")
    portable = schema_version == PORTABLE_MANIFEST_SCHEMA_VERSION
    if schema_version not in {MANIFEST_SCHEMA_VERSION, PORTABLE_MANIFEST_SCHEMA_VERSION}:
        raise ValueError("unsupported feature manifest schema_version")
    expected_discovery = PORTABLE_DISCOVERY_VERSION if portable else DISCOVERY_VERSION
    if payload.get("discovery_version") != expected_discovery:
        raise ValueError("unsupported feature manifest discovery_version")
    if portable and payload.get("path_mode") != PORTABLE_PATH_MODE:
        raise ValueError("unsupported portable manifest path_mode")
    if portable:
        if input_root is None:
            raise ValueError("portable manifest loading requires input_root")
        dataset_root = resolve_dataset_root(input_root, max_depth=max_root_depth)
        resolved_input_root = Path(input_root).resolve(strict=False)
    else:
        dataset_root = Path(str(payload.get("dataset_root", "")))
        resolved_input_root = Path(str(payload.get("input_root", "")))
    raw_videos = payload.get("videos")
    if not isinstance(raw_videos, list) or not raw_videos:
        raise ValueError("feature manifest videos must be a non-empty list")
    videos = [
        _parse_video(
            item,
            index=index,
            dataset_root=dataset_root if portable else None,
        )
        for index, item in enumerate(raw_videos)
    ]
    ordered = tuple(sorted(videos, key=lambda video: video.video_id.casefold()))
    if len({video.video_id for video in ordered}) != len(ordered):
        raise ValueError("feature manifest contains duplicate video_id")
    fingerprint = (
        _hash_payload([_portable_video_payload(video, dataset_root) for video in ordered])
        if portable
        else _fingerprint(ordered)
    )
    if payload.get("manifest_fingerprint") != fingerprint:
        raise ValueError("feature manifest fingerprint mismatch")
    if portable:
        identity = payload.get("dataset_identity")
        if not isinstance(identity, dict):
            raise ValueError("portable manifest dataset_identity must be an object")
        if identity.get("algorithm") != DATASET_IDENTITY_ALGORITHM:
            raise ValueError("unsupported portable dataset identity algorithm")
        if identity.get("fingerprint") != fingerprint:
            raise ValueError("portable manifest dataset identity mismatch")
    if validate_sources:
        _validate_manifest_sources(ordered)
    discovery_metrics = DiscoveryMetrics.from_payload(payload.get("discovery_metrics"))
    if portable:
        discovery_metrics = replace(
            discovery_metrics,
            family_root_traversals={
                str(_portable_path(dataset_root, root)): count
                for root, count in discovery_metrics.family_root_traversals.items()
            },
        )
    return CorpusManifest(
        input_root=resolved_input_root,
        dataset_root=dataset_root,
        fingerprint=fingerprint,
        videos=ordered,
        schema_version=int(schema_version),
        discovery_version=expected_discovery,
        portable=portable,
        dataset_identity=fingerprint if portable else None,
        validation_mode=DiscoveryValidation(
            payload.get("discovery_validation", DiscoveryValidation.STRICT.value)
        ),
        discovery_metrics=discovery_metrics,
    )


@dataclass(frozen=True, slots=True)
class ManifestCacheResult:
    manifest: CorpusManifest
    status: str
    cache_path: Path


def load_or_build_manifest_cache(
    cache_path: Path,
    *,
    input_root: Path,
    expected_dimension: int = 512,
    max_root_depth: int = 4,
    rebuild_invalid: bool = False,
    discoverer: Callable[..., CorpusManifest] = discover_corpus,
) -> ManifestCacheResult:
    cache = Path(cache_path)
    if cache.exists():
        try:
            manifest = load_corpus_manifest(
                cache,
                input_root=input_root,
                max_root_depth=max_root_depth,
            )
            return ManifestCacheResult(manifest, "CACHE_HIT", cache)
        except (CorpusDiscoveryError, FileNotFoundError, OSError, ValueError) as exc:
            if not rebuild_invalid:
                raise CorpusDiscoveryError(
                    f"manifest cache is invalid; explicit rebuild required: {cache}",
                    issues=(f"{type(exc).__name__}: {exc}",),
                ) from exc
            status = "CACHE_REBUILT"
    else:
        status = "CACHE_BUILT"
    manifest = discoverer(
        input_root,
        expected_dimension=expected_dimension,
        max_root_depth=max_root_depth,
        validation_mode=DiscoveryValidation.STRICT,
        portable=True,
    )
    manifest.write(cache, portable=True)
    return ManifestCacheResult(manifest, status, cache)


def with_manifest_write_seconds(manifest: CorpusManifest, seconds: float) -> CorpusManifest:
    return replace(
        manifest,
        discovery_metrics=replace(
            manifest.discovery_metrics,
            manifest_write_seconds=float(seconds),
        ),
    )
