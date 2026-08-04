"""Discover Dataset_AIC2026 artifacts without assuming a Kaggle dataset slug."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

DEFAULT_HINTS = (
    "Videos_L21_a",
    "clip-features-32-aic25-b1",
    "keyframes",
    "map-keyframes-aic25-b1",
    "media-info-aic25-b1",
    "objects-aic25-b1",
)
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".m4v"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class DiscoveryError(RuntimeError):
    """Raised when an artifact cannot be resolved uniquely."""

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.details = details or {}


def _limited_directories(root: Path, *, max_depth: int = 2) -> list[Path]:
    directories = [root]
    frontier = [(root, 0)]
    while frontier:
        current, depth = frontier.pop(0)
        if depth >= max_depth:
            continue
        try:
            children = sorted(
                (path for path in current.iterdir() if path.is_dir()),
                key=lambda path: str(path).lower(),
            )
        except OSError:
            continue
        for child in children:
            directories.append(child)
            frontier.append((child, depth + 1))
    return directories


def _family_kind(path: Path) -> str | None:
    name = path.name.lower()
    if "media" in name and "info" in name:
        return "media"
    if "object" in name:
        return "objects"
    if "clip" in name and ("feature" in name or "vit" in name):
        return "clip"
    if "map" in name and "keyframe" in name:
        return "mapping"
    if "keyframe" in name:
        return "keyframes"
    if "video" in name:
        return "videos"
    return None


def _candidate_families(root: Path, hints: Sequence[str]) -> dict[str, list[Path]]:
    families: dict[str, list[Path]] = {
        "videos": [],
        "mapping": [],
        "clip": [],
        "keyframes": [],
        "media": [],
        "objects": [],
    }
    hint_names = {hint.lower() for hint in hints}
    for directory in _limited_directories(root):
        kind = _family_kind(directory)
        if kind is None:
            continue
        if directory.name.lower() in hint_names or directory != root:
            families[kind].append(directory)
    for kind in families:
        families[kind] = sorted(set(families[kind]), key=lambda path: str(path).lower())
    return families


def _likely_dataset_roots(input_root: Path, *, max_depth: int = 4) -> list[Path]:
    candidates: list[Path] = []
    for directory in _limited_directories(input_root, max_depth=max_depth):
        try:
            child_kinds = {
                kind
                for child in directory.iterdir()
                if child.is_dir() and (kind := _family_kind(child)) is not None
            }
        except OSError:
            continue
        if len(child_kinds) >= 2:
            candidates.append(directory.resolve(strict=False))
    return sorted(set(candidates), key=lambda path: str(path).lower())


def _walk_target_files(
    roots: Iterable[Path],
    *,
    video_id: str,
    extensions: set[str],
    allow_prefix: bool = False,
) -> list[Path]:
    target = video_id.lower()
    matches: set[Path] = set()
    group = target.split("_", maxsplit=1)[0]
    for root in roots:
        for current, directories, files in os.walk(root):
            lowered = {name: name.lower() for name in directories}
            target_directories = [
                name for name, lowered_name in lowered.items() if lowered_name == target
            ]
            group_directories = [
                name for name, lowered_name in lowered.items() if lowered_name.startswith(group)
            ]
            if target_directories:
                directories[:] = target_directories
            elif group_directories:
                directories[:] = group_directories
            for filename in files:
                path = Path(current) / filename
                stem = path.stem.lower()
                name_matches = stem == target or (allow_prefix and stem.startswith(target + "_"))
                if name_matches and path.suffix.lower() in extensions:
                    matches.add(path.resolve(strict=False))
    return sorted(matches, key=lambda path: str(path).lower())


def _keyframe_sources(roots: Iterable[Path], video_id: str) -> tuple[list[Path], list[Path]]:
    target = video_id.lower()
    group = target.split("_", maxsplit=1)[0]
    directories: set[Path] = set()
    loose_images: set[Path] = set()
    for root in roots:
        for current, child_dirs, files in os.walk(root):
            current_path = Path(current)
            if current_path.name.lower() == target:
                directories.add(current_path.resolve(strict=False))
                child_dirs.clear()
                continue
            lowered = {name: name.lower() for name in child_dirs}
            target_directories = [
                name for name, lowered_name in lowered.items() if lowered_name == target
            ]
            group_directories = [
                name for name, lowered_name in lowered.items() if lowered_name.startswith(group)
            ]
            if target_directories:
                child_dirs[:] = target_directories
            elif group_directories:
                child_dirs[:] = group_directories
            for filename in files:
                path = current_path / filename
                if path.suffix.lower() in IMAGE_EXTENSIONS and path.stem.lower().startswith(
                    target + "_"
                ):
                    loose_images.add(path.resolve(strict=False))
    return (
        sorted(directories, key=lambda path: str(path).lower()),
        sorted(loose_images, key=lambda path: str(path).lower()),
    )


def _optional_family_file(
    roots: Iterable[Path],
    *,
    video_id: str,
    extensions: set[str],
) -> list[Path]:
    exact = _walk_target_files(
        roots,
        video_id=video_id,
        extensions=extensions,
        allow_prefix=True,
    )
    if exact:
        return exact
    all_files: list[Path] = []
    for root in roots:
        for current, _directories, files in os.walk(root):
            for filename in files:
                path = Path(current) / filename
                if path.suffix.lower() in extensions:
                    all_files.append(path.resolve(strict=False))
    unique = sorted(set(all_files), key=lambda path: str(path).lower())
    return unique


def _one(label: str, matches: Sequence[Path], *, required: bool) -> Path | None:
    if len(matches) > 1:
        raise DiscoveryError(
            f"ambiguous {label}: found {len(matches)} matches",
            details={label: [str(path) for path in matches]},
        )
    if not matches:
        if required:
            raise DiscoveryError(f"missing required {label}")
        return None
    return matches[0]


def _inspect_dataset_root(root: Path, video_id: str, hints: Sequence[str]) -> dict[str, Any]:
    families = _candidate_families(root, hints)
    videos = _walk_target_files(families["videos"], video_id=video_id, extensions=VIDEO_EXTENSIONS)
    mappings = _walk_target_files(families["mapping"], video_id=video_id, extensions={".csv"})
    clip_arrays = _walk_target_files(families["clip"], video_id=video_id, extensions={".npy"})
    keyframe_dirs, loose_keyframes = _keyframe_sources(families["keyframes"], video_id)
    media_files = _optional_family_file(
        families["media"], video_id=video_id, extensions={".json", ".csv", ".yaml", ".yml"}
    )
    object_files = _walk_target_files(
        families["objects"], video_id=video_id, extensions={".json"}, allow_prefix=True
    )
    return {
        "root": root.resolve(strict=False),
        "families": families,
        "videos": videos,
        "mappings": mappings,
        "clip_arrays": clip_arrays,
        "keyframe_dirs": keyframe_dirs,
        "loose_keyframes": loose_keyframes,
        "media_files": media_files,
        "object_files": object_files,
    }


def discover(
    input_root: Path,
    video_id: str,
    *,
    hints: Sequence[str] = DEFAULT_HINTS,
) -> dict[str, Any]:
    input_root = Path(input_root)
    if not input_root.is_dir():
        raise DiscoveryError(f"input root is not a directory: {input_root}")
    if not video_id.strip():
        raise DiscoveryError("video_id must not be empty")

    direct_children = sorted(
        (path for path in input_root.iterdir() if path.is_dir()),
        key=lambda path: str(path).lower(),
    )
    likely_roots = _likely_dataset_roots(input_root)
    inspections = [_inspect_dataset_root(root, video_id, hints) for root in likely_roots]
    candidates = [
        item
        for item in inspections
        if item["videos"] or item["mappings"] or item["clip_arrays"] or item["keyframe_dirs"]
    ]
    if not candidates:
        raise DiscoveryError(
            f"no Dataset_AIC2026 candidate contains artifacts for {video_id}",
            details={
                "scanned_children": [str(path) for path in direct_children],
                "likely_dataset_roots": [str(path) for path in likely_roots],
            },
        )
    if len(candidates) > 1:
        raise DiscoveryError(
            f"ambiguous dataset root for {video_id}: found {len(candidates)} candidates",
            details={"candidate_roots": [str(item["root"]) for item in candidates]},
        )

    selected = candidates[0]
    video = _one("original_video", selected["videos"], required=True)
    mapping = _one("mapping_csv", selected["mappings"], required=True)
    clip_npy = _one("clip_npy", selected["clip_arrays"], required=True)
    keyframe_dir = _one("keyframe_directory", selected["keyframe_dirs"], required=False)
    if keyframe_dir is not None and selected["loose_keyframes"]:
        raise DiscoveryError(
            "ambiguous keyframe sources: found both a directory and loose image files",
            details={
                "keyframe_directory": str(keyframe_dir),
                "loose_keyframes": [str(path) for path in selected["loose_keyframes"]],
            },
        )
    if keyframe_dir is None and not selected["loose_keyframes"]:
        raise DiscoveryError("missing required keyframe directory or image files")
    media_info = _one("media_info", selected["media_files"], required=False)
    object_json = _one("object_json", selected["object_files"], required=False)

    keyframe_images: list[Path]
    if keyframe_dir is not None:
        keyframe_images = sorted(
            (
                path.resolve(strict=False)
                for path in keyframe_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            ),
            key=lambda path: str(path).lower(),
        )
    else:
        keyframe_images = selected["loose_keyframes"]

    matched_families = {
        kind: [str(path) for path in paths] for kind, paths in selected["families"].items() if paths
    }
    matched_hints = sorted(
        {
            hint
            for hint in hints
            for paths in selected["families"].values()
            for path in paths
            if hint.lower() in path.name.lower() or path.name.lower() in hint.lower()
        },
        key=str.lower,
    )
    return {
        "status": "DISCOVERED",
        "input_root": str(input_root.resolve(strict=False)),
        "dataset_root": str(selected["root"]),
        "video_id": video_id,
        "matched_discovery_hints": matched_hints,
        "matched_families": matched_families,
        "artifacts": {
            "original_video": str(video),
            "mapping_csv": str(mapping),
            "clip_npy": str(clip_npy),
            "keyframe_directory": str(keyframe_dir) if keyframe_dir is not None else None,
            "keyframe_image_count": len(keyframe_images),
            "keyframe_image_examples": [str(path) for path in keyframe_images[:5]],
            "media_info": str(media_info) if media_info is not None else None,
            "object_json": str(object_json) if object_json is not None else None,
        },
        "copied_artifacts": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--hint", action="append", dest="hints")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = discover(args.input_root, args.video_id, hints=args.hints or DEFAULT_HINTS)
        exit_code = 0
    except DiscoveryError as exc:
        report = {
            "status": "ERROR",
            "input_root": str(args.input_root),
            "video_id": args.video_id,
            "error": str(exc),
            "details": exc.details,
            "copied_artifacts": False,
        }
        exit_code = 1
    serialized = json.dumps(report, ensure_ascii=False, separators=(",", ":"))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    return exit_code


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
