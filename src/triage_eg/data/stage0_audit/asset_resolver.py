"""Resolve known BTC asset layouts without recursive traversal."""

from __future__ import annotations

import re
from pathlib import Path

from triage_eg.data.stage0_audit.contracts import AssetPaths

VIDEO_ID_PATTERN = re.compile(r"^L\d+_V\d+$", re.ASCII)


def validate_video_id(video_id: str) -> None:
    if not VIDEO_ID_PATTERN.fullmatch(video_id):
        raise ValueError(f"Invalid video_id: {video_id}")


def discover_layout(root: Path) -> tuple[dict[str, str], dict[str, str]]:
    video_partitions: dict[str, str] = {}
    keyframe_partitions: dict[str, str] = {}
    for partition in sorted(root.glob("Videos_*"), key=lambda item: item.name):
        if not partition.is_dir() or partition.is_symlink():
            continue
        for path in sorted((partition / "video").glob("*.mp4"), key=lambda item: item.name):
            if VIDEO_ID_PATTERN.fullmatch(path.stem):
                video_partitions[path.stem] = partition.name
    keyframe_root = root / "keyframes" / "keyframes"
    if keyframe_root.is_dir():
        for partition in sorted(keyframe_root.iterdir(), key=lambda item: item.name):
            content = partition / "keyframes"
            if not content.is_dir() or partition.is_symlink() or content.is_symlink():
                continue
            for directory in sorted(content.iterdir(), key=lambda item: item.name):
                if (
                    directory.is_dir()
                    and not directory.is_symlink()
                    and VIDEO_ID_PATTERN.fullmatch(directory.name)
                ):
                    keyframe_partitions[directory.name] = partition.name
    return video_partitions, keyframe_partitions


def resolve_assets(
    root: Path, video_id: str, video_partitions: dict[str, str], keyframe_partitions: dict[str, str]
) -> AssetPaths:
    validate_video_id(video_id)
    video_partition = video_partitions.get(video_id)
    keyframe_partition = keyframe_partitions.get(video_id)
    return AssetPaths(
        video_id=video_id,
        video_partition=video_partition,
        keyframe_partition=keyframe_partition,
        video=root / (video_partition or "Videos_UNKNOWN") / "video" / f"{video_id}.mp4",
        mapping=root / "map-keyframes-aic25-b1" / "map-keyframes" / f"{video_id}.csv",
        keyframe_directory=root
        / "keyframes"
        / "keyframes"
        / (keyframe_partition or "Keyframes_UNKNOWN")
        / "keyframes"
        / video_id,
        clip=root / "clip-features-32-aic25-b1" / "clip-features-32" / f"{video_id}.npy",
        object_directory=root / "objects-aic25-b1" / "objects" / video_id,
        metadata=root / "media-info-aic25-b1" / "media-info" / f"{video_id}.json",
    )


def relative(root: Path, path: Path) -> str:
    return path.resolve(strict=False).relative_to(root.resolve(strict=False)).as_posix()
