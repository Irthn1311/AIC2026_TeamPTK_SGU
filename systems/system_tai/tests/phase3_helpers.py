from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def create_corpus(
    input_root: Path,
    videos: dict[str, tuple[list[int], np.ndarray]],
    *,
    include_raw_video: bool = True,
) -> Path:
    dataset = input_root / "datasets" / "owner" / "dataset-aic"
    mapping_root = dataset / "map-keyframes-aic25-b1" / "map-keyframes"
    clip_root = dataset / "clip-features-32-aic25-b1" / "clip-features-32"
    keyframe_root = dataset / "keyframes" / "keyframes"
    mapping_root.mkdir(parents=True)
    clip_root.mkdir(parents=True)
    keyframe_root.mkdir(parents=True)
    for video_id, (frame_ids, matrix) in videos.items():
        with (mapping_root / f"{video_id}.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as stream:
            writer = csv.writer(stream)
            writer.writerow(["n", "pts_time", "fps", "frame_idx"])
            for index, frame_id in enumerate(frame_ids, start=1):
                writer.writerow([index, (index - 1) / 30, 30.0, frame_id])
        np.save(clip_root / f"{video_id}.npy", matrix)
        group = video_id.split("_", maxsplit=1)[0]
        directory = keyframe_root / f"Keyframes_{group}" / "keyframes" / video_id
        directory.mkdir(parents=True)
        for index in range(1, len(frame_ids) + 1):
            (directory / f"{index:03d}.jpg").write_bytes(b"synthetic-not-an-image")
        if include_raw_video:
            video_root = dataset / f"Videos_{group}_a"
            video_root.mkdir(exist_ok=True)
            (video_root / f"{video_id}.mp4").write_bytes(b"synthetic-video-marker")
    return dataset


def feature_matrix(rows: list[tuple[int, float]]) -> np.ndarray:
    matrix = np.zeros((len(rows), 512), dtype=np.float32)
    for index, (dimension, value) in enumerate(rows):
        matrix[index, dimension] = value
    return matrix
