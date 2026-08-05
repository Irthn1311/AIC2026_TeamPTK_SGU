from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from system_tai.common.schemas import FrameMappingRecord, VideoFeatureStore
from system_tai.features.btc_clip_store import LoadedVideoFeatureStore


def write_mapping(path: Path, rows: list[tuple[int, float, float, int]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["n", "pts_time", "fps", "frame_idx"])
        writer.writerows(rows)


def make_store(
    video_id: str,
    matrix: np.ndarray,
    frame_ids: list[int],
) -> LoadedVideoFeatureStore:
    mappings = tuple(
        FrameMappingRecord(
            clip_row=row,
            keyframe_order=row + 1,
            frame_id=frame_id,
            pts_time=float(row),
            fps=30.0,
        )
        for row, frame_id in enumerate(frame_ids)
    )
    matrix.setflags(write=False)
    return LoadedVideoFeatureStore(
        descriptor=VideoFeatureStore(
            video_id=video_id,
            mapping_csv_path=Path(f"{video_id}.csv"),
            clip_npy_path=Path(f"{video_id}.npy"),
            row_count=len(frame_ids),
            embedding_dimension=matrix.shape[1],
            normalized=bool(np.allclose(np.linalg.norm(matrix.astype(np.float32), axis=1), 1.0)),
        ),
        matrix=matrix,
        mappings=mappings,
    )
