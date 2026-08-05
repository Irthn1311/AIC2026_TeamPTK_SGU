"""Deterministic, catalog-driven Stage 1B probe sampling."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from triage_eg.retrieval.stage1.search import CompactCatalog

REASONS = ("EARLY", "MIDDLE", "LATE", "PARTITION_SPREAD", "RANDOM")


def select_probe_samples(
    stage1_root: str | Path,
    dataset_root: str | Path,
    sample_size: int = 50,
    seed: int = 2026,
) -> tuple[list[dict], list[dict]]:
    if not 1 <= sample_size <= 100:
        raise ValueError("probe sample_size must be between 1 and 100")
    index_root = Path(stage1_root).resolve(strict=True) / "index"
    dataset = Path(dataset_root).resolve(strict=True)
    catalog = CompactCatalog(index_root)
    video_index_values = np.asarray(catalog.video_index)
    boundaries = np.flatnonzero(np.diff(video_index_values)) + 1
    video_rows = list(np.split(np.arange(len(video_index_values), dtype=np.int64), boundaries))
    chosen_videos = np.linspace(0, len(video_rows) - 1, sample_size, dtype=np.int64)
    randomizer = np.random.default_rng(seed)
    samples, issues = [], []
    for sample_index, video_index in enumerate(chosen_videos):
        rows = video_rows[int(video_index)]
        reason = REASONS[sample_index % len(REASONS)]
        if reason == "EARLY":
            local_index = 0
        elif reason == "MIDDLE":
            local_index = len(rows) // 2
        elif reason == "LATE":
            local_index = len(rows) - 1
        elif reason == "RANDOM":
            local_index = int(randomizer.integers(0, len(rows)))
        else:
            local_index = round((sample_index / max(1, sample_size - 1)) * (len(rows) - 1))
        global_row = int(rows[local_index])
        mapped = catalog.map_row(global_row)
        keyframe = dataset / mapped["keyframe_relative_path"]
        record = {
            "sample_index": sample_index,
            "global_row": global_row,
            "video_id": mapped["video_id"],
            "n": mapped["n"],
            "original_frame_idx": mapped["original_frame_idx"],
            "keyframe_path": str(keyframe),
            "stored_clip_row": global_row,
            "selection_reason": reason,
        }
        samples.append(record)
        if not keyframe.is_file():
            issues.append(
                {
                    "severity": "ERROR",
                    "code": "IMAGE_LOAD_FAILED",
                    "candidate_id": None,
                    "global_row": global_row,
                    "video_id": mapped["video_id"],
                    "path": str(keyframe),
                    "message": "Probe keyframe is missing",
                    "evidence": {},
                }
            )
    return samples, issues
