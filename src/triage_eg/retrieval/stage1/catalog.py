"""Deterministic compact global BTC frame catalog."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from triage_eg.retrieval.stage1.stage0_loader import Stage0Bundle, iter_jsonl

CATALOG_ARRAYS = {
    "frame_video_index.npy": np.int32,
    "frame_n.npy": np.int32,
    "frame_original_idx.npy": np.int64,
    "frame_pts_time.npy": np.float64,
    "frame_mapping_fps.npy": np.float32,
    "duplicate_group_size.npy": np.int16,
}


@dataclass(frozen=True)
class CatalogData:
    rows: tuple[dict[str, Any], ...]
    video_table: tuple[dict[str, Any], ...]


def load_catalog_rows(bundle: Stage0Bundle) -> CatalogData:
    rows = sorted(
        iter_jsonl(bundle.frame_manifest_path), key=lambda item: (item["video_id"], item["n"])
    )
    expected = int(bundle.summary["mapping_rows"])
    if len(rows) != expected:
        raise ValueError(f"Catalog row count {len(rows)} does not match Stage 0 {expected}")
    seen: set[tuple[str, int]] = set()
    videos: dict[str, dict[str, Any]] = {}
    previous: tuple[str, int] | None = None
    for global_row, row in enumerate(rows):
        video_id = str(row["video_id"])
        n = int(row["n"])
        key = (video_id, n)
        if key in seen:
            raise ValueError(f"Duplicate catalog key: {key}")
        seen.add(key)
        if int(row["clip_row_index"]) != n - 1:
            raise ValueError(f"clip_row_index contract failed for {video_id}/{n}")
        if previous is not None and key <= previous:
            raise ValueError("Catalog ordering must be video_id then n")
        previous = key
        row["global_row"] = global_row
        prefix = str(Path(row["keyframe_relative_path"]).parent).replace("\\", "/")
        current = videos.setdefault(video_id, {"video_id": video_id, "keyframe_prefix": prefix})
        if current["keyframe_prefix"] != prefix:
            raise ValueError(f"Inconsistent keyframe prefix for {video_id}")
    video_table = tuple(
        {"video_index": index, **videos[video_id]} for index, video_id in enumerate(sorted(videos))
    )
    video_indices = {item["video_id"]: item["video_index"] for item in video_table}
    for row in rows:
        row["video_index"] = video_indices[row["video_id"]]
    return CatalogData(tuple(rows), video_table)


def write_compact_catalog(data: CatalogData, index_root: Path) -> dict[str, Any]:
    index_root.mkdir(parents=True, exist_ok=True)
    rows = data.rows
    values = {
        "frame_video_index.npy": [row["video_index"] for row in rows],
        "frame_n.npy": [row["n"] for row in rows],
        "frame_original_idx.npy": [row["original_frame_idx"] for row in rows],
        "frame_pts_time.npy": [row["pts_time"] for row in rows],
        "frame_mapping_fps.npy": [row["mapping_fps"] for row in rows],
        "duplicate_group_size.npy": [row["duplicate_frame_idx_group_size"] for row in rows],
    }
    for name, dtype in CATALOG_ARRAYS.items():
        np.save(index_root / name, np.asarray(values[name], dtype=dtype), allow_pickle=False)
    (index_root / "video_table.json").write_text(
        json.dumps(list(data.video_table), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "rows": len(rows),
        "videos": len(data.video_table),
        "ordering": "video_id_then_n",
        "global_row_range": [0, len(rows) - 1] if rows else [],
        "arrays": {name: str(np.dtype(dtype)) for name, dtype in CATALOG_ARRAYS.items()},
        "keyframe_path_policy": "video_table.keyframe_prefix + / + n:03d.jpg",
    }
    (index_root / "catalog_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
