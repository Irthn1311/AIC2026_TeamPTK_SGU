"""Stage 1 exact search, catalog mapping, grouping, and candidate export."""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from time import monotonic
from typing import Any

import numpy as np

from triage_eg.retrieval.numpy_index import NumPyMemmapExactIndex
from triage_eg.retrieval.stage1.contracts import SearchConfig


class CompactCatalog:
    def __init__(self, index_root: Path) -> None:
        self.video_table = json.loads((index_root / "video_table.json").read_text(encoding="utf-8"))
        self.video_index = np.load(
            index_root / "frame_video_index.npy", mmap_mode="r", allow_pickle=False
        )
        self.n = np.load(index_root / "frame_n.npy", mmap_mode="r", allow_pickle=False)
        self.original_idx = np.load(
            index_root / "frame_original_idx.npy", mmap_mode="r", allow_pickle=False
        )
        self.pts_time = np.load(
            index_root / "frame_pts_time.npy", mmap_mode="r", allow_pickle=False
        )
        self.mapping_fps = np.load(
            index_root / "frame_mapping_fps.npy", mmap_mode="r", allow_pickle=False
        )
        self.duplicate_size = np.load(
            index_root / "duplicate_group_size.npy", mmap_mode="r", allow_pickle=False
        )
        lengths = {
            len(self.video_index),
            len(self.n),
            len(self.original_idx),
            len(self.pts_time),
            len(self.mapping_fps),
            len(self.duplicate_size),
        }
        if len(lengths) != 1:
            raise ValueError("Compact catalog arrays have inconsistent lengths")

    def map_row(self, global_row: int) -> dict[str, Any]:
        if not 0 <= global_row < len(self.n):
            raise IndexError(global_row)
        video = self.video_table[int(self.video_index[global_row])]
        n = int(self.n[global_row])
        return {
            "global_row": global_row,
            "video_id": video["video_id"],
            "n": n,
            "clip_row_index": n - 1,
            "pts_time": float(self.pts_time[global_row]),
            "mapping_fps": float(self.mapping_fps[global_row]),
            "original_frame_idx": int(self.original_idx[global_row]),
            "keyframe_relative_path": f"{video['keyframe_prefix']}/{n:03d}.jpg",
            "duplicate_frame_idx_group_size": int(self.duplicate_size[global_row]),
        }


def load_search_backend(config: SearchConfig) -> tuple[NumPyMemmapExactIndex, CompactCatalog]:
    index_root = config.stage1_root / "index"
    manifest = json.loads((index_root / "index_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "COMPLETE":
        raise ValueError("Stage 1 index is not COMPLETE")
    vectors = np.load(index_root / "clip_vectors.f16.npy", mmap_mode="r", allow_pickle=False)
    norms = np.load(index_root / "vector_norms.f32.npy", mmap_mode="r", allow_pickle=False)
    catalog = CompactCatalog(index_root)
    if len(vectors) != len(catalog.n) or tuple(vectors.shape) != (
        manifest["vector_count"],
        manifest["dimension"],
    ):
        raise ValueError("Index matrix/catalog/manifest mismatch")
    return NumPyMemmapExactIndex(
        vectors, norms, metric=config.metric, chunk_rows=config.search_chunk_rows
    ), catalog


def rank_query(
    query: np.ndarray, config: SearchConfig, *, encoder_status: str = "NOT_APPLICABLE_VECTOR_QUERY"
) -> tuple[list[dict[str, Any]], float]:
    backend, catalog = load_search_backend(config)
    started = monotonic()
    scores, rows = backend.search(query, config.top_k)
    if scores.shape[0] != 1:
        raise ValueError("Stage 1 query runner accepts exactly one query vector per run")
    latency = monotonic() - started
    candidates = []
    for rank, (score, global_row) in enumerate(zip(scores[0], rows[0], strict=True), start=1):
        candidates.append(
            {
                "rank": rank,
                **catalog.map_row(int(global_row)),
                "score": float(score),
                "metric": config.metric,
                "encoder_contract_status": encoder_status,
                "query_id": config.query_id,
                "issues": [],
            }
        )
    return candidates, latency


def group_videos(
    candidates: list[dict[str, Any]], *, strategy: str = "max", mean_top_k: int = 3
) -> list[dict[str, Any]]:
    if strategy not in {"max", "mean_top_k"} or mean_top_k <= 0:
        raise ValueError("Invalid video grouping configuration")
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate["video_id"]].append(candidate)
    output = []
    for video_id, items in grouped.items():
        ordered = sorted(items, key=lambda item: (-item["score"], item["global_row"]))
        selected = ordered[:mean_top_k]
        score = (
            ordered[0]["score"]
            if strategy == "max"
            else sum(item["score"] for item in selected) / len(selected)
        )
        best = ordered[0]
        output.append(
            {
                "video_id": video_id,
                "video_score": score,
                "best_global_row": best["global_row"],
                "best_n": best["n"],
                "best_original_frame_idx": best["original_frame_idx"],
                "top_frame_count": len(items),
            }
        )
    ordered_output = sorted(output, key=lambda item: (-item["video_score"], item["video_id"]))
    return [{"video_rank": rank, **item} for rank, item in enumerate(ordered_output, start=1)]


def deduplicate_kis(
    candidates: list[dict[str, Any]], max_predictions: int
) -> tuple[list[dict[str, Any]], dict[tuple[str, int], list[int]]]:
    if max_predictions <= 0:
        raise ValueError("max_predictions must be positive")
    supporting: defaultdict[tuple[str, int], list[int]] = defaultdict(list)
    best: dict[tuple[str, int], dict[str, Any]] = {}
    for item in candidates:
        key = (item["video_id"], item["original_frame_idx"])
        supporting[key].append(item["n"])
        current = best.get(key)
        if current is None or (-item["score"], item["global_row"]) < (
            -current["score"],
            current["global_row"],
        ):
            best[key] = item
    ordered = sorted(best.values(), key=lambda item: (-item["score"], item["global_row"]))[
        :max_predictions
    ]
    return [
        {"video_id": item["video_id"], "frame_id": item["original_frame_idx"]} for item in ordered
    ], dict(supporting)


def write_query_outputs(
    root: Path,
    config: SearchConfig,
    candidates: list[dict[str, Any]],
    *,
    search_latency_seconds: float,
    encoder_status: str,
    query_directory_name: str = "queries",
) -> dict[str, Path]:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", query_directory_name):
        raise ValueError("query_directory_name must be a safe path component")
    query_root = root / query_directory_name / config.query_id
    query_root.mkdir(parents=True, exist_ok=True)
    frames_path = query_root / "ranked_frames.jsonl"
    videos_path = query_root / "ranked_videos.jsonl"
    csv_path = query_root / "kis_candidates.csv"
    run_path = query_root / "search_run.json"
    frames_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in candidates),
        encoding="utf-8",
    )
    videos = group_videos(candidates, strategy=config.video_grouping)
    videos_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in videos), encoding="utf-8"
    )
    kis, supporting = deduplicate_kis(candidates, config.max_predictions)
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["video_id", "frame_id"])
        if config.csv_header:
            writer.writeheader()
        writer.writerows(kis)
    run_path.write_text(
        json.dumps(
            {
                "query_id": config.query_id,
                "metric": config.metric,
                "top_k": config.top_k,
                "search_latency_seconds": search_latency_seconds,
                "encoder_contract_status": encoder_status,
                "kis_candidate_count": len(kis),
                "supporting_ordinals": {
                    f"{video_id}:{frame_id}": values
                    for (video_id, frame_id), values in sorted(supporting.items())
                    if len(values) > 1
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "search_run": run_path,
        "ranked_frames": frames_path,
        "ranked_videos": videos_path,
        "kis_candidates": csv_path,
    }
