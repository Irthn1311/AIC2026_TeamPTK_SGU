from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from .mapping_loader import load_keyframe_mapping


@dataclass
class InventoryRow:
    video_id: str
    feature_path: str
    feature_shape: str
    feature_dtype: str
    embedding_dim: int
    num_features: int
    mapping_path: str
    num_mapping_rows: int
    keyframe_directory: str
    num_keyframes: int
    video_path: str
    fps: float | None
    duration_seconds: float | None
    ocr_status: str
    status: str
    notes: str


def _tree_lines(root: Path, max_depth: int) -> list[str]:
    lines: list[str] = []
    root = root.resolve()

    skip_names = {
        ".git",
        ".conda",
        ".conda_pkgs",
        ".cache",
        ".ocr_cache",
        ".pip_cache",
        "__pycache__",
        "outputs",
        "data",
    }

    def walk(path: Path, depth: int) -> None:
        if depth > max_depth:
            return
        try:
            children = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except Exception:
            return
        for child in children:
            if child.name in skip_names:
                continue
            indent = "  " * depth
            suffix = "/" if child.is_dir() else ""
            lines.append(f"{indent}{child.name}{suffix}")
            if child.is_dir():
                walk(child, depth + 1)

    lines.append(f"{root.name}/")
    walk(root, 1)
    return lines


def _count_files(path: Path, suffixes: tuple[str, ...]) -> int:
    return sum(1 for p in path.rglob("*") if p.is_file() and p.suffix.lower() in suffixes)


def _ocr_status_for_video(video_id: str, project_root: Path) -> str:
    candidates = [
        project_root / "data" / "interim" / "l21_ocr" / "per_video" / f"{video_id}.jsonl",
        project_root / "outputs" / "ocr" / f"{video_id}.jsonl",
        project_root / "outputs" / "ocr",
    ]
    for cand in candidates:
        if cand.is_file():
            return "existing_video_ocr"
        if cand.is_dir() and any(cand.glob(f"{video_id}*.jsonl")):
            return "existing_video_ocr"
    return "none_found"


def inspect_l21_dataset(dataset_root: str | Path, project_root: str | Path, output_dir: str | Path, max_tree_depth_dataset: int = 5, max_tree_depth_project: int = 4) -> tuple[pd.DataFrame, dict]:
    dataset_root = Path(dataset_root)
    project_root = Path(project_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_root = dataset_root / "clip-features-32-aic25-b1" / "clip-features-32"
    mapping_root = dataset_root / "map-keyframes-aic25-b1" / "map-keyframes"
    keyframe_root = dataset_root / "Keyframes_L21" / "keyframes"
    video_root = dataset_root / "Videos_L21_a" / "video"
    media_root = dataset_root / "media-info-aic25-b1" / "media-info"

    rows: list[InventoryRow] = []
    excluded: list[dict] = []

    feature_files = sorted(feature_root.glob("L21_V*.npy"))
    for feature_path in feature_files:
        video_id = feature_path.stem
        mapping_path = mapping_root / f"{video_id}.csv"
        media_path = media_root / f"{video_id}.json"
        video_path = video_root / f"{video_id}.mp4"
        keyframe_dir = keyframe_root / video_id
        status = "ok"
        notes = ""
        fps = None
        duration_seconds = None
        try:
            import numpy as np

            arr = np.load(feature_path, mmap_mode="r")
            feature_shape = str(tuple(arr.shape))
            feature_dtype = str(arr.dtype)
            embedding_dim = int(arr.shape[1]) if arr.ndim == 2 else -1
            num_features = int(arr.shape[0]) if arr.ndim == 2 else 0
        except Exception as exc:
            feature_shape = "error"
            feature_dtype = "error"
            embedding_dim = -1
            num_features = 0
            status = "feature_load_error"
            notes = str(exc)

        num_mapping_rows = 0
        if mapping_path.exists():
            try:
                mapping = load_keyframe_mapping(mapping_path, keyframe_root)
                num_mapping_rows = len(mapping)
            except Exception as exc:
                status = "mapping_error"
                notes = f"{notes}; {exc}".strip("; ")
        else:
            status = "missing_mapping"
            notes = f"{notes}; missing mapping".strip("; ")

        if media_path.exists():
            try:
                media = json.loads(media_path.read_text(encoding="utf-8"))
                fps = media.get("fps")
                duration_seconds = media.get("duration_seconds") or media.get("duration") or media.get("length")
            except Exception as exc:
                notes = f"{notes}; media error: {exc}".strip("; ")
        else:
            notes = f"{notes}; missing media".strip("; ")

        num_keyframes = _count_files(keyframe_dir, (".jpg", ".jpeg", ".png")) if keyframe_dir.exists() else 0
        ocr_status = _ocr_status_for_video(video_id, project_root)
        if num_features != num_mapping_rows:
            status = "mismatch"
            notes = f"{notes}; feature_rows={num_features} mapping_rows={num_mapping_rows}".strip("; ")

        rows.append(
            InventoryRow(
                video_id=video_id,
                feature_path=str(feature_path),
                feature_shape=feature_shape,
                feature_dtype=feature_dtype,
                embedding_dim=embedding_dim,
                num_features=num_features,
                mapping_path=str(mapping_path),
                num_mapping_rows=num_mapping_rows,
                keyframe_directory=str(keyframe_dir),
                num_keyframes=num_keyframes,
                video_path=str(video_path),
                fps=fps,
                duration_seconds=duration_seconds,
                ocr_status=ocr_status,
                status=status,
                notes=notes,
            )
        )

    df = pd.DataFrame([asdict(r) for r in rows])
    if not df.empty:
        df = df.sort_values("video_id").reset_index(drop=True)

    folder_tree = []
    folder_tree.append(f"PROJECT_ROOT: {project_root}")
    folder_tree.extend(_tree_lines(project_root, max_tree_depth_project))
    folder_tree.append("")
    folder_tree.append(f"DATASET_ROOT: {dataset_root}")
    folder_tree.extend(_tree_lines(dataset_root, max_tree_depth_dataset))

    summary = {
        "dataset_root": str(dataset_root),
        "project_root": str(project_root),
        "num_videos": int(len(df)),
        "total_keyframes": int(df["num_keyframes"].sum()) if not df.empty else 0,
        "total_features": int(df["num_features"].sum()) if not df.empty else 0,
        "feature_dim": int(df["embedding_dim"].mode().iloc[0]) if not df.empty else None,
        "videos_with_ocr": int((df["ocr_status"] != "none_found").sum()) if not df.empty else 0,
        "feature_files": [str(p) for p in feature_files],
    }
    return df, {"summary": summary, "folder_tree": "\n".join(folder_tree)}
