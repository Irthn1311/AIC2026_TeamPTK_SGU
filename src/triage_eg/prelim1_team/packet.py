"""Canonical coordinates, embeddings, contact sheets, and packet validation."""

from __future__ import annotations

import csv
import json
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from triage_eg.retrieval.stage1.search import CompactCatalog

SOURCE_SYSTEM = "MY_PRELIM1_R5_SAFE_TEAM"


class CatalogResolver:
    """Resolve arbitrary evidence coordinates to the nearest frozen BTC catalog row."""

    def __init__(self, stage1_root: str | Path) -> None:
        root = Path(stage1_root).expanduser().resolve(strict=True)
        self.index_root = root / "index" if (root / "index").is_dir() else root
        self.catalog = CompactCatalog(self.index_root)
        self._rows_by_video: dict[str, np.ndarray] = {}
        grouped: dict[str, list[int]] = defaultdict(list)
        for global_row, video_index in enumerate(np.asarray(self.catalog.video_index)):
            video_id = str(self.catalog.video_table[int(video_index)]["video_id"])
            grouped[video_id].append(global_row)
        for video_id, rows in grouped.items():
            self._rows_by_video[video_id] = np.asarray(rows, dtype=np.int64)

    def nearest_row(self, video_id: str, frame_id: int) -> int:
        rows = self._rows_by_video.get(str(video_id))
        if rows is None or not len(rows):
            raise KeyError(f"PRELIM1_UNKNOWN_VIDEO:{video_id}")
        frames = np.asarray(self.catalog.original_idx[rows], dtype=np.int64)
        distance = np.abs(frames - int(frame_id))
        best = np.flatnonzero(distance == distance.min())
        return int(rows[int(best[0])])

    def map_coordinate(self, video_id: str, frame_id: int) -> dict[str, Any]:
        row = self.nearest_row(video_id, frame_id)
        mapped = self.catalog.map_row(row)
        return {
            **mapped,
            "requested_frame_id": int(frame_id),
            "canonical_frame_delta": int(mapped["original_frame_idx"]) - int(frame_id),
        }

    def nearest_time_row(self, video_id: str, seconds: float) -> int:
        rows = self._rows_by_video.get(str(video_id))
        if rows is None or not len(rows):
            raise KeyError(f"PRELIM1_UNKNOWN_VIDEO:{video_id}")
        times = np.asarray(self.catalog.pts_time[rows], dtype=np.float64)
        distance = np.abs(times - float(seconds))
        return int(rows[int(np.flatnonzero(distance == distance.min())[0])])

    def image_path(self, dataset_root: str | Path, global_row: int) -> Path:
        relative = Path(self.catalog.map_row(global_row)["keyframe_relative_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError("PRELIM1_KEYFRAME_PATH_TRAVERSAL")
        path = Path(dataset_root).expanduser().resolve(strict=True) / relative
        return path.resolve(strict=True)

    def nearby_rows(self, global_row: int, seconds: tuple[float, ...]) -> list[int]:
        row = self.catalog.map_row(global_row)
        rows = self._rows_by_video[str(row["video_id"])]
        times = np.asarray(self.catalog.pts_time[rows], dtype=np.float64)
        target = float(row["pts_time"])
        output = []
        for offset in seconds:
            distances = np.abs(times - (target + float(offset)))
            output.append(int(rows[int(np.flatnonzero(distances == distances.min())[0])]))
        return output


def _unit(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(value))
    if not np.isfinite(value).all() or not np.isfinite(norm) or norm <= 0:
        raise RuntimeError("PRELIM1_EMBEDDING_INVALID")
    return value / norm


def export_candidate_embeddings(
    candidates: list[dict[str, Any]],
    resolver: CatalogResolver,
    siglip_index_root: str | Path,
    output_npz: str | Path,
    output_index_csv: str | Path,
) -> dict[str, Any]:
    """Export normalized A0/S1 vectors for every public candidate coordinate."""

    a0 = np.load(resolver.index_root / "clip_vectors.f16.npy", mmap_mode="r", allow_pickle=False)
    s1_root = Path(siglip_index_root).expanduser().resolve(strict=True)
    s1_index = s1_root / "index" if (s1_root / "index").is_dir() else s1_root
    s1 = np.load(s1_index / "siglip2_vectors.f16.npy", mmap_mode="r", allow_pickle=False)
    if len(a0) != len(s1) or len(a0) != len(resolver.catalog.n):
        raise RuntimeError("PRELIM1_A0_S1_CATALOG_ALIGNMENT_FAILED")
    arrays: dict[str, np.ndarray] = {}
    index_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        query_id = str(candidate["query_id"])
        task = str(candidate["task_type"]).upper()
        rank = int(candidate["candidate_rank"])
        video_id = str(candidate["video_id"])
        frames = (
            [int(value) for value in candidate["frame_ids"]]
            if task == "TRAKE"
            else [int(candidate["frame_id"])]
        )
        event_a0, event_s1 = [], []
        for event_index, frame_id in enumerate(frames):
            global_row = resolver.nearest_row(video_id, frame_id)
            suffix = f"_e{event_index + 1}" if task == "TRAKE" else ""
            base = f"{query_id}__r{rank}{suffix}"
            key_a0, key_s1 = f"a0__{base}", f"s1__{base}"
            arrays[key_a0] = _unit(a0[global_row])
            arrays[key_s1] = _unit(s1[global_row])
            event_a0.append(arrays[key_a0])
            event_s1.append(arrays[key_s1])
            index_rows.append(
                {
                    "query_id": query_id,
                    "task_type": task,
                    "candidate_rank": rank,
                    "video_id": video_id,
                    "frame_id": frame_id,
                    "chain_id": f"{query_id}:rank{rank}" if task == "TRAKE" else "",
                    "event_index": event_index if task == "TRAKE" else "",
                    "embedding_key_a0": key_a0,
                    "embedding_key_s1": key_s1,
                    "source_system": SOURCE_SYSTEM,
                }
            )
        if task == "TRAKE":
            base = f"{query_id}__r{rank}__chain_mean"
            key_a0, key_s1 = f"a0__{base}", f"s1__{base}"
            arrays[key_a0] = _unit(np.mean(np.stack(event_a0), axis=0))
            arrays[key_s1] = _unit(np.mean(np.stack(event_s1), axis=0))
            index_rows.append(
                {
                    "query_id": query_id,
                    "task_type": task,
                    "candidate_rank": rank,
                    "video_id": video_id,
                    "frame_id": "",
                    "chain_id": f"{query_id}:rank{rank}",
                    "event_index": "CHAIN_MEAN",
                    "embedding_key_a0": key_a0,
                    "embedding_key_s1": key_s1,
                    "source_system": SOURCE_SYSTEM,
                }
            )
    npz = Path(output_npz)
    npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz, **arrays)
    csv_path = Path(output_index_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(index_rows[0]) if index_rows else []
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(index_rows)
    return {
        "candidate_coordinate_count": sum(
            len(row.get("frame_ids", [row.get("frame_id")])) for row in candidates
        ),
        "embedding_array_count": len(arrays),
        "embedding_index_row_count": len(index_rows),
        "a0_dimension": int(a0.shape[1]),
        "s1_dimension": int(s1.shape[1]),
    }


def _font(size: int):
    from PIL import ImageFont

    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ):
        if Path(path).is_file():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def write_contact_sheet(
    path: str | Path,
    title: str,
    panels: list[tuple[Path, str]],
    *,
    panel_size: tuple[int, int] = (320, 200),
) -> Path:
    from PIL import Image, ImageDraw, ImageOps

    if not panels:
        raise ValueError("PRELIM1_CONTACT_SHEET_REQUIRES_PANELS")
    width = panel_size[0] * len(panels)
    title_lines = textwrap.wrap(" ".join(title.split()), width=max(45, width // 13))[:4]
    header_height = 38 + 25 * len(title_lines)
    canvas = Image.new("RGB", (width, header_height + panel_size[1] + 48), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 8), "\n".join(title_lines), fill="black", font=_font(18), spacing=4)
    for index, (image_path, label) in enumerate(panels):
        with Image.open(image_path) as source:
            image = ImageOps.fit(source.convert("RGB"), panel_size)
        x = index * panel_size[0]
        canvas.paste(image, (x, header_height))
        draw.rectangle(
            (x, header_height, x + panel_size[0] - 1, header_height + panel_size[1]),
            outline="black",
        )
        draw.text(
            (x + 6, header_height + panel_size[1] + 8),
            label[:52],
            fill="black",
            font=_font(14),
        )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target, quality=90)
    return target


def validate_team_packet(
    output_root: str | Path,
    manifest: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    root = Path(output_root).resolve(strict=True)
    queries = list(manifest["queries"])
    query_ids = {str(query["query_id"]) for query in queries}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        grouped[str(row["query_id"])].append(row)
    issues = []
    if set(grouped) != query_ids:
        issues.append("CANDIDATE_QUERY_ID_SET_MISMATCH")
    for query in queries:
        rows = sorted(
            grouped.get(str(query["query_id"]), []), key=lambda row: int(row["candidate_rank"])
        )
        if len(rows) != 5 or [int(row["candidate_rank"]) for row in rows] != list(range(1, 6)):
            issues.append(f"TOP5_INVALID:{query['query_id']}")
        if sum(bool(row.get("primary_candidate")) for row in rows) != 1:
            issues.append(f"PRIMARY_CARDINALITY:{query['query_id']}")
        if str(query["task"]) == "TRAKE":
            for row in rows:
                frames = [int(value) for value in row.get("frame_ids", [])]
                if len(frames) != int(query["event_count"]) or any(
                    left >= right for left, right in zip(frames, frames[1:], strict=False)
                ):
                    issues.append(f"TRAKE_STRUCTURE:{query['query_id']}:{row['candidate_rank']}")
    required = {
        "MY_PRELIM1_RESULTS.md",
        "my_prelim1_top5.csv",
        "my_prelim1_top5.json",
        "my_prelim1_primary.csv",
        "query_manifest.json",
        "query_view_provenance.jsonl",
        "candidate_evidence.jsonl",
        "team_candidate_embeddings.npz",
        "team_candidate_embedding_index.csv",
        "qa_evidence_review.csv",
        "trake_top5.json",
        "asset_status.json",
        "run_provenance.json",
    }
    missing = sorted(name for name in required if not (root / name).is_file())
    if missing:
        issues.append(f"MISSING_FILES:{missing}")
    for query in queries:
        query_id, task = str(query["query_id"]), str(query["task"])
        if task in {"KIS", "QA"}:
            target = root / "review" / task / f"{query_id}.jpg"
            if not target.is_file():
                issues.append(f"CONTACT_SHEET_MISSING:{query_id}")
        else:
            for rank in range(1, 6):
                target = root / "review" / "TRAKE" / f"{query_id}_rank{rank}.jpg"
                if not target.is_file():
                    issues.append(f"CONTACT_SHEET_MISSING:{query_id}:rank{rank}")
    forbidden = [
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and ("gt" in path.name.casefold() or "sealed" in path.name.casefold())
    ]
    if forbidden:
        issues.append(f"FORBIDDEN_ARTIFACT:{forbidden}")
    if (root / "team_candidate_embeddings.npz").is_file() and (
        root / "team_candidate_embedding_index.csv"
    ).is_file():
        with np.load(root / "team_candidate_embeddings.npz") as archive:
            keys = set(archive.files)
        with (root / "team_candidate_embedding_index.csv").open(
            encoding="utf-8-sig", newline=""
        ) as stream:
            index_rows = list(csv.DictReader(stream))
        referenced = {
            row[key]
            for row in index_rows
            for key in ("embedding_key_a0", "embedding_key_s1")
            if row.get(key)
        }
        if referenced != keys:
            issues.append("EMBEDDING_INDEX_KEY_SET_MISMATCH")
    return {
        "status": "PASS" if not issues else "FAIL",
        "query_count": len(queries),
        "candidate_count": len(candidates),
        "contact_sheet_count": len(list((root / "review").rglob("*.jpg"))),
        "issues": issues,
        "ground_truth_opened": False,
        "submission_uploaded": False,
    }


def write_json(path: str | Path, value: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return target


__all__ = [
    "SOURCE_SYSTEM",
    "CatalogResolver",
    "export_candidate_embeddings",
    "validate_team_packet",
    "write_contact_sheet",
    "write_json",
]
