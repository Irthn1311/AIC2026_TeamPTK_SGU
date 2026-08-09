"""Stage 1C qualitative artifacts, contact sheets, review template, and ZIP policy."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

from triage_eg.retrieval.stage1.search import deduplicate_kis, group_videos
from triage_eg.retrieval.stage1b.writers import write_json, write_jsonl
from triage_eg.retrieval.stage1c.contracts import QueryRecord

REVIEW_FIELDS = (
    "query_id",
    "pair_id",
    "language",
    "category",
    "difficulty",
    "query_text",
    "rank",
    "global_row",
    "video_id",
    "n",
    "original_frame_idx",
    "score",
    "review_label",
    "review_notes",
    "failure_tags",
)
CORE_BUNDLE_MEMBERS = (
    "run_manifest.json",
    "stage1c_summary.json",
    "stage1c_report.md",
    "query_suite/query_suite.jsonl",
    "query_suite/query_suite_manifest.json",
    "pairs/pair_diagnostics.jsonl",
    "review/review_template.csv",
    "review/review_instructions.md",
    "issues.jsonl",
)
QUERY_REQUIRED = (
    "query.json",
    "ranked_frames.jsonl",
    "ranked_videos.jsonl",
    "kis_candidates.csv",
    "retrieval_diagnostics.json",
)
FORBIDDEN_SUFFIXES = {".pt", ".pth", ".bin", ".npy", ".mp4", ".avi", ".mkv"}


def grouped_video_diagnostics(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = group_videos(frames, strategy="max")
    by_row = {item["global_row"]: item for item in frames}
    output = []
    for item in grouped:
        best = by_row[item["best_global_row"]]
        output.append(
            {
                "video_rank": item["video_rank"],
                "video_id": item["video_id"],
                "best_frame_rank": best["rank"],
                "best_global_row": item["best_global_row"],
                "best_n": item["best_n"],
                "best_original_frame_idx": item["best_original_frame_idx"],
                "best_score": item["video_score"],
                "frames_in_raw_top50": item["top_frame_count"],
            }
        )
    return output


def contact_sheet_label(item: dict[str, Any], *, rank_key: str = "rank") -> str:
    return (
        f"#{item[rank_key]} {item['video_id']} n={item['n']} "
        f"frame={item['original_frame_idx']} score={item['score']:.4f}"
    )


def render_contact_sheet(
    items: list[dict[str, Any]],
    dataset_root: Path,
    output_path: Path,
    *,
    columns: int = 4,
    thumbnail_width: int = 256,
) -> list[dict[str, Any]]:
    from PIL import Image, ImageDraw

    if columns <= 0 or thumbnail_width <= 0:
        raise ValueError("Invalid contact-sheet geometry")
    tile_height, label_height = 176, 44
    rows = max(1, math.ceil(len(items) / columns))
    sheet = Image.new(
        "RGB",
        (columns * thumbnail_width, rows * (tile_height + label_height)),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    issues: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        x = (index % columns) * thumbnail_width
        y = (index // columns) * (tile_height + label_height)
        source = dataset_root / item["keyframe_relative_path"]
        try:
            with Image.open(source) as image:
                preview = image.convert("RGB")
                preview.thumbnail((thumbnail_width, tile_height))
                left = x + (thumbnail_width - preview.width) // 2
                top = y + (tile_height - preview.height) // 2
                sheet.paste(preview, (left, top))
        except (FileNotFoundError, OSError) as error:
            draw.rectangle((x, y, x + thumbnail_width - 1, y + tile_height - 1), fill="#dddddd")
            draw.text((x + 8, y + 8), "IMAGE UNAVAILABLE", fill="black")
            issues.append(
                {
                    "severity": "WARNING",
                    "code": "KEYFRAME_RESOLUTION_FAILED",
                    "query_id": item.get("query_id"),
                    "global_row": item.get("global_row"),
                    "path": item["keyframe_relative_path"],
                    "message": str(error),
                    "evidence": {},
                }
            )
        draw.text((x + 4, y + tile_height + 3), contact_sheet_label(item), fill="black")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, format="JPEG", quality=85, optimize=True)
    return issues


def write_query_artifacts(
    output_root: Path,
    query: QueryRecord,
    encoding: dict[str, Any],
    frames: list[dict[str, Any]],
    kis_frames: list[dict[str, Any]],
    diagnostics: dict[str, Any],
    dataset_root: Path,
    *,
    contact_sheet_top_k: int,
    skip_contact_sheets: bool,
) -> tuple[dict[str, Path], list[dict[str, Any]]]:
    root = output_root / "queries" / query.query_id
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "query.json", {**query.__dict__, "encoding": encoding})
    write_jsonl(root / "ranked_frames.jsonl", frames)
    videos = grouped_video_diagnostics(frames)
    write_jsonl(root / "ranked_videos.jsonl", videos)
    try:
        kis, _ = deduplicate_kis(kis_frames, max_predictions=len(kis_frames))
        with (root / "kis_candidates.csv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=["video_id", "frame_id"])
            writer.writeheader()
            writer.writerows(kis)
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise RuntimeError(f"KIS_EXPORT_FAILED: {error}") from error
    write_json(root / "retrieval_diagnostics.json", diagnostics)
    issues: list[dict[str, Any]] = []
    if not skip_contact_sheets:
        try:
            issues.extend(
                render_contact_sheet(
                    frames[:contact_sheet_top_k],
                    dataset_root,
                    root / "contact_sheet_top20.jpg",
                )
            )
            representatives = []
            by_row = {item["global_row"]: item for item in frames}
            for video in videos[:12]:
                representative = dict(by_row[video["best_global_row"]])
                representative["rank"] = video["video_rank"]
                representatives.append(representative)
            issues.extend(
                render_contact_sheet(
                    representatives,
                    dataset_root,
                    root / "contact_sheet_top12_videos.jpg",
                    columns=4,
                )
            )
        except (ImportError, OSError, RuntimeError, ValueError) as error:
            issues.append(
                {
                    "severity": "WARNING",
                    "code": "CONTACT_SHEET_RENDER_FAILED",
                    "query_id": query.query_id,
                    "global_row": None,
                    "path": str(root),
                    "message": str(error),
                    "evidence": {},
                }
            )
    return (
        {
            "query": root / "query.json",
            "ranked_frames": root / "ranked_frames.jsonl",
            "ranked_videos": root / "ranked_videos.jsonl",
            "kis_candidates": root / "kis_candidates.csv",
            "retrieval_diagnostics": root / "retrieval_diagnostics.json",
        },
        issues,
    )


def write_review_template(
    path: Path,
    queries: list[QueryRecord],
    frames_by_query: dict[str, list[dict[str, Any]]],
    review_top_k: int,
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(REVIEW_FIELDS))
        writer.writeheader()
        for query in queries:
            for item in frames_by_query[query.query_id][:review_top_k]:
                writer.writerow(
                    {
                        "query_id": query.query_id,
                        "pair_id": query.pair_id,
                        "language": query.language,
                        "category": query.category,
                        "difficulty": query.difficulty,
                        "query_text": query.text,
                        "rank": item["rank"],
                        "global_row": item["global_row"],
                        "video_id": item["video_id"],
                        "n": item["n"],
                        "original_frame_idx": item["original_frame_idx"],
                        "score": item["score"],
                        "review_label": "",
                        "review_notes": "",
                        "failure_tags": "",
                    }
                )
                count += 1
    return count


def review_instructions() -> str:
    return """# Stage 1C Human Review Instructions

Review only the visible frame against the query text. Scores are diagnostics, not labels.

- `RELEVANT`: clearly satisfies the main semantic intent.
- `PARTIAL`: satisfies part of the query but misses or contradicts an important component.
- `IRRELEVANT`: does not satisfy the main semantic intent.
- `UNCERTAIN`: the frame alone is insufficient to decide; this is reported separately.

Do not edit identity columns. Fill `review_label`, optional `review_notes`, and optional
semicolon-separated `failure_tags`. Do not infer competition Recall@K from this review.
"""


def create_stage1c_bundle(root: str | Path, zip_path: str | Path) -> Path:
    source = Path(root).expanduser().resolve(strict=True)
    target = Path(zip_path).expanduser().resolve(strict=False)
    if source in target.parents:
        raise ValueError("Stage 1C ZIP must be outside the output root")
    missing = [name for name in CORE_BUNDLE_MEMBERS if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Stage 1C bundle artifacts: {missing}")
    members = list(CORE_BUNDLE_MEMBERS)
    for optional in ("review/review_metrics.json", "review/review_metrics.md"):
        if (source / optional).is_file():
            members.append(optional)
    query_roots = sorted(path for path in (source / "queries").iterdir() if path.is_dir())
    for query_root in query_roots:
        for name in QUERY_REQUIRED:
            path = query_root / name
            if not path.is_file():
                raise FileNotFoundError(f"Missing Stage 1C query artifact: {path}")
            members.append(path.relative_to(source).as_posix())
        members.extend(
            path.relative_to(source).as_posix()
            for path in sorted(query_root.glob("contact_sheet_*.jpg"))
        )
    if any(
        Path(name).suffix.lower() in FORBIDDEN_SUFFIXES
        or name.startswith(("logs/", "cache/", "caches/"))
        for name in members
    ):
        raise ValueError("Stage 1C ZIP allowlist contains a forbidden artifact")
    target.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(target, "w", compression=ZIP_DEFLATED) as archive:
        for name in members:
            archive.write(source / name, arcname=name)
    return target
