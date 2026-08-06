"""Static candidate JSON/Markdown and optional derived contact-sheet output."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from system_tai.common.schemas import KISResult
from system_tai.data.corpus_discovery import CorpusManifest
from system_tai.features.btc_clip_store import FeatureStoreRegistry

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True, slots=True)
class CandidateInspectionArtifact:
    records: tuple[dict[str, Any], ...]
    warnings: tuple[str, ...]
    json_path: Path
    markdown_path: Path
    contact_sheet_path: Path | None


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def resolve_keyframe_path(directory: Path, keyframe_order: int) -> Path | None:
    root = Path(directory)
    if not root.is_dir():
        return None
    matches: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.casefold() not in IMAGE_EXTENSIONS:
            continue
        stem = path.stem
        try:
            numeric_stem = int(stem)
        except ValueError:
            continue
        if numeric_stem == keyframe_order:
            matches.append(path.resolve(strict=False))
    if len(matches) != 1:
        return None
    return matches[0]


def _contact_sheet(
    records: tuple[dict[str, Any], ...],
    destination: Path,
) -> tuple[Path | None, str | None]:
    records_with_images = [record for record in records if record["thumbnail_path"]]
    if not records_with_images:
        return None, "contact sheet skipped because no candidate thumbnails resolved"
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return None, "contact sheet skipped because Pillow is unavailable"
    columns = 4
    cell_width = 260
    cell_height = 190
    rows = (len(records_with_images) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * cell_width, rows * cell_height), "white")
    draw = ImageDraw.Draw(canvas)
    for index, record in enumerate(records_with_images):
        x = (index % columns) * cell_width
        y = (index // columns) * cell_height
        try:
            with Image.open(record["thumbnail_path"]) as source:
                image = source.convert("RGB")
                image.thumbnail((240, 145))
                canvas.paste(image, (x + 10, y + 5))
        except OSError:
            continue
        draw.text(
            (x + 10, y + 153),
            f"#{record['rank']} {record['video_id']} frame={record['frame_id']}",
            fill="black",
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, format="JPEG", quality=82, optimize=True)
    return destination, None


def build_candidate_inspection(
    results: tuple[KISResult, ...],
    registry: FeatureStoreRegistry,
    manifest: CorpusManifest,
    output_directory: Path,
    *,
    top_n: int = 50,
    create_contact_sheet: bool = False,
) -> CandidateInspectionArtifact:
    if top_n <= 0:
        raise ValueError("inspection top_n must be positive")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    manifest_by_video = {video.video_id: video for video in manifest.videos}
    warnings: list[str] = []
    records: list[dict[str, Any]] = []
    for result in sorted(results, key=lambda item: item.query_id):
        for candidate in result.ranked_candidates:
            store = registry.get(candidate.video_id)
            mapping = store.frame_for_row(candidate.clip_row)
            if mapping.frame_id != candidate.frame_id:
                raise ValueError(
                    "candidate frame_id does not match physical mapping row: "
                    f"{candidate.video_id}/{candidate.frame_id}"
                )
            discovered = manifest_by_video[candidate.video_id]
            thumbnail = resolve_keyframe_path(
                discovered.keyframe_directory,
                mapping.keyframe_order,
            )
            if thumbnail is None:
                warnings.append(
                    f"thumbnail unresolved for {candidate.video_id} "
                    f"keyframe_order={mapping.keyframe_order}"
                )
            metadata = _json_value(candidate.diagnostic_metadata or {})
            records.append(
                {
                    "query_id": result.query_id,
                    "rank": candidate.rank,
                    "video_id": candidate.video_id,
                    "frame_id": candidate.frame_id,
                    "timestamp_seconds": mapping.pts_time,
                    "fusion_score": candidate.score,
                    "variant_hit_count": metadata.get("variant_hit_count"),
                    "best_individual_rank": metadata.get("best_individual_rank"),
                    "per_variant": metadata.get("per_variant", []),
                    "clip_row_diagnostic": candidate.clip_row,
                    "keyframe_order_diagnostic": mapping.keyframe_order,
                    "thumbnail_path": str(thumbnail) if thumbnail else None,
                }
            )
    record_tuple = tuple(records)
    json_path = output / "candidates.json"
    json_path.write_text(
        json.dumps(
            {"records": records, "warnings": sorted(set(warnings))},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    markdown_path = output / "candidate_inspection.md"
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    inspection_records = [record for record in records if record["rank"] <= top_n]
    for record in inspection_records:
        grouped[(record["query_id"], record["video_id"])].append(record)
    lines = ["# Candidate Inspection", ""]
    for (query_id, video_id), group in sorted(grouped.items()):
        lines.extend(
            [
                f"## {query_id} — {video_id}",
                "",
                "| Rank | frame_id | Timestamp | Fusion score | Variant hits | Thumbnail |",
                "|---:|---:|---:|---:|---:|---|",
            ]
        )
        for record in group:
            thumbnail_text = record["thumbnail_path"] or "unavailable"
            lines.append(
                f"| {record['rank']} | {record['frame_id']} | "
                f"{record['timestamp_seconds']:.6f} | {record['fusion_score']:.9f} | "
                f"{record['variant_hit_count']} | {thumbnail_text} |"
            )
        lines.append("")
    unique_warnings = tuple(sorted(set(warnings)))
    lines.extend(["## Warnings", ""])
    lines.extend(f"- {warning}" for warning in unique_warnings)
    if not unique_warnings:
        lines.append("- None")
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    contact_path: Path | None = None
    if create_contact_sheet:
        contact_path, contact_warning = _contact_sheet(
            tuple(inspection_records),
            output / "candidate_contact_sheet.jpg",
        )
        if contact_warning is not None:
            unique_warnings = tuple(sorted(set((*unique_warnings, contact_warning))))
            with json_path.open("r", encoding="utf-8") as stream:
                payload = json.load(stream)
            payload["warnings"] = list(unique_warnings)
            json_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            markdown_path.write_text(
                markdown_path.read_text(encoding="utf-8")
                + f"- Contact sheet: {contact_warning}\n",
                encoding="utf-8",
            )
    return CandidateInspectionArtifact(
        records=record_tuple,
        warnings=unique_warnings,
        json_path=json_path,
        markdown_path=markdown_path,
        contact_sheet_path=contact_path,
    )
