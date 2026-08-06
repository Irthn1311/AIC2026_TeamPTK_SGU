"""Fast static candidate inspection with bounded thumbnail indexing."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any

from system_tai.common.schemas import KISResult
from system_tai.data.corpus_discovery import CorpusManifest
from system_tai.features.btc_clip_store import FeatureStoreRegistry

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


class InspectionMode(StrEnum):
    NONE = "none"
    TOP_N = "top-n"
    ALL = "all"


@dataclass(frozen=True, slots=True)
class ThumbnailResolution:
    path: Path | None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ThumbnailResolverStats:
    directory_scan_count: int = 0
    resolve_count: int = 0
    index_seconds: float = 0.0
    resolve_seconds: float = 0.0


@dataclass(frozen=True, slots=True)
class KeyframeThumbnailIndex:
    directory: Path
    by_keyframe_order: Mapping[int, Path]
    ambiguous_orders: frozenset[int]
    warnings: tuple[str, ...]

    @classmethod
    def build(cls, directory: Path) -> KeyframeThumbnailIndex:
        root = Path(directory).resolve(strict=False)
        if not root.is_dir():
            return cls(
                directory=root,
                by_keyframe_order=MappingProxyType({}),
                ambiguous_orders=frozenset(),
                warnings=(f"keyframe directory unavailable: {root}",),
            )
        matches: dict[int, list[Path]] = defaultdict(list)
        paths = sorted(root.rglob("*"), key=lambda path: str(path).casefold())
        for path in paths:
            if not path.is_file() or path.suffix.casefold() not in IMAGE_EXTENSIONS:
                continue
            try:
                keyframe_order = int(path.stem)
            except ValueError:
                continue
            matches[keyframe_order].append(path.resolve(strict=False))
        ambiguous = frozenset(
            keyframe_order
            for keyframe_order, candidates in matches.items()
            if len(candidates) > 1
        )
        by_order = {
            keyframe_order: candidates[0]
            for keyframe_order, candidates in matches.items()
            if len(candidates) == 1
        }
        warnings = tuple(
            f"ambiguous thumbnail keyframe_order={keyframe_order} in {root}"
            for keyframe_order in sorted(ambiguous)
        )
        return cls(
            directory=root,
            by_keyframe_order=MappingProxyType(by_order),
            ambiguous_orders=ambiguous,
            warnings=warnings,
        )


class ThumbnailResolver:
    """Lazy per-directory Path-only cache; image objects are never retained."""

    def __init__(self, *, clock: Callable[[], float] = time.perf_counter) -> None:
        self._clock = clock
        self._indexes: dict[Path, KeyframeThumbnailIndex] = {}
        self._directory_scan_count = 0
        self._resolve_count = 0
        self._index_seconds = 0.0
        self._resolve_seconds = 0.0

    @property
    def stats(self) -> ThumbnailResolverStats:
        return ThumbnailResolverStats(
            directory_scan_count=self._directory_scan_count,
            resolve_count=self._resolve_count,
            index_seconds=self._index_seconds,
            resolve_seconds=self._resolve_seconds,
        )

    def resolve(self, directory: Path, keyframe_order: int) -> ThumbnailResolution:
        self._resolve_count += 1
        root = Path(directory).resolve(strict=False)
        index = self._indexes.get(root)
        if index is None:
            index_start = self._clock()
            index = KeyframeThumbnailIndex.build(root)
            self._index_seconds += self._clock() - index_start
            self._indexes[root] = index
            self._directory_scan_count += 1
        resolve_start = self._clock()
        path = index.by_keyframe_order.get(keyframe_order)
        warnings = list(index.warnings)
        if path is None and keyframe_order not in index.ambiguous_orders:
            warnings.append(
                f"thumbnail missing in {root} for keyframe_order={keyframe_order}"
            )
        self._resolve_seconds += self._clock() - resolve_start
        return ThumbnailResolution(path=path, warnings=tuple(warnings))


@dataclass(frozen=True, slots=True)
class CandidateInspectionTimings:
    candidate_json_seconds: float = 0.0
    thumbnail_index_seconds: float = 0.0
    thumbnail_resolve_seconds: float = 0.0
    markdown_seconds: float = 0.0
    contact_sheet_seconds: float = 0.0


@dataclass(frozen=True, slots=True)
class PreparedCandidateInspection:
    records: tuple[dict[str, Any], ...]
    warnings: tuple[str, ...]
    mode: InspectionMode
    top_n: int
    timings: CandidateInspectionTimings


@dataclass(frozen=True, slots=True)
class CandidateInspectionArtifact:
    records: tuple[dict[str, Any], ...]
    warnings: tuple[str, ...]
    json_path: Path
    markdown_path: Path
    contact_sheet_path: Path | None
    timings: CandidateInspectionTimings


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def resolve_keyframe_path(directory: Path, keyframe_order: int) -> Path | None:
    """Backward-compatible one-off resolver; run pipelines should share a resolver."""

    return ThumbnailResolver().resolve(directory, keyframe_order).path


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


def _stats_delta(
    before: ThumbnailResolverStats,
    after: ThumbnailResolverStats,
) -> CandidateInspectionTimings:
    return CandidateInspectionTimings(
        thumbnail_index_seconds=after.index_seconds - before.index_seconds,
        thumbnail_resolve_seconds=after.resolve_seconds - before.resolve_seconds,
    )


def prepare_candidate_inspection(
    results: tuple[KISResult, ...],
    registry: FeatureStoreRegistry,
    manifest: CorpusManifest,
    *,
    mode: InspectionMode = InspectionMode.TOP_N,
    top_n: int = 50,
    thumbnail_resolver: ThumbnailResolver | None = None,
) -> PreparedCandidateInspection:
    if top_n <= 0:
        raise ValueError("inspection top_n must be positive")
    resolver = thumbnail_resolver or ThumbnailResolver()
    before = resolver.stats
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
            should_resolve = mode is InspectionMode.ALL or (
                mode is InspectionMode.TOP_N and candidate.rank <= top_n
            )
            thumbnail: Path | None = None
            if should_resolve:
                discovered = manifest_by_video[candidate.video_id]
                resolution = resolver.resolve(
                    discovered.keyframe_directory,
                    mapping.keyframe_order,
                )
                thumbnail = resolution.path
                warnings.extend(resolution.warnings)
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
    return PreparedCandidateInspection(
        records=tuple(records),
        warnings=tuple(sorted(set(warnings))),
        mode=mode,
        top_n=top_n,
        timings=_stats_delta(before, resolver.stats),
    )


def combine_prepared_inspections(
    prepared: Sequence[PreparedCandidateInspection],
) -> PreparedCandidateInspection:
    if not prepared:
        raise ValueError("at least one prepared inspection is required")
    mode = prepared[0].mode
    top_n = prepared[0].top_n
    if any(item.mode is not mode or item.top_n != top_n for item in prepared):
        raise ValueError("prepared inspections must use the same mode and top_n")
    records = tuple(
        record
        for item in sorted(
            prepared,
            key=lambda inspection: inspection.records[0]["query_id"]
            if inspection.records
            else "",
        )
        for record in item.records
    )
    return PreparedCandidateInspection(
        records=records,
        warnings=tuple(
            sorted({warning for item in prepared for warning in item.warnings})
        ),
        mode=mode,
        top_n=top_n,
        timings=CandidateInspectionTimings(),
    )


def write_candidate_inspection(
    prepared: PreparedCandidateInspection,
    output_directory: Path,
    *,
    create_contact_sheet: bool = False,
    clock: Callable[[], float] = time.perf_counter,
) -> CandidateInspectionArtifact:
    if prepared.mode is InspectionMode.NONE and create_contact_sheet:
        raise ValueError("contact sheet requires inspection mode top-n or all")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    warnings = list(prepared.warnings)
    contact_path: Path | None = None
    contact_start = clock()
    if create_contact_sheet:
        contact_path, contact_warning = _contact_sheet(
            prepared.records,
            output / "candidate_contact_sheet.jpg",
        )
        if contact_warning is not None:
            warnings.append(contact_warning)
    contact_seconds = clock() - contact_start if create_contact_sheet else 0.0
    unique_warnings = tuple(sorted(set(warnings)))

    json_start = clock()
    json_path = output / "candidates.json"
    json_path.write_text(
        json.dumps(
            {
                "inspection_mode": prepared.mode.value,
                "inspection_top_n": prepared.top_n,
                "records": prepared.records,
                "warnings": unique_warnings,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    candidate_json_seconds = clock() - json_start

    markdown_start = clock()
    markdown_path = output / "candidate_inspection.md"
    lines = [
        "# Candidate Inspection",
        "",
        f"- Inspection mode: `{prepared.mode.value}`",
        f"- Candidate records: {len(prepared.records)}",
    ]
    if prepared.mode is InspectionMode.NONE:
        lines.extend(
            [
                "- Thumbnail inspection disabled; no keyframe directory was scanned.",
                "",
            ]
        )
    else:
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        display_records = (
            prepared.records
            if prepared.mode is InspectionMode.ALL
            else tuple(
                record for record in prepared.records if record["rank"] <= prepared.top_n
            )
        )
        for record in display_records:
            grouped[(record["query_id"], record["video_id"])].append(record)
        lines.append("")
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
    lines.extend(["## Warnings", ""])
    lines.extend(f"- {warning}" for warning in unique_warnings)
    if not unique_warnings:
        lines.append("- None")
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    markdown_seconds = clock() - markdown_start

    timings = CandidateInspectionTimings(
        candidate_json_seconds=candidate_json_seconds,
        thumbnail_index_seconds=prepared.timings.thumbnail_index_seconds,
        thumbnail_resolve_seconds=prepared.timings.thumbnail_resolve_seconds,
        markdown_seconds=markdown_seconds,
        contact_sheet_seconds=contact_seconds,
    )
    return CandidateInspectionArtifact(
        records=prepared.records,
        warnings=unique_warnings,
        json_path=json_path,
        markdown_path=markdown_path,
        contact_sheet_path=contact_path,
        timings=timings,
    )


def build_candidate_inspection(
    results: tuple[KISResult, ...],
    registry: FeatureStoreRegistry,
    manifest: CorpusManifest,
    output_directory: Path,
    *,
    top_n: int = 50,
    create_contact_sheet: bool = False,
    mode: InspectionMode = InspectionMode.TOP_N,
    thumbnail_resolver: ThumbnailResolver | None = None,
) -> CandidateInspectionArtifact:
    prepared = prepare_candidate_inspection(
        results,
        registry,
        manifest,
        mode=mode,
        top_n=top_n,
        thumbnail_resolver=thumbnail_resolver,
    )
    return write_candidate_inspection(
        prepared,
        output_directory,
        create_contact_sheet=create_contact_sheet,
    )
