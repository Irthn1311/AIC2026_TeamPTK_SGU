"""MB1 Mode A: prepare raw-video evidence for interval annotation.

This module deliberately performs no semantic inference. RT2 references are used only as
deterministic source anchors around which chronological raw frames are decoded.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from triage_eg.data.stage0_audit.asset_resolver import discover_layout, resolve_assets
from triage_eg.experiments.moment_m1 import DecodedFrame, OpenCVRawVideoDecoder
from triage_eg.experiments.reference_rt2 import RT2BenchmarkQuery, load_rt2_benchmark
from triage_eg.retrieval.stage1b.writers import write_json, write_jsonl

MB1_VERSION = "0.1.0"
MB1_MODE = "RAW_VIDEO_MOMENT_CANDIDATE_PREPARATION"
SOURCE_ANCHOR_TYPE = "RT2_REFERENCE_ORIGINAL_FRAME"
ANNOTATION_GENERATOR = "GPT-5.6 Sol"
MOMENT_TYPES = (
    "STATE",
    "FIRST_OCCURRENCE",
    "TRANSITION_ONSET",
    "TRANSITION_OFFSET",
    "CONTACT",
    "SEPARATION",
    "EXTREMUM",
    "ACTION_VISIBILITY",
)
ANNOTATION_CONFIDENCE = ("HIGH", "MEDIUM", "LOW")

# The first ten entries are the user-prioritized MB1 sources. The final four are RT2
# sources containing procedural/action evidence and keep the default run at 14 videos.
DEFAULT_PREFERRED_VIDEO_IDS = (
    "L23_V005",
    "L24_V022",
    "L26_V012",
    "L26_V065",
    "L26_V094",
    "L26_V156",
    "L26_V251",
    "L26_V409",
    "L30_V043",
    "L30_V080",
    "L25_V064",
    "L25_V067",
    "L22_V004",
    "L25_V057",
)


@dataclass(frozen=True)
class MB1Settings:
    seed: int = 2026
    selected_video_count: int = 14
    windows_per_video: int = 2
    max_candidate_windows: int = 30
    window_seconds: float = 4.0
    target_displayed_frames: int = 32
    max_frames_per_sheet: int = 16
    jpeg_quality: int = 90
    preferred_video_ids: tuple[str, ...] = DEFAULT_PREFERRED_VIDEO_IDS

    def __post_init__(self) -> None:
        if not 12 <= self.selected_video_count <= 16:
            raise ValueError("MB1 selected_video_count must be between 12 and 16")
        if self.windows_per_video != 2:
            raise ValueError("MB1 Mode A prepares two candidate windows per video")
        if not 24 <= self.max_candidate_windows <= 30:
            raise ValueError("MB1 max_candidate_windows must be between 24 and 30")
        if not 3.0 <= self.window_seconds <= 5.0:
            raise ValueError("MB1 window_seconds must be between 3 and 5 seconds")
        if not 24 <= self.target_displayed_frames <= 40:
            raise ValueError("MB1 target_displayed_frames must be between 24 and 40")
        if not 8 <= self.max_frames_per_sheet <= 20:
            raise ValueError("MB1 max_frames_per_sheet must be between 8 and 20")
        if not 70 <= self.jpeg_quality <= 95:
            raise ValueError("MB1 jpeg_quality must be between 70 and 95")
        if len(set(self.preferred_video_ids)) != len(self.preferred_video_ids):
            raise ValueError("MB1 preferred_video_ids must be unique")


DEFAULT_SETTINGS = MB1Settings()


class RawVideoDecoder(Protocol):
    info: Any

    def decode_indices(self, frame_indices: list[int]) -> list[DecodedFrame]: ...

    def close(self) -> None: ...


def _stable_fallback_key(video_id: str, seed: int) -> tuple[bytes, str]:
    return hashlib.sha256(f"MB1:{seed}:{video_id}".encode()).digest(), video_id


def _query_by_video(queries: list[RT2BenchmarkQuery]) -> dict[str, RT2BenchmarkQuery]:
    by_video: dict[str, RT2BenchmarkQuery] = {}
    for query in queries:
        if query.source_video_id in by_video:
            raise ValueError(f"Duplicate RT2 source video: {query.source_video_id}")
        by_video[query.source_video_id] = query
    return by_video


def select_mb1_sources(
    queries: list[RT2BenchmarkQuery],
    available_video_ids: set[str],
    settings: MB1Settings = DEFAULT_SETTINGS,
) -> list[RT2BenchmarkQuery]:
    """Select preferred available videos, followed by a deterministic seeded fallback."""

    by_video = _query_by_video(queries)
    eligible = {
        video_id
        for video_id, query in by_video.items()
        if video_id in available_video_ids and len(query.events) >= settings.windows_per_video
    }
    preferred = [video_id for video_id in settings.preferred_video_ids if video_id in eligible]
    fallback = sorted(
        eligible.difference(preferred), key=lambda value: _stable_fallback_key(value, settings.seed)
    )
    selected_ids = (preferred + fallback)[: settings.selected_video_count]
    if len(selected_ids) < settings.selected_video_count:
        raise RuntimeError(
            "MB1_CANDIDATE_SOURCE_SHORTAGE: "
            f"need {settings.selected_video_count}, found {len(selected_ids)}"
        )
    return [by_video[video_id] for video_id in selected_ids]


def _source_events(query: RT2BenchmarkQuery, count: int) -> list[Any]:
    """Take chronologically separated RT2 identities without reusing their text labels."""

    if len(query.events) < count:
        raise ValueError(f"Not enough source anchors for {query.source_video_id}")
    if count == 1:
        return [query.events[len(query.events) // 2]]
    positions = np.rint(np.linspace(0, len(query.events) - 1, count)).astype(np.int64)
    return [query.events[int(position)] for position in positions]


def clipped_window(
    anchor_frame: int, *, fps: float, total_frames: int, seconds: float
) -> tuple[int, int]:
    if not math.isfinite(fps) or fps <= 0 or total_frames <= 0:
        raise ValueError("video FPS and total_frames must be valid")
    if not 0 <= anchor_frame < total_frames:
        raise IndexError("source anchor is outside the raw video")
    target_span = max(1, int(round(seconds * fps)))
    start = anchor_frame - target_span // 2
    end = start + target_span
    if start < 0:
        end = min(total_frames - 1, end - start)
        start = 0
    if end >= total_frames:
        start = max(0, start - (end - total_frames + 1))
        end = total_frames - 1
    return start, end


def displayed_frame_indices(
    start: int, end: int, anchor: int, *, target_count: int = 32
) -> list[int]:
    """Return approximately target_count chronological raw-frame coordinates."""

    if not 0 <= start <= anchor <= end or target_count < 2:
        raise ValueError("invalid displayed-frame sampling bounds")
    count = min(target_count, end - start + 1, 40)
    indices = [
        int(value)
        for value in np.rint(np.linspace(start, end, count)).astype(np.int64).tolist()
    ]
    indices = sorted(set(indices))
    if anchor not in indices:
        nearest = min(range(len(indices)), key=lambda index: abs(indices[index] - anchor))
        indices[nearest] = anchor
        indices = sorted(set(indices))
    if indices != sorted(set(indices)) or not all(start <= value <= end for value in indices):
        raise RuntimeError("displayed frame mapping is not chronological and bounded")
    return indices


def _font(size: int):
    from PIL import ImageFont

    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _render_sheet_page(
    path: Path,
    *,
    candidate_id: str,
    video_id: str,
    fps: float,
    frames: list[DecodedFrame],
    page_number: int,
    page_count: int,
    quality: int,
) -> None:
    from PIL import Image, ImageDraw, ImageOps

    columns = 4
    rows = int(math.ceil(len(frames) / columns))
    tile_width, image_height, label_height, header_height = 320, 180, 38, 68
    sheet = Image.new(
        "RGB",
        (columns * tile_width, header_height + rows * (image_height + label_height)),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (12, 10),
        (
            f"{candidate_id} | video_id={video_id} | chronological raw frames\n"
            f"page {page_number}/{page_count} | timestamps are display-only conveniences"
        ),
        fill="black",
        font=_font(20),
    )
    for slot, frame in enumerate(frames):
        x = (slot % columns) * tile_width
        y = header_height + (slot // columns) * (image_height + label_height)
        image = Image.fromarray(np.asarray(frame.image, dtype=np.uint8), mode="RGB")
        fitted = ImageOps.fit(image, (tile_width, image_height), method=Image.Resampling.LANCZOS)
        sheet.paste(fitted, (x, y))
        timestamp = frame.actual_frame_idx / fps
        draw.text(
            (x + 5, y + image_height + 5),
            f"actual_frame_idx={frame.actual_frame_idx}  t={timestamp:.3f}s",
            fill="black",
            font=_font(15),
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, format="JPEG", quality=quality, optimize=True)


def _annotation_schema() -> dict[str, Any]:
    example = {
        "moment_id": "mb1_001",
        "video_id": "Lxx_Vxxx",
        "source_candidate_id": "mb1_c001",
        "query_text": "object begins being cut",
        "moment_definition": "First visible frame where the blade begins cutting the object.",
        "moment_type": "TRANSITION_ONSET",
        "acceptable_start_frame": 100,
        "acceptable_end_frame": 106,
        "preferred_frame": 103,
        "annotation_confidence": "HIGH",
        "generator": ANNOTATION_GENERATOR,
        "human_reviewed": False,
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "TRIAGE-EG MB1 semantic moment annotation",
        "type": "object",
        "additionalProperties": False,
        "required": [name for name in example if name != "preferred_frame"],
        "properties": {
            "moment_id": {"type": "string", "minLength": 1},
            "video_id": {"type": "string", "pattern": "^L[0-9]+_V[0-9]+$"},
            "source_candidate_id": {"type": "string", "pattern": "^mb1_c[0-9]{3}$"},
            "query_text": {"type": "string", "minLength": 1},
            "moment_definition": {"type": "string", "minLength": 1},
            "moment_type": {"enum": list(MOMENT_TYPES)},
            "acceptable_start_frame": {"type": "integer", "minimum": 0},
            "acceptable_end_frame": {"type": "integer", "minimum": 0},
            "preferred_frame": {"type": ["integer", "null"], "minimum": 0},
            "annotation_confidence": {"enum": list(ANNOTATION_CONFIDENCE)},
            "generator": {"const": ANNOTATION_GENERATOR},
            "human_reviewed": {"const": False},
        },
        "x-invariants": [
            "acceptable_start_frame <= acceptable_end_frame",
            (
                "preferred_frame is null or acceptable_start_frame <= preferred_frame "
                "<= acceptable_end_frame"
            ),
            "all annotated frames must be visible within the source candidate window",
        ],
        "examples": [example],
    }


def validate_annotation(value: dict[str, Any]) -> None:
    required = set(_annotation_schema()["required"])
    allowed = set(_annotation_schema()["properties"])
    if not required.issubset(value) or not set(value).issubset(allowed):
        raise ValueError(
            f"MB1 annotation requires {sorted(required)} and allows only {sorted(allowed)}"
        )
    if value["moment_type"] not in MOMENT_TYPES:
        raise ValueError("invalid MB1 moment_type")
    if value["annotation_confidence"] not in ANNOTATION_CONFIDENCE:
        raise ValueError("invalid MB1 annotation_confidence")
    if value["generator"] != ANNOTATION_GENERATOR or value["human_reviewed"] is not False:
        raise ValueError("invalid MB1 annotation provenance")
    start, end, preferred = (
        value["acceptable_start_frame"],
        value["acceptable_end_frame"],
        value.get("preferred_frame"),
    )
    if not isinstance(start, int) or not isinstance(end, int) or start < 0 or start > end:
        raise ValueError("MB1 acceptable interval is invalid")
    if preferred is not None and (
        not isinstance(preferred, int) or not start <= preferred <= end
    ):
        raise ValueError("MB1 preferred_frame must be inside the acceptable interval")


def _annotation_readme() -> str:
    return """# MB1 semantic-moment annotation instructions

This pack contains chronological raw-video evidence for interval annotation. It does not
contain ground truth, retrieval scores, method comparisons, or model output.

## Inputs

- `mb1_candidate_manifest.jsonl`: window bounds, raw-frame coordinates, FPS, and sheet paths.
- `contact_sheets/`: chronological pages. Every tile prints `actual_frame_idx`.
- `annotation_schema.json`: exact output fields, enums, example, and interval invariants.

## GPT-5.6 Sol task

Inspect each candidate independently. Define a visually observable semantic moment only when
the sheets show enough before/during/after evidence. Write one or more JSONL annotations using
the schema. Do not infer hidden actions, identities, audio, intent, or off-screen events.

Use an interval rather than copying an arbitrary BTC frame. The interval must satisfy
`acceptable_start_frame <= acceptable_end_frame`. If `preferred_frame` is not null, it must be
inside that interval. Use LOW confidence or omit an unusable candidate when boundaries are not
visually defensible. Keep `human_reviewed` false.

Suggested output filename: `mb1_ai_semantic_moments.jsonl`.
"""


def preflight_mb1(
    dataset_root: str | Path,
    benchmark_path: str | Path,
    settings: MB1Settings = DEFAULT_SETTINGS,
) -> dict[str, Any]:
    dataset = Path(dataset_root).expanduser().resolve(strict=True)
    benchmark = Path(benchmark_path).expanduser().resolve(strict=True)
    queries = load_rt2_benchmark(benchmark)
    video_partitions, _ = discover_layout(dataset)
    selected = select_mb1_sources(queries, set(video_partitions), settings)
    candidate_count = min(
        len(selected) * settings.windows_per_video, settings.max_candidate_windows
    )
    if not 24 <= candidate_count <= 30:
        raise RuntimeError(f"MB1 candidate target is not satisfied: {candidate_count}")
    return {
        "status": "READY",
        "mode": MB1_MODE,
        "dataset_root": str(dataset),
        "benchmark_path": str(benchmark),
        "selected_video_count": len(selected),
        "candidate_window_count": candidate_count,
        "selected_video_ids": [query.source_video_id for query in selected],
        "model_inference_required": False,
        "network_required": False,
    }


def prepare_mb1_candidates(
    dataset_root: str | Path,
    benchmark_path: str | Path,
    output_root: str | Path,
    *,
    settings: MB1Settings = DEFAULT_SETTINGS,
    build_git_commit: str | None = None,
    decoder_factory: Any = OpenCVRawVideoDecoder,
) -> dict[str, Any]:
    dataset = Path(dataset_root).expanduser().resolve(strict=True)
    benchmark = Path(benchmark_path).expanduser().resolve(strict=True)
    output = Path(output_root).expanduser().resolve(strict=False)
    if output.exists():
        raise FileExistsError(f"MB1 output already exists: {output}")
    queries = load_rt2_benchmark(benchmark)
    video_partitions, keyframe_partitions = discover_layout(dataset)
    selected = select_mb1_sources(queries, set(video_partitions), settings)
    output.mkdir(parents=True)

    manifest_rows: list[dict[str, Any]] = []
    selected_video_rows: list[dict[str, Any]] = []
    candidate_number = 0
    for query in selected:
        assets = resolve_assets(
            dataset, query.source_video_id, video_partitions, keyframe_partitions
        )
        if not assets.video.is_file():
            raise FileNotFoundError(f"MB1 raw video missing: {assets.video}")
        decoder: RawVideoDecoder = decoder_factory(query.source_video_id, assets.video)
        video_candidates = []
        try:
            for event in _source_events(query, settings.windows_per_video):
                if candidate_number >= settings.max_candidate_windows:
                    break
                anchor = int(event.reference_original_frame_idx)
                start, end = clipped_window(
                    anchor,
                    fps=float(decoder.info.fps),
                    total_frames=int(decoder.info.total_frames),
                    seconds=settings.window_seconds,
                )
                displayed = displayed_frame_indices(
                    start, end, anchor, target_count=settings.target_displayed_frames
                )
                frames = decoder.decode_indices(displayed)
                if [frame.actual_frame_idx for frame in frames] != displayed:
                    raise RuntimeError("MB1 raw-frame identity mapping is incomplete")
                candidate_number += 1
                candidate_id = f"mb1_c{candidate_number:03d}"
                pages = [
                    frames[index : index + settings.max_frames_per_sheet]
                    for index in range(0, len(frames), settings.max_frames_per_sheet)
                ]
                sheet_paths = []
                for page_index, page_frames in enumerate(pages, 1):
                    suffix = chr(ord("A") + page_index - 1)
                    relative = Path("contact_sheets") / f"{candidate_id}_page_{suffix}.jpg"
                    _render_sheet_page(
                        output / relative,
                        candidate_id=candidate_id,
                        video_id=query.source_video_id,
                        fps=float(decoder.info.fps),
                        frames=page_frames,
                        page_number=page_index,
                        page_count=len(pages),
                        quality=settings.jpeg_quality,
                    )
                    sheet_paths.append(relative.as_posix())
                row = {
                    "candidate_id": candidate_id,
                    "video_id": query.source_video_id,
                    "window_start_frame": start,
                    "window_end_frame": end,
                    "fps": float(decoder.info.fps),
                    "displayed_frames": displayed,
                    "source_anchor_frame": anchor,
                    "source_anchor_type": SOURCE_ANCHOR_TYPE,
                    "image_sheet_paths": sheet_paths,
                }
                manifest_rows.append(row)
                video_candidates.append(candidate_id)
        finally:
            decoder.close()
        selected_video_rows.append(
            {"video_id": query.source_video_id, "candidate_ids": video_candidates}
        )

    if not 24 <= len(manifest_rows) <= 30:
        raise RuntimeError(f"MB1 candidate target is not satisfied: {len(manifest_rows)}")
    selection = {
        "experiment": "MB1",
        "mb1_version": MB1_VERSION,
        "mode": MB1_MODE,
        "status": "READY",
        "created_at": datetime.now(UTC).isoformat(),
        "build_git_commit": build_git_commit,
        "seed": settings.seed,
        "settings": {**asdict(settings), "preferred_video_ids": list(settings.preferred_video_ids)},
        "selection_policy": (
            "USER_PRIORITIZED_ACTION_RICH_RT2_SOURCE_ORDER_THEN_SEEDED_FALLBACK"
        ),
        "anchor_policy": "CHRONOLOGICALLY_SEPARATED_RT2_FRAME_IDENTITIES_WITHOUT_LABELS",
        "semantic_labels_assigned": False,
        "selected_video_count": len(selected_video_rows),
        "candidate_window_count": len(manifest_rows),
        "selected_videos": selected_video_rows,
        "model_inference_required": False,
        "network_required": False,
        "optional_clips_exported": False,
    }
    write_jsonl(output / "mb1_candidate_manifest.jsonl", manifest_rows)
    write_json(output / "candidate_selection.json", selection)
    write_json(output / "annotation_schema.json", _annotation_schema())
    (output / "README_AI_ANNOTATION.md").write_text(_annotation_readme(), encoding="utf-8")
    return selection


def create_mb1_bundle(root: str | Path, zip_path: str | Path) -> Path:
    source = Path(root).expanduser().resolve(strict=True)
    target = Path(zip_path).expanduser().resolve(strict=False)
    if source in target.parents:
        raise ValueError("MB1 ZIP must be outside the candidate output root")
    required = (
        "mb1_candidate_manifest.jsonl",
        "candidate_selection.json",
        "annotation_schema.json",
        "README_AI_ANNOTATION.md",
    )
    missing = [name for name in required if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing MB1 candidate artifacts: {missing}")
    manifest = [
        json.loads(line)
        for line in (source / "mb1_candidate_manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    sheet_names = [name for row in manifest for name in row["image_sheet_paths"]]
    if len(sheet_names) != len(set(sheet_names)):
        raise ValueError("MB1 manifest contains duplicate sheet paths")
    members = [source / name for name in required] + [source / name for name in sheet_names]
    missing_sheets = [path for path in members if not path.is_file()]
    if missing_sheets:
        raise FileNotFoundError(f"Missing MB1 bundle members: {missing_sheets}")
    allowed_suffixes = {".jsonl", ".json", ".md", ".jpg"}
    if any(path.suffix.lower() not in allowed_suffixes for path in members):
        raise ValueError("MB1 bundle contains a forbidden artifact type")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_suffix(target.suffix + ".building")
    staging.unlink(missing_ok=True)
    try:
        with ZipFile(staging, "w", compression=ZIP_DEFLATED) as archive:
            for path in sorted(members, key=lambda item: item.relative_to(source).as_posix()):
                archive.write(path, arcname=path.relative_to(source).as_posix())
        shutil.move(staging, target)
    finally:
        staging.unlink(missing_ok=True)
    return target
