"""Candidate preparation and strict pseudo-GT contracts for reference experiment RT2."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from triage_eg.experiments.reference_rt1.scoring import VideoRows, build_video_row_groups
from triage_eg.retrieval.stage1.search import CompactCatalog
from triage_eg.retrieval.stage1b.writers import write_json, write_jsonl
from triage_eg.retrieval.stage1d.artifacts import _paste_frame

RT2_VERSION = "0.1.0"
BENCHMARK_TYPE = "AI_CURATED_INTERNAL_PSEUDO_GT"
MIN_CANONICAL_KEYFRAMES = 12
TEMPORALLY_DIVERSE = "TEMPORALLY_DIVERSE"
GENERAL_ELIGIBLE = "GENERAL_ELIGIBLE"
FORBIDDEN_BUNDLE_SUFFIXES = {".pt", ".pth", ".bin", ".npy", ".npz", ".mp4"}


def _safe_id(value: str, name: str) -> str:
    import re

    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value):
        raise ValueError(f"{name} must be a safe non-empty identifier")
    return value


@dataclass(frozen=True)
class RT2ReferenceEvent:
    event_id: str
    text: str
    reference_slot: str
    reference_catalog_position: int
    reference_global_row: int
    reference_n: int
    reference_original_frame_idx: int

    def __post_init__(self) -> None:
        import re

        _safe_id(self.event_id, "event_id")
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("event text must be non-empty")
        if not re.fullmatch(r"S\d{2}", self.reference_slot):
            raise ValueError("reference_slot must use S01-style notation")
        values = (
            self.reference_catalog_position,
            self.reference_global_row,
            self.reference_n,
            self.reference_original_frame_idx,
        )
        if any(not isinstance(value, int) or value < 0 for value in values):
            raise ValueError("reference frame identity values must be non-negative integers")
        if self.reference_n < 1:
            raise ValueError("reference_n is one-based and must be positive")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> RT2ReferenceEvent:
        required = {
            "event_id",
            "text",
            "reference_slot",
            "reference_catalog_position",
            "reference_global_row",
            "reference_n",
            "reference_original_frame_idx",
        }
        if set(value) != required:
            raise ValueError(f"RT2 event fields must be exactly {sorted(required)}")
        return cls(
            event_id=str(value["event_id"]),
            text=str(value["text"]),
            reference_slot=str(value["reference_slot"]),
            reference_catalog_position=value["reference_catalog_position"],
            reference_global_row=value["reference_global_row"],
            reference_n=value["reference_n"],
            reference_original_frame_idx=value["reference_original_frame_idx"],
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "text": self.text,
            "reference_slot": self.reference_slot,
            "reference_catalog_position": self.reference_catalog_position,
            "reference_global_row": self.reference_global_row,
            "reference_n": self.reference_n,
            "reference_original_frame_idx": self.reference_original_frame_idx,
        }


@dataclass(frozen=True)
class RT2BenchmarkQuery:
    query_id: str
    benchmark_type: str
    source_video_id: str
    language: str
    events: tuple[RT2ReferenceEvent, ...]
    difficulty_tags: tuple[str, ...]
    generator: str
    human_reviewed: bool

    def __post_init__(self) -> None:
        _safe_id(self.query_id, "query_id")
        _safe_id(self.source_video_id, "source_video_id")
        if self.benchmark_type != BENCHMARK_TYPE:
            raise ValueError(f"benchmark_type must be {BENCHMARK_TYPE}")
        if self.language not in {"en", "vi"}:
            raise ValueError("RT2 language must be explicit en or vi")
        if not 2 <= len(self.events) <= 4:
            raise ValueError("RT2 benchmark queries require 2 to 4 events")
        event_ids = [event.event_id for event in self.events]
        if len(set(event_ids)) != len(event_ids):
            raise ValueError("event_id values must be unique within a query")
        positions = [event.reference_catalog_position for event in self.events]
        if any(left >= right for left, right in zip(positions[:-1], positions[1:], strict=True)):
            raise ValueError("reference_catalog_position values must be strictly increasing")
        if "MULTI_EVENT" not in self.difficulty_tags:
            raise ValueError("difficulty_tags must include MULTI_EVENT")
        if not self.generator.strip():
            raise ValueError("generator must be non-empty")
        if self.human_reviewed is not False:
            raise ValueError("RT2 AI-curated input must declare human_reviewed=false")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> RT2BenchmarkQuery:
        required = {
            "query_id",
            "benchmark_type",
            "source_video_id",
            "language",
            "events",
            "difficulty_tags",
            "generator",
            "human_reviewed",
        }
        if set(value) != required:
            raise ValueError(f"RT2 benchmark fields must be exactly {sorted(required)}")
        if not isinstance(value["events"], list) or not isinstance(value["difficulty_tags"], list):
            raise ValueError("events and difficulty_tags must be lists")
        if not isinstance(value["human_reviewed"], bool):
            raise ValueError("human_reviewed must be a boolean")
        if any(not isinstance(tag, str) or not tag for tag in value["difficulty_tags"]):
            raise ValueError("difficulty_tags must contain non-empty strings")
        return cls(
            query_id=str(value["query_id"]),
            benchmark_type=str(value["benchmark_type"]),
            source_video_id=str(value["source_video_id"]),
            language=str(value["language"]),
            events=tuple(RT2ReferenceEvent.from_dict(item) for item in value["events"]),
            difficulty_tags=tuple(value["difficulty_tags"]),
            generator=str(value["generator"]),
            human_reviewed=value["human_reviewed"],
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "benchmark_type": self.benchmark_type,
            "source_video_id": self.source_video_id,
            "language": self.language,
            "events": [event.as_dict() for event in self.events],
            "difficulty_tags": list(self.difficulty_tags),
            "generator": self.generator,
            "human_reviewed": self.human_reviewed,
        }


def load_rt2_benchmark(path: str | Path) -> list[RT2BenchmarkQuery]:
    source = Path(path).expanduser().resolve(strict=True)
    queries: list[RT2BenchmarkQuery] = []
    try:
        for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"line {line_number} is not an object")
            queries.append(RT2BenchmarkQuery.from_dict(value))
    except (json.JSONDecodeError, OSError, TypeError) as error:
        raise ValueError(f"Invalid RT2 benchmark {source}: {error}") from error
    if not queries:
        raise ValueError("RT2 benchmark must not be empty")
    if len({query.query_id for query in queries}) != len(queries):
        raise ValueError("RT2 benchmark query_id values must be unique")
    return queries


def evenly_sample_positions(total: int, count: int) -> tuple[int, ...]:
    """Return unique chronological zero-based positions spanning a video."""

    if total <= 0 or count <= 0:
        raise ValueError("total and count must be positive")
    if total <= count:
        return tuple(range(total))
    positions = np.rint(np.linspace(0, total - 1, count)).astype(np.int64)
    if len(np.unique(positions)) != count:
        raise RuntimeError("even sampling produced duplicate positions")
    return tuple(int(value) for value in positions)


def _visual_diversity(vectors: np.ndarray, rows: np.ndarray, sample_count: int = 16) -> float:
    positions = evenly_sample_positions(len(rows), min(sample_count, len(rows)))
    sampled = np.asarray(vectors[rows[list(positions)]], dtype=np.float32)
    if not np.isfinite(sampled).all():
        raise ValueError("candidate diversity received non-finite Stage 1 vectors")
    norms = np.linalg.norm(sampled, axis=1, keepdims=True)
    normalized = sampled / np.maximum(norms, np.finfo(np.float32).eps)
    adjacent_cosine = np.sum(normalized[:-1] * normalized[1:], axis=1)
    return float(np.mean(1.0 - adjacent_cosine, dtype=np.float64))


def select_candidate_videos(
    groups: list[VideoRows],
    vectors: np.ndarray,
    *,
    candidate_count: int = 36,
    seed: int = 2026,
) -> list[dict[str, Any]]:
    """Select half visual-diversity leaders and half seeded corpus-spread videos."""

    if candidate_count <= 1:
        raise ValueError("candidate_count must be at least two")
    eligible = [group for group in groups if len(group.rows) >= MIN_CANONICAL_KEYFRAMES]
    if len(eligible) < candidate_count:
        raise ValueError(
            f"Need {candidate_count} eligible videos with at least "
            f"{MIN_CANONICAL_KEYFRAMES} keyframes; found {len(eligible)}"
        )
    scored = [
        {
            "video_id": group.video_id,
            "group": group,
            "total_btc_keyframes": len(group.rows),
            "visual_diversity_score": _visual_diversity(vectors, group.rows),
        }
        for group in eligible
    ]
    diverse_count = candidate_count // 2
    general_count = candidate_count - diverse_count
    diverse = sorted(scored, key=lambda item: (-item["visual_diversity_score"], item["video_id"]))[
        :diverse_count
    ]
    diverse_ids = {item["video_id"] for item in diverse}
    remaining = sorted(
        (item for item in scored if item["video_id"] not in diverse_ids),
        key=lambda item: item["video_id"],
    )
    rng = np.random.default_rng(seed)
    general = []
    for stratum in np.array_split(np.arange(len(remaining)), general_count):
        if len(stratum) == 0:
            raise RuntimeError("general candidate strata cannot be empty")
        general.append(remaining[int(rng.choice(stratum))])
    output = []
    for bucket, items in ((TEMPORALLY_DIVERSE, diverse), (GENERAL_ELIGIBLE, general)):
        output.extend(
            {
                "video_id": item["video_id"],
                "sampling_bucket": bucket,
                "total_btc_keyframes": item["total_btc_keyframes"],
                "visual_diversity_score": item["visual_diversity_score"],
                "group": item["group"],
            }
            for item in sorted(items, key=lambda value: value["video_id"])
        )
    if len({item["video_id"] for item in output}) != candidate_count:
        raise RuntimeError("candidate selection did not produce unique videos")
    return output


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


def render_candidate_contact_sheet(
    output_path: Path,
    *,
    dataset_root: Path,
    catalog: Any,
    group: VideoRows,
    sampling_bucket: str,
    frames_per_sheet: int = 16,
) -> list[dict[str, Any]]:
    """Render a chronological 4x4 candidate sheet and return its manifest rows."""

    from PIL import Image, ImageDraw

    positions = evenly_sample_positions(len(group.rows), min(frames_per_sheet, len(group.rows)))
    columns, rows = 4, 4
    tile_width, image_height, label_height, header_height = 360, 205, 86, 92
    sheet = Image.new(
        "RGB",
        (columns * tile_width, header_height + rows * (image_height + label_height)),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    header = (
        f"video_id: {group.video_id}    total BTC keyframes: {len(group.rows)}\n"
        f"sampling bucket: {sampling_bucket}    chronological canonical order"
    )
    draw.multiline_text((12, 10), header, fill="black", font=_font(22), spacing=6)
    manifest_rows: list[dict[str, Any]] = []
    issues = []
    for slot_index, catalog_position in enumerate(positions, 1):
        global_row = int(group.rows[catalog_position])
        mapped = catalog.map_row(global_row)
        column, row = (slot_index - 1) % columns, (slot_index - 1) // columns
        x = column * tile_width
        y = header_height + row * (image_height + label_height)
        issue = _paste_frame(
            sheet,
            draw,
            item=mapped,
            dataset_root=dataset_root,
            x=x,
            y=y,
            width=tile_width,
            height=image_height,
        )
        if issue:
            issues.append(issue)
        slot = f"S{slot_index:02d}"
        draw.multiline_text(
            (x + 6, y + image_height + 4),
            (
                f"slot: {slot}  catalog_position: {catalog_position}\n"
                f"global_row: {global_row}  n: {mapped['n']}\n"
                f"original_frame_idx: {mapped['original_frame_idx']}"
            ),
            fill="black",
            font=_font(16),
            spacing=2,
        )
        manifest_rows.append(
            {
                "video_id": group.video_id,
                "sampling_bucket": sampling_bucket,
                "total_btc_keyframes": len(group.rows),
                "sheet_slot": slot,
                "catalog_position": catalog_position,
                "global_row": global_row,
                "n": int(mapped["n"]),
                "original_frame_idx": int(mapped["original_frame_idx"]),
                "image_path": str(mapped["keyframe_relative_path"]).replace("\\", "/"),
            }
        )
    if issues:
        raise FileNotFoundError(f"Missing {len(issues)} canonical keyframes for {group.video_id}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, format="JPEG", quality=90, optimize=True)
    return manifest_rows


def _creation_readme() -> str:
    example = {
        "query_id": "rt2_001",
        "benchmark_type": BENCHMARK_TYPE,
        "source_video_id": "Lxx_Vxxx",
        "language": "en",
        "events": [
            {
                "event_id": "E1",
                "text": "...",
                "reference_slot": "S03",
                "reference_catalog_position": 17,
                "reference_global_row": 123,
                "reference_n": 18,
                "reference_original_frame_idx": 456,
            },
            {
                "event_id": "E2",
                "text": "...",
                "reference_slot": "S11",
                "reference_catalog_position": 80,
                "reference_global_row": 186,
                "reference_n": 81,
                "reference_original_frame_idx": 1770,
            },
        ],
        "difficulty_tags": ["MULTI_EVENT"],
        "generator": "GPT-5.6 Sol",
        "human_reviewed": False,
    }
    return """# RT2 AI benchmark creation instructions

This candidate pack is an internal research artifact, not official ground truth.

For each usable video, inspect only the visible chronological contact-sheet evidence:

1. Choose 2-4 visually distinct events in strictly increasing slot order.
2. Write concise English event descriptions that are visually searchable.
3. Do not use video IDs, filenames, or external content metadata as semantic hints.
4. Reject videos without a usable multi-event sequence.
5. Copy the exact slot and frame identity fields printed on the sheet for each event.
6. Keep `human_reviewed` false and use benchmark type `AI_CURATED_INTERNAL_PSEUDO_GT`.

Return JSONL named `rt2_ai_benchmark.jsonl`. Each row must follow this shape:

```json
EXAMPLE_JSON
```

Use 20-24 usable videos if the visible evidence supports them. Never invent hidden events.
""".replace("EXAMPLE_JSON", json.dumps(example, ensure_ascii=False, separators=(",", ":")))


def prepare_benchmark_candidates(
    stage1_root: str | Path,
    dataset_root: str | Path,
    output_root: str | Path,
    *,
    candidate_count: int = 36,
    seed: int = 2026,
    frames_per_sheet: int = 16,
    build_git_commit: str | None = None,
) -> dict[str, Any]:
    stage1 = Path(stage1_root).expanduser().resolve(strict=True)
    dataset = Path(dataset_root).expanduser().resolve(strict=True)
    output = Path(output_root).expanduser().resolve(strict=False)
    if output.exists():
        raise FileExistsError(f"RT2 candidate output already exists: {output}")
    output.mkdir(parents=True)
    catalog = CompactCatalog(stage1 / "index")
    vectors = np.load(stage1 / "index/clip_vectors.f16.npy", mmap_mode="r", allow_pickle=False)
    if len(vectors) != len(catalog.n):
        raise ValueError("Stage 1 vectors and catalog have different row counts")
    groups = build_video_row_groups(catalog)
    selected = select_candidate_videos(groups, vectors, candidate_count=candidate_count, seed=seed)
    stage1_summary_path = stage1 / "stage1_summary.json"
    stage1_summary = (
        json.loads(stage1_summary_path.read_text(encoding="utf-8"))
        if stage1_summary_path.is_file()
        else {}
    )
    manifest_rows = []
    selected_metadata = []
    for item in selected:
        manifest_rows.extend(
            render_candidate_contact_sheet(
                output / "candidates" / f"{item['video_id']}.jpg",
                dataset_root=dataset,
                catalog=catalog,
                group=item["group"],
                sampling_bucket=item["sampling_bucket"],
                frames_per_sheet=frames_per_sheet,
            )
        )
        selected_metadata.append({key: value for key, value in item.items() if key != "group"})
    selection = {
        "reference_experiment": "RT2",
        "rt2_version": RT2_VERSION,
        "status": "READY",
        "created_at": datetime.now(UTC).isoformat(),
        "build_git_commit": build_git_commit,
        "stage1_index_fingerprint": stage1_summary.get("index_fingerprint"),
        "seed": seed,
        "selection_rules": {
            "minimum_canonical_btc_keyframes": MIN_CANONICAL_KEYFRAMES,
            "canonical_order": "N_ASCENDING_THEN_GLOBAL_ROW",
            "frames_per_sheet": frames_per_sheet,
            "temporally_diverse": (
                "highest mean adjacent cosine distance over evenly sampled frozen BTC CLIP "
                "image features"
            ),
            "general_eligible": (
                "seeded one-per-stratum sample over video-id-sorted remaining eligible videos"
            ),
            "retrieval_ranking_use": False,
        },
        "eligible_video_count": sum(len(group.rows) >= MIN_CANONICAL_KEYFRAMES for group in groups),
        "selected_video_count": len(selected),
        "bucket_counts": {
            TEMPORALLY_DIVERSE: sum(
                item["sampling_bucket"] == TEMPORALLY_DIVERSE for item in selected
            ),
            GENERAL_ELIGIBLE: sum(item["sampling_bucket"] == GENERAL_ELIGIBLE for item in selected),
        },
        "selected_videos": selected_metadata,
    }
    write_jsonl(output / "candidate_manifest.jsonl", manifest_rows)
    write_json(output / "candidate_selection.json", selection)
    (output / "README_AI_BENCHMARK_CREATION.md").write_text(_creation_readme(), encoding="utf-8")
    return selection


def resolve_benchmark_identities(
    queries: list[RT2BenchmarkQuery], catalog: Any
) -> list[RT2BenchmarkQuery]:
    """Fail closed unless every pseudo-GT identity is canonical and self-consistent."""

    groups = {group.video_id: group for group in build_video_row_groups(catalog)}
    for query in queries:
        group = groups.get(query.source_video_id)
        if group is None:
            raise ValueError(f"Unknown source_video_id: {query.source_video_id}")
        for event in query.events:
            if event.reference_catalog_position >= len(group.rows):
                raise ValueError(f"Reference position outside {query.source_video_id}")
            expected_global_row = int(group.rows[event.reference_catalog_position])
            mapped = catalog.map_row(expected_global_row)
            identity = (
                event.reference_global_row,
                event.reference_n,
                event.reference_original_frame_idx,
            )
            expected = (
                expected_global_row,
                int(mapped["n"]),
                int(mapped["original_frame_idx"]),
            )
            if identity != expected or mapped["video_id"] != query.source_video_id:
                raise ValueError(
                    f"Canonical frame identity mismatch for {query.query_id}/{event.event_id}"
                )
    return queries


def create_candidate_bundle(root: str | Path, zip_path: str | Path) -> Path:
    source = Path(root).expanduser().resolve(strict=True)
    target = Path(zip_path).expanduser().resolve(strict=False)
    if source in target.parents:
        raise ValueError("RT2 candidate ZIP must be outside the candidate output root")
    required = (
        "candidate_manifest.jsonl",
        "candidate_selection.json",
        "README_AI_BENCHMARK_CREATION.md",
    )
    missing = [name for name in required if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing RT2 candidate artifacts: {missing}")
    selection = json.loads((source / "candidate_selection.json").read_text(encoding="utf-8"))
    sheets = sorted((source / "candidates").glob("*.jpg"))
    if len(sheets) != int(selection["selected_video_count"]):
        raise ValueError("RT2 candidate sheet count does not match candidate_selection.json")
    members = [source / name for name in required] + sheets
    relative = [path.relative_to(source).as_posix() for path in members]
    if any(Path(name).suffix.lower() in FORBIDDEN_BUNDLE_SUFFIXES for name in relative):
        raise ValueError("RT2 candidate bundle contains a forbidden artifact")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_suffix(target.suffix + ".building")
    staging.unlink(missing_ok=True)
    try:
        with ZipFile(staging, "w", compression=ZIP_DEFLATED) as archive:
            for path, name in sorted(zip(members, relative, strict=True), key=lambda item: item[1]):
                archive.write(path, arcname=name)
        shutil.move(staging, target)
    finally:
        staging.unlink(missing_ok=True)
    return target


def stable_query_order(query: RT2BenchmarkQuery, seed: int) -> bytes:
    return hashlib.sha256(f"{seed}:{len(query.events)}:{query.query_id}".encode()).digest()


__all__ = [
    "BENCHMARK_TYPE",
    "GENERAL_ELIGIBLE",
    "MIN_CANONICAL_KEYFRAMES",
    "RT2BenchmarkQuery",
    "RT2ReferenceEvent",
    "RT2_VERSION",
    "TEMPORALLY_DIVERSE",
    "create_candidate_bundle",
    "evenly_sample_positions",
    "load_rt2_benchmark",
    "prepare_benchmark_candidates",
    "render_candidate_contact_sheet",
    "resolve_benchmark_identities",
    "select_candidate_videos",
    "stable_query_order",
]
