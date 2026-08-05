"""Bounded, non-authoritative helpers for manual KIS annotation review."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from system_tai.common.schemas import KISResult
from system_tai.evaluation.benchmark_schema import BenchmarkQuery

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True, slots=True)
class AnnotationCandidate:
    query_id: str
    video_id: str
    frame_id: int
    keyframe_order: int
    score: float
    image_path: str | None
    decision: str = "unreviewed"


def build_annotation_candidates(
    result: KISResult,
    keyframe_directories: dict[str, Path],
    *,
    limit: int = 20,
) -> tuple[AnnotationCandidate, ...]:
    if limit <= 0:
        raise ValueError("annotation candidate limit must be positive")
    resolved_images: dict[tuple[str, int], str | None] = {}
    output: list[AnnotationCandidate] = []
    for candidate in result.ranked_candidates[:limit]:
        key = (candidate.video_id, candidate.keyframe_order)
        if key not in resolved_images:
            directory = keyframe_directories.get(candidate.video_id)
            resolved_images[key] = _resolve_image(directory, candidate.keyframe_order)
        output.append(
            AnnotationCandidate(
                query_id=result.query_id,
                video_id=candidate.video_id,
                frame_id=candidate.frame_id,
                keyframe_order=candidate.keyframe_order,
                score=candidate.score,
                image_path=resolved_images[key],
            )
        )
    return tuple(output)


def write_draft_annotation_review(
    query: BenchmarkQuery,
    candidates: tuple[AnnotationCandidate, ...],
    destination: Path,
) -> Path:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "query_id": query.query_id,
        "language": query.language.value,
        "text": query.text,
        "semantic_group_id": query.semantic_group_id,
        "variant_type": query.variant_type.value,
        "annotation_status": "draft",
        "relevant_frames": [],
        "annotation_notes": (
            "Candidate review only. A human must copy confirmed official frame_id "
            "labels into the benchmark and explicitly set annotation_status=verified."
        ),
        "candidates": [asdict(candidate) for candidate in candidates],
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _resolve_image(directory: Path | None, keyframe_order: int) -> str | None:
    if directory is None:
        return None
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"keyframe directory not found: {root}")
    matches = sorted(
        (
            path
            for path in root.iterdir()
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
            and path.stem.isdigit()
            and int(path.stem) == keyframe_order
        ),
        key=lambda path: str(path).lower(),
    )
    if len(matches) > 1:
        raise ValueError(
            f"ambiguous keyframe image for order {keyframe_order} in {root}"
        )
    return str(matches[0]) if matches else None
