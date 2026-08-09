"""Small Stage 2A operational artifacts and allowlisted report bundle."""

from __future__ import annotations

from pathlib import Path
from time import monotonic
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

from triage_eg.retrieval.stage1.search import write_kis_candidates
from triage_eg.retrieval.stage1b.writers import write_json, write_jsonl

from .contracts import QueryRequest, Stage2RuntimeError
from .language import LanguageResolution
from .results import grouped_video_view

QUERY_FILES = (
    "query_request.json",
    "language_resolution.json",
    "encoding_manifest.json",
    "ranked_frames.jsonl",
    "ranked_videos.jsonl",
    "kis_candidates.csv",
    "query_summary.json",
)
ROOT_REPORT_FILES = (
    "run_manifest.json",
    "preflight.json",
    "runtime_manifest.json",
    "smoke_results.jsonl",
    "latency_summary.json",
    "issues.jsonl",
)
FORBIDDEN_SUFFIXES = {".pt", ".pth", ".bin", ".npy", ".mp4", ".avi", ".mkv", ".jpg"}


def write_operational_query_artifacts(
    output_root: Path,
    request: QueryRequest,
    resolution: LanguageResolution,
    encoding: dict[str, Any],
    frames: list[dict[str, Any]],
    latencies_ms: dict[str, float],
) -> tuple[Path, list[dict[str, Any]], float]:
    started = monotonic()
    root = output_root / request.query_id
    if root.exists():
        raise FileExistsError(f"Query output already exists: {root}")
    root.mkdir(parents=True)
    videos = grouped_video_view(frames)
    write_json(
        root / "query_request.json",
        {
            "query_id": request.query_id,
            "text": request.text.strip(),
            "language": request.language,
            "top_k": request.top_k,
        },
    )
    write_json(root / "language_resolution.json", resolution.as_dict())
    write_json(root / "encoding_manifest.json", encoding)
    write_jsonl(root / "ranked_frames.jsonl", frames)
    write_jsonl(root / "ranked_videos.jsonl", videos)
    try:
        supporting = write_kis_candidates(
            root / "kis_candidates.csv",
            frames,
            max_predictions=request.top_k,
        )
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise Stage2RuntimeError("KIS_EXPORT_FAILED", str(error)) from error
    artifact_ms = (monotonic() - started) * 1000
    final_latencies = {
        **latencies_ms,
        "artifact_write_ms": artifact_ms,
        "total_ms": latencies_ms["total_ms"] + artifact_ms,
    }
    write_json(
        root / "query_summary.json",
        {
            "query_id": request.query_id,
            "status": "COMPLETE",
            "raw_results": len(frames),
            "video_groups": len(videos),
            "kis_candidates": len(supporting),
            "duplicate_mapping_groups": sum(len(values) > 1 for values in supporting.values()),
            "ranking_policy": "FROZEN_STAGE1A_EXACT_COSINE_NO_RERANKING",
            "latencies_ms": final_latencies,
        },
    )
    return root, videos, artifact_ms


def create_stage2_report_bundle(root: str | Path, zip_path: str | Path) -> Path:
    source = Path(root).expanduser().resolve(strict=True)
    target = Path(zip_path).expanduser().resolve(strict=False)
    if source in target.parents:
        raise ValueError("Stage 2 report ZIP must be outside the output root")
    missing = [name for name in ROOT_REPORT_FILES if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Stage 2 report artifacts: {missing}")
    members = list(ROOT_REPORT_FILES)
    for query_root in sorted(path for path in source.iterdir() if path.is_dir()):
        if query_root.name.startswith((".", "runtime_cache")):
            continue
        for name in QUERY_FILES:
            path = query_root / name
            if path.is_file():
                members.append(path.relative_to(source).as_posix())
    if any(
        Path(name).suffix.lower() in FORBIDDEN_SUFFIXES
        or name.startswith(("logs/", "cache/", "runtime_cache/"))
        for name in members
    ):
        raise ValueError("Stage 2 report allowlist contains forbidden artifacts")
    target.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(target, "w", compression=ZIP_DEFLATED) as archive:
        for name in sorted(set(members)):
            archive.write(source / name, arcname=name)
    return target


__all__ = ["create_stage2_report_bundle", "write_operational_query_artifacts"]
