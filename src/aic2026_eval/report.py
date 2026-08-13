"""Report writers and strict compact bundle builders."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

from .io import write_json, write_jsonl

BOOTSTRAP_REQUIRED = {
    "corpus_summary.json",
    "corpus_inventory.jsonl",
    "video_usage_summary.json",
    "video_usage_census.jsonl",
    "heldout_candidate_manifest.jsonl",
    "heldout_selection_report.json",
    "atlas_manifest.jsonl",
    "atlas_index.json",
    "evaluation_contract.json",
    "README.md",
    "run_manifest.json",
    "issues.jsonl",
}
DENSE_REQUIRED = {
    "anchor_requests.jsonl",
    "dense_manifest.jsonl",
    "run_manifest.json",
    "issues.jsonl",
    "README.md",
}
FORBIDDEN_PARTS = {
    "runtime_cache",
    "__pycache__",
    "raw_videos",
    "checkpoints",
    "models",
}
FORBIDDEN_SUFFIXES = {".mp4", ".avi", ".mkv", ".npy", ".npz", ".pt", ".pth"}


def _safe_members(root: Path, *, dense: bool) -> list[Path]:
    required = DENSE_REQUIRED if dense else BOOTSTRAP_REQUIRED
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise ValueError(f"bundle missing required files: {missing}")
    allowed_prefix = "dense/" if dense else "atlas/"
    members = [root / name for name in sorted(required)]
    optional = ("l21_mapping_audit.jsonl", "l21_mapping_summary.json")
    if not dense:
        members.extend(root / name for name in optional if (root / name).is_file())
    members.extend(
        path for path in sorted((root / allowed_prefix.rstrip("/")).glob("*.jpg")) if path.is_file()
    )
    for path in members:
        relative = path.relative_to(root)
        if set(relative.parts) & FORBIDDEN_PARTS or path.suffix.lower() in FORBIDDEN_SUFFIXES:
            raise ValueError(f"forbidden bundle member: {relative}")
    return members


def create_bundle(root: str | Path, zip_path: str | Path, *, dense: bool = False) -> Path:
    source, target = Path(root), Path(zip_path)
    members = _safe_members(source, dense=dense)
    target.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(target, "w", ZIP_DEFLATED) as archive:
        for path in members:
            archive.write(path, path.relative_to(source).as_posix())
    return target


def write_evaluation_reports(
    root: str | Path,
    summary: dict[str, Any],
    by_query: list[dict[str, Any]],
    by_slice: dict[str, Any],
    issues: list[dict[str, Any]],
) -> None:
    output = Path(root)
    write_json(output / "evaluation_summary.json", summary)
    write_jsonl(output / "evaluation_by_query.jsonl", by_query)
    write_json(output / "evaluation_by_slice.json", by_slice)
    write_jsonl(output / "issues.jsonl", issues)
    with (output / "evaluation_summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)


__all__ = ["create_bundle", "write_evaluation_reports"]
