"""Deterministic and atomic Stage 0 artifact writers."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

FINAL_ARTIFACTS = (
    "run_manifest.json",
    "audit_summary.json",
    "audit_report.md",
    "video_manifest.jsonl",
    "btc_frame_manifest.jsonl",
    "clip_manifest.jsonl",
    "object_manifest.jsonl",
    "metadata_manifest.jsonl",
    "cross_asset_manifest.jsonl",
    "audit_issues.jsonl",
    "contract_notes.json",
)


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, default=json_default) + "\n")


def write_jsonl(path: Path, values: list[dict[str, Any]], *, sort_key=None) -> None:
    ordered = sorted(values, key=sort_key) if sort_key else values
    text = "".join(
        json.dumps(value, ensure_ascii=False, default=json_default) + "\n" for value in ordered
    )
    atomic_text(path, text)


def markdown_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# TRIAGE-EG Stage 0 BTC Data Audit",
            "",
            "This audit validates BTC assets only; it does not run retrieval or frame extraction.",
            "",
            "## Execution",
            "",
            f"- Mode: `{summary['mode']}`",
            "- Videos: "
            f"selected={summary['videos_selected']}, "
            f"completed={summary['videos_completed']}, "
            f"resumed={summary['videos_resumed']}, failed={summary['videos_failed']}",
            f"- Mapping rows: {summary['mapping_rows']}",
            f"- Detections observed: {summary['detections_observed']}",
            "",
            "## Gates",
            "",
            f"- BTC baseline: **{summary['gates']['btc_baseline']}**",
            f"- Raw video: **{summary['gates']['raw_video']}**",
            "",
            "## Issues",
            "",
            f"- Total: {summary['issues']['total']}",
            f"- By severity: `{summary['issues']['by_severity']}`",
            f"- By code: `{summary['issues']['by_code']}`",
            "",
            "## Contract boundaries",
            "",
            "- Mapping `frame_idx` is the authoritative original-frame coordinate.",
            "- Duplicate `frame_idx` rows are preserved.",
            "- Object numeric ingestion is fail closed and preserves raw evidence.",
            "- Bounding-box order remains `UNKNOWN`.",
            "- CLIP model compatibility remains unverified.",
            "",
        ]
    )


def create_bundle(output_root: str | Path, zip_path: str | Path) -> Path:
    root = Path(output_root).resolve(strict=True)
    target = Path(zip_path).resolve(strict=False)
    if target == root or root in target.parents:
        raise ValueError("ZIP must be outside the audit output directory")
    missing = [name for name in FINAL_ARTIFACTS if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing final artifacts: {', '.join(missing)}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(target, "w", compression=ZIP_DEFLATED) as archive:
        for name in FINAL_ARTIFACTS:
            archive.write(root / name, arcname=name)
    return target
