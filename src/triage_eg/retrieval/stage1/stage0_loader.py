"""Fail-closed Stage 0 artifact loading without raw-layout discovery."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REQUIRED_FILES = (
    "audit_summary.json",
    "run_manifest.json",
    "btc_frame_manifest.jsonl",
    "clip_manifest.jsonl",
    "contract_notes.json",
)


@dataclass(frozen=True)
class Stage0Bundle:
    root: Path
    summary: dict[str, Any]
    run_manifest: dict[str, Any]
    contract_notes: dict[str, Any]
    clip_records: tuple[dict[str, Any], ...]

    @property
    def frame_manifest_path(self) -> Path:
        return self.root / "btc_frame_manifest.jsonl"


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid Stage 0 JSON artifact {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Stage 0 artifact must contain an object: {path}")
    return value


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Malformed JSONL {path}:{line_number}: {error}") from error
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            yield value


def load_stage0_bundle(root: str | Path, *, require_full: bool = True) -> Stage0Bundle:
    resolved = Path(root).expanduser().resolve(strict=False)
    missing = [name for name in REQUIRED_FILES if not (resolved / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Stage 0 artifacts: {', '.join(missing)}")
    summary = _json(resolved / "audit_summary.json")
    run_manifest = _json(resolved / "run_manifest.json")
    notes = _json(resolved / "contract_notes.json")
    if run_manifest.get("status") != "COMPLETE":
        raise ValueError("Stage 0 run status must be COMPLETE")
    if summary.get("audit_version") != "0.1.0":
        raise ValueError("Unsupported Stage 0 audit version")
    if require_full and (
        summary.get("mode") != "full"
        or summary.get("videos_completed") != summary.get("videos_discovered")
        or summary.get("videos_completed") != 873
    ):
        raise ValueError("Stage 0 must be a complete 873-video full audit")
    if summary.get("gates", {}).get("btc_baseline") == "FAIL":
        raise ValueError("Stage 0 BTC baseline gate is FAIL")
    if summary.get("mapping_rows") != summary.get("clip_rows"):
        raise ValueError("Stage 0 mapping_rows must equal clip_rows")
    if notes.get("original_frame_policy") != (
        "CSV frame_idx is authoritative; never reconstruct from pts_time*fps"
    ):
        raise ValueError("Stage 0 original-frame policy is incompatible")
    unknown = {str(item).lower() for item in summary.get("unknown_contracts", [])}
    if not any("clip" in item and "compat" in item for item in unknown):
        raise ValueError("Stage 0 must retain unknown CLIP model compatibility")
    clips = tuple(iter_jsonl(resolved / "clip_manifest.jsonl"))
    if len(clips) != summary.get("videos_completed"):
        raise ValueError("clip_manifest row count must equal completed videos")
    if sum(int(item.get("row_count", -1)) for item in clips) != summary.get("clip_rows"):
        raise ValueError("clip_manifest total rows do not match audit summary")
    return Stage0Bundle(resolved, summary, run_manifest, notes, clips)
