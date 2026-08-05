"""Stage 1B JSONL/report writers and safe report-only ZIP packaging."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

REPORT_MEMBERS = (
    "run_manifest.json",
    "stage1b_summary.json",
    "stage1b_report.md",
    "evidence/evidence_inventory.jsonl",
    "evidence/evidence_summary.json",
    "probe/probe_samples.jsonl",
    "probe/candidate_results.jsonl",
    "probe/candidate_summaries.jsonl",
    "probe/compatibility_matrix.json",
    "encoder/candidate_contracts.jsonl",
    "encoder/selected_encoder_contract.json",
    "encoder/runtime_adapter_manifest.json",
    "smoke/smoke_queries.jsonl",
    "smoke/smoke_results.jsonl",
    "issues.jsonl",
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in values),
        encoding="utf-8",
    )


def create_stage1b_report_bundle(root: str | Path, zip_path: str | Path) -> Path:
    source = Path(root).resolve(strict=True)
    target = Path(zip_path).resolve(strict=False)
    if source in target.parents:
        raise ValueError("Stage 1B report ZIP must be outside output root")
    members = [name for name in REPORT_MEMBERS if (source / name).is_file()]
    if not members:
        raise FileNotFoundError("No Stage 1B report artifacts found")
    target.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(target, "w", compression=ZIP_DEFLATED) as archive:
        for name in members:
            archive.write(source / name, arcname=name)
    return target
