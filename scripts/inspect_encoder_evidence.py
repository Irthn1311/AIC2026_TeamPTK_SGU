#!/usr/bin/env python3
"""Run only bounded Stage 1B encoder evidence discovery."""

from __future__ import annotations

import argparse
from pathlib import Path

from triage_eg.data.dataset_survey import KAGGLE_INPUT_ROOT, _is_within
from triage_eg.retrieval.stage1b.assets import issue
from triage_eg.retrieval.stage1b.evidence import discover_encoder_evidence
from triage_eg.retrieval.stage1b.writers import write_json, write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--strict-root", action="store_true")
    args = parser.parse_args()
    if args.strict_root and not _is_within(
        args.dataset_root.resolve(strict=False), KAGGLE_INPUT_ROOT
    ):
        raise ValueError("Strict dataset root must be below /kaggle/input")
    records, summary = discover_encoder_evidence(args.repo_root, args.dataset_root)
    write_jsonl(args.output_root / "evidence/evidence_inventory.jsonl", records)
    write_json(args.output_root / "evidence/evidence_summary.json", summary)
    issues = []
    if not summary["authoritative_metadata_found"]:
        issues.append(issue("WARNING", "AUTHORITATIVE_ENCODER_METADATA_NOT_FOUND", None, None))
    write_jsonl(args.output_root / "issues.jsonl", issues)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
