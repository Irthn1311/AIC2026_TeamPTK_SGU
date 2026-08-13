"""Summarize the first N files from each Kaggle output root.

This is a lightweight sanity check for downloaded Kaggle artifacts:
it prints a compact table and writes JSON/Markdown manifests so you can
inspect whether the run produced the expected file families before pulling
everything back to local disk.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class OutputGroup:
    name: str
    root: Path


def _resolve(path: str | Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def _format_size(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f} KB"
    if num_bytes < 1024 * 1024 * 1024:
        return f"{num_bytes / (1024 * 1024):.1f} MB"
    return f"{num_bytes / (1024 * 1024 * 1024):.2f} GB"


def _collect_files(root: Path, limit: int) -> list[Path]:
    if not root.exists():
        return []
    files = [path for path in root.rglob("*") if path.is_file()]
    files.sort(key=lambda path: (path.stat().st_mtime, str(path)), reverse=True)
    return files[:limit]


def _count_files(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(1 for path in root.rglob("*") if path.is_file())


def _rel(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _build_groups() -> list[OutputGroup]:
    output_root = _resolve(os.environ.get("AIC_OUTPUT_ROOT", "artifacts"))
    return [
        OutputGroup("artifacts", output_root),
        OutputGroup("keyframes", _resolve(os.environ.get("AIC_KEYFRAME_OUTPUT_ROOT", output_root / "keyframe_v2_full"))),
        OutputGroup("ocr_v2_selected", _resolve(os.environ.get("AIC_OCR_V2_OUTPUT_ROOT", output_root / "ocr_v2_selected_keyframes"))),
        OutputGroup("ocr_temporal_v3", _resolve(os.environ.get("AIC_OCR_TEMPORAL_OUTPUT_ROOT", output_root / "ocr_temporal_v3_full_tracking"))),
        OutputGroup("asr", _resolve(os.environ.get("AIC_ASR_OUTPUT_ROOT", output_root / "asr"))),
        OutputGroup("objects", _resolve(os.environ.get("AIC_OBJECT_OUTPUT_ROOT", output_root / "keyframe_v2_full" / "object_v2"))),
        OutputGroup("indexes", _resolve(os.environ.get("AIC_INDEX_OUTPUT_ROOT", output_root / "indexes"))),
        OutputGroup("packages", output_root),
        OutputGroup("stage0_audit", _resolve(os.environ.get("AIC_AUDIT_OUTPUT_ROOT", "triage_eg_stage0_audit"))),
        OutputGroup("stage1_baseline", _resolve(os.environ.get("AIC_STAGE1_OUTPUT_ROOT", "triage_eg_stage1_baseline"))),
    ]


def _render_markdown(groups: Iterable[dict[str, object]], output_root: Path, limit: int) -> str:
    lines = [
        "# Kaggle output manifest",
        f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"Limit per group: {limit}",
        "",
    ]
    for group in groups:
        lines.append(f"## {group['name']}")
        lines.append(f"Root: `{group['root']}`")
        lines.append(f"Files found: {group['file_count']}")
        lines.append(f"Samples shown: {group['sample_count']}")
        lines.append("")
        lines.append("| # | File | Size | Modified UTC |")
        lines.append("|---:|---|---:|---|")
        for row in group["samples"]:
            lines.append(
                f"| {row['rank']} | `{row['path']}` | {row['size']} | {row['modified_utc']} |"
            )
        lines.append("")
    lines.append(f"Manifest root: `{output_root}`")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=20, help="Maximum files to list per output root.")
    parser.add_argument(
        "--output-root",
        default=os.environ.get("AIC_OUTPUT_ROOT", "artifacts"),
        help="Where to write the manifest files.",
    )
    args = parser.parse_args()

    output_root = _resolve(args.output_root)
    manifest_dir = output_root / "kaggle_manifest"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    payload: dict[str, object] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "limit_per_group": int(args.limit),
        "groups": [],
    }

    group_rows: list[dict[str, object]] = []
    for group in _build_groups():
        total_files = _count_files(group.root)
        files = _collect_files(group.root, args.limit)
        samples = []
        for idx, path in enumerate(files, start=1):
            stat = path.stat()
            samples.append(
                {
                    "rank": idx,
                    "path": _rel(group.root, path),
                    "size_bytes": stat.st_size,
                    "size": _format_size(stat.st_size),
                    "modified_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(timespec="seconds"),
                }
            )
        group_data = {
            "name": group.name,
            "root": str(group.root),
            "exists": group.root.exists(),
            "file_count": total_files,
            "sample_count": len(files),
            "samples": samples,
        }
        group_rows.append(group_data)
        payload["groups"].append(group_data)

    (manifest_dir / "kaggle_output_manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (manifest_dir / "kaggle_output_manifest.md").write_text(
        _render_markdown(group_rows, manifest_dir, args.limit),
        encoding="utf-8",
    )

    for group in group_rows:
        print(
            f"[{group['name']}] root={group['root']} "
            f"files={group['file_count']} sample_count={group['sample_count']}"
        )
        for row in group["samples"]:
            print(f"  {row['rank']:>2}. {row['path']}  ({row['size']})")
    print(f"Wrote manifest to {manifest_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
