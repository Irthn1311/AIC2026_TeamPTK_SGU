"""Review-only Stage 1D v0.1.1 blinded visual patching.

This module consumes frozen Stage 1D records and canonical keyframes. It does
not import or invoke translation, text encoding, or retrieval code.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import textwrap
from collections import defaultdict
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from triage_eg.retrieval.stage1b.writers import write_json

from .artifacts import _paste_frame
from .contracts import ARMS, STAGE1D_REVIEW_PATCH_VERSION
from .review import REVIEW_FIELDS, review_instructions

CONDITIONS = ("C01", "C02", "C03")
ARM_FILES = {
    "EN_DIRECT": "en_direct_top20.jsonl",
    "VI_DIRECT": "vi_direct_top20.jsonl",
    "VI_TRANSLATED_EN": "vi_translated_en_top20.jsonl",
}
INDEX_FIELDS = (
    "pair_id",
    "sheet_path",
    "condition_codes",
    "ranks",
    "en_reference_text",
    "vi_original_text",
)
PATCH_REPORT_MARKER = "<!-- STAGE1D_REVIEW_VISUAL_PATCH_V0_1_1 -->"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid JSON artifact {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        values = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid JSONL artifact {path}: {error}") from error
    if any(not isinstance(item, dict) for item in values):
        raise ValueError(f"Expected JSON objects in {path}")
    return values


def _read_review_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != REVIEW_FIELDS:
            raise ValueError("REVIEW_IDENTITY_MISMATCH: review schema changed")
        return list(reader)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _frozen_inventory(root: Path) -> dict[str, str]:
    members = [
        root / "translations/translations.jsonl",
        root / "comparisons/pair_comparisons.jsonl",
    ]
    for pair_root in sorted(path for path in (root / "comparisons").iterdir() if path.is_dir()):
        members.extend(pair_root / name for name in ARM_FILES.values())
    for query_root in sorted(
        path for path in (root / "translated_queries").iterdir() if path.is_dir()
    ):
        members.extend(
            query_root / name
            for name in (
                "translation.json",
                "query.json",
                "ranked_frames.jsonl",
                "ranked_videos.jsonl",
                "kis_candidates.csv",
                "retrieval_diagnostics.json",
            )
        )
    missing = [str(path) for path in members if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Frozen Stage 1D artifact missing: {missing}")
    return {path.relative_to(root).as_posix(): _sha256(path) for path in sorted(set(members))}


def _inventory_fingerprint(inventory: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for path, file_hash in sorted(inventory.items()):
        digest.update(f"{path}\0{file_hash}\n".encode())
    return digest.hexdigest()


def _font(size: int):
    from PIL import ImageFont

    candidates = [
        os.environ.get("AIC_REVIEW_FONT_PATH"),
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "DejaVuSans.ttf",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def blinded_header_text(pair_id: str, en_text: str, vi_text: str) -> str:
    """Return the complete, intentionally arm-free sheet header."""

    lines = [f"Pair: {pair_id}"]
    lines.extend(textwrap.wrap(f"English intent: {en_text}", width=105))
    lines.extend(textwrap.wrap(f"Vietnamese intent: {vi_text}", width=105))
    return "\n".join(lines)


def blinded_tile_label(condition_code: str, item: dict[str, Any]) -> str:
    """Return the identity label used to map one tile to the review CSV."""

    return (
        f"{condition_code} · #{int(item['rank'])}\n"
        f"{item['video_id']} · frame={int(item['original_frame_idx'])}"
    )


def render_blinded_review_sheet(
    output_path: Path,
    *,
    dataset_root: Path,
    pair_id: str,
    en_text: str,
    vi_text: str,
    conditions: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Render one C01/C02/C03 Top-5 sheet without accepting arm labels or MT text."""

    from PIL import Image, ImageDraw

    if tuple(conditions) != CONDITIONS:
        raise ValueError("Blinded conditions must be ordered C01, C02, C03")
    for code, records in conditions.items():
        if len(records) < 5 or [int(item["rank"]) for item in records[:5]] != list(range(1, 6)):
            raise ValueError(f"Frozen Top-5 is invalid for {pair_id}/{code}")
    width, image_height, label_height, header_height = 360, 200, 56, 150
    sheet = Image.new(
        "RGB",
        (len(CONDITIONS) * width, header_height + 5 * (image_height + label_height)),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    header_font, condition_font, label_font = _font(18), _font(22), _font(15)
    draw.multiline_text(
        (12, 8),
        blinded_header_text(pair_id, en_text, vi_text),
        fill="black",
        font=header_font,
        spacing=4,
    )
    issues: list[dict[str, Any]] = []
    for column, condition_code in enumerate(CONDITIONS):
        x = column * width
        draw.rectangle((x, 108, x + width - 1, header_height - 1), fill="#eaf0f6")
        draw.text((x + 12, 116), condition_code, fill="black", font=condition_font)
        for index, item in enumerate(conditions[condition_code][:5]):
            y = header_height + index * (image_height + label_height)
            issue = _paste_frame(
                sheet,
                draw,
                item=item,
                dataset_root=dataset_root,
                x=x,
                y=y,
                width=width,
                height=image_height,
            )
            if issue:
                issues.append(issue)
            draw.multiline_text(
                (x + 7, y + image_height + 4),
                blinded_tile_label(condition_code, item),
                fill="black",
                font=label_font,
                spacing=2,
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, format="JPEG", quality=88, optimize=True)
    return issues


def _decimal_equal(left: Any, right: Any) -> bool:
    try:
        return Decimal(str(left)) == Decimal(str(right))
    except InvalidOperation:
        return False


def _validate_review_identity(
    pair_id: str,
    condition_code: str,
    rows: list[dict[str, str]],
    records: list[dict[str, Any]],
) -> None:
    if len(rows) != 5 or [int(row["rank"]) for row in rows] != list(range(1, 6)):
        raise ValueError(f"REVIEW_IDENTITY_MISMATCH: invalid ranks for {pair_id}/{condition_code}")
    for row, item in zip(rows, records[:5], strict=True):
        expected = {
            "video_id": str(item["video_id"]),
            "global_row": str(item["global_row"]),
            "n": str(item["n"]),
            "original_frame_idx": str(item["original_frame_idx"]),
        }
        if any(row[name] != value for name, value in expected.items()) or not _decimal_equal(
            row["score"], item["score"]
        ):
            raise ValueError(
                f"REVIEW_IDENTITY_MISMATCH: frozen record differs for "
                f"{pair_id}/{condition_code}/rank={row['rank']}"
            )


def _report_patch_section(sheet_count: int, frozen_fingerprint: str) -> str:
    return f"""

{PATCH_REPORT_MARKER}
## Stage 1D v0.1.1 Review Presentation Patch

- Patch scope: `REVIEW_PRESENTATION_ONLY`
- Retrieval source: `FROZEN_STAGE1D_V0_1_0`
- Retrieval regenerated: false
- Translation regenerated: false
- Baseline regenerated: false
- `ENGINEERING_UNBLINDED_SHEET`: `comparisons/<pair_id>/comparison_top5.jpg`
- `HUMAN_REVIEW_BLINDED_SHEET`: `review/blinded_sheets/<pair_id>_top5.jpg`
- Blinded sheets: {sheet_count}
- Frozen retrieval artifact fingerprint: `{frozen_fingerprint}`
- Formal human review executability: `READY`
- Human review status: `NOT_REVIEWED`
- Language bridge quality status: `NOT_REVIEWED`

Do not open `review/review_key.json` or engineering comparison sheets while
performing the blinded review.
"""


def patch_blinded_review_visuals(
    stage1d_root: str | Path,
    dataset_root: str | Path,
    *,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    """Patch a frozen v0.1.0 output in place or copy it to a new output root."""

    source = Path(stage1d_root).expanduser().resolve(strict=True)
    dataset = Path(dataset_root).expanduser().resolve(strict=True)
    destination = (
        Path(output_root).expanduser().resolve(strict=False) if output_root is not None else source
    )
    if destination != source:
        if destination.exists():
            raise FileExistsError(f"Review patch output already exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, destination)
    summary_path = destination / "stage1d_summary.json"
    manifest_path = destination / "run_manifest.json"
    report_path = destination / "stage1d_report.md"
    template_path = destination / "review/review_template_blinded.csv"
    key_path = destination / "review/review_key.json"
    for required in (summary_path, manifest_path, report_path, template_path, key_path):
        if not required.is_file():
            raise FileNotFoundError(f"Frozen Stage 1D artifact missing: {required}")
    summary, manifest, key = (
        _read_json(summary_path),
        _read_json(manifest_path),
        _read_json(key_path),
    )
    if summary.get("execution_status") not in {"COMPLETE", "COMPLETE_WITH_WARNINGS"}:
        raise ValueError("Frozen Stage 1D execution is not complete")
    if summary.get("human_review", {}).get("status") != "NOT_REVIEWED":
        raise ValueError("Review visual patch requires an unreviewed frozen run")
    original_version = str(
        summary.get("original_retrieval_run", {}).get("stage1d_version")
        or summary.get("stage1d_version")
    )
    if original_version != "0.1.0":
        raise ValueError(f"Expected frozen Stage 1D v0.1.0, found {original_version}")
    rows = _read_review_rows(template_path)
    if any(row["review_label"].strip() or row["review_notes"].strip() for row in rows):
        raise ValueError("Review visual patch requires a blank review template")
    mappings = {
        item["pair_id"]: item["conditions"]
        for item in key.get("pairs", [])
        if isinstance(item, dict)
    }
    if not mappings or any(
        tuple(sorted(value)) != CONDITIONS or set(value.values()) != set(ARMS)
        for value in mappings.values()
    ):
        raise ValueError("REVIEW_IDENTITY_MISMATCH: invalid review key")
    expected_pairs = int(summary.get("stage1c_frozen_baseline", {}).get("pairs_selected", 0))
    if len(mappings) != expected_pairs or len(rows) != expected_pairs * 15:
        raise ValueError("REVIEW_IDENTITY_MISMATCH: pair or row count mismatch")
    rows_by_pair_condition: defaultdict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        rows_by_pair_condition[(row["pair_id"], row["condition_code"])].append(row)
    before = _frozen_inventory(destination)
    before_key_hash = _sha256(key_path)
    staging = destination / "review/.blinded_sheets.building"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    index_rows: list[dict[str, str]] = []
    try:
        for pair_id, condition_map in sorted(mappings.items()):
            pair_rows = [row for row in rows if row["pair_id"] == pair_id]
            if len(pair_rows) != 15:
                raise ValueError(f"REVIEW_IDENTITY_MISMATCH: expected 15 rows for {pair_id}")
            en_text = pair_rows[0]["en_reference_text"]
            vi_text = pair_rows[0]["vi_original_text"]
            if any(
                row["en_reference_text"] != en_text or row["vi_original_text"] != vi_text
                for row in pair_rows
            ):
                raise ValueError(f"REVIEW_IDENTITY_MISMATCH: intent drift for {pair_id}")
            conditions: dict[str, list[dict[str, Any]]] = {}
            for code in CONDITIONS:
                arm = condition_map[code]
                records = _read_jsonl(destination / "comparisons" / pair_id / ARM_FILES[arm])
                review_rows = sorted(
                    rows_by_pair_condition[(pair_id, code)], key=lambda item: int(item["rank"])
                )
                _validate_review_identity(pair_id, code, review_rows, records)
                conditions[code] = records[:5]
            sheet_name = f"{pair_id}_top5.jpg"
            issues = render_blinded_review_sheet(
                staging / sheet_name,
                dataset_root=dataset,
                pair_id=pair_id,
                en_text=en_text,
                vi_text=vi_text,
                conditions=conditions,
            )
            if issues:
                raise RuntimeError(f"Blinded review rendering failed for {pair_id}: {issues}")
            index_rows.append(
                {
                    "pair_id": pair_id,
                    "sheet_path": f"review/blinded_sheets/{sheet_name}",
                    "condition_codes": "C01|C02|C03",
                    "ranks": "1|2|3|4|5",
                    "en_reference_text": en_text,
                    "vi_original_text": vi_text,
                }
            )
        target = destination / "review/blinded_sheets"
        if target.exists():
            shutil.rmtree(target)
        os.replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    index_path = destination / "review/blinded_sheet_index.csv"
    with index_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(INDEX_FIELDS))
        writer.writeheader()
        writer.writerows(index_rows)
    (destination / "review/review_instructions.md").write_text(
        review_instructions(blinded_visuals=True), encoding="utf-8"
    )
    after = _frozen_inventory(destination)
    if before != after or before_key_hash != _sha256(key_path):
        raise RuntimeError("Frozen rankings, translations, or review key changed during patch")
    frozen_fingerprint = _inventory_fingerprint(before)
    patch_metadata = {
        "patch_version": STAGE1D_REVIEW_PATCH_VERSION,
        "patch_scope": "REVIEW_PRESENTATION_ONLY",
        "retrieval_source": "FROZEN_STAGE1D_V0_1_0",
        "retrieval_regenerated": False,
        "translation_regenerated": False,
        "baseline_regenerated": False,
        "frozen_artifact_fingerprint": frozen_fingerprint,
        "blinded_sheet_count": len(index_rows),
        "blinded_sheet_index": "review/blinded_sheet_index.csv",
        "formal_human_review_executability": "READY",
        "patched_at": datetime.now(UTC).isoformat(),
    }
    summary["original_retrieval_run"] = {
        "stage1d_version": original_version,
        "build_git_commit": summary.get("build_git_commit"),
        "execution_status": summary.get("execution_status"),
        "stage1_index_fingerprint": summary.get("stage1_index_fingerprint"),
        "query_suite_fingerprint": summary.get("stage1c_frozen_baseline", {}).get(
            "query_suite_fingerprint"
        ),
    }
    summary["stage1d_version"] = STAGE1D_REVIEW_PATCH_VERSION
    summary["review_visual_patch"] = patch_metadata
    summary["human_review"]["formal_executability"] = "READY"
    summary["human_review"]["blinded_sheet_count"] = len(index_rows)
    summary["human_review"]["blinded_sheet_index"] = "review/blinded_sheet_index.csv"
    write_json(summary_path, summary)
    manifest["original_retrieval_run"] = {
        "stage1d_version": original_version,
        "build_git_commit": manifest.get("build_git_commit"),
        "started_at": manifest.get("started_at"),
        "completed_at": manifest.get("completed_at"),
    }
    manifest["stage1d_version"] = STAGE1D_REVIEW_PATCH_VERSION
    manifest["review_visual_patch"] = patch_metadata
    write_json(manifest_path, manifest)
    report = report_path.read_text(encoding="utf-8")
    if PATCH_REPORT_MARKER in report:
        report = report.split(PATCH_REPORT_MARKER, 1)[0].rstrip()
    report_path.write_text(
        report + _report_patch_section(len(index_rows), frozen_fingerprint), encoding="utf-8"
    )
    return {
        "output_root": destination,
        **patch_metadata,
        "review_rows": len(rows),
        "review_key_sha256": before_key_hash,
    }


__all__ = [
    "blinded_header_text",
    "blinded_tile_label",
    "patch_blinded_review_visuals",
    "render_blinded_review_sheet",
]
