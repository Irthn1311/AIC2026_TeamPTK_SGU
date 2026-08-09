"""Stage 1D translated-query, comparison-sheet, and bundle artifacts."""

from __future__ import annotations

import csv
import textwrap
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

from triage_eg.retrieval.stage1.search import deduplicate_kis
from triage_eg.retrieval.stage1b.writers import write_json, write_jsonl
from triage_eg.retrieval.stage1c.artifacts import (
    contact_sheet_label,
    grouped_video_diagnostics,
    render_contact_sheet,
)
from triage_eg.retrieval.stage1c.contracts import QueryRecord

CORE_BUNDLE_MEMBERS = (
    "run_manifest.json",
    "stage1d_summary.json",
    "stage1d_report.md",
    "translator/translator_contract.json",
    "translator/translator_runtime_manifest.json",
    "translator/asset_validation.json",
    "translations/translations.jsonl",
    "comparisons/pair_comparisons.jsonl",
    "review/review_template_blinded.csv",
    "review/review_key.json",
    "review/review_instructions.md",
    "issues.jsonl",
)
TRANSLATED_REQUIRED = (
    "translation.json",
    "query.json",
    "ranked_frames.jsonl",
    "ranked_videos.jsonl",
    "kis_candidates.csv",
    "retrieval_diagnostics.json",
)
COMPARISON_REQUIRED = (
    "en_direct_top20.jsonl",
    "vi_direct_top20.jsonl",
    "vi_translated_en_top20.jsonl",
)
FORBIDDEN_SUFFIXES = {
    ".pt", ".pth", ".bin", ".npy", ".npz", ".mp4", ".avi", ".mkv", ".mov"
}


def write_translated_query_artifacts(
    output_root: Path,
    pair_id: str,
    query: QueryRecord,
    encoding: dict[str, Any],
    translation: dict[str, Any],
    frames: list[dict[str, Any]],
    kis_frames: list[dict[str, Any]],
    diagnostics: dict[str, Any],
    dataset_root: Path,
    *,
    contact_sheet_top_k: int,
    skip_contact_sheets: bool,
) -> tuple[dict[str, Path], list[dict[str, Any]]]:
    root = output_root / "translated_queries" / pair_id
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "translation.json", translation)
    write_json(root / "query.json", {**query.__dict__, "encoding": encoding})
    write_jsonl(root / "ranked_frames.jsonl", frames)
    write_jsonl(root / "ranked_videos.jsonl", grouped_video_diagnostics(frames))
    try:
        kis, _ = deduplicate_kis(kis_frames, max_predictions=len(kis_frames))
        with (root / "kis_candidates.csv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=["video_id", "frame_id"])
            writer.writeheader()
            writer.writerows(kis)
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise RuntimeError(f"KIS_EXPORT_FAILED: {error}") from error
    write_json(root / "retrieval_diagnostics.json", diagnostics)
    issues: list[dict[str, Any]] = []
    if not skip_contact_sheets:
        try:
            issues.extend(
                render_contact_sheet(
                    frames[:contact_sheet_top_k],
                    dataset_root,
                    root / "contact_sheet_top20.jpg",
                )
            )
        except (ImportError, OSError, RuntimeError, ValueError) as error:
            issues.append(
                {
                    "severity": "WARNING",
                    "code": "CONTACT_SHEET_RENDER_FAILED",
                    "query_id": query.query_id,
                    "global_row": None,
                    "path": str(root),
                    "message": str(error),
                    "evidence": {},
                }
            )
    return (
        {
            "translation": root / "translation.json",
            "query": root / "query.json",
            "ranked_frames": root / "ranked_frames.jsonl",
            "ranked_videos": root / "ranked_videos.jsonl",
            "kis_candidates": root / "kis_candidates.csv",
            "retrieval_diagnostics": root / "retrieval_diagnostics.json",
        },
        issues,
    )


def _paste_frame(
    sheet: Any,
    draw: Any,
    *,
    item: dict[str, Any],
    dataset_root: Path,
    x: int,
    y: int,
    width: int,
    height: int,
) -> dict[str, Any] | None:
    from PIL import Image

    source = dataset_root / item["keyframe_relative_path"]
    try:
        with Image.open(source) as image:
            preview = image.convert("RGB")
            preview.thumbnail((width, height))
            sheet.paste(
                preview,
                (x + (width - preview.width) // 2, y + (height - preview.height) // 2),
            )
    except (FileNotFoundError, OSError) as error:
        draw.rectangle((x, y, x + width - 1, y + height - 1), fill="#dddddd")
        draw.text((x + 8, y + 8), "IMAGE UNAVAILABLE", fill="black")
        return {
            "severity": "WARNING",
            "code": "KEYFRAME_RESOLUTION_FAILED",
            "query_id": item.get("query_id"),
            "global_row": item.get("global_row"),
            "path": item.get("keyframe_relative_path"),
            "message": str(error),
            "evidence": {},
        }
    return None


def render_comparison_sheet(
    output_path: Path,
    *,
    dataset_root: Path,
    en_text: str,
    vi_text: str,
    translated_text: str,
    arms: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    from PIL import Image, ImageDraw

    width, image_height, label_height, header_height = 320, 180, 34, 150
    arm_names = ("EN_DIRECT", "VI_DIRECT", "VI_TRANSLATED_EN")
    sheet = Image.new(
        "RGB",
        (len(arm_names) * width, header_height + 5 * (image_height + label_height)),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    header = f"EN: {en_text}\nVI: {vi_text}\nMT: {translated_text}"
    draw.multiline_text((8, 6), "\n".join(textwrap.wrap(header, width=120)), fill="black")
    issues: list[dict[str, Any]] = []
    for column, arm in enumerate(arm_names):
        x = column * width
        draw.rectangle((x, 104, x + width - 1, header_height - 1), fill="#eaf0f6")
        draw.text((x + 8, 118), arm, fill="black")
        for index, item in enumerate(arms[arm][:5]):
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
            draw.text(
                (x + 4, y + image_height + 3),
                contact_sheet_label(item),
                fill="black",
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, format="JPEG", quality=85, optimize=True)
    return issues


def write_pair_comparison_artifacts(
    output_root: Path,
    pair_id: str,
    *,
    en_frames: list[dict[str, Any]],
    vi_frames: list[dict[str, Any]],
    translated_frames: list[dict[str, Any]],
    dataset_root: Path,
    en_text: str,
    vi_text: str,
    translated_text: str,
    stage1c_root: Path,
    query_suite_fingerprint: str,
    skip_contact_sheets: bool,
) -> list[dict[str, Any]]:
    root = output_root / "comparisons" / pair_id
    root.mkdir(parents=True, exist_ok=True)
    write_jsonl(root / "en_direct_top20.jsonl", en_frames[:20])
    write_jsonl(root / "vi_direct_top20.jsonl", vi_frames[:20])
    write_jsonl(root / "vi_translated_en_top20.jsonl", translated_frames[:20])
    write_json(
        root / "baseline_provenance.json",
        {
            "stage1c_root": str(stage1c_root),
            "stage1c_query_suite_fingerprint": query_suite_fingerprint,
            "stage1c_query_artifact_paths": {
                "EN_DIRECT": str(
                    stage1c_root
                    / "queries"
                    / str(en_frames[0].get("query_id"))
                    / "ranked_frames.jsonl"
                ),
                "VI_DIRECT": str(
                    stage1c_root
                    / "queries"
                    / str(vi_frames[0].get("query_id"))
                    / "ranked_frames.jsonl"
                ),
            },
            "baseline_records_copied_without_recomputation": True,
        },
    )
    if skip_contact_sheets:
        return []
    try:
        return render_comparison_sheet(
            root / "comparison_top5.jpg",
            dataset_root=dataset_root,
            en_text=en_text,
            vi_text=vi_text,
            translated_text=translated_text,
            arms={
                "EN_DIRECT": en_frames,
                "VI_DIRECT": vi_frames,
                "VI_TRANSLATED_EN": translated_frames,
            },
        )
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        return [
            {
                "severity": "WARNING",
                "code": "CONTACT_SHEET_RENDER_FAILED",
                "query_id": pair_id,
                "global_row": None,
                "path": str(root),
                "message": str(error),
                "evidence": {},
            }
        ]


def create_stage1d_bundle(root: str | Path, zip_path: str | Path) -> Path:
    source = Path(root).expanduser().resolve(strict=True)
    target = Path(zip_path).expanduser().resolve(strict=False)
    if source in target.parents:
        raise ValueError("Stage 1D ZIP must be outside the output root")
    missing = [name for name in CORE_BUNDLE_MEMBERS if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Stage 1D bundle artifacts: {missing}")
    members = list(CORE_BUNDLE_MEMBERS)
    for optional in ("review/review_metrics.json", "review/review_metrics.md"):
        if (source / optional).is_file():
            members.append(optional)
    for query_root in sorted(
        path for path in (source / "translated_queries").iterdir() if path.is_dir()
    ):
        for name in TRANSLATED_REQUIRED:
            path = query_root / name
            if not path.is_file():
                raise FileNotFoundError(f"Missing translated-query artifact: {path}")
            members.append(path.relative_to(source).as_posix())
        sheet = query_root / "contact_sheet_top20.jpg"
        if sheet.is_file():
            members.append(sheet.relative_to(source).as_posix())
    for pair_root in sorted(
        path for path in (source / "comparisons").iterdir() if path.is_dir()
    ):
        for name in COMPARISON_REQUIRED:
            path = pair_root / name
            if not path.is_file():
                raise FileNotFoundError(f"Missing comparison artifact: {path}")
            members.append(path.relative_to(source).as_posix())
        for optional in ("baseline_provenance.json", "comparison_top5.jpg"):
            path = pair_root / optional
            if path.is_file():
                members.append(path.relative_to(source).as_posix())
    if any(
        Path(name).suffix.lower() in FORBIDDEN_SUFFIXES
        or name.startswith(("logs/", "cache/", "caches/"))
        for name in members
    ):
        raise ValueError("Stage 1D ZIP allowlist contains a forbidden artifact")
    target.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(target, "w", compression=ZIP_DEFLATED) as archive:
        for name in sorted(set(members)):
            archive.write(source / name, arcname=name)
    return target
