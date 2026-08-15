"""Bounded evaluation-only D1 review sheets with explicit GT labels."""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import MAX_REVIEW_CASES


def select_review_cases(
    single_rows: list[dict[str, Any]],
    trake_rows: list[dict[str, Any]],
    *,
    max_cases: int = MAX_REVIEW_CASES,
) -> list[tuple[str, str]]:
    if max_cases != MAX_REVIEW_CASES:
        raise ValueError("D1 review cap is frozen at 18")
    selected: list[tuple[str, str]] = []

    def take(rows: list[dict[str, Any]], task: str, reasons: set[str], limit: int) -> None:
        candidates = sorted(
            (
                row
                for row in rows
                if row["task"] == task and row["primary_failure_reason"] in reasons
            ),
            key=lambda row: row["query_id"],
        )
        for row in candidates[:limit]:
            key = task, row["query_id"]
            if key not in selected and len(selected) < max_cases:
                selected.append(key)

    take(single_rows, "KIS", {"TARGET_SEMANTIC_SCORE_WEAK", "T3_REGION_REPRESENTATIVE_GAP"}, 3)
    take(single_rows, "KIS", {"BTC_REPRESENTATION_GAP"}, 2)
    take(single_rows, "KIS", {"GLOBAL_VIDEO_RANKING_GAP", "G1_ALLOCATION_GAP"}, 2)
    take(single_rows, "QA", {"TARGET_SEMANTIC_SCORE_WEAK", "T3_REGION_REPRESENTATIVE_GAP"}, 2)
    take(
        single_rows,
        "QA",
        {"BTC_REPRESENTATION_GAP", "GLOBAL_VIDEO_RANKING_GAP", "G1_ALLOCATION_GAP"},
        1,
    )
    take(trake_rows, "TRAKE", {"EVENT_SEMANTIC_SCORE_GAP", "T3_EVENT_POOL_GAP"}, 3)
    take(trake_rows, "TRAKE", {"MONOTONIC_COMPOSITION_GAP"}, 2)
    take(trake_rows, "TRAKE", {"GLOBAL_CHAIN_RANKING_GAP"}, 2)
    controls = sorted(
        [
            row
            for row in [*single_rows, *trake_rows]
            if row["primary_failure_reason"] in {"SUCCESS_G1_TARGET_HIT", "SUCCESS_FULL_CHAIN"}
        ],
        key=lambda row: (row["task"], row["query_id"]),
    )
    if controls and len(selected) < max_cases:
        selected.append((controls[0]["task"], controls[0]["query_id"]))
    return selected


def _font(ImageFont: Any, size: int) -> Any:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_text(draw: Any, position: tuple[int, int], text: str, *, font: Any, fill: str) -> None:
    try:
        draw.text(position, text, font=font, fill=fill)
    except UnicodeEncodeError:
        draw.text(position, text.encode("ascii", "replace").decode(), font=font, fill=fill)


def _panel(
    pipeline: Any,
    Image: Any,
    ImageDraw: Any,
    font: Any,
    *,
    video_id: str,
    frame: int | None,
    label: str,
    score: float | None = None,
) -> Any:
    output = Image.new("RGB", (330, 245), "white")
    draw = ImageDraw.Draw(output)
    _draw_text(draw, (8, 6), label, font=font, fill="black")
    _draw_text(
        draw,
        (8, 24),
        f"{video_id} frame={frame} score={score}",
        font=font,
        fill="black",
    )
    if frame is None:
        return output
    try:
        image = Image.fromarray(pipeline._decode_image(video_id, int(frame))).convert("RGB")
        image.thumbnail((314, 196))
        output.paste(image, ((330 - image.width) // 2, 45))
    except (IndexError, OSError, RuntimeError, ValueError):
        _draw_text(draw, (8, 55), "FRAME UNAVAILABLE", font=font, fill="red")
    return output


def _header(Image: Any, ImageDraw: Any, font: Any, lines: list[str], width: int) -> Any:
    wrapped = [part for line in lines for part in textwrap.wrap(line, width=145) or [""]]
    output = Image.new("RGB", (width, 28 + 18 * len(wrapped)), "white")
    draw = ImageDraw.Draw(output)
    for index, line in enumerate(wrapped):
        _draw_text(
            draw, (8, 6 + 18 * index), line, font=font, fill="red" if index == 0 else "black"
        )
    return output


def _translation_map(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["unit_id"]: row for row in rows}


def render_review_sheets(
    pipeline: Any,
    audit: dict[str, Any],
    blind_rows: list[dict[str, Any]],
    output_root: str | Path,
) -> list[Path]:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return []
    selected = select_review_cases(audit["single_rows"], audit["trake_query_rows"])
    single = {row["query_id"]: row for row in audit["single_rows"]}
    trake = {row["query_id"]: row for row in audit["trake_query_rows"]}
    translations = _translation_map(blind_rows)
    root = Path(output_root)
    font = _font(ImageFont, 13)
    rendered: list[Path] = []
    task_images: dict[str, list[Any]] = {"KIS": [], "QA": [], "TRAKE": []}
    for task, query_id in selected:
        if task in {"KIS", "QA"}:
            row = single[query_id]
            translation = translations[f"{query_id}:E1"]
            top = row.get("g1_top_prediction") or {}
            panels = [
                _panel(
                    pipeline,
                    Image,
                    ImageDraw,
                    font,
                    video_id=row["correct_video"],
                    frame=row["best_target_frame_idx"],
                    label="BEST TARGET BTC INSIDE GT",
                    score=row["best_target_score"],
                ),
                _panel(
                    pipeline,
                    Image,
                    ImageDraw,
                    font,
                    video_id=row["correct_video"],
                    frame=row["correct_video_best_frame_idx"],
                    label="CORRECT VIDEO HIGHEST SCORE",
                    score=row["correct_video_best_score"],
                ),
                _panel(
                    pipeline,
                    Image,
                    ImageDraw,
                    font,
                    video_id=row["correct_video"],
                    frame=row["nearest_t3_candidate_frame"],
                    label="NEAREST FROZEN T3 CANDIDATE",
                ),
                _panel(
                    pipeline,
                    Image,
                    ImageDraw,
                    font,
                    video_id=str(top.get("video_id", "NONE")),
                    frame=top.get("frame_id"),
                    label="CURRENT G1 TOP1 / DISTRACTOR",
                    score=top.get("retrieval_score"),
                ),
            ]
            width = sum(panel.width for panel in panels)
            header = _header(
                Image,
                ImageDraw,
                font,
                [
                    "EVALUATION ONLY - GT EXPOSED",
                    f"{query_id} {task} reason={row['primary_failure_reason']}",
                    f"VI: {translation['source_vi']}",
                    f"OPUS: {translation['opus_en']}",
                    f"GT video={row['correct_video']} intervals={row['gt_intervals']}",
                ],
                width,
            )
            canvas = Image.new("RGB", (width, header.height + 245), "white")
            canvas.paste(header, (0, 0))
            x = 0
            for panel in panels:
                canvas.paste(panel, (x, header.height))
                x += panel.width
        else:
            row = trake[query_id]
            current = (
                row.get("g1_best_correct_video_prediction") or row.get("g1_top_prediction") or {}
            )
            frames = current.get("frame_ids", [])
            panels = []
            text_lines = [
                "EVALUATION ONLY - GT EXPOSED",
                f"{query_id} TRAKE reason={row['primary_failure_reason']}",
                f"BTC_TARGET_CHAIN_EXISTS={row['btc_target_chain_exists']} "
                f"T3_TARGET_CHAIN_EXISTS={row['t3_target_chain_exists']} "
                f"BEST_CORRECT_VIDEO_CHAIN_GLOBAL_RANK={row['best_correct_video_chain_global_rank']}",
            ]
            for index, event in enumerate(row["events"]):
                translation = translations[f"{query_id}:{event['event_id']}"]
                text_lines.extend(
                    [
                        f"{event['event_id']} VI: {translation['source_vi']}",
                        f"{event['event_id']} OPUS: {translation['opus_en']} "
                        f"GT={event['gt_intervals']}",
                    ]
                )
                panels.extend(
                    [
                        _panel(
                            pipeline,
                            Image,
                            ImageDraw,
                            font,
                            video_id=row["correct_video"],
                            frame=event["best_target_frame_idx"],
                            label=f"{event['event_id']} BEST TARGET",
                            score=event["best_target_score"],
                        ),
                        _panel(
                            pipeline,
                            Image,
                            ImageDraw,
                            font,
                            video_id=row["correct_video"],
                            frame=event["nearest_t3_candidate_frame"],
                            label=f"{event['event_id']} NEAREST T3",
                        ),
                        _panel(
                            pipeline,
                            Image,
                            ImageDraw,
                            font,
                            video_id=str(current.get("video_id", "NONE")),
                            frame=frames[index] if index < len(frames) else None,
                            label=f"{event['event_id']} CURRENT CHAIN",
                        ),
                    ]
                )
            width = 990
            header = _header(Image, ImageDraw, font, text_lines, width)
            rows = int(np.ceil(len(panels) / 3))
            canvas = Image.new("RGB", (width, header.height + rows * 245), "white")
            canvas.paste(header, (0, 0))
            for index, panel in enumerate(panels):
                canvas.paste(panel, ((index % 3) * 330, header.height + (index // 3) * 245))
        path = root / "review" / f"{task.casefold()}_{query_id}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(path, quality=86)
        rendered.append(path)
        task_images[task].append(canvas)
    for task, images in task_images.items():
        if not images:
            continue
        width = max(image.width for image in images)
        total_height = sum(image.height for image in images)
        montage = Image.new("RGB", (width, total_height), "white")
        y = 0
        for image in images:
            montage.paste(image, (0, y))
            y += image.height
        path = root / "montages" / f"d1_{task.casefold()}_review.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        montage.save(path, quality=82)
        rendered.append(path)
    return rendered


__all__ = ["render_review_sheets", "select_review_cases"]
