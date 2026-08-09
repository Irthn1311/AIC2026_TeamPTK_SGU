"""Compact semantic/temporal visual sheets for RT1 review."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from triage_eg.retrieval.stage1d.artifacts import _paste_frame


def _font(size: int):
    from PIL import ImageFont

    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _anchors(item: dict[str, Any], method: str) -> list[dict[str, Any]]:
    return item["event_best"] if method == "UNORDERED_EVENT_MAX" else item["chain"]


def _render_temporal_sheet(
    output_path: Path,
    *,
    dataset_root: Path,
    query_id: str,
    method_label: str,
    method: str,
    ranked: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    from PIL import Image, ImageDraw

    selected = ranked[:top_k]
    if not selected:
        raise ValueError(f"No {method} videos available for visualization")
    event_count = len(_anchors(selected[0], method))
    tile_width, image_height, label_height, header_height = 300, 170, 54, 66
    sheet = Image.new(
        "RGB",
        (
            event_count * tile_width,
            header_height + len(selected) * (image_height + label_height),
        ),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    draw.text((10, 8), f"{query_id} · {method_label}", fill="black", font=_font(22))
    issues = []
    for row_index, video in enumerate(selected):
        anchors = _anchors(video, method)
        for column, anchor in enumerate(anchors):
            x = column * tile_width
            y = header_height + row_index * (image_height + label_height)
            issue = _paste_frame(
                sheet,
                draw,
                item={**anchor, "query_id": query_id},
                dataset_root=dataset_root,
                x=x,
                y=y,
                width=tile_width,
                height=image_height,
            )
            if issue:
                issues.append(issue)
            label = (
                f"{video['video_id']} · {anchor['event_id']} · rank={video['video_rank']}\n"
                f"frame={anchor['original_frame_idx']}"
            )
            draw.multiline_text(
                (x + 6, y + image_height + 4),
                label,
                fill="black",
                font=_font(15),
                spacing=2,
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, format="JPEG", quality=88, optimize=True)
    return issues


def render_whole_query_frames(
    output_path: Path,
    *,
    dataset_root: Path,
    query_id: str,
    frames: list[dict[str, Any]],
    top_k: int = 10,
) -> list[dict[str, Any]]:
    from PIL import Image, ImageDraw

    selected = frames[:top_k]
    columns = min(5, len(selected))
    rows = (len(selected) + columns - 1) // columns
    tile_width, image_height, label_height, header_height = 300, 170, 52, 64
    sheet = Image.new(
        "RGB",
        (columns * tile_width, header_height + rows * (image_height + label_height)),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    draw.text((10, 8), f"{query_id} · WHOLE_QUERY", fill="black", font=_font(22))
    issues = []
    for index, item in enumerate(selected):
        column, row = index % columns, index // columns
        x, y = column * tile_width, header_height + row * (image_height + label_height)
        issue = _paste_frame(
            sheet,
            draw,
            item=item,
            dataset_root=dataset_root,
            x=x,
            y=y,
            width=tile_width,
            height=image_height,
        )
        if issue:
            issues.append(issue)
        draw.multiline_text(
            (x + 6, y + image_height + 4),
            f"{item['video_id']} · rank={item['rank']}\nframe={item['original_frame_idx']}",
            fill="black",
            font=_font(15),
            spacing=2,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, format="JPEG", quality=88, optimize=True)
    return issues


def blinded_method_mapping(query_id: str, seed: int) -> dict[str, str]:
    digest = hashlib.sha256(f"{seed}:{query_id}".encode()).digest()
    ordered = (
        ("DANTE_DP", "UNORDERED_EVENT_MAX")
        if digest[0] & 1
        else ("UNORDERED_EVENT_MAX", "DANTE_DP")
    )
    return {"METHOD_A": ordered[0], "METHOD_B": ordered[1]}


def render_rt1_visuals(
    output_root: Path,
    *,
    dataset_root: Path,
    query_id: str,
    whole_frames: list[dict[str, Any]],
    unordered: list[dict[str, Any]],
    dante: list[dict[str, Any]],
    top_k: int,
    review_seed: int,
) -> tuple[dict[str, Path], dict[str, str], list[dict[str, Any]]]:
    from PIL import Image

    root = output_root / "visuals" / query_id
    unordered_path = root / "unordered_top3.jpg"
    dante_path = root / "dante_top3.jpg"
    whole_path = root / "whole_query_top_frames.jpg"
    issues = []
    issues.extend(
        _render_temporal_sheet(
            unordered_path,
            dataset_root=dataset_root,
            query_id=query_id,
            method_label="UNORDERED_EVENT_MAX",
            method="UNORDERED_EVENT_MAX",
            ranked=unordered,
            top_k=top_k,
        )
    )
    issues.extend(
        _render_temporal_sheet(
            dante_path,
            dataset_root=dataset_root,
            query_id=query_id,
            method_label="DANTE_DP",
            method="DANTE_DP",
            ranked=dante,
            top_k=top_k,
        )
    )
    issues.extend(
        render_whole_query_frames(
            whole_path,
            dataset_root=dataset_root,
            query_id=query_id,
            frames=whole_frames,
        )
    )
    mapping = blinded_method_mapping(query_id, review_seed)
    method_paths = {
        "UNORDERED_EVENT_MAX": unordered_path,
        "DANTE_DP": dante_path,
    }
    with (
        Image.open(method_paths[mapping["METHOD_A"]]) as left,
        Image.open(method_paths[mapping["METHOD_B"]]) as right,
    ):
        header_height = 54
        left_body = left.crop((0, 66, left.width, left.height))
        right_body = right.crop((0, 66, right.width, right.height))
        sheet = Image.new(
            "RGB",
            (
                left_body.width + right_body.width,
                header_height + max(left_body.height, right_body.height),
            ),
            "white",
        )
        from PIL import ImageDraw

        draw = ImageDraw.Draw(sheet)
        draw.text((10, 8), "METHOD_A", fill="black", font=_font(22))
        draw.text((left_body.width + 10, 8), "METHOD_B", fill="black", font=_font(22))
        sheet.paste(left_body, (0, header_height))
        sheet.paste(right_body, (left_body.width, header_height))
        ab_path = root / "ab_temporal_comparison.jpg"
        sheet.save(ab_path, format="JPEG", quality=88, optimize=True)
    return (
        {
            "unordered": unordered_path,
            "dante": dante_path,
            "whole_query": whole_path,
            "ab": ab_path,
        },
        mapping,
        issues,
    )


__all__ = ["blinded_method_mapping", "render_rt1_visuals"]
