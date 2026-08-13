"""Evaluation-only compact review sheets for M3."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


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


def _inside(frame: int, intervals: list[list[int]]) -> bool:
    return any(start <= frame <= end for start, end in intervals)


def render_review_sheet(
    path: str | Path,
    *,
    case_id: str,
    video_id: str,
    moment_type: str,
    accepted_intervals: list[list[int]],
    predictions: dict[str, int],
    images: dict[int, np.ndarray],
) -> dict[str, Any]:
    """Render GT labels after inference; this module never produces a prediction."""
    from PIL import Image, ImageDraw, ImageOps

    selected: list[int] = []
    for arm in ("m1", "m3_a1", "m3_a2"):
        center = int(predictions[arm])
        available = sorted(images)
        nearest = sorted(available, key=lambda frame: (abs(frame - center), frame))[:3]
        selected.extend(nearest)
    frames = sorted(set(selected))
    if not frames:
        raise ValueError("M3 review sheet has no decoded images")
    tile_w, tile_h, header_h, label_h = 320, 180, 92, 46
    columns = min(6, len(frames))
    rows = (len(frames) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * tile_w, header_h + rows * (tile_h + label_h)), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 10), f"{case_id} | {video_id} | {moment_type}", fill="black", font=_font(20))
    draw.text(
        (12, 45),
        "GT intervals={} | M1={} | A1={} | A2={}".format(
            accepted_intervals,
            predictions["m1"],
            predictions["m3_a1"],
            predictions["m3_a2"],
        ),
        fill="black",
        font=_font(16),
    )
    labels_by_frame: dict[int, list[str]] = {}
    for arm, frame in predictions.items():
        labels_by_frame.setdefault(int(frame), []).append(arm.upper())
    for index, frame in enumerate(frames):
        row, column = divmod(index, columns)
        x, y = column * tile_w, header_h + row * (tile_h + label_h)
        image = Image.fromarray(np.asarray(images[frame], dtype=np.uint8), mode="RGB")
        tile = ImageOps.fit(image, (tile_w, tile_h), method=Image.Resampling.LANCZOS)
        canvas.paste(tile, (x, y))
        if _inside(frame, accepted_intervals):
            draw.rectangle(
                (x + 2, y + 2, x + tile_w - 3, y + tile_h - 3), outline="#00a651", width=6
            )
        tags = ",".join(labels_by_frame.get(frame, [])) or "context"
        gt = "GT-HIT" if _inside(frame, accepted_intervals) else "GT-OUT"
        draw.text(
            (x + 8, y + tile_h + 7), f"frame={frame} | {tags} | {gt}", fill="black", font=_font(14)
        )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target, format="JPEG", quality=88, optimize=True)
    return {"case_id": case_id, "visual_path": target.name, "displayed_frames": frames}


def render_overview_montage(paths: list[Path], target: str | Path) -> Path | None:
    """Stack a small number of already-rendered primary sheets."""
    from PIL import Image, ImageOps

    existing = [path for path in paths if path.is_file()]
    if not existing:
        return None
    opened = [Image.open(path).convert("RGB") for path in existing]
    width = 1280
    resized = []
    for image in opened:
        height = max(1, int(round(image.height * width / image.width)))
        resized.append(ImageOps.fit(image, (width, height)))
    canvas = Image.new("RGB", (width, sum(image.height for image in resized)), "white")
    y = 0
    for image in resized:
        canvas.paste(image, (0, y))
        y += image.height
    output = Path(target)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="JPEG", quality=84, optimize=True)
    for image in opened:
        image.close()
    return output


__all__ = ["render_overview_montage", "render_review_sheet"]
