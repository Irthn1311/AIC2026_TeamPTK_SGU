"""Deterministic blinded visuals for M1 frame-localization review."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

M0_METHOD = "BTC_TECHNICAL_KEYFRAME_ANCHOR"
M1_METHOD = "LOCAL_RAW_CLIP_COARSE_TO_FINE"


def blinded_mapping(query_id: str, event_id: str, seed: int = 2026) -> dict[str, str]:
    digest = hashlib.sha256(f"{seed}|{query_id}|{event_id}".encode()).digest()
    if digest[0] % 2:
        return {"METHOD_A": M1_METHOD, "METHOD_B": M0_METHOD}
    return {"METHOD_A": M0_METHOD, "METHOD_B": M1_METHOD}


def _font(size: int):
    from PIL import ImageFont

    for name in ("DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _rgb_image(value: np.ndarray, size: tuple[int, int]):
    from PIL import Image

    frame = np.asarray(value, dtype=np.uint8)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("M1 visual frame must be an RGB HxWx3 array")
    image = Image.fromarray(frame, mode="RGB")
    image.thumbnail(size)
    canvas = Image.new("RGB", size, "white")
    canvas.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return canvas


def render_blinded_event_sheet(
    output_path: Path,
    *,
    query_id: str,
    event_id: str,
    event_text: str,
    video_id: str,
    coarse_frame_idx: int,
    coarse_image: np.ndarray,
    refined_frame_idx: int,
    refined_image: np.ndarray,
    mapping: dict[str, str],
) -> None:
    """Render one blind A/B sheet without method, score, reference, or error labels."""

    from PIL import Image, ImageDraw

    method_values = {
        M0_METHOD: (coarse_frame_idx, coarse_image),
        M1_METHOD: (refined_frame_idx, refined_image),
    }
    tile_size = (640, 360)
    header_height = 130
    label_height = 48
    sheet = Image.new(
        "RGB", (tile_size[0] * 2, header_height + tile_size[1] + label_height), "white"
    )
    draw = ImageDraw.Draw(sheet)
    title_font, body_font, label_font = _font(24), _font(18), _font(17)
    draw.text((16, 12), f"{query_id} / {event_id} - BLIND REVIEW", fill="black", font=title_font)
    draw.multiline_text((16, 50), event_text, fill="black", font=body_font, spacing=4)
    for column, side in enumerate(("METHOD_A", "METHOD_B")):
        frame_idx, frame = method_values[mapping[side]]
        left = column * tile_size[0]
        draw.text((left + 16, header_height - 28), side, fill="black", font=label_font)
        sheet.paste(_rgb_image(frame, tile_size), (left, header_height))
        draw.text(
            (left + 16, header_height + tile_size[1] + 12),
            f"video_id={video_id}  actual_frame_idx={frame_idx}",
            fill="black",
            font=label_font,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=90)


def render_debug_strip(
    output_path: Path,
    *,
    video_id: str,
    frames: list[tuple[str, int, np.ndarray]],
) -> None:
    """Render the optional engineering strip; no reference frame is included."""

    from PIL import Image, ImageDraw

    tile_size = (420, 236)
    header = 50
    sheet = Image.new("RGB", (tile_size[0] * len(frames), header + tile_size[1]), "white")
    draw = ImageDraw.Draw(sheet)
    font = _font(16)
    for column, (label, frame_idx, frame) in enumerate(frames):
        left = column * tile_size[0]
        draw.text(
            (left + 8, 14), f"{label} | {video_id} | frame={frame_idx}", fill="black", font=font
        )
        sheet.paste(_rgb_image(frame, tile_size), (left, header))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=88)


__all__ = [
    "M0_METHOD",
    "M1_METHOD",
    "blinded_mapping",
    "render_blinded_event_sheet",
    "render_debug_strip",
]
