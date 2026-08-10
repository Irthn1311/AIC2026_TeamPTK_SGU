"""Blinded MB1-E1 A/B sheets with the method mapping kept in a separate key."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

M0_METHOD = "SOURCE_ANCHOR_FRAME"
M1_METHOD = "LOCAL_RAW_CLIP_COARSE_TO_FINE"


def blinded_mapping(moment_id: str, seed: int = 2026) -> dict[str, str]:
    digest = hashlib.sha256(f"MB1-E1:{seed}:{moment_id}".encode()).digest()
    if digest[0] % 2:
        return {"METHOD_A": M1_METHOD, "METHOD_B": M0_METHOD}
    return {"METHOD_A": M0_METHOD, "METHOD_B": M1_METHOD}


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


def render_blinded_sheet(
    path: Path,
    *,
    moment_id: str,
    query_text: str,
    m0_frame: int,
    m1_frame: int,
    images: dict[int, np.ndarray],
    seed: int = 2026,
) -> dict[str, Any]:
    """Render only query text, blind labels, images, and raw frame identities."""

    from PIL import Image, ImageDraw, ImageOps

    mapping = blinded_mapping(moment_id, seed)
    method_frames = {M0_METHOD: m0_frame, M1_METHOD: m1_frame}
    width, tile_height, header_height, label_height = 640, 360, 92, 48
    sheet = Image.new("RGB", (width * 2, header_height + tile_height + label_height), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((14, 12), query_text, fill="black", font=_font(22))
    for index, blind_label in enumerate(("METHOD_A", "METHOD_B")):
        frame = method_frames[mapping[blind_label]]
        image = Image.fromarray(np.asarray(images[frame], dtype=np.uint8), mode="RGB")
        fitted = ImageOps.fit(image, (width, tile_height), method=Image.Resampling.LANCZOS)
        x = index * width
        sheet.paste(fitted, (x, header_height))
        draw.text(
            (x + 14, header_height + tile_height + 10),
            f"{blind_label}  frame={frame}",
            fill="black",
            font=_font(20),
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, format="JPEG", quality=90, optimize=True)
    return {
        "moment_id": moment_id,
        "seed": seed,
        "mapping": mapping,
        "frames": {
            "METHOD_A": method_frames[mapping["METHOD_A"]],
            "METHOD_B": method_frames[mapping["METHOD_B"]],
        },
    }
