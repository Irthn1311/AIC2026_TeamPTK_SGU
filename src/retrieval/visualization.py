from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _load_font(size: int = 16) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\segoeui.ttf",
        r"C:\Windows\Fonts\tahoma.ttf",
    ]
    for fp in candidates:
        if Path(fp).exists():
            try:
                return ImageFont.truetype(fp, size=size)
            except Exception:
                pass
    return ImageFont.load_default()


def make_contact_sheet(items, output_path: str | Path, columns: int = 4, tile_size=(360, 240)) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not items:
        Image.new("RGB", (tile_size[0], tile_size[1]), "white").save(output_path)
        return output_path
    rows = math.ceil(len(items) / columns)
    label_h = 96
    sheet = Image.new("RGB", (columns * tile_size[0], rows * (tile_size[1] + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    font = _load_font(16)
    small = _load_font(13)
    for i, item in enumerate(items):
        r, c = divmod(i, columns)
        x = c * tile_size[0]
        y = r * (tile_size[1] + label_h)
        img = item.get("image")
        if img is None:
            img = Image.new("RGB", tile_size, (230, 230, 230))
        else:
            img = img.copy()
            img.thumbnail(tile_size)
        canvas = Image.new("RGB", tile_size, "white")
        ox = (tile_size[0] - img.width) // 2
        oy = (tile_size[1] - img.height) // 2
        canvas.paste(img, (ox, oy))
        sheet.paste(canvas, (x, y))
        text = item.get("label", "")
        draw.multiline_text((x + 8, y + tile_size[1] + 4), text, fill="black", font=small, spacing=3)
    sheet.save(output_path)
    return output_path

