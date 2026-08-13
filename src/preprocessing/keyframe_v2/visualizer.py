from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _font(size: int = 13):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def make_contact_sheet(items: list[dict], image_key: str, output_path: Path, title: str, cols: int = 4, thumb_w: int = 300, thumb_h: int = 170) -> None:
    if not items:
        return
    rows = (len(items) + cols - 1) // cols
    label_h = 82
    pad = 8
    header_h = 34
    sheet = Image.new("RGB", (cols * (thumb_w + pad) + pad, header_h + rows * (thumb_h + label_h + pad) + pad), (24, 26, 32))
    draw = ImageDraw.Draw(sheet)
    font = _font(13)
    small = _font(11)
    draw.text((pad, 8), title, fill=(245, 245, 245), font=font)
    for idx, item in enumerate(items):
        path = Path(str(item[image_key]))
        if not path.exists():
            continue
        img = Image.open(path).convert("RGB")
        img.thumbnail((thumb_w, thumb_h), Image.LANCZOS)
        canvas = Image.new("RGB", (thumb_w, thumb_h), (10, 10, 10))
        canvas.paste(img, ((thumb_w - img.width) // 2, (thumb_h - img.height) // 2))
        r, c = divmod(idx, cols)
        x = pad + c * (thumb_w + pad)
        y = header_h + pad + r * (thumb_h + label_h + pad)
        sheet.paste(canvas, (x, y))
        lines = item.get("label_lines", [])
        for line_i, line in enumerate(lines[:5]):
            draw.text((x + 2, y + thumb_h + 4 + line_i * 14), str(line), fill=(220, 220, 220), font=small)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)
