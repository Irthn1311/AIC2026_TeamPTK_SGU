from __future__ import annotations

import argparse
import base64
import html
import io
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def parse_jsonish(value: Any, fallback: Any) -> Any:
    if value is None:
        return fallback
    if isinstance(value, float) and pd.isna(value):
        return fallback
    if isinstance(value, (list, dict)):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return fallback
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return fallback
    return fallback


def escape(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def format_seconds(value: Any) -> str:
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return "?"
    minutes = int(seconds // 60)
    remain = seconds - minutes * 60
    return f"{minutes:02d}:{remain:05.2f}"


def image_to_data_uri(path: Path, max_width: int = 960) -> str:
    with Image.open(path) as img:
        img = img.convert("RGB")
        if img.width > max_width:
            ratio = max_width / float(img.width)
            img = img.resize((max_width, max(1, int(img.height * ratio))))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
    data = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{data}"


def draw_overlay(
    image_path: Path,
    boxes: list[dict[str, Any]],
    output_path: Path,
    *,
    label_key: str,
    score_key: str = "confidence",
) -> Path | None:
    if not image_path.is_file():
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as img:
        canvas = img.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    colors = [
        (230, 57, 70),
        (29, 53, 87),
        (42, 157, 143),
        (244, 162, 97),
        (131, 56, 236),
        (0, 119, 182),
        (255, 183, 3),
        (90, 24, 154),
    ]
    width = max(2, int(min(canvas.size) / 240))
    for idx, box in enumerate(boxes):
        bbox = parse_jsonish(box.get("bbox"), [])
        if len(bbox) < 4:
            continue
        try:
            x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
        except (TypeError, ValueError):
            continue
        color = colors[idx % len(colors)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
        label = str(box.get(label_key, "")).strip() or "?"
        if score_key and box.get(score_key) not in (None, ""):
            try:
                label += f" {float(box.get(score_key)):.2f}"
            except (TypeError, ValueError):
                pass
        tx, ty = x1 + 3, max(0, y1 - 15)
        draw.rectangle([tx - 2, ty, tx + len(label) * 6 + 6, ty + 14], fill=color)
        draw.text((tx, ty), label, fill=(255, 255, 255), font=font)
    canvas.save(output_path, quality=90)
    return output_path


def card(title: str, body: str, extra_class: str = "") -> str:
    return f'<article class="card {extra_class}"><h3>{escape(title)}</h3>{body}</article>'


def metric_cards(summary: dict[str, Any]) -> str:
    items = []
    for key, value in summary.items():
        items.append(f"<div><b>{escape(value)}</b><span>{escape(key)}</span></div>")
    return '<section class="metrics">' + "".join(items) + "</section>"


def select_evenly(rows: list[Any], limit: int) -> list[Any]:
    if len(rows) <= limit:
        return rows
    if limit <= 1:
        return rows[:1]
    step = (len(rows) - 1) / float(limit - 1)
    return [rows[round(idx * step)] for idx in range(limit)]


def render_keyframes(output_root: Path, report_dir: Path, limit: int) -> tuple[str, dict[str, Any]]:
    keyframe_root = Path(os.environ.get("AIC_KEYFRAME_OUTPUT_ROOT", output_root / "keyframe_v2_full"))
    btc_path = keyframe_root / "indexes" / "keyframe_btc_global_map.parquet"
    v2_path = keyframe_root / "indexes" / "keyframe_v2_global_map.parquet"
    path = btc_path if btc_path.is_file() else v2_path
    if not path.is_file():
        return card("Keyframes", "<p class='muted'>Chưa có global map.</p>"), {"keyframes": "missing"}
    df = pd.read_parquet(path)
    rows = select_evenly(df.to_dict("records"), min(limit, 12))
    cards = []
    for row in rows:
        image_path = Path(str(row.get("image_path", "")))
        if not image_path.is_file():
            continue
        img_uri = image_to_data_uri(image_path)
        title = f"{row.get('video_id')} | frame {row.get('actual_frame_id', row.get('frame_idx'))}"
        meta = f"{format_seconds(row.get('timestamp_sec', row.get('timestamp_seconds')))} | {Path(image_path).name}"
        cards.append(f"<figure><img src='{img_uri}'><figcaption><b>{escape(title)}</b><span>{escape(meta)}</span></figcaption></figure>")
    body = "<div class='gallery'>" + "".join(cards) + "</div>"
    return card("Keyframes", body), {"keyframes": len(df)}


def render_object(output_root: Path, report_dir: Path, limit: int) -> tuple[str, dict[str, Any]]:
    object_root = Path(os.environ.get("AIC_OBJECT_OUTPUT_ROOT", output_root / "keyframe_v2_full" / "object_v2"))
    btc_path = object_root / "l21_objects_btc_detections.parquet"
    v2_path = object_root / "l21_objects_v2_detections.parquet"
    path = btc_path if btc_path.is_file() else v2_path
    if not path.is_file():
        return card("Object V2", "<p class='muted'>Chưa có object detections.</p>"), {"object_detections": "missing"}
    df = pd.read_parquet(path)
    if df.empty:
        return card("Object V2", "<p class='muted'>Object parquet rỗng.</p>"), {"object_detections": 0}
    gallery = []
    for idx, (image_path_text, group) in enumerate(df.groupby("image_path", sort=False)):
        if idx >= min(limit, 10):
            break
        image_path = Path(str(image_path_text))
        rows = group.sort_values("confidence", ascending=False).head(8).to_dict("records")
        overlay = draw_overlay(image_path, rows, report_dir / "media" / "objects" / f"{image_path.stem}.jpg", label_key="object_label")
        if overlay is None:
            continue
        labels = ", ".join(f"{r.get('object_label')}:{float(r.get('confidence', 0.0)):.2f}" for r in rows[:6])
        gallery.append(
            f"<figure><img src='{image_to_data_uri(overlay)}'><figcaption><b>{escape(image_path.name)}</b><span>{escape(labels)}</span></figcaption></figure>"
        )
    top_labels = []
    if "object_label" in df.columns:
        top_labels = df["object_label"].astype(str).value_counts().head(12).items()
    pills = "".join(f"<span>{escape(label)} <b>{count}</b></span>" for label, count in top_labels)
    body = f"<div class='pills'>{pills}</div><div class='gallery'>{''.join(gallery)}</div>"
    return card("Object V2", body), {"object_detections": len(df), "object_images": int(df["image_path"].nunique())}


def render_ocr_v2(output_root: Path, report_dir: Path, limit: int) -> tuple[str, dict[str, Any]]:
    path = output_root / "ocr_v2_selected_keyframes" / "l21_keyframe_ocr.jsonl"
    rows = [row for row in read_jsonl(path) if row.get("detections")]
    if not rows:
        return card("OCR V2", "<p class='muted'>Chưa có OCR V2 hoặc chưa detect được text box.</p>"), {"ocr_v2_rows": "missing"}
    gallery = []
    for row in rows[: min(limit, 10)]:
        image_path = Path(str(row.get("keyframe_path", "")))
        boxes = parse_jsonish(row.get("detections"), [])
        overlay = draw_overlay(image_path, boxes[:10], report_dir / "media" / "ocr" / f"{image_path.stem}.jpg", label_key="text")
        if overlay is None:
            continue
        text = str(row.get("combined_text", "")).strip()
        title = f"{row.get('video_id')} | {Path(image_path).name}"
        gallery.append(f"<figure><img src='{image_to_data_uri(overlay)}'><figcaption><b>{escape(title)}</b><span>{escape(text[:180])}</span></figcaption></figure>")
    body = "<div class='gallery'>" + "".join(gallery) + "</div>"
    return card("OCR V2 Boxes", body), {"ocr_v2_rows": len(rows)}


def render_ocr_temporal(output_root: Path, limit: int) -> tuple[str, dict[str, Any]]:
    path = output_root / "ocr_temporal_v3_full_tracking" / "l21_ocr_tracks.parquet"
    if not path.is_file():
        return card("OCR Temporal V3", "<p class='muted'>Chưa có OCR temporal tracks.</p>"), {"ocr_tracks": "missing"}
    df = pd.read_parquet(path)
    if df.empty:
        return card("OCR Temporal V3", "<p class='muted'>Track parquet rỗng.</p>"), {"ocr_tracks": 0}
    sort_col = "num_observations" if "num_observations" in df.columns else None
    sample = df.sort_values(sort_col, ascending=False).head(min(limit, 20)) if sort_col else df.head(min(limit, 20))
    rows = []
    for _, row in sample.iterrows():
        text = row.get("corrected_text") or row.get("consensus_text") or row.get("semantic_search_text") or ""
        rows.append(
            "<tr>"
            f"<td>{escape(row.get('video_id'))}</td>"
            f"<td>{format_seconds(row.get('start_time'))} - {format_seconds(row.get('end_time'))}</td>"
            f"<td>{escape(row.get('num_observations', '?'))}</td>"
            f"<td>{escape(text)}</td>"
            "</tr>"
        )
    body = "<table><thead><tr><th>Video</th><th>Time</th><th>Obs</th><th>Text</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    return card("OCR Temporal V3", body), {"ocr_tracks": len(df)}


def render_asr(output_root: Path, data_root: Path | None, report_dir: Path, limit: int, video_preview_count: int) -> tuple[str, dict[str, Any]]:
    asr_dir = output_root / "asr"
    files = sorted(asr_dir.glob("L*_V*_asr.json"))
    if not files:
        return card("ASR", "<p class='muted'>Chưa có ASR JSON.</p>"), {"asr_segments": "missing"}
    total_segments = 0
    blocks = []
    clip_count = 0
    for file_idx, path in enumerate(files[: min(limit, 6)]):
        video_id = path.name.replace("_asr.json", "")
        segments = read_json(path)
        total_segments += len(segments)
        clip_html = ""
        if data_root and clip_count < video_preview_count:
            clip = make_video_clip(data_root, report_dir, video_id, segments)
            if clip:
                clip_count += 1
                clip_html = f"<video controls preload='metadata' src='media/video/{escape(clip.name)}'></video>"
        lines = []
        for seg in segments[:8]:
            lines.append(f"<li><b>{format_seconds(seg.get('start'))}-{format_seconds(seg.get('end'))}</b> {escape(seg.get('text_raw'))}</li>")
        blocks.append(f"<section class='asr-block'><h4>{escape(video_id)} · {len(segments)} segments</h4>{clip_html}<ol>{''.join(lines)}</ol></section>")
    return card("ASR", "".join(blocks)), {"asr_videos": len(files), "asr_segments_sampled": total_segments, "video_clips": clip_count}


def make_video_clip(data_root: Path, report_dir: Path, video_id: str, segments: list[dict[str, Any]]) -> Path | None:
    ffmpeg = shutil.which("ffmpeg")
    video_path = next(
        (path for path in sorted(data_root.glob("Videos_L*_*/video/*.mp4")) if path.stem == video_id),
        data_root / "Videos_L21_a" / "video" / f"{video_id}.mp4",
    )
    if not ffmpeg or not video_path.is_file():
        return None
    start = 0.0
    if segments:
        try:
            start = max(0.0, float(segments[0].get("start", 0.0)) - 1.0)
        except (TypeError, ValueError):
            start = 0.0
    out_dir = report_dir / "media" / "video"
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / f"{video_id}_preview.mp4"
    if output.is_file() and output.stat().st_size > 0:
        return output
    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start:.3f}",
        "-i",
        str(video_path),
        "-t",
        "12",
        "-vf",
        "scale=640:-2",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-c:a",
        "aac",
        str(output),
    ]
    try:
        subprocess.run(cmd, check=True)
    except Exception:
        return None
    return output if output.is_file() else None


def render_package(output_root: Path) -> tuple[str, dict[str, Any]]:
    path = output_root / "kaggle_package_validation.json"
    if not path.is_file():
        return card("Package", "<p class='muted'>Chưa validate/package.</p>"), {"package": "missing"}
    report = read_json(path)
    missing = report.get("missing_or_empty", [])
    body = f"<p>Status: <b>{escape(report.get('status'))}</b></p><p>Missing: <b>{len(missing)}</b></p>"
    if missing:
        body += "<ul>" + "".join(f"<li>{escape(item.get('path'))}</li>" for item in missing[:10]) + "</ul>"
    return card("Package", body), {"package": report.get("status", "unknown")}


def write_html(report_dir: Path, sections: list[str], summary: dict[str, Any]) -> Path:
    html_path = report_dir / "index.html"
    html_path.parent.mkdir(parents=True, exist_ok=True)
    css = """
    body{margin:0;background:#f4f6f8;color:#15202b;font-family:Segoe UI,Arial,sans-serif}
    header{padding:22px 28px;background:#132238;color:white}
    header h1{margin:0;font-size:24px} header p{margin:6px 0 0;color:#c8d3df}
    main{padding:22px 28px;display:grid;gap:18px}
    .metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}
    .metrics div{background:white;border:1px solid #dbe3ea;border-radius:8px;padding:12px}
    .metrics b{display:block;font-size:22px}.metrics span{color:#52616f;font-size:12px}
    .card{background:white;border:1px solid #dbe3ea;border-radius:8px;padding:16px;overflow:hidden}
    .card h3{margin:0 0 12px;font-size:18px}.gallery{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px}
    figure{margin:0;border:1px solid #e1e7ee;border-radius:8px;overflow:hidden;background:#fbfcfd}
    img{width:100%;display:block}figcaption{display:grid;gap:4px;padding:9px;font-size:12px;color:#4d5b68}
    figcaption b{color:#162331}.pills{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:12px}
    .pills span{border:1px solid #d8e1e8;border-radius:999px;padding:5px 9px;background:#f8fafc;font-size:12px}
    table{width:100%;border-collapse:collapse;font-size:13px}th,td{border-bottom:1px solid #e5ebf0;text-align:left;padding:8px;vertical-align:top}
    th{color:#536170;background:#f8fafc}.muted{color:#6b7785}.asr-block{border-top:1px solid #e5ebf0;padding-top:10px;margin-top:10px}
    .asr-block:first-child{border-top:0;margin-top:0}.asr-block h4{margin:0 0 8px}video{width:100%;max-width:640px;border-radius:8px;background:#111}
    """
    html_path.write_text(
        "<!doctype html><meta charset='utf-8'>"
        "<title>Kaggle Pipeline Visual Report</title>"
        f"<style>{css}</style>"
        "<header><h1>Kaggle Pipeline Visual Report</h1>"
        "<p>Keyframe · OCR · OCR Temporal · ASR · Object · Package</p></header>"
        "<main>"
        + metric_cards(summary)
        + "".join(sections)
        + "</main>",
        encoding="utf-8",
    )
    return html_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a compact HTML visual report for Kaggle preprocessing artifacts.")
    parser.add_argument("--output-root", default=os.environ.get("AIC_OUTPUT_ROOT", "/kaggle/working/artifacts"))
    parser.add_argument("--data-root", default=os.environ.get("AIC_DATA_ROOT", ""))
    parser.add_argument("--report-dir", default=None)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--video-preview-count", type=int, default=1)
    args = parser.parse_args()

    output_root = resolve_path(args.output_root)
    data_root = resolve_path(args.data_root) if args.data_root else None
    report_dir = resolve_path(args.report_dir or (output_root / "kaggle_visual_report"))
    report_dir.mkdir(parents=True, exist_ok=True)

    sections: list[str] = []
    summary: dict[str, Any] = {}
    for section_html, section_summary in [
        render_keyframes(output_root, report_dir, args.limit),
        render_object(output_root, report_dir, args.limit),
        render_ocr_v2(output_root, report_dir, args.limit),
        render_ocr_temporal(output_root, args.limit),
        render_asr(output_root, data_root, report_dir, args.limit, args.video_preview_count),
        render_package(output_root),
    ]:
        sections.append(section_html)
        summary.update(section_summary)

    html_path = write_html(report_dir, sections, summary)
    summary_path = report_dir / "visual_report_summary.json"
    summary_path.write_text(json.dumps({"html": str(html_path), **summary}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"visual_report": str(html_path), "summary": summary}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
