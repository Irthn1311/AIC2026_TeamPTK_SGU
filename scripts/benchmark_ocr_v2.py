"""
OCR Pipeline V2 - Crop, Upscale, Preprocess & ROI Classification Benchmark
==========================================================================
Tests and compares 3 configurations on a single video dataset (L21_V001):
  Config A: Current Baseline (Raw Crop without padding/upscaling)
  Config B: Crop (Standard 32px Height, Aspect Ratio Preserved)
  Config C: Crop + 2x/3x Lanczos Upscale + 5px Padding + Contrast Preprocessing

Generates:
  1. outputs/benchmark_ocr_comparison.csv
  2. outputs/benchmark_ocr_report.html
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from _bootstrap import PROJECT_ROOT


def classify_roi(bbox: list[int], img_w: int, img_h: int, text: str = "") -> str:
    """
    Classifies a text bounding box [x1, y1, x2, y2] into structural news video ROIs:
      - logo_channel
      - clock_time
      - headline
      - ticker
      - scene_text
    """
    x1, y1, x2, y2 = bbox
    x1_n = x1 / max(1, img_w)
    y1_n = y1 / max(1, img_h)
    x2_n = x2 / max(1, img_w)
    y2_n = y2 / max(1, img_h)
    w_n = (x2 - x1) / max(1, img_w)
    h_n = (y2 - y1) / max(1, img_h)

    # 1. Logo / Channel mark (Top-left or Top-right corners, small height)
    if y1_n < 0.25 and (x1_n < 0.30 or x2_n > 0.70) and h_n < 0.15 and w_n < 0.35:
        return "logo_channel"

    # 2. Clock / Timestamp (Small width/height, contains digits or time markers)
    if y1_n < 0.30 and h_n < 0.10 and w_n < 0.25:
        return "clock_time"

    # 3. News Headline / Lower-Third (Main lower banner area, wide box) - HIGHEST RETRIEVAL PRIORITY
    if 0.52 <= y1_n <= 0.88 and w_n >= 0.25:
        return "headline"

    # 4. Ticker (Scrolling ticker at the very bottom)
    if y1_n >= 0.85 and w_n >= 0.35:
        return "ticker"

    return "scene_text"


def preprocess_crop(crop_pil: Image.Image, scale_factor: float = 2.0, pad_px: int = 6) -> Image.Image:
    """
    Applies padding, Lanczos upscaling, and subtle contrast enhancement to text crop.
    """
    # Add border padding
    w, h = crop_pil.size
    padded = Image.new("RGB", (w + 2 * pad_px, h + 2 * pad_px), color=(255, 255, 255))
    padded.paste(crop_pil, (pad_px, pad_px))

    # Upscale using Lanczos (high quality anti-aliased interpolation)
    new_w = max(1, int(padded.width * scale_factor))
    new_h = max(1, int(padded.height * scale_factor))
    upscaled = padded.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)

    # Subtle contrast enhancement
    enhancer = ImageEnhance.Contrast(upscaled)
    enhanced = enhancer.enhance(1.2)

    return enhanced


def run_benchmark(video_id: str = "L21_V001", sample_count: int = 75):
    print("=" * 80)
    print(f" 🧪 OCR PIPELINE BENCHMARK (Video: {video_id}, Target Samples: {sample_count})")
    print("=" * 80)

    # 1. Load models
    import easyocr
    import torch
    from vietocr.tool.config import Cfg
    from vietocr.tool.predictor import Predictor

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Loading VietOCR (vgg_transformer) on {device}...")
    config = Cfg.load_config_from_name("vgg_transformer")
    config["device"] = device
    recognizer = Predictor(config)

    print(f"Loading CRAFT Detector (EasyOCR) on GPU: {torch.cuda.is_available()}...")
    detector = easyocr.Reader(["vi", "en"], gpu=torch.cuda.is_available())

    # 2. Find keyframes
    keyframe_dir = PROJECT_ROOT / "datasets_L21" / "Keyframes_L21" / "keyframes" / video_id
    kf_images = sorted(list(keyframe_dir.glob("*.jpg")))
    if not kf_images:
        print(f"❌ Keyframe directory not found: {keyframe_dir}")
        return

    print(f"Found {len(kf_images)} keyframe images in {video_id}.")

    # Collect sample text boxes across keyframes
    records = []
    box_counter = 0

    for kf_path in kf_images:
        if box_counter >= sample_count:
            break

        frame_id = int(kf_path.stem) if kf_path.stem.isdigit() else 0
        img_pil = Image.open(kf_path).convert("RGB")
        img_w, img_h = img_pil.size

        # Run detection
        raw_res = detector.readtext(str(kf_path))
        if not raw_res:
            continue

        for bbox, raw_text, easy_conf in raw_res:
            pts = np.array(bbox, dtype=np.int32)
            x, y, w, h = cv2.boundingRect(pts)
            if w <= 6 or h <= 6:
                continue

            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(img_w, x + w)
            y2 = min(img_h, y + h)

            region_type = classify_roi([x1, y1, x2, y2], img_w, img_h, raw_text)

            # CONFIG A: Current Raw Crop (No padding, direct crop)
            crop_A = img_pil.crop((x1, y1, x2, y2))
            text_A = recognizer.predict(crop_A).strip()

            # CONFIG B: Crop with Standard Padding (no upscaling)
            crop_B = preprocess_crop(crop_A, scale_factor=1.0, pad_px=4)
            text_B = recognizer.predict(crop_B).strip()

            # CONFIG C: Crop + 2.5x Lanczos Upscale + Padding + Preprocessing
            crop_C = preprocess_crop(crop_A, scale_factor=2.5, pad_px=6)
            text_C = recognizer.predict(crop_C).strip()

            # Save crop image for visual HTML report
            crop_out_dir = PROJECT_ROOT / "outputs" / "benchmark_crops"
            crop_out_dir.mkdir(parents=True, exist_ok=True)
            crop_filename = f"{video_id}_f{frame_id:04d}_b{box_counter:03d}.jpg"
            crop_filepath = crop_out_dir / crop_filename
            crop_C.save(crop_filepath)

            rec = {
                "video_id": video_id,
                "frame_id": frame_id,
                "bbox": [x1, y1, x2, y2],
                "region_type": region_type,
                "raw_easyocr": str(raw_text).strip(),
                "easyocr_conf": round(float(easy_conf), 3),
                "text_config_A": text_A,
                "text_config_B": text_B,
                "text_config_C": text_C,
                "crop_path": str(crop_filepath),
                "crop_uri": crop_filepath.as_uri(),
            }
            records.append(rec)
            box_counter += 1

            print(f" Box {box_counter:2d}/{sample_count} | ROI: {region_type:<13} | A: '{text_A[:25]}' -> C: '{text_C[:25]}'", end="\r")

            if box_counter >= sample_count:
                break

    print(f"\n✅ Successfully processed {len(records)} test text boxes!")

    # 3. Export CSV comparison
    df = pd.DataFrame(records)
    csv_out = PROJECT_ROOT / "outputs" / "benchmark_ocr_comparison.csv"
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_out, index=False, encoding="utf-8-sig")
    print(f"📊 CSV comparison exported to: {csv_out}")

    # 4. Export Visual HTML Report
    html_items = []
    for idx, row in enumerate(records, start=1):
        badge_color = "#1a73e8" if row["region_type"] == "headline" else "#5f6368"
        if row["region_type"] == "logo_channel":
            badge_color = "#ea4335"
        elif row["region_type"] == "clock_time":
            badge_color = "#fbbc04"

        html_items.append(f"""
        <div style="border: 1px solid #dadce0; padding: 12px; margin: 12px 0; border-radius: 8px; background: white; font-family: system-ui, sans-serif;">
            <div style="display: flex; justify-space-between; align-items: center;">
                <h4 style="margin: 0; color: #202124;">#{idx} | Frame {row['frame_id']} | BBox {row['bbox']}</h4>
                <span style="background: {badge_color}; color: white; padding: 3px 8px; border-radius: 4px; font-weight: bold; font-size: 12px;">{row['region_type'].upper()}</span>
            </div>
            <div style="margin-top: 10px; display: flex; gap: 15px; align-items: center;">
                <img src="{row['crop_uri']}" style="border: 1px solid #ccc; max-height: 70px; border-radius: 4px; background: #f1f3f4;" />
                <table style="width: 100%; font-size: 13px; border-collapse: collapse;">
                    <tr style="background: #f8f9fa;">
                        <th style="padding: 4px 8px; border: 1px solid #eee;">Config A (Raw Crop)</th>
                        <th style="padding: 4px 8px; border: 1px solid #eee;">Config B (Padded Crop)</th>
                        <th style="padding: 4px 8px; border: 1px solid #eee; color: #1a73e8;">Config C (Padded + 2.5x Upscale) ⭐</th>
                    </tr>
                    <tr>
                        <td style="padding: 6px; border: 1px solid #eee; color: #5f6368;">"{row['text_config_A']}"</td>
                        <td style="padding: 6px; border: 1px solid #eee; color: #3c4043;">"{row['text_config_B']}"</td>
                        <td style="padding: 6px; border: 1px solid #eee; font-weight: bold; color: #188038; background: #e6f4ea;">"{row['text_config_C']}"</td>
                    </tr>
                </table>
            </div>
        </div>
        """)

    html_out = PROJECT_ROOT / "outputs" / "benchmark_ocr_report.html"
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>OCR Pipeline V2 Benchmark Report</title>
</head>
<body style="background: #f8f9fa; padding: 20px; font-family: system-ui, sans-serif; max-width: 1000px; margin: 0 auto;">
    <h2 style="color: #1a73e8;">🧪 OCR Pipeline V2 Benchmark Report ({video_id})</h2>
    <p>So sánh 3 cấu hình bóc tách OCR trên {len(records)} text boxes:</p>
    <ul>
        <li><b>Config A:</b> Direct Raw Crop (Không padding, không upscale)</li>
        <li><b>Config B:</b> Padded Crop (Thêm 4px padding)</li>
        <li><b>Config C:</b> Padded + 2.5x Lanczos Upscaling + Contrast Enhancement (Cải tiến V2) ⭐</li>
    </ul>
    {"".join(html_items)}
</body>
</html>"""
    html_out.write_text(html_content, encoding="utf-8")
    print(f"🌐 Visual HTML report exported to: {html_out}")


if __name__ == "__main__":
    run_benchmark(video_id="L21_V001", sample_count=75)
