#!/usr/bin/env python3
"""Visual Inspector for BTC THUNGHIEM_20-8 Top Predictions.

Extracts and displays the actual video frame images for Top 1..3 candidates of each query.
Ultra-fast single-pass video indexing and real-time progress streaming.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
THUNGHIEM_DIR = REPO_ROOT / "systems" / "system_tai" / "THUNGHIEM_20-8"
OUTPUT_DIR = Path("/kaggle/working/output/thunghiem_20_8/requests") if Path("/kaggle/working").exists() else REPO_ROOT / "scratch" / "thunghiem_20_8" / "requests"

_VIDEO_MAP_CACHE: dict[str, Path] = {}


def get_video_map() -> dict[str, Path]:
    """Single-pass fast resolution of all video files."""
    global _VIDEO_MAP_CACHE
    if _VIDEO_MAP_CACHE:
        return _VIDEO_MAP_CACHE

    # 1. Fast resolve via manifest cache
    for mf in [
        Path("/kaggle/working/manifest_cache.json"),
        Path("/kaggle/input/system-tai-manifest/feature_manifest.json"),
        Path("/kaggle/working/feature_manifest.json"),
    ]:
        if mf.exists():
            try:
                data = json.loads(mf.read_text(encoding="utf-8"))
                for v in data.get("videos", []):
                    vid = v.get("video_id")
                    vpath = v.get("raw_video_path")
                    if vid and vpath and Path(vpath).exists():
                        _VIDEO_MAP_CACHE[vid] = Path(vpath)
                if _VIDEO_MAP_CACHE:
                    return _VIDEO_MAP_CACHE
            except Exception:
                pass

    # 2. Fast single-pass os.walk
    for root_dir in [
        Path("/kaggle/input/datasets/videos"),
        Path("/kaggle/input/videos"),
        Path("/kaggle/input"),
    ]:
        if root_dir.exists():
            for dirpath, _, filenames in os.walk(str(root_dir)):
                for fname in filenames:
                    if fname.endswith(".mp4"):
                        vid = fname[:-4]
                        if vid not in _VIDEO_MAP_CACHE:
                            _VIDEO_MAP_CACHE[vid] = Path(dirpath) / fname

    return _VIDEO_MAP_CACHE


def find_video_path(video_id: str) -> Path | None:
    return get_video_map().get(video_id)


def extract_frame_base64(video_path: Path, frame_id: int, max_width: int = 400) -> str | None:
    try:
        import cv2
        from PIL import Image

        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            return None

        # Convert BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)

        # Resize for fast HTML rendering
        if img.width > max_width:
            h = int(img.height * (max_width / img.width))
            img = img.resize((max_width, h), Image.Resampling.LANCZOS)

        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=80)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    except Exception:
        return None


def generate_html_gallery(top_n_images: int = 3) -> str:
    query_files = sorted(THUNGHIEM_DIR.glob("*.txt"))
    html_cards = []

    html_cards.append("""
    <div style="font-family: Arial, sans-serif; max-width: 1200px; margin: auto;">
        <h1 style="text-align: center; color: #1a73e8; border-bottom: 2px solid #1a73e8; padding-bottom: 10px;">
            🖼️ TRỰC QUAN HÓA KẾT QUẢ TOP PREDICTIONS (BTC THỬ NGHIỆM 20-8)
        </h1>
    """)

    # Pre-index video paths once
    v_map = get_video_map()
    print(f"• Đã lập chỉ mục {len(v_map)} video từ dataset. Bắt đầu trích xuất hình ảnh...", flush=True)

    for idx, f in enumerate(query_files, start=1):
        filename = f.name
        qid = filename.replace(".txt", "")
        text = f.read_text(encoding="utf-8").strip()

        # Find matching request output folder
        matches = list(OUTPUT_DIR.glob(f"*{qid}*"))
        top_preds = []
        if matches:
            q_dir = matches[0]
            top_file = q_dir / "refined_top100.jsonl"
            if not top_file.exists():
                top_file = q_dir / "top100.jsonl"
            if not top_file.exists():
                top_file = q_dir / "qa_predictions.jsonl"

            if top_file.exists():
                preds = [json.loads(l) for l in top_file.read_text(encoding="utf-8").splitlines() if l.strip()]
                top_preds = preds[:top_n_images]

        card_html = f"""
        <div style="background: #ffffff; border: 1px solid #e0e0e0; border-radius: 8px; margin-bottom: 25px; padding: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.08);">
            <div style="font-size: 16px; font-weight: bold; color: #202124; margin-bottom: 8px;">
                <span style="background: #e8f0fe; color: #1967d2; padding: 3px 8px; border-radius: 4px; margin-right: 8px;">#{idx:02d} | {qid}</span>
            </div>
            <div style="font-size: 14px; color: #3c4043; background: #f8f9fa; padding: 10px; border-left: 4px solid #1a73e8; border-radius: 4px; margin-bottom: 15px; line-height: 1.5;">
                <b>Truy vấn:</b> {text}
            </div>
            <div style="display: flex; gap: 15px; flex-wrap: wrap;">
        """

        if not top_preds:
            card_html += """<div style="color: #d93025; font-style: italic;">Chưa tìm thấy file kết quả output cho query này.</div>"""
        else:
            for p in top_preds:
                rank = p["rank"]
                v_id = p["video_id"]
                f_id = p.get("frame_id", 0)
                sec = f_id // 25
                time_str = f"{sec // 60:02d}:{sec % 60:02d}"
                ans = p.get("answer", "")

                v_path = find_video_path(v_id)
                img_b64 = extract_frame_base64(v_path, f_id) if v_path else None

                img_tag = (
                    f'<img src="data:image/jpeg;base64,{img_b64}" style="width: 100%; height: auto; border-radius: 4px; border: 1px solid #ddd;" />'
                    if img_b64
                    else '<div style="height: 180px; background: #eee; display: flex; align-items: center; justify-content: center; color: #888;">(Không tìm thấy file video MP4)</div>'
                )

                ans_tag = f'<div style="font-size: 12px; color: #188038; margin-top: 4px;"><b>Answer:</b> {ans}</div>' if ans else ''

                card_html += f"""
                <div style="flex: 1; min-width: 260px; max-width: 360px; background: #fafafa; border: 1px solid #ddd; border-radius: 6px; padding: 10px; text-align: center;">
                    <div style="font-weight: bold; font-size: 13px; color: #1a73e8; margin-bottom: 6px;">
                        Rank @{rank} | {v_id}
                    </div>
                    {img_tag}
                    <div style="font-size: 12px; color: #5f6368; margin-top: 6px;">
                        Frame: <b>{f_id}</b> (~{time_str})
                    </div>
                    {ans_tag}
                </div>
                """

        card_html += """
            </div>
        </div>
        """
        html_cards.append(card_html)
        print(f"[{idx:02d}/{len(query_files)}] Trích xuất xong {qid}", flush=True)

    html_cards.append("</div>")
    return "\n".join(html_cards)


if __name__ == "__main__":
    pass
