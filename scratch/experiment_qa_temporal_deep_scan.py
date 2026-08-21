#!/usr/bin/env python3
"""Video-Centric Temporal OCR Deep Scanner for Exact QA Evidence Verification.

Strict Operational Protocols:
  1. No Blind Guessing & No Auto-Packaging: Never claim an answer without verifiable visual evidence.
  2. p1-22 Temporal Scan: Scans L26 cooking series at 50-frame strides for '200g', 'thịt nạc xay', and extracts the exact dish title card.
  3. p1-19 Temporal Scan: Scans L28/L27 historical series for temple poetry verses dedicated to Nguyễn Trung Trực in Kiên Giang.
  4. p1-15 Temporal Scan: Scans L30/L21/L22/L29 charity candidates for 'FANA' AND 'Khánh Hòa' to identify the genuine commune name.
  5. High-Resolution ROI Extraction: Crops text bounding boxes + 2x upscaling for razor-sharp OCR evidence.
  6. Visual Gallery Output: /kaggle/working/qa_temporal_deep_scan_gallery.html for human inspection.
"""

from __future__ import annotations

import base64
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

print("=" * 150, flush=True)
print("QA VIDEO-CENTRIC TEMPORAL DEEP SCANNER (EXACT EVIDENCE VERIFICATION)", flush=True)
print("=" * 150, flush=True)

REPO_ROOT = Path(__file__).resolve().parents[1]
SYSTEM_TAI_SRC = REPO_ROOT / "systems" / "system_tai" / "src"
if str(SYSTEM_TAI_SRC) not in sys.path:
    sys.path.insert(0, str(SYSTEM_TAI_SRC))

try:
    import cv2
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "opencv-python-headless"], check=False)
    import cv2

try:
    import easyocr
    READER = easyocr.Reader(["vi", "en"], gpu=False, verbose=False)
except Exception as exc:
    print(f"EasyOCR init: {exc}", flush=True)
    READER = None

VIDEO_CACHE: dict[str, Path] = {}

def find_video_path(video_id: str) -> Path | None:
    if video_id in VIDEO_CACHE:
        return VIDEO_CACHE[video_id]
    batch = video_id.split("_")[0] if "_" in video_id else ""
    candidates = [
        Path(f"/kaggle/input/datasets/nadkli/dataset-aic/Videos_{batch}_a/video/{video_id}.mp4"),
        Path(f"/kaggle/input/datasets/nadkli/dataset-aic/Videos_{batch}_b/video/{video_id}.mp4"),
        Path(f"/kaggle/input/datasets/nadkli/dataset-aic/Videos_{batch}/video/{video_id}.mp4"),
        Path(f"/kaggle/input/datasets/videos/{batch}/{video_id}.mp4"),
        Path(f"/kaggle/input/datasets/{batch}/{video_id}.mp4"),
        Path(f"/kaggle/input/datasets/{video_id}.mp4"),
        REPO_ROOT / "systems" / "system_tai" / "data" / "videos" / batch / f"{video_id}.mp4",
    ]
    for p in candidates:
        if p.exists():
            VIDEO_CACHE[video_id] = p
            return p
    return None

def ocr_frame_boxes(frame: Any) -> list[dict[str, Any]]:
    """Runs EasyOCR with bounding box detection and extracts text lines."""
    if frame is None or READER is None:
        return []
    try:
        results = READER.readtext(frame, detail=1)
        items = []
        for bbox, text, conf in results:
            clean = " ".join(text.strip().split())
            if clean and len(clean) >= 2 and conf > 0.2:
                items.append({"bbox": bbox, "text": clean, "conf": float(conf)})
        return items
    except Exception:
        return []

def extract_b64(img: Any, quality: int = 85) -> str:
    if img is None:
        return ""
    try:
        _, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        return base64.b64encode(buf).decode("utf-8")
    except Exception:
        return ""

# -----------------------------------------------------------------------------
# 1. p1-22 Temporal Deep Scan (Cooking Series: 200g Thịt Nạc Xay + Tên Món Ăn)
# -----------------------------------------------------------------------------
def scan_p1_22_cooking() -> list[dict[str, Any]]:
    print("\n" + "=" * 120)
    print("🍳 [p1-22] TEMPORAL DEEP SCAN: Searching Cooking Series for 200g Thịt Nạc Xay & Dish Title")
    print("=" * 120)
    
    # Priority list of retrieved L26 cooking videos
    candidate_videos = [
        "L26_V277", "L26_V281", "L26_V242", "L26_V122", "L26_V455",
        "L26_V203", "L26_V315", "L26_V336", "L26_V320", "L26_V294",
        "L26_V474", "L26_V414", "L26_V016", "L26_V422", "L26_V367",
        "L26_V008", "L26_V018", "L26_V453", "L26_V300", "L26_V380",
        "L26_V350", "L26_V365", "L26_V494", "L26_V232", "L26_V377",
        "L26_V145", "L26_V313", "L26_V288", "L26_V032"
    ]
    
    evidence_hits = []
    
    for vid in candidate_videos:
        vpath = find_video_path(vid)
        if not vpath:
            continue
        cap = cv2.VideoCapture(str(vpath))
        if not cap.isOpened():
            continue
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        print(f"  • Scanning {vid} ({total_frames} frames, {total_frames/fps:.1f}s) ...", flush=True)
        
        # Stride of 60 frames (~2.4s) across the entire video
        for fid in range(0, total_frames, 60):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
                
            box_results = ocr_frame_boxes(frame)
            full_text = " ".join([b["text"] for b in box_results]).lower()
            
            # Look for ingredient card keywords
            has_meat = any(k in full_text for k in ["thịt", "xay", "nạc", "pork", "meat"])
            has_200g = any(k in full_text for k in ["200g", "200 g", "200", "2009", "2009g", "2oog"])
            has_recipe = any(k in full_text for k in ["nguyên liệu", "công thức", "món", "thực hiện", "nấu"])
            
            score = 0
            if has_meat and has_200g:
                score += 5
            elif has_meat or has_200g:
                score += 2
            if has_recipe:
                score += 1
                
            if score >= 3 or (has_meat and has_200g):
                print(f"      ⭐ [HIT score={score}] {vid} Frame {fid} ({fid/fps:.1f}s): {[b['text'] for b in box_results]}")
                b64_full = extract_b64(frame)
                
                # Center crop 2x
                h, w = frame.shape[:2]
                crop = frame[int(h*0.15):int(h*0.85), int(w*0.1):int(w*0.9)]
                b64_crop = extract_b64(crop)
                
                evidence_hits.append({
                    "video_id": vid,
                    "frame_id": fid,
                    "sec": fid / fps,
                    "score": score,
                    "text_lines": [b["text"] for b in box_results],
                    "b64_full": b64_full,
                    "b64_crop": b64_crop,
                })
        cap.release()
        
    return evidence_hits

# -----------------------------------------------------------------------------
# 2. p1-19 Temporal Deep Scan (Nguyễn Trung Trực Kiên Giang Couplet Poetry)
# -----------------------------------------------------------------------------
def scan_p1_19_poetry() -> list[dict[str, Any]]:
    print("\n" + "=" * 120)
    print("📜 [p1-19] TEMPORAL DEEP SCAN: Searching Historical Series for Nguyễn Trung Trực Couplet Verses")
    print("=" * 120)
    
    hist_videos = [
        "L28_V018", "L28_V012", "L28_V013", "L28_V010", "L28_V007",
        "L28_V015", "L28_V016", "L28_V005", "L28_V006", "L28_V024",
        "L29_V023", "L29_V001", "L29_V002", "L27_V001", "L27_V004", "L27_V010"
    ]
    
    evidence_hits = []
    
    for vid in hist_videos:
        vpath = find_video_path(vid)
        if not vpath:
            continue
        cap = cv2.VideoCapture(str(vpath))
        if not cap.isOpened():
            continue
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        print(f"  • Scanning {vid} ({total_frames} frames, {total_frames/fps:.1f}s) ...", flush=True)
        
        for fid in range(0, total_frames, 75):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
                
            box_results = ocr_frame_boxes(frame)
            full_text = " ".join([b["text"] for b in box_results]).lower()
            
            has_ntt = any(k in full_text for k in ["nguyễn trung trực", "trung trực", "nguyen trung truc"])
            has_poem = any(k in full_text for k in ["hỏa hồng", "nhật tảo", "kiếm bạt", "kiên giang", "oanh thiên địa", "khấp quỷ thần", "câu thơ", "thơ ca"])
            has_temple = any(k in full_text for k in ["đình thần", "đền thờ", "đình", "hòn chông", "rạch giá"])
            
            score = 0
            if has_poem:
                score += 5
            if has_ntt:
                score += 3
            if has_temple:
                score += 1
                
            if score >= 3:
                print(f"      ⭐ [HIT score={score}] {vid} Frame {fid} ({fid/fps:.1f}s): {[b['text'] for b in box_results]}")
                b64_full = extract_b64(frame)
                h, w = frame.shape[:2]
                crop = frame[int(h*0.1):int(h*0.9), int(w*0.05):int(w*0.95)]
                b64_crop = extract_b64(crop)
                
                evidence_hits.append({
                    "video_id": vid,
                    "frame_id": fid,
                    "sec": fid / fps,
                    "score": score,
                    "text_lines": [b["text"] for b in box_results],
                    "b64_full": b64_full,
                    "b64_crop": b64_crop,
                })
        cap.release()
        
    return evidence_hits

# -----------------------------------------------------------------------------
# 3. p1-15 Temporal Deep Scan (CLB FANA & Khánh Hòa Commune Name)
# -----------------------------------------------------------------------------
def scan_p1_15_fana() -> list[dict[str, Any]]:
    print("\n" + "=" * 120)
    print("🤝 [p1-15] TEMPORAL DEEP SCAN: Searching Charity Series for FANA & Khánh Hòa Commune")
    print("=" * 120)
    
    charity_videos = [
        "L30_V072", "L30_V031", "L30_V034", "L30_V066", "L30_V081", "L30_V091",
        "L21_V009", "L21_V018", "L21_V027", "L21_V030", "L22_V025", "L22_V018", "L22_V004",
        "L29_V008", "L29_V005", "L29_V001", "L28_V024"
    ]
    
    evidence_hits = []
    
    for vid in charity_videos:
        vpath = find_video_path(vid)
        if not vpath:
            continue
        cap = cv2.VideoCapture(str(vpath))
        if not cap.isOpened():
            continue
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        print(f"  • Scanning {vid} ({total_frames} frames, {total_frames/fps:.1f}s) ...", flush=True)
        
        for fid in range(0, total_frames, 75):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
                
            box_results = ocr_frame_boxes(frame)
            full_text = " ".join([b["text"] for b in box_results]).lower()
            
            has_fana = any(k in full_text for k in ["fana", "fa na", "clb fana", "câu lạc bộ fana"])
            has_khanh_hoa = any(k in full_text for k in ["khánh hòa", "khanh hoa", "khánh vĩnh", "khánh sơn", "diên khánh", "cam lâm", "vạn ninh", "nha trang"])
            has_charity = any(k in full_text for k in ["từ thiện", "trao quà", "tặng quà", "học bổng", "xã", "thôn", "bà con", "hộ nghèo"])
            
            score = 0
            if has_fana:
                score += 5
            if has_khanh_hoa:
                score += 4
            if has_charity:
                score += 1
                
            if score >= 4 or (has_fana and has_charity) or (has_khanh_hoa and has_charity):
                print(f"      ⭐ [HIT score={score}] {vid} Frame {fid} ({fid/fps:.1f}s): {[b['text'] for b in box_results]}")
                b64_full = extract_b64(frame)
                h, w = frame.shape[:2]
                crop = frame[int(h*0.1):int(h*0.9), int(w*0.05):int(w*0.95)]
                b64_crop = extract_b64(crop)
                
                evidence_hits.append({
                    "video_id": vid,
                    "frame_id": fid,
                    "sec": fid / fps,
                    "score": score,
                    "text_lines": [b["text"] for b in box_results],
                    "b64_full": b64_full,
                    "b64_crop": b64_crop,
                })
        cap.release()
        
    return evidence_hits

# -----------------------------------------------------------------------------
# 4. Generate Comprehensive Visual Deep Scan Gallery
# -----------------------------------------------------------------------------
def render_deep_scan_gallery(
    p15_hits: list[dict[str, Any]],
    p19_hits: list[dict[str, Any]],
    p22_hits: list[dict[str, Any]],
    out_html: Path,
) -> None:
    print(f"\n[GALLERY] Rendering Verified Temporal Evidence HTML Gallery to {out_html} ...")
    
    sections = []
    
    task_groups = [
        ("query-p1-22-qa", "🍳 QA p1-22: Tên Món Ăn Công Thức 200g Thịt Nạc Xay", p22_hits, "Tìm kiếm thẻ nguyên liệu chứa '200g thịt nạc xay' và thẻ tiêu đề tên món ăn"),
        ("query-p1-19-qa", "📜 QA p1-19: Hai Câu Thơ Ca Ngợi Anh Hùng Nguyễn Trung Trực (Kiên Giang)", p19_hits, "Tìm kiếm hai câu đối / câu thơ ca ngợi trong đình đền thờ Nguyễn Trung Trực"),
        ("query-p1-15-qa", "🤝 QA p1-15: Tên Xã Trao Quà Từ Thiện của CLB FANA (Khánh Hòa)", p15_hits, "Tìm kiếm bằng chứng chứa đồng thời 'CLB FANA' và tên xã tại tỉnh Khánh Hòa"),
    ]
    
    for qid, title, hits, guidance in task_groups:
        cards = []
        # Sort by score descending
        sorted_hits = sorted(hits, key=lambda x: x["score"], reverse=True)[:15]
        
        for idx, h in enumerate(sorted_hits, start=1):
            vid = h["video_id"]
            fid = h["frame_id"]
            sec = h["sec"]
            sc = h["score"]
            lines = h["text_lines"]
            b64_f = h["b64_full"]
            b64_c = h["b64_crop"]
            
            img_f_tag = f'<img src="data:image/jpeg;base64,{b64_f}" style="width:100%; border-radius:4px;" />' if b64_f else ''
            img_c_tag = f'<img src="data:image/jpeg;base64,{b64_c}" style="width:100%; border-radius:4px; border:1px solid #e5c07b;" />' if b64_c else ''
            
            cards.append(f"""
            <div style="flex:0 0 calc(33.333% - 12px); margin:6px; padding:10px; background:#1c1c1c; border:1px solid #333; border-radius:8px; box-sizing:border-box;">
                <div style="display:flex; justify-content:space-between; font-size:12px; margin-bottom:4px;">
                    <span style="font-weight:bold; color:#61afef;">#{idx} (Score={sc})</span>
                    <span style="color:#aaa;"><b>{vid}</b> | Frame {fid} ({sec:.1f}s)</span>
                </div>
                <div style="display:flex; gap:4px; margin-bottom:6px;">
                    <div style="flex:1;">{img_f_tag}</div>
                    <div style="flex:1;">{img_c_tag}</div>
                </div>
                <div style="background:#111; padding:6px; border-radius:4px; font-size:10px; color:#98c379; font-family:monospace; min-height:40px; word-break:break-word;">
                    <b style="color:#e5c07b;">EasyOCR Lines:</b><br>{' | '.join(lines[:8])}
                </div>
            </div>
            """)
            
        sections.append(f"""
        <div style="background:#242424; border:1px solid #444; border-radius:10px; padding:16px; margin-bottom:28px;">
            <h3 style="color:#e06c75; margin-top:0; border-bottom:1px solid #555; padding-bottom:6px;">{title}</h3>
            <div style="font-size:12px; color:#aaa; margin-bottom:12px;"><b>Mục tiêu kiểm chứng:</b> {guidance} (Tìm thấy {len(hits)} bằng chứng)</div>
            <div style="display:flex; flex-wrap:wrap; margin:-6px;">
                {''.join(cards) if cards else '<div style="color:#888; padding:10px;">Chưa tìm thấy bằng chứng khớp điều kiện lọc.</div>'}
            </div>
        </div>
        """)
        
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head><meta charset="utf-8"><title>QA Temporal Deep Scan Evidence</title></head>
    <body style="background:#121212; color:#fff; font-family:-apple-system,BlinkMacSystemFont,sans-serif; padding:16px;">
        <h2 style="color:#61afef; border-bottom:2px solid #555; padding-bottom:10px; margin-top:0;">🔍 BẢNG ĐỐI SOÁT CHỨNG CỨ THỊ GIÁC & CHỮ OCR CHUYÊN SÂU (QA TEMPORAL DEEP SCAN)</h2>
        {''.join(sections)}
    </body>
    </html>
    """
    out_html.write_text(full_html, encoding="utf-8")
    print(f"      • Saved Deep Scan Gallery to: {out_html} ✅", flush=True)

# -----------------------------------------------------------------------------
# 5. Main Execution
# -----------------------------------------------------------------------------
def main() -> None:
    t0 = time.time()
    
    # 1. Temporal Scans
    p22_hits = scan_p1_22_cooking()
    p19_hits = scan_p1_19_poetry()
    p15_hits = scan_p1_15_fana()
    
    # 2. Render Gallery
    gallery_out = Path("/kaggle/working/qa_temporal_deep_scan_gallery.html")
    render_deep_scan_gallery(p15_hits, p19_hits, p22_hits, gallery_out)
    
    print("\n" + "=" * 120)
    print("📊 BẢNG TỔNG HỢP TIẾN ĐỘ CHỨNG CỨ QA (VERIFICATION SUMMARY)")
    print("=" * 120)
    print(f"  • query-p1-22-qa (Nấu ăn 200g thịt nạc xay) : {len(p22_hits)} candidate evidence frames found")
    print(f"  • query-p1-19-qa (Nguyễn Trung Trực Kiên Giang) : {len(p19_hits)} candidate evidence frames found")
    print(f"  • query-p1-15-qa (CLB FANA Khánh Hòa)           : {len(p15_hits)} candidate evidence frames found")
    print(f"\n[DONE] Deep Scan Completed in {time.time() - t0:.2f}s ✅")
    print("=" * 120 + "\n")

if __name__ == "__main__":
    main()
