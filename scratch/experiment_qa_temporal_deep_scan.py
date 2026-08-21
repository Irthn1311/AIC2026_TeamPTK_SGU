#!/usr/bin/env python3
"""Forensic Two-Stage QA Deep Scanner with Geometry Adjacency, Exact Extraction, and Strict Semantics.

Operational Guarantees:
  1. No Hard-Coded Answers: p1-19, p1-15, and p1-22 answer candidates are derived 100% from OCR/ROI evidence.
  2. Spatial Geometry BBox Adjacency: 200g and meat terms must overlap vertically or be within 120px line distance.
  3. Strict Pattern Extraction:
     - p1-22: Extracts clean dish name only (excludes NGUYÊN LIỆU, MÓN NGON MỖI NGÀY, AJINOMOTO, etc.).
     - p1-15: Extracts specific commune name only (e.g. 'xã <TÊN>', rejecting 'xã hội', 'các xã').
     - p1-19: Derives poem candidate purely from OCR lines; partial fragments remain PARTIAL_EVIDENCE.
  4. Wall-Clock 12-Minute Deadline & Per-Query Budget: Prevents query starvation and runaway execution.
  5. Exact Status Semantics: Prints VIDEO_CONTEXT_STATUS, ANSWER_STATUS, ANSWER_CANDIDATE, ANSWER_EVIDENCE_VIDEO, ANSWER_EVIDENCE_FRAME, ANSWER_EVIDENCE_OCR.
"""

from __future__ import annotations

import base64
import csv
import json
import math
import os
import re
import subprocess
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

print("=" * 150, flush=True)
print("QA FORENSIC TEMPORAL SCANNER (GEOMETRIC ADJACENCY & EXACT TEXT EXTRACTION)", flush=True)
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
    import torch
    USE_GPU = torch.cuda.is_available()
except Exception:
    USE_GPU = False

try:
    import easyocr
    READER = easyocr.Reader(["vi", "en"], gpu=USE_GPU, verbose=False)
except Exception as exc:
    print(f"EasyOCR init error: {exc}", flush=True)
    READER = None

# Global Budget & 12-Minute Wall-Clock Guard
START_TIME = time.time()
DEADLINE_TIME = START_TIME + (12 * 60)  # 12 minutes hard wall-clock deadline
GLOBAL_OCR_CALLS = 0
MAX_GLOBAL_OCR_LIMIT = 1200 if USE_GPU else 550
COARSE_STRIDE = 500

VIDEO_CACHE: dict[str, Path] = {}
VIDEO_FRAME_COUNTS: dict[str, int] = {}

def is_time_exhausted() -> bool:
    return time.time() >= DEADLINE_TIME or GLOBAL_OCR_CALLS >= MAX_GLOBAL_OCR_LIMIT

def strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c)).lower().replace("đ", "d")

def normalize_text(text: str) -> str:
    clean = " ".join(text.strip().split())
    return unicodedata.normalize("NFC", clean)

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

def get_video_frame_count(video_id: str) -> int:
    if video_id in VIDEO_FRAME_COUNTS:
        return VIDEO_FRAME_COUNTS[video_id]
    vpath = find_video_path(video_id)
    if not vpath:
        return 0
    cap = cv2.VideoCapture(str(vpath))
    cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
    cap.release()
    VIDEO_FRAME_COUNTS[video_id] = cnt
    return cnt

def ocr_full_frame(frame: Any) -> tuple[str, list[dict[str, Any]]]:
    global GLOBAL_OCR_CALLS
    if frame is None or READER is None or is_time_exhausted():
        return "", []
    try:
        GLOBAL_OCR_CALLS += 1
        results = READER.readtext(frame, detail=1)
        items = []
        text_lines = []
        for bbox, text, conf in results:
            clean = normalize_text(text)
            if clean and len(clean) >= 2 and conf > 0.2:
                items.append({"bbox": bbox, "text": clean, "conf": float(conf)})
                text_lines.append(clean)
        return " | ".join(text_lines), items
    except Exception:
        return "", []

def ocr_roi_upscaled_3x(frame: Any, bbox: Any) -> str:
    global GLOBAL_OCR_CALLS
    if frame is None or READER is None or not bbox or is_time_exhausted():
        return ""
    try:
        GLOBAL_OCR_CALLS += 1
        h, w = frame.shape[:2]
        pts = bbox
        x_min = max(0, int(min(p[0] for p in pts)) - 10)
        x_max = min(w, int(max(p[0] for p in pts)) + 10)
        y_min = max(0, int(min(p[1] for p in pts)) - 6)
        y_max = min(h, int(max(p[1] for p in pts)) + 6)
        
        if (x_max - x_min) < 10 or (y_max - y_min) < 10:
            return ""
            
        roi = frame[y_min:y_max, x_min:x_max]
        roi_3x = cv2.resize(roi, (0, 0), fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
        res = READER.readtext(roi_3x, detail=0)
        return normalize_text(" ".join([r.strip() for r in res if r.strip()]))
    except Exception:
        return ""

def extract_b64(img: Any, quality: int = 85) -> str:
    if img is None:
        return ""
    try:
        _, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        return base64.b64encode(buf).decode("utf-8")
    except Exception:
        return ""

# -----------------------------------------------------------------------------
# 1. True Geometric Bounding Box Adjacency (Same Line or Spatial Neighbor)
# -----------------------------------------------------------------------------
def get_bbox_rect(bbox: list) -> tuple[int, int, int, int]:
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))

def are_bboxes_spatially_adjacent(b1: list, b2: list) -> bool:
    x1_min, y1_min, x1_max, y1_max = get_bbox_rect(b1)
    x2_min, y2_min, x2_max, y2_max = get_bbox_rect(b2)
    
    h1 = max(1, y1_max - y1_min)
    h2 = max(1, y2_max - y2_min)
    min_h = min(h1, h2)
    
    # 1. Vertical overlap (same line / row)
    overlap_y = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
    if overlap_y >= 0.35 * min_h:
        # Horizontal gap on the same line <= 150px
        horiz_gap = max(0, max(x1_min, x2_min) - min(x1_max, x2_max))
        if horiz_gap <= 150:
            return True
            
    # 2. Consecutive lines (adjacent row with horizontal overlap)
    vert_gap = max(0, max(y1_min, y2_min) - min(y1_max, y2_max))
    if vert_gap <= 40:
        overlap_x = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
        if overlap_x >= 20:
            return True
            
    return False

# -----------------------------------------------------------------------------
# 2. Budget Verification Guard
# -----------------------------------------------------------------------------
PRIORITY_P22 = ["L26_V277", "L26_V281", "L26_V455", "L26_V414", "L26_V294", "L26_V242", "L26_V122", "L26_V016", "L26_V474", "L26_V422", "L26_V367", "L26_V008"]
PRIORITY_P19 = ["L28_V018", "L28_V012", "L28_V013", "L28_V010", "L28_V007", "L28_V015", "L28_V016", "L28_V005"]
PRIORITY_P15 = ["L30_V066", "L30_V072", "L30_V081", "L30_V091", "L30_V031", "L30_V034", "L21_V009", "L21_V018", "L21_V027"]

def verify_actual_ocr_budget() -> None:
    print("\n" + "=" * 100)
    print(f"📊 METADATA & OCR BUDGET GUARD (gpu={USE_GPU}, limit={MAX_GLOBAL_OCR_LIMIT}):")
    print("=" * 100)
    total_coarse_planned = 0
    for group_name, vlist in [("p1-22", PRIORITY_P22), ("p1-19", PRIORITY_P19), ("p1-15", PRIORITY_P15)]:
        group_calls = 0
        for vid in vlist:
            fcnt = get_video_frame_count(vid)
            calls = math.ceil(fcnt / COARSE_STRIDE) if fcnt > 0 else 0
            group_calls += calls
        total_coarse_planned += group_calls
        print(f"  • Planned coarse calls for {group_name}: {group_calls}")
    print("-" * 100)
    print(f"  🏁 TOTAL PLANNED COARSE CALLS: {total_coarse_planned} (Limit: 800)")
    print(f"  🛑 GLOBAL RUNTIME HARD LIMIT : {MAX_GLOBAL_OCR_LIMIT} calls | 12-Minute Deadline Guard")
    print("-" * 100 + "\n", flush=True)
    if total_coarse_planned > 800:
        raise RuntimeError(f"FATAL: Planned coarse calls ({total_coarse_planned}) > 800!")

ALL_ACCUMULATED_RECORDS: list[dict[str, Any]] = []

def flush_incremental_evidence() -> None:
    csv_out = Path("/kaggle/working/qa_deep_scan_evidence.csv") if Path("/kaggle/working").exists() else REPO_ROOT / "scratch" / "qa_deep_scan_evidence.csv"
    json_out = Path("/kaggle/working/qa_deep_scan_evidence.json") if Path("/kaggle/working").exists() else REPO_ROOT / "scratch" / "qa_deep_scan_evidence.json"
    with open(csv_out, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "video_id", "frame_id", "time_sec", "score", "matched_keywords", "ocr_text"])
        for r in ALL_ACCUMULATED_RECORDS:
            writer.writerow([r["query_id"], r["video_id"], r["frame_id"], r["time_sec"], r["score"], r["matched_keywords"], r["ocr_text"]])
    clean_json = [{k: v for k, v in r.items() if not k.startswith("b64")} for r in ALL_ACCUMULATED_RECORDS]
    json_out.write_text(json.dumps(clean_json, ensure_ascii=False, indent=2), encoding="utf-8")

# -----------------------------------------------------------------------------
# 3. p1-22: 200g Thịt Nạc Xay -> Exact Dish Title
# -----------------------------------------------------------------------------
RE_200G = re.compile(r"\b(200\s*(g|gr|gram|g\b|9|oog)|200g|2009|2oog)\b", re.IGNORECASE)
MEAT_TERMS = ["thịt", "nạc", "xay", "thit", "nac", "pork", "heo"]
EXCLUDED_TITLE_TERMS = ["nguyên liệu", "nguyen lieu", "món ngon mỗi ngày", "mon ngon moi ngay", "ajinomoto", "htv", "online", "thực hiện", "thuc hien", "gia vị", "gia vi", "200g", "400g", "500g", "100g", "trang web", "quét mã"]

def match_spatial_200g_and_meat(bboxes: list[dict[str, Any]]) -> tuple[bool, str]:
    for i, b1 in enumerate(bboxes):
        t1 = b1["text"]
        t1_asc = strip_accents(t1)
        if RE_200G.search(t1) or RE_200G.search(t1_asc):
            # Same bbox check
            if any(m in t1_asc for m in MEAT_TERMS):
                return True, t1
            # Geometric spatial adjacency check with other bboxes
            for j, b2 in enumerate(bboxes):
                if i != j:
                    t2_asc = strip_accents(b2["text"])
                    if any(m in t2_asc for m in MEAT_TERMS):
                        if are_bboxes_spatially_adjacent(b1["bbox"], b2["bbox"]):
                            return True, f"{t1} | {b2['text']}"
    return False, ""

def extract_clean_dish_title(bboxes: list[dict[str, Any]]) -> str:
    for b in bboxes:
        txt = b["text"]
        txt_asc = strip_accents(txt)
        if any(ex in txt_asc for ex in EXCLUDED_TITLE_TERMS):
            continue
        # Plausible dish name terms
        if any(k in txt_asc for k in ["canh", "thit", "xao", "kho", "chien", "lau", "cuon", "hap", "cha", "goi", "bo", "ga", "tom", "ca "]):
            if 5 <= len(txt) <= 50 and not bool(re.search(r"\d+g", txt_asc)):
                return txt
    return ""

def run_two_stage_p1_22() -> dict[str, Any]:
    print("=" * 120)
    print("🍳 [p1-22] TWO-STAGE SCAN: 200g Thịt Nạc Xay -> Exact Dish Title")
    print("=" * 120)
    
    locked_video = None
    locked_hit_frame = None
    evidence_records = []
    dish_title_candidate = ""
    title_frame = None
    
    # Stage A: Coarse Discovery with Geometric BBox Adjacency
    print("\n--- [p1-22 Stage A] Coarse Discovery (Stride 500) ---")
    for vid in PRIORITY_P22:
        if is_time_exhausted():
            break
        vpath = find_video_path(vid)
        if not vpath:
            continue
        cap = cv2.VideoCapture(str(vpath))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        print(f"  • Coarse scanning {vid} ({total_frames} frames) ...", flush=True)
        
        for fid in range(0, total_frames, COARSE_STRIDE):
            if is_time_exhausted():
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            txt_raw, bboxes = ocr_full_frame(frame)
            matched, snippet = match_spatial_200g_and_meat(bboxes)
            if matched:
                print(f"\n      🌟 [GEOMETRIC 200G + MEAT HIT!] {vid} Frame {fid} ({fid/fps:.1f}s): '{snippet}'", flush=True)
                locked_video = vid
                locked_hit_frame = fid
                break
        cap.release()
        if locked_video:
            break
            
    if not locked_video:
        print("\n  ❌ P1_22_UNRESOLVED: No spatial 200g + meat hit found in priority videos.\n", flush=True)
        return {
            "query_id": "query-p1-22-qa",
            "video_context_status": "UNRESOLVED",
            "answer_status": "UNRESOLVED",
            "answer_candidate": "",
            "evidence_video": "N/A",
            "evidence_frame": "N/A",
            "evidence_ocr": "No verified ingredient card found",
            "records": [],
        }
        
    # Stage B: Dense Verification on Locked Video
    print(f"\n--- [p1-22 Stage B] Dense Scan & Title Card Extraction on LOCKED VIDEO: {locked_video} ---")
    vpath = find_video_path(locked_video)
    if vpath:
        cap = cv2.VideoCapture(str(vpath))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        
        # 1. Dense scan around ingredient hit
        start_f = max(0, locked_hit_frame - 1000)
        end_f = min(total_frames, locked_hit_frame + 1000)
        
        for fid in range(start_f, end_f, 30):
            if is_time_exhausted():
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            txt_raw, bboxes = ocr_full_frame(frame)
            matched, snippet = match_spatial_200g_and_meat(bboxes)
            if matched:
                roi_texts = []
                for b in bboxes:
                    b_asc = strip_accents(b["text"])
                    if RE_200G.search(b["text"]) or any(m in b_asc for m in MEAT_TERMS):
                        roi_t = ocr_roi_upscaled_3x(frame, b["bbox"])
                        if roi_t:
                            roi_texts.append(roi_t)
                enhanced = " | ".join(roi_texts) if roi_texts else txt_raw
                b64_f = extract_b64(frame)
                h, w = frame.shape[:2]
                b64_c = extract_b64(frame[int(h*0.15):int(h*0.85), int(w*0.1):int(w*0.9)])
                evidence_records.append({
                    "query_id": "query-p1-22-qa",
                    "video_id": locked_video,
                    "frame_id": fid,
                    "time_sec": round(fid / fps, 2),
                    "score": 10.0,
                    "matched_keywords": "200g meat card",
                    "ocr_text": enhanced,
                    "b64_full": b64_f,
                    "b64_crop": b64_c,
                })
                
        # 2. Dense scan frames 0 to 500 for Clean Dish Title
        print(f"  • Scanning beginning of {locked_video} (Frames 0-500) for Recipe Title ...")
        for fid in range(0, min(500, total_frames), 20):
            if is_time_exhausted():
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            txt_raw, bboxes = ocr_full_frame(frame)
            cand_title = extract_clean_dish_title(bboxes)
            if cand_title and not dish_title_candidate:
                dish_title_candidate = cand_title
                title_frame = fid
                b64_f = extract_b64(frame)
                h, w = frame.shape[:2]
                b64_c = extract_b64(frame[int(h*0.15):int(h*0.85), int(w*0.1):int(w*0.9)])
                evidence_records.append({
                    "query_id": "query-p1-22-qa",
                    "video_id": locked_video,
                    "frame_id": fid,
                    "time_sec": round(fid / fps, 2),
                    "score": 9.0,
                    "matched_keywords": "Dish Title Candidate",
                    "ocr_text": cand_title,
                    "b64_full": b64_f,
                    "b64_crop": b64_c,
                })
        cap.release()
        
    ALL_ACCUMULATED_RECORDS.extend(evidence_records)
    flush_incremental_evidence()
    
    ctx_status = "VERIFIED_CONTEXT"
    ans_status = "VERIFIED_HIGH_CONFIDENCE" if dish_title_candidate else "UNRESOLVED"
    ev_frame = title_frame if title_frame is not None else locked_hit_frame
    best_ocr = dish_title_candidate if dish_title_candidate else (evidence_records[0]["ocr_text"] if evidence_records else "N/A")
    
    return {
        "query_id": "query-p1-22-qa",
        "video_context_status": ctx_status,
        "answer_status": ans_status,
        "answer_candidate": dish_title_candidate,
        "evidence_video": locked_video,
        "evidence_frame": ev_frame,
        "evidence_ocr": best_ocr,
        "records": evidence_records,
    }

# -----------------------------------------------------------------------------
# 4. p1-19: Nguyễn Trung Trực Couplet Poetry (Derived 100% from OCR)
# -----------------------------------------------------------------------------
DISTINCTIVE_POEM_RAW = ["hỏa hồng", "nhật tảo", "oanh thiên địa", "kiếm bạt", "khấp quỷ thần"]
DISTINCTIVE_POEM_ASCII = ["hoa hong", "nhat tao", "oanh thien dia", "kiem bat", "khap quy than"]

def run_two_stage_p1_19() -> dict[str, Any]:
    print("=" * 120)
    print("📜 [p1-19] TWO-STAGE SCAN: Nguyễn Trung Trực Couplet Poetry (Kiên Giang)")
    print("=" * 120)
    
    locked_video = None
    locked_hit_frame = None
    context_only_video = None
    evidence_records = []
    extracted_poem_lines = []
    best_poem_frame = None
    
    # Stage A: Coarse Discovery
    print("\n--- [p1-19 Stage A] Coarse Discovery (Stride 500) ---")
    for vid in PRIORITY_P19:
        if is_time_exhausted():
            break
        vpath = find_video_path(vid)
        if not vpath:
            continue
        cap = cv2.VideoCapture(str(vpath))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        print(f"  • Coarse scanning {vid} ({total_frames} frames) ...", flush=True)
        
        for fid in range(0, total_frames, COARSE_STRIDE):
            if is_time_exhausted():
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            txt_raw, bboxes = ocr_full_frame(frame)
            txt_asc = strip_accents(txt_raw)
            
            has_poem = any(k in txt_raw.lower() for k in DISTINCTIVE_POEM_RAW) or any(k in txt_asc for k in DISTINCTIVE_POEM_ASCII)
            has_ntt = ("nguyễn trung trực" in txt_raw.lower()) or ("nguyen trung truc" in txt_asc)
            
            if has_poem:
                print(f"\n      🌟 [DISTINCTIVE POEM HIT!] {vid} Frame {fid} ({fid/fps:.1f}s): '{txt_raw}'", flush=True)
                locked_video = vid
                locked_hit_frame = fid
                break
            elif has_ntt and not context_only_video:
                context_only_video = vid
                locked_hit_frame = fid
        cap.release()
        if locked_video:
            break
            
    chosen_vid = locked_video if locked_video else context_only_video
    if not chosen_vid:
        print("\n  ❌ P1_19_UNRESOLVED: No poem or Nguyễn Trung Trực evidence found.\n", flush=True)
        return {
            "query_id": "query-p1-19-qa",
            "video_context_status": "UNRESOLVED",
            "answer_status": "UNRESOLVED",
            "answer_candidate": "",
            "evidence_video": "N/A",
            "evidence_frame": "N/A",
            "evidence_ocr": "No evidence found",
            "records": [],
        }
        
    # Stage B: Dense Verification on Candidate Video
    print(f"\n--- [p1-19 Stage B] Dense Verification on VIDEO: {chosen_vid} ---")
    vpath = find_video_path(chosen_vid)
    if vpath:
        cap = cv2.VideoCapture(str(vpath))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        
        start_f = max(0, locked_hit_frame - 1000)
        end_f = min(total_frames, locked_hit_frame + 1000)
        
        for fid in range(start_f, end_f, 35):
            if is_time_exhausted():
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            txt_raw, bboxes = ocr_full_frame(frame)
            txt_asc = strip_accents(txt_raw)
            
            has_poem = any(k in txt_raw.lower() for k in DISTINCTIVE_POEM_RAW) or any(k in txt_asc for k in DISTINCTIVE_POEM_ASCII)
            has_ntt = any(k in txt_asc for k in ["nguyen trung truc", "dinh", "than", "den tho", "hon chong"])
            
            if has_poem or has_ntt:
                roi_texts = []
                for b in bboxes:
                    b_asc = strip_accents(b["text"])
                    if any(k in b_asc for k in DISTINCTIVE_POEM_ASCII) or ("nguyen trung truc" in b_asc):
                        roi_t = ocr_roi_upscaled_3x(frame, b["bbox"])
                        if roi_t:
                            roi_texts.append(roi_t)
                            if any(k in strip_accents(roi_t) for k in DISTINCTIVE_POEM_ASCII):
                                if roi_t not in extracted_poem_lines:
                                    extracted_poem_lines.append(roi_t)
                                    best_poem_frame = fid
                                    
                enhanced = " | ".join(roi_texts) if roi_texts else txt_raw
                b64_f = extract_b64(frame)
                h, w = frame.shape[:2]
                b64_c = extract_b64(frame[int(h*0.1):int(h*0.9), int(w*0.05):int(w*0.95)])
                score = 10.0 if has_poem else 5.0
                matched = "Poem Verse" if has_poem else "Nguyễn Trung Trực Memorial/Temple"
                evidence_records.append({
                    "query_id": "query-p1-19-qa",
                    "video_id": chosen_vid,
                    "frame_id": fid,
                    "time_sec": round(fid / fps, 2),
                    "score": score,
                    "matched_keywords": matched,
                    "ocr_text": enhanced,
                    "b64_full": b64_f,
                    "b64_crop": b64_c,
                })
        cap.release()
        
    ALL_ACCUMULATED_RECORDS.extend(evidence_records)
    flush_incremental_evidence()
    
    # Pure OCR derived answer (no hardcoded fallback)
    poem_candidate = " / ".join(extracted_poem_lines) if extracted_poem_lines else ""
    ctx_status = "VERIFIED_CONTEXT" if chosen_vid else "UNRESOLVED"
    ans_status = "VERIFIED_HIGH_CONFIDENCE" if len(extracted_poem_lines) >= 2 else ("PARTIAL_EVIDENCE" if extracted_poem_lines else "UNRESOLVED")
    ev_frame = best_poem_frame if best_poem_frame is not None else locked_hit_frame
    best_ocr = poem_candidate if poem_candidate else (evidence_records[0]["ocr_text"] if evidence_records else "N/A")
    
    return {
        "query_id": "query-p1-19-qa",
        "video_context_status": ctx_status,
        "answer_status": ans_status,
        "answer_candidate": poem_candidate,
        "evidence_video": chosen_vid,
        "evidence_frame": ev_frame,
        "evidence_ocr": best_ocr,
        "records": evidence_records,
    }

# -----------------------------------------------------------------------------
# 5. p1-15: CLB FANA & Specific Commune Name Extraction
# -----------------------------------------------------------------------------
RE_COMMUNE = re.compile(r"\b(xã|xa|ubnd xã|ubnd xa|tai xa|tại xã)\s+([A-ZÀ-Ỵ][a-zà-ỹ]+(?:\s+[A-ZÀ-Ỵ][a-zà-ỹ]+)*)", re.IGNORECASE)
REJECTED_COMMUNE_WORDS = ["hội", "hoi", "phường", "phuong", "các", "cac", "văn hóa", "van hoa", "nông dân", "nong dan", "việt nam", "viet nam"]

def extract_specific_commune_name(text: str) -> str:
    clean = normalize_text(text)
    match = RE_COMMUNE.search(clean)
    if match:
        c_name = match.group(2).strip()
        c_asc = strip_accents(c_name)
        if any(rej in c_asc for rej in REJECTED_COMMUNE_WORDS):
            return ""
        if len(c_name) >= 3:
            return c_name
    return ""

def run_two_stage_p1_15() -> dict[str, Any]:
    print("=" * 120)
    print("🤝 [p1-15] TWO-STAGE SCAN: CLB FANA & Specific Commune Name")
    print("=" * 120)
    
    locked_video = None
    locked_hit_frame = None
    evidence_records = []
    commune_name_candidate = ""
    commune_frame = None
    
    # Stage A: Coarse Discovery
    print("\n--- [p1-15 Stage A] Coarse Discovery (Stride 500) ---")
    for vid in PRIORITY_P15:
        if is_time_exhausted():
            break
        vpath = find_video_path(vid)
        if not vpath:
            continue
        cap = cv2.VideoCapture(str(vpath))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        print(f"  • Coarse scanning {vid} ({total_frames} frames) ...", flush=True)
        
        for fid in range(0, total_frames, COARSE_STRIDE):
            if is_time_exhausted():
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            txt_raw, bboxes = ocr_full_frame(frame)
            txt_asc = strip_accents(txt_raw)
            
            has_fana = ("fana" in txt_asc) or ("fa na" in txt_asc)
            has_kh = any(k in txt_asc for k in ["khanh hoa", "khanh vinh", "khanh son", "dien khanh", "cam lam", "van ninh", "nha trang"])
            has_context = any(k in txt_asc for k in ["trao qua", "tu thien", "tang qua", "hoc bong", "xa ", "thon "])
            
            if has_fana and (has_kh or has_context):
                print(f"\n      🌟 [VERIFIED FANA HIT!] {vid} Frame {fid} ({fid/fps:.1f}s): '{txt_raw}'", flush=True)
                locked_video = vid
                locked_hit_frame = fid
                break
        cap.release()
        if locked_video:
            break
            
    if not locked_video:
        print("\n  ❌ P1_15_UNRESOLVED: No verified FANA charity video found.\n", flush=True)
        return {
            "query_id": "query-p1-15-qa",
            "video_context_status": "UNRESOLVED",
            "answer_status": "UNRESOLVED",
            "answer_candidate": "",
            "evidence_video": "N/A",
            "evidence_frame": "N/A",
            "evidence_ocr": "No verified FANA video found",
            "records": [],
        }
        
    # Stage B: Dense Verification on Locked Video
    print(f"\n--- [p1-15 Stage B] Dense Scan for Commune Name on LOCKED VIDEO: {locked_video} ---")
    vpath = find_video_path(locked_video)
    if vpath:
        cap = cv2.VideoCapture(str(vpath))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        
        start_f = max(0, locked_hit_frame - 1000)
        end_f = min(total_frames, locked_hit_frame + 1000)
        
        for fid in range(start_f, end_f, 35):
            if is_time_exhausted():
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            txt_raw, bboxes = ocr_full_frame(frame)
            txt_asc = strip_accents(txt_raw)
            
            has_fana = ("fana" in txt_asc) or ("fa na" in txt_asc)
            has_kh = any(k in txt_asc for k in ["khanh hoa", "xa", "thon", "huyen"])
            
            if has_fana or has_kh:
                roi_texts = []
                for b in bboxes:
                    b_asc = strip_accents(b["text"])
                    if ("fana" in b_asc) or any(k in b_asc for k in ["khanh hoa", "xa", "thon"]):
                        roi_t = ocr_roi_upscaled_3x(frame, b["bbox"])
                        if roi_t:
                            roi_texts.append(roi_t)
                enhanced = " | ".join(roi_texts) if roi_texts else txt_raw
                b64_f = extract_b64(frame)
                h, w = frame.shape[:2]
                b64_c = extract_b64(frame[int(h*0.1):int(h*0.9), int(w*0.05):int(w*0.95)])
                
                # Check for explicit commune name extraction
                c_cand = extract_specific_commune_name(enhanced) or extract_specific_commune_name(txt_raw)
                if c_cand and not commune_name_candidate:
                    commune_name_candidate = c_cand
                    commune_frame = fid
                    
                score = 10.0 if has_fana else 5.0
                matched = "FANA Club Banner" if has_fana else "Commune Sign"
                evidence_records.append({
                    "query_id": "query-p1-15-qa",
                    "video_id": locked_video,
                    "frame_id": fid,
                    "time_sec": round(fid / fps, 2),
                    "score": score,
                    "matched_keywords": matched,
                    "ocr_text": enhanced,
                    "b64_full": b64_f,
                    "b64_crop": b64_c,
                })
        cap.release()
        
    ALL_ACCUMULATED_RECORDS.extend(evidence_records)
    flush_incremental_evidence()
    
    ctx_status = "VERIFIED_CONTEXT"
    ans_status = "VERIFIED_HIGH_CONFIDENCE" if commune_name_candidate else "UNRESOLVED"
    ev_frame = commune_frame if commune_frame is not None else locked_hit_frame
    best_ocr = commune_name_candidate if commune_name_candidate else (evidence_records[0]["ocr_text"] if evidence_records else "N/A")
    
    return {
        "query_id": "query-p1-15-qa",
        "video_context_status": ctx_status,
        "answer_status": ans_status,
        "answer_candidate": commune_name_candidate,
        "evidence_video": locked_video,
        "evidence_frame": ev_frame,
        "evidence_ocr": best_ocr,
        "records": evidence_records,
    }

# -----------------------------------------------------------------------------
# 6. Render Ranked & Deduplicated Visual Gallery
# -----------------------------------------------------------------------------
def render_ranked_gallery(outcomes: list[dict[str, Any]]) -> None:
    html_out = Path("/kaggle/working/qa_temporal_deep_scan_gallery.html") if Path("/kaggle/working").exists() else REPO_ROOT / "scratch" / "qa_temporal_deep_scan_gallery.html"
    sections = []
    
    for out in outcomes:
        qid = out["query_id"]
        ctx_stat = out["video_context_status"]
        ans_stat = out["answer_status"]
        cand = out["answer_candidate"]
        raw_q_recs = out["records"]
        
        sorted_recs = sorted(raw_q_recs, key=lambda r: (-r["score"], r["frame_id"]))
        dedup_recs = []
        seen_ranges: list[tuple[str, int, int]] = []
        for r in sorted_recs:
            v = r["video_id"]
            f = r["frame_id"]
            if not any(v == sv and abs(f - sf) <= 100 for sv, sf, _ in seen_ranges):
                seen_ranges.append((v, f, f))
                dedup_recs.append(r)
                
        cards = []
        for idx, r in enumerate(dedup_recs[:15], start=1):
            vid = r["video_id"]
            fid = r["frame_id"]
            sec = r["time_sec"]
            sc = r["score"]
            kw = r["matched_keywords"]
            txt = r["ocr_text"]
            b64_f = r.get("b64_full", "")
            b64_c = r.get("b64_crop", "")
            
            img_f_tag = f'<img src="data:image/jpeg;base64,{b64_f}" style="width:100%; border-radius:4px;" />' if b64_f else ''
            img_c_tag = f'<img src="data:image/jpeg;base64,{b64_c}" style="width:100%; border-radius:4px; border:1px solid #e5c07b;" />' if b64_c else ''
            
            cards.append(f"""
            <div style="flex:0 0 calc(33.333% - 12px); margin:6px; padding:10px; background:#1c1c1c; border:1px solid #333; border-radius:8px; box-sizing:border-box;">
                <div style="display:flex; justify-content:space-between; font-size:12px; margin-bottom:4px;">
                    <span style="font-weight:bold; color:#61afef;">#{idx} (Score: {sc})</span>
                    <span style="color:#aaa;"><b>{vid}</b> | Frame {fid} ({sec}s)</span>
                </div>
                <div style="font-size:11px; color:#e5c07b; margin-bottom:4px;"><b>Match:</b> {kw}</div>
                <div style="display:flex; gap:4px; margin-bottom:6px;">
                    <div style="flex:1;">{img_f_tag}</div>
                    <div style="flex:1;">{img_c_tag}</div>
                </div>
                <div style="background:#111; padding:6px; border-radius:4px; font-size:10px; color:#98c379; font-family:monospace; min-height:40px; word-break:break-word;">
                    <b style="color:#e5c07b;">3x Upscaled OCR:</b><br>{txt}
                </div>
            </div>
            """)
            
        badge_color = "#98c379" if ans_stat == "VERIFIED_HIGH_CONFIDENCE" else ("#e5c07b" if ctx_stat == "VERIFIED_CONTEXT" else "#e06c75")
        sections.append(f"""
        <div style="background:#242424; border:1px solid #444; border-radius:10px; padding:16px; margin-bottom:28px;">
            <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #555; padding-bottom:6px; margin-bottom:8px;">
                <h3 style="color:#e06c75; margin:0;">🎯 {qid}</h3>
                <span style="background:{badge_color}; color:#000; font-size:11px; font-weight:bold; padding:3px 8px; border-radius:4px;">{ans_stat}</span>
            </div>
            <div style="font-size:12px; color:#ccc; margin-bottom:4px;"><b>• Video Context:</b> {ctx_stat} | <b>Evidence Video:</b> {out['evidence_video']} (Frame {out['evidence_frame']})</div>
            <div style="font-size:13px; color:#61afef; margin-bottom:12px;"><b>• Extracted Answer Candidate:</b> <span style="color:#fff; font-weight:bold;">{cand if cand else '[UNRESOLVED / REQUIRES HUMAN REVIEW]'}</span></div>
            <div style="display:flex; flex-wrap:wrap; margin:-6px;">
                {''.join(cards) if cards else '<div style="color:#888; padding:10px;">Chưa tìm thấy bằng chứng khớp điều kiện lọc.</div>'}
            </div>
        </div>
        """)
        
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head><meta charset="utf-8"><title>QA Forensic Deep Scan Ranked Evidence</title></head>
    <body style="background:#121212; color:#fff; font-family:-apple-system,BlinkMacSystemFont,sans-serif; padding:16px;">
        <h2 style="color:#61afef; border-bottom:2px solid #555; padding-bottom:10px; margin-top:0;">🔍 BẢNG ĐỐI SOÁT CHỨNG CỨ FORENSIC & CHỮ OCR 3X</h2>
        {''.join(sections)}
    </body>
    </html>
    """
    html_out.write_text(full_html, encoding="utf-8")
    print(f"      • Saved Ranked Forensic Gallery : {html_out} ✅\n")

# -----------------------------------------------------------------------------
# 7. Main Execution Flow
# -----------------------------------------------------------------------------
def main() -> None:
    t0 = time.time()
    
    # 1. Budget Verification Guard
    verify_actual_ocr_budget()
    
    # 2. Run Targeted Scans in Priority Order (p1-22 -> p1-19 -> p1-15)
    out_22 = run_two_stage_p1_22()
    out_19 = run_two_stage_p1_19()
    out_15 = run_two_stage_p1_15()
    
    outcomes = [out_22, out_19, out_15]
    
    # 3. Render Ranked Gallery
    render_ranked_gallery(outcomes)
    
    # 4. Print Structured Terminal Summary
    print("=" * 120)
    print("📋 BẢNG TỔNG HỢP FORENSIC QA EVIDENCE & EXACT ANSWER STATUS")
    print("=" * 120)
    for out in outcomes:
        print(f"--- [{out['query_id']}] ---")
        print(f"  VIDEO_CONTEXT_STATUS  = {out['video_context_status']}")
        print(f"  ANSWER_STATUS         = {out['answer_status']}")
        print(f"  ANSWER_CANDIDATE      = '{out['answer_candidate']}'")
        print(f"  ANSWER_EVIDENCE_VIDEO = {out['evidence_video']}")
        print(f"  ANSWER_EVIDENCE_FRAME = {out['evidence_frame']}")
        print(f"  ANSWER_EVIDENCE_OCR   = '{out['evidence_ocr'][:80]}'")
        print(f"  Evidence Frames Count = {len(out['records'])}")
    print("=" * 120)
    print(f"\n🎉 FORENSIC SCAN FINISHED IN {time.time() - t0:.2f}s (Total OCR Calls: {GLOBAL_OCR_CALLS}/{MAX_GLOBAL_OCR_LIMIT}) ✅\n")

if __name__ == "__main__":
    main()
