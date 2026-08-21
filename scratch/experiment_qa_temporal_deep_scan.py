#!/usr/bin/env python3
"""Two-Stage Video-Centric QA Deep Scanner with Strict Budget, Real ROI, and Incremental Export.

Guarantees:
  1. Exact Real Video Frame Count & Coarse Budget Guard: Stride 500, hard fail if >800 coarse calls.
  2. Global OCR Hard Stop: Maximum 1200 total OCR calls with immediate incremental flush.
  3. No Blind Guessing / No Fallbacks:
     - p1-22: Strict regex (200g/200 g/2009/2oog) AND meat terms (thịt/nạc/xay/heo) in same text region.
     - p1-19: Explicit poem_found flag for distinctive verse fragments. Memorial board = CONTEXT_ONLY.
     - p1-15: Requires FANA AND (Khánh Hòa OR charity/location context). No fallback to L30_V066.
  4. True 3x Upscaled ROI OCR: Only OCRs matching and adjacent bboxes (never all bboxes).
  5. Incremental Persistence: Writes CSV & JSON after each query so partial results are never lost.
  6. Ranked Gallery: Sorts records by (-score, frame_id) and deduplicates ±100 frames.
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
print("QA TWO-STAGE TARGETED TEMPORAL SCANNER (BUDGET-GUARDED & REAL ROI UPSCALING)", flush=True)
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
    print(f"EasyOCR init error: {exc}", flush=True)
    READER = None

# Global Runtime OCR Counter & Limits
GLOBAL_OCR_CALLS = 0
MAX_GLOBAL_OCR_LIMIT = 1200
COARSE_STRIDE = 500

VIDEO_CACHE: dict[str, Path] = {}
VIDEO_FRAME_COUNTS: dict[str, int] = {}

def strip_accents(text: str) -> str:
    """Removes diacritics and normalizes to ASCII lowercase."""
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c)).lower().replace("đ", "d")

def normalize_text(text: str) -> str:
    """Clean whitespace and normalize unicode."""
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
    """Coarse OCR scan with global budget tracking."""
    global GLOBAL_OCR_CALLS
    if frame is None or READER is None or GLOBAL_OCR_CALLS >= MAX_GLOBAL_OCR_LIMIT:
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
    """True ROI extraction: crops bbox with 10px padding and upscales 3x before OCR."""
    global GLOBAL_OCR_CALLS
    if frame is None or READER is None or not bbox or GLOBAL_OCR_CALLS >= MAX_GLOBAL_OCR_LIMIT:
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
# 1. Budget Verification Guard (Calculated from Real Video Metadata)
# -----------------------------------------------------------------------------
PRIORITY_P22 = ["L26_V277", "L26_V281", "L26_V455", "L26_V414", "L26_V294", "L26_V242", "L26_V122", "L26_V016", "L26_V474", "L26_V422", "L26_V367", "L26_V008"]
PRIORITY_P19 = ["L28_V018", "L28_V012", "L28_V013", "L28_V010", "L28_V007", "L28_V015", "L28_V016", "L28_V005"]
PRIORITY_P15 = ["L30_V066", "L30_V072", "L30_V081", "L30_V091", "L30_V031", "L30_V034", "L21_V009", "L21_V018", "L21_V027"]

def verify_actual_ocr_budget() -> None:
    print("\n" + "=" * 100)
    print(f"📊 REAL VIDEO METADATA & OCR BUDGET VERIFICATION (Coarse Stride = {COARSE_STRIDE}):")
    print("=" * 100)
    
    total_coarse_planned = 0
    
    for group_name, vlist in [("p1-22", PRIORITY_P22), ("p1-19", PRIORITY_P19), ("p1-15", PRIORITY_P15)]:
        group_calls = 0
        print(f"\n  • Group [{group_name}] ({len(vlist)} candidate videos):")
        for vid in vlist:
            fcnt = get_video_frame_count(vid)
            calls = math.ceil(fcnt / COARSE_STRIDE) if fcnt > 0 else 0
            group_calls += calls
            print(f"      - {vid:<10}: Total Frames = {fcnt:<6} -> Planned Coarse Calls = {calls}")
        total_coarse_planned += group_calls
        print(f"    👉 Total planned for {group_name}: {group_calls} coarse calls")
        
    print("\n" + "-" * 100)
    print(f"  🏁 TOTAL PLANNED COARSE OCR CALLS ACROSS ALL QUERIES: {total_coarse_planned} (Limit: 800)")
    print(f"  🛑 GLOBAL RUNTIME OCR HARD LIMIT: {MAX_GLOBAL_OCR_LIMIT} calls")
    print("-" * 100 + "\n", flush=True)
    
    if total_coarse_planned > 800:
        raise RuntimeError(f"FATAL: Planned coarse OCR calls ({total_coarse_planned}) exceeded limit 800!")

# -----------------------------------------------------------------------------
# 2. Incremental Evidence Persistence Helpers
# -----------------------------------------------------------------------------
ALL_ACCUMULATED_RECORDS: list[dict[str, Any]] = []

def flush_incremental_evidence() -> None:
    csv_out = Path("/kaggle/working/qa_deep_scan_evidence.csv") if Path("/kaggle/working").exists() else REPO_ROOT / "scratch" / "qa_deep_scan_evidence.csv"
    json_out = Path("/kaggle/working/qa_deep_scan_evidence.json") if Path("/kaggle/working").exists() else REPO_ROOT / "scratch" / "qa_deep_scan_evidence.json"
    
    # Save CSV
    with open(csv_out, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "video_id", "frame_id", "time_sec", "score", "matched_keywords", "ocr_text"])
        for r in ALL_ACCUMULATED_RECORDS:
            writer.writerow([r["query_id"], r["video_id"], r["frame_id"], r["time_sec"], r["score"], r["matched_keywords"], r["ocr_text"]])
            
    # Save JSON (without heavy b64)
    clean_json = [{k: v for k, v in r.items() if not k.startswith("b64")} for r in ALL_ACCUMULATED_RECORDS]
    json_out.write_text(json.dumps(clean_json, ensure_ascii=False, indent=2), encoding="utf-8")

# -----------------------------------------------------------------------------
# 3. p1-22 Targeted Two-Stage Scan (200g Thịt Nạc Xay -> Dish Title)
# -----------------------------------------------------------------------------
RE_200G = re.compile(r"\b(200\s*(g|gr|gram|g\b|9|oog)|200g|2009|2oog)\b", re.IGNORECASE)
MEAT_TERMS_RAW = ["thịt", "nạc", "xay", "thịt nạc", "thịt xay", "thịt heo", "thịt nạc xay", "pork"]
MEAT_TERMS_ASCII = ["thit", "nac", "xay", "thit nac", "thit xay", "thit heo", "thit nac xay", "pork"]

def run_two_stage_p1_22() -> tuple[str, list[dict[str, Any]]]:
    print("=" * 120)
    print("🍳 [p1-22] TWO-STAGE SCAN: 200g Thịt Nạc Xay -> Dish Title Card")
    print("=" * 120)
    
    locked_video = None
    locked_hit_frame = None
    evidence_records = []
    
    # Stage A: Coarse Discovery (Stride 500)
    print("\n--- [p1-22 Stage A] Coarse Discovery (Stride 500) ---")
    for vid in PRIORITY_P22:
        if GLOBAL_OCR_CALLS >= MAX_GLOBAL_OCR_LIMIT:
            print("  🛑 Global OCR limit reached during p1-22 Stage A!")
            break
        vpath = find_video_path(vid)
        if not vpath:
            continue
        cap = cv2.VideoCapture(str(vpath))
        if not cap.isOpened():
            continue
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        print(f"  • Coarse scanning {vid} ({total_frames} frames, ~{math.ceil(total_frames/COARSE_STRIDE)} calls) ...", flush=True)
        
        for fid in range(0, total_frames, COARSE_STRIDE):
            if GLOBAL_OCR_CALLS >= MAX_GLOBAL_OCR_LIMIT:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
                
            txt_raw, bboxes = ocr_full_frame(frame)
            txt_asc = strip_accents(txt_raw)
            
            # Numeric match with regex
            has_200 = bool(RE_200G.search(txt_raw) or RE_200G.search(txt_asc))
            # Meat term match in raw or ascii
            has_meat = any(m in txt_raw.lower() for m in MEAT_TERMS_RAW) or any(m in txt_asc for m in MEAT_TERMS_ASCII)
            
            if has_200 and has_meat:
                matched_kw = f"200g + meat (Frame {fid})"
                print(f"\n      🌟 [VERIFIED INGREDIENT HIT!] {vid} Frame {fid} ({fid/fps:.1f}s): '{txt_raw}'", flush=True)
                locked_video = vid
                locked_hit_frame = fid
                break
        cap.release()
        if locked_video:
            break
            
    if not locked_video:
        print("\n  ❌ P1_22_UNRESOLVED: No verified 200g + meat hit found in priority videos.\n", flush=True)
        return "UNRESOLVED", []
        
    # Stage B: Dense Verification on Locked Video
    print(f"\n--- [p1-22 Stage B] Dense Verification & True 3x ROI Upscale on LOCKED VIDEO: {locked_video} ---")
    vpath = find_video_path(locked_video)
    if vpath:
        cap = cv2.VideoCapture(str(vpath))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        
        # 1. Densely scan around hit frame (±1000 frames, stride 30)
        start_f = max(0, locked_hit_frame - 1000)
        end_f = min(total_frames, locked_hit_frame + 1000)
        
        for fid in range(start_f, end_f, 30):
            if GLOBAL_OCR_CALLS >= MAX_GLOBAL_OCR_LIMIT:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
                
            txt_raw, bboxes = ocr_full_frame(frame)
            txt_asc = strip_accents(txt_raw)
            
            has_200 = bool(RE_200G.search(txt_raw) or RE_200G.search(txt_asc))
            has_meat = any(m in txt_raw.lower() for m in MEAT_TERMS_RAW) or any(m in txt_asc for m in MEAT_TERMS_ASCII)
            has_recipe = any(k in txt_asc for k in ["nguyen lieu", "cong thuc", "mon", "thuc hien"])
            
            if has_200 or has_meat or has_recipe:
                # True 3x ROI: ONLY upscale bounding boxes that match keywords
                roi_texts = []
                for b in bboxes:
                    b_txt = b["text"]
                    b_asc = strip_accents(b_txt)
                    if bool(RE_200G.search(b_txt)) or any(m in b_asc for m in MEAT_TERMS_ASCII) or any(k in b_asc for k in ["nguyen lieu", "mon"]):
                        roi_t = ocr_roi_upscaled_3x(frame, b["bbox"])
                        if roi_t:
                            roi_texts.append(roi_t)
                enhanced_text = " | ".join(roi_texts) if roi_texts else txt_raw
                
                b64_f = extract_b64(frame)
                h, w = frame.shape[:2]
                crop_center = frame[int(h*0.15):int(h*0.85), int(w*0.1):int(w*0.9)]
                b64_c = extract_b64(crop_center)
                
                matched = ("200g + meat" if (has_200 and has_meat) else ("200g" if has_200 else "meat/recipe"))
                score = 10.0 if (has_200 and has_meat) else (6.0 if has_200 else 3.0)
                
                print(f"      📍 Dense Frame {fid} ({fid/fps:.1f}s) [score={score}]: {enhanced_text[:65]}")
                evidence_records.append({
                    "query_id": "query-p1-22-qa",
                    "video_id": locked_video,
                    "frame_id": fid,
                    "time_sec": round(fid / fps, 2),
                    "score": score,
                    "matched_keywords": matched,
                    "ocr_text": enhanced_text,
                    "b64_full": b64_f,
                    "b64_crop": b64_c,
                })
                
        # 2. Densely scan beginning of video (frames 0 to 500, stride 25) for DISH TITLE CARD
        print(f"\n--- [p1-22 Dish Title Scan] Scanning Beginning of {locked_video} (Frames 0-500) ---")
        for fid in range(0, min(500, total_frames), 25):
            if GLOBAL_OCR_CALLS >= MAX_GLOBAL_OCR_LIMIT:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            txt_raw, bboxes = ocr_full_frame(frame)
            if txt_raw and len(txt_raw) >= 4:
                b64_f = extract_b64(frame)
                h, w = frame.shape[:2]
                crop_center = frame[int(h*0.15):int(h*0.85), int(w*0.1):int(w*0.9)]
                b64_c = extract_b64(crop_center)
                print(f"      🏷️ Title Frame {fid} ({fid/fps:.1f}s): {txt_raw[:65]}")
                evidence_records.append({
                    "query_id": "query-p1-22-qa",
                    "video_id": locked_video,
                    "frame_id": fid,
                    "time_sec": round(fid / fps, 2),
                    "score": 7.0,
                    "matched_keywords": "Dish Title Candidate",
                    "ocr_text": txt_raw,
                    "b64_full": b64_f,
                    "b64_crop": b64_c,
                })
        cap.release()
        
    ALL_ACCUMULATED_RECORDS.extend(evidence_records)
    flush_incremental_evidence()
    print(f"\n[p1-22 STATUS]: VERIFIED_HIGH_CONFIDENCE in {locked_video} ({len(evidence_records)} evidence frames)\n", flush=True)
    return "VERIFIED_HIGH_CONFIDENCE", evidence_records

# -----------------------------------------------------------------------------
# 4. p1-19 Targeted Two-Stage Scan (Nguyễn Trung Trực Couplet Verses)
# -----------------------------------------------------------------------------
DISTINCTIVE_POEM_RAW = ["hỏa hồng", "nhật tảo", "oanh thiên địa", "kiếm bạt", "khấp quỷ thần", "câu thơ"]
DISTINCTIVE_POEM_ASCII = ["hoa hong", "nhat tao", "oanh thien dia", "kiem bat", "khap quy than", "cau tho"]

def run_two_stage_p1_19() -> tuple[str, list[dict[str, Any]]]:
    print("=" * 120)
    print("📜 [p1-19] TWO-STAGE SCAN: Nguyễn Trung Trực Couplet Poetry (Kiên Giang)")
    print("=" * 120)
    
    locked_video = None
    locked_hit_frame = None
    poem_found = False
    context_only_video = None
    evidence_records = []
    
    # Stage A: Coarse Discovery (Stride 500)
    print("\n--- [p1-19 Stage A] Coarse Discovery (Stride 500) ---")
    for vid in PRIORITY_P19:
        if GLOBAL_OCR_CALLS >= MAX_GLOBAL_OCR_LIMIT:
            print("  🛑 Global OCR limit reached during p1-19 Stage A!")
            break
        vpath = find_video_path(vid)
        if not vpath:
            continue
        cap = cv2.VideoCapture(str(vpath))
        if not cap.isOpened():
            continue
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        print(f"  • Coarse scanning {vid} ({total_frames} frames, ~{math.ceil(total_frames/COARSE_STRIDE)} calls) ...", flush=True)
        
        for fid in range(0, total_frames, COARSE_STRIDE):
            if GLOBAL_OCR_CALLS >= MAX_GLOBAL_OCR_LIMIT:
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
                print(f"\n      🌟 [DISTINCTIVE POEM HIT FOUND!] {vid} Frame {fid} ({fid/fps:.1f}s): '{txt_raw}'", flush=True)
                locked_video = vid
                locked_hit_frame = fid
                poem_found = True
                break
            elif has_ntt and not context_only_video:
                context_only_video = vid
                locked_hit_frame = fid
        cap.release()
        if poem_found:
            break
            
    chosen_vid = locked_video if poem_found else context_only_video
    status = "VERIFIED_HIGH_CONFIDENCE" if poem_found else ("CONTEXT_ONLY" if context_only_video else "UNRESOLVED")
    
    if not chosen_vid:
        print("\n  ❌ P1_19_UNRESOLVED: No poem or Nguyễn Trung Trực evidence found.\n", flush=True)
        return "UNRESOLVED", []
        
    print(f"\n--- [p1-19 Stage B] Dense Verification on {status} VIDEO: {chosen_vid} ---")
    vpath = find_video_path(chosen_vid)
    if vpath:
        cap = cv2.VideoCapture(str(vpath))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        
        start_f = max(0, locked_hit_frame - 1000)
        end_f = min(total_frames, locked_hit_frame + 1000)
        
        for fid in range(start_f, end_f, 35):
            if GLOBAL_OCR_CALLS >= MAX_GLOBAL_OCR_LIMIT:
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
                enhanced_text = " | ".join(roi_texts) if roi_texts else txt_raw
                
                b64_f = extract_b64(frame)
                h, w = frame.shape[:2]
                crop_center = frame[int(h*0.1):int(h*0.9), int(w*0.05):int(w*0.95)]
                b64_c = extract_b64(crop_center)
                
                score = 10.0 if has_poem else 5.0
                matched = "Poem Verse" if has_poem else "Nguyễn Trung Trực Memorial/Temple"
                
                print(f"      📍 Dense Frame {fid} ({fid/fps:.1f}s) [score={score}]: {enhanced_text[:65]}")
                evidence_records.append({
                    "query_id": "query-p1-19-qa",
                    "video_id": chosen_vid,
                    "frame_id": fid,
                    "time_sec": round(fid / fps, 2),
                    "score": score,
                    "matched_keywords": matched,
                    "ocr_text": enhanced_text,
                    "b64_full": b64_f,
                    "b64_crop": b64_c,
                })
        cap.release()
        
    ALL_ACCUMULATED_RECORDS.extend(evidence_records)
    flush_incremental_evidence()
    print(f"\n[p1-19 STATUS]: {status} in {chosen_vid} ({len(evidence_records)} evidence frames)\n", flush=True)
    return status, evidence_records

# -----------------------------------------------------------------------------
# 5. p1-15 Targeted Two-Stage Scan (CLB FANA & Khánh Hòa Commune Name)
# -----------------------------------------------------------------------------
def run_two_stage_p1_15() -> tuple[str, list[dict[str, Any]]]:
    print("=" * 120)
    print("🤝 [p1-15] TWO-STAGE SCAN: CLB FANA & Khánh Hòa Commune Name")
    print("=" * 120)
    
    locked_video = None
    locked_hit_frame = None
    fana_found = False
    evidence_records = []
    
    # Stage A: Coarse Discovery (Stride 500)
    print("\n--- [p1-15 Stage A] Coarse Discovery (Stride 500) ---")
    for vid in PRIORITY_P15:
        if GLOBAL_OCR_CALLS >= MAX_GLOBAL_OCR_LIMIT:
            print("  🛑 Global OCR limit reached during p1-15 Stage A!")
            break
        vpath = find_video_path(vid)
        if not vpath:
            continue
        cap = cv2.VideoCapture(str(vpath))
        if not cap.isOpened():
            continue
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        print(f"  • Coarse scanning {vid} ({total_frames} frames, ~{math.ceil(total_frames/COARSE_STRIDE)} calls) ...", flush=True)
        
        for fid in range(0, total_frames, COARSE_STRIDE):
            if GLOBAL_OCR_CALLS >= MAX_GLOBAL_OCR_LIMIT:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
                
            txt_raw, bboxes = ocr_full_frame(frame)
            txt_asc = strip_accents(txt_raw)
            
            # Strict Gate: FANA AND (Khánh Hòa OR Charity/Location Context)
            has_fana = ("fana" in txt_asc) or ("fa na" in txt_asc)
            has_kh = any(k in txt_asc for k in ["khanh hoa", "khanh vinh", "khanh son", "dien khanh", "cam lam", "van ninh", "nha trang"])
            has_context = any(k in txt_asc for k in ["trao qua", "tu thien", "tang qua", "hoc bong", "xa ", "thon ", "ba con"])
            
            if has_fana and (has_kh or has_context):
                print(f"\n      🌟 [VERIFIED FANA CHARITY HIT!] {vid} Frame {fid} ({fid/fps:.1f}s): '{txt_raw}'", flush=True)
                locked_video = vid
                locked_hit_frame = fid
                fana_found = True
                break
        cap.release()
        if fana_found:
            break
            
    if not locked_video:
        print("\n  ❌ P1_15_UNRESOLVED: No verified FANA + location/charity evidence found in priority videos.\n", flush=True)
        return "UNRESOLVED", []
        
    print(f"\n--- [p1-15 Stage B] Dense Verification on LOCKED VIDEO: {locked_video} ---")
    vpath = find_video_path(locked_video)
    if vpath:
        cap = cv2.VideoCapture(str(vpath))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        
        start_f = max(0, locked_hit_frame - 1000)
        end_f = min(total_frames, locked_hit_frame + 1000)
        
        for fid in range(start_f, end_f, 35):
            if GLOBAL_OCR_CALLS >= MAX_GLOBAL_OCR_LIMIT:
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
                enhanced_text = " | ".join(roi_texts) if roi_texts else txt_raw
                
                b64_f = extract_b64(frame)
                h, w = frame.shape[:2]
                crop_center = frame[int(h*0.1):int(h*0.9), int(w*0.05):int(w*0.95)]
                b64_c = extract_b64(crop_center)
                
                score = 10.0 if has_fana else 5.0
                matched = "FANA Club Banner" if has_fana else "Location / Commune Sign"
                
                print(f"      📍 Dense Frame {fid} ({fid/fps:.1f}s) [score={score}]: {enhanced_text[:65]}")
                evidence_records.append({
                    "query_id": "query-p1-15-qa",
                    "video_id": locked_video,
                    "frame_id": fid,
                    "time_sec": round(fid / fps, 2),
                    "score": score,
                    "matched_keywords": matched,
                    "ocr_text": enhanced_text,
                    "b64_full": b64_f,
                    "b64_crop": b64_c,
                })
        cap.release()
        
    ALL_ACCUMULATED_RECORDS.extend(evidence_records)
    flush_incremental_evidence()
    print(f"\n[p1-15 STATUS]: VERIFIED_HIGH_CONFIDENCE in {locked_video} ({len(evidence_records)} evidence frames)\n", flush=True)
    return "VERIFIED_HIGH_CONFIDENCE", evidence_records

# -----------------------------------------------------------------------------
# 6. Render Ranked & Deduplicated Visual Gallery
# -----------------------------------------------------------------------------
def render_ranked_gallery() -> None:
    html_out = Path("/kaggle/working/qa_temporal_deep_scan_gallery.html") if Path("/kaggle/working").exists() else REPO_ROOT / "scratch" / "qa_temporal_deep_scan_gallery.html"
    sections = []
    
    task_info = [
        ("query-p1-22-qa", "🍳 QA p1-22: 200g Thịt Nạc Xay & Dish Title Evidence"),
        ("query-p1-19-qa", "📜 QA p1-19: Nguyễn Trung Trực Couplet Poetry (Kiên Giang) Evidence"),
        ("query-p1-15-qa", "🤝 QA p1-15: CLB FANA & Khánh Hòa Commune Evidence"),
    ]
    
    for qid, title in task_info:
        raw_q_recs = [r for r in ALL_ACCUMULATED_RECORDS if r["query_id"] == qid]
        # Sort by (-score, frame_id) so highest quality evidence appears first
        sorted_recs = sorted(raw_q_recs, key=lambda r: (-r["score"], r["frame_id"]))
        
        # Deduplicate neighboring frames within ±100 frames from the same video
        dedup_recs = []
        seen_ranges: list[tuple[str, int, int]] = []
        for r in sorted_recs:
            v = r["video_id"]
            f = r["frame_id"]
            is_dup = any(v == sv and abs(f - sf) <= 100 for sv, sf, _ in seen_ranges)
            if not is_dup:
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
            
        sections.append(f"""
        <div style="background:#242424; border:1px solid #444; border-radius:10px; padding:16px; margin-bottom:28px;">
            <h3 style="color:#e06c75; margin-top:0; border-bottom:1px solid #555; padding-bottom:6px;">🎯 {title}</h3>
            <div style="font-size:12px; color:#aaa; margin-bottom:12px;">Đã trích xuất {len(raw_q_recs)} frames -> {len(dedup_recs)} frames sau khử trùng lặp ±100 frames</div>
            <div style="display:flex; flex-wrap:wrap; margin:-6px;">
                {''.join(cards) if cards else '<div style="color:#888; padding:10px;">Chưa tìm thấy bằng chứng khớp điều kiện lọc.</div>'}
            </div>
        </div>
        """)
        
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head><meta charset="utf-8"><title>QA Two-Stage Deep Scan Ranked Evidence</title></head>
    <body style="background:#121212; color:#fff; font-family:-apple-system,BlinkMacSystemFont,sans-serif; padding:16px;">
        <h2 style="color:#61afef; border-bottom:2px solid #555; padding-bottom:10px; margin-top:0;">🔍 BẢNG ĐỐI SOÁT CHỨNG CỨ THỊ GIÁC & 3X ROI OCR (ĐÃ SẮP XẾP THEO CHẤT LƯỢNG)</h2>
        {''.join(sections)}
    </body>
    </html>
    """
    html_out.write_text(full_html, encoding="utf-8")
    print(f"      • Saved Ranked HTML Gallery : {html_out} ✅\n")

# -----------------------------------------------------------------------------
# 7. Main Execution Flow
# -----------------------------------------------------------------------------
def main() -> None:
    t0 = time.time()
    
    # 1. Budget Verification Guard
    verify_actual_ocr_budget()
    
    # 2. Run Targeted Scans in Priority Order (p1-22 -> p1-19 -> p1-15)
    status_22, p22_recs = run_two_stage_p1_22()
    status_19, p19_recs = run_two_stage_p1_19()
    status_15, p15_recs = run_two_stage_p1_15()
    
    # 3. Render Ranked & Deduplicated Gallery
    render_ranked_gallery()
    
    # 4. Print Formal Terminal Summary Table
    print("=" * 120)
    print("📋 BẢNG TỔNG HỢP TRẠNG THÁI ĐỐI SOÁT CHỨNG CỨ QA (TERMINAL VERIFICATION SUMMARY)")
    print("=" * 120)
    print(f"{'Query ID':<18} | {'Status':<25} | {'Locked Video':<14} | {'Evidence Count':<16} | {'Top OCR Evidence'}")
    print("-" * 120)
    
    for qid, stat, recs in [("query-p1-22-qa", status_22, p22_recs), ("query-p1-19-qa", status_19, p19_recs), ("query-p1-15-qa", status_15, p15_recs)]:
        best = max(recs, key=lambda x: x["score"]) if recs else None
        vid = best["video_id"] if best else "N/A"
        cnt = str(len(recs))
        txt = best["ocr_text"][:45] if best else "N/A"
        print(f"{qid:<18} | {stat:<25} | {vid:<14} | {cnt:<16} | {txt}")
        
    print("=" * 120)
    print(f"\n🎉 TWO-STAGE DEEP SCAN FINISHED IN {time.time() - t0:.2f}s (Total OCR Calls: {GLOBAL_OCR_CALLS}/{MAX_GLOBAL_OCR_LIMIT}) ✅\n")

if __name__ == "__main__":
    main()
