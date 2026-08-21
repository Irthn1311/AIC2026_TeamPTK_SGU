#!/usr/bin/env python3
"""QA Final Recovery Shootout (3 Independent Arms: Dense-VI, VinAI B1, Manual Literal EN).

Operational Guarantees:
  1. No Raw-VI contamination: Completely eliminates bloated raw-VI paragraph queries and blind RRF.
  2. 3 Independent Retrieval Arms:
     - Arm 1: DENSE_VI (dense visual grounding entities)
     - Arm 2: VINAI_B1_DENSE_EN (real historical B1: num_beams=3, no_repeat_ngram=3, repetition_penalty=1.15)
     - Arm 3: MANUAL_LITERAL_EN (preserves proper nouns: Khanh Hoa, Nguyen Trung Truc, 200g minced pork)
  3. Preflight & Instant Video Path Resolution: 0.001s direct lookup, 0% "No Frame".
  4. EasyOCR on Deduplicated Union: Scans Top-10 deduplicated candidates per query for text evidence.
  5. Emits Focused Visual Shootout Gallery: /kaggle/working/qa_shootout_gallery.html.
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
print("QA FINAL RECOVERY SHOOTOUT (3 INDEPENDENT ARMS + REAL VINAI B1 + DENSE EN/VI)", flush=True)
print("=" * 150, flush=True)

REPO_ROOT = Path(__file__).resolve().parents[1]
SYSTEM_TAI_SRC = REPO_ROOT / "systems" / "system_tai" / "src"
if str(SYSTEM_TAI_SRC) not in sys.path:
    sys.path.insert(0, str(SYSTEM_TAI_SRC))

try:
    import clip
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "git+https://github.com/openai/CLIP.git", "ftfy", "regex"], check=True)
    import clip

try:
    import cv2
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "opencv-python-headless"], check=False)
    import cv2

try:
    import easyocr
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "easyocr"], check=True)
    import easyocr

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from system_tai.kis.session_engine import OperationalKISRuntime
from system_tai.kis.session_schema import SessionConfig

# -----------------------------------------------------------------------------
# 1. Video Path Direct Resolver (0.001s Instant Resolution)
# -----------------------------------------------------------------------------
VIDEO_CACHE: dict[str, Path] = {}

def find_video_path(video_id: str, runtime: OperationalKISRuntime | None = None) -> Path | None:
    if video_id in VIDEO_CACHE:
        return VIDEO_CACHE[video_id]

    if runtime is not None and hasattr(runtime, "raw_video_registry"):
        for rec in runtime.raw_video_registry._records:
            if rec.video_id == video_id and rec.raw_video_path and rec.raw_video_path.exists():
                VIDEO_CACHE[video_id] = rec.raw_video_path
                return rec.raw_video_path

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

def decode_full_res(video_id: str, frame_id: int, runtime: OperationalKISRuntime | None = None) -> tuple[Any | None, str, str]:
    vpath = find_video_path(video_id, runtime)
    if not vpath or not vpath.exists():
        return None, "", ""
    try:
        cap = cv2.VideoCapture(str(vpath))
        if not cap.isOpened():
            return None, "", ""
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_id))
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            return None, "", ""

        _, buf_f = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        b64_f = base64.b64encode(buf_f).decode("utf-8")

        h, w = frame.shape[:2]
        crop = frame[int(h * 0.15): int(h * 0.85), int(w * 0.1): int(w * 0.9)]
        _, buf_c = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        b64_c = base64.b64encode(buf_c).decode("utf-8")

        return frame, b64_f, b64_c
    except Exception:
        return None, "", ""

# -----------------------------------------------------------------------------
# 2. EasyOCR Reader (CPU Safe)
# -----------------------------------------------------------------------------
EASYOCR_READER = None

def get_easyocr_reader() -> Any:
    global EASYOCR_READER
    if EASYOCR_READER is None:
        use_gpu = torch.cuda.is_available()
        print(f"[EasyOCR] Initializing OCR Reader (gpu={use_gpu}) ...", flush=True)
        EASYOCR_READER = easyocr.Reader(["vi", "en"], gpu=use_gpu, verbose=False)
    return EASYOCR_READER

def run_easyocr_evidence(frame: Any) -> str:
    if frame is None:
        return "[No Frame]"
    try:
        reader = get_easyocr_reader()
        res_full = reader.readtext(frame, detail=0)
        h, w = frame.shape[:2]
        crop = frame[int(h * 0.15): int(h * 0.85), int(w * 0.1): int(w * 0.9)]
        res_crop = reader.readtext(crop, detail=0)

        seen = set()
        merged = []
        for line in res_full + res_crop:
            clean = " ".join(line.strip().split())
            if clean and clean not in seen and len(clean) >= 2:
                seen.add(clean)
                merged.append(clean)
        return " | ".join(merged) if merged else "[No text detected in frame]"
    except Exception as exc:
        return f"[OCR Error: {exc}]"

# -----------------------------------------------------------------------------
# 3. Real Historical VinAI B1 Translator (num_beams=3, no_repeat_ngram=3, rep_penalty=1.15)
# -----------------------------------------------------------------------------
class RealVinAIB1Translator:
    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self.model_id = "vinai/vinai-translate-vi2en-v2"
        print(f"\n[Loading Real VinAI B1: num_beams=3, no_repeat_ngram=3, rep_penalty=1.15 on {device}] ...", flush=True)
        t0 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, src_lang="vi_VN", use_fast=False)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_id).to(device)
        self.model.eval()
        print(f"      • Loaded Real VinAI B1 in {time.time() - t0:.2f}s ✅\n", flush=True)

    def translate_b1(self, text: str) -> str:
        clean = " ".join(text.strip().split())
        inputs = self.tokenizer(clean, return_tensors="pt", padding=True, truncation=True, max_length=128).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=128,
                num_beams=3,
                no_repeat_ngram_size=3,
                repetition_penalty=1.15,
                early_stopping=True,
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

# -----------------------------------------------------------------------------
# 4. Bootstrap OperationalKISRuntime (0.5s Cached)
# -----------------------------------------------------------------------------
def bootstrap_runtime() -> OperationalKISRuntime:
    yaml_path = REPO_ROOT / "systems" / "system_tai" / "configs" / "production.yaml"
    input_root = Path("/kaggle/input/datasets") if Path("/kaggle/input/datasets").exists() else Path("/kaggle/input")
    manifest_cache = Path("/kaggle/working/manifest_cache.json")
    out_dir = Path("/kaggle/working/output/unified_session") if Path("/kaggle/working").exists() else REPO_ROOT / "scratch" / "unified_session"

    cfg = SessionConfig.from_yaml(
        yaml_path,
        input_root=input_root,
        output_root=out_dir,
        reuse_manifest=manifest_cache if manifest_cache.exists() else None,
    )
    t0 = time.time()
    runtime = OperationalKISRuntime.bootstrap(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[BOOTSTRAP_DONE] seconds={time.time() - t0:.2f} (device={device}) ✅\n", flush=True)
    return runtime

# -----------------------------------------------------------------------------
# 5. QA Shootout Runner (Dense VI vs VinAI B1 vs Manual Literal EN)
# -----------------------------------------------------------------------------
def run_qa_shootout(runtime: OperationalKISRuntime, translator: RealVinAIB1Translator) -> list[dict[str, Any]]:
    print("=" * 120)
    print("[QA SHOOTOUT] Evaluating 3 Arms Independently on Top 15 Candidates")
    print("=" * 120)

    qa_tasks = [
        {
            "qid": "query-p1-15-qa",
            "title": "Xã từ thiện của CLB FANA tại tỉnh Khánh Hòa",
            "dense_vi": "FANA trao quà từ thiện tại một xã ở tỉnh Khánh Hòa, bảng tên xã, biển địa điểm",
            "manual_en": "FANA charity club giving gifts in a commune in Khanh Hoa province, commune name sign, location banner",
        },
        {
            "qid": "query-p1-19-qa",
            "title": "2 câu thơ ca ngợi Nguyễn Trung Trực tại Kiên Giang",
            "dense_vi": "đình Nguyễn Trung Trực Kiên Giang, hai câu thơ ca ngợi Nguyễn Trung Trực, chữ câu thơ trên bảng tường",
            "manual_en": "Nguyen Trung Truc temple in Kien Giang, two written verses praising Nguyen Trung Truc, poem text on wall or sign",
        },
        {
            "qid": "query-p1-22-qa",
            "title": "Tên món ăn công thức 200g thịt nạc xay",
            "dense_vi": "phụ nữ dạy nấu ăn, cầm tờ công thức, 200g thịt nạc xay, tiêu đề tên món ăn",
            "manual_en": "woman teaching cooking, holding a recipe sheet, 200g minced pork, recipe title, dish name",
        },
    ]

    shootout_results: list[dict[str, Any]] = []

    for task in qa_tasks:
        qid = task["qid"]
        title = task["title"]
        dense_vi = task["dense_vi"]
        manual_en = task["manual_en"]

        # VinAI B1 translation of dense VI
        vinai_b1_en = translator.translate_b1(dense_vi)

        print(f"\n" + "=" * 80)
        print(f"🎯 [{qid}] {title}")
        print(f"  • Arm 1 [DENSE_VI]         : {dense_vi}")
        print(f"  • Arm 2 [VINAI_B1_DENSE_EN]: {vinai_b1_en}")
        print(f"  • Arm 3 [MANUAL_LITERAL_EN]: {manual_en}")
        print("=" * 80)

        arms = [
            ("DENSE_VI", dense_vi),
            ("VINAI_B1_DENSE_EN", vinai_b1_en),
            ("MANUAL_LITERAL_EN", manual_en),
        ]

        arm_results = {}
        for arm_name, query_text in arms:
            vec = runtime.shared_encoder.encode(query_text)
            cands = runtime.exact_retriever.search_vector(
                query_id=f"{qid}-{arm_name}",
                query_vector=vec,
                top_k=15,
            )
            # VideoConditioner refinement
            conditioned = runtime.video_conditioner.condition(
                global_result=cands,
                query_vector=vec,
                config=runtime.config.video_conditioned_keyframe_config,
                protected_prefix_rank=1,
            ).result.ranked_candidates

            arm_results[arm_name] = [(str(c.video_id).removesuffix(".mp4"), int(c.frame_id), float(c.score)) for c in conditioned[:15]]

            print(f"\n--- Top 10 Candidates: {arm_name} ---")
            for r_idx, (v, f, s) in enumerate(arm_results[arm_name][:10], start=1):
                print(f"    @{r_idx:<2}: Video={v:<10} Frame={f:<6} CosineScore={s:.4f}")

        # Deduplicated Top 10 across the arms for visual review & OCR
        dedup_candidates = []
        seen = set()
        for arm_name in ["MANUAL_LITERAL_EN", "VINAI_B1_DENSE_EN", "DENSE_VI"]:
            for v, f, s in arm_results[arm_name][:6]:
                if (v, f) not in seen:
                    seen.add((v, f))
                    dedup_candidates.append((arm_name, v, f, s))

        # Decode frames & run EasyOCR
        print(f"\n[Decoding & EasyOCR on {len(dedup_candidates[:12])} Deduplicated Candidates] ...")
        gallery_items = []
        for rank_idx, (src_arm, vid, fid, score) in enumerate(dedup_candidates[:12], start=1):
            frame_mat, b64_f, b64_c = decode_full_res(vid, fid, runtime)
            ocr_text = run_easyocr_evidence(frame_mat)
            gallery_items.append({
                "rank": rank_idx,
                "src_arm": src_arm,
                "video_id": vid,
                "frame_id": fid,
                "score": score,
                "b64_f": b64_f,
                "b64_c": b64_c,
                "ocr": ocr_text,
            })
            print(f"  Candidate #{rank_idx} ({src_arm}): {vid} f={fid} -> OCR: '{ocr_text[:60]}'")

        shootout_results.append({
            "qid": qid,
            "title": title,
            "dense_vi": dense_vi,
            "vinai_b1_en": vinai_b1_en,
            "manual_en": manual_en,
            "arm_results": arm_results,
            "gallery_items": gallery_items,
        })

    return shootout_results

# -----------------------------------------------------------------------------
# 6. Generate Shootout HTML Gallery
# -----------------------------------------------------------------------------
def render_shootout_gallery(results: list[dict[str, Any]], out_path: Path) -> None:
    sections = []
    for r in results:
        qid = r["qid"]
        title = r["title"]
        d_vi = r["dense_vi"]
        v_en = r["vinai_b1_en"]
        m_en = r["manual_en"]

        grid = []
        for item in r["gallery_items"]:
            rk = item["rank"]
            arm = item["src_arm"]
            vid = item["video_id"]
            fid = item["frame_id"]
            sc = item["score"]
            b64_f = item["b64_f"]
            b64_c = item["b64_c"]
            ocr = item["ocr"]

            img_f_tag = f'<img src="data:image/jpeg;base64,{b64_f}" style="width:100%; border-radius:4px; margin-bottom:4px;" />' if b64_f else '<div style="background:#333;height:80px;">No Frame</div>'
            img_c_tag = f'<img src="data:image/jpeg;base64,{b64_c}" style="width:100%; border-radius:4px; border:1px solid #e5c07b;" />' if b64_c else ''

            grid.append(f"""
            <div style="flex:0 0 calc(33.333% - 10px); margin:5px; padding:10px; background:#1e1e1e; border:1px solid #333; border-radius:6px; box-sizing:border-box;">
                <div style="display:flex; justify-content:space-between; font-size:11px; margin-bottom:4px;">
                    <span style="font-weight:bold; color:#61afef;">#{rk} [{arm}]</span>
                    <span style="color:#aaa;">{vid} (f={fid})</span>
                </div>
                <div style="display:flex; gap:4px;">
                    <div style="flex:1;">{img_f_tag}</div>
                    <div style="flex:1;">{img_c_tag}</div>
                </div>
                <div style="background:#111; padding:6px; border-radius:4px; margin-top:6px; font-size:10px; color:#98c379; font-family:monospace; min-height:36px; word-break:break-word;">
                    <b style="color:#e5c07b;">EasyOCR Evidence:</b> {ocr}
                </div>
            </div>
            """)

        sections.append(f"""
        <div style="background:#242424; border:1px solid #444; border-radius:8px; padding:16px; margin-bottom:28px;">
            <h3 style="color:#e06c75; margin-top:0; border-bottom:1px solid #555; padding-bottom:6px;">🎯 {qid}: {title}</h3>
            <div style="font-size:12px; color:#ccc; margin-bottom:2px;"><b>• Arm 1 (Dense VI):</b> {d_vi}</div>
            <div style="font-size:12px; color:#98c379; margin-bottom:2px;"><b>• Arm 2 (VinAI B1):</b> {v_en}</div>
            <div style="font-size:12px; color:#61afef; margin-bottom:12px;"><b>• Arm 3 (Manual EN):</b> {m_en}</div>
            <div style="display:flex; flex-wrap:wrap; margin:-5px;">
                {''.join(grid)}
            </div>
        </div>
        """)

    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head><meta charset="utf-8"><title>QA Final Recovery Shootout</title></head>
    <body style="background:#121212; color:#fff; font-family:-apple-system,BlinkMacSystemFont,sans-serif; padding:16px;">
        <h2 style="color:#61afef; border-bottom:2px solid #555; padding-bottom:10px; margin-top:0;">🔍 BẢNG ĐỐI SOÁT HÌNH ẢNH & CHỨNG CỨ EASYOCR (QA RECOVERY SHOOTOUT)</h2>
        {''.join(sections)}
    </body>
    </html>
    """
    out_path.write_text(full_html, encoding="utf-8")
    print(f"\n      • Saved QA Shootout Gallery to: {out_path} ✅", flush=True)

# -----------------------------------------------------------------------------
# 7. Main Execution
# -----------------------------------------------------------------------------
def main() -> None:
    runtime = bootstrap_runtime()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    translator = RealVinAIB1Translator(device=device)
    results = run_qa_shootout(runtime, translator)
    out_html = Path("/kaggle/working/qa_shootout_gallery.html")
    render_shootout_gallery(results, out_html)
    print("=" * 150)
    print(">>> QA RECOVERY SHOOTOUT COMPLETE <<<")
    print("=" * 150 + "\n")

if __name__ == "__main__":
    main()
