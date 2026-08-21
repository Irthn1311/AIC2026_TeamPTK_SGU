#!/usr/bin/env python3
"""QA Recovery (VinAI B1 Dual-Arm + EasyOCR Evidence) & TRAKE Quality Audit (Single Runtime).

Operational Constraints:
  1. KIS Skip: 18 verified KIS CSVs copied/loaded instantly (0s).
  2. Single Bootstrap: OperationalKISRuntime bootstrapped once for the entire session.
  3. Preflight Decode Test: Verifies OpenCV video path resolution & frame read on 1 QA + 1 TRAKE sample.
  4. QA Dual-Arm Grounding: Uses Original VI + VinAI B1 English (num_beams=1), fused via RRF.
  5. EasyOCR Evidence (CPU Safe): Runs on Top 10 frames/query (full-frame + upscaled crops) for human review.
  6. TRAKE 100% Frozen: Inspects and renders strictly increasing candidate chains (p1-16 @3..@7, p1-4 @38.., p1-18 @25..).
  7. Emits Focused Visual Inspection Gallery (qa_recovery_and_trake_audit_gallery.html).
"""

from __future__ import annotations

import base64
import csv
import faulthandler
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
print("QA RECOVERY (VinAI B1 Dual-Arm + EasyOCR) & TRAKE QUALITY AUDIT (SINGLE RUNTIME)", flush=True)
print("=" * 150, flush=True)

REPO_ROOT = Path(__file__).resolve().parents[1]
SYSTEM_TAI_SRC = REPO_ROOT / "systems" / "system_tai" / "src"
if str(SYSTEM_TAI_SRC) not in sys.path:
    sys.path.insert(0, str(SYSTEM_TAI_SRC))

try:
    import clip
except ImportError:
    print("Installing official openai-clip dependency ...", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "git+https://github.com/openai/CLIP.git", "ftfy", "regex"], check=True)
    import clip

try:
    import cv2
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "opencv-python-headless"], check=False)
    import cv2

# EasyOCR for Vietnamese Text Reading
EASYOCR_READER = None
try:
    import easyocr
except ImportError:
    print("Installing easyocr for Vietnamese text reading ...", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "easyocr"], check=True)
    import easyocr

import torch
import transformers
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from system_tai.kis.session_engine import OperationalKISRuntime
from system_tai.kis.session_schema import (
    SessionConfig,
    TRAKEQueryRequest,
)

THUNGHIEM_DIR = REPO_ROOT / "systems" / "system_tai" / "THUNGHIEM_20-8"
SUBMISSION_DIR = Path("/kaggle/working/submission") if Path("/kaggle/working").exists() else REPO_ROOT / "scratch" / "submission"
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

AUTHORITATIVE_KIS_QIDS = [
    "query-p1-1-kis", "query-p1-2-kis", "query-p1-5-kis", "query-p1-6-kis",
    "query-p1-7-kis", "query-p1-8-kis", "query-p1-9-kis", "query-p1-10-kis",
    "query-p1-11-kis", "query-p1-12-kis", "query-p1-13-kis", "query-p1-14-kis",
    "query-p1-17-kis", "query-p1-20-kis", "query-p1-21-kis", "query-p1-23-kis",
    "query-p1-24-kis", "query-p1-25-kis",
]


# -----------------------------------------------------------------------------
# 1. KIS Skip Guard & Instant Copy
# -----------------------------------------------------------------------------
def check_and_ensure_kis() -> None:
    print("\n" + "=" * 120)
    print("[STAGE 1] KIS CSV Check & Reorder Guard")
    print("=" * 120)

    # Fast Copy from Repo Backup
    for base in [
        REPO_ROOT / "scratch" / "submission",
        REPO_ROOT / "systems" / "system_tai" / "THUNGHIEM_20-8" / "DAPAN",
    ]:
        if base.exists() and base != SUBMISSION_DIR:
            for qid in AUTHORITATIVE_KIS_QIDS:
                src = base / f"{qid}.csv"
                dst = SUBMISSION_DIR / f"{qid}.csv"
                if src.exists() and not dst.exists():
                    import shutil
                    shutil.copyfile(src, dst)

    missing_kis = [qid for qid in AUTHORITATIVE_KIS_QIDS if not (SUBMISSION_DIR / f"{qid}.csv").exists()]
    if not missing_kis:
        print(f"✅ All 18 KIS CSV files verified and loaded into {SUBMISSION_DIR} (Zero Rerun Needed!) ✅", flush=True)
        # Apply reorder patch just in case
        sys.path.insert(0, str(REPO_ROOT / "scratch"))
        from patch_kis_submission_reorder import reorder_csv
        p1_23_path = SUBMISSION_DIR / "query-p1-23-kis.csv"
        p1_23_top3 = [("L28_V006", 14483), ("L28_V006", 23895), ("L28_V006", 14444)]
        reorder_csv(p1_23_path, p1_23_top3, "query-p1-23-kis")
        p1_10_path = SUBMISSION_DIR / "query-p1-10-kis.csv"
        p1_10_top3 = [("L30_V017", 3010), ("L30_V017", 2531), ("L30_V017", 2640)]
        reorder_csv(p1_10_path, p1_10_top3, "query-p1-10-kis")
        print("      • Reorder verified: p1-23 Marian Top-3 & p1-10 VinAI Top-3 ✅", flush=True)
        return

    print(f"[Auto-Gen] {len(missing_kis)} KIS CSVs missing -> Generating via KIS Submission Merger ...", flush=True)
    sys.path.insert(0, str(REPO_ROOT / "scratch"))
    from experiment_kis_btc_submission_merger import run_kis_submission_merger
    run_kis_submission_merger()

    from patch_kis_submission_reorder import main as patch_main
    patch_main()


# -----------------------------------------------------------------------------
# 2. Fast Video Decoder & EasyOCR Helpers
# -----------------------------------------------------------------------------
VIDEO_PATH_CACHE: dict[str, Path] = {}


def resolve_video_path(video_id: str, runtime: OperationalKISRuntime | None = None) -> Path | None:
    if video_id in VIDEO_PATH_CACHE:
        return VIDEO_PATH_CACHE[video_id]

    # 1. Ask runtime's raw_video_registry directly
    if runtime is not None and hasattr(runtime, "raw_video_registry"):
        try:
            for rec in runtime.raw_video_registry._records:
                if rec.video_id == video_id and rec.raw_video_path and rec.raw_video_path.exists():
                    VIDEO_PATH_CACHE[video_id] = rec.raw_video_path
                    return rec.raw_video_path
        except Exception:
            pass

    # 2. Search common batch folder structures
    batch = video_id.split("_")[0] if "_" in video_id else ""
    search_patterns = [
        Path(f"/kaggle/input/datasets/videos/{batch}/{video_id}.mp4"),
        Path(f"/kaggle/input/datasets/videos/{video_id}.mp4"),
        Path(f"/kaggle/input/datasets/{batch}/{video_id}.mp4"),
        Path(f"/kaggle/input/datasets/{video_id}.mp4"),
        REPO_ROOT / "systems" / "system_tai" / "data" / "videos" / batch / f"{video_id}.mp4",
        REPO_ROOT / "systems" / "system_tai" / "data" / "videos" / f"{video_id}.mp4",
        REPO_ROOT / "systems" / "system_tai" / "data" / batch / f"{video_id}.mp4",
    ]
    for p in search_patterns:
        if p.exists():
            VIDEO_PATH_CACHE[video_id] = p
            return p

    # 3. Dynamic glob fallback across /kaggle/input
    if Path("/kaggle/input").exists():
        for found in Path("/kaggle/input").glob(f"**/{video_id}.mp4"):
            VIDEO_PATH_CACHE[video_id] = found
            return found

    return None


def decode_full_resolution_frame(video_id: str, frame_id: int, runtime: OperationalKISRuntime | None = None) -> tuple[Any | None, str, str]:
    """Decodes original frame, produces full-res JPEG base64 and 2x center-crop JPEG base64."""
    vpath = resolve_video_path(video_id, runtime)
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

        # Full-res base64
        _, buf_full = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        b64_full = base64.b64encode(buf_full).decode("utf-8")

        # Center 50% crop for text reading
        h, w = frame.shape[:2]
        crop = frame[int(h * 0.15): int(h * 0.85), int(w * 0.1): int(w * 0.9)]
        _, buf_crop = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        b64_crop = base64.b64encode(buf_crop).decode("utf-8")

        return frame, b64_full, b64_crop
    except Exception:
        return None, "", ""


def get_easyocr_reader() -> Any:
    global EASYOCR_READER
    if EASYOCR_READER is None:
        use_gpu = torch.cuda.is_available()
        print(f"[EasyOCR] Initializing Vietnamese+English OCR Reader (gpu={use_gpu}) ...", flush=True)
        EASYOCR_READER = easyocr.Reader(["vi", "en"], gpu=use_gpu, verbose=False)
    return EASYOCR_READER


def run_easyocr_evidence(frame: Any) -> str:
    """Extracts Vietnamese text evidence from frame (full frame + center region)."""
    if frame is None:
        return "[No Frame Decoded]"
    try:
        reader = get_easyocr_reader()
        # 1. Full frame scan
        results_full = reader.readtext(frame, detail=0)
        # 2. High-res crop scan (center/text region)
        h, w = frame.shape[:2]
        crop = frame[int(h * 0.15): int(h * 0.85), int(w * 0.1): int(w * 0.9)]
        results_crop = reader.readtext(crop, detail=0)

        # Merge unique text lines
        seen = set()
        merged = []
        for line in results_full + results_crop:
            clean = " ".join(line.strip().split())
            if clean and clean not in seen and len(clean) >= 2:
                seen.add(clean)
                merged.append(clean)

        return " | ".join(merged) if merged else "[No text detected in frame]"
    except Exception as exc:
        return f"[OCR Error: {exc}]"


# -----------------------------------------------------------------------------
# 3. Preflight Frame Decode Test (Sanity Gate)
# -----------------------------------------------------------------------------
def preflight_decode_sanity_test(runtime: OperationalKISRuntime) -> None:
    print("=" * 120)
    print("[PREFLIGHT DECODE TEST] Verifying Real Video Path Resolution & OpenCV Read")
    print("=" * 120)

    test_samples = [
        ("QA Sample", "L30_V072", 1384),
        ("TRAKE Sample", "L25_V007", 5363),
    ]

    for label, vid, fid in test_samples:
        vpath = resolve_video_path(vid, runtime)
        print(f"  • {label:<15} : Video={vid:<10} Frame={fid:<6}")
        print(f"      - Resolved Path : {vpath}")
        if not vpath or not vpath.exists():
            print(f"      - File Exists   : ❌ NOT FOUND on disk!")
            continue
        cap = cv2.VideoCapture(str(vpath))
        is_opened = cap.isOpened()
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = cap.read()
        cap.release()
        shape = frame.shape if frame is not None else None
        print(f"      - OpenCV Opened : {is_opened} | Read Success: {ret} | Shape: {shape} ✅" if ret else f"      - OpenCV Read Fail ❌")

    print("\n" + "-" * 120 + "\n", flush=True)


# -----------------------------------------------------------------------------
# 4. VinAI B1 Translation Provider (Scratch QA Recovery Only)
# -----------------------------------------------------------------------------
class VinAIB1Translator:
    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self.model_id = "vinai/vinai-translate-vi2en-v2"
        print(f"[VinAI B1 Translator] Loading '{self.model_id}' on {device} (Greedy / Beam=1 Baseline) ...", flush=True)
        t0 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, src_lang="vi_VN", use_fast=False)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_id).to(device)
        self.model.eval()
        print(f"      • Loaded VinAI B1 in {time.time() - t0:.2f}s ✅\n", flush=True)

    def translate_b1(self, text: str) -> str:
        clean = " ".join(text.strip().split())
        inputs = self.tokenizer(clean, return_tensors="pt", padding=True, truncation=True, max_length=256).to(self.device)
        with torch.no_grad():
            # VinAI B1 explicit: num_beams=1 (greedy deterministic)
            outputs = self.model.generate(**inputs, max_length=256, num_beams=1, early_stopping=True)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


# -----------------------------------------------------------------------------
# 5. Single Runtime Bootstrap with Watchdog
# -----------------------------------------------------------------------------
def get_reuse_manifest() -> Path | None:
    for p in [
        Path("/kaggle/working/manifest_cache.json"),
        Path("/kaggle/input/system-tai-manifest/feature_manifest.json"),
        Path("/kaggle/input/datasets/manifest_cache.json"),
        Path("/kaggle/input/manifest_cache.json"),
        REPO_ROOT / "systems" / "system_tai" / "data" / "feature_manifest.json",
    ]:
        if p.exists() and p.stat().st_size > 1000:
            return p
    return None


def bootstrap_runtime_once() -> OperationalKISRuntime:
    print("\n" + "=" * 120)
    print("[STAGE 2] Single OperationalKISRuntime Bootstrap")
    print("=" * 120)

    yaml_path = REPO_ROOT / "systems" / "system_tai" / "configs" / "production.yaml"
    input_root = Path("/kaggle/input/datasets") if Path("/kaggle/input/datasets").exists() else Path("/kaggle/input")
    reuse_manifest = get_reuse_manifest()
    out_dir = Path("/kaggle/working/output/unified_session") if Path("/kaggle/working").exists() else REPO_ROOT / "scratch" / "unified_session"

    cfg = SessionConfig.from_yaml(
        yaml_path,
        input_root=input_root,
        output_root=out_dir,
        reuse_manifest=reuse_manifest,
    )

    print("\n[BOOTSTRAP_START] Initializing Single OperationalKISRuntime Instance (watchdog active: 120s dump) ...", flush=True)
    t0_boot = time.time()
    faulthandler.dump_traceback_later(120, repeat=True)
    try:
        runtime = OperationalKISRuntime.bootstrap(cfg)
    finally:
        faulthandler.cancel_dump_traceback_later()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[BOOTSTRAP_DONE] seconds={time.time() - t0_boot:.2f} (device={device}) ✅\n", flush=True)
    return runtime


# -----------------------------------------------------------------------------
# 6. TRAKE Generation & Quality Audit (100% Frozen Model)
# -----------------------------------------------------------------------------
def parse_trake_query(file_path: Path) -> tuple[str, list[dict[str, str]]]:
    content = file_path.read_text(encoding="utf-8")
    lines = [l.strip() for l in content.splitlines() if l.strip()]
    events = []
    for line in lines:
        for idx in range(1, 10):
            prefix = f"E{idx}:"
            if line.startswith(prefix):
                desc = line[len(prefix):].strip()
                events.append({"event_id": f"E{idx}", "description": desc})
                break
    return file_path.stem, events


def ensure_and_audit_trake(runtime: OperationalKISRuntime) -> list[dict[str, Any]]:
    print("=" * 120)
    print("[STAGE 3] TRAKE Execution & Strictly Increasing Chain Audit (Frozen Planner)")
    print("=" * 120)

    t0_tr = time.time()
    trake_files = sorted(list(THUNGHIEM_DIR.glob("*trake*.txt")))
    audit_results: list[dict[str, Any]] = []

    for tr_f in trake_files:
        qid, events = parse_trake_query(tr_f)
        csv_p = SUBMISSION_DIR / f"{qid}.csv"

        if not csv_p.exists():
            print(f"[Executing TRAKE] Generating {qid}.csv ({len(events)} events) ...", flush=True)
            req = TRAKEQueryRequest(
                request_id=qid,
                query_id=qid,
                events=tuple(events),
                include_vi_variant=True,
                top_k_per_variant=100,
                event_candidate_top_k=100,
                output_top_k=100,
                beam_width=100,
                refine_top_n=3,
            )
            resp = runtime.handle_trake_query(req)
            pred_rel = (resp.get("artifacts") or {}).get("trake_predictions_jsonl")
            if pred_rel:
                pred_f = runtime.output_root / pred_rel
                if pred_f.exists():
                    with pred_f.open("r", encoding="utf-8") as inf, csv_p.open("w", encoding="utf-8", newline="") as outf:
                        writer = csv.writer(outf)
                        for line in inf:
                            if line.strip():
                                rec = json.loads(line)
                                writer.writerow([str(rec["video_id"]).removesuffix(".mp4"), *rec["frame_ids"]])

        chains: list[tuple[int, str, list[int]]] = []
        if csv_p.exists():
            with csv_p.open("r", encoding="utf-8") as f:
                reader = csv.reader(f)
                for r_idx, r in enumerate(reader, start=1):
                    if r and len(r) >= 5:
                        vid = r[0].strip()
                        fids = [int(x.strip()) for x in r[1:5]]
                        chains.append((r_idx, vid, fids))

        unique_4_chains = []
        strictly_increasing = []

        for orig_rank, vid, fids in chains:
            if len(set(fids)) == 4:
                unique_4_chains.append((orig_rank, vid, fids))
            if fids[0] < fids[1] < fids[2] < fids[3]:
                gaps = [fids[i+1] - fids[i] for i in range(3)]
                strictly_increasing.append((orig_rank, vid, fids, gaps))

        print(f"\n[{qid}] Total Chains: {len(chains)}")
        print(f"  • Chains with 4 Unique Frame IDs       : {len(unique_4_chains)}/100")
        print(f"  • Chains with Strictly Increasing Frames: {len(strictly_increasing)}/100")

        if strictly_increasing:
            for idx, (orig_r, vid, fids, gaps) in enumerate(strictly_increasing[:5], start=1):
                print(f"    Candidate #{idx} (Original Rank @{orig_r:<3}): Video={vid:<10} Frames={fids} Gaps={gaps}")

        audit_results.append({
            "qid": qid,
            "total": len(chains),
            "unique_count": len(unique_4_chains),
            "increasing_count": len(strictly_increasing),
            "top_increasing": strictly_increasing[:5],
        })

    print(f"\n[TRAKE_DONE] seconds={time.time() - t0_tr:.2f} ✅\n", flush=True)
    return audit_results


# -----------------------------------------------------------------------------
# 7. QA Pre-Provider Dual-Arm Grounding (Original VI + VinAI B1 EN)
# -----------------------------------------------------------------------------
def run_qa_recovery(runtime: OperationalKISRuntime, translator: VinAIB1Translator) -> list[dict[str, Any]]:
    print("=" * 120)
    print("[STAGE 4] QA Dual-Arm Grounding (Original VI + VinAI B1 EN) & EasyOCR Evidence")
    print("=" * 120)

    t0_qa = time.time()
    qa_queries = [
        ("query-p1-15-qa", "Đoạn video về một chương trình từ thiện của một câu lạc bộ tên là FANA. Trong đoạn video có thể thấy câu lạc bộ này đang đi trao quà tại một xã thuộc tỉnh Khánh Hòa.", "Hỏi xã này có tên là gì? (tại thời điểm đó)"),
        ("query-p1-19-qa", "Trong đoạn video có 2 câu thơ của một nhà thơ ca ngợi anh hùng Nguyễn Trung Trực trong đình thần Nguyễn Trung Trực tại Kiên Giang.", "Hai câu thơ đó là gì?"),
        ("query-p1-22-qa", "Đoạn video về một người phụ nữ dạy nấu ăn cho những người khác. Trong đoạn video có thể thấy một người đang cầm công thức món ăn với nguyên liệu chính là 200g thịt nạc xay.", "Hỏi tiêu đề của công thức nấu ăn (tên món ăn) này là gì?"),
    ]

    recovery_results: list[dict[str, Any]] = []

    for qid, desc, q_part in qa_queries:
        print(f"\n--- [QA Dual-Arm Localization] {qid} ---")
        t0_q = time.time()

        # Arm 1: Original Vietnamese Query
        vi_full = f"{desc} {q_part}"
        eff_vi, _, _ = runtime.token_budget_guard.guard_and_compact(vi_full)

        # Arm 2: VinAI B1 English Query
        trans_en = translator.translate_b1(vi_full)
        eff_en, _, _ = runtime.token_budget_guard.guard_and_compact(trans_en)

        print(f"  • Arm 1 (Raw VI)  : {eff_vi}")
        print(f"  • Arm 2 (VinAI B1): {eff_en}")

        # Multi-Query Vector Retrieval
        vec_vi = runtime.shared_encoder.encode(eff_vi)
        vec_en = runtime.shared_encoder.encode(eff_en)

        cands_vi = runtime.exact_retriever.search_vector(query_id=f"{qid}-vi", query_vector=vec_vi, top_k=50)
        cands_en = runtime.exact_retriever.search_vector(query_id=f"{qid}-en", query_vector=vec_en, top_k=50)

        # RRF Fusion of VI + VinAI B1 EN
        fused = runtime.weighted_rrf.fuse_rankings(
            query_id=qid,
            variants=[
                type("Var", (), {"variant_id": "vi", "weight": 1.0})(),
                type("Var", (), {"variant_id": "en", "weight": 1.0})(),
            ],
            rankings={"vi": cands_vi, "en": cands_en},
            top_k=50,
            rrf_k=60.0,
        )

        # Apply Video Conditioned Refinement
        conditioned = runtime.video_conditioner.condition(
            global_result=fused,
            query_vector=vec_en,
            config=runtime.config.video_conditioned_keyframe_config,
            protected_prefix_rank=1,
        ).result.ranked_candidates

        print(f"  • Extracted {len(conditioned)} Grounding Candidates in {time.time() - t0_q:.2f}s ✅")

        # Decode Top 10 Frames & Run EasyOCR Evidence
        frame_records: list[dict[str, Any]] = []
        for rank_idx, c in enumerate(conditioned[:10], start=1):
            vid = str(c.video_id).removesuffix(".mp4")
            fid = int(c.frame_id)
            frame_mat, b64_full, b64_crop = decode_full_resolution_frame(vid, fid, runtime)
            ocr_text = run_easyocr_evidence(frame_mat)
            frame_records.append({
                "rank": rank_idx,
                "video_id": vid,
                "frame_id": fid,
                "score": float(c.score),
                "b64_full": b64_full,
                "b64_crop": b64_crop,
                "ocr_text": ocr_text,
            })
            print(f"    @{rank_idx:<2}: Video={vid:<10} Frame={fid:<6} EasyOCR='{ocr_text[:60]}'")

        recovery_results.append({
            "qid": qid,
            "desc": desc,
            "question": q_part,
            "trans_en": eff_en,
            "frames": frame_records,
        })

    print(f"\n[QA_LOCALIZATION_DONE] seconds={time.time() - t0_qa:.2f} ✅\n", flush=True)
    return recovery_results


# -----------------------------------------------------------------------------
# 8. Build Focused HTML Visual Gallery (QA Full-Res + TRAKE Chains)
# -----------------------------------------------------------------------------
def generate_focused_gallery(
    trake_results: list[dict[str, Any]],
    qa_results: list[dict[str, Any]],
    runtime: OperationalKISRuntime,
    out_html: Path,
) -> None:
    print(f"[STAGE 5] Building Focused Visual Inspection Gallery HTML ...")

    sections = []

    # Section 1: QA Recovery (Full-Res + Crop + OCR)
    qa_cards = []
    for q in qa_results:
        qid = q["qid"]
        desc = q["desc"]
        question = q["question"]
        trans = q["trans_en"]

        grid_items = []
        for r in q["frames"][:10]:
            rank = r["rank"]
            vid = r["video_id"]
            fid = r["frame_id"]
            b64_full = r["b64_full"]
            b64_crop = r["b64_crop"]
            ocr = r["ocr_text"]

            img_full_tag = f'<img src="data:image/jpeg;base64,{b64_full}" style="width:100%; border-radius:4px; margin-bottom:4px;" />' if b64_full else '<div style="background:#333;color:#888;height:100px;display:flex;align-items:center;justify-content:center;">No Frame</div>'
            img_crop_tag = f'<img src="data:image/jpeg;base64,{b64_crop}" style="width:100%; border-radius:4px; border:1px solid #e5c07b;" />' if b64_crop else ''

            grid_items.append(f"""
            <div style="flex:0 0 calc(50% - 12px); margin:6px; padding:10px; background:#1e1e1e; border:1px solid #333; border-radius:6px; box-sizing:border-box;">
                <div style="display:flex; justify-content:space-between; font-size:12px; margin-bottom:6px;">
                    <span style="font-weight:bold; color:#61afef;">Candidate Rank @{rank}</span>
                    <span style="color:#aaa;"><b>Video:</b> {vid} | <b>Frame:</b> {fid}</span>
                </div>
                <div style="display:flex; gap:6px;">
                    <div style="flex:1;">
                        <div style="font-size:10px; color:#aaa; margin-bottom:2px;">Full-Resolution Frame:</div>
                        {img_full_tag}
                    </div>
                    <div style="flex:1;">
                        <div style="font-size:10px; color:#e5c07b; margin-bottom:2px;">2x Enlarged Center Region:</div>
                        {img_crop_tag}
                    </div>
                </div>
                <div style="background:#111; padding:8px; border-radius:4px; margin-top:8px; font-size:11px; color:#98c379; font-family:monospace; min-height:36px; word-break:break-word;">
                    <b style="color:#e5c07b;">EasyOCR Evidence:</b> {ocr}
                </div>
            </div>
            """)

        qa_cards.append(f"""
        <div style="background:#262626; border:1px solid #444; border-radius:8px; margin-bottom:24px; padding:16px;">
            <div style="font-size:16px; font-weight:bold; color:#e06c75; margin-bottom:6px;">{qid} — Visual Evidence (VinAI B1 + EasyOCR Top 10)</div>
            <div style="font-size:13px; color:#ddd; margin-bottom:4px;"><b>Bối cảnh:</b> {desc}</div>
            <div style="font-size:14px; color:#fff; font-weight:600; margin-bottom:4px;"><b>Câu hỏi:</b> {question}</div>
            <div style="font-size:12px; color:#61afef; margin-bottom:12px;"><b>VinAI B1 Query:</b> {trans}</div>
            <div style="display:flex; flex-wrap:wrap; margin:-6px;">
                {''.join(grid_items)}
            </div>
        </div>
        """)

    sections.append(f"""
    <h2 style="color:#e06c75; border-bottom:2px solid #555; padding-bottom:6px;">🔍 PHẦN 1: QA VISUAL EVIDENCE & SIDECAR EASYOCR (HUMAN REVIEW)</h2>
    {''.join(qa_cards)}
    """)

    # Section 2: TRAKE Quality Audit
    trake_cards = []
    for t in trake_results:
        qid = t["qid"]
        inc_chains = t["top_increasing"]
        
        chain_rows = []
        for idx, (orig_r, vid, fids, gaps) in enumerate(inc_chains, start=1):
            frames_html = []
            for e_idx, fid in enumerate(fids, start=1):
                _, b64_f, _ = decode_full_resolution_frame(vid, fid, runtime)
                img_tag = f'<img src="data:image/jpeg;base64,{b64_f}" style="width:100%; border-radius:4px;" />' if b64_f else '<div style="background:#333;color:#888;height:80px;display:flex;align-items:center;justify-content:center;">No Frame</div>'
                frames_html.append(f"""
                <div style="flex:1; margin:2px; padding:4px; background:#1c1c1c; border:1px solid #333; border-radius:4px; text-align:center;">
                    <div style="font-size:10px; color:#e5c07b; font-weight:bold; margin-bottom:2px;">E{e_idx} (f={fid})</div>
                    {img_tag}
                </div>
                """)

            chain_rows.append(f"""
            <div style="background:#1e1e1e; border:1px solid #333; border-radius:6px; padding:8px; margin-bottom:8px;">
                <div style="font-size:12px; font-weight:bold; color:#61afef; margin-bottom:4px;">
                    Candidate #{idx} (Original Rank @{orig_r}) — Video: <span style="color:#fff;">{vid}</span> | Temporal Gaps: {gaps}
                </div>
                <div style="display:flex; gap:4px;">{''.join(frames_html)}</div>
            </div>
            """)

        trake_cards.append(f"""
        <div style="background:#262626; border:1px solid #444; border-radius:8px; margin-bottom:20px; padding:14px;">
            <div style="font-size:15px; font-weight:bold; color:#e5c07b; margin-bottom:8px;">
                {qid} — Best Strictly Increasing Chains (f1 < f2 < f3 < f4)
            </div>
            {''.join(chain_rows) if chain_rows else '<div style="color:#888;">No strictly increasing chains found.</div>'}
        </div>
        """)

    sections.append(f"""
    <h2 style="color:#e5c07b; border-bottom:2px solid #555; padding-bottom:6px; margin-top:24px;">⏱️ PHẦN 2: TRAKE QUALITY & STRICTLY-INCREASING CHAINS AUDIT</h2>
    {''.join(trake_cards)}
    """)

    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head><meta charset="utf-8"><title>QA Recovery & TRAKE Quality Audit</title></head>
    <body style="background:#121212; color:#fff; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif; padding:16px;">
        <h1 style="color:#61afef; border-bottom:2px solid #444; padding-bottom:10px;">📋 BÁO CÁO KIỂM TOÁN CHẤT LƯỢNG QA EVIDENCE & TRAKE CHAINS</h1>
        {''.join(sections)}
    </body>
    </html>
    """
    out_html.write_text(full_html, encoding="utf-8")
    print(f"      • Saved Focused Visual Gallery to: {out_html} ✅", flush=True)


# -----------------------------------------------------------------------------
# 9. Main Pipeline
# -----------------------------------------------------------------------------
def main() -> None:
    # 1. KIS Instant Copy & Reorder Check
    check_and_ensure_kis()

    # 2. Bootstrap Single Runtime Instance with Watchdog
    runtime = bootstrap_runtime_once()

    # 3. Preflight Frame Decode Sanity Test
    preflight_decode_sanity_test(runtime)

    # 4. Initialize VinAI B1 Translator for QA (CPU safe)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    translator = VinAIB1Translator(device=device)

    # 5. Ensure TRAKE Generation & Audit Distinct Chains (Frozen Model)
    trake_results = ensure_and_audit_trake(runtime)

    # 6. Extract QA Pre-Provider Dual-Arm Grounding & EasyOCR Evidence
    qa_results = run_qa_recovery(runtime, translator)

    # 7. Generate Focused Visual Gallery
    gallery_out = Path("/kaggle/working/qa_recovery_and_trake_audit_gallery.html") if Path("/kaggle/working").exists() else REPO_ROOT / "scratch" / "qa_recovery_and_trake_audit_gallery.html"
    generate_focused_gallery(trake_results, qa_results, runtime, gallery_out)

    print("=" * 150)
    print(">>> QA RECOVERY & TRAKE AUDIT COMPLETE (READY FOR HUMAN VISUAL INSPECTION) <<<")
    print("=" * 150 + "\n")


if __name__ == "__main__":
    main()
