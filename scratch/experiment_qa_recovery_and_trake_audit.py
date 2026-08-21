#!/usr/bin/env python3
"""QA Recovery & TRAKE Quality Audit (Zero Core Algorithm Changes).

Objectives:
  1. KIS Reorder Reaffirmation:
     - Applies final reorder patch (p1-10 Handpan Top-3, p1-23 Marian Top-3) to submission/ CSVs.
  2. TRAKE Quality & Distinct Chain Audit:
     - Inspects the existing 100 chains per query for p1-4, p1-16, p1-18.
     - Identifies all chains with 4 unique frame IDs and strictly increasing order (f1 < f2 < f3 < f4).
     - Reports their original ranks, temporal gaps, and generates high-res visual comparison.
  3. QA Visual Localization Recovery & Scratch OCR Sidecar:
     - Traces exact code path (UNSUPPORTED_NO_PROVIDER in QARuntimePipeline).
     - Extracts Top 30 visual localization candidate frames (CLIP + RRF) BEFORE provider gating.
     - Decodes full-resolution video frames (with high-res crops).
     - Executes experimental scratch OCR sidecar (pytesseract vie+eng if available).
     - Emits QA Recovery & TRAKE Audit Visual Gallery (qa_recovery_and_trake_audit_gallery.html).
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
print("QA RECOVERY SIDECAR & TRAKE QUALITY AUDIT (NO PRODUCTION ALGORITHM CHANGES)", flush=True)
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

# Optional sidecar OCR
TESSERACT_AVAILABLE = False
try:
    import pytesseract
    # Check if tesseract binary exists
    res = subprocess.run(["which", "tesseract"], capture_output=True, text=True)
    if res.returncode == 0:
        TESSERACT_AVAILABLE = True
except Exception:
    pass

import torch
from system_tai.kis.session_engine import OperationalKISRuntime
from system_tai.kis.session_schema import (
    QAQueryRequest,
    SessionConfig,
    TRAKEQueryRequest,
)

THUNGHIEM_DIR = REPO_ROOT / "systems" / "system_tai" / "THUNGHIEM_20-8"
SUBMISSION_DIR = Path("/kaggle/working/submission") if Path("/kaggle/working").exists() else REPO_ROOT / "scratch" / "submission"


# -----------------------------------------------------------------------------
# 1. KIS Reorder Reaffirmation
# -----------------------------------------------------------------------------
def apply_kis_reorder_patch() -> None:
    print("\n" + "=" * 120)
    print("[1/3] Re-applying Final KIS Routing Patch to submission/ CSVs")
    print("=" * 120)
    sys.path.insert(0, str(REPO_ROOT / "scratch"))
    from patch_kis_submission_reorder import main as patch_main
    patch_main()


# -----------------------------------------------------------------------------
# 2. Fast Video Decoder & OCR Sidecar Helpers
# -----------------------------------------------------------------------------
VIDEO_PATH_CACHE: dict[str, Path] = {}


def resolve_video_path(video_id: str) -> Path | None:
    if video_id in VIDEO_PATH_CACHE:
        return VIDEO_PATH_CACHE[video_id]
    for base in [
        Path("/kaggle/input/datasets/videos"),
        Path("/kaggle/input/datasets"),
        REPO_ROOT / "systems" / "system_tai" / "data" / "videos",
        REPO_ROOT / "systems" / "system_tai" / "data",
    ]:
        p = base / f"{video_id}.mp4"
        if p.exists():
            VIDEO_PATH_CACHE[video_id] = p
            return p
    if Path("/kaggle/input").exists():
        for sub in Path("/kaggle/input").iterdir():
            if sub.is_dir():
                for p in [sub / "videos" / f"{video_id}.mp4", sub / f"{video_id}.mp4"]:
                    if p.exists():
                        VIDEO_PATH_CACHE[video_id] = p
                        return p
    return None


def decode_full_resolution_frame(video_id: str, frame_id: int) -> tuple[Any | None, str, str]:
    """Decodes original frame, produces full-res JPEG base64 and 2x center-crop JPEG base64."""
    vpath = resolve_video_path(video_id)
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
        crop = frame[int(h * 0.2): int(h * 0.8), int(w * 0.15): int(w * 0.85)]
        _, buf_crop = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        b64_crop = base64.b64encode(buf_crop).decode("utf-8")

        return frame, b64_full, b64_crop
    except Exception:
        return None, "", ""


def run_scratch_ocr(frame: Any) -> str:
    """Experimental scratch OCR sidecar (does not touch production core)."""
    if frame is None or not TESSERACT_AVAILABLE:
        return "OCR unavailable (tesseract binary not active)"
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Try vie+eng then eng
        try:
            text = pytesseract.image_to_string(gray, lang="vie+eng", config="--psm 6").strip()
        except Exception:
            text = pytesseract.image_to_string(gray, config="--psm 6").strip()
        clean = " ".join(text.split())
        return clean if clean else "[No text detected]"
    except Exception as exc:
        return f"[OCR Error: {exc}]"


# -----------------------------------------------------------------------------
# 3. TRAKE Quality & Distinct Chains Audit
# -----------------------------------------------------------------------------
def audit_trake_quality() -> list[dict[str, Any]]:
    print("\n" + "=" * 120)
    print("[2/3] Auditing TRAKE Candidate Chains Quality (Distinct & Strictly Increasing)")
    print("=" * 120)

    trake_qids = ["query-p1-4-trake", "query-p1-16-trake", "query-p1-18-trake"]
    audit_results: list[dict[str, Any]] = []

    for qid in trake_qids:
        csv_p = SUBMISSION_DIR / f"{qid}.csv"
        if not csv_p.exists():
            print(f"❌ {qid}.csv missing in {SUBMISSION_DIR}")
            continue

        chains: list[tuple[int, str, list[int]]] = []
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
        print(f"  • Chains with 4 Unique Frame IDs     : {len(unique_4_chains)}/100")
        print(f"  • Chains with Strictly Increasing Ranks: {len(strictly_increasing)}/100")

        print(f"\n  Top 10 Strictly Increasing Chains for {qid}:")
        if strictly_increasing:
            for idx, (orig_r, vid, fids, gaps) in enumerate(strictly_increasing[:10], start=1):
                print(f"    #{idx:<2} (Original Rank @{orig_r:<3}): Video={vid:<10} Frames={fids} Gaps={gaps}")
        else:
            print("    [None found with strict f1 < f2 < f3 < f4 in current top 100]")

        audit_results.append({
            "qid": qid,
            "total": len(chains),
            "unique_count": len(unique_4_chains),
            "increasing_count": len(strictly_increasing),
            "top_increasing": strictly_increasing[:5],
            "top_unique": unique_4_chains[:5],
        })

    return audit_results


# -----------------------------------------------------------------------------
# 4. QA Visual Localization Recovery & Sidecar Extraction
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


def run_qa_recovery(runtime: OperationalKISRuntime) -> list[dict[str, Any]]:
    print("\n" + "=" * 120)
    print("[3/3] Executing QA Visual Localization Extraction & Scratch OCR Sidecar")
    print("=" * 120)

    qa_queries = [
        ("query-p1-15-qa", "Đoạn video về một chương trình từ thiện của một câu lạc bộ tên là FANA. Trong đoạn video có thể thấy câu lạc bộ này đang đi trao quà tại một xã thuộc tỉnh Khánh Hòa.", "Hỏi xã này có tên là gì? (tại thời điểm đó)"),
        ("query-p1-19-qa", "Trong đoạn video có 2 câu thơ của một nhà thơ ca ngợi anh hùng Nguyễn Trung Trực trong đình thần Nguyễn Trung Trực tại Kiên Giang.", "Hai câu thơ đó là gì?"),
        ("query-p1-22-qa", "Đoạn video về một người phụ nữ dạy nấu ăn cho những người khác. Trong đoạn video có thể thấy một người đang cầm công thức món ăn với nguyên liệu chính là 200g thịt nạc xay.", "Hỏi tiêu đề của công thức nấu ăn (tên món ăn) này là gì?"),
    ]

    recovery_results: list[dict[str, Any]] = []

    for qid, desc, q_part in qa_queries:
        print(f"\n--- Extracting Visual Localization Candidates for {qid} ---")
        t0 = time.time()

        # 1. Translate Query to English
        translator = runtime.translation_provider
        trans_en = translator.translate(f"{desc} {q_part}").strip()
        eff_en, _, _ = runtime.token_budget_guard.guard_and_compact(trans_en)
        print(f"  • VI Query  : {desc} {q_part}")
        print(f"  • EN Query  : {eff_en}")

        # 2. Multi-Query Retrieval (VI + EN vectors)
        vec_en = runtime.shared_encoder.encode(eff_en)
        coarse_candidates = runtime.exact_retriever.search_vector(
            query_id=f"rec-{qid}",
            query_vector=vec_en,
            top_k=50,
        )

        # 3. Apply Video Conditioned Refinement
        conditioned = runtime.video_conditioner.condition(
            global_result=coarse_candidates,
            query_vector=vec_en,
            config=runtime.config.video_conditioned_keyframe_config,
            protected_prefix_rank=1,
        ).result.ranked_candidates

        print(f"  • Extracted {len(conditioned)} Visual Grounding Candidates in {time.time() - t0:.2f}s ✅")

        # 4. Decode Top 25 Frames & Run Scratch Sidecar OCR
        frame_records: list[dict[str, Any]] = []
        for rank_idx, c in enumerate(conditioned[:25], start=1):
            vid = str(c.video_id).removesuffix(".mp4")
            fid = int(c.frame_id)
            frame_mat, b64_full, b64_crop = decode_full_resolution_frame(vid, fid)
            ocr_text = run_scratch_ocr(frame_mat)
            frame_records.append({
                "rank": rank_idx,
                "video_id": vid,
                "frame_id": fid,
                "score": float(c.score),
                "b64_full": b64_full,
                "b64_crop": b64_crop,
                "ocr_text": ocr_text,
            })
            if rank_idx <= 5:
                print(f"    @{rank_idx:<2}: Video={vid:<10} Frame={fid:<6} OCR='{ocr_text[:40]}'")

        recovery_results.append({
            "qid": qid,
            "desc": desc,
            "question": q_part,
            "trans_en": eff_en,
            "frames": frame_records,
        })

    return recovery_results


# -----------------------------------------------------------------------------
# 5. Build Unified QA Recovery & TRAKE Quality Visual HTML Gallery
# -----------------------------------------------------------------------------
def generate_qa_trake_audit_gallery(
    trake_results: list[dict[str, Any]],
    qa_results: list[dict[str, Any]],
    out_html: Path,
) -> None:
    print(f"\n[Visual Report] Building QA Recovery & TRAKE Quality Audit HTML Report...")

    sections = []

    # QA Recovery Section (High-Res & OCR)
    qa_cards = []
    for q in qa_results:
        qid = q["qid"]
        desc = q["desc"]
        question = q["question"]
        trans = q["trans_en"]

        grid_items = []
        for r in q["frames"][:20]:
            rank = r["rank"]
            vid = r["video_id"]
            fid = r["frame_id"]
            b64_full = r["b64_full"]
            b64_crop = r["b64_crop"]
            ocr = r["ocr_text"]

            img_full_tag = f'<img src="data:image/jpeg;base64,{b64_full}" style="width:100%; border-radius:4px; margin-bottom:4px;" />' if b64_full else '<div style="background:#333;color:#888;height:100px;display:flex;align-items:center;justify-content:center;">No Frame</div>'
            img_crop_tag = f'<img src="data:image/jpeg;base64,{b64_crop}" style="width:100%; border-radius:4px; border:1px solid #e5c07b;" />' if b64_crop else ''

            grid_items.append(f"""
            <div style="flex:0 0 calc(25% - 8px); margin:4px; padding:8px; background:#1e1e1e; border:1px solid #333; border-radius:6px; box-sizing:border-box;">
                <div style="display:flex; justify-content:space-between; font-size:11px; margin-bottom:4px;">
                    <span style="font-weight:bold; color:#61afef;">Rank @{rank}</span>
                    <span style="color:#aaa;">{vid} (f={fid})</span>
                </div>
                {img_full_tag}
                <div style="font-size:10px; color:#e5c07b; margin:2px 0;"><b>Center Crop:</b></div>
                {img_crop_tag}
                <div style="background:#111; padding:4px; border-radius:3px; margin-top:4px; font-size:10px; color:#98c379; font-family:monospace; min-height:28px; word-break:break-all;">
                    <b>OCR:</b> {ocr}
                </div>
            </div>
            """)

        qa_cards.append(f"""
        <div style="background:#262626; border:1px solid #444; border-radius:8px; margin-bottom:24px; padding:16px;">
            <div style="font-size:16px; font-weight:bold; color:#e06c75; margin-bottom:6px;">{qid} — Visual Localization Candidates (Top 20 Full-Res)</div>
            <div style="font-size:12px; color:#ddd; margin-bottom:4px;"><b>Bối cảnh:</b> {desc}</div>
            <div style="font-size:13px; color:#fff; font-weight:600; margin-bottom:4px;"><b>Câu hỏi:</b> {question}</div>
            <div style="font-size:11px; color:#888; margin-bottom:12px;"><b>CLIP Query:</b> {trans}</div>
            <div style="display:flex; flex-wrap:wrap; margin:-4px;">
                {''.join(grid_items)}
            </div>
        </div>
        """)

    sections.append(f"""
    <h2 style="color:#e06c75; border-bottom:2px solid #555; padding-bottom:6px;">🔍 QA VISUAL LOCALIZATION RECOVERY & SIDECAR OCR AUDIT</h2>
    {''.join(qa_cards)}
    """)

    # TRAKE Quality Section (Top Increasing Chains)
    trake_cards = []
    for t in trake_results:
        qid = t["qid"]
        inc_chains = t["top_increasing"]
        
        chain_rows = []
        for idx, (orig_r, vid, fids, gaps) in enumerate(inc_chains, start=1):
            frames_html = []
            for e_idx, fid in enumerate(fids, start=1):
                _, b64_f, _ = decode_full_resolution_frame(vid, fid)
                img_tag = f'<img src="data:image/jpeg;base64,{b64_f}" style="width:100%; border-radius:4px;" />' if b64_f else '<div style="background:#333;color:#888;height:80px;">No Frame</div>'
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
    <h2 style="color:#e5c07b; border-bottom:2px solid #555; padding-bottom:6px; margin-top:24px;">⏱️ TRAKE QUALITY & STRICTLY-INCREASING CHAINS AUDIT</h2>
    {''.join(trake_cards)}
    """)

    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head><meta charset="utf-8"><title>QA Recovery & TRAKE Quality Audit</title></head>
    <body style="background:#121212; color:#fff; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif; padding:16px;">
        <h1 style="color:#61afef; border-bottom:2px solid #444; padding-bottom:10px;">📋 BÁO CÁO KIỂM TOÁN CHẤT LƯỢNG QA RECOVERY & TRAKE CHAINS</h1>
        {''.join(sections)}
    </body>
    </html>
    """
    out_html.write_text(full_html, encoding="utf-8")
    print(f"      • Saved QA Recovery & TRAKE Quality Audit Gallery to: {out_html} ✅", flush=True)


# -----------------------------------------------------------------------------
# 6. Main Execution Pipeline
# -----------------------------------------------------------------------------
def main() -> None:
    # 1. Re-apply KIS Reorder Patch
    apply_kis_reorder_patch()

    # 2. Audit TRAKE Quality on existing CSVs
    trake_results = audit_trake_quality()

    # 3. Bootstrap Runtime for QA Localization Extraction
    print("\n" + "=" * 120)
    print("Bootstrapping OperationalKISRuntime for QA Localization Recovery...")
    print("=" * 120)
    yaml_path = REPO_ROOT / "systems" / "system_tai" / "configs" / "production.yaml"
    input_root = Path("/kaggle/input/datasets") if Path("/kaggle/input/datasets").exists() else Path("/kaggle/input")
    reuse_manifest = get_reuse_manifest()
    out_dir = Path("/kaggle/working/output/qa_recovery_session") if Path("/kaggle/working").exists() else REPO_ROOT / "scratch" / "qa_recovery_session"

    cfg = SessionConfig.from_yaml(
        yaml_path,
        input_root=input_root,
        output_root=out_dir,
        reuse_manifest=reuse_manifest,
    )
    runtime = OperationalKISRuntime.bootstrap(cfg)

    # 4. Extract QA Localization Candidates & Run Sidecar OCR
    qa_results = run_qa_recovery(runtime)

    # 5. Build Comprehensive Visual Inspection Gallery
    gallery_out = Path("/kaggle/working/qa_recovery_and_trake_audit_gallery.html") if Path("/kaggle/working").exists() else REPO_ROOT / "scratch" / "qa_recovery_and_trake_audit_gallery.html"
    generate_qa_trake_audit_gallery(trake_results, qa_results, gallery_out)

    print("\n" + "=" * 150)
    print(">>> QA RECOVERY & TRAKE AUDIT COMPLETE (READY FOR HUMAN VISUAL INSPECTION) <<<")
    print("=" * 150 + "\n")


if __name__ == "__main__":
    main()
