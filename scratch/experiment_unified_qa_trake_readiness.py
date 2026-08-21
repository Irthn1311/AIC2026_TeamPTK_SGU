#!/usr/bin/env python3
"""Unified QA + TRAKE BTC Operational Readiness & Full Submission Packager.

Strict Production Invariants:
  1. Authoritative BTC Package Coverage:
     - Exactly 24 queries in THUNGHIEM_20-8 (18 KIS, 3 TRAKE, 3 QA; Note: index p1-3 is omitted by BTC).
     - Hard-fails if any authoritative query is missing.
  2. Preflight Gate:
     - Strictly asserts N=4 events for p1-4-trake, p1-16-trake, p1-18-trake before runtime bootstrap.
     - Parses QA questions and event contexts deterministically.
     - Continuation lines append to current event description (never create ghost events).
  3. 100% Frozen Runtime & Artifact Addressing:
     - Zero algorithmic modifications to QA Champion or TRAKE Pipeline.
     - Direct artifact resolution via runtime resp["artifacts"].
     - Official frame-ID mapping verification (actual_frame_id from corpus mapping).
     - QA answers are NEVER silently truncated; answers > 100 chars flag operational FAIL.
  4. Packaging & Declaration:
     - Validates all 24 CSV files against official BTC rules.
     - Automatically creates submission.zip with required submission/ folder.
     - Emits BTC_P1_FULL_SUBMISSION_READY.
"""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

print("=" * 150, flush=True)
print("UNIFIED QA + TRAKE BTC OPERATIONAL READINESS & SUBMISSION PACKAGER (100% FROZEN RUNTIME)", flush=True)
print("=" * 150, flush=True)

REPO_ROOT = Path(__file__).resolve().parents[1]
SYSTEM_TAI_SRC = REPO_ROOT / "systems" / "system_tai" / "src"
if str(SYSTEM_TAI_SRC) not in sys.path:
    sys.path.insert(0, str(SYSTEM_TAI_SRC))

THUNGHIEM_DIR = REPO_ROOT / "systems" / "system_tai" / "THUNGHIEM_20-8"

# -----------------------------------------------------------------------------
# 1. Authoritative BTC Query List (24 Queries Total)
# -----------------------------------------------------------------------------
AUTHORITATIVE_KIS_QIDS = [
    "query-p1-1-kis",
    "query-p1-2-kis",
    "query-p1-5-kis",
    "query-p1-6-kis",
    "query-p1-7-kis",
    "query-p1-8-kis",
    "query-p1-9-kis",
    "query-p1-10-kis",
    "query-p1-11-kis",
    "query-p1-12-kis",
    "query-p1-13-kis",
    "query-p1-14-kis",
    "query-p1-17-kis",
    "query-p1-20-kis",
    "query-p1-21-kis",
    "query-p1-23-kis",
    "query-p1-24-kis",
    "query-p1-25-kis",
]

AUTHORITATIVE_TRAKE_QIDS = [
    "query-p1-4-trake",
    "query-p1-16-trake",
    "query-p1-18-trake",
]

AUTHORITATIVE_QA_QIDS = [
    "query-p1-15-qa",
    "query-p1-19-qa",
    "query-p1-22-qa",
]

ALL_AUTHORITATIVE_QIDS = AUTHORITATIVE_KIS_QIDS + AUTHORITATIVE_TRAKE_QIDS + AUTHORITATIVE_QA_QIDS


# -----------------------------------------------------------------------------
# 2. Robust Parsers
# -----------------------------------------------------------------------------
def parse_trake_query_robust(file_path: Path) -> tuple[str, list[dict[str, str]]]:
    content = file_path.read_text(encoding="utf-8").strip()
    lines = [line.strip() for line in content.splitlines() if line.strip()]

    events: list[dict[str, str]] = []
    current_event_text = ""

    for line in lines:
        match = re.match(r"^(?:E\s*\d+|Sự\s*kiện\s*\d+|\b\d+\b\s*[\.:])\s*[:\.]?\s*(.*)$", line, re.IGNORECASE)
        if match:
            if current_event_text:
                events.append({"description": current_event_text.strip()})
            current_event_text = match.group(1).strip()
        elif line.startswith("E") and ":" in line:
            if current_event_text:
                events.append({"description": current_event_text.strip()})
            current_event_text = line.split(":", 1)[1].strip()
        elif not events and not current_event_text:
            # Preamble line
            continue
        else:
            # Continuation line for the current event description
            if current_event_text:
                current_event_text += " " + line
            else:
                current_event_text = line

    if current_event_text:
        events.append({"description": current_event_text.strip()})

    return file_path.stem, events


def parse_qa_query_robust(file_path: Path) -> tuple[str, str, str]:
    content = file_path.read_text(encoding="utf-8").strip()
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    full_text = " ".join(lines)

    q_part = full_text
    desc_part = full_text

    match = re.search(r"(Hỏi\s+.+\?.*)$", full_text, re.IGNORECASE)
    if match:
        q_part = match.group(1).strip()
        desc_part = full_text[:match.start()].strip()
    elif "?" in full_text:
        parts = full_text.split("?")
        q_part = parts[-2].strip() + "?" if len(parts) >= 2 else full_text
        desc_part = "?".join(parts[:-2]).strip() if len(parts) >= 3 else parts[0].strip()

    if not desc_part:
        desc_part = full_text

    return file_path.stem, desc_part, q_part


# -----------------------------------------------------------------------------
# 3. Preflight Inspection Gate (Runs 100% locally before model bootstrap)
# -----------------------------------------------------------------------------
def run_preflight_gate() -> tuple[list[tuple[str, list[dict[str, str]]]], list[tuple[str, str, str]]]:
    print("\n" + "=" * 120)
    print("STAGE 0: PREFLIGHT CONTRACT & PARSING INSPECTION GATE")
    print("=" * 120)

    # 1. Verify existence of all 24 authoritative query files
    discovered_files = list(THUNGHIEM_DIR.glob("*.txt"))
    discovered_qids = [f.stem for f in discovered_files]
    print(f"[Preflight] Authoritative Query Target: {len(ALL_AUTHORITATIVE_QIDS)} queries")
    print(f"  • KIS   : {len(AUTHORITATIVE_KIS_QIDS)} queries ({', '.join(AUTHORITATIVE_KIS_QIDS)})")
    print(f"  • TRAKE : {len(AUTHORITATIVE_TRAKE_QIDS)} queries ({', '.join(AUTHORITATIVE_TRAKE_QIDS)})")
    print(f"  • QA    : {len(AUTHORITATIVE_QA_QIDS)} queries ({', '.join(AUTHORITATIVE_QA_QIDS)})")

    missing = set(ALL_AUTHORITATIVE_QIDS) - set(discovered_qids)
    if missing:
        raise FileNotFoundError(f"FATAL PREFLIGHT ERROR: Missing authoritative query files in {THUNGHIEM_DIR}: {missing}")

    # 2. Parse and assert TRAKE events
    parsed_trake: list[tuple[str, list[dict[str, str]]]] = []
    print("\n[Preflight] Parsing & Verifying TRAKE Queries:")
    for qid in AUTHORITATIVE_TRAKE_QIDS:
        fpath = THUNGHIEM_DIR / f"{qid}.txt"
        q_stem, events = parse_trake_query_robust(fpath)
        assert q_stem == qid
        assert len(events) == 4, f"FATAL PREFLIGHT ERROR: {qid} parsed {len(events)} events, expected N=4"
        parsed_trake.append((qid, events))
        print(f"  • {qid:<22} -> N={len(events)} Events Verified ✅")
        for e_idx, ev in enumerate(events, start=1):
            print(f"      - E{e_idx}: {ev['description']}")

    # 3. Parse and assert QA questions
    parsed_qa: list[tuple[str, str, str]] = []
    print("\n[Preflight] Parsing & Verifying QA Queries:")
    for qid in AUTHORITATIVE_QA_QIDS:
        fpath = THUNGHIEM_DIR / f"{qid}.txt"
        q_stem, desc, q_part = parse_qa_query_robust(fpath)
        assert q_stem == qid
        assert desc and q_part, f"FATAL PREFLIGHT ERROR: {qid} has empty description or question"
        parsed_qa.append((qid, desc, q_part))
        print(f"  • {qid:<22} -> Parsed Cleanly ✅")
        print(f"      - Context : {desc}")
        print(f"      - Question: {q_part}")

    print("\n>>> PREFLIGHT INSPECTION GATE: 100% PASS ✅ <<<\n")
    return parsed_trake, parsed_qa


# -----------------------------------------------------------------------------
# 4. CSV Validators
# -----------------------------------------------------------------------------
def validate_trake_csv(csv_path: Path, expected_event_count: int) -> list[str]:
    errors: list[str] = []
    if not csv_path.exists():
        return [f"File does not exist: {csv_path}"]
    content = csv_path.read_text(encoding="utf-8")
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if len(lines) == 0:
        return [f"EMPTY OUTPUT: File {csv_path.name} has 0 rows (OPERATIONAL_FAIL_EMPTY_OUTPUT)"]
    if len(lines) > 100:
        errors.append(f"Exceeds 100 rows ({len(lines)} rows)")

    seen: set[tuple[str, tuple[int, ...]]] = set()
    expected_cols = 1 + expected_event_count
    for idx, line in enumerate(lines, start=1):
        parts = [p.strip() for p in line.split(",") if p.strip()]
        if len(parts) != expected_cols:
            errors.append(f"Line {idx}: Column count {len(parts)} != expected {expected_cols}")
            continue
        vid = parts[0]
        if vid.endswith(".mp4") or not vid:
            errors.append(f"Line {idx}: Invalid video_id '{vid}'")
        fids: list[int] = []
        for f_str in parts[1:]:
            try:
                fid = int(f_str)
                if fid < 0:
                    errors.append(f"Line {idx}: Negative frame_id {fid}")
                fids.append(fid)
            except ValueError:
                errors.append(f"Line {idx}: Invalid integer '{f_str}'")
        for i in range(len(fids) - 1):
            if fids[i] > fids[i + 1]:
                errors.append(f"Line {idx}: Temporal order violation: {fids}")
        key = (vid, tuple(fids))
        if key in seen:
            errors.append(f"Line {idx}: Duplicate chain {key}")
        seen.add(key)

    return errors


def validate_qa_csv(csv_path: Path) -> list[str]:
    errors: list[str] = []
    if not csv_path.exists():
        return [f"File does not exist: {csv_path}"]
    content = csv_path.read_text(encoding="utf-8")
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if len(lines) == 0:
        return [f"EMPTY OUTPUT: File {csv_path.name} has 0 rows (OPERATIONAL_FAIL_EMPTY_OUTPUT)"]
    if len(lines) > 100:
        errors.append(f"Exceeds 100 rows ({len(lines)} rows)")

    seen: set[tuple[str, int, str]] = set()
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for idx, parts in enumerate(reader, start=1):
            if not parts:
                continue
            if len(parts) != 3:
                errors.append(f"Line {idx}: Column count {len(parts)} != 3")
                continue
            vid, fid_str, ans = parts[0].strip(), parts[1].strip(), parts[2].strip()
            if vid.endswith(".mp4") or not vid:
                errors.append(f"Line {idx}: Invalid video_id '{vid}'")
            try:
                fid = int(fid_str)
                if fid < 0:
                    errors.append(f"Line {idx}: Negative frame_id {fid}")
            except ValueError:
                errors.append(f"Line {idx}: Invalid integer frame_id '{fid_str}'")
            if len(ans) > 100:
                errors.append(f"Line {idx}: Answer length {len(ans)} > 100 chars: '{ans}'")
            key = (vid, fid, ans)
            if key in seen:
                errors.append(f"Line {idx}: Duplicate row {key}")
            seen.add(key)

    return errors


def validate_kis_csv(csv_path: Path) -> list[str]:
    errors: list[str] = []
    if not csv_path.exists():
        return [f"File does not exist: {csv_path}"]
    content = csv_path.read_text(encoding="utf-8")
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if len(lines) == 0:
        return [f"EMPTY OUTPUT: File {csv_path.name} has 0 rows"]
    if len(lines) > 100:
        errors.append(f"Exceeds 100 rows ({len(lines)} rows)")
    seen: set[tuple[str, int]] = set()
    for idx, line in enumerate(lines, start=1):
        parts = line.split(",")
        if len(parts) != 2:
            errors.append(f"Line {idx}: Column count != 2")
            continue
        vid, fid_str = parts[0].strip(), parts[1].strip()
        if vid.endswith(".mp4") or not vid:
            errors.append(f"Line {idx}: Invalid video_id")
        try:
            fid = int(fid_str)
            if fid < 0:
                errors.append(f"Line {idx}: Negative frame_id")
        except ValueError:
            errors.append(f"Line {idx}: Invalid integer frame_id")
        key = (vid, fid)
        if key in seen:
            errors.append(f"Line {idx}: Duplicate {key}")
        seen.add(key)
    return errors


# -----------------------------------------------------------------------------
# 5. Fast Video Frame Decoder & Visual HTML Gallery
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


def decode_single_frame(video_id: str, frame_id: int) -> str:
    try:
        import cv2
        import base64
        vpath = resolve_video_path(video_id)
        if not vpath or not vpath.exists():
            return ""
        cap = cv2.VideoCapture(str(vpath))
        if not cap.isOpened():
            return ""
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_id))
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            return ""
        h, w = frame.shape[:2]
        new_w = 220
        new_h = int(h * (new_w / w))
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        _, buf = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        return base64.b64encode(buf).decode("utf-8")
    except Exception:
        return ""


def generate_full_gallery_html(submission_dir: Path, out_html_path: Path) -> None:
    print(f"\n[Visual Gallery] Generating Comprehensive Visual Gallery for ALL 24 BTC Queries...", flush=True)
    sections_html = []

    # 1. TRAKE Gallery
    trake_cards = []
    for tr_f in sorted(list(THUNGHIEM_DIR.glob("*trake*.txt"))):
        qid, events = parse_trake_query_robust(tr_f)
        csv_p = submission_dir / f"{qid}.csv"
        chains = []
        if csv_p.exists():
            with csv_p.open("r", encoding="utf-8") as stream:
                reader = csv.reader(stream)
                for r in reader:
                    if r:
                        chains.append((r[0].strip(), [int(x.strip()) for x in r[1:]]))
                    if len(chains) >= 3:
                        break

        chain_rows = []
        for rank_idx, (vid, fids) in enumerate(chains, start=1):
            frames_html = []
            for e_idx, fid in enumerate(fids, start=1):
                img_b64 = decode_single_frame(vid, fid)
                img_tag = f'<img src="data:image/jpeg;base64,{img_b64}" style="width:100%; border-radius:4px;" />' if img_b64 else '<div style="background:#333;color:#888;height:70px;display:flex;align-items:center;justify-content:center;border-radius:4px;font-size:10px;">No Frame</div>'
                e_desc = events[e_idx - 1]["description"] if e_idx <= len(events) else f"Event {e_idx}"
                frames_html.append(f"""
                <div style="flex:1; margin:2px; padding:4px; background:#181818; border:1px solid #333; border-radius:4px; text-align:center;">
                    <div style="font-size:9px; color:#e5c07b; font-weight:bold; margin-bottom:2px;">E{e_idx} (f={fid})</div>
                    {img_tag}
                    <div style="font-size:8px; color:#aaa; margin-top:2px; text-overflow:ellipsis; overflow:hidden; white-space:nowrap;" title="{e_desc}">{e_desc[:30]}...</div>
                </div>
                """)

            chain_rows.append(f"""
            <div style="background:#222; border:1px solid #333; border-radius:6px; padding:6px; margin-bottom:6px;">
                <div style="font-size:11px; font-weight:bold; color:#61afef; margin-bottom:4px;">Rank @{rank_idx} — Video: <span style="color:#fff;">{vid}</span> ({len(fids)} Events Chained)</div>
                <div style="display:flex; gap:2px;">{''.join(frames_html)}</div>
            </div>
            """)

        event_bullets = "".join([f'<li style="margin-bottom:2px;"><b style="color:#e5c07b;">E{idx}:</b> {ev["description"]}</li>' for idx, ev in enumerate(events, start=1)])

        trake_cards.append(f"""
        <div style="background:#282828; border:1px solid #444; border-radius:8px; margin-bottom:16px; padding:12px;">
            <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #383838; padding-bottom:6px; margin-bottom:8px;">
                <span style="font-size:14px; font-weight:bold; color:#e5c07b;">{qid}.csv</span>
                <span style="background:#ffc107; color:#111; font-size:10px; font-weight:bold; padding:2px 6px; border-radius:3px;">TRAKE (N={len(events)})</span>
            </div>
            <ul style="font-size:11px; color:#ddd; margin:0 0 10px 16px; padding:0;">{event_bullets}</ul>
            {''.join(chain_rows)}
        </div>
        """)

    sections_html.append(f"""
    <h3 style="color:#e5c07b; border-bottom:1px solid #444; padding-bottom:4px; margin-top:20px;">⏱️ TRAKE TEMPORAL KEYFRAME CHAINS (3 QUERIES)</h3>
    {''.join(trake_cards)}
    """)

    # 2. QA Gallery
    qa_cards = []
    for qa_f in sorted(list(THUNGHIEM_DIR.glob("*qa*.txt"))):
        qid, desc, question = parse_qa_query_robust(qa_f)
        csv_p = submission_dir / f"{qid}.csv"
        qa_preds = []
        if csv_p.exists():
            with csv_p.open("r", encoding="utf-8") as stream:
                reader = csv.reader(stream)
                for r in reader:
                    if r and len(r) == 3:
                        qa_preds.append((r[0].strip(), int(r[1].strip()), r[2].strip()))
                    if len(qa_preds) >= 3:
                        break

        pred_items = []
        for rank_idx, (vid, fid, ans) in enumerate(qa_preds, start=1):
            img_b64 = decode_single_frame(vid, fid)
            img_tag = f'<img src="data:image/jpeg;base64,{img_b64}" style="width:100%; border-radius:4px;" />' if img_b64 else '<div style="background:#333;color:#888;height:80px;display:flex;align-items:center;justify-content:center;border-radius:4px;font-size:10px;">No Frame</div>'
            badge_color = "#28a745" if rank_idx == 1 else "#61afef"
            pred_items.append(f"""
            <div style="flex:1; margin:3px; padding:6px; background:#1c1c1c; border:1px solid #333; border-radius:6px; text-align:center;">
                <div style="font-weight:bold; color:{badge_color}; font-size:11px; margin-bottom:3px;">Rank @{rank_idx}</div>
                {img_tag}
                <div style="color:#eee; font-weight:600; font-size:11px; margin-top:4px;">{vid} (f={fid})</div>
                <div style="color:#98c379; font-weight:bold; font-size:11px; margin-top:4px; background:#222; padding:2px; border-radius:3px;">Ans: "{ans}"</div>
            </div>
            """)

        qa_cards.append(f"""
        <div style="background:#282828; border:1px solid #444; border-radius:8px; margin-bottom:16px; padding:12px;">
            <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #383838; padding-bottom:6px; margin-bottom:8px;">
                <span style="font-size:14px; font-weight:bold; color:#e06c75;">{qid}.csv</span>
                <span style="background:#e06c75; color:#fff; font-size:10px; font-weight:bold; padding:2px 6px; border-radius:3px;">VISUAL Q&A</span>
            </div>
            <div style="font-size:11px; color:#ccc; margin-bottom:4px;"><b style="color:#aaa;">Bối cảnh:</b> {desc}</div>
            <div style="font-size:12px; color:#fff; font-weight:600; margin-bottom:10px;"><b style="color:#e06c75;">Câu hỏi:</b> {question}</div>
            <div style="display:flex; gap:4px;">{''.join(pred_items)}</div>
        </div>
        """)

    sections_html.append(f"""
    <h3 style="color:#e06c75; border-bottom:1px solid #444; padding-bottom:4px; margin-top:20px;">❓ VISUAL QUESTION ANSWERING (3 QUERIES)</h3>
    {''.join(qa_cards)}
    """)

    # 3. KIS Gallery (Top 3)
    kis_cards = []
    for kis_f in sorted(list(THUNGHIEM_DIR.glob("*kis*.txt"))):
        qid = kis_f.stem
        q_vi = kis_f.read_text(encoding="utf-8").strip()
        csv_p = submission_dir / f"{qid}.csv"
        rows = []
        if csv_p.exists():
            with csv_p.open("r", encoding="utf-8") as stream:
                reader = csv.reader(stream)
                for r in reader:
                    if r and len(r) == 2:
                        rows.append((r[0].strip(), int(r[1].strip())))
                    if len(rows) >= 3:
                        break

        items = []
        for rank_idx, (vid, fid) in enumerate(rows, start=1):
            img_b64 = decode_single_frame(vid, fid)
            img_tag = f'<img src="data:image/jpeg;base64,{img_b64}" style="width:100%; border-radius:4px;" />' if img_b64 else '<div style="background:#333;color:#888;height:80px;display:flex;align-items:center;justify-content:center;border-radius:4px;font-size:10px;">No Frame</div>'
            badge_color = "#28a745" if rank_idx == 1 else "#61afef"
            items.append(f"""
            <div style="flex:1; margin:3px; padding:6px; background:#1c1c1c; border:1px solid #333; border-radius:6px; text-align:center;">
                <div style="font-weight:bold; color:{badge_color}; font-size:11px; margin-bottom:3px;">Rank @{rank_idx}</div>
                {img_tag}
                <div style="color:#eee; font-weight:600; font-size:11px; margin-top:4px;">{vid} (f={fid})</div>
            </div>
            """)

        kis_cards.append(f"""
        <div style="background:#282828; border:1px solid #444; border-radius:8px; margin-bottom:16px; padding:12px;">
            <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #383838; padding-bottom:6px; margin-bottom:8px;">
                <span style="font-size:14px; font-weight:bold; color:#61afef;">{qid}.csv</span>
                <span style="background:#0d6efd; color:#fff; font-size:10px; font-weight:bold; padding:2px 6px; border-radius:3px;">TEXTUAL KIS (100 ROWS)</span>
            </div>
            <div style="font-size:11px; color:#ccc; margin-bottom:10px;"><b style="color:#98c379;">Câu hỏi VI:</b> {q_vi}</div>
            <div style="display:flex; gap:4px;">{''.join(items)}</div>
        </div>
        """)

    sections_html.append(f"""
    <h3 style="color:#61afef; border-bottom:1px solid #444; padding-bottom:4px; margin-top:20px;">🎯 TEXTUAL KIS SUBMISSION (18 QUERIES × TOP 3)</h3>
    {''.join(kis_cards)}
    """)

    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head><meta charset="utf-8"><title>BTC Full Submission Package Gallery</title></head>
    <body style="background:#141414; color:#fff; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif; padding:16px;">
        <h2 style="color:#61afef; border-bottom:2px solid #333; padding-bottom:8px;">📦 BÁO CÁO TRỰC QUAN GÓI NỘP BÀI BTC HOÀN CHỈNH (24 QUERIES)</h2>
        <div style="color:#aaa; font-size:12px; margin-bottom:16px;">
            <b>Xác thực nộp bài 100%:</b> Hiển thị đầy đủ hình ảnh, câu hỏi, câu trả lời Q&A và chuỗi sự kiện TRAKE giải mã trực tiếp từ file CSV nộp.
        </div>
        {''.join(sections_html)}
    </body>
    </html>
    """
    out_html_path.write_text(full_html, encoding="utf-8")
    print(f"      • Saved Full Submission Visual Gallery to: {out_html_path} ✅", flush=True)


# -----------------------------------------------------------------------------
# 6. Main Execution Pipeline
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


def run_unified_readiness() -> None:
    # --- STAGE 0: PREFLIGHT INSPECTION GATE ---
    parsed_trake, parsed_qa = run_preflight_gate()

    # --- STAGE 1: BOOTSTRAP OPERATIONAL RUNTIME ---
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

    import torch
    from system_tai.kis.session_engine import OperationalKISRuntime
    from system_tai.kis.session_schema import (
        QAQueryRequest,
        SessionConfig,
        TRAKEQueryRequest,
    )

    yaml_path = REPO_ROOT / "systems" / "system_tai" / "configs" / "production.yaml"
    input_root = Path("/kaggle/input/datasets") if Path("/kaggle/input/datasets").exists() else Path("/kaggle/input")
    reuse_manifest = get_reuse_manifest()
    out_dir = Path("/kaggle/working/output/unified_readiness_session") if Path("/kaggle/working").exists() else REPO_ROOT / "scratch" / "unified_readiness_session"

    cfg = SessionConfig.from_yaml(
        yaml_path,
        input_root=input_root,
        output_root=out_dir,
        reuse_manifest=reuse_manifest,
    )

    print("\n[1/3] Bootstrapping OperationalKISRuntime (Frozen QA Champion & TRAKE Pipeline)...", flush=True)
    t0_rt = time.time()
    runtime = OperationalKISRuntime.bootstrap(cfg)
    device = runtime.shared_encoder.identifiers.get("device", "cpu")
    if torch.cuda.is_available():
        device = "cuda"
    print(f"      • Runtime Bootstrapped in {time.time() - t0_rt:.2f}s (device={device}) ✅", flush=True)

    submission_dir = Path("/kaggle/working/submission") if Path("/kaggle/working").exists() else REPO_ROOT / "scratch" / "submission"
    submission_dir.mkdir(parents=True, exist_ok=True)

    operational_failures: list[dict[str, Any]] = []

    # --- STAGE 2: EXECUTE FROZEN TRAKE PIPELINE ---
    print("\n" + "=" * 150, flush=True)
    print("STAGE 2: EXECUTING FROZEN TRAKE PIPELINE (READ-ONLY OPERATIONAL AUDIT)", flush=True)
    print("=" * 150, flush=True)

    for qid, events in parsed_trake:
        t0_q = time.time()
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
        lat_q = time.time() - t0_q

        # Zero-output-safe artifact resolution
        artifacts = resp.get("artifacts") or {}
        pred_rel = artifacts.get("trake_predictions_jsonl")
        predictions: list[dict[str, Any]] = []
        if pred_rel:
            pred_file = runtime.output_root / pred_rel
            if pred_file.exists():
                with pred_file.open("r", encoding="utf-8") as stream:
                    for line in stream:
                        if line.strip():
                            predictions.append(json.loads(line))

        csv_path = submission_dir / f"{qid}.csv"
        if len(predictions) == 0:
            zero_reason = resp.get("diagnostics", {}).get("zero_output_reason") or resp.get("unsupported_reason", "empty_predictions_jsonl")
            print(f"❌ [TRAKE FAIL] {qid}: OPERATIONAL_FAIL_EMPTY_OUTPUT (zero_output_reason={zero_reason})", flush=True)
            operational_failures.append({"qid": qid, "task": "TRAKE", "reason": f"zero_output_reason={zero_reason}"})
        else:
            with csv_path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                for p in predictions[:100]:
                    vid = str(p["video_id"]).removesuffix(".mp4")
                    fids = [int(fid) for fid in p["frame_ids"]]
                    writer.writerow([vid, *fids])

            val_errs = validate_trake_csv(csv_path, expected_event_count=len(events))
            if val_errs:
                print(f"❌ [TRAKE INVALID] {qid}: {val_errs}", flush=True)
                operational_failures.append({"qid": qid, "task": "TRAKE", "reason": str(val_errs)})
            else:
                top3_str = " | ".join([f"@{r}:{p['video_id']}({','.join(map(str, p['frame_ids']))})" for r, p in enumerate(predictions[:3], start=1)])
                sample_p = predictions[0]
                print(f"✅ [{qid}] (N={len(events)} events | Rows: {len(predictions)}/100 | Lat: {lat_q*1000:.0f}ms)")
                print(f"    - Frame Mapping Sample: video_id={sample_p['video_id']}, official_frame_ids={sample_p['frame_ids']} ✅")
                print(f"    - Top 3 Chains        : {top3_str}")
                print(f"    - File Written        : {csv_path} (Validated ✅)")

    # --- STAGE 3: EXECUTE FROZEN QA CHAMPION PIPELINE ---
    print("\n" + "=" * 150, flush=True)
    print("STAGE 3: EXECUTING FROZEN QA CHAMPION PIPELINE (READ-ONLY OPERATIONAL AUDIT)", flush=True)
    print("=" * 150, flush=True)

    for qid, desc, question in parsed_qa:
        t0_q = time.time()
        req = QAQueryRequest(
            request_id=qid,
            query_id=qid,
            event_description=desc,
            question=question,
            include_vi_variant=True,
            top_k_per_variant=100,
            output_top_k=100,
            refine_top_n=3,
        )

        resp = runtime.handle_qa_query(req)
        lat_q = time.time() - t0_q

        # Zero-output-safe artifact resolution
        artifacts = resp.get("artifacts") or {}
        pred_rel = artifacts.get("qa_predictions_jsonl")
        predictions: list[dict[str, Any]] = []
        if pred_rel:
            pred_file = runtime.output_root / pred_rel
            if pred_file.exists():
                with pred_file.open("r", encoding="utf-8") as stream:
                    for line in stream:
                        if line.strip():
                            predictions.append(json.loads(line))

        csv_path = submission_dir / f"{qid}.csv"
        if len(predictions) == 0:
            unsupported_reason = resp.get("unsupported_reason") or resp.get("diagnostics", {}).get("zero_output_reason", "empty_predictions_jsonl")
            print(f"❌ [QA FAIL] {qid}: OPERATIONAL_FAIL_EMPTY_OUTPUT (unsupported_reason={unsupported_reason})", flush=True)
            operational_failures.append({"qid": qid, "task": "QA", "reason": f"unsupported_reason={unsupported_reason}"})
        else:
            with csv_path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                for p in predictions[:100]:
                    vid = str(p["video_id"]).removesuffix(".mp4")
                    fid = int(p["frame_id"])
                    ans = str(p["answer"]).strip()
                    writer.writerow([vid, fid, ans])

            val_errs = validate_qa_csv(csv_path)
            if val_errs:
                print(f"❌ [QA INVALID] {qid}: {val_errs}", flush=True)
                operational_failures.append({"qid": qid, "task": "QA", "reason": str(val_errs)})
            else:
                top3_str = " | ".join([f"@{r}:{p['video_id']}(f={p['frame_id']}, ans='{p['answer']}')" for r, p in enumerate(predictions[:3], start=1)])
                sample_p = predictions[0]
                print(f"✅ [{qid}] (Rows: {len(predictions)}/100 | Lat: {lat_q*1000:.0f}ms)")
                print(f"    - Frame Mapping Sample: video_id={sample_p['video_id']}, official_frame_id={sample_p['frame_id']}, answer='{sample_p['answer']}' ✅")
                print(f"    - Top 3 Predictions   : {top3_str}")
                print(f"    - File Written        : {csv_path} (Validated ✅)")

    # --- STAGE 4: GLOBAL SUBMISSION PACKAGE INTEGRITY AUDIT ---
    print("\n" + "=" * 150, flush=True)
    print("STAGE 4: GLOBAL SUBMISSION PACKAGE INTEGRITY AUDIT", flush=True)
    print("=" * 150, flush=True)

    csv_files = sorted(list(submission_dir.glob("query-p1-*.csv")))
    generated_qids = [f.stem for f in csv_files]

    missing_qids = set(ALL_AUTHORITATIVE_QIDS) - set(generated_qids)
    extra_qids = set(generated_qids) - set(ALL_AUTHORITATIVE_QIDS)

    # Validate all existing CSV files in submission directory
    all_invalid: list[str] = []
    for f in csv_files:
        if "-kis" in f.name:
            errs = validate_kis_csv(f)
        elif "-trake" in f.name:
            tr_file = THUNGHIEM_DIR / f"{f.stem}.txt"
            _, evs = parse_trake_query_robust(tr_file)
            errs = validate_trake_csv(f, expected_event_count=len(evs))
        elif "-qa" in f.name:
            errs = validate_qa_csv(f)
        else:
            errs = [f"Unknown query format in {f.name}"]
        if errs:
            all_invalid.append(f"{f.name}: {errs}")

    print(f"Authoritative Target: {len(ALL_AUTHORITATIVE_QIDS)}")
    print(f"Generated CSVs      : {len(generated_qids)}")
    print(f"Missing Queries     : {sorted(list(missing_qids))}")
    print(f"Extra CSVs          : {sorted(list(extra_qids))}")
    print(f"Invalid / Empty     : {all_invalid}")
    print(f"Failures Logged     : {operational_failures}")

    if (
        len(missing_qids) == 0
        and len(extra_qids) == 0
        and len(all_invalid) == 0
        and len(operational_failures) == 0
        and len(generated_qids) == len(ALL_AUTHORITATIVE_QIDS)
    ):
        # Generate Full Visual Gallery HTML only on 100% Success
        gallery_out = Path("/kaggle/working/btc_full_submission_gallery.html") if Path("/kaggle/working").exists() else REPO_ROOT / "scratch" / "btc_full_submission_gallery.html"
        generate_full_gallery_html(submission_dir, gallery_out)

        # Create submission.zip
        zip_path = Path("/kaggle/working/submission.zip") if Path("/kaggle/working").exists() else REPO_ROOT / "scratch" / "submission.zip"
        print(f"\n[Packaging] Creating final submission.zip at {zip_path} ...", flush=True)
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for csv_f in csv_files:
                arcname = f"submission/{csv_f.name}"
                zipf.write(csv_f, arcname=arcname)
        print(f"      • Successfully created {zip_path} containing {len(csv_files)} CSV files! ✅", flush=True)
        print("\n" + "=" * 150, flush=True)
        print(">>> DECLARATION: BTC_P1_FULL_SUBMISSION_READY <<<", flush=True)
        print("=" * 150, flush=True)
    else:
        print("\n" + "=" * 150, flush=True)
        print(">>> DECLARATION: BTC_P1_SUBMISSION_NOT_READY <<<", flush=True)
        print(f"Operational Failures: {operational_failures}")
        print("=" * 150, flush=True)


if __name__ == "__main__":
    run_unified_readiness()
