#!/usr/bin/env python3
"""Run BTC Practice Test Queries (THUNGHIEM_20-8) End-to-End.

Covers all 24 queries:
  - 18 KIS queries
  - 3 QA queries
  - 3 TRAKE queries
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

# Ensure openai-clip is available
try:
    import clip
except ImportError:
    print("Installing openai-clip dependency in Python environment...", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "openai-clip", "ftfy", "regex", "tqdm"], check=False)

REPO_ROOT = Path(__file__).resolve().parents[1]
SYSTEM_TAI_SRC = REPO_ROOT / "systems" / "system_tai" / "src"
if str(SYSTEM_TAI_SRC) not in sys.path:
    sys.path.insert(0, str(SYSTEM_TAI_SRC))

from system_tai.kis.session_engine import OperationalKISRuntime
from system_tai.kis.session_schema import (
    QAQueryRequest,
    QueryRequest,
    SessionConfig,
    TRAKEQueryRequest,
)

THUNGHIEM_DIR = REPO_ROOT / "systems" / "system_tai" / "THUNGHIEM_20-8"


def parse_qa_text(text: str) -> tuple[str, str]:
    """Extract event_description and question from Vietnamese QA text."""
    text = text.strip()
    if "Hỏi " in text:
        idx = text.find("Hỏi ")
        event_desc = text[:idx].strip()
        question = text[idx:].strip()
        if not event_desc:
            event_desc = text
        return event_desc, question

    # Fallback to sentence split
    sentences = [s.strip() for s in re.split(r"(?<=[.?!])\s+", text) if s.strip()]
    if len(sentences) > 1 and sentences[-1].endswith("?"):
        event_desc = " ".join(sentences[:-1]).strip()
        question = sentences[-1].strip()
        return event_desc, question

    return text, text


def parse_trake_text(text: str) -> list[dict[str, str]]:
    """Extract ordered events from TRAKE query text."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    events = []
    for idx, line in enumerate(lines, start=1):
        m = re.match(r"^E?(\d+)[:.]\s*(.+)$", line, re.IGNORECASE)
        if m:
            ev_id = f"E{m.group(1)}"
            ev_desc = m.group(2).strip()
        else:
            ev_id = f"E{idx}"
            ev_desc = line
        events.append({"event_id": ev_id, "description": ev_desc})
    if not events:
        events.append({"event_id": "E1", "description": text.strip()})
    return events


def run_thunghiem(target_pattern: str | None = None) -> None:
    print("=" * 145, flush=True)
    print("RUNNING BTC TEST QUERIES: THUNGHIEM_20-8 (All Tracks: KIS, QA, TRAKE)", flush=True)
    print("=" * 145, flush=True)

    if not THUNGHIEM_DIR.exists():
        raise FileNotFoundError(f"Directory not found: {THUNGHIEM_DIR}")

    files = sorted(THUNGHIEM_DIR.glob("*.txt"))
    if target_pattern:
        files = [f for f in files if target_pattern.lower() in f.name.lower()]

    print(f"• Total Query Files Selected       : {len(files)} queries", flush=True)
    print(f"• Query Directory                  : {THUNGHIEM_DIR.relative_to(REPO_ROOT)}", flush=True)

    session_output = Path("/kaggle/working/output/thunghiem_20_8") if Path("/kaggle/working").exists() else REPO_ROOT / "scratch" / "thunghiem_20_8"
    session_output.mkdir(parents=True, exist_ok=True)

    # Manifest auto-detection
    reuse_manifest_path: Path | None = None
    manifest_cache_path: Path | None = None

    for p in [
        Path("/kaggle/working/manifest_cache.json"),
        Path("/kaggle/input/system-tai-manifest/feature_manifest.json"),
        Path("/kaggle/input/datasets/manifest_cache.json"),
        Path("/kaggle/input/manifest_cache.json"),
    ]:
        if p.exists() and p.stat().st_size > 1000:
            reuse_manifest_path = p
            break

    if reuse_manifest_path is None:
        manifest_cache_path = (
            Path("/kaggle/working/manifest_cache.json")
            if Path("/kaggle/working").exists()
            else REPO_ROOT / "scratch" / "manifest_cache.json"
        )

    config = SessionConfig(
        input_root=Path("/kaggle/input/datasets") if Path("/kaggle/input/datasets").exists() else Path("/kaggle/input"),
        reuse_manifest=reuse_manifest_path,
        manifest_cache=manifest_cache_path,
        output_root=session_output,
        device="auto",
        allow_model_download=True,
        default_output_top_k=100,
        default_refine_top_n=3,
        rrf_constant=60.0,
    )

    print("\n--- BOOTSTRAPPING RUNTIME ---", flush=True)
    t0 = time.time()
    runtime = OperationalKISRuntime.bootstrap(config)
    print(f"Runtime bootstrap completed in {time.time() - t0:.2f}s.", flush=True)

    results_summary = []

    for idx, f in enumerate(files, start=1):
        filename = f.name
        raw_text = f.read_text(encoding="utf-8").strip()
        query_id = filename.replace(".txt", "")

        # Determine track
        if "qa" in filename.lower():
            track = "QA"
        elif "trake" in filename.lower():
            track = "TRAKE"
        else:
            track = "KIS"

        print("\n" + "-" * 145, flush=True)
        print(f"[{idx:02d}/{len(files):02d}] Processing {track:<5} | ID: {query_id} ({filename})", flush=True)
        print(f"      Text: {raw_text[:110]}...", flush=True)

        t_q0 = time.time()
        preds = []
        top_preview = []

        try:
            if track == "KIS":
                req = QueryRequest(
                    request_id=f"run-{query_id}",
                    query_id=query_id,
                    query_vi=raw_text,
                    include_vi_variant=True,
                    output_top_k=100,
                    refine_top_n=3,
                )
                res = runtime.handle_query(req)
                top100_rel = res["artifacts"].get("refined_top100_jsonl", res["artifacts"]["top100_jsonl"])
                top100_path = runtime.output_root / top100_rel
                preds = [json.loads(l) for l in top100_path.read_text(encoding="utf-8").splitlines() if l.strip()]

                for p in preds[:5]:
                    f_id = p["frame_id"]
                    sec = f_id // 25
                    top_preview.append(f"Rank @{p['rank']}: {p['video_id']} (frame {f_id}, ~{sec//60:02d}:{sec%60:02d})")

            elif track == "QA":
                event_desc, question = parse_qa_text(raw_text)
                req_qa = QAQueryRequest(
                    request_id=f"run-{query_id}",
                    query_id=query_id,
                    event_description=event_desc,
                    question=question,
                    include_vi_variant=True,
                    output_top_k=100,
                    refine_top_n=3,
                )
                res = runtime.handle_qa_query(req_qa)
                top100_rel = res["artifacts"].get("refined_top100_jsonl", res["artifacts"]["top100_jsonl"])
                top100_path = runtime.output_root / top100_rel
                preds = [json.loads(l) for l in top100_path.read_text(encoding="utf-8").splitlines() if l.strip()]

                for p in preds[:5]:
                    f_id = p["frame_id"]
                    ans = p.get("answer", "")
                    sec = f_id // 25
                    top_preview.append(f"Rank @{p['rank']}: {p['video_id']} (frame {f_id}, ~{sec//60:02d}:{sec%60:02d}) -> Answer: '{ans}'")

            elif track == "TRAKE":
                events = parse_trake_text(raw_text)
                req_trake = TRAKEQueryRequest(
                    request_id=f"run-{query_id}",
                    query_id=query_id,
                    events=tuple(events),
                    include_vi_variant=True,
                    output_top_k=100,
                    refine_top_n=3,
                )
                res = runtime.handle_trake_query(req_trake)
                top100_rel = res["artifacts"].get("refined_top100_jsonl", res["artifacts"]["top100_jsonl"])
                top100_path = runtime.output_root / top100_rel
                preds = [json.loads(l) for l in top100_path.read_text(encoding="utf-8").splitlines() if l.strip()]

                for p in preds[:5]:
                    f_ids = p.get("frame_ids", [p.get("frame_id")])
                    top_preview.append(f"Rank @{p['rank']}: {p['video_id']} (frames {f_ids})")

            elapsed = time.time() - t_q0
            print(f"      • Emitted {len(preds)} candidates in {elapsed:.2f}s", flush=True)
            for prev in top_preview:
                print(f"        - {prev}", flush=True)

            results_summary.append({
                "query_id": query_id,
                "track": track,
                "emitted": len(preds),
                "top1": top_preview[0] if top_preview else "-",
                "elapsed": elapsed,
                "status": "SUCCESS",
            })

        except Exception as exc:
            elapsed = time.time() - t_q0
            print(f"      • ERROR on {query_id}: {exc}", flush=True)
            results_summary.append({
                "query_id": query_id,
                "track": track,
                "emitted": 0,
                "top1": f"ERROR: {exc}",
                "elapsed": elapsed,
                "status": "FAILED",
            })

    print("\n" + "=" * 145, flush=True)
    print("BTC THUNGHIEM_20-8 RUN SUMMARY", flush=True)
    print("=" * 145, flush=True)
    print(f"{'Query ID':<25} | {'Track':<6} | {'Emitted':<7} | {'Time':<8} | {'Top 1 Prediction'}", flush=True)
    print("-" * 145, flush=True)
    for r in results_summary:
        print(f"{r['query_id']:<25} | {r['track']:<6} | {r['emitted']:<7} | {r['elapsed']:5.2f}s  | {r['top1']}", flush=True)
    print("=" * 145, flush=True)
    print(f"Artifacts exported to: {session_output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", "-q", type=str, default=None, help="Filter specific query (e.g. 'p1-1' or 'qa')")
    args = parser.parse_args()
    run_thunghiem(args.query)
