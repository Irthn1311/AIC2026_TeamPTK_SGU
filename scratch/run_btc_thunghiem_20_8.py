#!/usr/bin/env python3
"""Canonical BTC Blind Practice Test Runner (THUNGHIEM_20-8).

Production-like Blind Run Compliance (5 Strict Gates):
  1. Zero DEV Semantic Contamination:
     - Absolutely no loading of kis_dev_gt.json, q2_kis_dev_en_translation.json,
       qa_dev_translations_en.json, benchmark.json, accepted_answers, or DEV intervals.
     - Only technical infra cache (manifest_cache.json, CLIP weights) is reused.
  2. Fail-Closed Deterministic Task Routing:
     - Suffix based: '-kis.txt' -> KIS, '-qa.txt' -> QA, '-trake.txt' -> TRAKE.
     - Unknown suffixes raise immediate ValueError (fail-closed, never default).
  3. Effective Production Config Resolution:
     - Uses canonical SessionConfig with production defaults.
     - KIS: include_vi_variant=True, query_en=None, rrf_k=60.0, refine_top_n=3.
     - QA: canonical production QA pipeline (no synthetic DEV champion substitution).
     - TRAKE: canonical locked video_first + bounded_beam solver.
  4. Task-Specific Output & Export Contract Validation:
     - Asserts 1 <= N <= 100, ranks strictly contiguous 1..N.
     - Validates task-specific schemas (QA answer field, TRAKE event trajectories).
  5. Pure Blind Inference (Zero Evaluator / Zero GT Matching):
     - No GT lookup, no scoring or hit/miss computation.
     - Live preview Top 5 predictions + per-track latency and export summary.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from enum import Enum
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


class BTCTaskType(Enum):
    KIS = "KIS"
    QA = "QA"
    TRAKE = "TRAKE"


def get_git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "UNKNOWN"


def resolve_task_type(filename: str) -> BTCTaskType:
    """Fail-closed deterministic task resolution based on filename suffix contract."""
    name_lower = filename.lower()
    if name_lower.endswith("-kis.txt"):
        return BTCTaskType.KIS
    elif name_lower.endswith("-qa.txt"):
        return BTCTaskType.QA
    elif name_lower.endswith("-trake.txt"):
        return BTCTaskType.TRAKE
    else:
        raise ValueError(f"FATAL: Unrecognized task suffix in filename '{filename}'. Fail-closed policy.")


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


def run_btc_blind_benchmark(target_pattern: str | None = None) -> None:
    print("=" * 145, flush=True)
    print("BTC THUNGHIEM_20-8: CANONICAL PRODUCTION-LIKE BLIND RUNNER", flush=True)
    print("=" * 145, flush=True)
    print(f"• Git HEAD Commit                  : {get_git_head()}", flush=True)
    print(f"• Ingestion Mode                   : PRODUCTION BLIND INFERENCE (Zero DEV Semantic Contamination)", flush=True)
    print(f"• Evaluator Status                 : DISABLED (No GT matching, purely blind predictions)", flush=True)

    if not THUNGHIEM_DIR.exists():
        raise FileNotFoundError(f"BTC Query Directory not found: {THUNGHIEM_DIR}")

    all_files = sorted(THUNGHIEM_DIR.glob("*.txt"))
    if target_pattern:
        patterns = [p.strip().lower() for p in target_pattern.split(",") if p.strip()]
        files = [f for f in all_files if any(p in f.name.lower() for p in patterns)]
    else:
        files = all_files

    print(f"• Total Query Files Selected       : {len(files)} / {len(all_files)} queries", flush=True)
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

    print("\n--- EFFECTIVE PRODUCTION CONFIGURATION SUMMARY ---", flush=True)
    print(f"• Manifest Strategy                : {'REUSE: ' + str(reuse_manifest_path) if reuse_manifest_path else 'BUILD & CACHE: ' + str(manifest_cache_path)}", flush=True)
    print(f"• KIS Production Profile           : include_vi_variant=True, query_en=None, RRF=60.0, TopK=100, RefineTopN=3", flush=True)
    print(f"• QA Production Profile            : canonical production pipeline, visual ontology enabled, ranking=decision_10_top100", flush=True)
    print(f"• TRAKE Production Profile         : video_first restricted candidate search, bounded_beam solver, beam_width=100", flush=True)
    print(f"• Initial Status Marker            : BTC_BLIND_RUNNER_PARTIAL_PASS (QA/TRAKE artifact contract being verified) ⚠️", flush=True)

    print("\n--- BOOTSTRAPPING RUNTIME ---", flush=True)
    t0 = time.time()
    runtime = OperationalKISRuntime.bootstrap(config)
    print(f"Runtime bootstrap completed in {time.time() - t0:.2f}s.", flush=True)

    results_summary = []
    latencies_by_task: dict[str, list[float]] = {"KIS": [], "QA": [], "TRAKE": []}
    export_validation_failures = 0

    for idx, f in enumerate(files, start=1):
        filename = f.name
        raw_text = f.read_text(encoding="utf-8").strip()
        query_id = filename.replace(".txt", "")

        task_type = resolve_task_type(filename)
        track_str = task_type.value

        print("\n" + "-" * 145, flush=True)
        print(f"[{idx:02d}/{len(files):02d}] Processing {track_str:<5} | ID: {query_id} ({filename})", flush=True)
        print(f"      Input Text: {raw_text[:110]}...", flush=True)

        t_q0 = time.time()
        preds = []
        top_preview = []

        try:
            if task_type == BTCTaskType.KIS:
                req = QueryRequest(
                    request_id=f"btc-{query_id}",
                    query_id=query_id,
                    query_vi=raw_text,
                    query_en=None,  # Pure blind input, zero translation sidecar injection
                    include_vi_variant=True,
                    output_top_k=100,
                    refine_top_n=3,
                )
                res = runtime.handle_query(req)
                top100_rel = res["artifacts"].get("refined_top100_jsonl", res["artifacts"].get("top100_jsonl"))
                if not top100_rel:
                    raise KeyError(f"No valid KIS predictions artifact in {res['artifacts']}")
                top100_path = runtime.output_root / top100_rel
                preds = [json.loads(l) for l in top100_path.read_text(encoding="utf-8").splitlines() if l.strip()]

                # Export Schema & Contract Validation
                assert 1 <= len(preds) <= 100, f"Contract violation: {len(preds)} predictions"
                ranks = [p["rank"] for p in preds]
                assert ranks == list(range(1, len(preds) + 1)), "Ranks not strictly contiguous 1..N"
                for p in preds:
                    assert p["query_id"] == query_id, f"Query ID mismatch in prediction: {p}"
                    assert isinstance(p["video_id"], str) and p["video_id"].startswith("L"), f"Invalid video_id: {p}"
                    assert isinstance(p["frame_id"], int) and p["frame_id"] >= 0, f"Invalid frame_id: {p}"

                for p in preds[:5]:
                    f_id = p["frame_id"]
                    sec = f_id // 25
                    top_preview.append(f"Rank @{p['rank']}: {p['video_id']} (frame {f_id}, ~{sec//60:02d}:{sec%60:02d})")

            elif task_type == BTCTaskType.QA:
                event_desc, question = parse_qa_text(raw_text)
                req_qa = QAQueryRequest(
                    request_id=f"btc-{query_id}",
                    query_id=query_id,
                    event_description=event_desc,
                    question=question,
                    event_description_en=None,
                    question_en=None,
                    include_vi_variant=True,
                    output_top_k=100,
                    refine_top_n=3,
                )
                res = runtime.handle_qa_query(req_qa)

                # Correct Artifact Resolution from handle_qa_query() schema contract
                if "qa_predictions_jsonl" in res["artifacts"]:
                    qa_path = runtime.output_root / res["artifacts"]["qa_predictions_jsonl"]
                    preds = [json.loads(l) for l in qa_path.read_text(encoding="utf-8").splitlines() if l.strip()]
                elif "predictions" in res and res["predictions"]:
                    preds = res["predictions"]
                else:
                    raise KeyError(f"No valid QA predictions artifact in {res['artifacts']}")

                # Export Schema & Contract Validation
                assert 1 <= len(preds) <= 100, f"Contract violation: {len(preds)} predictions"
                ranks = [p["rank"] for p in preds]
                assert ranks == list(range(1, len(preds) + 1)), "Ranks not strictly contiguous 1..N"
                for p in preds:
                    assert p["query_id"] == query_id, f"Query ID mismatch: {p}"
                    assert isinstance(p["video_id"], str) and p["video_id"].startswith("L"), f"Invalid video_id: {p}"
                    assert isinstance(p["frame_id"], int) and p["frame_id"] >= 0, f"Invalid frame_id: {p}"
                    assert "answer" in p, f"Missing answer field in QA prediction: {p}"

                for p in preds[:5]:
                    f_id = p["frame_id"]
                    ans = p.get("answer", "")
                    sec = f_id // 25
                    top_preview.append(f"Rank @{p['rank']}: {p['video_id']} (frame {f_id}, ~{sec//60:02d}:{sec%60:02d}) -> Answer: '{ans}'")

            elif task_type == BTCTaskType.TRAKE:
                events = parse_trake_text(raw_text)
                req_trake = TRAKEQueryRequest(
                    request_id=f"btc-{query_id}",
                    query_id=query_id,
                    events=tuple(events),
                    include_vi_variant=True,
                    output_top_k=100,
                    refine_top_n=3,
                )
                res = runtime.handle_trake_query(req_trake)

                # Correct Artifact Resolution from handle_trake_query() schema contract
                if "trake_predictions_jsonl" in res["artifacts"]:
                    trake_path = runtime.output_root / res["artifacts"]["trake_predictions_jsonl"]
                    preds = [json.loads(l) for l in trake_path.read_text(encoding="utf-8").splitlines() if l.strip()]
                elif "predictions" in res and res["predictions"]:
                    preds = res["predictions"]
                else:
                    raise KeyError(f"No valid TRAKE predictions artifact in {res['artifacts']}")

                # Export Schema & Contract Validation
                assert 1 <= len(preds) <= 100, f"Contract violation: {len(preds)} predictions"
                ranks = [p["rank"] for p in preds]
                assert ranks == list(range(1, len(preds) + 1)), "Ranks not strictly contiguous 1..N"
                for p in preds:
                    assert p["query_id"] == query_id, f"Query ID mismatch: {p}"
                    assert isinstance(p["video_id"], str) and p["video_id"].startswith("L"), f"Invalid video_id: {p}"
                    f_ids = p.get("frame_ids", [p.get("frame_id")])
                    assert isinstance(f_ids, (list, tuple)) and len(f_ids) > 0, f"Invalid frame_ids in TRAKE: {p}"

                for p in preds[:5]:
                    f_ids = p.get("frame_ids", [p.get("frame_id")])
                    top_preview.append(f"Rank @{p['rank']}: {p['video_id']} (frames {f_ids})")

            elapsed = time.time() - t_q0
            latencies_by_task[track_str].append(elapsed)

            print(f"      • Emitted {len(preds)} candidates in {elapsed:.2f}s | Export Schema: PASS ✅", flush=True)
            for prev in top_preview:
                print(f"        - {prev}", flush=True)

            results_summary.append({
                "query_id": query_id,
                "track": track_str,
                "emitted": len(preds),
                "top1": top_preview[0] if top_preview else "-",
                "elapsed": elapsed,
                "status": "SUCCESS",
                "validation": "PASS",
            })

        except Exception as exc:
            elapsed = time.time() - t_q0
            export_validation_failures += 1
            latencies_by_task[track_str].append(elapsed)
            print(f"      • ERROR on {query_id}: {exc}", flush=True)
            results_summary.append({
                "query_id": query_id,
                "track": track_str,
                "emitted": 0,
                "top1": f"ERROR: {exc}",
                "elapsed": elapsed,
                "status": "FAILED",
                "validation": "FAIL",
            })

    # ==============================================================================================================
    # 6. SUMMARY & LATENCY STATISTICS
    # ==============================================================================================================
    print("\n" + "=" * 145, flush=True)
    print("BTC THUNGHIEM_20-8 BLIND EXECUTION SUMMARY", flush=True)
    print("=" * 145, flush=True)

    kis_completed = sum(1 for r in results_summary if r["track"] == "KIS" and r["status"] == "SUCCESS")
    qa_completed = sum(1 for r in results_summary if r["track"] == "QA" and r["status"] == "SUCCESS")
    trake_completed = sum(1 for r in results_summary if r["track"] == "TRAKE" and r["status"] == "SUCCESS")

    kis_total = sum(1 for r in results_summary if r["track"] == "KIS")
    qa_total = sum(1 for r in results_summary if r["track"] == "QA")
    trake_total = sum(1 for r in results_summary if r["track"] == "TRAKE")

    n_zero_queries = [r["query_id"] for r in results_summary if r["emitted"] == 0]
    n_under_100_queries = [f"{r['query_id']} (N={r['emitted']})" for r in results_summary if 0 < r["emitted"] < 100]

    all_passed = (export_validation_failures == 0) and (kis_completed == kis_total) and (qa_completed == qa_total) and (trake_completed == trake_total)

    print(f"• KIS   Completed                  : {kis_completed}/{kis_total} | Exceptions: {kis_total - kis_completed}", flush=True)
    print(f"• QA    Completed                  : {qa_completed}/{qa_total} | Exceptions: {qa_total - qa_completed}", flush=True)
    print(f"• TRAKE Completed                  : {trake_completed}/{trake_total} | Exceptions: {trake_total - trake_completed}", flush=True)
    print(f"• Queries with N=0                 : {len(n_zero_queries)} {n_zero_queries if n_zero_queries else 'None'}", flush=True)
    print(f"• Queries with N<100 (dedup/short) : {len(n_under_100_queries)} {n_under_100_queries if n_under_100_queries else 'None'}", flush=True)
    print(f"• Export Validation Status         : {'ALL PASS ✅' if export_validation_failures == 0 else f'{export_validation_failures} FAILURES ❌'}", flush=True)
    print(f"• Overall Status Marker            : {'BTC_BLIND_RUNNER_READY ✅' if all_passed else 'BTC_BLIND_RUNNER_PARTIAL_PASS ⚠️'}", flush=True)

    print("\n--- LATENCY BY TASK ---", flush=True)
    for task_name, l_list in latencies_by_task.items():
        if l_list:
            mean_lat = sum(l_list) / len(l_list)
            sorted_l = sorted(l_list)
            p95_idx = min(len(sorted_l) - 1, math.ceil(0.95 * len(sorted_l)) - 1)
            p95_lat = sorted_l[p95_idx]
            print(f"• {task_name:<5} (N={len(l_list):2d})                   : Mean = {mean_lat:5.2f}s | p95 = {p95_lat:5.2f}s", flush=True)

    print("\n" + "=" * 145, flush=True)
    print(f"{'Query ID':<22} | {'Track':<6} | {'Emitted':<7} | {'Time':<8} | {'Validation':<10} | {'Top 1 Prediction'}", flush=True)
    print("-" * 145, flush=True)
    for r in results_summary:
        print(f"{r['query_id']:<22} | {r['track']:<6} | {r['emitted']:<7} | {r['elapsed']:5.2f}s  | {r['validation']:<10} | {r['top1']}", flush=True)
    print("=" * 145, flush=True)
    print(f"Artifacts exported to: {session_output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", "-q", type=str, default=None, help="Filter specific query (e.g. 'qa,trake' or 'p1-15')")
    args = parser.parse_args()
    run_btc_blind_benchmark(args.query)
