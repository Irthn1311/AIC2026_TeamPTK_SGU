#!/usr/bin/env python3
"""Canonical KIS Marian Dynamic Translation Benchmark Runner (Full-38 DEV & 5-Query Smoke).

Strict Contract & Parity with b49f628:
  1. Exact same OperationalKISRuntime.handle_query execution path:
     - Includes default_refine_top_n=3 (Phase 4 multi-scale raw-video decoding & refinement).
     - Same candidate materialization & contiguous Top100 JSONL export.
     - Same round-trip frame semantic assertion (internal frame == exported frame == evaluator frame).
     - Same evaluator against official kis_dev_gt.json intervals.
  2. Pure Vietnamese Ingestion -> Marian Dynamic Translation -> EN_ONLY variant:
     - query_vi fed directly to MarianOfflineTranslator (zero human-reviewed English sidecar access).
     - TokenBudgetGuard validation (<= 75 content tokens).
     - EN_ONLY retrieval (zero Vietnamese ranking fusion).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

REPO_ROOT = Path(__file__).resolve().parents[1]
SYSTEM_TAI_SRC = REPO_ROOT / "systems" / "system_tai" / "src"
if str(SYSTEM_TAI_SRC) not in sys.path:
    sys.path.insert(0, str(SYSTEM_TAI_SRC))

try:
    import clip
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "openai-clip", "ftfy", "regex", "tqdm", "transformers", "sentencepiece"], check=False)
    import clip

try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers", "sentencepiece"], check=False)
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Purge stale system_tai modules from sys.modules if running interactively in Jupyter
for mod in list(sys.modules.keys()):
    if mod.startswith("system_tai"):
        del sys.modules[mod]

from system_tai.kis.session_engine import OperationalKISRuntime
from system_tai.kis.session_schema import QueryRequest, SessionConfig

FROZEN_KIS_DEV_GT_SHA256 = "7d25708b7243ca2b9964bad9a2b65b63354acd74eddb100167f49e1166f8e5b2"
OFFICIAL_K = (1, 5, 20, 50, 100)


def get_git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "UNKNOWN"


def run_canonical_marian_dev(limit: int | None = None, device: str = "auto") -> None:
    gt_path = REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "l21_150_diagnostic" / "kis_dev_gt.json"
    sidecar_path = REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "l21_150_diagnostic" / "q2_kis_dev_en_translation.json"

    gt_bytes = gt_path.read_bytes()
    gt_sha = hashlib.sha256(gt_bytes).hexdigest()

    print("=" * 150, flush=True)
    print(f"CANONICAL KIS MARIAN EN_ONLY BENCHMARK RUNNER ({'5-QUERY SMOKE' if limit == 5 else 'FULL-38 DEV'})", flush=True)
    print("=" * 150, flush=True)
    print(f"• Git HEAD Commit                  : {get_git_head()}", flush=True)
    print(f"• DEV GT Source Provenance         : USER_TEAM_PROVIDED_KIS_DEV_DIAGNOSTIC_GT", flush=True)
    print(f"• DEV-Only GT Path                 : {gt_path.relative_to(REPO_ROOT)}", flush=True)
    print(f"• DEV-Only GT SHA256               : {gt_sha} ({'MATCH ✅' if gt_sha == FROZEN_KIS_DEV_GT_SHA256 else 'MISMATCH ❌'})", flush=True)
    print(f"• Ingestion Strategy               : PURE VIETNAMESE -> MARIAN DYNAMIC TRANSLATION -> EN_ONLY RETRIEVAL", flush=True)
    print(f"• Zero Sidecar English Access      : EN sidecar ignored (only source_vi extracted)", flush=True)

    if gt_sha != FROZEN_KIS_DEV_GT_SHA256:
        raise ValueError(f"CRITICAL: DEV GT SHA256 mismatch! Got {gt_sha}")

    gt_data = json.loads(gt_bytes.decode("utf-8"))
    raw_queries = gt_data.get("queries", gt_data)
    sidecar_data = json.loads(sidecar_path.read_text(encoding="utf-8"))
    vi_lookup = {rec["query_id"]: rec["source_vi"] for rec in sidecar_data.get("records", [])}

    selected_queries = raw_queries[:limit] if limit else raw_queries
    print(f"• Total Queries to Execute         : {len(selected_queries)} queries", flush=True)

    session_output = Path("/kaggle/working/output/kis_marian_canonical") if Path("/kaggle/working").exists() else REPO_ROOT / "scratch" / "kis_marian_canonical"
    if session_output.exists():
        shutil.rmtree(session_output, ignore_errors=True)
    session_output.mkdir(parents=True, exist_ok=True)

    reuse_manifest_path: Path | None = None
    for p in [
        Path("/kaggle/working/manifest_cache.json"),
        Path("/kaggle/input/system-tai-manifest/feature_manifest.json"),
        Path("/kaggle/input/datasets/manifest_cache.json"),
        Path("/kaggle/input/manifest_cache.json"),
    ]:
        if p.exists() and p.stat().st_size > 1000:
            reuse_manifest_path = p
            break

    config = SessionConfig(
        input_root=Path("/kaggle/input/datasets") if Path("/kaggle/input/datasets").exists() else Path("/kaggle/input"),
        reuse_manifest=reuse_manifest_path,
        output_root=session_output,
        device=device,
        allow_model_download=True,
        default_output_top_k=100,
        default_refine_top_n=3,
        rrf_constant=60.0,
        enable_dynamic_translation=True,
        translation_device="auto",
    )

    print("\n--- EFFECTIVE CANONICAL RUNTIME PARAMETERS ---", flush=True)
    print(f"• Dynamic Translation Feature      : {config.enable_dynamic_translation} (Helsinki-NLP/opus-mt-vi-en)", flush=True)
    print(f"• Retrieval Variant Policy         : EN_ONLY (Never fuse Vietnamese branch)", flush=True)
    print(f"• Raw-Video Refinement (Phase 4)   : ENABLED on Top {config.default_refine_top_n} candidates", flush=True)
    print(f"• Refinement Config                : window=±{config.refinement_config.window_before_seconds}s, stride={config.refinement_config.coarse_stride_frames}f, radius={config.refinement_config.fine_radius_frames}f", flush=True)

    print("\n--- BOOTSTRAPPING RUNTIME ---", flush=True)
    t_boot = time.time()
    runtime = OperationalKISRuntime.bootstrap(config)
    print(f"Runtime bootstrap completed in {time.time() - t_boot:.2f}s.\n", flush=True)

    print("=" * 150, flush=True)
    print("EXECUTING CANONICAL KIS DEV EVALUATION VIA runtime.handle_query()", flush=True)
    print("=" * 150, flush=True)

    total_numerator = 0
    strict_hits_count = 0
    video_hits_count = 0
    r_at_k_counts = {k: 0 for k in OFFICIAL_K}
    total_exec_time = 0.0
    query_reports = []
    strict_hit_details = []

    for idx, q in enumerate(selected_queries, start=1):
        qid = q["query_id"]
        q_vi = vi_lookup.get(qid, "").strip()
        target_vid = q["video_id"]
        start_f = int(q.get("start_frame", q.get("start_frame_id", 0)))
        end_f = int(q.get("end_frame", q.get("end_frame_id", 0)))

        # Pure Vietnamese query_vi, query_en=None, dynamic translation handles translation + EN_ONLY variant
        req = QueryRequest(
            request_id=f"kis-marian-{qid}",
            query_id=qid,
            query_vi=q_vi,
            query_en=None,
            include_vi_variant=True,
            output_top_k=100,
            refine_top_n=3,
        )

        t_q0 = time.time()
        res = runtime.handle_query(req)
        elapsed = time.time() - t_q0
        total_exec_time += elapsed

        # Load canonical exported JSONL artifact
        top100_rel = res["artifacts"].get("refined_top100_jsonl", res["artifacts"]["top100_jsonl"])
        top100_path = runtime.output_root / top100_rel
        preds = [
            json.loads(line)
            for line in top100_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        # Strict contract assertions
        assert 1 <= len(preds) <= 100, f"Failure on {qid}: Expected 1 <= len(preds) <= 100, got {len(preds)}"
        ranks = [p["rank"] for p in preds]
        assert ranks == list(range(1, len(preds) + 1)), f"Failure on {qid}: Non-contiguous ranks {ranks}"

        # Evaluate Strict Frame Hit & Video Hit
        first_strict_hit_rank = None
        first_video_hit_rank = None
        first_strict_hit_frame = None

        for p in preds:
            v_id = p["video_id"]
            f_id = p["frame_id"]
            rank = p["rank"]

            if v_id == target_vid and first_video_hit_rank is None:
                first_video_hit_rank = rank

            if v_id == target_vid and (start_f <= f_id <= end_f) and first_strict_hit_rank is None:
                first_strict_hit_rank = rank
                first_strict_hit_frame = f_id

        # Calculate Score for this query
        q_score = 0
        if first_strict_hit_rank is not None:
            strict_hits_count += 1
            strict_hit_details.append((qid, first_strict_hit_rank, first_strict_hit_frame, target_vid))
            for k in OFFICIAL_K:
                if first_strict_hit_rank <= k:
                    r_at_k_counts[k] += 1
                    q_score += 1

        if first_video_hit_rank is not None:
            video_hits_count += 1

        total_numerator += q_score

        trans_meta = res.get("diagnostics", {})
        marian_en = res.get("variants", [{}])[0].get("text", "") if "variants" in res else ""

        status_str = f"STRICT HIT @{first_strict_hit_rank} (f={first_strict_hit_frame}) ✅" if first_strict_hit_rank is not None else ("VIDEO HIT ✅" if first_video_hit_rank is not None else "MISS ❌")
        print(f"[{idx:02d}/{len(selected_queries)}] {qid:<8} in {elapsed:5.1f}s | {status_str:<32} | Score: {q_score}/5 | Target: {target_vid} [{start_f}..{end_f}]", flush=True)

    mean_latency = total_exec_time / max(len(selected_queries), 1)
    max_possible_numerator = len(selected_queries) * 5
    macro_score = total_numerator / float(max_possible_numerator)

    print("\n" + "=" * 150, flush=True)
    print(f"CANONICAL MARIAN DEV BENCHMARK SUMMARY ({'5-QUERY SMOKE' if limit == 5 else 'FULL-38 DEV'})", flush=True)
    print("=" * 150, flush=True)
    print(f"• Completed Queries    : {len(selected_queries)} / {len(selected_queries)} (100.0%, 0 exceptions)")
    print(f"• Mean Query Latency   : {mean_latency:.2f} s / query")
    print(f"• Total Execution Time : {total_exec_time:.2f} s ({total_exec_time/60.0:.2f} min)")
    print(f"• Strict Hits List     : {strict_hit_details}\n")

    print(f"{'Metric':<25} | {'Reference b49f628 (Arm-B)':<30} | {'Canonical Dynamic Marian EN_ONLY':<30}")
    print("-" * 95)
    print(f"{'Recall @1':<25} | {0:<30} | {r_at_k_counts[1]:<30}")
    print(f"{'Recall @5':<25} | {0:<30} | {r_at_k_counts[5]:<30}")
    print(f"{'Recall @20':<25} | {2:<30} | {r_at_k_counts[20]:<30}")
    print(f"{'Recall @50':<25} | {5:<30} | {r_at_k_counts[50]:<30}")
    print(f"{'Recall @100':<25} | {5:<30} | {r_at_k_counts[100]:<30}")
    print("-" * 95)
    print(f"{'Numerator / 190':<25} | {'12 / 190':<30} | {f'{total_numerator} / {max_possible_numerator}':<30}")
    print(f"{'Macro Score':<25} | {'0.063158':<30} | {f'{macro_score:.6f}':<30}")
    print(f"{'Strict Hit @100':<25} | {'5 / 38 (13.16%)':<30} | {f'{strict_hits_count} / {len(selected_queries)} ({strict_hits_count/len(selected_queries)*100:.2f}%)':<30}")
    print(f"{'Video Hit @100':<25} | {'34 / 38 (89.47%)':<30} | {f'{video_hits_count} / {len(selected_queries)} ({video_hits_count/len(selected_queries)*100:.2f}%)':<30}")
    print("=" * 150, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Canonical KIS Marian Benchmark")
    parser.add_argument("--smoke", action="store_true", help="Run 5-query parity smoke first")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args, _ = parser.parse_known_args()

    limit = 5 if args.smoke else None
    run_canonical_marian_dev(limit=limit, device=args.device)


if __name__ == "__main__":
    main()
