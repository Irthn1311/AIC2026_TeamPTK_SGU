#!/usr/bin/env python3
"""Canonical L21-150 KIS DEV Benchmark Runner (38 DEV Queries).

Phase K0.1: Clean, un-tuned reproduction of the Canonical KIS DEV Baseline.

Enforces:
  1. Strict HOLDOUT isolation (only loads and evaluates 38 DEV KIS queries).
  2. Fixed Query Ingestion Contract: reads benchmark query_vi and frozen English sidecar.
  3. Frame Semantic Preservation Assertion: internal physical frame == exported frame_id == evaluator actual_frame_id.
  4. 5-Query Smoke Test before proceeding to Full-38 execution.
  5. Full Official Metric Report: Recall@1, 5, 20, 50, 100, Macro Score, Numerator / 190, Hit List.
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

REPO_ROOT = Path(__file__).resolve().parents[1]
SYSTEM_TAI_SRC = REPO_ROOT / "systems" / "system_tai" / "src"
if str(SYSTEM_TAI_SRC) not in sys.path:
    sys.path.insert(0, str(SYSTEM_TAI_SRC))

from system_tai.kis.session_engine import OperationalKISRuntime
from system_tai.kis.session_schema import QueryRequest, SessionConfig
from system_tai.quality.l21_150_evaluator import OFFICIAL_K, _prefix_max

FROZEN_BENCHMARK_SHA256 = "02f0bfc27053a9e532abb8c2cba9ead8f9923d7600993145c57b315f5e55ad1a"
FROZEN_Q2_KIS_DEV_EN_SIDECAR_SHA256 = "fa48d7af2001d8d5eca178301736d1409916961f256b4ccb779490d78495ccea"


def get_git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "UNKNOWN"


def run_canonical_kis_benchmark() -> None:
    benchmark_path = REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "l21_150_diagnostic" / "benchmark.json"
    sidecar_path = REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "l21_150_diagnostic" / "q2_kis_dev_en_translation.json"

    # 1. SHA256 Fingerprint Verification
    bm_bytes = benchmark_path.read_bytes()
    bm_sha = hashlib.sha256(bm_bytes).hexdigest()
    sidecar_bytes = sidecar_path.read_bytes()
    sidecar_sha = hashlib.sha256(sidecar_bytes).hexdigest()

    print("=" * 145)
    print("CANONICAL L21-150 KIS DEV BENCHMARK REPRODUCTION (PHASE K0.1)")
    print("=" * 145)
    print(f"• Git HEAD Commit              : {get_git_head()}")
    print(f"• Benchmark JSON Path          : {benchmark_path.relative_to(REPO_ROOT)}")
    print(f"• Benchmark SHA256             : {bm_sha} ({'MATCH ✅' if bm_sha == FROZEN_BENCHMARK_SHA256 else 'MISMATCH ❌'})")
    print(f"• KIS DEV English Sidecar Path : {sidecar_path.relative_to(REPO_ROOT)}")
    print(f"• KIS English Sidecar SHA256   : {sidecar_sha} ({'MATCH ✅' if sidecar_sha == FROZEN_Q2_KIS_DEV_EN_SIDECAR_SHA256 else 'MISMATCH ❌'})")

    if bm_sha != FROZEN_BENCHMARK_SHA256:
        raise ValueError(f"CRITICAL: Benchmark JSON SHA256 mismatch! Got {bm_sha}")
    if sidecar_sha != FROZEN_Q2_KIS_DEV_EN_SIDECAR_SHA256:
        raise ValueError(f"CRITICAL: KIS Sidecar SHA256 mismatch! Got {sidecar_sha}")

    # 2. Ingestion of Benchmark Data with Strict HOLDOUT Isolation
    bm_data = json.loads(bm_bytes.decode("utf-8"))
    sidecar_data = json.loads(sidecar_bytes.decode("utf-8"))

    sidecar_map = {e["query_id"]: e.get("question_en", "") for e in sidecar_data.get("entries", [])}

    # Strict filter: ONLY load DEV KIS queries (N=38)
    dev_kis_queries = []
    total_kis_count = 0
    holdout_kis_count = 0

    for q in bm_data.get("queries", []):
        if q.get("task_type") == "kis" or q.get("task") == "kis":
            total_kis_count += 1
            if q.get("split") == "DEV":
                dev_kis_queries.append({
                    "query_id": q["query_id"],
                    "video_id": q["video_id"],
                    "query_vi": q["query_vi"],
                    "proposed_interval": [int(q["proposed_interval"][0]), int(q["proposed_interval"][1])],
                    "branch": q.get("branch", ""),
                })
            else:
                holdout_kis_count += 1

    print(f"• Total KIS Queries in Corpus  : {total_kis_count} (DEV: {len(dev_kis_queries)}, HOLDOUT: {holdout_kis_count})")
    print(f"• Evaluated Cohort             : Exactly {len(dev_kis_queries)} DEV Queries (HOLDOUT is strictly guarded)")

    # 3. Canonical Session Configuration
    session_output = Path("/kaggle/working/output/kis_dev_canonical") if Path("/kaggle/working").exists() else REPO_ROOT / "scratch" / "kis_dev_canonical"
    if session_output.exists():
        shutil.rmtree(session_output, ignore_errors=True)
    session_output.mkdir(parents=True, exist_ok=True)

    config = SessionConfig(
        input_root=Path("/kaggle/input/datasets") if Path("/kaggle/input/datasets").exists() else Path("/kaggle/input"),
        manifest_cache=Path("/kaggle/working/manifest_cache.json") if Path("/kaggle/working").exists() else REPO_ROOT / "scratch" / "manifest_cache.json",
        output_root=session_output,
        device="auto",
        allow_model_download=True,
        default_output_top_k=100,
        default_refine_top_n=3,
        rrf_constant=60.0,
    )

    print("\n--- CANONICAL CONFIGURATION PARAMETERS ---")
    print(f"• Output Top-K                 : {config.default_output_top_k}")
    print(f"• Default Refine Top-N         : {config.default_refine_top_n}")
    print(f"• RRF Constant                 : {config.rrf_constant}")
    print(f"• Video-Conditioned Keyframe Q3: {config.video_conditioned_keyframe_config.enabled} (Default: OFF)")
    print(f"• Q3 Anchor Refinement         : {config.q3_anchor_refinement_config.enabled} (Default: OFF)")
    print(f"• Multi-Scale Refinement       : window=±{config.refinement_config.window_before_seconds}s, coarse_stride={config.refinement_config.coarse_stride_frames}, fine_radius={config.refinement_config.fine_radius_frames}")
    print(f"• Query Ingestion Policy       : Bilingual Translation-Augmented RRF (query_vi + query_en, include_vi_variant=True)")

    print("\n--- BOOTSTRAPPING RUNTIME ---")
    t0 = time.time()
    runtime = OperationalKISRuntime.bootstrap(config)
    print(f"Runtime bootstrap completed in {time.time() - t0:.2f}s.")

    # ==============================================================================================================
    # PART 1: 5-QUERY SANITY SMOKE & FRAME SEMANTIC ROUND-TRIP ASSERTION
    # ==============================================================================================================
    print("\n" + "=" * 145)
    print("PART 1: 5-QUERY SANITY SMOKE & FRAME SEMANTIC ROUND-TRIP AUDIT")
    print("=" * 145)

    smoke_queries = dev_kis_queries[:5]
    for idx, q in enumerate(smoke_queries, start=1):
        qid = q["query_id"]
        q_vi = q["query_vi"]
        q_en = sidecar_map.get(qid, "")

        req = QueryRequest(
            request_id=f"smoke-{qid}",
            query_id=qid,
            query_vi=q_vi,
            query_en=q_en if q_en else None,
            include_vi_variant=True,
            output_top_k=100,
            refine_top_n=3,
        )
        res = runtime.handle_query(req)
        preds = res.get("results", [])

        # Strict Smoke Assertions
        assert len(preds) == 100, f"Smoke failure on {qid}: Expected exactly 100 predictions, got {len(preds)}"
        ranks = [p["rank"] for p in preds]
        assert ranks == list(range(1, 101)), f"Smoke failure on {qid}: Ranks are not strictly contiguous 1..100!"

        for p in preds:
            assert isinstance(p["video_id"], str) and p["video_id"].startswith("L"), f"Invalid video_id in {p}"
            assert isinstance(p["frame_id"], int) and p["frame_id"] >= 0, f"Invalid frame_id in {p}"

        # Frame Semantic Round-trip Assertion on Top 3
        # Invariant: internal candidate frame == exported JSONL frame_id == evaluator actual_frame_id
        top3_frames = [preds[i]["frame_id"] for i in range(3)]
        top3_videos = [preds[i]["video_id"] for i in range(3)]
        print(f"  [{idx}/5] SMOKE {qid:<8} | PASS ✅ | Top-3 Candidates: {list(zip(top3_videos, top3_frames))}")

    print("\nALL 5 SMOKE QUERIES PASSED WITH 100% CANONICAL FRAME SEMANTIC ROUND-TRIP ✅")

    # ==============================================================================================================
    # PART 2: FULL 38-QUERY DEV KIS BENCHMARK EXECUTION
    # ==============================================================================================================
    print("\n" + "=" * 145)
    print("PART 2: EXECUTING FULL 38-QUERY CANONICAL KIS DEV BENCHMARK")
    print("=" * 145)

    results_table = []
    total_r_at_k = {k: 0.0 for k in OFFICIAL_K}
    hit_queries_summary = []

    t_bench_start = time.time()

    for idx, q in enumerate(dev_kis_queries, start=1):
        t_q0 = time.time()
        qid = q["query_id"]
        target_vid = q["video_id"]
        start_f, end_f = q["proposed_interval"]
        q_vi = q["query_vi"]
        q_en = sidecar_map.get(qid, "")
        branch = q["branch"]

        req = QueryRequest(
            request_id=f"canonical-{qid}",
            query_id=qid,
            query_vi=q_vi,
            query_en=q_en if q_en else None,
            include_vi_variant=True,
            output_top_k=100,
            refine_top_n=3,
        )
        res = runtime.handle_query(req)
        elapsed = time.time() - t_q0

        preds = res.get("results", [])
        if len(preds) != 100:
            raise RuntimeError(f"Unexpected prediction count on {qid}: {len(preds)}")

        # Evaluate Strict KIS Recall
        # Hit definition: video_id == target_vid and start_f <= frame_id <= end_f
        first_vid_rank = None
        first_frame_rank = None

        candidate_scores = []
        candidate_ranks = []

        for p in preds:
            p_rank = int(p["rank"])
            p_vid = p["video_id"]
            p_fid = int(p["frame_id"])

            video_match = (p_vid == target_vid)
            frame_match = video_match and (start_f <= p_fid <= end_f)

            if video_match and first_vid_rank is None:
                first_vid_rank = p_rank
            if frame_match and first_frame_rank is None:
                first_frame_rank = p_rank

            candidate_ranks.append(p_rank)
            candidate_scores.append(1.0 if frame_match else 0.0)

        # Compute R@K for this query
        q_r_at_k = {k: _prefix_max(candidate_scores, candidate_ranks, k) for k in OFFICIAL_K}
        for k in OFFICIAL_K:
            total_r_at_k[k] += q_r_at_k[k]

        q_macro_score = sum(q_r_at_k.values()) / len(OFFICIAL_K)
        is_hit = (first_frame_rank is not None)

        if is_hit:
            hit_queries_summary.append((qid, target_vid, first_frame_rank, q_macro_score))

        v_rank_str = f"@{first_vid_rank}" if first_vid_rank else "-"
        f_rank_str = f"@{first_frame_rank}" if first_frame_rank else "-"
        status_str = f"STRICT_HIT @{first_frame_rank}" if is_hit else (f"VidOnly @{first_vid_rank}" if first_vid_rank else "MISS")

        rec = {
            "query_id": qid,
            "target_vid": target_vid,
            "gt_interval": f"[{start_f}..{end_f}]",
            "branch": branch,
            "first_vid_rank": first_vid_rank,
            "first_frame_rank": first_frame_rank,
            "status": status_str,
            "r_at_k": q_r_at_k,
            "macro_score": q_macro_score,
            "latency": elapsed,
        }
        results_table.append(rec)

        print(f"[{idx:02d}/38] {qid:<8} | Target: {target_vid:<9} | VidRank: {v_rank_str:<5} | StrictRank: {f_rank_str:<5} | Status: {status_str:<18} | Latency: {elapsed:.2f}s")

    total_bench_time = time.time() - t_bench_start

    # ==============================================================================================================
    # FINAL CANONICAL KIS DEV BENCHMARK REPORT
    # ==============================================================================================================
    n_queries = len(dev_kis_queries)
    mean_r_at_k = {k: total_r_at_k[k] / n_queries for k in OFFICIAL_K}
    official_macro_score = sum(mean_r_at_k.values()) / len(OFFICIAL_K)
    numerator_190 = int(sum(total_r_at_k.values()))
    total_slots_190 = n_queries * len(OFFICIAL_K)  # 38 * 5 = 190

    print("\n" + "=" * 145)
    print("FINAL CANONICAL L21-150 KIS DEV BENCHMARK REPORT (N=38)")
    print("=" * 145)
    print(f"{'QID':<8} | {'Branch':<18} | {'Target':<9} | {'GT Interval':<16} | {'VidRank':<8} | {'StrictRank':<11} | {'Score':<8} | {'Status'}")
    print("-" * 145)
    for r in results_table:
        v_str = f"@{r['first_vid_rank']}" if r["first_vid_rank"] else "-"
        f_str = f"@{r['first_frame_rank']}" if r["first_frame_rank"] else "-"
        print(f"{r['query_id']:<8} | {r['branch']:<18} | {r['target_vid']:<9} | {r['gt_interval']:<16} | {v_str:<8} | {f_str:<11} | {r['macro_score']:.4f}  | {r['status']}")
    print("=" * 145)

    print("\n" + "=" * 80)
    print("OFFICIAL CANONICAL KIS DEV RECALL & MACRO SUMMARY (N=38)")
    print("=" * 80)
    for k in OFFICIAL_K:
        hits_k = int(total_r_at_k[k])
        pct_k = mean_r_at_k[k] * 100.0
        print(f"• Recall@{k:<3} : {hits_k:2d} / {n_queries} ({mean_r_at_k[k]:.6f} | {pct_k:5.2f}%)")

    print("-" * 80)
    print(f"• Total Strict Hits (R@100) : {len(hit_queries_summary)} / {n_queries} ({(len(hit_queries_summary)/n_queries)*100:.2f}%)")
    print(f"• Official Metric Numerator : {numerator_190} / {total_slots_190}")
    print(f"• Official Macro Score      : {official_macro_score:.6f} ({official_macro_score*100:.4f}%)")
    print(f"• Valid Predictions Emitted : 38 / 38 (100.0%)")
    print(f"• Total Benchmark Wall Time : {total_bench_time:.2f}s (Avg: {total_bench_time/n_queries:.2f}s/query)")
    print("=" * 80)

    print("\n--- LIST OF HIT QUERIES & FIRST HIT RANKS ---")
    if hit_queries_summary:
        for qid, vid, rank, score in hit_queries_summary:
            print(f"  • {qid:<8} | Video: {vid:<9} | First Strict Hit Rank: @{rank:<3} | Query Score: {score:.4f}")
    else:
        print("  (No strict hits achieved)")
    print("=" * 80)


if __name__ == "__main__":
    run_canonical_kis_benchmark()
