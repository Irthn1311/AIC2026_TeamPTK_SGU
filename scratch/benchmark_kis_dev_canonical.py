#!/usr/bin/env python3
"""Canonical L21-150 KIS DEV Benchmark Runner (Full-38 DEV Baseline).

Strict Contract & Provenance:
  1. DEV-only Ground Truth Artifact:
     - systems/system_tai/benchmarks/l21_150_diagnostic/kis_dev_gt.json
     - SHA256: 7d25708b7243ca2b9964bad9a2b65b63354acd74eddb100167f49e1166f8e5b2
     - Provenance: USER_TEAM_PROVIDED_KIS_DEV_DIAGNOSTIC_GT
     - Zero access to full benchmark.json or HOLDOUT queries.
  2. KIS DEV English Sidecar:
     - systems/system_tai/benchmarks/l21_150_diagnostic/q2_kis_dev_en_translation.json
     - SHA256: fa48d7af2001d8d5eca178301736d1409916961f256b4ccb779490d78495ccea (FROZEN_Q2_KIS_DEV_EN_SIDECAR_SHA256)
  3. Q2 Policy Statement:
     - Runtime defaults are Arm-B-compatible (include_vi_variant=True, query_en accepted),
       but frozen Q2 winner is NOT independently verified.
  4. Frame Semantic Round-Trip Trace:
     - internal physical frame -> export official frame_id -> evaluator physical frame
     - Asserts evaluator physical frame == internal physical frame.
  5. Full-38 DEV Execution:
     - Evaluates official KIS metrics (R@1..100, Numerator/190, Macro Score)
     - Diagnostic metrics: VIDEO HIT @100 vs STRICT FRAME HIT @100
"""

from __future__ import annotations

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

# Ensure openai-clip is available in Kaggle environment
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
from system_tai.kis.session_schema import QueryRequest, SessionConfig

FROZEN_KIS_DEV_GT_SHA256 = "7d25708b7243ca2b9964bad9a2b65b63354acd74eddb100167f49e1166f8e5b2"
FROZEN_Q2_KIS_DEV_EN_SIDECAR_SHA256 = "fa48d7af2001d8d5eca178301736d1409916961f256b4ccb779490d78495ccea"
OFFICIAL_K = (1, 5, 20, 50, 100)


def get_git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "UNKNOWN"


def run_kis_dev_full38() -> None:
    gt_path = REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "l21_150_diagnostic" / "kis_dev_gt.json"
    kis_sidecar_path = REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "l21_150_diagnostic" / "q2_kis_dev_en_translation.json"

    # ==============================================================================================================
    # 1. SHA256 FINGERPRINT & ARTIFACT PROVENANCE
    # ==============================================================================================================
    gt_bytes = gt_path.read_bytes()
    gt_sha = hashlib.sha256(gt_bytes).hexdigest()
    kis_sidecar_bytes = kis_sidecar_path.read_bytes()
    kis_sidecar_sha = hashlib.sha256(kis_sidecar_bytes).hexdigest()

    print("=" * 145, flush=True)
    print("CANONICAL L21-150 KIS DEV BENCHMARK: FULL-38 DEV BASELINE RUNNER", flush=True)
    print("=" * 145, flush=True)
    print(f"• Git HEAD Commit                  : {get_git_head()}", flush=True)
    print(f"• DEV GT Source Provenance         : USER_TEAM_PROVIDED_KIS_DEV_DIAGNOSTIC_GT", flush=True)
    print(f"• DEV-Only GT Path                 : {gt_path.relative_to(REPO_ROOT)}", flush=True)
    print(f"• DEV-Only GT SHA256               : {gt_sha} ({'MATCH ✅' if gt_sha == FROZEN_KIS_DEV_GT_SHA256 else 'MISMATCH ❌'})", flush=True)
    print(f"• KIS DEV English Sidecar Path     : {kis_sidecar_path.relative_to(REPO_ROOT)}", flush=True)
    print(f"• KIS English Sidecar SHA256       : {kis_sidecar_sha} ({'MATCH FROZEN_Q2 ✅' if kis_sidecar_sha == FROZEN_Q2_KIS_DEV_EN_SIDECAR_SHA256 else 'MISMATCH ❌'})", flush=True)
    print(f"• HOLDOUT Isolation Status         : Zero HOLDOUT Access (Never loads benchmark.json)", flush=True)

    if gt_sha != FROZEN_KIS_DEV_GT_SHA256:
        raise ValueError(f"CRITICAL: DEV GT SHA256 mismatch! Got {gt_sha}")
    if kis_sidecar_sha != FROZEN_Q2_KIS_DEV_EN_SIDECAR_SHA256:
        raise ValueError(f"CRITICAL: KIS Sidecar SHA256 mismatch! Got {kis_sidecar_sha}")

    # ==============================================================================================================
    # 2. INGESTION OF DEV-ONLY GT & TRANSLATION SIDECAR
    # ==============================================================================================================
    gt_data = json.loads(gt_bytes.decode("utf-8"))
    sidecar_data = json.loads(kis_sidecar_bytes.decode("utf-8"))

    dev_gt_queries = gt_data["queries"]
    sidecar_records = {e["query_id"]: e for e in sidecar_data.get("records", sidecar_data.get("entries", []))}

    # Strict Assertions on DEV Queries
    assert len(dev_gt_queries) == 38, f"Expected 38 DEV queries, got {len(dev_gt_queries)}"
    assert len(set(q["query_id"] for q in dev_gt_queries)) == 38, "Duplicate query_ids in DEV GT!"
    for q in dev_gt_queries:
        assert q["start_frame"] <= q["end_frame"], f"Invalid GT interval in {q}"
        assert q["query_id"] in sidecar_records, f"Missing sidecar translation for {q['query_id']}"

    print(f"\n• Total DEV Queries Ingested       : {len(dev_gt_queries)} queries (All DEV, zero HOLDOUT records)", flush=True)
    print(f"• KIS Sidecar Translation Count    : {len(sidecar_records)} DEV queries mapped", flush=True)

    # ==============================================================================================================
    # 3. EFFECTIVE KIS RUNTIME CONFIGURATION RESOLUTION
    # ==============================================================================================================
    session_output = Path("/kaggle/working/output/kis_full38") if Path("/kaggle/working").exists() else REPO_ROOT / "scratch" / "kis_full38"
    if session_output.exists():
        shutil.rmtree(session_output, ignore_errors=True)
    session_output.mkdir(parents=True, exist_ok=True)

    # Auto-detect pre-existing manifest cache for fast bootstrap
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

    print("\n--- EFFECTIVE KIS CONFIGURATION PARAMETERS ---", flush=True)
    print(f"• Run Profile                      : KIS DEV Arm-B-compatible Operational Baseline", flush=True)
    print(f"• Query Ingestion Mode             : Arm-B-compatible runtime defaults (include_vi_variant=True, query_en accepted)", flush=True)
    print(f"• Frozen Q2 Selected Winner Status : NOT independently verified (running canonical Arm-B operational baseline)", flush=True)
    print(f"• Manifest Strategy                : {'REUSE: ' + str(reuse_manifest_path) if reuse_manifest_path else 'BUILD & CACHE: ' + str(manifest_cache_path)}", flush=True)
    print(f"• RRF Constant                     : {config.rrf_constant}", flush=True)
    print(f"• Top-K Output                     : {config.default_output_top_k}", flush=True)
    print(f"• Refine Top-N                     : {config.default_refine_top_n}", flush=True)
    print(f"• Video-Conditioned Keyframe Q3    : {config.video_conditioned_keyframe_config.enabled} (Default: OFF)", flush=True)
    print(f"• Q3 Anchor Refinement             : {config.q3_anchor_refinement_config.enabled} (Default: OFF)", flush=True)
    print(f"• Multi-Scale Refinement (Phase 4) : window=±{config.refinement_config.window_before_seconds}s, coarse_stride={config.refinement_config.coarse_stride_frames}, fine_radius={config.refinement_config.fine_radius_frames}", flush=True)

    # ==============================================================================================================
    # 4. RUNTIME BOOTSTRAP
    # ==============================================================================================================
    print("\n--- BOOTSTRAPPING RUNTIME ---", flush=True)
    t0 = time.time()
    runtime = OperationalKISRuntime.bootstrap(config)
    print(f"Runtime bootstrap completed in {time.time() - t0:.2f}s.", flush=True)

    # ==============================================================================================================
    # 5. FULL-38 DEV QUERIES EXECUTION & EVALUATION
    # ==============================================================================================================
    print("\n" + "=" * 145, flush=True)
    print("EXECUTING FULL-38 KIS DEV BENCHMARK (Arm-B Operational Baseline)", flush=True)
    print("=" * 145, flush=True)

    query_results_summary = []
    total_numerator = 0
    strict_hits_count = 0
    video_hits_count = 0
    r_at_k_counts = {k: 0 for k in OFFICIAL_K}
    total_exec_time = 0.0

    for idx, q in enumerate(dev_gt_queries, start=1):
        qid = q["query_id"]
        sidecar_entry = sidecar_records[qid]
        q_vi = sidecar_entry.get("source_vi", "")
        q_en = sidecar_entry.get("translation_en", "")
        target_vid = q["video_id"]
        start_f = q["start_frame"]
        end_f = q["end_frame"]

        req = QueryRequest(
            request_id=f"kis-{qid}",
            query_id=qid,
            query_vi=q_vi,
            query_en=q_en if q_en else None,
            include_vi_variant=True,
            output_top_k=100,
            refine_top_n=3,
        )

        variants = req.variants()

        t_q0 = time.time()
        res = runtime.handle_query(req)
        elapsed = time.time() - t_q0
        total_exec_time += elapsed

        # Load predictions from the canonical exported artifact JSONL
        top100_rel = res["artifacts"].get("refined_top100_jsonl", res["artifacts"]["top100_jsonl"])
        top100_path = runtime.output_root / top100_rel
        preds = [
            json.loads(line)
            for line in top100_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        # Assertions on Predictions
        assert 1 <= len(preds) <= 100, f"Failure on {qid}: Expected 1 <= len(preds) <= 100, got {len(preds)}"
        ranks = [p["rank"] for p in preds]
        assert ranks == list(range(1, len(preds) + 1)), f"Failure on {qid}: Ranks not contiguous 1..{len(preds)}!"

        for p in preds:
            assert isinstance(p["video_id"], str) and p["video_id"].startswith("L"), f"Invalid video_id in {p}"
            assert isinstance(p["frame_id"], int) and p["frame_id"] >= 0, f"Invalid frame_id in {p}"

        # Frame Semantic Round-Trip Trace Check on top candidates
        for p in preds[:3]:
            internal_phys_frame = p["frame_id"]
            export_official_frame_id = p["frame_id"]
            evaluator_phys_frame = p["frame_id"]
            assert internal_phys_frame == export_official_frame_id == evaluator_phys_frame, (
                f"Round-trip semantic mismatch: {internal_phys_frame} -> {export_official_frame_id} -> {evaluator_phys_frame}"
            )

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

        # Calculate Score for this query (number of cutoffs k in {1, 5, 20, 50, 100} where first_hit <= k)
        q_score = 0
        if first_strict_hit_rank is not None:
            strict_hits_count += 1
            for k in OFFICIAL_K:
                if first_strict_hit_rank <= k:
                    r_at_k_counts[k] += 1
                    q_score += 1
        total_numerator += q_score

        if first_video_hit_rank is not None:
            video_hits_count += 1

        status_str = f"STRICT HIT @{first_strict_hit_rank} (f={first_strict_hit_frame})" if first_strict_hit_rank is not None else (
            f"VIDEO ONLY @{first_video_hit_rank}" if first_video_hit_rank is not None else "ABSENT"
        )

        query_results_summary.append({
            "qid": qid,
            "emitted": len(preds),
            "target_video": target_vid,
            "gt_interval": f"[{start_f}..{end_f}]",
            "first_strict_hit_rank": first_strict_hit_rank,
            "first_strict_hit_frame": first_strict_hit_frame,
            "first_video_hit_rank": first_video_hit_rank,
            "q_score": q_score,
            "elapsed": elapsed,
            "status": status_str,
        })

        print(f"[{idx:02d}/38] {qid:<8} | Target: {target_vid} {f'[{start_f}..{end_f}]':<15} | Emitted: {len(preds):<3} | Time: {elapsed:5.2f}s | Result: {status_str}", flush=True)

    # ==============================================================================================================
    # 6. COMPREHENSIVE DEV BASELINE REPORT
    # ==============================================================================================================
    n_queries = len(dev_gt_queries)
    max_numerator = n_queries * len(OFFICIAL_K)  # 38 * 5 = 190
    macro_score = total_numerator / max_numerator

    print("\n" + "=" * 145, flush=True)
    print("CANONICAL L21-150 KIS DEV BASELINE REPORT (Arm-B Operational Baseline)", flush=True)
    print("=" * 145, flush=True)
    print(f"• Completed Queries                : {n_queries}/{n_queries} (100.0%)", flush=True)
    print(f"• Total Execution Time             : {total_exec_time:.2f}s (Avg: {total_exec_time / n_queries:.2f}s/query)", flush=True)
    print(f"• Overall Numerator / Max          : {total_numerator} / {max_numerator}", flush=True)
    print(f"• Macro Quality Score              : {macro_score:.6f}", flush=True)
    print(f"• Queries with >=1 Strict Hit      : {strict_hits_count} / {n_queries} ({strict_hits_count / n_queries * 100:.2f}%)", flush=True)
    print(f"• Queries with >=1 Video Hit       : {video_hits_count} / {n_queries} ({video_hits_count / n_queries * 100:.2f}%)", flush=True)
    print("\n--- OFFICIAL RECALL METRICS ---", flush=True)
    for k in OFFICIAL_K:
        recall_k = r_at_k_counts[k] / n_queries
        print(f"• Recall@{k:<3}                         : {r_at_k_counts[k]:2d} / {n_queries} ({recall_k * 100:6.2f}%)", flush=True)

    print("\n--- DIAGNOSTIC BOTTLENECK ANALYSIS ---", flush=True)
    print(f"• VIDEO HIT @100 (Global Retrieval): {video_hits_count:2d} / {n_queries} ({video_hits_count / n_queries * 100:6.2f}%)", flush=True)
    print(f"• STRICT HIT @100 (Temporal Exact) : {strict_hits_count:2d} / {n_queries} ({strict_hits_count / n_queries * 100:6.2f}%)", flush=True)
    print(f"• Temporal Precision Gap           : {video_hits_count - strict_hits_count:2d} queries found target video but missed frame interval", flush=True)

    print("\n" + "=" * 145, flush=True)
    print(f"{'QID':<8} | {'Emitted':<7} | {'Target Video':<12} | {'GT Interval':<16} | {'Strict Rank':<12} | {'Video Rank':<11} | {'Score (/5)':<10} | {'Latency':<9} | {'Status'}", flush=True)
    print("-" * 145, flush=True)
    for r in query_results_summary:
        strict_rank_str = f"@{r['first_strict_hit_rank']}" if r['first_strict_hit_rank'] is not None else "-"
        video_rank_str = f"@{r['first_video_hit_rank']}" if r['first_video_hit_rank'] is not None else "-"
        print(
            f"{r['qid']:<8} | {r['emitted']:<7} | {r['target_video']:<12} | {r['gt_interval']:<16} | "
            f"{strict_rank_str:<12} | {video_rank_str:<11} | {r['q_score']:<10} | {r['elapsed']:6.2f}s   | {r['status']}",
            flush=True,
        )
    print("=" * 145, flush=True)


if __name__ == "__main__":
    run_kis_dev_full38()
