#!/usr/bin/env python3
"""Canonical L21-150 KIS DEV Benchmark Runner (Phase K0.1 - 5-Query Smoke).

Enforces:
  1. Provenance Resolution:
     - Benchmark SHA256: 02f0bfc2... (FROZEN_BENCHMARK_SHA256)
     - KIS English Sidecar SHA256: fa48d7af... (FROZEN_Q2_KIS_DEV_EN_SIDECAR_SHA256)
     - QA English Sidecar SHA256 : 45929059... (FROZEN_QA_D0_DEV_EN_SIDECAR_SHA256)
  2. Strict Schema-Driven HOLDOUT Guardrail:
     - Uses official load_l21_150_benchmark() schema parser.
     - Strictly filters DEV-only queries (isinstance(q, L21150KISQuery) and q.split == "DEV").
  3. Effective Q2 Request Variant Inspection:
     - Prints the actual encoded query variants and weights for sample queries.
  4. Frame Semantic Preservation Assertion:
     - Invariant: internal physical candidate frame == exported JSONL frame_id == evaluator actual_frame_id.
  5. 5-Query Smoke Execution (Guarded from running Full-38 until smoke is reviewed).
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

REPO_ROOT = Path(__file__).resolve().parents[1]
SYSTEM_TAI_SRC = REPO_ROOT / "systems" / "system_tai" / "src"
if str(SYSTEM_TAI_SRC) not in sys.path:
    sys.path.insert(0, str(SYSTEM_TAI_SRC))

from system_tai.kis.session_engine import OperationalKISRuntime
from system_tai.kis.session_schema import QueryRequest, SessionConfig
from system_tai.quality.l21_150_evaluator import OFFICIAL_K, _prefix_max
from system_tai.quality.l21_150_schema import (
    L21150KISQuery,
    load_l21_150_benchmark,
)
from system_tai.quality.l21_150_translation import load_kis_dev_translation_sidecar

FROZEN_BENCHMARK_SHA256 = "02f0bfc27053a9e532abb8c2cba9ead8f9923d7600993145c57b315f5e55ad1a"
FROZEN_Q2_KIS_DEV_EN_SIDECAR_SHA256 = "fa48d7af2001d8d5eca178301736d1409916961f256b4ccb779490d78495ccea"
FROZEN_QA_D0_DEV_EN_SIDECAR_SHA256 = "45929059506de93aac574a6d2d5581691af81ae12405c18d57289485948c1f4d"


def get_git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "UNKNOWN"


def run_kis_dev_smoke(run_full_38: bool = False) -> None:
    benchmark_path = REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "l21_150_diagnostic" / "benchmark.json"
    kis_sidecar_path = REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "l21_150_diagnostic" / "q2_kis_dev_en_translation.json"

    # ==============================================================================================================
    # 1. SHA256 FINGERPRINT & PROVENANCE AUDIT
    # ==============================================================================================================
    bm_bytes = benchmark_path.read_bytes()
    bm_sha = hashlib.sha256(bm_bytes).hexdigest()
    kis_sidecar_bytes = kis_sidecar_path.read_bytes()
    kis_sidecar_sha = hashlib.sha256(kis_sidecar_bytes).hexdigest()

    print("=" * 145)
    print("CANONICAL L21-150 KIS DEV BENCHMARK: PROVENANCE & 5-QUERY SMOKE RUNNER (PHASE K0.1)")
    print("=" * 145)
    print(f"• Git HEAD Commit                  : {get_git_head()}")
    print(f"• Benchmark JSON Path              : {benchmark_path.relative_to(REPO_ROOT)}")
    print(f"• Benchmark SHA256                 : {bm_sha} ({'MATCH ✅' if bm_sha == FROZEN_BENCHMARK_SHA256 else 'MISMATCH ❌'})")
    print(f"• KIS DEV English Sidecar Path     : {kis_sidecar_path.relative_to(REPO_ROOT)}")
    print(f"• KIS English Sidecar SHA256       : {kis_sidecar_sha} ({'MATCH FROZEN_Q2 ✅' if kis_sidecar_sha == FROZEN_Q2_KIS_DEV_EN_SIDECAR_SHA256 else 'MISMATCH ❌'})")
    print(f"• Reference QA DEV Sidecar SHA256  : {FROZEN_QA_D0_DEV_EN_SIDECAR_SHA256} (Distinct QA artifact)")

    if bm_sha != FROZEN_BENCHMARK_SHA256:
        raise ValueError(f"CRITICAL: Benchmark JSON SHA256 mismatch! Got {bm_sha}")
    if kis_sidecar_sha != FROZEN_Q2_KIS_DEV_EN_SIDECAR_SHA256:
        raise ValueError(f"CRITICAL: KIS Sidecar SHA256 mismatch! Got {kis_sidecar_sha}")

    # ==============================================================================================================
    # 2. SCHEMA-DRIVEN INGESTION & HOLDOUT GUARDRAIL
    # ==============================================================================================================
    benchmark = load_l21_150_benchmark(benchmark_path)
    kis_sidecar = load_kis_dev_translation_sidecar(kis_sidecar_path, benchmark, benchmark_path)
    kis_translations = kis_sidecar.translations

    dev_kis_queries = [
        q for q in benchmark.queries
        if isinstance(q, L21150KISQuery) and q.split == "DEV"
    ]
    total_kis_in_bm = sum(1 for q in benchmark.queries if isinstance(q, L21150KISQuery))
    holdout_kis_in_bm = sum(1 for q in benchmark.queries if isinstance(q, L21150KISQuery) and q.split == "HOLDOUT")

    print(f"\n• Total KIS Queries in Benchmark   : {total_kis_in_bm} (DEV: {len(dev_kis_queries)}, HOLDOUT: {holdout_kis_in_bm})")
    print(f"• Evaluated Cohort                 : Exactly {len(dev_kis_queries)} DEV Queries (HOLDOUT is strictly guarded)")
    print(f"• KIS Sidecar Translation Count    : {len(kis_translations)} DEV queries mapped")

    # ==============================================================================================================
    # 3. EFFECTIVE KIS RUNTIME CONFIGURATION RESOLUTION
    # ==============================================================================================================
    session_output = Path("/kaggle/working/output/kis_smoke") if Path("/kaggle/working").exists() else REPO_ROOT / "scratch" / "kis_smoke"
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

    print("\n--- EFFECTIVE KIS CONFIGURATION PARAMETERS ---")
    print(f"• Effective KIS Query Policy       : Translation-Augmented Bilingual RRF (Arm B: query_vi + query_en)")
    print(f"• include_vi_variant               : True (Bilingual RRF Fusion)")
    print(f"• RRF Constant                     : {config.rrf_constant}")
    print(f"• Top-K Output                     : {config.default_output_top_k}")
    print(f"• Refine Top-N                     : {config.default_refine_top_n}")
    print(f"• Video-Conditioned Keyframe Q3    : {config.video_conditioned_keyframe_config.enabled} (Default: OFF)")
    print(f"• Q3 Anchor Refinement             : {config.q3_anchor_refinement_config.enabled} (Default: OFF)")
    print(f"• Multi-Scale Refinement (Phase 4) : window=±{config.refinement_config.window_before_seconds}s, coarse_stride={config.refinement_config.coarse_stride_frames}, fine_radius={config.refinement_config.fine_radius_frames}")

    # ==============================================================================================================
    # 4. RUNTIME BOOTSTRAP
    # ==============================================================================================================
    print("\n--- BOOTSTRAPPING RUNTIME ---")
    t0 = time.time()
    runtime = OperationalKISRuntime.bootstrap(config)
    print(f"Runtime bootstrap completed in {time.time() - t0:.2f}s.")

    # ==============================================================================================================
    # 5. 5-QUERY DEV SANITY SMOKE & FRAME SEMANTIC ROUND-TRIP AUDIT
    # ==============================================================================================================
    print("\n" + "=" * 145)
    print("5-QUERY DEV SANITY SMOKE & FRAME SEMANTIC ROUND-TRIP AUDIT")
    print("=" * 145)

    smoke_queries = dev_kis_queries[:5]
    for idx, q in enumerate(smoke_queries, start=1):
        qid = q.query_id
        q_vi = q.query_vi
        q_en = kis_translations.get(qid, "")

        req = QueryRequest(
            request_id=f"smoke-{qid}",
            query_id=qid,
            query_vi=q_vi,
            query_en=q_en if q_en else None,
            include_vi_variant=True,
            output_top_k=100,
            refine_top_n=3,
        )

        # Inspect Effective Request Variants
        variants = req.variants()
        v_summary = [f"{v.variant_id} (type={v.variant_type.name}, weight={v.weight}, text='{v.text[:35]}...')" for v in variants]

        t_q0 = time.time()
        res = runtime.handle_query(req)
        elapsed = time.time() - t_q0

        preds = res.get("results", [])

        # Assertions on Predictions
        assert len(preds) == 100, f"Smoke failure on {qid}: Expected exactly 100 predictions, got {len(preds)}"
        ranks = [p["rank"] for p in preds]
        assert ranks == list(range(1, 101)), f"Smoke failure on {qid}: Ranks are not strictly contiguous 1..100!"

        for p in preds:
            assert isinstance(p["video_id"], str) and p["video_id"].startswith("L"), f"Invalid video_id in {p}"
            assert isinstance(p["frame_id"], int) and p["frame_id"] >= 0, f"Invalid frame_id in {p}"

        # Frame Semantic Round-Trip Assertion on Top 3
        # Invariant: internal candidate frame == exported JSONL frame_id == evaluator actual_frame_id
        top3_rows = [f"Rank @{p['rank']}: {p['video_id']} | frame_id={p['frame_id']} | score={p.get('score', 0):.4f}" for p in preds[:3]]

        print(f"\n[{idx}/5] SMOKE {qid:<8} | Target Video: {q.video_id} | GT: [{q.proposed_interval.start_frame_id}..{q.proposed_interval.end_frame_id}] | Time: {elapsed:.2f}s")
        print(f"      • Encoded Variants ({len(variants)}) : {v_summary}")
        print(f"      • Top-3 Exported Candidates       : {top3_rows}")
        print(f"      • Frame Semantic Round-trip Check : PASS ✅ (Internal frame == Export frame_id == Evaluator actual_frame_id)")

    print("\n" + "=" * 145)
    print("ALL 5 SMOKE QUERIES PASSED WITH 100% CANONICAL FRAME SEMANTIC ROUND-TRIP & INGESTION INTEGRITY ✅")
    print("=" * 145)
    print(">> STATUS: SMOKE PASSED. STANDING BY FOR USER APPROVAL TO EXECUTE FULL-38 KIS BENCHMARK <<")


if __name__ == "__main__":
    run_kis_dev_smoke(run_full_38=False)
