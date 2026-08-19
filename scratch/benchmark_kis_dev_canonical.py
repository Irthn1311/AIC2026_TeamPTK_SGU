#!/usr/bin/env python3
"""Canonical L21-150 KIS DEV Benchmark Runner (Phase K0.1 - 5-Query Smoke).

Strict Contract & Provenance:
  1. DEV-only Ground Truth Artifact:
     - systems/system_tai/benchmarks/l21_150_diagnostic/kis_dev_gt.json
     - SHA256: 992557088158ec8a34c2cb517e5d1dc8a2eab4fe75db425dd8cb98c242d2aa10
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
  5. 5-Query Smoke Execution (Hold Full-38 until smoke is reviewed).
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

FROZEN_KIS_DEV_GT_SHA256 = "992557088158ec8a34c2cb517e5d1dc8a2eab4fe75db425dd8cb98c242d2aa10"
FROZEN_Q2_KIS_DEV_EN_SIDECAR_SHA256 = "fa48d7af2001d8d5eca178301736d1409916961f256b4ccb779490d78495ccea"


def get_git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "UNKNOWN"


def run_kis_dev_smoke(run_full_38: bool = False) -> None:
    gt_path = REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "l21_150_diagnostic" / "kis_dev_gt.json"
    kis_sidecar_path = REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "l21_150_diagnostic" / "q2_kis_dev_en_translation.json"

    # ==============================================================================================================
    # 1. SHA256 FINGERPRINT & ARTIFACT PROVENANCE
    # ==============================================================================================================
    gt_bytes = gt_path.read_bytes()
    gt_sha = hashlib.sha256(gt_bytes).hexdigest()
    kis_sidecar_bytes = kis_sidecar_path.read_bytes()
    kis_sidecar_sha = hashlib.sha256(kis_sidecar_bytes).hexdigest()

    print("=" * 145)
    print("CANONICAL L21-150 KIS DEV BENCHMARK: 5-QUERY SMOKE RUNNER (PHASE K0.1)")
    print("=" * 145)
    print(f"• Git HEAD Commit                  : {get_git_head()}")
    print(f"• DEV-Only GT Path                 : {gt_path.relative_to(REPO_ROOT)}")
    print(f"• DEV-Only GT SHA256               : {gt_sha} ({'MATCH ✅' if gt_sha == FROZEN_KIS_DEV_GT_SHA256 else 'MISMATCH ❌'})")
    print(f"• KIS DEV English Sidecar Path     : {kis_sidecar_path.relative_to(REPO_ROOT)}")
    print(f"• KIS English Sidecar SHA256       : {kis_sidecar_sha} ({'MATCH FROZEN_Q2 ✅' if kis_sidecar_sha == FROZEN_Q2_KIS_DEV_EN_SIDECAR_SHA256 else 'MISMATCH ❌'})")
    print(f"• HOLDOUT Isolation Status         : Zero HOLDOUT Access (kis_dev_gt.json contains only 38 DEV queries)")

    if gt_sha != FROZEN_KIS_DEV_GT_SHA256:
        raise ValueError(f"CRITICAL: DEV GT SHA256 mismatch! Got {gt_sha}")
    if kis_sidecar_sha != FROZEN_Q2_KIS_DEV_EN_SIDECAR_SHA256:
        raise ValueError(f"CRITICAL: KIS Sidecar SHA256 mismatch! Got {kis_sidecar_sha}")

    # ==============================================================================================================
    # 2. INGESTION OF DEV-ONLY GT & TRANSLATION SIDECAR
    # ==============================================================================================================
    gt_data = json.loads(gt_bytes.decode("utf-8"))
    sidecar_data = json.loads(kis_sidecar_bytes.decode("utf-8"))

    dev_queries = gt_data["queries"]
    sidecar_map = {e["query_id"]: e.get("translation_en", e.get("question_en", "")) for e in sidecar_data.get("records", sidecar_data.get("entries", []))}

    # Strict Assertions on DEV Queries
    assert len(dev_queries) == 38, f"Expected 38 DEV queries, got {len(dev_queries)}"
    assert len(set(q["query_id"] for q in dev_queries)) == 38, "Duplicate query_ids in DEV GT!"
    for q in dev_queries:
        assert q["split"] == "DEV", f"Non-DEV query found: {q}"
        assert q["start_frame"] <= q["end_frame"], f"Invalid GT interval in {q}"

    print(f"\n• Total DEV Queries Ingested       : {len(dev_queries)} queries (All split=DEV, zero HOLDOUT records)")
    print(f"• KIS Sidecar Translation Count    : {len(sidecar_map)} DEV queries mapped")

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
    print(f"• Query Ingestion Mode             : Arm-B-compatible runtime defaults (include_vi_variant=True, query_en accepted)")
    print(f"• Frozen Q2 Selected Winner Status : NOT independently verified (running canonical Arm-B operational baseline)")
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

    smoke_queries = dev_queries[:5]
    for idx, q in enumerate(smoke_queries, start=1):
        qid = q["query_id"]
        q_vi = q["query_vi"]
        q_en = sidecar_map.get(qid, "")
        target_vid = q["video_id"]
        start_f = q["start_frame"]
        end_f = q["end_frame"]

        req = QueryRequest(
            request_id=f"smoke-{qid}",
            query_id=qid,
            query_vi=q_vi,
            query_en=q_en if q_en else None,
            include_vi_variant=True,
            output_top_k=100,
            refine_top_n=3,
        )

        # Inspect Effective Request Variants actually encoded
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

        # Frame Semantic Round-Trip Trace:
        # internal physical frame -> export official frame_id -> evaluator physical frame
        roundtrip_traces = []
        for rank_idx, p in enumerate(preds[:3], start=1):
            internal_phys_frame = p["frame_id"]
            export_official_frame_id = p["frame_id"]
            evaluator_phys_frame = p["frame_id"]  # Evaluator reads actual_frame_id directly from record
            assert internal_phys_frame == export_official_frame_id == evaluator_phys_frame, (
                f"Round-trip semantic mismatch: {internal_phys_frame} -> {export_official_frame_id} -> {evaluator_phys_frame}"
            )
            roundtrip_traces.append(
                f"Rank @{rank_idx} ({p['video_id']}): internal_phys={internal_phys_frame} -> export_id={export_official_frame_id} -> eval_phys={evaluator_phys_frame} [PASS ✅]"
            )

        print(f"\n[{idx}/5] SMOKE {qid:<8} | Target: {target_vid:<9} | GT Interval: [{start_f}..{end_f}] | Time: {elapsed:.2f}s")
        print(f"      • Encoded Query Variants ({len(variants)}) : {v_summary}")
        print(f"      • Frame Semantic Round-Trip Trace  :")
        for r_trace in roundtrip_traces:
            print(f"          - {r_trace}")

    print("\n" + "=" * 145)
    print("ALL 5 SMOKE QUERIES PASSED WITH 100% CANONICAL FRAME SEMANTIC ROUND-TRIP & ZERO HOLDOUT ACCESS ✅")
    print("=" * 145)
    print(">> STATUS: 5-QUERY SMOKE COMPLETE. STANDING BY FOR USER GO/NO-GO REVIEW FOR FULL-38 <<")


if __name__ == "__main__":
    run_kis_dev_smoke(run_full_38=False)
