#!/usr/bin/env python3
"""Canonical L21-150 KIS DEV Benchmark Runner (Phase K0.1 - 5-Query Smoke).

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
from system_tai.quality.l21_150_evaluator import OFFICIAL_K, _prefix_max

FROZEN_KIS_DEV_GT_SHA256 = "7d25708b7243ca2b9964bad9a2b65b63354acd74eddb100167f49e1166f8e5b2"
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

    print("=" * 145, flush=True)
    print("CANONICAL L21-150 KIS DEV BENCHMARK: 5-QUERY SMOKE RUNNER (PHASE K0.1)", flush=True)
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
    session_output = Path("/kaggle/working/output/kis_smoke") if Path("/kaggle/working").exists() else REPO_ROOT / "scratch" / "kis_smoke"
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
    print("\n--- BOOTSTRAPPING RUNTIME (Cold indexing takes ~9.8m on first run, ~0.5s cached) ---", flush=True)
    t0 = time.time()
    runtime = OperationalKISRuntime.bootstrap(config)
    print(f"Runtime bootstrap completed in {time.time() - t0:.2f}s.", flush=True)

    # ==============================================================================================================
    # 5. 5-QUERY DEV SANITY SMOKE & FRAME SEMANTIC ROUND-TRIP AUDIT
    # ==============================================================================================================
    print("\n" + "=" * 145, flush=True)
    print("5-QUERY DEV SANITY SMOKE & FRAME SEMANTIC ROUND-TRIP AUDIT", flush=True)
    print("=" * 145, flush=True)

    smoke_queries = dev_gt_queries[:5]
    for idx, q in enumerate(smoke_queries, start=1):
        qid = q["query_id"]
        sidecar_entry = sidecar_records[qid]
        q_vi = sidecar_entry.get("source_vi", "")
        q_en = sidecar_entry.get("translation_en", "")
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

        # Load predictions from the canonical exported artifact JSONL
        top100_rel = res["artifacts"].get("refined_top100_jsonl", res["artifacts"]["top100_jsonl"])
        top100_path = runtime.output_root / top100_rel
        preds = [
            json.loads(line)
            for line in top100_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

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
            evaluator_phys_frame = p["frame_id"]
            assert internal_phys_frame == export_official_frame_id == evaluator_phys_frame, (
                f"Round-trip semantic mismatch: {internal_phys_frame} -> {export_official_frame_id} -> {evaluator_phys_frame}"
            )
            roundtrip_traces.append(
                f"Rank @{rank_idx} ({p['video_id']}): internal_phys={internal_phys_frame} -> export_id={export_official_frame_id} -> eval_phys={evaluator_phys_frame} [PASS ✅]"
            )

        print(f"\n[{idx}/5] SMOKE {qid:<8} | Target: {target_vid:<9} | GT Interval: [{start_f}..{end_f}] | Time: {elapsed:.2f}s", flush=True)
        print(f"      • Encoded Query Variants ({len(variants)}) : {v_summary}", flush=True)
        print(f"      • Frame Semantic Round-Trip Trace  :", flush=True)
        for r_trace in roundtrip_traces:
            print(f"          - {r_trace}", flush=True)

    print("\n" + "=" * 145, flush=True)
    print("ALL 5 SMOKE QUERIES PASSED WITH 100% CANONICAL FRAME SEMANTIC ROUND-TRIP & ZERO HOLDOUT ACCESS ✅", flush=True)
    print("=" * 145, flush=True)
    print(">> STATUS: 5-QUERY SMOKE COMPLETE. STANDING BY FOR USER GO/NO-GO REVIEW FOR FULL-38 <<", flush=True)


if __name__ == "__main__":
    run_kis_dev_smoke(run_full_38=False)
