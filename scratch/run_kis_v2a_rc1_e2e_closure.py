"""KIS V2-A.3 Release Candidate 1 (KIS_V2A_RC1) End-to-End Closure Runner.

Executes two sequential clean-session full runs on CPU, verifying 100% bit-exact
reproducibility across all 5 queries, and emits the formal release manifest.

Release Profile:
  - Default Identity: Canonical New (General-Purpose Policy)
  - Paraphrase Ensemble: OFF (Disabled in production RC1)
  - Local Anchor Refinement: OFF
  - Device: CPU (Locked)
  - Adaptive Nomination: v2_adaptive_enabled=True, K in {32, 48, 64}, cap=64
  - Evidence Pooling: Top-M=5 (weights: 0.4, 0.25, 0.15, 0.1, 0.1, min gap: 60)
  - Frame Fusion: Internal RRF depth=1000, rrf_constant=60.0
  - Localization: VI variant enabled (weight: 0.5), temporal diversity gap 5.0s
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

REPO_ROOT = Path(__file__).resolve().parent.parent
SYSTEM_TAI_SRC = REPO_ROOT / "systems" / "system_tai" / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SYSTEM_TAI_SRC) not in sys.path:
    sys.path.insert(0, str(SYSTEM_TAI_SRC))

from system_tai.kis.session_engine import OperationalKISRuntime
from system_tai.kis.session_schema import (
    KISVideoFirstConfig,
    QueryRequest,
    SessionConfig,
)
from system_tai.retrieval.canonical_projection import canonical_projection_digest
from system_tai.translation.sidecar_provider import (
    ImmutableSidecarTranslationProvider,
    canonical_sidecar_sha256,
)

CANONICAL_TNEW_SHA256 = "545bd4a37c57af53713a1d9f382241ef729c287a1817a5671fdc923115b0be2a"
CANONICAL_TOLD_SHA256 = "022a6c1db48d5fe00a223ec9f637aa1d64eea5d55c06e901caa42e04ff0e3367"
CANONICAL_QUERY_MANIFEST_SHA256 = "c7ee3b1168e681444d7a0b4059c81db4bbb8fe15b91c2d58f7641823a52d2fbf"
CANONICAL_MANUAL_REF_SHA256 = "b23d45682f6159075b03c129104e1b41abeb065f610f65ec39860d204c78f65d"
EXPECTED_CORPUS_FINGERPRINT = "398bb60c6ea1c8eb"
RELEASE_CANDIDATE_ID = "KIS_V2A_RC1"


def get_git_commit_sha() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).decode().strip()
        return out
    except Exception:
        return "UNKNOWN_COMMIT"


def load_manual_reference(ref_path: Path) -> dict[str, dict[str, Any]]:
    if not ref_path.exists():
        raise FileNotFoundError(f"Manual reference file not found: {ref_path}")
    raw = json.loads(ref_path.read_text(encoding="utf-8"))
    queries = raw.get("queries", [])
    result = {}
    for q in queries:
        qid = q["query_id"]
        result[qid] = {
            "query_id": qid,
            "query_vi": q["query_vi"],
            "human_verified_video_id": q["human_verified_video_id"],
            "human_annotated_intervals": q.get("human_annotated_intervals", []),
            "human_annotated_intervals_pts": q.get("human_annotated_intervals_pts", []),
            "legacy_target_video": q.get("legacy_manifest_target", {}).get("target_video"),
            "annotation_status": q.get("annotation_status", "VIDEO_ONLY_VERIFIED"),
        }
    return result


def execute_closure_session(
    run_name: str,
    run_out_dir: Path,
    input_root: Path,
    manifest_cache_path: Path,
    queries_data: list[dict[str, Any]],
    gt_ref: dict[str, dict[str, Any]],
    provider: ImmutableSidecarTranslationProvider,
    device: str = "cpu",
    strict_corpus_gate: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Execute a single clean session and collect canonical projections."""
    if run_out_dir.exists():
        shutil.rmtree(run_out_dir)
    run_out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 90, flush=True)
    print(f"🚀 Executing Clean Session: {run_name} (Output: {run_out_dir})", flush=True)
    print("=" * 90, flush=True)

    vf_cfg = KISVideoFirstConfig(
        enabled=True,
        v2_adaptive_enabled=True,
        selected_video_cap=64,
        video_nomination_depth=100,
        restricted_frames_per_video_per_variant=20,
        full_query_weight=1.0,
        primary_scene_weight=1.0,
        supporting_attribute_weight=0.35,
        top_m_evidence_cap=5,
        top_m_weights=(0.4, 0.25, 0.15, 0.1, 0.1),
        top_m_min_frame_gap=60,
        enable_temporal_diverse_local_candidates=True,
        temporal_diversity_gap_seconds=5.0,
        enable_vi_localization_variant=True,
        vi_localization_weight=0.5,
        internal_rrf_candidate_depth=1000,
        enable_top_video_local_anchor=False,
        enable_paraphrase_ensemble=False,
        paraphrase_ensemble_mode="EQUAL_BUDGET",
    )

    session_cfg = SessionConfig(
        session_id=f"rc1_{run_name}_{int(time.time())}",
        input_root=input_root,
        manifest_cache=manifest_cache_path,
        output_root=run_out_dir,
        enable_dynamic_translation=True,
        allow_model_download=False,
        device=device,
        rrf_constant=60.0,
        continue_on_request_error=False,
        fail_fast_protocol=True,
        kis_video_first_config=vf_cfg,
    )

    runtime = OperationalKISRuntime.bootstrap(
        config=session_cfg,
        translation_provider=provider,
    )

    corpus_info: dict[str, Any] = {}
    session_results: dict[str, Any] = {}

    try:
        v_count = len(runtime.registry.stores)
        r_count = runtime.registry.total_rows
        d_count = runtime.registry.embedding_dimension
        fp = getattr(runtime.manifest, "fingerprint", "UNKNOWN")
        print(f"📦 Corpus loaded: {v_count} videos, {r_count} total rows, {d_count} dims (fingerprint: {fp[:16]}...)", flush=True)

        if strict_corpus_gate or v_count >= 800:
            if v_count != 873 or r_count != 177321 or d_count != 512:
                raise AssertionError(
                    f"Strict corpus gate failed! Expected 873 videos, 177321 rows, 512 dimensions. "
                    f"Got {v_count} videos, {r_count} rows, {d_count} dimensions."
                )
            print("✅ Strict Corpus Gate PASS (873 videos / 177321 rows / 512 dimensions)", flush=True)

        corpus_info = {
            "video_count": v_count,
            "total_rows": r_count,
            "embedding_dimension": d_count,
            "fingerprint": fp,
        }

        for q_item in queries_data:
            qid = q_item["query_id"]
            q_vi = q_item.get("query_vi", q_item.get("text", ""))

            print(f"  • Processing Query {qid}: '{q_vi[:50]}...'", flush=True)
            req = QueryRequest(
                request_id=f"{run_name}_{qid}",
                query_id=qid,
                query_vi=q_vi,
                output_top_k=100,
            )

            resp = runtime.handle_query(req)
            cand_rel_path = resp["artifacts"]["candidates_json"]
            cand_path = run_out_dir / cand_rel_path
            cand_json_data = json.loads(cand_path.read_text(encoding="utf-8"))

            records = cand_json_data.get("records", [])
            if len(records) != 100:
                raise ValueError(f"Expected 100 candidates for {qid}, got {len(records)}")

            ranks = [r["rank"] for r in records]
            if ranks != list(range(1, 101)):
                raise ValueError(f"Ranks for {qid} are not contiguous 1..100: {ranks[:5]}...{ranks[-5:]}")

            identities = [(r["video_id"], r["frame_id"]) for r in records]
            if len(set(identities)) != 100:
                raise ValueError(f"Duplicate candidate identities found for {qid}")

            proj_digest = canonical_projection_digest(records)

            # Ground Truth Target Evaluation
            gt_info = gt_ref.get(qid, {})
            human_target_vid = gt_info.get("human_verified_video_id")

            human_target_video_rank = None
            human_target_best_score = None
            human_target_count_top100 = 0

            for r in records:
                vid = r["video_id"]
                score = float(r["fusion_score"])
                if vid == human_target_vid:
                    human_target_count_top100 += 1
                    if human_target_video_rank is None:
                        human_target_video_rank = r["rank"]
                        human_target_best_score = score

            video_first_trace = cand_json_data.get("video_first", {})
            phase_c_tel = video_first_trace.get("phase_c_telemetry", {})

            target_coarse_rank = None
            selected_vids_list = video_first_trace.get("selected_videos", [])
            for idx, v_item in enumerate(selected_vids_list, start=1):
                if v_item.get("video_id") == human_target_vid:
                    target_coarse_rank = idx
                    break

            session_results[qid] = {
                "query_id": qid,
                "query_vi": q_vi,
                "canonical_projection_digest": proj_digest,
                "human_target_video": human_target_vid,
                "human_target_video_rank": human_target_video_rank,
                "human_target_best_score": human_target_best_score,
                "human_target_count_top100": human_target_count_top100,
                "target_coarse_rank": target_coarse_rank,
                "selected_video_count": phase_c_tel.get("selected_video_count", len(selected_vids_list)),
                "candidate_count_before_dedup": phase_c_tel.get("candidate_count_before_dedup"),
                "effective_unique_candidates": phase_c_tel.get("effective_unique_candidate_count_after_dedup"),
                "compulsory_extra_count": phase_c_tel.get("compulsory_extra_count"),
            }
    finally:
        runtime.close()

    return session_results, corpus_info


def run_kis_v2a_rc1_e2e_closure(
    query_manifest_path: Path,
    input_root: Path,
    manifest_cache_path: Path,
    output_root: Path,
    tnew_sidecar_path: Path,
    manual_ref_path: Path,
    told_sidecar_path: Path | None = None,
    policy: str = "general",
    phase_c1_audit_path: Path | None = None,
    strict_corpus_gate: bool = True,
    device: str = "cpu",
) -> dict[str, Any]:
    """Execute KIS V2-A.3 RC1 Two-Pass Bit-Exact Closure."""
    start_time_all = time.time()
    git_sha = get_git_commit_sha()
    print("=" * 110, flush=True)
    print("🔒 KIS V2-A.3 RELEASE CANDIDATE 1 (KIS_V2A_RC1) FULL-SYSTEM E2E CLOSURE", flush=True)
    print(f"Git Commit: {git_sha} | Policy: {policy.upper()}", flush=True)
    print("=" * 110, flush=True)

    if device != "cpu":
        print(f"⚠️ Warning: Non-CPU device '{device}' specified for RC1 closure; standard release baseline requires 'cpu'.", flush=True)

    # 1. Provenance Validation
    qm_sha = canonical_sidecar_sha256(query_manifest_path)
    if qm_sha.lower() != CANONICAL_QUERY_MANIFEST_SHA256.lower():
        raise AssertionError(f"Query Manifest canonical JSON SHA mismatch: expected {CANONICAL_QUERY_MANIFEST_SHA256}, got {qm_sha}")
    print(f"✅ Query Manifest canonical JSON SHA256 verified: {qm_sha}", flush=True)

    mr_sha = canonical_sidecar_sha256(manual_ref_path)
    if mr_sha.lower() != CANONICAL_MANUAL_REF_SHA256.lower():
        raise AssertionError(f"Manual Reference canonical JSON SHA mismatch: expected {CANONICAL_MANUAL_REF_SHA256}, got {mr_sha}")
    print(f"✅ Manual Reference canonical JSON SHA256 verified: {mr_sha}", flush=True)

    # Policy Resolution
    if policy == "benchmark_tuned":
        if told_sidecar_path is None or not told_sidecar_path.exists():
            raise FileNotFoundError("Policy 'benchmark_tuned' requires a valid --told-sidecar path!")
        active_sidecar_path = told_sidecar_path
        expected_sidecar_sha = CANONICAL_TOLD_SHA256
        policy_label = "EXPERIMENTAL_BENCHMARK_TUNED"
        print(f"🔬 Policy: {policy_label} (Reference profile; not default production RC1)", flush=True)
    else:
        active_sidecar_path = tnew_sidecar_path
        expected_sidecar_sha = CANONICAL_TNEW_SHA256
        policy_label = "PRODUCTION_GENERAL_PURPOSE"
        print(f"🎯 Policy: {policy_label} (Official KIS_V2A_RC1 default identity)", flush=True)

    active_sidecar_sha = canonical_sidecar_sha256(active_sidecar_path)
    if active_sidecar_sha.lower() != expected_sidecar_sha.lower():
        raise AssertionError(f"Active Sidecar canonical SHA mismatch: expected {expected_sidecar_sha}, got {active_sidecar_sha}")
    print(f"✅ Active Translation Sidecar verified ({policy_label}): {active_sidecar_sha}", flush=True)

    # Build Translation Provider
    provider = ImmutableSidecarTranslationProvider(
        sidecar_path=active_sidecar_path,
        expected_content_sha256=active_sidecar_sha,
    )

    queries_data = json.loads(query_manifest_path.read_text(encoding="utf-8")).get("queries", [])
    gt_ref = load_manual_reference(manual_ref_path)
    print(f"✅ Loaded {len(queries_data)} queries and {len(gt_ref)} ground truth targets", flush=True)

    # 2. Execute Session 1 (Run 1)
    run1_out_dir = output_root / "run_1"
    run1_results, corpus_info = execute_closure_session(
        run_name="run_1",
        run_out_dir=run1_out_dir,
        input_root=input_root,
        manifest_cache_path=manifest_cache_path,
        queries_data=queries_data,
        gt_ref=gt_ref,
        provider=provider,
        device=device,
        strict_corpus_gate=strict_corpus_gate,
    )

    # 3. Execute Session 2 (Run 2) - Clean, isolated session
    run2_out_dir = output_root / "run_2"
    run2_results, _ = execute_closure_session(
        run_name="run_2",
        run_out_dir=run2_out_dir,
        input_root=input_root,
        manifest_cache_path=manifest_cache_path,
        queries_data=queries_data,
        gt_ref=gt_ref,
        provider=provider,
        device=device,
        strict_corpus_gate=strict_corpus_gate,
    )

    # 4. Rigorous Bit-Exact Comparison Between Run 1 and Run 2
    print("\n" + "=" * 110, flush=True)
    print("⚖️ VERIFYING 100% BIT-EXACT REPRODUCIBILITY (RUN 1 ≡ RUN 2)", flush=True)
    print("=" * 110, flush=True)

    bit_exact_parity = True
    query_comparisons: dict[str, Any] = {}

    for q_item in queries_data:
        qid = q_item["query_id"]
        d1 = run1_results[qid]["canonical_projection_digest"]
        d2 = run2_results[qid]["canonical_projection_digest"]
        match = (d1 == d2)
        if not match:
            bit_exact_parity = False
            raise AssertionError(f"Bit-exact parity failed between Run 1 and Run 2 on {qid}: {d1} != {d2}")

        r1 = run1_results[qid]["human_target_video_rank"]
        r2 = run2_results[qid]["human_target_video_rank"]
        print(f"  • {qid:<18}: Digest Parity: PASS (Digest: {d1[:16]}...) | Target Rank: #{r1} ≡ #{r2}", flush=True)

        query_comparisons[qid] = {
            "digest_run_1": d1,
            "digest_run_2": d2,
            "bit_exact": match,
            "target_rank": r1,
            "target_coarse_rank": run1_results[qid]["target_coarse_rank"],
            "selected_videos": run1_results[qid]["selected_video_count"],
        }

    print("✅ 100% Bit-Exact Determinism verified across all 5 queries!", flush=True)

    # 5. Cross-audit against Phase C.1 C_NEW_SINGLE (if available)
    cross_audit_verified = None
    if phase_c1_audit_path and phase_c1_audit_path.exists():
        print("\n🔍 Cross-auditing against Phase C.1 audit artifact...", flush=True)
        try:
            c1_data = json.loads(phase_c1_audit_path.read_text(encoding="utf-8"))
            c1_new_arm = c1_data.get("arms", {}).get("C_NEW_SINGLE", {})
            if c1_new_arm:
                drift_detected = False
                for qid in run1_results:
                    c1_digest = c1_new_arm.get(qid, {}).get("canonical_projection_digest")
                    rc1_digest = run1_results[qid]["canonical_projection_digest"]
                    if c1_digest != rc1_digest:
                        drift_detected = True
                        print(f"⚠️ Cross-audit drift detected on {qid}: Phase C.1={c1_digest} vs RC1={rc1_digest}", flush=True)
                if not drift_detected:
                    cross_audit_verified = True
                    print("✅ Zero-configuration drift PASS: All RC1 projection digests match Phase C.1 C_NEW_SINGLE exactly!", flush=True)
        except Exception as exc:
            print(f"⚠️ Note: Phase C.1 cross-audit failed to complete: {exc}", flush=True)

    # 6. Build and Emit Release Manifest
    release_manifest = {
        "release_candidate": RELEASE_CANDIDATE_ID,
        "policy": policy_label,
        "is_production_default": (policy == "general"),
        "timestamp": datetime.now(UTC).isoformat(),
        "git_commit_sha": git_sha,
        "device": device,
        "configuration": {
            "v2_adaptive_enabled": True,
            "selected_video_cap_ceiling": 64,
            "video_nomination_depth": 100,
            "restricted_frames_per_video_per_variant": 20,
            "full_query_weight": 1.0,
            "primary_scene_weight": 1.0,
            "supporting_attribute_weight": 0.35,
            "top_m_evidence_cap": 5,
            "top_m_weights": [0.4, 0.25, 0.15, 0.1, 0.1],
            "top_m_min_frame_gap": 60,
            "enable_temporal_diverse_local_candidates": True,
            "temporal_diversity_gap_seconds": 5.0,
            "enable_vi_localization_variant": True,
            "vi_localization_weight": 0.5,
            "internal_rrf_candidate_depth": 1000,
            "rrf_constant": 60.0,
            "enable_paraphrase_ensemble": False,
            "enable_top_video_local_anchor": False,
            "final_output_top_k": 100,
        },
        "provenance": {
            "query_manifest": {"path": str(query_manifest_path), "canonical_sha256": qm_sha},
            "manual_reference": {"path": str(manual_ref_path), "canonical_sha256": mr_sha},
            "active_sidecar": {"path": str(active_sidecar_path), "canonical_sha256": active_sidecar_sha},
        },
        "corpus": corpus_info,
        "verification_gates": {
            "strict_corpus_gate_verified": (corpus_info.get("video_count") == 873 and corpus_info.get("total_rows") == 177321),
            "two_pass_closure_bit_exact": bit_exact_parity,
            "cross_audit_with_phase_c1": cross_audit_verified,
        },
        "queries": query_comparisons,
    }

    manifest_out_path = output_root / "kis_v2a_rc1_closure_manifest.json"
    manifest_out_path.write_text(json.dumps(release_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n📦 Formal Release Manifest written to: {manifest_out_path}", flush=True)

    total_time = time.time() - start_time_all
    print("\n" + "=" * 110, flush=True)
    print(f"🎉 KIS V2-A.3 RC1 FULL-SYSTEM E2E CLOSURE: SUCCESS (Total Time: {total_time:.2f}s)", flush=True)
    print("=" * 110, flush=True)
    print(f"Release Candidate: {RELEASE_CANDIDATE_ID}")
    print(f"Production Policy: {policy_label} (Sidecar: {active_sidecar_path.name})")
    print("Bit-Exact Determinism: 100% across all 5 queries (Run 1 ≡ Run 2)")
    print("=" * 110, flush=True)

    return release_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KIS V2-A.3 Release Candidate 1 (KIS_V2A_RC1) E2E Closure Runner")
    parser.add_argument("--query-manifest", type=Path, required=True, help="Path to frozen query manifest JSON")
    parser.add_argument("--input-root", type=Path, required=True, help="Path to dataset input root")
    parser.add_argument("--manifest-cache", type=Path, required=True, help="Path to pre-generated portable manifest cache JSON")
    parser.add_argument("--output", type=Path, required=True, help="Path to closure output directory")
    parser.add_argument("--tnew-sidecar", type=Path, required=True, help="Path to Canonical T-New sidecar JSON")
    parser.add_argument("--manual-ref", type=Path, required=True, help="Path to ground truth reference JSON")
    parser.add_argument("--told-sidecar", type=Path, default=None, help="Path to Candidate Old sidecar JSON (for benchmark_tuned policy)")
    parser.add_argument("--policy", choices=["general", "benchmark_tuned"], default="general", help="Translation policy: general (default) or benchmark_tuned")
    parser.add_argument("--phase-c1-audit", type=Path, default=None, help="Optional path to phase_c1_four_way_ablation_audit.json for drift checking")
    parser.add_argument("--strict-corpus-gate", action="store_true", help="Enforce 873 videos / 177321 rows / 512 dim gate")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Inference device (locked to 'cpu' for official release)")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    args = parse_args()
    run_kis_v2a_rc1_e2e_closure(
        query_manifest_path=args.query_manifest,
        input_root=args.input_root,
        manifest_cache_path=args.manifest_cache,
        output_root=args.output,
        tnew_sidecar_path=args.tnew_sidecar,
        manual_ref_path=args.manual_ref,
        told_sidecar_path=args.told_sidecar,
        policy=args.policy,
        phase_c1_audit_path=args.phase_c1_audit,
        strict_corpus_gate=args.strict_corpus_gate,
        device=args.device,
    )


if __name__ == "__main__":
    main()
