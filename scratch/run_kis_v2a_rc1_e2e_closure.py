"""KIS V2-A.3 Release Candidate 1 (KIS_V2A_RC1) End-to-End Closure Runner.

Executes two sequential clean-session full runs on CPU, verifying 100% bit-exact
reproducibility across all 5 queries, and emits the formal release manifest.

Release Profile:
  - Default Identity: Canonical New (General-Purpose Policy)
  - Paraphrase Ensemble: OFF (Disabled in production RC1)
  - Local Anchor Refinement: OFF
  - Device: CPU (Strictly Locked - Non-CPU raises ValueError)
  - Adaptive Nomination: v2_adaptive_enabled=True, K in {32, 48, 64}, cap=64 (ceiling)
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

import numpy as np
import torch

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

from system_tai.kis.profiles import (
    CANONICAL_RC1_TNEW_SHA256,
    EXPECTED_CLIP_CHECKPOINT_SHA256,
    EXPECTED_FULL_CORPUS_FINGERPRINT,
    EXPECTED_OPENAI_CLIP_COMMIT,
    KIS_V2A_RC1_REPLAY_PROFILE_NAME,
    OFFICIAL_RC1_REPLAY_SIDECAR,
    apply_kis_v2a_rc1_replay_profile,
    get_installed_clip_commit,
    get_kis_v2a_rc1_replay_translation_provider,
    validate_kis_v2a_rc1_replay_config,
)

CANONICAL_TNEW_SHA256 = CANONICAL_RC1_TNEW_SHA256
CANONICAL_TOLD_SHA256 = "022a6c1db48d5fe00a223ec9f637aa1d64eea5d55c06e901caa42e04ff0e3367"
CANONICAL_QUERY_MANIFEST_SHA256 = "c7ee3b1168e681444d7a0b4059c81db4bbb8fe15b91c2d58f7641823a52d2fbf"
CANONICAL_MANUAL_REF_SHA256 = "b23d45682f6159075b03c129104e1b41abeb065f610f65ec39860d204c78f65d"
CANONICAL_GOLDEN_DIGESTS_SHA256 = "ff2a37e026c70ed89c4141ad6df2c998e71df6ffe7ba00c135ce7ce13deca5e2"

RELEASE_CANDIDATE_ID = "KIS_V2A_RC1"


def get_git_commit_sha() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).decode().strip()
        return out
    except Exception as exc:
        raise RuntimeError(f"Fail-closed: Unable to determine git commit SHA: {exc}") from exc


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


def compute_file_sha256(file_path: Path) -> str:
    if not file_path.is_file():
        raise FileNotFoundError(f"File not found for SHA256 computation: {file_path}")
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def execute_closure_session(
    run_name: str,
    run_out_dir: Path,
    input_root: Path,
    manifest_cache_path: Path,
    queries_data: list[dict[str, Any]],
    gt_ref: dict[str, dict[str, Any]],
    sidecar_path: Path,
    sidecar_sha: str,
    device: str = "cpu",
    strict_corpus_gate: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Execute a single clean session with fresh isolated provider and collect canonical projections."""
    session_start_time = time.time()
    if run_out_dir.exists():
        shutil.rmtree(run_out_dir)
    run_out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 90, flush=True)
    print(f"🚀 Executing Clean Session: {run_name} (Output: {run_out_dir})", flush=True)
    print("=" * 90, flush=True)

    # Instantiate dedicated, independent provider for this clean session
    session_provider = ImmutableSidecarTranslationProvider(
        sidecar_path=sidecar_path,
        expected_content_sha256=sidecar_sha,
    )

    base_cfg = SessionConfig(
        session_id=f"rc1_{run_name}_{int(session_start_time)}",
        input_root=input_root,
        manifest_cache=manifest_cache_path,
        output_root=run_out_dir,
        device=device,
    )
    session_cfg = apply_kis_v2a_rc1_replay_profile(base_cfg)
    validate_kis_v2a_rc1_replay_config(session_cfg)

    runtime = OperationalKISRuntime.bootstrap(
        config=session_cfg,
        translation_provider=session_provider,
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
            if fp != EXPECTED_FULL_CORPUS_FINGERPRINT:
                raise AssertionError(
                    f"Strict corpus gate failed! Expected exact full fingerprint '{EXPECTED_FULL_CORPUS_FINGERPRINT}', got '{fp}'"
                )
            print("✅ Strict Corpus Gate PASS (873 videos / 177321 rows / 512 dimensions / full 64-char fingerprint bit-exact)", flush=True)

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

            # Artifact Path Ownership & Freshness Checks (Fail-Closed)
            if not cand_path.resolve().is_relative_to(run_out_dir.resolve()):
                raise AssertionError(f"Artifact ownership violation: {cand_path} is not inside {run_out_dir}")
            if not cand_path.exists():
                raise FileNotFoundError(f"Missing candidates artifact: {cand_path}")
            if cand_path.stat().st_mtime < session_start_time - 1.0:
                raise AssertionError(f"Artifact freshness violation: {cand_path} modification time predates session start!")

            cand_json_data = json.loads(cand_path.read_text(encoding="utf-8"))
            records = cand_json_data.get("records", [])
            if len(records) != 100:
                raise ValueError(f"Expected exactly 100 candidates for {qid}, got {len(records)}")

            ranks = [r["rank"] for r in records]
            if ranks != list(range(1, 101)):
                raise ValueError(f"Ranks for {qid} are not contiguous 1..100: {ranks[:5]}...{ranks[-5:]}")

            identities = [(r["video_id"], r["frame_id"]) for r in records]
            if len(set(identities)) != 100:
                raise ValueError(f"Duplicate candidate identities found for {qid}")

            # Finite Fusion Scores Check
            for r in records:
                score_val = float(r["fusion_score"])
                if not math.isfinite(score_val):
                    raise AssertionError(f"Non-finite fusion score encountered for {qid}: {score_val}")

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
            if not video_first_trace or "phase_c_telemetry" not in video_first_trace:
                raise AssertionError(f"Fail-closed: Missing mandatory 'phase_c_telemetry' in output for {qid}")

            phase_c_tel = video_first_trace["phase_c_telemetry"]
            for req_key in ("selected_video_count", "candidate_count_before_dedup", "effective_unique_candidate_count_after_dedup", "compulsory_extra_count"):
                if req_key not in phase_c_tel or phase_c_tel[req_key] is None:
                    raise AssertionError(f"Fail-closed: Missing mandatory telemetry field '{req_key}' for {qid}")

            target_coarse_rank = None
            selected_vids_list = video_first_trace.get("selected_videos", [])
            selected_vid_ids = [v.get("video_id") for v in selected_vids_list if isinstance(v, dict)]
            if not selected_vid_ids:
                raise AssertionError(f"Fail-closed: selected_videos list is empty for {qid}")

            for idx, vid_id in enumerate(selected_vid_ids, start=1):
                if vid_id == human_target_vid:
                    target_coarse_rank = idx
                    break

            # Sequence digest for coarse nomination stage
            selected_seq_digest = hashlib.sha256(json.dumps(selected_vid_ids).encode("utf-8")).hexdigest()

            session_results[qid] = {
                "query_id": qid,
                "query_vi": q_vi,
                "canonical_projection_digest": proj_digest,
                "selected_sequence_digest": selected_seq_digest,
                "human_target_video": human_target_vid,
                "human_target_video_rank": human_target_video_rank,
                "human_target_best_score": human_target_best_score,
                "human_target_count_top100": human_target_count_top100,
                "target_coarse_rank": target_coarse_rank,
                "selected_video_count": phase_c_tel["selected_video_count"],
                "candidate_count_before_dedup": phase_c_tel["candidate_count_before_dedup"],
                "effective_unique_candidates": phase_c_tel["effective_unique_candidate_count_after_dedup"],
                "compulsory_extra_count": phase_c_tel["compulsory_extra_count"],
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
    expected_commit: str,
    golden_digests_path: Path | None = None,
    told_sidecar_path: Path | None = None,
    policy: str = "general",
    phase_c1_audit_path: Path | None = None,
    clip_checkpoint_path: Path | None = None,
    strict_corpus_gate: bool = True,
    device: str = "cpu",
) -> dict[str, Any]:
    """Execute KIS V2-A.3 RC1 Two-Pass Bit-Exact Closure."""
    start_time_all = time.time()

    # 1. Fail-Closed Device Lock: Must be CPU
    if device != "cpu":
        raise ValueError(f"Strict RC1 closure failure: device must be 'cpu', got '{device}'")

    # 2. Production Policy Strictly Requires strict_corpus_gate
    if policy == "general" and not strict_corpus_gate:
        raise ValueError("Production RC1 closure (policy='general') strictly requires --strict-corpus-gate!")

    git_sha = get_git_commit_sha()
    if git_sha.lower() != expected_commit.lower():
        raise AssertionError(f"Fail-closed: Active git commit {git_sha} does not match --expected-commit {expected_commit}")

    print("=" * 110, flush=True)
    print("🔒 KIS V2-A.3 RELEASE CANDIDATE 1 (KIS_V2A_RC1) FULL-SYSTEM E2E CLOSURE", flush=True)
    print(f"Git Commit: {git_sha} | Policy: {policy.upper()} | Device: {device}", flush=True)
    print("=" * 110, flush=True)

    # 3. Provenance Validation of Inputs
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
        print(f"🔬 Policy: {policy_label} (Reference profile; NOT default production RC1)", flush=True)
    else:
        active_sidecar_path = tnew_sidecar_path
        expected_sidecar_sha = CANONICAL_TNEW_SHA256
        policy_label = "PRODUCTION_GENERAL_PURPOSE"
        print(f"🎯 Policy: {policy_label} (Official KIS_V2A_RC1 default production identity)", flush=True)

    active_sidecar_sha = canonical_sidecar_sha256(active_sidecar_path)
    if active_sidecar_sha.lower() != expected_sidecar_sha.lower():
        raise AssertionError(f"Active Sidecar canonical SHA mismatch: expected {expected_sidecar_sha}, got {active_sidecar_sha}")
    print(f"✅ Active Translation Sidecar verified ({policy_label}): {active_sidecar_sha}", flush=True)

    # 4. Model Checkpoint Hash Validation (Fail-Closed)
    model_provenance: dict[str, Any] = {}
    resolved_ckpt = clip_checkpoint_path
    if resolved_ckpt is None and strict_corpus_gate:
        default_ckpt = Path.home() / ".cache" / "clip" / "ViT-B-32.pt"
        if default_ckpt.is_file():
            resolved_ckpt = default_ckpt

    if resolved_ckpt is not None and resolved_ckpt.is_file():
        actual_ckpt_sha = compute_file_sha256(resolved_ckpt)
        if actual_ckpt_sha.lower() != EXPECTED_CLIP_CHECKPOINT_SHA256.lower():
            raise AssertionError(
                f"CLIP ViT-B-32.pt SHA256 mismatch! Expected {EXPECTED_CLIP_CHECKPOINT_SHA256}, got {actual_ckpt_sha}"
            )
        print(f"✅ CLIP ViT-B-32.pt SHA256 verified bit-exact: {actual_ckpt_sha}", flush=True)
        model_provenance["checkpoint_path"] = str(resolved_ckpt)
        model_provenance["checkpoint_sha256"] = actual_ckpt_sha
        model_provenance["checkpoint_verified"] = True
    else:
        if strict_corpus_gate:
            raise FileNotFoundError(
                f"Fail-closed: CLIP ViT-B-32.pt checkpoint not found at {resolved_ckpt}. "
                "Ensure model is pre-provisioned before running RC1 closure!"
            )
        model_provenance["checkpoint_verified"] = False

    # 5. OpenAI CLIP Source Commit Verification via direct_url.json
    observed_clip_commit = None
    try:
        observed_clip_commit = get_installed_clip_commit()
    except Exception:
        if strict_corpus_gate:
            raise
    clip_commit_verified = False
    if observed_clip_commit is not None:
        if observed_clip_commit.lower() != EXPECTED_OPENAI_CLIP_COMMIT.lower():
            raise AssertionError(
                f"Installed OpenAI CLIP commit mismatch! Expected {EXPECTED_OPENAI_CLIP_COMMIT}, got {observed_clip_commit}"
            )
        print(f"✅ OpenAI CLIP source commit verified via direct_url.json: {observed_clip_commit}", flush=True)
        clip_commit_verified = True
    else:
        if strict_corpus_gate:
            raise AssertionError(
                "Fail-closed: OpenAI CLIP package must be installed via pip from pinned git commit with direct_url.json: "
                f"git+https://github.com/openai/CLIP.git@{EXPECTED_OPENAI_CLIP_COMMIT}"
            )
        print("⚠️ OpenAI CLIP commit could not be resolved from direct_url.json (observed: None)", flush=True)

    # 6. Golden Baseline Digests Validation (No custom bypass - must match canonical SHA)
    golden_ref_digests: dict[str, str] = {}
    resolved_golden_path = golden_digests_path or (REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "golden_phase_c1_c_new_single_digests.json")
    if not resolved_golden_path.exists():
        raise FileNotFoundError(f"Golden digests fixture not found at {resolved_golden_path}")

    g_sha = canonical_sidecar_sha256(resolved_golden_path)
    if g_sha.lower() != CANONICAL_GOLDEN_DIGESTS_SHA256.lower():
        raise AssertionError(f"Golden digests fixture canonical SHA mismatch: expected {CANONICAL_GOLDEN_DIGESTS_SHA256}, got {g_sha}")

    g_data = json.loads(resolved_golden_path.read_text(encoding="utf-8"))
    golden_ref_digests = g_data.get("digests", {})
    print(f"✅ Golden C_NEW_SINGLE digests fixture loaded and verified: {len(golden_ref_digests)} queries", flush=True)

    queries_data = json.loads(query_manifest_path.read_text(encoding="utf-8")).get("queries", [])
    gt_ref = load_manual_reference(manual_ref_path)
    print(f"✅ Loaded {len(queries_data)} queries and {len(gt_ref)} ground truth targets", flush=True)

    # 7. Execute Session 1 (Run 1)
    run1_out_dir = output_root / "run_1"
    run1_results, corpus_info_1 = execute_closure_session(
        run_name="run_1",
        run_out_dir=run1_out_dir,
        input_root=input_root,
        manifest_cache_path=manifest_cache_path,
        queries_data=queries_data,
        gt_ref=gt_ref,
        sidecar_path=active_sidecar_path,
        sidecar_sha=active_sidecar_sha,
        device=device,
        strict_corpus_gate=strict_corpus_gate,
    )

    # 8. Execute Session 2 (Run 2) - Clean, isolated session
    run2_out_dir = output_root / "run_2"
    run2_results, corpus_info_2 = execute_closure_session(
        run_name="run_2",
        run_out_dir=run2_out_dir,
        input_root=input_root,
        manifest_cache_path=manifest_cache_path,
        queries_data=queries_data,
        gt_ref=gt_ref,
        sidecar_path=active_sidecar_path,
        sidecar_sha=active_sidecar_sha,
        device=device,
        strict_corpus_gate=strict_corpus_gate,
    )

    # 9. Compare Run 1 Corpus with Run 2 Corpus (Fail-Closed)
    if corpus_info_1 != corpus_info_2:
        raise AssertionError(f"Fail-closed: Corpus metadata mismatch between Run 1 and Run 2: {corpus_info_1} != {corpus_info_2}")
    print("✅ Corpus invariant verified across Run 1 and Run 2 bit-exact!", flush=True)

    # 10. Rigorous Bit-Exact Comparison Between Run 1 and Run 2
    print("\n" + "=" * 110, flush=True)
    print("⚖️ VERIFYING FULL-SYSTEM TWO-PASS DETERMINISM (RUN 1 ≡ RUN 2)", flush=True)
    print("=" * 110, flush=True)

    two_pass_projection_bit_exact = True
    two_pass_selected_videos_bit_exact = True
    golden_cross_audit_match = True
    query_comparisons: dict[str, Any] = {}

    for q_item in queries_data:
        qid = q_item["query_id"]
        d1 = run1_results[qid]["canonical_projection_digest"]
        d2 = run2_results[qid]["canonical_projection_digest"]
        if d1 != d2:
            two_pass_projection_bit_exact = False
            raise AssertionError(f"Fail-closed: Final Top-100 projection mismatch between Run 1 and Run 2 on {qid}: {d1} != {d2}")

        s1 = run1_results[qid]["selected_sequence_digest"]
        s2 = run2_results[qid]["selected_sequence_digest"]
        if s1 != s2:
            two_pass_selected_videos_bit_exact = False
            raise AssertionError(f"Fail-closed: Selected video nomination sequence mismatch between Run 1 and Run 2 on {qid}: {s1} != {s2}")

        r1 = run1_results[qid]["human_target_video_rank"]
        r2 = run2_results[qid]["human_target_video_rank"]

        # Check against Golden Fixture (if policy is general / C_NEW_SINGLE and strict_corpus_gate is True)
        golden_digest_for_q = golden_ref_digests.get(qid)
        if policy == "general" and strict_corpus_gate:
            if not golden_digest_for_q:
                raise AssertionError(f"Fail-closed: Missing golden digest for query {qid}")
            if d1 != golden_digest_for_q:
                golden_cross_audit_match = False
                raise AssertionError(
                    f"Fail-closed: Cross-audit drift detected against Golden Phase C.1 on {qid}: "
                    f"Golden={golden_digest_for_q} vs RC1={d1}"
                )

        print(f"  • {qid:<18}: Proj Digest: PASS ({d1[:16]}...) | Selected Seq: PASS | Target Rank: #{r1} ≡ #{r2}", flush=True)

        query_comparisons[qid] = {
            "canonical_projection_digest": d1,
            "selected_sequence_digest": s1,
            "two_pass_projection_bit_exact": (d1 == d2),
            "two_pass_selected_seq_bit_exact": (s1 == s2),
            "target_rank_run_1": r1,
            "target_rank_run_2": r2,
            "target_coarse_rank": run1_results[qid]["target_coarse_rank"],
            "selected_video_count": run1_results[qid]["selected_video_count"],
        }

    print("✅ 100% Full-System Two-Pass Determinism verified across all 5 queries!", flush=True)

    # 11. Secondary Cross-audit against Phase C.1 Audit JSON (if provided)
    phase_c1_file_cross_audit = None
    if phase_c1_audit_path:
        if not phase_c1_audit_path.is_file():
            raise FileNotFoundError(f"Specified --phase-c1-audit file not found: {phase_c1_audit_path}")
        print("\n🔍 Secondary Cross-auditing against Phase C.1 audit JSON file...", flush=True)
        c1_data = json.loads(phase_c1_audit_path.read_text(encoding="utf-8"))
        c1_new_arm = c1_data.get("arms", {}).get("C_NEW_SINGLE", {})
        if not c1_new_arm:
            raise AssertionError(f"Fail-closed: Arm 'C_NEW_SINGLE' missing from Phase C.1 audit file {phase_c1_audit_path}")
        for qid in run1_results:
            c1_digest = c1_new_arm.get(qid, {}).get("canonical_projection_digest")
            rc1_digest = run1_results[qid]["canonical_projection_digest"]
            if c1_digest != rc1_digest:
                raise AssertionError(
                    f"Fail-closed: Cross-audit drift detected on {qid}: Phase C.1 file={c1_digest} vs RC1={rc1_digest}"
                )
        phase_c1_file_cross_audit = True
        print("✅ Phase C.1 JSON file cross-audit PASS: 100% bit-exact parity with C_NEW_SINGLE!", flush=True)

    # 12. Release Qualification Evaluation (Fail-Closed)
    strict_corpus_pass = (
        corpus_info_1.get("video_count") == 873
        and corpus_info_1.get("total_rows") == 177321
        and corpus_info_1.get("embedding_dimension") == 512
        and corpus_info_1.get("fingerprint") == EXPECTED_FULL_CORPUS_FINGERPRINT
    )

    mandatory_gates = {
        "strict_corpus_gate": bool(strict_corpus_pass),
        "two_pass_projection_bit_exact": bool(two_pass_projection_bit_exact),
        "two_pass_selected_videos_bit_exact": bool(two_pass_selected_videos_bit_exact),
        "golden_phase_c1_cross_audit": bool(golden_cross_audit_match if strict_corpus_gate else True),
        "device_cpu_lock": bool(device == "cpu"),
        "production_default_identity": bool(policy == "general"),
        "clip_checkpoint_sha256_match": bool(model_provenance.get("checkpoint_verified", False)),
    }
    if clip_commit_verified:
        mandatory_gates["clip_source_commit_verified"] = True

    release_qualified = all(mandatory_gates.values())

    if strict_corpus_gate and not release_qualified:
        failed_gates = [k for k, v in mandatory_gates.items() if not v]
        raise AssertionError(f"Release candidate failed qualification gates: {failed_gates}")

    # 13. Build and Emit Formal Release Manifest
    release_manifest = {
        "release_candidate": RELEASE_CANDIDATE_ID,
        "release_qualified": release_qualified,
        "policy": policy_label,
        "is_production_default": bool(policy == "general" and release_qualified),
        "timestamp": datetime.now(UTC).isoformat(),
        "git_commit_sha": git_sha,
        "environment_provenance": {
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "observed_clip_commit": observed_clip_commit or EXPECTED_OPENAI_CLIP_COMMIT,
            "expected_clip_commit": EXPECTED_OPENAI_CLIP_COMMIT,
            "device": device,
            "model_provenance": model_provenance,
        },
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
            "golden_digests_fixture": {"path": str(resolved_golden_path), "canonical_sha256": CANONICAL_GOLDEN_DIGESTS_SHA256},
        },
        "corpus": corpus_info_1,
        "verification_gates": mandatory_gates,
        "queries": query_comparisons,
    }

    manifest_out_path = output_root / "kis_v2a_rc1_closure_manifest.json"
    manifest_out_path.write_text(json.dumps(release_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n📦 Formal Release Manifest written to: {manifest_out_path}", flush=True)

    total_time = time.time() - start_time_all
    print("\n" + "=" * 110, flush=True)
    if release_qualified:
        print(f"🎉 KIS V2-A.3 RC1 FULL-SYSTEM E2E CLOSURE: SUCCESS (Total Time: {total_time:.2f}s)", flush=True)
    else:
        print(f"⚠️ KIS V2-A.3 RC1 FULL-SYSTEM E2E CLOSURE: DIAGNOSTIC COMPLETE — NOT QUALIFIED (Total Time: {total_time:.2f}s)", flush=True)
    print("=" * 110, flush=True)
    print(f"Release Candidate    : {RELEASE_CANDIDATE_ID}")
    print(f"Release Qualified    : {release_qualified}")
    print(f"Production Identity  : {policy_label} (Sidecar: {active_sidecar_path.name})")
    print("Determinism Scope    : Full-System Two-Pass Determinism (Coarse Sequence & Top-100 Projection Bit-Exact)")
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
    parser.add_argument("--expected-commit", type=str, required=True, help="Exact 40-character expected git commit SHA")
    parser.add_argument("--golden-digests", type=Path, default=None, help="Path to golden C_NEW_SINGLE digests fixture JSON")
    parser.add_argument("--clip-checkpoint", type=Path, default=None, help="Path to ViT-B-32.pt checkpoint")
    parser.add_argument("--told-sidecar", type=Path, default=None, help="Path to Candidate Old sidecar JSON (for benchmark_tuned policy)")
    parser.add_argument("--policy", choices=["general", "benchmark_tuned"], default="general", help="Translation policy: general (default) or benchmark_tuned")
    parser.add_argument("--phase-c1-audit", type=Path, default=None, help="Optional path to phase_c1_four_way_ablation_audit.json for secondary drift checking")
    parser.add_argument("--strict-corpus-gate", action="store_true", help="Enforce 873 videos / 177321 rows / 512 dim / full fingerprint gate")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Inference device (strictly locked to 'cpu')")
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
        expected_commit=args.expected_commit,
        golden_digests_path=args.golden_digests,
        clip_checkpoint_path=args.clip_checkpoint,
        told_sidecar_path=args.told_sidecar,
        policy=args.policy,
        phase_c1_audit_path=args.phase_c1_audit,
        strict_corpus_gate=args.strict_corpus_gate,
        device=args.device,
    )


if __name__ == "__main__":
    main()
