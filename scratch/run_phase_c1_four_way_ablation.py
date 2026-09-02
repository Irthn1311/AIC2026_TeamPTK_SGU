"""Phase C.1: Four-Way Controlled Ablation Benchmark & Dilution Diagnosis Runner.

Experimental Arms:
  1. C_NEW_SINGLE: Canonical New-only Single Baseline (Current Control)
  2. C_OLD_SINGLE: Candidate Old-only Single Baseline
  3. C_NEW_DUP_SHAM: Sham Paraphrase Ensemble (2x Canonical New, Quota Split)
  4. C_NEW_OLD_50_50: True Paraphrase Ensemble (Canonical New + Candidate Old 50/50)

Evaluates case-level causal diagnosis for query P1-5 on full/frozen corpus.
Zero network access, fail-closed golden validation, invariant enforcement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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
from system_tai.translation.paraphrase_sidecar_provider import (
    ImmutableParaphraseEnsembleSidecarProvider,
)
from system_tai.translation.sidecar_provider import (
    ImmutableSidecarTranslationProvider,
    canonical_sidecar_sha256,
)

CANONICAL_TNEW_SHA256 = "545bd4a37c57af53713a1d9f382241ef729c287a1817a5671fdc923115b0be2a"
CANONICAL_TOLD_SHA256 = "022a6c1db48d5fe00a223ec9f637aa1d64eea5d55c06e901caa42e04ff0e3367"
CANONICAL_SHAM_SHA256 = "baeded42652804068378831b7478eee9535844e7d1c3acb40284b4c875d1a9a3"
CANONICAL_PARAPHRASE_SHA256 = "1bb2a15e7f55d9b1947552cdd33f5dba52b4316444781ff8d883aa359f163cf2"
CANONICAL_QUERY_MANIFEST_SHA256 = "c7ee3b1168e681444d7a0b4059c81db4bbb8fe15b91c2d58f7641823a52d2fbf"
CANONICAL_MANUAL_REF_SHA256 = "b23d45682f6159075b03c129104e1b41abeb065f610f65ec39860d204c78f65d"
DIAGNOSTIC_RANK_DELTA_THRESHOLD = 2


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


def compute_pairwise_comparison(
    records_a: list[dict[str, Any]],
    records_b: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute exact intersection, Jaccard distance, and median rank shifts."""
    ids_a = {(r["video_id"], r["frame_id"]): r["rank"] for r in records_a}
    ids_b = {(r["video_id"], r["frame_id"]): r["rank"] for r in records_b}

    set_a = set(ids_a.keys())
    set_b = set(ids_b.keys())

    inter = set_a & set_b
    union = set_a | set_b

    inter_cnt = len(inter)
    union_cnt = len(union)
    jaccard_sim = inter_cnt / union_cnt if union_cnt > 0 else 1.0
    jaccard_dist = 1.0 - jaccard_sim
    replaced_cnt = 100 - inter_cnt
    replaced_ratio = replaced_cnt / 100.0

    shifts: list[int] = [abs(ids_a[k] - ids_b[k]) for k in inter]
    median_shift = float(statistics.median(shifts)) if shifts else 0.0
    mean_shift = float(statistics.mean(shifts)) if shifts else 0.0

    return {
        "intersection_count": inter_cnt,
        "union_count": union_cnt,
        "jaccard_similarity": round(jaccard_sim, 6),
        "jaccard_distance": round(jaccard_dist, 6),
        "membership_replaced_count": replaced_cnt,
        "membership_replaced_ratio": round(replaced_ratio, 6),
        "survivor_count": len(shifts),
        "median_rank_shift": median_shift,
        "mean_rank_shift": round(mean_shift, 4),
    }


def classify_p15_findings(
    r_new: int | None,
    r_old: int | None,
    r_sham: int | None,
    r_mixed: int | None,
    selected_video_count_parity: bool,
    threshold: int = DIAGNOSTIC_RANK_DELTA_THRESHOLD,
) -> tuple[dict[str, Any], str]:
    """Evaluate independent causal contrasts and primary verdict."""
    rr_new = 1.0 / r_new if (r_new is not None and r_new > 0) else 0.0
    rr_old = 1.0 / r_old if (r_old is not None and r_old > 0) else 0.0
    rr_sham = 1.0 / r_sham if (r_sham is not None and r_sham > 0) else 0.0
    rr_mixed = 1.0 / r_mixed if (r_mixed is not None and r_mixed > 0) else 0.0

    delta_mechanics = (r_sham - r_new) if (r_sham is not None and r_new is not None) else None
    delta_wording = (r_old - r_new) if (r_old is not None and r_new is not None) else None
    delta_replacement = (r_mixed - r_sham) if (r_mixed is not None and r_sham is not None) else None

    delta_rr_mechanics = rr_sham - rr_new
    delta_rr_wording = rr_old - rr_new
    delta_rr_replacement = rr_mixed - rr_sham

    findings = {
        "ensemble_mechanics_effect": bool(
            (r_sham is not None and r_new is not None and r_sham > r_new + threshold)
            or (r_sham is None and r_new is not None)
        ),
        "old_wording_weaker_than_new": bool(
            (r_old is not None and r_new is not None and r_old > r_new + threshold)
            or (r_old is None and r_new is not None)
        ),
        "old_wording_stronger_than_new": bool(
            r_old is not None and r_new is not None and r_old < r_new - threshold
        ),
        "semantic_replacement_degradation": bool(
            (r_mixed is not None and r_sham is not None and r_mixed > r_sham + threshold)
            or (r_mixed is None and r_sham is not None)
        ),
        "destructive_interference": bool(
            r_old is not None and r_new is not None and r_mixed is not None
            and r_old <= r_new + threshold
            and (r_sham is None or r_sham <= r_new + threshold)
            and r_mixed > max(r_new, r_old) + threshold
        ),
        "complementarity": bool(
            r_mixed is not None and r_new is not None and r_old is not None
            and r_mixed < min(r_new, r_old) - threshold
        ),
        "adaptive_video_budget_confound": not selected_video_count_parity,
    }

    if findings["adaptive_video_budget_confound"]:
        primary_verdict = "ADAPTIVE_VIDEO_BUDGET_CONFOUND"
    elif findings["destructive_interference"]:
        primary_verdict = "DESTRUCTIVE_INTER_GROUP_INTERFERENCE"
    elif findings["ensemble_mechanics_effect"] and findings["semantic_replacement_degradation"]:
        primary_verdict = "COMPOUND_MECHANICS_AND_REPLACEMENT_DEGRADATION"
    elif findings["ensemble_mechanics_effect"]:
        primary_verdict = "ENSEMBLE_MECHANICS_CONFOUND"
    elif findings["old_wording_weaker_than_new"] and findings["semantic_replacement_degradation"]:
        primary_verdict = "WEAK_OLD_GROUP_DILUTION_SUPPORTED"
    elif findings["complementarity"]:
        primary_verdict = "ENSEMBLE_COMPLEMENTARITY"
    else:
        primary_verdict = "INCONCLUSIVE"

    contrasts = {
        "ranks": {
            "C_NEW_SINGLE": r_new,
            "C_OLD_SINGLE": r_old,
            "C_NEW_DUP_SHAM": r_sham,
            "C_NEW_OLD_50_50": r_mixed,
        },
        "reciprocal_ranks": {
            "C_NEW_SINGLE": round(rr_new, 6),
            "C_OLD_SINGLE": round(rr_old, 6),
            "C_NEW_DUP_SHAM": round(rr_sham, 6),
            "C_NEW_OLD_50_50": round(rr_mixed, 6),
        },
        "rank_deltas": {
            "delta_mechanics_sham_minus_new": delta_mechanics,
            "delta_wording_old_minus_new": delta_wording,
            "delta_replacement_mixed_minus_sham": delta_replacement,
        },
        "reciprocal_rank_deltas": {
            "delta_rr_mechanics": round(delta_rr_mechanics, 6),
            "delta_rr_wording": round(delta_rr_wording, 6),
            "delta_rr_replacement": round(delta_rr_replacement, 6),
        },
        "findings": findings,
        "primary_verdict": primary_verdict,
    }

    return contrasts, primary_verdict


def run_phase_c1_four_way_ablation(
    query_manifest_path: Path,
    input_root: Path,
    manifest_cache_path: Path,
    output_root: Path,
    tnew_sidecar_path: Path,
    told_sidecar_path: Path,
    sham_sidecar_path: Path,
    paraphrase_sidecar_path: Path,
    manual_ref_path: Path,
    strict_corpus_gate: bool = True,
    device: str = "cpu",
    allow_model_download: bool = False,
) -> dict[str, Any]:
    """Execute Phase C.1 Four-Way Controlled Ablation Benchmark."""
    start_time_all = time.time()
    git_sha = get_git_commit_sha()
    print("=" * 110, flush=True)
    print("🔬 KIS V2-A.3 PHASE C.1: FOUR-WAY CONTROLLED ABLATION BENCHMARK", flush=True)
    print(f"Git Commit: {git_sha}", flush=True)
    print("=" * 110, flush=True)

    # 1. Provenance & Sidecar Validation
    tnew_sha = canonical_sidecar_sha256(tnew_sidecar_path)
    if tnew_sha.lower() != CANONICAL_TNEW_SHA256.lower():
        raise AssertionError(f"T-New Sidecar SHA mismatch: expected {CANONICAL_TNEW_SHA256}, got {tnew_sha}")
    print(f"✅ Canonical T-New Sidecar SHA256 verified: {tnew_sha}", flush=True)

    told_sha = canonical_sidecar_sha256(told_sidecar_path)
    if told_sha.lower() != CANONICAL_TOLD_SHA256.lower():
        raise AssertionError(f"T-Old Sidecar SHA mismatch: expected {CANONICAL_TOLD_SHA256}, got {told_sha}")
    print(f"✅ Canonical T-Old Sidecar SHA256 verified: {told_sha}", flush=True)

    sham_sha = canonical_sidecar_sha256(sham_sidecar_path)
    if sham_sha.lower() != CANONICAL_SHAM_SHA256.lower():
        raise AssertionError(f"Sham Sidecar SHA mismatch: expected {CANONICAL_SHAM_SHA256}, got {sham_sha}")
    print(f"✅ Canonical Sham Sidecar SHA256 verified: {sham_sha}", flush=True)

    para_sha = canonical_sidecar_sha256(paraphrase_sidecar_path)
    if para_sha.lower() != CANONICAL_PARAPHRASE_SHA256.lower():
        raise AssertionError(f"Paraphrase Sidecar SHA mismatch: expected {CANONICAL_PARAPHRASE_SHA256}, got {para_sha}")
    print(f"✅ Canonical Paraphrase Sidecar SHA256 verified: {para_sha}", flush=True)

    qm_sha = canonical_sidecar_sha256(query_manifest_path)
    if qm_sha.lower() != CANONICAL_QUERY_MANIFEST_SHA256.lower():
        raise AssertionError(f"Query Manifest canonical JSON SHA mismatch: expected {CANONICAL_QUERY_MANIFEST_SHA256}, got {qm_sha}")
    print(f"✅ Query Manifest canonical JSON SHA256 verified: {qm_sha}", flush=True)

    mr_sha = canonical_sidecar_sha256(manual_ref_path)
    if mr_sha.lower() != CANONICAL_MANUAL_REF_SHA256.lower():
        raise AssertionError(f"Manual Reference canonical JSON SHA mismatch: expected {CANONICAL_MANUAL_REF_SHA256}, got {mr_sha}")
    print(f"✅ Manual Reference canonical JSON SHA256 verified: {mr_sha}", flush=True)

    # 2. Build Dedicated Providers for Each Arm
    tnew_provider = ImmutableSidecarTranslationProvider(
        sidecar_path=tnew_sidecar_path,
        expected_content_sha256=tnew_sha,
    )
    told_provider = ImmutableSidecarTranslationProvider(
        sidecar_path=told_sidecar_path,
        expected_content_sha256=told_sha,
    )
    sham_provider = ImmutableParaphraseEnsembleSidecarProvider(
        sidecar_path=sham_sidecar_path,
        expected_content_sha256=sham_sha,
    )
    paraphrase_provider = ImmutableParaphraseEnsembleSidecarProvider(
        sidecar_path=paraphrase_sidecar_path,
        expected_content_sha256=para_sha,
    )

    providers: dict[str, Any] = {
        "C_NEW_SINGLE": tnew_provider,
        "C_OLD_SINGLE": told_provider,
        "C_NEW_DUP_SHAM": sham_provider,
        "C_NEW_OLD_50_50": paraphrase_provider,
    }

    # Load queries and ground truth
    queries_data = json.loads(query_manifest_path.read_text(encoding="utf-8")).get("queries", [])
    gt_ref = load_manual_reference(manual_ref_path)
    print(f"✅ Loaded {len(gt_ref)} human-verified ground truth targets from {manual_ref_path.name}", flush=True)

    arms_def = [
        {
            "name": "C_NEW_SINGLE",
            "description": "Canonical New-only Single Baseline (Current Control)",
            "enable_ensemble": False,
            "mode": "EQUAL_BUDGET",
        },
        {
            "name": "C_OLD_SINGLE",
            "description": "Candidate Old-only Single Baseline",
            "enable_ensemble": False,
            "mode": "EQUAL_BUDGET",
        },
        {
            "name": "C_NEW_DUP_SHAM",
            "description": "Sham Paraphrase Ensemble (2x Canonical New, Quota Split)",
            "enable_ensemble": True,
            "mode": "EQUAL_BUDGET",
        },
        {
            "name": "C_NEW_OLD_50_50",
            "description": "True Paraphrase Ensemble (Canonical New + Candidate Old 50/50)",
            "enable_ensemble": True,
            "mode": "EQUAL_BUDGET",
        },
    ]

    audit_results: dict[str, Any] = {
        "metadata": {
            "title": "KIS V2-A.3 Phase C.1: Four-Way Controlled Ablation Benchmark",
            "scope": "Case-level causal diagnosis for query P1-5 on locked corpus, retrieval stack, and commit.",
            "timestamp": datetime.now(UTC).isoformat(),
            "git_commit_sha": git_sha,
            "device": device,
            "diagnostic_rank_delta_threshold": DIAGNOSTIC_RANK_DELTA_THRESHOLD,
        },
        "provenance": {
            "query_manifest": {"path": str(query_manifest_path), "canonical_sha256": qm_sha},
            "manual_reference": {"path": str(manual_ref_path), "canonical_sha256": mr_sha},
            "sidecar_tnew": {"path": str(tnew_sidecar_path), "canonical_sha256": tnew_sha},
            "sidecar_told": {"path": str(told_sidecar_path), "canonical_sha256": told_sha},
            "sidecar_sham": {"path": str(sham_sidecar_path), "canonical_sha256": sham_sha},
            "sidecar_paraphrase": {"path": str(paraphrase_sidecar_path), "canonical_sha256": para_sha},
        },
        "corpus": {},
        "arms": {},
        "sham_invariants": {},
        "assertions": {},
        "comparisons": {},
        "diagnosis": {},
    }

    corpus_verified = False

    for arm in arms_def:
        arm_name = arm["name"]
        arm_out_dir = output_root / arm_name
        if arm_out_dir.exists():
            shutil.rmtree(arm_out_dir)
        arm_out_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 80, flush=True)
        print(f"🚀 Running Arm: {arm_name} ({arm['description']})", flush=True)
        print("=" * 80, flush=True)

        arm_provider = providers[arm_name]

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
            enable_paraphrase_ensemble=arm["enable_ensemble"],
            paraphrase_ensemble_mode=arm["mode"],
        )

        session_cfg = SessionConfig(
            session_id=f"phase_c1_{arm_name}_{int(time.time())}",
            input_root=input_root,
            manifest_cache=manifest_cache_path,
            output_root=arm_out_dir,
            enable_dynamic_translation=True,
            allow_model_download=allow_model_download,
            device=device,
            rrf_constant=60.0,
            continue_on_request_error=False,
            fail_fast_protocol=True,
            kis_video_first_config=vf_cfg,
        )

        runtime = OperationalKISRuntime.bootstrap(
            config=session_cfg,
            translation_provider=arm_provider,
        )

        try:
            if not corpus_verified:
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

                audit_results["corpus"] = {
                    "dataset_root": str(input_root),
                    "video_count": v_count,
                    "total_rows": r_count,
                    "embedding_dimension": d_count,
                    "fingerprint": fp,
                    "strict_corpus_gate_verified": (v_count == 873 and r_count == 177321 and d_count == 512),
                }
                corpus_verified = True

            arm_query_results: dict[str, Any] = {}

            for q_item in queries_data:
                qid = q_item["query_id"]
                q_vi = q_item.get("query_vi", q_item.get("text", ""))

                print(f"  • Processing Query {qid}: '{q_vi[:50]}...'", flush=True)
                req = QueryRequest(
                    request_id=f"{arm_name}_{qid}",
                    query_id=qid,
                    query_vi=q_vi,
                    output_top_k=100,
                )

                resp = runtime.handle_query(req)
                cand_rel_path = resp["artifacts"]["candidates_json"]
                cand_path = arm_out_dir / cand_rel_path
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
                phase_c_tel = video_first_trace.get("phase_c_telemetry")
                if not phase_c_tel:
                    raise RuntimeError(f"Missing required phase_c_telemetry for query '{qid}' in arm '{arm_name}'!")

                # Coarse nomination rank of target video
                target_coarse_rank = None
                selected_vids_list = video_first_trace.get("selected_videos", [])
                for idx, v_item in enumerate(selected_vids_list, start=1):
                    if v_item.get("video_id") == human_target_vid:
                        target_coarse_rank = idx
                        break

                arm_query_results[qid] = {
                    "query_id": qid,
                    "query_vi": q_vi,
                    "candidates_json_path": str(cand_path),
                    "records": records,
                    "canonical_projection_digest": proj_digest,
                    "human_target_video": human_target_vid,
                    "human_target_video_rank": human_target_video_rank,
                    "human_target_best_score": human_target_best_score,
                    "human_target_count_top100": human_target_count_top100,
                    "target_coarse_rank": target_coarse_rank,
                    "phase_c_telemetry": phase_c_tel,
                }

            audit_results["arms"][arm_name] = arm_query_results
        finally:
            runtime.close()

    # 3. Sham Invariants Verification on Treatment P1-5 (FAIL-CLOSED)
    p15_sham_data = audit_results["arms"]["C_NEW_DUP_SHAM"]["query-p1-5-kis"]
    p15_sham_tel = p15_sham_data["phase_c_telemetry"]

    sham_prov = providers["C_NEW_DUP_SHAM"]
    sham_exp_hashes = sham_prov.expected_group_hashes("query-p1-5-kis") if hasattr(sham_prov, "expected_group_hashes") else {}
    sham_group_a_hash = sham_exp_hashes.get("group_sham_a")
    sham_group_b_hash = sham_exp_hashes.get("group_sham_b")

    hash_match = bool(
        sham_group_a_hash is not None
        and sham_group_a_hash == sham_group_b_hash
        and sham_group_a_hash.lower() == "243b0f915c63"
    )

    group_mass_map = p15_sham_tel.get("normalized_weight_mass_by_group", {})
    if "group_sham_a" not in group_mass_map or "group_sham_b" not in group_mass_map:
        raise AssertionError(f"Fail-closed: group_sham_a or group_sham_b missing from normalized_weight_mass_by_group: {group_mass_map}")

    mass_a = float(group_mass_map["group_sham_a"])
    mass_b = float(group_mass_map["group_sham_b"])
    total_mass = float(p15_sham_tel.get("total_normalized_weight_mass", 0.0))

    if mass_a <= 0.0 or not math.isfinite(mass_a):
        raise AssertionError(f"Fail-closed: mass_a must be positive and finite, got {mass_a}")
    if mass_b <= 0.0 or not math.isfinite(mass_b):
        raise AssertionError(f"Fail-closed: mass_b must be positive and finite, got {mass_b}")
    if total_mass <= 0.0 or not math.isfinite(total_mass):
        raise AssertionError(f"Fail-closed: total_mass must be positive and finite, got {total_mass}")

    mass_parity = bool(
        math.isclose(mass_a, mass_b, rel_tol=1e-5)
        and math.isclose(mass_a, total_mass / 2.0, rel_tol=1e-5)
    )

    # Quota validation per variant
    compiled_grp_count = p15_sham_tel.get("compiled_group_count")
    compiled_var_count = p15_sham_tel.get("compiled_variant_count")
    sem_nominal_budget = p15_sham_tel.get("semantic_nominal_budget")
    req_quotas = p15_sham_tel.get("requested_quota_by_variant", {})
    sem_quotas = [q for vid, q in req_quotas.items() if "vi_local" not in vid]

    quotas_valid = bool(
        compiled_grp_count == 2
        and compiled_var_count == 6
        and sem_nominal_budget == 60
        and len(sem_quotas) == 6
        and sum(sem_quotas) == 60
        and all(q == 10 for q in sem_quotas)
    )

    audit_results["sham_invariants"] = {
        "group_count": compiled_grp_count,
        "compiled_variant_count": compiled_var_count,
        "semantic_nominal_budget": sem_nominal_budget,
        "expected_group_hashes": sham_exp_hashes,
        "hashes_identical_canonical_new": hash_match,
        "normalized_weight_mass_by_group": group_mass_map,
        "group_mass_parity": mass_parity,
        "quotas_valid_10_per_variant": quotas_valid,
        "all_invariants_pass": bool(hash_match and mass_parity and quotas_valid),
    }

    if not audit_results["sham_invariants"]["all_invariants_pass"]:
        raise AssertionError(f"Sham Invariants failed for C_NEW_DUP_SHAM on P1-5: {audit_results['sham_invariants']}")
    print("\n✅ Sham Invariants strictly verified (hash=243b0f915c63, mass=W0/2, quota=10/var, total=60)!", flush=True)

    # 4. Negative Controls Bit-Exact Parity across ALL 4 Arms
    arm_names = [a["name"] for a in arms_def]
    neg_control_assertions: dict[str, Any] = {}
    all_neg_match = True

    for qid in ["query-p1-1-kis", "query-p1-2-kis", "query-p1-4-kis", "query-p1-6-kis"]:
        digests = {aname: audit_results["arms"][aname][qid]["canonical_projection_digest"] for aname in arm_names}
        unique_digests = set(digests.values())
        is_exact = (len(unique_digests) == 1)
        neg_control_assertions[qid] = {
            "digests": digests,
            "bit_exact_parity": is_exact,
        }
        if not is_exact:
            all_neg_match = False
            raise AssertionError(f"Negative control {qid} violation: Digests differ across 4 arms: {digests}")

    audit_results["assertions"]["negative_controls"] = {
        "all_match": all_neg_match,
        "queries": neg_control_assertions,
    }
    print("✅ All Negative Controls strictly verified with 100% Bit-Exact Parity across ALL 4 arms!", flush=True)

    # 5. ALL SIX Pairwise Comparisons on Treatment P1-5
    p15_arms = {aname: audit_results["arms"][aname]["query-p1-5-kis"] for aname in arm_names}

    pairs_to_compare = [
        ("C_NEW_SINGLE", "C_OLD_SINGLE"),
        ("C_NEW_SINGLE", "C_NEW_DUP_SHAM"),
        ("C_NEW_SINGLE", "C_NEW_OLD_50_50"),
        ("C_OLD_SINGLE", "C_NEW_DUP_SHAM"),
        ("C_OLD_SINGLE", "C_NEW_OLD_50_50"),
        ("C_NEW_DUP_SHAM", "C_NEW_OLD_50_50"),
    ]

    pairwise_comparisons: dict[str, Any] = {}
    for a1, a2 in pairs_to_compare:
        pw = compute_pairwise_comparison(p15_arms[a1]["records"], p15_arms[a2]["records"])
        pairwise_comparisons[f"{a1}__vs__{a2}"] = pw

    audit_results["comparisons"]["p1-5_pairwise"] = pairwise_comparisons

    # Selected video count parity check
    sel_counts = {aname: p15_arms[aname]["phase_c_telemetry"]["selected_video_count"] for aname in arm_names}
    sel_parity = (len(set(sel_counts.values())) == 1)

    r_new = p15_arms["C_NEW_SINGLE"]["human_target_video_rank"]
    r_old = p15_arms["C_OLD_SINGLE"]["human_target_video_rank"]
    r_sham = p15_arms["C_NEW_DUP_SHAM"]["human_target_video_rank"]
    r_mixed = p15_arms["C_NEW_OLD_50_50"]["human_target_video_rank"]

    contrasts, primary_verdict = classify_p15_findings(
        r_new=r_new,
        r_old=r_old,
        r_sham=r_sham,
        r_mixed=r_mixed,
        selected_video_count_parity=sel_parity,
    )

    audit_results["diagnosis"]["p1-5"] = contrasts
    audit_results["assertions"]["selected_video_count_parity"] = sel_parity

    # Per-arm summary table for P1-5
    p15_summary_table = {
        aname: {
            "target_rank": p15_arms[aname]["human_target_video_rank"],
            "target_best_score": p15_arms[aname]["human_target_best_score"],
            "target_count_top100": p15_arms[aname]["human_target_count_top100"],
            "target_coarse_rank": p15_arms[aname]["target_coarse_rank"],
            "nominal_semantic_budget": p15_arms[aname]["phase_c_telemetry"]["semantic_nominal_budget"],
            "nominal_vi_budget": p15_arms[aname]["phase_c_telemetry"]["vi_nominal_budget"],
            "total_nominal_budget": p15_arms[aname]["phase_c_telemetry"]["total_nominal_budget"],
            "candidate_count_before_dedup": p15_arms[aname]["phase_c_telemetry"]["candidate_count_before_dedup"],
            "compulsory_extra_count": p15_arms[aname]["phase_c_telemetry"]["compulsory_extra_count"],
            "effective_unique_candidates_after_dedup": p15_arms[aname]["phase_c_telemetry"]["effective_unique_candidate_count_after_dedup"],
            "duplication_rate": round(p15_arms[aname]["phase_c_telemetry"]["duplication_rate"] * 100, 2),
            "selected_video_count": p15_arms[aname]["phase_c_telemetry"]["selected_video_count"],
            "compiled_group_count": p15_arms[aname]["phase_c_telemetry"]["compiled_group_count"],
            "compiled_variant_count": p15_arms[aname]["phase_c_telemetry"]["compiled_variant_count"],
            "text_embedding_count": p15_arms[aname]["phase_c_telemetry"]["text_embedding_count"],
            "logical_similarity_evaluations": p15_arms[aname]["phase_c_telemetry"]["logical_similarity_evaluations"],
            "projection_digest": p15_arms[aname]["canonical_projection_digest"],
        }
        for aname in arm_names
    }
    audit_results["summary_table_p1-5"] = p15_summary_table

    # Write Final Audit JSON
    audit_json_path = output_root / "phase_c1_four_way_ablation_audit.json"
    audit_json_path.write_text(json.dumps(audit_results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n📊 Full Four-Way Audit JSON written to: {audit_json_path}", flush=True)

    total_time = time.time() - start_time_all
    print("\n" + "=" * 110, flush=True)
    print(f"🎯 PHASE C.1 FOUR-WAY ABLATION SUMMARY (Total Time: {total_time:.2f}s)", flush=True)
    print("=" * 110, flush=True)
    print(f"Target Video (P1-5): {p15_arms['C_NEW_SINGLE']['human_target_video']}", flush=True)
    for aname in arm_names:
        row = p15_summary_table[aname]
        r_str = f"#{row['target_rank']}" if row['target_rank'] else "MISS"
        c_str = f"#{row['target_coarse_rank']}" if row['target_coarse_rank'] else "MISS"
        print(
            f"  • {aname:<16}: Top-100 Rank: {r_str:<6} | Coarse Rank: {c_str:<6} | "
            f"Candidates (Pre/Dedup): {row['candidate_count_before_dedup']}/{row['effective_unique_candidates_after_dedup']} | "
            f"Extras: {row['compulsory_extra_count']}",
            flush=True,
        )

    print("-" * 110, flush=True)
    print(f"🔬 Causal Contrasts on P1-5:", flush=True)
    print(f"   • Delta Mechanics (Sham - New)      : {contrasts['rank_deltas']['delta_mechanics_sham_minus_new']}", flush=True)
    print(f"   • Delta Wording (Old - New)         : {contrasts['rank_deltas']['delta_wording_old_minus_new']}", flush=True)
    print(f"   • Delta Replacement (Mixed - Sham)  : {contrasts['rank_deltas']['delta_replacement_mixed_minus_sham']}", flush=True)
    print(f"   • Multi-label Findings              : {json.dumps(contrasts['findings'])}", flush=True)
    print(f"   • PRIMARY VERDICT                   : {primary_verdict}", flush=True)
    print("=" * 110, flush=True)

    return audit_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KIS V2-A.3 Phase C.1 Four-Way Controlled Ablation Benchmark")
    parser.add_argument("--query-manifest", type=Path, required=True, help="Path to frozen query manifest JSON")
    parser.add_argument("--input-root", type=Path, required=True, help="Path to dataset input root")
    parser.add_argument("--manifest-cache", type=Path, required=True, help="Path to pre-generated portable manifest cache JSON")
    parser.add_argument("--output", type=Path, required=True, help="Path to audit output directory")
    parser.add_argument("--tnew-sidecar", type=Path, required=True, help="Path to Canonical T-New sidecar JSON")
    parser.add_argument("--told-sidecar", type=Path, required=True, help="Path to Candidate Old sidecar JSON")
    parser.add_argument("--sham-sidecar", type=Path, required=True, help="Path to Sham Duplicate New sidecar JSON")
    parser.add_argument("--paraphrase-sidecar", type=Path, required=True, help="Path to Paraphrase Ensemble sidecar JSON")
    parser.add_argument("--manual-ref", type=Path, required=True, help="Path to ground truth reference JSON")
    parser.add_argument("--strict-corpus-gate", action="store_true", help="Enforce 873 videos / 177321 rows / 512 dim gate")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Inference device")
    parser.add_argument("--allow-model-download", action="store_true", help="Allow CLIP weights download if missing")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    args = parse_args()
    run_phase_c1_four_way_ablation(
        query_manifest_path=args.query_manifest,
        input_root=args.input_root,
        manifest_cache_path=args.manifest_cache,
        output_root=args.output,
        tnew_sidecar_path=args.tnew_sidecar,
        told_sidecar_path=args.told_sidecar,
        sham_sidecar_path=args.sham_sidecar,
        paraphrase_sidecar_path=args.paraphrase_sidecar,
        manual_ref_path=args.manual_ref,
        strict_corpus_gate=args.strict_corpus_gate,
        device=args.device,
        allow_model_download=args.allow_model_download,
    )


if __name__ == "__main__":
    main()
