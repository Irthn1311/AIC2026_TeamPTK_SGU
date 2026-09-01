#!/usr/bin/env python3
"""
KIS V2-A.3 PHASE C PARAPHRASE ENSEMBLE & INVARIANT FUSION AUDIT
================================================================================
Scientific Evaluation Protocol:
- Sidecar T-New (C0): scratch/benchmarks/translation_ablation/translation_p1_focus_v2_new.json
  Canonical Content SHA256: 69b76e1f0e47087611a5118546b5278c2e64ca6583907c1bb43b9df09c8e10d2
- Sidecar Paraphrase Ensemble (C1, C2): scratch/benchmarks/translation_ablation/paraphrase_ensemble_p1_focus_v1.json
  Canonical Content SHA256: 1bb2a15e7f55d9b1947552cdd33f5dba52b4316444781ff8d883aa359f163cf2
- Ground Truth Reference: systems/system_tai/benchmarks/manual_kis_reference_v1.json
- Arms:
  * C0: Single baseline translation (T_new), nominal semantic quota 20 per variant, Run G config
  * C1: Strict Equal-Budget Paraphrase Ensemble (B_sem = 20 * |V0|, hierarchical divmod quotas, normalized weight mass)
  * C2: Expanded Candidate-Retention Upper Bound (B_sem = 20 * sum|Vi|, quota 20 per variant, normalized weight mass)
- Negative Controls: P1-1, P1-2, P1-4, P1-6 (N=1 group, strict bit-exact parity verified across C0/C1/C2 and historical Run G)
- Treatment Target: P1-5 (N=2 groups, target L26_V035)
================================================================================
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
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
from system_tai.translation.paraphrase_sidecar_provider import (
    ImmutableParaphraseEnsembleSidecarProvider,
    canonical_sidecar_sha256,
)
from system_tai.translation.sidecar_provider import (
    ImmutableSidecarTranslationProvider,
)

CANONICAL_TNEW_SHA256 = "545bd4a37c57af53713a1d9f382241ef729c287a1817a5671fdc923115b0be2a"
CANONICAL_PARAPHRASE_SHA256 = "1bb2a15e7f55d9b1947552cdd33f5dba52b4316444781ff8d883aa359f163cf2"

HISTORICAL_RUN_G_DIGESTS = {
    "query-p1-1-kis": "73d7185cfd35b032ef29c35a24e6ed6be6e8683423ca2221700e22f281b5bac0",
    "query-p1-2-kis": "05d3af38138cd7ac3b9aa70aab138c23b2136c2f723ca190729d90db8c64b95f",
    "query-p1-4-kis": "65cd8647f4f1009715191abc3e743dd83d1d81c49406a8414d69558415696bc5",
    "query-p1-5-kis": "445c6b0f590b9ecf3a34d3fe499c04a2d399e27e4ca782b5ec87f15ccf688b8f",
    "query-p1-6-kis": "f955bec8d023bdd30bc9f62a21975763c4bbc071366e6c5d0b9a962bc9457084",
}


def float_to_ieee754_hex(val: float) -> str:
    """Convert 64-bit IEEE-754 float to exact uppercase 16-hex string."""
    return f"{struct.unpack('>Q', struct.pack('>d', float(val)))[0]:016X}"


def canonical_projection_digest(candidates: list[dict[str, Any]]) -> str:
    """Compute deterministic SHA256 digest of top-100 (rank, video_id, frame_id, score_bits)."""
    rows = []
    for c in candidates:
        rank = int(c["rank"])
        vid = str(c["video_id"]).strip()
        fid = int(c.get("frame_id", c.get("actual_frame_id", 0)))
        score_val = float(c.get("fusion_score", c.get("score", 0.0)))
        s_hex = float_to_ieee754_hex(score_val)
        rows.append(f"{rank}:{vid}:{fid}:{s_hex}")
    payload = "\n".join(rows).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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


def run_phase_c_audit(
    *,
    manifest_path: Path,
    feature_root: Path,
    output_dir: Path,
    tnew_sidecar_path: Path,
    paraphrase_sidecar_path: Path,
    manual_ref_path: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Sidecar integrity verification
    tnew_sha = canonical_sidecar_sha256(tnew_sidecar_path)
    if tnew_sha != CANONICAL_TNEW_SHA256:
        raise ValueError(
            f"T-New Sidecar SHA256 mismatch: expected {CANONICAL_TNEW_SHA256}, got {tnew_sha}"
        )
    print(f"✅ Canonical T-New Sidecar SHA256 verified: {tnew_sha}")

    para_sha = canonical_sidecar_sha256(paraphrase_sidecar_path)
    if para_sha != CANONICAL_PARAPHRASE_SHA256:
        raise ValueError(
            f"Paraphrase Sidecar SHA256 mismatch: expected {CANONICAL_PARAPHRASE_SHA256}, got {para_sha}"
        )
    print(f"✅ Canonical Paraphrase Ensemble Sidecar SHA256 verified: {para_sha}")

    # 2. Providers
    tnew_provider = ImmutableSidecarTranslationProvider(
        sidecar_path=tnew_sidecar_path,
        expected_content_sha256=CANONICAL_TNEW_SHA256,
    )
    para_provider = ImmutableParaphraseEnsembleSidecarProvider(
        sidecar_path=paraphrase_sidecar_path,
        expected_content_sha256=CANONICAL_PARAPHRASE_SHA256,
    )

    # 3. Ground Truth Reference
    gt_ref = load_manual_reference(manual_ref_path)
    print(f"✅ Loaded {len(gt_ref)} human-verified ground truth targets from {manual_ref_path.name}")

    manifest_raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    queries_data = manifest_raw.get("queries", manifest_raw)

    arms = [
        ("C0_baseline", False, "EQUAL_BUDGET", tnew_provider),
        ("C1_equal_budget_ensemble", True, "EQUAL_BUDGET", para_provider),
        ("C2_expanded_retention_ensemble", True, "EXPANDED_RETENTION", para_provider),
    ]

    audit_results: dict[str, Any] = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "sidecar_tnew_canonical_sha256": tnew_sha,
        "sidecar_paraphrase_canonical_sha256": para_sha,
        "arms": {},
        "comparisons": {},
        "assertions": {},
    }

    for arm_name, enable_ens, ens_mode, provider in arms:
        print(f"\n================================================================================")
        print(f"🚀 Running Arm: {arm_name} (enable_ensemble={enable_ens}, mode={ens_mode})")
        print(f"================================================================================")

        arm_out_dir = output_dir / arm_name
        arm_out_dir.mkdir(parents=True, exist_ok=True)

        # Run G exact feature configuration
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
            enable_paraphrase_ensemble=enable_ens,
            paraphrase_ensemble_mode=ens_mode,
        )

        session_cfg = SessionConfig(
            session_id=f"phase_c_{arm_name}_{int(time.time())}",
            input_root=feature_root,
            reuse_manifest=manifest_path,
            output_root=arm_out_dir,
            enable_dynamic_translation=True,
            allow_model_download=False,
            rrf_constant=60.0,
            kis_video_first_config=vf_cfg,
        )

        runtime = OperationalKISRuntime(
            config=session_cfg,
            translation_provider=provider,
        )

        arm_query_results: dict[str, Any] = {}

        for q_item in queries_data:
            qid = q_item["query_id"]
            q_vi = q_item.get("query_vi", q_item.get("text", ""))

            print(f"  • Query {qid}: '{q_vi[:50]}...'")
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

            # Validate contiguous ranks and unique identities
            ranks = [r["rank"] for r in records]
            if ranks != list(range(1, 101)):
                raise ValueError(f"Ranks for {qid} are not contiguous 1..100: {ranks[:5]}...{ranks[-5:]}")

            identities = [(r["video_id"], r["frame_id"]) for r in records]
            if len(set(identities)) != 100:
                raise ValueError(f"Duplicate candidate identities found for {qid}")

            proj_digest = canonical_projection_digest(records)

            # Evaluate Ground Truth targets
            gt_info = gt_ref.get(qid, {})
            human_target_vid = gt_info.get("human_verified_video_id")
            legacy_target_vid = gt_info.get("legacy_target_video")
            human_intervals = gt_info.get("human_annotated_intervals", [])
            status = gt_info.get("annotation_status", "VIDEO_ONLY_VERIFIED")

            human_target_video_rank = None
            human_target_best_score = None
            human_target_count_top100 = 0
            human_target_interval_rank = None
            legacy_target_video_rank = None

            for r in records:
                vid = r["video_id"]
                fid = r["frame_id"]
                score = float(r["fusion_score"])

                if vid == human_target_vid:
                    human_target_count_top100 += 1
                    if human_target_video_rank is None:
                        human_target_video_rank = r["rank"]
                        human_target_best_score = score
                    if human_intervals:
                        for (f_start, f_end) in human_intervals:
                            if f_start <= fid <= f_end and human_target_interval_rank is None:
                                human_target_interval_rank = r["rank"]

                if vid == legacy_target_vid and legacy_target_video_rank is None:
                    legacy_target_video_rank = r["rank"]

            # Extract telemetry from candidates.json
            video_first_trace = cand_json_data.get("video_first", {})
            phase_c_tel = video_first_trace.get("phase_c_telemetry")
            if not phase_c_tel:
                raise RuntimeError(f"Missing required phase_c_telemetry for query '{qid}' in arm '{arm_name}'!")

            # Validate required telemetry fields
            required_fields = [
                "semantic_nominal_budget", "vi_nominal_budget", "total_nominal_budget",
                "requested_quota_by_variant", "effective_quota_by_variant_video",
                "compulsory_extra_count", "candidate_count_before_dedup",
                "effective_unique_candidate_count_after_dedup", "duplication_rate",
                "selected_video_count", "compiled_group_count", "compiled_variant_count",
                "text_embedding_count", "logical_similarity_evaluations",
            ]
            for f in required_fields:
                if f not in phase_c_tel:
                    raise RuntimeError(f"Missing required telemetry field '{f}' for query '{qid}' in arm '{arm_name}'!")

            arm_query_results[qid] = {
                "query_id": qid,
                "query_vi": q_vi,
                "canonical_projection_digest": proj_digest,
                "human_target_video": human_target_vid,
                "human_target_video_rank": human_target_video_rank,
                "human_target_best_score": human_target_best_score,
                "human_target_count_top100": human_target_count_top100,
                "human_target_interval_rank": human_target_interval_rank,
                "legacy_target_video": legacy_target_vid,
                "legacy_target_video_rank": legacy_target_video_rank,
                "annotation_status": status,
                "phase_c_telemetry": phase_c_tel,
                "records_summary": records[:5],
            }

        audit_results["arms"][arm_name] = arm_query_results

    # 4. Assertions & Parity Checks
    c0 = audit_results["arms"]["C0_baseline"]
    c1 = audit_results["arms"]["C1_equal_budget_ensemble"]
    c2 = audit_results["arms"]["C2_expanded_retention_ensemble"]

    # Historical Run G Bit-Exact Parity Check for C0
    historical_parity: dict[str, Any] = {}
    for qid, exp_digest in HISTORICAL_RUN_G_DIGESTS.items():
        actual_c0_digest = c0[qid]["canonical_projection_digest"]
        matches = (actual_c0_digest == exp_digest)
        historical_parity[qid] = {
            "expected_run_g_digest": exp_digest,
            "c0_actual_digest": actual_c0_digest,
            "historical_run_g_parity": matches,
        }
        if not matches:
            raise AssertionError(
                f"Historical Run G parity failure for {qid}: expected {exp_digest}, got {actual_c0_digest}"
            )
    audit_results["assertions"]["historical_run_g_parity"] = historical_parity
    print("\n✅ Arm C0 matches historical Run G canonical projection 100% BIT-EXACT across all 5 queries!")

    # Negative Controls Bit-Exact Parity across C0 / C1 / C2
    neg_controls = ["query-p1-1-kis", "query-p1-2-kis", "query-p1-4-kis", "query-p1-6-kis"]
    neg_control_assertions: dict[str, Any] = {}

    for qid in neg_controls:
        d0 = c0[qid]["canonical_projection_digest"]
        d1 = c1[qid]["canonical_projection_digest"]
        d2 = c2[qid]["canonical_projection_digest"]
        is_exact = (d0 == d1 == d2)
        neg_control_assertions[qid] = {
            "c0_digest": d0,
            "c1_digest": d1,
            "c2_digest": d2,
            "bit_exact_parity": is_exact,
        }
        if not is_exact:
            raise AssertionError(f"Negative control {qid} violation: C0/C1/C2 digests differ!")

    audit_results["assertions"]["negative_controls"] = neg_control_assertions
    print("✅ All 4 Negative Controls strictly verified with 100% Bit-Exact Parity across C0/C1/C2.")

    # Treatment query P1-5 analysis
    p15_c0_path = list((output_dir / "C0_baseline" / "requests").glob("*p1-5*/candidates.json"))[0]
    p15_c1_path = list((output_dir / "C1_equal_budget_ensemble" / "requests").glob("*p1-5*/candidates.json"))[0]
    p15_c2_path = list((output_dir / "C2_expanded_retention_ensemble" / "requests").glob("*p1-5*/candidates.json"))[0]

    p15_c0_records = json.loads(p15_c0_path.read_text(encoding="utf-8"))["records"]
    p15_c1_records = json.loads(p15_c1_path.read_text(encoding="utf-8"))["records"]
    p15_c2_records = json.loads(p15_c2_path.read_text(encoding="utf-8"))["records"]

    def compute_set_and_rank_metrics(recs_a, recs_b):
        map_a = {(r["video_id"], r["frame_id"]): r for r in recs_a}
        map_b = {(r["video_id"], r["frame_id"]): r for r in recs_b}
        set_a = set(map_a.keys())
        set_b = set(map_b.keys())
        inter = set_a & set_b
        union = set_a | set_b
        jaccard_sim = len(inter) / len(union) if union else 1.0
        jaccard_dist = 1.0 - jaccard_sim
        replaced_ratio = (len(set_a) - len(inter)) / len(set_a) if set_a else 0.0

        rank_shifts = [abs(map_a[k]["rank"] - map_b[k]["rank"]) for k in inter]
        score_deltas = [map_b[k]["fusion_score"] - map_a[k]["fusion_score"] for k in inter]

        return {
            "intersection_count": len(inter),
            "union_count": len(union),
            "jaccard_similarity": round(jaccard_sim, 4),
            "jaccard_distance": round(jaccard_dist, 4),
            "membership_replaced_ratio": round(replaced_ratio, 4),
            "common_candidates_count": len(inter),
            "mean_absolute_rank_shift": round(float(sum(rank_shifts) / len(rank_shifts)), 2) if rank_shifts else 0.0,
            "median_rank_shift": round(float(sorted(rank_shifts)[len(rank_shifts) // 2]), 2) if rank_shifts else 0.0,
            "max_rank_shift": max(rank_shifts) if rank_shifts else 0,
            "mean_score_delta": round(float(sum(score_deltas) / len(score_deltas)), 6) if score_deltas else 0.0,
        }

    p15_comparison = {
        "c0_vs_c1": compute_set_and_rank_metrics(p15_c0_records, p15_c1_records),
        "c0_vs_c2": compute_set_and_rank_metrics(p15_c0_records, p15_c2_records),
        "c1_vs_c2": compute_set_and_rank_metrics(p15_c1_records, p15_c2_records),
        "target_metrics": {
            "human_target_video": "L26_V035",
            "C0_baseline": {
                "target_video_rank": c0["query-p1-5-kis"]["human_target_video_rank"],
                "target_candidate_count": c0["query-p1-5-kis"]["human_target_count_top100"],
                "best_target_score": c0["query-p1-5-kis"]["human_target_best_score"],
            },
            "C1_equal_budget": {
                "target_video_rank": c1["query-p1-5-kis"]["human_target_video_rank"],
                "target_candidate_count": c1["query-p1-5-kis"]["human_target_count_top100"],
                "best_target_score": c1["query-p1-5-kis"]["human_target_best_score"],
            },
            "C2_expanded_retention": {
                "target_video_rank": c2["query-p1-5-kis"]["human_target_video_rank"],
                "target_candidate_count": c2["query-p1-5-kis"]["human_target_count_top100"],
                "best_target_score": c2["query-p1-5-kis"]["human_target_best_score"],
            },
        },
    }
    audit_results["comparisons"]["p1-5"] = p15_comparison

    # Budget and Compulsory Extras Gate Audit
    c0_p15_tel = c0["query-p1-5-kis"]["phase_c_telemetry"]
    c1_p15_tel = c1["query-p1-5-kis"]["phase_c_telemetry"]
    c2_p15_tel = c2["query-p1-5-kis"]["phase_c_telemetry"]

    c0_extra = c0_p15_tel["compulsory_extra_count"]
    c1_extra = c1_p15_tel["compulsory_extra_count"]
    c2_extra = c2_p15_tel["compulsory_extra_count"]

    nominal_parity = (c0_p15_tel["semantic_nominal_budget"] == c1_p15_tel["semantic_nominal_budget"])
    extra_delta = c1_extra - c0_extra
    retention_claim_status = (
        "STRICT_EQUAL_RETENTION_PASS"
        if (nominal_parity and extra_delta == 0)
        else "EQUAL_NOMINAL_ONLY"
    )

    budget_gate_audit = {
        "c0_semantic_nominal_budget": c0_p15_tel["semantic_nominal_budget"],
        "c1_semantic_nominal_budget": c1_p15_tel["semantic_nominal_budget"],
        "c2_semantic_nominal_budget": c2_p15_tel["semantic_nominal_budget"],
        "nominal_budget_parity_c0_c1": nominal_parity,
        "c0_compulsory_extra_count": c0_extra,
        "c1_compulsory_extra_count": c1_extra,
        "c2_compulsory_extra_count": c2_extra,
        "compulsory_extra_delta_c1_c0": extra_delta,
        "retention_claim_verdict": retention_claim_status,
        "candidate_count_before_dedup": {
            "C0": c0_p15_tel["candidate_count_before_dedup"],
            "C1": c1_p15_tel["candidate_count_before_dedup"],
            "C2": c2_p15_tel["candidate_count_before_dedup"],
        },
        "effective_unique_candidates_after_dedup": {
            "C0": c0_p15_tel["effective_unique_candidate_count_after_dedup"],
            "C1": c1_p15_tel["effective_unique_candidate_count_after_dedup"],
            "C2": c2_p15_tel["effective_unique_candidate_count_after_dedup"],
        },
        "duplication_rates": {
            "C0": c0_p15_tel["duplication_rate"],
            "C1": c1_p15_tel["duplication_rate"],
            "C2": c2_p15_tel["duplication_rate"],
        },
        "logical_similarity_evaluations": {
            "C0": c0_p15_tel["logical_similarity_evaluations"],
            "C1": c1_p15_tel["logical_similarity_evaluations"],
            "C2": c2_p15_tel["logical_similarity_evaluations"],
        },
    }
    audit_results["assertions"]["budget_gate_audit"] = budget_gate_audit

    audit_file = output_dir / "phase_c_paraphrase_ensemble_audit.json"
    audit_file.write_text(json.dumps(audit_results, indent=2), encoding="utf-8")
    print(f"\n📊 Full Audit JSON written to: {audit_file}")

    # Output Summary Table
    print("\n" + "=" * 110)
    print("📊 KIS V2-A.3 PHASE C PARAPHRASE ENSEMBLE SUMMARY TABLE")
    print("=" * 110)
    print(f"{'Query ID':<18} | {'Target Video':<12} | {'C0 (Run G)':<10} | {'C1 (Equal)':<10} | {'C2 (Upper)':<10} | {'Parity / Status'}")
    print("-" * 110)
    for q_item in queries_data:
        qid = q_item["query_id"]
        t_vid = gt_ref.get(qid, {}).get("human_verified_video_id", "N/A")
        r0 = c0[qid]["human_target_video_rank"]
        r1 = c1[qid]["human_target_video_rank"]
        r2 = c2[qid]["human_target_video_rank"]
        status_str = "BIT-EXACT ✅" if qid in neg_controls else f"TREATMENT (Jaccard Dist: {p15_comparison['c0_vs_c1']['jaccard_distance']*100:.1f}%)"
        print(f"{qid:<18} | {t_vid:<12} | #{str(r0):<9} | #{str(r1):<9} | #{str(r2):<9} | {status_str}")
    print("-" * 110)
    print(f"Compulsory Extras Gate: {retention_claim_status} (C0 extra={c0_extra}, C1 extra={c1_extra})")
    print("=" * 110)

    return audit_results


def main():
    parser = argparse.ArgumentParser(description="Phase C Paraphrase Ensemble Audit")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("scratch/phase_c_audit"))
    parser.add_argument(
        "--tnew-sidecar",
        type=Path,
        default=REPO_ROOT / "scratch/benchmarks/translation_ablation/translation_p1_focus_v2_new.json",
    )
    parser.add_argument(
        "--paraphrase-sidecar",
        type=Path,
        default=REPO_ROOT / "scratch/benchmarks/translation_ablation/paraphrase_ensemble_p1_focus_v1.json",
    )
    parser.add_argument(
        "--manual-ref",
        type=Path,
        default=REPO_ROOT / "systems/system_tai/benchmarks/manual_kis_reference_v1.json",
    )
    args = parser.parse_args()

    run_phase_c_audit(
        manifest_path=args.manifest,
        feature_root=args.features,
        output_dir=args.output,
        tnew_sidecar_path=args.tnew_sidecar,
        paraphrase_sidecar_path=args.paraphrase_sidecar,
        manual_ref_path=args.manual_ref,
    )


if __name__ == "__main__":
    main()
