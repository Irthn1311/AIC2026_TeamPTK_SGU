#!/usr/bin/env python3
"""
KIS V2-A.3 PHASE C PARAPHRASE ENSEMBLE & INVARIANT FUSION AUDIT
================================================================================
Scientific Evaluation Protocol:
- Sidecar: scratch/benchmarks/translation_ablation/paraphrase_ensemble_p1_focus_v1.json
  Canonical Content SHA256: 1bb2a15e7f55d9b1947552cdd33f5dba52b4316444781ff8d883aa359f163cf2
- Arms:
  * C0: Single baseline translation (group_canonical_new), nominal semantic quota 20 per variant
  * C1: Strict Equal-Budget Paraphrase Ensemble (B_sem = 20 * |V0|, hierarchical divmod quotas, normalized weight mass)
  * C2: Expanded Candidate-Retention Upper Bound (B_sem = 20 * sum|Vi|, quota 20 per variant, normalized weight mass)
- Negative Controls: P1-1, P1-2, P1-4, P1-6 (N=1 group, strict bit-exact parity verified across C0/C1/C2)
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

TARGET_GROUND_TRUTH = {
    "query-p1-1-kis": {"target_video": "L26_V016", "gt_frames": [4215, 4290, 4365, 4440, 4515]},
    "query-p1-2-kis": {"target_video": "L26_V017", "gt_frames": [4710, 4785, 4860, 4935, 5010]},
    "query-p1-4-kis": {"target_video": "L26_V029", "gt_frames": [600, 675, 750, 825, 900]},
    "query-p1-5-kis": {"target_video": "L26_V035", "gt_frames": [14925, 15000, 15075, 15150, 15225]},
    "query-p1-6-kis": {"target_video": "L26_V026", "gt_frames": [3900, 3975, 4050, 4125, 4200]},
}

CANONICAL_SIDECAR_SHA256 = "1bb2a15e7f55d9b1947552cdd33f5dba52b4316444781ff8d883aa359f163cf2"


def float_to_ieee754_hex(val: float) -> str:
    """Convert 64-bit IEEE-754 float to exact uppercase 16-hex string."""
    return f"{struct.unpack('>Q', struct.pack('>d', float(val)))[0]:016X}"


def canonical_projection_digest(candidates: list[dict[str, Any]]) -> str:
    """Compute deterministic SHA256 digest of top-100 (rank, video_id, frame_id, score_bits)."""
    rows = []
    for c in candidates:
        rank = int(c["rank"])
        vid = str(c["video_id"]).strip()
        fid = int(c["actual_frame_id"])
        s_hex = float_to_ieee754_hex(float(c["score"]))
        rows.append(f"{rank}:{vid}:{fid}:{s_hex}")
    payload = "\n".join(rows).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def run_phase_c_audit(
    *,
    manifest_path: Path,
    feature_root: Path,
    output_dir: Path,
    sidecar_path: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Sidecar integrity verification
    actual_sha = canonical_sidecar_sha256(sidecar_path)
    if actual_sha != CANONICAL_SIDECAR_SHA256:
        raise ValueError(
            f"Sidecar SHA256 mismatch: expected {CANONICAL_SIDECAR_SHA256}, got {actual_sha}"
        )
    print(f"✅ Canonical Sidecar SHA256 verified: {actual_sha}")

    sidecar_provider = ImmutableParaphraseEnsembleSidecarProvider(
        sidecar_path=sidecar_path,
        expected_content_sha256=CANONICAL_SIDECAR_SHA256,
    )

    manifest_raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    queries_data = manifest_raw.get("queries", manifest_raw)

    arms = [
        ("C0_baseline", False, "EQUAL_BUDGET"),
        ("C1_equal_budget_ensemble", True, "EQUAL_BUDGET"),
        ("C2_expanded_retention_ensemble", True, "EXPANDED_RETENTION"),
    ]

    audit_results: dict[str, Any] = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "sidecar_id": sidecar_provider.sidecar_id,
        "sidecar_canonical_sha256": actual_sha,
        "arms": {},
        "comparisons": {},
        "assertions": {},
    }

    for arm_name, enable_ens, ens_mode in arms:
        print(f"\n================================================================================")
        print(f"🚀 Running Arm: {arm_name} (enable_ensemble={enable_ens}, mode={ens_mode})")
        print(f"================================================================================")

        arm_out_dir = output_dir / arm_name
        arm_out_dir.mkdir(parents=True, exist_ok=True)

        vf_cfg = KISVideoFirstConfig(
            enabled=True,
            v2_adaptive_enabled=True,
            selected_video_cap=32,
            video_nomination_depth=100,
            restricted_frames_per_video_per_variant=20,
            full_query_weight=1.0,
            primary_scene_weight=1.0,
            supporting_attribute_weight=0.35,
            enable_vi_localization_variant=True,
            vi_localization_weight=0.5,
            enable_paraphrase_ensemble=enable_ens,
            paraphrase_ensemble_mode=ens_mode,
        )

        session_cfg = SessionConfig(
            session_id=f"phase_c_{arm_name}_{int(time.time())}",
            manifest_path=manifest_path,
            feature_store_root=feature_root,
            runtime_output_root=arm_out_dir,
            enable_dynamic_translation=True,
            rrf_constant=60.0,
            kis_video_first_config=vf_cfg,
        )

        runtime = OperationalKISRuntime(
            config=session_cfg,
            translation_provider=sidecar_provider,
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

            res = runtime.process_query(req)
            ranked_cands = [
                {
                    "rank": c.rank,
                    "video_id": c.video_id,
                    "actual_frame_id": c.frame_id,
                    "score": float(c.score),
                    "score_bits": float_to_ieee754_hex(c.score),
                    "source": c.source,
                }
                for c in res.ranked_candidates
            ]

            proj_digest = canonical_projection_digest(ranked_cands)

            # Evaluate target metrics
            gt_info = TARGET_GROUND_TRUTH.get(qid, {})
            target_vid = gt_info.get("target_video")
            gt_frames = gt_info.get("gt_frames", [])

            target_video_rank = None
            target_frame_rank = None
            target_frame_id = None
            min_gt_distance = None

            for c in ranked_cands:
                if c["video_id"] == target_vid:
                    if target_video_rank is None:
                        target_video_rank = c["rank"]
                    dist = min(abs(c["actual_frame_id"] - gf) for gf in gt_frames) if gt_frames else 0
                    if min_gt_distance is None or dist < min_gt_distance:
                        min_gt_distance = dist
                        target_frame_rank = c["rank"]
                        target_frame_id = c["actual_frame_id"]

            cands_file = arm_out_dir / f"{qid}_candidates.json"
            cands_file.write_text(json.dumps(ranked_cands, indent=2), encoding="utf-8")

            # Extract telemetry from trace
            trace_obj = res.ranked_candidates[0].diagnostic_metadata if res.ranked_candidates else {}
            phase_c_tel = {}
            if trace_obj:
                phase_c_tel = trace_obj.get("phase_c_telemetry", {})

            arm_query_results[qid] = {
                "query_id": qid,
                "query_vi": q_vi,
                "canonical_projection_digest": proj_digest,
                "target_video": target_vid,
                "target_video_rank": target_video_rank,
                "target_frame_rank": target_frame_rank,
                "target_frame_id": target_frame_id,
                "min_gt_distance": min_gt_distance,
                "candidate_count": len(ranked_cands),
                "phase_c_telemetry": phase_c_tel,
            }

        audit_results["arms"][arm_name] = arm_query_results

    # 4. Comparative Metrics and Invariant Assertions
    c0 = audit_results["arms"]["C0_baseline"]
    c1 = audit_results["arms"]["C1_equal_budget_ensemble"]
    c2 = audit_results["arms"]["C2_expanded_retention_ensemble"]

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
    print("\n✅ All 4 Negative Controls strictly verified with 100% Bit-Exact IEEE-754 Parity across C0/C1/C2.")

    # Treatment query P1-5 comparisons
    p15_c0_cands = json.loads((output_dir / "C0_baseline" / "query-p1-5-kis_candidates.json").read_text(encoding="utf-8"))
    p15_c1_cands = json.loads((output_dir / "C1_equal_budget_ensemble" / "query-p1-5-kis_candidates.json").read_text(encoding="utf-8"))
    p15_c2_cands = json.loads((output_dir / "C2_expanded_retention_ensemble" / "query-p1-5-kis_candidates.json").read_text(encoding="utf-8"))

    def compute_set_metrics(cands_a, cands_b):
        set_a = {(c["video_id"], c["actual_frame_id"]) for c in cands_a}
        set_b = {(c["video_id"], c["actual_frame_id"]) for c in cands_b}
        inter = len(set_a & set_b)
        union = len(set_a | set_b)
        jaccard_sim = inter / union if union > 0 else 1.0
        jaccard_dist = 1.0 - jaccard_sim
        replaced_ratio = (len(set_a) - inter) / len(set_a) if len(set_a) > 0 else 0.0
        return {
            "intersection_count": inter,
            "union_count": union,
            "jaccard_similarity": round(jaccard_sim, 4),
            "jaccard_distance": round(jaccard_dist, 4),
            "membership_replaced_ratio": round(replaced_ratio, 4),
        }

    p15_comparison = {
        "c0_vs_c1": compute_set_metrics(p15_c0_cands, p15_c1_cands),
        "c0_vs_c2": compute_set_metrics(p15_c0_cands, p15_c2_cands),
        "c1_vs_c2": compute_set_metrics(p15_c1_cands, p15_c2_cands),
        "target_video_ranks": {
            "C0_baseline": c0["query-p1-5-kis"]["target_video_rank"],
            "C1_equal_budget": c1["query-p1-5-kis"]["target_video_rank"],
            "C2_expanded_retention": c2["query-p1-5-kis"]["target_video_rank"],
        },
        "target_frame_ranks": {
            "C0_baseline": c0["query-p1-5-kis"]["target_frame_rank"],
            "C1_equal_budget": c1["query-p1-5-kis"]["target_frame_rank"],
            "C2_expanded_retention": c2["query-p1-5-kis"]["target_frame_rank"],
        }
    }
    audit_results["comparisons"]["p1-5"] = p15_comparison

    # Budget and Compulsory Extras Gate Audit
    c0_p15_tel = c0["query-p1-5-kis"]["phase_c_telemetry"]
    c1_p15_tel = c1["query-p1-5-kis"]["phase_c_telemetry"]
    c2_p15_tel = c2["query-p1-5-kis"]["phase_c_telemetry"]

    c0_extra = c0_p15_tel.get("compulsory_extra_count", 0)
    c1_extra = c1_p15_tel.get("compulsory_extra_count", 0)
    c2_extra = c2_p15_tel.get("compulsory_extra_count", 0)

    nominal_parity = (c0_p15_tel.get("semantic_nominal_budget") == c1_p15_tel.get("semantic_nominal_budget"))
    extra_delta = c1_extra - c0_extra
    retention_claim_status = "STRICT_EQUAL_RETENTION_PASS" if (nominal_parity and extra_delta == 0) else "EQUAL_NOMINAL_ONLY"

    budget_gate_audit = {
        "c0_semantic_nominal_budget": c0_p15_tel.get("semantic_nominal_budget"),
        "c1_semantic_nominal_budget": c1_p15_tel.get("semantic_nominal_budget"),
        "c2_semantic_nominal_budget": c2_p15_tel.get("semantic_nominal_budget"),
        "nominal_budget_parity_c0_c1": nominal_parity,
        "c0_compulsory_extra_count": c0_extra,
        "c1_compulsory_extra_count": c1_extra,
        "c2_compulsory_extra_count": c2_extra,
        "compulsory_extra_delta_c1_c0": extra_delta,
        "retention_claim_verdict": retention_claim_status,
        "effective_candidates_before_dedup": {
            "C0": c0_p15_tel.get("candidate_count_before_dedup"),
            "C1": c1_p15_tel.get("candidate_count_before_dedup"),
            "C2": c2_p15_tel.get("candidate_count_before_dedup"),
        },
        "effective_unique_candidates_after_dedup": {
            "C0": c0_p15_tel.get("effective_unique_candidate_count_after_dedup"),
            "C1": c1_p15_tel.get("effective_unique_candidate_count_after_dedup"),
            "C2": c2_p15_tel.get("effective_unique_candidate_count_after_dedup"),
        },
        "duplication_rates": {
            "C0": c0_p15_tel.get("duplication_rate"),
            "C1": c1_p15_tel.get("duplication_rate"),
            "C2": c2_p15_tel.get("duplication_rate"),
        },
        "logical_similarity_evaluations": {
            "C0": c0_p15_tel.get("logical_similarity_evaluations"),
            "C1": c1_p15_tel.get("logical_similarity_evaluations"),
            "C2": c2_p15_tel.get("logical_similarity_evaluations"),
        }
    }
    audit_results["assertions"]["budget_gate_audit"] = budget_gate_audit

    audit_file = output_dir / "phase_c_paraphrase_ensemble_audit.json"
    audit_file.write_text(json.dumps(audit_results, indent=2), encoding="utf-8")
    print(f"\n📊 Audit written to: {audit_file}")

    # Output Summary Table
    print("\n" + "=" * 100)
    print("📊 KIS V2-A.3 PHASE C PARAPHRASE ENSEMBLE SUMMARY")
    print("=" * 100)
    print(f"{'Query ID':<18} | {'Target Video':<12} | {'C0 Rank':<8} | {'C1 Rank':<8} | {'C2 Rank':<8} | {'Parity / Status'}")
    print("-" * 100)
    for qid in queries_data:
        q_name = qid["query_id"]
        t_vid = TARGET_GROUND_TRUTH.get(q_name, {}).get("target_video", "N/A")
        r0 = c0[q_name]["target_video_rank"]
        r1 = c1[q_name]["target_video_rank"]
        r2 = c2[q_name]["target_video_rank"]
        status = "BIT_EXACT ✅" if q_name in neg_controls else f"TREATMENT (Jaccard Dist: {p15_comparison['c0_vs_c1']['jaccard_distance']*100:.1f}%)"
        print(f"{q_name:<18} | {t_vid:<12} | #{str(r0):<7} | #{str(r1):<7} | #{str(r2):<7} | {status}")
    print("-" * 100)
    print(f"Compulsory Extras Gate: {retention_claim_status} (C0 extra={c0_extra}, C1 extra={c1_extra})")
    print("=" * 100)

    return audit_results


def main():
    parser = argparse.ArgumentParser(description="Phase C Paraphrase Ensemble Audit")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("scratch/phase_c_audit"))
    parser.add_argument(
        "--sidecar",
        type=Path,
        default=Path("scratch/benchmarks/translation_ablation/paraphrase_ensemble_p1_focus_v1.json"),
    )
    args = parser.parse_args()

    run_phase_c_audit(
        manifest_path=args.manifest,
        feature_root=args.features,
        output_dir=args.output,
        sidecar_path=args.sidecar,
    )


if __name__ == "__main__":
    main()
