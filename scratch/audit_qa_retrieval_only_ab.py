#!/usr/bin/env python3
"""Retrieval-Only A/B Diagnostic on 14 Queries (R3 Generic English Query Expansion).

Compares Video Nomination (Top-16 selected_video_ids) under two configurations:
  - Arm A (Baseline Canonical Champion) : Single literal English query (weight=1.0)
  - Arm B (Existing R3 Generic Expansion): Literal English (weight=1.0) + compact_keywords (weight=0.8) multi-variant RRF fusion

Both arms preserve canonical language policy:
  qa_localization_language_policy = 'en_only'
  include_vi_variant = False

Evaluates:
  - 6 Provider-Ready Rescue Probes : QA-09, QA-22, QA-42, QA-12, QA-29, QA-44
  - 1 Historical Retrieval Control : QA-26 (Positive control for R3 retrieval rescue)
  - 7 Protected Champion Controls  : QA-08, QA-10, QA-13, QA-23, QA-27, QA-45, QA-46

Promotion Gate:
  Gate 0: Arm A reproduces baseline nomination of protected controls
  Gate 1: Arm B retains 7/7 protected target videos in Top-16
  Gate 2: QA-26 target video L21_V009 rescued into Top-16
  Gate 3: At least 3/6 rescue probes newly enter Top-16
"""

from __future__ import annotations

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
from system_tai.kis.session_schema import QueryLanguage, QueryVariant, QueryVariantType, SessionConfig
from system_tai.qa.grounding import (
    QA_CANDIDATE_ORDER_ROUND_ROBIN,
    QAVideoConditionedEvidenceConfig,
    nominate_qa_videos,
)
from system_tai.retrieval.query_decomposition import decompose_query


def run_r3_retrieval_ab() -> None:
    benchmark_path = REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "l21_150_diagnostic" / "benchmark.json"
    sidecar_path = REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "l21_150_diagnostic" / "qa_dev_translations_en.json"

    bm_data = json.loads(benchmark_path.read_text(encoding="utf-8"))
    sidecar_data = json.loads(sidecar_path.read_text(encoding="utf-8"))

    en_map = {e["query_id"]: e.get("question_en", "") for e in sidecar_data.get("entries", [])}

    rescue_qids = ["QA-09", "QA-22", "QA-42", "QA-12", "QA-29", "QA-44"]
    hist_qids = ["QA-26"]
    control_qids = ["QA-08", "QA-10", "QA-13", "QA-23", "QA-27", "QA-45", "QA-46"]
    target_qids = rescue_qids + hist_qids + control_qids

    qa_queries = [q for q in bm_data["queries"] if q["query_id"] in target_qids]
    qa_queries.sort(key=lambda q: target_qids.index(q["query_id"]))

    print("=" * 150)
    print("RETRIEVAL-ONLY A/B EXPERIMENT: CANONICAL (Arm A) vs EXISTING R3 GENERIC EXPANSION (Arm B)")
    print(f"Target Cohort (N={len(qa_queries)}):")
    print(f"  • 6 Rescue Probes        : {', '.join(rescue_qids)}")
    print(f"  • 1 Historical Control   : {', '.join(hist_qids)}")
    print(f"  • 7 Protected Controls   : {', '.join(control_qids)}")
    print("=" * 150)

    session_output = Path("/kaggle/working/output/retrieval_r3_ab") if Path("/kaggle/working").exists() else REPO_ROOT / "scratch" / "retrieval_r3_ab"
    if session_output.exists():
        shutil.rmtree(session_output, ignore_errors=True)
    session_output.mkdir(parents=True, exist_ok=True)

    evidence_config = QAVideoConditionedEvidenceConfig(
        enabled=True,
        selected_video_cap=16,
        anchors_per_video=5,
        video_rrf_constant=60.0,
        candidate_ordering_policy=QA_CANDIDATE_ORDER_ROUND_ROBIN,
    )

    config = SessionConfig(
        input_root=Path("/kaggle/input/datasets") if Path("/kaggle/input/datasets").exists() else Path("/kaggle/input"),
        manifest_cache=Path("/kaggle/working/manifest_cache.json") if Path("/kaggle/working").exists() else REPO_ROOT / "scratch" / "manifest_cache.json",
        output_root=session_output,
        device="auto",
        allow_model_download=True,
        default_output_top_k=100,
        default_refine_top_n=3,
        qa_video_conditioned_evidence_config=evidence_config,
    )

    print("\n--- BOOTSTRAPPING RUNTIME RETRIEVAL COMPONENTS ---")
    t0 = time.time()
    runtime = OperationalKISRuntime.bootstrap(config)
    searcher = runtime.qa_pipeline.video_restricted_searcher
    encoder = runtime.qa_pipeline.shared_encoder
    print(f"Bootstrap completed in {time.time() - t0:.2f}s.")

    results: list[dict[str, Any]] = []

    print("\n--- EXECUTING RETRIEVAL A/B VIDEO NOMINATION ---")
    for idx, q in enumerate(qa_queries, start=1):
        qid = q["query_id"]
        target_vid = q.get("video_id")
        q_vi = q.get("question_vi", "")
        q_en = en_map.get(qid, "")
        base_en = q_en.strip() if q_en else q_vi.strip()

        if qid in control_qids:
            cohort = "PROTECTED_CONTROL"
        elif qid in hist_qids:
            cohort = "HISTORICAL_CONTROL"
        else:
            cohort = "RESCUE_PROBE"

        # Decompose English Query into Literal and Compact Keywords
        variants_obj = decompose_query(query_text_vi=q_vi, query_text_en=base_en)
        decomp_map = dict(variants_obj.as_list())
        lit_text = decomp_map.get("literal", base_en).strip()
        cmp_text = decomp_map.get("compact_keywords", "").strip()

        # Build Arm A (Single Literal English Query)
        v_lit = QueryVariant(
            variant_id=f"{qid}::literal",
            text=lit_text,
            language=QueryLanguage.ENGLISH,
            variant_type=QueryVariantType.ENGLISH_TRANSLATION,
            weight=1.0,
        )
        vec_lit = encoder.encode_texts([lit_text])
        maxima_lit = searcher.search_video_maxima(
            query_ids=[v_lit.variant_id],
            query_vectors=vec_lit,
        )
        noms_lit = nominate_qa_videos(
            variants=[v_lit],
            maxima=maxima_lit,
            config=evidence_config,
        )
        vids_a = [n.video_id for n in noms_lit[:16]]
        selected_a = (target_vid in vids_a)
        rank_a = vids_a.index(target_vid) + 1 if selected_a else None

        # Build Arm B (Literal 1.0 + Compact Keywords 0.8 Multi-Variant Expansion)
        if cmp_text and cmp_text != lit_text:
            v_cmp = QueryVariant(
                variant_id=f"{qid}::compact_keywords",
                text=cmp_text,
                language=QueryLanguage.ENGLISH,
                variant_type=QueryVariantType.ENGLISH_TRANSLATION,
                weight=0.8,
            )
            vec_cmp = encoder.encode_texts([cmp_text])
            maxima_fused = searcher.search_video_maxima(
                query_ids=[v_lit.variant_id, v_cmp.variant_id],
                query_vectors=list(vec_lit) + list(vec_cmp),
            )
            noms_fused = nominate_qa_videos(
                variants=[v_lit, v_cmp],
                maxima=maxima_fused,
                config=evidence_config,
            )
            vids_b = [n.video_id for n in noms_fused[:16]]
        else:
            vids_b = list(vids_a)

        selected_b = (target_vid in vids_b)
        rank_b = vids_b.index(target_vid) + 1 if selected_b else None

        overlap = len(set(vids_a) & set(vids_b))
        rescued = (not selected_a and selected_b)
        regressed = (selected_a and not selected_b)

        entering = [v for v in vids_b if v not in vids_a]
        leaving = [v for v in vids_a if v not in vids_b]

        if rescued:
            delta_str = f"RESCUED 🎯 (@{rank_b})"
        elif regressed:
            delta_str = f"REGRESSED ❌ (was @{rank_a} -> Absent)"
        elif selected_a and selected_b:
            delta_str = f"RETAINED ✅ (@{rank_a} -> @{rank_b})"
        else:
            delta_str = "ABSENT (No change)"

        record = {
            "query_id": qid,
            "cohort": cohort,
            "target_vid": target_vid,
            "lit_text": lit_text,
            "cmp_text": cmp_text,
            "selected_a": selected_a,
            "rank_a": rank_a,
            "selected_b": selected_b,
            "rank_b": rank_b,
            "vids_a": vids_a,
            "vids_b": vids_b,
            "overlap_top16": overlap,
            "entering": entering,
            "leaving": leaving,
            "rescued": rescued,
            "regressed": regressed,
            "delta_str": delta_str,
        }
        results.append(record)

        print(f"[{idx:2d}/14] {qid:<5} ({cohort:<18}) | Target: {target_vid:<8} | Arm A: {f'@{rank_a}' if rank_a else 'ABSENT':<7} -> Arm B: {f'@{rank_b}' if rank_b else 'ABSENT':<7} | Delta: {delta_str}")
        if cmp_text:
            print(f"        • Literal Query : \"{lit_text}\"")
            print(f"        • Compact Query : \"{cmp_text}\"")
            print(f"        • Top-16 Overlap: {overlap}/16 | Entering: {entering} | Leaving: {leaving}")

    # ==============================================================================================================
    # A/B SUMMARY MATRIX & PROMOTION GATE EVALUATION
    # ==============================================================================================================
    print("\n" + "=" * 150)
    print("RETRIEVAL R3 A/B EXPERIMENT: DETAILED AUDIT MATRIX")
    print("=" * 150)
    print(f"{'QID':<6} | {'Cohort':<18} | {'Target':<9} | {'Arm A Rank':<11} | {'Arm B Rank':<11} | {'Top-16 Overlap':<15} | {'Entering Top-16':<18} | {'Retrieval Status'}")
    print("-" * 150)
    for r in results:
        rank_a_str = f"@{r['rank_a']}" if r["rank_a"] else "ABSENT"
        rank_b_str = f"@{r['rank_b']}" if r["rank_b"] else "ABSENT"
        overlap_str = f"{r['overlap_top16']}/16 ({r['overlap_top16']/16*100:.0f}%)"
        ent_str = ", ".join(r["entering"]) if r["entering"] else "-"
        print(f"{r['query_id']:<6} | {r['cohort']:<18} | {r['target_vid']:<9} | {rank_a_str:<11} | {rank_b_str:<11} | {overlap_str:<15} | {ent_str:<18} | {r['delta_str']}")
    print("=" * 150)

    control_recs = [r for r in results if r["cohort"] == "PROTECTED_CONTROL"]
    hist_recs = [r for r in results if r["cohort"] == "HISTORICAL_CONTROL"]
    rescue_recs = [r for r in results if r["cohort"] == "RESCUE_PROBE"]

    # Baseline parity check for Gate 0
    canonical_ranks = {
        "QA-46": 10, "QA-13": 6, "QA-08": 6, "QA-27": 2, "QA-23": 1, "QA-10": 6, "QA-45": 9
    }
    parity_matches = sum(1 for r in control_recs if r["rank_a"] == canonical_ranks.get(r["query_id"]))

    controls_retained = sum(1 for r in control_recs if r["selected_b"])
    qa26_rescued = any(r["selected_b"] for r in hist_recs)
    qa26_rank = hist_recs[0]["rank_b"] if hist_recs and hist_recs[0]["selected_b"] else None
    rescued_count = sum(1 for r in rescue_recs if r["rescued"])
    regressed_count = sum(1 for r in control_recs if r["regressed"])

    print("\n" + "=" * 125)
    print("PROMOTION GATE EVALUATION (R3 GENERIC ENGLISH QUERY EXPANSION):")
    print("=" * 125)
    print(f"GATE 0. Baseline Parity Controls Match (7/7): {parity_matches}/7 ({'PASS ✅' if parity_matches == 7 else 'FAIL ❌ (Parity mismatch on Arm A)'})")
    print(f"GATE 1. Protected Controls Retained (7/7)    : {controls_retained}/7 ({'PASS ✅' if controls_retained == 7 else 'FAIL ❌ - Control Lost'})")
    print(f"GATE 2. QA-26 Historical Rescue (Positive)  : {'PASS ✅ (Nominated @' + str(qa26_rank) + ')' if qa26_rescued else 'FAIL ❌ (L21_V009 Absent)'}")
    print(f"GATE 3. Rescue Probes Rescued (>=3/6)        : {rescued_count}/6 ({'PASS ✅' if rescued_count >= 3 else 'INSUFFICIENT ⚠️ (<3 rescued)'})")
    print(f"Control Regressions Count                    : {regressed_count} (Must be 0)")
    print("-" * 125)

    gate_pass = (parity_matches == 7 and controls_retained == 7 and qa26_rescued and rescued_count >= 3)
    if gate_pass:
        print(">> VERDICT: ALL 4 GATES PASSED 🏆 (Eligible for single frozen QA verification run) <<")
    elif parity_matches < 7:
        print(">> VERDICT: GATE 0 FAILED ❌ (Baseline parity mismatch -> DO NOT interpret Arm B, FREEZE QA, MOVE KIS) <<")
    elif controls_retained < 7:
        print(">> VERDICT: GATE 1 FAILED ❌ (Control lost -> DROP expansion, FREEZE QA, MOVE KIS) <<")
    elif not qa26_rescued:
        print(">> VERDICT: GATE 2 FAILED ❌ (QA-26 positive control not reproduced -> DROP expansion, FREEZE QA, MOVE KIS) <<")
    else:
        print(f">> VERDICT: GATE 3 INSUFFICIENT ⚠️ ({rescued_count}/6 rescued < 3 -> FREEZE QA, MOVE KIS) <<")
    print("=" * 125)


if __name__ == "__main__":
    run_r3_retrieval_ab()
