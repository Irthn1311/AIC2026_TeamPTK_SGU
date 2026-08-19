#!/usr/bin/env python3
"""Retrieval-Only A/B Diagnostic on 13 Queries (6 Rescue Candidates + 7 Controls).

Compares Video Nomination (Top-16 selected_video_ids) under two configurations:
  - Arm A (Baseline Canonical Champion) : include_vi_variant = False (en_only)
  - Arm B (Bilingual Variant Expansion) : include_vi_variant = True  (en + vi variant RRF fusion)

Evaluates:
  - 6 Rescue Candidates (VIDEO_ABSENT + Provider Ready) : QA-09, QA-22, QA-42, QA-12, QA-29, QA-44
  - 7 Protected Champion Controls                       : QA-08, QA-10, QA-13, QA-23, QA-27, QA-45, QA-46

Promotion Gate:
  1. Protected Controls : 7/7 target video nominations RETAINED in Top-16
  2. Rescue Candidates  : At least 3/6 target videos NEWLY ENTER Top-16
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SYSTEM_TAI_SRC = REPO_ROOT / "systems" / "system_tai" / "src"
if str(SYSTEM_TAI_SRC) not in sys.path:
    sys.path.insert(0, str(SYSTEM_TAI_SRC))

from system_tai.kis.session_engine import OperationalKISRuntime
from system_tai.kis.session_schema import QAQueryRequest, SessionConfig
from system_tai.qa.grounding import (
    QA_CANDIDATE_ORDER_ROUND_ROBIN,
    QAVideoConditionedEvidenceConfig,
)
from system_tai.qa.object_provider import ObjectAnswerProviderConfig
from system_tai.qa.ocr_provider import OCRAnswerProviderConfig
from system_tai.qa.visual_ontology import VisualOntologyConfig


def normalize_text(text: str | None) -> str:
    if text is None:
        return ""
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    without_marks = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return " ".join(without_marks.split())


def resolve_ocr_config() -> OCRAnswerProviderConfig:
    tess_path = shutil.which("tesseract")
    available_langs: list[str] = []
    if tess_path:
        try:
            res = subprocess.run([tess_path, "--list-langs"], capture_output=True, text=True, check=False)
            available_langs = [l.strip() for l in res.stdout.splitlines()[1:] if l.strip()]
        except Exception:
            pass

    desired = ("eng", "vie")
    supported = tuple(l for l in desired if l in available_langs)
    if not supported:
        supported = tuple(available_langs[:2]) if available_langs else ("eng",)

    if not available_langs:
        return OCRAnswerProviderConfig(enabled=False, languages=("eng",))

    return OCRAnswerProviderConfig(
        enabled=True,
        languages=supported,
        evidence_frame_budget=8,
    )


def resolve_visual_ontology_config() -> VisualOntologyConfig:
    candidates = [
        REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "l21_150_diagnostic" / "qa_dev_visual_ontology.json",
        Path("/kaggle/working/AIC2026_TeamPTK_SGU/systems/system_tai/benchmarks/l21_150_diagnostic/qa_dev_visual_ontology.json"),
    ]
    for p in candidates:
        if p.exists():
            return VisualOntologyConfig(
                enabled=True,
                ontology_path=p,
                evidence_frame_budget=100,
                max_active_domains=2,
            )
    return VisualOntologyConfig(enabled=False)


def run_retrieval_ab() -> None:
    benchmark_path = REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "l21_150_diagnostic" / "benchmark.json"
    sidecar_path = REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "l21_150_diagnostic" / "qa_dev_translations_en.json"

    bm_data = json.loads(benchmark_path.read_text(encoding="utf-8"))
    sidecar_data = json.loads(sidecar_path.read_text(encoding="utf-8"))

    en_map = {e["query_id"]: e.get("question_en", "") for e in sidecar_data.get("entries", [])}

    rescue_qids = ["QA-09", "QA-22", "QA-42", "QA-12", "QA-29", "QA-44"]
    control_qids = ["QA-08", "QA-10", "QA-13", "QA-23", "QA-27", "QA-45", "QA-46"]
    target_qids = rescue_qids + control_qids

    qa_queries = [q for q in bm_data["queries"] if q["query_id"] in target_qids]
    qa_queries.sort(key=lambda q: target_qids.index(q["query_id"]))

    print("=" * 145)
    print("RETRIEVAL-ONLY A/B EXPERIMENT: CANONICAL (Arm A: en_only) vs BILINGUAL EXPANSION (Arm B: vi_variant)")
    print(f"Target Cohort: 6 Rescue Candidates ({', '.join(rescue_qids)}) + 7 Controls ({', '.join(control_qids)})")
    print("=" * 145)

    session_output = Path("/kaggle/working/output/retrieval_ab_13") if Path("/kaggle/working").exists() else REPO_ROOT / "scratch" / "retrieval_ab_13"
    if session_output.exists():
        shutil.rmtree(session_output, ignore_errors=True)
    session_output.mkdir(parents=True, exist_ok=True)

    evidence_config = QAVideoConditionedEvidenceConfig(
        enabled=True,
        selected_video_cap=16,
        anchors_per_video=5,
        video_rrf_constant=60.0,
        candidate_ordering_policy=QA_CANDIDATE_ORDER_ROUND_ROBIN,
        preserve_keyframe_evidence=True,
        keyframe_evidence_video_cap=16,
        keyframe_evidence_anchors_per_video=1,
        temporal_refinement_enabled=True,
        temporal_seed_anchors_per_video=2,
        temporal_refinement_video_cap=8,
        temporal_refinement_total_seed_cap=16,
        secondary_temporal_micro_budget=True,
        primary_11_12_micro_coverage=True,
        tier3_primary_first=True,
        tier3_negative_offset_first=True,
        count_far_alt_micro=False,
        top1_secondary_refined_rescue_enabled=True,
        top1_secondary_refined_rescue_span_candidateizer=True,
        top1_secondary_refined_rescue_tail_budget=5,
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
        qa_visual_ontology_config=resolve_visual_ontology_config(),
        qa_ocr_answer_provider_config=resolve_ocr_config(),
        qa_object_answer_provider_config=ObjectAnswerProviderConfig(enabled=False),
        qa_unsupported_provider_fallback=True,
    )

    print("\n--- BOOTSTRAPPING RUNTIME ---")
    t0 = time.time()
    runtime = OperationalKISRuntime.bootstrap(config)
    print(f"Runtime bootstrap completed in {time.time() - t0:.2f}s.")

    results: list[dict[str, Any]] = []

    print("\n--- EXECUTING A/B VIDEO NOMINATION EVALUATION ---")
    for idx, q in enumerate(qa_queries, start=1):
        qid = q["query_id"]
        target_vid = q.get("video_id")
        q_vi = q.get("question_vi", "")
        q_en = en_map.get(qid)
        cohort = "RESCUE_CANDIDATE" if qid in rescue_qids else "PROTECTED_CONTROL"

        # Arm A: Baseline Canonical (include_vi_variant = False)
        req_a = QAQueryRequest(
            request_id=f"ret-a-{qid}",
            query_id=qid,
            event_description=q_vi,
            question=q_vi,
            event_description_en=q_en if q_en else None,
            question_en=None,
            include_vi_variant=False,
            output_top_k=100,
            refine_top_n=3,
        )
        _, _, diag_a = runtime.qa_pipeline.process_qa_query(req_a)
        vids_a = diag_a.get("selected_video_ids", [])
        selected_a = (target_vid in vids_a)
        rank_a = vids_a.index(target_vid) + 1 if selected_a else None

        # Arm B: Bilingual Expansion (include_vi_variant = True)
        req_b = QAQueryRequest(
            request_id=f"ret-b-{qid}",
            query_id=qid,
            event_description=q_vi,
            question=q_vi,
            event_description_en=q_en if q_en else None,
            question_en=None,
            include_vi_variant=True,
            output_top_k=100,
            refine_top_n=3,
        )
        _, _, diag_b = runtime.qa_pipeline.process_qa_query(req_b)
        vids_b = diag_b.get("selected_video_ids", [])
        selected_b = (target_vid in vids_b)
        rank_b = vids_b.index(target_vid) + 1 if selected_b else None

        overlap = len(set(vids_a) & set(vids_b))
        rescued = (not selected_a and selected_b)
        regressed = (selected_a and not selected_b)

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
            "selected_a": selected_a,
            "rank_a": rank_a,
            "selected_b": selected_b,
            "rank_b": rank_b,
            "vids_a": vids_a,
            "vids_b": vids_b,
            "overlap_top16": overlap,
            "rescued": rescued,
            "regressed": regressed,
            "delta_str": delta_str,
        }
        results.append(record)

        print(f"[{idx:2d}/13] {qid:<5} ({cohort:<17}) | Target: {target_vid} | Arm A: {f'@{rank_a}' if rank_a else 'ABSENT':<7} -> Arm B: {f'@{rank_b}' if rank_b else 'ABSENT':<7} | Delta: {delta_str}")

    # ==============================================================================================================
    # A/B SUMMARY MATRIX & PROMOTION GATE EVALUATION
    # ==============================================================================================================
    print("\n" + "=" * 145)
    print("RETRIEVAL A/B EXPERIMENT: DETAILED MATRIX")
    print("=" * 145)
    print(f"{'QID':<6} | {'Cohort':<18} | {'Target':<10} | {'Arm A Rank':<12} | {'Arm B Rank':<12} | {'Top-16 Overlap':<15} | {'Retrieval Status'}")
    print("-" * 145)
    for r in results:
        rank_a_str = f"@{r['rank_a']}" if r["rank_a"] else "ABSENT"
        rank_b_str = f"@{r['rank_b']}" if r["rank_b"] else "ABSENT"
        overlap_str = f"{r['overlap_top16']}/16 ({r['overlap_top16']/16*100:.0f}%)"
        print(f"{r['query_id']:<6} | {r['cohort']:<18} | {r['target_vid']:<10} | {rank_a_str:<12} | {rank_b_str:<12} | {overlap_str:<15} | {r['delta_str']}")
    print("=" * 145)

    control_recs = [r for r in results if r["cohort"] == "PROTECTED_CONTROL"]
    rescue_recs = [r for r in results if r["cohort"] == "RESCUE_CANDIDATE"]

    controls_retained = sum(1 for r in control_recs if r["selected_b"])
    rescued_count = sum(1 for r in rescue_recs if r["rescued"])
    regressed_count = sum(1 for r in control_recs if r["regressed"])

    print("\n" + "=" * 115)
    print("PROMOTION GATE EVALUATION:")
    print("=" * 115)
    print(f"1. Protected Controls Retained : {controls_retained}/7 ({'PASS ✅' if controls_retained == 7 else 'FAIL ❌ - Lost Control'})")
    print(f"2. Rescue Candidates Rescued   : {rescued_count}/6 ({'PASS ✅ (>=3 rescued)' if rescued_count >= 3 else 'INSUFFICIENT ⚠️ (<3 rescued)'})")
    print(f"3. Control Regressions Count   : {regressed_count} (Must be 0)")
    print("-" * 115)

    if controls_retained == 7 and rescued_count >= 3:
        print(">> VERDICT: PROMOTION GATE PASSED 🏆 (Eligible for single frozen QA verification run) <<")
    elif controls_retained < 7:
        print(">> VERDICT: PROMOTION GATE FAILED ❌ (Control lost -> DROP expansion, FREEZE QA, MOVE KIS) <<")
    else:
        print(f">> VERDICT: PROMOTION GATE INSUFFICIENT ⚠️ ({rescued_count}/6 rescued < 3 -> FREEZE QA, MOVE KIS) <<")
    print("=" * 115)


if __name__ == "__main__":
    run_retrieval_ab()
