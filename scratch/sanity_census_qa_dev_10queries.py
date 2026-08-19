#!/usr/bin/env python3
"""10-Query Sanity Census Test for QA DEV First-Failure Taxonomy.

Evaluates 10 benchmark queries:
  - 7 Positives (Known Champion Controls): QA-46, QA-13, QA-08, QA-27, QA-23, QA-10, QA-45
  - 3 Known Failure Controls:
      * QA-01 -> Expected: VIDEO_ABSENT
      * QA-02 -> Expected: TEMPORAL_MISS
      * QA-34 -> Expected: UNSUPPORTED_OR_ERROR (Early Bail-out / N=0)

Mutually-Exclusive First-Failure Taxonomy Hierarchy:
  0. STRICT_HIT                : Official strict tuple exists in final Top-100.
  1. UNSUPPORTED_OR_ERROR      : Pipeline bail-out / unsupported / unexecuted retrieval / N=0.
  2. VIDEO_ABSENT              : Retrieval executed, but target video absent from selected_video_ids.
  3. TARGET_VIDEO_NO_EVIDENCE  : Target video selected, but no anchor seeds or refinement records created.
  4. TEMPORAL_MISS             : Target video present, but no physical pre-evidence candidate/refined frame in GT.
  5. EVIDENCE_SELECTION_MISS   : In-GT physical frame existed before evidence selection, but excluded from usable evidence.
  6. ANSWER_MISS               : Usable in-GT target evidence exists, but bound answer hypotheses on in-GT frame fail answer_matches().
  7. ALLOCATION_MISS           : Bound tuple (target video, in-GT frame, exact gold answer) existed pre-Top100, but displaced from final Top-100.
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
from system_tai.quality.l21_150_answers import answer_matches


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


def run_sanity_census() -> None:
    benchmark_path = REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "l21_150_diagnostic" / "benchmark.json"
    sidecar_path = REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "l21_150_diagnostic" / "qa_dev_translations_en.json"

    bm_data = json.loads(benchmark_path.read_text(encoding="utf-8"))
    sidecar_data = json.loads(sidecar_path.read_text(encoding="utf-8"))

    en_map = {e["query_id"]: e.get("question_en", "") for e in sidecar_data.get("entries", [])}
    target_qids = [
        "QA-46", "QA-13", "QA-08", "QA-27", "QA-23", "QA-10", "QA-45",  # 7 Positives
        "QA-01", "QA-02", "QA-34",                                       # 3 Known Failure Controls
    ]
    qa_queries = [q for q in bm_data["queries"] if q["query_id"] in target_qids]
    # Keep target_qids order
    qa_queries.sort(key=lambda q: target_qids.index(q["query_id"]))

    print("=" * 135)
    print("10-QUERY SANITY CENSUS TEST (7 POSITIVE CONTROLS + 3 FAILURE CONTROLS)")
    print(f"Queries to evaluate: {', '.join(target_qids)}")
    print("=" * 135)

    session_output = Path("/kaggle/working/output/sanity_census_10") if Path("/kaggle/working").exists() else REPO_ROOT / "scratch" / "sanity_census_10"
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

    census_records: list[dict[str, Any]] = []

    print("\n--- EXECUTING INFERENCE & CLASSIFYING 10 SANITY QUERIES ---")
    for idx, q in enumerate(qa_queries, start=1):
        qid = q["query_id"]
        target_vid = q.get("video_id")
        start_f, end_f = int(q["proposed_interval"][0]), int(q["proposed_interval"][1])
        accepted_answers = q.get("accepted_answers", [])
        q_vi = q.get("question_vi", "")
        q_en = en_map.get(qid)
        branch = q.get("branch", "")

        t_q0 = time.time()
        req = QAQueryRequest(
            request_id=f"sanity-{qid}",
            query_id=qid,
            event_description=q_vi,
            question=q_vi,
            event_description_en=q_en if q_en else None,
            question_en=None,
            include_vi_variant=False,
            output_top_k=100,
            refine_top_n=3,
        )
        res, timings, diag = runtime.qa_pipeline.process_qa_query(req)
        elapsed = time.time() - t_q0

        q_type_str = res.question_type.value if hasattr(res.question_type, "value") else str(res.question_type)
        preds = res.predictions

        # Telemetry Extraction
        selected_vids = diag.get("selected_video_ids", [])
        target_selected = (target_vid in selected_vids)
        target_nom_rank = selected_vids.index(target_vid) + 1 if target_selected else None

        seed_cands = diag.get("temporal_seed_candidates", [])
        target_seeds = [s for s in seed_cands if s.get("video_id") == target_vid]

        ref_cands = diag.get("refined_candidates", [])
        target_refined = [r for r in ref_cands if r.get("video_id") == target_vid]

        usable_cands = diag.get("usable_evidence_candidates", [])
        target_usable = [e for e in usable_cands if e.get("video_id") == target_vid]

        # Extract Physical Frame Collections
        pre_ev_fids = set()
        for s in target_seeds:
            if s.get("frame_id") is not None:
                pre_ev_fids.add(int(s["frame_id"]))
        for r in target_refined:
            if r.get("refined_frame_id") is not None:
                pre_ev_fids.add(int(r["refined_frame_id"]))
            if r.get("candidate_frame_id") is not None:
                pre_ev_fids.add(int(r["candidate_frame_id"]))

        usable_fids = {int(e["frame_id"]) for e in target_usable if e.get("frame_id") is not None}

        pre_ev_in_gt = [f for f in sorted(pre_ev_fids) if start_f <= f <= end_f]
        usable_in_gt = [f for f in sorted(usable_fids) if start_f <= f <= end_f]

        # Check Official Strict Hit in Predictions
        strict_hit_rank = None
        strict_hit_frame = None
        strict_hit_answer = None

        for p in preds:
            p_vid = p.video_id
            p_frame = int(p.frame_id)
            p_ans = str(p.answer)
            p_rank = int(p.rank)

            video_match = (p_vid == target_vid)
            frame_match = video_match and (start_f <= p_frame <= end_f)
            ans_match = answer_matches(p_ans, accepted_answers)
            if frame_match and ans_match:
                strict_hit_rank = p_rank
                strict_hit_frame = p_frame
                strict_hit_answer = p_ans
                break

        # Check BOUND Tuple (Target Video + in-GT physical frame + exact gold answer) on Usable Evidence
        bound_exact_ans_exists_pre_top100 = False
        bound_full_tuple_pre_top100_exists = False
        bound_answers_sample = []
        matching_bound_answer = None

        for e in target_usable:
            e_fid = int(e.get("frame_id", -1))
            e_in_gt = (start_f <= e_fid <= end_f)
            e_answers = e.get("answers") or ([e.get("answer")] if e.get("answer") else [])
            if e_in_gt:
                for a in e_answers:
                    if a:
                        bound_answers_sample.append(str(a))
                        if answer_matches(str(a), accepted_answers):
                            bound_exact_ans_exists_pre_top100 = True
                            bound_full_tuple_pre_top100_exists = True
                            matching_bound_answer = str(a)

        # Mutually Exclusive First-Failure Classification Hierarchy
        label = "UNSPECIFIED"
        causal_detail = ""

        # Tier 0: Strict Hit
        if strict_hit_rank is not None:
            label = "STRICT_HIT"
            causal_detail = f"Strict hit @{strict_hit_rank} on {target_vid} (f={strict_hit_frame}, ans='{strict_hit_answer}')"

        # Tier 1: Unsupported / Pre-retrieval Bail-out / Pipeline exception / N=0
        elif (
            len(preds) == 0
            or diag.get("unsupported_reason") is not None
            or (len(selected_vids) == 0 and len(preds) == 0)
            or q_type_str == "unsupported"
        ):
            label = "UNSUPPORTED_OR_ERROR"
            reason = diag.get("unsupported_reason") or "EARLY_UNSUPPORTED_BAILOUT"
            causal_detail = f"Runtime bail-out before/without valid predictions (Reason={reason}, N={len(preds)}, QuestionType={q_type_str})"

        # Tier 2: Video Absent (Retrieval ran, but target video absent from selected_video_ids)
        elif not target_selected:
            label = "VIDEO_ABSENT"
            causal_detail = f"Target video '{target_vid}' absent from nominated top-{len(selected_vids)} selected videos"

        # Tier 3: Target Video Selected but No Evidence Records Created
        elif len(target_seeds) == 0 and len(target_refined) == 0 and len(target_usable) == 0:
            label = "TARGET_VIDEO_NO_EVIDENCE"
            causal_detail = f"Target video nominated @{target_nom_rank}, but 0 anchors/seeds/refinements recorded"

        # Tier 4: Temporal Miss (Target records exist, but no physical pre-evidence candidate/refined frame in GT)
        elif len(pre_ev_in_gt) == 0:
            label = "TEMPORAL_MISS"
            causal_detail = f"Nominated @{target_nom_rank}, pre-evidence frames {sorted(pre_ev_fids)} miss GT [{start_f}..{end_f}]"

        # Tier 5: Evidence Selection Miss (In-GT physical frame existed pre-evidence, but excluded from usable evidence)
        elif len(pre_ev_in_gt) > 0 and len(usable_in_gt) == 0:
            label = "EVIDENCE_SELECTION_MISS"
            causal_detail = f"In-GT physical frame(s) {pre_ev_in_gt} generated pre-evidence but excluded/truncated before usable evidence {sorted(usable_fids)}"

        # Tier 6: Answer Miss (Usable in-GT target evidence exists, but bound answer hypotheses on in-GT frame fail answer_matches)
        elif len(usable_in_gt) > 0 and not bound_exact_ans_exists_pre_top100:
            label = "ANSWER_MISS"
            causal_detail = f"Usable in-GT frame(s) {usable_in_gt} present, but bound answers on in-GT frame {bound_answers_sample[:3]} fail answer_matches({accepted_answers})"

        # Tier 7: Allocation Miss (Full bound tuple: target video + in-GT frame + exact answer existed pre-Top100, but displaced from Top-100)
        elif bound_full_tuple_pre_top100_exists and strict_hit_rank is None:
            label = "ALLOCATION_MISS"
            causal_detail = f"Full bound tuple (video={target_vid}, frame in {usable_in_gt}, ans='{matching_bound_answer}') existed pre-Top100 but displaced from final Top-100"

        # Tier 8: Fallback Catch-all
        else:
            label = "UNSUPPORTED_OR_ERROR"
            causal_detail = f"Unclassified failure path (QuestionType={q_type_str}, N={len(preds)})"

        # SANITY ASSERTIONS
        if label == "STRICT_HIT":
            if not target_selected:
                raise RuntimeError(f"CENSUS_TELEMETRY_ERROR on {qid}: Strict hit achieved but target reported absent!")
            if not (start_f <= strict_hit_frame <= end_f):
                raise RuntimeError(f"CENSUS_TELEMETRY_ERROR on {qid}: Strict hit frame {strict_hit_frame} outside GT [{start_f}..{end_f}]!")
            if not answer_matches(strict_hit_answer, accepted_answers):
                raise RuntimeError(f"CENSUS_TELEMETRY_ERROR on {qid}: Strict hit answer '{strict_hit_answer}' fails answer_matches({accepted_answers})!")
        else:
            if strict_hit_rank is not None:
                raise RuntimeError(f"CENSUS_TELEMETRY_ERROR on {qid}: Classified as failure '{label}' but strict hit @{strict_hit_rank} exists!")

        record = {
            "query_id": qid,
            "branch": branch,
            "q_type": q_type_str,
            "target_vid": target_vid,
            "gt_interval": f"[{start_f}..{end_f}]",
            "target_selected": target_selected,
            "target_nom_rank": target_nom_rank,
            "pre_ev_fids": sorted(pre_ev_fids),
            "usable_fids": sorted(usable_fids),
            "pre_ev_in_gt": len(pre_ev_in_gt) > 0,
            "usable_in_gt": len(usable_in_gt) > 0,
            "bound_exact_ans_pre_top100": bound_exact_ans_exists_pre_top100,
            "bound_full_tuple_pre_top100": bound_full_tuple_pre_top100_exists,
            "strict_hit_rank": strict_hit_rank,
            "strict_hit_frame": strict_hit_frame,
            "strict_hit_answer": strict_hit_answer,
            "first_failure_label": label,
            "causal_detail": causal_detail,
        }
        census_records.append(record)

        print(f"[{idx:2d}/10] {qid:<5} | Branch: {branch:<14} | NomRank: {str(target_nom_rank):<4} | Label: {label:<26} | Time: {elapsed:.2f}s")
        print(f"        -> Detail: {causal_detail}")

    # ==============================================================================================================
    # 10-QUERY AUDIT MATRIX TABLE
    # ==============================================================================================================
    print("\n" + "=" * 135)
    print("10-QUERY SANITY CENSUS AUDIT MATRIX")
    print("=" * 135)
    print(f"{'QID':<6} | {'Phân nhánh':<14} | {'Target':<10} | {'Nom':<5} | {'Pre-GT':<7} | {'Usable-GT':<10} | {'Strict Rank':<12} | {'First-Failure Label':<26} | {'Causal Detail'}")
    print("-" * 135)
    for r in census_records:
        pre_gt_str = "YES" if r["pre_ev_in_gt"] else "NO"
        use_gt_str = "YES" if r["usable_in_gt"] else "NO"
        nom_str = f"@{r['target_nom_rank']}" if r["target_nom_rank"] else "-"
        rank_str = f"@{r['strict_hit_rank']}" if r["strict_hit_rank"] else "-"
        print(f"{r['query_id']:<6} | {r['branch']:<14} | {r['target_vid']:<10} | {nom_str:<5} | {pre_gt_str:<7} | {use_gt_str:<10} | {rank_str:<12} | {r['first_failure_label']:<26} | {r['causal_detail']}")
    print("=" * 135)

    # Sanity checks on known expectations
    rec_by_qid = {r["query_id"]: r for r in census_records}

    # 1. 7 Positives must be STRICT_HIT
    for qid in ["QA-46", "QA-13", "QA-08", "QA-27", "QA-23", "QA-10", "QA-45"]:
        assert rec_by_qid[qid]["first_failure_label"] == "STRICT_HIT", f"Sanity failed: {qid} expected STRICT_HIT, got {rec_by_qid[qid]['first_failure_label']}"

    # 2. Known Failures
    assert rec_by_qid["QA-01"]["first_failure_label"] == "VIDEO_ABSENT", f"Sanity failed: QA-01 expected VIDEO_ABSENT, got {rec_by_qid['QA-01']['first_failure_label']}"
    assert rec_by_qid["QA-02"]["first_failure_label"] == "TEMPORAL_MISS", f"Sanity failed: QA-02 expected TEMPORAL_MISS, got {rec_by_qid['QA-02']['first_failure_label']}"
    assert rec_by_qid["QA-34"]["first_failure_label"] == "UNSUPPORTED_OR_ERROR", f"Sanity failed: QA-34 expected UNSUPPORTED_OR_ERROR, got {rec_by_qid['QA-34']['first_failure_label']}"

    print("\nALL 10/10 SANITY CONTROLS MATCHED EXPECTATIONS EXACTLY (100% PASS) ✅")


if __name__ == "__main__":
    run_sanity_census()
