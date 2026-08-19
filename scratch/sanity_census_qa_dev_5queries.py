#!/usr/bin/env python3
"""5-Query Targeted Sanity Census Test with Full Post-Evidence Tracking.

Evaluates 5 benchmark queries:
  - QA-10: Micro-offset strict hit control (expected post-evidence frame 28135 -> STRICT_HIT @88)
  - QA-23: S2E1 rescue strict hit control (expected post-evidence frame 29018 -> STRICT_HIT @63)
  - QA-08: Downstream temporal offset strict hit control (expected post-evidence frame 552 -> STRICT_HIT @43)
  - QA-02: True temporal miss negative control (expected no physical frames in GT -> TEMPORAL_MISS)
  - QA-34: Early unsupported bail-out control (expected N=0 -> UNSUPPORTED_OR_ERROR)
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


def run_5query_sanity_census() -> None:
    benchmark_path = REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "l21_150_diagnostic" / "benchmark.json"
    sidecar_path = REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "l21_150_diagnostic" / "qa_dev_translations_en.json"

    bm_data = json.loads(benchmark_path.read_text(encoding="utf-8"))
    sidecar_data = json.loads(sidecar_path.read_text(encoding="utf-8"))

    en_map = {e["query_id"]: e.get("question_en", "") for e in sidecar_data.get("entries", [])}
    target_qids = ["QA-10", "QA-23", "QA-08", "QA-02", "QA-34"]
    qa_queries = [q for q in bm_data["queries"] if q["query_id"] in target_qids]
    qa_queries.sort(key=lambda q: target_qids.index(q["query_id"]))

    print("=" * 135)
    print("5-QUERY TARGETED SANITY CENSUS TEST (FULL POST-EVIDENCE & TUPLE PROVENANCE)")
    print(f"Queries to evaluate: {', '.join(target_qids)}")
    print("=" * 135)

    session_output = Path("/kaggle/working/output/sanity_census_5") if Path("/kaggle/working").exists() else REPO_ROOT / "scratch" / "sanity_census_5"
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
            request_id=f"sanity5-{qid}",
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

        # Stage 1: Retrieval Telemetry
        selected_vids = diag.get("selected_video_ids", [])
        target_selected = (target_vid in selected_vids)
        target_nom_rank = selected_vids.index(target_vid) + 1 if target_selected else None

        # Stage 2: Pre-Evidence Frames (Seeds & Refined)
        seed_cands = diag.get("temporal_seed_candidates", [])
        target_seeds = [s for s in seed_cands if s.get("video_id") == target_vid]
        ref_cands = diag.get("refined_candidates", [])
        target_refined = [r for r in ref_cands if r.get("video_id") == target_vid]

        pre_ev_fids = set()
        for s in target_seeds:
            if s.get("frame_id") is not None:
                pre_ev_fids.add(int(s["frame_id"]))
        for r in target_refined:
            if r.get("refined_frame_id") is not None:
                pre_ev_fids.add(int(r["refined_frame_id"]))
            if r.get("candidate_frame_id") is not None:
                pre_ev_fids.add(int(r["candidate_frame_id"]))

        # Stage 3: Usable Evidence Frames
        usable_cands = diag.get("usable_evidence_candidates", [])
        target_usable = [e for e in usable_cands if e.get("video_id") == target_vid]
        usable_fids = {int(e["frame_id"]) for e in target_usable if e.get("frame_id") is not None}

        # Stage 4: Post-Evidence Materialized Candidate Tuples
        post_ev_tuples: list[dict[str, Any]] = []
        for p in preds:
            post_ev_tuples.append({
                "video_id": p.video_id,
                "frame_id": int(p.frame_id),
                "answer": str(p.answer),
                "rank": int(p.rank),
            })

        target_post_ev_tuples = [t for t in post_ev_tuples if t["video_id"] == target_vid]
        target_post_ev_fids = {t["frame_id"] for t in target_post_ev_tuples}

        all_target_physical_fids = pre_ev_fids | usable_fids | target_post_ev_fids

        # Check Ground Truth Intersections across all 4 stages
        pre_ev_in_gt = [f for f in sorted(pre_ev_fids) if start_f <= f <= end_f]
        usable_in_gt = [f for f in sorted(usable_fids) if start_f <= f <= end_f]
        post_ev_in_gt = [f for f in sorted(target_post_ev_fids) if start_f <= f <= end_f]
        any_in_gt = [f for f in sorted(all_target_physical_fids) if start_f <= f <= end_f]

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

        # Check BOUND Tuple (Target Video + in-GT physical frame + exact gold answer) in Materialized Candidates
        bound_exact_ans_exists = False
        matching_bound_answer = None
        bound_in_gt_answers_sample = []

        for t in target_post_ev_tuples:
            t_fid = t["frame_id"]
            t_ans = t["answer"]
            if start_f <= t_fid <= end_f:
                bound_in_gt_answers_sample.append(t_ans)
                if answer_matches(t_ans, accepted_answers):
                    bound_exact_ans_exists = True
                    matching_bound_answer = t_ans

        # Mutually Exclusive First-Failure Classification Hierarchy
        label = "UNSPECIFIED"
        causal_detail = ""

        # Tier 0: Strict Hit
        if strict_hit_rank is not None:
            label = "STRICT_HIT"
            causal_detail = f"Strict hit @{strict_hit_rank} on {target_vid} (f={strict_hit_frame}, ans='{strict_hit_answer}')"

        # Tier 1: Unsupported / Early Bail-out / Exception / N=0
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

        # Tier 3: Target Video Selected but No Canonical Records Anywhere
        elif len(all_target_physical_fids) == 0:
            label = "TARGET_VIDEO_NO_EVIDENCE"
            causal_detail = f"Target video nominated @{target_nom_rank}, but 0 physical candidate records created"

        # Tier 4: Temporal Miss (No physical frame anywhere in the entire candidate universe in GT)
        elif len(any_in_gt) == 0:
            label = "TEMPORAL_MISS"
            causal_detail = f"Nominated @{target_nom_rank}, all physical candidate frames {sorted(all_target_physical_fids)} miss GT [{start_f}..{end_f}]"

        # Tier 5: Evidence Selection Miss (In-GT frame existed pre-evidence, but disappeared before post-evidence materialization)
        elif len(pre_ev_in_gt) > 0 and len(post_ev_in_gt) == 0:
            label = "EVIDENCE_SELECTION_MISS"
            causal_detail = f"In-GT frame(s) {pre_ev_in_gt} existed pre-evidence but excluded from post-evidence candidates"

        # Tier 6: Answer Miss (In-GT post-evidence target candidate exists, but bound answer fails answer_matches)
        elif len(post_ev_in_gt) > 0 and not bound_exact_ans_exists:
            label = "ANSWER_MISS"
            causal_detail = f"In-GT target candidate frame(s) {post_ev_in_gt} present, but bound answers {bound_in_gt_answers_sample[:3]} fail answer_matches({accepted_answers})"

        # Tier 7: Allocation Miss (Full bound tuple existed pre-Top100, but displaced from final Top-100)
        elif bound_exact_ans_exists and strict_hit_rank is None:
            label = "ALLOCATION_MISS"
            causal_detail = f"Full bound tuple (video={target_vid}, frame in {post_ev_in_gt}, ans='{matching_bound_answer}') existed pre-Top100 but displaced from final Top-100"

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
            if strict_hit_frame not in all_target_physical_fids:
                raise RuntimeError(f"CENSUS_TELEMETRY_ERROR on {qid}: Strict hit frame {strict_hit_frame} not traceable in target physical candidate universe!")
        else:
            if strict_hit_rank is not None:
                raise RuntimeError(f"CENSUS_TELEMETRY_ERROR on {qid}: Classified as failure '{label}' but strict hit @{strict_hit_rank} exists!")

        record = {
            "query_id": qid,
            "branch": branch,
            "target_vid": target_vid,
            "gt_interval": f"[{start_f}..{end_f}]",
            "target_nom_rank": target_nom_rank,
            "pre_ev_fids": sorted(pre_ev_fids),
            "usable_fids": sorted(usable_fids),
            "post_ev_fids": sorted(target_post_ev_fids),
            "post_ev_in_gt_fids": post_ev_in_gt,
            "any_in_gt_fids": any_in_gt,
            "strict_hit_rank": strict_hit_rank,
            "strict_hit_frame": strict_hit_frame,
            "strict_hit_answer": strict_hit_answer,
            "first_failure_label": label,
            "causal_detail": causal_detail,
        }
        census_records.append(record)

        in_gt_rows = [f"f={t['frame_id']} | '{t['answer']}' | @{t['rank']}" for t in target_post_ev_tuples if start_f <= t['frame_id'] <= end_f]

        print(f"\n[{idx:2d}/5] {qid:<5} | NomRank: {str(target_nom_rank):<4} | Label: {label:<26} | Time: {elapsed:.2f}s")
        print(f"       • Target Video: {target_vid} | GT Interval: [{start_f}..{end_f}]")
        print(f"       • Stage 1 (Pre-Evidence Anchors/Refined) : {sorted(pre_ev_fids)}")
        print(f"       • Stage 2 (Usable Evidence Bank)        : {sorted(usable_fids)}")
        print(f"       • Stage 3 (Post-Evidence Materialized)  : {sorted(target_post_ev_fids)[:10]} ... (Total: {len(target_post_ev_fids)})")
        print(f"       • Stage 4 (In-GT Materialized Rows)     : {in_gt_rows if in_gt_rows else 'NONE'}")
        print(f"       • Final Classification Verdict          : {label} -> {causal_detail}")

    print("\n" + "=" * 135)
    print("5-QUERY TARGETED SANITY CENSUS AUDIT MATRIX")
    print("=" * 135)
    print(f"{'QID':<6} | {'Target':<10} | {'Nom':<5} | {'Pre-GT':<7} | {'Post-GT':<8} | {'Strict Rank':<12} | {'First-Failure Label':<26} | {'Causal Detail'}")
    print("-" * 135)
    for r in census_records:
        pre_gt_str = "YES" if r["pre_ev_fids"] and (start_f <= r["pre_ev_fids"][0] <= end_f) else "NO"
        post_gt_str = "YES" if r["post_ev_in_gt_fids"] else "NO"
        nom_str = f"@{r['target_nom_rank']}" if r["target_nom_rank"] else "-"
        rank_str = f"@{r['strict_hit_rank']}" if r["strict_hit_rank"] else "-"
        print(f"{r['query_id']:<6} | {r['target_vid']:<10} | {nom_str:<5} | {str(len(r['any_in_gt_fids'])>0):<7} | {post_gt_str:<8} | {rank_str:<12} | {r['first_failure_label']:<26} | {r['causal_detail']}")
    print("=" * 135)

    rec_by_qid = {r["query_id"]: r for r in census_records}
    assert rec_by_qid["QA-10"]["first_failure_label"] == "STRICT_HIT", "QA-10 must be STRICT_HIT"
    assert rec_by_qid["QA-10"]["strict_hit_frame"] == 28135, "QA-10 physical frame must be 28135"
    assert 28135 in rec_by_qid["QA-10"]["post_ev_in_gt_fids"], "QA-10 post-evidence frame 28135 must be captured in post_ev_in_gt_fids"

    assert rec_by_qid["QA-23"]["first_failure_label"] == "STRICT_HIT", "QA-23 must be STRICT_HIT"
    assert rec_by_qid["QA-23"]["strict_hit_frame"] == 29018, "QA-23 physical frame must be 29018"
    assert 29018 in rec_by_qid["QA-23"]["post_ev_in_gt_fids"], "QA-23 post-evidence frame 29018 must be captured in post_ev_in_gt_fids"

    assert rec_by_qid["QA-08"]["first_failure_label"] == "STRICT_HIT", "QA-08 must be STRICT_HIT"
    assert rec_by_qid["QA-08"]["strict_hit_frame"] == 552, "QA-08 physical frame must be 552"
    assert 552 in rec_by_qid["QA-08"]["post_ev_in_gt_fids"], "QA-08 post-evidence frame 552 must be captured in post_ev_in_gt_fids"

    assert rec_by_qid["QA-02"]["first_failure_label"] == "TEMPORAL_MISS", "QA-02 must be TEMPORAL_MISS"
    assert len(rec_by_qid["QA-02"]["any_in_gt_fids"]) == 0, "QA-02 must have 0 physical frames in GT"

    assert rec_by_qid["QA-34"]["first_failure_label"] == "UNSUPPORTED_OR_ERROR", "QA-34 must be UNSUPPORTED_OR_ERROR"

    print("\nALL 5/5 MINI-SANITY ASSERTIONS PASSED WITH 100% MATERIALIZED PROVENANCE TRACING ✅")


if __name__ == "__main__":
    run_5query_sanity_census()
