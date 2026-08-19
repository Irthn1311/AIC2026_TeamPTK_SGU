#!/usr/bin/env python3
"""12-Query Targeted QA Salvage Triage & Near-Miss Diagnostics.

Evaluates the 12 high-ROI candidate queries under the frozen QA DEV Champion
configuration (commit a8d6631):
  - Tier A (Temporal near-miss candidates) : QA-26, QA-30, QA-31
  - Tier B (Simple answer space / Visual) : QA-09, QA-22, QA-42, QA-12, QA-25, QA-29, QA-35, QA-44
  - Tier C (Structural opportunity)      : QA-34

Tracks all 4 candidate stages and computes the exact physical signed distance to GT:
  - Stage 1: Pre-Evidence Anchors/Refined
  - Stage 2: Usable Evidence Bank
  - Stage 3: Post-Evidence Materialized Frames (Constructor offsets & rescue)
  - Stage 4: Provider Answer Hypotheses & Bound In-GT Tuples

Outputs a prioritized Salvageability Ranking (Tier A..E) to guide the next generic sprint.
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


def run_12query_salvage_triage() -> None:
    benchmark_path = REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "l21_150_diagnostic" / "benchmark.json"
    sidecar_path = REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "l21_150_diagnostic" / "qa_dev_translations_en.json"

    bm_data = json.loads(benchmark_path.read_text(encoding="utf-8"))
    sidecar_data = json.loads(sidecar_path.read_text(encoding="utf-8"))

    en_map = {e["query_id"]: e.get("question_en", "") for e in sidecar_data.get("entries", [])}
    target_qids = [
        "QA-26", "QA-30", "QA-31",                                      # Tier A: Near-miss temporal
        "QA-09", "QA-22", "QA-42", "QA-12", "QA-25", "QA-29", "QA-35", "QA-44",  # Tier B: Simple visual/object
        "QA-34",                                                        # Tier C: Structural unsupported
    ]
    qa_queries = [q for q in bm_data["queries"] if q["query_id"] in target_qids]
    qa_queries.sort(key=lambda q: target_qids.index(q["query_id"]))

    print("=" * 145)
    print("RAPID QA SALVAGE: 12-QUERY FIRST-FAILURE & NEAR-MISS TRIAGE")
    print(f"Target Cohort: {', '.join(target_qids)} (N={len(qa_queries)})")
    print("=" * 145)

    session_output = Path("/kaggle/working/output/salvage_triage_12") if Path("/kaggle/working").exists() else REPO_ROOT / "scratch" / "salvage_triage_12"
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

    triage_records: list[dict[str, Any]] = []

    for idx, q in enumerate(qa_queries, start=1):
        qid = q["query_id"]
        target_vid = q.get("video_id")
        start_f, end_f = int(q["proposed_interval"][0]), int(q["proposed_interval"][1])
        accepted_answers = q.get("accepted_answers", [])
        gold_ans = q.get("answer", "")
        q_vi = q.get("question_vi", "")
        q_en = en_map.get(qid)
        branch = q.get("branch", "")

        t_q0 = time.time()
        req = QAQueryRequest(
            request_id=f"triage-{qid}",
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

        # Ground Truth Intersections
        pre_ev_in_gt = [f for f in sorted(pre_ev_fids) if start_f <= f <= end_f]
        usable_in_gt = [f for f in sorted(usable_fids) if start_f <= f <= end_f]
        post_ev_in_gt = [f for f in sorted(target_post_ev_fids) if start_f <= f <= end_f]
        any_in_gt = [f for f in sorted(all_target_physical_fids) if start_f <= f <= end_f]

        # Minimum Physical Distance to GT
        min_dist_to_gt = None
        closest_fid = None
        if all_target_physical_fids:
            dists = []
            for f in all_target_physical_fids:
                if f < start_f:
                    d = start_f - f
                elif f > end_f:
                    d = f - end_f
                else:
                    d = 0
                dists.append((d, f))
            dists.sort()
            min_dist_to_gt, closest_fid = dists[0]

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

        # Check if Gold Answer exists in provider hypotheses
        provider_all_hypotheses = set()
        for t in post_ev_tuples:
            provider_all_hypotheses.add(t["answer"])
        for e in usable_cands:
            for a in e.get("answers", []) or []:
                provider_all_hypotheses.add(str(a))

        gold_in_provider_hypotheses = any(answer_matches(h, accepted_answers) for h in provider_all_hypotheses)

        # Check Bound Tuple (Target Video + in-GT physical frame + exact gold answer) in Materialized Candidates
        bound_exact_ans_exists = False
        matching_bound_answer = None
        for t in target_post_ev_tuples:
            t_fid = t["frame_id"]
            t_ans = t["answer"]
            if start_f <= t_fid <= end_f:
                if answer_matches(t_ans, accepted_answers):
                    bound_exact_ans_exists = True
                    matching_bound_answer = t_ans
                    break

        # Mutually Exclusive First-Failure Classification Hierarchy
        label = "UNSPECIFIED"
        causal_detail = ""

        if strict_hit_rank is not None:
            label = "STRICT_HIT"
            causal_detail = f"Strict hit @{strict_hit_rank} on {target_vid} (f={strict_hit_frame}, ans='{strict_hit_answer}')"
        elif (
            len(preds) == 0
            or diag.get("unsupported_reason") is not None
            or (len(selected_vids) == 0 and len(preds) == 0)
            or q_type_str == "unsupported"
        ):
            label = "UNSUPPORTED_OR_ERROR"
            reason = diag.get("unsupported_reason") or "EARLY_UNSUPPORTED_BAILOUT"
            causal_detail = f"Runtime bail-out before valid predictions (Reason={reason}, N={len(preds)}, QuestionType={q_type_str})"
        elif not target_selected:
            label = "VIDEO_ABSENT"
            causal_detail = f"Target video '{target_vid}' absent from nominated top-{len(selected_vids)} selected videos"
        elif len(all_target_physical_fids) == 0:
            label = "TARGET_VIDEO_NO_EVIDENCE"
            causal_detail = f"Target video nominated @{target_nom_rank}, but 0 physical candidate records created"
        elif len(any_in_gt) == 0:
            label = "TEMPORAL_MISS"
            causal_detail = f"Nominated @{target_nom_rank}, closest frame {closest_fid} is {min_dist_to_gt} frames from GT [{start_f}..{end_f}]"
        elif len(pre_ev_in_gt) > 0 and len(post_ev_in_gt) == 0:
            label = "EVIDENCE_SELECTION_MISS"
            causal_detail = f"In-GT frame(s) {pre_ev_in_gt} existed pre-evidence but excluded from post-evidence candidates"
        elif len(post_ev_in_gt) > 0 and not bound_exact_ans_exists:
            label = "ANSWER_MISS"
            causal_detail = f"In-GT target candidate frame(s) {post_ev_in_gt} present, but bound answers fail answer_matches({accepted_answers})"
        elif bound_exact_ans_exists and strict_hit_rank is None:
            label = "ALLOCATION_MISS"
            causal_detail = f"Full bound tuple (video={target_vid}, frame in {post_ev_in_gt}, ans='{matching_bound_answer}') existed pre-Top100 but displaced from final Top-100"
        else:
            label = "UNSUPPORTED_OR_ERROR"
            causal_detail = f"Unclassified failure path (QuestionType={q_type_str}, N={len(preds)})"

        # Salvageability Tier Determination:
        # A: Valid tuple already exists preTop100 (Allocation miss)
        # B: In-GT materialized frame exists but answer missing (Answer miss / missing provider candidate)
        # C: Target video present and nearest physical candidate is very close (min_dist <= 300 frames ~ 10s)
        # D: Target video present but temporal miss far away (min_dist > 300 frames)
        # E: Video absent / unsupported
        if strict_hit_rank is not None:
            salvage_tier = "ALREADY_HIT"
            salvage_score = 0
        elif label == "ALLOCATION_MISS":
            salvage_tier = "TIER_A (Allocation Miss - Immediate Win)"
            salvage_score = 1
        elif label in ("ANSWER_MISS", "EVIDENCE_SELECTION_MISS") or (len(post_ev_in_gt) > 0 and gold_in_provider_hypotheses):
            salvage_tier = "TIER_B (In-GT Frame Exists - Answer/Selection)"
            salvage_score = 2
        elif target_selected and min_dist_to_gt is not None and min_dist_to_gt <= 300:
            salvage_tier = f"TIER_C (Near-Miss Temporal <=300f: dist={min_dist_to_gt}f)"
            salvage_score = 3
        elif target_selected and min_dist_to_gt is not None and min_dist_to_gt > 300:
            salvage_tier = f"TIER_D (Far-Miss Temporal >300f: dist={min_dist_to_gt}f)"
            salvage_score = 4
        else:
            salvage_tier = f"TIER_E ({label})"
            salvage_score = 5

        record = {
            "query_id": qid,
            "branch": branch,
            "q_type": q_type_str,
            "target_vid": target_vid,
            "gt_interval": f"[{start_f}..{end_f}]",
            "accepted_answers": accepted_answers,
            "target_nom_rank": target_nom_rank,
            "pre_ev_fids": sorted(pre_ev_fids),
            "usable_fids": sorted(usable_fids),
            "post_ev_fids": sorted(target_post_ev_fids),
            "post_ev_in_gt_fids": post_ev_in_gt,
            "any_in_gt_fids": any_in_gt,
            "min_dist_to_gt": min_dist_to_gt,
            "closest_fid": closest_fid,
            "gold_in_provider": gold_in_provider_hypotheses,
            "bound_exact_ans_exists": bound_exact_ans_exists,
            "strict_hit_rank": strict_hit_rank,
            "first_failure_label": label,
            "causal_detail": causal_detail,
            "salvage_tier": salvage_tier,
            "salvage_score": salvage_score,
        }
        triage_records.append(record)

        print(f"\n[{idx:2d}/12] {qid:<5} | NomRank: {str(target_nom_rank):<4} | Label: {label:<26} | Salvage: {salvage_tier} | Time: {elapsed:.2f}s")
        print(f"        • Target: {target_vid} | GT: [{start_f}..{end_f}] | Gold: {accepted_answers}")
        print(f"        • Min Dist to GT: {min_dist_to_gt} frames (Closest Frame: {closest_fid})")
        print(f"        • Gold Answer in Provider Hypotheses?: {gold_in_provider_hypotheses}")
        print(f"        • Pre-Ev: {sorted(pre_ev_fids)} | Usable: {sorted(usable_fids)} | Post-Ev: {sorted(target_post_ev_fids)[:8]}")
        print(f"        • Detail: {causal_detail}")

    # ==============================================================================================================
    # PRIORITIZED SALVAGEABILITY RANKING TABLE
    # ==============================================================================================================
    print("\n" + "=" * 145)
    print("PRIORITIZED SALVAGEABILITY RANKING TABLE (ORDERED BY EASE OF RECOVERY / ROI)")
    print("=" * 145)
    triage_records.sort(key=lambda r: (r["salvage_score"], r["min_dist_to_gt"] if r["min_dist_to_gt"] is not None else 999999))

    print(f"{'Rank':<5} | {'QID':<6} | {'Phân nhánh':<14} | {'Target':<10} | {'Nom':<5} | {'Min Dist':<10} | {'Gold in Prov':<14} | {'First Failure Label':<25} | {'Salvageability Tier'}")
    print("-" * 145)
    for idx, r in enumerate(triage_records, start=1):
        nom_str = f"@{r['target_nom_rank']}" if r["target_nom_rank"] else "-"
        dist_str = f"{r['min_dist_to_gt']}f" if r["min_dist_to_gt"] is not None else "-"
        gold_prov_str = "YES ✅" if r["gold_in_provider"] else "NO ❌"
        print(f"{idx:<5} | {r['query_id']:<6} | {r['branch']:<14} | {r['target_vid']:<10} | {nom_str:<5} | {dist_str:<10} | {gold_prov_str:<14} | {r['first_failure_label']:<25} | {r['salvage_tier']}")
    print("=" * 145)


if __name__ == "__main__":
    run_12query_salvage_triage()
