#!/usr/bin/env python3
"""QA-10 Comprehensive Temporal Parity & Trace Audit.

Inspects every temporal step for QA-10 (L21_V003):
1. Initial video nominations & all keyframe anchors for L21_V003.
2. Temporal seed selection (local_rank 1, local_rank 2, etc.).
3. Refinement execution & refined physical frames.
4. Evidence candidate assembly & filtering.
5. Answer scoring & Top-100 construction.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
import unicodedata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SYSTEM_TAI_SRC = REPO_ROOT / "systems" / "system_tai" / "src"
if str(SYSTEM_TAI_SRC) not in sys.path:
    sys.path.insert(0, str(SYSTEM_TAI_SRC))

from system_tai.kis.session_engine import OperationalKISRuntime
from system_tai.kis.session_schema import QAQueryRequest, SessionConfig
from system_tai.qa.grounding import QAVideoConditionedEvidenceConfig
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
            import subprocess
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
    return OCRAnswerProviderConfig(enabled=True, languages=supported, evidence_frame_budget=8)


def resolve_visual_ontology_config() -> VisualOntologyConfig:
    candidates = [
        REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "l21_150_diagnostic" / "qa_dev_visual_ontology.json",
        Path("/kaggle/working/AIC2026_TeamPTK_SGU/systems/system_tai/benchmarks/l21_150_diagnostic/qa_dev_visual_ontology.json"),
    ]
    for p in candidates:
        if p.exists():
            return VisualOntologyConfig(enabled=True, ontology_path=p, evidence_frame_budget=16, max_active_domains=2)
    return VisualOntologyConfig(enabled=False)


def run_audit() -> None:
    print("=" * 115)
    print("QA-10 COMPREHENSIVE TEMPORAL PARITY & TRACE AUDIT")
    print("=" * 115)

    benchmark_path = REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "l21_150_diagnostic" / "benchmark.json"
    sidecar_path = REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "l21_150_diagnostic" / "qa_dev_translations_en.json"

    bm_data = json.loads(benchmark_path.read_text(encoding="utf-8"))
    sidecar_data = json.loads(sidecar_path.read_text(encoding="utf-8"))
    en_map = {e["query_id"]: e.get("question_en", "") for e in sidecar_data.get("entries", [])}

    q_info = next(q for q in bm_data["queries"] if q.get("query_id") == "QA-10")
    qid = "QA-10"
    target_vid = q_info["video_id"]
    s_gt, e_gt = map(int, q_info["proposed_interval"])
    raw_accepted = q_info.get("accepted_answers") or [q_info.get("answer", "")]
    gold_answers = [normalize_text(a) for a in raw_accepted if a]
    q_vi = q_info.get("question_vi", "")
    q_en = en_map.get(qid, "")

    print(f"Query ID             : {qid}")
    print(f"Target Video         : {target_vid}")
    print(f"Ground Truth Interval: [{s_gt}..{e_gt}]")
    print(f"Gold Answers         : {gold_answers}")
    print(f"Question VI          : {q_vi}")
    print(f"Question EN          : {q_en}")
    print("=" * 115)

    session_output = Path("/kaggle/working/qa10_temporal_output") if Path("/kaggle/working").exists() else REPO_ROOT / "scratch" / "qa10_temporal_output"
    if session_output.exists():
        shutil.rmtree(session_output, ignore_errors=True)
    session_output.mkdir(parents=True, exist_ok=True)

    evidence_config = QAVideoConditionedEvidenceConfig(
        enabled=True,
        selected_video_cap=16,
        anchors_per_video=5,
        video_rrf_constant=60.0,
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
        manifest_cache=Path("/kaggle/working/manifest_cache.json") if Path("/kaggle/working").exists() else Path("scratch/manifest_cache.json"),
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
    runtime = OperationalKISRuntime.bootstrap(config)

    req = QAQueryRequest(
        request_id=f"audit-temporal-{qid}",
        query_id=qid,
        event_description=q_vi,
        question=q_vi,
        event_description_en=q_en if q_en else None,
        question_en=None,
        include_vi_variant=False,
        output_top_k=100,
        refine_top_n=3,
    )

    print("\n--- RUNNING INFERENCE WITH FULL TEMPORAL TELEMETRY ---")
    res, timings, diag = runtime.qa_pipeline.process_qa_query(req)

    # 1. Video Nominations
    print("\n1. GLOBAL VIDEO NOMINATIONS:")
    selected_vids = diag.get("selected_video_ids", [])
    for idx, vid in enumerate(selected_vids, start=1):
        mark = " <--- TARGET VIDEO!" if vid == target_vid else ""
        print(f"   [{idx:02d}] Video: {vid}{mark}")

    # 2. Keyframe Anchors / Temporal Seed Candidates
    print("\n2. ALL TEMPORAL SEED CANDIDATES (Before Refinement):")
    seed_cands = diag.get("temporal_seed_candidates", [])
    target_seeds = [s for s in seed_cands if s.get("video_id") == target_vid]
    print(f"   Total Temporal Seed Candidates: {len(seed_cands)}")
    print(f"   Temporal Seeds for {target_vid}: {len(target_seeds)}")
    for idx, sc in enumerate(seed_cands, start=1):
        vid = sc.get("video_id")
        fid = sc.get("frame_id")
        nom_r = sc.get("video_nomination_rank")
        loc_r = sc.get("local_anchor_rank")
        mark = f" <--- TARGET {target_vid} (local_rank={loc_r})!" if vid == target_vid else ""
        print(f"   [{idx:02d}] Video: {vid:<10} | Frame: {fid:<6} | NomRank: {str(nom_r):<3} | LocalRank: {str(loc_r):<3}{mark}")

    # 3. Refined Candidates Output
    print("\n3. REFINED CANDIDATES OUTPUT (After Refinement):")
    refined_cands = diag.get("refined_candidates", [])
    target_refined = [rc for rc in refined_cands if rc.get("video_id") == target_vid]
    print(f"   Total Refined Candidates: {len(refined_cands)}")
    print(f"   Refined Candidates for {target_vid}: {len(target_refined)}")
    for idx, rc in enumerate(refined_cands, start=1):
        vid = rc.get("video_id")
        orig_f = rc.get("candidate_frame_id")
        ref_f = rc.get("refined_frame_id")
        status = rc.get("status")
        orig_r = rc.get("original_rank")
        in_gt = (s_gt <= ref_f <= e_gt) if ref_f is not None else False
        mark = f" <--- TARGET {target_vid} (In GT={in_gt})!" if vid == target_vid else ""
        print(f"   [{idx:02d}] Video: {vid:<10} | OrigFrame: {orig_f:<6} | RefinedFrame: {str(ref_f):<6} | Status: {status:<8} | OrigRank: {orig_r:<3}{mark}")

    # 4. Usable Evidence Candidates
    print("\n4. USABLE EVIDENCE CANDIDATES (Fed to Answer Provider):")
    usable_cands = diag.get("usable_evidence_candidates", [])
    target_usable = [uc for uc in usable_cands if uc.get("video_id") == target_vid]
    print(f"   Total Usable Evidence: {len(usable_cands)}")
    print(f"   Usable Evidence for {target_vid}: {len(target_usable)}")
    for idx, uc in enumerate(usable_cands, start=1):
        vid = uc.get("video_id")
        fid = uc.get("frame_id")
        src = uc.get("evidence_source")
        r = uc.get("rank")
        in_gt = (s_gt <= fid <= e_gt)
        mark = f" <--- TARGET {target_vid} (In GT={in_gt})!" if vid == target_vid else ""
        print(f"   [{idx:02d}] Video: {vid:<10} | Frame: {fid:<6} | Source: {src:<18} | Rank: {str(r):<3}{mark}")

    # 5. Predictions Summary
    print("\n5. FINAL PREDICTIONS FOR TARGET VIDEO:")
    target_preds = [p for p in res.predictions if p.video_id == target_vid]
    print(f"   Total Predictions for {target_vid}: {len(target_preds)}")
    for p in target_preds:
        fid = p.frame_id
        ans = p.answer
        in_gt = (s_gt <= fid <= e_gt)
        match_a = (normalize_text(ans) in gold_answers)
        is_hit = in_gt and match_a
        print(f"   Rank={p.rank:<3} | Frame={fid:<6} | In GT={str(in_gt):<5} | Answer='{ans}' | STRICT HIT={is_hit}")

    # 6. Failure Taxonomy
    print("\n" + "=" * 115)
    print("AUDIT SUMMARY & FIRST TEMPORAL DIVERGENCE ANALYSIS")
    print("=" * 115)
    if any(s_gt <= p.frame_id <= e_gt and normalize_text(p.answer) in gold_answers for p in target_preds):
        print("RESULT: STRICT HIT ✅")
    else:
        print("RESULT: NO HIT ❌")
        # Check if local_rank=2 anchor was in temporal_seed_candidates
        has_sec_seed = any(s.get("video_id") == target_vid and s.get("local_anchor_rank") == 2 for s in seed_cands)
        has_sec_refined = any(rc.get("video_id") == target_vid and rc.get("candidate_frame_id") != target_seeds[0].get("frame_id") for rc in refined_cands) if target_seeds else False
        has_sec_usable = len(target_usable) > 1

        print(f"  • Video Nomination Rank        : {selected_vids.index(target_vid)+1 if target_vid in selected_vids else 'ABSENT'}")
        print(f"  • Secondary Seed Anchor Present : {has_sec_seed}")
        print(f"  • Secondary Refinement Present  : {has_sec_refined}")
        print(f"  • Secondary Usable Evidence     : {has_sec_usable}")
        print(f"  • Closest Refined Frame to GT   : {min([rc.get('refined_frame_id') or 999999 for rc in target_refined], key=lambda f: min(abs(f-s_gt), abs(f-e_gt))) if target_refined else 'N/A'}")


if __name__ == "__main__":
    run_audit()
