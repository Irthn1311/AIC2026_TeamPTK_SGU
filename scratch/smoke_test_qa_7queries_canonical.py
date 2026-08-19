#!/usr/bin/env python3
"""Canonical 7-Query QA DEV Promotion Smoke Test & Budget Audit.

Covers the 6 historical protected queries + QA-23 (S2E1 Target Gain):
  - QA-46: Action (@13)
  - QA-13: Object (@18)
  - QA-08: OCR + Visual (@43)
  - QA-27: Visual Color (@49)
  - QA-23: OCR S2E1 Gain (@63)
  - QA-45: Object Fallback (@92)
  - QA-10: Object Secondary Temporal Micro-Offset (@88)

Part 1: QA-10 Evidence Frame Budget Differential (16 vs 32 vs 100)
Part 2: Full 7-Query Canonical Promotion Smoke Test under default evidence_frame_budget=100
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

REPO_ROOT = Path(__file__).resolve().parents[1]
SYSTEM_TAI_SRC = REPO_ROOT / "systems" / "system_tai" / "src"
if str(SYSTEM_TAI_SRC) not in sys.path:
    sys.path.insert(0, str(SYSTEM_TAI_SRC))

from system_tai.kis.session_engine import OperationalKISRuntime
from system_tai.kis.session_schema import QAQueryRequest, SessionConfig
from system_tai.preliminary.matching import answer_matches
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


def resolve_visual_ontology_config(evidence_frame_budget: int = 100) -> VisualOntologyConfig:
    candidates = [
        REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "l21_150_diagnostic" / "qa_dev_visual_ontology.json",
        Path("/kaggle/working/AIC2026_TeamPTK_SGU/systems/system_tai/benchmarks/l21_150_diagnostic/qa_dev_visual_ontology.json"),
    ]
    for p in candidates:
        if p.exists():
            return VisualOntologyConfig(
                enabled=True,
                ontology_path=p,
                evidence_frame_budget=evidence_frame_budget,
                max_active_domains=2,
            )
    return VisualOntologyConfig(enabled=False)


def run_budget_experiment_qa10(bm_data: dict, en_map: dict) -> None:
    print("=" * 115)
    print("PART 1: QA-10 EVIDENCE FRAME BUDGET EXPERIMENT (16 vs 32 vs 100)")
    print("=" * 115)

    q_info = next(q for q in bm_data["queries"] if q.get("query_id") == "QA-10")
    q_vi = q_info["question_vi"]
    q_en = en_map.get("QA-10", "")
    target_vid = "L21_V003"
    s_gt, e_gt = 28100, 28150
    gold_answers = ["xich đu"]

    budgets = [16, 32, 100]
    for b in budgets:
        print(f"\n--- Testing VisualOntology evidence_frame_budget = {b} ---")
        session_out = Path(f"/kaggle/working/test_qa10_b{b}") if Path("/kaggle/working").exists() else REPO_ROOT / "scratch" / f"test_qa10_b{b}"
        if session_out.exists():
            shutil.rmtree(session_out, ignore_errors=True)
        session_out.mkdir(parents=True, exist_ok=True)

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
            manifest_cache=Path("/kaggle/working/manifest_cache.json") if Path("/kaggle/working").exists() else REPO_ROOT / "scratch" / "manifest_cache.json",
            output_root=session_out,
            device="auto",
            allow_model_download=True,
            default_output_top_k=100,
            default_refine_top_n=3,
            qa_video_conditioned_evidence_config=evidence_config,
            qa_visual_ontology_config=resolve_visual_ontology_config(evidence_frame_budget=b),
            qa_ocr_answer_provider_config=resolve_ocr_config(),
            qa_object_answer_provider_config=ObjectAnswerProviderConfig(enabled=False),
            qa_unsupported_provider_fallback=True,
        )

        t0 = time.time()
        runtime = OperationalKISRuntime.bootstrap(config)
        req = QAQueryRequest(
            request_id=f"test-b{b}-QA-10",
            query_id="QA-10",
            event_description=q_vi,
            question=q_vi,
            event_description_en=q_en if q_en else None,
            question_en=None,
            include_vi_variant=False,
            output_top_k=100,
            refine_top_n=3,
        )
        res, timings, diag = runtime.qa_pipeline.process_qa_query(req)
        elapsed = time.time() - t0

        usable_cands = diag.get("usable_evidence_candidates", [])
        target_usable = [c for c in usable_cands if c.get("video_id") == target_vid]
        target_fids = [c.get("frame_id") for c in target_usable]
        has_28165 = (28165 in target_fids)

        preds = res.predictions
        t_preds = [p for p in preds if p.video_id == target_vid]
        t_pred_fids = [p.frame_id for p in t_preds]
        has_28135 = (28135 in t_pred_fids)

        hit_preds = [
            p for p in t_preds
            if s_gt <= p.frame_id <= e_gt and normalize_text(p.answer) in gold_answers
        ]

        hit_str = f"STRICT HIT @{hit_preds[0].rank} (Frame={hit_preds[0].frame_id}, Answer='{hit_preds[0].answer}') ✅" if hit_preds else "NO HIT ❌"
        print(f"  • Usable Evidence Total Count    : {len(usable_cands)}")
        print(f"  • Usable Evidence Frames for {target_vid}: {target_fids}")
        print(f"  • Secondary Refined Frame 28165 Present? : {has_28165}")
        print(f"  • Micro-Offset Frame 28135 Materialized? : {has_28135}")
        print(f"  • QA-10 Official Strict Verdict  : {hit_str} (Latency: {elapsed:.2f}s)")


def run_7query_smoke_test(bm_data: dict, en_map: dict) -> None:
    print("\n" + "=" * 115)
    print("PART 2: CANONICAL 7-QUERY PROMOTION SMOKE TEST (evidence_frame_budget=100)")
    print("=" * 115)

    target_qids = ["QA-46", "QA-13", "QA-08", "QA-27", "QA-23", "QA-45", "QA-10"]
    q_map = {q["query_id"]: q for q in bm_data["queries"] if q["query_id"] in target_qids}

    session_out = Path("/kaggle/working/output_7query_smoke") if Path("/kaggle/working").exists() else REPO_ROOT / "scratch" / "output_7query_smoke"
    if session_out.exists():
        shutil.rmtree(session_out, ignore_errors=True)
    session_out.mkdir(parents=True, exist_ok=True)

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
        output_root=session_out,
        device="auto",
        allow_model_download=True,
        default_output_top_k=100,
        default_refine_top_n=3,
        qa_video_conditioned_evidence_config=evidence_config,
        qa_visual_ontology_config=resolve_visual_ontology_config(evidence_frame_budget=100),
        qa_ocr_answer_provider_config=resolve_ocr_config(),
        qa_object_answer_provider_config=ObjectAnswerProviderConfig(enabled=False),
        qa_unsupported_provider_fallback=True,
    )

    print("\n--- BOOTSTRAPPING RUNTIME FOR 7-QUERY SMOKE ---")
    t0 = time.time()
    runtime = OperationalKISRuntime.bootstrap(config)
    print(f"Runtime bootstrap completed in {time.time() - t0:.2f}s.")

    results_table = []

    for idx, qid in enumerate(target_qids, start=1):
        q = q_map[qid]
        target_vid = q["video_id"]
        start_f, end_f = int(q["proposed_interval"][0]), int(q["proposed_interval"][1])
        accepted_answers = q.get("accepted_answers", [])
        q_vi = q.get("question_vi", "")
        q_en = en_map.get(qid)
        branch = q.get("branch", "")

        t_q0 = time.time()
        req = QAQueryRequest(
            request_id=f"smoke7-{qid}",
            query_id=qid,
            event_description=q_vi,
            question=q_vi,
            event_description_en=q_en if q_en else None,
            question_en=None,
            include_vi_variant=False,
            output_top_k=100,
            refine_top_n=3,
        )
        res = runtime.handle_qa_query(req)
        elapsed = time.time() - t_q0
        preds = res.get("predictions", [])

        hit_rank = None
        hit_frame = None
        hit_ans = None

        for p in preds:
            p_vid = p.get("video_id")
            p_frame = int(p.get("frame_id", -1))
            p_ans = str(p.get("answer", ""))
            p_rank = int(p.get("rank", 101))

            video_match = (p_vid == target_vid)
            frame_match = video_match and (start_f <= p_frame <= end_f)
            ans_match = answer_matches(p_ans, accepted_answers)
            if frame_match and ans_match:
                hit_rank = p_rank
                hit_frame = p_frame
                hit_ans = p_ans
                break

        status = f"STRICT_HIT @{hit_rank}" if hit_rank is not None else "NO HIT ❌"
        print(f"[{idx}/7] {qid:<5} | Branch: {branch:<14} | N={len(preds):<3} | Status: {status:<18} | Time: {elapsed:.2f}s -> Frame={hit_frame} | Ans='{hit_ans}'")

        results_table.append({
            "qid": qid,
            "branch": branch,
            "target_vid": target_vid,
            "gt": f"[{start_f}..{end_f}]",
            "n": len(preds),
            "hit_rank": f"@{hit_rank}" if hit_rank else "-",
            "frame": hit_frame or "-",
            "answer": hit_ans or "-",
            "status": "STRICT HIT ✅" if hit_rank else "NO HIT ❌",
        })

    print("\n" + "=" * 115)
    print("CANONICAL 7-QUERY PROMOTION SMOKE TEST: FINAL AUDIT MATRIX")
    print("=" * 115)
    print(f"{'Query ID':<8} | {'Phân nhánh':<14} | {'N':<4} | {'Hit Rank':<10} | {'Status':<16} | {'Target Video':<12} | {'Frame':<8} | {'Answer'}")
    print("-" * 115)
    for r in results_table:
        print(f"{r['qid']:<8} | {r['branch']:<14} | {r['n']:<4} | {r['hit_rank']:<10} | {r['status']:<16} | {r['target_vid']:<12} | {r['frame']:<8} | {r['answer']}")
    print("=" * 115)


def main() -> None:
    bm_path = REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "l21_150_diagnostic" / "benchmark.json"
    sidecar_path = REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "l21_150_diagnostic" / "qa_dev_translations_en.json"

    bm_data = json.loads(bm_path.read_text(encoding="utf-8"))
    sidecar_data = json.loads(sidecar_path.read_text(encoding="utf-8"))
    en_map = {e["query_id"]: e.get("question_en", "") for e in sidecar_data.get("entries", [])}

    run_budget_experiment_qa10(bm_data, en_map)
    run_7query_smoke_test(bm_data, en_map)


if __name__ == "__main__":
    main()
