#!/usr/bin/env python3
"""Canonical 4-Query Smoke Test for QA DEV.

Evaluates the exact canonical contract on 4 critical queries:
  - QA-13: Protected Object Hit
  - QA-46: Protected Action Hit
  - QA-45: Historical Fallback Case ("cần cẩu")
  - QA-23: S2E1 Target OCR Positive Control

Contract:
  - VisualOntologyConfig.enabled = True
  - qa_unsupported_provider_fallback = True
  - event_description_en = q_en
  - question_en = None (Canonical routing for candidate provider)
  - include_vi_variant = False (en_only visual localization)
  - S2D1 + S2E1 treatment frozen
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
    tesseract_available = shutil.which("tesseract") is not None
    if not tesseract_available:
        try:
            print("Installing tesseract-ocr-vie packages...")
            subprocess.run(["apt-get", "update", "-y"], check=True)
            subprocess.run(["apt-get", "install", "-y", "tesseract-ocr", "tesseract-ocr-vie"], check=True)
            tesseract_available = shutil.which("tesseract") is not None
        except Exception as exc:
            print(f"Warning: OCR install failed ({exc}). OCR provider disabled.")
            tesseract_available = False

    return OCRAnswerProviderConfig(
        enabled=tesseract_available,
        evidence_frame_budget=16,
        tesseract_binary="tesseract",
        languages="vie+eng",
        min_word_confidence=40.0,
        page_segmentation_mode=11,
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
                evidence_frame_budget=16,
                max_active_domains=2,
            )
    return VisualOntologyConfig(enabled=False)


def main() -> None:
    print("=" * 115)
    print("CANONICAL 4-QUERY QA DEV SMOKE TEST (QA-13, QA-46, QA-45, QA-23)")
    print("• Language Policy               : qa_localization_language_policy = 'en_only'")
    print("• Query Localization / Routing   : event_description_en = q_en, question_en = None")
    print("• Visual Ontology               : Enabled = True")
    print("• Unsupported Provider Fallback : Enabled = True")
    print("• S2D1 + S2E1 Treatment         : Enabled (tail_budget = 5)")
    print("=" * 115)

    benchmark_path = REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "l21_150_diagnostic" / "benchmark.json"
    sidecar_path = REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "l21_150_diagnostic" / "qa_dev_translations_en.json"

    bm_data = json.loads(benchmark_path.read_text(encoding="utf-8"))
    sidecar_data = json.loads(sidecar_path.read_text(encoding="utf-8"))
    en_map = {e["query_id"]: e.get("question_en") for e in sidecar_data.get("entries", [])}

    target_qids = ["QA-13", "QA-46", "QA-45", "QA-23"]
    selected_queries = [
        q for q in bm_data.get("queries", [])
        if q.get("query_id") in target_qids and str(q.get("task_type", q.get("task", ""))).lower() == "qa"
    ]
    # Keep canonical order QA-13, QA-46, QA-45, QA-23
    selected_queries.sort(key=lambda q: target_qids.index(q["query_id"]))

    session_output = Path("/kaggle/working/smoke_test_runtime_output") if Path("/kaggle/working").exists() else REPO_ROOT / "scratch" / "smoke_test_runtime_output"
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
    t0 = time.time()
    runtime = OperationalKISRuntime.bootstrap(config)
    print(f"Runtime bootstrap completed in {time.time() - t0:.2f}s.")

    print("\n--- EXECUTING 4 TARGET QUERIES ---")
    results = []

    for idx, q in enumerate(selected_queries, start=1):
        qid = q["query_id"]
        target_vid = q.get("video_id")
        start_f, end_f = int(q["proposed_interval"][0]), int(q["proposed_interval"][1])
        raw_accepted = q.get("accepted_answers") or [q.get("answer", "")]
        gold_answers = [normalize_text(a) for a in raw_accepted if a]
        q_vi = q.get("question_vi", "")
        q_en = en_map.get(qid)
        branch = q.get("branch", "")

        t_q0 = time.time()
        # Canonical contract: event_description_en=q_en, question_en=None
        req = QAQueryRequest(
            request_id=f"smoke-treat-{qid}",
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
        t_q = time.time() - t_q0
        preds = res.get("predictions", [])

        # Evaluate Strict Hit Criteria
        hit_rank = None
        hit_frame = None
        hit_ans = None

        for p in preds:
            r = int(p.get("rank"))
            vid = p.get("video_id")
            fid = int(p.get("frame_id", -1))
            ans = p.get("answer", "")
            norm_a = normalize_text(ans)

            if vid == target_vid and (start_f <= fid <= end_f) and (norm_a in gold_answers):
                hit_rank = r
                hit_frame = fid
                hit_ans = ans
                break

        status_str = f"STRICT_HIT @{hit_rank}" if hit_rank is not None else "NO HIT"
        print(f"[{idx}/4] {qid:<5} | Branch: {branch:<12} | N={len(preds):3d} | Status: {status_str:<18} | Time: {t_q:.2f}s")
        if hit_rank is not None:
            print(f"       -> Physical Hit: Target={target_vid} | Frame={hit_frame} (GT=[{start_f}..{end_f}]) | Answer='{hit_ans}'")

        results.append({
            "query_id": qid,
            "branch": branch,
            "target_vid": target_vid,
            "gt_interval": f"[{start_f}..{end_f}]",
            "gold_answers": gold_answers,
            "n_preds": len(preds),
            "hit_rank": hit_rank,
            "hit_frame": hit_frame,
            "hit_answer": hit_ans,
            "status": status_str,
            "time_sec": t_q,
        })

    print("\n" + "=" * 115)
    print("CANONICAL 4-QUERY SMOKE TEST: SUMMARY AUDIT TABLE")
    print("=" * 115)
    print(f"{'Query ID':<8} | {'Branch':<12} | {'N':<5} | {'Hit Rank':<10} | {'Status':<16} | {'Target Video':<12} | {'Physical Frame':<14} | {'Answer'}")
    print("-" * 115)
    for r in results:
        hr_str = f"@{r['hit_rank']}" if r["hit_rank"] is not None else "-"
        hf_str = str(r["hit_frame"]) if r["hit_frame"] is not None else "-"
        ha_str = f"'{r['hit_answer']}'" if r["hit_answer"] is not None else "-"
        status_icon = "STRICT HIT ✅" if r["hit_rank"] is not None else "NO HIT ❌"
        print(f"{r['query_id']:<8} | {r['branch']:<12} | {r['n_preds']:<5} | {hr_str:<10} | {status_icon:<16} | {r['target_vid']:<12} | {hf_str:<14} | {ha_str}")
    print("=" * 115)


if __name__ == "__main__":
    main()
