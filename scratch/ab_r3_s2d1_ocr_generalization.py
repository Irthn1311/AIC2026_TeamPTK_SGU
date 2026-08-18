# ==============================================================================================================
# Sprint 2D.1: OCR-Only Generalization Validation Runner under Locked en_only Champion Config
# ==============================================================================================================

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import unicodedata
from dataclasses import replace
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

if Path("/kaggle").exists():
    try:
        import clip
    except ImportError:
        print("Installing openai-clip dependency...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "openai-clip", "ftfy", "regex", "tqdm"], check=False)

    tess_path = shutil.which("tesseract")
    if not tess_path or not (Path("/usr/share/tesseract-ocr/5/tessdata/vie.traineddata").exists() or Path("/usr/share/tesseract-ocr/4.00/tessdata/vie.traineddata").exists()):
        print("Installing tesseract-ocr-vie packages...")
        subprocess.run(["apt-get", "update", "-qq"], check=False)
        subprocess.run(["apt-get", "install", "-y", "-qq", "tesseract-ocr", "tesseract-ocr-vie", "tesseract-ocr-eng"], check=False)

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
from system_tai.qa.question_types import QuestionType, classify_question
from system_tai.qa.visual_ontology import VisualOntologyConfig


def normalize_text(t: str) -> str:
    if not t:
        return ""
    t = unicodedata.normalize("NFKC", str(t)).casefold()
    return "".join(c for c in t if c.isalnum() or c.isspace()).strip()


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


from system_tai.qa.runtime import classify_runtime_question


def run_ocr_generalization_validation():
    benchmark_path = REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "l21_150_diagnostic" / "benchmark.json"
    sidecar_path = REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "l21_150_diagnostic" / "qa_dev_translations_en.json"
    ontology_path = REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "l21_150_diagnostic" / "qa_dev_visual_ontology.json"

    with open(benchmark_path, encoding="utf-8") as f:
        bm_data = json.load(f)
    with open(sidecar_path, encoding="utf-8") as f:
        en_sidecar = json.load(f)

    en_map = {e["query_id"]: e.get("question_en", "") for e in en_sidecar.get("entries", [])}
    all_qa_queries = {q["query_id"]: q for q in bm_data["queries"] if q.get("task_type") == "qa" and q.get("split") == "DEV"}

    # Filter all queries classified as QuestionType.OCR using canonical runtime classifier
    ocr_queries = {}
    for qid, q in all_qa_queries.items():
        q_vi = q.get("question_vi", "")
        q_en = en_map.get(qid)
        classification, strat = classify_runtime_question(q_vi, q_en, qa_a2_enabled=True, qa_ocr_enabled=True)
        if classification.question_type == QuestionType.OCR:
            ocr_queries[qid] = q

    print("=" * 140)
    print("SPRINT 2D.1: OCR-ONLY GENERALIZATION VALIDATION UNDER LOCKED 'en_only' CHAMPION CONFIG")
    print("  • Language Policy             : qa_localization_language_policy = 'en_only'")
    print("  • Query Variant Setting       : include_vi_variant = False (Locked Champion)")
    print("  • Selection Source            : Canonical classify_runtime_question (QuestionType.OCR)")
    print(f"  • Total DEV QA Queries        : {len(all_qa_queries)}")
    print(f"  • Total OCR Queries to Run    : {len(ocr_queries)} -> {list(ocr_queries.keys())}")
    print("  • Arm Isolation               : Control (S2D1 OFF) vs Treatment (S2D1 ON), 100% Identical Payload")
    print("=" * 140)

    session_output = Path("/kaggle/working/output/ocr_generalization") if Path("/kaggle/working").exists() else REPO_ROOT / "scratch" / "ocr_generalization"
    session_output.mkdir(parents=True, exist_ok=True)

    base_evidence_config = QAVideoConditionedEvidenceConfig(
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
        consensus_novel_rescue_enabled=False,
        bounded_negative_temporal_rescue_enabled=False,
        top1_secondary_refined_rescue_enabled=False,
        top1_secondary_refined_rescue_tail_budget=5,
    )

    config = SessionConfig(
        input_root=Path("/kaggle/input/datasets") if Path("/kaggle/input/datasets").exists() else Path("/kaggle/input"),
        manifest_cache=Path("/kaggle/working/manifest_cache.json"),
        output_root=session_output,
        device="auto",
        allow_model_download=True,
        default_output_top_k=100,
        default_refine_top_n=3,
        qa_video_conditioned_evidence_config=base_evidence_config,
        qa_visual_ontology_config=VisualOntologyConfig(enabled=ontology_path.exists(), ontology_path=ontology_path if ontology_path.exists() else None),
        qa_ocr_answer_provider_config=resolve_ocr_config(),
        qa_object_answer_provider_config=ObjectAnswerProviderConfig(enabled=False),
    )

    print("\n--- BOOTSTRAPPING RUNTIME ---")
    t0 = time.time()
    runtime = OperationalKISRuntime.bootstrap(config)
    print(f"Runtime bootstrap completed in {time.time() - t0:.2f}s.")

    results_table = []
    gains = 0
    regressions = 0
    neutrals = 0
    eligible_count = 0
    fire_count = 0
    n_less_95 = 0
    n_ge_95 = 0
    tail_displacement_cases = 0

    for qid, q in ocr_queries.items():
        target_vid = q.get("video_id")
        start_f, end_f = int(q["proposed_interval"][0]), int(q["proposed_interval"][1])
        gt_answers = [normalize_text(a) for a in q.get("accepted_answers", [])]
        q_vi = q.get("question_vi", "")
        q_en = en_map.get(qid, "")

        print("\n" + "-" * 140)
        print(f"EVALUATING {qid} [Target: {target_vid}, GT: [{start_f}..{end_f}], Accepted: {gt_answers}]")
        print(f"  VI : '{q_vi}'")
        print(f"  EN : '{q_en}'")

        # ---------------------------------------------------------------------
        # ARM A: CONTROL (Rescue OFF)
        # ---------------------------------------------------------------------
        runtime.qa_pipeline.video_conditioned_evidence_config = replace(
            base_evidence_config,
            top1_secondary_refined_rescue_enabled=False,
        )
        req_ctrl = QAQueryRequest(
            request_id=f"ocr-gen-ctrl-{qid}",
            query_id=qid,
            event_description=q_vi,
            question=q_vi,
            event_description_en=q_en if q_en else None,
            include_vi_variant=False,  # Locked en_only policy
            output_top_k=100,
            refine_top_n=3,
        )
        t_ctrl0 = time.time()
        res_ctrl = runtime.handle_qa_query(req_ctrl)
        t_ctrl = time.time() - t_ctrl0
        preds_ctrl = res_ctrl.get("predictions", [])
        n_ctrl = len(preds_ctrl)

        ctrl_hit_rank = None
        ctrl_hit_frame = None
        ctrl_hit_ans = None
        for p in preds_ctrl:
            p_vid = p.get("video_id")
            p_frame = int(p.get("frame_id", -1))
            p_ans = normalize_text(str(p.get("answer", "")))
            if p_vid == target_vid and start_f <= p_frame <= end_f and any(gt in p_ans for gt in gt_answers):
                ctrl_hit_rank = p.get("rank")
                ctrl_hit_frame = p_frame
                ctrl_hit_ans = p_ans
                break

        # ---------------------------------------------------------------------
        # ARM B: TREATMENT (Rescue ON)
        # ---------------------------------------------------------------------
        runtime.qa_pipeline.video_conditioned_evidence_config = replace(
            base_evidence_config,
            top1_secondary_refined_rescue_enabled=True,
            top1_secondary_refined_rescue_tail_budget=5,
        )
        req_treat = QAQueryRequest(
            request_id=f"ocr-gen-treat-{qid}",
            query_id=qid,
            event_description=q_vi,
            question=q_vi,
            event_description_en=q_en if q_en else None,
            include_vi_variant=False,  # Locked en_only policy
            output_top_k=100,
            refine_top_n=3,
        )
        t_treat0 = time.time()
        res_treat = runtime.handle_qa_query(req_treat)
        t_treat = time.time() - t_treat0
        preds_treat = res_treat.get("predictions", [])
        n_treat = len(preds_treat)

        # Read treatment diagnostics
        diag_file = runtime.output_root / res_treat.get("artifacts", {}).get("qa_evidence_json", "")
        diags_treat = {}
        if diag_file.exists():
            with open(diag_file, encoding="utf-8") as f:
                diags_treat = json.load(f)

        rescue_meta = diags_treat.get("top1_secondary_refined_rescue", {})
        is_eligible = rescue_meta.get("eligible", False)
        adm_count = rescue_meta.get("admitted_tail_count", 0)
        adm_tuples = rescue_meta.get("tail", {}).get("admitted_tuples", [])

        if is_eligible:
            eligible_count += 1
        if adm_count > 0:
            fire_count += 1

        if n_ctrl < 95:
            n_less_95 += 1
        else:
            n_ge_95 += 1

        # Check treatment hit
        treat_hit_rank = None
        treat_hit_frame = None
        treat_hit_ans = None
        for p in preds_treat:
            p_vid = p.get("video_id")
            p_frame = int(p.get("frame_id", -1))
            p_ans = normalize_text(str(p.get("answer", "")))
            if p_vid == target_vid and start_f <= p_frame <= end_f and any(gt in p_ans for gt in gt_answers):
                treat_hit_rank = p.get("rank")
                treat_hit_frame = p_frame
                treat_hit_ans = p_ans
                break

        # Safety Check:
        # If N < 95: All N original Champion rows must be 100% bit-identical
        # If N >= 95: Ranks 1..95 must be 100% bit-identical
        safety_pass = True
        prefix_limit = min(n_ctrl, 95)
        for idx in range(prefix_limit):
            pc = preds_ctrl[idx]
            pt = preds_treat[idx]
            if (pc["rank"], pc["video_id"], pc["frame_id"], pc["answer"]) != (pt["rank"], pt["video_id"], pt["frame_id"], pt["answer"]):
                safety_pass = False
                break

        # Delta classification
        if ctrl_hit_rank is None and treat_hit_rank is not None:
            delta = "GAIN 🟢"
            gains += 1
        elif ctrl_hit_rank is not None and treat_hit_rank is None:
            delta = "REGRESSION 🔴"
            regressions += 1
        elif ctrl_hit_rank != treat_hit_rank:
            delta = f"SHIFT ({ctrl_hit_rank} -> {treat_hit_rank})"
            neutrals += 1
        else:
            delta = "NEUTRAL ⚪"
            neutrals += 1

        if n_ctrl >= 95 and adm_count > 0 and delta != "REGRESSION 🔴":
            tail_displacement_cases += 1

        print(f"  • Baseline Count N           : {n_ctrl} (Treatment N = {n_treat})")
        print(f"  • Top-1 Video                 : {rescue_meta.get('top1_video')}")
        print(f"  • Primary Refined Anchor      : {rescue_meta.get('primary_refined_anchor')}")
        print(f"  • Secondary Refined Anchor    : {rescue_meta.get('secondary_refined_anchor')}")
        print(f"  • Eligible / Reason           : {is_eligible} / {rescue_meta.get('reason')}")
        print(f"  • Admitted Rescue Tuples      : {adm_count} -> {[t.get('rank') for t in adm_tuples]}")
        print(f"  • Control Strict Hit          : {f'Rank {ctrl_hit_rank}' if ctrl_hit_rank else 'NO HIT'}")
        print(f"  • Treatment Strict Hit        : {f'Rank {treat_hit_rank}' if treat_hit_rank else 'NO HIT'}")
        print(f"  • Strict Delta                : {delta}")
        print(f"  • Safety Invariant (1..{prefix_limit})   : {'PASS ✅' if safety_pass else 'FAIL ❌'}")

        results_table.append({
            "qid": qid,
            "target": target_vid,
            "n_ctrl": n_ctrl,
            "top1": str(rescue_meta.get("top1_video")),
            "sec_frame": str(rescue_meta.get("secondary_refined_anchor")),
            "admitted_ranks": str([t.get("rank") for t in adm_tuples]),
            "ctrl_hit": f"Rank {ctrl_hit_rank}" if ctrl_hit_rank else "NO",
            "treat_hit": f"Rank {treat_hit_rank}" if treat_hit_rank else "NO",
            "delta": delta,
            "safety": "PASS ✅" if safety_pass else "FAIL ❌",
        })

    # =========================================================================
    # SUMMARY REPORT
    # =========================================================================
    print("\n" + "=" * 140)
    print("OCR-ONLY GENERALIZATION VALIDATION: PER-QUERY SUMMARY TABLE")
    print("=" * 140)
    print(f"{'Query ID':<8} | {'Target':<10} | {'Base N':<6} | {'Top-1 Video':<11} | {'Sec Frame':<10} | {'Admitted Ranks':<16} | {'Control':<10} | {'Treatment':<10} | {'Delta':<15} | {'Safety'}")
    print("-" * 140)
    for r in results_table:
        print(f"{r['qid']:<8} | {r['target']:<10} | {r['n_ctrl']:<6} | {r['top1']:<11} | {r['sec_frame']:<10} | {r['admitted_ranks']:<16} | {r['ctrl_hit']:<10} | {r['treat_hit']:<10} | {r['delta']:<15} | {r['safety']}")
    print("=" * 140)

    print("\n" + "=" * 140)
    print("OCR-ONLY GENERALIZATION: AGGREGATE SUMMARY METRICS")
    print("=" * 140)
    print(f"  • Total OCR Queries Evaluated : {len(ocr_queries)}")
    print(f"  • Eligible Queries            : {eligible_count} / {len(ocr_queries)} ({eligible_count / len(ocr_queries):.1%})")
    print(f"  • Rescue Fired Queries        : {fire_count} / {len(ocr_queries)} ({fire_count / len(ocr_queries):.1%})")
    print(f"  • Distribution by Base N      : N < 95: {n_less_95} queries | N >= 95: {n_ge_95} queries")
    print(f"  • Tail Displacement Cases     : {tail_displacement_cases} (Ranks 96..100 replaced safely with ZERO strict regression)")
    print(f"  • Strict GAINS                : {gains} 🟢")
    print(f"  • Strict REGRESSIONS          : {regressions} 🔴")
    print(f"  • NEUTRAL Queries             : {neutrals} ⚪")
    print(f"  • Net Strict Delta            : +{gains - regressions}")
    print(f"  • Safety Invariant Across All : {'ALL PASSED (100% Zero-Regression on Protected Prefix) ✅' if all(r['safety'] == 'PASS ✅' for r in results_table) else 'FAIL ❌'}")
    print("=" * 140)


if __name__ == "__main__":
    run_ocr_generalization_validation()
