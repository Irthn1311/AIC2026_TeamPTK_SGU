# ==============================================================================================================
# Phase R3-S2B1: Consensus Novel Video Rescue Targeted A/B Validation Runner
# ==============================================================================================================

import argparse
import hashlib
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
from system_tai.qa.visual_ontology import VisualOntologyConfig

TARGETED_QUERIES = [
    "QA-46",  # Strict positive control (must stay STRICT_HIT @13)
    "QA-26",  # Treatment target (white vehicles convoy)
    "QA-01",  # Blast-radius / False-positive control
    "QA-23",  # Temporal near miss invariance control
]


def normalize_text(t: str) -> str:
    if not t:
        return ""
    t = unicodedata.normalize("NFKC", str(t)).casefold()
    return "".join(c for c in t if c.isalnum() or c.isspace()).strip()


def interval_distance(f: int, start: int, end: int) -> int:
    if start <= f <= end:
        return 0
    return start - f if f < start else f - end


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


def run_ab_validation(
    benchmark_path: Path,
    dev_en_sidecar_path: Path,
    ontology_path: Path,
    manifest_cache_path: Path,
    input_root: Path = Path("/kaggle/input"),
    device: str = "auto",
):
    print("=" * 135)
    print("ROUND-3 SPRINT 2B.1: CONSENSUS NOVEL VIDEO RESCUE TARGETED A/B EXPERIMENT")
    print(f"Targeted Queries: {TARGETED_QUERIES}")
    print("=" * 135)

    with open(benchmark_path, encoding="utf-8") as f:
        bm_data = json.load(f)

    with open(dev_en_sidecar_path, encoding="utf-8") as f:
        en_sidecar = json.load(f)

    en_map = {e["query_id"]: e.get("question_en", "") for e in en_sidecar.get("entries", [])}
    all_qa_queries = {q["query_id"]: q for q in bm_data["queries"] if q.get("task_type") == "qa"}

    session_output = Path("/kaggle/working/output/ab_s2b1") if Path("/kaggle/working").exists() else REPO_ROOT / "scratch" / "ab_s2b1"
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
        consensus_novel_rescue_enabled=False,  # Default OFF
        consensus_novel_rescue_max_videos=1,
        consensus_novel_rescue_tail_budget=5,
    )

    visual_config = VisualOntologyConfig(
        enabled=ontology_path.exists(),
        ontology_path=ontology_path if ontology_path.exists() else None,
    )
    ocr_config = resolve_ocr_config()
    object_config = ObjectAnswerProviderConfig(enabled=False)

    config = SessionConfig(
        input_root=input_root,
        manifest_cache=manifest_cache_path,
        output_root=session_output,
        device=device,
        allow_model_download=True,
        default_output_top_k=100,
        default_refine_top_n=3,
        qa_video_conditioned_evidence_config=base_evidence_config,
        qa_visual_ontology_config=visual_config,
        qa_ocr_answer_provider_config=ocr_config,
        qa_object_answer_provider_config=object_config,
    )

    print("\n--- BOOTSTRAPPING RUNTIME (SINGLE INSTANCE) ---")
    t0 = time.time()
    runtime = OperationalKISRuntime.bootstrap(config)
    print(f"Runtime bootstrap completed in {time.time() - t0:.2f}s.")

    results_table = []

    for qid in TARGETED_QUERIES:
        q = all_qa_queries[qid]
        target_vid = q.get("video_id")
        start_f, end_f = int(q["proposed_interval"][0]), int(q["proposed_interval"][1])
        gt_answers = [normalize_text(a) for a in q.get("accepted_answers", [])]
        q_vi = q.get("question_vi", "")
        q_en = en_map.get(qid, "")

        print("\n" + "=" * 135)
        print(f"EVALUATING {qid} [Target: {target_vid}, GT: [{start_f}..{end_f}], Accepted: {gt_answers}]")
        print("=" * 135)

        # ---------------------------------------------------------------------
        # ARM A: CONTROL (Consensus Rescue OFF)
        # ---------------------------------------------------------------------
        runtime.qa_pipeline.video_conditioned_evidence_config = replace(
            base_evidence_config,
            consensus_novel_rescue_enabled=False,
        )

        req_ctrl = QAQueryRequest(
            request_id=f"r3s2b1-control-{qid}",
            query_id=qid,
            event_description=q_vi,
            question=q_vi,
            event_description_en=q_en if q_en else None,
            question_en=None,
            include_vi_variant=False if q_en else True,
            output_top_k=100,
            refine_top_n=3,
        )

        t_ctrl0 = time.time()
        res_ctrl = runtime.handle_qa_query(req_ctrl)
        t_ctrl = time.time() - t_ctrl0
        preds_ctrl = res_ctrl.get("predictions", [])

        # Check hit in control
        ctrl_hit_rank = None
        ctrl_hit_frame = None
        ctrl_hit_ans = None
        for p in preds_ctrl:
            p_vid = p.get("video_id")
            p_frame = int(p.get("frame_id", -1))
            p_ans = normalize_text(str(p.get("answer", "")))
            if p_vid == target_vid and start_f <= p_frame <= end_f and p_ans in gt_answers:
                ctrl_hit_rank = p.get("rank")
                ctrl_hit_frame = p_frame
                ctrl_hit_ans = p_ans
                break

        # ---------------------------------------------------------------------
        # ARM B: TREATMENT (Consensus Rescue ON, max_rescue=1, tail_budget=5)
        # ---------------------------------------------------------------------
        runtime.qa_pipeline.video_conditioned_evidence_config = replace(
            base_evidence_config,
            consensus_novel_rescue_enabled=True,
            consensus_novel_rescue_max_videos=1,
            consensus_novel_rescue_tail_budget=5,
        )

        req_treat = QAQueryRequest(
            request_id=f"r3s2b1-treatment-{qid}",
            query_id=qid,
            event_description=q_vi,
            question=q_vi,
            event_description_en=q_en if q_en else None,
            question_en=None,
            include_vi_variant=False if q_en else True,
            output_top_k=100,
            refine_top_n=3,
        )

        t_treat0 = time.time()
        res_treat = runtime.handle_qa_query(req_treat)
        t_treat = time.time() - t_treat0
        preds_treat = res_treat.get("predictions", [])

        # Read treatment diagnostics
        diag_file = runtime.output_root / res_treat.get("artifacts", {}).get("qa_evidence_json", "")
        diags_treat = {}
        if diag_file.exists():
            with open(diag_file, encoding="utf-8") as f:
                diags_treat = json.load(f)

        rescue_meta = diags_treat.get("consensus_novel_rescue", {})
        chosen_cand = rescue_meta.get("chosen_candidate")
        chosen_vid = chosen_cand.get("video_id") if chosen_cand else None
        admitted_tuples = rescue_meta.get("admitted_tail_tuples", [])

        # Check hit in treatment
        treat_hit_rank = None
        treat_hit_frame = None
        treat_hit_ans = None
        for p in preds_treat:
            p_vid = p.get("video_id")
            p_frame = int(p.get("frame_id", -1))
            p_ans = normalize_text(str(p.get("answer", "")))
            if p_vid == target_vid and start_f <= p_frame <= end_f and p_ans in gt_answers:
                treat_hit_rank = p.get("rank")
                treat_hit_frame = p_frame
                treat_hit_ans = p_ans
                break

        # Verification of exact preservation for ranks 1..95
        prefix_identical = True
        for idx in range(min(95, len(preds_ctrl), len(preds_treat))):
            p_c = preds_ctrl[idx]
            p_t = preds_treat[idx]
            if (p_c.get("video_id"), p_c.get("frame_id"), p_c.get("answer")) != (p_t.get("video_id"), p_t.get("frame_id"), p_t.get("answer")):
                prefix_identical = False
                break

        print(f"\n[CONTROL]   {qid}: Hit = {'Rank ' + str(ctrl_hit_rank) if ctrl_hit_rank else 'NO HIT'} in {t_ctrl:.2f}s")
        print(f"[TREATMENT] {qid}: Hit = {'Rank ' + str(treat_hit_rank) if treat_hit_rank else 'NO HIT'} in {t_treat:.2f}s")
        print(f"  - Consensus Rescue Fired?     : {'YES (Chosen: ' + str(chosen_vid) + ')' if chosen_vid else 'NO (' + rescue_meta.get('reason', '') + ')'}")
        print(f"  - Eligible Consensus List     : {[c['video_id'] for c in rescue_meta.get('all_consensus_candidates', [])]}")
        print(f"  - Admitted Tail Tuples (96..100) : {len(admitted_tuples)} tuples -> {[p.get('video_id') + ':' + str(p.get('frame_id')) + ':' + str(p.get('answer')) for p in admitted_tuples]}")
        print(f"  - Ranks 1..95 Exact Parity    : {'PASS ✅' if prefix_identical else 'FAIL ❌'}")

        results_table.append({
            "qid": qid,
            "target": target_vid,
            "ctrl_hit": f"Rank {ctrl_hit_rank}" if ctrl_hit_rank else "NO",
            "treat_hit": f"Rank {treat_hit_rank}" if treat_hit_rank else "NO",
            "chosen_rescue": chosen_vid if chosen_vid else "None",
            "target_rescued": "YES ✅" if chosen_vid == target_vid else ("NO (Non-target)" if chosen_vid else "NO"),
            "admitted_tail_count": len(admitted_tuples),
            "prefix_parity": "PASS ✅" if prefix_identical else "FAIL ❌",
        })

    # =========================================================================
    # SUMMARY REPORT
    # =========================================================================
    print("\n" + "=" * 135)
    print("SPRINT 2B.1 CONSENSUS NOVEL VIDEO RESCUE: A/B SUMMARY TABLE")
    print("=" * 135)
    print(f"{'Query ID':<8} | {'Target':<10} | {'Control Hit':<14} | {'Treatment Hit':<14} | {'Rescued Video':<15} | {'Target Rescued?':<18} | {'Tail Admits':<12} | {'Prefix 1..95 Parity'}")
    print("-" * 135)
    for r in results_table:
        print(f"{r['qid']:<8} | {r['target']:<10} | {r['ctrl_hit']:<14} | {r['treat_hit']:<14} | {r['chosen_rescue']:<15} | {r['target_rescued']:<18} | {str(r['admitted_tail_count']) + ' tuples':<12} | {r['prefix_parity']}")
    print("=" * 135)

    # Sanity checks
    qa46_r = next(r for r in results_table if r["qid"] == "QA-46")
    qa26_r = next(r for r in results_table if r["qid"] == "QA-26")
    qa01_r = next(r for r in results_table if r["qid"] == "QA-01")

    print("\n--- CAUSAL FINDINGS ---")
    if qa46_r["ctrl_hit"] == qa46_r["treat_hit"] and qa46_r["prefix_parity"] == "PASS ✅":
        print("✅ SANITY GATE PASS: QA-46 strict hit is 100% preserved.")
    else:
        print("❌ SANITY GATE FAIL: QA-46 regression detected!")

    if qa26_r["chosen_rescue"] == "L21_V009":
        print(f"🎯 QA-26 CAUSAL TREATMENT: Target L21_V009 successfully admitted by consensus rescue rule! Final Outcome: {qa26_r['treat_hit']}")
    else:
        print(f"⚠️ QA-26 WARNING: Chosen rescue video was {qa26_r['chosen_rescue']}")

    if qa01_r["chosen_rescue"] != "L21_V001":
        print(f"🛡️ QA-01 BLAST-RADIUS CONTROL: Target L21_V001 was correctly NOT admitted. Chosen side-video: {qa01_r['chosen_rescue']}")

    print("=" * 135)


if __name__ == "__main__":
    default_input = Path("/kaggle/input/datasets") if Path("/kaggle/input/datasets").exists() else Path("/kaggle/input")
    parser = argparse.ArgumentParser(description="Run R3-S2B1 Consensus Novel Video Rescue A/B Experiment")
    parser.add_argument("--benchmark", type=Path, default=REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "l21_150_diagnostic" / "benchmark.json")
    parser.add_argument("--sidecar", type=Path, default=REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "l21_150_diagnostic" / "qa_dev_translations_en.json")
    parser.add_argument("--ontology", type=Path, default=REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "l21_150_diagnostic" / "qa_dev_visual_ontology.json")
    parser.add_argument("--manifest-cache", type=Path, default=Path("/kaggle/working/manifest_cache.json"))
    parser.add_argument("--input-root", type=Path, default=default_input)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    run_ab_validation(
        benchmark_path=args.benchmark,
        dev_en_sidecar_path=args.sidecar,
        ontology_path=args.ontology,
        manifest_cache_path=args.manifest_cache,
        input_root=args.input_root,
        device=args.device,
    )
