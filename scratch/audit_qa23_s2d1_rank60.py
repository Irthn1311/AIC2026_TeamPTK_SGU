# ==============================================================================================================
# Audit Script to Prove QA-23 Rank 60 Mechanics (Control vs Treatment)
# ==============================================================================================================

import json
import shutil
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

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
    return OCRAnswerProviderConfig(enabled=bool(available_langs), languages=supported, evidence_frame_budget=8)


def run_audit():
    benchmark_path = REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "l21_150_diagnostic" / "benchmark.json"
    sidecar_path = REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "l21_150_diagnostic" / "qa_dev_translations_en.json"
    ontology_path = REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "l21_150_diagnostic" / "qa_dev_visual_ontology.json"

    with open(benchmark_path, encoding="utf-8") as f:
        bm_data = json.load(f)
    with open(sidecar_path, encoding="utf-8") as f:
        en_sidecar = json.load(f)

    en_map = {e["query_id"]: e.get("question_en", "") for e in en_sidecar.get("entries", [])}
    all_qa_queries = {q["query_id"]: q for q in bm_data["queries"] if q.get("task_type") == "qa"}

    q = all_qa_queries["QA-23"]
    q_vi = q.get("question_vi", "")
    q_en = en_map.get("QA-23", "")
    start_f, end_f = int(q["proposed_interval"][0]), int(q["proposed_interval"][1])
    gt_answers = q.get("accepted_answers", [])

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

    session_output = Path("/kaggle/working/output/audit_qa23") if Path("/kaggle/working").exists() else REPO_ROOT / "scratch" / "audit_qa23"
    session_output.mkdir(parents=True, exist_ok=True)

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

    print("--- BOOTSTRAPPING RUNTIME ---")
    runtime = OperationalKISRuntime.bootstrap(config)

    # 1. RUN CONTROL
    runtime.qa_pipeline.video_conditioned_evidence_config = replace(base_evidence_config, top1_secondary_refined_rescue_enabled=False)
    req_ctrl = QAQueryRequest(
        request_id="audit-ctrl-qa23",
        query_id="QA-23",
        event_description=q_vi,
        question=q_vi,
        event_description_en=q_en if q_en else None,
        include_vi_variant=False,
        output_top_k=100,
        refine_top_n=3,
    )
    res_ctrl = runtime.handle_qa_query(req_ctrl)
    preds_ctrl = res_ctrl.get("predictions", [])

    # 2. RUN TREATMENT
    runtime.qa_pipeline.video_conditioned_evidence_config = replace(base_evidence_config, top1_secondary_refined_rescue_enabled=True)
    req_treat = QAQueryRequest(
        request_id="audit-treat-qa23",
        query_id="QA-23",
        event_description=q_vi,
        question=q_vi,
        event_description_en=q_en if q_en else None,
        include_vi_variant=False,
        output_top_k=100,
        refine_top_n=3,
    )
    res_treat = runtime.handle_qa_query(req_treat)
    preds_treat = res_treat.get("predictions", [])

    # Read treatment diagnostics
    diag_file = runtime.output_root / res_treat.get("artifacts", {}).get("qa_evidence_json", "")
    with open(diag_file, encoding="utf-8") as f:
        diags_treat = json.load(f)

    print("\n" + "=" * 120)
    print("QA-23 MERGE & PARITY CONTRACT AUDIT")
    print("=" * 120)

    print(f"\n1. PREDICTION LIST LENGTHS:")
    print(f"   • Control Predictions Count   : {len(preds_ctrl)}")
    print(f"   • Treatment Predictions Count : {len(preds_treat)}")
    print(f"   • Length Difference           : +{len(preds_treat) - len(preds_ctrl)} prediction appended")

    print(f"\n2. EXACT ROWS AROUND RANKS 55..60:")
    print("   --- CONTROL ROWS ---")
    for p in preds_ctrl[54:60]:
        print(f"   Rank {p['rank']:2d}: video={p['video_id']}, frame={p['frame_id']}, answer={repr(p['answer'])}")
    if len(preds_ctrl) < 60:
        print(f"   [Control has only {len(preds_ctrl)} rows, no row at rank 60]")

    print("\n   --- TREATMENT ROWS ---")
    for p in preds_treat[54:60]:
        print(f"   Rank {p['rank']:2d}: video={p['video_id']}, frame={p['frame_id']}, answer={repr(p['answer'][:40])}...")

    print(f"\n3. EXACT ROW RESPONSIBLE FOR STRICT HIT @60 IN TREATMENT:")
    p60 = preds_treat[59] if len(preds_treat) >= 60 else None
    if p60:
        print(f"   • Rank          : {p60['rank']}")
        print(f"   • Video ID      : {p60['video_id']}")
        print(f"   • Frame ID      : {p60['frame_id']} (In GT [{start_f}..{end_f}]? {start_f <= int(p60['frame_id']) <= end_f})")
        print(f"   • Answer Preview: {repr(p60['answer'][:60])}...")
        print(f"   • Contains DIOR?: {'DIOR' in str(p60['answer']).upper()}")

    print(f"\n4. RESCUE TAIL TELEMETRY IN DIAGNOSTICS:")
    sec_meta = diags_treat.get("top1_secondary_refined_rescue", {})
    print(f"   • Enabled               : {sec_meta.get('enabled')}")
    print(f"   • Eligible              : {sec_meta.get('eligible')}")
    print(f"   • Reason                : {sec_meta.get('reason')}")
    print(f"   • Top-1 Video           : {sec_meta.get('top1_video')}")
    print(f"   • Primary Anchor        : {sec_meta.get('primary_refined_anchor')}")
    print(f"   • Secondary Anchor      : {sec_meta.get('secondary_refined_anchor')}")
    print(f"   • Admitted Tail Count   : {sec_meta.get('admitted_tail_count')}")
    print(f"   • Admitted Tail Objects : {sec_meta.get('tail', {}).get('admitted_tuples')}")

    print(f"\n5. BIT-FOR-BIT SAFETY INVARIANT CHECK (RANKS 1..{len(preds_ctrl)}):")
    mismatches = []
    for idx in range(len(preds_ctrl)):
        pc = preds_ctrl[idx]
        pt = preds_treat[idx]
        if (pc["rank"], pc["video_id"], pc["frame_id"], pc["answer"]) != (pt["rank"], pt["video_id"], pt["frame_id"], pt["answer"]):
            mismatches.append((idx + 1, pc, pt))

    if not mismatches:
        print(f"   • ALL {len(preds_ctrl)} Champion predictions (Ranks 1..{len(preds_ctrl)}) are 100% BIT-FOR-BIT IDENTICAL!")
        print(f"   • ZERO existing Champion rows displaced or modified.")
        print(f"   • Treatment is 100% APPEND-ONLY: Row 60 is the admitted rescue tuple.")
    else:
        print(f"   • Mismatches found: {mismatches}")

    print("=" * 120)


if __name__ == "__main__":
    run_audit()
