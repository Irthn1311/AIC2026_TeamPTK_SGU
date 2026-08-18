# ==============================================================================================================
# Comprehensive QA-23 Evidence & OCR Audit Probe (Tracing Canonical In-GT Frame 29018)
# ==============================================================================================================

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
import numpy as np

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
from system_tai.qa.models import QAEvidenceCandidate
from system_tai.qa.object_provider import ObjectAnswerProviderConfig
from system_tai.qa.ocr_provider import OCRAnswerProviderConfig
from system_tai.qa.question_types import QuestionType
from system_tai.qa.visual_ontology import VisualOntologyConfig
from system_tai.refinement.video import DecodeRequest


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


def run_qa23_deep_audit():
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
        consensus_novel_rescue_enabled=False,
        bounded_negative_temporal_rescue_enabled=False,
    )

    visual_config = VisualOntologyConfig(
        enabled=ontology_path.exists(),
        ontology_path=ontology_path if ontology_path.exists() else None,
    )
    ocr_config = resolve_ocr_config()
    object_config = ObjectAnswerProviderConfig(enabled=False)

    session_output = Path("/kaggle/working/output/probe_qa23") if Path("/kaggle/working").exists() else REPO_ROOT / "scratch" / "probe_qa23"
    session_output.mkdir(parents=True, exist_ok=True)

    config = SessionConfig(
        input_root=Path("/kaggle/input/datasets") if Path("/kaggle/input/datasets").exists() else Path("/kaggle/input"),
        manifest_cache=Path("/kaggle/working/manifest_cache.json"),
        output_root=session_output,
        device="auto",
        allow_model_download=True,
        default_output_top_k=100,
        default_refine_top_n=3,
        qa_video_conditioned_evidence_config=evidence_config,
        qa_visual_ontology_config=visual_config,
        qa_ocr_answer_provider_config=ocr_config,
        qa_object_answer_provider_config=object_config,
    )

    print("--- BOOTSTRAPPING RUNTIME ---")
    t0 = time.time()
    runtime = OperationalKISRuntime.bootstrap(config)
    print(f"Bootstrapped in {time.time() - t0:.2f}s")

    req = QAQueryRequest(
        request_id="probe-qa23-champion",
        query_id="QA-23",
        event_description=q_vi,
        question=q_vi,
        event_description_en=q_en if q_en else None,
        question_en=None,
        include_vi_variant=False if q_en else True,
        output_top_k=100,
        refine_top_n=3,
    )

    print("\n--- RUNNING CANONICAL CHAMPION FOR QA-23 ---")
    res = runtime.handle_qa_query(req)

    diag_file = runtime.output_root / res.get("artifacts", {}).get("qa_evidence_json", "")
    with open(diag_file, encoding="utf-8") as f:
        diags = json.load(f)

    print("\n" + "=" * 120)
    print(f"QA-23 CANONICAL EVIDENCE & OCR AUDIT (GT: [{start_f}..{end_f}], Accepted: {gt_answers})")
    print("=" * 120)

    # 1. Selected Videos
    sel_vids = diags.get("selected_video_ids", [])
    top1_vid = sel_vids[0] if sel_vids else None
    print(f"\n1. Top-1 Nominated Video: {top1_vid}")

    # 2. Refined candidates on Top-1
    refined_cands = diags.get("refined_candidates", [])
    top1_refined = [c for c in refined_cands if c.get("video_id") == top1_vid]
    print(f"\n2. ALL Refined Candidates on Top-1 ({top1_vid}) (Count: {len(top1_refined)}):")
    for idx, c in enumerate(top1_refined, 1):
        in_gt = (start_f <= int(c.get("refined_frame_id") or -1) <= end_f)
        print(f"   [{idx}] orig_rank={c.get('original_rank')}, cand_frame={c.get('candidate_frame_id')}, refined_frame={c.get('refined_frame_id')}, status={c.get('status')} | In GT [{start_f}..{end_f}]? {'YES ✅' if in_gt else 'NO ❌'}")

    # 3. ALL evidence records
    ev_records = diags.get("evidence", [])
    print(f"\n3. ALL Evidence Records in diagnostics['evidence'] (Total: {len(ev_records)}):")
    for idx, e in enumerate(ev_records, 1):
        is_top1 = (e.get("video_id") == top1_vid)
        in_gt = (start_f <= int(e.get("output_frame_id") or e.get("candidate_frame_id") or -1) <= end_f)
        print(f"   [{idx:2d}] rank={e.get('rank')}, video={e.get('video_id')}, cand_frame={e.get('candidate_frame_id')}, out_frame={e.get('output_frame_id')}, status={e.get('refinement_status')}, ans='{e.get('answer')}', skip={e.get('skip_reason')} | Top1? {is_top1} | In GT? {in_gt}")

    # 4. OCR Frames Requested & Processed
    ocr_req_count = diags.get("ocr_frames_requested", 0)
    print(f"\n4. OCR Stage Telemetry:")
    print(f"   • OCR Frames Requested (Budget): {ocr_req_count}")
    print(f"   • OCR Languages: {diags.get('ocr_languages')}")
    print(f"   • OCR Predictions Count: {len(diags.get('ocr_predictions', []))}")
    print(f"   • OCR Predictions: {diags.get('ocr_predictions')}")

    # 5. Dedicated In-GT Frame 29018 Direct OCR Diagnostic
    print("\n" + "=" * 120)
    print(f"5. DIRECT OCR EVALUATION ON CANONICAL IN-GT FRAME 29018")
    print("=" * 120)

    try:
        video_record = runtime.qa_pipeline.raw_video_registry.get(top1_vid)
        probe = runtime.qa_pipeline.decoder.probe(video_record)
        dec_req = DecodeRequest(probe=probe, frame_ids=(29018,), max_decoded_frames=100)
        dec_res = runtime.qa_pipeline.decoder.decode(dec_req)

        if dec_res.frames:
            decoded_29018 = dec_res.frames[0]
            print(f"   • Successfully decoded frame 29018 from raw video {top1_vid} (timestamp={decoded_29018.timestamp_seconds:.2f}s)")

            # Call OCR Answer Provider directly on 29018
            ev_cand = QAEvidenceCandidate(
                query_id="QA-23",
                rank=1,
                video_id=top1_vid,
                frame_id=29018,
                retrieval_score=1.0,
                source_status="CANONICAL_REFINED_IN_GT",
            )
            ocr_res, ocr_tel = runtime.qa_pipeline.ocr_answer_provider.answer(
                query_id="QA-23",
                question_type=QuestionType.OCR,
                evidence=((ev_cand, decoded_29018.image),),
                output_top_k=1,
                warnings=[],
            )

            preds = ocr_res.predictions
            ans_29018 = preds[0].answer if preds else None
            is_match = (ans_29018 and any(gt in ans_29018.lower() for gt in ["dior"]))

            print(f"   • Direct OCR Raw Prediction on Frame 29018: '{ans_29018}'")
            print(f"   • OCR Telemetry on 29018: {ocr_tel.get('ocr_predictions')}")
            print(f"   • Accepted GT Match? {'YES ✅ (STRICT HIT PROVEN)' if is_match else 'NO ❌ (TRUE OCR ANSWER MISS)'}")
        else:
            print("   • Failed to decode frame 29018 (empty frames returned)")
    except Exception as exc:
        print(f"   • Exception during direct 29018 evaluation: {exc}")

    print("=" * 120)


if __name__ == "__main__":
    run_qa23_deep_audit()
