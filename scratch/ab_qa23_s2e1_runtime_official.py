# ==============================================================================================================
# Sprint R3-S2E1: QA-23 Official Runtime Control vs Treatment A/B Validation
# ==============================================================================================================

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

if Path("/kaggle").exists():
    try:
        import clip
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "openai-clip", "ftfy", "regex", "tqdm"], check=False)

    tess_path = shutil.which("tesseract")
    if not tess_path or not (Path("/usr/share/tesseract-ocr/5/tessdata/vie.traineddata").exists() or Path("/usr/share/tesseract-ocr/4.00/tessdata/vie.traineddata").exists()):
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
from system_tai.quality.l21_150_answers import answer_matches, normalize_answer


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


def evaluate_predictions(preds: list[dict[str, Any]], target_vid: str, start_f: int, end_f: int, accepted: tuple[str, ...]) -> tuple[bool, int | None, dict[str, Any] | None]:
    for idx, p in enumerate(preds, start=1):
        vid = p.get("video_id")
        fid = int(p.get("frame_id", -1))
        ans = str(p.get("answer", ""))
        in_gt = (vid == target_vid and start_f <= fid <= end_f)
        if in_gt and answer_matches(ans, accepted):
            return True, idx, p
    return False, None, None


def run_qa23_ab():
    benchmark_path = REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "l21_150_diagnostic" / "benchmark.json"
    sidecar_path = REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "l21_150_diagnostic" / "qa_dev_translations_en.json"

    with open(benchmark_path, encoding="utf-8") as f:
        bm_data = json.load(f)
    with open(sidecar_path, encoding="utf-8") as f:
        en_sidecar = json.load(f)

    en_map = {e["query_id"]: e.get("question_en", "") for e in en_sidecar.get("entries", [])}
    qa_dev_queries = [q for q in bm_data["queries"] if q.get("task_type") == "qa" and q.get("split") == "DEV"]

    q = next(q for q in qa_dev_queries if q["query_id"] == "QA-23")
    qid = "QA-23"
    target_vid = q.get("video_id")
    start_f, end_f = int(q["proposed_interval"][0]), int(q["proposed_interval"][1])
    accepted_answers = tuple(q.get("accepted_answers", []))
    q_vi = q.get("question_vi", "")
    q_en = en_map.get(qid, "")

    print("=" * 130)
    print("SPRINT R3-S2E1: QA-23 OFFICIAL RUNTIME CONTROL VS TREATMENT A/B")
    print(f"  • Query                       : {qid} [Target: {target_vid}, GT: [{start_f}..{end_f}], Accepted: {accepted_answers}]")
    print(f"  • Policy                      : Locked en_only (include_vi_variant=False)")
    print(f"  • Parser Fix Status           : Quoting=csv.QUOTE_NONE (Both)")
    print(f"  • S2D1 Tail Budget            : 5 Slots (Both)")
    print(f"  • Control                     : S2E1 OFF (Line-Level OCR Answers)")
    print(f"  • Treatment                   : S2E1 ON (Canonical Generic Top-5 Spans)")
    print("=" * 130)

    # 1. SETUP RUNTIME
    session_output = Path("/kaggle/working/output/qa23_ab") if Path("/kaggle/working").exists() else REPO_ROOT / "scratch" / "qa23_ab"
    session_output.mkdir(parents=True, exist_ok=True)

    base_evidence_kwargs = dict(
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
        top1_secondary_refined_rescue_tail_budget=5,
    )

    ctrl_ev_config = QAVideoConditionedEvidenceConfig(
        **base_evidence_kwargs,
        top1_secondary_refined_rescue_span_candidateizer=False,
    )

    treat_ev_config = QAVideoConditionedEvidenceConfig(
        **base_evidence_kwargs,
        top1_secondary_refined_rescue_span_candidateizer=True,
    )

    base_config_kwargs = dict(
        input_root=Path("/kaggle/input/datasets") if Path("/kaggle/input/datasets").exists() else Path("/kaggle/input"),
        manifest_cache=Path("/kaggle/working/manifest_cache.json"),
        output_root=session_output,
        device="auto",
        allow_model_download=True,
        default_output_top_k=100,
        default_refine_top_n=3,
        qa_visual_ontology_config=VisualOntologyConfig(enabled=False),
        qa_ocr_answer_provider_config=resolve_ocr_config(),
        qa_object_answer_provider_config=ObjectAnswerProviderConfig(enabled=False),
    )

    print("\n--- BOOTSTRAPPING RUNTIME (CONTROL: S2E1 OFF) ---")
    ctrl_runtime = OperationalKISRuntime.bootstrap(SessionConfig(**base_config_kwargs, qa_video_conditioned_evidence_config=ctrl_ev_config))

    print("\n--- BOOTSTRAPPING RUNTIME (TREATMENT: S2E1 ON) ---")
    treat_runtime = OperationalKISRuntime.bootstrap(SessionConfig(**base_config_kwargs, qa_video_conditioned_evidence_config=treat_ev_config))

    req = QAQueryRequest(
        request_id=f"qa23-ab",
        query_id=qid,
        event_description=q_vi,
        question=q_vi,
        event_description_en=q_en if q_en else None,
        include_vi_variant=False,
        output_top_k=100,
        refine_top_n=3,
    )

    print("\n" + "=" * 130)
    print("EXECUTING QA-23 RUNTIME PIPELINE...")
    print("=" * 130)
    ctrl_res = ctrl_runtime.handle_qa_query(req)
    treat_res = treat_runtime.handle_qa_query(req)

    ctrl_preds = ctrl_res.get("predictions", [])
    treat_preds = treat_res.get("predictions", [])

    # Read diagnostics
    ctrl_diag_file = ctrl_runtime.output_root / ctrl_res.get("artifacts", {}).get("qa_evidence_json", "")
    ctrl_sec_meta = {}
    if ctrl_diag_file.is_file():
        with open(ctrl_diag_file, encoding="utf-8") as f:
            ctrl_sec_meta = json.load(f).get("top1_secondary_refined_rescue", {})

    treat_diag_file = treat_runtime.output_root / treat_res.get("artifacts", {}).get("qa_evidence_json", "")
    treat_sec_meta = {}
    if treat_diag_file.is_file():
        with open(treat_diag_file, encoding="utf-8") as f:
            treat_sec_meta = json.load(f).get("top1_secondary_refined_rescue", {})

    ctrl_admitted = ctrl_sec_meta.get("tail", {}).get("admitted_tuples", [])
    treat_admitted = treat_sec_meta.get("tail", {}).get("admitted_tuples", [])

    # 1. Pipeline Routing Information
    print("\n[PIPELINE ROUTING TELEMETRY]")
    print(f"  • Top-1 Nominated Video       : Control = {ctrl_sec_meta.get('top1_video')} | Treatment = {treat_sec_meta.get('top1_video')}")
    print(f"  • Primary Refined Anchor      : Control = {ctrl_sec_meta.get('primary_refined_anchor')} | Treatment = {treat_sec_meta.get('primary_refined_anchor')}")
    print(f"  • Secondary Refined Anchor    : Control = {ctrl_sec_meta.get('secondary_refined_anchor')} | Treatment = {treat_sec_meta.get('secondary_refined_anchor')} (In GT? {ctrl_sec_meta.get('top1_video') == target_vid and start_f <= int(ctrl_sec_meta.get('secondary_refined_anchor', -1)) <= end_f})")
    print(f"  • S2E1 Span Candidateizer ON? : Control = {ctrl_sec_meta.get('span_candidateizer_enabled', False)} | Treatment = {treat_sec_meta.get('span_candidateizer_enabled', False)}")
    print(f"  • Total Candidates Ranked     : Control = N/A (Line-Level) | Treatment = {treat_sec_meta.get('total_spans_ranked', 0)} spans")

    # 2. Protected Prefix Parity Check (Ranks 1..59)
    prefix_len = min(len(ctrl_preds), len(treat_preds), 59)
    prefix_identical = all(
        ctrl_preds[i].get("video_id") == treat_preds[i].get("video_id")
        and ctrl_preds[i].get("frame_id") == treat_preds[i].get("frame_id")
        and ctrl_preds[i].get("answer") == treat_preds[i].get("answer")
        for i in range(prefix_len)
    )

    print("\n[PROTECTED PREFIX PARITY CHECK (RANKS 1..59)]")
    print(f"  • Total Control Predictions   : {len(ctrl_preds)}")
    print(f"  • Total Treatment Predictions : {len(treat_preds)}")
    print(f"  • Prefix Invariant Match      : {'PASS ✅ (Ranks 1..59 are 100% identical)' if prefix_identical else 'FAIL ❌'}")

    # 3. Admitted Tail Rescue Rows Comparison
    print("\n" + "=" * 130)
    print("CONTROL FINAL ADMITTED RESCUE ROWS (S2E1 OFF - LINE LEVEL):")
    print("=" * 130)
    print(f"{'Rank':<6} | {'Video':<10} | {'Frame':<8} | {'Official Match?':<16} | {'Answer Text'}")
    print("-" * 130)
    for t in ctrl_admitted:
        ans = str(t.get("answer", ""))
        m = answer_matches(ans, accepted_answers)
        print(f"@{t.get('rank'):<5} | {t.get('video_id'):<10} | {t.get('frame_id'):<8} | {'MATCH ✅' if m else 'NO_MATCH ❌':<16} | {repr(ans)}")

    print("\n" + "=" * 130)
    print("TREATMENT FINAL ADMITTED RESCUE ROWS (S2E1 ON - TOP 5 SPANS):")
    print("=" * 130)
    print(f"{'Rank':<6} | {'Video':<10} | {'Frame':<8} | {'Official Match?':<16} | {'Answer Text'}")
    print("-" * 130)
    for t in treat_admitted:
        ans = str(t.get("answer", ""))
        m = answer_matches(ans, accepted_answers)
        print(f"@{t.get('rank'):<5} | {t.get('video_id'):<10} | {t.get('frame_id'):<8} | {'MATCH ✅' if m else 'NO_MATCH ❌':<16} | {repr(ans)}")
    print("=" * 130)

    # 4. OFFICIAL EVALUATION RESULTS
    ctrl_hit, ctrl_rank, ctrl_hit_obj = evaluate_predictions(ctrl_preds, target_vid, start_f, end_f, accepted_answers)
    treat_hit, treat_rank, treat_hit_obj = evaluate_predictions(treat_preds, target_vid, start_f, end_f, accepted_answers)

    print("\n" + "=" * 130)
    print("OFFICIAL BENCHMARK EVALUATION SUMMARY:")
    print("=" * 130)
    print(f"  • Accepted Answers in GT      : {accepted_answers}")
    print(f"  • Control Result (S2E1 OFF)   : {'STRICT HIT ✅ @' + str(ctrl_rank) if ctrl_hit else 'NO HIT ❌'}")
    print(f"  • Treatment Result (S2E1 ON)  : {'STRICT HIT ✅ @' + str(treat_rank) if treat_hit else 'NO HIT ❌'}")

    if treat_hit and treat_hit_obj:
        print(f"\n  🎯 TREATMENT STRICT WINNING ROW DETAILS:")
        print(f"     • Final Admitted Rank      : #{treat_rank}")
        print(f"     • Video ID                 : {treat_hit_obj.get('video_id')} (Target = {target_vid})")
        print(f"     • Frame ID                 : {treat_hit_obj.get('frame_id')} (GT = [{start_f}..{end_f}])")
        print(f"     • Exact Prediction Answer  : {repr(treat_hit_obj.get('answer'))}")
        print(f"     • Normalized Prediction    : {repr(normalize_answer(treat_hit_obj.get('answer')))}")
        print(f"     • Official Match Outcome   : {answer_matches(treat_hit_obj.get('answer'), accepted_answers)}")

    print("\n" + "=" * 130)
    print("CAUSAL VERIFICATION CONCLUSION:")
    print("=" * 130)
    if not ctrl_hit and treat_hit and prefix_identical:
        conclusion = f"OFFICIAL STRICT CAUSAL WIN CONFIRMED! (Control=NO_HIT -> Treatment=STRICT_HIT @{treat_rank}) 🏆"
    elif ctrl_hit and treat_hit:
        conclusion = "BOTH_HIT (No delta)"
    else:
        conclusion = f"STRICT_WIN_NOT_CONFIRMED (Control={ctrl_hit}, Treatment={treat_hit})"

    print(f"  • Final Outcome               : {conclusion}")
    print("=" * 130)


if __name__ == "__main__":
    run_qa23_ab()
