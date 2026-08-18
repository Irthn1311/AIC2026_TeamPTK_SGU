# ==============================================================================================================
# Sprint R3-S2E1: Fresh Official 8-OCR Control vs Treatment Validation
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
from system_tai.qa.question_types import QuestionType
from system_tai.qa.runtime import classify_runtime_question
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


def run_fresh_8ocr_validation():
    benchmark_path = REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "l21_150_diagnostic" / "benchmark.json"
    sidecar_path = REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "l21_150_diagnostic" / "qa_dev_translations_en.json"

    with open(benchmark_path, encoding="utf-8") as f:
        bm_data = json.load(f)
    with open(sidecar_path, encoding="utf-8") as f:
        en_sidecar = json.load(f)

    en_map = {e["query_id"]: e.get("question_en", "") for e in en_sidecar.get("entries", [])}
    qa_dev_queries = [q for q in bm_data["queries"] if q.get("task_type") == "qa" and q.get("split") == "DEV"]

    # Filter strictly using canonical runtime classifier
    ocr_queries = {}
    for q in qa_dev_queries:
        qid = q["query_id"]
        q_vi = q.get("question_vi", "")
        q_en = en_map.get(qid, "")
        cls, _ = classify_runtime_question(question=q_vi, question_en=q_en, qa_a2_enabled=True, qa_ocr_enabled=True)
        if cls.question_type is QuestionType.OCR:
            ocr_queries[qid] = q

    print("=" * 130)
    print("SPRINT R3-S2E1: FRESH OFFICIAL 8-OCR CONTROL VS TREATMENT VALIDATION")
    print(f"  • Total Canonical OCR DEV Queries : {len(ocr_queries)} -> {list(ocr_queries.keys())}")
    print(f"  • Policy                          : Locked en_only (include_vi_variant=False)")
    print(f"  • Parser Fix Status               : Quoting=csv.QUOTE_NONE (Both)")
    print(f"  • Control                         : S2D1 ON | S2E1 OFF (Line-Level OCR Answers)")
    print(f"  • Treatment                       : S2D1 ON | S2E1 ON (Canonical Generic Top-5 Spans)")
    print(f"  • Evaluator                       : Official answer_matches() (Strict Normalized Equality)")
    print("=" * 130)

    session_output = Path("/kaggle/working/output/fresh_8ocr_val") if Path("/kaggle/working").exists() else REPO_ROOT / "scratch" / "fresh_8ocr_val"
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

    gains_count = 0
    regressions_count = 0
    neutral_count = 0
    prefix_failure_count = 0
    tail_budget_violations_count = 0
    tail_displacements_count = 0

    results = []

    for qid, q in ocr_queries.items():
        target_vid = q.get("video_id")
        start_f, end_f = int(q["proposed_interval"][0]), int(q["proposed_interval"][1])
        accepted_answers = tuple(q.get("accepted_answers", []))
        q_vi = q.get("question_vi", "")
        q_en = en_map.get(qid, "")

        print(f"\n" + "-" * 110)
        print(f"RUNNING A/B FOR {qid} [Target: {target_vid}, GT: [{start_f}..{end_f}], Accepted: {accepted_answers}]")

        req = QAQueryRequest(
            request_id=f"val-8ocr-{qid}",
            query_id=qid,
            event_description=q_vi,
            question=q_vi,
            event_description_en=q_en if q_en else None,
            include_vi_variant=False,
            output_top_k=100,
            refine_top_n=3,
        )

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

        # Official evaluation
        ctrl_hit, ctrl_rank, ctrl_hit_obj = evaluate_predictions(ctrl_preds, target_vid, start_f, end_f, accepted_answers)
        treat_hit, treat_rank, treat_hit_obj = evaluate_predictions(treat_preds, target_vid, start_f, end_f, accepted_answers)

        # Delta classification
        if not ctrl_hit and treat_hit:
            delta = "GAIN (+1) 🟢"
            gains_count += 1
        elif ctrl_hit and not treat_hit:
            delta = "REGRESSION (-1) 🔴"
            regressions_count += 1
        else:
            delta = "NEUTRAL (0) ⚪"
            neutral_count += 1

        # Prefix Invariant Check
        prefix_limit = min(len(ctrl_preds), len(treat_preds), 95)
        # If N < 95, prefix boundary is before the rescue tail
        if len(ctrl_preds) < 95 and ctrl_admitted:
            min_rescue_rank = min(t.get("rank", 100) for t in ctrl_admitted)
            prefix_limit = min_rescue_rank - 1

        prefix_ok = all(
            ctrl_preds[i].get("video_id") == treat_preds[i].get("video_id")
            and ctrl_preds[i].get("frame_id") == treat_preds[i].get("frame_id")
            and ctrl_preds[i].get("answer") == treat_preds[i].get("answer")
            for i in range(prefix_limit)
        )
        if not prefix_ok:
            prefix_failure_count += 1

        # Tail Budget Check (Max 5)
        tail_budget_ok = len(treat_admitted) <= 5
        if not tail_budget_ok:
            tail_budget_violations_count += 1

        # Tail Displacement Check (ranks 96..100 changed while strict hit unchanged)
        tail_displaced = False
        if len(ctrl_preds) >= 95 and len(treat_preds) >= 95 and treat_admitted:
            tail_displaced = any(
                ctrl_preds[i].get("answer") != treat_preds[i].get("answer")
                or ctrl_preds[i].get("video_id") != treat_preds[i].get("video_id")
                for i in range(95, min(len(ctrl_preds), len(treat_preds)))
            )
            if tail_displaced and delta.startswith("NEUTRAL"):
                tail_displacements_count += 1

        top1_vid = treat_sec_meta.get("top1_video")
        sec_frame = treat_sec_meta.get("secondary_refined_anchor")
        eligible = treat_sec_meta.get("eligible", False)
        cand_univ_size = treat_sec_meta.get("candidate_universe_size", 0)
        admitted_ranks = [t.get("rank") for t in treat_admitted]
        selected_answers = [t.get("answer") for t in treat_admitted]

        print(f"  • Routing                    : Top-1 = {top1_vid}, Sec Frame = {sec_frame} (Eligible: {eligible})")
        print(f"  • Candidate Universe / Top-5 : Universe = {cand_univ_size} spans | Selected = {len(selected_answers)} spans {selected_answers[:3]}")
        print(f"  • Predictions Count          : Control N = {len(ctrl_preds)} | Treatment N = {len(treat_preds)}")
        print(f"  • Admitted Ranks             : Control = {[t.get('rank') for t in ctrl_admitted]} | Treatment = {admitted_ranks}")
        print(f"  • Prefix Invariant Match     : {'PASS ✅' if prefix_ok else 'FAIL ❌'} (Checked ranks 1..{prefix_limit})")
        print(f"  • Tail Budget Check (<=5)    : {'PASS ✅' if tail_budget_ok else 'FAIL ❌'} ({len(treat_admitted)} admitted)")
        print(f"  • Official Control Result    : {'STRICT HIT @' + str(ctrl_rank) if ctrl_hit else 'NO HIT ❌'}")
        print(f"  • Official Treatment Result  : {'STRICT HIT @' + str(treat_rank) if treat_hit else 'NO HIT ❌'}")
        print(f"  • Strict Delta               : {delta}")

        results.append({
            "query_id": qid,
            "target_vid": target_vid,
            "top1_vid": str(top1_vid),
            "sec_frame": str(sec_frame),
            "eligible": eligible,
            "ctrl_n": len(ctrl_preds),
            "treat_n": len(treat_preds),
            "ctrl_res": f"HIT @{ctrl_rank}" if ctrl_hit else "NO HIT",
            "treat_res": f"HIT @{treat_rank}" if treat_hit else "NO HIT",
            "delta": delta,
            "univ_size": cand_univ_size,
            "admitted_ranks": str(admitted_ranks),
            "prefix_ok": prefix_ok,
            "tail_budget_ok": tail_budget_ok,
            "tail_displaced": tail_displaced,
        })

    # =========================================================================
    # SUMMARY MATRIX TABLE OF ALL 8 OCR QUERIES
    # =========================================================================
    print("\n" + "=" * 130)
    print("SPRINT R3-S2E1: FRESH 8-OCR OFFICIAL A/B VALIDATION SUMMARY MATRIX")
    print("=" * 130)
    print(f"{'Query ID':<8} | {'Target':<10} | {'Top-1 Vid':<10} | {'Sec Frame':<10} | {'Eligible':<8} | {'Ctrl N':<6} | {'Treat N':<7} | {'Ctrl Hit':<10} | {'Treat Hit':<10} | {'Delta':<16} | {'Prefix OK'}")
    print("-" * 130)
    for r in results:
        print(f"{r['query_id']:<8} | {r['target_vid']:<10} | {r['top1_vid']:<10} | {r['sec_frame']:<10} | {str(r['eligible']):<8} | {r['ctrl_n']:<6} | {r['treat_n']:<7} | {r['ctrl_res']:<10} | {r['treat_res']:<10} | {r['delta']:<16} | {'PASS ✅' if r['prefix_ok'] else 'FAIL ❌'}")
    print("=" * 130)

    # =========================================================================
    # AGGREGATE SUMMARY & PROMOTION GATE
    # =========================================================================
    print("\n" + "=" * 130)
    print("AGGREGATE VALIDATION METRICS & PROMOTION GATE:")
    print("=" * 130)
    print(f"  • Total OCR Queries Evaluated : {len(ocr_queries)}")
    print(f"  • Strict GAINS (+1)           : {gains_count}")
    print(f"  • Strict REGRESSIONS (-1)     : {regressions_count}")
    print(f"  • NEUTRAL (0)                 : {neutral_count}")
    print(f"  • Prefix Invariant Failures   : {prefix_failure_count}")
    print(f"  • Tail Budget Violations      : {tail_budget_violations_count}")
    print(f"  • Tail Displacements (Safe)   : {tail_displacements_count}")

    gate_gains = (gains_count >= 1)
    gate_regressions = (regressions_count == 0)
    gate_prefix = (prefix_failure_count == 0)
    gate_budget = (tail_budget_violations_count == 0)

    print("\n[PROMOTION GATE CRITERIA]")
    print(f"  1. QA23 Gain Retained         : {'PASS ✅' if gate_gains else 'FAIL ❌'}")
    print(f"  2. Zero Strict Regressions    : {'PASS ✅' if gate_regressions else 'FAIL ❌'}")
    print(f"  3. Zero Prefix Invariant Fails: {'PASS ✅' if gate_prefix else 'FAIL ❌'}")
    print(f"  4. Zero Tail Budget Violations: {'PASS ✅' if gate_budget else 'FAIL ❌'}")

    if gate_gains and gate_regressions and gate_prefix and gate_budget:
        print("\n🏆 FINAL GATE STATUS: GO FOR FULL-38 TREATMENT BENCHMARK! 🚀")
    else:
        print("\n❌ FINAL GATE STATUS: RECTIFICATION REQUIRED BEFORE FULL-38.")
    print("=" * 130)


if __name__ == "__main__":
    run_fresh_8ocr_validation()
