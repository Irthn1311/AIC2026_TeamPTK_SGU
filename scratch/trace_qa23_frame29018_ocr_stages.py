# ==============================================================================================================
# Diagnostic Trace on QA-23 Secondary Frame 29018 OCR Stages
# ==============================================================================================================

import json
import re
import shutil
import subprocess
import sys
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
from system_tai.qa.models import QAEvidenceCandidate
from system_tai.qa.object_provider import ObjectAnswerProviderConfig
from system_tai.qa.ocr_provider import (
    OCRAnswerProviderConfig,
    _portable_pixmap,
    parse_tesseract_tsv,
)
from system_tai.qa.question_types import QuestionType
from system_tai.qa.visual_ontology import VisualOntologyConfig
from system_tai.refinement.video import DecodeRequest


def check_tsv_numeric_metadata(text: str) -> bool:
    """Check if string contains raw Tesseract TSV column patterns like '5 1 1 1 14 3 401 593'."""
    pattern = r"\b\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+\b"
    return bool(re.search(pattern, str(text)))


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


def run_trace():
    print("=" * 120)
    print("STAGE-BY-STAGE OCR MATERIALIZATION TRACE: QA-23 FRAME 29018 (L21_V008)")
    print("=" * 120)

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
        top1_secondary_refined_rescue_tail_budget=5,
    )

    session_output = Path("/kaggle/working/output/trace_qa23") if Path("/kaggle/working").exists() else REPO_ROOT / "scratch" / "trace_qa23"
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
        qa_visual_ontology_config=VisualOntologyConfig(enabled=False),
        qa_ocr_answer_provider_config=resolve_ocr_config(),
        qa_object_answer_provider_config=ObjectAnswerProviderConfig(enabled=False),
    )

    print("\n--- BOOTSTRAPPING RUNTIME ---")
    runtime = OperationalKISRuntime.bootstrap(config)

    ocr_provider = runtime.qa_pipeline.ocr_answer_provider
    print(f"\n[PROVIDER RUNTIME IDENTITY]")
    print(f"  • OCR Answer Provider Class : {ocr_provider.__class__.__module__}.{ocr_provider.__class__.__name__}")
    print(f"  • OCR Backend Class         : {ocr_provider.backend.__class__.__module__}.{ocr_provider.backend.__class__.__name__}")
    print(f"  • Backend Identifiers       : {dict(ocr_provider.identifiers)}")

    # Decode Frame 29018 on L21_V008
    target_vid = "L21_V008"
    target_frame = 29018
    video_record = runtime.raw_video_registry.get(target_vid)
    probe = runtime.decoder.probe(video_record)
    dec_req = DecodeRequest(probe=probe, frame_ids=(target_frame,), max_decoded_frames=10)
    dec_res = runtime.decoder.decode(dec_req)
    frame_image = dec_res.frames[0].image
    print(f"\n[FRAME DECODE] Successfully decoded {target_vid} frame {target_frame}: image shape = {frame_image.shape}")

    # =========================================================================
    # STAGE A: raw Tesseract subprocess stdout
    # =========================================================================
    payload = _portable_pixmap(frame_image)
    proc = ocr_provider.backend._invoke(
        (
            "stdin",
            "stdout",
            "-l",
            "+".join(ocr_provider.config.languages),
            "--psm",
            str(ocr_provider.config.page_segmentation_mode),
            "tsv",
        ),
        input_bytes=payload,
    )
    raw_stdout_bytes = proc.stdout
    raw_stdout_str = raw_stdout_bytes.decode("utf-8", errors="replace")
    has_meta_A = check_tsv_numeric_metadata(raw_stdout_str)

    print("\n" + "=" * 100)
    print("STAGE A: raw Tesseract subprocess stdout")
    print("=" * 100)
    print(f"  • Type                        : {type(raw_stdout_bytes)} (Decoded len: {len(raw_stdout_str)} chars)")
    print(f"  • TSV Numeric Metadata Present: {has_meta_A} (Expected for raw TSV stdout)")
    print(f"  • First 500 chars repr        :\n{repr(raw_stdout_str[:500])}")

    # =========================================================================
    # STAGE B: parsed TSV structured words/lines
    # =========================================================================
    detections = parse_tesseract_tsv(raw_stdout_bytes)
    has_meta_B = any(check_tsv_numeric_metadata(d.text) for d in detections)

    print("\n" + "=" * 100)
    print("STAGE B: parsed TSV structured words/lines")
    print("=" * 100)
    print(f"  • Type                        : tuple[{len(detections)} OCRDetection objects]")
    print(f"  • TSV Numeric Metadata Present: {has_meta_B}")
    print(f"  • First 5 Detections text repr:")
    for idx, d in enumerate(detections[:5], start=1):
        print(f"    [{idx}] conf={d.confidence:5.1f} | text={repr(d.text)}")

    # =========================================================================
    # STAGE C: canonical ocr_answer_provider.answer(...) return value
    # =========================================================================
    ev_cand = QAEvidenceCandidate(
        query_id="QA-23",
        rank=1,
        video_id=target_vid,
        frame_id=target_frame,
        retrieval_score=1.0,
        source_status="TOP1_SECONDARY_REFINED_RESCUE",
    )
    ocr_res, _ = ocr_provider.answer(
        query_id="QA-23",
        question_type=QuestionType.OCR,
        evidence=((ev_cand, frame_image),),
        output_top_k=5,
        warnings=[],
    )
    has_meta_C = any(check_tsv_numeric_metadata(p.answer) for p in ocr_res.predictions)

    print("\n" + "=" * 100)
    print("STAGE C: canonical ocr_answer_provider.answer(...) return value")
    print("=" * 100)
    print(f"  • Type                        : QAResult with {len(ocr_res.predictions)} QAPrediction objects")
    print(f"  • TSV Numeric Metadata Present: {has_meta_C}")
    print(f"  • Predictions emitted:")
    for p in ocr_res.predictions:
        print(f"    Rank {p.rank}: video={p.video_id}, frame={p.frame_id}, answer={repr(p.answer)}")

    # =========================================================================
    # STAGE D & E: End-to-End Pipeline Execution (Rescue Candidates & Admitted Ranks)
    # =========================================================================
    req_treat = QAQueryRequest(
        request_id="trace-full-qa23",
        query_id="QA-23",
        event_description="Thương hiệu thời trang nào được nhắc trong dòng tiêu đề bên dưới?",
        question="Thương hiệu thời trang nào được nhắc trong dòng tiêu đề bên dưới?",
        event_description_en="Which fashion brand is mentioned in the headline below?",
        include_vi_variant=False,
        output_top_k=100,
        refine_top_n=3,
    )
    res_treat = runtime.handle_qa_query(req_treat)
    preds_treat = res_treat.get("predictions", [])

    # Read treatment diagnostics
    diag_file = runtime.output_root / res_treat.get("artifacts", {}).get("qa_evidence_json", "")
    sec_meta = {}
    if diag_file.is_file():
        with open(diag_file, encoding="utf-8") as f:
            diags_treat = json.load(f)
            sec_meta = diags_treat.get("top1_secondary_refined_rescue", {})

    produced_answers = sec_meta.get("produced_answers", [])
    admitted_tuples = sec_meta.get("tail", {}).get("admitted_tuples", [])
    has_meta_D = any(check_tsv_numeric_metadata(ans) for ans in produced_answers)

    print("\n" + "=" * 100)
    print("STAGE D: S2D1 Rescue Candidates Generated from Frame 29018")
    print("=" * 100)
    print(f"  • Provider Predictions Count  : {len(ocr_res.predictions)}")
    print(f"  • Rescue Candidates Count     : {len(produced_answers)}")
    print(f"  • TSV Numeric Metadata Present: {has_meta_D}")
    print(f"  • Produced Answers            :")
    for idx, ans in enumerate(produced_answers, start=1):
        print(f"    [{idx}] {repr(ans)}")

    print("\n" + "=" * 100)
    print("STAGE E: Admitted Final Rescue Rows in Top-100 Predictions")
    print("=" * 100)
    print(f"  • Total Treatment Predictions : {len(preds_treat)}")
    print(f"  • Admitted Tuples Count       : {len(admitted_tuples)}")
    print(f"  • Admitted Rows Detail        :")
    for t in admitted_tuples:
        print(f"    Rank {t.get('rank')}: video={t.get('video_id')}, frame={t.get('frame_id')}, answer={repr(t.get('answer'))}")

    # =========================================================================
    # OFFICIAL EVALUATION OF ALL ADMITTED ROWS
    # =========================================================================
    from system_tai.quality.l21_150_answers import answer_matches, normalize_answer
    accepted = ("dior",)
    any_strict_match = False
    print("\n" + "=" * 100)
    print("OFFICIAL EVALUATION RESULT (ALL ADMITTED ROWS VS ACCEPTED ANSWERS):")
    print("=" * 100)
    print(f"  • Accepted Answers in Benchmark: {accepted}")
    for t in admitted_tuples:
        r = t.get("rank")
        vid = t.get("video_id")
        fid = int(t.get("frame_id", -1))
        ans = str(t.get("answer", ""))
        in_gt = (vid == target_vid and 28975 <= fid <= 29025)
        ans_match = answer_matches(ans, accepted)
        is_strict = bool(in_gt and ans_match)
        if is_strict:
            any_strict_match = True
        print(f"  • Row @{r}: in_gt={in_gt} | norm_ans={repr(normalize_answer(ans))} | strict_match={'PASS ✅' if is_strict else 'FAIL ❌'}")

    has_meta_E = any(check_tsv_numeric_metadata(t.get("answer", "")) for t in admitted_tuples)

    # =========================================================================
    # DIAGNOSTIC CONCLUSION
    # =========================================================================
    print("\n" + "=" * 100)
    print("DIAGNOSTIC CONCLUSION")
    print("=" * 100)
    if has_meta_B or has_meta_C:
        conclusion = "RAW_TSV_LEAK_IN_PROVIDER"
    elif has_meta_D:
        conclusion = "RAW_TSV_LEAK_IN_S2D1"
    elif has_meta_E:
        conclusion = "RAW_TSV_LEAK_AFTER_RESCUE"
    else:
        conclusion = "CLEAN_LINE_LEVEL_END_TO_END"

    print(f"  • Required Audit Conclusion  : {conclusion}")
    print(f"  • Official Strict Hit Status : {'STRICT HIT @60' if any_strict_match else 'NO HIT (Granularity investigation required)'}")
    print("=" * 100)


if __name__ == "__main__":
    run_trace()
