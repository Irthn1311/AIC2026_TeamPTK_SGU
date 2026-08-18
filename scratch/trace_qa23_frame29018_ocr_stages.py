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
    # STAGE D: S2D1 RescueCandidate.answer before merge
    # =========================================================================
    from system_tai.qa.secondary_refined_rescue import execute_top1_secondary_refined_rescue
    from system_tai.refinement.models import RefinedCandidate, RefinementStatus

    ref_cands = [
        RefinedCandidate(
            video_id=target_vid,
            initial_frame_id=29237,
            refined_frame_id=29237,
            initial_score=0.9,
            refined_score=0.95,
            status=RefinementStatus.SUCCESS,
        ),
        RefinedCandidate(
            video_id=target_vid,
            initial_frame_id=29018,
            refined_frame_id=29018,
            initial_score=0.85,
            refined_score=0.92,
            status=RefinementStatus.SUCCESS,
        ),
    ]

    _, rescue_candidates, admitted_tuples, telemetry = execute_top1_secondary_refined_rescue(
        request=QAQueryRequest(
            request_id="trace-qa23",
            query_id="QA-23",
            event_description="Thương hiệu thời trang nào được nhắc trong dòng tiêu đề bên dưới?",
            question="Thương hiệu thời trang nào được nhắc trong dòng tiêu đề bên dưới?",
            include_vi_variant=False,
            output_top_k=100,
        ),
        q_type=QuestionType.OCR,
        champion_selected_video_ids=[target_vid],
        champion_refined_candidates=ref_cands,
        champion_predictions=[],
        canonical_evidence_cands=(),
        raw_video_registry=runtime.raw_video_registry,
        decoder=runtime.decoder,
        ocr_answer_provider=ocr_provider,
        ocr_provider_supported=True,
        config=evidence_config,
    )
    has_meta_D = any(check_tsv_numeric_metadata(c.answer) for c in rescue_candidates)

    print("\n" + "=" * 100)
    print("STAGE D: S2D1 RescueCandidate.answer before merge")
    print("=" * 100)
    print(f"  • Type                        : list[{len(rescue_candidates)} RescueCandidate objects]")
    print(f"  • TSV Numeric Metadata Present: {has_meta_D}")
    print(f"  • Rescue Candidates:")
    for idx, c in enumerate(rescue_candidates, start=1):
        print(f"    [{idx}] score={c.rescue_score} | video={c.video_id} | frame={c.frame_id} | answer={repr(c.answer)}")

    # =========================================================================
    # STAGE E: final QAPrediction.answer at admitted rank (End-to-End Pipeline)
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
    rank60_pred = preds_treat[59] if len(preds_treat) >= 60 else None
    has_meta_E = check_tsv_numeric_metadata(rank60_pred["answer"]) if rank60_pred else False

    print("\n" + "=" * 100)
    print("STAGE E: final QAPrediction.answer at admitted rank (Rank 60)")
    print("=" * 100)
    print(f"  • Total Predictions Count     : {len(preds_treat)}")
    print(f"  • Admitted Row at Rank 60     : {rank60_pred}")
    print(f"  • TSV Numeric Metadata Present: {has_meta_E}")
    if rank60_pred:
        print(f"  • First 500 chars repr        :\n{repr(str(rank60_pred['answer'])[:500])}")

    # =========================================================================
    # DIAGNOSTIC CONCLUSION
    # =========================================================================
    print("\n" + "=" * 100)
    print("DIAGNOSTIC CONCLUSION")
    print("=" * 100)
    if has_meta_B:
        conclusion = "RAW_TSV_LEAK_IN_PROVIDER"
    elif has_meta_C:
        conclusion = "RAW_TSV_LEAK_IN_PROVIDER"
    elif has_meta_D:
        conclusion = "RAW_TSV_LEAK_IN_S2D1"
    elif has_meta_E:
        conclusion = "RAW_TSV_LEAK_AFTER_RESCUE"
    else:
        conclusion = "CLEAN_LINE_LEVEL_END_TO_END"

    print(f"  • Required Audit Conclusion  : {conclusion}")
    print("=" * 100)


if __name__ == "__main__":
    run_trace()
