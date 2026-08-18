# ==============================================================================================================
# Sprint R3-S2E0: DEV-Only Generic OCR Span Coverage Audit (Contiguous 1..4 Token Spans)
# ==============================================================================================================

import argparse
import json
import re
import shutil
import subprocess
import sys
import unicodedata
from collections import defaultdict
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
from system_tai.qa.ocr_provider import (
    OCRAnswerProviderConfig,
    _portable_pixmap,
    parse_tesseract_tsv,
)
from system_tai.qa.question_types import QuestionType
from system_tai.qa.runtime import classify_runtime_question
from system_tai.qa.visual_ontology import VisualOntologyConfig
from system_tai.quality.l21_150_answers import answer_matches, normalize_answer
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
    return OCRAnswerProviderConfig(enabled=bool(available_langs), languages=supported, evidence_frame_budget=8)


def extract_contiguous_spans(clean_lines: list[str], max_n: int = 4) -> dict[int, set[str]]:
    """
    Enumerate all normalized contiguous 1..max_n token spans from clean OCR lines.
    Does NOT use ground truth or accepted answers.
    """
    spans_by_n: dict[int, set[str]] = defaultdict(set)
    for line in clean_lines:
        # Tokenize line into words
        # Clean punctuation from edges of tokens while preserving hyphenated words
        raw_tokens = [t.strip() for t in line.split() if t.strip()]
        tokens = []
        for t in raw_tokens:
            # Strip standard outer punctuation
            cleaned = t.strip(".,;:!?\"'“”‘’()[]{}<>~`|/\\-_#@$")
            if cleaned:
                tokens.append(cleaned)

        # Generate contiguous n-grams (1..max_n)
        num_tokens = len(tokens)
        for n in range(1, max_n + 1):
            for i in range(num_tokens - n + 1):
                span = " ".join(tokens[i : i + n])
                norm_span = normalize_answer(span)
                if norm_span:
                    spans_by_n[n].add(norm_span)

    return spans_by_n


def run_span_coverage_audit():
    benchmark_path = REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "l21_150_diagnostic" / "benchmark.json"
    sidecar_path = REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "l21_150_diagnostic" / "qa_dev_translations_en.json"
    ontology_path = REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "l21_150_diagnostic" / "qa_dev_visual_ontology.json"

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
    print("SPRINT R3-S2E0: DEV-ONLY GENERIC OCR SPAN COVERAGE AUDIT (CONTIGUOUS 1..4 TOKEN SPANS)")
    print(f"  • Total Canonical OCR DEV Queries : {len(ocr_queries)} -> {list(ocr_queries.keys())}")
    print(f"  • Policy                          : Locked en_only (include_vi_variant=False)")
    print(f"  • TSV Parser Status               : Fixed (quoting=csv.QUOTE_NONE)")
    print("=" * 130)

    session_output = Path("/kaggle/working/output/span_audit") if Path("/kaggle/working").exists() else REPO_ROOT / "scratch" / "span_audit"
    session_output.mkdir(parents=True, exist_ok=True)

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

    coverage_1 = 0
    coverage_le_2 = 0
    coverage_le_3 = 0
    coverage_le_4 = 0
    no_coverage = 0

    results = []

    for qid, q in ocr_queries.items():
        target_vid = q.get("video_id")
        start_f, end_f = int(q["proposed_interval"][0]), int(q["proposed_interval"][1])
        accepted_answers = tuple(q.get("accepted_answers", []))
        q_vi = q.get("question_vi", "")
        q_en = en_map.get(qid, "")

        print(f"\n" + "-" * 110)
        print(f"AUDITING {qid} [Target: {target_vid}, GT: [{start_f}..{end_f}], Accepted: {accepted_answers}]")

        req = QAQueryRequest(
            request_id=f"audit-e0-{qid}",
            query_id=qid,
            event_description=q_vi,
            question=q_vi,
            event_description_en=q_en if q_en else None,
            include_vi_variant=False,
            output_top_k=100,
            refine_top_n=3,
        )
        res = runtime.handle_qa_query(req)

        # Retrieve diagnostics for Top-1 secondary refined anchor
        diag_file = runtime.output_root / res.get("artifacts", {}).get("qa_evidence_json", "")
        sec_meta = {}
        if diag_file.is_file():
            with open(diag_file, encoding="utf-8") as f:
                diags = json.load(f)
                sec_meta = diags.get("top1_secondary_refined_rescue", {})

        top1_vid = sec_meta.get("top1_video")
        sec_frame = sec_meta.get("secondary_refined_anchor")
        eligible = sec_meta.get("eligible", False)
        reason = sec_meta.get("reason", "UNKNOWN")

        if not top1_vid or sec_frame is None or not eligible:
            print(f"  • Secondary Refined Anchor   : NOT ELIGIBLE ({reason})")
            results.append({
                "query_id": qid,
                "target_vid": target_vid,
                "top1_vid": str(top1_vid),
                "sec_frame": str(sec_frame),
                "in_gt": False,
                "num_lines": 0,
                "spans_1": 0,
                "spans_2": 0,
                "spans_3": 0,
                "spans_4": 0,
                "total_spans": 0,
                "covered": "NO (NOT_ELIGIBLE)",
                "min_n": "-",
                "matching_spans": [],
            })
            no_coverage += 1
            continue

        in_gt = (top1_vid == target_vid and start_f <= int(sec_frame) <= end_f)

        # Decode secondary frame and run clean OCR
        video_rec = runtime.raw_video_registry.get(top1_vid)
        probe = runtime.decoder.probe(video_rec)
        dec_req = DecodeRequest(probe=probe, frame_ids=(int(sec_frame),), max_decoded_frames=10)
        dec_res = runtime.decoder.decode(dec_req)
        frame_image = dec_res.frames[0].image

        # Run Tesseract on frame image and get clean lines
        ocr_backend = runtime.qa_pipeline.ocr_answer_provider.backend
        raw_stdout = ocr_backend._invoke(
            (
                "stdin",
                "stdout",
                "-l",
                "+".join(runtime.qa_pipeline.ocr_answer_provider.config.languages),
                "--psm",
                str(runtime.qa_pipeline.ocr_answer_provider.config.page_segmentation_mode),
                "tsv",
            ),
            input_bytes=_portable_pixmap(frame_image),
        ).stdout

        detections = parse_tesseract_tsv(raw_stdout)
        clean_lines = [d.text for d in detections if d.text and d.text.strip()]

        # Generate candidate span universe (1..4 tokens)
        spans_by_n = extract_contiguous_spans(clean_lines, max_n=4)
        total_unique_spans = set().union(*spans_by_n.values())

        # Check coverage against accepted answers using official answer_matches
        covered = False
        min_n = None
        matching_spans = []

        for n in (1, 2, 3, 4):
            for span in spans_by_n[n]:
                if answer_matches(span, accepted_answers):
                    covered = True
                    matching_spans.append((n, span))
                    if min_n is None:
                        min_n = n

        print(f"  • Top-1 Video / Frame        : {top1_vid} / {sec_frame} (In GT? {'YES ✅' if in_gt else 'NO ❌'})")
        print(f"  • Clean OCR Lines Detected   : {len(clean_lines)}")
        print(f"  • Span Counts                : 1-gram={len(spans_by_n[1])}, 2-gram={len(spans_by_n[2])}, 3-gram={len(spans_by_n[3])}, 4-gram={len(spans_by_n[4])} | Total Unique={len(total_unique_spans)}")
        print(f"  • Accepted Answer Covered?   : {'YES 🟢' if covered else 'NO ⚪'}")
        if covered:
            print(f"  • Minimum n-gram Length      : {min_n}-gram")
            print(f"  • Exact Matching Spans       : {matching_spans}")

        # Update aggregate coverage
        if covered:
            if min_n == 1:
                coverage_1 += 1
                coverage_le_2 += 1
                coverage_le_3 += 1
                coverage_le_4 += 1
            elif min_n == 2:
                coverage_le_2 += 1
                coverage_le_3 += 1
                coverage_le_4 += 1
            elif min_n == 3:
                coverage_le_3 += 1
                coverage_le_4 += 1
            elif min_n == 4:
                coverage_le_4 += 1
        else:
            no_coverage += 1

        results.append({
            "query_id": qid,
            "target_vid": target_vid,
            "top1_vid": str(top1_vid),
            "sec_frame": str(sec_frame),
            "in_gt": in_gt,
            "num_lines": len(clean_lines),
            "spans_1": len(spans_by_n[1]),
            "spans_2": len(spans_by_n[2]),
            "spans_3": len(spans_by_n[3]),
            "spans_4": len(spans_by_n[4]),
            "total_spans": len(total_unique_spans),
            "covered": "YES 🟢" if covered else "NO ⚪",
            "min_n": str(min_n) if min_n else "-",
            "matching_spans": matching_spans,
        })

    # =========================================================================
    # SUMMARY TABLE OF ALL 8 OCR QUERIES
    # =========================================================================
    print("\n" + "=" * 130)
    print("SPRINT R3-S2E0: PER-QUERY SPAN COVERAGE MATRIX")
    print("=" * 130)
    print(f"{'Query ID':<8} | {'Target':<10} | {'Top-1 Vid':<10} | {'Sec Frame':<10} | {'In GT?':<7} | {'Lines':<5} | {'1-gram':<6} | {'2-gram':<6} | {'3-gram':<6} | {'4-gram':<6} | {'Total':<6} | {'Covered?':<9} | {'Min n'}")
    print("-" * 130)
    for r in results:
        print(f"{r['query_id']:<8} | {r['target_vid']:<10} | {r['top1_vid']:<10} | {r['sec_frame']:<10} | {str(r['in_gt']):<7} | {r['num_lines']:<5} | {r['spans_1']:<6} | {r['spans_2']:<6} | {r['spans_3']:<6} | {r['spans_4']:<6} | {r['total_spans']:<6} | {r['covered']:<9} | {r['min_n']}")
    print("=" * 130)

    # =========================================================================
    # AGGREGATE SUMMARY METRICS
    # =========================================================================
    total_q = len(ocr_queries)
    print("\n" + "=" * 130)
    print("AGGREGATE SPAN COVERAGE METRICS (8 CANONICAL OCR DEV QUERIES):")
    print("=" * 130)
    print(f"  • Total OCR Queries Evaluated : {total_q}")
    print(f"  • Coverage @ 1-gram           : {coverage_1} / {total_q} ({coverage_1 / total_q:.1%})")
    print(f"  • Coverage @ <= 2-gram        : {coverage_le_2} / {total_q} ({coverage_le_2 / total_q:.1%})")
    print(f"  • Coverage @ <= 3-gram        : {coverage_le_3} / {total_q} ({coverage_le_3 / total_q:.1%})")
    print(f"  • Coverage @ <= 4-gram        : {coverage_le_4} / {total_q} ({coverage_le_4 / total_q:.1%})")
    print(f"  • No Coverage Queries         : {no_coverage} / {total_q} ({no_coverage / total_q:.1%})")
    print("=" * 130)


if __name__ == "__main__":
    run_span_coverage_audit()
