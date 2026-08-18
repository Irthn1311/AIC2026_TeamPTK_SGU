# ==============================================================================================================
# Sprint R3-S2E1: DEV-Only Generic OCR Span Ranking-Feasibility Audit
# ==============================================================================================================

import argparse
import json
import math
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


# =============================================================================
# FROZEN GENERIC RUNTIME-SAFE SPAN EXTRACTION & SCORING RULE (ZERO GT KNOWLEDGE)
# =============================================================================

class GenericSpanScorer:
    """
    Evaluates candidates using only runtime-safe features:
    1. Word-level OCR confidence from Tesseract
    2. Character cleanliness and alphabetic density
    3. Junk and broken-symbol penalty
    4. Bounded concise length prior
    5. Line-level prominence and rank
    """

    @staticmethod
    def is_junk_token(tok: str) -> bool:
        # Punctuation/symbols only or non-alphanumeric junk
        cleaned = re.sub(r"[^\w\s]", "", tok, flags=re.UNICODE).strip()
        if not cleaned:
            return True
        # Strings with weird symbol density like '/-PW/ĐMMAANWS' or '†.Il¿1f#!'
        alphanumeric_count = sum(c.isalnum() for c in tok)
        if alphanumeric_count / len(tok) < 0.5:
            return True
        return False

    @classmethod
    def score_span(
        cls,
        *,
        tokens: list[str],
        confidences: list[float],
        line_idx: int,
        line_conf: float,
    ) -> tuple[float, dict[str, float]]:
        raw_span_text = " ".join(tokens)
        norm_span = normalize_answer(raw_span_text)

        if not norm_span:
            return 0.0, {"valid": 0.0}

        # 1. Average Word Confidence (0.0 .. 1.0)
        mean_conf = sum(confidences) / max(len(confidences), 1)
        conf_factor = mean_conf / 100.0

        # 2. Cleanliness Ratio (Alphanumeric Density)
        total_chars = len(raw_span_text)
        alpha_chars = sum(c.isalpha() for c in raw_span_text)
        cleanliness = alpha_chars / max(total_chars, 1)

        # 3. Junk Penalty
        junk_count = sum(cls.is_junk_token(t) for t in tokens)
        junk_factor = 1.0 / (1.0 + 2.0 * junk_count)
        if any(cls.is_junk_token(t) for t in tokens):
            junk_factor *= 0.5

        # 4. Length Prior (Concise 1..3 words preferred over long sentences)
        num_words = len(tokens)
        if num_words == 1:
            len_prior = 1.0
        elif num_words == 2:
            len_prior = 0.95
        elif num_words == 3:
            len_prior = 0.85
        elif num_words == 4:
            len_prior = 0.75
        else:
            len_prior = 0.5

        # 5. Line Prominence Factor
        line_factor = (line_conf / 100.0) * (1.0 / (1.0 + 0.1 * line_idx))

        # 6. Proper Capitalization / Entity Bonus (Generic: Capitalized tokens in OCR often denote entities/names)
        cap_bonus = 1.0
        if any(t.isupper() and len(t) >= 2 for t in tokens):
            cap_bonus = 1.15

        # Final Combined Generic Score
        final_score = conf_factor * cleanliness * junk_factor * len_prior * line_factor * cap_bonus

        components = {
            "mean_conf": mean_conf,
            "conf_factor": conf_factor,
            "cleanliness": cleanliness,
            "junk_factor": junk_factor,
            "len_prior": len_prior,
            "line_factor": line_factor,
            "cap_bonus": cap_bonus,
            "final_score": final_score,
        }
        return final_score, components


def extract_and_score_all_spans(
    tsv_payload: bytes,
    max_n: int = 4,
) -> list[dict[str, Any]]:
    """
    Parses Tesseract TSV, extracts all word tokens with their exact confidences,
    and enumerates all contiguous 1..max_n spans scored by the frozen GenericSpanScorer.
    """
    import csv
    import io

    text = tsv_payload.decode("utf-8", errors="replace")
    reader = csv.DictReader(io.StringIO(text), delimiter="\t", quoting=csv.QUOTE_NONE)

    # Group words by line: (page_num, block_num, par_num, line_num)
    lines_dict: dict[tuple[int, int, int, int], list[tuple[int, str, float]]] = defaultdict(list)
    for row in reader:
        try:
            raw_text = str(row.get("text", "")).strip()
            conf = float(row.get("conf", -1))
            if not raw_text or conf < 0:
                continue
            key = (int(row["page_num"]), int(row["block_num"]), int(row["par_num"]), int(row["line_num"]))
            word_num = int(row["word_num"])
            lines_dict[key].append((word_num, raw_text, conf))
        except Exception:
            continue

    candidate_spans_by_norm: dict[str, dict[str, Any]] = {}

    for line_idx, (line_key, words) in enumerate(sorted(lines_dict.items())):
        sorted_words = sorted(words, key=lambda x: x[0])
        raw_tokens = [w[1] for w in sorted_words]
        confs = [w[2] for w in sorted_words]
        line_mean_conf = sum(confs) / max(len(confs), 1)

        num_words = len(sorted_words)
        for n in range(1, max_n + 1):
            for i in range(num_words - n + 1):
                span_tokens = raw_tokens[i : i + n]
                span_confs = confs[i : i + n]
                span_text = " ".join(span_tokens)
                norm_span = normalize_answer(span_text)

                if not norm_span:
                    continue

                score, components = GenericSpanScorer.score_span(
                    tokens=span_tokens,
                    confidences=span_confs,
                    line_idx=line_idx,
                    line_conf=line_mean_conf,
                )

                # Deduplicate by keeping highest generic score per normalized span
                if norm_span not in candidate_spans_by_norm or score > candidate_spans_by_norm[norm_span]["score"]:
                    candidate_spans_by_norm[norm_span] = {
                        "norm_span": norm_span,
                        "raw_span": span_text,
                        "n_gram": n,
                        "score": score,
                        "components": components,
                        "line_idx": line_idx,
                    }

    # Sort descending by score, tie-break by shorter n-gram and alphabetical norm_span
    ranked_candidates = sorted(
        candidate_spans_by_norm.values(),
        key=lambda x: (-x["score"], x["n_gram"], x["norm_span"]),
    )
    return ranked_candidates


def run_ranking_feasibility_audit():
    benchmark_path = REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "l21_150_diagnostic" / "benchmark.json"
    sidecar_path = REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "l21_150_diagnostic" / "qa_dev_translations_en.json"

    with open(benchmark_path, encoding="utf-8") as f:
        bm_data = json.load(f)
    with open(sidecar_path, encoding="utf-8") as f:
        en_sidecar = json.load(f)

    en_map = {e["query_id"]: e.get("question_en", "") for e in en_sidecar.get("entries", [])}
    qa_dev_queries = [q for q in bm_data["queries"] if q.get("task_type") == "qa" and q.get("split") == "DEV"]

    # Focus especially on QA-23 (the relevant evidence recovery case)
    q = next(q for q in qa_dev_queries if q["query_id"] == "QA-23")
    qid = "QA-23"
    target_vid = q.get("video_id")
    start_f, end_f = int(q["proposed_interval"][0]), int(q["proposed_interval"][1])
    accepted_answers = tuple(q.get("accepted_answers", []))
    q_vi = q.get("question_vi", "")
    q_en = en_map.get(qid, "")

    print("=" * 130)
    print("SPRINT R3-S2E1: DEV-ONLY GENERIC OCR SPAN RANKING-FEASIBILITY AUDIT (ZERO GT KNOWLEDGE)")
    print(f"  • Auditing Query              : {qid} [Target: {target_vid}, GT: [{start_f}..{end_f}], Accepted: {accepted_answers}]")
    print(f"  • Scoring Rule                : Frozen GenericSpanScorer (Conf * Cleanliness * JunkPenalty * LenPrior * LineFactor)")
    print(f"  • Candidate Universe          : Contiguous 1..4 Token Spans (Deduplicated)")
    print("=" * 130)

    session_output = Path("/kaggle/working/output/ranking_audit") if Path("/kaggle/working").exists() else REPO_ROOT / "scratch" / "ranking_audit"
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

    # 1. RUN QUERY TO GET TOP-1 SECONDARY REFINED ANCHOR FROM RUNTIME
    req = QAQueryRequest(
        request_id=f"audit-e1-{qid}",
        query_id=qid,
        event_description=q_vi,
        question=q_vi,
        event_description_en=q_en if q_en else None,
        include_vi_variant=False,
        output_top_k=100,
        refine_top_n=3,
    )
    res = runtime.handle_qa_query(req)

    diag_file = runtime.output_root / res.get("artifacts", {}).get("qa_evidence_json", "")
    sec_meta = {}
    if diag_file.is_file():
        with open(diag_file, encoding="utf-8") as f:
            diags = json.load(f)
            sec_meta = diags.get("top1_secondary_refined_rescue", {})

    top1_vid = sec_meta.get("top1_video")
    sec_frame = sec_meta.get("secondary_refined_anchor")
    print(f"\n[RUNTIME RETRIEVAL & REFINEMENT RESULT]")
    print(f"  • Top-1 Nominated Video       : {top1_vid}")
    print(f"  • Secondary Refined Frame     : {sec_frame} (In GT? {top1_vid == target_vid and start_f <= int(sec_frame) <= end_f})")

    # 2. DECODE FRAME AND INVOKE OCR BACKEND
    video_rec = runtime.raw_video_registry.get(top1_vid)
    probe = runtime.decoder.probe(video_rec)
    dec_req = DecodeRequest(probe=probe, frame_ids=(int(sec_frame),), max_decoded_frames=10)
    dec_res = runtime.decoder.decode(dec_req)
    frame_image = dec_res.frames[0].image

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

    # 3. EXTRACT AND RANK ALL CONTIGUOUS 1..4 SPANS (FROZEN GENERIC SCORER)
    ranked_candidates = extract_and_score_all_spans(raw_stdout, max_n=4)
    total_candidates = len(ranked_candidates)

    print("\n" + "=" * 130)
    print(f"GENERIC RANKING RESULT: {total_candidates} UNIQUE SPANS RANKED (WITHOUT GT)")
    print("=" * 130)
    print(f"{'Rank':<5} | {'Normalized Span':<32} | {'Raw Span':<32} | {'n-gram':<6} | {'Score':<8} | {'Conf':<6} | {'Clean':<6} | {'Line'}")
    print("-" * 130)
    for rank, c in enumerate(ranked_candidates[:20], start=1):
        comp = c["components"]
        print(f"#{rank:<4} | {c['norm_span'][:32]:<32} | {c['raw_span'][:32]:<32} | {c['n_gram']:<6} | {c['score']:.4f} | {comp['mean_conf']:5.1f}% | {comp['cleanliness']:5.2f} | Line {c['line_idx']}")
    print("=" * 130)

    # 4. OFFLINE EVALUATION OF ACCEPTED ANSWER RANK (DEV DIAGNOSTIC ONLY)
    matching_ranks = []
    for rank, c in enumerate(ranked_candidates, start=1):
        if answer_matches(c["norm_span"], accepted_answers):
            matching_ranks.append((rank, c))

    print("\n" + "=" * 130)
    print("OFFICIAL EVALUATION OF ACCEPTED ANSWER IN GENERIC RANKING:")
    print("=" * 130)
    print(f"  • Accepted Answers            : {accepted_answers}")
    print(f"  • Total Candidates Ranked     : {total_candidates}")
    print(f"  • Matches Found in Universe   : {len(matching_ranks)}")

    if matching_ranks:
        best_match_rank, best_match_obj = matching_ranks[0]
        comp = best_match_obj["components"]
        print(f"\n  🎯 BEST MATCHING CANDIDATE:")
        print(f"     • Normalized Span          : {repr(best_match_obj['norm_span'])}")
        print(f"     • Raw Span Text            : {repr(best_match_obj['raw_span'])}")
        print(f"     • Generic Score Rank       : #{best_match_rank} / {total_candidates}")
        print(f"     • Final Generic Score      : {best_match_obj['score']:.4f}")
        print(f"     • Mean Word Confidence     : {comp['mean_conf']:.1f}%")
        print(f"     • Cleanliness Factor       : {comp['cleanliness']:.2f}")
        print(f"     • Length Prior             : {comp['len_prior']:.2f}")
        print(f"     • Capitalization Bonus     : {comp['cap_bonus']:.2f}")

        cov_1 = best_match_rank <= 1
        cov_5 = best_match_rank <= 5
        cov_10 = best_match_rank <= 10
        cov_20 = best_match_rank <= 20

        print(f"\n  📊 COVERAGE AT TOP-K SLOTS:")
        print(f"     • Covered @ Top-1 (Rank <= 1)  : {'YES 🟢' if cov_1 else 'NO ⚪'} (Rank = {best_match_rank})")
        print(f"     • Covered @ Top-5 (Rank <= 5)  : {'YES 🟢' if cov_5 else 'NO ⚪'} (Rank = {best_match_rank})")
        print(f"     • Covered @ Top-10 (Rank <= 10): {'YES 🟢' if cov_10 else 'NO ⚪'} (Rank = {best_match_rank})")
        print(f"     • Covered @ Top-20 (Rank <= 20): {'YES 🟢' if cov_20 else 'NO ⚪'} (Rank = {best_match_rank})")

        print("\n" + "=" * 130)
        print("RANKING FEASIBILITY AUDIT CONCLUSION:")
        print("=" * 130)
        if cov_5:
            conclusion = "RANKING_FEASIBILITY_PASS (Answer placed in Top 5 rescue slots without GT!) ✅"
        else:
            conclusion = f"RANKING_FEASIBILITY_ANALYSIS_REQUIRED (Answer rank = #{best_match_rank}, outside Top 5 slots)"
        print(f"  • Audit Conclusion            : {conclusion}")
        print("=" * 130)
    else:
        print("  • Matching Candidate Not Found in Universe ❌")


if __name__ == "__main__":
    run_ranking_feasibility_audit()
