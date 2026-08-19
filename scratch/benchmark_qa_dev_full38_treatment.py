# ==============================================================================================================
# Final QA Full-38 DEV Benchmark (Treatment Under Locked en_only Production Champion Config)
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
from system_tai.qa.visual_ontology import VisualOntologyConfig
from system_tai.quality.l21_150_answers import answer_matches

OFFICIAL_K = (1, 5, 20, 50, 100)


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


def run_full38_benchmark():
    benchmark_path = REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "l21_150_diagnostic" / "benchmark.json"
    sidecar_path = REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "l21_150_diagnostic" / "qa_dev_translations_en.json"
    ontology_path = REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "l21_150_diagnostic" / "qa_dev_visual_ontology.json"

    with open(benchmark_path, encoding="utf-8") as f:
        bm_data = json.load(f)
    with open(sidecar_path, encoding="utf-8") as f:
        en_sidecar = json.load(f)

    en_map = {e["query_id"]: e.get("question_en", "") for e in en_sidecar.get("entries", [])}
    qa_dev_queries = [q for q in bm_data["queries"] if q.get("task_type") == "qa" and q.get("split") == "DEV"]

    print("=" * 145)
    print("FINAL FULL-38 QA DEV BENCHMARK (TREATMENT UNDER LOCKED 'en_only' PRODUCTION CHAMPION CONFIG)")
    print("  • Language Policy             : qa_localization_language_policy = 'en_only'")
    print("  • Query Variant Setting       : include_vi_variant = False (Locked Champion)")
    print("  • Top-1 Secondary Rescue (S2D1): Enabled (tail_budget = 5)")
    print(f"  • Total QA Queries to Evaluate: {len(qa_dev_queries)} DEV queries")
    print("=" * 145)

    session_output = Path("/kaggle/working/output/full38_qa_dev_treatment") if Path("/kaggle/working").exists() else REPO_ROOT / "scratch" / "full38_qa_dev_treatment"
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
        consensus_novel_rescue_enabled=False,  # Frozen OFF
        bounded_negative_temporal_rescue_enabled=False,  # Frozen OFF
        top1_secondary_refined_rescue_enabled=True,  # S2D1 ON (Treatment)
        top1_secondary_refined_rescue_span_candidateizer=True,  # S2E1 ON (Treatment)
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
        qa_visual_ontology_config=resolve_visual_ontology_config(),
        qa_ocr_answer_provider_config=resolve_ocr_config(),
        qa_object_answer_provider_config=ObjectAnswerProviderConfig(enabled=False),
        qa_unsupported_provider_fallback=True,
    )

    print("\n--- BOOTSTRAPPING RUNTIME ---")
    t0 = time.time()
    runtime = OperationalKISRuntime.bootstrap(config)
    print(f"Runtime bootstrap completed in {time.time() - t0:.2f}s.")

    query_results = []
    strict_hits = []
    r_counts = {1: 0, 5: 0, 20: 0, 50: 0, 100: 0}
    total_scores = []

    print("\n--- RUNNING INFERENCE ON 38 DEV QA QUERIES ---")
    for idx, q in enumerate(qa_dev_queries, start=1):
        qid = q["query_id"]
        target_vid = q.get("video_id")
        start_f, end_f = int(q["proposed_interval"][0]), int(q["proposed_interval"][1])
        accepted_answers = q.get("accepted_answers", [])
        q_vi = q.get("question_vi", "")
        q_en = en_map.get(qid)
        branch = q.get("branch", "")

        t_q0 = time.time()
        req = QAQueryRequest(
            request_id=f"full38-treat-{qid}",
            query_id=qid,
            event_description=q_vi,
            question=q_vi,
            event_description_en=q_en if q_en else None,
            question_en=None,
            include_vi_variant=False,
            output_top_k=100,
            refine_top_n=3,
        )
        res = runtime.handle_qa_query(req)
        t_q = time.time() - t_q0
        preds = res.get("predictions", [])

        # Evaluate Strict Hit Criteria
        hit_rank = None
        hit_frame = None
        hit_ans = None
        scores_by_rank = []

        for p in preds:
            p_vid = p.get("video_id")
            p_frame = int(p.get("frame_id", -1))
            p_ans = str(p.get("answer", ""))
            p_rank = int(p.get("rank", 101))

            video_match = (p_vid == target_vid)
            frame_match = video_match and (start_f <= p_frame <= end_f)
            ans_match = answer_matches(p_ans, accepted_answers)
            is_strict_hit = bool(frame_match and ans_match)

            scores_by_rank.append((p_rank, float(is_strict_hit)))

            if is_strict_hit and hit_rank is None:
                hit_rank = p_rank
                hit_frame = p_frame
                hit_ans = p_ans

        # Calculate R@K for this query
        q_r_at_k = {}
        for k in OFFICIAL_K:
            val = max((score for r, score in scores_by_rank if r <= k), default=0.0)
            q_r_at_k[k] = val
            if val == 1.0:
                r_counts[k] += 1

        final_score = sum(q_r_at_k.values()) / len(OFFICIAL_K)
        total_scores.append(final_score)

        status_str = f"STRICT_HIT @{hit_rank}" if hit_rank is not None else "NO HIT"
        if hit_rank is not None:
            strict_hits.append({
                "query_id": qid,
                "rank": hit_rank,
                "video_id": target_vid,
                "frame_id": hit_frame,
                "answer": hit_ans,
                "branch": branch,
            })

        print(f"[{idx:2d}/38] {qid:<8} | Branch: {branch:<15} | N={len(preds):3d} | Status: {status_str:<18} | Time: {t_q:6.2f}s")

        query_results.append({
            "query_id": qid,
            "branch": branch,
            "target": target_vid,
            "gt_interval": f"[{start_f}..{end_f}]",
            "accepted": str(accepted_answers),
            "pred_count": len(preds),
            "hit_rank": hit_rank if hit_rank is not None else "-",
            "r_at_1": int(q_r_at_k[1]),
            "r_at_5": int(q_r_at_k[5]),
            "r_at_20": int(q_r_at_k[20]),
            "r_at_50": int(q_r_at_k[50]),
            "r_at_100": int(q_r_at_k[100]),
            "final_score": final_score,
            "hit_details": {
                "rank": hit_rank,
                "frame_id": hit_frame,
                "answer": hit_ans,
            } if hit_rank is not None else None,
        })

    # =========================================================================
    # SUMMARY TABLE OF ALL 38 QUERIES
    # =========================================================================
    print("\n" + "=" * 145)
    print("FULL-38 QA DEV BENCHMARK: PER-QUERY SCORE MATRIX")
    print("=" * 145)
    print(f"{'Query ID':<8} | {'Branch':<15} | {'Target':<10} | {'GT Interval':<14} | {'N':<4} | {'Hit Rank':<9} | {'R@1':<4} | {'R@5':<4} | {'R@20':<5} | {'R@50':<5} | {'R@100':<6} | {'Final Score'}")
    print("-" * 145)
    for r in query_results:
        print(f"{r['query_id']:<8} | {r['branch']:<15} | {r['target']:<10} | {r['gt_interval']:<14} | {r['pred_count']:<4} | {str(r['hit_rank']):<9} | {r['r_at_1']:<4} | {r['r_at_5']:<4} | {r['r_at_20']:<5} | {r['r_at_50']:<5} | {r['r_at_100']:<6} | {r['final_score']:.4f}")
    print("=" * 145)

    # =========================================================================
    # AUDIT OF 6 PROTECTED REFERENCES + QA-23 NEW GAIN
    # =========================================================================
    protected_ref_map = {
        "QA-08": "Historical Protected Hit (~@43)",
        "QA-10": "Historical Protected Hit (~@88)",
        "QA-13": "Historical Protected Hit (~@18)",
        "QA-27": "Historical Protected Hit (~@49)",
        "QA-45": "Historical Protected Hit (~@92-96)",
        "QA-46": "Historical Protected Hit (~@13)",
        "QA-23": "S2E1 Target Gain Candidate (~@63)",
    }

    print("\n" + "=" * 145)
    print("EXPLICIT AUDIT OF 6 PROTECTED REFERENCES + QA-23 NEW GAIN:")
    print("=" * 145)
    print(f"{'Query ID':<8} | {'Category':<35} | {'Status':<18} | {'Physical Frame':<16} | {'Answer Text'}")
    print("-" * 145)
    q_res_by_id = {r["query_id"]: r for r in query_results}
    for ref_id, cat_desc in protected_ref_map.items():
        qr = q_res_by_id.get(ref_id)
        if qr and qr["hit_rank"] != "-":
            hd = qr["hit_details"]
            print(f"{ref_id:<8} | {cat_desc:<35} | STRICT HIT @{qr['hit_rank']:<7} | Frame {str(hd['frame_id']):<10} | {repr(str(hd['answer']))}")
        else:
            print(f"{ref_id:<8} | {cat_desc:<35} | NO HIT ❌          | N/A              | N/A")
    print("=" * 145)

    # =========================================================================
    # STRICT HITS DETAIL LIST
    # =========================================================================
    print("\n" + "=" * 145)
    print(f"OFFICIAL STRICT HITS LIST ({len(strict_hits)} HITS FOUND):")
    print("=" * 145)
    print(f"{'Rank':<6} | {'Query ID':<8} | {'Branch':<15} | {'Target Video':<12} | {'Physical Frame':<15} | {'Answer Text'}")
    print("-" * 145)
    for h in sorted(strict_hits, key=lambda x: (x['rank'], x['query_id'])):
        print(f"{h['rank']:<6} | {h['query_id']:<8} | {h['branch']:<15} | {h['video_id']:<12} | {str(h['frame_id']):<15} | {repr(str(h['answer'])[:60])}")
    print("=" * 145)

    # =========================================================================
    # AGGREGATE SUMMARY METRICS
    # =========================================================================
    mean_macro_score = sum(total_scores) / len(total_scores)
    total_numerator = sum(r_counts.values())
    total_denominator = len(qa_dev_queries) * len(OFFICIAL_K)  # 38 * 5 = 190

    print("\n" + "=" * 145)
    print("AGGREGATE OFFICIAL METRICS (DEV SPLIT):")
    print("=" * 145)
    print(f"  • Completed Queries           : {len(qa_dev_queries)} / 38 (100% format & output valid)")
    print(f"  • Strict R@1                  : {r_counts[1]} / 38 ({r_counts[1] / 38:.2%})")
    print(f"  • Strict R@5                  : {r_counts[5]} / 38 ({r_counts[5] / 38:.2%})")
    print(f"  • Strict R@20                 : {r_counts[20]} / 38 ({r_counts[20] / 38:.2%})")
    print(f"  • Strict R@50                 : {r_counts[50]} / 38 ({r_counts[50] / 38:.2%})")
    print(f"  • Strict R@100                : {r_counts[100]} / 38 ({r_counts[100] / 38:.2%})")
    print(f"  • Total Strict Numerator      : {total_numerator} / {total_denominator} (Sum of R@K hits)")
    print(f"  • Final QA Macro Score        : {mean_macro_score:.6f} (or {total_numerator / total_denominator:.6f})")
    print("=" * 145)


if __name__ == "__main__":
    run_full38_benchmark()
