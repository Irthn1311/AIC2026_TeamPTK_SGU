# ==============================================================================================================
# Comprehensive Full Benchmark Runner: KIS (38) + QA (38 with S2E1) + TRAKE (38) on DEV
# ==============================================================================================================

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
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
from system_tai.kis.session_schema import (
    QAQueryRequest,
    QueryRequest,
    SessionConfig,
    TRAKEQueryRequest,
)
from system_tai.qa.grounding import (
    QA_CANDIDATE_ORDER_ROUND_ROBIN,
    QAVideoConditionedEvidenceConfig,
)
from system_tai.qa.object_provider import ObjectAnswerProviderConfig
from system_tai.qa.ocr_provider import OCRAnswerProviderConfig
from system_tai.qa.visual_ontology import VisualOntologyConfig
from system_tai.quality.l21_150_answers import answer_matches, normalize_answer

OFFICIAL_K = (1, 5, 20, 50, 100)


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


def run_comprehensive_benchmark():
    benchmark_path = REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "l21_150_diagnostic" / "benchmark.json"
    qa_sidecar_path = REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "l21_150_diagnostic" / "qa_dev_translations_en.json"
    trake_sidecar_path = REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "l21_150_diagnostic" / "trake_dev_translations_en.json"
    ontology_path = REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "l21_150_diagnostic" / "qa_dev_visual_ontology.json"

    with open(benchmark_path, encoding="utf-8") as f:
        bm_data = json.load(f)

    qa_en_map = {}
    if qa_sidecar_path.is_file():
        with open(qa_sidecar_path, encoding="utf-8") as f:
            qa_en_map = {e["query_id"]: e.get("question_en", "") for e in json.load(f).get("entries", [])}

    trake_en_map = {}
    if trake_sidecar_path.is_file():
        with open(trake_sidecar_path, encoding="utf-8") as f:
            trake_en_map = {e["query_id"]: e.get("events_en", []) for e in json.load(f).get("entries", [])}

    dev_queries = [q for q in bm_data["queries"] if q.get("split") == "DEV"]
    kis_queries = [q for q in dev_queries if q.get("task_type") == "kis"]
    qa_queries = [q for q in dev_queries if q.get("task_type") == "qa"]
    trake_queries = [q for q in dev_queries if q.get("task_type") == "trake"]

    print("=" * 145)
    print("ALL-TASK COMPREHENSIVE OVERNIGHT BENCHMARK RUNNER (KIS + QA + TRAKE DEV)")
    print(f"  • Total KIS Queries   : {len(kis_queries)}")
    print(f"  • Total QA Queries    : {len(qa_queries)} (with S2E1 Top-5 Spans + Parser Fix)")
    print(f"  • Total TRAKE Queries : {len(trake_queries)}")
    print(f"  • Total Queries       : {len(dev_queries)}")
    print("=" * 145)

    session_output = Path("/kaggle/working/output/comprehensive_benchmark") if Path("/kaggle/working").exists() else REPO_ROOT / "scratch" / "comprehensive_benchmark"
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
        top1_secondary_refined_rescue_span_candidateizer=True,
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
        qa_visual_ontology_config=VisualOntologyConfig(enabled=ontology_path.exists(), ontology_path=ontology_path if ontology_path.exists() else None),
        qa_ocr_answer_provider_config=resolve_ocr_config(),
        qa_object_answer_provider_config=ObjectAnswerProviderConfig(enabled=False),
    )

    print("\n--- BOOTSTRAPPING RUNTIME ---")
    t_boot_start = time.time()
    runtime = OperationalKISRuntime.bootstrap(config)
    print(f"Runtime bootstrap completed in {time.time() - t_boot_start:.2f}s.")

    # =========================================================================
    # PART 1: KIS BENCHMARK (38 QUERIES)
    # =========================================================================
    print("\n" + "=" * 145)
    print("PART 1: RUNNING KIS BENCHMARK (38 DEV QUERIES)...")
    print("=" * 145)

    kis_results = []
    kis_video_hits = {k: 0 for k in OFFICIAL_K}
    kis_frame_hits = {k: 0 for k in OFFICIAL_K}

    for idx, q in enumerate(kis_queries, start=1):
        t_q_start = time.time()
        qid = q["query_id"]
        target_vid = q.get("video_id")
        start_f, end_f = int(q["proposed_interval"][0]), int(q["proposed_interval"][1])
        text_vi = q.get("text", "")

        req = QueryRequest(
            request_id=f"kis-{qid}",
            query_id=qid,
            query_vi=text_vi,
            output_top_k=100,
            refine_top_n=3,
        )
        res = runtime.handle_query(req)
        preds = res.get("results", [])

        # Evaluate KIS Recall
        v_hit_rank = None
        f_hit_rank = None
        for r_idx, p in enumerate(preds, start=1):
            vid = p.get("video_id")
            fid = int(p.get("frame_id", -1))
            if vid == target_vid and v_hit_rank is None:
                v_hit_rank = r_idx
            if vid == target_vid and start_f <= fid <= end_f and f_hit_rank is None:
                f_hit_rank = r_idx

        for k in OFFICIAL_K:
            if v_hit_rank and v_hit_rank <= k:
                kis_video_hits[k] += 1
            if f_hit_rank and f_hit_rank <= k:
                kis_frame_hits[k] += 1

        elapsed = time.time() - t_q_start
        status_str = f"FrameHit @{f_hit_rank}" if f_hit_rank else (f"VidHit @{v_hit_rank}" if v_hit_rank else "MISS")
        print(f"  [{idx:02d}/{len(kis_queries):02d}] {qid:<8} | Target: {target_vid} | Status: {status_str:<15} | Latency: {elapsed:.2f}s")

        kis_results.append({
            "query_id": qid,
            "target_vid": target_vid,
            "v_hit_rank": v_hit_rank,
            "f_hit_rank": f_hit_rank,
            "latency": elapsed,
        })

    # =========================================================================
    # PART 2: QA BENCHMARK (38 QUERIES with S2E1 Treatment)
    # =========================================================================
    print("\n" + "=" * 145)
    print("PART 2: RUNNING QA BENCHMARK (38 DEV QUERIES WITH S2E1 TREATMENT)...")
    print("=" * 145)

    qa_results = []
    qa_strict_hits = {k: 0 for k in OFFICIAL_K}

    for idx, q in enumerate(qa_queries, start=1):
        t_q_start = time.time()
        qid = q["query_id"]
        target_vid = q.get("video_id")
        start_f, end_f = int(q["proposed_interval"][0]), int(q["proposed_interval"][1])
        accepted_answers = tuple(q.get("accepted_answers", []))
        q_vi = q.get("question_vi", "")
        q_en = qa_en_map.get(qid)

        req = QAQueryRequest(
            request_id=f"qa-{qid}",
            query_id=qid,
            event_description=q_vi,
            question=q_vi,
            event_description_en=q_en if q_en else None,
            include_vi_variant=False,
            output_top_k=100,
            refine_top_n=3,
        )
        res = runtime.handle_qa_query(req)
        preds = res.get("predictions", [])

        # Evaluate QA Strict Recall
        strict_hit_rank = None
        for r_idx, p in enumerate(preds, start=1):
            vid = p.get("video_id")
            fid = int(p.get("frame_id", -1))
            ans = str(p.get("answer", ""))
            in_gt = (vid == target_vid and start_f <= fid <= end_f)
            if in_gt and answer_matches(ans, accepted_answers) and strict_hit_rank is None:
                strict_hit_rank = r_idx

        for k in OFFICIAL_K:
            if strict_hit_rank and strict_hit_rank <= k:
                qa_strict_hits[k] += 1

        elapsed = time.time() - t_q_start
        status_str = f"STRICT HIT @{strict_hit_rank}" if strict_hit_rank else "NO HIT"
        print(f"  [{idx:02d}/{len(qa_queries):02d}] {qid:<8} | Target: {target_vid} | Status: {status_str:<18} | Latency: {elapsed:.2f}s")

        qa_results.append({
            "query_id": qid,
            "target_vid": target_vid,
            "strict_hit_rank": strict_hit_rank,
            "latency": elapsed,
        })

    # =========================================================================
    # PART 3: TRAKE BENCHMARK (38 QUERIES)
    # =========================================================================
    print("\n" + "=" * 145)
    print("PART 3: RUNNING TRAKE BENCHMARK (38 DEV QUERIES)...")
    print("=" * 145)

    trake_results = []
    trake_chain_hits = {k: 0 for k in OFFICIAL_K}

    for idx, q in enumerate(trake_queries, start=1):
        t_q_start = time.time()
        qid = q["query_id"]
        target_vid = q.get("video_id")
        events_gt = q.get("events", [])
        events_vi = [e.get("description", "") for e in events_gt]
        events_en = trake_en_map.get(qid, [])

        req = TRAKEQueryRequest(
            request_id=f"trake-{qid}",
            query_id=qid,
            event_descriptions=tuple(events_vi),
            event_descriptions_en=tuple(events_en) if events_en else None,
            include_vi_variant=False,
            output_top_k=100,
            refine_top_n=3,
        )
        res = runtime.handle_trake_query(req)
        preds = res.get("predictions", [])

        # Evaluate TRAKE Recall
        chain_hit_rank = None
        for r_idx, p in enumerate(preds, start=1):
            vid = p.get("video_id")
            frame_ids = p.get("actual_frame_ids", [])
            if vid == target_vid and len(frame_ids) == len(events_gt):
                all_events_in_gt = True
                for f_id, ev_gt in zip(frame_ids, events_gt):
                    gt_start = int(ev_gt["interval"][0])
                    gt_end = int(ev_gt["interval"][1])
                    if not (gt_start <= int(f_id) <= gt_end):
                        all_events_in_gt = False
                        break
                if all_events_in_gt and chain_hit_rank is None:
                    chain_hit_rank = r_idx

        for k in OFFICIAL_K:
            if chain_hit_rank and chain_hit_rank <= k:
                trake_chain_hits[k] += 1

        elapsed = time.time() - t_q_start
        status_str = f"CHAIN HIT @{chain_hit_rank}" if chain_hit_rank else "NO HIT"
        print(f"  [{idx:02d}/{len(trake_queries):02d}] {qid:<8} | Target: {target_vid} | Status: {status_str:<18} | Latency: {elapsed:.2f}s")

        trake_results.append({
            "query_id": qid,
            "target_vid": target_vid,
            "chain_hit_rank": chain_hit_rank,
            "latency": elapsed,
        })

    # =========================================================================
    # COMPREHENSIVE OVERNIGHT SUMMARY REPORT
    # =========================================================================
    n_kis = len(kis_queries)
    n_qa = len(qa_queries)
    n_trake = len(trake_queries)

    print("\n" + "=" * 145)
    print("🏆 FINAL COMPREHENSIVE BENCHMARK EVALUATION RESULTS SUMMARY (114 DEV QUERIES)")
    print("=" * 145)

    print(f"\n📊 1. KIS EVALUATION METRICS ({n_kis} Queries):")
    print(f"  • Video Recall@1   : {kis_video_hits[1]:2d}/{n_kis} ({kis_video_hits[1]/n_kis:.1%})  |  Frame Recall@1   : {kis_frame_hits[1]:2d}/{n_kis} ({kis_frame_hits[1]/n_kis:.1%})")
    print(f"  • Video Recall@5   : {kis_video_hits[5]:2d}/{n_kis} ({kis_video_hits[5]/n_kis:.1%})  |  Frame Recall@5   : {kis_frame_hits[5]:2d}/{n_kis} ({kis_frame_hits[5]/n_kis:.1%})")
    print(f"  • Video Recall@20  : {kis_video_hits[20]:2d}/{n_kis} ({kis_video_hits[20]/n_kis:.1%})  |  Frame Recall@20  : {kis_frame_hits[20]:2d}/{n_kis} ({kis_frame_hits[20]/n_kis:.1%})")
    print(f"  • Video Recall@50  : {kis_video_hits[50]:2d}/{n_kis} ({kis_video_hits[50]/n_kis:.1%})  |  Frame Recall@50  : {kis_frame_hits[50]:2d}/{n_kis} ({kis_frame_hits[50]/n_kis:.1%})")
    print(f"  • Video Recall@100 : {kis_video_hits[100]:2d}/{n_kis} ({kis_video_hits[100]/n_kis:.1%})  |  Frame Recall@100 : {kis_frame_hits[100]:2d}/{n_kis} ({kis_frame_hits[100]:2d}/{n_kis} -> {kis_frame_hits[100]/n_kis:.1%})")

    print(f"\n📊 2. QA EVALUATION METRICS ({n_qa} Queries with S2E1 + Parser Fix):")
    print(f"  • Strict QA Recall@1   : {qa_strict_hits[1]:2d}/{n_qa} ({qa_strict_hits[1]/n_qa:.1%})")
    print(f"  • Strict QA Recall@5   : {qa_strict_hits[5]:2d}/{n_qa} ({qa_strict_hits[5]/n_qa:.1%})")
    print(f"  • Strict QA Recall@20  : {qa_strict_hits[20]:2d}/{n_qa} ({qa_strict_hits[20]/n_qa:.1%})")
    print(f"  • Strict QA Recall@50  : {qa_strict_hits[50]:2d}/{n_qa} ({qa_strict_hits[50]/n_qa:.1%})")
    print(f"  • Strict QA Recall@100 : {qa_strict_hits[100]:2d}/{n_qa} ({qa_strict_hits[100]/n_qa:.1%}) -> Total Strict Hits: {qa_strict_hits[100]}")

    print(f"\n📊 3. TRAKE EVALUATION METRICS ({n_trake} Queries):")
    print(f"  • Chain Recall@1   : {trake_chain_hits[1]:2d}/{n_trake} ({trake_chain_hits[1]/n_trake:.1%})")
    print(f"  • Chain Recall@5   : {trake_chain_hits[5]:2d}/{n_trake} ({trake_chain_hits[5]/n_trake:.1%})")
    print(f"  • Chain Recall@20  : {trake_chain_hits[20]:2d}/{n_trake} ({trake_chain_hits[20]/n_trake:.1%})")
    print(f"  • Chain Recall@50  : {trake_chain_hits[50]:2d}/{n_trake} ({trake_chain_hits[50]/n_trake:.1%})")
    print(f"  • Chain Recall@100 : {trake_chain_hits[100]:2d}/{n_trake} ({trake_chain_hits[100]/n_trake:.1%}) -> Total Chain Hits: {trake_chain_hits[100]}")

    print("\n" + "=" * 145)
    print("COMPREHENSIVE OVERNIGHT BENCHMARK RUN COMPLETED SUCCESSFULLY! 🎉")
    print("=" * 145)

    # Save summary report artifact
    summary_report = {
        "kis": {
            "total_queries": n_kis,
            "video_recall": {f"R@{k}": kis_video_hits[k] for k in OFFICIAL_K},
            "frame_recall": {f"R@{k}": kis_frame_hits[k] for k in OFFICIAL_K},
        },
        "qa": {
            "total_queries": n_qa,
            "strict_recall": {f"R@{k}": qa_strict_hits[k] for k in OFFICIAL_K},
        },
        "trake": {
            "total_queries": n_trake,
            "chain_recall": {f"R@{k}": trake_chain_hits[k] for k in OFFICIAL_K},
        },
    }
    report_file = session_output / "final_benchmark_summary.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2, ensure_ascii=False)
    print(f"Summary JSON saved to {report_file}")


if __name__ == "__main__":
    run_comprehensive_benchmark()
