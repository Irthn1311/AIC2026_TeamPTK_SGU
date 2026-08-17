# ==============================================================================================================
# Phase R3-S2A: Targeted Conversion Diagnostic (Fast Bounded Forensic with Refined Taxonomy)
# ==============================================================================================================

import argparse
import hashlib
import json
import math
import os
import sys
import time
import unicodedata
from collections import Counter
from enum import StrEnum
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

try:
    import clip
except ImportError:
    import subprocess
    print("Installing openai-clip dependency...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "openai-clip", "ftfy", "regex", "tqdm"], check=False)

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

# 7 Targeted Bounded Queries (Skipping QA-20 to avoid CPU refinement hang)
DEFAULT_TARGETED_QUERIES = [
    "QA-01",  # S1A / OCR branch (biển cảnh báo)
    "QA-02",  # Object branch
    "QA-23",  # Color branch
    "QA-26",  # S1A / Visual branch
    "QA-30",  # Action branch
    "QA-31",  # Visual branch
    "QA-46",  # Positive control / Action branch (thủ công thổ cẩm)
]


class FailureClass(StrEnum):
    STRICT_HIT = "STRICT_HIT"
    VIDEO_ABSENT = "VIDEO_ABSENT"
    TARGET_VIDEO_NO_EVIDENCE = "TARGET_VIDEO_NO_EVIDENCE"
    TEMPORAL_MISS = "TEMPORAL_MISS"
    ANSWER_MISS = "ANSWER_MISS"
    ALLOCATION_MISS = "ALLOCATION_MISS"
    QUERY_TIMEOUT = "QUERY_TIMEOUT"
    OTHER_UNKNOWN = "OTHER_UNKNOWN"


def normalize_text(t: str) -> str:
    if not t:
        return ""
    t = unicodedata.normalize("NFKC", str(t)).casefold()
    return "".join(c for c in t if c.isalnum() or c.isspace()).strip()


def parse_benchmark_gt_interval(q: dict[str, Any]) -> tuple[int, int]:
    if "proposed_interval" in q and isinstance(q["proposed_interval"], (list, tuple)) and len(q["proposed_interval"]) == 2:
        return int(q["proposed_interval"][0]), int(q["proposed_interval"][1])
    if "start_frame_id" in q and "end_frame_id" in q:
        return int(q["start_frame_id"]), int(q["end_frame_id"])
    return -1, -1


def run_targeted_diagnostic(
    benchmark_path: Path,
    dev_en_sidecar_path: Path,
    manifest_cache_path: Path,
    input_root: Path = Path("/kaggle/input"),
    device: str = "auto",
    selected_queries: list[str] | None = None,
):
    target_qids = selected_queries or DEFAULT_TARGETED_QUERIES

    print("=" * 125)
    print("ROUND-3 SPRINT 2A: TARGETED 7-QUERY CONVERSION FORENSIC (BOUNDED PER-QUERY)")
    print("=" * 125)

    benchmark_bytes = benchmark_path.read_bytes()
    benchmark_sha = hashlib.sha256(benchmark_bytes).hexdigest()
    sidecar_bytes = dev_en_sidecar_path.read_bytes()
    sidecar_sha = hashlib.sha256(sidecar_bytes).hexdigest()

    print(f"Benchmark File  : {benchmark_path.name} (SHA256: {benchmark_sha[:16]}...)")
    print(f"Sidecar File    : {dev_en_sidecar_path.name} (SHA256: {sidecar_sha[:16]}...)")
    print(f"Targeted Queries: {target_qids}")

    with open(benchmark_path, encoding="utf-8") as f:
        bm_data = json.load(f)

    with open(dev_en_sidecar_path, encoding="utf-8") as f:
        en_sidecar = json.load(f)

    en_map = {e["query_id"]: e.get("question_en", "") for e in en_sidecar.get("entries", [])}
    all_qa_queries = {q["query_id"]: q for q in bm_data["queries"] if q.get("task_type") == "qa"}

    # 1. Bootstrap Runtime with Exact Champion R2G1 Configuration
    print("\n--- BOOTSTRAPPING CHAMPION RUNTIME ---")
    session_output = Path("/kaggle/working/output/targeted_diagnostic") if Path("/kaggle/working").exists() else REPO_ROOT / "scratch" / "targeted_diagnostic"
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
    )

    config = SessionConfig(
        input_root=input_root,
        manifest_cache=manifest_cache_path,
        output_root=session_output,
        device=device,
        allow_model_download=True,
        default_output_top_k=100,
        qa_video_conditioned_evidence_config=evidence_config,
    )
    runtime = OperationalKISRuntime.bootstrap(config)
    print("Runtime bootstrapped successfully.")

    # 2. Execute Targeted Queries with Exact Stage Logging
    print("\n" + "=" * 125)
    print("EXECUTING TARGETED DIAGNOSTIC QUERIES")
    print("=" * 125)

    forensic_results = []
    classification_counts: Counter[str] = Counter()

    for idx, qid in enumerate(target_qids, start=1):
        if qid not in all_qa_queries:
            print(f"[{idx}/{len(target_qids)}] {qid} NOT FOUND in benchmark queries!")
            continue

        q = all_qa_queries[qid]
        target_vid = q.get("video_id")
        start_f, end_f = parse_benchmark_gt_interval(q)
        gt_answers = [normalize_text(a) for a in q.get("accepted_answers", [])]
        q_vi = q.get("question_vi", "")
        q_en = en_map.get(qid, "")
        branch = q.get("branch", "General")

        print(f"\n[START] [{idx}/{len(target_qids)}] {qid} ({branch:<12}) [Target: {target_vid}, GT: [{start_f}..{end_f}]]...")
        t_q0 = time.time()
        req = QAQueryRequest(
            request_id=f"targeted-{qid}",
            query_id=qid,
            event_description=q_vi,
            question=q_vi,
            event_description_en=q_en if q_en else None,
            question_en=q_en if q_en else None,
            include_vi_variant=False if q_en else True,
            output_top_k=100,
        )

        res = runtime.handle_qa_query(req)
        preds = res.get("predictions", [])
        t_elapsed = time.time() - t_q0

        # Read evidence diagnostics JSON artifact created by runtime
        diagnostics = {}
        ev_rel_path = res.get("artifacts", {}).get("qa_evidence_json")
        if ev_rel_path:
            ev_file = runtime.output_root / ev_rel_path
            if ev_file.exists():
                try:
                    with open(ev_file, encoding="utf-8") as ef:
                        diagnostics = json.load(ef)
                except Exception as e:
                    print(f"      Warning reading {ev_file}: {e}")

        # Stage A: Champion Nomination Pool
        selected_video_ids = diagnostics.get("selected_video_ids", [])
        if not selected_video_ids and preds:
            selected_video_ids = list(dict.fromkeys([p.get("video_id") for p in preds if p.get("video_id")]))
        vid_rank_in_pool = selected_video_ids.index(target_vid) + 1 if target_vid in selected_video_ids else None

        # Stage B: Evidence Bank Records (Internal Candidates before Top100)
        evidence_records = diagnostics.get("evidence", [])
        usable_candidates = diagnostics.get("usable_candidates", [])

        target_frames: list[int] = []
        target_answers: list[tuple[int, list[str]]] = []

        for ev in evidence_records:
            if ev.get("video_id") == target_vid:
                f_id = ev.get("output_frame_id") or ev.get("refined_frame_id") or ev.get("candidate_frame_id")
                if f_id is not None:
                    target_frames.append(int(f_id))
                    ans_list = [normalize_text(str(ev.get("answer")))] if ev.get("answer") else []
                    target_answers.append((int(f_id), ans_list))

        if not target_frames:
            for uc in usable_candidates:
                if uc.get("video_id") == target_vid and uc.get("frame_id") is not None:
                    f_id = int(uc.get("frame_id"))
                    target_frames.append(f_id)
                    ans_list = [normalize_text(str(a)) for a in uc.get("answers", []) if a]
                    target_answers.append((f_id, ans_list))

        # Check Final Top 100 hit
        hit_rank = None
        for p in preds:
            p_vid = p.get("video_id")
            p_frame = int(p.get("frame_id", -1))
            p_ans = normalize_text(str(p.get("answer", "")))
            if p_vid == target_vid and start_f <= p_frame <= end_f and p_ans in gt_answers:
                hit_rank = p.get("rank")
                break

        # Classify Causal Bottleneck with Refined Taxonomy
        in_gt_frames = [f for f in target_frames if start_f <= f <= end_f]
        nearest_dist = min([abs(f - start_f) for f in target_frames] + [abs(f - end_f) for f in target_frames]) if target_frames else 999999
        nearest_f = min(target_frames, key=lambda f: min(abs(f - start_f), abs(f - end_f))) if target_frames else None

        if hit_rank is not None:
            stage_failure = FailureClass.STRICT_HIT
            forensic = f"Rank {hit_rank} ✅"
        elif target_vid not in selected_video_ids:
            stage_failure = FailureClass.VIDEO_ABSENT
            pool_preview = ", ".join(selected_video_ids[:8])
            forensic = f"Target {target_vid} absent from pool ({len(selected_video_ids)} nominated: [{pool_preview}...])"
        elif len(target_frames) == 0:
            stage_failure = FailureClass.TARGET_VIDEO_NO_EVIDENCE
            forensic = f"Target @Nomination Rank {vid_rank_in_pool}, but 0 target-video candidate frames materialized in evidence"
        elif not in_gt_frames:
            stage_failure = FailureClass.TEMPORAL_MISS
            forensic = f"Target @Nomination Rank {vid_rank_in_pool}, 0/{len(target_frames)} frames in GT. Nearest f={nearest_f} (dist={nearest_dist})"
        else:
            # In-GT frame exists! Check answers produced
            all_answers_on_in_gt = []
            for f_id, ans_list in target_answers:
                if start_f <= f_id <= end_f:
                    all_answers_on_in_gt.extend(ans_list)
            for p in preds:
                if p.get("video_id") == target_vid and start_f <= int(p.get("frame_id", -1)) <= end_f:
                    all_answers_on_in_gt.append(normalize_text(str(p.get("answer", ""))))

            correct_answers = [a for a in all_answers_on_in_gt if a in gt_answers]
            if not correct_answers:
                stage_failure = FailureClass.ANSWER_MISS
                unique_ans = list(set(all_answers_on_in_gt))[:3]
                forensic = f"{len(in_gt_frames)} in-GT frames, but wrong answers: {unique_ans} (GT: {gt_answers})"
            else:
                stage_failure = FailureClass.ALLOCATION_MISS
                forensic = f"Valid tuple existed pre-Top100, but dropped/ranked > 100"

        classification_counts[stage_failure.value] += 1
        forensic_results.append({
            "qid": qid,
            "branch": branch,
            "target": target_vid,
            "gt_interval": f"[{start_f}..{end_f}]",
            "vid_rank": vid_rank_in_pool if vid_rank_in_pool is not None else "ABSENT",
            "candidate_frames": len(target_frames),
            "in_gt_frames": len(in_gt_frames),
            "nearest_frame": f"{nearest_f} (d={nearest_dist})" if nearest_f is not None else "None",
            "stage_failure": stage_failure.value,
            "forensic": forensic,
            "time_s": f"{t_elapsed:.2f}s",
        })

        print(f"[END]   [{idx}/{len(target_qids)}] {qid} -> {stage_failure.value:<26} [{t_elapsed:.2f}s]: {forensic}")

    # 3. Print Forensic Summary Table
    print("\n" + "=" * 125)
    print(f"{'Query ID':<8} | {'Branch':<12} | {'Target':<10} | {'GT Interval':<16} | {'VidRank':<8} | {'CandFs':<8} | {'InGT':<6} | {'Failure Class':<26} | {'Time'}")
    print("-" * 125)
    for r in forensic_results:
        mark = "✅" if r["stage_failure"] == FailureClass.STRICT_HIT.value else "❌"
        print(f"{r['qid']:<8} | {r['branch']:<12} | {r['target']:<10} | {r['gt_interval']:<16} | {str(r['vid_rank']):<8} | {r['candidate_frames']:<8} | {r['in_gt_frames']:<6} | {mark} {r['stage_failure']:<24} | {r['time_s']}")
    print("=" * 125)

    # 4. Strategic Decision Analysis
    print("\n" + "=" * 125)
    print("TARGETED CONVERSION SUMMARY & STRATEGY SELECTION")
    print("=" * 125)
    for fc in FailureClass:
        count = classification_counts[fc.value]
        if count > 0:
            print(f"  {fc.value:<26}: {count} / {len(target_qids)} ({count/len(target_qids)*100:.1f}%)")

    temp_count = classification_counts[FailureClass.TEMPORAL_MISS.value]
    no_ev_count = classification_counts[FailureClass.TARGET_VIDEO_NO_EVIDENCE.value]
    ans_count = classification_counts[FailureClass.ANSWER_MISS.value]
    absent_count = classification_counts[FailureClass.VIDEO_ABSENT.value]

    print("\n--- ACTIONABLE ROADMAP ---")
    if (temp_count + no_ev_count) > ans_count:
        print(f"🎯 DOMINANT BOTTLENECK IS TEMPORAL / EVIDENCE LOCALIZATION ({temp_count + no_ev_count}/{len(target_qids)} queries).")
        print("   -> SPRINT 2A: Prioritize Temporal Localization Rescue (Multi-Anchor + Bounded Window Expansion).")
    elif ans_count > (temp_count + no_ev_count):
        print(f"🎯 DOMINANT BOTTLENECK IS ANSWER / VISUAL REASONING ({ans_count}/{len(target_qids)} queries).")
        print("   -> SPRINT 2A: Prioritize Multi-Crop / Contextual Visual Answer Scorer / OCR Rescue.")
    else:
        print(f"🎯 BALANCED BOTTLENECK: TEMPORAL ({temp_count + no_ev_count}) vs ANSWER ({ans_count}).")
    print("=" * 125)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Targeted Conversion Diagnostic")
    parser.add_argument("--benchmark", type=Path, default=REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "l21_150_diagnostic" / "benchmark.json")
    parser.add_argument("--sidecar", type=Path, default=REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "l21_150_diagnostic" / "qa_dev_translations_en.json")
    parser.add_argument("--manifest-cache", type=Path, default=Path("/kaggle/working/manifest_cache.json"))
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--query-id", type=str, default=None, help="Optional single query_id to run")
    args = parser.parse_args()

    selected = [args.query_id] if args.query_id else None
    run_targeted_diagnostic(
        benchmark_path=args.benchmark,
        dev_en_sidecar_path=args.sidecar,
        manifest_cache_path=args.manifest_cache,
        input_root=args.input_root,
        device=args.device,
        selected_queries=selected,
    )
