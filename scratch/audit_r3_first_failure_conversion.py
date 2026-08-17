# ==============================================================================================================
# Phase R3-S2A: First-Failure Conversion Audit across all 38 DEV queries
# ==============================================================================================================

import argparse
import hashlib
import json
import math
import sys
import time
from collections import Counter
from enum import StrEnum
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parents[1]
SYSTEM_TAI_SRC = REPO_ROOT / "systems" / "system_tai" / "src"
if str(SYSTEM_TAI_SRC) not in sys.path:
    sys.path.insert(0, str(SYSTEM_TAI_SRC))

from system_tai.common.schemas import CandidateFrame
from system_tai.kis.session_engine import OperationalKISRuntime
from system_tai.kis.session_schema import QAQueryRequest, SessionConfig
from system_tai.qa.grounding import (
    QA_CANDIDATE_ORDER_ROUND_ROBIN,
    QAVideoConditionedEvidenceConfig,
)


class FailureClass(StrEnum):
    STRICT_HIT = "STRICT_HIT"
    VIDEO_ABSENT = "VIDEO_ABSENT"
    TEMPORAL_MISS = "TEMPORAL_MISS"
    ANSWER_MISS = "ANSWER_MISS"
    ALLOCATION_MISS = "ALLOCATION_MISS"
    OTHER_UNKNOWN = "OTHER_UNKNOWN"


def normalize_ans(a: str) -> str:
    return " ".join(a.strip().lower().split())


def run_first_failure_audit(
    benchmark_path: Path,
    dev_en_sidecar_path: Path,
    manifest_cache_path: Path,
    input_root: Path = Path("/kaggle/input"),
    device: str = "auto",
):
    print("=" * 125)
    print("ROUND-3 SPRINT 2A: FIRST-FAILURE CONVERSION AUDIT (STRICT THREE-STAGE PROVENANCE)")
    print("=" * 125)

    benchmark_bytes = benchmark_path.read_bytes()
    benchmark_sha = hashlib.sha256(benchmark_bytes).hexdigest()
    sidecar_bytes = dev_en_sidecar_path.read_bytes()
    sidecar_sha = hashlib.sha256(sidecar_bytes).hexdigest()

    print(f"Benchmark File: {benchmark_path.name} (SHA256: {benchmark_sha[:16]}...)")
    print(f"Sidecar File  : {dev_en_sidecar_path.name} (SHA256: {sidecar_sha[:16]}...)")

    with open(benchmark_path, encoding="utf-8") as f:
        bm_data = json.load(f)

    with open(dev_en_sidecar_path, encoding="utf-8") as f:
        en_sidecar = json.load(f)

    en_map = {e["query_id"]: e.get("question_en", "") for e in en_sidecar.get("entries", [])}
    qa_dev_queries = [q for q in bm_data["queries"] if q.get("task_type") == "qa" and q.get("split") == "DEV"]
    print(f"Loaded {len(qa_dev_queries)} QA DEV queries from benchmark.")

    # 1. Bootstrap Runtime with Champion R2G1 Configuration
    print("\n--- CHAMPION REFERENCE CONTRACT ---")
    print("  Configuration : Champion R2G1 (qa_video_conditioned_evidence=ON, selected_video_cap=16, temporal_refine=ON)")
    print("  Language      : EN_ONLY via frozen QA-D0 sidecar")
    print("  Evidence Bank : preserve_keyframe_evidence=ON, keyframe_anchors=1, temporal_seeds=3, total_seed_cap=48")
    print("  Micro budgets : secondary_temporal=ON, primary_11_12=ON, tier3_primary_first=ON, tier3_negative_offset=ON")

    session_output = Path("/kaggle/working/output/r3_failure_audit_runtime") if Path("/kaggle/working").exists() else REPO_ROOT / "scratch" / "r3_failure_audit_runtime"
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
        temporal_seed_anchors_per_video=3,
        temporal_refinement_video_cap=16,
        temporal_refinement_total_seed_cap=48,
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

    failure_records = []
    class_counts: Counter[str] = Counter()
    class_queries: dict[str, list[str]] = {fc.value: [] for fc in FailureClass}

    t0 = time.time()
    print("\nExecuting Champion inference and auditing failure stages across all 38 queries...")

    for idx, q in enumerate(qa_dev_queries, start=1):
        qid = q["query_id"]
        target_vid = q.get("video_id")
        start_f = int(q.get("start_frame_id", -1))
        end_f = int(q.get("end_frame_id", -1))
        gt_answers = [normalize_ans(a) for a in q.get("accepted_answers", [])]
        q_vi = q.get("question_vi", "")
        q_en = en_map.get(qid, "")

        req = QAQueryRequest(
            request_id=f"audit-{qid}",
            query_id=qid,
            question=q_vi,
            question_en=q_en,
            top_k=100,
        )

        res = runtime.handle_qa_query(req)
        preds = res.get("predictions", [])
        diagnostics = res.get("diagnostics", {})

        # Stage A: Champion Nomination Pool
        selected_video_ids = diagnostics.get("selected_video_ids", [])

        # Stage B: Internal Localized & Scored Candidates Pre-Top100
        # Check both evidence bank records and usable candidates
        evidence_records = diagnostics.get("evidence", [])
        usable_candidates = diagnostics.get("usable_candidates", [])

        # Extract all internal candidate frames produced for target video
        target_internal_frames: list[int] = []
        target_internal_tuples: list[tuple[int, list[str]]] = []  # (frame_id, [answers])

        for ev in evidence_records:
            if ev.get("video_id") == target_vid:
                f_id = ev.get("output_frame_id") or ev.get("refined_frame_id") or ev.get("candidate_frame_id")
                if f_id is not None:
                    target_internal_frames.append(int(f_id))
                    ans_list = []
                    if ev.get("answer"):
                        ans_list.append(normalize_ans(str(ev.get("answer"))))
                    target_internal_tuples.append((int(f_id), ans_list))

        if not target_internal_frames:
            for uc in usable_candidates:
                if uc.get("video_id") == target_vid and uc.get("frame_id") is not None:
                    f_id = int(uc.get("frame_id"))
                    target_internal_frames.append(f_id)
                    ans_list = [normalize_ans(str(a)) for a in uc.get("answers", []) if a]
                    target_internal_tuples.append((f_id, ans_list))

        # Check 1: STRICT_HIT in final predictions (Ranks 1..100)
        hit_rank = None
        hit_frame = None
        hit_ans = None
        for p in preds:
            p_vid = p.get("video_id")
            p_frame = int(p.get("frame_id", -1))
            p_ans = normalize_ans(str(p.get("answer", "")))
            if p_vid == target_vid and start_f <= p_frame <= end_f and p_ans in gt_answers:
                hit_rank = p.get("rank")
                hit_frame = p_frame
                hit_ans = p_ans
                break

        if hit_rank is not None:
            failure_class = FailureClass.STRICT_HIT
            detail = f"Hit at Rank {hit_rank} (f={hit_frame}, ans='{hit_ans}')"
        elif not preds and not selected_video_ids:
            failure_class = FailureClass.OTHER_UNKNOWN
            detail = "Zero predictions & missing provenance (runtime error)"
        elif target_vid not in selected_video_ids:
            # Check 2: VIDEO_ABSENT
            failure_class = FailureClass.VIDEO_ABSENT
            detail = f"Target video absent from champion nomination pool ({len(selected_video_ids)} videos in pool)"
        else:
            # Target video is in pool. Check internal frames before Top100
            in_gt_internal_frames = [f for f in target_internal_frames if start_f <= f <= end_f]

            if not in_gt_internal_frames:
                # Check 3: TEMPORAL_MISS
                failure_class = FailureClass.TEMPORAL_MISS
                nearest_dist = min([abs(f - start_f) for f in target_internal_frames] + [abs(f - end_f) for f in target_internal_frames]) if target_internal_frames else 999999
                nearest_f = min(target_internal_frames, key=lambda f: min(abs(f - start_f), abs(f - end_f))) if target_internal_frames else None
                detail = f"Video present, but 0/{len(target_internal_frames)} internal frames in GT [{start_f}..{end_f}]. Nearest f={nearest_f} (dist={nearest_dist})"
            else:
                # Target video + In-GT internal frame exists!
                # Check if an accepted answer was generated on in-GT candidates
                in_gt_answers_pre_top100: list[str] = []
                for f_id, ans_list in target_internal_tuples:
                    if start_f <= f_id <= end_f:
                        in_gt_answers_pre_top100.extend(ans_list)

                # Also check final predictions for in-GT answers
                for p in preds:
                    if p.get("video_id") == target_vid and start_f <= int(p.get("frame_id", -1)) <= end_f:
                        in_gt_answers_pre_top100.append(normalize_ans(str(p.get("answer", ""))))

                correct_answers_seen = [a for a in in_gt_answers_pre_top100 if a in gt_answers]

                if not correct_answers_seen:
                    # Check 4: ANSWER_MISS
                    failure_class = FailureClass.ANSWER_MISS
                    unique_seen_ans = list(set(in_gt_answers_pre_top100))[:3]
                    detail = f"In-GT internal frame exists ({len(in_gt_internal_frames)} frames), but wrong answers: {unique_seen_ans} (GT: {gt_answers})"
                else:
                    # Check 5: ALLOCATION_MISS
                    # Valid tuple (target, in_GT_frame, correct_answer) existed internally, but was dropped from Top 100!
                    failure_class = FailureClass.ALLOCATION_MISS
                    detail = f"Valid full tuple existed pre-Top100 (f in GT & ans='{correct_answers_seen[0]}'), but was dropped or ranked > 100"

        class_counts[failure_class.value] += 1
        class_queries[failure_class.value].append(qid)

        failure_records.append({
            "qid": qid,
            "target_vid": target_vid,
            "gt_interval": f"[{start_f}..{end_f}]",
            "gt_answers": gt_answers,
            "failure_class": failure_class.value,
            "detail": detail,
        })

    elapsed = time.time() - t0
    print(f"Audit completed in {elapsed:.2f}s.")

    # 2. Print Detailed Summary Table
    print("\n" + "=" * 125)
    print(f"{'Query ID':<8} | {'Target':<10} | {'GT Interval':<16} | {'First Failure Class':<18} | {'Forensic Detail'}")
    print("-" * 125)
    for r in failure_records:
        mark = "✅" if r["failure_class"] == "STRICT_HIT" else "❌"
        print(f"{r['qid']:<8} | {r['target_vid']:<10} | {r['gt_interval']:<16} | {mark} {r['failure_class']:<16} | {r['detail']}")
    print("=" * 125)

    # 3. Print Aggregate Failure Breakdown
    print("\n" + "=" * 125)
    print("AGGREGATE FIRST-FAILURE BREAKDOWN (N = 38 DEV QUERIES)")
    print("=" * 125)
    print(f"{'Failure Class':<20} | {'Count':<10} | {'Percentage':<12} | {'Query IDs'}")
    print("-" * 125)
    for fc in FailureClass:
        count = class_counts[fc.value]
        pct = f"{count / 38 * 100:.1f}%"
        q_list = ", ".join(class_queries[fc.value]) if class_queries[fc.value] else "None"
        print(f"{fc.value:<20} | {count:<10} | {pct:<12} | {q_list}")
    print("=" * 125)

    # 4. Strategic Decision Analysis
    print("\n--- STRATEGIC ACTION DECISION BASED ON DOMINANT BOTTLENECK ---")
    miss_counts = {
        FailureClass.VIDEO_ABSENT.value: class_counts[FailureClass.VIDEO_ABSENT.value],
        FailureClass.TEMPORAL_MISS.value: class_counts[FailureClass.TEMPORAL_MISS.value],
        FailureClass.ANSWER_MISS.value: class_counts[FailureClass.ANSWER_MISS.value],
        FailureClass.ALLOCATION_MISS.value: class_counts[FailureClass.ALLOCATION_MISS.value],
    }
    dominant = max(miss_counts, key=miss_counts.get)
    print(f"Dominant Miss Category: {dominant} ({miss_counts[dominant]} queries)")
    if dominant == FailureClass.TEMPORAL_MISS.value:
        print("🎯 STRATEGY: Prioritize Sprint 2A on Temporal Localization Rescue (Multi-Anchor + Bounded Interval Expansion).")
    elif dominant == FailureClass.ANSWER_MISS.value:
        print("🎯 STRATEGY: Prioritize Sprint 2A on Answer Scoring / Multi-Crop Visual Reasoning / OCR.")
    elif dominant == FailureClass.VIDEO_ABSENT.value:
        print("🎯 STRATEGY: Video retrieval pool expansion required for the specific absent queries.")
    elif dominant == FailureClass.ALLOCATION_MISS.value:
        print("🎯 STRATEGY: Prioritize Tail Allocation tuning.")
    print("=" * 125)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run First-Failure Conversion Audit")
    parser.add_argument("--benchmark", type=Path, default=REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "l21_150_diagnostic" / "benchmark.json")
    parser.add_argument("--sidecar", type=Path, default=REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "l21_150_diagnostic" / "qa_dev_translations_en.json")
    parser.add_argument("--manifest-cache", type=Path, default=Path("/kaggle/working/manifest_cache.json"))
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    run_first_failure_audit(
        benchmark_path=args.benchmark,
        dev_en_sidecar_path=args.sidecar,
        manifest_cache_path=args.manifest_cache,
        input_root=args.input_root,
        device=args.device,
    )
