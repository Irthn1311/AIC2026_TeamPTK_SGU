# ==============================================================================================================
# Phase R3-S2A: Artifact-First First-Failure Conversion Audit (Zero-Decode, Pure Artifact Post-Hoc)
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

REPO_ROOT = Path(__file__).resolve().parents[1]
SYSTEM_TAI_SRC = REPO_ROOT / "systems" / "system_tai" / "src"
if str(SYSTEM_TAI_SRC) not in sys.path:
    sys.path.insert(0, str(SYSTEM_TAI_SRC))


class FailureClass(StrEnum):
    STRICT_HIT = "STRICT_HIT"
    VIDEO_ABSENT = "VIDEO_ABSENT"
    TEMPORAL_MISS = "TEMPORAL_MISS"
    ANSWER_MISS = "ANSWER_MISS"
    ALLOCATION_MISS = "ALLOCATION_MISS"
    NO_PREDICTIONS = "NO_PREDICTIONS"


def normalize_text(t: str) -> str:
    if not t:
        return ""
    t = unicodedata.normalize("NFKC", str(t)).casefold()
    return "".join(c for c in t if c.isalnum() or c.isspace()).strip()


def find_champion_artifact_roots() -> list[Path]:
    """Find all potential artifact roots containing qa_predictions.jsonl on Kaggle or locally."""
    search_dirs = [
        Path("/kaggle/working/output/qa_r2h1_control_cap16"),
        Path("/kaggle/working/output/qa_r2g1_control"),
        Path("/kaggle/working/output"),
        Path("/kaggle/working"),
        REPO_ROOT / "output",
        REPO_ROOT / "scratch",
    ]
    found = []
    for d in search_dirs:
        if d.exists():
            # Check if there are any qa_predictions.jsonl files
            files = list(d.rglob("qa_predictions.jsonl"))
            if files:
                found.append(d)
    return found


def run_artifact_first_audit(
    benchmark_path: Path,
    dev_en_sidecar_path: Path,
    artifact_dir: Path | None = None,
):
    print("=" * 125)
    print("ROUND-3 SPRINT 2A: ARTIFACT-FIRST FIRST-FAILURE CONVERSION AUDIT (POST-HOC AUDIT)")
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

    # 1. Resolve Champion Artifact Directory
    search_roots = [artifact_dir] if artifact_dir else find_champion_artifact_roots()
    print(f"\n--- ARTIFACT SOURCE RESOLUTION ---")
    print(f"  Searching artifact roots: {[str(r) for r in search_roots]}")

    predictions_by_query: dict[str, list[dict]] = {}
    diagnostics_by_query: dict[str, dict] = {}
    evidence_by_query: dict[str, dict] = {}

    all_pred_files = []
    for r in search_roots:
        if r and r.exists():
            all_pred_files.extend(list(r.rglob("qa_predictions.jsonl")))
            # Also check top100.jsonl or predictions.jsonl
            all_pred_files.extend(list(r.rglob("predictions.jsonl")))

    # Deduplicate files
    unique_pred_files = list(dict.fromkeys(all_pred_files))
    print(f"  Found {len(unique_pred_files)} prediction files across search roots.")

    for pf in unique_pred_files:
        try:
            with open(pf, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        p = json.loads(line)
                        qid = p.get("query_id")
                        if qid:
                            # Keep only one clean set per query (prioritizing control arm)
                            if qid not in predictions_by_query or "control" in str(pf):
                                if qid not in predictions_by_query:
                                    predictions_by_query[qid] = []
                                predictions_by_query[qid].append(p)

            # Check sibling qa_evidence.json or qa_request_manifest.json
            ev_file = pf.parent / "qa_evidence.json"
            if ev_file.exists():
                try:
                    with open(ev_file, encoding="utf-8") as ef:
                        ev_data = json.load(ef)
                        eqid = ev_data.get("query_id")
                        if eqid:
                            evidence_by_query[eqid] = ev_data
                except Exception:
                    pass

            diag_file = pf.parent / "qa_timings.json"
            if diag_file.exists():
                try:
                    with open(diag_file, encoding="utf-8") as df:
                        df_data = json.load(df)
                        dqid = df_data.get("query_id")
                        if dqid:
                            diagnostics_by_query[dqid] = df_data
                except Exception:
                    pass

        except Exception as e:
            print(f"  Warning reading {pf}: {e}")

    # Also search for any requests/ directory
    for r in search_roots:
        if r and r.exists():
            for req_file in r.rglob("requests/**/*.json"):
                if req_file.name == "qa_evidence.json":
                    try:
                        with open(req_file, encoding="utf-8") as ef:
                            ev_data = json.load(ef)
                            eqid = ev_data.get("query_id")
                            if eqid:
                                evidence_by_query[eqid] = ev_data
                    except Exception:
                        pass

    print(f"  Loaded predictions for {len(predictions_by_query)} queries.")
    print(f"  Loaded pre-Top100 evidence for {len(evidence_by_query)} queries.")

    # 2. Classify Each Query
    failure_records = []
    class_counts: Counter[str] = Counter()
    class_queries: dict[str, list[str]] = {fc.value: [] for fc in FailureClass}

    for idx, q in enumerate(qa_dev_queries, start=1):
        qid = q["query_id"]
        target_vid = q.get("video_id")
        start_f = int(q.get("start_frame_id", -1))
        end_f = int(q.get("end_frame_id", -1))
        gt_answers = [normalize_text(a) for a in q.get("accepted_answers", [])]

        preds = predictions_by_query.get(qid, [])
        ev_data = evidence_by_query.get(qid, {})

        # Stage A: Champion Nomination Pool
        selected_video_ids = ev_data.get("selected_video_ids", [])
        # If evidence file not serialized, extract all unique video_ids from final predictions
        if not selected_video_ids and preds:
            selected_video_ids = list(dict.fromkeys([p.get("video_id") for p in preds if p.get("video_id")]))

        # Stage B: Internal candidates pre-Top100
        evidence_records = ev_data.get("evidence", [])
        usable_candidates = ev_data.get("usable_candidates", [])

        target_internal_frames: list[int] = []
        target_internal_tuples: list[tuple[int, list[str]]] = []

        for ev in evidence_records:
            if ev.get("video_id") == target_vid:
                f_id = ev.get("output_frame_id") or ev.get("refined_frame_id") or ev.get("candidate_frame_id")
                if f_id is not None:
                    target_internal_frames.append(int(f_id))
                    ans_list = [normalize_text(str(ev.get("answer")))] if ev.get("answer") else []
                    target_internal_tuples.append((int(f_id), ans_list))

        if not target_internal_frames:
            for uc in usable_candidates:
                if uc.get("video_id") == target_vid and uc.get("frame_id") is not None:
                    f_id = int(uc.get("frame_id"))
                    target_internal_frames.append(f_id)
                    ans_list = [normalize_text(str(a)) for a in uc.get("answers", []) if a]
                    target_internal_tuples.append((f_id, ans_list))

        # Check 1: STRICT_HIT
        hit_rank = None
        hit_frame = None
        hit_ans = None
        for p in preds:
            p_vid = p.get("video_id")
            p_frame = int(p.get("frame_id", -1))
            p_ans = normalize_text(str(p.get("answer", "")))
            if p_vid == target_vid and start_f <= p_frame <= end_f and p_ans in gt_answers:
                hit_rank = p.get("rank")
                hit_frame = p_frame
                hit_ans = p_ans
                break

        if hit_rank is not None:
            failure_class = FailureClass.STRICT_HIT
            detail = f"Hit at Rank {hit_rank} (f={hit_frame}, ans='{hit_ans}')"
        elif not preds:
            failure_class = FailureClass.NO_PREDICTIONS
            detail = "Zero predictions in artifact"
        elif target_vid not in selected_video_ids:
            failure_class = FailureClass.VIDEO_ABSENT
            detail = f"Target {target_vid} not in nomination pool ({len(selected_video_ids)} videos)"
        else:
            # Video is in pool. Check frames
            all_target_frames = list(dict.fromkeys(target_internal_frames + [int(p.get("frame_id", -1)) for p in preds if p.get("video_id") == target_vid]))
            in_gt_frames = [f for f in all_target_frames if start_f <= f <= end_f]

            if not in_gt_frames:
                failure_class = FailureClass.TEMPORAL_MISS
                nearest_dist = min([abs(f - start_f) for f in all_target_frames] + [abs(f - end_f) for f in all_target_frames]) if all_target_frames else 999999
                nearest_f = min(all_target_frames, key=lambda f: min(abs(f - start_f), abs(f - end_f))) if all_target_frames else None
                detail = f"Video present, but 0/{len(all_target_frames)} frames in GT [{start_f}..{end_f}]. Nearest f={nearest_f} (dist={nearest_dist})"
            else:
                # In-GT frame exists. Check answers
                answers_on_in_gt = []
                for p in preds:
                    if p.get("video_id") == target_vid and start_f <= int(p.get("frame_id", -1)) <= end_f:
                        answers_on_in_gt.append(normalize_text(str(p.get("answer", ""))))

                for f_id, ans_list in target_internal_tuples:
                    if start_f <= f_id <= end_f:
                        answers_on_in_gt.extend(ans_list)

                correct_answers_seen = [a for a in answers_on_in_gt if a in gt_answers]

                if not correct_answers_seen:
                    failure_class = FailureClass.ANSWER_MISS
                    unique_seen = list(set(answers_on_in_gt))[:3]
                    detail = f"In-GT frame exists ({len(in_gt_frames)} frames), but wrong answers: {unique_seen} (GT: {gt_answers})"
                else:
                    failure_class = FailureClass.ALLOCATION_MISS
                    detail = f"Valid full tuple existed internally (f in GT & ans='{correct_answers_seen[0]}'), but was dropped from Top 100"

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

    # 3. Print Detailed Table
    print("\n" + "=" * 125)
    print(f"{'Query ID':<8} | {'Target':<10} | {'GT Interval':<16} | {'First Failure Class':<20} | {'Forensic Detail'}")
    print("-" * 125)
    for r in failure_records:
        mark = "✅" if r["failure_class"] == "STRICT_HIT" else "❌"
        print(f"{r['qid']:<8} | {r['target_vid']:<10} | {r['gt_interval']:<16} | {mark} {r['failure_class']:<18} | {r['detail']}")
    print("=" * 125)

    # 4. Print Aggregate Breakdown
    print("\n" + "=" * 125)
    print("AGGREGATE FIRST-FAILURE BREAKDOWN (N = 38 DEV QUERIES)")
    print("=" * 125)
    print(f"{'Failure Class':<22} | {'Count':<10} | {'Percentage':<12} | {'Query IDs'}")
    print("-" * 125)
    for fc in FailureClass:
        count = class_counts[fc.value]
        pct = f"{count / 38 * 100:.1f}%"
        q_list = ", ".join(class_queries[fc.value]) if class_queries[fc.value] else "None"
        print(f"{fc.value:<22} | {count:<10} | {pct:<12} | {q_list}")
    print("=" * 125)

    # 5. Strategic Decision Analysis
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
    parser = argparse.ArgumentParser(description="Run Artifact-First First-Failure Conversion Audit")
    parser.add_argument("--benchmark", type=Path, default=REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "l21_150_diagnostic" / "benchmark.json")
    parser.add_argument("--sidecar", type=Path, default=REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "l21_150_diagnostic" / "qa_dev_translations_en.json")
    parser.add_argument("--artifact-dir", type=Path, default=None)
    args = parser.parse_args()

    run_artifact_first_audit(
        benchmark_path=args.benchmark,
        dev_en_sidecar_path=args.sidecar,
        artifact_dir=args.artifact_dir,
    )
