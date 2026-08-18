# ==============================================================================================================
# Phase R3-S2A: Targeted Conversion Diagnostic (Canonical Scored Evidence Contract)
# ==============================================================================================================

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import unicodedata
from collections import Counter
from enum import StrEnum
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

    # Ensure tesseract language packages are installed
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

DEFAULT_TARGETED_QUERIES = [
    "QA-46",  # Positive control / Sanity gate (Action branch, thủ công thổ cẩm) - RUN FIRST
    "QA-01",  # S1A / OCR branch (biển cảnh báo)
    "QA-02",  # Object branch
    "QA-23",  # Color branch
    "QA-26",  # S1A / Visual branch
    "QA-30",  # Action branch
    "QA-31",  # Visual branch
]


class FailureClass(StrEnum):
    STRICT_HIT = "STRICT_HIT"
    VIDEO_ABSENT = "VIDEO_ABSENT"
    TARGET_VIDEO_NO_EVIDENCE = "TARGET_VIDEO_NO_EVIDENCE"
    TEMPORAL_MISS = "TEMPORAL_MISS"
    ANSWER_MISS = "ANSWER_MISS"
    ALLOCATION_MISS = "ALLOCATION_MISS"
    OTHER_UNKNOWN = "OTHER_UNKNOWN"


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


def parse_benchmark_gt_interval(q: dict[str, Any]) -> tuple[int, int]:
    if "proposed_interval" in q and isinstance(q["proposed_interval"], (list, tuple)) and len(q["proposed_interval"]) == 2:
        return int(q["proposed_interval"][0]), int(q["proposed_interval"][1])
    if "start_frame_id" in q and "end_frame_id" in q:
        return int(q["start_frame_id"]), int(q["end_frame_id"])
    return -1, -1


def run_targeted_diagnostic(
    benchmark_path: Path,
    dev_en_sidecar_path: Path,
    ontology_path: Path,
    manifest_cache_path: Path,
    input_root: Path = Path("/kaggle/input"),
    device: str = "auto",
    selected_queries: list[str] | None = None,
):
    target_qids = selected_queries or DEFAULT_TARGETED_QUERIES

    print("=" * 135)
    print("ROUND-3 SPRINT 2A: TARGETED 7-QUERY CONVERSION FORENSIC (f39f63c CONTROL-COMPATIBLE RUNTIME)")
    print("=" * 135)

    benchmark_bytes = benchmark_path.read_bytes()
    benchmark_sha = hashlib.sha256(benchmark_bytes).hexdigest()
    sidecar_bytes = dev_en_sidecar_path.read_bytes()
    sidecar_sha = hashlib.sha256(sidecar_bytes).hexdigest()

    print(f"Benchmark File  : {benchmark_path.name} (SHA256: {benchmark_sha[:16]}...)")
    print(f"Sidecar File    : {dev_en_sidecar_path.name} (SHA256: {sidecar_sha[:16]}...)")
    print(f"Ontology File   : {ontology_path.name} (exists={ontology_path.exists()})")
    print(f"Targeted Queries: {target_qids}")

    with open(benchmark_path, encoding="utf-8") as f:
        bm_data = json.load(f)

    with open(dev_en_sidecar_path, encoding="utf-8") as f:
        en_sidecar = json.load(f)

    en_map = {e["query_id"]: e.get("question_en", "") for e in en_sidecar.get("entries", [])}
    all_qa_queries = {q["query_id"]: q for q in bm_data["queries"] if q.get("task_type") == "qa"}

    # 1. Bootstrap Runtime with Full Exact Champion Configuration
    print("\n--- BOOTSTRAPPING CHAMPION RUNTIME (SINGLE INSTANCE) ---")
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

    visual_config = VisualOntologyConfig(
        enabled=ontology_path.exists(),
        ontology_path=ontology_path if ontology_path.exists() else None,
    )
    ocr_config = resolve_ocr_config()
    object_config = ObjectAnswerProviderConfig(enabled=False)

    config = SessionConfig(
        input_root=input_root,
        manifest_cache=manifest_cache_path,
        output_root=session_output,
        device=device,
        allow_model_download=True,
        default_output_top_k=100,
        default_refine_top_n=3,
        qa_video_conditioned_evidence_config=evidence_config,
        qa_visual_ontology_config=visual_config,
        qa_ocr_answer_provider_config=ocr_config,
        qa_object_answer_provider_config=object_config,
    )

    t_boot0 = time.time()
    runtime = OperationalKISRuntime.bootstrap(config)
    print(f"Runtime bootstrap completed in {time.time() - t_boot0:.2f}s.")

    # 2. Execute Targeted Queries Sequentially in Same Process
    print("\n" + "=" * 135)
    print("EXECUTING TARGETED DIAGNOSTIC QUERIES")
    print("=" * 135)

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

        print(f"\n[START] [{idx}/{len(target_qids)}] {qid} ({branch:<12}) [Target: {target_vid}, GT: [{start_f}..{end_f}], Expected Ans: {gt_answers}]...")
        t_q0 = time.time()
        req = QAQueryRequest(
            request_id=f"targeted-{qid}",
            query_id=qid,
            event_description=q_vi,
            question=q_vi,
            event_description_en=q_en if q_en else None,
            question_en=None,
            include_vi_variant=False if q_en else True,
            output_top_k=100,
            refine_top_n=3,
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

        # Stage A: Nomination Pool
        selected_video_ids = diagnostics.get("selected_video_ids", [])
        if not selected_video_ids and preds:
            selected_video_ids = list(dict.fromkeys([p.get("video_id") for p in preds if p.get("video_id")]))
        target_in_pool = target_vid in selected_video_ids
        vid_rank_in_pool = selected_video_ids.index(target_vid) + 1 if target_in_pool else None

        # Stage B: Canonical Scored Pre-Top100 Evidence Records
        scored_evidence = diagnostics.get("evidence", [])
        target_evidence = [r for r in scored_evidence if r.get("video_id") == target_vid]

        target_ev_records = []
        for ev in target_evidence:
            # Use candidate_frame_id or evidence_frame_id for frame matching (do NOT use output_frame_id)
            f_id = ev.get("candidate_frame_id") or ev.get("evidence_frame_id") or ev.get("frame_id")
            if f_id is not None:
                ans = normalize_text(str(ev.get("answer", "")))
                score = float(ev.get("answer_score", 0.0))
                target_ev_records.append({"frame_id": int(f_id), "answer": ans, "score": score})

        target_ev_frames = [r["frame_id"] for r in target_ev_records]

        # Target frames in final predictions
        target_preds = [p for p in preds if p.get("video_id") == target_vid]
        target_pred_frames = [int(p.get("frame_id", -1)) for p in target_preds if p.get("frame_id") is not None]

        all_target_frames = list(dict.fromkeys(target_ev_frames + target_pred_frames))

        # Check Final Top 100 Strict Hit
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

        # Classify Causal Bottleneck with Strict Ground-Truth Taxonomy
        in_gt_ev_frames = [f for f in target_ev_frames if start_f <= f <= end_f]
        in_gt_all_frames = [f for f in all_target_frames if start_f <= f <= end_f]
        nearest_dist = min([abs(f - start_f) for f in all_target_frames] + [abs(f - end_f) for f in all_target_frames]) if all_target_frames else 999999
        nearest_f = min(all_target_frames, key=lambda f: min(abs(f - start_f), abs(f - end_f))) if all_target_frames else None

        valid_pre_top100 = [
            r for r in target_ev_records
            if start_f <= r["frame_id"] <= end_f and r["answer"] in gt_answers
        ]

        if hit_rank is not None:
            stage_failure = FailureClass.STRICT_HIT
            forensic = f"STRICT HIT at Rank {hit_rank} (f={hit_frame}, ans='{hit_ans}') ✅"
        elif not target_in_pool:
            stage_failure = FailureClass.VIDEO_ABSENT
            pool_str = ", ".join(selected_video_ids[:16])
            forensic = f"Target {target_vid} ABSENT from pool (Nominated 16: [{pool_str}])"
        elif len(target_evidence) == 0 and len(target_preds) == 0:
            stage_failure = FailureClass.TARGET_VIDEO_NO_EVIDENCE
            forensic = f"Target @Nomination Rank {vid_rank_in_pool}, but 0 target-video candidate frames in evidence/predictions"
        elif len(in_gt_all_frames) == 0:
            stage_failure = FailureClass.TEMPORAL_MISS
            forensic = f"Target @Nomination Rank {vid_rank_in_pool}, 0/{len(all_target_frames)} frames in GT. Frames: {all_target_frames[:6]}, Nearest f={nearest_f} (dist={nearest_dist})"
        else:
            # In-GT frame exists! Check if valid tuple existed pre-Top100
            if len(valid_pre_top100) > 0:
                stage_failure = FailureClass.ALLOCATION_MISS
                top_valid = valid_pre_top100[0]
                forensic = f"Valid tuple existed pre-Top100 (f={top_valid['frame_id']}, ans='{top_valid['answer']}'), but dropped/ranked > 100"
            else:
                stage_failure = FailureClass.ANSWER_MISS
                # Collect answers on in-GT frames
                ans_on_in_gt = [r["answer"] for r in target_ev_records if start_f <= r["frame_id"] <= end_f]
                ans_on_in_gt += [normalize_text(str(p.get("answer", ""))) for p in target_preds if start_f <= int(p.get("frame_id", -1)) <= end_f]
                unique_ans = list(set(ans_on_in_gt))[:3]
                forensic = f"{len(in_gt_all_frames)} in-GT frames evaluated, but wrong answers: {unique_ans} (GT: {gt_answers})"

        classification_counts[stage_failure.value] += 1
        has_pre_tuple = "YES" if len(valid_pre_top100) > 0 else "NO"
        has_final_hit = f"Rank {hit_rank}" if hit_rank is not None else "NO"

        forensic_results.append({
            "qid": qid,
            "branch": branch,
            "target": target_vid,
            "gt_interval": f"[{start_f}..{end_f}]",
            "vid_rank": vid_rank_in_pool if vid_rank_in_pool is not None else "ABSENT",
            "ev_records": len(target_evidence),
            "cand_frames": str(all_target_frames[:5]),
            "in_gt_frames": len(in_gt_all_frames),
            "pre_tuple": has_pre_tuple,
            "final_hit": has_final_hit,
            "stage_failure": stage_failure.value,
            "forensic": forensic,
            "time_s": f"{t_elapsed:.2f}s",
        })

        print(f"[END]   [{idx}/{len(target_qids)}] {qid} -> {stage_failure.value:<26} [{t_elapsed:.2f}s]: {forensic}")

    # 3. Print Forensic Summary Table
    print("\n" + "=" * 135)
    print(f"{'Query ID':<8} | {'Branch':<12} | {'Target':<10} | {'GT Interval':<16} | {'VidRank':<8} | {'EvRecs':<6} | {'InGT':<5} | {'PreTuple':<8} | {'FinalHit':<10} | {'Failure Class':<24} | {'Time'}")
    print("-" * 135)
    for r in forensic_results:
        mark = "✅" if r["stage_failure"] == FailureClass.STRICT_HIT.value else "❌"
        print(f"{r['qid']:<8} | {r['branch']:<12} | {r['target']:<10} | {r['gt_interval']:<16} | {str(r['vid_rank']):<8} | {r['ev_records']:<6} | {r['in_gt_frames']:<5} | {r['pre_tuple']:<8} | {r['final_hit']:<10} | {mark} {r['stage_failure']:<22} | {r['time_s']}")
    print("=" * 135)

    # 4. Strategic Decision Analysis (Guarded by Positive Control QA-46)
    print("\n" + "=" * 135)
    print("TARGETED 7-QUERY SAMPLE CONVERSION SUMMARY (f39f63c CONTROL-COMPATIBLE RUNTIME)")
    print("=" * 135)
    for fc in FailureClass:
        count = classification_counts[fc.value]
        if count > 0:
            print(f"  {fc.value:<26}: {count} / {len(target_qids)} ({count/len(target_qids)*100:.1f}%)")

    qa46_res = next((r for r in forensic_results if r["qid"] == "QA-46"), None)
    qa46_passed = qa46_res is not None and qa46_res["stage_failure"] == FailureClass.STRICT_HIT.value

    if not qa46_passed and "QA-46" in target_qids:
        print("\n⚠️ SANITY GATE FAILED: Positive control QA-46 did not strict hit. Strategic recommendation suppressed.")
        print("=" * 135)
        return

    temp_count = classification_counts[FailureClass.TEMPORAL_MISS.value]
    no_ev_count = classification_counts[FailureClass.TARGET_VIDEO_NO_EVIDENCE.value]
    ans_count = classification_counts[FailureClass.ANSWER_MISS.value]
    alloc_count = classification_counts[FailureClass.ALLOCATION_MISS.value]
    absent_count = classification_counts[FailureClass.VIDEO_ABSENT.value]

    print(f"\n--- ACTIONABLE SAMPLE ROADMAP (Positive Control QA-46: {'PASS' if qa46_passed else 'N/A'}) ---")
    if (temp_count + no_ev_count) > (ans_count + alloc_count):
        print(f"🎯 IN THIS TARGETED SAMPLE: TEMPORAL LOCALIZATION is the leading failure mode ({temp_count + no_ev_count}/{len(target_qids)} queries).")
        print("   -> SPRINT 2A CANDIDATE: Bounded Temporal Localization Rescue (Multi-Anchor + Local Window Expansion).")
    elif (ans_count + alloc_count) > (temp_count + no_ev_count):
        print(f"🎯 IN THIS TARGETED SAMPLE: ANSWER REASONING / ALLOCATION is the leading failure mode ({ans_count + alloc_count}/{len(target_qids)} queries).")
        print("   -> SPRINT 2A CANDIDATE: Multi-Crop / Contextual Visual Answer Scorer / OCR Subset Rescue.")
    else:
        print(f"🎯 BALANCED SAMPLE DISTRIBUTION: TEMPORAL ({temp_count + no_ev_count}) vs ANSWER/ALLOCATION ({ans_count + alloc_count}).")
    print("=" * 135)


if __name__ == "__main__":
    default_input = Path("/kaggle/input/datasets") if Path("/kaggle/input/datasets").exists() else Path("/kaggle/input")
    parser = argparse.ArgumentParser(description="Run Targeted Conversion Diagnostic")
    parser.add_argument("--benchmark", type=Path, default=REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "l21_150_diagnostic" / "benchmark.json")
    parser.add_argument("--sidecar", type=Path, default=REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "l21_150_diagnostic" / "qa_dev_translations_en.json")
    parser.add_argument("--ontology", type=Path, default=REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "l21_150_diagnostic" / "qa_dev_visual_ontology.json")
    parser.add_argument("--manifest-cache", type=Path, default=Path("/kaggle/working/manifest_cache.json"))
    parser.add_argument("--input-root", type=Path, default=default_input)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--query-id", type=str, default=None, help="Optional single query_id to run")
    args = parser.parse_args()

    selected = [args.query_id] if args.query_id else None
    run_targeted_diagnostic(
        benchmark_path=args.benchmark,
        dev_en_sidecar_path=args.sidecar,
        ontology_path=args.ontology,
        manifest_cache_path=args.manifest_cache,
        input_root=args.input_root,
        device=args.device,
        selected_queries=selected,
    )
