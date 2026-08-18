# ==============================================================================================================
# Phase R3-S2A: Triplet Forensic Probe (Strict Runtime-Safe Contract, No Oracle Contamination)
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
    nominate_qa_videos,
)
from system_tai.qa.object_provider import ObjectAnswerProviderConfig
from system_tai.qa.ocr_provider import OCRAnswerProviderConfig
from system_tai.qa.visual_ontology import VisualOntologyConfig
from system_tai.retrieval.multi_query import QueryLanguage, QueryVariant, QueryVariantType
from system_tai.retrieval.query_decomposition import decompose_query


def normalize_text(t: str) -> str:
    if not t:
        return ""
    t = unicodedata.normalize("NFKC", str(t)).casefold()
    return "".join(c for c in t if c.isalnum() or c.isspace()).strip()


def interval_distance(f: int, start: int, end: int) -> int:
    if start <= f <= end:
        return 0
    return start - f if f < start else f - end


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


def run_triplet_probe(
    benchmark_path: Path,
    dev_en_sidecar_path: Path,
    ontology_path: Path,
    manifest_cache_path: Path,
    input_root: Path = Path("/kaggle/input"),
    device: str = "auto",
    enabled_tasks: set[int] | None = None,
):
    active_tasks = enabled_tasks or {1, 2, 3}

    print("=" * 135)
    print(f"ROUND-3 SPRINT 2A: TRIPLET FORENSIC PROBE (ACTIVE TASKS: {sorted(active_tasks)})")
    print("  Task 1: QA-31 Telemetry & Canonical Taxonomy Classification (including ALLOCATION_MISS)")
    print("  Task 2: QA-01 & QA-26 Runtime-Safe Query Decomposition Nomination Novelty Probe (NO ORACLE STRINGS)")
    print("  Task 3: QA-02, QA-23, QA-30 Temporal Provenance & True Interval Dispersion Analysis")
    print("=" * 135)

    with open(benchmark_path, encoding="utf-8") as f:
        bm_data = json.load(f)

    with open(dev_en_sidecar_path, encoding="utf-8") as f:
        en_sidecar = json.load(f)

    en_map = {e["query_id"]: e.get("question_en", "") for e in en_sidecar.get("entries", [])}
    all_qa_queries = {q["query_id"]: q for q in bm_data["queries"] if q.get("task_type") == "qa"}

    session_output = Path("/kaggle/working/output/triplet_probe") if Path("/kaggle/working").exists() else REPO_ROOT / "scratch" / "triplet_probe"
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

    print("\n--- BOOTSTRAPPING RUNTIME ---")
    t0 = time.time()
    runtime = OperationalKISRuntime.bootstrap(config)
    print(f"Runtime bootstrap completed in {time.time() - t0:.2f}s.")

    # =========================================================================
    # TASK 1: QA-31 Telemetry & Canonical Taxonomy Classification
    # =========================================================================
    if 1 in active_tasks:
        print("\n" + "=" * 135)
        print("TASK 1: QA-31 DIRECT EXECUTION & CANONICAL TAXONOMY AUDIT")
        print("=" * 135)
        try:
            q31 = all_qa_queries.get("QA-31")
            if q31:
                target_vid = q31.get("video_id")
                start_f, end_f = int(q31["proposed_interval"][0]), int(q31["proposed_interval"][1])
                gt_answers = [normalize_text(a) for a in q31.get("accepted_answers", [])]
                q_vi = q31.get("question_vi", "")
                q_en = en_map.get("QA-31", "")

                print(f"Executing QA-31 [Target: {target_vid}, GT: [{start_f}..{end_f}], Expected Ans: {gt_answers}]...")
                t_q0 = time.time()
                req31 = QAQueryRequest(
                    request_id="triplet-QA-31",
                    query_id="QA-31",
                    event_description=q_vi,
                    question=q_vi,
                    event_description_en=q_en if q_en else None,
                    question_en=None,
                    include_vi_variant=False if q_en else True,
                    output_top_k=100,
                    refine_top_n=3,
                )
                res31 = runtime.handle_qa_query(req31)
                preds31 = res31.get("predictions", [])
                t_elapsed31 = time.time() - t_q0

                diag_file = runtime.output_root / res31.get("artifacts", {}).get("qa_evidence_json", "")
                diags31 = {}
                if diag_file.exists():
                    with open(diag_file, encoding="utf-8") as f:
                        diags31 = json.load(f)

                selected_videos31 = diags31.get("selected_video_ids", [])
                target_in_pool31 = target_vid in selected_videos31
                rank31 = selected_videos31.index(target_vid) + 1 if target_in_pool31 else None

                scored_ev31 = diags31.get("evidence", [])
                target_ev31 = [r for r in scored_ev31 if r.get("video_id") == target_vid]

                target_ev_records31 = []
                for ev in target_ev31:
                    f_id = ev.get("candidate_frame_id") or ev.get("evidence_frame_id") or ev.get("frame_id")
                    if f_id is not None:
                        ans = normalize_text(str(ev.get("answer", "")))
                        score = float(ev.get("answer_score") or 0.0)
                        target_ev_records31.append({"frame_id": int(f_id), "answer": ans, "score": score})

                target_ev_frames31 = [r["frame_id"] for r in target_ev_records31]
                target_preds31 = [int(p.get("frame_id", -1)) for p in preds31 if p.get("video_id") == target_vid]
                all_frames31 = list(dict.fromkeys(target_ev_frames31 + target_preds31))
                in_gt31 = [f for f in all_frames31 if start_f <= f <= end_f]

                valid_pre_top100_31 = [
                    r for r in target_ev_records31
                    if start_f <= r["frame_id"] <= end_f and r["answer"] in gt_answers
                ]

                hit_rank31 = None
                hit_row31 = None
                for p in preds31:
                    if p.get("video_id") == target_vid and start_f <= int(p.get("frame_id", -1)) <= end_f and normalize_text(str(p.get("answer", ""))) in gt_answers:
                        hit_rank31 = p.get("rank")
                        hit_row31 = p
                        break

                nearest_dist31 = min([interval_distance(f, start_f, end_f) for f in all_frames31]) if all_frames31 else 999999
                nearest_f31 = min(all_frames31, key=lambda f: interval_distance(f, start_f, end_f)) if all_frames31 else None

                if hit_rank31 is not None:
                    failure31 = "STRICT_HIT"
                elif not target_in_pool31:
                    failure31 = "VIDEO_ABSENT"
                elif len(target_ev31) == 0 and len(target_preds31) == 0:
                    failure31 = "TARGET_VIDEO_NO_EVIDENCE"
                elif len(in_gt31) == 0:
                    failure31 = "TEMPORAL_MISS"
                elif len(valid_pre_top100_31) > 0:
                    failure31 = "ALLOCATION_MISS"
                else:
                    failure31 = "ANSWER_MISS"

                print(f"\n[TASK 1 RESULT] QA-31 -> {failure31} in {t_elapsed31:.2f}s:")
                print(f"  VidRank (Nomination)         : {rank31 if rank31 else 'ABSENT'}")
                print(f"  EvRecs (Evidence Records)    : {len(target_ev31)}")
                print(f"  Evidence Frame/Answer Pairs  : {[(r['frame_id'], r['answer']) for r in target_ev_records31]}")
                print(f"  InGT Frames Count            : {len(in_gt31)}")
                print(f"  Nearest Frame to GT          : {nearest_f31} (Distance={nearest_dist31} frames)")
                print(f"  PreTuple Exists in Evidence? : {'YES ✅' if valid_pre_top100_31 else 'NO ❌'}")
                print(f"  FinalHit in Top 100?         : {'Rank ' + str(hit_rank31) + ' ✅' if hit_rank31 else 'NO ❌'}")
                print(f"  Canonical Failure Class      : {failure31}")
        except Exception as exc:
            print(f"⚠️ Task 1 encountered exception: {exc}")
            import traceback
            traceback.print_exc()

    # =========================================================================
    # TASK 2: QA-01 & QA-26 Deterministic Runtime-Safe Query Decomposition Novelty
    # =========================================================================
    if 2 in active_tasks:
        print("\n" + "=" * 135)
        print("TASK 2: QA-01 & QA-26 DETERMINISTIC RUNTIME-SAFE QUERY NOVELTY PROBE")
        print("=" * 135)
        try:
            searcher = runtime.qa_pipeline.video_restricted_searcher
            encoder = runtime.shared_encoder

            target_queries = ["QA-01", "QA-26"]
            for qid in target_queries:
                q = all_qa_queries[qid]
                target_vid = q.get("video_id")
                q_vi = q.get("question_vi", "")
                q_en = en_map.get(qid, "")

                print(f"\n" + "-" * 110)
                print(f"PROBING NOMINATION NOVELTY FOR {qid} [Target: {target_vid}]")
                print(f"  Question VI: '{q_vi}'")
                print(f"  Question EN: '{q_en}'")
                print("-" * 110)

                # 1. Champion actual baseline nomination pool
                req_champ = QAQueryRequest(
                    request_id=f"probe-champ-{qid}",
                    query_id=qid,
                    event_description=q_vi,
                    question=q_vi,
                    event_description_en=q_en if q_en else None,
                    question_en=None,
                    include_vi_variant=False if q_en else True,
                    output_top_k=100,
                )
                res_champ = runtime.handle_qa_query(req_champ)
                diag_file = runtime.output_root / res_champ.get("artifacts", {}).get("qa_evidence_json", "")
                champ_pool = []
                if diag_file.exists():
                    with open(diag_file, encoding="utf-8") as f:
                        champ_pool = json.load(f).get("selected_video_ids", [])
                champ_rank = champ_pool.index(target_vid) + 1 if target_vid in champ_pool else "ABSENT"

                print(f"\n1. Champion Baseline Top 16 Pool (all 16 IDs):")
                print(f"   {champ_pool}")
                print(f"   -> Target {target_vid} Rank in Champion Baseline: {champ_rank}")

                # 2. Derive deterministic runtime-safe query decomposition (NO ORACLE / NO GT STRINGS)
                variants_obj = decompose_query(query_text_vi=q_vi, query_text_en=q_en)
                decomp_list = variants_obj.as_list()

                print(f"\n2. Derived Runtime-Safe Query Variants (from query text only):")
                for v_name, v_text in decomp_list:
                    print(f"   - {v_name:<18}: '{v_text}'")

                # 3. Search Individual Variants using production search_video_maxima
                print(f"\n3. Individual Variant Top 16 Nomination Rankings:")
                variant_objects: list[QueryVariant] = []
                variant_vectors = []
                for v_name, v_text in decomp_list:
                    v_lang = QueryLanguage.VIETNAMESE if v_name == "literal" and not q_en else QueryLanguage.ENGLISH
                    v_type = QueryVariantType.VIETNAMESE_DIRECT if v_lang == QueryLanguage.VIETNAMESE else QueryVariantType.ENGLISH_TRANSLATION
                    v_id = f"{qid}::{v_name}"
                    v_obj = QueryVariant(variant_id=v_id, text=v_text, language=v_lang, variant_type=v_type, weight=1.0)
                    v_vec = encoder.encode(v_text)
                    variant_objects.append(v_obj)
                    variant_vectors.append(v_vec)

                    single_maxima = searcher.search_video_maxima(
                        query_ids=[v_id],
                        query_vectors=[v_vec],
                    )
                    single_noms = nominate_qa_videos(
                        variants=[v_obj],
                        maxima=single_maxima,
                        config=evidence_config,
                    )
                    v_pool = [n.video_id for n in single_noms]
                    v_rank = v_pool.index(target_vid) + 1 if target_vid in v_pool else "ABSENT"
                    print(f"   - {v_name:<18} -> Target Rank: {str(v_rank):<7} | Top 16: {v_pool}")

                # 4. Fused Multi-Variant Top 16 Nomination
                multi_qids = [v.variant_id for v in variant_objects]
                multi_maxima = searcher.search_video_maxima(
                    query_ids=multi_qids,
                    query_vectors=variant_vectors,
                )
                nominations = nominate_qa_videos(
                    variants=variant_objects,
                    maxima=multi_maxima,
                    config=evidence_config,
                )
                fused_pool = [n.video_id for n in nominations]
                fused_rank = fused_pool.index(target_vid) + 1 if target_vid in fused_pool else "ABSENT"

                print(f"\n4. Fused Runtime-Safe Multi-Variant Top 16 Pool (all 16 IDs):")
                print(f"   {fused_pool}")
                print(f"   -> Target {target_vid} Rank in Fused Multi-Variant: {fused_rank}")

                novel_recovery = "YES (Target Recovered into Pool) ✅" if fused_rank != "ABSENT" and champ_rank == "ABSENT" else ("TARGET ALREADY PRESENT" if champ_rank != "ABSENT" else "NO (Target Still Absent) ❌")
                print(f"   👉 NOVEL RECOVERY STATUS FOR {qid}: {novel_recovery}")
        except Exception as exc:
            print(f"⚠️ Task 2 encountered exception: {exc}")
            import traceback
            traceback.print_exc()

    # =========================================================================
    # TASK 3: QA-02, QA-23, QA-30 Temporal Provenance & True Dispersion Analysis
    # =========================================================================
    if 3 in active_tasks:
        print("\n" + "=" * 135)
        print("TASK 3: QA-02, QA-23, QA-30 TEMPORAL PROVENANCE & DISPERSION AUDIT")
        print("=" * 135)
        try:
            temporal_qids = ["QA-23", "QA-30", "QA-02"]
            for qid in temporal_qids:
                q = all_qa_queries[qid]
                target_vid = q.get("video_id")
                start_f, end_f = int(q["proposed_interval"][0]), int(q["proposed_interval"][1])
                gt_answers = [normalize_text(a) for a in q.get("accepted_answers", [])]
                q_vi = q.get("question_vi", "")
                q_en = en_map.get(qid, "")

                print(f"\n" + "-" * 110)
                print(f"TEMPORAL PROVENANCE & CLUSTERING FOR {qid} [Target: {target_vid}, GT: [{start_f}..{end_f}]]")
                print("-" * 110)

                req = QAQueryRequest(
                    request_id=f"triplet-{qid}",
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

                diag_file = runtime.output_root / res.get("artifacts", {}).get("qa_evidence_json", "")
                diags = {}
                if diag_file.exists():
                    with open(diag_file, encoding="utf-8") as f:
                        diags = json.load(f)

                scored_ev = diags.get("evidence", [])
                target_ev = [r for r in scored_ev if r.get("video_id") == target_vid]

                print(f"1. Target Canonical Evidence Records ({len(target_ev)} records):")
                frame_list = []
                for i, ev in enumerate(target_ev, 1):
                    cand_f = ev.get("candidate_frame_id")
                    ev_f = ev.get("evidence_frame_id")
                    source = ev.get("evidence_source", "UNKNOWN")
                    local_rank = ev.get("local_anchor_rank", 1)
                    loc_score = ev.get("localization_score", 0.0)
                    var_ids = ev.get("source_localization_variant_ids", [])
                    ref_status = ev.get("refinement_status", "NOT_REFINED")
                    ans = ev.get("answer", "")
                    raw_ans_score = ev.get("answer_score")
                    ans_score = float(raw_ans_score) if raw_ans_score is not None else 0.0

                    frame_val = int(cand_f or ev_f or 0)
                    if frame_val:
                        frame_list.append(frame_val)

                    in_gt_mark = "IN_GT ✅" if start_f <= frame_val <= end_f else "OUT ❌"
                    dist_to_gt = interval_distance(frame_val, start_f, end_f)
                    print(f"   [{i}] f={frame_val:<6} ({in_gt_mark}, dist={dist_to_gt:<5}) | src={source:<16} | rank={local_rank} | loc_score={loc_score:.4f} | ref={ref_status:<12} | ans='{ans}' (score={ans_score:.3f}) | vars={var_ids}")

                if frame_list:
                    min_f = min(frame_list)
                    max_f = max(frame_list)
                    spread = max_f - min_f
                    nearest_dist = min([interval_distance(f, start_f, end_f) for f in frame_list])
                    nearest_f = min(frame_list, key=lambda f: interval_distance(f, start_f, end_f))

                    # Identify clusters (cluster distance threshold = 500 frames)
                    sorted_frames = sorted(set(frame_list))
                    clusters = [[sorted_frames[0]]]
                    for f in sorted_frames[1:]:
                        if f - clusters[-1][-1] <= 500:
                            clusters[-1].append(f)
                        else:
                            clusters.append([f])

                    # Correct Diagnosis Ordering (Near Miss checked first!)
                    if nearest_dist <= 250:
                        diagnosis = f"BOUNDED NEAR MISS (Distance {nearest_dist} <= 250 frames -> Ideal candidate for Local Window Expansion)"
                    elif len(clusters) == 1 and spread < 500:
                        diagnosis = f"WRONG SEGMENT MODE COLLAPSE (Distance {nearest_dist} frames, all candidates clustered at ~{min_f}..{max_f} -> Needs Diverse Temporal Anchors)"
                    else:
                        diagnosis = f"CATASTROPHIC WRONG SEGMENT / MULTI-CLUSTER (Distance {nearest_dist} frames, spread {spread} frames -> Semantic Localization Bias)"

                    print(f"\n2. Cluster & Dispersion Metrics for {qid}:")
                    print(f"   - Unique Temporal Clusters : {len(clusters)} -> {[f'{min(c)}..{max(c)} (n={len(c)})' for c in clusters]}")
                    print(f"   - Min / Max Frame          : {min_f} .. {max_f}")
                    print(f"   - Temporal Spread          : {spread} frames")
                    print(f"   - Nearest Frame to GT      : Frame {nearest_f} (Distance = {nearest_dist} frames)")
                    print(f"   👉 DIAGNOSIS               : {diagnosis}")
        except Exception as exc:
            print(f"⚠️ Task 3 encountered exception: {exc}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 135)
    print("TRIPLET FORENSIC PROBE COMPLETED")
    print("=" * 135)


if __name__ == "__main__":
    default_input = Path("/kaggle/input/datasets") if Path("/kaggle/input/datasets").exists() else Path("/kaggle/input")
    parser = argparse.ArgumentParser(description="Run Triplet Forensic Probe")
    parser.add_argument("--benchmark", type=Path, default=REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "l21_150_diagnostic" / "benchmark.json")
    parser.add_argument("--sidecar", type=Path, default=REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "l21_150_diagnostic" / "qa_dev_translations_en.json")
    parser.add_argument("--ontology", type=Path, default=REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "l21_150_diagnostic" / "qa_dev_visual_ontology.json")
    parser.add_argument("--manifest-cache", type=Path, default=Path("/kaggle/working/manifest_cache.json"))
    parser.add_argument("--input-root", type=Path, default=default_input)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--task", type=str, default="all", help="Tasks to run: 1, 2, 3, 2,3, or all")
    parser.add_argument("--skip-task1", action="store_true", help="Skip Task 1")
    args = parser.parse_args()

    if args.skip_task1:
        task_set = {2, 3}
    elif args.task == "all":
        task_set = {1, 2, 3}
    else:
        task_set = {int(t.strip()) for t in args.task.split(",") if t.strip().isdigit()}

    run_triplet_probe(
        benchmark_path=args.benchmark,
        dev_en_sidecar_path=args.sidecar,
        ontology_path=args.ontology,
        manifest_cache_path=args.manifest_cache,
        input_root=args.input_root,
        device=args.device,
        enabled_tasks=task_set,
    )
