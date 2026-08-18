# ==============================================================================================================
# Phase R3-S2A: Triplet Forensic Probe (QA-31 Telemetry + QA01/26 Novelty + QA02/23/30 Provenance)
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
)
from system_tai.qa.object_provider import ObjectAnswerProviderConfig
from system_tai.qa.ocr_provider import OCRAnswerProviderConfig
from system_tai.qa.visual_ontology import VisualOntologyConfig
from system_tai.retrieval.multi_query import QueryLanguage, QueryVariant, QueryVariantType


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


def run_triplet_probe(
    benchmark_path: Path,
    dev_en_sidecar_path: Path,
    ontology_path: Path,
    manifest_cache_path: Path,
    input_root: Path = Path("/kaggle/input"),
    device: str = "auto",
):
    print("=" * 125)
    print("ROUND-3 SPRINT 2A: TRIPLET FORENSIC PROBE")
    print("  Task 1: QA-31 Telemetry & Classification Repair")
    print("  Task 2: QA-01 & QA-26 Actual-Runtime Nomination Novelty Probe")
    print("  Task 3: QA-02, QA-23, QA-30 Temporal Provenance & Cluster Dispersion Analysis")
    print("=" * 125)

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
    # TASK 1: QA-31 Telemetry & Classification
    # =========================================================================
    print("\n" + "=" * 125)
    print("TASK 1: QA-31 DIRECT EXECUTION & TELEMETRY")
    print("=" * 125)
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
        target_frames31 = []
        for ev in target_ev31:
            f_id = ev.get("candidate_frame_id") or ev.get("evidence_frame_id") or ev.get("frame_id")
            if f_id is not None:
                target_frames31.append(int(f_id))

        target_preds31 = [int(p.get("frame_id", -1)) for p in preds31 if p.get("video_id") == target_vid]
        all_frames31 = list(dict.fromkeys(target_frames31 + target_preds31))
        in_gt31 = [f for f in all_frames31 if start_f <= f <= end_f]

        hit_rank31 = None
        for p in preds31:
            if p.get("video_id") == target_vid and start_f <= int(p.get("frame_id", -1)) <= end_f and normalize_text(str(p.get("answer", ""))) in gt_answers:
                hit_rank31 = p.get("rank")
                break

        nearest_dist31 = min([abs(f - start_f) for f in all_frames31] + [abs(f - end_f) for f in all_frames31]) if all_frames31 else 999999
        nearest_f31 = min(all_frames31, key=lambda f: min(abs(f - start_f), abs(f - end_f))) if all_frames31 else None

        if hit_rank31 is not None:
            failure31 = "STRICT_HIT"
        elif not target_in_pool31:
            failure31 = "VIDEO_ABSENT"
        elif len(target_ev31) == 0 and len(target_preds31) == 0:
            failure31 = "TARGET_VIDEO_NO_EVIDENCE"
        elif len(in_gt31) == 0:
            failure31 = "TEMPORAL_MISS"
        else:
            failure31 = "ANSWER_MISS"

        print(f"\n[TASK 1 RESULT] QA-31 -> {failure31} in {t_elapsed31:.2f}s:")
        print(f"  Target Video Present?        : {'YES (Rank ' + str(rank31) + ')' if target_in_pool31 else 'NO (ABSENT)'}")
        print(f"  Target Evidence Records      : {len(target_ev31)}")
        print(f"  Target Frames Evaluated      : {all_frames31}")
        print(f"  In-GT Frames                 : {len(in_gt31)}")
        print(f"  Nearest Frame                : {nearest_f31} (dist={nearest_dist31})")
        print(f"  Final Top100 Hit             : {hit_rank31 if hit_rank31 else 'NO'}")

    # =========================================================================
    # TASK 2: QA-01 & QA-26 Actual-Runtime Nomination Novelty Probe
    # =========================================================================
    print("\n" + "=" * 125)
    print("TASK 2: QA-01 & QA-26 ACTUAL-RUNTIME NOMINATION NOVELTY PROBE")
    print("=" * 125)

    expansion_probes = {
        "QA-01": {
            "target": "L21_V001",
            "q_vi": all_qa_queries["QA-01"]["question_vi"],
            "q_en": en_map.get("QA-01", ""),
            "variants": [
                ("v_literal_vi", all_qa_queries["QA-01"]["question_vi"], QueryLanguage.VIETNAMESE, QueryVariantType.VIETNAMESE_DIRECT),
                ("v_literal_en", en_map.get("QA-01", ""), QueryLanguage.ENGLISH, QueryVariantType.ENGLISH_TRANSLATION),
                ("v_compact", "warning sign yellow red danger road hazard", QueryLanguage.ENGLISH, QueryVariantType.ENGLISH_TRANSLATION),
                ("v_entity", "yellow red triangle hazard warning plate sign", QueryLanguage.ENGLISH, QueryVariantType.ENGLISH_TRANSLATION),
                ("v_action", "traffic road sign landslide warning cliff mountain", QueryLanguage.ENGLISH, QueryVariantType.ENGLISH_TRANSLATION),
            ],
        },
        "QA-26": {
            "target": "L21_V009",
            "q_vi": all_qa_queries["QA-26"]["question_vi"],
            "q_en": en_map.get("QA-26", ""),
            "variants": [
                ("v_literal_vi", all_qa_queries["QA-26"]["question_vi"], QueryLanguage.VIETNAMESE, QueryVariantType.VIETNAMESE_DIRECT),
                ("v_literal_en", en_map.get("QA-26", ""), QueryLanguage.ENGLISH, QueryVariantType.ENGLISH_TRANSLATION),
                ("v_compact", "taxi car vehicle street road driving", QueryLanguage.ENGLISH, QueryVariantType.ENGLISH_TRANSLATION),
                ("v_entity", "green white taxi cab driving street intersection", QueryLanguage.ENGLISH, QueryVariantType.ENGLISH_TRANSLATION),
                ("v_action", "passenger car vehicle street road driving daytime", QueryLanguage.ENGLISH, QueryVariantType.ENGLISH_TRANSLATION),
            ],
        },
    }

    for qid, pdata in expansion_probes.items():
        target_vid = pdata["target"]
        print(f"\n--- Probing Nomination for {qid} (Target: {target_vid}) ---")

        # 1. Champion actual nomination pool
        req_champ = QAQueryRequest(
            request_id=f"probe-champ-{qid}",
            query_id=qid,
            event_description=pdata["q_vi"],
            question=pdata["q_vi"],
            event_description_en=pdata["q_en"] if pdata["q_en"] else None,
            question_en=None,
            include_vi_variant=False if pdata["q_en"] else True,
            output_top_k=100,
        )
        res_champ = runtime.handle_qa_query(req_champ)
        diag_file = runtime.output_root / res_champ.get("artifacts", {}).get("qa_evidence_json", "")
        champ_pool = []
        if diag_file.exists():
            with open(diag_file, encoding="utf-8") as f:
                champ_pool = json.load(f).get("selected_video_ids", [])
        champ_rank = champ_pool.index(target_vid) + 1 if target_vid in champ_pool else "ABSENT"

        print(f"  Champion Baseline Top 16 Pool: {champ_pool[:8]}...")
        print(f"  Target {target_vid} in Champion Pool: {champ_rank}")

        # 2. Individual Variant Nominations
        searcher = runtime.qa_pipeline.video_restricted_searcher
        encoder = runtime.shared_encoder

        print(f"\n  Individual Expansion Variant Rankings for {target_vid}:")
        variant_rankings = {}
        for var_name, var_text, var_lang, var_type in pdata["variants"]:
            v_obj = QueryVariant(variant_id=f"{qid}::{var_name}", text=var_text, language=var_lang, variant_type=var_type, weight=1.0)
            v_vec = encoder.encode(var_text)
            maxima = searcher.search_maxima(
                query_ids=[v_obj.variant_id],
                query_vectors=[v_vec],
                per_query_cap=16,
            )
            v_pool = [item.video_id for item in maxima.rankings.get(v_obj.variant_id, [])]
            v_rank = v_pool.index(target_vid) + 1 if target_vid in v_pool else "ABSENT"
            variant_rankings[var_name] = (v_rank, v_pool)
            print(f"    - {var_name:<14} -> Target Rank: {str(v_rank):<7} | Top 4: {v_pool[:4]}")

        # 3. Fused Multi-Variant Top 16 Nomination
        multi_qids = [f"{qid}::{v[0]}" for v in pdata["variants"]]
        multi_vecs = [encoder.encode(v[1]) for v in pdata["variants"]]
        multi_maxima = searcher.search_maxima(
            query_ids=multi_qids,
            query_vectors=multi_vecs,
            per_query_cap=16,
        )
        nominations = runtime.qa_pipeline.video_conditioned_evidence_fuser.nominate_videos(
            multi_maxima,
            config=evidence_config,
        )
        fused_pool = [n.video_id for n in nominations]
        fused_rank = fused_pool.index(target_vid) + 1 if target_vid in fused_pool else "ABSENT"

        print(f"\n  Fused Multi-Variant Top 16 Pool: {fused_pool[:8]}...")
        print(f"  Target {target_vid} in Fused Multi-Variant Pool: {fused_rank}")
        novel_recovery = "YES ✅" if fused_rank != "ABSENT" and champ_rank == "ABSENT" else "NO ❌"
        print(f"  👉 NOVEL RECOVERY STATUS: {novel_recovery}")

    # =========================================================================
    # TASK 3: QA-02, QA-23, QA-30 Temporal Provenance & Cluster Analysis
    # =========================================================================
    print("\n" + "=" * 125)
    print("TASK 3: QA-02, QA-23, QA-30 TEMPORAL PROVENANCE & CLUSTER DISPERSION ANALYSIS")
    print("=" * 125)

    temporal_qids = ["QA-02", "QA-23", "QA-30"]
    for qid in temporal_qids:
        q = all_qa_queries[qid]
        target_vid = q.get("video_id")
        start_f, end_f = int(q["proposed_interval"][0]), int(q["proposed_interval"][1])
        gt_answers = [normalize_text(a) for a in q.get("accepted_answers", [])]
        q_vi = q.get("question_vi", "")
        q_en = en_map.get(qid, "")

        print(f"\n--- Temporal Provenance for {qid} [Target: {target_vid}, GT: [{start_f}..{end_f}]] ---")
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

        print(f"  Target Canonical Evidence Records ({len(target_ev)} records):")
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
            print(f"    [{i}] f={frame_val:<6} ({in_gt_mark}) | src={source:<16} | rank={local_rank} | loc_score={loc_score:.4f} | ref={ref_status:<12} | ans='{ans}' (score={ans_score:.3f}) | vars={var_ids}")

        if frame_list:
            min_f = min(frame_list)
            max_f = max(frame_list)
            spread = max_f - min_f
            nearest_dist = min([abs(f - start_f) for f in frame_list] + [abs(f - end_f) for f in frame_list])
            nearest_f = min(frame_list, key=lambda f: min(abs(f - start_f), abs(f - end_f)))

            # Identify clusters (distance threshold = 500 frames)
            sorted_frames = sorted(set(frame_list))
            clusters = [[sorted_frames[0]]]
            for f in sorted_frames[1:]:
                if f - clusters[-1][-1] <= 500:
                    clusters[-1].append(f)
                else:
                    clusters.append([f])

            print(f"\n  Cluster & Dispersion Metrics for {qid}:")
            print(f"    - Unique Temporal Clusters : {len(clusters)} -> {[f'{min(c)}..{max(c)} (n={len(c)})' for c in clusters]}")
            print(f"    - Min / Max Frame          : {min_f} .. {max_f}")
            print(f"    - Temporal Spread          : {spread} frames")
            print(f"    - Nearest to GT Interval   : Frame {nearest_f} (Distance = {nearest_dist} frames)")
            if len(clusters) == 1 and spread < 500:
                print(f"    - Diagnosis                : SEVERE TEMPORAL MODE COLLAPSE (All candidates tightly clustered at ~{min_f})")
            elif nearest_dist <= 250:
                print(f"    - Diagnosis                : BOUNDED NEAR MISS (Distance {nearest_dist} <= 250 frames -> Ideal candidate for Local Negative Expansion)")
            else:
                print(f"    - Diagnosis                : CATASTROPHIC WRONG SEGMENT (Distance {nearest_dist} frames -> Needs Diverse Anchors across video)")

    print("\n" + "=" * 125)
    print("TRIPLET FORENSIC PROBE COMPLETED")
    print("=" * 125)


if __name__ == "__main__":
    default_input = Path("/kaggle/input/datasets") if Path("/kaggle/input/datasets").exists() else Path("/kaggle/input")
    parser = argparse.ArgumentParser(description="Run Triplet Forensic Probe")
    parser.add_argument("--benchmark", type=Path, default=REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "l21_150_diagnostic" / "benchmark.json")
    parser.add_argument("--sidecar", type=Path, default=REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "l21_150_diagnostic" / "qa_dev_translations_en.json")
    parser.add_argument("--ontology", type=Path, default=REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "l21_150_diagnostic" / "qa_dev_visual_ontology.json")
    parser.add_argument("--manifest-cache", type=Path, default=Path("/kaggle/working/manifest_cache.json"))
    parser.add_argument("--input-root", type=Path, default=default_input)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    run_triplet_probe(
        benchmark_path=args.benchmark,
        dev_en_sidecar_path=args.sidecar,
        ontology_path=args.ontology,
        manifest_cache_path=args.manifest_cache,
        input_root=args.input_root,
        device=args.device,
    )
