# ==============================================================================================================
# QA-23 Champion Evidence Diagnostic Probe
# ==============================================================================================================

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

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


def run_qa23_probe():
    benchmark_path = REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "l21_150_diagnostic" / "benchmark.json"
    sidecar_path = REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "l21_150_diagnostic" / "qa_dev_translations_en.json"
    ontology_path = REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "l21_150_diagnostic" / "qa_dev_visual_ontology.json"

    with open(benchmark_path, encoding="utf-8") as f:
        bm_data = json.load(f)
    with open(sidecar_path, encoding="utf-8") as f:
        en_sidecar = json.load(f)

    en_map = {e["query_id"]: e.get("question_en", "") for e in en_sidecar.get("entries", [])}
    all_qa_queries = {q["query_id"]: q for q in bm_data["queries"] if q.get("task_type") == "qa"}

    q = all_qa_queries["QA-23"]
    q_vi = q.get("question_vi", "")
    q_en = en_map.get("QA-23", "")

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
        consensus_novel_rescue_enabled=False,
        bounded_negative_temporal_rescue_enabled=False,
    )

    visual_config = VisualOntologyConfig(
        enabled=ontology_path.exists(),
        ontology_path=ontology_path if ontology_path.exists() else None,
    )
    ocr_config = resolve_ocr_config()
    object_config = ObjectAnswerProviderConfig(enabled=False)

    session_output = Path("/kaggle/working/output/probe_qa23") if Path("/kaggle/working").exists() else REPO_ROOT / "scratch" / "probe_qa23"
    session_output.mkdir(parents=True, exist_ok=True)

    config = SessionConfig(
        input_root=Path("/kaggle/input/datasets") if Path("/kaggle/input/datasets").exists() else Path("/kaggle/input"),
        manifest_cache=Path("/kaggle/working/manifest_cache.json"),
        output_root=session_output,
        device="auto",
        allow_model_download=True,
        default_output_top_k=100,
        default_refine_top_n=3,
        qa_video_conditioned_evidence_config=evidence_config,
        qa_visual_ontology_config=visual_config,
        qa_ocr_answer_provider_config=ocr_config,
        qa_object_answer_provider_config=object_config,
    )

    print("--- BOOTSTRAPPING RUNTIME ---")
    t0 = time.time()
    runtime = OperationalKISRuntime.bootstrap(config)
    print(f"Bootstrapped in {time.time() - t0:.2f}s")

    req = QAQueryRequest(
        request_id="probe-qa23-champion",
        query_id="QA-23",
        event_description=q_vi,
        question=q_vi,
        event_description_en=q_en if q_en else None,
        question_en=None,
        include_vi_variant=False if q_en else True,
        output_top_k=100,
        refine_top_n=3,
    )

    print("\n--- RUNNING CANONICAL CHAMPION FOR QA-23 ---")
    res = runtime.handle_qa_query(req)

    diag_file = runtime.output_root / res.get("artifacts", {}).get("qa_evidence_json", "")
    with open(diag_file, encoding="utf-8") as f:
        diags = json.load(f)

    print("\n" + "=" * 100)
    print("QA-23 CANONICAL CHAMPION AUDIT TELEMETRY")
    print("=" * 100)

    sel_vids = diags.get("selected_video_ids", [])
    print(f"\n1. Selected Videos (Count: {len(sel_vids)}):")
    for rank, vid in enumerate(sel_vids, 1):
        print(f"   Rank {rank:2d}: {vid}")

    top1_vid = sel_vids[0] if sel_vids else None
    print(f"\n2. Top-1 Nominated Video: {top1_vid}")

    # Inspect temporal seed candidates
    temporal_seeds = diags.get("temporal_seed_candidates", [])
    print(f"\n3. Temporal Seed Candidates (Count: {len(temporal_seeds)}):")
    for s in temporal_seeds:
        if s.get("video_id") == top1_vid:
            print(f"   • Top-1 Seed: video={s.get('video_id')}, frame={s.get('frame_id')}, nomination_rank={s.get('video_nomination_rank')}, local_anchor_rank={s.get('local_anchor_rank')}")

    # Inspect refined candidates
    refined_cands = diags.get("refined_candidates", [])
    print(f"\n4. Refined Candidates for Top-1 ({top1_vid}) (Count: {len(refined_cands)}):")
    for c in refined_cands:
        if c.get("video_id") == top1_vid:
            print(f"   • Candidate: orig_rank={c.get('original_rank')}, cand_frame={c.get('candidate_frame_id')}, refined_frame={c.get('refined_frame_id')}, status={c.get('status')}")

    # Inspect evidence records
    ev_records = diags.get("evidence", [])
    print(f"\n5. Evidence Records for Top-1 ({top1_vid}) (Count: {len(ev_records)}):")
    for e in ev_records:
        if e.get("video_id") == top1_vid:
            print(f"   • Evidence Record: rank={e.get('rank')}, cand_frame={e.get('candidate_frame_id')}, output_frame={e.get('output_frame_id')}, status={e.get('refinement_status')}, answer={e.get('answer')}, skip={e.get('skip_reason')}")

    print("\n" + "=" * 100)


if __name__ == "__main__":
    run_qa23_probe()
