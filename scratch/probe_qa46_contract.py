# ==============================================================================================================
# Phase R3-S2A: QA-46 Contract Probe (Positive Control Parity Verification)
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


def run_qa46_probe(
    benchmark_path: Path,
    dev_en_sidecar_path: Path,
    ontology_path: Path,
    manifest_cache_path: Path,
    input_root: Path = Path("/kaggle/input"),
    device: str = "auto",
):
    print("=" * 110)
    print("QA-46 CONTRACT PARITY PROBE (FOUR-LAYER CAUSAL INSPECTION)")
    print("=" * 110)

    benchmark_bytes = benchmark_path.read_bytes()
    benchmark_sha = hashlib.sha256(benchmark_bytes).hexdigest()
    sidecar_bytes = dev_en_sidecar_path.read_bytes()
    sidecar_sha = hashlib.sha256(sidecar_bytes).hexdigest()

    with open(benchmark_path, encoding="utf-8") as f:
        bm_data = json.load(f)

    with open(dev_en_sidecar_path, encoding="utf-8") as f:
        en_sidecar = json.load(f)

    en_map = {e["query_id"]: e.get("question_en", "") for e in en_sidecar.get("entries", [])}
    qa46_query = next(q for q in bm_data["queries"] if q.get("query_id") == "QA-46")

    target_vid = qa46_query.get("video_id")  # L21_V016
    start_f = int(qa46_query["proposed_interval"][0])  # 8190
    end_f = int(qa46_query["proposed_interval"][1])  # 8250
    gt_answers = [normalize_text(a) for a in qa46_query.get("accepted_answers", [])]
    q_vi = qa46_query.get("question_vi", "")
    q_en = en_map.get("QA-46", "")

    # Layer A: Resolved Champion Configuration and R2G1 Feature Flags
    print("\n[LAYER A] RESOLVED CHAMPION CONFIGURATION & R2G1 FEATURE FLAGS")
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

    session_output = Path("/kaggle/working/output/qa46_contract_probe") if Path("/kaggle/working").exists() else REPO_ROOT / "scratch" / "qa46_contract_probe"
    session_output.mkdir(parents=True, exist_ok=True)

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

    print(f"  input_root                         : {input_root}")
    print(f"  qa_video_conditioned_evidence      : {evidence_config.enabled} (cap={evidence_config.selected_video_cap})")
    print(f"  qa_keyframe_evidence_bank          : {evidence_config.preserve_keyframe_evidence} (cap={evidence_config.keyframe_evidence_video_cap}, anchors={evidence_config.keyframe_evidence_anchors_per_video})")
    print(f"  qa_temporal_refinement             : {evidence_config.temporal_refinement_enabled} (seeds_per_video={evidence_config.temporal_seed_anchors_per_video}, total_seed_cap={evidence_config.temporal_refinement_total_seed_cap})")
    print(f"  secondary_temporal_micro_budget    : {evidence_config.secondary_temporal_micro_budget}")
    print(f"  primary_11_12_micro_coverage       : {evidence_config.primary_11_12_micro_coverage}")
    print(f"  tier3_primary_first                : {evidence_config.tier3_primary_first}")
    print(f"  tier3_negative_offset_first        : {evidence_config.tier3_negative_offset_first}")
    print(f"  qa_visual_ontology                 : {visual_config.enabled} (path={visual_config.ontology_path})")
    print(f"  qa_ocr_evidence                    : {ocr_config.enabled} (languages={ocr_config.languages}, budget={ocr_config.evidence_frame_budget})")
    print(f"  qa_object_evidence                 : {object_config.enabled}")

    print("\nBootstrapping runtime...")
    t_boot0 = time.time()
    runtime = OperationalKISRuntime.bootstrap(config)
    print(f"Runtime bootstrap completed in {time.time() - t_boot0:.2f}s.")

    req = QAQueryRequest(
        request_id="probe-QA-46",
        query_id="QA-46",
        event_description=q_vi,
        question=q_vi,
        event_description_en=q_en if q_en else None,
        question_en=None,
        include_vi_variant=False if q_en else True,
        output_top_k=100,
        refine_top_n=3,
    )

    print("\nExecuting QA-46 request...")
    t_q0 = time.time()
    res = runtime.handle_qa_query(req)
    t_elapsed = time.time() - t_q0
    print(f"Execution completed in {t_elapsed:.2f}s.")

    # Layer B: Nomination Pool
    print("\n" + "-" * 110)
    print("[LAYER B] EXACT NOMINATION POOL INSPECTION")
    print("-" * 110)
    diag_file = runtime.output_root / res.get("artifacts", {}).get("qa_evidence_json", "")
    diagnostics = {}
    if diag_file.exists():
        with open(diag_file, encoding="utf-8") as f:
            diagnostics = json.load(f)

    selected_video_ids = diagnostics.get("selected_video_ids", [])
    target_in_pool = target_vid in selected_video_ids
    target_rank_in_pool = selected_video_ids.index(target_vid) + 1 if target_in_pool else None

    print(f"  Target Video                 : {target_vid}")
    print(f"  Target Present in Pool?      : {'YES' if target_in_pool else 'NO'}")
    print(f"  Target Nomination Rank       : {target_rank_in_pool} / {len(selected_video_ids)}")
    print(f"  Full 16 Nominated Video IDs  : {selected_video_ids}")

    # Layer C: qa_evidence.json Records
    print("\n" + "-" * 110)
    print("[LAYER C] qa_evidence.json INTERNAL CANDIDATE REPRESENTATION")
    print("-" * 110)
    print(f"  Artifact Path                : {diag_file}")
    evidence_records = diagnostics.get("evidence", [])
    usable_candidates = diagnostics.get("usable_candidates", [])
    keyframe_records = diagnostics.get("keyframe_evidence_candidates", [])
    generic_records = diagnostics.get("generic_evidence_bank_candidates", [])

    print(f"  Total evidence records       : {len(evidence_records)}")
    print(f"  Total usable candidates      : {len(usable_candidates)}")
    print(f"  Total keyframe records       : {len(keyframe_records)}")
    print(f"  Total generic bank records   : {len(generic_records)}")

    target_ev_records = [r for r in evidence_records if r.get("video_id") == target_vid]
    target_usable = [r for r in usable_candidates if r.get("video_id") == target_vid]
    target_kf = [r for r in keyframe_records if r.get("video_id") == target_vid]
    target_generic = [r for r in generic_records if r.get("video_id") == target_vid]

    print(f"\n  Target ({target_vid}) records across diagnostic fields:")
    print(f"    - In 'evidence'            : {len(target_ev_records)} -> {target_ev_records}")
    print(f"    - In 'usable_candidates'   : {len(target_usable)} -> {target_usable}")
    print(f"    - In 'keyframe_candidates' : {len(target_kf)} -> {target_kf}")
    print(f"    - In 'generic_bank'        : {len(target_generic)} -> {target_generic}")

    # Layer D: Final Top100 Output
    print("\n" + "-" * 110)
    print("[LAYER D] FINAL TOP100 PREDICTIONS INSPECTION")
    print("-" * 110)
    pred_file = runtime.output_root / res.get("artifacts", {}).get("qa_predictions_jsonl", "")
    print(f"  Artifact Path                : {pred_file}")

    preds = res.get("predictions", [])
    print(f"  Total Predictions            : {len(preds)}")

    target_preds = [p for p in preds if p.get("video_id") == target_vid]
    print(f"  Total Rows for {target_vid}  : {len(target_preds)}")

    hit_rank = None
    hit_row = None
    for p in preds:
        p_vid = p.get("video_id")
        p_frame = int(p.get("frame_id", -1))
        p_ans = normalize_text(str(p.get("answer", "")))
        if p_vid == target_vid and start_f <= p_frame <= end_f and p_ans in gt_answers:
            hit_rank = p.get("rank")
            hit_row = p
            break

    print(f"\n  Top Predictions for {target_vid} (first 10 rows):")
    for p in target_preds[:10]:
        in_gt = "IN_GT" if start_f <= int(p.get("frame_id", -1)) <= end_f else "OUT"
        ans_match = "CORRECT" if normalize_text(str(p.get("answer", ""))) in gt_answers else "WRONG"
        print(f"    Rank {p.get('rank'):<3} | f={p.get('frame_id'):<6} ({in_gt}) | ans='{p.get('answer')}' ({ans_match})")

    # Probe Verdict
    print("\n" + "=" * 110)
    print("PROBE VERDICT & SANITY GATE AUDIT")
    print("=" * 110)
    if hit_rank is not None:
        print(f"✅ QA-46 POSITIVE CONTROL: STRICT HIT at Rank {hit_rank}!")
        print(f"   Hit Details: video={target_vid}, frame={hit_row.get('frame_id')} (in GT [{start_f}..{end_f}]), answer='{hit_row.get('answer')}'")
        print("   -> CHAMPION PARITY VERIFIED: SUCCESS")
    else:
        print(f"❌ QA-46 POSITIVE CONTROL: FAILED (No strict hit in Top 100)")
        print(f"   Target Video Rows: {len(target_preds)}, GT Interval: [{start_f}..{end_f}], Accepted Answers: {gt_answers}")
        print("   -> CHAMPION PARITY VERIFIED: FAILED (Config or provider discrepancy detected)")
    print("=" * 110)


if __name__ == "__main__":
    default_input = Path("/kaggle/input/datasets") if Path("/kaggle/input/datasets").exists() else Path("/kaggle/input")
    parser = argparse.ArgumentParser(description="Run QA-46 Contract Probe")
    parser.add_argument("--benchmark", type=Path, default=REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "l21_150_diagnostic" / "benchmark.json")
    parser.add_argument("--sidecar", type=Path, default=REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "l21_150_diagnostic" / "qa_dev_translations_en.json")
    parser.add_argument("--ontology", type=Path, default=REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "l21_150_diagnostic" / "qa_dev_visual_ontology.json")
    parser.add_argument("--manifest-cache", type=Path, default=Path("/kaggle/working/manifest_cache.json"))
    parser.add_argument("--input-root", type=Path, default=default_input)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    run_qa46_probe(
        benchmark_path=args.benchmark,
        dev_en_sidecar_path=args.sidecar,
        ontology_path=args.ontology,
        manifest_cache_path=args.manifest_cache,
        input_root=args.input_root,
        device=args.device,
    )
