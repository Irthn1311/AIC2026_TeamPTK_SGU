# ==============================================================================================================
# Visualizer for QA Query: Decoded Frame Image & Video Clip Snippet (Interactive Kaggle Demo)
# ==============================================================================================================

import argparse
import base64
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from PIL import Image as PILImage

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
    available_langs = []
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
    return OCRAnswerProviderConfig(enabled=bool(available_langs), languages=supported, evidence_frame_budget=8)


def extract_video_clip(video_path: Path, output_clip_path: Path, center_seconds: float, duration_seconds: float = 6.0):
    start_sec = max(0.0, center_seconds - duration_seconds / 2.0)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss", str(start_sec),
        "-i", str(video_path),
        "-t", str(duration_seconds),
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "23",
        "-c:a", "aac",
        str(output_clip_path),
    ]
    subprocess.run(cmd, capture_output=True, check=False)


def run_visualization(query_id: str = "QA-23"):
    benchmark_path = REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "l21_150_diagnostic" / "benchmark.json"
    sidecar_path = REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "l21_150_diagnostic" / "qa_dev_translations_en.json"
    ontology_path = REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "l21_150_diagnostic" / "qa_dev_visual_ontology.json"

    with open(benchmark_path, encoding="utf-8") as f:
        bm_data = json.load(f)
    with open(sidecar_path, encoding="utf-8") as f:
        en_sidecar = json.load(f)

    en_map = {e["query_id"]: e.get("question_en", "") for e in en_sidecar.get("entries", [])}
    all_qa_queries = {q["query_id"]: q for q in bm_data["queries"] if q.get("task_type") == "qa"}

    if query_id not in all_qa_queries:
        raise ValueError(f"Query {query_id} not found in benchmark! Available: {list(all_qa_queries.keys())}")

    q = all_qa_queries[query_id]
    q_vi = q.get("question_vi", "")
    q_en = en_map.get(query_id, "")
    target_vid = q.get("video_id")
    start_f, end_f = int(q["proposed_interval"][0]), int(q["proposed_interval"][1])
    accepted_answers = q.get("accepted_answers", [])

    print("=" * 100)
    print(f"VISUALIZING QA QUERY: {query_id}")
    print(f"  • Question (VI)     : '{q_vi}'")
    print(f"  • Question (EN)     : '{q_en}'")
    print(f"  • Target Video      : {target_vid}")
    print(f"  • Ground Truth Range: Frames [{start_f}..{end_f}]")
    print(f"  • Accepted Answers  : {accepted_answers}")
    print("=" * 100)

    session_output = Path("/kaggle/working/output/visualize") if Path("/kaggle/working").exists() else REPO_ROOT / "scratch" / "visualize"
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

    print("\n--- Bootstrapping Runtime ---")
    runtime = OperationalKISRuntime.bootstrap(config)

    req = QAQueryRequest(
        request_id=f"vis-{query_id}",
        query_id=query_id,
        event_description=q_vi,
        question=q_vi,
        event_description_en=q_en if q_en else None,
        include_vi_variant=False,
        output_top_k=100,
        refine_top_n=3,
    )
    res = runtime.handle_qa_query(req)
    preds = res.get("predictions", [])

    # Find the top hit or top prediction
    top_pred = preds[0] if preds else None
    strict_hit_pred = None
    for p in preds:
        if p.get("video_id") == target_vid and start_f <= int(p.get("frame_id", -1)) <= end_f:
            strict_hit_pred = p
            break

    chosen_pred = strict_hit_pred or top_pred
    if not chosen_pred:
        print("No prediction generated!")
        return

    chosen_vid = chosen_pred.get("video_id")
    chosen_frame = int(chosen_pred.get("frame_id"))
    chosen_ans = chosen_pred.get("answer")
    chosen_rank = chosen_pred.get("rank")

    print(f"\n--- SELECTED PREDICTION TO SHOW ---")
    print(f"  • Rank       : #{chosen_rank}")
    print(f"  • Video ID   : {chosen_vid} (Target: {target_vid} -> {'CORRECT ✅' if chosen_vid == target_vid else 'MISMATCH ❌'})")
    print(f"  • Frame ID   : {chosen_frame} (GT: [{start_f}..{end_f}] -> {'INSIDE GT ✅' if start_f <= chosen_frame <= end_f else 'OUTSIDE ❌'})")
    print(f"  • Model Ans  : {chosen_ans}")

    # Decode Frame Image
    video_rec = runtime.raw_video_registry.get(chosen_vid)
    if not video_rec or not video_rec.raw_video_path or not video_rec.raw_video_path.is_file():
        print(f"Raw video path not found for {chosen_vid}")
        return

    probe = runtime.decoder.probe(video_rec)
    fps = probe.fps if probe.fps and probe.fps > 0 else 25.0
    center_sec = chosen_frame / fps

    from system_tai.refinement.video import DecodeRequest
    dec_res = runtime.decoder.decode(DecodeRequest(probe=probe, frame_ids=(chosen_frame,), max_decoded_frames=10))

    out_img_path = session_output / f"{query_id}_{chosen_vid}_{chosen_frame}.png"
    out_clip_path = session_output / f"{query_id}_{chosen_vid}_{chosen_frame}_clip.mp4"

    if dec_res.frames:
        img_arr = dec_res.frames[0].image
        pil_img = PILImage.fromarray(img_arr)
        pil_img.save(out_img_path)
        print(f"\nSaved decoded high-res frame image to: {out_img_path}")

    # Extract 6-second video clip around this frame
    extract_video_clip(video_rec.raw_video_path, out_clip_path, center_seconds=center_sec, duration_seconds=6.0)
    print(f"Extracted 6-second video snippet to : {out_clip_path}")

    return {
        "query_id": query_id,
        "question_vi": q_vi,
        "question_en": q_en,
        "target_vid": target_vid,
        "chosen_vid": chosen_vid,
        "chosen_frame": chosen_frame,
        "chosen_rank": chosen_rank,
        "chosen_ans": chosen_ans,
        "in_gt": (start_f <= chosen_frame <= end_f),
        "img_path": str(out_img_path),
        "clip_path": str(out_clip_path),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, default="QA-23", help="Query ID to visualize (e.g. QA-23, QA-46)")
    args = parser.parse_args()
    run_visualization(args.query)
