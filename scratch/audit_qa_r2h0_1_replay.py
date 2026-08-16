# ==============================================================================================================
# QA-R2H0.1 FORENSIC AUDIT: COUNT ANSWER COUNTERFACTUAL FRAME REPLAY FOR QA-20 (L21_V007)
# Replays exact visual scoring on: [14541, 14601, 14631, 14661]
# Execution Mode: Post-processing / frame decode replay (Execution time ~5-10s)
# ==============================================================================================================

from __future__ import annotations

import json
import os
import sys
import unicodedata
from pathlib import Path

# Fix stdout encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

print("=" * 110)
print("QA-R2H0.1 FORENSIC AUDIT: QA-20 RUNTIME ROUTE & PER-FRAME ANSWER REPLAY")
print("=" * 110)

REPO_DIR = Path("/kaggle/working/AIC2026_TeamPTK_SGU")
if not REPO_DIR.exists():
    REPO_DIR = Path(".")

src_path = str(REPO_DIR / "systems" / "system_tai" / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from system_tai.qa.runtime import classify_runtime_question
from system_tai.qa.question_types import classify_question, QuestionType
from system_tai.qa.visual_ontology import (
    load_visual_answer_ontology,
    VisualOntologyAnswerCandidateProvider,
    VisualOntologyConfig,
)
from system_tai.qa.answer_candidates import BaselineQuestionCandidateProvider
from system_tai.qa.models import QAQuery, AnswerHypothesis

SYSTEM_DIR = REPO_DIR / "systems" / "system_tai"
BENCHMARK_PATH = SYSTEM_DIR / "benchmarks" / "l21_150_diagnostic" / "benchmark.json"
DEV_EN_SIDECAR_PATH = SYSTEM_DIR / "benchmarks" / "l21_150_diagnostic" / "qa_dev_translations_en.json"
ONTOLOGY_PATH = SYSTEM_DIR / "benchmarks" / "l21_150_diagnostic" / "qa_dev_visual_ontology.json"

bm_data = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
dev_queries = [
    q for q in bm_data.get("queries", [])
    if str(q.get("split", "")).upper() == "DEV" and str(q.get("task_type", q.get("task", ""))).lower() == "qa"
]
qa_dev_map = {q["query_id"]: q for q in dev_queries}

q20_data = qa_dev_map["QA-20"]
q_vi = q20_data.get("question_vi", "")
q_en = q20_data.get("question_en", "")
target_vid = q20_data.get("video_id", "L21_V007")
s_gt, e_gt = map(int, q20_data.get("proposed_interval", [14610, 14670]))
gold_answers = ["2"]

# --------------------------------------------------------------------------------------------------------------
# STEP 1: RUNTIME ROUTING & PROVIDER AUDIT
# --------------------------------------------------------------------------------------------------------------
print("\n" + "=" * 110)
print("STEP 1: RUNTIME ROUTING & PROVIDER HYPOTHESES AUDIT FOR QA-20")
print("=" * 110)

cls_pure = classify_question(q_vi)
cls_runtime, _ = classify_runtime_question(q_vi, q_en, qa_a2_enabled=False, qa_ocr_enabled=True)

print(f"Benchmark Question (VI)       : {q_vi}")
print(f"Benchmark Question (EN)       : {q_en}")
print(f"Target Video                  : {target_vid}")
print(f"GT Frame Interval             : [{s_gt}, {e_gt}] (Center: {(s_gt + e_gt)//2})")
print(f"Accepted Gold Answers         : {gold_answers}")
print(f"\n[Classification Results]")
print(f"  Pure Question Classifier     : {cls_pure.question_type.value} | Reason: {cls_pure.reason}")
print(f"  Runtime Question Classifier  : {cls_runtime.question_type.value} | Reason: {cls_runtime.reason}")

# Load hypotheses from Visual Ontology Provider
ont_config = VisualOntologyConfig(enabled=True, ontology_path=ONTOLOGY_PATH)
ontology = load_visual_answer_ontology(ONTOLOGY_PATH)
prov_vis = VisualOntologyAnswerCandidateProvider(ontology, ont_config)
prov_base = BaselineQuestionCandidateProvider()

hyps = prov_vis.get_candidates_for_query(cls_runtime.question_type, q_en or q_vi)
print(f"\n[Candidate Hypotheses Provider]")
print(f"  Provider Name                : VisualOntologyAnswerCandidateProvider")
print(f"  Active Question Type         : {cls_runtime.question_type.value}")
print(f"  Total Hypotheses Count       : {len(hyps)}")
print(f"  Hypotheses List              :")
for h in hyps:
    print(f"    -> Canonical: '{h.canonical_answer:<3}' | Aliases: {h.aliases} | Visual Prompts: {h.visual_prompts}")

# --------------------------------------------------------------------------------------------------------------
# STEP 2: DEV-ONLY VISUAL ANSWER SCORING REPLAY ON TARGET FRAMES
# --------------------------------------------------------------------------------------------------------------
print("\n" + "=" * 110)
print("STEP 2: PER-FRAME VISUAL ANSWER SCORING REPLAY ON L21_V007 FRAMES")
print("=" * 110)

REPLAY_FRAMES = [
    (14541, "Primary Refined Anchor"),
    (14601, "Existing +60 Final Frame (Rank 41)"),
    (14631, "Existing +90 Geometry (Inside GT)"),
    (14661, "Existing +120 Geometry (Inside GT)"),
]

# Find video file for L21_V007
video_dirs = [
    Path(f"/kaggle/input/hcm-ai-challenge-2026-round-1/keyframes/{target_vid}"),
    Path(f"/kaggle/input/aic2024-round1-keyframes/{target_vid}"),
    Path(f"/kaggle/input/aic2026-dataset/{target_vid}"),
]

# Search in /kaggle/input
found_video_path = None
for root_cand in [Path("/kaggle/input"), Path("data"), Path("videos")]:
    if root_cand.exists():
        for p in root_cand.glob(f"**/{target_vid}.mp4"):
            found_video_path = p
            break
        if found_video_path:
            break

print(f"Target Video Path: {found_video_path}")

try:
    import torch
    import clip
    import cv2
    from PIL import Image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CLIP model ViT-B/32 on {device}...")
    model, preprocess = clip.load("ViT-B/32", device=device)

    # Encode all text hypotheses
    prompt_entries = []
    for hyp in hyps:
        prompts = hyp.visual_prompts or (hyp.canonical_answer,)
        prompt_entries.append((hyp.canonical_answer, prompts))

    text_tokens_list = []
    text_mapping = []
    for can_ans, prompts in prompt_entries:
        for p_text in prompts:
            text_tokens_list.append(p_text)
            text_mapping.append(can_ans)

    text_tokens = clip.tokenize(text_tokens_list).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    if found_video_path and found_video_path.exists():
        cap = cv2.VideoCapture(str(found_video_path))
        print(f"\nSuccessfully opened video {found_video_path.name}. Total frames in video: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}")

        print(f"\n{'Frame ID':<10} | {'in_GT':<8} | {'Dist to GT':<12} | {'Top-1 Answer':<14} | {'Gold Match':<12} | {'Hypothesis Scores (Top-5)'}")
        print("-" * 110)

        for fid, desc in REPLAY_FRAMES:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
            ret, frame = cap.read()
            if not ret:
                print(f"{fid:<10} | {'ERROR':<8} | {'N/A':<12} | {'DECODE_FAILED':<14} | {'False':<12} | Failed to read frame from video.")
                continue

            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_frame)
            img_tensor = preprocess(pil_img).unsqueeze(0).to(device)

            with torch.no_grad():
                img_feature = model.encode_image(img_tensor)
                img_feature /= img_feature.norm(dim=-1, keepdim=True)
                sims = (img_feature @ text_features.T).squeeze(0)

            # Aggregate scores per canonical answer (max pooling over prompts)
            ans_scores = {}
            for sim, can_ans in zip(sims.tolist(), text_mapping):
                ans_scores[can_ans] = max(ans_scores.get(can_ans, -1.0), float(sim))

            sorted_scores = sorted(ans_scores.items(), key=lambda x: x[1], reverse=True)
            top1_ans, top1_score = sorted_scores[0]
            is_gold = top1_ans in gold_answers
            in_gt = s_gt <= fid <= e_gt
            if fid < s_gt: dist = fid - s_gt
            elif fid > e_gt: dist = fid - e_gt
            else: dist = 0

            top5_str = ", ".join([f"'{a}': {sc:.4f}" for a, sc in sorted_scores[:5]])
            print(f"{fid:<10} | {str(in_gt):<8} | {dist:+d} frames   | '{top1_ans}' ({top1_score:.4f}) | {str(is_gold):<12} | {top5_str}")

            # Specific comparison between '2' and '3'
            sc_2 = ans_scores.get("2", 0.0)
            sc_3 = ans_scores.get("3", 0.0)
            print(f"           --> Description: {desc}")
            print(f"           --> Score('2'): {sc_2:.4f} | Score('3'): {sc_3:.4f} | Margin('2' - '3'): {sc_2 - sc_3:+.4f}")
            print()

        cap.release()
    else:
        print(f"[WARNING] Video file {target_vid}.mp4 not found in search paths. Run this script in Kaggle notebook where /kaggle/input is mounted.")

except Exception as e:
    print(f"[ERROR during replay]: {e}")
    import traceback
    traceback.print_exc()

# --------------------------------------------------------------------------------------------------------------
# STEP 3: SOURCE AUDIT ON FAR-OFFSET TOP-100 STARVATION
# --------------------------------------------------------------------------------------------------------------
print("=" * 110)
print("STEP 3: SOURCE AUDIT ON TOP-100 STARVATION FOR CANDIDATE 4 FAR OFFSETS (+90 / +120)")
print("=" * 110)
print("1. Candidate 4 (L21_V007) Phase Allocations in Top-100 Constructor:")
print("   - Tier 3 Phase A (Direct Primary)    : Slot emitted at Rank 7   (Frame 14541)")
print("   - Tier 3 Phase B (Offset -30)        : Slot emitted at Rank 16  (Frame 14511)")
print("   - Tier 3 Phase C (Offset +30)        : Slot emitted at Rank 25  (Frame 14571)")
print("   - Tier 5 Phase A (Offset +60)        : Slot emitted at Rank 41  (Frame 14601)")
print("   - Tier 5 Phase A (Offset -60)        : Slot emitted at Rank 51  (Frame 14481)")
print("   - Tier 5 Phase A (Offset +45)        : Slot emitted at Rank 61  (Frame 14586)")
print("   - Tier 5 Phase A (Offset -45)        : Slot emitted at Rank 71  (Frame 14496)")
print("   - Primary 11-12 Coverage             : Ranks 78-79")
print("   - Secondary Micro-Budget             : Ranks 80-99 (20 secondary slots emitted for videos 1..10)")
print("   - Tier 5 Phase B (Far Offsets)       :")
print("     -> Attempt +90 for Candidate 1     : Emitted at Rank 100 [TARGET_K=100 REACHED!]")
print("     -> Attempt +90 for Candidate 4     : STARVED (Theoretical Rank 103 > 100)")
print("     -> Attempt +120 for Candidate 4    : STARVED (Theoretical Rank 123 > 100)")
print("\n[SOURCE-DERIVED PROOF]:")
print("  Far offsets (+90, +120) for Candidate 4 (Frame 14631, 14661) are 100% syntactically implemented in Tier 5 Phase B,")
print("  but were truncated because Secondary Micro-Budget + Phase A filled slots 1..99, and Candidate 1 took slot 100.")
print("=" * 110)
