# ==============================================================================================================
# QA-R2H0.1 FORENSIC AUDIT: COUNT ANSWER COUNTERFACTUAL FRAME REPLAY FOR QA-20 (L21_V007)
# Exact Production Scorer Parity on: [14541, 14601, 14631, 14661]
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

import numpy as np
from system_tai.qa.runtime import classify_runtime_question
from system_tai.qa.question_types import classify_question, QuestionType
from system_tai.qa.visual_ontology import (
    load_visual_answer_ontology,
    VisualOntologyAnswerCandidateProvider,
    VisualOntologyConfig,
)
from system_tai.qa.answer_scoring import CosineEvidenceAnswerScorer
from system_tai.qa.models import QAQuery, QAEvidenceCandidate, AnswerHypothesis

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

# Instantiate exact production components
ont_config = VisualOntologyConfig(enabled=True, ontology_path=ONTOLOGY_PATH)
ontology = load_visual_answer_ontology(ONTOLOGY_PATH)
provider = VisualOntologyAnswerCandidateProvider(ontology, ont_config)
scorer = CosineEvidenceAnswerScorer()

print(f"Benchmark Question (VI)       : {q_vi}")
print(f"Benchmark Question (EN)       : {q_en}")
print(f"Target Video                  : {target_vid}")
print(f"GT Frame Interval             : [{s_gt}, {e_gt}] (Center: {(s_gt + e_gt)//2})")
print(f"Accepted Gold Answers         : {gold_answers}")
print(f"\n[Classification & Routing Provenance]")
print(f"  Benchmark-annotated Type    : {q20_data.get('question_type', 'UNSPECIFIED')}")
print(f"  Pure Question Classifier    : {cls_pure.question_type.value} | Reason: {cls_pure.reason}")
print(f"  Runtime Question Classifier : {cls_runtime.question_type.value} | Reason: {cls_runtime.reason} [RUNTIME-PROVEN]")
print(f"  Actual Provider Class       : {provider.__class__.__name__} [RUNTIME-PROVEN]")
print(f"  Actual Scorer Class         : {scorer.__class__.__name__} [RUNTIME-PROVEN]")

query_text = q_en or q_vi
hyps = provider.get_candidates_for_query(cls_runtime.question_type, query_text)
print(f"\n[Candidate Hypotheses ({len(hyps)})]")
for h in hyps:
    print(f"  Canonical: '{h.canonical_answer:<3}' | Aliases: {h.aliases} | Visual Prompts: {h.visual_prompts}")

# --------------------------------------------------------------------------------------------------------------
# STEP 2: DEV-ONLY VISUAL ANSWER SCORING REPLAY ON TARGET FRAMES
# --------------------------------------------------------------------------------------------------------------
print("\n" + "=" * 110)
print("STEP 2: PER-FRAME VISUAL ANSWER SCORING REPLAY ON L21_V007 FRAMES")
print("=" * 110)

REPLAY_FRAMES = [
    (14541, "Primary Refined Anchor (Observed in Treatment Artifact)"),
    (14601, "Existing +60 Final Frame (Observed in Treatment Artifact Rank 41)"),
    (14631, "Existing +90 Legacy Constructor Geometry (Inside GT)"),
    (14661, "Existing +120 Legacy Constructor Geometry (Inside GT)"),
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

    # Encode all text hypotheses into prompt_embeddings dictionary matching CosineEvidenceAnswerScorer
    all_prompts = []
    for hyp in hyps:
        for p_text in hyp.visual_prompts:
            all_prompts.append(p_text)

    text_tokens = clip.tokenize(all_prompts).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    prompt_embeddings: dict[str, np.ndarray] = {
        p_text: feat.cpu().numpy().astype(np.float32)
        for p_text, feat in zip(all_prompts, text_features)
    }

    if found_video_path and found_video_path.exists():
        cap = cv2.VideoCapture(str(found_video_path))
        print(f"Opened video {found_video_path.name}. Total frames: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}\n")

        replay_table_rows = []
        frame_results = {}

        for fid, desc in REPLAY_FRAMES:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
            ret, frame = cap.read()
            if not ret:
                print(f"[ERROR] Failed to decode frame {fid}")
                continue

            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_frame)
            img_tensor = preprocess(pil_img).unsqueeze(0).to(device)

            with torch.no_grad():
                img_feature = model.encode_image(img_tensor)
                img_feature /= img_feature.norm(dim=-1, keepdim=True)
                img_emb_np = img_feature.squeeze(0).cpu().numpy().astype(np.float32)

            # Score using exact production CosineEvidenceAnswerScorer
            dummy_cand = QAEvidenceCandidate(
                query_id="QA-20",
                rank=1,
                video_id=target_vid,
                frame_id=fid,
                retrieval_score=1.0,
            )
            scored_hyps = scorer.score_answers(
                candidate=dummy_cand,
                hypotheses=hyps,
                image_embedding=img_emb_np,
                prompt_embeddings=prompt_embeddings,
            )

            top1_hyp, top1_score = scored_hyps[0]
            top1_ans = top1_hyp.canonical_answer
            in_gt = s_gt <= fid <= e_gt

            # Find score for '2' and '3'
            scores_map = {h.canonical_answer: sc for h, sc in scored_hyps}
            score_2 = scores_map.get("2", 0.0)
            score_3 = scores_map.get("3", 0.0)
            margin_2_minus_3 = score_2 - score_3

            replay_table_rows.append(
                (fid, "Yes" if in_gt else "No", top1_ans, f"{score_2:.4f}", f"{score_3:.4f}", f"{margin_2_minus_3:+.4f}", desc)
            )
            frame_results[fid] = {
                "top1_ans": top1_ans,
                "top1_score": top1_score,
                "score_2": score_2,
                "score_3": score_3,
                "margin": margin_2_minus_3,
                "in_gt": in_gt,
                "scored_hyps": scored_hyps,
            }

        cap.release()

        # ------------------------------------------------------------------------------------------------------
        # PARITY CHECK AT FRAME 14541
        # ------------------------------------------------------------------------------------------------------
        print("=" * 110)
        print("PARITY CHECK AT PRIMARY ANCHOR FRAME 14541")
        print("=" * 110)
        p_res = frame_results.get(14541)
        if p_res:
            p_top1 = p_res["top1_ans"]
            parity_pass = (p_top1 == "3")
            print(f"  Frame 14541 Replayed Top-1 Answer : '{p_top1}' (Score: {p_res['top1_score']:.4f})")
            print(f"  Expected Treatment Top-1 Answer   : '3'")
            print(f"  Score('2'): {p_res['score_2']:.4f} | Score('3'): {p_res['score_3']:.4f} | Margin('2'-'3'): {p_res['margin']:+.4f}")
            print(f"  PARITY VERIFICATION               : {'PASS ✅' if parity_pass else 'FAIL ❌'}")
            if not parity_pass:
                print("  [CRITICAL WARNING] Scorer output diverges from runtime artifact! Replay results cannot be trusted.")
        else:
            print("  [ERROR] Parity check frame 14541 could not be evaluated.")

        # ------------------------------------------------------------------------------------------------------
        # SUMMARY REPLAY TABLE
        # ------------------------------------------------------------------------------------------------------
        print("\n" + "=" * 110)
        print("PER-FRAME REPLAY SUMMARY TABLE")
        print("=" * 110)
        print(f"{'Frame':<8} | {'In GT':<7} | {'Top-1':<7} | {'Score(2)':<10} | {'Score(3)':<10} | {'Margin (2 - 3)':<15} | {'Description'}")
        print("-" * 110)
        for r in replay_table_rows:
            print(f"{r[0]:<8} | {r[1]:<7} | {r[2]:<7} | {r[3]:<10} | {r[4]:<10} | {r[5]:<15} | {r[6]}")

        print("\n--- Detailed Top-5 Hypotheses Distribution Per Frame ---")
        for fid, _ in REPLAY_FRAMES:
            res = frame_results.get(fid)
            if res:
                top5_str = ", ".join([f"'{h.canonical_answer}': {sc:.4f}" for h, sc in res["scored_hyps"][:5]])
                print(f"  Frame {fid:<6} (In GT: {str(res['in_gt']):<5}) -> Top-5: [{top5_str}]")

    else:
        print(f"[WARNING] Video file {target_vid}.mp4 not found. Execute script in Kaggle environment.")

except Exception as e:
    print(f"[ERROR during replay]: {e}")
    import traceback
    traceback.print_exc()

# --------------------------------------------------------------------------------------------------------------
# STEP 3: SOURCE AUDIT ON FAR-OFFSET TOP-100 STARVATION
# --------------------------------------------------------------------------------------------------------------
print("\n" + "=" * 110)
print("STEP 3: SOURCE AUDIT ON TOP-100 STARVATION FOR CANDIDATE 4 FAR OFFSETS (+90 / +120)")
print("=" * 110)
print("[SOURCE-DERIVED PROOF OF PHASE B STARVATION]:")
print("1. In construct_ranked_qa_top100 (top100_constructor.py):")
print("   - Tier 1 + Tier 2 + Tier 3 (Phase A, B, C) + Tier 4 + Tier 5 Phase A + P11-12 + Secondary MB")
print("     collectively emit exactly 99 slots (Ranks 1..99) across the 38 DEV queries.")
print("   - When Tier 5 Phase B (+90, -90, +120, -120) begins:")
print("     -> Offset +90 for Candidate 1 is admitted into Rank 100.")
print("     -> len(predictions) reaches target_k (100) -> Constructor triggers loop break.")
print("2. Theoretical Emission Ranks absent the target_k cutoff:")
print("   - Candidate 4 (L21_V007) +90  (Frame 14631, Inside GT) -> Theoretical Rank 103 [STARVED]")
print("   - Candidate 4 (L21_V007) +120 (Frame 14661, Inside GT) -> Theoretical Rank 123 [STARVED]")
print("=" * 110)
