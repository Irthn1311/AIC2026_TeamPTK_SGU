# ==============================================================================================================
# QA-R2H0 ARTIFACT-GROUNDED FORENSIC AUDIT: DYNAMIC VIDEO-HIT / ANSWER-MISS DEEP DIVE
# Target Artifact Directory: /kaggle/working/output/qa_r2g1_treatment_cap16
# Execution Mode: Post-processing only (Execution time < 2 seconds)
# ==============================================================================================================

from __future__ import annotations

import hashlib
import json
import os
import sys
import unicodedata
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

print("=" * 110)
print("QA-R2H0 FORENSIC AUDIT: VIDEO-HIT / ANSWER-MISS INVESTIGATION")
print("=" * 110)

CANDIDATE_DIRS = [
    Path("/kaggle/working/output/qa_r2g1_treatment_cap16"),
    Path("/kaggle/working/output/qa_r2f3_treatment_neg_first"),
    Path("output/qa_r2g1_treatment_cap16"),
]

OUT_DIR = None
for d in CANDIDATE_DIRS:
    if d.exists() and len(list(d.glob("**/qa_predictions.jsonl"))) == 38:
        OUT_DIR = d
        break

if OUT_DIR is None:
    raise FileNotFoundError("Exact QA-R2G1 treatment artifact directory not found at candidate paths.")

REPO_DIR = Path("/kaggle/working/AIC2026_TeamPTK_SGU")
if not REPO_DIR.exists():
    REPO_DIR = Path(".")

SYSTEM_DIR = REPO_DIR / "systems" / "system_tai"
BENCHMARK_PATH = SYSTEM_DIR / "benchmarks" / "l21_150_diagnostic" / "benchmark.json"
DEV_EN_SIDECAR_PATH = SYSTEM_DIR / "benchmarks" / "l21_150_diagnostic" / "qa_dev_translations_en.json"
ONTOLOGY_PATH = SYSTEM_DIR / "benchmarks" / "l21_150_diagnostic" / "qa_dev_visual_ontology.json"

def normalize_text(t: str) -> str:
    if not t:
        return ""
    t = unicodedata.normalize("NFKC", str(t)).casefold()
    return "".join(c for c in t if c.isalnum() or c.isspace()).strip()

bm_data = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
dev_queries = [
    q for q in bm_data.get("queries", [])
    if str(q.get("split", "")).upper() == "DEV" and str(q.get("task_type", q.get("task", ""))).lower() == "qa"
]
qa_dev_map = {q["query_id"]: q for q in dev_queries}

print(f"Loaded {len(dev_queries)} QA DEV queries from benchmark.")
print(f"Active Artifact Directory: {OUT_DIR}")

def run_audit():
    pred_files = list(OUT_DIR.glob("**/qa_predictions.jsonl"))
    ev_files = list(OUT_DIR.glob("**/qa_evidence.json"))
    assert len(pred_files) == 38, f"Expected 38 prediction files, got {len(pred_files)}"
    assert len(ev_files) == 38, f"Expected 38 evidence files, got {len(ev_files)}"

    preds_by_qid = {}
    for pf in pred_files:
        for line in pf.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                preds_by_qid.setdefault(row["query_id"], []).append(row)
    for qid in preds_by_qid:
        preds_by_qid[qid].sort(key=lambda x: int(x["rank"]))

    ev_by_qid = {}
    for ef in ev_files:
        d = json.loads(ef.read_text(encoding="utf-8"))
        ev_by_qid[d["query_id"]] = d

    # 1. Recompute baseline metrics across all 38 queries dynamically
    video_match_qids = []
    answer_match_qids = []
    full_hit_qids = []
    video_hit_answer_miss_qids = []
    full_hit_details = {}
    pred_producing_count = 0

    for q in dev_queries:
        qid = q["query_id"]
        t_vid = q["video_id"]
        s_gt, e_gt = map(int, q.get("proposed_interval", [0, 0]))
        raw_answers = q.get("accepted_answers") or [q.get("answer", "")]
        gold_answers = [normalize_text(a) for a in raw_answers if a]

        preds = preds_by_qid.get(qid, [])
        if len(preds) > 0:
            pred_producing_count += 1

        t_preds = [p for p in preds if p.get("video_id") == t_vid]
        has_vid = len(t_preds) > 0
        has_ans = any(normalize_text(p.get("answer", "")) in gold_answers for p in t_preds)

        full_tuple_preds = [
            p for p in t_preds
            if s_gt <= int(p.get("frame_id", -1)) <= e_gt and normalize_text(p.get("answer", "")) in gold_answers
        ]
        has_full = len(full_tuple_preds) > 0

        if has_vid:
            video_match_qids.append(qid)
        if has_ans:
            answer_match_qids.append(qid)
        if has_full:
            full_hit_qids.append(qid)
            full_hit_details[qid] = int(full_tuple_preds[0]["rank"])
        elif has_vid and not has_ans:
            video_hit_answer_miss_qids.append(qid)

    print("\n" + "=" * 110)
    print("STEP 1: R2G1 TREATMENT BASELINE RECOMPUTATION & RECONCILIATION")
    print("=" * 110)
    print(f"Predictions-Producing Queries : {pred_producing_count} / 38 ({pred_producing_count/38*100:.1f}%)")
    print(f"Video Match Queries           : {len(video_match_qids)} / 38 ({len(video_match_qids)/38*100:.1f}%) -> {video_match_qids}")
    print(f"Answer Match Queries          : {len(answer_match_qids)} / 38 ({len(answer_match_qids)/38*100:.1f}%) -> {answer_match_qids}")
    print(f"Full QA Tuple Hit Queries     : {len(full_hit_qids)} / 38 ({len(full_hit_qids)/38*100:.1f}%) -> {sorted(full_hit_details.items(), key=lambda x: x[1])}")
    print(f"\nDynamically Derived Video-Hit / Answer-Miss Set ({len(video_hit_answer_miss_qids)}): {video_hit_answer_miss_qids}")

    assert pred_producing_count == 37, f"Sanity check failed: Expected 37 pred-producing, got {pred_producing_count}"
    assert len(video_match_qids) == 11, f"Sanity check failed: Expected 11 video matches, got {len(video_match_qids)}"
    assert len(answer_match_qids) == 6, f"Sanity check failed: Expected 6 answer matches, got {len(answer_match_qids)}"
    assert len(full_hit_qids) == 6, f"Sanity check failed: Expected 6 full hits, got {len(full_hit_qids)}"
    assert len(video_hit_answer_miss_qids) == 5, f"Sanity check failed: Expected exactly 5 video-hit/answer-miss queries, got {len(video_hit_answer_miss_qids)}"

    print("\n" + "=" * 110)
    print("STEP 2: DETAILED PER-QUERY FORENSIC AUDIT FOR THE 5 TARGET QUERIES")
    print("=" * 110)

    query_diagnostics = []

    for qid in video_hit_answer_miss_qids:
        q_gt = qa_dev_map[qid]
        t_vid = q_gt["video_id"]
        s_gt, e_gt = map(int, q_gt.get("proposed_interval", [0, 0]))
        raw_answers = q_gt.get("accepted_answers") or [q_gt.get("answer", "")]
        gold_answers = [normalize_text(a) for a in raw_answers if a]
        q_vi = q_gt.get("question_vi", "")
        q_en = q_gt.get("question_en", "")
        q_type = q_gt.get("question_type") or "UNSPECIFIED"

        preds = preds_by_qid.get(qid, [])
        t_preds = [p for p in preds if p.get("video_id") == t_vid]

        frame_hit_any_ans = any(s_gt <= int(p.get("frame_id", -1)) <= e_gt for p in t_preds)
        ans_hit_any_frame = any(normalize_text(p.get("answer", "")) in gold_answers for p in t_preds)
        full_hit = any(s_gt <= int(p.get("frame_id", -1)) <= e_gt and normalize_text(p.get("answer", "")) in gold_answers for p in t_preds)

        # Find closest prediction
        closest_pred = None
        min_dist = float("inf")
        min_signed_dist = 0

        for p in t_preds:
            fid = int(p.get("frame_id", -1))
            if s_gt <= fid <= e_gt:
                d_abs = 0
                d_signed = 0
            elif fid < s_gt:
                d_abs = s_gt - fid
                d_signed = fid - s_gt
            else:
                d_abs = fid - e_gt
                d_signed = fid - e_gt

            if d_abs < min_dist:
                min_dist = d_abs
                min_signed_dist = d_signed
                closest_pred = p

        ev_data = ev_by_qid.get(qid, {})
        usable = ev_data.get("usable_evidence_candidates", [])
        t_usable = [c for c in usable if c.get("video_id") == t_vid]

        diag_entry = {
            "qid": qid,
            "q_vi": q_vi,
            "q_en": q_en,
            "q_type": q_type,
            "t_vid": t_vid,
            "gt_interval": [s_gt, e_gt],
            "gold_answers": gold_answers,
            "t_preds": t_preds,
            "frame_hit_any_ans": frame_hit_any_ans,
            "ans_hit_any_frame": ans_hit_any_frame,
            "full_hit": full_hit,
            "min_dist": min_dist,
            "min_signed_dist": min_signed_dist,
            "closest_pred": closest_pred,
            "t_usable": t_usable,
            "ev_data": ev_data,
        }
        query_diagnostics.append(diag_entry)

        print("\n" + "-" * 110)
        print(f"QUERY AUDIT: {qid} | Question Type: {q_type} | Target Video: {t_vid}")
        print("-" * 110)
        print(f"Question (VI):         {q_vi}")
        print(f"Question (EN):         {q_en}")
        print(f"Target Video:          {t_vid}")
        print(f"GT Frame Interval:     [{s_gt}, {e_gt}] (Center: {(s_gt + e_gt)//2}, Width: {e_gt - s_gt} frames)")
        print(f"Accepted Gold Answers: {gold_answers}")
        print(f"\n[Status Flags & Localization Summary]")
        print(f"  FRAME_HIT_ANY_ANSWER : {frame_hit_any_ans}")
        print(f"  ANSWER_HIT_ANY_FRAME : {ans_hit_any_frame}")
        print(f"  FULL_HIT             : {full_hit}")
        print(f"  Total Predictions    : {len(t_preds)} slots in Top-100")
        if closest_pred is not None:
            print(f"  Closest Prediction   : Rank {closest_pred.get('rank')} | Frame {closest_pred.get('frame_id')} | Signed Dist: {min_signed_dist:+d} frames | Answer: '{closest_pred.get('answer')}'")

        print(f"\n--- [Target Video {t_vid} Final Predictions (All {len(t_preds)})] ---")
        print(f"{'Rank':<6} | {'Frame ID':<10} | {'Answer':<25} | {'in_GT':<8} | {'Ans Match':<10} | {'Signed Dist'}")
        print("-" * 80)
        for p in t_preds:
            fid = int(p.get("frame_id", -1))
            in_gt = s_gt <= fid <= e_gt
            ans = p.get("answer", "")
            ans_match = normalize_text(ans) in gold_answers
            if fid < s_gt: dist = fid - s_gt
            elif fid > e_gt: dist = fid - e_gt
            else: dist = 0
            print(f"{p.get('rank'):<6} | {fid:<10} | {ans:<25} | {str(in_gt):<8} | {str(ans_match):<10} | {dist:+d} frames")

        print(f"\n--- [Target Video {t_vid} Usable Evidence Candidates ({len(t_usable)}/{len(usable)})] ---")
        print(f"{'Pos':<5} | {'Frame ID':<10} | {'Nom Rank':<10} | {'Local Rank':<12} | {'Ev Source':<18} | {'Signed Dist'}")
        print("-" * 75)
        for idx, c in enumerate(t_usable, start=1):
            fid = int(c.get("frame_id", -1))
            if fid < s_gt: dist = fid - s_gt
            elif fid > e_gt: dist = fid - e_gt
            else: dist = 0
            print(f"{idx:<5} | {fid:<10} | {str(c.get('video_nomination_rank')):<10} | {str(c.get('local_anchor_rank')):<12} | {str(c.get('evidence_source')):<18} | {dist:+d} frames")

    # Step 3: Rank candidates primarily by smallest blocker
    print("\n" + "=" * 110)
    print("STEP 3: CANDIDATE RANKING BY SMALLEST REMAINING BLOCKER")
    print("=" * 110)
    query_diagnostics.sort(key=lambda x: (not x["frame_hit_any_ans"], x["min_dist"]))

    print(f"{'Pri':<4} | {'QID':<8} | {'Type':<12} | {'Target Video':<14} | {'Frame in GT':<12} | {'Min Dist to GT':<18} | {'Primary Blocker'}")
    print("-" * 105)
    for idx, d in enumerate(query_diagnostics, start=1):
        if d["frame_hit_any_ans"]:
            blocker = "ANSWER_SCORING_ONLY (Frame already inside GT!)"
        elif d["min_dist"] <= 120:
            blocker = f"NEAR_GT_LOCALIZATION (Dist: {d['min_dist']} frames) + ANSWER_SCORING"
        else:
            blocker = f"FAR_LOCALIZATION (Dist: {d['min_dist']} frames)"
        print(f"{idx:<4} | {d['qid']:<8} | {d['q_type']:<12} | {d['t_vid']:<14} | {str(d['frame_hit_any_ans']):<12} | {d['min_dist']:<18} | {blocker}")

    # Step 4: Deep dive into the strongest candidate
    strongest = query_diagnostics[0]
    s_qid = strongest["qid"]
    s_ev = strongest["ev_data"]
    s_vid = strongest["t_vid"]
    print("\n" + "=" * 110)
    print(f"STEP 4: DEEP UPSTREAM PIPELINE AUDIT FOR STRONGEST CANDIDATE: {s_qid}")
    print("=" * 110)
    print(f"Strongest Candidate Query ID: {s_qid} ({strongest['q_type']})")
    print(f"Question (VI): {strongest['q_vi']}")
    print(f"Question (EN): {strongest['q_en']}")
    print(f"Target Video:  {s_vid} | GT Interval: {strongest['gt_interval']} | Gold Answers: {strongest['gold_answers']}")
    
    # Check upstream stages for strongest candidate
    fused = s_ev.get("fused_retrieval_candidates", [])
    t_fused = [c for c in fused if c.get("video_id") == s_vid]
    print(f"\n1. Fused Retrieval Candidates for {s_vid}: {len(t_fused)} frames")
    for c in t_fused:
        print(f"   -> Frame {c.get('frame_id')} (Rank {c.get('rank', 'N/A')})")

    keyframe_cands = s_ev.get("keyframe_evidence_candidates", [])
    t_keyframe = [c for c in keyframe_cands if c.get("video_id") == s_vid]
    print(f"\n2. Keyframe Evidence Candidates for {s_vid}: {len(t_keyframe)} frames")
    for c in t_keyframe:
        print(f"   -> Frame {c.get('frame_id')} (Nomination Rank {c.get('video_nomination_rank')}, Local Rank {c.get('local_anchor_rank')})")

    temporal_seeds = s_ev.get("temporal_seed_candidates", [])
    t_temp = [c for c in temporal_seeds if c.get("video_id") == s_vid]
    print(f"\n3. Temporal Seeds for {s_vid}: {len(t_temp)} frames")
    for c in t_temp:
        print(f"   -> Frame {c.get('frame_id')}")

    refined_selected = s_ev.get("refinement_selected_candidates", [])
    t_ref_sel = [c for c in refined_selected if c.get("video_id") == s_vid]
    print(f"\n4. Refinement Selected for {s_vid}: {len(t_ref_sel)} frames")
    for c in t_ref_sel:
        print(f"   -> Frame {c.get('frame_id')} (Status: {c.get('status')})")

    refined_success = s_ev.get("refinement_success_candidates", [])
    t_ref_succ = [c for c in refined_success if c.get("video_id") == s_vid]
    print(f"\n5. Refinement Success for {s_vid}: {len(t_ref_succ)} frames")
    for c in t_ref_succ:
        print(f"   -> Original Frame {c.get('candidate_frame_id')} -> Refined Frame {c.get('frame_id')} (Status: {c.get('status')})")

if __name__ == "__main__":
    run_audit()
