# ==============================================================================================================
# QA-R2H0 ARTIFACT-GROUNDED FORENSIC AUDIT: VIDEO-HIT / ANSWER-MISS INVESTIGATION (QA-02, QA-20, QA-23, QA-30, QA-31)
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
print("QA-R2H0 FORENSIC AUDIT: VIDEO-HIT / ANSWER-MISS DEEP DIVE")
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

print(f"Loaded {len(dev_queries)} QA DEV queries.")
print(f"Active Artifact Directory: {OUT_DIR}")

TARGET_QIDS = ["QA-02", "QA-20", "QA-23", "QA-30", "QA-31"]

def run_audit():
    if OUT_DIR is None:
        print("[ERROR] Artifact directory not found.")
        return

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

    query_diagnostics = []

    for qid in TARGET_QIDS:
        q_gt = qa_dev_map[qid]
        t_vid = q_gt["video_id"]
        s_gt, e_gt = map(int, q_gt.get("proposed_interval", [0, 0]))
        raw_answers = q_gt.get("accepted_answers") or [q_gt.get("answer", "")]
        gold_answers = [normalize_text(a) for a in raw_answers if a]
        q_vi = q_gt.get("question_vi", "")
        q_en = q_gt.get("question_en", "")

        preds = preds_by_qid.get(qid, [])
        t_preds = [p for p in preds if p.get("video_id") == t_vid]

        frame_hit_any_ans = any(s_gt <= int(p.get("frame_id", -1)) <= e_gt for p in t_preds)
        ans_hit_any_frame = any(normalize_text(p.get("answer", "")) in gold_answers for p in t_preds)
        full_hit = any(s_gt <= int(p.get("frame_id", -1)) <= e_gt and normalize_text(p.get("answer", "")) in gold_answers for p in t_preds)

        # Compute min distance to GT
        min_dist = float("inf")
        closest_frame = None
        for p in t_preds:
            fid = int(p.get("frame_id", -1))
            if s_gt <= fid <= e_gt:
                dist = 0
            elif fid < s_gt:
                dist = s_gt - fid
            else:
                dist = fid - e_gt
            if dist < min_dist:
                min_dist = dist
                closest_frame = fid

        ev_data = ev_by_qid.get(qid, {})
        usable = ev_data.get("usable_evidence_candidates", [])
        t_usable = [c for c in usable if c.get("video_id") == t_vid]

        diag_entry = {
            "qid": qid,
            "q_vi": q_vi,
            "q_en": q_en,
            "t_vid": t_vid,
            "gt_interval": [s_gt, e_gt],
            "gold_answers": gold_answers,
            "t_preds": t_preds,
            "frame_hit_any_ans": frame_hit_any_ans,
            "ans_hit_any_frame": ans_hit_any_frame,
            "full_hit": full_hit,
            "min_dist": min_dist,
            "closest_frame": closest_frame,
            "t_usable": t_usable,
            "ev_data": ev_data,
        }
        query_diagnostics.append(diag_entry)

        print("\n" + "=" * 110)
        print(f"FORENSIC REPORT FOR QUERY: {qid}")
        print("=" * 110)
        print(f"Query ID:             {qid}")
        print(f"Question (VI):        {q_vi}")
        print(f"Question (EN):        {q_en}")
        print(f"Target Video:         {t_vid}")
        print(f"Ground Truth Interval:[{s_gt}, {e_gt}] (Center: {(s_gt + e_gt)//2}, Width: {e_gt - s_gt})")
        print(f"Accepted Gold Answers:{gold_answers}")
        print(f"\n[Status Flags]")
        print(f"  VIDEO_MATCH            : True ({len(t_preds)} predictions in Top-100)")
        print(f"  FRAME_HIT_ANY_ANSWER   : {frame_hit_any_ans}")
        print(f"  ANSWER_HIT_ANY_FRAME   : {ans_hit_any_frame}")
        print(f"  FULL_HIT               : {full_hit}")
        print(f"  Min Distance to GT     : {min_dist} frames (Closest frame: {closest_frame})")

        print(f"\n--- [A. TARGET VIDEO {t_vid} FINAL PREDICTIONS (Total: {len(t_preds)})] ---")
        print(f"{'Rank':<6} | {'Frame ID':<10} | {'Answer':<25} | {'in_GT':<8} | {'Ans Match':<10} | {'Signed Dist to GT'}")
        print("-" * 85)
        for p in t_preds:
            fid = int(p.get("frame_id", -1))
            in_gt = s_gt <= fid <= e_gt
            ans = p.get("answer", "")
            ans_match = normalize_text(ans) in gold_answers
            if fid < s_gt: dist = fid - s_gt
            elif fid > e_gt: dist = fid - e_gt
            else: dist = 0
            print(f"{p.get('rank'):<6} | {fid:<10} | {ans:<25} | {str(in_gt):<8} | {str(ans_match):<10} | {dist:+d} frames")

        print(f"\n--- [B. USABLE EVIDENCE CANDIDATES FOR {t_vid} (Total: {len(t_usable)})] ---")
        print(f"{'Pos':<5} | {'Frame ID':<10} | {'Nom Rank':<10} | {'Local Rank':<12} | {'Ev Source':<18} | {'Distance to GT'}")
        print("-" * 75)
        for idx, c in enumerate(t_usable, start=1):
            fid = int(c.get("frame_id", -1))
            if fid < s_gt: dist = fid - s_gt
            elif fid > e_gt: dist = fid - e_gt
            else: dist = 0
            print(f"{idx:<5} | {fid:<10} | {str(c.get('video_nomination_rank')):<10} | {str(c.get('local_anchor_rank')):<12} | {str(c.get('evidence_source')):<18} | {dist:+d} frames")

    # Rank the 5 queries by smallest remaining blocker
    print("\n" + "=" * 110)
    print("RANKING OF THE 5 VIDEO-HIT / ANSWER-MISS QUERIES BY SMALLEST REMAINING BLOCKER")
    print("=" * 110)

    # Sort key: (not frame_hit_any_ans, min_dist)
    query_diagnostics.sort(key=lambda x: (not x["frame_hit_any_ans"], x["min_dist"]))
    print(f"{'Rank':<5} | {'QID':<8} | {'Target Video':<14} | {'Frame Hit (Any Ans)':<22} | {'Min Distance to GT':<20} | {'Primary Blocker'}")
    print("-" * 95)
    for idx, d in enumerate(query_diagnostics, start=1):
        if d["frame_hit_any_ans"]:
            blocker = "ANSWER_SCORING_ONLY (Frame already inside GT!)"
        elif d["min_dist"] <= 100:
            blocker = f"NEAR_GT_LOCALIZATION (Dist: {d['min_dist']} frames) + ANSWER_SCORING"
        else:
            blocker = f"FAR_LOCALIZATION (Dist: {d['min_dist']} frames)"
        print(f"{idx:<5} | {d['qid']:<8} | {d['t_vid']:<14} | {str(d['frame_hit_any_ans']):<22} | {d['min_dist']:<20} | {blocker}")

if __name__ == "__main__":
    run_audit()
