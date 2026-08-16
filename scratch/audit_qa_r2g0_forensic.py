# ==============================================================================================================
# QA-R2G0 ARTIFACT-GROUNDED FORENSIC AUDIT: 6TH ANSWER-MATCH / FRAME-MISS DEEP DIVE
# Target Artifact Directory: /kaggle/working/output/qa_r2f3_treatment_neg_first (or fallback to any QA output)
# Mode: Post-processing only (Execution time < 2 seconds)
# ==============================================================================================================

from __future__ import annotations

import hashlib
import json
import sys
import unicodedata
from pathlib import Path

# Fix stdout encoding for safe printing
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

print("=" * 110)
print("QA-R2G0 FORENSIC AUDIT: ANSWER-HIT / FRAME-MISS INVESTIGATION")
print("=" * 110)

# Paths to search for artifacts
CANDIDATE_DIRS = [
    Path("/kaggle/working/output/qa_r2f3_treatment_neg_first"),
    Path("/kaggle/working/output/qa_r2f3_control_pos_first"),
    Path("/kaggle/working/output/qa_r2f2_treatment_t3_on"),
    Path("/kaggle/working/output/qa_r2f1_treatment_p11_12_on"),
    Path("output/qa_r2f3_treatment_neg_first"),
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

if not BENCHMARK_PATH.exists():
    raise FileNotFoundError(f"Missing benchmark file: {BENCHMARK_PATH}")

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
if OUT_DIR is not None:
    print(f"Active Artifact Directory: {OUT_DIR}")
else:
    print("[WARNING] Kaggle output directory not found in candidate paths.")
    print("Script will run in standalone mode or wait for execution on Kaggle.")

def run_audit():
    if OUT_DIR is None:
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

    # 1. Evaluate all 38 queries to classify Video Match, Answer Match, and Full Tuple Hit
    video_matches = []
    answer_matches = []
    full_hits = []
    target_miss_queries = []  # Video Match + Answer Match + Frame Miss

    for q in dev_queries:
        qid = q["query_id"]
        t_vid = q["video_id"]
        s_gt, e_gt = map(int, q.get("proposed_interval", [0, 0]))
        raw_answers = q.get("accepted_answers") or [q.get("answer", "")]
        gold_answers = [normalize_text(a) for a in raw_answers if a]

        preds = preds_by_qid.get(qid, [])
        t_preds = [p for p in preds if p.get("video_id") == t_vid]

        has_vid = len(t_preds) > 0
        has_ans = any(normalize_text(p.get("answer", "")) in gold_answers for p in t_preds)
        
        full_tuple_preds = [
            p for p in t_preds
            if s_gt <= int(p.get("frame_id", -1)) <= e_gt and normalize_text(p.get("answer", "")) in gold_answers
        ]
        has_full = len(full_tuple_preds) > 0

        if has_vid:
            video_matches.append((qid, t_vid, len(t_preds)))
        if has_ans:
            answer_matches.append((qid, t_vid))
        if has_full:
            best_rank = int(full_tuple_preds[0]["rank"])
            full_hits.append((qid, best_rank))
        elif has_ans and not has_full:
            target_miss_queries.append(qid)

    print("\n" + "=" * 110)
    print(f"AUDIT SUMMARY: {len(video_matches)}/38 Video Matches | {len(answer_matches)}/38 Answer Matches | {len(full_hits)}/38 Full Hits")
    print("=" * 110)
    print(f"Video Matches (11):  {[v[0] for v in video_matches]}")
    print(f"Answer Matches (6):  {[a[0] for a in answer_matches]}")
    print(f"Full Tuple Hits (5): {sorted(full_hits, key=lambda x: x[1])}")
    print(f"\n🎯 TARGET QUERY WITH ANSWER MATCH BUT FRAME MISS ({len(target_miss_queries)}): {target_miss_queries}")

    # 2. Deep Dive into the Target Query
    for target_qid in target_miss_queries:
        q_gt = qa_dev_map[target_qid]
        t_vid = q_gt["video_id"]
        s_gt, e_gt = map(int, q_gt.get("proposed_interval", [0, 0]))
        raw_answers = q_gt.get("accepted_answers") or [q_gt.get("answer", "")]
        gold_answers = [normalize_text(a) for a in raw_answers if a]
        q_vi = q_gt.get("question_vi", "")
        q_en = q_gt.get("question_en", "")

        print("\n" + "=" * 110)
        print(f"DEEP FORENSIC AUDIT FOR TARGET QUERY: {target_qid}")
        print("=" * 110)
        print(f"Query ID:             {target_qid}")
        print(f"Question (VI):        {q_vi}")
        print(f"Question (EN):        {q_en}")
        print(f"Target Video:         {t_vid}")
        print(f"Ground Truth Interval:[{s_gt}, {e_gt}] (Center: {(s_gt + e_gt)//2})")
        print(f"Accepted Gold Answers:{gold_answers}")

        # A. All final predictions for Target Video
        preds = preds_by_qid.get(target_qid, [])
        t_preds = [p for p in preds if p.get("video_id") == t_vid]
        print(f"\n--- [A. FINAL TOP-100 PREDICTIONS FOR TARGET VIDEO {t_vid}] (Total: {len(t_preds)}) ---")
        print(f"{'Rank':<6} | {'Frame ID':<10} | {'Answer':<16} | {'Answer Match':<14} | {'in_GT Interval':<16} | {'Signed Distance to GT'}")
        print("-" * 95)
        for p in t_preds:
            fid = int(p.get("frame_id", -1))
            ans = p.get("answer", "")
            ans_match = normalize_text(ans) in gold_answers
            in_gt = s_gt <= fid <= e_gt
            if fid < s_gt:
                dist = fid - s_gt  # Negative: before GT
            elif fid > e_gt:
                dist = fid - e_gt  # Positive: after GT
            else:
                dist = 0
            print(f"{p.get('rank'):<6} | {fid:<10} | {ans:<16} | {str(ans_match):<14} | {str(in_gt):<16} | {dist:+d} frames")

        # B. Usable Evidence Candidates Audit
        ev_data = ev_by_qid.get(target_qid, {})
        usable = ev_data.get("usable_evidence_candidates", [])
        t_usable = [c for c in usable if c.get("video_id") == t_vid]
        print(f"\n--- [B. USABLE EVIDENCE CANDIDATES FOR TARGET VIDEO {t_vid}] (Total in usable: {len(t_usable)}/{len(usable)}) ---")
        print(f"{'Pos':<5} | {'Frame ID':<10} | {'Nom Rank':<10} | {'Local Rank':<12} | {'Ev Source':<15} | {'Distance to GT'}")
        print("-" * 80)
        for idx, c in enumerate(t_usable, start=1):
            fid = int(c.get("frame_id", -1))
            nom_r = c.get("video_nomination_rank", "N/A")
            loc_r = c.get("local_anchor_rank", "N/A")
            src = c.get("evidence_source", "KEYFRAME_BANK")
            if fid < s_gt: dist = fid - s_gt
            elif fid > e_gt: dist = fid - e_gt
            else: dist = 0
            print(f"{idx:<5} | {fid:<10} | {str(nom_r):<10} | {str(loc_r):<12} | {str(src):<15} | {dist:+d} frames")

        # C. Upstream Stage Audit: Fused Retrieval, Keyframe Bank, Temporal Seeds
        print(f"\n--- [C. UPSTREAM STAGES AUDIT FOR TARGET VIDEO {t_vid}] ---")
        fused = ev_data.get("fused_retrieval_candidates", [])
        t_fused = [c for c in fused if c.get("video_id") == t_vid]
        print(f"1. Fused Retrieval Candidates for {t_vid}: {len(t_fused)} frames")
        for c in t_fused[:10]:
            fid = int(c.get("frame_id", -1))
            if fid < s_gt: dist = fid - s_gt
            elif fid > e_gt: dist = fid - e_gt
            else: dist = 0
            print(f"   -> Frame {fid} (retrieval_rank={c.get('retrieval_rank', 'N/A')}, dist={dist:+d} frames)")

        keyframe_cands = ev_data.get("keyframe_evidence_candidates", [])
        t_keyframe = [c for c in keyframe_cands if c.get("video_id") == t_vid]
        print(f"2. Keyframe Evidence Bank Candidates for {t_vid}: {len(t_keyframe)} frames")
        for c in t_keyframe[:10]:
            fid = int(c.get("frame_id", -1))
            if fid < s_gt: dist = fid - s_gt
            elif fid > e_gt: dist = fid - e_gt
            else: dist = 0
            print(f"   -> Frame {fid} (local_anchor_rank={c.get('local_anchor_rank', 'N/A')}, dist={dist:+d} frames)")

        temporal_seeds = ev_data.get("temporal_seed_candidates", [])
        t_temp = [c for c in temporal_seeds if c.get("video_id") == t_vid]
        print(f"3. Temporal Refinement Seeds for {t_vid}: {len(t_temp)} frames")
        for c in t_temp[:10]:
            fid = int(c.get("frame_id", -1))
            if fid < s_gt: dist = fid - s_gt
            elif fid > e_gt: dist = fid - e_gt
            else: dist = 0
            print(f"   -> Frame {fid} (dist={dist:+d} frames)")

if __name__ == "__main__":
    run_audit()
