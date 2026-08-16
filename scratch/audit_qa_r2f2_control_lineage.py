# ==============================================================================================================
# QA-R2F2.1 FORENSIC AUDIT: QA-13 EXACT CONTROL LINEAGE & PROTECTED HITS SAFETY AUDIT
# ==============================================================================================================

from __future__ import annotations

import json
from pathlib import Path

# Paths to artifacts if run on Kaggle
CONTROL_DIR = Path("/kaggle/working/output/qa_r2f2_control_t3_off")
TREATMENT_DIR = Path("/kaggle/working/output/qa_r2f2_treatment_t3_on")

def load_run(out_dir: Path):
    if not out_dir.exists():
        return None, None
    pred_files = list(out_dir.glob("**/qa_predictions.jsonl"))
    ev_files = list(out_dir.glob("**/qa_evidence.json"))
    
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
        
    return preds_by_qid, ev_by_qid

def audit_qa13_lineage():
    print("=" * 110)
    print("QA-R2F2.1 FORENSIC: QA-13 EXACT CONTROL & TREATMENT TARGET PREDICTIONS")
    print("=" * 110)
    
    preds_ctrl, ev_ctrl = load_run(CONTROL_DIR)
    preds_treat, ev_treat = load_run(TREATMENT_DIR)
    
    if preds_ctrl is None:
        print("[INFO] Kaggle output directory not found locally. Running deterministic derivation.")
        return

    q13_preds_ctrl = preds_ctrl.get("QA-13", [])
    q13_preds_treat = preds_treat.get("QA-13", [])
    q13_ev = ev_ctrl.get("QA-13", {})
    
    target_vid = "L21_V005"
    gt_s, gt_e = 23160, 23220
    gold_answer = "trâu"
    
    print("\n--- CONTROL PREDICTIONS FOR QA-13 (TARGET VIDEO L21_V005) ---")
    ctrl_target_preds = [p for p in q13_preds_ctrl if p.get("video_id") == target_vid]
    print(f"{'Rank':<6} | {'Frame ID':<10} | {'Answer':<12} | {'in_GT [23160, 23220]':<22} | {'Gold Match'}")
    print("-" * 75)
    for p in ctrl_target_preds:
        fid = int(p.get("frame_id", -1))
        in_gt = gt_s <= fid <= gt_e
        ans = p.get("answer", "")
        gm = (ans == gold_answer)
        print(f"{p.get('rank'):<6} | {fid:<10} | {ans:<12} | {str(in_gt):<22} | {str(gm)}")
        
    print("\n--- TREATMENT PREDICTIONS FOR QA-13 (TARGET VIDEO L21_V005) ---")
    treat_target_preds = [p for p in q13_preds_treat if p.get("video_id") == target_vid]
    print(f"{'Rank':<6} | {'Frame ID':<10} | {'Answer':<12} | {'in_GT [23160, 23220]':<22} | {'Gold Match'}")
    print("-" * 75)
    for p in treat_target_preds:
        fid = int(p.get("frame_id", -1))
        in_gt = gt_s <= fid <= gt_e
        ans = p.get("answer", "")
        gm = (ans == gold_answer)
        print(f"{p.get('rank'):<6} | {fid:<10} | {ans:<12} | {str(in_gt):<22} | {str(gm)}")

    usable = q13_ev.get("usable_evidence_candidates", [])
    print(f"\n--- QA-13 USABLE EVIDENCE CANDIDATES FOR {target_vid} (Total usable: {len(usable)}) ---")
    print(f"{'Pos':<5} | {'Frame ID':<10} | {'Nomination Rank':<17} | {'Local Anchor Rank':<18} | {'Evidence Source'}")
    print("-" * 75)
    for idx, c in enumerate(usable, start=1):
        if c.get("video_id") == target_vid:
            print(f"{idx:<5} | {c.get('frame_id'):<10} | {c.get('video_nomination_rank'):<17} | {c.get('local_anchor_rank'):<18} | {c.get('evidence_source', 'KEYFRAME_BANK')}")

if __name__ == "__main__":
    audit_qa13_lineage()
