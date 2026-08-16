# ==============================================================================================================
# QA-R2H1 KAGGLE RUNNER & STRICT EVALUATOR (A/B RUNNER)
# Control Arm   : QA-R2G1 Champion (total_seed_cap=16, count_far_alt_micro=False)
# Treatment Arm : QA-R2H1 (total_seed_cap=16, count_far_alt_micro=True)
# ==============================================================================================================

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

# Fix stdout encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

print("=" * 110)
print("QA-R2H1 STRICT A/B COMPARISON ON KAGGLE")
print("=" * 110)

REPO_DIR = Path("/kaggle/working/AIC2026_TeamPTK_SGU")
if not REPO_DIR.exists():
    REPO_DIR = Path(".")

BENCHMARK_PATH = REPO_DIR / "systems" / "system_tai" / "benchmarks" / "l21_150_diagnostic" / "benchmark.json"
DEV_EN_SIDECAR_PATH = REPO_DIR / "systems" / "system_tai" / "benchmarks" / "l21_150_diagnostic" / "qa_dev_translations_en.json"
VISUAL_ONTOLOGY_PATH = REPO_DIR / "systems" / "system_tai" / "benchmarks" / "l21_150_diagnostic" / "qa_dev_visual_ontology.json"

CONTROL_OUT_DIR = Path("/kaggle/working/output/qa_r2h1_control_cap16")
TREATMENT_OUT_DIR = Path("/kaggle/working/output/qa_r2h1_treatment_count_far_alt")

CONTROL_OUT_DIR.mkdir(parents=True, exist_ok=True)
TREATMENT_OUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------------------------------------------
# STEP 1: BASE COMMAND DEFINITION
# --------------------------------------------------------------------------------------------------------------
BASE_CMD = [
    sys.executable,
    str(REPO_DIR / "systems" / "system_tai" / "scripts" / "l21_150_run_baseline.py"),
    "--split", "dev",
    "--task", "qa",
    "--benchmark", str(BENCHMARK_PATH),
    "--qa-video-conditioned-evidence",
    "--qa-grounding-candidate-ordering", "fused_temporal_first",
    "--qa-localization-language-policy", "en_only",
    "--qa-dev-en-sidecar", str(DEV_EN_SIDECAR_PATH),
    "--qa-keyframe-evidence-bank",
    "--qa-keyframe-evidence-video-cap", "8",
    "--qa-keyframe-evidence-anchors-per-video", "3",
    "--qa-multi-seed-temporal-refinement",
    "--qa-temporal-seeds-per-video", "2",
    "--qa-temporal-refinement-video-cap", "8",
    "--qa-temporal-refinement-total-seed-cap", "16",
    "--qa-secondary-temporal-micro-budget",
    "--qa-primary-11-12-micro-coverage",
    "--qa-tier3-primary-first",
    "--qa-tier3-negative-offset-first",
    "--qa-unsupported-provider-fallback",
    "--qa-visual-ontology", str(VISUAL_ONTOLOGY_PATH),
]

# --------------------------------------------------------------------------------------------------------------
# STEP 2: EXECUTE CONTROL ARM
# --------------------------------------------------------------------------------------------------------------
print("\n" + "=" * 110)
print("RUNNING CONTROL ARM: QA-R2G1 (total_seed_cap=16, count_far_alt_micro=OFF)")
print("=" * 110)
t0_ctl = time.time()
ctl_cmd = list(BASE_CMD) + ["--output-dir", str(CONTROL_OUT_DIR)]
subprocess.run(ctl_cmd, check=True)
t_ctl = time.time() - t0_ctl
print(f"Control Arm finished in {t_ctl:.2f}s.")

# --------------------------------------------------------------------------------------------------------------
# STEP 3: EXECUTE TREATMENT ARM
# --------------------------------------------------------------------------------------------------------------
print("\n" + "=" * 110)
print("RUNNING TREATMENT ARM: QA-R2H1 (total_seed_cap=16, count_far_alt_micro=ON)")
print("=" * 110)
t0_trt = time.time()
trt_cmd = list(BASE_CMD) + ["--qa-count-far-alt-micro", "--output-dir", str(TREATMENT_OUT_DIR)]
subprocess.run(trt_cmd, check=True)
t_trt = time.time() - t0_trt
print(f"Treatment Arm finished in {t_trt:.2f}s.")

# --------------------------------------------------------------------------------------------------------------
# STEP 4: STRICT EVALUATION & COMPARATIVE AUDIT
# --------------------------------------------------------------------------------------------------------------
print("\n" + "=" * 110)
print("STRICT EVALUATION & COMPARATIVE AUDIT (CONTROL vs TREATMENT)")
print("=" * 110)

def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text)).casefold()
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = "".join(ch if ch.isalnum() else " " for ch in text)
    return " ".join(text.split())

bm_data = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
dev_queries = [
    q for q in bm_data.get("queries", [])
    if str(q.get("split", "")).upper() == "DEV" and str(q.get("task_type", q.get("task", ""))).lower() == "qa"
]
qa_dev_map = {q["query_id"]: q for q in dev_queries}

def evaluate_predictions(preds_path: Path):
    if not preds_path.exists():
        return {}
    preds_by_query = {}
    with open(preds_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            row = json.loads(line)
            preds_by_query.setdefault(row["query_id"], []).append(row)

    eval_res = {}
    for qid, qdata in qa_dev_map.items():
        q_preds = preds_by_query.get(qid, [])
        t_vid = qdata.get("video_id")
        gt_s, gt_e = map(int, qdata.get("proposed_interval", [0, 0]))
        gold_ans = set(normalize_text(a) for a in qdata.get("accepted_answers", []))
        if qdata.get("answer"):
            gold_ans.add(normalize_text(qdata["answer"]))

        hit_rank = None
        hit_frame = None
        hit_ans = None
        vid_match = False
        ans_match = False

        for p in q_preds:
            r = p["rank"]
            v = p["video_id"]
            f_id = p["frame_id"]
            ans_norm = normalize_text(p.get("answer", ""))

            if v == t_vid:
                vid_match = True
            if ans_norm in gold_ans:
                ans_match = True

            if hit_rank is None and v == t_vid and ans_norm in gold_ans and (gt_s <= f_id <= gt_e):
                hit_rank = r
                hit_frame = f_id
                hit_ans = p.get("answer")

        eval_res[qid] = {
            "query_id": qid,
            "has_preds": len(q_preds) > 0,
            "pred_count": len(q_preds),
            "video_match": vid_match,
            "answer_match": ans_match,
            "hit_rank": hit_rank,
            "hit_frame": hit_frame,
            "hit_answer": hit_ans,
            "predictions": q_preds,
        }
    return eval_res

ctl_preds_path = CONTROL_OUT_DIR / "qa_predictions.jsonl"
trt_preds_path = TREATMENT_OUT_DIR / "qa_predictions.jsonl"

ctl_eval = evaluate_predictions(ctl_preds_path)
trt_eval = evaluate_predictions(trt_preds_path)

def calc_score(eval_res):
    n_queries = len(qa_dev_map)
    weights = {1: 1.0, 5: 1.0, 20: 1.0, 50: 1.0, 100: 1.0}
    r_counts = {1: 0, 5: 0, 20: 0, 50: 0, 100: 0}
    for qid, res in eval_res.items():
        hr = res["hit_rank"]
        if hr is not None:
            for k in [1, 5, 20, 50, 100]:
                if hr <= k:
                    r_counts[k] += 1
    weighted_sum = sum(r_counts[k] * weights[k] for k in weights)
    total_possible = n_queries * sum(weights.values())
    score = weighted_sum / total_possible
    return score, r_counts

ctl_score, ctl_rc = calc_score(ctl_eval)
trt_score, trt_rc = calc_score(trt_eval)

print(f"\n{'Metric':<25} | {'Control (R2G1)':<20} | {'Treatment (R2H1)':<20} | {'Delta'}")
print("-" * 80)
print(f"{'Execution Time':<25} | {t_ctl:.2f}s{'':<15} | {t_trt:.2f}s{'':<15} | {t_trt - t_ctl:+.2f}s")
print(f"{'Predictions-Producing':<25} | {sum(1 for r in ctl_eval.values() if r['has_preds'])}/38{'':<16} | {sum(1 for r in trt_eval.values() if r['has_preds'])}/38{'':<16} | 0")
print(f"{'Video Matches':<25} | {sum(1 for r in ctl_eval.values() if r['video_match'])}/38{'':<16} | {sum(1 for r in trt_eval.values() if r['video_match'])}/38{'':<16} | {sum(1 for r in trt_eval.values() if r['video_match']) - sum(1 for r in ctl_eval.values() if r['video_match']):+d}")
print(f"{'Answer Matches':<25} | {sum(1 for r in ctl_eval.values() if r['answer_match'])}/38{'':<16} | {sum(1 for r in trt_eval.values() if r['answer_match'])}/38{'':<16} | {sum(1 for r in trt_eval.values() if r['answer_match']) - sum(1 for r in ctl_eval.values() if r['answer_match']):+d}")
print(f"{'Full Hits (R<=100)':<25} | {ctl_rc[100]}/38{'':<16} | {trt_rc[100]}/38{'':<16} | {trt_rc[100] - ctl_rc[100]:+d}")
print(f"{'R@20':<25} | {ctl_rc[20]}/38{'':<16} | {trt_rc[20]}/38{'':<16} | {trt_rc[20] - ctl_rc[20]:+d}")
print(f"{'R@50':<25} | {ctl_rc[50]}/38{'':<16} | {trt_rc[50]}/38{'':<16} | {trt_rc[50] - ctl_rc[50]:+d}")
print(f"{'R@100':<25} | {ctl_rc[100]}/38{'':<16} | {trt_rc[100]}/38{'':<16} | {trt_rc[100] - ctl_rc[100]:+d}")
print(f"{'Strict Score':<25} | {ctl_score:.6f} ({int(ctl_score*190)}/190){'':<4} | {trt_score:.6f} ({int(trt_score*190)}/190){'':<4} | {trt_score - ctl_score:+.6f}")

print("\n" + "=" * 110)
print("PROTECTED HITS REGRESSION AUDIT (ALL 6 PREVIOUS STRICT HITS)")
print("=" * 110)
PROTECTED_HITS = ["QA-13", "QA-08", "QA-27", "QA-46", "QA-10", "QA-45"]
print(f"{'Query ID':<10} | {'Control Hit Rank':<18} | {'Treatment Hit Rank':<20} | {'Status'}")
print("-" * 75)
for qid in PROTECTED_HITS:
    c_hr = ctl_eval.get(qid, {}).get("hit_rank")
    t_hr = trt_eval.get(qid, {}).get("hit_rank")
    c_f = ctl_eval.get(qid, {}).get("hit_frame")
    t_f = trt_eval.get(qid, {}).get("hit_frame")
    status = "PROTECTED ✅" if t_hr is not None and t_hr <= 100 and (c_hr is None or t_hr <= c_hr or t_hr <= 100) else "REGRESSION ❌"
    print(f"{qid:<10} | Rank {str(c_hr):<5} (Frame {str(c_f)}) | Rank {str(t_hr):<5} (Frame {str(t_f)})  | {status}")

print("\n" + "=" * 110)
print("TARGET QUERY AUDIT: QA-20")
print("=" * 110)
q20_c = ctl_eval.get("QA-20", {})
q20_t = trt_eval.get("QA-20", {})
print(f"Control Arm   -> Hit Rank: {q20_c.get('hit_rank')} | Hit Frame: {q20_c.get('hit_frame')} | Hit Answer: {q20_c.get('hit_answer')}")
print(f"Treatment Arm -> Hit Rank: {q20_t.get('hit_rank')} | Hit Frame: {q20_t.get('hit_frame')} | Hit Answer: {q20_t.get('hit_answer')}")

# Print Top-10 predictions for QA-20 in Treatment
print("\n--- QA-20 Treatment Final Predictions (Ranks 80..100) ---")
q20_preds = q20_t.get("predictions", [])
for p in q20_preds:
    if p["rank"] >= 80:
        print(f"  Rank {p['rank']:<3}: video={p['video_id']}, frame={p['frame_id']}, answer='{p.get('answer')}'")

print("=" * 110)
