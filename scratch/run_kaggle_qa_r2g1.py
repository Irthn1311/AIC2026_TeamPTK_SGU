# ==============================================================================================================
# QA-R2G1 PRODUCTION A/B EXPERIMENT: SECONDARY REFINEMENT CAP COMPLETION (TOTAL_SEED_CAP: 8 -> 16)
# Target Branch: feat/system-tai-quality-q1b
# Exact Commit: 138d443 (or latest runner commit)
# Both Arms: --qa-temporal-seeds-per-video 2 --qa-temporal-refinement-video-cap 8
#            --qa-secondary-temporal-micro-budget --qa-primary-11-12-micro-coverage
#            --qa-tier3-primary-first --qa-tier3-negative-offset-first --qa-unsupported-provider-fallback
# Control:   --qa-temporal-refinement-total-seed-cap 8
# Treatment: --qa-temporal-refinement-total-seed-cap 16
# ==============================================================================================================

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

# Silence FFmpeg / OpenCV C-level decoding warnings
os.environ["OPENCV_LOG_LEVEL"] = "FATAL"
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"
os.environ["AV_LOG_FORCE_NOCOLOR"] = "1"

# ==============================================================================================================
# STEP 0: SYSTEM ENVIRONMENT & PYTHON PACKAGES CHECK
# ==============================================================================================================
print("\n" + "=" * 110)
print("STEP 0: SYSTEM ENVIRONMENT & PYTHON PACKAGES CHECK")
print("=" * 110)

def run_cmd(cmd: list[str], cwd: Path | None = None) -> str:
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, check=True)
    return res.stdout.strip()

print(f"Python interpreter: {sys.executable}")
print(f"Working directory:  {Path.cwd()}")

print("Checking and installing Python dependencies...")
subprocess.run(
    [
        sys.executable, "-m", "pip", "install", "-q",
        "openai-clip", "ftfy", "regex", "tqdm", "pytesseract"
    ],
    check=True,
)

print("Checking and installing system Tesseract OCR dependencies...")
subprocess.run(["apt-get", "update", "-qq"], check=False)
subprocess.run(
    [
        "apt-get", "install", "-y", "-qq",
        "tesseract-ocr",
        "tesseract-ocr-vie",
        "tesseract-ocr-eng",
        "libtesseract-dev",
    ],
    check=False,
)

# ==============================================================================================================
# STEP 1: GIT REPOSITORY SYNC TO LATEST QA-R2G1 COMMIT
# ==============================================================================================================
print("\n" + "=" * 110)
print("STEP 1: SYNCING CODEBASE")
print("=" * 110)

REPO_DIR = Path("/kaggle/working/AIC2026_TeamPTK_SGU")
REPO_URL = "https://github.com/Irthn1311/AIC2026_TeamPTK_SGU.git"

if not (REPO_DIR / ".git").exists():
    print(f"Cloning repository from {REPO_URL}...")
    run_cmd(["git", "clone", REPO_URL, str(REPO_DIR)])
else:
    print("Fetching latest commits...")
    run_cmd(["git", "fetch", "origin"], cwd=REPO_DIR)

run_cmd(["git", "checkout", "-f", "origin/feat/system-tai-quality-q1b"], cwd=REPO_DIR)

current_commit = run_cmd(["git", "rev-parse", "HEAD"], cwd=REPO_DIR)
print(f"Verified HEAD commit: {current_commit}")

src_path = str(REPO_DIR / "systems" / "system_tai" / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# ==============================================================================================================
# STEP 2: VERIFY BENCHMARK & SIDECAR ARTIFACTS SHA256
# ==============================================================================================================
print("\n" + "=" * 110)
print("STEP 2: VALIDATING FROZEN BENCHMARK & SIDECAR ARTIFACTS SHA256")
print("=" * 110)

SYSTEM_DIR = REPO_DIR / "systems" / "system_tai"
BENCHMARK_PATH = SYSTEM_DIR / "benchmarks" / "l21_150_diagnostic" / "benchmark.json"
DEV_EN_SIDECAR_PATH = SYSTEM_DIR / "benchmarks" / "l21_150_diagnostic" / "qa_dev_translations_en.json"
ONTOLOGY_PATH = SYSTEM_DIR / "benchmarks" / "l21_150_diagnostic" / "qa_dev_visual_ontology.json"
MANIFEST_CACHE_PATH = Path("/kaggle/working/manifest_cache.json")

EXPECTED_SHA = {
    "benchmark.json": "02f0bfc27053a9e532abb8c2cba9ead8f9923d7600993145c57b315f5e55ad1a",
    "qa_dev_translations_en.json": "45929059506de93aac574a6d2d5581691af81ae12405c18d57289485948c1f4d",
    "qa_dev_visual_ontology.json": "fc19f4ca1ce2e4960463ba054be2f9c351cf874867eb65a2ef9ce2252d644ddc",
}

for p, name in [
    (BENCHMARK_PATH, "benchmark.json"),
    (DEV_EN_SIDECAR_PATH, "qa_dev_translations_en.json"),
    (ONTOLOGY_PATH, "qa_dev_visual_ontology.json"),
]:
    if not p.exists():
        raise FileNotFoundError(f"Missing required artifact: {p}")
    actual_sha = hashlib.sha256(p.read_bytes()).hexdigest()
    expected = EXPECTED_SHA[name]
    if actual_sha != expected:
        raise ValueError(f"SHA256 mismatch for {name}:\n  expected: {expected}\n  actual:   {actual_sha}")
    print(f"  [PASS] {name:<30} SHA256: {actual_sha[:16]}...")

bm_data = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
qa_dev_queries = [
    q for q in bm_data.get("queries", [])
    if str(q.get("split", "")).upper() == "DEV" and str(q.get("task_type", q.get("task", ""))).lower() == "qa"
]
qa_dev_map = {q["query_id"]: q for q in qa_dev_queries}
assert len(qa_dev_queries) == 38, f"Expected 38 QA DEV queries, got {len(qa_dev_queries)}"

def normalize_text(t: str) -> str:
    if not t:
        return ""
    t = unicodedata.normalize("NFKC", str(t)).casefold()
    return "".join(c for c in t if c.isalnum() or c.isspace()).strip()

RUNNER_SCRIPT = SYSTEM_DIR / "scripts" / "l21_150_run_baseline.py"

def run_fast_arm(total_seed_cap: int, out_dir: Path) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(RUNNER_SCRIPT),
        "--benchmark", str(BENCHMARK_PATH),
        "--manifest-cache", str(MANIFEST_CACHE_PATH),
        "--input-root", "/kaggle/input",
        "--split", "dev",
        "--task", "qa",
        "--device", "auto",
        "--allow-model-download",
        "--output-dir", str(out_dir),
        "--top-k", "100",
        "--qa-video-conditioned-evidence",
        "--qa-keyframe-evidence-bank",
        "--qa-localization-language-policy", "en_only",
        "--qa-dev-en-sidecar", str(DEV_EN_SIDECAR_PATH),
        "--qa-multi-seed-temporal-refinement",
        "--qa-temporal-seeds-per-video", "2",
        "--qa-temporal-refinement-video-cap", "8",
        "--qa-temporal-refinement-total-seed-cap", str(total_seed_cap),
        "--qa-secondary-temporal-micro-budget",
        "--qa-primary-11-12-micro-coverage",
        "--qa-tier3-primary-first",
        "--qa-tier3-negative-offset-first",
        "--qa-visual-ontology", str(ONTOLOGY_PATH),
        "--qa-ocr-evidence",
        "--qa-ocr-languages", "eng+vie",
        "--qa-ocr-evidence-frame-budget", "8",
        "--qa-unsupported-provider-fallback",
    ]

    desc = f"total_seed_cap={total_seed_cap}"
    print(f"Executing Fast Arm ({desc}) -> {out_dir.name}...")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print("[ERROR] Runner stderr:")
        print(res.stderr)
        raise RuntimeError(f"Runner failed with exit code {res.returncode}")
    print(f"Run completed successfully for {out_dir.name}.")

# ==============================================================================================================
# STEP 3: EXECUTE CLEAN CONTROL (CAP=8) & TREATMENT (CAP=16)
# ==============================================================================================================
print("\n" + "=" * 110)
print("STEP 3: EXECUTING CLEAN CONTROL (TOTAL_SEED_CAP=8) & TREATMENT (TOTAL_SEED_CAP=16)")
print("=" * 110)

CONTROL_DIR = Path("/kaggle/working/output/qa_r2g1_control_cap8")
TREATMENT_DIR = Path("/kaggle/working/output/qa_r2g1_treatment_cap16")

print("Executing clean Control (total_seed_cap=8)...")
run_fast_arm(total_seed_cap=8, out_dir=CONTROL_DIR)

print("Executing clean Treatment (total_seed_cap=16)...")
run_fast_arm(total_seed_cap=16, out_dir=TREATMENT_DIR)

# ==============================================================================================================
# STEP 4: STRICT BENCHMARK EVALUATOR
# ==============================================================================================================
def evaluate_run(out_dir: Path) -> dict:
    pred_files = list(out_dir.glob("**/qa_predictions.jsonl"))
    ev_files = list(out_dir.glob("**/qa_evidence.json"))
    assert len(pred_files) == 38, f"Expected 38 prediction files in {out_dir}, got {len(pred_files)}"
    assert len(ev_files) == 38, f"Expected 38 evidence files in {out_dir}, got {len(ev_files)}"
    
    preds_by_qid: dict[str, list[dict]] = {}
    for pf in pred_files:
        for line in pf.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                preds_by_qid.setdefault(row["query_id"], []).append(row)
    for qid in preds_by_qid:
        preds_by_qid[qid].sort(key=lambda x: int(x["rank"]))
        
    ev_by_qid: dict[str, dict] = {}
    max_usable_cardinality = 0
    for ef in ev_files:
        d = json.loads(ef.read_text(encoding="utf-8"))
        qid = d["query_id"]
        ev_by_qid[qid] = d
        usable_cnt = len(d.get("usable_evidence_candidates", []))
        if usable_cnt > max_usable_cardinality:
            max_usable_cardinality = usable_cnt

    r1_hits, r5_hits, r20_hits, r50_hits, r100_hits = 0, 0, 0, 0, 0
    video_hits, answer_hits, full_hits, pred_producing_queries = 0, 0, 0, 0
    full_hit_details: dict[str, int] = {}
    
    for q in qa_dev_queries:
        qid = q["query_id"]
        target_vid = q["video_id"]
        s, e = map(int, q.get("proposed_interval", [0, 0]))
        raw_answers = q.get("accepted_answers") or [q.get("answer", "")]
        gold_answers = [normalize_text(a) for a in raw_answers if a]
        
        preds = preds_by_qid.get(qid, [])
        if len(preds) > 0:
            pred_producing_queries += 1
            
        vid_hit = any(p.get("video_id") == target_vid for p in preds)
        ans_hit = any(p.get("video_id") == target_vid and normalize_text(p.get("answer", "")) in gold_answers for p in preds)
        
        best_rank = None
        for p in preds:
            pf = int(p.get("frame_id", -1))
            if p.get("video_id") == target_vid and s <= pf <= e and normalize_text(p.get("answer", "")) in gold_answers:
                best_rank = int(p["rank"])
                break
                
        if vid_hit: video_hits += 1
        if ans_hit: answer_hits += 1
        if best_rank is not None:
            full_hits += 1
            full_hit_details[qid] = best_rank
            if best_rank <= 1: r1_hits += 1
            if best_rank <= 5: r5_hits += 1
            if best_rank <= 20: r20_hits += 1
            if best_rank <= 50: r50_hits += 1
            if best_rank <= 100: r100_hits += 1

    avg_r1 = r1_hits / 38.0
    avg_r5 = r5_hits / 38.0
    avg_r20 = r20_hits / 38.0
    avg_r50 = r50_hits / 38.0
    avg_r100 = r100_hits / 38.0
    score = (avg_r1 + avg_r5 + avg_r20 + avg_r50 + avg_r100) / 5.0
    
    return {
        "pred_producing_queries": pred_producing_queries,
        "video_hits": video_hits,
        "answer_hits": answer_hits,
        "full_hits": full_hits,
        "r1_hits": r1_hits,
        "r5_hits": r5_hits,
        "r20_hits": r20_hits,
        "r50_hits": r50_hits,
        "r100_hits": r100_hits,
        "avg_r1": avg_r1,
        "avg_r5": avg_r5,
        "avg_r20": avg_r20,
        "avg_r50": avg_r50,
        "avg_r100": avg_r100,
        "score": score,
        "full_hit_details": full_hit_details,
        "preds_by_qid": preds_by_qid,
        "ev_by_qid": ev_by_qid,
        "max_usable_cardinality": max_usable_cardinality,
    }

print("Evaluating Control (total_seed_cap=8)...")
ctrl_eval = evaluate_run(CONTROL_DIR)
print("Evaluating Treatment (total_seed_cap=16)...")
treat_eval = evaluate_run(TREATMENT_DIR)

# ==============================================================================================================
# STEP 5: QA-10 REFINEMENT & PREDICTIONS DEEP FORENSIC
# ==============================================================================================================
print("\n" + "=" * 110)
print("STEP 5: QA-10 REFINEMENT & PREDICTIONS DEEP FORENSIC")
print("=" * 110)

q10_gt = qa_dev_map["QA-10"]
t_vid = q10_gt["video_id"]
s_gt, e_gt = map(int, q10_gt.get("proposed_interval", [0, 0]))
gold_answers = {
    normalize_text(a)
    for a in (q10_gt.get("accepted_answers") or [q10_gt.get("answer", "")])
    if a
}

ev_ctrl = ctrl_eval["ev_by_qid"].get("QA-10", {})
ev_treat = treat_eval["ev_by_qid"].get("QA-10", {})

print(f"Target Video: {t_vid} | GT Interval: [{s_gt}, {e_gt}] | Gold Answers: {gold_answers}")
print(f"\n--- Refinement Execution Diagnostics for QA-10 ---")
print(f"Control   (cap=8)  -> Total Seeds Sent: {ev_ctrl.get('temporal_refinement_seed_count', 'N/A')} | Refinement Successes: {ev_ctrl.get('temporal_refinement_success_count', 'N/A')}")
print(f"Treatment (cap=16) -> Total Seeds Sent: {ev_treat.get('temporal_refinement_seed_count', 'N/A')} | Refinement Successes: {ev_treat.get('temporal_refinement_success_count', 'N/A')}")

# Check QA-10 secondary anchor 28185 refinement status in Treatment
ref_selected = ev_treat.get("refinement_selected_candidates", [])
ref_success = ev_treat.get("refinement_success_candidates", [])
q10_28185_selected = [c for c in ref_selected if c.get("video_id") == t_vid and c.get("frame_id") == 28185]
q10_28185_success = [c for c in ref_success if c.get("video_id") == t_vid and c.get("candidate_frame_id") == 28185]

print(f"\nSecondary Anchor 28185 Refinement Status in Treatment:")
print(f"  Was 28185 sent to refiner?        : {len(q10_28185_selected) > 0}")
if q10_28185_success:
    print(f"  Refined Frame ID                  : {q10_28185_success[0].get('frame_id')}")
    print(f"  Refinement Status                 : {q10_28185_success[0].get('status')}")
else:
    print(f"  Refinement Success Record         : None (Status: {q10_28185_selected[0].get('status') if q10_28185_selected else 'NOT_SENT'})")

# Print Usable Evidence for QA-10 in Treatment
usable_treat = ev_treat.get("usable_evidence_candidates", [])
t_usable_treat = [c for c in usable_treat if c.get("video_id") == t_vid]
print(f"\nTreatment Usable Evidence Candidates for {t_vid} (Total in usable: {len(t_usable_treat)}/{len(usable_treat)}):")
print(f"{'Pos':<5} | {'Frame ID':<10} | {'Nom Rank':<10} | {'Local Rank':<12} | {'Ev Source':<18} | {'Answers'}")
print("-" * 85)
for idx, c in enumerate(t_usable_treat, start=1):
    print(f"{idx:<5} | {c.get('frame_id'):<10} | {str(c.get('video_nomination_rank')):<10} | {str(c.get('local_anchor_rank')):<12} | {str(c.get('evidence_source')):<18} | {c.get('answers', [])[:3]}")

# Print Treatment Target Video Predictions List for QA-10
preds_treat = treat_eval["preds_by_qid"].get("QA-10", [])
t_preds_treat = [p for p in preds_treat if p.get("video_id") == t_vid]
print(f"\nTreatment {t_vid} Final Predictions List for QA-10 (Total: {len(t_preds_treat)}):")
print(f"{'Rank':<6} | {'Frame ID':<10} | {'Answer':<16} | {'Answer Match':<14} | {'in_GT Interval':<16} | {'Signed Distance'}")
print("-" * 85)
for p in t_preds_treat:
    fid = int(p.get("frame_id", -1))
    in_int = (s_gt <= fid <= e_gt)
    ans = p.get("answer", "")
    is_gold = normalize_text(ans) in gold_answers
    dist = fid - s_gt if fid < s_gt else (fid - e_gt if fid > e_gt else 0)
    print(f"{p.get('rank'):<6} | {fid:<10} | {ans:<16} | {str(is_gold):<14} | {str(in_int):<16} | {dist:+d} frames")

# ==============================================================================================================
# STEP 6: QUERY-LEVEL REGRESSION DIFF & FULL A/B METRICS TABLE
# ==============================================================================================================
print("\n" + "=" * 110)
print("STEP 6: QUERY-LEVEL A/B REGRESSION DIFF & STRICT BENCHMARK METRICS")
print("=" * 110)

ctrl_hits = ctrl_eval["full_hit_details"]
treat_hits = treat_eval["full_hit_details"]

gained_hits = sorted(set(treat_hits.keys()) - set(ctrl_hits.keys()))
lost_hits = sorted(set(ctrl_hits.keys()) - set(treat_hits.keys()))
common_hits = sorted(set(ctrl_hits.keys()) & set(treat_hits.keys()))

rank_improved = [(qid, ctrl_hits[qid], treat_hits[qid]) for qid in common_hits if treat_hits[qid] < ctrl_hits[qid]]
rank_regressed = [(qid, ctrl_hits[qid], treat_hits[qid]) for qid in common_hits if treat_hits[qid] > ctrl_hits[qid]]
rank_neutral = [(qid, ctrl_hits[qid], treat_hits[qid]) for qid in common_hits if treat_hits[qid] == ctrl_hits[qid]]

print(f"Control Full Hits ({len(ctrl_hits)}): {sorted(ctrl_hits.items())}")
print(f"Treatment Full Hits ({len(treat_hits)}): {sorted(treat_hits.items())}")
print(f"  GAINED FULL HITS  : {gained_hits if gained_hits else 'None'}")
print(f"  LOST FULL HITS    : {lost_hits if lost_hits else 'None'}")
print(f"  RANK IMPROVED     : {rank_improved if rank_improved else 'None'}")
print(f"  RANK REGRESSED    : {rank_regressed if rank_regressed else 'None'}")
print(f"  RANK UNCHANGED    : {rank_neutral if rank_neutral else 'None'}")
print(f"\nMax Usable Cardinality : Control={ctrl_eval['max_usable_cardinality']} | Treatment={treat_eval['max_usable_cardinality']} (Limit <= 100 PASS: {treat_eval['max_usable_cardinality'] <= 100})")

# Verify Control matches established QA-R2F3 winner reference
r2f3_ref_hits = {"QA-13": 19, "QA-08": 43, "QA-27": 49, "QA-46": 78, "QA-45": 96}
ctrl_match_ref = (ctrl_hits == r2f3_ref_hits)
print(f"\nControl Baseline Verification vs R2F3 Reference ({r2f3_ref_hits}):")
print(f"  Matches Exact Reference Hits & Ranks: {ctrl_match_ref}")
if not ctrl_match_ref:
    print(f"  WARNING: Control differs from reference! Control: {ctrl_hits}")

metrics_rows = [
    (
        "Predictions-Producing",
        f"{ctrl_eval['pred_producing_queries']}/38 ({ctrl_eval['pred_producing_queries']/38*100:.1f}%)",
        f"{treat_eval['pred_producing_queries']}/38 ({treat_eval['pred_producing_queries']/38*100:.1f}%)",
        f"{treat_eval['pred_producing_queries'] - ctrl_eval['pred_producing_queries']:+d}",
    ),
    (
        "Video Match",
        f"{ctrl_eval['video_hits']}/38 ({ctrl_eval['video_hits']/38*100:.1f}%)",
        f"{treat_eval['video_hits']}/38 ({treat_eval['video_hits']/38*100:.1f}%)",
        f"{treat_eval['video_hits'] - ctrl_eval['video_hits']:+d}",
    ),
    (
        "Answer Match",
        f"{ctrl_eval['answer_hits']}/38 ({ctrl_eval['answer_hits']/38*100:.1f}%)",
        f"{treat_eval['answer_hits']}/38 ({treat_eval['answer_hits']/38*100:.1f}%)",
        f"{treat_eval['answer_hits'] - ctrl_eval['answer_hits']:+d}",
    ),
    (
        "Full QA Tuple Hit",
        f"{ctrl_eval['full_hits']}/38 ({ctrl_eval['full_hits']/38*100:.1f}%)",
        f"{treat_eval['full_hits']}/38 ({treat_eval['full_hits']/38*100:.1f}%)",
        f"{treat_eval['full_hits'] - ctrl_eval['full_hits']:+d}",
    ),
    (
        "R@1",
        f"{ctrl_eval['r1_hits']}/38 ({ctrl_eval['avg_r1']:.6f})",
        f"{treat_eval['r1_hits']}/38 ({treat_eval['avg_r1']:.6f})",
        f"{treat_eval['r1_hits'] - ctrl_eval['r1_hits']:+d}",
    ),
    (
        "R@5",
        f"{ctrl_eval['r5_hits']}/38 ({ctrl_eval['avg_r5']:.6f})",
        f"{treat_eval['r5_hits']}/38 ({treat_eval['avg_r5']:.6f})",
        f"{treat_eval['r5_hits'] - ctrl_eval['r5_hits']:+d}",
    ),
    (
        "R@20",
        f"{ctrl_eval['r20_hits']}/38 ({ctrl_eval['avg_r20']:.6f})",
        f"{treat_eval['r20_hits']}/38 ({treat_eval['avg_r20']:.6f})",
        f"{treat_eval['r20_hits'] - ctrl_eval['r20_hits']:+d}",
    ),
    (
        "R@50",
        f"{ctrl_eval['r50_hits']}/38 ({ctrl_eval['avg_r50']:.6f})",
        f"{treat_eval['r50_hits']}/38 ({treat_eval['avg_r50']:.6f})",
        f"{treat_eval['r50_hits'] - ctrl_eval['r50_hits']:+d}",
    ),
    (
        "R@100",
        f"{ctrl_eval['r100_hits']}/38 ({ctrl_eval['avg_r100']:.6f})",
        f"{treat_eval['r100_hits']}/38 ({treat_eval['avg_r100']:.6f})",
        f"{treat_eval['r100_hits'] - ctrl_eval['r100_hits']:+d}",
    ),
    (
        "Strict Final Score",
        f"{ctrl_eval['score']:.6f}",
        f"{treat_eval['score']:.6f}",
        f"{treat_eval['score'] - ctrl_eval['score']:+.6f}",
    ),
]

print("\n" + "-" * 110)
print(f"{'Metric':<30} | {'Control (CAP=8)':<33} | {'Treatment (CAP=16)':<33} | {'Delta'}")
print("-" * 110)
for name, c_val, t_val, diff in metrics_rows:
    print(f"{name:<30} | {c_val:<33} | {t_val:<33} | {diff}")
print("=" * 110)

# ==============================================================================================================
# STEP 7: FINAL ACCEPTANCE CLASSIFICATION
# ==============================================================================================================
print("\n--- QA-R2G1 FINAL ACCEPTANCE CLASSIFICATION ---")
if len(q10_28185_selected) > 0:
    print(f"  [REFINEMENT COVERAGE PASS] QA-10 secondary anchor 28185 was successfully sent to refiner!")
else:
    print(f"  [REFINEMENT COVERAGE FAIL] QA-10 secondary anchor 28185 was NOT sent to refiner.")

q10_full_treat = [
    p for p in treat_eval["preds_by_qid"].get("QA-10", [])
    if p.get("video_id") == t_vid
    and s_gt <= int(p.get("frame_id", -1)) <= e_gt
    and normalize_text(p.get("answer", "")) in gold_answers
]

if q10_full_treat:
    best_q10 = min(q10_full_treat, key=lambda p: int(p["rank"]))
    q10_rank = int(best_q10["rank"])
    print(f"  [QA10 STRICT PASS] QA-10 full tuple achieved at rank {q10_rank}: frame={best_q10['frame_id']}, answer='{best_q10.get('answer', '')}'.")
else:
    print(f"  [QA10 STRICT NEUTRAL/FAIL] QA-10 did not produce a strict full tuple.")

score_diff = treat_eval["score"] - ctrl_eval["score"]
if score_diff > 1e-9:
    print(f"  [SCORE PASS] Strict score improved from {ctrl_eval['score']:.6f} to {treat_eval['score']:.6f} (+{score_diff:.6f})! (NEW CHAMPION)")
elif abs(score_diff) <= 1e-9:
    print(f"  [SCORE NEUTRAL] Aggregate strict score identical to Control ({ctrl_eval['score']:.6f}).")
else:
    print(f"  [SCORE REGRESSION] Strict score decreased from {ctrl_eval['score']:.6f} to {treat_eval['score']:.6f} ({score_diff:.6f}).")

if not lost_hits and not rank_regressed and q10_full_treat:
    print("  [KEEP] QA-10 gained with NO lost hits and NO rank regressions!")
print("=" * 110)
