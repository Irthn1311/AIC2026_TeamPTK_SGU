# ==============================================================================================================
# QA-R2H1 KAGGLE RUNNER & STRICT EVALUATOR (A/B RUNNER)
# Control Arm   : QA-R2G1 Champion (total_seed_cap=16, count_far_alt_micro=OFF)
# Treatment Arm : QA-R2H1 (total_seed_cap=16, count_far_alt_micro=ON)
# ==============================================================================================================

from __future__ import annotations

import hashlib
import json
import os
import shutil
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

SYSTEM_DIR = REPO_DIR / "systems" / "system_tai"
BENCHMARK_PATH = SYSTEM_DIR / "benchmarks" / "l21_150_diagnostic" / "benchmark.json"
DEV_EN_SIDECAR_PATH = SYSTEM_DIR / "benchmarks" / "l21_150_diagnostic" / "qa_dev_translations_en.json"
ONTOLOGY_PATH = SYSTEM_DIR / "benchmarks" / "l21_150_diagnostic" / "qa_dev_visual_ontology.json"
MANIFEST_CACHE_PATH = Path("/kaggle/working/manifest_cache.json")
RUNNER_SCRIPT = SYSTEM_DIR / "scripts" / "l21_150_run_baseline.py"

CONTROL_DIR = Path("/kaggle/working/output/qa_r2h1_control_cap16")
TREATMENT_DIR = Path("/kaggle/working/output/qa_r2h1_treatment_count_far_alt")

# ==============================================================================================================
# STEP 0: SYSTEM ENVIRONMENT & PYTHON PACKAGES CHECK
# ==============================================================================================================
print("\n" + "=" * 110)
print("STEP 0: SYSTEM ENVIRONMENT & PYTHON PACKAGES CHECK")
print("=" * 110)

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
# STEP 1: VALIDATING FROZEN BENCHMARK & SIDECAR ARTIFACTS SHA256
# ==============================================================================================================
print("\n" + "=" * 110)
print("STEP 1: VALIDATING FROZEN BENCHMARK & SIDECAR ARTIFACTS SHA256")
print("=" * 110)

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

# ==============================================================================================================
# STEP 2: EXECUTION FUNCTION FOR EXPERIMENT ARMS
# ==============================================================================================================
def run_arm(count_far_alt_micro: bool, out_dir: Path) -> float:
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
        "--qa-temporal-refinement-total-seed-cap", "16",
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
    if count_far_alt_micro:
        cmd.append("--qa-count-far-alt-micro")

    desc = "QA-R2H1 Treatment (count_far_alt_micro=ON)" if count_far_alt_micro else "QA-R2G1 Control (count_far_alt_micro=OFF)"
    print(f"\nExecuting {desc} -> {out_dir.name}...")
    t0 = time.time()
    subprocess.run(cmd, check=True)
    t_elapsed = time.time() - t0
    print(f"Finished in {t_elapsed:.2f}s.")
    return t_elapsed

# ==============================================================================================================
# STEP 3: RUN CONTROL AND TREATMENT ARMS
# ==============================================================================================================
print("\n" + "=" * 110)
print("RUNNING EXPERIMENT ARMS (A/B)")
print("=" * 110)

t_ctrl = run_arm(count_far_alt_micro=False, out_dir=CONTROL_DIR)
t_treat = run_arm(count_far_alt_micro=True, out_dir=TREATMENT_DIR)

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
    for ef in ev_files:
        d = json.loads(ef.read_text(encoding="utf-8"))
        ev_by_qid[d["query_id"]] = d

    r1_hits, r5_hits, r20_hits, r50_hits, r100_hits = 0, 0, 0, 0, 0
    video_hits, answer_hits, full_hits, pred_producing_queries = 0, 0, 0, 0
    full_hit_details: dict[str, dict] = {}

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
        best_frame = None
        best_ans = None
        for p in preds:
            pf = int(p.get("frame_id", -1))
            if p.get("video_id") == target_vid and s <= pf <= e and normalize_text(p.get("answer", "")) in gold_answers:
                best_rank = int(p["rank"])
                best_frame = pf
                best_ans = p.get("answer")
                break

        if vid_hit: video_hits += 1
        if ans_hit: answer_hits += 1
        if best_rank is not None:
            full_hits += 1
            full_hit_details[qid] = {"rank": best_rank, "frame_id": best_frame, "answer": best_ans}
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
    }

print("\nEvaluating Control Arm...")
ctrl_eval = evaluate_run(CONTROL_DIR)
print("Evaluating Treatment Arm...")
treat_eval = evaluate_run(TREATMENT_DIR)

# ==============================================================================================================
# STEP 5: COMPARATIVE AUDIT & SCORE SUMMARY
# ==============================================================================================================
print("\n" + "=" * 110)
print("QA-R2H1 COMPARATIVE SUMMARY TABLE (CONTROL vs TREATMENT)")
print("=" * 110)
print(f"{'Metric':<25} | {'Control (R2G1)':<20} | {'Treatment (R2H1)':<20} | {'Delta'}")
print("-" * 80)
print(f"{'Execution Time':<25} | {t_ctrl:.2f}s{'':<15} | {t_treat:.2f}s{'':<15} | {t_treat - t_ctrl:+.2f}s")
print(f"{'Predictions-Producing':<25} | {ctrl_eval['pred_producing_queries']}/38{'':<16} | {treat_eval['pred_producing_queries']}/38{'':<16} | 0")
print(f"{'Video Matches':<25} | {ctrl_eval['video_hits']}/38{'':<16} | {treat_eval['video_hits']}/38{'':<16} | {treat_eval['video_hits'] - ctrl_eval['video_hits']:+d}")
print(f"{'Answer Matches':<25} | {ctrl_eval['answer_hits']}/38{'':<16} | {treat_eval['answer_hits']}/38{'':<16} | {treat_eval['answer_hits'] - ctrl_eval['answer_hits']:+d}")
print(f"{'Full Hits (R<=100)':<25} | {ctrl_eval['full_hits']}/38{'':<16} | {treat_eval['full_hits']}/38{'':<16} | {treat_eval['full_hits'] - ctrl_eval['full_hits']:+d}")
print(f"{'R@20':<25} | {ctrl_eval['r20_hits']}/38{'':<16} | {treat_eval['r20_hits']}/38{'':<16} | {treat_eval['r20_hits'] - ctrl_eval['r20_hits']:+d}")
print(f"{'R@50':<25} | {ctrl_eval['r50_hits']}/38{'':<16} | {treat_eval['r50_hits']}/38{'':<16} | {treat_eval['r50_hits'] - ctrl_eval['r50_hits']:+d}")
print(f"{'R@100':<25} | {ctrl_eval['r100_hits']}/38{'':<16} | {treat_eval['r100_hits']}/38{'':<16} | {treat_eval['r100_hits'] - ctrl_eval['r100_hits']:+d}")
print(f"{'Strict Score':<25} | {ctrl_eval['score']:.6f} ({int(ctrl_eval['score']*190)}/190){'':<4} | {treat_eval['score']:.6f} ({int(treat_eval['score']*190)}/190){'':<4} | {treat_eval['score'] - ctrl_eval['score']:+.6f}")

# ==============================================================================================================
# STEP 6: PROTECTED HITS REGRESSION AUDIT
# ==============================================================================================================
print("\n" + "=" * 110)
print("PROTECTED HITS REGRESSION AUDIT (ALL 6 PREVIOUS STRICT HITS)")
print("=" * 110)
PROTECTED_HITS = ["QA-13", "QA-08", "QA-27", "QA-46", "QA-10", "QA-45"]
print(f"{'Query ID':<10} | {'Control Hit Rank':<22} | {'Treatment Hit Rank':<22} | {'Status'}")
print("-" * 80)
for qid in PROTECTED_HITS:
    c_info = ctrl_eval["full_hit_details"].get(qid)
    t_info = treat_eval["full_hit_details"].get(qid)
    c_str = f"Rank {c_info['rank']} (f={c_info['frame_id']})" if c_info else "NONE"
    t_str = f"Rank {t_info['rank']} (f={t_info['frame_id']})" if t_info else "NONE"
    status = "PROTECTED ✅" if (t_info and t_info["rank"] <= 100) else "REGRESSION ❌"
    print(f"{qid:<10} | {c_str:<22} | {t_str:<22} | {status}")

# ==============================================================================================================
# STEP 7: TARGET QUERY AUDIT & CAUSAL TRACE: QA-20
# ==============================================================================================================
print("\n" + "=" * 110)
print("TARGET QUERY AUDIT & CAUSAL TRACE: QA-20 (L21_V007)")
print("=" * 110)

q20_c = ctrl_eval["full_hit_details"].get("QA-20")
q20_t = treat_eval["full_hit_details"].get("QA-20")
print(f"Control Arm   -> Hit: {q20_c}")
print(f"Treatment Arm -> Hit: {q20_t}")

ev_q20 = treat_eval["ev_by_qid"].get("QA-20", {})
usable_cands = ev_q20.get("usable_evidence_candidates", [])
print(f"\nQA-20 Usable Evidence Candidates (Total: {len(usable_cands)}):")
for idx, c in enumerate(usable_cands, start=1):
    print(f"  [{idx:02d}] Video: {c.get('video_id'):<10} Frame: {c.get('frame_id'):<8} NomRank: {str(c.get('video_nomination_rank')):<3} LocalRank: {str(c.get('local_anchor_rank')):<3} Answers: {c.get('answers', [])[:3]}")

q20_preds = treat_eval["preds_by_qid"].get("QA-20", [])
print("\n--- QA-20 Treatment Final Predictions (Ranks 75..100) ---")
for p in q20_preds:
    if int(p["rank"]) >= 75:
        is_hit = (p["video_id"] == "L21_V007" and 14610 <= int(p["frame_id"]) <= 14670 and normalize_text(p["answer"]) == "2")
        hit_mark = " <--- STRICT FULL HIT! 🎯" if is_hit else ""
        print(f"  Rank {p['rank']:<3}: video={p['video_id']}, frame={p['frame_id']}, answer='{p.get('answer')}'{hit_mark}")

# ==============================================================================================================
# STEP 8: ALL 38 DEV QUERIES FULL-HIT AUDIT & RANK SHIFTS
# ==============================================================================================================
print("\n" + "=" * 110)
print("ALL 38 DEV QUERIES FULL-HIT AUDIT & RANK SHIFTS")
print("=" * 110)
print(f"{'Query ID':<10} | {'Control Hit Rank':<18} | {'Treatment Hit Rank':<20} | {'Delta':<10} | {'Status'}")
print("-" * 75)
for qid in sorted(qa_dev_map.keys()):
    c_hr = ctrl_eval["full_hit_details"].get(qid, {}).get("rank")
    t_hr = treat_eval["full_hit_details"].get(qid, {}).get("rank")
    if c_hr is not None or t_hr is not None:
        delta_str = f"{t_hr - c_hr:+d}" if (c_hr is not None and t_hr is not None) else ("+NEW" if c_hr is None else "-LOST")
        if c_hr is None and t_hr is not None: status = "GAINED HIT 🎯"
        elif c_hr is not None and t_hr is None: status = "LOST HIT ❌"
        elif t_hr < c_hr: status = "IMPROVED 🚀"
        elif t_hr > c_hr: status = "REGRESSED ⚠️"
        else: status = "IDENTICAL ✅"
        print(f"{qid:<10} | Rank {str(c_hr):<13} | Rank {str(t_hr):<15} | {delta_str:<10} | {status}")
print("=" * 110)
