#!/usr/bin/env python3
"""KIS P1A2b: HEAD_TAIL_77 Canonical Subset A/B Benchmark Runner.

Strict Scope:
  1. PART 1: 5 Over-Budget BTC Queries Canonical A/B (p1-5, p1-11, p1-12, p1-17, p1-20)
     - Arm A: prefix_77 (Current P0 baseline)
     - Arm B: head_tail_77 (Experimental bifurcated packing)
     - Evaluates Top10 candidates, Top3 thumbnail comparisons, latency, and regression guards (p1-5, p1-11, p1-20) vs rescue probes (p1-12, p1-17).
     - Generates Side-by-Side visual comparison gallery.
  2. PART 2: Full-38 DEV Benchmark Census & Over-Budget Subset Metric Delta Evaluation
     - Censuses all 38 DEV queries for raw Marian tokens > 77.
     - Runs Arm A vs Arm B on the over-budget DEV subset against official kis_dev_gt.json.
     - Reports exact strict hit and score deltas.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

# Purge stale system_tai modules
for mod in list(sys.modules.keys()):
    if mod.startswith("system_tai"):
        del sys.modules[mod]

REPO_ROOT = Path(__file__).resolve().parents[1]
SYSTEM_TAI_SRC = REPO_ROOT / "systems" / "system_tai" / "src"
if str(SYSTEM_TAI_SRC) not in sys.path:
    sys.path.insert(0, str(SYSTEM_TAI_SRC))

try:
    import clip
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "openai-clip", "ftfy", "regex", "opencv-python-headless"], check=False)
    import clip

import cv2
from system_tai.kis.session_engine import OperationalKISRuntime
from system_tai.kis.session_schema import QueryRequest, SessionConfig
from system_tai.translation.provider import MarianOfflineTranslator, TokenBudgetGuard

FROZEN_KIS_DEV_GT_SHA256 = "7d25708b7243ca2b9964bad9a2b65b63354acd74eddb100167f49e1166f8e5b2"
OFFICIAL_K = (1, 5, 20, 50, 100)

BTC_OVER_BUDGET_QUERIES = [
    {
        "qid": "query-p1-5-kis",
        "name": "Hai người phụ nữ cho dê ăn trong trại (Regression Guard)",
        "file": "query-p1-5-kis.txt",
        "role": "REGRESSION_GUARD",
    },
    {
        "qid": "query-p1-11-kis",
        "name": "Đổ bóng tạo hình chân dung người đàn ông mặc vest (Regression Guard)",
        "file": "query-p1-11-kis.txt",
        "role": "REGRESSION_GUARD",
    },
    {
        "qid": "query-p1-12-kis",
        "name": "Trang trí bánh rán, rưới chocolate dâu tây chuối (Rescue Probe)",
        "file": "query-p1-12-kis.txt",
        "role": "RESCUE_PROBE",
    },
    {
        "qid": "query-p1-17-kis",
        "name": "Trao quà từ thiện tại bệnh viện cho 4 em nhỏ nhận biển COVID-19 (Rescue Probe)",
        "file": "query-p1-17-kis.txt",
        "role": "RESCUE_PROBE",
    },
    {
        "qid": "query-p1-20-kis",
        "name": "Đặt thêm 2 ly panna cotta, trang trí hoa ăn được (Regression Guard)",
        "file": "query-p1-20-kis.txt",
        "role": "REGRESSION_GUARD",
    },
]

VIDEO_PATH_CACHE: dict[str, Path] = {}


def populate_video_index_once() -> None:
    if VIDEO_PATH_CACHE:
        return
    for search_root in [Path("/kaggle/input"), REPO_ROOT / "systems" / "system_tai" / "data"]:
        if not search_root.exists():
            continue
        for root_dir, _, files in os.walk(str(search_root)):
            for fname in files:
                if fname.endswith(".mp4"):
                    vid = fname[:-4]
                    if vid not in VIDEO_PATH_CACHE:
                        VIDEO_PATH_CACHE[vid] = Path(root_dir) / fname


def resolve_video_path(video_id: str, raw_video_registry: Any = None) -> Path | None:
    if raw_video_registry:
        try:
            rec = raw_video_registry.get(video_id)
            if rec and rec.raw_video_path and rec.raw_video_path.exists():
                return rec.raw_video_path
        except Exception:
            pass
    if video_id in VIDEO_PATH_CACHE:
        return VIDEO_PATH_CACHE[video_id]
    populate_video_index_once()
    return VIDEO_PATH_CACHE.get(video_id)


def extract_thumbnail_base64(video_id: str, frame_id: int, raw_video_registry: Any = None) -> str:
    vpath = resolve_video_path(video_id, raw_video_registry)
    if not vpath or not vpath.exists():
        return ""
    try:
        cap = cv2.VideoCapture(str(vpath))
        if not cap.isOpened():
            return ""
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_id))
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            return ""
        h, w = frame.shape[:2]
        new_w = 240
        new_h = int(h * (new_w / w))
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        _, buf = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        return base64.b64encode(buf).decode("utf-8")
    except Exception:
        return ""


def get_reuse_manifest() -> Path | None:
    for p in [
        Path("/kaggle/working/manifest_cache.json"),
        Path("/kaggle/input/system-tai-manifest/feature_manifest.json"),
        Path("/kaggle/input/datasets/manifest_cache.json"),
        Path("/kaggle/input/manifest_cache.json"),
    ]:
        if p.exists() and p.stat().st_size > 1000:
            return p
    return None


def run_btc_5query_ab_benchmark() -> list[dict[str, Any]]:
    print("=" * 150, flush=True)
    print("PART 1: CANONICAL 5-QUERY BTC OVER-BUDGET SUBSET A/B BENCHMARK", flush=True)
    print("=" * 150, flush=True)

    yaml_path = REPO_ROOT / "systems" / "system_tai" / "configs" / "production.yaml"
    input_root = Path("/kaggle/input/datasets") if Path("/kaggle/input/datasets").exists() else Path("/kaggle/input")
    reuse_manifest = get_reuse_manifest()

    # 1. Bootstrap Arm A (PREFIX_77 Baseline)
    print("\n[1/2] Bootstrapping OperationalKISRuntime for ARM A (PREFIX_77 P0 Baseline)...", flush=True)
    out_a = Path("/kaggle/working/output/kis_p1a2b_arm_a") if Path("/kaggle/working").exists() else REPO_ROOT / "scratch" / "kis_p1a2b_arm_a"
    cfg_a = SessionConfig.from_yaml(
        yaml_path,
        input_root=input_root,
        output_root=out_a,
        reuse_manifest=reuse_manifest,
        translation_packing_policy="prefix_77",
    )
    t0_a = time.time()
    runtime_a = OperationalKISRuntime.bootstrap(cfg_a)
    print(f"      ARM A Bootstrap Complete in {time.time() - t0_a:.2f}s", flush=True)

    # 2. Bootstrap Arm B (HEAD_TAIL_77 Experimental)
    print("\n[2/2] Bootstrapping OperationalKISRuntime for ARM B (HEAD_TAIL_77 Experimental)...", flush=True)
    out_b = Path("/kaggle/working/output/kis_p1a2b_arm_b") if Path("/kaggle/working").exists() else REPO_ROOT / "scratch" / "kis_p1a2b_arm_b"
    cfg_b = SessionConfig.from_yaml(
        yaml_path,
        input_root=input_root,
        output_root=out_b,
        reuse_manifest=reuse_manifest,
        translation_packing_policy="head_tail_77",
    )
    t0_b = time.time()
    runtime_b = OperationalKISRuntime.bootstrap(cfg_b)
    print(f"      ARM B Bootstrap Complete in {time.time() - t0_b:.2f}s", flush=True)

    thunghiem_dir = REPO_ROOT / "systems" / "system_tai" / "THUNGHIEM_20-8"
    results: list[dict[str, Any]] = []

    print("\n" + "=" * 150, flush=True)
    print("EXECUTING A/B RUNTIME COMPARISON ACROSS 5 BTC OVER-BUDGET QUERIES", flush=True)
    print("=" * 150, flush=True)

    for idx, q_info in enumerate(BTC_OVER_BUDGET_QUERIES, start=1):
        qid = q_info["qid"]
        name = q_info["name"]
        role = q_info["role"]
        q_file = thunghiem_dir / q_info["file"]
        q_vi = q_file.read_text(encoding="utf-8").strip() if q_file.exists() else ""

        # Run Arm A
        req_a = QueryRequest(request_id=f"a-{qid}", query_id=qid, query_vi=q_vi, output_top_k=100, refine_top_n=3)
        t_start_a = time.time()
        res_a = runtime_a.handle_query(req_a)
        elapsed_a = time.time() - t_start_a

        top100_rel_a = res_a["artifacts"].get("refined_top100_jsonl", res_a["artifacts"]["top100_jsonl"])
        preds_a = [json.loads(line) for line in (runtime_a.output_root / top100_rel_a).read_text(encoding="utf-8").splitlines() if line.strip()]
        eff_en_a = res_a.get("variants", [{}])[0].get("text", "")

        # Run Arm B
        req_b = QueryRequest(request_id=f"b-{qid}", query_id=qid, query_vi=q_vi, output_top_k=100, refine_top_n=3)
        t_start_b = time.time()
        res_b = runtime_b.handle_query(req_b)
        elapsed_b = time.time() - t_start_b

        top100_rel_b = res_b["artifacts"].get("refined_top100_jsonl", res_b["artifacts"]["top100_jsonl"])
        preds_b = [json.loads(line) for line in (runtime_b.output_root / top100_rel_b).read_text(encoding="utf-8").splitlines() if line.strip()]
        eff_en_b = res_b.get("variants", [{}])[0].get("text", "")

        top10_desc_a = [f"@{p['rank']}: {p['video_id']} (f={p['frame_id']})" for p in preds_a[:10]]
        top10_desc_b = [f"@{p['rank']}: {p['video_id']} (f={p['frame_id']})" for p in preds_b[:10]]

        top3_vids_a = [p["video_id"] for p in preds_a[:3]]
        top3_vids_b = [p["video_id"] for p in preds_b[:3]]
        top10_vids_a = set(p["video_id"] for p in preds_a[:10])
        top10_vids_b = set(p["video_id"] for p in preds_b[:10])
        overlap_10 = len(top10_vids_a.intersection(top10_vids_b))

        results.append({
            "qid": qid,
            "name": name,
            "role": role,
            "query_vi": q_vi,
            "eff_en_a": eff_en_a,
            "eff_en_b": eff_en_b,
            "elapsed_a": elapsed_a,
            "elapsed_b": elapsed_b,
            "preds_a": preds_a,
            "preds_b": preds_b,
            "top10_desc_a": top10_desc_a,
            "top10_desc_b": top10_desc_b,
            "overlap_10": overlap_10,
        })

        print(f"\n--- [{idx}/5] {qid} ({role}) ---", flush=True)
        print(f"• Name               : {name}", flush=True)
        print(f"• Arm A Effective EN : \"{eff_en_a}\" (Latency: {elapsed_a:.2f}s)", flush=True)
        print(f"• Arm B Effective EN : \"{eff_en_b}\" (Latency: {elapsed_b:.2f}s)", flush=True)
        print(f"• Arm A Top 10       : {top10_desc_a}", flush=True)
        print(f"• Arm B Top 10       : {top10_desc_b}", flush=True)
        print(f"• Top 10 Overlap     : {overlap_10}/10 shared video entities", flush=True)

    # Generate HTML gallery
    gallery_out = Path("/kaggle/working/kis_p1a2b_btc_gallery.html")
    generate_ab_gallery_html(results, gallery_out, runtime_b.raw_video_registry)
    print(f"\nSaved Comparative Side-by-Side Gallery to: {gallery_out}", flush=True)
    return results


def generate_ab_gallery_html(results: list[dict[str, Any]], out_path: Path, raw_video_registry: Any = None) -> None:
    html_cards = []
    for r in results:
        qid = r["qid"]
        name = r["name"]
        role = r["role"]
        q_vi = r["query_vi"]
        en_a = r["eff_en_a"]
        en_b = r["eff_en_b"]
        preds_a = r["preds_a"][:3]
        preds_b = r["preds_b"][:3]

        def render_top3_grid(preds: list[dict[str, Any]], label: str, color: str) -> str:
            items = []
            for p in preds:
                rank = p["rank"]
                vid = p["video_id"]
                fid = p["frame_id"]
                img_b64 = extract_thumbnail_base64(vid, fid, raw_video_registry)
                img_tag = f'<img src="data:image/jpeg;base64,{img_b64}" style="width:100%; border-radius:4px;" />' if img_b64 else '<div style="background:#333;color:#888;height:80px;display:flex;align-items:center;justify-content:center;">No Frame</div>'
                items.append(f"""
                <div style="flex:1; margin:4px; padding:6px; background:#181818; border:1px solid #333; border-radius:6px; text-align:center; font-size:11px;">
                    <div style="font-weight:bold; color:{color};">Rank @{rank}</div>
                    {img_tag}
                    <div style="color:#eee; font-weight:600; margin-top:2px;">{vid}</div>
                    <div style="color:#888; font-size:10px;">f={fid}</div>
                </div>
                """)
            return f"""
            <div style="flex:1; padding:8px; background:#222; border-radius:6px; margin:4px;">
                <div style="font-weight:bold; color:{color}; margin-bottom:6px;">{label}</div>
                <div style="display:flex;">{''.join(items)}</div>
            </div>
            """

        grid_a = render_top3_grid(preds_a, "ARM A: PREFIX_77 (P0 Baseline)", "#0d6efd")
        grid_b = render_top3_grid(preds_b, "ARM B: HEAD_TAIL_77 (Experimental)", "#28a745")

        role_badge_color = "#ffc107; color:#111" if role == "REGRESSION_GUARD" else "#17a2b8; color:#fff"

        html_cards.append(f"""
        <div style="background:#2b2b2b; border:1px solid #444; border-radius:8px; margin-bottom:20px; padding:16px;">
            <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #3c3c3c; padding-bottom:8px; margin-bottom:12px;">
                <span style="font-size:15px; font-weight:bold; color:#61afef;">{qid}</span>
                <span style="font-size:13px; font-weight:bold; color:#fff;">{name}</span>
                <span style="background:{role_badge_color}; font-weight:bold; font-size:11px; padding:3px 8px; border-radius:4px;">{role}</span>
            </div>
            <div style="font-size:12px; color:#ccc; margin-bottom:4px;"><b style="color:#aaa;">VI Query:</b> {q_vi}</div>
            <div style="font-size:11px; color:#9cdcfe; margin-bottom:4px;"><b style="color:#0d6efd;">Arm A Effective EN:</b> "{en_a}"</div>
            <div style="font-size:11px; color:#98c379; margin-bottom:12px;"><b style="color:#28a745;">Arm B Effective EN:</b> "{en_b}"</div>
            <div style="display:flex; gap:8px;">
                {grid_a}
                {grid_b}
            </div>
        </div>
        """)

    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head><meta charset="utf-8"><title>KIS P1A2b A/B Comparison</title></head>
    <body style="background:#121212; color:#fff; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif; padding:20px;">
        <h2 style="color:#61afef; border-bottom:2px solid #333; padding-bottom:8px;">🔬 KIS P1A2b: HEAD_TAIL_77 CANONICAL SUBSET A/B GALLERY</h2>
        <div style="color:#aaa; font-size:13px; margin-bottom:16px;">Side-by-side Top 3 candidate comparison between ARM A (PREFIX_77) and ARM B (HEAD_TAIL_77).</div>
        {''.join(html_cards)}
    </body>
    </html>
    """
    out_path.write_text(full_html, encoding="utf-8")


def run_dev38_overbudget_census_and_ab() -> None:
    print("\n" + "=" * 150, flush=True)
    print("PART 2: FULL-38 DEV BENCHMARK CENSUS & OVER-BUDGET SUBSET METRIC EVALUATION", flush=True)
    print("=" * 150, flush=True)

    gt_path = REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "l21_150_diagnostic" / "kis_dev_gt.json"
    sidecar_path = REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "l21_150_diagnostic" / "q2_kis_dev_en_translation.json"

    gt_bytes = gt_path.read_bytes()
    gt_sha = hashlib.sha256(gt_bytes).hexdigest()
    assert gt_sha == FROZEN_KIS_DEV_GT_SHA256, f"DEV GT SHA256 mismatch! Got {gt_sha}"

    gt_data = json.loads(gt_bytes.decode("utf-8"))
    raw_queries = gt_data.get("queries", gt_data)
    sidecar_data = json.loads(sidecar_path.read_text(encoding="utf-8"))
    vi_lookup = {rec["query_id"]: rec["source_vi"] for rec in sidecar_data.get("records", [])}

    yaml_path = REPO_ROOT / "systems" / "system_tai" / "configs" / "production.yaml"
    cfg = SessionConfig.from_yaml(yaml_path)
    translator = MarianOfflineTranslator(revision=cfg.translation_revision, local_files_only=True)
    guard = TokenBudgetGuard()

    print(f"Scanning all {len(raw_queries)} DEV benchmark queries for Marian raw token count > 77 ...\n", flush=True)

    dev_overbudget_queries: list[dict[str, Any]] = []

    for q in raw_queries:
        qid = q["query_id"]
        q_vi = vi_lookup.get(qid, "").strip()
        raw_en = translator.translate(q_vi)
        raw_tokens = guard.count_tokens(raw_en)
        is_over = raw_tokens > 77

        status = "⚠️ OVER BUDGET" if is_over else "OK"
        print(f"  • {qid:<10}: {raw_tokens:3d} tokens [{status}] -> \"{raw_en[:65]}...\"", flush=True)

        if is_over:
            dev_overbudget_queries.append({
                "query_record": q,
                "query_id": qid,
                "source_vi": q_vi,
                "raw_marian_en": raw_en,
                "raw_tokens": raw_tokens,
            })

    print(f"\nDEV Benchmark Census Result: {len(dev_overbudget_queries)} / {len(raw_queries)} DEV queries exceed 77 tokens.", flush=True)

    if not dev_overbudget_queries:
        print(">>> Zero DEV queries exceed the 77 token budget. The Full-38 DEV benchmark is 100% unaffected by token packing policy!", flush=True)
        return

    # If over-budget DEV queries exist, run exact metric evaluation A vs B
    print("\n" + "=" * 150, flush=True)
    print(f"EVALUATING METRIC DELTAS ON {len(dev_overbudget_queries)} OVER-BUDGET DEV QUERIES", flush=True)
    print("=" * 150, flush=True)

    input_root = Path("/kaggle/input/datasets") if Path("/kaggle/input/datasets").exists() else Path("/kaggle/input")
    reuse_manifest = get_reuse_manifest()

    out_dev_a = Path("/kaggle/working/output/dev_ab_arm_a") if Path("/kaggle/working").exists() else REPO_ROOT / "scratch" / "dev_ab_arm_a"
    out_dev_b = Path("/kaggle/working/output/dev_ab_arm_b") if Path("/kaggle/working").exists() else REPO_ROOT / "scratch" / "dev_ab_arm_b"

    runtime_dev_a = OperationalKISRuntime.bootstrap(SessionConfig.from_yaml(yaml_path, input_root=input_root, output_root=out_dev_a, reuse_manifest=reuse_manifest, translation_packing_policy="prefix_77"))
    runtime_dev_b = OperationalKISRuntime.bootstrap(SessionConfig.from_yaml(yaml_path, input_root=input_root, output_root=out_dev_b, reuse_manifest=reuse_manifest, translation_packing_policy="head_tail_77"))

    def evaluate_arm(runtime_inst: OperationalKISRuntime, arm_label: str) -> tuple[int, int, list[Any]]:
        score_sum = 0
        strict_hits = 0
        hit_details = []
        for item in dev_overbudget_queries:
            q = item["query_record"]
            qid = item["query_id"]
            q_vi = item["source_vi"]
            target_vid = q["video_id"]
            start_f = int(q.get("start_frame", q.get("start_frame_id", 0)))
            end_f = int(q.get("end_frame", q.get("end_frame_id", 0)))

            req = QueryRequest(request_id=f"eval-{arm_label}-{qid}", query_id=qid, query_vi=q_vi, output_top_k=100, refine_top_n=3)
            res = runtime_inst.handle_query(req)
            top100_rel = res["artifacts"].get("refined_top100_jsonl", res["artifacts"]["top100_jsonl"])
            preds = [json.loads(line) for line in (runtime_inst.output_root / top100_rel).read_text(encoding="utf-8").splitlines() if line.strip()]

            first_strict_rank = None
            first_strict_frame = None
            for p in preds:
                if p["video_id"] == target_vid and (start_f <= p["frame_id"] <= end_f) and first_strict_rank is None:
                    first_strict_rank = p["rank"]
                    first_strict_frame = p["frame_id"]

            q_score = 0
            if first_strict_rank is not None:
                strict_hits += 1
                hit_details.append((qid, first_strict_rank, first_strict_frame))
                for k in OFFICIAL_K:
                    if first_strict_rank <= k:
                        q_score += 1
            score_sum += q_score
            status_tag = f"STRICT HIT @{first_strict_rank} ✅" if first_strict_rank else "MISS ❌"
            print(f"  [{arm_label}] {qid:<10} | {status_tag:<20} | Score: {q_score}/5 | Target: {target_vid} [{start_f}..{end_f}]", flush=True)

        return strict_hits, score_sum, hit_details

    print("\n--- Evaluating Arm A (PREFIX_77) on DEV Over-Budget Subset ---", flush=True)
    hits_a, score_a, details_a = evaluate_arm(runtime_dev_a, "ARM_A")

    print("\n--- Evaluating Arm B (HEAD_TAIL_77) on DEV Over-Budget Subset ---", flush=True)
    hits_b, score_b, details_b = evaluate_arm(runtime_dev_b, "ARM_B")

    max_score = len(dev_overbudget_queries) * 5
    print("\n" + "=" * 150, flush=True)
    print("DEV OVER-BUDGET SUBSET A/B METRIC DELTA REPORT", flush=True)
    print("=" * 150, flush=True)
    print(f"• Subset Query Count        : {len(dev_overbudget_queries)} queries")
    print(f"• Arm A Strict Hits         : {hits_a} / {len(dev_overbudget_queries)}")
    print(f"• Arm B Strict Hits         : {hits_b} / {len(dev_overbudget_queries)} (Δ Strict Hits: {hits_b - hits_a:+d})")
    print(f"• Arm A Score Numerator     : {score_a} / {max_score}")
    print(f"• Arm B Score Numerator     : {score_b} / {max_score} (Δ Score: {score_b - score_a:+d})")
    print(f"• Arm A Strict Hit Details  : {details_a}")
    print(f"• Arm B Strict Hit Details  : {details_b}")
    print("=" * 150, flush=True)


def main() -> None:
    # 1. Run Part 1: BTC Over-Budget Subset A/B
    run_btc_5query_ab_benchmark()

    # 2. Run Part 2: DEV 38 Census & Over-Budget Subset A/B
    run_dev38_overbudget_census_and_ab()


if __name__ == "__main__":
    main()
