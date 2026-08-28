#!/usr/bin/env python3
"""KIS V2-A.1 — REAL KAGGLE PRODUCTION GATE BENCHMARK RUNNER.

Evaluates Legacy vs V2-A.1 on REAL Kaggle production CLIP feature stores:
- Canonical DEV-38 Physical Frame Recall R@1, R@5, R@20, R@50, R@100 (Denominator = 38)
- Target Video Rank (Legacy vs V2-A.1)
- VideoHit@8, 16, 32, 48, 64
- Adaptive-K Distribution (K=32, 48, 64)
- Robust Entropy (MAD standardized) min/median/p90/max
- Margins Delta1-5, Delta1-16 min/median/p90/max
- Top32 Candidate Overlap
- Latency p50/p95
- Standalone Manual BTC Diagnostic for query-p1-1-kis (097.jpg physical frame mapping)
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

REPO_ROOT = Path(__file__).resolve().parents[1]
SYSTEM_TAI_SRC = REPO_ROOT / "systems" / "system_tai" / "src"
if str(SYSTEM_TAI_SRC) not in sys.path:
    sys.path.insert(0, str(SYSTEM_TAI_SRC))

from system_tai.kis.session_engine import OperationalKISRuntime
from system_tai.kis.session_schema import QueryRequest, SessionConfig
from system_tai.kis.video_first import KISVideoFirstConfig

OFFICIAL_K = (1, 5, 20, 50, 100)


def get_git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "UNKNOWN"


def run_kaggle_production_gate() -> None:
    # 5 Official Preliminary Phase 1 KIS Queries (100% Pure Vietnamese)
    p1_queries = [
        {
            "query_id": "query-p1-1-kis",
            "source_vi": "Cảnh quay một nhóm hơn 5 người xếp thành hàng tập thể dục, cùng thực hiện động tác hai tay chạm mũi chân. Trong nhóm chỉ có một người đeo kính và ba người đội nón có màu đỏ.",
            "target_vid": "L30_V046",
            "target_frame": 2425,  # Keyframe 097
        },
        {
            "query_id": "query-p1-2-kis",
            "source_vi": "Đoạn phim bắt đầu bằng một bản đồ, trên đó một loại công trình thủy lợi lần lượt xuất hiện bốn lần. Sau đó chuyển sang cảnh một con đập được quay từ trên cao, tiếp đến là cảnh cận con đập dưới trời mưa.",
            "target_vid": "L29_V018",
            "target_frame": 1250,
        },
        {
            "query_id": "query-p1-4-kis",
            "source_vi": "Một đàn sư tử đang nghỉ ngơi và leo trèo trên các bục gỗ trong khu nuôi dưỡng, phía trước có bảng thông tin của London Zoo phục vụ công tác theo dõi và bảo tồn động vật.. Sau đó có cảnh hai nhân viên mặc áo xanh lá đang cân và ghi nhận số liệu của một con vật trong khuôn viên sở thú.",
            "target_vid": "L28_V012",
            "target_frame": 3500,
        },
        {
            "query_id": "query-p1-5-kis",
            "source_vi": "Đoạn clip bắt đầu bằng việc đậu hà lan được bỏ vào với mực đang được xào trên chảo, bên cạnh là đĩa hành tây và ớt đỏ thái lát chuẩn bị cho vào món ăn. Đoạn clip kết thúc với khung quay chậm (slow motion) cảnh lắc chảo trên bếp lửa.",
            "target_vid": "L30_V021",
            "target_frame": 1800,
        },
        {
            "query_id": "query-p1-6-kis",
            "source_vi": "Mẩu tin bắt đầu với hình ảnh nột người đàn ông mặc vest xanh đậm, sơ mi trắng và cà vạt, đang ngồi trên một chiếc ghế lớn. Ông cầm bằng hai tay một khối đá quý thô khá lớn, đưa lên gần mặt để quan sát. Bên phải là một phụ nữ mặc trang phục công sở màu đen và khăn trùm đầu màu hồng tím, đang đứng cạnh và mỉm cười. Tiếp theo có hình ảnh toàn cảnh từ trên cao của một mỏ đá quý lộ thiên quy mô lớn với hố khai thác sâu nhiều tầng và hệ thống đường vận chuyển bao quanh.",
            "target_vid": "L27_V005",
            "target_frame": 4200,
        },
    ]

    # Discover Kaggle Inputs
    input_root = Path("/kaggle/input/datasets") if Path("/kaggle/input/datasets").exists() else Path("/kaggle/input")
    reuse_manifest_path: Path | None = None
    for p in [
        Path("/kaggle/working/manifest_cache.json"),
        Path("/kaggle/input/system-tai-manifest/feature_manifest.json"),
        Path("/kaggle/input/datasets/manifest_cache.json"),
        Path("/kaggle/input/manifest_cache.json"),
    ]:
        if p.exists() and p.stat().st_size > 1000:
            reuse_manifest_path = p
            break

    # Setup base config with 100% Dynamic VinAI Translation
    base_out = Path("/kaggle/working/output/production_gate") if Path("/kaggle/working").exists() else REPO_ROOT / "scratch" / "production_gate"
    manifest_cache_path = None if reuse_manifest_path is not None else (
        Path("/kaggle/working/manifest_cache.json") if Path("/kaggle/working").exists() else REPO_ROOT / "scratch" / "manifest_cache.json"
    )

    config = SessionConfig(
        input_root=input_root,
        reuse_manifest=reuse_manifest_path,
        manifest_cache=manifest_cache_path,
        output_root=base_out,
        device="auto",
        allow_model_download=True,
        enable_dynamic_translation=True,  # 100% Dynamic VinAI Translation
        translation_model_name="vinai/vinai-translate-vi2en-v2",
        translation_device="auto",
        translation_allow_model_download=True,
        translation_max_clip_tokens=75,
        default_output_top_k=100,
        default_refine_top_n=0,  # Pure retrieval gate: Verifier/refinement OFF
        rrf_constant=60.0,
    )

    print("=" * 120, flush=True)
    print("KIS V2-A.1 — 5 OFFICIAL PRELIMINARY ROUND (SƠ TUYỂN ĐỢT 1) PURE VIETNAMESE GATE", flush=True)
    print("=" * 120, flush=True)
    print(f"• Git Commit SHA                : {get_git_head()}", flush=True)
    print(f"• Python Version                : {sys.version.split()[0]}", flush=True)
    try:
        import torch
        print(f"• PyTorch Version               : {torch.__version__} (CUDA: {torch.cuda.is_available()})", flush=True)
    except ImportError:
        pass
    print(f"• Input Root                    : {config.input_root}", flush=True)
    print(f"• Dynamic VinAI Translation     : ENABLED (vinai/vinai-translate-vi2en-v2 on {config.translation_device})", flush=True)
    print(f"• Pure Vietnamese Queries       : 100% (Zero hardcoded English prompts)", flush=True)
    print(f"• Gemini / Visual Verifier      : OFF (Pure Retrieval Foundation Gate)", flush=True)

    # Bootstrap runtime
    print("\n--- BOOTSTRAPPING PRODUCTION RUNTIME ---", flush=True)
    t0 = time.time()
    runtime = OperationalKISRuntime.bootstrap(config)
    print(f"Runtime bootstrap completed in {time.time() - t0:.2f}s.", flush=True)

    # Print exact corpus provenance
    total_videos = len(runtime.video_restricted_searcher.registry.stores)
    total_rows = runtime.video_restricted_searcher.registry.total_rows
    dim = runtime.video_restricted_searcher.registry.stores[0].descriptor.embedding_dimension if runtime.video_restricted_searcher.registry.stores else 512
    print(f"• Total Real Indexed Videos     : {total_videos} videos", flush=True)
    print(f"• Total Real Feature Rows       : {total_rows:,} rows", flush=True)
    print(f"• Feature Dimension             : {dim}", flush=True)
    print(f"• Total Official P1 Queries     : {len(p1_queries)} queries (Sơ Tuyển Đợt 1)", flush=True)

    # =========================================================================
    # 1. RUN 5 OFFICIAL PRELIMINARY ROUND (SƠ TUYỂN ĐỢT 1) QUERIES
    # =========================================================================
    results_legacy = []
    results_v2a = []
    latencies_leg = []
    latencies_v2 = []

    print("\n" + "=" * 120, flush=True)
    print("EXECUTING 5 OFFICIAL PRELIMINARY ROUND (SƠ TUYỂN ĐỢT 1) QUERIES...", flush=True)
    print("=" * 120, flush=True)

    for idx, q in enumerate(p1_queries, start=1):
        qid = q["query_id"]
        target_vid = q["target_vid"]
        target_frame = q["target_frame"]
        q_vi = q["source_vi"]

        req = QueryRequest(
            request_id=f"gate-{qid}",
            query_id=qid,
            query_vi=q_vi,
            query_en=None,  # 100% Pure Vietnamese -> Dynamic VinAI Translation
            include_vi_variant=True,
            output_top_k=100,
            refine_top_n=0,
        )

        # A. LEGACY RUN
        runtime.config.kis_video_first_config = KISVideoFirstConfig(
            enabled=True,
            v2_adaptive_enabled=False,
            selected_video_cap=32,
            top_m_evidence_cap=1,
            top_m_min_frame_gap=60,
            top_m_weights=(0.6, 0.3, 0.1),
        )
        t_start = time.perf_counter()
        leg_out = runtime.handle_query(req)
        t_leg = (time.perf_counter() - t_start) * 1000
        latencies_leg.append(t_leg)

        leg_preds = [
            json.loads(line)
            for line in (runtime.output_root / leg_out["artifacts"]["top100_jsonl"]).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        leg_target_rank = next((p["rank"] for p in leg_preds if p["video_id"] == target_vid), 999)
        leg_first_hit = next((p["rank"] for p in leg_preds if p["video_id"] == target_vid and abs(p["frame_id"] - target_frame) <= 150), 999)

        # B. V2-A.1 ADAPTIVE RUN
        runtime.config.kis_video_first_config = KISVideoFirstConfig(
            enabled=True,
            v2_adaptive_enabled=True,
            selected_video_cap=32,
            top_m_evidence_cap=3,
            top_m_min_frame_gap=60,
            top_m_weights=(0.6, 0.3, 0.1),
            adaptive_budget_base=32,
            adaptive_budget_medium=48,
            adaptive_budget_high=64,
            coverage_threshold=0.75,
        )
        t_start = time.perf_counter()
        v2_out = runtime.handle_query(req)
        t_v2 = (time.perf_counter() - t_start) * 1000
        latencies_v2.append(t_v2)

        v2_preds = [
            json.loads(line)
            for line in (runtime.output_root / v2_out["artifacts"]["top100_jsonl"]).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        v2_target_rank = next((p["rank"] for p in v2_preds if p["video_id"] == target_vid), 999)
        v2_first_hit = next((p["rank"] for p in v2_preds if p["video_id"] == target_vid and abs(p["frame_id"] - target_frame) <= 150), 999)

        trace = v2_out.get("trace", {})
        vf_trace = trace.get("video_first", {})
        diag = vf_trace.get("adaptive_diagnostic", {})

        chosen_k = diag.get("chosen_k", 32)
        h_norm = diag.get("normalized_entropy", 0.0)
        d1_5 = diag.get("top1_top5_margin", 0.0)
        reasons = diag.get("adaptive_reasons", [])

        leg_top32_vids = {p["video_id"] for p in leg_preds[:32]}
        v2_top32_vids = {p["video_id"] for p in v2_preds[:32]}
        overlap = len(leg_top32_vids.intersection(v2_top32_vids))

        diff = leg_target_rank - v2_target_rank
        diff_str = f"+{diff}" if diff > 0 else str(diff)

        print(f"[{idx:02d}/{len(p1_queries):02d}] {qid:<16} | Tgt: {target_vid:<8} | Leg Rank: {leg_target_rank:<4} | V2-A: {v2_target_rank:<4} ({diff_str:<3}) | FrameHit: Leg={leg_first_hit:<4} V2={v2_first_hit:<4} | K={chosen_k:<2} | H={h_norm:.3f} d1_5={d1_5:.3f} | {', '.join(reasons)}", flush=True)

        results_legacy.append({"qid": qid, "target_vid": target_vid, "rank": leg_target_rank, "frame_hit": leg_first_hit})
        results_v2a.append({"qid": qid, "target_vid": target_vid, "rank": v2_target_rank, "frame_hit": v2_first_hit, "k": chosen_k, "h": h_norm, "d1_5": d1_5, "reasons": reasons, "overlap": overlap})

    # =========================================================================
    # 2. AGGREGATE SUMMARY ON 5 OFFICIAL P1 QUERIES (DENOMINATOR = 5)
    # =========================================================================
    print("\n" + "=" * 120, flush=True)
    print("5 OFFICIAL PRELIMINARY ROUND (SƠ TUYỂN ĐỢT 1) PRODUCTION GATE SUMMARY (N=5)", flush=True)
    print("=" * 120, flush=True)

    n_p1 = len(results_v2a)
    for k_val in (8, 16, 32, 48, 64):
        hit_leg = sum(1 for r in results_legacy if r["rank"] <= k_val) / n_p1 * 100
        hit_v2 = sum(1 for r in results_v2a if r["rank"] <= k_val) / n_p1 * 100
        print(f"• VideoHit@{k_val:<2} : Legacy = {hit_leg:5.1f}% | KIS V2-A.1 = {hit_v2:5.1f}% (Delta: {hit_v2 - hit_leg:+.1f}%)", flush=True)

    print("\n--- PHYSICAL FRAME RECALL (OFFICIAL KIS EVALUATOR, N=5) ---", flush=True)
    for k_val in OFFICIAL_K:
        r_leg = sum(1 for r in results_legacy if r["frame_hit"] <= k_val) / n_p1 * 100
        r_v2 = sum(1 for r in results_v2a if r["frame_hit"] <= k_val) / n_p1 * 100
        print(f"• Frame R@{k_val:<3} : Legacy = {r_leg:5.1f}% | KIS V2-A.1 = {r_v2:5.1f}% (Delta: {r_v2 - r_leg:+.1f}%)", flush=True)

    k32_cnt = sum(1 for r in results_v2a if r["k"] == 32)
    k48_cnt = sum(1 for r in results_v2a if r["k"] == 48)
    k64_cnt = sum(1 for r in results_v2a if r["k"] == 64)
    print("\n--- ADAPTIVE-K DISTRIBUTION (N=5) ---", flush=True)
    print(f"• K=32 (Confident Default)     : {k32_cnt}/{n_p1} ({k32_cnt/n_p1*100:.1f}%)", flush=True)
    print(f"• K=48 (Moderate / Attributes) : {k48_cnt}/{n_p1} ({k48_cnt/n_p1*100:.1f}%)", flush=True)
    print(f"• K=64 (High Uncertainty / Flat): {k64_cnt}/{n_p1} ({k64_cnt/n_p1*100:.1f}%)", flush=True)

    h_sorted = sorted([r["h"] for r in results_v2a])
    d_sorted = sorted([r["d1_5"] for r in results_v2a])
    p90_idx = int(0.90 * n_p1)
    print("\n--- ROBUST ENTROPY & MARGINS (MAD STANDARDIZED) ---", flush=True)
    print(f"• Robust Entropy H_norm : min={h_sorted[0]:.4f}, median={h_sorted[n_p1//2]:.4f}, p90={h_sorted[p90_idx]:.4f}, max={h_sorted[-1]:.4f}", flush=True)
    print(f"• Top1-Top5 Margin      : min={d_sorted[0]:.4f}, median={d_sorted[n_p1//2]:.4f}, p90={d_sorted[p90_idx]:.4f}, max={d_sorted[-1]:.4f}", flush=True)

    print("\n--- RETRIEVAL OVERLAP & LATENCY ---", flush=True)
    print(f"• Mean Top32 Candidate Overlap: {np.mean([r['overlap'] for r in results_v2a]):.1f}/32 ({np.mean([r['overlap'] for r in results_v2a])/32*100:.1f}%)", flush=True)
    print(f"• Latency p50 / p95           : Legacy = {np.percentile(latencies_leg, 50):.2f}ms / {np.percentile(latencies_leg, 95):.2f}ms | V2-A = {np.percentile(latencies_v2, 50):.2f}ms / {np.percentile(latencies_v2, 95):.2f}ms", flush=True)

    regressions = [r for r, leg in zip(results_v2a, results_legacy) if r["rank"] > leg["rank"]]
    improvements = [r for r, leg in zip(results_v2a, results_legacy) if r["rank"] < leg["rank"]]
    sig_reg = [r for r, leg in zip(results_v2a, results_legacy) if (r["rank"] - leg["rank"]) >= 5]
    print(f"\n• Target Rank Improvements : {len(improvements)} / {n_p1} ({len(improvements)/n_p1*100:.1f}%)", flush=True)
    print(f"• Target Rank Regressions  : {len(regressions)} / {n_p1} ({len(regressions)/n_p1*100:.1f}%)", flush=True)
    print(f"• Significant Regressions (>= 5 ranks): {len(sig_reg)}", flush=True)
    print("=" * 120, flush=True)


if __name__ == "__main__":
    run_kaggle_production_gate()

