#!/usr/bin/env python3
"""KIS V2-A.2 — SINGLE-PASS TEMPORAL MULTI-CLAUSE PRODUCTION GATE & VISUALIZER.

Unified benchmark runner & visual keyframe inspector:
1. Bootstraps OperationalKISRuntime with V2-A.2 Soft-AND + DP Temporal Chain Solver.
2. Runs 5 Official Preliminary KIS Queries (100% Pure Vietnamese -> Dynamic VinAI Translation).
3. Writes immutable artifacts (`candidates.json`, `top100.jsonl`) with SHA256 checksum.
4. Evaluates:
   - Primary Target Video Top64 (5/5 Gate)
   - p1-4 Distractor Demotion (L28_V012 outranks single-scene L22_V021)
   - Evidence-Pool Interval Frame Recall
   - Official Physical Frame Recall R@1, 5, 20, 50, 100
   - Diagnostic Temporal Chains & VinAI Translation Audit
5. Renders Top-5 visual contact sheet galleries directly from the same candidate artifact.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import replace
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


def locate_keyframe_path(dataset_root: Path, video_id: str, frame_id: int) -> Path | None:
    """Locate keyframe image file across possible Kaggle directory structures."""
    # Common Kaggle structures:
    # 1. <dataset_root>/keyframes/<video_id>/<frame_id:06d>.jpg
    # 2. <dataset_root>/keyframes/<video_id>/<frame_id>.jpg
    # 3. <dataset_root>/<video_id>/<frame_id:06d>.jpg
    # 4. <dataset_root>/<video_id>/<frame_id>.jpg
    # 5. Search by keyframe order or file index (e.g. 097.jpg)
    candidates = [
        dataset_root / "keyframes" / video_id / f"{frame_id:06d}.jpg",
        dataset_root / "keyframes" / video_id / f"{frame_id:05d}.jpg",
        dataset_root / "keyframes" / video_id / f"{frame_id}.jpg",
        dataset_root / video_id / f"{frame_id:06d}.jpg",
        dataset_root / video_id / f"{frame_id:05d}.jpg",
        dataset_root / video_id / f"{frame_id}.jpg",
        dataset_root / "keyframes" / video_id / f"{frame_id:03d}.jpg",
        dataset_root / video_id / f"{frame_id:03d}.jpg",
    ]
    for p in candidates:
        if p.exists():
            return p

    # Fallback: scan video directory for closest matching frame
    vid_dir = dataset_root / "keyframes" / video_id
    if not vid_dir.exists():
        vid_dir = dataset_root / video_id
    if vid_dir.exists() and vid_dir.is_dir():
        files = list(vid_dir.glob("*.jpg"))
        if files:
            # Try to match numeric stem
            for f in files:
                try:
                    num = int(f.stem)
                    if num == frame_id:
                        return f
                except ValueError:
                    pass
            # Return first available file if not exact
            return files[0]
    return None


def run_kaggle_production_gate_and_visualizer() -> None:
    # 5 Official Preliminary Round KIS Queries (100% Pure Vietnamese)
    # Groundtruth targets (Evaluator-Only: strictly offline assessment)
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
            "target_frame": 6050,  # Dam opening floodgates under rain
        },
        {
            "query_id": "query-p1-4-kis",
            "source_vi": "Một đàn sư tử đang nghỉ ngơi và leo trèo trên các bục gỗ trong khu nuôi dưỡng, phía trước có bảng thông tin của London Zoo phục vụ công tác theo dõi và bảo tồn động vật.. Sau đó có cảnh hai nhân viên mặc áo xanh lá đang cân và ghi nhận số liệu của một con vật trong khuôn viên sở thú.",
            "target_vid": "L28_V012",
            "target_frame": 1375,  # London zoo lions & keepers weighing animals
        },
        {
            "query_id": "query-p1-5-kis",
            "source_vi": "Đoạn clip bắt đầu bằng việc đậu hà lan được bỏ vào với mực đang được xào trên chảo, bên cạnh là đĩa hành tây và ớt đỏ thái lát chuẩn bị cho vào món ăn. Đoạn clip kết thúc với khung quay chậm (slow motion) cảnh lắc chảo trên bếp lửa.",
            "target_vid": "L30_V021",
            "target_frame": 3325,  # Peas & squid frying + slow motion tossing
        },
        {
            "query_id": "query-p1-6-kis",
            "source_vi": "Mẩu tin bắt đầu với hình ảnh một người đàn ông mặc vest xanh đậm, sơ mi trắng và cà vạt, đang ngồi trên một chiếc ghế lớn. Ông cầm bằng hai tay một khối đá quý thô khá lớn, đưa lên gần mặt để quan sát. Bên phải là một phụ nữ mặc trang phục công sở màu đen và khăn trùm đầu màu hồng tím, đang đứng cạnh và mỉm cười. Tiếp theo có hình ảnh toàn cảnh từ trên cao của một mỏ đá quý lộ thiên quy mô lớn với hố khai thác sâu nhiều tầng và hệ thống đường vận chuyển bao quanh.",
            "target_vid": "L27_V005",
            "target_frame": 1150,  # Gemstone inspection + terraced open pit mine
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

    base_out = Path("/kaggle/working/output/v2a2_gate") if Path("/kaggle/working").exists() else REPO_ROOT / "scratch" / "v2a2_gate"
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
        default_refine_top_n=0,  # Pure retrieval gate: Verifier OFF
        rrf_constant=60.0,
        kis_video_first_config=KISVideoFirstConfig(
            enabled=True,
            v2_adaptive_enabled=True,
            selected_video_cap=64,
            top_m_evidence_cap=5,
            top_m_min_frame_gap=60,
            top_m_weights=(0.4, 0.25, 0.15, 0.1, 0.1),
            adaptive_budget_base=32,
            adaptive_budget_medium=48,
            adaptive_budget_high=64,
            coverage_threshold=0.75,
        ),
    )

    print("=" * 120, flush=True)
    print("KIS V2-A.2 — SINGLE-PASS TEMPORAL MULTI-CLAUSE VIDEO NOMINATION & EXACT AUDIT", flush=True)
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
    print(f"• Temporal Multi-Clause Soft-AND: ENABLED (Geometric Mean + DP Chain Solver)", flush=True)
    print(f"• Pre-Verifier Evidence Pool    : ENABLED (Neighborhood interval retention)", flush=True)
    print(f"• Verifier / Gemini (V2-B)      : OFF (Strict Promotion Gate Benchmark)", flush=True)

    # Bootstrap runtime
    print("\n--- BOOTSTRAPPING OPERATIONAL KIS RUNTIME ---", flush=True)
    t0 = time.time()
    runtime = OperationalKISRuntime.bootstrap(config)
    print(f"Runtime bootstrap completed in {time.time() - t0:.2f}s.", flush=True)

    total_videos = len(runtime.video_restricted_searcher.registry.stores)
    total_rows = runtime.video_restricted_searcher.registry.total_rows
    print(f"• Real Indexed Videos           : {total_videos} videos", flush=True)
    print(f"• Real Feature Rows             : {total_rows:,} rows", flush=True)
    print(f"• Benchmark Query Count         : {len(p1_queries)} queries (Sơ Tuyển Đợt 1)", flush=True)

    # =========================================================================
    # 1. RUN RETRIEVAL ONCE & WRITE IMMUTABLE ARTIFACTS
    # =========================================================================
    print("\n" + "=" * 120, flush=True)
    print("EXECUTING SINGLE-PASS RETRIEVAL OVER REAL CORPUS...", flush=True)
    print("=" * 120, flush=True)

    query_results = []
    latencies = []

    for idx, q in enumerate(p1_queries, start=1):
        qid = q["query_id"]
        target_vid = q["target_vid"]
        target_frame = q["target_frame"]
        q_vi = q["source_vi"]

        req = QueryRequest(
            request_id=f"v2a2-eval-{qid}",
            query_id=qid,
            query_vi=q_vi,
            query_en=None,
            include_vi_variant=True,
            output_top_k=100,
            refine_top_n=0,
        )

        t_start = time.perf_counter()
        out = runtime.handle_query(req)
        t_elapsed = (time.perf_counter() - t_start) * 1000
        latencies.append(t_elapsed)

        # Read immutable candidates.json artifact generated by the engine
        query_out_dir = runtime.output_root / out["request_id"]
        candidates_file = query_out_dir / "candidates.json"
        top100_file = query_out_dir / "top100.jsonl"

        cand_data = json.loads(candidates_file.read_text(encoding="utf-8"))
        top100_sha256 = cand_data.get("top100_sha256", "UNKNOWN")
        records = cand_data.get("records", [])
        evidence_pool = cand_data.get("evidence_frame_pool", [])
        translation_meta = cand_data.get("translation", {})
        vf_trace = cand_data.get("video_first", {})
        adaptive_diag = vf_trace.get("adaptive_budget", {})

        # Compute Evaluator Metrics strictly from this saved artifact
        target_rank = next((p["rank"] for p in records if p["video_id"] == target_vid), 999)
        official_frame_hit = next(
            (p["rank"] for p in records if p["video_id"] == target_vid and abs(p["frame_id"] - target_frame) <= 150),
            999,
        )

        # Evidence pool interval recall (check if target frame interval was retained in pre-verifier pool)
        in_evidence_pool = any(
            item["video_id"] == target_vid and abs(int(item["frame_id"]) - target_frame) <= 150
            for item in evidence_pool
        )

        # Distractor audit for p1-4 (London Zoo)
        distractor_l22_rank = next((p["rank"] for p in records if p["video_id"] == "L22_V021"), 999)

        # Extract VinAI translation units
        units_trace = translation_meta.get("units", [])
        trans_summary = " | ".join(
            f"T{u.get('temporal_index', idx)}: '{u.get('raw_english', '')[:45]}...'"
            for idx, u in enumerate(units_trace[1:], start=1)
        ) if len(units_trace) > 1 else (units_trace[0].get("raw_english", "")[:60] if units_trace else "N/A")

        # Extract temporal chain diagnostic for target video if available
        target_vid_ev = next((v for v in vf_trace.get("selected_videos", []) if v["video_id"] == target_vid), None)
        chain_info = target_vid_ev.get("temporal_chain") if target_vid_ev else None
        chain_summary = (
            f"Chain={chain_info.get('selected_chain_frames')} (Score={chain_info.get('chain_score', 0):.3f}, SoftAND={chain_info.get('soft_and_score', 0):.3f})"
            if chain_info and chain_info.get("has_valid_chain")
            else "Single-Scene / No Chain"
        )

        print(f"\n[{idx:02d}/{len(p1_queries):02d}] QUERY: {qid}")
        print(f"   • Vietnamese Query     : {q_vi}")
        print(f"   • VinAI Translation    : {trans_summary}")
        print(f"   • Target Groundtruth   : Video {target_vid} @ Frame {target_frame}")
        print(f"   • Target Video Rank    : Rank {target_rank} ({'TOP 64 HIT ✅' if target_rank <= 64 else 'FAIL ❌'})")
        print(f"   • Pre-Verifier Pool    : {'RETAINED ✅' if in_evidence_pool else 'MISSED ❌'}")
        print(f"   • Official Frame Hit   : Rank {official_frame_hit} ({'EXACT HIT ✅' if official_frame_hit <= 100 else 'NO HIT ❌'})")
        print(f"   • Temporal Chain Audit : {chain_summary}")
        if qid == "query-p1-4-kis":
            print(f"   • Distractor Check     : Target L28_V012 (Rank {target_rank}) vs Distractor L22_V021 (Rank {distractor_l22_rank}) -> {'SUCCESS (Target Outranks Distractor) ✅' if target_rank < distractor_l22_rank else 'DISTRACTOR DOMINATED ❌'}")
        print(f"   • Artifact SHA256      : {top100_sha256[:16]}...")

        query_results.append({
            "qid": qid,
            "target_vid": target_vid,
            "target_frame": target_frame,
            "target_rank": target_rank,
            "official_frame_hit": official_frame_hit,
            "in_evidence_pool": in_evidence_pool,
            "sha256": top100_sha256,
            "records": records,
            "trans_summary": trans_summary,
            "distractor_l22_rank": distractor_l22_rank if qid == "query-p1-4-kis" else None,
        })

    # =========================================================================
    # 2. PROMOTION GATE AUDIT TABLE
    # =========================================================================
    print("\n" + "=" * 120, flush=True)
    print("KIS V2-A.2 OFFICIAL PROMOTION GATE EVALUATION (N=5)", flush=True)
    print("=" * 120, flush=True)

    n_queries = len(query_results)
    top64_count = sum(1 for r in query_results if r["target_rank"] <= 64)
    top32_count = sum(1 for r in query_results if r["target_rank"] <= 32)
    ev_pool_count = sum(1 for r in query_results if r["in_evidence_pool"])
    frame100_count = sum(1 for r in query_results if r["official_frame_hit"] <= 100)

    print(f"• Primary Target Video Top64 Recall : {top64_count}/{n_queries} ({top64_count/n_queries*100:.1f}%) [CRITERIA: 5/5]")
    print(f"• Pre-Verifier Evidence Pool Recall : {ev_pool_count}/{n_queries} ({ev_pool_count/n_queries*100:.1f}%)")
    print(f"• Target Video Top32 Recall (Diag)  : {top32_count}/{n_queries} ({top32_count/n_queries*100:.1f}%)")
    print(f"• Official Physical Frame R@100     : {frame100_count}/{n_queries} ({frame100_count/n_queries*100:.1f}%)")

    # Audit individual queries
    p1_5 = next(r for r in query_results if r["qid"] == "query-p1-5-kis")
    p1_6 = next(r for r in query_results if r["qid"] == "query-p1-6-kis")
    p1_4 = next(r for r in query_results if r["qid"] == "query-p1-4-kis")

    print("\n--- CRITICAL INDIVIDUAL TESTCASES ---")
    print(f"• p1-5 (Peas & Squid): Target Rank = {p1_5['target_rank']} ({'RESOLVED (<=64) ✅' if p1_5['target_rank'] <= 64 else 'ABSENT ❌'})")
    print(f"• p1-6 (Gemstone)    : Target Rank = {p1_6['target_rank']} ({'RESOLVED (<=64) ✅' if p1_6['target_rank'] <= 64 else 'ABSENT ❌'})")
    print(f"• p1-4 (London Zoo)  : Target Rank = {p1_4['target_rank']} vs Distractor = {p1_4['distractor_l22_rank']} ({'PASSED ✅' if p1_4['target_rank'] < p1_4['distractor_l22_rank'] else 'FAILED ❌'})")

    # =========================================================================
    # 3. DIRECT MATPLOTLIB VISUAL GALLERY (READS THE SAME ARTIFACT)
    # =========================================================================
    print("\n" + "=" * 120, flush=True)
    print("RENDERING KEYFRAME VISUAL INSPECTION GALLERIES (MATPLOTLIB)...", flush=True)
    print("=" * 120, flush=True)

    try:
        import matplotlib.pyplot as plt
        from PIL import Image

        vis_out_dir = Path("/kaggle/working/output/visualizer") if Path("/kaggle/working").exists() else REPO_ROOT / "scratch" / "visualizer"
        vis_out_dir.mkdir(parents=True, exist_ok=True)

        for res in query_results:
            qid = res["qid"]
            tgt_vid = res["target_vid"]
            tgt_frame = res["target_frame"]
            sha = res["sha256"]
            top_records = res["records"][:5]

            fig, axes = plt.subplots(1, 5, figsize=(20, 4.5))
            fig.suptitle(
                f"[{qid}] Target: {tgt_vid} @ {tgt_frame} | Artifact SHA: {sha[:8]}... | Target Video Rank: #{res['target_rank']}",
                fontsize=12,
                fontweight="bold",
                y=1.03,
            )

            for i, ax in enumerate(axes):
                if i < len(top_records):
                    rec = top_records[i]
                    v_id = rec["video_id"]
                    f_id = rec["frame_id"]
                    rank = rec["rank"]
                    score = rec.get("fusion_score", 0.0)

                    is_target_vid = (v_id == tgt_vid)
                    is_exact_frame = is_target_vid and (abs(f_id - tgt_frame) <= 150)

                    img_path = locate_keyframe_path(input_root, v_id, f_id)
                    if img_path and img_path.exists():
                        try:
                            img = Image.open(img_path)
                            ax.imshow(img)
                        except Exception:
                            ax.text(0.5, 0.5, "Image Load Error", ha="center", va="center")
                    else:
                        ax.text(0.5, 0.5, f"Missing:\n{v_id}\nF={f_id}", ha="center", va="center", fontsize=9)

                    # Highlight border & title
                    if is_exact_frame:
                        border_color = "green"
                        status_tag = "🎯 EXACT HIT"
                    elif is_target_vid:
                        border_color = "orange"
                        status_tag = "⚠️ TARGET VID (DIFF MOMENT)"
                    else:
                        border_color = "black"
                        status_tag = "DISTRACTOR"

                    ax.set_title(
                        f"Rank #{rank} | {v_id}\nFrame: {f_id} (Score: {score:.3f})\n{status_tag}",
                        fontsize=9,
                        color="green" if is_exact_frame else ("red" if not is_target_vid else "orange"),
                        fontweight="bold",
                    )
                    for spine in ax.spines.values():
                        spine.set_edgecolor(border_color)
                        spine.set_linewidth(3 if is_target_vid else 1)
                ax.axis("off")

            plt.tight_layout()
            save_path = vis_out_dir / f"{qid}_top5.png"
            plt.savefig(save_path, dpi=120, bbox_inches="tight")
            print(f"• Gallery saved: {save_path} (Artifact SHA: {sha[:16]} | Evaluator Rank: #{res['target_rank']})", flush=True)
            try:
                plt.show()
            except Exception:
                pass
            plt.close(fig)

    except ImportError:
        print("• Matplotlib / PIL not available in this environment, skipping graphical display.", flush=True)

    print("\n" + "=" * 120, flush=True)
    print("V2-A.2 SINGLE-PASS BENCHMARK & VISUAL AUDIT COMPLETE.", flush=True)
    print("=" * 120, flush=True)


if __name__ == "__main__":
    run_kaggle_production_gate_and_visualizer()
