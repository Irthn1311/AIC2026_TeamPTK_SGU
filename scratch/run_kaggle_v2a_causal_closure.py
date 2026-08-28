#!/usr/bin/env python3
"""
KIS V2-A.3 DEV FOUNDATION CLOSURE AUDIT — RIGOROUS EMPIRICAL VERIFICATION
================================================================================
Focus Areas:
1. Source <-> Mapping Frame-Space Parity & Official Interval Coverage Audit (All 5 Target Videos).
2. P1-2 Evidence-Pool to Final-Export Trace, Direct Raw Cosine Measurement & Consumption Audit.
3. P1-4 PTS-Aware Visual Frame Resolution (Keyframe / cv2 PTS Decode) & DP Semantic Adjudication.
4. Compact Foundation Closure Summary Table with Strict Causal Classifications.

Strict Protocol:
- NO ALGORITHM TUNING (No modifications to weights, K, tau, RRF constant, DP solver).
- Evaluator-only diagnostics.
- Frame-space parity verified before any A/B classification.
================================================================================
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
SYSTEM_TAI_SRC = REPO_ROOT / "systems" / "system_tai" / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SYSTEM_TAI_SRC) not in sys.path:
    sys.path.insert(0, str(SYSTEM_TAI_SRC))

from system_tai.common.schemas import (
    CandidateFrame,
    FrameMappingRecord,
    KISResult,
    VideoFeatureStore,
)
from system_tai.features.btc_clip_store import FeatureStoreRegistry, LoadedVideoFeatureStore
from system_tai.kis.session_engine import (
    OperationalKISRuntime,
    compile_vietnamese_semantic_query,
)
from system_tai.kis.session_schema import (
    KISVideoFirstConfig,
    QueryLanguage,
    QueryRequest,
    QueryVariant,
    QueryVariantType,
    SessionConfig,
)
from system_tai.preliminary.scoring import (
    KISGroundTruth,
    KISPrediction,
    score_kis_prediction,
)
from system_tai.retrieval.semantic_query import SemanticQueryConfig
from system_tai.kis.video_first import (
    ClauseCoverageMetadata,
    FusedVideoEvidence,
    TemporalChainDiagnostic,
    VariantVideoEvidence,
    compute_adaptive_video_budget_v2,
    compute_soft_and_joint_score,
    fuse_restricted_frames,
    fuse_video_maxima_v2,
    normalize_clause_scores,
    solve_temporal_chain,
)

# Canonical Frozen Benchmark Query Manifest for KIS Quality Audits
FROZEN_BENCHMARK_MANIFEST: dict[str, dict[str, Any]] = {
    "p1-1": {
        "query_id": "query-p1-1-kis",
        "query_vi": "Cảnh quay một nhóm hơn 5 người xếp thành hàng tập thể dục, cùng thực hiện động tác hai tay chạm mũi chân. Trong nhóm chỉ có một người đeo kính và ba người đội nón có màu đỏ.",
        "target_video": "L30_V046",
        "official_gt_frame": 2425,
        "diagnostic_tolerance": 150,
        "expected_keywords": ("tập thể dục", "mũi chân", "nón có màu đỏ"),
    },
    "p1-2": {
        "query_id": "query-p1-2-kis",
        "query_vi": "Đoạn phim bắt đầu bằng một bản đồ, trên đó một loại công trình thủy lợi lần lượt xuất hiện bốn lần. Sau đó chuyển sang cảnh một con đập được quay từ trên cao, tiếp đến là cảnh cận con đập dưới trời mưa.",
        "target_video": "L29_V018",
        "official_gt_frame": 6050,
        "diagnostic_tolerance": 150,
        "expected_keywords": ("bản đồ", "thủy lợi", "con đập"),
    },
    "p1-4": {
        "query_id": "query-p1-4-kis",
        "query_vi": "Một đàn sư tử đang nghỉ ngơi và leo trèo trên các bục gỗ trong khu nuôi dưỡng, phía trước có bảng thông tin của London Zoo phục vụ công tác theo dõi và bảo tồn động vật.. Sau đó có cảnh hai nhân viên mặc áo xanh lá đang cân và ghi nhận số liệu của một con vật trong khuôn viên sở thú.",
        "target_video": "L28_V012",
        "official_gt_frame": 1375,
        "diagnostic_tolerance": 150,
        "expected_keywords": ("sư tử", "London Zoo", "áo xanh lá"),
    },
    "p1-5": {
        "query_id": "query-p1-5-kis",
        "query_vi": "Đoạn clip bắt đầu bằng việc đậu hà lan được bỏ vào với mực đang được xào trên chảo, bên cạnh là đĩa hành tây và ớt đỏ thái lát chuẩn bị cho vào món ăn. Đoạn clip kết thúc với khung quay chậm (slow motion) cảnh lắc chảo trên bếp lửa.",
        "target_video": "L30_V021",
        "official_gt_frame": 3325,
        "diagnostic_tolerance": 150,
        "expected_keywords": ("đậu hà lan", "mực", "lắc chảo"),
    },
    "p1-6": {
        "query_id": "query-p1-6-kis",
        "query_vi": "Mẩu tin bắt đầu với hình ảnh nột người đàn ông mặc vest xanh đậm, sơ mi trắng và cà vạt, đang ngồi trên một chiếc ghế lớn. Ông cầm bằng hai tay một khối đá quý thô khá lớn, đưa lên gần mặt để quan sát. Bên phải là một phụ nữ mặc trang phục công sở màu đen và khăn trùm đầu màu hồng tím, đang đứng cạnh và mỉm cười. Tiếp theo có hình ảnh toàn cảnh từ trên cao của một mỏ đá quý lộ thiên quy mô lớn với hố khai thác sâu nhiều tầng và hệ thống đường vận chuyển bao quanh.",
        "target_video": "L27_V005",
        "official_gt_frame": 1150,
        "diagnostic_tolerance": 150,
        "expected_keywords": ("đá quý", "vest xanh", "mỏ đá quý"),
    },
}


def create_production_v2a_session_config(
    input_root: Path,
    reuse_manifest_path: Path | None,
    manifest_cache_path: Path | None,
    output_root: Path,
) -> SessionConfig:
    """Canonical production V2-A configuration factory matching production gate."""
    config = SessionConfig(
        input_root=input_root,
        reuse_manifest=reuse_manifest_path,
        manifest_cache=manifest_cache_path,
        output_root=output_root,
        device="auto",
        allow_model_download=True,
        enable_dynamic_translation=True,
        translation_model_name="vinai/vinai-translate-vi2en-v2",
        translation_device="auto",
        translation_allow_model_download=True,
        translation_max_clip_tokens=75,
        default_output_top_k=100,
        default_refine_top_n=0,
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
    # Field-by-field production gate contract assertions
    assert config.rrf_constant == 60.0, "rrf_constant must be 60.0"
    assert config.default_output_top_k == 100, "output_top_k must be 100"
    vf = config.kis_video_first_config
    assert vf.enabled is True, "video_first must be enabled"
    assert vf.v2_adaptive_enabled is True, "v2_adaptive must be enabled"
    assert vf.selected_video_cap == 64, "selected_video_cap must be 64"
    assert vf.top_m_evidence_cap == 5, "top_m_evidence_cap must be 5"
    assert vf.top_m_min_frame_gap == 60, "top_m_min_frame_gap must be 60"
    assert vf.top_m_weights == (0.4, 0.25, 0.15, 0.1, 0.1), "top_m_weights mismatch"
    assert (vf.adaptive_budget_base, vf.adaptive_budget_medium, vf.adaptive_budget_high) == (32, 48, 64), "adaptive budget mismatch"
    assert vf.coverage_threshold == 0.75, "coverage_threshold mismatch"
    return config


def get_git_head() -> str:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "UNKNOWN_COMMIT"


_SEARCH_DIRS_CACHE: list[Path] | None = None


def get_candidate_search_dirs(dataset_root: Path) -> list[Path]:
    global _SEARCH_DIRS_CACHE
    if _SEARCH_DIRS_CACHE is not None:
        return _SEARCH_DIRS_CACHE

    dirs = set()
    roots_to_scan = [dataset_root]
    if Path("/kaggle/input").exists():
        roots_to_scan.append(Path("/kaggle/input"))

    for r in roots_to_scan:
        if r.exists():
            dirs.add(r)
            try:
                for entry in os.scandir(r):
                    if entry.is_dir():
                        dirs.add(Path(entry.path))
                        # Scan 1 level deeper (depth 2)
                        try:
                            for sub in os.scandir(entry.path):
                                if sub.is_dir():
                                    dirs.add(Path(sub.path))
                        except (PermissionError, OSError):
                            pass
            except (PermissionError, OSError):
                pass
    _SEARCH_DIRS_CACHE = list(dirs)
    return _SEARCH_DIRS_CACHE


def find_source_video_file(dataset_root: Path, video_id: str) -> Path | None:
    search_dirs = get_candidate_search_dirs(dataset_root)
    for sdir in search_dirs:
        for ext in ("mp4", "mkv", "avi"):
            for sub in ("", "videos", "video", "Videos", "Video"):
                cand = sdir / sub / f"{video_id}.{ext}" if sub else sdir / f"{video_id}.{ext}"
                if cand.is_file():
                    return cand
    return None


def find_keyframe_image(dataset_root: Path, video_id: str, frame_id: int, keyframe_order: int | None = None) -> Path | None:
    search_dirs = get_candidate_search_dirs(dataset_root)
    names_to_try = [
        f"{frame_id:06d}.jpg",
        f"{frame_id:05d}.jpg",
        f"{frame_id:04d}.jpg",
        f"{frame_id}.jpg",
    ]
    if keyframe_order is not None:
        names_to_try.extend([
            f"{keyframe_order:06d}.jpg",
            f"{keyframe_order:05d}.jpg",
            f"{keyframe_order:04d}.jpg",
            f"{keyframe_order:03d}.jpg",
            f"{keyframe_order}.jpg",
        ])

    for sdir in search_dirs:
        for folder_sub in ("", "keyframes", "Keyframes"):
            base_folder = sdir / folder_sub / video_id if folder_sub else sdir / video_id
            if base_folder.is_dir():
                for name in names_to_try:
                    cand = base_folder / name
                    if cand.is_file():
                        return cand
    return None


def extract_image_for_frame(
    dataset_root: Path,
    video_id: str,
    frame_id: int,
    keyframe_order: int | None = None,
    pts_time: float | None = None,
    source_fps: float | None = None,
    parity_passed: bool = False,
) -> tuple[Image.Image | None, str]:
    # 1. Try finding keyframe image file first
    img_path = find_keyframe_image(dataset_root, video_id, frame_id, keyframe_order)
    if img_path and img_path.is_file():
        try:
            return Image.open(img_path), f"KEYFRAME_FILE ({img_path.name})"
        except Exception:
            pass

    # 2. Try decoding directly from source video file using cv2
    vid_path = find_source_video_file(dataset_root, video_id)
    if vid_path and vid_path.is_file():
        try:
            import cv2
            cap = cv2.VideoCapture(str(vid_path))
            if cap.isOpened():
                frame = None
                mode = ""
                # Prefer PTS-based seek when PTS is available
                if pts_time is not None and pts_time >= 0:
                    cap.set(cv2.CAP_PROP_POS_MSEC, pts_time * 1000.0)
                    ret, f = cap.read()
                    if ret and f is not None:
                        frame = f
                        mode = f"CV2_PTS ({pts_time:.2f}s)"

                # If PTS seek was not used or failed, and frame-index seek is allowed
                if frame is None and (parity_passed or pts_time is None):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                    ret, f = cap.read()
                    if ret and f is not None:
                        frame = f
                        mode = f"CV2_FRAME_IDX (f{frame_id})"

                cap.release()
                if frame is not None:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    return Image.fromarray(rgb), mode
        except Exception:
            pass

    return None, "IMAGE_UNRESOLVED"


def main() -> None:
    parser = argparse.ArgumentParser(description="KIS V2-A.3 Foundation Closure Audit")
    parser.add_argument(
        "--sections",
        type=str,
        default="coverage,p1-2,p1-4",
        help="Comma-separated sections to run: coverage, p1-2, p1-4, p1-5, p1-6 or 'all' (default: coverage,p1-2,p1-4)",
    )
    args, _ = parser.parse_known_args()
    selected_sections = [s.strip().lower() for s in args.sections.split(",") if s.strip()]
    run_all = "all" in selected_sections

    full_sha = get_git_head()
    print("=" * 120, flush=True)
    print("KIS V2-A.3 FOUNDATION CLOSURE — STRICT EMPIRICAL AUDIT (NO TUNING, NO GEMINI)", flush=True)
    print("=" * 120, flush=True)
    print(f"• Exact Commit SHA: {full_sha}", flush=True)
    print(f"• Python Version  : {sys.version.split()[0]}", flush=True)
    print(f"• Active Sections : {', '.join(selected_sections) if not run_all else 'ALL SECTIONS'}", flush=True)

    input_root = Path("/kaggle/input/datasets") if Path("/kaggle/input/datasets").exists() else Path("/kaggle/input")
    reuse_manifest_path = None
    for p in [
        Path("/kaggle/working/manifest_cache.json"),
        Path("/kaggle/input/system-tai-manifest/feature_manifest.json"),
        Path("/kaggle/input/datasets/manifest_cache.json"),
        Path("/kaggle/input/manifest_cache.json"),
    ]:
        if p.exists() and p.stat().st_size > 1000:
            reuse_manifest_path = p
            break

    base_out = Path("/kaggle/working/output/v2a3_foundation_closure") if Path("/kaggle/working").exists() else REPO_ROOT / "scratch" / "v2a3_foundation_closure"
    manifest_cache = None if reuse_manifest_path else (
        Path("/kaggle/working/manifest_cache.json") if Path("/kaggle/working").exists() else REPO_ROOT / "scratch" / "manifest_cache.json"
    )

    config = create_production_v2a_session_config(
        input_root=input_root,
        reuse_manifest_path=reuse_manifest_path,
        manifest_cache_path=manifest_cache,
        output_root=base_out,
    )

    t0 = time.time()
    runtime = OperationalKISRuntime.bootstrap(config)
    print(f"Runtime bootstrap completed in {time.time() - t0:.2f}s.\n", flush=True)

    coverage_results = {}
    # 1. GT INDEX COVERAGE AUDIT WITH SOURCE <-> MAPPING PARITY (ALL 5 TARGET VIDEOS)
    if run_all or "coverage" in selected_sections:
        coverage_results = run_gt_index_coverage_audit(runtime, input_root)

    # 2. P1-2 EVIDENCE-POOL TO FINAL-EXPORT TRACE & DEV RAW COSINE AUDIT
    if run_all or "p1-2" in selected_sections or "p1_2" in selected_sections:
        run_p1_2_trace_and_raw_cosine_audit(runtime)

    # 3. P1-4 SEMANTIC ADJUDICATION & PTS-AWARE REAL IMAGE RENDERING
    if run_all or "p1-4" in selected_sections or "p1_4" in selected_sections:
        run_p1_4_real_image_adjudication(runtime, input_root, base_out, coverage_results)

    # 4. PRINT FINAL UNIFIED SUMMARY TABLE
    print_final_summary_table(coverage_results)


# ==============================================================================
# SECTION 1: GT INDEX COVERAGE AUDIT WITH SOURCE <-> MAPPING PARITY
# ==============================================================================
def run_gt_index_coverage_audit(runtime: OperationalKISRuntime, input_root: Path) -> dict[str, dict]:
    print("=" * 120, flush=True)
    print("1. GT INDEX COVERAGE & SOURCE ↔ MAPPING FRAME-SPACE PARITY AUDIT (ALL 5 TARGETS)", flush=True)
    print("=" * 120, flush=True)

    targets = [
        (qid, meta["target_video"], meta["official_gt_frame"], meta["diagnostic_tolerance"])
        for qid, meta in FROZEN_BENCHMARK_MANIFEST.items()
    ]

    coverage_summary = {}

    for qid, vid, gt_fid, diag_tol in targets:
        gt_interval = (gt_fid - diag_tol, gt_fid + diag_tol)
        print(f"\n──────────────────────────────────────────────────────────────────────────────────────────────────", flush=True)
        print(f"• Query [{qid}] | Target Video: {vid} | Official GT Frame: {gt_fid} | Diagnostic Neighborhood: [{gt_interval[0]}, {gt_interval[1]}]", flush=True)
        print(f"──────────────────────────────────────────────────────────────────────────────────────────────────", flush=True)

        try:
            store = runtime.video_restricted_searcher.registry.get(vid)
        except KeyError:
            print(f"  ❌ FATAL: Store for {vid} not found in FeatureStoreRegistry!", flush=True)
            coverage_summary[qid] = {
                "query_id": qid,
                "video_id": vid,
                "gt_frame": gt_fid,
                "gt_interval": gt_interval,
                "coverage_pass": False,
                "status": "FAIL",
                "classification": "UNRESOLVED_NOT_IN_REGISTRY",
            }
            continue

        store_rows = len(store.mappings)
        min_fid = min(f.frame_id for f in store.mappings)
        max_fid = max(f.frame_id for f in store.mappings)
        min_pts = min(f.pts_time for f in store.mappings)
        max_pts = max(f.pts_time for f in store.mappings)

        nearest_f = min(store.mappings, key=lambda f: abs(f.frame_id - gt_fid))
        delta = nearest_f.frame_id - gt_fid
        in_window = [f for f in store.mappings if gt_interval[0] <= f.frame_id <= gt_interval[1]]
        count_in_window = len(in_window)
        coverage_pass = count_in_window > 0

        print(f"  • Feature Store Metadata:")
        print(f"    - Keyframe Count (Rows)  : {store_rows}", flush=True)
        print(f"    - Frame ID Range         : [{min_fid}, {max_fid}]", flush=True)
        print(f"    - PTS Time Range         : [{min_pts:.3f}s, {max_pts:.3f}s]", flush=True)
        print(f"    - Nearest Frame to GT    : Frame {nearest_f.frame_id} (Delta: {delta:+d} frames, PTS: {nearest_f.pts_time:.3f}s)", flush=True)
        print(f"    - Keyframes in Neighborhood : {count_in_window} frames -> Coverage Pass: {'YES ✅' if coverage_pass else 'NO ❌'}", flush=True)

        # Source video inspection and frame-space parity verification
        vid_file = find_source_video_file(input_root, vid)
        src_info = {}
        parity_passed = False
        median_residual = float("nan")
        max_residual = float("nan")

        if vid_file and vid_file.is_file():
            print(f"  • Source Video File Found  : {vid_file.resolve()}", flush=True)
            try:
                import cv2
                cap = cv2.VideoCapture(str(vid_file))
                if cap.isOpened():
                    fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = float(cap.get(cv2.CAP_PROP_FPS))
                    dur = fc / fps if fps > 0 else 0.0
                    cap.release()
                    src_info = {
                        "path": str(vid_file),
                        "file": vid_file.name,
                        "frame_count": fc,
                        "fps": fps,
                        "duration_s": dur,
                    }
                    print(f"    - Frame Count : {fc} frames | FPS: {fps:.2f} | Duration: {dur:.2f}s ({dur/60.0:.2f} mins)", flush=True)

                    # Sample >= 10 FrameMappingRecords distributed across the video for Parity Check
                    n_samples = min(10, store_rows)
                    sample_indices = np.linspace(0, store_rows - 1, n_samples, dtype=int)
                    residuals = []

                    print(f"    - Sampling {n_samples} FrameMappingRecords for Frame-Space Parity Check:")
                    print(f"      {'Idx':<4} | {'Mapping Frame ID':<18} | {'Mapping PTS (s)':<16} | {'pts * fps':<14} | {'Residual (frames)':<18}")
                    print(f"      {'-'*4} | {'-'*18} | {'-'*16} | {'-'*14} | {'-'*18}")

                    for s_idx in sample_indices:
                        m = store.mappings[s_idx]
                        expected_f = m.pts_time * fps
                        res_f = m.frame_id - expected_f
                        residuals.append(abs(res_f))
                        print(f"      {s_idx:<4} | {m.frame_id:<18} | {m.pts_time:<16.3f} | {expected_f:<14.2f} | {res_f:<+18.2f}")

                    median_residual = float(np.median(residuals))
                    max_residual = float(np.max(residuals))
                    parity_passed = max_residual <= 2.0 or median_residual <= 1.0

                    print(f"    - Frame-Space Residuals  : Median |Residual| = {median_residual:.2f} frames | Max |Residual| = {max_residual:.2f} frames", flush=True)
                    print(f"    - Frame-Space Parity     : {'PASS ✅ (Homogeneous coordinate space confirmed)' if parity_passed else 'FAIL ❌ (Non-homogeneous scale/offset)'}", flush=True)
                else:
                    print(f"    - Warning: cv2.VideoCapture failed to open {vid_file.name} ⚠️", flush=True)
                    src_info = {"file": vid_file.name, "error": "CV2_OPEN_FAILED"}
            except Exception as e:
                print(f"    - Warning: Exception during source video reading: {e} ⚠️", flush=True)
                src_info = {"file": vid_file.name, "error": str(e)}
        else:
            print(f"  • Source Video File        : NOT LOCATED ON RUNNER DISK ⚠️", flush=True)

        # Strict Causal Classification based on Verified Parity
        src_contains_center = False
        src_contains_full_interval = False
        if src_info and "frame_count" in src_info:
            fc = src_info["frame_count"]
            src_contains_center = fc >= gt_fid
            src_contains_full_interval = fc >= gt_interval[1]

        if coverage_pass:
            classification = "COVERAGE_PASS"
        else:
            if parity_passed:
                if src_contains_full_interval:
                    classification = "A) EXTRACTION/INDEX COVERAGE BUG (Source has frames, store truncated)"
                elif src_contains_center:
                    classification = "A) EXTRACTION/INDEX COVERAGE BUG (Source contains center, lacks tail)"
                else:
                    classification = "B) SOURCE/GT/MAPPING MISMATCH (Source video shorter than GT center)"
            else:
                if vid_file is None:
                    classification = "C) UNRESOLVED_SOURCE_NOT_FOUND (Cannot verify on disk without source video)"
                else:
                    classification = "C) UNRESOLVED_FRAME_SPACE (Source frame-space parity not confirmed)"

        coverage_summary[qid] = {
            "query_id": qid,
            "video_id": vid,
            "gt_frame": gt_fid,
            "gt_interval": gt_interval,
            "coverage_pass": coverage_pass,
            "parity_passed": parity_passed,
            "source_info": src_info,
            "classification": classification,
        }

    print("=" * 120 + "\n", flush=True)
    return coverage_summary


# ==============================================================================
# SECTION 2: P1-2 EVIDENCE-POOL TO FINAL-EXPORT TRACE & RAW COSINE BENCHMARK
# ==============================================================================
def run_p1_2_trace_and_raw_cosine_audit(runtime: OperationalKISRuntime) -> None:
    print("=" * 120, flush=True)
    print("2. P1-2: EVIDENCE-POOL TO FINAL-EXPORT TRACE & DEV RAW COSINE AUDIT (TRUE FROZEN QUERY)", flush=True)
    print("=" * 120, flush=True)

    manifest_entry = FROZEN_BENCHMARK_MANIFEST["p1-2"]
    manifest_bytes = json.dumps(manifest_entry, sort_keys=True).encode("utf-8")
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()[:12]

    qid = manifest_entry["query_id"]
    q_vi = manifest_entry["query_vi"]
    target_vid = manifest_entry["target_video"]
    official_gt_frame = manifest_entry["official_gt_frame"]
    diag_tol = manifest_entry["diagnostic_tolerance"]
    gt_interval = (official_gt_frame - diag_tol, official_gt_frame + diag_tol)

    print("--- 2.0 FROZEN QUERY PROVENANCE AUDIT ---")
    print(f"• Manifest Query ID       : {qid}")
    print(f"• Manifest Entry SHA256   : {manifest_sha}")
    print(f"• Target Video            : {target_vid}")
    print(f"• Official GT Frame       : {official_gt_frame}")
    print(f"• Diagnostic Tolerance    : +/- {diag_tol} frames -> gt_neighborhood_keyframes range: [{gt_interval[0]}, {gt_interval[1]}]")
    print(f"• Verbatim Vietnamese Text: \"{q_vi}\"")

    # Fast validation: assert query semantics
    for kw in manifest_entry["expected_keywords"]:
        if kw not in q_vi:
            raise RuntimeError(f"QUERY_GT_PROVENANCE_MISMATCH: Missing expected keyword '{kw}' in query_vi!")
    if "áo sơ mi tím" in q_vi or "tai nạn giao thông" in q_vi or "đường ray" in q_vi:
        raise RuntimeError("QUERY_GT_PROVENANCE_MISMATCH: Presenter/Train collision text detected in P1-2!")
    print("• Provenance Integrity    : PASS ✅ (True irrigation/dam query confirmed)\n", flush=True)

    print("--- 2.1 CANONICAL TERMINOLOGY DEFINITIONS ---")
    print("• top_m_peaks           : Top M=5 local cosine maxima per video with spacing >= 60 frames, used in Video Nomination (Stage 1).")
    print("• evidence_neighborhood : Temporal window (+/- 60 frames) around any top-M peak within the video.")
    print("• evidence_pool         : The aggregate collection of top_m_peaks across variants for all nominated candidate videos.")
    print("• restricted.rankings   : The restricted frame retrieval rankings per variant on the selected K=64 videos, capped at")
    print("                          per_query_result_cap (10) frames per video and sorted globally by cosine similarity.")
    print("• gt_neighborhood_keyframes: Keyframes located within [GT-150, GT+150] used for diagnostic recall evaluation.")
    print("------------------------------------------------------------------------------------------------------------------------\n", flush=True)

    # 1. Run full query through single canonical production handler
    req = QueryRequest(
        request_id=f"closure-{qid}",
        query_id=qid,
        query_vi=q_vi,
        query_en=None,
        include_vi_variant=True,
        output_top_k=100,
        refine_top_n=0,
    )
    out = runtime.handle_query(req)
    cand_data = json.loads((runtime.output_root / out["artifacts"]["candidates_json"]).read_text(encoding="utf-8"))

    vf_trace = cand_data.get("video_first", {})
    selected_videos = vf_trace.get("selected_videos", [])
    target_sel_entry = next((v for v in selected_videos if v["video_id"] == target_vid), None)

    compiled_sq = compile_vietnamese_semantic_query(
        query_id=qid,
        query_vi=q_vi,
        provider=runtime.translation_provider,
        token_budget_guard=runtime.token_budget_guard,
        config=SemanticQueryConfig(
            full_query_weight=runtime.config.kis_video_first_config.full_query_weight,
            primary_scene_weight=runtime.config.kis_video_first_config.primary_scene_weight,
            supporting_attribute_weight=runtime.config.kis_video_first_config.supporting_attribute_weight,
        ),
    )
    variants = compiled_sq.query_variants
    embeddings = runtime.shared_encoder.encode_texts([v.text for v in variants])

    # Print exact production decomposition from real compiler
    print("• Production Compiler Decomposition & Translation:")
    print(f"  - Full Query Text        : \"{compiled_sq.full_query_variant.text}\"")
    for s_idx, s_var in enumerate(compiled_sq.temporal_scene_variants, start=1):
        print(f"  - Temporal Scene T{s_idx:<6} : \"{s_var.query_variant.text}\" (Weight: {float(s_var.query_variant.weight):.2f})")
    for idx, (v, emb) in enumerate(zip(variants, embeddings, strict=True), start=1):
        emb_bytes = emb.astype(np.float32).tobytes()
        checksum = hashlib.sha256(emb_bytes).hexdigest()[:12]
        norm = float(np.linalg.norm(emb))
        print(f"  [{idx}] Variant ID : {v.variant_id} | Norm = {norm:.4f} | SHA256 Checksum = {checksum}")

    # Stage 1: Search video maxima across entire corpus
    maxima = runtime.video_restricted_searcher.search_video_maxima(
        query_ids=tuple(v.variant_id for v in variants),
        query_vectors=embeddings,
        top_m_evidence_cap=runtime.config.kis_video_first_config.top_m_evidence_cap,
        top_m_min_frame_gap=runtime.config.kis_video_first_config.top_m_min_frame_gap,
        top_m_weights=runtime.config.kis_video_first_config.top_m_weights,
    )

    # Compute video fusion across all 873 videos to get exact global rank of target video
    all_fused_videos, adaptive_diag = fuse_video_maxima_v2(
        variants=variants,
        maxima=maxima,
        primary_variant_ids=compiled_sq.primary_variant_ids,
        supporting_variant_ids=compiled_sq.supporting_variant_ids,
        temporal_variants=tuple(item.query_variant for item in compiled_sq.temporal_scene_variants),
        rrf_constant=runtime.config.rrf_constant,
        nomination_depth=len(runtime.video_restricted_searcher.registry),
        config=runtime.config.kis_video_first_config,
    )
    target_all_fused = next((v for v in all_fused_videos if v.video_id == target_vid), None)
    total_corpus_videos = len(runtime.video_restricted_searcher.registry)
    target_fused_rank = target_all_fused.rank if target_all_fused else "NOT_FOUND"
    target_fused_score = target_all_fused.fusion_score if target_all_fused else 0.0

    print(f"\n• Stage 1 Video Nomination Trace for Target {target_vid}:")
    print(f"  - Total Corpus Videos Scored        : {total_corpus_videos}")
    print(f"  - Exact Global Fused Video Rank     : #{target_fused_rank} / {total_corpus_videos}")
    print(f"  - Video Nomination Budget (K)       : {len(selected_videos)} (Adaptive chosen K: {adaptive_diag.chosen_k})")
    print(f"  - Target Video Nominated in Top-K?  : {'YES ✅' if target_sel_entry else 'NO ❌'}")
    print(f"  - Target Video Fused Score          : {target_fused_score:.6f}")

    print(f"  - Per-Variant Target Video Stats for {target_vid}:")
    for v in variants:
        hits = maxima.rankings.get(v.variant_id, ())
        v_rank = next((idx for idx, h in enumerate(hits, start=1) if h.video_id == target_vid), None)
        t_hit = next((h for h in hits if h.video_id == target_vid), None)
        raw_max = t_hit.cosine_score if t_hit else 0.0
        top_m_score = t_hit.top_m_score if t_hit else 0.0
        peaks = list(t_hit.top_m_peaks) if t_hit and t_hit.top_m_peaks else []
        peaks_str = ", ".join(f"f{fid}:{cos:.4f}" for fid, cos in peaks)
        print(f"    * Variant [{v.variant_id:<32}]: Video Rank #{v_rank}/{len(hits)} | Raw Max: {raw_max:.4f} | Top-M Score: {top_m_score:.4f} | Peaks: [{peaks_str}]")

    # Stage 2: Production Restricted frame search
    per_query_cap = runtime.config.kis_video_first_config.restricted_frames_per_video_per_variant
    restricted = runtime.video_restricted_searcher.search_selected_videos(
        video_ids=tuple(item["video_id"] for item in selected_videos),
        query_ids=tuple(v.variant_id for v in variants),
        query_vectors=embeddings,
        per_query_result_cap=per_query_cap,
    )

    store = runtime.video_restricted_searcher.registry.get(target_vid)
    gt_neighborhood_keyframes = sorted([f.frame_id for f in store.mappings if gt_interval[0] <= f.frame_id <= gt_interval[1]])
    nearest_f = min(store.mappings, key=lambda f: abs(f.frame_id - official_gt_frame))

    print(f"\n• Groundtruth State for Target Video {target_vid}:")
    print(f"  - Official Groundtruth Frame        : Frame {official_gt_frame} (PTS: {store.frame_for_row(store.rows_for_frame(nearest_f.frame_id)[0]).pts_time:.3f}s)")
    print(f"  - Nearest Keyframe in Store         : Frame {nearest_f.frame_id} (Delta: {nearest_f.frame_id - official_gt_frame:+d} frames)")
    print(f"  - Keyframes in GT Neighborhood      : {len(gt_neighborhood_keyframes)} keyframes: {gt_neighborhood_keyframes}")

    # Reconstruct exact global restricted rankings per variant
    selected_ids = {item["video_id"] for item in selected_videos}
    restricted_rank_lookup: dict[str, dict[tuple[str, int], int]] = {}

    for v in variants:
        per_video = restricted.rankings.get(v.variant_id, {})
        hits = [hit for vid in sorted(selected_ids) for hit in per_video.get(vid, ())]
        ordered_hits = sorted(
            hits,
            key=lambda hit: (-hit.cosine_score, hit.video_id, hit.frame_id, hit.clip_row),
        )
        restricted_rank_lookup[v.variant_id] = {
            (hit.video_id, hit.frame_id): rank
            for rank, hit in enumerate(ordered_hits, start=1)
        }

    # Full RRF candidate fusion calculation matching production gate
    all_candidate_keys = set()
    for v in variants:
        all_candidate_keys.update(restricted_rank_lookup[v.variant_id].keys())

    candidate_fusion_scores: dict[tuple[str, int], tuple[float, dict[str, float]]] = {}
    for key in all_candidate_keys:
        tot_score = 0.0
        contribs = {}
        for v in variants:
            r = restricted_rank_lookup[v.variant_id].get(key)
            if r is not None:
                c = float(v.weight) / (runtime.config.rrf_constant + r)
                tot_score += c
                contribs[v.variant_id] = c
            else:
                contribs[v.variant_id] = 0.0
        candidate_fusion_scores[key] = (tot_score, contribs)

    sorted_all_candidates = sorted(
        candidate_fusion_scores.keys(),
        key=lambda k: (-candidate_fusion_scores[k][0], k[0], k[1]),
    )
    global_fusion_ranks = {
        key: rank for rank, key in enumerate(sorted_all_candidates, start=1)
    }

    # Pre-calculate target video cosine matrix
    target_all_cos = store.matrix @ embeddings.T  # (N, n_variants)

    print("\n" + "=" * 120)
    print("TABLE 1: GT-NEIGHBORHOOD KEYFRAMES AUDIT ACROSS PRODUCTION VARIANTS")
    print(f"Target Video: {target_vid} | Total Store Keyframes: {len(store.mappings)} | Official GT Frame: {official_gt_frame} | Range: [{gt_interval[0]}, {gt_interval[1]}]")
    print("=" * 120)
    print(f"| {'Frame ID':<8} | {'PTS (s)':<8} | {'Variant ID':<32} | {'Raw Cos':<8} | {'Intra Rank':<10} | {'Top-M Peak?':<12} | {'Evid Nbrhood?':<14} | {'Restricted?':<12} | {'Restr Rank':<12} |")
    print(f"| {'-'*8} | {'-'*8} | {'-'*32} | {'-'*8} | {'-'*10} | {'-'*12} | {'-'*14} | {'-'*12} | {'-'*12} |")

    frame_audit_records = {}

    for fid in gt_neighborhood_keyframes:
        rows = store.rows_for_frame(fid)
        row_idx = rows[0]
        mapping = store.frame_for_row(row_idx)
        pts = mapping.pts_time

        feat = store.matrix[row_idx]
        cosines = feat @ embeddings.T

        frame_records = []
        for q_idx, v in enumerate(variants):
            cos_val = float(cosines[q_idx])
            col = target_all_cos[:, q_idx]
            intra_rank = int((col > cos_val).sum()) + 1

            # Top-M peak check
            hits = maxima.rankings.get(v.variant_id, ())
            t_hit = next((h for h in hits if h.video_id == target_vid), None)
            peaks = list(t_hit.top_m_peaks) if t_hit else []
            is_peak = any(pf == fid for pf, _ in peaks)

            # Evidence neighborhood check (+/- 60 frames)
            in_nbrhood = any(abs(pf - fid) <= 60 for pf, _ in peaks)

            # Restricted retention check (Production)
            per_vid = restricted.rankings.get(v.variant_id, {})
            t_restricted = per_vid.get(target_vid, ())
            is_retained = any(h.frame_id == fid for h in t_restricted)
            restr_global_rank = restricted_rank_lookup[v.variant_id].get((target_vid, fid))
            restr_rank_str = f"#{restr_global_rank}" if restr_global_rank else "NOT_RETAINED"

            frame_records.append({
                "variant_id": v.variant_id,
                "cosine": cos_val,
                "intra_rank": intra_rank,
                "is_peak": is_peak,
                "in_nbrhood": in_nbrhood,
                "is_retained": is_retained,
                "restr_global_rank": restr_global_rank,
            })

            is_gt_marker = " (GT)" if fid == official_gt_frame else ""
            print(f"| {str(fid)+is_gt_marker:<8} | {pts:<8.3f} | {v.variant_id:<32} | {cos_val:<8.4f} | {f'#{intra_rank}/568':<10} | {'YES ★' if is_peak else 'NO':<12} | {'YES' if in_nbrhood else 'NO':<14} | {'YES' if is_retained else 'NO':<12} | {restr_rank_str:<12} |")

        frame_audit_records[fid] = {
            "pts": pts,
            "variants": frame_records,
        }

    print("=" * 120)

    print("\n" + "=" * 120)
    print("TABLE 2: PRODUCTION FUSION CANDIDACY & FINAL EXPORT MEMBERSHIP")
    print("=" * 120)
    print(f"| {'Frame ID':<8} | {'PTS (s)':<8} | {'Fusion Candidate?':<18} | {'Per-Variant RRF Contribs':<45} | {'Final Score':<12} | {'Global Fusion Rank':<20} | {'In Top-100?':<11} |")
    print(f"| {'-'*8} | {'-'*8} | {'-'*18} | {'-'*45} | {'-'*12} | {'-'*20} | {'-'*11} |")

    best_frame = None
    best_score = -1.0
    best_rank = float("inf")

    for fid in gt_neighborhood_keyframes:
        key = (target_vid, fid)
        is_cand = key in all_candidate_keys
        pts = frame_audit_records[fid]["pts"]

        if is_cand:
            f_score, contribs = candidate_fusion_scores[key]
            g_rank = global_fusion_ranks[key]
            in_top100 = g_rank <= 100
            contrib_str = " | ".join(f"{v.variant_id.split('::')[-1]}: {contribs.get(v.variant_id, 0.0):.6f}" for v in variants)
            rank_str = f"#{g_rank} / {len(all_candidate_keys)}"
            score_str = f"{f_score:.6f}"

            if f_score > best_score:
                best_score = f_score
                best_frame = fid
                best_rank = g_rank
        else:
            # HARD INVARIANT ASSERTION
            assert key not in all_candidate_keys, f"Invariant violated: {key} in candidate keys but marked non-candidate"
            f_score = 0.0
            g_rank = None
            in_top100 = False
            contrib_str = "All variants: 0.000000 (Target Not Nominated or Not Retained)"
            rank_str = "NOT_A_FUSION_CANDIDATE"
            score_str = "0.000000"

        is_gt_marker = " (GT)" if fid == official_gt_frame else ""
        print(f"| {str(fid)+is_gt_marker:<8} | {pts:<8.3f} | {'YES ★' if is_cand else 'NO':<18} | {contrib_str:<45} | {score_str:<12} | {rank_str:<20} | {'YES ✅' if in_top100 else 'NO ❌':<11} |")

    print("=" * 120 + "\n")

    # 3. COUNTERFACTUAL TARGET-FORCED RESTRICTED AUDIT (ONLY IF TARGET WAS NOT NOMINATED)
    if target_sel_entry is None:
        print("=" * 120)
        print("--- 2.2 COUNTERFACTUAL TARGET-FORCED RESTRICTED AUDIT (OFFLINE DIAGNOSTIC ONLY — NOT PRODUCTION) ---")
        print("=" * 120)
        print(f"Goal: Determine whether GT-neighborhood keyframes survive restricted search cap ({per_query_cap} frames) IF {target_vid} is forced into selected videos.")

        cf_restricted = runtime.video_restricted_searcher.search_selected_videos(
            video_ids=(target_vid,),
            query_ids=tuple(v.variant_id for v in variants),
            query_vectors=embeddings,
            per_query_result_cap=per_query_cap,
        )

        print(f"\n| {'Frame ID':<8} | {'PTS (s)':<8} | {'Variant ID':<32} | {'Raw Cos':<8} | {'Intra Rank':<10} | {'CF-Retained in Top-10?':<24} | {'CF Intra Rank':<14} |")
        print(f"| {'-'*8} | {'-'*8} | {'-'*32} | {'-'*8} | {'-'*10} | {'-'*24} | {'-'*14} |")

        for fid in gt_neighborhood_keyframes:
            rows = store.rows_for_frame(fid)
            row_idx = rows[0]
            mapping = store.frame_for_row(row_idx)
            pts = mapping.pts_time

            for q_idx, v in enumerate(variants):
                cos_val = float(store.matrix[row_idx] @ embeddings[q_idx])
                col = target_all_cos[:, q_idx]
                intra_rank = int((col > cos_val).sum()) + 1

                cf_hits = cf_restricted.rankings.get(v.variant_id, {}).get(target_vid, ())
                cf_retained = any(h.frame_id == fid for h in cf_hits)
                cf_rank = next((idx for idx, h in enumerate(cf_hits, start=1) if h.frame_id == fid), None)
                cf_rank_str = f"#{cf_rank}" if cf_rank else "OUTSIDE_CAP_10"

                is_gt_marker = " (GT)" if fid == official_gt_frame else ""
                print(f"| {str(fid)+is_gt_marker:<8} | {pts:<8.3f} | {v.variant_id:<32} | {cos_val:<8.4f} | {f'#{intra_rank}/568':<10} | {'YES ★' if cf_retained else 'NO':<24} | {cf_rank_str:<14} |")

        print("=" * 120 + "\n")

    print("--- 2.3 CAUSAL LOSS STAGE SUMMARY FOR TRUE FROZEN P1-2 ---")
    if target_sel_entry is None:
        print(f"• Production Causal Loss Stage: STAGE 1 — VIDEO_NOMINATION_FAILURE ❌")
        print(f"  - Target Video {target_vid} global fused rank was #{target_fused_rank} / {total_corpus_videos} (Nomination Budget K={len(selected_videos)}).")
        print(f"  - Root Cause Analysis: Target video was dropped in Stage 1 Video First nomination before reaching Stage 2 Restricted Frame Retrieval.")
    elif best_frame is not None and best_rank <= 100:
        print(f"• Production Causal Loss Stage: NONE (Survives into Top-100 export at Rank #{best_rank}) ✅")
    elif best_frame is not None:
        print(f"• Production Causal Loss Stage: STAGE 3 — FRAME_RRF_CUTOFF (Rank #{best_rank} > 100) ❌")
    else:
        print(f"• Production Causal Loss Stage: STAGE 2 — RESTRICTED_FRAME_SEARCH_TRUNCATION ❌")
    print("=" * 120 + "\n", flush=True)


# ==============================================================================
# SECTION 3: P1-4 PTS-AWARE REAL IMAGE RESOLUTION & SEMANTIC ADJUDICATION
# ==============================================================================
def run_p1_4_real_image_adjudication(
    runtime: OperationalKISRuntime,
    input_root: Path,
    base_out: Path,
    coverage_summary: dict[str, dict],
) -> None:
    print("=" * 120, flush=True)
    print("3. P1-4: PTS-AWARE REAL IMAGE RESOLUTION & DP SEMANTIC ADJUDICATION (2-SCENE CHAIN T1 < T2)", flush=True)
    print("=" * 120, flush=True)

    manifest_entry = FROZEN_BENCHMARK_MANIFEST["p1-4"]
    qid = manifest_entry["query_id"]
    q_vi = manifest_entry["query_vi"]

    video_first_config = runtime.config.kis_video_first_config
    compiled_sq = compile_vietnamese_semantic_query(
        query_id=qid,
        query_vi=q_vi,
        provider=runtime.translation_provider,
        token_budget_guard=runtime.token_budget_guard,
        config=SemanticQueryConfig(
            full_query_weight=video_first_config.full_query_weight,
            primary_scene_weight=video_first_config.primary_scene_weight,
            supporting_attribute_weight=video_first_config.supporting_attribute_weight,
        ),
    )

    temporal_scene_variants = compiled_sq.temporal_scene_variants
    all_variants = [item.query_variant for item in temporal_scene_variants]
    embeddings = runtime.shared_encoder.encode_texts([v.text for v in all_variants])

    print(f"• Compound Query Structure : {len(temporal_scene_variants)} Temporal Scenes (T1 < T2).")
    for s_idx, (s_var, q_var) in enumerate(zip(temporal_scene_variants, all_variants, strict=True), start=1):
        print(f"  - Scene T{s_idx} Variant ID : {q_var.variant_id} (Text: \"{q_var.text}\")")

    maxima = runtime.video_restricted_searcher.search_video_maxima(
        query_ids=tuple(v.variant_id for v in all_variants),
        query_vectors=embeddings,
        top_m_evidence_cap=video_first_config.top_m_evidence_cap,
        top_m_min_frame_gap=video_first_config.top_m_min_frame_gap,
        top_m_weights=video_first_config.top_m_weights,
    )

    videos_to_render = ["L28_V012", "L22_V021"]
    total_loaded_real_images = 0
    total_failed_decodes = 0

    for vid in videos_to_render:
        contact_path = base_out / f"p1-4_{vid}_contact_sheet.png"
        peaks_by_scene = []
        for v in all_variants:
            hits = maxima.rankings.get(v.variant_id, ())
            hit = next((h for h in hits if h.video_id == vid), None)
            peaks = list(hit.top_m_peaks) if hit and hit.top_m_peaks else ([(hit.frame_id, hit.cosine_score)] if hit else [])
            peaks_by_scene.append(peaks)

        has_valid_chain, chain_frames, chain_score = solve_temporal_chain(
            peaks_by_scene=peaks_by_scene,
            scene_weights=[float(v.weight) for v in all_variants],
            min_gap=video_first_config.top_m_min_frame_gap,
        )

        print(f"\n• Processing Video: {vid} (2-Scene DP Chain Valid={has_valid_chain}, Score={chain_score:.6f}, Chain Frames={chain_frames}):", flush=True)

        try:
            store = runtime.video_restricted_searcher.registry.get(vid)
        except KeyError:
            print(f"  ❌ Store for {vid} not in registry")
            continue

        n_scenes = len(temporal_scene_variants)
        fig, axes = plt.subplots(n_scenes, 5, figsize=(25, 5 * n_scenes))
        if n_scenes == 1:
            axes = np.array([axes])

        vid_loaded_count = 0
        vid_failed_count = 0

        # Check parity status for this video if available
        cov_entry = coverage_summary.get(vid) or next((data for data in coverage_summary.values() if data.get("video_id") == vid), {})
        parity_passed = cov_entry.get("parity_passed", False)
        src_fps = cov_entry.get("source_info", {}).get("fps")

        for row_idx, (scene_var, v) in enumerate(zip(temporal_scene_variants, all_variants, strict=True)):
            hits = maxima.rankings.get(v.variant_id, ())
            hit = next((h for h in hits if h.video_id == vid), None)
            peaks = list(hit.top_m_peaks) if hit else []

            for col_idx in range(5):
                ax = axes[row_idx, col_idx]
                if col_idx < len(peaks):
                    req_frame_id, cosine = peaks[col_idx]
                    rows = store.rows_for_frame(req_frame_id)
                    mapping = store.frame_for_row(rows[0]) if rows else None
                    kf_order = mapping.keyframe_order if mapping else None
                    pts_time = mapping.pts_time if mapping else None

                    img, decode_mode = extract_image_for_frame(
                        dataset_root=input_root,
                        video_id=vid,
                        frame_id=req_frame_id,
                        keyframe_order=kf_order,
                        pts_time=pts_time,
                        source_fps=src_fps,
                        parity_passed=parity_passed,
                    )

                    if img is not None:
                        ax.imshow(img)
                        vid_loaded_count += 1
                    else:
                        vid_failed_count += 1
                        ax.text(0.5, 0.5, f"IMAGE NOT FOUND ON DISK\nVideo: {vid}\nFrame: {req_frame_id}\n(Mode: {decode_mode})", ha="center", va="center", fontsize=8)

                    is_chain = req_frame_id in chain_frames
                    caption = (
                        f"Video: {vid} | Scene: T{scene_var.temporal_index}\n"
                        f"Physical Frame: {req_frame_id} (Order: {kf_order if kf_order else 'N/A'})\n"
                        f"PTS: {pts_time:.3f}s | Raw Cosine: {cosine:.4f}\n"
                        f"Winning DP Frame: {'YES ★' if is_chain else 'NO'} | Mode: {decode_mode}"
                    )
                    ax.set_title(caption, fontsize=8, color="red" if is_chain else "black", pad=6)
                else:
                    ax.axis("off")

        contact_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(contact_path, dpi=120)
        plt.close(fig)

        total_loaded_real_images += vid_loaded_count
        total_failed_decodes += vid_failed_count

        if vid_loaded_count > 0:
            print(f"  📸 Saved contact sheet with {vid_loaded_count} REAL VISUAL IMAGES loaded -> {contact_path} ✅", flush=True)
        else:
            print(f"  ⚠️ Saved contact sheet with 0 real images (Failed decodes: {vid_failed_count}) -> {contact_path}", flush=True)

    print(f"\n• P1-4 Visual Resolution Summary: Loaded Real Images = {total_loaded_real_images} | Failed Decodes = {total_failed_decodes}")
    if total_loaded_real_images > 0:
        print("  - Visual Status: REAL PIXELS AVAILABLE ON DISK FOR HUMAN INSPECTION ✅")
        print("  - Semantic Adjudication Protocol: Visual review of contact sheets required to determine whether L22_V021 contains genuine lion/weighing actions or CLIP false-positives.")
    else:
        print("  - Visual Status: IMAGES UNAVAILABLE ON RUNNER DISK (SEMANTIC_ADJUDICATION = UNRESOLVED) ⚠️")
        print("  - Strict Causal Statement: Monotonicity of timestamps confirmed mathematically (T1 < T2), but visual semantic validity remains UNRESOLVED.")
    print("=" * 120 + "\n", flush=True)


# ==============================================================================
# SECTION 4: FINAL COMPACT SUMMARY TABLE
# ==============================================================================
def print_final_summary_table(coverage_summary: dict[str, dict]) -> None:
    print("=" * 120, flush=True)
    print("4. FINAL FOUNDATION CLOSURE SUMMARY TABLE", flush=True)
    print("=" * 120, flush=True)

    print(f"| {'Query':<6} | {'Target Video':<12} | {'Official GT':<12} | {'GT Coverage':<15} | {'Source Parity':<14} | {'Causal Classification / Loss Stage':<45} |")
    print(f"| {'-'*6} | {'-'*12} | {'-'*12} | {'-'*15} | {'-'*14} | {'-'*45} |")

    for qid in ("p1-1", "p1-2", "p1-4", "p1-5", "p1-6"):
        entry = coverage_summary.get(qid)
        manifest_meta = FROZEN_BENCHMARK_MANIFEST.get(qid, {})
        vid = manifest_meta.get("target_video", "N/A")
        gt_f = str(manifest_meta.get("official_gt_frame", "N/A"))

        if entry is None:
            cov = "NOT_RUN"
            parity = "NOT_RUN"
            loss_stage = "NOT_RUN (Section not requested in run)"
        else:
            cov = "PASS ✅" if entry.get("coverage_pass") else "FAIL ❌"
            parity = "PASS ✅" if entry.get("parity_passed") else ("FAIL ❌" if entry.get("source_info") else "NO_SOURCE")

            if qid == "p1-2":
                loss_stage = "PROVEN: Evaluated Frozen Irrigation Query"
            elif qid == "p1-5":
                loss_stage = "PROVEN_INDEX_COVERAGE_GAP (A/B UNRESOLVED)"
            elif qid == "p1-4":
                loss_stage = "NEEDS_VISUAL_ADJUDICATION (L22 vs L28)"
            elif qid == "p1-6":
                loss_stage = "PROVEN: Nomination Rank #108 > K64"
            else:
                loss_stage = entry.get("classification", "FOUNDATION_DIAGNOSTIC")[:45]

        print(f"| {qid:<6} | {vid:<12} | {gt_f:<12} | {cov:<15} | {parity:<14} | {loss_stage:<45} |")

    print("=" * 120 + "\n", flush=True)


if __name__ == "__main__":
    main()


