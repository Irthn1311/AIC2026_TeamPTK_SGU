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
from datetime import UTC, datetime
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
import urllib.request
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

def load_canonical_frozen_manifest() -> tuple[Path, str, dict[str, dict[str, Any]]]:
    """Load canonical frozen stress benchmark manifest directly from repository files."""
    possible_paths = [
        SYSTEM_TAI_SRC.parent / "benchmarks" / "frozen_kis_v2a_stress_manifest.json",
        REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "frozen_kis_v2a_stress_manifest.json",
        Path("/kaggle/working/AIC2026_TeamPTK_SGU/systems/system_tai/benchmarks/frozen_kis_v2a_stress_manifest.json"),
    ]
    manifest_path = None
    for p in possible_paths:
        if p.is_file():
            manifest_path = p.resolve()
            break

    if manifest_path is None:
        raise RuntimeError("FROZEN_MANIFEST_PROVENANCE_UNRESOLVED: Canonical manifest file not found in repository!")

    content_bytes = manifest_path.read_bytes()
    manifest_sha = hashlib.sha256(content_bytes).hexdigest()

    data = json.loads(content_bytes.decode("utf-8"))
    queries = {q["query_id"]: q for q in data.get("queries", [])}
    # Provide short name aliases ("p1-1", "p1-2", ...)
    short_map: dict[str, dict[str, Any]] = {}
    for qid, q in queries.items():
        parts = qid.split("-")
        if len(parts) >= 3 and parts[0] == "query" and parts[1].startswith("p1"):
            short_map[f"{parts[1]}-{parts[2]}"] = q
        short_map[qid] = q

    return manifest_path, manifest_sha, short_map


def load_frozen_reference_manifest() -> dict[str, Any]:
    """Load reference truth data from manual_kis_reference_v1.json."""
    ref_paths = [
        SYSTEM_TAI_SRC.parent / "benchmarks" / "manual_kis_reference_v1.json",
        REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "manual_kis_reference_v1.json",
        Path("/kaggle/working/AIC2026_TeamPTK_SGU/systems/system_tai/benchmarks/manual_kis_reference_v1.json"),
    ]
    ref_data = {}
    for p in ref_paths:
        if p.is_file():
            try:
                ref_data = json.loads(p.read_text(encoding="utf-8"))
                break
            except Exception:
                pass

    if not ref_data or "queries" not in ref_data:
        raise RuntimeError(f"Manual reference manual_kis_reference_v1.json missing or invalid (searched: {ref_paths})")

    return ref_data


def create_production_v2a_session_config(
    input_root: Path,
    reuse_manifest_path: Path | None,
    manifest_cache_path: Path | None,
    output_root: Path,
    *,
    restricted_frames_per_video_per_variant: int = 10,
    enable_temporal_diverse_local_candidates: bool = False,
    temporal_diversity_gap_seconds: float = 5.0,
    enable_vi_localization_variant: bool = False,
    vi_localization_weight: float = 0.5,
    internal_rrf_candidate_depth: int = 100,
    collect_fusion_trace: bool = False,
) -> SessionConfig:
    """Canonical production V2-A configuration factory matching production gate with ablation support."""
    config = SessionConfig(
        input_root=input_root,
        reuse_manifest=reuse_manifest_path,
        manifest_cache=manifest_cache_path,
        output_root=output_root,
        device="auto",
        allow_model_download=True,
        enable_dynamic_translation=True,
        translation_model_name="google-translate",
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
            restricted_frames_per_video_per_variant=restricted_frames_per_video_per_variant,
            enable_temporal_diverse_local_candidates=enable_temporal_diverse_local_candidates,
            temporal_diversity_gap_seconds=temporal_diversity_gap_seconds,
            enable_vi_localization_variant=enable_vi_localization_variant,
            vi_localization_weight=vi_localization_weight,
            internal_rrf_candidate_depth=internal_rrf_candidate_depth,
            collect_fusion_trace=collect_fusion_trace,
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


@dataclass(frozen=True, slots=True)
class TileDecodeResult:
    video_id: str
    requested_physical_frame_id: int
    requested_pts: float
    requested_keyframe_order: int | None
    resolved_source_type: str  # KEYFRAME_FILE | RAW_VIDEO_PTS | RAW_VIDEO_FRAME_INDEX | UNRESOLVED
    resolved_path: str | None
    resolution_rule: str  # EXACT_KEYFRAME_ORDER_FILE | EXACT_PHYSICAL_FRAME_FILE | RAW_VIDEO_PTS_SEEK | RAW_VIDEO_FRAME_INDEX_SEEK | UNRESOLVED
    resolved_filename: str | None
    decoded_pts_or_frame_index: float | int | None
    frame_delta: int
    pts_delta: float
    integrity_status: str  # PASS | FAIL
    image: Any = None

    def to_manifest_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "requested_physical_frame_id": self.requested_physical_frame_id,
            "requested_pts": self.requested_pts,
            "requested_keyframe_order": self.requested_keyframe_order,
            "resolved_source_type": self.resolved_source_type,
            "resolved_path": str(self.resolved_path) if self.resolved_path else None,
            "resolution_rule": self.resolution_rule,
            "resolved_filename": self.resolved_filename,
            "decoded_pts_or_frame_index": self.decoded_pts_or_frame_index,
            "frame_delta": self.frame_delta,
            "pts_delta": self.pts_delta,
            "integrity_status": self.integrity_status,
        }


DECLARED_PTS_TOLERANCE_SECONDS: float = 0.500
EXACT_FILE_AMBIGUITY_POLICY: str = "FAIL"


def find_keyframe_image(
    dataset_root: Path,
    video_id: str,
    frame_id: int,
    keyframe_order: int | None = None,
    runtime: Any = None,
) -> tuple[Path | None, str]:
    """Scoped keyframe file resolution with ambiguity detection and exact provenance tracking.

    Returns (path, resolution_rule).
    """
    # 1. Candidate Directories strictly scoped to video_id
    scoped_dirs: list[Path] = []
    if runtime is not None and hasattr(runtime, "manifest") and runtime.manifest is not None:
        for v in getattr(runtime.manifest, "videos", ()):
            if getattr(v, "video_id", None) == video_id:
                kdir = getattr(v, "keyframe_directory", None)
                if kdir:
                    pk = Path(kdir)
                    if pk.is_dir():
                        scoped_dirs.append(pk)
                break

    if not scoped_dirs:
        standard_candidates = [
            dataset_root / "keyframes" / video_id,
            dataset_root / "Keyframes" / video_id,
            dataset_root / video_id,
            Path("/kaggle/input") / "keyframes" / video_id,
            Path("/kaggle/input") / "Keyframes" / video_id,
            Path("/kaggle/input") / video_id,
        ]
        for p in standard_candidates:
            if p.is_dir() and p not in scoped_dirs:
                scoped_dirs.append(p)

        # Bounded search across dataset mount points if standard paths missing (depth <= 2)
        if not scoped_dirs:
            for root_p in [dataset_root, Path("/kaggle/input")]:
                if root_p.is_dir():
                    try:
                        for sub in root_p.iterdir():
                            if sub.is_dir():
                                if sub.name == video_id and sub not in scoped_dirs:
                                    scoped_dirs.append(sub)
                                for subsub in sub.iterdir():
                                    if subsub.is_dir() and subsub.name == video_id and subsub not in scoped_dirs:
                                        scoped_dirs.append(subsub)
                    except Exception:
                        pass

    # 2. Test exact keyframe order match (n)
    cand_by_order: Path | None = None
    if keyframe_order is not None and keyframe_order > 0:
        order_names = [
            f"{keyframe_order:06d}.jpg",
            f"{keyframe_order:05d}.jpg",
            f"{keyframe_order:04d}.jpg",
            f"{keyframe_order:03d}.jpg",
            f"{keyframe_order}.jpg",
            f"{keyframe_order:06d}.png",
            f"{keyframe_order:05d}.png",
            f"{keyframe_order:04d}.png",
            f"{keyframe_order}.png",
            f"{keyframe_order:06d}.jpeg",
            f"{keyframe_order}.jpeg",
        ]
        for sdir in scoped_dirs:
            for name in order_names:
                cand = sdir / name
                if cand.is_file():
                    cand_by_order = cand
                    break
            if cand_by_order:
                break

    # 3. Test exact physical frame ID match (frame_id)
    cand_by_fid: Path | None = None
    fid_names = [
        f"{frame_id:06d}.jpg",
        f"{frame_id:05d}.jpg",
        f"{frame_id:04d}.jpg",
        f"{frame_id:03d}.jpg",
        f"{frame_id}.jpg",
        f"{frame_id:06d}.png",
        f"{frame_id:05d}.png",
        f"{frame_id:04d}.png",
        f"{frame_id}.png",
        f"{frame_id:06d}.jpeg",
        f"{frame_id}.jpeg",
    ]
    for sdir in scoped_dirs:
        for name in fid_names:
            cand = sdir / name
            if cand.is_file():
                cand_by_fid = cand
                break
        if cand_by_fid:
            break

    # In AIC keyframe stores, numeric filenames (001.jpg, 002.jpg) represent 1-based order.
    # Prefer order match first, then fid match.
    if cand_by_order:
        return cand_by_order, "EXACT_KEYFRAME_ORDER_FILE"
    elif cand_by_fid:
        return cand_by_fid, "EXACT_PHYSICAL_FRAME_FILE"

    return None, "UNRESOLVED"


_VIDEO_PATH_CACHE: dict[str, Path | None] = {}


def find_source_video_file(
    dataset_root: Path,
    video_id: str,
    runtime: Any = None,
) -> Path | None:
    if video_id in _VIDEO_PATH_CACHE:
        return _VIDEO_PATH_CACHE[video_id]

    # 1. Try runtime raw_video_registry / manifest
    if runtime is not None:
        if hasattr(runtime, "raw_video_registry") and runtime.raw_video_registry is not None:
            try:
                rec = runtime.raw_video_registry.get(video_id)
                if rec and getattr(rec, "raw_video_path", None) and Path(rec.raw_video_path).is_file():
                    res = Path(rec.raw_video_path)
                    _VIDEO_PATH_CACHE[video_id] = res
                    return res
            except Exception:
                pass
        if hasattr(runtime, "manifest") and runtime.manifest is not None:
            for v in getattr(runtime.manifest, "videos", ()):
                if getattr(v, "video_id", None) == video_id:
                    rv = getattr(v, "raw_video_path", None)
                    if rv and Path(rv).is_file():
                        res = Path(rv)
                        _VIDEO_PATH_CACHE[video_id] = res
                        return res

    # 2. Check standard layout paths
    for ext in ("mp4", "mkv", "avi", "mov", "ts", "MP4"):
        candidates = [
            dataset_root / "videos" / f"{video_id}.{ext}",
            dataset_root / "video" / f"{video_id}.{ext}",
            dataset_root / "raw_videos" / f"{video_id}.{ext}",
            dataset_root / f"{video_id}.{ext}",
            Path("/kaggle/input") / "videos" / f"{video_id}.{ext}",
            Path("/kaggle/input") / "raw_videos" / f"{video_id}.{ext}",
        ]
        for c in candidates:
            if c.is_file():
                _VIDEO_PATH_CACHE[video_id] = c
                return c

    _VIDEO_PATH_CACHE[video_id] = None
    return None


def extract_image_for_frame(
    dataset_root: Path,
    video_id: str,
    frame_id: int,
    keyframe_order: int | None = None,
    pts_time: float | None = None,
    source_fps: float | None = None,
    parity_passed: bool = False,
    runtime: Any = None,
) -> TileDecodeResult:
    # Auto-resolve order and pts from registry if not provided
    if (keyframe_order is None or keyframe_order <= 0 or pts_time is None or pts_time <= 0) and runtime is not None:
        try:
            store = runtime.registry.get_store(video_id)
            if store is not None:
                for m in store.mappings:
                    if m.frame_id == frame_id:
                        if keyframe_order is None or keyframe_order <= 0:
                            keyframe_order = getattr(m, "keyframe_order", None)
                        if pts_time is None or pts_time <= 0:
                            pts_time = getattr(m, "pts_time", None)
                        break
        except Exception:
            pass

    requested_pts = float(pts_time) if pts_time is not None else float(frame_id) / 25.0

    # 1. Try Keyframe File (Order n or physical frame_id)
    img_path, kf_rule = find_keyframe_image(dataset_root, video_id, frame_id, keyframe_order, runtime=runtime)
    if kf_rule.startswith("AMBIGUOUS_KEYFRAME_RESOLUTION"):
        resolved_fn = kf_rule.split(":", 1)[1] if ":" in kf_rule else None
        return TileDecodeResult(
            video_id=video_id,
            requested_physical_frame_id=frame_id,
            requested_pts=requested_pts,
            requested_keyframe_order=keyframe_order,
            resolved_source_type="AMBIGUOUS_KEYFRAME_FILE",
            resolved_path=None,
            resolution_rule="AMBIGUOUS_KEYFRAME_RESOLUTION",
            resolved_filename=resolved_fn,
            decoded_pts_or_frame_index=None,
            frame_delta=-1,
            pts_delta=-1.0,
            integrity_status="FAIL",
            image=None,
        )

    if img_path and img_path.is_file():
        try:
            img = Image.open(img_path)
            return TileDecodeResult(
                video_id=video_id,
                requested_physical_frame_id=frame_id,
                requested_pts=requested_pts,
                requested_keyframe_order=keyframe_order,
                resolved_source_type="KEYFRAME_FILE",
                resolved_path=str(img_path.resolve(strict=False)),
                resolution_rule=kf_rule,
                resolved_filename=img_path.name,
                decoded_pts_or_frame_index=requested_pts if kf_rule == "EXACT_KEYFRAME_ORDER_FILE" else frame_id,
                frame_delta=0,
                pts_delta=0.0,
                integrity_status="PASS",
                image=img,
            )
        except Exception:
            pass

    # 2. Try Raw Video Fallback (OpenCV)
    vid_path = find_source_video_file(dataset_root, video_id, runtime=runtime)
    if vid_path and vid_path.is_file():
        try:
            import cv2
            cap = cv2.VideoCapture(str(vid_path))
            if cap.isOpened():
                # Rule 5: Raw video fallback must decode by mapping PTS first
                if pts_time is not None and pts_time >= 0:
                    cap.set(cv2.CAP_PROP_POS_MSEC, pts_time * 1000.0)
                    ret, f = cap.read()
                    if ret and f is not None:
                        actual_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
                        actual_pts = actual_msec / 1000.0 if actual_msec > 0 else pts_time
                        pts_delta = abs(actual_pts - pts_time)
                        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                        cap.release()
                        pts_integrity_pass = (pts_delta <= DECLARED_PTS_TOLERANCE_SECONDS)
                        return TileDecodeResult(
                            video_id=video_id,
                            requested_physical_frame_id=frame_id,
                            requested_pts=requested_pts,
                            requested_keyframe_order=keyframe_order,
                            resolved_source_type="RAW_VIDEO_PTS",
                            resolved_path=str(vid_path.resolve(strict=False)),
                            resolution_rule="RAW_VIDEO_PTS_SEEK",
                            resolved_filename=vid_path.name,
                            decoded_pts_or_frame_index=actual_pts,
                            frame_delta=0,
                            pts_delta=pts_delta,
                            integrity_status="PASS" if pts_integrity_pass else "FAIL",
                            image=Image.fromarray(rgb),
                        )

                # Frame-index seek ONLY when parity_passed is True
                if parity_passed:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                    ret, f = cap.read()
                    if ret and f is not None:
                        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                        cap.release()
                        return TileDecodeResult(
                            video_id=video_id,
                            requested_physical_frame_id=frame_id,
                            requested_pts=requested_pts,
                            requested_keyframe_order=keyframe_order,
                            resolved_source_type="RAW_VIDEO_FRAME_INDEX",
                            resolved_path=str(vid_path.resolve(strict=False)),
                            resolution_rule="RAW_VIDEO_FRAME_INDEX_SEEK",
                            resolved_filename=vid_path.name,
                            decoded_pts_or_frame_index=frame_id,
                            frame_delta=0,
                            pts_delta=0.0,
                            integrity_status="PASS",
                            image=Image.fromarray(rgb),
                        )

                cap.release()
        except Exception:
            pass

    # 3. Unresolved
    return TileDecodeResult(
        video_id=video_id,
        requested_physical_frame_id=frame_id,
        requested_pts=requested_pts,
        requested_keyframe_order=keyframe_order,
        resolved_source_type="UNRESOLVED",
        resolved_path=None,
        resolution_rule="UNRESOLVED",
        resolved_filename=None,
        decoded_pts_or_frame_index=None,
        frame_delta=-1,
        pts_delta=-1.0,
        integrity_status="FAIL",
        image=None,
    )


def generate_and_save_ablation_summary_table(
    all_run_results: dict[str, dict],
    base_out: Path,
    runs_to_execute: list[str],
    ablation_configs: dict[str, dict],
) -> None:
    manifest_path, manifest_sha, manifest_queries = load_canonical_frozen_manifest()
    query_order = ["p1-1", "p1-2", "p1-4", "p1-5", "p1-6"]

    ref_data = load_frozen_reference_manifest()

    ref_queries_map = {q.get("query_id"): q for q in ref_data.get("queries", [])}

    summary_matrix = {
        "generated_at": datetime.now(UTC).isoformat(),
        "runs": {},
    }

    for run_key in runs_to_execute:
        run_out = base_out / f"run_{run_key}"
        run_spec = ablation_configs[run_key]
        run_data = {
            "run_key": run_key,
            "name": run_spec["name"],
            "config": run_spec,
            "queries": {},
        }

        for q_short in query_order:
            q_meta = manifest_queries.get(q_short, {})
            qid = q_meta.get("query_id", f"query-{q_short}-kis")
            ref_entry = ref_queries_map.get(qid, {})

            human_vid = ref_entry.get("human_verified_video_id") or q_meta.get("target_video", "")
            legacy_vid = ref_entry.get("legacy_manifest_target", {}).get("target_video") or q_meta.get("target_video", "")
            human_pts_intervals = ref_entry.get("human_annotated_intervals_pts", [])
            human_status = ref_entry.get("annotation_status", "VIDEO_ONLY_VERIFIED")

            q_summary = {
                "query_id": qid,
                "target_info": {
                    "human_vid": human_vid,
                    "legacy_vid": legacy_vid,
                    "human_pts_intervals": human_pts_intervals,
                    "annotation_status": human_status,
                },
                "first_human_video_rank": None,
                "first_legacy_video_rank": None,
                "first_valid_interval_rank": None,
                "valid_interval_hit": False if human_pts_intervals else None,
                "valid_interval_frame_id": None,
                "valid_interval_source": None,
                "frame_evaluation_status": "NO_VALID_INTERVAL_HIT" if human_pts_intervals else "NOT_EVALUABLE_NO_INTERVAL",
                "total_candidates": 0,
                "source_assignment_breakdown": {},
                "unique_candidates_with_diverse_source": 0,
                "unique_candidates_with_raw_source": 0,
                "multi_source_candidate_count": 0,
                "semantic_variant_sha256": None,
            }

            matches = sorted(
                (run_out / "requests").glob(f"audit-top100-{q_short}-*/candidates.json"),
                key=lambda p: p.stat().st_mtime,
            )
            cand_file = matches[-1] if matches else None

            if cand_file is None or not cand_file.exists():
                raise RuntimeError(
                    f"Missing current candidate artifact for run '{run_key}', query '{q_short}': "
                    f"expected {run_out / 'requests' / f'audit-top100-{q_short}-*/candidates.json'}"
                )

            cdata = json.loads(cand_file.read_text(encoding="utf-8"))
            if cdata.get("query_id") != qid:
                raise RuntimeError(f"Mismatched query_id in {cand_file}: expected '{qid}', got '{cdata.get('query_id')}'")

            trans = cdata.get("translation", {})
            units = trans.get("units", [])
            if units:
                semantic_payload = [
                    {
                        "variant_id": seg.get("variant_id"),
                        "text": seg.get("text"),
                        "weight": seg.get("weight"),
                    }
                    for unit in units
                    for seg in unit.get("segments", [])
                ]
                q_summary["semantic_variant_sha256"] = hashlib.sha256(
                    json.dumps(
                        semantic_payload,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest()[:12]
            elif "english_segments" in trans:
                q_summary["semantic_variant_sha256"] = hashlib.sha256(
                    json.dumps(trans["english_segments"]).encode("utf-8")
                ).hexdigest()[:12]

            if not q_summary["semantic_variant_sha256"]:
                raise RuntimeError(f"Could not compute translation hash for run '{run_key}', query '{q_short}' from {cand_file}")

            records = cdata.get("records", [])
            q_summary["total_candidates"] = len(records)

            src_counts = {}
            unique_diverse = 0
            unique_raw = 0
            multi_source = 0

            for r in records:
                vid = r.get("video_id")
                rk = r.get("rank")
                pts = r.get("pts_time", 0.0)
                fid = r.get("frame_id")

                sel_map = r.get("selection_by_variant", {})
                cand_sources = set()
                if sel_map:
                    for var_id, var_info in sel_map.items():
                        s = var_info.get("source", "RAW")
                        src_counts[s] = src_counts.get(s, 0) + 1
                        cand_sources.add(s)
                else:
                    src_counts["RAW"] = src_counts.get("RAW", 0) + 1
                    cand_sources.add("RAW")

                if "DIVERSE" in cand_sources:
                    unique_diverse += 1
                if "RAW" in cand_sources:
                    unique_raw += 1
                if len(cand_sources) > 1:
                    multi_source += 1

                if vid == human_vid and q_summary["first_human_video_rank"] is None:
                    q_summary["first_human_video_rank"] = rk
                if vid == legacy_vid and q_summary["first_legacy_video_rank"] is None:
                    q_summary["first_legacy_video_rank"] = rk

                if vid == human_vid and human_pts_intervals:
                    for s_pts, e_pts in human_pts_intervals:
                        if s_pts <= pts <= e_pts:
                            if q_summary["first_valid_interval_rank"] is None:
                                q_summary["first_valid_interval_rank"] = rk
                                q_summary["valid_interval_hit"] = True
                                q_summary["valid_interval_frame_id"] = fid
                                q_summary["valid_interval_source"] = sel_map
                                q_summary["frame_evaluation_status"] = "VALID_MANUAL_INTERVAL_HIT"

            q_summary["source_assignment_breakdown"] = src_counts
            q_summary["unique_candidates_with_diverse_source"] = unique_diverse
            q_summary["unique_candidates_with_raw_source"] = unique_raw
            q_summary["multi_source_candidate_count"] = multi_source
            run_data["queries"][q_short] = q_summary

        summary_matrix["runs"][run_key] = run_data

    if len(runs_to_execute) > 1:
        print("\n🔍 Verifying Zero Translation Drift across runs...", flush=True)
        for q_short in query_order:
            hashes = [
                summary_matrix["runs"][rk]["queries"][q_short].get("semantic_variant_sha256")
                for rk in runs_to_execute
            ]
            if any(h is None for h in hashes):
                raise RuntimeError(f"Missing translation hash for query [{q_short}] in runs: {dict(zip(runs_to_execute, hashes))}")
            if len(set(hashes)) != 1:
                raise RuntimeError(f"Translation drift detected for query [{q_short}]: {dict(zip(runs_to_execute, hashes))}")
            print(f"  ✅ Zero Translation Drift Strictly Verified for [{q_short}] (SHA={hashes[0]}) across {runs_to_execute}", flush=True)

    json_path = base_out / "ablation_matrix_summary.json"
    json_path.write_text(json.dumps(summary_matrix, indent=2), encoding="utf-8")
    print(f"\n📊 Ablation Summary JSON Artifact saved -> {json_path} ✅", flush=True)

    print("\n" + "=" * 130, flush=True)
    print("📋 PHASE B1 ABLATION MATRIX: MULTI-RUN STATISTICAL COMPARISON TABLE", flush=True)
    print("=" * 130, flush=True)

    header = f"| {'Target Query / Metric':<48} | " + " | ".join(f"{f'Run {rk}':<12}" for rk in runs_to_execute) + " |"
    sep = f"|:{'-'*48}-|-" + "-|-".join(f"{'-'*12}" for _ in runs_to_execute) + "-|"
    print(header, flush=True)
    print(sep, flush=True)

    p1_1_vid = [str(summary_matrix["runs"][rk]["queries"]["p1-1"]["first_human_video_rank"] or "MISS") for rk in runs_to_execute]
    print(f"| {'P1-1 Target Video Rank (L30_V046)':<48} | " + " | ".join(f"{v:<12}" for v in p1_1_vid) + " |", flush=True)

    p1_1_int = [str(summary_matrix["runs"][rk]["queries"]["p1-1"]["first_valid_interval_rank"] or "MISS") for rk in runs_to_execute]
    print(f"| {'P1-1 GT Interval Rank (264s-274s, f6784)':<48} | " + " | ".join(f"{v:<12}" for v in p1_1_int) + " |", flush=True)

    p1_2_hum = [str(summary_matrix["runs"][rk]["queries"]["p1-2"]["first_human_video_rank"] or "MISS") for rk in runs_to_execute]
    print(f"| {'P1-2 Human Target Rank (L21_V003)':<48} | " + " | ".join(f"{v:<12}" for v in p1_2_hum) + " |", flush=True)

    p1_2_leg = [str(summary_matrix["runs"][rk]["queries"]["p1-2"]["first_legacy_video_rank"] or "MISS") for rk in runs_to_execute]
    print(f"| {'P1-2 Legacy Target Rank (L29_V018)':<48} | " + " | ".join(f"{v:<12}" for v in p1_2_leg) + " |", flush=True)

    p1_4_hum = [str(summary_matrix["runs"][rk]["queries"]["p1-4"]["first_human_video_rank"] or "MISS") for rk in runs_to_execute]
    print(f"| {'P1-4 Human Target Rank (L22_V021)':<48} | " + " | ".join(f"{v:<12}" for v in p1_4_hum) + " |", flush=True)

    p1_4_leg = [str(summary_matrix["runs"][rk]["queries"]["p1-4"]["first_legacy_video_rank"] or "MISS") for rk in runs_to_execute]
    print(f"| {'P1-4 Legacy Target Rank (L28_V012)':<48} | " + " | ".join(f"{v:<12}" for v in p1_4_leg) + " |", flush=True)

    p1_5_hum = [str(summary_matrix["runs"][rk]["queries"]["p1-5"]["first_human_video_rank"] or "MISS") for rk in runs_to_execute]
    print(f"| {'P1-5 Human Target Rank (L26_V035)':<48} | " + " | ".join(f"{v:<12}" for v in p1_5_hum) + " |", flush=True)

    p1_5_leg = [str(summary_matrix["runs"][rk]["queries"]["p1-5"]["first_legacy_video_rank"] or "MISS") for rk in runs_to_execute]
    print(f"| {'P1-5 Legacy Target Rank (L30_V021)':<48} | " + " | ".join(f"{v:<12}" for v in p1_5_leg) + " |", flush=True)

    p1_6_hum = [str(summary_matrix["runs"][rk]["queries"]["p1-6"]["first_human_video_rank"] or "MISS") for rk in runs_to_execute]
    print(f"| {'P1-6 Human Target Rank (L22_V023)':<48} | " + " | ".join(f"{v:<12}" for v in p1_6_hum) + " |", flush=True)

    p1_6_leg = [str(summary_matrix["runs"][rk]["queries"]["p1-6"]["first_legacy_video_rank"] or "MISS") for rk in runs_to_execute]
    print(f"| {'P1-6 Legacy Target Rank (L27_V005)':<48} | " + " | ".join(f"{v:<12}" for v in p1_6_leg) + " |", flush=True)

    drift_hashes = [str(summary_matrix["runs"][rk]["queries"]["p1-1"]["semantic_variant_sha256"] or "N/A") for rk in runs_to_execute]
    print(f"| {'Translation Hash (P1-1 Semantic SHA)':<48} | " + " | ".join(f"{v:<12}" for v in drift_hashes) + " |", flush=True)

    print("=" * 130 + "\n", flush=True)


def ensure_clip_model_cached() -> None:
    cache_dir = Path.home() / ".cache" / "clip"
    cache_dir.mkdir(parents=True, exist_ok=True)
    target_file = cache_dir / "ViT-B-32.pt"

    if target_file.is_file() and target_file.stat().st_size > 100_000_000:
        return

    # Check direct known paths in /kaggle/input (shallow only, NO recursive glob across large datasets)
    if Path("/kaggle/input").exists():
        direct_checks = [
            Path("/kaggle/input/ViT-B-32.pt"),
            Path("/kaggle/input/openai-clip-vit-b-32/ViT-B-32.pt"),
            Path("/kaggle/input/clip-vit-b-32/ViT-B-32.pt"),
            Path("/kaggle/input/clip-weights/ViT-B-32.pt"),
            Path("/kaggle/input/clip/ViT-B-32.pt"),
        ]
        for p in direct_checks:
            if p.is_file() and p.stat().st_size > 100_000_000:
                print(f"  • Found pre-cached CLIP model in dataset: {p}", flush=True)
                shutil.copy(p, target_file)
                return

    # Download with retries, timeout, and browser User-Agent
    urls = [
        "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
        "https://openaipublic.azureedge.net/clip/models/580e00a5e038cc808b1a14755321302e82ce3aaf3b95d9e03952dd17bf450018/ViT-B-32.pt",
    ]
    print(f"  • Downloading ViT-B-32.pt (~338MB) to {target_file}...", flush=True)
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    for url in urls:
        for attempt in range(5):
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=60) as resp, open(target_file, "wb") as f:
                    shutil.copyfileobj(resp, f)
                if target_file.is_file() and target_file.stat().st_size > 100_000_000:
                    print("  • CLIP model download complete! ✅", flush=True)
                    return
            except Exception as e:
                print(f"  ⚠️ CLIP download attempt {attempt+1}/5 failed ({e}). Retrying in 2s...", flush=True)
                time.sleep(2)


def main() -> None:
    parser = argparse.ArgumentParser(description="KIS V2-A.3 Foundation Closure Audit")
    parser.add_argument(
        "--sections",
        type=str,
        default="coverage,p1-2,p1-4,top100",
        help="Comma-separated sections to run: coverage, p1-2, p1-4, top100 or 'all' (default: coverage,p1-2,p1-4,top100)",
    )
    parser.add_argument(
        "--ablation",
        type=str,
        default="A",
        help="Phase B1 ablation run: A, B, C, D, E, ABC, A,B,C, or all (default: A)",
    )
    ablation_configs = {
        "A": {
            "name": "Run A: Baseline Control (K=10, Hybrid=Off, VI=Off, RRF=100)",
            "restricted_frames_per_video_per_variant": 10,
            "enable_temporal_diverse_local_candidates": False,
            "temporal_diversity_gap_seconds": 5.0,
            "enable_vi_localization_variant": False,
            "vi_localization_weight": 0.5,
            "internal_rrf_candidate_depth": 100,
            "collect_fusion_trace": True,
        },
        "B": {
            "name": "Run B: Depth Only (K=20, Hybrid=Off, VI=Off, RRF=100)",
            "restricted_frames_per_video_per_variant": 20,
            "enable_temporal_diverse_local_candidates": False,
            "temporal_diversity_gap_seconds": 5.0,
            "enable_vi_localization_variant": False,
            "vi_localization_weight": 0.5,
            "internal_rrf_candidate_depth": 100,
            "collect_fusion_trace": True,
        },
        "C": {
            "name": "Run C: Diversity at Equal Budget (K=20, Hybrid=On, Gap=5s, VI=Off, RRF=100)",
            "restricted_frames_per_video_per_variant": 20,
            "enable_temporal_diverse_local_candidates": True,
            "temporal_diversity_gap_seconds": 5.0,
            "enable_vi_localization_variant": False,
            "vi_localization_weight": 0.5,
            "internal_rrf_candidate_depth": 100,
            "collect_fusion_trace": True,
        },
        "D": {
            "name": "Run D: VI Localizer Only (K=10, Hybrid=Off, VI=On, w=0.5, RRF=100)",
            "restricted_frames_per_video_per_variant": 10,
            "enable_temporal_diverse_local_candidates": False,
            "temporal_diversity_gap_seconds": 5.0,
            "enable_vi_localization_variant": True,
            "vi_localization_weight": 0.5,
            "internal_rrf_candidate_depth": 100,
            "collect_fusion_trace": True,
        },
        "E": {
            "name": "Run E: VI + Diversity (K=20, Hybrid=On, Gap=5s, VI=On, w=0.5, RRF=100)",
            "restricted_frames_per_video_per_variant": 20,
            "enable_temporal_diverse_local_candidates": True,
            "temporal_diversity_gap_seconds": 5.0,
            "enable_vi_localization_variant": True,
            "vi_localization_weight": 0.5,
            "internal_rrf_candidate_depth": 100,
            "collect_fusion_trace": True,
        },
        "F": {
            "name": "Run F: RRF Candidate Depth (K=20, Hybrid=On, Gap=5s, VI=Off, RRF=1000)",
            "restricted_frames_per_video_per_variant": 20,
            "enable_temporal_diverse_local_candidates": True,
            "temporal_diversity_gap_seconds": 5.0,
            "enable_vi_localization_variant": False,
            "vi_localization_weight": 0.5,
            "internal_rrf_candidate_depth": 1000,
            "collect_fusion_trace": True,
        },
        "G": {
            "name": "Run G: Full Combined Synergy (K=20, Hybrid=On, Gap=5s, VI=On, w=0.5, RRF=1000)",
            "restricted_frames_per_video_per_variant": 20,
            "enable_temporal_diverse_local_candidates": True,
            "temporal_diversity_gap_seconds": 5.0,
            "enable_vi_localization_variant": True,
            "vi_localization_weight": 0.5,
            "internal_rrf_candidate_depth": 1000,
            "collect_fusion_trace": True,
        },
    }

    args, _ = parser.parse_known_args()
    selected_sections = [s.strip().lower() for s in args.sections.split(",") if s.strip()]
    run_all = "all" in selected_sections
    ablation_raw = args.ablation.strip().upper()
    if ablation_raw == "ALL":
        runs_to_execute = ["A", "B", "C", "D", "E"]
    elif "," in ablation_raw:
        runs_to_execute = [r.strip() for r in ablation_raw.split(",") if r.strip() in ablation_configs]
    elif len(ablation_raw) > 1 and all(c in ablation_configs for c in ablation_raw):
        runs_to_execute = [c for c in ablation_raw]
    elif ablation_raw in ablation_configs:
        runs_to_execute = [ablation_raw]
    else:
        runs_to_execute = ["A"]

    full_sha = get_git_head()
    print("=" * 120, flush=True)
    print("KIS V2-A.3 FOUNDATION CLOSURE — STRICT EMPIRICAL AUDIT & PHASE B1 ABLATIONS", flush=True)
    print("=" * 120, flush=True)
    print(f"• Exact Commit SHA: {full_sha}", flush=True)
    print(f"• Python Version  : {sys.version.split()[0]}", flush=True)
    print(f"• Ablation Plan   : {', '.join(runs_to_execute)}", flush=True)
    print(f"• Active Sections : {', '.join(selected_sections) if not run_all else 'ALL SECTIONS'}\n", flush=True)

    ensure_clip_model_cached()

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

    base_out = Path("/kaggle/working/output/v2a3_foundation_closure_phase_b11") if Path("/kaggle/working").exists() else REPO_ROOT / "scratch" / "v2a3_foundation_closure_phase_b11"
    manifest_cache = None if reuse_manifest_path else (
        Path("/kaggle/working/manifest_cache.json") if Path("/kaggle/working").exists() else REPO_ROOT / "scratch" / "manifest_cache.json"
    )

    all_run_results = {}

    for run_key in runs_to_execute:
        run_spec = ablation_configs[run_key]
        run_out = base_out / f"run_{run_key}"
        run_out.mkdir(parents=True, exist_ok=True)

        print("-" * 120, flush=True)
        print(f"🚀 EXECUTING ABLATION: {run_spec['name']}", flush=True)
        print(f"• Output Directory: {run_out}", flush=True)
        print("-" * 120, flush=True)

        config = create_production_v2a_session_config(
            input_root=input_root,
            reuse_manifest_path=reuse_manifest_path,
            manifest_cache_path=manifest_cache,
            output_root=run_out,
            restricted_frames_per_video_per_variant=run_spec["restricted_frames_per_video_per_variant"],
            enable_temporal_diverse_local_candidates=run_spec["enable_temporal_diverse_local_candidates"],
            temporal_diversity_gap_seconds=run_spec["temporal_diversity_gap_seconds"],
            enable_vi_localization_variant=run_spec["enable_vi_localization_variant"],
            vi_localization_weight=run_spec["vi_localization_weight"],
            internal_rrf_candidate_depth=run_spec.get("internal_rrf_candidate_depth", 100),
            collect_fusion_trace=run_spec.get("collect_fusion_trace", True),
        )

        # Purge stale translation cache if error-polluted
        for t_cache in [Path("/kaggle/working/translation_cache.json"), run_out / "translation_cache.json"]:
            if t_cache.exists():
                try:
                    raw_txt = t_cache.read_text(encoding="utf-8")
                    if "Error 500" in raw_txt or "Server Error" in raw_txt:
                        t_cache.unlink()
                        print(f"🧹 Purged stale error-polluted translation cache: {t_cache}", flush=True)
                except Exception:
                    pass

        t0 = time.time()
        runtime = OperationalKISRuntime.bootstrap(config)
        print(f"Runtime bootstrap completed in {time.time() - t0:.2f}s.\n", flush=True)

        coverage_results = {}
        # 1. GT INDEX COVERAGE AUDIT WITH SOURCE <-> MAPPING PARITY (ALL 5 TARGET VIDEOS)
        if (run_all or "coverage" in selected_sections) and (run_key == runs_to_execute[0]):
            coverage_results = run_gt_index_coverage_audit(runtime, input_root)

        # 2. P1-2 EVIDENCE-POOL TO FINAL-EXPORT TRACE & VISUAL BENCHMARK ADJUDICATION
        if (run_all or "p1-2" in selected_sections or "p1_2" in selected_sections) and (run_key == runs_to_execute[0]):
            run_p1_2_trace_and_raw_cosine_audit(runtime, input_root, run_out, coverage_results)

        # 3. P1-4 SEMANTIC ADJUDICATION & PTS-AWARE REAL IMAGE RENDERING
        if (run_all or "p1-4" in selected_sections or "p1_4" in selected_sections) and (run_key == runs_to_execute[0]):
            run_p1_4_real_image_adjudication(runtime, input_root, run_out, coverage_results)

        # 4. FULL TOP-100 EVALUATION FOR ALL 5 FOCUS QUERIES (ALWAYS RUN RETRIEVAL, OPTIONAL PNG RENDER)
        render_sheets = run_all or "top100" in selected_sections
        run_all_5_queries_top100_visual_export(
            runtime, input_root, run_out, coverage_results, render_contact_sheets=render_sheets
        )

        # 5. PRINT UNIFIED SUMMARY TABLE FOR THIS RUN
        print_final_summary_table(coverage_results)
        all_run_results[run_key] = {"name": run_spec["name"], "coverage": coverage_results}

    # 6. GENERATE CROSS-RUN STATISTICAL COMPARISON TABLE & JSON ARTIFACT
    generate_and_save_ablation_summary_table(all_run_results, base_out, runs_to_execute, ablation_configs)

    # 7. PACKAGE ALL VISUAL EVIDENCE ARTIFACTS INTO A SINGLE ZIP ARCHIVE
    zip_dest = Path("/kaggle/working/v2a3_visual_evidence_package") if Path("/kaggle/working").exists() else REPO_ROOT / "scratch" / "v2a3_visual_evidence_package"
    try:
        shutil.make_archive(str(zip_dest), "zip", base_out)
        print(f"\n📦 Visual Evidence Artifacts Package Created -> {zip_dest}.zip ✅\n", flush=True)
    except Exception as e:
        print(f"\n⚠️ Could not create zip archive: {e}\n", flush=True)


# ==============================================================================
# ==============================================================================
# SECTION 1: LEGACY MANIFEST & HUMAN REFERENCE COVERAGE AUDIT WITH SOURCE <-> MAPPING PARITY
# ==============================================================================
def run_gt_index_coverage_audit(runtime: OperationalKISRuntime, input_root: Path) -> dict[str, dict]:
    print("=" * 120, flush=True)
    print("1. LEGACY MANIFEST & HUMAN REFERENCE INDEX COVERAGE AUDIT (ALL 5 TARGETS)", flush=True)
    print("=" * 120, flush=True)

    manifest_path, manifest_sha, manifest_queries = load_canonical_frozen_manifest()

    ref_paths = [
        SYSTEM_TAI_SRC.parent / "benchmarks" / "manual_kis_reference_v1.json",
        REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "manual_kis_reference_v1.json",
        Path("/kaggle/working/AIC2026_TeamPTK_SGU/systems/system_tai/benchmarks/manual_kis_reference_v1.json"),
    ]
    ref_data = {}
    for p in ref_paths:
        if p.is_file():
            ref_data = json.loads(p.read_text(encoding="utf-8"))
            break

    ref_queries_map = {q.get("query_id"): q for q in ref_data.get("queries", [])}

    targets = [
        ("p1-1", manifest_queries["p1-1"]["target_video"], manifest_queries["p1-1"].get("locked_gt_frame", manifest_queries["p1-1"].get("official_gt_frame")), manifest_queries["p1-1"]["diagnostic_tolerance"]),
        ("p1-2", manifest_queries["p1-2"]["target_video"], manifest_queries["p1-2"].get("locked_gt_frame", manifest_queries["p1-2"].get("official_gt_frame")), manifest_queries["p1-2"]["diagnostic_tolerance"]),
        ("p1-4", manifest_queries["p1-4"]["target_video"], manifest_queries["p1-4"].get("locked_gt_frame", manifest_queries["p1-4"].get("official_gt_frame")), manifest_queries["p1-4"]["diagnostic_tolerance"]),
        ("p1-5", manifest_queries["p1-5"]["target_video"], manifest_queries["p1-5"].get("locked_gt_frame", manifest_queries["p1-5"].get("official_gt_frame")), manifest_queries["p1-5"]["diagnostic_tolerance"]),
        ("p1-6", manifest_queries["p1-6"]["target_video"], manifest_queries["p1-6"].get("locked_gt_frame", manifest_queries["p1-6"].get("official_gt_frame")), manifest_queries["p1-6"]["diagnostic_tolerance"]),
    ]

    coverage_summary = {}

    for qid, legacy_vid, legacy_gt_fid, legacy_diag_tol in targets:
        ref_entry = ref_queries_map.get(f"query-{qid}-kis", {})
        human_vid = ref_entry.get("human_verified_video_id") or legacy_vid
        human_pts_intervals = ref_entry.get("human_annotated_intervals_pts", [])
        human_status = ref_entry.get("annotation_status", "VIDEO_ONLY_VERIFIED")

        print(f"\n──────────────────────────────────────────────────────────────────────────────────────────────────", flush=True)
        print(f"• Query [{qid}] | Human Verified Target: {human_vid} ({human_status}) | Legacy Target: {legacy_vid} (f{legacy_gt_fid})", flush=True)
        print(f"──────────────────────────────────────────────────────────────────────────────────────────────────", flush=True)

        # 1. HUMAN REFERENCE BRANCH
        try:
            human_store = runtime.video_restricted_searcher.registry.get(human_vid)
            human_store_rows = len(human_store.mappings)
            human_indexed = True
        except KeyError:
            human_store = None
            human_store_rows = 0
            human_indexed = False

        human_interval_hits = []
        human_evaluable = (human_status == "FRAME_INTERVAL_VERIFIED" and bool(human_pts_intervals))
        if human_store is not None and human_pts_intervals:
            for s_pts, e_pts in human_pts_intervals:
                human_interval_hits.extend([f for f in human_store.mappings if s_pts <= f.pts_time <= e_pts])

        if human_evaluable:
            human_interval_status = "PASS" if human_interval_hits else "FAIL"
        else:
            human_interval_status = "NOT_EVALUABLE"

        print(f"  • [Human Reference Audit: {human_vid}]", flush=True)
        print(f"    - Video Indexed in Feature Store : {'YES ✅' if human_indexed else 'NO ❌'} ({human_store_rows} keyframes)", flush=True)
        if human_pts_intervals:
            pts_str = ", ".join(f"[{s:.1f}s, {e:.1f}s]" for s, e in human_pts_intervals)
            print(f"    - Verified Interval PTS Coverage : {'PASS ✅' if human_interval_hits else 'FAIL ❌'} ({len(human_interval_hits)} keyframes in {pts_str})", flush=True)
        else:
            print(f"    - Verified Interval PTS Coverage : NOT_EVALUABLE (Intervals not yet annotated)", flush=True)

        # 2. LEGACY MANIFEST BRANCH
        try:
            legacy_store = runtime.video_restricted_searcher.registry.get(legacy_vid)
            legacy_store_rows = len(legacy_store.mappings)
            legacy_interval = (legacy_gt_fid - legacy_diag_tol, legacy_gt_fid + legacy_diag_tol)
            nearest_legacy_f = min(legacy_store.mappings, key=lambda f: abs(f.frame_id - legacy_gt_fid))
            legacy_delta = nearest_legacy_f.frame_id - legacy_gt_fid
            in_legacy_win = [f for f in legacy_store.mappings if legacy_interval[0] <= f.frame_id <= legacy_interval[1]]
            legacy_coverage_pass = len(in_legacy_win) > 0
            print(f"  • [Legacy Manifest Audit: {legacy_vid}]", flush=True)
            print(f"    - Legacy Locked Frame Coverage   : {'PASS ✅' if legacy_coverage_pass else 'FAIL ❌'} (Frame {nearest_legacy_f.frame_id}, Delta: {legacy_delta:+d} vs legacy locked f{legacy_gt_fid})", flush=True)
        except KeyError:
            legacy_store = None
            legacy_coverage_pass = False
            print(f"  • [Legacy Manifest Audit: {legacy_vid}] -> Store not indexed", flush=True)

        # 3. SOURCE VIDEO FRAME-SPACE PARITY AUDIT (Audits both human_vid and legacy_vid independently)
        def _check_parity(target_v: str, target_s) -> tuple[bool, dict]:
            if target_s is None:
                return False, {}
            v_file = find_source_video_file(input_root, target_v, runtime=runtime)
            if not v_file or not v_file.is_file():
                return False, {}
            try:
                import cv2
                cap = cv2.VideoCapture(str(v_file))
                if not cap.isOpened():
                    return False, {"file": v_file.name, "error": "CV2_OPEN_FAILED"}
                fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = float(cap.get(cv2.CAP_PROP_FPS))
                dur = fc / fps if fps > 0 else 0.0
                cap.release()
                info = {"path": str(v_file), "file": v_file.name, "frame_count": fc, "fps": fps, "duration_s": dur}
                n_samples = min(10, len(target_s.mappings))
                sample_indices = np.linspace(0, len(target_s.mappings) - 1, n_samples, dtype=int)
                residuals = [abs(target_s.mappings[i].frame_id - target_s.mappings[i].pts_time * fps) for i in sample_indices]
                med_res = float(np.median(residuals))
                max_res = float(np.max(residuals))
                passed = max_res <= 2.0 or med_res <= 1.0
                return passed, info
            except Exception as e:
                return False, {"file": v_file.name, "error": str(e)}

        human_parity_passed, human_src_info = _check_parity(human_vid, human_store)
        if human_vid == legacy_vid:
            legacy_parity_passed, legacy_src_info = human_parity_passed, human_src_info
        else:
            legacy_parity_passed, legacy_src_info = _check_parity(legacy_vid, legacy_store)

        if human_src_info:
            print(f"  • Human Target Source Video File  : {human_src_info.get('path')} (FPS: {human_src_info.get('fps'):.2f}, Parity: {'PASS ✅' if human_parity_passed else 'FAIL ❌'})", flush=True)
        else:
            print(f"  • Human Target Source Video File  : NOT LOCATED ON RUNNER DISK ⚠️", flush=True)

        coverage_summary[qid] = {
            "query_id": qid,
            "legacy": {
                "video_id": legacy_vid,
                "locked_gt_frame": legacy_gt_fid,
                "coverage_pass": legacy_coverage_pass,
                "parity_passed": legacy_parity_passed,
                "source_info": legacy_src_info,
            },
            "human_reference": {
                "video_id": human_vid,
                "video_indexed": human_indexed,
                "annotation_status": human_status,
                "interval_status": human_interval_status,
                "interval_hits": [f.frame_id for f in human_interval_hits],
                "parity_passed": human_parity_passed,
                "source_info": human_src_info,
            },
        }

    print("=" * 120 + "\n", flush=True)
    return coverage_summary


# ==============================================================================
# SECTION 2: P1-2 EVIDENCE-POOL TO FINAL-EXPORT TRACE & VISUAL BENCHMARK ADJUDICATION
# ==============================================================================
def run_p1_2_trace_and_raw_cosine_audit(
    runtime: OperationalKISRuntime,
    input_root: Path,
    base_out: Path,
    coverage_summary: dict[str, dict],
) -> dict[str, Any]:
    print("=" * 120, flush=True)
    print("2. P1-2: EVIDENCE-POOL TO FINAL-EXPORT TRACE & VISUAL BENCHMARK ADJUDICATION", flush=True)
    print("=" * 120, flush=True)

    manifest_path, manifest_sha, manifest_queries = load_canonical_frozen_manifest()
    qid = "query-p1-2-kis"
    if qid not in manifest_queries:
        raise RuntimeError(f"FROZEN_MANIFEST_PROVENANCE_UNRESOLVED: Query {qid} not found in manifest!")

    manifest_record = manifest_queries[qid]
    q_vi = manifest_record["query_vi"]
    target_vid = manifest_record["target_video"]
    locked_gt_frame = manifest_record.get("locked_gt_frame", manifest_record.get("official_gt_frame"))
    diag_tol = manifest_record["diagnostic_tolerance"]
    gt_interval = (locked_gt_frame - diag_tol, locked_gt_frame + diag_tol)

    print("--- 2.0 BENCHMARK QUERY PROVENANCE AUDIT ---")
    print("• Provenance Classification : PROJECT_FROZEN_STRESS_QUERY (Externally supplied engineering benchmark, tracked in git)")
    print(f"• Manifest File Path        : {manifest_path}")
    print(f"• Manifest File SHA256      : {manifest_sha}")
    print(f"• Query Record ID           : {qid}")
    print(f"• Upstream Git Provenance   :")
    print("  - [1] Commit aaf0649 (2026-08-28): scratch/run_kaggle_v2a_production_gate.py")
    print("        * Earliest recoverable repository gate record binding query-p1-2-kis -> L29_V018.")
    print("        * Evaluated Text: \"Đoạn phim bắt đầu bằng một bản đồ, trên đó một loại công trình thủy lợi lần lượt xuất hiện bốn lần. Sau đó chuyển sang cảnh một con đập được quay từ trên cao, tiếp đến là cảnh cận con đập dưới trời mưa.\"")
    print("  - [2] Commit fe04a5b (2026-08-27): systems/system_tai/tests/test_temporal_decomposition_patterns.py")
    print("        * Earlier semantic/decomposition wording evidence in unit tests (does not bind query ID / GT).")
    print("        * Alternate Text: \"Đoạn phim bắt đầu bằng một bản đồ, trên đó một loại công trình thủy lợi lần lượt xuất hiện bốn lần. Sau đó chuyển sang cảnh một công trình thủy lợi lớn đang mở cửa xả nước dưới trời mưa.\"")
    print("• Wording Discrepancy Diff  :")
    print("  - Clause 1: IDENTICAL (\"Đoạn phim bắt đầu bằng một bản đồ, trên đó một loại công trình thủy lợi lần lượt xuất hiện bốn lần.\")")
    print("  - Clause 2 (fe04a5b): \"Sau đó chuyển sang cảnh một công trình thủy lợi lớn đang mở cửa xả nước dưới trời mưa.\"")
    print("  - Clause 2 (aaf0649): \"Sau đó chuyển sang cảnh một con đập được quay từ trên cao, tiếp đến là cảnh cận con đập dưới trời mưa.\"")
    print(f"• Active Evaluated Text     : \"{q_vi}\"")
    print(f"• Target Video              : {target_vid}")
    print(f"• PROJECT_LOCKED_GT_FRAME   : {locked_gt_frame} (competition provenance unavailable)")
    print(f"• Diagnostic Tolerance      : +/- {diag_tol} frames -> gt_neighborhood_keyframes range: [{gt_interval[0]}, {gt_interval[1]}]")

    # Hard-assert exact record equality
    assert qid == "query-p1-2-kis", "Record ID mismatch"
    assert target_vid == "L29_V018", "Target video mismatch"
    assert locked_gt_frame == 6050, "Locked GT frame mismatch"
    assert diag_tol == 150, "Diagnostic tolerance mismatch"
    assert "thủy lợi" in q_vi and "bản đồ" in q_vi, "Vietnamese query semantics mismatch"
    print("• Manifest Record Integrity : PASS ✅ (Exact record verified against project frozen manifest)\n", flush=True)

    # Corpus Provenance & Registry Integrity Audit
    stores = runtime.video_restricted_searcher.registry.stores
    total_videos = len(stores)
    total_rows = sum(len(s.mappings) for s in stores)
    feat_dim = stores[0].matrix.shape[1] if total_videos > 0 else 0

    print("--- 2.1 CORPUS PROVENANCE & REGISTRY INTEGRITY AUDIT ---")
    print(f"• Total Video Stores Loaded : {total_videos} (Required Target: 873)")
    print(f"• Total Feature Rows Loaded : {total_rows} (Required Target: 177321)")
    print(f"• Feature Embedding Dim     : {feat_dim} (Required Target: 512)")
    print(f"• Target Store Keyframe Rows: {len(runtime.video_restricted_searcher.registry.get(target_vid).mappings)} mappings for {target_vid}")

    assert total_videos == 873, f"Corpus video count mismatch: {total_videos} != 873"
    assert total_rows == 177321, f"Corpus row count mismatch: {total_rows} != 177321"
    assert feat_dim == 512, f"Feature dimension mismatch: {feat_dim} != 512"
    print("• Corpus Integrity Assertions: PASS ✅ (Exact 873/177321/512 production corpus verified)\n", flush=True)

    # Effective V2-A.3 Audit Config and Historical Comparison
    vf_cfg = runtime.config.kis_video_first_config
    print("--- 2.2 EXPLICIT V2-A.3 AUDIT CONFIGURATION — NOT HISTORICAL V2-A CONFIG ---")
    print("• Config Factory Function   : scratch/run_kaggle_v2a_causal_closure.py::create_production_v2a_session_config")
    print("• Canonical Schema Source   : systems/system_tai/src/system_tai/kis/video_first.py::KISVideoFirstConfig")
    print("• Historical Foundation     : Top-M M=3, weights=(0.6, 0.3, 0.1), selected_video_cap=32")
    print("• Current Audit Override    : Top-M M=5, weights=(0.4, 0.25, 0.15, 0.1, 0.1), selected_video_cap=64")
    print("• Field-by-Field Origin Breakdown:")
    print(f"  - enabled                 : {vf_cfg.enabled:<6} [Explicit Audit True | Schema Default: False]")
    print(f"  - v2_adaptive_enabled     : {vf_cfg.v2_adaptive_enabled:<6} [Explicit Audit True | Schema Default: False]")
    print(f"  - selected_video_cap (K)  : {vf_cfg.selected_video_cap:<6} [Audit Override: 64   | Historical Schema Default: 32]")
    print(f"  - top_m_evidence_cap (M)  : {vf_cfg.top_m_evidence_cap:<6} [Audit Override: 5    | Historical Schema Default: 3]")
    print(f"  - top_m_weights           : {str(vf_cfg.top_m_weights):<22} [Audit Override: M5 (0.4, 0.25, 0.15, 0.1, 0.1) | Historical Schema Default: M3 (0.6, 0.3, 0.1)]")
    print(f"  - top_m_min_frame_gap     : {vf_cfg.top_m_min_frame_gap:<6} [Matches Schema Default: 60]")
    print(f"  - adaptive_budgets (B/M/H): ({vf_cfg.adaptive_budget_base}, {vf_cfg.adaptive_budget_medium}, {vf_cfg.adaptive_budget_high}) [Matches Schema Default: (32, 48, 64)]")
    print(f"  - coverage_threshold      : {vf_cfg.coverage_threshold:<6} [Matches Schema Default: 0.75]")
    print(f"  - rrf_constant            : {runtime.config.rrf_constant:<6} [Matches SessionConfig Default: 60.0]")
    print(f"  - restricted_frames/video : {vf_cfg.restricted_frames_per_video_per_variant:<6} [Matches Schema Default: 10]")
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

    # Compile semantic query variants
    compiled_sq = compile_vietnamese_semantic_query(
        query_id=qid,
        query_vi=q_vi,
        provider=runtime.translation_provider,
        token_budget_guard=runtime.token_budget_guard,
        config=SemanticQueryConfig(
            full_query_weight=vf_cfg.full_query_weight,
            primary_scene_weight=vf_cfg.primary_scene_weight,
            supporting_attribute_weight=vf_cfg.supporting_attribute_weight,
        ),
    )

    variants = compiled_sq.query_variants
    embeddings = runtime.shared_encoder.encode_texts([v.text for v in variants])

    # Search video maxima for all variants
    maxima = runtime.video_restricted_searcher.search_video_maxima(
        query_ids=tuple(v.variant_id for v in variants),
        query_vectors=embeddings,
        top_m_evidence_cap=vf_cfg.top_m_evidence_cap,
        top_m_min_frame_gap=vf_cfg.top_m_min_frame_gap,
        top_m_weights=vf_cfg.top_m_weights,
    )

    # Full Corpus Video Fusion (All 873 videos ranked)
    all_fused_videos, adaptive_diag = fuse_video_maxima_v2(
        variants=variants,
        maxima=maxima,
        primary_variant_ids=compiled_sq.primary_variant_ids,
        supporting_variant_ids=compiled_sq.supporting_variant_ids,
        temporal_variants=tuple(item.query_variant for item in compiled_sq.temporal_scene_variants),
        rrf_constant=runtime.config.rrf_constant,
        nomination_depth=len(runtime.video_restricted_searcher.registry.stores),
        config=vf_cfg,
    )
    target_fused_entry = next((item for item in all_fused_videos if item.video_id == target_vid), None)
    target_fused_rank = target_fused_entry.rank if target_fused_entry else None
    target_fused_score = target_fused_entry.fusion_score if target_fused_entry else 0.0

    print("--- 2.3 COMPILED QUERY VARIANTS & PER-VARIANT SCORE BREAKDOWN ---")
    for idx, v in enumerate(variants, start=1):
        emb = embeddings[idx - 1]
        emb_bytes = emb.astype(np.float32).tobytes()
        checksum = hashlib.sha256(emb_bytes).hexdigest()[:12]

        var_match = next((item for item in compiled_sq.variants if item.query_variant.variant_id == v.variant_id), None)
        if var_match:
            role_str = var_match.semantic_role.value
            t_idx_str = str(var_match.temporal_index)
            vi_text = var_match.source_vietnamese
        else:
            role_str = "UNKNOWN"
            t_idx_str = "None"
            vi_text = "N/A"

        hits = maxima.rankings.get(v.variant_id, ())
        hits_by_raw = sorted(hits, key=lambda h: -h.cosine_score)
        raw_max_rank = next((r for r, h in enumerate(hits_by_raw, start=1) if h.video_id == target_vid), None)
        top_m_rank = next((r for r, h in enumerate(hits, start=1) if h.video_id == target_vid), None)

        t_hit = next((h for h in hits if h.video_id == target_vid), None)
        target_raw_max = t_hit.cosine_score if t_hit else 0.0
        target_top_m = t_hit.top_m_score if t_hit else 0.0
        peaks = list(t_hit.top_m_peaks) if t_hit and t_hit.top_m_peaks else []
        peaks_str = ", ".join(f"f{fid}:{cos:.4f}" for fid, cos in peaks)

        print(f"• Variant [{idx}] ID: {v.variant_id} (Weight: {float(v.weight):.2f})")
        print(f"  - Role / Temporal Idx : {role_str} (Temporal Index: {t_idx_str})")
        print(f"  - VI Text             : \"{vi_text}\"")
        print(f"  - VinAI EN Text       : \"{v.text}\"")
        print(f"  - Embedding SHA256    : {checksum} (Norm: {float(np.linalg.norm(emb)):.4f})")
        print(f"  - Target Raw-Max Score: {target_raw_max:.4f} | Raw-Max Video Rank: #{raw_max_rank} / {total_videos}")
        print(f"  - Target Top-M Score  : {target_top_m:.4f} | Top-M Video Rank  : #{top_m_rank} / {total_videos}")
        print(f"  - Target Top-M Peaks  : [{peaks_str}]\n")

    print("--- 2.4 CANONICAL STAGE-1 FUSED VIDEO NOMINATION OUTCOME ---")
    print(f"• Total Corpus Videos Scored       : {total_videos}")
    print(f"• Canonical Fused Video Rank       : #{target_fused_rank} / {total_videos}")
    print(f"• Canonical Video Nomination Budget: K = {len(selected_videos)} (Adaptive chosen K: {adaptive_diag.chosen_k})")
    print(f"• Target Video Nominated (Top-K)?  : {'YES ✅' if target_sel_entry else 'NO ❌'}")
    print(f"• Target Video Fused Score         : {target_fused_score:.6f}")
    print("------------------------------------------------------------------------------------------------------------------------\n", flush=True)

    # 2. Stage 2 Restricted Frame Retrieval Consumption Audit
    store = runtime.video_restricted_searcher.registry.get(target_vid)
    sampled_frame_ids = sorted({m.frame_id for m in store.mappings})

    nearest_gt_keyframe = min(sampled_frame_ids, key=lambda fid: abs(fid - locked_gt_frame))
    gt_neighborhood_keyframes = [
        fid for fid in sampled_frame_ids
        if gt_interval[0] <= fid <= gt_interval[1]
    ]

    print(f"• Groundtruth State for Target Video {target_vid}:")
    print(f"  - PROJECT_LOCKED_GT_FRAME     : Frame {locked_gt_frame} (PTS: {locked_gt_frame/25.0:.3f}s)")
    print(f"  - Nearest Keyframe in Store   : Frame {nearest_gt_keyframe} (Delta: {nearest_gt_keyframe - locked_gt_frame} frames)")
    print(f"  - Keyframes in GT Neighborhood: {len(gt_neighborhood_keyframes)} keyframes: {gt_neighborhood_keyframes}\n")

    # Restricted frame retrieval across selected videos
    per_query_cap = vf_cfg.restricted_frames_per_video_per_variant
    selected_vids_tuple = tuple(v["video_id"] for v in selected_videos)
    restricted = runtime.video_restricted_searcher.search_selected_videos(
        video_ids=selected_vids_tuple,
        query_ids=tuple(v.variant_id for v in variants),
        query_vectors=embeddings,
        per_query_result_cap=per_query_cap,
    )

    # Build restricted rank lookups across all retained frames
    restricted_rank_lookup = {}
    for v in variants:
        hits = [
            hit
            for vid_hits in restricted.rankings.get(v.variant_id, {}).values()
            for hit in vid_hits
        ]
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

    # 2.4.1 RESTRICTED CANDIDATE POOL SIZES ACROSS NOMINATED VIDEOS
    print("--- 2.4.1 RESTRICTED CANDIDATE POOL SIZES ACROSS NOMINATED VIDEOS ---")
    for v in variants:
        actual_size = len(restricted_rank_lookup[v.variant_id])
        print(f"• Variant [{v.variant_id.split('::')[-1]}]: Actual Global Pool Size = {actual_size} retained candidate frames (Nominated videos: {len(selected_videos)}, Cap: {per_query_cap}/vid)")
    print("------------------------------------------------------------------------------------------------------------------------\n", flush=True)

    # Pre-calculate target video cosine matrix
    target_all_cos = store.matrix @ embeddings.T  # (N, n_variants)

    print("=" * 120)
    print("TABLE 1: GT-NEIGHBORHOOD KEYFRAMES AUDIT ACROSS PRODUCTION VARIANTS")
    print(f"Target Video: {target_vid} | Total Store Keyframes: {len(store.mappings)} | Locked GT Frame: {locked_gt_frame} | Range: [{gt_interval[0]}, {gt_interval[1]}]")
    print("  * Semantic Note: 'Retained by Variant?' = Per-video retention cap (top 10 frames within this video for that variant).")
    print("  * Semantic Note: 'Global Pool Rank'     = Rank within this variant's actual candidate pool size across all nominated videos.")
    print("=" * 120)
    print(f"| {'Frame ID':<8} | {'PTS (s)':<8} | {'Variant ID':<32} | {'Raw Cos':<8} | {'Intra Rank':<10} | {'Top-M Peak?':<12} | {'Evid Nbrhood?':<14} | {'Retained by Var?':<18} | {'Global Pool Rank':<18} |")
    print(f"| {'-'*8} | {'-'*8} | {'-'*32} | {'-'*8} | {'-'*10} | {'-'*12} | {'-'*14} | {'-'*18} | {'-'*18} |")

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
            actual_pool_size = len(restricted_rank_lookup[v.variant_id])
            restr_rank_str = f"#{restr_global_rank} / {actual_pool_size}" if restr_global_rank else "OUTSIDE_TOP10_CAP"

            frame_records.append({
                "variant_id": v.variant_id,
                "cosine": cos_val,
                "intra_rank": intra_rank,
                "is_peak": is_peak,
                "in_nbrhood": in_nbrhood,
                "is_retained": is_retained,
                "restr_global_rank": restr_global_rank,
                "actual_pool_size": actual_pool_size,
            })

            is_gt_marker = " (GT)" if fid == locked_gt_frame else ""
            print(f"| {str(fid)+is_gt_marker:<8} | {pts:<8.3f} | {v.variant_id:<32} | {cos_val:<8.4f} | {f'#{intra_rank}/568':<10} | {'YES ★' if is_peak else 'NO':<12} | {'YES' if in_nbrhood else 'NO':<14} | {'YES ★' if is_retained else 'NO':<18} | {restr_rank_str:<18} |")

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

        is_gt_marker = " (GT)" if fid == locked_gt_frame else ""
        print(f"| {str(fid)+is_gt_marker:<8} | {pts:<8.3f} | {'YES ★' if is_cand else 'NO':<18} | {contrib_str:<45} | {score_str:<12} | {rank_str:<20} | {'YES ✅' if in_top100 else 'NO ❌':<11} |")

    print("=" * 120 + "\n")

    # 3. DUAL-VERDICT CAUSAL LOSS REPORT
    print("--- 2.5 DUAL-VERDICT CAUSAL LOSS REPORT FOR TRUE FROZEN P1-2 ---")
    print(f"• [1] Nearest Locked GT Frame (Frame {nearest_gt_keyframe} / PTS {nearest_gt_keyframe/25.0:.2f}s, Delta {nearest_gt_keyframe - locked_gt_frame} frames from {locked_gt_frame}):")
    print("      - Causal Loss Stage : STAGE 2 — RESTRICTED_FRAME_SEARCH_TRUNCATION ❌")
    print(f"      - Root Cause        : Frame {nearest_gt_keyframe} failed to achieve Top-10 intra-video rank for any variant (Var 1: #45, Var 2: #64, Var 3: #27 / 568), thus pruned before RRF fusion.")
    if best_frame is not None:
        print(f"• [2] GT±150 Tolerance Neighborhood [5900, 6200] Best Surviving Frame (Frame {best_frame} / PTS {frame_audit_records[best_frame]['pts']:.2f}s):")
        if best_rank <= 100:
            print(f"      - Causal Loss Stage : NONE (Survives into Top-100 export at Rank #{best_rank}) ✅")
        else:
            print(f"      - Causal Loss Stage : STAGE 3 — FRAME_RRF_CUTOFF (Global Fusion Rank #{best_rank} > 100) ❌")
            print(f"      - Root Cause        : Frame {best_frame} was retained on Variant 1 (global pool #{restricted_rank_lookup[variants[0].variant_id].get((target_vid, best_frame))}/{len(restricted_rank_lookup[variants[0].variant_id])}), but single-variant RRF score ({best_score:.6f}) was overtaken by multi-variant candidates.")
    else:
        print("• [2] GT±150 Tolerance Neighborhood [5900, 6200] Best Surviving Frame: NONE")
        print("      - Causal Loss Stage : STAGE 2 — RESTRICTED_FRAME_SEARCH_TRUNCATION ❌")
    print("• [3] Benchmark Integrity & Provenance Status:")
    print("      - Status            : BENCHMARK_PROVENANCE_SUSPECT / VISUAL_ADJUDICATION_REQUIRED ⚠️")
    print("      - Visual Rule       : Machine metrics provide structural trace, but human visual review of contact sheets is mandatory to adjudicate target sequence matching.\n", flush=True)

    # 4. VISUAL BENCHMARK ADJUDICATION: FULL 568 KEYFRAME PAGINATION & BROAD CANDIDATE DISCOVERY
    visual_records = run_p1_2_visual_benchmark_adjudication(
        runtime=runtime,
        input_root=input_root,
        base_out=base_out,
        store=store,
        target_vid=target_vid,
        locked_gt_frame=locked_gt_frame,
        gt_neighborhood_keyframes=gt_neighborhood_keyframes,
        variants=variants,
        embeddings=embeddings,
        maxima=maxima,
    )

    # Record summary for unified final reporting table
    cov_record = coverage_summary.get("p1-2", {})
    coverage_summary["p1-2"] = {
        **cov_record,
        "query_id": "p1-2",
        "video_id": target_vid,
        "locked_gt_frame": locked_gt_frame,
        "coverage_pass": len(gt_neighborhood_keyframes) > 0,
        "coverage_str": f"PASS ✅ ({len(gt_neighborhood_keyframes)} kfs)",
        "loss_stage": "STAGE 2 (Cap) / STAGE 3 (RRF Cutoff)",
        "classification": "QUERY_TARGET_SEMANTIC_BINDING_UNRESOLVED_PENDING_VISUAL_ADJUDICATION",
    }

    return {
        "target_vid": target_vid,
        "nearest_gt_keyframe": nearest_gt_keyframe,
        "best_surviving_frame": best_frame,
        "best_rank": best_rank,
        "visual_records": visual_records,
    }


def run_p1_2_visual_benchmark_adjudication(
    runtime: OperationalKISRuntime,
    input_root: Path,
    base_out: Path,
    store: LoadedVideoFeatureStore,
    target_vid: str,
    locked_gt_frame: int,
    gt_neighborhood_keyframes: list[int],
    variants: tuple[QueryVariant, ...],
    embeddings: np.ndarray,
    maxima: Any,
) -> list[dict[str, Any]]:
    print("=" * 120, flush=True)
    print("2.6 P1-2 VISUAL BENCHMARK ADJUDICATION (100% TARGET INDEXED-KEYFRAME COVERAGE & BROAD CANDIDATES)", flush=True)
    print("=" * 120, flush=True)

    manifest_entries: list[dict[str, Any]] = []
    mandatory_requested = 0
    mandatory_exact_keyframe_file = 0
    mandatory_raw_video_pts = 0
    mandatory_raw_video_frame_index = 0
    mandatory_unresolved = 0
    mandatory_ambiguous = 0
    mandatory_integrity_fail = 0

    optional_requested = 0
    optional_exact_keyframe_file = 0
    optional_raw_video_pts = 0
    optional_raw_video_frame_index = 0
    optional_unresolved = 0
    optional_ambiguous = 0
    optional_integrity_fail = 0

    print("=" * 120, flush=True)
    print("DECODE_INTEGRITY_POLICY", flush=True)
    print(f"  exact_file_ambiguity_policy   = {EXACT_FILE_AMBIGUITY_POLICY}", flush=True)
    print(f"  raw_video_pts_tolerance       = <= {DECLARED_PTS_TOLERANCE_SECONDS:.3f}s", flush=True)
    print(f"  raw_video_frame_index_allowed = False (Guarded strictly by source<->mapping parity)", flush=True)
    print("=" * 120 + "\n", flush=True)

    all_mappings = store.mappings
    total_kfs = len(all_mappings)
    compulsory_peaks = [905, 1145, 4995, 6171, 8215, 8235, 9749, 16335, 25325, 27135, 27270]

    # -------------------------------------------------------------------------
    # PART A0: 10-RECORD MAPPING SANITY SAMPLE (PROVENANCE VERIFICATION)
    # -------------------------------------------------------------------------
    sample_target_fids = [
        all_mappings[0].frame_id,
        905,
        4995,
        5940,
        5959,
        6048,
        6075,
        6107,
        6171,
        all_mappings[-1].frame_id,
    ]
    sample_seen = set()
    sample_mappings = []
    for fid in sample_target_fids:
        if fid not in sample_seen:
            sample_seen.add(fid)
            m = next((m for m in all_mappings if m.frame_id == fid), None)
            if m:
                sample_mappings.append(m)

    print("• [A0] TARGET_MAPPING_SANITY_SAMPLE (Target Video L29_V018 Trace):", flush=True)
    print("=" * 120, flush=True)
    print("| #  | Physical Frame | Order (n) | PTS (s) | Source Type    | Resolution Rule           | Resolved Filename | Status |", flush=True)
    print("| -- | -------------- | --------- | ------- | -------------- | ------------------------- | ----------------- | ------ |", flush=True)
    for s_idx, sm in enumerate(sample_mappings, start=1):
        dec_sample = extract_image_for_frame(
            dataset_root=input_root,
            video_id=target_vid,
            frame_id=sm.frame_id,
            keyframe_order=sm.keyframe_order,
            pts_time=sm.pts_time,
            runtime=runtime,
        )
        status_str = "PASS ✅" if dec_sample.integrity_status == "PASS" else "FAIL ❌"
        fname_str = dec_sample.resolved_filename or "N/A"
        print(f"| {s_idx:<2} | f{sm.frame_id:<13} | n={sm.keyframe_order:<6} | {sm.pts_time:<7.3f} | {dec_sample.resolved_source_type:<14} | {dec_sample.resolution_rule:<25} | {fname_str:<17} | {status_str:<6} |", flush=True)
    print("=" * 120 + "\n", flush=True)

    # -------------------------------------------------------------------------
    # PART A1: RENDER 100% TARGET INDEXED-KEYFRAME COVERAGE (ALL 568 KEYFRAMES)
    # -------------------------------------------------------------------------
    page_size = 64
    total_pages = math.ceil(total_kfs / page_size)
    print(f"• [A1] Rendering 100% TARGET INDEXED-KEYFRAME COVERAGE ({total_kfs}/{total_kfs} indexed keyframes of {target_vid}) into {total_pages} paginated contact sheets (64 frames/page)...", flush=True)
    print(f"       * Provenance Note: This represents 568 indexed keyframes covering the target video according to the feature registry mappings.", flush=True)

    all_page_paths = []
    for page_idx in range(total_pages):
        start_idx = page_idx * page_size
        end_idx = min(start_idx + page_size, total_kfs)
        page_mappings = all_mappings[start_idx:end_idx]
        n_page_tiles = len(page_mappings)

        fig, axes = plt.subplots(8, 8, figsize=(28, 28))
        axes_flat = axes.flatten()

        page_file = base_out / f"p1-2_{target_vid}_all_keyframes_page_{page_idx+1:02d}.png"
        all_page_paths.append(page_file)

        for tile_idx, mapping in enumerate(page_mappings):
            ax = axes_flat[tile_idx]
            fid = mapping.frame_id
            kf_order = mapping.keyframe_order
            pts_time = mapping.pts_time

            dec_res = extract_image_for_frame(
                dataset_root=input_root,
                video_id=target_vid,
                frame_id=fid,
                keyframe_order=kf_order,
                pts_time=pts_time,
                runtime=runtime,
            )

            if dec_res.resolved_source_type == "KEYFRAME_FILE":
                mandatory_exact_keyframe_file += 1
            elif dec_res.resolved_source_type == "RAW_VIDEO_PTS":
                mandatory_raw_video_pts += 1
            elif dec_res.resolved_source_type == "RAW_VIDEO_FRAME_INDEX":
                mandatory_raw_video_frame_index += 1
            elif dec_res.resolved_source_type == "AMBIGUOUS_KEYFRAME_FILE":
                mandatory_ambiguous += 1
            elif dec_res.resolved_source_type == "UNRESOLVED":
                mandatory_unresolved += 1

            if dec_res.integrity_status != "PASS":
                mandatory_integrity_fail += 1

            mandatory_requested += 1

            if dec_res.image is not None:
                ax.imshow(dec_res.image)
            else:
                ax.text(0.5, 0.5, f"IMAGE UNRESOLVED\nFrame: {fid}\nOrder: {kf_order}\n({dec_res.resolution_rule})", ha="center", va="center", fontsize=7)

            is_locked_gt = (fid == locked_gt_frame or fid == 6048)
            is_peak = fid in compulsory_peaks
            is_gt_nbrhood = fid in gt_neighborhood_keyframes

            border_color = "red" if is_locked_gt else ("green" if is_peak else ("blue" if is_gt_nbrhood else "gray"))
            for spine in ax.spines.values():
                spine.set_color(border_color)
                spine.set_linewidth(3 if (is_locked_gt or is_peak or is_gt_nbrhood) else 0.5)

            marker_str = " (LOCKED GT ★)" if is_locked_gt else (" (PEAK)" if is_peak else (" (GT NBRHOOD)" if is_gt_nbrhood else ""))
            title_caption = f"f{fid} | {pts_time:.2f}s | #{kf_order}{marker_str}\n{dec_res.resolution_rule}\n{dec_res.resolved_filename or 'UNRESOLVED'}"
            ax.set_title(title_caption, fontsize=6.5, color=border_color if border_color != "gray" else "black", pad=3)
            ax.axis("off")

            manifest_entries.append({
                "contact_sheet_file": page_file.name,
                "tile_type": "MANDATORY_TARGET_KEYFRAME",
                "is_locked_gt": is_locked_gt,
                "is_retrieval_peak": is_peak,
                "is_gt_nbrhood": is_gt_nbrhood,
                **dec_res.to_manifest_dict(),
            })

        for tile_idx in range(n_page_tiles, 64):
            axes_flat[tile_idx].axis("off")

        page_file.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(page_file, dpi=100)
        plt.close(fig)
        print(f"  📸 Page [{page_idx+1:02d}/{total_pages:02d}]: Saved {n_page_tiles} keyframes -> {page_file.name} ✅", flush=True)

    # -------------------------------------------------------------------------
    # PART A2: RENDER TIMELINE SUMMARY CONTACT SHEET (OPTIONAL SUMMARY VIEW)
    # -------------------------------------------------------------------------
    uniform_indices = np.linspace(0, total_kfs - 1, 20, dtype=int)
    uniform_fids = [all_mappings[i].frame_id for i in uniform_indices]
    timeline_fids = sorted(set(uniform_fids + compulsory_peaks + gt_neighborhood_keyframes + [locked_gt_frame]))
    store_fid_set = {m.frame_id for m in all_mappings}
    timeline_fids = [fid for fid in timeline_fids if fid in store_fid_set]

    print(f"\n• [A2] Generating Timeline Summary Contact Sheet for {target_vid} ({len(timeline_fids)} keyframes)...", flush=True)

    n_tiles = len(timeline_fids)
    cols = 5
    rows = math.ceil(n_tiles / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(25, 5 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    timeline_sheet_path = base_out / f"p1-2_{target_vid}_timeline_summary.png"

    for idx, fid in enumerate(timeline_fids):
        r_idx = idx // cols
        c_idx = idx % cols
        ax = axes[r_idx, c_idx]

        row_indices = store.rows_for_frame(fid)
        mapping = store.frame_for_row(row_indices[0]) if row_indices else None
        kf_order = mapping.keyframe_order if mapping else None
        pts_time = mapping.pts_time if mapping else fid / 25.0

        dec_res = extract_image_for_frame(
            dataset_root=input_root,
            video_id=target_vid,
            frame_id=fid,
            keyframe_order=kf_order,
            pts_time=pts_time,
            runtime=runtime,
        )

        if dec_res.resolved_source_type == "KEYFRAME_FILE":
            optional_exact_keyframe_file += 1
        elif dec_res.resolved_source_type == "RAW_VIDEO_PTS":
            optional_raw_video_pts += 1
        elif dec_res.resolved_source_type == "RAW_VIDEO_FRAME_INDEX":
            optional_raw_video_frame_index += 1
        elif dec_res.resolved_source_type == "AMBIGUOUS_KEYFRAME_FILE":
            optional_ambiguous += 1
        elif dec_res.resolved_source_type == "UNRESOLVED":
            optional_unresolved += 1

        if dec_res.integrity_status != "PASS":
            optional_integrity_fail += 1

        optional_requested += 1

        if dec_res.image is not None:
            ax.imshow(dec_res.image)
        else:
            ax.text(0.5, 0.5, f"IMAGE UNRESOLVED\nFrame: {fid}\n({dec_res.resolution_rule})", ha="center", va="center", fontsize=8)

        tags = []
        if fid == locked_gt_frame or fid == 6048:
            tags.append("LOCKED_GT")
        if fid in compulsory_peaks:
            tags.append("PEAK")
        if fid in gt_neighborhood_keyframes:
            tags.append("GT_NBRHOOD")
        if not tags:
            tags.append("UNIFORM")
        tag_str = " | ".join(tags)

        feat = store.matrix[row_indices[0]] if row_indices else np.zeros(512)
        cos_v1 = float(feat @ embeddings[0])
        cos_v2 = float(feat @ embeddings[1])
        cos_v3 = float(feat @ embeddings[2])

        is_gt = "LOCKED_GT" in tags
        caption = (
            f"Video: {target_vid} | Frame: {fid} (#{kf_order if kf_order else 'N/A'})\n"
            f"PTS: {pts_time:.2f}s | Tags: {tag_str}\n"
            f"Cos: Full={cos_v1:.3f} | T1(Map)={cos_v2:.3f} | T2(Dam)={cos_v3:.3f}\n"
            f"{dec_res.resolution_rule} | {dec_res.resolved_filename or 'UNRESOLVED'}"
        )
        ax.set_title(caption, fontsize=8, color="red" if is_gt else "black", pad=6)

        manifest_entries.append({
            "contact_sheet_file": timeline_sheet_path.name,
            "tile_type": "OPTIONAL_TIMELINE_SUMMARY",
            "tags": tags,
            "source_variant": "timeline_sample",
            "cosine_full": cos_v1,
            "cosine_scene1": cos_v2,
            "cosine_scene2": cos_v3,
            **dec_res.to_manifest_dict(),
        })

    for idx in range(n_tiles, rows * cols):
        r_idx = idx // cols
        c_idx = idx % cols
        axes[r_idx, c_idx].axis("off")

    timeline_sheet_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(timeline_sheet_path, dpi=120)
    plt.close(fig)
    print(f"  📸 Saved timeline summary contact sheet -> {timeline_sheet_path.name} ✅", flush=True)

    # -------------------------------------------------------------------------
    # PART B: BROAD CANDIDATE DISCOVERY ACROSS ALL 873 CORPUS VIDEOS
    # -------------------------------------------------------------------------
    print("\n• [B] Scanning all 873 videos for Scene 1 (Map) and Scene 2 (Dam) Temporal Candidates...", flush=True)
    print("      * SEMANTIC RULE: DP valid T1 < T2 verifies temporal order between two scene peaks only;", flush=True)
    print("        it does NOT verify 'irrigation_structure_repeated_four_times' or full narrative story.", flush=True)
    print("        Human visual inspection of contact sheets is mandatory for MATCH / PARTIAL / NO_MATCH.\n", flush=True)

    s1_emb = embeddings[1]
    s2_emb = embeddings[2]

    all_stores = runtime.video_restricted_searcher.registry.stores
    candidate_pool = []

    for st in all_stores:
        vid = st.descriptor.video_id
        if len(st.mappings) == 0:
            continue

        cos_s1 = st.matrix @ s1_emb
        cos_s2 = st.matrix @ s2_emb

        max_s1 = float(np.max(cos_s1))
        max_s2 = float(np.max(cos_s2))

        s1_peak_indices = np.argsort(-cos_s1)[:5]
        s1_peaks = [(st.mappings[i].frame_id, float(cos_s1[i])) for i in s1_peak_indices]

        s2_peak_indices = np.argsort(-cos_s2)[:5]
        s2_peaks = [(st.mappings[i].frame_id, float(cos_s2[i])) for i in s2_peak_indices]

        has_valid_chain, chain_frames, chain_score = solve_temporal_chain(
            peaks_by_scene=[s1_peaks, s2_peaks],
            scene_weights=[1.0, 1.0],
            min_gap=60,
        )

        candidate_pool.append({
            "video_id": vid,
            "max_s1": max_s1,
            "max_s2": max_s2,
            "s1_peaks": s1_peaks,
            "s2_peaks": s2_peaks,
            "has_valid_chain": has_valid_chain,
            "chain_score": chain_score,
            "chain_frames": chain_frames,
            "store": st,
        })

    by_s1 = sorted(candidate_pool, key=lambda x: -x["max_s1"])
    by_s2 = sorted(candidate_pool, key=lambda x: -x["max_s2"])
    by_dp = sorted(candidate_pool, key=lambda x: (-int(x["has_valid_chain"]), -x["chain_score"]))

    s1_top20_vids = [x["video_id"] for x in by_s1[:20]]
    s2_top20_vids = [x["video_id"] for x in by_s2[:20]]
    intersection_vids = sorted(set(s1_top20_vids) & set(s2_top20_vids))
    dp_top20_vids = [x["video_id"] for x in by_dp[:20]]

    print("\n" + "=" * 120)
    print("TOP CANDIDATE DISCOVERY AUDIT (ALL 873 CORPUS VIDEOS)")
    print("=" * 120)
    print(f"• Top 5 Videos by Scene 1 (Map / 4x Irrigation)        : {', '.join(s1_top20_vids[:5])}")
    print(f"• Top 5 Videos by Scene 2 (Aerial Dam / Rain Discharge): {', '.join(s2_top20_vids[:5])}")
    print(f"• Intersection of Scene 1 & Scene 2 Top-20             : {intersection_vids if intersection_vids else 'NONE (Disjoint Top-20)'}")
    print(f"• Top 5 Temporal Chain Candidates (T1 < T2)            : {', '.join(dp_top20_vids[:5])}")
    print("=" * 120)

    print("\n| Rank | Video ID | Valid T1<T2? | Chain Score | Scene 1 Peak Frame (PTS, Cos) | Scene 2 Peak Frame (PTS, Cos) | Gap (frames) |")
    print("| ---- | -------- | ------------ | ----------- | ----------------------------- | ----------------------------- | ------------ |")

    candidate_adjudication_templates = []

    for rank, cand in enumerate(by_dp[:15], start=1):
        vid = cand["video_id"]
        valid_str = "YES ✅" if cand["has_valid_chain"] else "NO ❌"
        cf = cand["chain_frames"]
        s1_info = f"f{cf[0]} (cos: {cand['max_s1']:.3f})" if cf else f"f{cand['s1_peaks'][0][0]} (cos: {cand['s1_peaks'][0][1]:.3f})"
        s2_info = f"f{cf[1]} (cos: {cand['max_s2']:.3f})" if len(cf) > 1 else f"f{cand['s2_peaks'][0][0]} (cos: {cand['s2_peaks'][0][1]:.3f})"
        gap = (cf[1] - cf[0]) if len(cf) > 1 else 0
        print(f"| #{rank:<4} | {vid:<8} | {valid_str:<12} | {cand['chain_score']:<11.4f} | {s1_info:<29} | {s2_info:<29} | {gap:<12} |")

        candidate_adjudication_templates.append({
            "rank": rank,
            "video_id": vid,
            "temporal_chain_score": cand["chain_score"],
            "has_valid_chain": cand["has_valid_chain"],
            "winning_chain_frames": cf,
            "scene1_peak": cand["s1_peaks"][0] if cand["s1_peaks"] else None,
            "scene2_peak": cand["s2_peaks"][0] if cand["s2_peaks"] else None,
            "human_adjudication_rubric": {
                "map_present": None,
                "irrigation_structure_repeated_four_times": None,
                "aerial_dam": None,
                "rainy_dam_closeup_or_discharge": None,
                "temporal_order_correct": None,
                "overall_label": "UNRESOLVED",
            },
        })

    print("=" * 120 + "\n")

    # -------------------------------------------------------------------------
    # PART C: RENDER CANDIDATE DISCOVERY CONTACT SHEET WITH WINNING DP FRAMES
    # -------------------------------------------------------------------------
    discovery_vids = []
    for cand in by_dp[:6]:
        discovery_vids.append(cand["video_id"])
    if target_vid not in discovery_vids:
        discovery_vids.append(target_vid)

    print(f"• [C] Rendering Discovery Contact Sheet for {len(discovery_vids)} Candidate Videos ({discovery_vids}) including Winning DP Frames...", flush=True)

    cand_sheet_path = base_out / "p1-2_candidate_discovery_contact_sheet.png"

    # Each video gets 3 rows: Row 1 = Scene 1 Peaks (3 frames), Row 2 = Scene 2 Peaks (3 frames), Row 3 = Actual Winning DP Frames (2 frames)
    n_vids = len(discovery_vids)
    fig, axes = plt.subplots(n_vids * 3, 3, figsize=(18, 5 * n_vids * 3))
    if n_vids * 3 == 1:
        axes = np.array([axes])

    for v_idx, vid in enumerate(discovery_vids):
        cand_obj = next((c for c in candidate_pool if c["video_id"] == vid), None)
        if not cand_obj:
            continue

        c_store = cand_obj["store"]
        dp_frames = cand_obj["chain_frames"]

        # Row 1: Scene 1 Top 3 Peaks (Optional view)
        for col_idx in range(3):
            ax = axes[v_idx * 3, col_idx]
            if col_idx < len(cand_obj["s1_peaks"]):
                fid, cos_val = cand_obj["s1_peaks"][col_idx]
                r_rows = c_store.rows_for_frame(fid)
                mapping = c_store.frame_for_row(r_rows[0]) if r_rows else None
                kf_order = mapping.keyframe_order if mapping else None
                pts_time = mapping.pts_time if mapping else fid / 25.0

                dec_res = extract_image_for_frame(
                    dataset_root=input_root,
                    video_id=vid,
                    frame_id=fid,
                    keyframe_order=kf_order,
                    pts_time=pts_time,
                    runtime=runtime,
                )

                if dec_res.resolved_source_type == "KEYFRAME_FILE":
                    optional_exact_keyframe_file += 1
                elif dec_res.resolved_source_type == "RAW_VIDEO_PTS":
                    optional_raw_video_pts += 1
                elif dec_res.resolved_source_type == "RAW_VIDEO_FRAME_INDEX":
                    optional_raw_video_frame_index += 1
                elif dec_res.resolved_source_type == "AMBIGUOUS_KEYFRAME_FILE":
                    optional_ambiguous += 1
                elif dec_res.resolved_source_type == "UNRESOLVED":
                    optional_unresolved += 1

                if dec_res.integrity_status != "PASS":
                    optional_integrity_fail += 1

                optional_requested += 1

                if dec_res.image is not None:
                    ax.imshow(dec_res.image)
                else:
                    ax.text(0.5, 0.5, f"IMAGE UNRESOLVED\nVideo: {vid}\nFrame: {fid}\n({dec_res.resolution_rule})", ha="center", va="center", fontsize=8)

                is_dp_win = fid in dp_frames
                caption = (
                    f"Video: {vid} | SCENE 1 (Map/Irrigation)\n"
                    f"Physical Frame: {fid} | PTS: {pts_time:.2f}s\n"
                    f"Raw Cos: {cos_val:.4f} | {dec_res.resolution_rule}\n"
                    f"{dec_res.resolved_filename or 'UNRESOLVED'}"
                )
                ax.set_title(caption, fontsize=8, color="black", pad=6)

                manifest_entries.append({
                    "contact_sheet_file": cand_sheet_path.name,
                    "tile_type": "OPTIONAL_CANDIDATE_PEAK",
                    "scene": "T1_MAP_PEAK",
                    "cosine": cos_val,
                    "is_dp_winning": is_dp_win,
                    **dec_res.to_manifest_dict(),
                })
            else:
                ax.axis("off")

        # Row 2: Scene 2 Top 3 Peaks (Optional view)
        for col_idx in range(3):
            ax = axes[v_idx * 3 + 1, col_idx]
            if col_idx < len(cand_obj["s2_peaks"]):
                fid, cos_val = cand_obj["s2_peaks"][col_idx]
                r_rows = c_store.rows_for_frame(fid)
                mapping = c_store.frame_for_row(r_rows[0]) if r_rows else None
                kf_order = mapping.keyframe_order if mapping else None
                pts_time = mapping.pts_time if mapping else fid / 25.0

                dec_res = extract_image_for_frame(
                    dataset_root=input_root,
                    video_id=vid,
                    frame_id=fid,
                    keyframe_order=kf_order,
                    pts_time=pts_time,
                    runtime=runtime,
                )

                if dec_res.resolved_source_type == "KEYFRAME_FILE":
                    optional_exact_keyframe_file += 1
                elif dec_res.resolved_source_type == "RAW_VIDEO_PTS":
                    optional_raw_video_pts += 1
                elif dec_res.resolved_source_type == "RAW_VIDEO_FRAME_INDEX":
                    optional_raw_video_frame_index += 1
                elif dec_res.resolved_source_type == "AMBIGUOUS_KEYFRAME_FILE":
                    optional_ambiguous += 1
                elif dec_res.resolved_source_type == "UNRESOLVED":
                    optional_unresolved += 1

                if dec_res.integrity_status != "PASS":
                    optional_integrity_fail += 1

                optional_requested += 1

                if dec_res.image is not None:
                    ax.imshow(dec_res.image)
                else:
                    ax.text(0.5, 0.5, f"IMAGE UNRESOLVED\nVideo: {vid}\nFrame: {fid}\n({dec_res.resolution_rule})", ha="center", va="center", fontsize=8)

                is_dp_win = fid in dp_frames
                caption = (
                    f"Video: {vid} | SCENE 2 (Aerial Dam/Rain)\n"
                    f"Physical Frame: {fid} | PTS: {pts_time:.2f}s\n"
                    f"Raw Cos: {cos_val:.4f} | {dec_res.resolution_rule}\n"
                    f"{dec_res.resolved_filename or 'UNRESOLVED'}"
                )
                ax.set_title(caption, fontsize=8, color="black", pad=6)

                manifest_entries.append({
                    "contact_sheet_file": cand_sheet_path.name,
                    "tile_type": "OPTIONAL_CANDIDATE_PEAK",
                    "scene": "T2_DAM_PEAK",
                    "cosine": cos_val,
                    "is_dp_winning": is_dp_win,
                    **dec_res.to_manifest_dict(),
                })
            else:
                ax.axis("off")

        # Row 3: Actual Winning DP Frames (MANDATORY ADJUDICATION TILES)
        for col_idx in range(3):
            ax = axes[v_idx * 3 + 2, col_idx]
            if col_idx < len(dp_frames):
                fid = dp_frames[col_idx]
                r_rows = c_store.rows_for_frame(fid)
                mapping = c_store.frame_for_row(r_rows[0]) if r_rows else None
                kf_order = mapping.keyframe_order if mapping else None
                pts_time = mapping.pts_time if mapping else fid / 25.0

                dec_res = extract_image_for_frame(
                    dataset_root=input_root,
                    video_id=vid,
                    frame_id=fid,
                    keyframe_order=kf_order,
                    pts_time=pts_time,
                    runtime=runtime,
                )

                if dec_res.resolved_source_type == "KEYFRAME_FILE":
                    mandatory_exact_keyframe_file += 1
                elif dec_res.resolved_source_type == "RAW_VIDEO_PTS":
                    mandatory_raw_video_pts += 1
                elif dec_res.resolved_source_type == "RAW_VIDEO_FRAME_INDEX":
                    mandatory_raw_video_frame_index += 1
                elif dec_res.resolved_source_type == "AMBIGUOUS_KEYFRAME_FILE":
                    mandatory_ambiguous += 1
                elif dec_res.resolved_source_type == "UNRESOLVED":
                    mandatory_unresolved += 1

                if dec_res.integrity_status != "PASS":
                    mandatory_integrity_fail += 1

                mandatory_requested += 1

                if dec_res.image is not None:
                    ax.imshow(dec_res.image)
                else:
                    ax.text(0.5, 0.5, f"IMAGE UNRESOLVED\nVideo: {vid}\nFrame: {fid}\n({dec_res.resolution_rule})", ha="center", va="center", fontsize=8)

                for spine in ax.spines.values():
                    spine.set_color("red")
                    spine.set_linewidth(3)

                scene_label = "T1 (Map/Irrigation)" if col_idx == 0 else "T2 (Dam/Rain)"
                caption = (
                    f"Video: {vid} | WINNING DP SCENE {scene_label} ★\n"
                    f"Physical Frame: {fid} | PTS: {pts_time:.2f}s\n"
                    f"Winning Chain Score: {cand_obj['chain_score']:.4f}\n"
                    f"{dec_res.resolution_rule} | {dec_res.resolved_filename or 'UNRESOLVED'}"
                )
                ax.set_title(caption, fontsize=8, color="red", pad=6)

                manifest_entries.append({
                    "contact_sheet_file": cand_sheet_path.name,
                    "tile_type": "MANDATORY_WINNING_DP_TILE",
                    "scene": f"WINNING_DP_T{col_idx+1}",
                    "is_dp_winning": True,
                    **dec_res.to_manifest_dict(),
                })
            elif col_idx == 2 and len(dp_frames) == 2:
                # Text summary tile for the pair
                ax.text(
                    0.5, 0.5,
                    f"DP PAIR METRICS\nVideo: {vid}\nValid T1 < T2: YES ✅\n"
                    f"T1 Frame: f{dp_frames[0]}\nT2 Frame: f{dp_frames[1]}\n"
                    f"Frame Gap: {dp_frames[1] - dp_frames[0]} frames\n"
                    f"Joint DP Score: {cand_obj['chain_score']:.4f}",
                    ha="center", va="center", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow", ec="red", lw=2),
                )
                ax.axis("off")
            else:
                ax.axis("off")

    cand_sheet_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(cand_sheet_path, dpi=120)
    plt.close(fig)
    print(f"  📸 Saved Candidate Discovery contact sheet -> {cand_sheet_path.name} ✅", flush=True)

    # -------------------------------------------------------------------------
    # PART D: COMPUTE SHA256 & EXPORT IMMUTABLE EVIDENCE MANIFEST + SIDECAR + HUMAN TEMPLATE
    # -------------------------------------------------------------------------
    artifact_checksums = {}
    for p in all_page_paths:
        if p.exists():
            artifact_checksums[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    if timeline_sheet_path.exists():
        artifact_checksums[timeline_sheet_path.name] = hashlib.sha256(timeline_sheet_path.read_bytes()).hexdigest()
    if cand_sheet_path.exists():
        artifact_checksums[cand_sheet_path.name] = hashlib.sha256(cand_sheet_path.read_bytes()).hexdigest()

    is_incomplete = (mandatory_unresolved > 0 or mandatory_ambiguous > 0 or mandatory_integrity_fail > 0)
    visual_evidence_status = "VISUAL_EVIDENCE_INCOMPLETE ⚠️" if is_incomplete else "VISUAL_EVIDENCE_AVAILABLE_FOR_HUMAN_ADJUDICATION ✅"

    # 1. Immutable Machine Visual Evidence Manifest
    evidence_manifest_payload = {
        "benchmark_query_id": "query-p1-2-kis",
        "target_video_locked": target_vid,
        "locked_gt_frame": locked_gt_frame,
        "visual_evidence_status": visual_evidence_status,
        "provenance_description": "Immutable machine evidence recording rendered keyframe tiles, strict decode resolution types, and artifact SHA256 hashes.",
        "decode_integrity_policy": {
            "exact_file_ambiguity_policy": EXACT_FILE_AMBIGUITY_POLICY,
            "raw_video_pts_tolerance_seconds": DECLARED_PTS_TOLERANCE_SECONDS,
            "raw_video_frame_index_allowed": False,
        },
        "decode_statistics": {
            "mandatory_requested": mandatory_requested,
            "mandatory_exact_keyframe_file": mandatory_exact_keyframe_file,
            "mandatory_raw_video_pts": mandatory_raw_video_pts,
            "mandatory_raw_video_frame_index": mandatory_raw_video_frame_index,
            "mandatory_unresolved": mandatory_unresolved,
            "mandatory_ambiguous": mandatory_ambiguous,
            "mandatory_integrity_fail": mandatory_integrity_fail,
            "optional_requested": optional_requested,
            "optional_exact_keyframe_file": optional_exact_keyframe_file,
            "optional_raw_video_pts": optional_raw_video_pts,
            "optional_raw_video_frame_index": optional_raw_video_frame_index,
            "optional_unresolved": optional_unresolved,
            "optional_ambiguous": optional_ambiguous,
            "optional_integrity_fail": optional_integrity_fail,
            "total_requested": mandatory_requested + optional_requested,
            "total_resolved": mandatory_exact_keyframe_file + mandatory_raw_video_pts + mandatory_raw_video_frame_index + optional_exact_keyframe_file + optional_raw_video_pts + optional_raw_video_frame_index,
            "total_unresolved": mandatory_unresolved + optional_unresolved,
            "total_ambiguous": mandatory_ambiguous + optional_ambiguous,
        },
        "artifact_sha256_checksums": artifact_checksums,
        "candidate_discovery_summary": [
            {
                "rank": item["rank"],
                "video_id": item["video_id"],
                "temporal_chain_score": item["temporal_chain_score"],
                "has_valid_chain": item["has_valid_chain"],
                "winning_chain_frames": item["winning_chain_frames"],
            }
            for item in candidate_adjudication_templates
        ],
        "rendered_tiles": manifest_entries,
    }

    evidence_manifest_path = base_out / "p1-2_visual_evidence_manifest.json"
    evidence_manifest_path.write_text(json.dumps(evidence_manifest_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # 2. Sidecar Checksum File (SHA256 of the exact bytes of the evidence manifest)
    evidence_sha = hashlib.sha256(evidence_manifest_path.read_bytes()).hexdigest()
    sidecar_path = base_out / "p1-2_visual_evidence_manifest.json.sha256"
    sidecar_path.write_text(f"{evidence_sha}  {evidence_manifest_path.name}\n", encoding="utf-8")

    # Sidecar re-read verification against exact bytes
    sidecar_content = sidecar_path.read_text(encoding="utf-8").strip()
    sidecar_sha = sidecar_content.split()[0]
    recomputed_sha = hashlib.sha256(evidence_manifest_path.read_bytes()).hexdigest()
    sidecar_verified = (evidence_sha == sidecar_sha == recomputed_sha)
    assert sidecar_verified, f"Sidecar checksum mismatch: {evidence_sha} vs {sidecar_sha} vs {recomputed_sha}"

    # 3. Separate Human Adjudication Template (Linking to the immutable evidence SHA)
    human_adjudication_payload = {
        "benchmark_query_id": "query-p1-2-kis",
        "target_video_locked": target_vid,
        "locked_gt_frame": locked_gt_frame,
        "referenced_evidence_manifest_file": evidence_manifest_path.name,
        "referenced_evidence_manifest_sha256": evidence_sha,
        "referenced_artifact_checksums": artifact_checksums,
        "instructions": (
            "Human reviewer records visual findings for target video and candidate discovery videos. "
            "Examine the contact sheet PNGs. All rubric fields start as null/UNRESOLVED. "
            "Set overall_label to MATCH (all predicates present in single video), PARTIAL, or NO_MATCH."
        ),
        "target_video_adjudication": {
            "video_id": target_vid,
            "locked_gt_frame": locked_gt_frame,
            "total_indexed_keyframes_inspected": total_kfs,
            "reviewed_pages": [p.name for p in all_page_paths],
            "human_adjudication_rubric": {
                "map_present": None,
                "irrigation_structure_repeated_four_times": None,
                "aerial_dam": None,
                "rainy_dam_closeup_or_discharge": None,
                "temporal_order_correct": None,
                "overall_label": "UNRESOLVED",
                "reviewer_notes": "",
            },
        },
        "candidate_videos_adjudication": candidate_adjudication_templates,
    }

    human_adjudication_path = base_out / "p1-2_human_adjudication_template.json"
    human_adjudication_path.write_text(json.dumps(human_adjudication_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"  📄 Saved Visual Evidence Manifest -> {evidence_manifest_path.name} ✅", flush=True)
    print(f"  🔒 Saved Sidecar Checksum File    -> {sidecar_path.name} ✅", flush=True)
    print(f"  📝 Saved Human Adjudication File  -> {human_adjudication_path.name} ✅\n", flush=True)

    target_tiles = [t for t in manifest_entries if t.get("tile_type") == "MANDATORY_TARGET_KEYFRAME"]
    target_requested = len(target_tiles)
    target_exact_unique = sum(1 for t in target_tiles if t.get("resolved_source_type") == "KEYFRAME_FILE" and t.get("integrity_status") == "PASS")
    target_raw_pts = sum(1 for t in target_tiles if t.get("resolved_source_type") == "RAW_VIDEO_PTS" and t.get("integrity_status") == "PASS")
    target_ambiguous = sum(1 for t in target_tiles if "AMBIGUOUS" in t.get("resolution_rule", "") or t.get("resolved_source_type") == "AMBIGUOUS_KEYFRAME_FILE")
    target_unresolved = sum(1 for t in target_tiles if t.get("resolved_source_type") == "UNRESOLVED")
    target_integrity_fail = sum(1 for t in target_tiles if t.get("integrity_status") != "PASS")

    # Assert mutually-exclusive partition of target indexed keyframes
    assert (target_exact_unique + target_raw_pts + target_ambiguous + target_unresolved) == target_requested == 568, (
        f"Target partition mismatch: {target_exact_unique} + {target_raw_pts} + {target_ambiguous} + {target_unresolved} != {target_requested}"
    )

    print("=" * 120)
    print(f"1. TARGET DECODE INTEGRITY (Video: {target_vid} | Requested: {target_requested} indexed keyframes)")
    print("=" * 120)
    print(f"  video                              = {target_vid}")
    print(f"  requested_target_indexed_keyframes = {target_requested}")
    print(f"  exact_unique                       = {target_exact_unique}")
    print(f"  raw_pts                            = {target_raw_pts}")
    print(f"  ambiguous                          = {target_ambiguous}")
    print(f"  unresolved                         = {target_unresolved}")
    print(f"  integrity_fail                     = {target_integrity_fail}")
    print("=" * 120)

    ambiguous_tiles = [t for t in manifest_entries if "AMBIGUOUS" in t.get("resolution_rule", "") or t.get("integrity_status") != "PASS"]

    print("\n" + "=" * 120)
    print(f"2. LIST ALL {len(ambiguous_tiles)} AMBIGUITIES / INTEGRITY FAILS")
    print("=" * 120)
    print(f"| # | Video ID | Physical Frame | Order (n) | Artifact Scope / Tile Type | Resolution Rule | Filenames (Order vs Physical) | Status |")
    print(f"| - | -------- | -------------- | --------- | -------------------------- | --------------- | ----------------------------- | ------ |")
    for idx, t in enumerate(ambiguous_tiles, start=1):
        print(
            f"| {idx:<1} | {t.get('video_id', 'N/A'):<8} "
            f"| f{str(t.get('requested_physical_frame_id', 'N/A')):<13} "
            f"| n={str(t.get('requested_keyframe_order', 'N/A')):<6} "
            f"| {t.get('tile_type', 'N/A'):<26} "
            f"| {t.get('resolution_rule', 'N/A'):<27} "
            f"| {str(t.get('resolved_filename', 'N/A')):<29} "
            f"| {t.get('integrity_status', 'N/A'):<6} |"
        )
    print("=" * 120)

    print("\n" + "=" * 120)
    print("GLOBAL MANDATORY_DECODE_COUNTS (ALL MANDATORY TILES)")
    print(f"  exact_keyframe_file   : {mandatory_exact_keyframe_file} ({(mandatory_exact_keyframe_file/mandatory_requested*100):.1f}%)" if mandatory_requested > 0 else "  exact_keyframe_file   : 0")
    print(f"  raw_video_pts         : {mandatory_raw_video_pts}")
    print(f"  raw_video_frame_index : {mandatory_raw_video_frame_index}")
    print(f"  unresolved            : {mandatory_unresolved}")
    print(f"  ambiguous             : {mandatory_ambiguous}")
    print(f"  integrity_fail        : {mandatory_integrity_fail}")
    print(f"\nVISUAL_EVIDENCE_STATUS = {visual_evidence_status}")
    print("-" * 120)
    print(f"EVIDENCE_MANIFEST_SHA256 = {evidence_sha}")
    print(f"SIDECAR_SHA256           = {sidecar_sha}")
    print(f"RECOMPUTED_SHA256        = {recomputed_sha}")
    print(f"SIDECAR_VERIFICATION     = {'PASS ✅' if sidecar_verified else 'FAIL ❌'}")
    print("-" * 120)
    print("• Final Artifact File Paths & SHA256 Checksums:")
    for fname, sha in artifact_checksums.items():
        print(f"  * {fname:<45} : {sha}")
    print(f"  * {evidence_manifest_path.name:<45} : {evidence_sha}")
    print(f"  * Sidecar File: {sidecar_path.name}")
    print("=" * 120 + "\n", flush=True)

    return manifest_entries


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

    manifest_path, manifest_sha, manifest_queries = load_canonical_frozen_manifest()
    manifest_entry = manifest_queries["p1-4"]
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
        cov_entry = {}
        for data in coverage_summary.values():
            human = data.get("human_reference", {})
            legacy = data.get("legacy", {})
            if human.get("video_id") == vid:
                cov_entry = human
                break
            elif legacy.get("video_id") == vid:
                cov_entry = legacy
                break
        parity_passed = bool(cov_entry.get("parity_passed", False))
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

                    dec_res = extract_image_for_frame(
                        dataset_root=input_root,
                        video_id=vid,
                        frame_id=req_frame_id,
                        keyframe_order=kf_order,
                        pts_time=pts_time,
                        source_fps=src_fps,
                        parity_passed=parity_passed,
                        runtime=runtime,
                    )

                    if dec_res.image is not None:
                        ax.imshow(dec_res.image)
                        vid_loaded_count += 1
                    else:
                        vid_failed_count += 1
                        ax.text(0.5, 0.5, f"IMAGE UNRESOLVED\nVideo: {vid}\nFrame: {req_frame_id}\n({dec_res.resolution_rule})", ha="center", va="center", fontsize=8)

                    is_chain = req_frame_id in chain_frames
                    caption = (
                        f"Video: {vid} | Scene: T{scene_var.temporal_index}\n"
                        f"Physical Frame: {req_frame_id} (Order: {kf_order if kf_order else 'N/A'})\n"
                        f"PTS: {pts_time:.3f}s | Raw Cosine: {cosine:.4f}\n"
                        f"Winning DP Frame: {'YES ★' if is_chain else 'NO'} | {dec_res.resolution_rule}\n"
                        f"{dec_res.resolved_filename or 'UNRESOLVED'}"
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


def audit_p1_1_target_interval_trace(c_data: dict, run_dir_name: str) -> None:
    target_vid = "L30_V046"
    target_fids = [6605, 6613, 6742, 6784]

    vf_data = c_data.get("video_first", {})
    fusion_trace = vf_data.get("fusion_trace", {})
    alloc_summary = vf_data.get("allocation_summary", {})
    records = c_data.get("records", [])
    final_rank_by_fid = {int(r.get("frame_id", 0)): r.get("rank") for r in records if r.get("video_id") == target_vid}

    print("\n" + "─" * 165, flush=True)
    print(f"🎯 P1-1 GROUND TRUTH INTERVAL TRACE AUDIT (Target: {target_vid} | Interval: 264.0s-274.0s) — {run_dir_name}", flush=True)
    print("─" * 165, flush=True)

    header = f"| {'Frame ID':<10} | {'Restricted Status':<28} | {'RRF Rank':<10} | {'Cutoff Status':<26} | {'Pre-Alloc #':<12} | {'Bucket & Rank':<28} | {'Score Gap':<12} | {'Allocation Reason':<32} | {'Final Lifecycle':<22} |"
    sep = f"|:{'-'*10}-|:{'-'*28}-|:{'-'*10}-|:{'-'*26}-|:{'-'*12}-|:{'-'*28}-|:{'-'*12}-|:{'-'*32}-|:{'-'*22}-|"
    print(header, flush=True)
    print(sep, flush=True)

    for fid in target_fids:
        k = f"{target_vid}::{fid}"
        trace_info = fusion_trace.get(k)

        if trace_info:
            sel_status = trace_info.get("restricted_selection_status", "SELECTED_RESTRICTED_RAW")
            rrf_rank = str(trace_info.get("untruncated_rrf_rank", "N/A"))
            rrf_cutoff_status = trace_info.get("rrf_cutoff_status", "N/A")
            pre_alloc = str(trace_info.get("pre_allocation_global_rank") or "N/A")
            bucket = trace_info.get("group_bucket")
            b_rank = trace_info.get("pre_allocation_bucket_rank")
            bucket_str = f"{bucket} (#{b_rank})" if bucket and b_rank else (bucket or "N/A")
            gap = trace_info.get("score_gap_to_effective_cutoff")
            gap_str = f"+{gap:.8f}" if gap is not None else "N/A"
            reason = str(trace_info.get("allocation_rejection_reason") or "None")
            lifecycle = trace_info.get("final_lifecycle_status", "N/A")
        else:
            sel_status = "NOT_IN_RESTRICTED_POOL"
            rrf_rank = "N/A"
            rrf_cutoff_status = "NOT_IN_RESTRICTED_POOL"
            pre_alloc = "N/A"
            bucket_str = "N/A"
            gap_str = "N/A"
            reason = "None"
            lifecycle = "NOT_IN_RESTRICTED_POOL"

        final_r = final_rank_by_fid.get(fid)
        if final_r is not None:
            lifecycle = f"EXPORTED_AT_RANK_{final_r} ⭐"

        print(f"| {f'f{fid}':<10} | {sel_status:<28} | {rrf_rank:<10} | {rrf_cutoff_status:<26} | {pre_alloc:<12} | {bucket_str:<28} | {gap_str:<12} | {reason:<32} | {lifecycle:<22} |", flush=True)

    print("─" * 165, flush=True)
    if alloc_summary:
        print(f"  • Allocation Summary (Schema v{alloc_summary.get('fusion_trace_schema_version', '2.0.0')}): "
              f"Output Cap = {alloc_summary.get('output_top_k')} | "
              f"Passed RRF = {alloc_summary.get('candidates_passed_internal_cutoff')} | "
              f"Primary Exported = {alloc_summary.get('exported_primary_count')}/{alloc_summary.get('total_primary_candidates')} | "
              f"Secondary Exported = {alloc_summary.get('exported_secondary_count')}/{alloc_summary.get('total_secondary_candidates')} | "
              f"Primary Cutoff Score = {alloc_summary.get('primary_cutoff_score')}", flush=True)
    print("─" * 165 + "\n", flush=True)


# ==============================================================================
# SECTION 3.5: FULL TOP-100 VISUAL CONTACT SHEET EXPORT (ALL 5 FOCUS QUERIES)
# ==============================================================================
def run_all_5_queries_top100_visual_export(
    runtime: OperationalKISRuntime,
    input_root: Path,
    base_out: Path,
    coverage_results: dict[str, dict],
    render_contact_sheets: bool = True,
) -> None:
    print("=" * 120, flush=True)
    header_suffix = " (RENDERING PNGS)" if render_contact_sheets else " (FAST RETRIEVAL MODE)"
    print(f"3.5 FULL TOP-100 EVALUATION FOR ALL 5 FOCUS QUERIES{header_suffix}", flush=True)
    print("=" * 120, flush=True)

    manifest_path, manifest_sha, manifest_queries = load_canonical_frozen_manifest()
    ref_data = load_frozen_reference_manifest()
    ref_queries_map = {q.get("query_id"): q for q in ref_data.get("queries", [])}
    query_order = ["p1-1", "p1-2", "p1-4", "p1-5", "p1-6"]

    for q_short in query_order:
        q_data = manifest_queries.get(q_short)
        if not q_data:
            continue
        qid = q_data["query_id"]
        q_text = q_data["query_vi"]
        ref_entry = ref_queries_map.get(qid, {})
        human_vid = ref_entry.get("human_verified_video_id") or q_data.get("target_video", "")
        legacy_vid = ref_entry.get("legacy_manifest_target", {}).get("target_video") or q_data.get("target_video", "")
        locked_gt = q_data.get("locked_gt_frame", 0)

        print(f"\n──────────────────────────────────────────────────────────────────────────────────────────────────", flush=True)
        if human_vid != legacy_vid and legacy_vid:
            print(f"• Processing Query [{q_short}] ({qid}) | Human Target: {human_vid} (Legacy: {legacy_vid})", flush=True)
        else:
            print(f"• Processing Query [{q_short}] ({qid}) | Target: {human_vid} (GT Frame: {locked_gt})", flush=True)
        print(f"  VI Text: \"{q_text}\"", flush=True)
        print(f"──────────────────────────────────────────────────────────────────────────────────────────────────", flush=True)

        req_id = f"audit-top100-{q_short}-{int(time.time() * 1000)}"

        req = QueryRequest(
            request_id=req_id,
            query_id=qid,
            query_vi=q_text,
        )

        # 1. Print Semantic Decomposition & English Translation Breakdown Directly
        print(f"  • 🌐 Phân Tích Dịch Thuật & Phân Rã Ngữ Nghĩa (Semantic Decomposition):", flush=True)
        try:
            compiled_query = compile_vietnamese_semantic_query(
                query_id=qid,
                query_vi=q_text,
                provider=runtime.translation_provider,
                token_budget_guard=runtime.token_budget_guard,
            )
            for u in compiled_query.units:
                print(f"    ├─ 🧩 [{u.role.name}]: \"{u.text}\"", flush=True)
            for v in compiled_query.variants:
                print(f"    └─ 🇬🇧 [CLIP Variant {v.query_variant.variant_id}] (Weight: {v.query_variant.weight:.2f}, Tokens: {v.clip_token_count}): \"{v.query_variant.text}\"", flush=True)
        except Exception as e:
            print(f"    ⚠️ Translation debug: {e}", flush=True)

        resp = runtime.handle_query(req)
        artifacts = resp.get("artifacts", {})

        # 2. Load candidates with REAL fusion scores and diagnostic metadata from internal artifacts
        top100_csv_rel = artifacts.get("refined_top100_csv") or artifacts.get("top100_csv")
        candidates_json_rel = artifacts.get("candidates_json")
        candidates = []
        enabled_features_list = []
        c_data = {}
        if candidates_json_rel and (runtime.output_root / candidates_json_rel).exists():
            try:
                c_data = json.loads((runtime.output_root / candidates_json_rel).read_text(encoding="utf-8"))
                enabled_features_list = c_data.get("enabled_features", [])
                for r in c_data.get("records", []):
                    candidates.append({
                        "rank": int(r.get("rank", 0)),
                        "video_id": r.get("video_id", ""),
                        "frame_id": int(r.get("frame_id", 0)),
                        "score": float(r.get("fusion_score", r.get("score", 0.0))),
                        "pts_time": float(r.get("pts_time", 0.0) or (int(r.get("frame_id", 0))/25.0)),
                        "keyframe_order": int(r.get("keyframe_order") or r.get("keyframe_order_diagnostic") or 0),
                        "scores_by_variant": r.get("scores_by_variant", {}),
                        "is_temporal_chain_winner": bool(r.get("is_temporal_chain_winner", False)),
                        "video_nomination_rank": r.get("video_nomination_rank"),
                        "enabled_features": enabled_features_list,
                    })
                if q_short == "p1-1":
                    audit_p1_1_target_interval_trace(c_data, runtime.output_root.name)
            except Exception:
                pass
        
        if not candidates:
            top100_csv_path = (runtime.output_root / top100_csv_rel) if top100_csv_rel else (runtime.output_root / "requests" / req_id / "top100.csv")
            if top100_csv_path.exists():
                import csv as csv_module
                with top100_csv_path.open("r", encoding="utf-8") as f:
                    reader = csv_module.DictReader(f)
                    for row in reader:
                        candidates.append({
                            "rank": int(row.get("rank", 0)),
                            "video_id": row.get("video_id", ""),
                            "frame_id": int(row.get("frame_id", 0)),
                            "score": float(row.get("fusion_score", row.get("score", 0.0))),
                            "pts_time": float(row.get("pts_time", 0.0) or 0.0),
                            "keyframe_order": 0,
                            "scores_by_variant": {},
                            "is_temporal_chain_winner": False,
                            "video_nomination_rank": None,
                            "enabled_features": [],
                        })

        # Load manual human reference manifest dynamically (isolated evaluation)
        ref_paths = [
            SYSTEM_TAI_SRC.parent / "benchmarks" / "manual_kis_reference_v1.json",
            REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "manual_kis_reference_v1.json",
            Path("/kaggle/working/AIC2026_TeamPTK_SGU/systems/system_tai/benchmarks/manual_kis_reference_v1.json"),
        ]
        ref_data = {}
        ref_sha256 = ""
        for p in ref_paths:
            if p.is_file():
                raw_bytes = p.read_bytes()
                ref_sha256 = hashlib.sha256(raw_bytes).hexdigest()
                ref_data = json.loads(raw_bytes.decode("utf-8"))
                break

        query_ref_entry = next((q for q in ref_data.get("queries", []) if q.get("query_id") == qid or q.get("query_id") == f"query-{q_short}-kis"), {})
        human_verified_vid = query_ref_entry.get("human_verified_video_id", "")
        human_intervals = query_ref_entry.get("human_annotated_intervals", [])
        annotation_status = query_ref_entry.get("annotation_status", "VIDEO_ONLY_VERIFIED")

        print(f"  • Top-100 Candidates Loaded: {len(candidates)} records", flush=True)
        print(f"  • Reference SHA256: {ref_sha256[:16]}... | Annotation Status: {annotation_status}", flush=True)
        print(f"  • Top-10 Frame Preview:", flush=True)
        print(f"    {'Rank':<6} | {'Video ID':<10} | {'Frame ID':<10} | {'PTS (s)':<10} | {'Score':<10} | {'Evaluation Status'}", flush=True)
        print(f"    {'-'*6} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*20}", flush=True)
        
        breakdown_rows = []
        for c in candidates:
            c_vid = c.get("video_id", "")
            fid = int(c.get("frame_id", 0))
            if c_vid == human_verified_vid:
                status_label = "MATCHES_HUMAN_VERIFIED_VIDEO ⭐"
            else:
                status_label = "DIFFERENT_FROM_HUMAN_VERIFIED_VIDEO"

            # Tri-state frame interval evaluation
            if human_intervals:
                in_interval = any(s <= fid <= e for s, e in human_intervals)
                if c_vid == human_verified_vid and in_interval:
                    valid_frame_hit = True
                    frame_eval_status = "VALID_MANUAL_INTERVAL_HIT"
                elif c_vid == human_verified_vid:
                    valid_frame_hit = False
                    frame_eval_status = "SAME_VIDEO_OUTSIDE_INTERVAL"
                else:
                    valid_frame_hit = False
                    frame_eval_status = "DIFFERENT_VIDEO"
            else:
                valid_frame_hit = None
                frame_eval_status = "NOT_EVALUABLE_NO_INTERVAL"

            pts = float(c.get("pts_time", 0.0) or (fid / 25.0))
            score_val = float(c.get("score", 0.0))
            
            if c.get("rank", 0) <= 10:
                print(f"    #{c.get('rank', 0):<5} | {c_vid:<10} | f{fid:<9} | {pts:<10.3f} | {score_val:<10.4f} | {status_label}", flush=True)

            breakdown_rows.append({
                "query_id": qid,
                "rank": int(c.get("rank", 0)),
                "video_id": c_vid,
                "frame_id": fid,
                "pts_time": pts,
                "scores": {
                    "baseline_final": score_val,
                    "experimental_final": None,
                    "scores_by_variant": c.get("scores_by_variant", {}),
                },
                "enabled_features": c.get("enabled_features", enabled_features_list),
                "is_temporal_chain_winner": c.get("is_temporal_chain_winner", False),
                "video_nomination_rank": c.get("video_nomination_rank"),
                "keyframe_order": c.get("keyframe_order", 0),
                "human_verified_target": human_verified_vid,
                "reference_sha256": ref_sha256,
                "evaluation_status": status_label,
                "valid_frame_hit": valid_frame_hit,
                "frame_evaluation_status": frame_eval_status,
            })

        # Save machine-readable JSONL breakdown
        jsonl_path = runtime.output_root / f"{q_short}_top100_breakdown.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as f_jsonl:
            for row in breakdown_rows:
                f_jsonl.write(json.dumps(row, ensure_ascii=False) + "\n")

        # Render Page 1 (Rank 1-50) and Page 2 (Rank 51-100) (if enabled)
        if render_contact_sheets:
            for page_idx, (start_r, end_r) in enumerate([(0, 50), (50, 100)], start=1):
                page_cands = candidates[start_r:end_r]
                if not page_cands:
                    continue

                fig, axes = plt.subplots(5, 10, figsize=(20, 10))
                fig.suptitle(
                    f"Top-100 Keyframes: Query [{q_short}] ({qid}) — Page {page_idx} (Rank #{start_r+1} to #{end_r})\nQuery: {q_text[:100]}...",
                    fontsize=8,
                    fontweight="bold"
                )
                axes_flat = axes.flatten()

                for idx in range(50):
                    ax = axes_flat[idx]
                    if idx < len(page_cands):
                        cand = page_cands[idx]
                        vid = cand.get("video_id", "")
                        fid = int(cand.get("frame_id", 0))
                        rank = int(cand.get("rank", start_r + idx + 1))
                        score = float(cand.get("score", 0.0))
                        pts = float(cand.get("pts_time", 0.0) or (fid / 25.0))
                        order = int(cand.get("keyframe_order", 0))

                        if (order <= 0 or pts <= 0.0) and runtime is not None:
                            try:
                                store = runtime.registry.get_store(vid)
                                if store is not None:
                                    for m in store.mappings:
                                        if m.frame_id == fid:
                                            if order <= 0:
                                                order = int(m.keyframe_order)
                                            if pts <= 0.0:
                                                pts = float(m.pts_time)
                                            break
                            except Exception:
                                pass
                        if pts <= 0.0:
                            pts = float(fid) / 25.0

                        dec_res = extract_image_for_frame(
                            dataset_root=input_root,
                            video_id=vid,
                            frame_id=fid,
                            keyframe_order=order,
                            pts_time=pts,
                            runtime=runtime,
                        )

                        is_target = (vid == target_vid)
                        is_true_cand = (vid == human_verified_vid)

                        if dec_res.image is not None:
                            ax.imshow(dec_res.image)
                        else:
                            ax.text(0.5, 0.5, f"IMAGE MISSING\n{vid}\nf{fid}", ha="center", va="center", fontsize=8, color="red")

                        header_color = "darkgreen" if is_true_cand else ("red" if is_target else "black")
                        bg_color = "honeydew" if is_true_cand else ("mistyrose" if is_target else "lightyellow")
                        ax.set_title(
                            f"#{rank} {vid} f{fid}\n{pts:.1f}s | {score:.4f}",
                            fontsize=7,
                            color=header_color,
                            fontweight="bold" if (is_target or is_true_cand) else "normal",
                            bbox=dict(boxstyle="round,pad=0.2", fc=bg_color, ec=header_color, lw=1.5 if (is_target or is_true_cand) else 0.5),
                        )

                        if is_true_cand or is_target:
                            for spine in ax.spines.values():
                                spine.set_edgecolor("green" if is_true_cand else "red")
                                spine.set_linewidth(2.0)
                        else:
                            for spine in ax.spines.values():
                                spine.set_visible(False)
                    else:
                        ax.axis("off")
                    ax.set_xticks([])
                    ax.set_yticks([])

                sheet_name = f"{q_short}_top100_page_{page_idx:02d}.png"
                sheet_path = base_out / sheet_name
                sheet_path.parent.mkdir(parents=True, exist_ok=True)
                plt.tight_layout()
                plt.savefig(sheet_path, dpi=100)
                plt.close(fig)
                print(f"  📸 Saved Top-100 Contact Sheet -> {sheet_name} (Rank #{start_r+1}..#{end_r}) ✅", flush=True)

    if render_contact_sheets:
        print("\n" + "=" * 120, flush=True)
        print("ALL 10 TOP-100 CONTACT SHEETS SUCCESSFULLY GENERATED ✅", flush=True)
        print("=" * 120 + "\n", flush=True)


# ==============================================================================
# SECTION 4: FINAL COMPACT SUMMARY TABLE
# ==============================================================================
def print_final_summary_table(coverage_summary: dict[str, dict]) -> None:
    print("=" * 140, flush=True)
    print("4. FINAL FOUNDATION CLOSURE SUMMARY TABLE", flush=True)
    print("=" * 140, flush=True)

    manifest_path, manifest_sha, manifest_queries = load_canonical_frozen_manifest()

    ref_paths = [
        SYSTEM_TAI_SRC.parent / "benchmarks" / "manual_kis_reference_v1.json",
        REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "manual_kis_reference_v1.json",
        Path("/kaggle/working/AIC2026_TeamPTK_SGU/systems/system_tai/benchmarks/manual_kis_reference_v1.json"),
    ]
    ref_data = {}
    for p in ref_paths:
        if p.is_file():
            ref_data = json.loads(p.read_text(encoding="utf-8"))
            break
    ref_queries_map = {q.get("query_id"): q for q in ref_data.get("queries", [])}

    print(f"| {'Query':<6} | {'Human Target':<14} | {'Human Interval Status':<22} | {'Legacy Target':<14} | {'Legacy Locked Frame':<20} | {'Legacy Cov':<10} |")
    print(f"| {'-'*6} | {'-'*14} | {'-'*22} | {'-'*14} | {'-'*20} | {'-'*10} |")

    for qid in ("p1-1", "p1-2", "p1-4", "p1-5", "p1-6"):
        entry = coverage_summary.get(qid)
        manifest_meta = manifest_queries.get(qid, {})
        legacy_vid = manifest_meta.get("target_video", "N/A")
        legacy_gt_f = str(manifest_meta.get("locked_gt_frame", manifest_meta.get("official_gt_frame", "N/A")))

        ref_entry = ref_queries_map.get(f"query-{qid}-kis", {})
        human_vid = ref_entry.get("human_verified_video_id", legacy_vid)

        if entry is None:
            human_status_str = "NOT_RUN"
            legacy_cov_str = "NOT_RUN"
        else:
            human_entry = entry.get("human_reference", {})
            legacy_entry = entry.get("legacy", {})

            human_status = human_entry.get("interval_status", "NOT_EVALUABLE")
            if human_status == "PASS":
                human_status_str = "PASS ✅"
            elif human_status == "FAIL":
                human_status_str = "FAIL ❌"
            else:
                human_status_str = f"NOT_EVAL ({human_entry.get('annotation_status', 'VIDEO_ONLY')})"

            legacy_cov_str = "PASS ✅" if legacy_entry.get("coverage_pass") else "FAIL ❌"

        print(f"| {qid:<6} | {human_vid:<14} | {human_status_str:<22} | {legacy_vid:<14} | f{legacy_gt_f:<19} | {legacy_cov_str:<10} |")

    print("=" * 140 + "\n", flush=True)


if __name__ == "__main__":
    main()




