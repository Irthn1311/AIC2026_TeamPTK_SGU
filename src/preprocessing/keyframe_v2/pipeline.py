from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

from .candidate_sampler import make_candidate_frames, make_targets, margin_guard
from .clip_scorer import ImageEmbeddingScorer
from .deduplicator import cross_shot_deduplicate
from .exact_decoder import ExactFrameDecoder, compare_images
from .frame_mapper import FrameMapper
from .quality_scorer import score_quality
from .selector import duplicate_penalty, final_score, representative_scores, temporal_score
from .shot_detector import detect_shots
from .video_metadata import probe_video
from .visualizer import make_contact_sheet


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_keyframe_v2(video_path: str | Path, config_path: str | Path, output_root: str | Path, validate_btc_mapping: bool = True, debug: bool = True) -> dict:
    started = time.time()
    project_root = Path.cwd()
    cfg = load_config(config_path)
    _force_e_local_env(project_root, cfg)
    if os.environ.get("AIC_ALLOW_HISTDIFF_FALLBACK", "0") == "1":
        cfg.setdefault("shot_detection", {})["require_transnetv2"] = False

    video_path = Path(video_path).resolve()
    video_id = video_path.stem
    out_dir = Path(output_root).resolve() / video_id
    debug_dir = out_dir / "debug"
    keyframe_dir = out_dir / "keyframes"
    out_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)
    keyframe_dir.mkdir(parents=True, exist_ok=True)

    timings: dict[str, float] = {}
    warnings: list[str] = []
    t = time.time()
    meta = probe_video(video_path, float(cfg["video"].get("cfr_tolerance", 0.001)))
    meta.to_json(out_dir / "video_metadata.json")
    timings["metadata_fps_check"] = time.time() - t

    discovery = discover_project(project_root, cfg, video_id)
    btc_map = pd.read_csv(discovery["btc_mapping"])
    if validate_btc_mapping:
        t = time.time()
        val_df, convention, validation_counts = validate_frame_convention(video_path, btc_map, discovery["btc_keyframe_video_root"], meta.reported_fps, cfg, debug_dir)
        val_df.to_csv(out_dir / "frame_id_validation_extended.csv", index=False, encoding="utf-8-sig")
        val_df.to_csv(out_dir / "frame_id_validation.csv", index=False, encoding="utf-8-sig")
        timings["frame_id_validation"] = time.time() - t
    else:
        convention = "0-based"
        validation_counts = {}
        warnings.append("BTC mapping validation disabled; defaulted to 0-based")
    mapper = FrameMapper(convention, meta.reported_fps)

    t = time.time()
    shots, shot_warnings = detect_shots(video_path, meta, cfg["shot_detection"])
    warnings.extend(shot_warnings)
    shots_df = pd.DataFrame([s.asdict() for s in shots])
    shots_df.to_csv(out_dir / "shots.csv", index=False, encoding="utf-8-sig")
    timings["shot_detection"] = time.time() - t

    t = time.time()
    decoder = ExactFrameDecoder(video_path)
    embedder = ImageEmbeddingScorer(project_root, cfg.get("clip", {}))
    if embedder.warning:
        warnings.append(embedder.warning)

    candidates, selected_rows, frame_embeddings = score_and_select_candidates(
        decoder, mapper, shots, meta.reported_fps, cfg, embedder, debug_dir, write_candidate_images=debug
    )
    candidates_df = pd.DataFrame(candidates)
    selected_df = pd.DataFrame(selected_rows)
    timings["candidate_decoding_clip_quality_selection"] = time.time() - t

    t = time.time()
    final_df, dedup_df = cross_shot_deduplicate(selected_df, frame_embeddings, cfg["scoring"])
    timings["dedup"] = time.time() - t

    t = time.time()
    final_rows = save_final_keyframes(decoder, mapper, final_df, keyframe_dir)
    decoder.close()
    final_df = pd.DataFrame(final_rows)
    final_df.to_csv(out_dir / "final_keyframes.csv", index=False, encoding="utf-8-sig")
    selected_df.to_csv(out_dir / "selected_before_dedup.csv", index=False, encoding="utf-8-sig")
    candidates_df.to_csv(out_dir / "candidates.csv", index=False, encoding="utf-8-sig")
    dedup_df.to_csv(out_dir / "cross_shot_dedup.csv", index=False, encoding="utf-8-sig")
    write_v2_map(final_df, out_dir)
    timings["save_outputs"] = time.time() - t

    t = time.time()
    if debug:
        make_debug_sheets(candidates_df, final_df, debug_dir, cfg)
    timings["visualization"] = time.time() - t

    t = time.time()
    final_sample_count = int(cfg.get("frame_validation", {}).get("final_sample_count", 20))
    validation_df = validate_final_frames(video_path, final_df, int(min(final_sample_count, len(final_df))), cfg)
    validation_df.to_csv(out_dir / "final_frame_validation.csv", index=False, encoding="utf-8-sig")
    timings["final_frame_validation"] = time.time() - t

    summary = make_summary(video_id, meta, convention, validation_counts, shots_df, candidates_df, selected_df, dedup_df, final_df, btc_map, timings, discovery, embedder.backend, embedder.info, warnings, started)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def _force_e_local_env(project_root: Path, cfg: dict) -> None:
    model_cache = project_root / cfg["paths"].get("model_cache", ".model_cache")
    cache = project_root / ".cache"
    for key, path in {
        "HF_HOME": cache / "huggingface",
        "HUGGINGFACE_HUB_CACHE": cache / "huggingface" / "hub",
        "TRANSFORMERS_CACHE": cache / "huggingface",
        "TORCH_HOME": model_cache / "torch",
        "XDG_CACHE_HOME": cache,
        "TMP": project_root / "outputs" / "tmp",
        "TEMP": project_root / "outputs" / "tmp",
    }.items():
        os.environ[str(key)] = str(path)
        Path(path).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def discover_project(project_root: Path, cfg: dict, video_id: str) -> dict:
    paths = cfg["paths"]
    video_root = project_root / paths["video_root"]
    btc_root = project_root / paths["btc_keyframe_root"]
    mapping_root = project_root / paths["btc_mapping_root"]
    global_id_map = project_root / paths["global_id_map"]
    clip_feature_root = project_root / paths["clip_feature_root"]
    return {
        "video_root": str(video_root),
        "btc_keyframe_root": str(btc_root),
        "btc_keyframe_video_root": str(btc_root / video_id),
        "btc_mapping": str(mapping_root / f"{video_id}.csv"),
        "global_id_map": str(global_id_map),
        "clip_feature_source": str(clip_feature_root / f"{video_id}.npy"),
        "clip_model_current": "BTC precomputed CLIP ViT-B/32 features; local clip package installed; image weights only used if cached",
        "transnetv2": "transnetv2_pytorch if installed, otherwise histdiff_fallback with warning",
        "btc_filename_convention": "n column -> %03d.jpg; frame_idx is original frame id",
    }


def validate_frame_convention(video_path: Path, btc_map: pd.DataFrame, btc_kf_root: str, fps: float, cfg: dict, debug_dir: Path) -> tuple[pd.DataFrame, str, dict]:
    sample_count = int(cfg["frame_validation"].get("sample_count", 5))
    idxs = _validation_sample_indices(len(btc_map), cfg["frame_validation"])
    decoder = ExactFrameDecoder(video_path)
    records = []
    sheet_items = []
    mismatch_items = []
    for idx in idxs:
        row = btc_map.iloc[idx]
        n = int(row["n"])
        mapped = int(row["frame_idx"])
        btc_img_path = Path(btc_kf_root) / f"{n:03d}.jpg"
        btc_img = cv2.imread(str(btc_img_path))
        if btc_img is None:
            records.append({"btc_keyframe_name": f"{n:03d}.jpg", "status": "missing_btc_image"})
            continue
        sims = {}
        images = {"BTC ORIGINAL": btc_img}
        for label, fid in [("minus_1", mapped - 1), ("exact", mapped), ("plus_1", mapped + 1)]:
            if fid < 0:
                sims[label] = np.nan
                continue
            try:
                dec = decoder.decode(fid)
                sims[label] = compare_images(btc_img, dec.image_bgr, int(cfg["frame_validation"].get("ssim_resize_width", 320)))
                images[f"decoded({fid})"] = dec.image_bgr
            except Exception:
                sims[label] = np.nan
        best_label = max((k for k, v in sims.items() if not pd.isna(v)), key=lambda k: sims[k], default="unknown")
        best_frame = {"minus_1": mapped - 1, "exact": mapped, "plus_1": mapped + 1}.get(best_label)
        convention = "1-based" if best_label == "minus_1" else ("0-based" if best_label == "exact" else ("plus_1_anomaly" if best_label == "plus_1" else "uncertain"))
        records.append(
            {
                "btc_keyframe_name": f"{n:03d}.jpg",
                "btc_keyframe_idx": n,
                "btc_mapped_frame_id": mapped,
                "test_frame_minus_1": mapped - 1 if mapped > 0 else "",
                "similarity_minus_1": sims.get("minus_1"),
                "test_frame_exact": mapped,
                "similarity_exact": sims.get("exact"),
                "test_frame_plus_1": mapped + 1,
                "similarity_plus_1": sims.get("plus_1"),
                "best_matching_frame": best_frame,
                "detected_convention": convention,
                "status": "ok" if convention != "uncertain" else "uncertain",
            }
        )
        items = _validation_sheet_items(debug_dir, n, mapped, images, sims, best_frame, convention)
        sheet_items.extend(items)
        if best_label != "exact":
            mismatch_items.extend(items)
    decoder.close()
    df = pd.DataFrame(records)
    counts = df["detected_convention"].value_counts().to_dict()
    exact = int(counts.get("0-based", 0))
    one_based = int(counts.get("1-based", 0))
    plus_one = int(counts.get("plus_1_anomaly", 0))
    uncertain = int(counts.get("uncertain", 0))
    total = int(len(df))
    if exact <= max(one_based, plus_one, uncertain):
        raise RuntimeError(
            "BTC frame-id validation did not find a dominant exact 0-based convention: "
            f"exact={exact}, fid-1={one_based}, fid+1={plus_one}, uncertain={uncertain}, total={total}"
        )
    final = "0-based"
    make_contact_sheet(sheet_items, "image_path", debug_dir / "frame_id_validation_contact_sheet.jpg", "BTC frame-id validation", cols=4, thumb_w=260, thumb_h=146)
    if mismatch_items:
        make_contact_sheet(mismatch_items, "image_path", debug_dir / "frame_id_validation_mismatches_contact_sheet.jpg", "BTC frame-id validation mismatches", cols=4, thumb_w=260, thumb_h=146)
    return df, final, {
        "total": total,
        "exact_matches": exact,
        "fid_minus_1_matches": one_based,
        "fid_plus_1_matches": plus_one,
        "uncertain": uncertain,
    }


def _validation_sample_indices(total_rows: int, cfg: dict) -> list[int]:
    if total_rows <= 0:
        return []
    sample_count = max(25, int(cfg.get("sample_count", 25)))
    tail_count = max(0, int(cfg.get("tail_sample_count", 10)))
    tail_start_ratio = float(cfg.get("tail_start_ratio", 0.90))
    base_count = max(1, sample_count - tail_count)
    base = np.linspace(0, total_rows - 1, min(base_count, total_rows)).round().astype(int).tolist()
    tail_start = min(total_rows - 1, max(0, int(round((total_rows - 1) * tail_start_ratio))))
    tail = np.linspace(tail_start, total_rows - 1, min(tail_count, total_rows - tail_start)).round().astype(int).tolist()
    idxs = sorted(set(base + tail))
    if len(idxs) < min(sample_count, total_rows):
        extras = np.linspace(0, total_rows - 1, min(sample_count, total_rows)).round().astype(int).tolist()
        idxs = sorted(set(idxs + extras))
    return idxs


def _validation_sheet_items(debug_dir: Path, n: int, mapped: int, images: dict, sims: dict, best_frame: int | None, convention: str) -> list[dict]:
    temp_dir = debug_dir / "_validation_frames"
    temp_dir.mkdir(parents=True, exist_ok=True)
    items = []
    for label, img in images.items():
        out = temp_dir / f"{n:03d}_{label.replace(' ', '_').replace('(', '').replace(')', '')}.jpg"
        cv2.imwrite(str(out), img)
        items.append(
            {
                "image_path": str(out),
                "label_lines": [
                    f"BTC {n:03d} map={mapped}",
                    label,
                    f"sims m/e/p={sims.get('minus_1', np.nan):.3f}/{sims.get('exact', np.nan):.3f}/{sims.get('plus_1', np.nan):.3f}",
                    f"match={best_frame} {convention}",
                ],
            }
        )
    return items


def score_and_select_candidates(
    decoder: ExactFrameDecoder,
    mapper: FrameMapper,
    shots: list,
    fps: float,
    cfg: dict,
    embedder: ImageEmbeddingScorer,
    debug_dir: Path,
    write_candidate_images: bool = False,
):
    candidates = []
    selected_rows = []
    frame_embeddings: dict[int, np.ndarray] = {}
    soft_dup = float(cfg["scoring"].get("soft_duplicate_threshold", 0.90))
    hard_dup = float(cfg["scoring"].get("hard_duplicate_threshold", 0.965))
    window_frames = max(1, int(round(float(cfg["candidates"].get("candidate_window_seconds", 0.5)) * fps)))
    candidate_img_dir = debug_dir / "candidate_frames"
    if write_candidate_images:
        candidate_img_dir.mkdir(parents=True, exist_ok=True)

    for shot in shots:
        selected_in_shot: list[np.ndarray] = []
        for target in make_targets(shot.start_frame, shot.end_frame, fps, cfg["targets"]):
            guard = margin_guard(shot.start_frame, shot.end_frame, cfg["margin_guard"])
            frames = make_candidate_frames(target, shot.start_frame, shot.end_frame, fps, cfg["candidates"], guard)
            duration = (shot.end_frame - shot.start_frame + 1) / fps
            if float(cfg["targets"]["short_max_seconds"]) <= duration < float(cfg["targets"]["medium_max_seconds"]):
                ratio_frames = [
                    int(round(shot.start_frame + (shot.end_frame - shot.start_frame) * float(r)))
                    for r in cfg["targets"].get("medium_candidate_ratios", [0.35, 0.50, 0.65])
                ]
                frames = sorted(set(frames + [max(shot.start_frame, min(shot.end_frame, f)) for f in ratio_frames]))
            images = []
            quality_rows = []
            for fid in frames:
                dec = decoder.decode(fid)
                candidate_image_path = candidate_img_dir / f"shot_{int(shot.shot_id):06d}_target_{int(target.target_id):03d}_frame_{fid:06d}.jpg"
                if write_candidate_images:
                    cv2.imwrite(str(candidate_image_path), dec.image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
                images.append(dec.image_bgr)
                q = score_quality(dec.image_bgr, cfg["quality"])
                row = {
                    "video_id": shot.video_id,
                    "shot_id": int(shot.shot_id),
                    "shot_start_frame": int(shot.start_frame),
                    "shot_end_frame": int(shot.end_frame),
                    "target_id": int(target.target_id),
                    "target_ratio": float(target.target_ratio),
                    "target_frame": int(target.target_frame),
                    "candidate_frame_internal": int(fid),
                    "candidate_actual_frame_id": mapper.internal_to_btc_frame_id(fid),
                    "timestamp": mapper.frame_to_timestamp(fid),
                    "distance_from_target": abs(int(fid) - int(target.target_frame)),
                    "inside_margin_guard": bool(shot.start_frame + guard <= fid <= shot.end_frame - guard) if shot.start_frame + guard <= shot.end_frame - guard else True,
                    "image_path": str(candidate_image_path) if write_candidate_images else "",
                    "selected": False,
                    "reject_reason": "",
                    **q,
                }
                quality_rows.append(row)
            embs = embedder.embed_images(images)
            reps = representative_scores(embs)
            for row, emb, rep in zip(quality_rows, embs, reps):
                frame_embeddings[int(row["candidate_frame_internal"])] = emb
                max_sim, penalty, dup_status = duplicate_penalty(emb, selected_in_shot, soft_dup)
                row["representative_score"] = float(rep)
                row["temporal_score"] = temporal_score(int(row["candidate_frame_internal"]), int(row["target_frame"]), window_frames)
                row["duplicate_similarity"] = max_sim
                row["duplicate_penalty"] = penalty
                row["duplicate_status"] = "duplicate_hard" if max_sim >= hard_dup else dup_status
                row["final_score"] = final_score(row, cfg["scoring"])
            valid = [r for r in quality_rows if r["inside_margin_guard"] and r["duplicate_status"] != "duplicate_hard"]
            if not valid:
                valid = [r for r in quality_rows if r["inside_margin_guard"]] or quality_rows
            winner = max(valid, key=lambda r: r["final_score"])
            selected_in_shot.append(frame_embeddings[int(winner["candidate_frame_internal"])])
            for row in quality_rows:
                if row is winner:
                    row["selected"] = True
                    row["reject_reason"] = ""
                    selected_rows.append(row.copy())
                elif row["duplicate_status"] == "duplicate_hard":
                    row["reject_reason"] = "duplicate_hard"
                elif not row["inside_margin_guard"]:
                    row["reject_reason"] = "boundary_guard"
                else:
                    row["reject_reason"] = "lower_score"
                candidates.append(row)
    return candidates, selected_rows, frame_embeddings


def save_final_keyframes(decoder: ExactFrameDecoder, mapper: FrameMapper, final_df: pd.DataFrame, keyframe_dir: Path) -> list[dict]:
    rows = []
    for idx, row in enumerate(final_df.sort_values("candidate_frame_internal").to_dict("records")):
        internal = int(row["candidate_frame_internal"])
        actual = mapper.internal_to_btc_frame_id(internal)
        img_path = keyframe_dir / f"kf_{idx:06d}_frame_{actual:06d}.jpg"
        decoder.save(internal, img_path)
        new = dict(row)
        new.update(
            {
                "keyframe_v2_idx": idx,
                "actual_frame_id": actual,
                "actual_frame_id_internal": internal,
                "btc_frame_id_convention": mapper.btc_convention,
                "timestamp_sec": mapper.frame_to_timestamp(internal),
                "timestamp_ms": int(round(mapper.frame_to_timestamp(internal) * 1000)),
                "shot_start_frame": int(row.get("shot_start_frame", -1)) if "shot_start_frame" in row else "",
                "shot_end_frame": int(row.get("shot_end_frame", -1)) if "shot_end_frame" in row else "",
                "target_position": float(row["target_ratio"]),
                "image_path": str(img_path),
            }
        )
        rows.append(new)
    return rows


def write_v2_map(final_df: pd.DataFrame, out_dir: Path) -> None:
    cols = ["video_id", "keyframe_v2_idx", "actual_frame_id", "timestamp_ms", "shot_id", "image_path"]
    m = final_df[cols].copy() if not final_df.empty else pd.DataFrame(columns=cols)
    m.insert(0, "global_id", range(len(m)))
    m.to_csv(out_dir / "keyframe_v2_map.csv", index=False, encoding="utf-8-sig")
    m.to_parquet(out_dir / "keyframe_v2_map.parquet", index=False)


def make_debug_sheets(candidates_df: pd.DataFrame, final_df: pd.DataFrame, debug_dir: Path, cfg: dict) -> None:
    vis = cfg["visualization"]
    final_items = []
    for row in final_df.to_dict("records"):
        final_items.append(
            {
                "image_path": row["image_path"],
                "label_lines": [
                    f"KF #{int(row['keyframe_v2_idx']):03d} frame {int(row['actual_frame_id'])}",
                    f"{float(row['timestamp_sec']):.3f}s shot {int(row['shot_id'])}",
                    f"Q={float(row['quality_score']):.2f} R={float(row['representative_score']):.2f} T={float(row['temporal_score']):.2f}",
                    f"D={float(row['duplicate_penalty']):.2f} S={float(row['final_score']):.3f}",
                ],
            }
        )
    make_contact_sheet(final_items, "image_path", debug_dir / "timeline_contact_sheet.jpg", "FINAL KEYFRAME V2 timeline", int(vis["contact_sheet_cols"]), int(vis["thumb_width"]), int(vis["thumb_height"]))

    max_sheets = int(vis.get("max_shot_sheets", 9999))
    for count, (shot_id, group) in enumerate(candidates_df.groupby("shot_id")):
        if count >= max_sheets:
            break
        items = []
        for row in group.to_dict("records"):
            items.append(
                {
                    "image_path": row["image_path"],
                    "label_lines": [
                        f"shot {int(shot_id)} target {int(row['target_id'])}",
                        f"frame {int(row['candidate_actual_frame_id'])} {float(row['timestamp']):.2f}s",
                        f"Q={float(row['quality_score']):.2f} R={float(row['representative_score']):.2f} T={float(row['temporal_score']):.2f}",
                        f"D={float(row['duplicate_penalty']):.2f} S={float(row['final_score']):.3f}",
                        "SELECTED" if row["selected"] else str(row["reject_reason"]),
                    ],
                }
            )
        make_contact_sheet(items, "image_path", debug_dir / f"shot_{int(shot_id):06d}_contact_sheet.jpg", f"SHOT {int(shot_id)} candidates", int(vis["contact_sheet_cols"]), int(vis["thumb_width"]), int(vis["thumb_height"]))


def validate_final_frames(video_path: Path, final_df: pd.DataFrame, sample_count: int, cfg: dict) -> pd.DataFrame:
    if final_df.empty:
        return pd.DataFrame()
    idxs = np.linspace(0, len(final_df) - 1, sample_count).round().astype(int).tolist()
    decoder = ExactFrameDecoder(video_path)
    records = []
    for idx in idxs:
        row = final_df.iloc[idx]
        img = cv2.imread(str(row["image_path"]))
        dec = decoder.decode(int(row["actual_frame_id_internal"]))
        sim = compare_images(img, dec.image_bgr, int(cfg["frame_validation"].get("ssim_resize_width", 320))) if img is not None else np.nan
        pix = normalized_pixel_error(img, dec.image_bgr) if img is not None else np.nan
        min_ssim = float(cfg["frame_validation"].get("final_match_min_ssim", 0.98))
        max_pixel_error = float(cfg["frame_validation"].get("final_match_max_pixel_error", 0.01))
        status = "ok" if sim >= min_ssim and pix <= max_pixel_error else "mismatch"
        records.append(
            {
                "video_id": str(row["video_id"]),
                "keyframe_v2_idx": int(row["keyframe_v2_idx"]),
                "actual_frame_id": int(row["actual_frame_id"]),
                "actual_frame_id_internal": int(row["actual_frame_id_internal"]),
                "timestamp_sec": float(row["timestamp_sec"]),
                "pixel_error": pix,
                "ssim": sim,
                "similarity_to_decoded_original": sim,
                "validation_status": status,
                "status": status,
            }
        )
    decoder.close()
    return pd.DataFrame(records)


def normalized_pixel_error(a_bgr: np.ndarray, b_bgr: np.ndarray) -> float:
    if a_bgr.shape[:2] != b_bgr.shape[:2]:
        b_bgr = cv2.resize(b_bgr, (a_bgr.shape[1], a_bgr.shape[0]), interpolation=cv2.INTER_AREA)
    return float(np.mean(np.abs(a_bgr.astype(np.float32) - b_bgr.astype(np.float32))) / 255.0)


def make_summary(video_id: str, meta, convention: str, validation_counts: dict, shots_df: pd.DataFrame, candidates_df: pd.DataFrame, selected_df: pd.DataFrame, dedup_df: pd.DataFrame, final_df: pd.DataFrame, btc_map: pd.DataFrame, timings: dict, discovery: dict, repr_backend: str, clip_info: dict, warnings: list[str], started: float) -> dict:
    durations = shots_df["duration_sec"] if not shots_df.empty else pd.Series(dtype=float)
    timings = {k: round(float(v), 3) for k, v in timings.items()}
    timings["total"] = round(time.time() - started, 3)
    return {
        "video": {
            "video_id": video_id,
            "duration": meta.duration_sec,
            "resolution": f"{meta.width}x{meta.height}",
            "fps": meta.reported_fps,
            "fps_exact_fraction": meta.avg_frame_rate,
            "total_original_frames": meta.total_frames,
            "detected_frame_convention": convention,
            "probe_backend": meta.probe_backend,
        },
        "project_discovery": discovery,
        "shot": {
            "total_shots": int(len(shots_df)),
            "<2s": int((durations < 2).sum()),
            "2-8s": int(((durations >= 2) & (durations < 8)).sum()),
            "8-20s": int(((durations >= 8) & (durations < 20)).sum()),
            ">=20s": int((durations >= 20).sum()),
            "detector_backend": shots_df["detector_backend"].iloc[0] if not shots_df.empty else "",
        },
        "btc_validation": validation_counts,
        "candidates": {
            "targets": int(selected_df.shape[0]),
            "candidate_frames": int(candidates_df.shape[0]),
            "boundary_rejected": int((candidates_df.get("reject_reason", pd.Series(dtype=str)) == "boundary_guard").sum()),
            "quality_rejected": 0,
            "duplicate_rejected": int(candidates_df.get("reject_reason", pd.Series(dtype=str)).astype(str).str.contains("duplicate").sum()) if not candidates_df.empty else 0,
        },
        "keyframes": {
            "selected_before_dedup": int(len(selected_df)),
            "cross_shot_removed": int(len(dedup_df)),
            "final_keyframes": int(len(final_df)),
            "avg_keyframes_per_shot": round(float(len(final_df) / max(1, len(shots_df))), 3),
            "avg_seconds_per_keyframe": round(float(meta.duration_sec / max(1, len(final_df))), 3),
        },
        "btc_comparison": {
            "btc_keyframes_same_video": int(len(btc_map)),
            "v2_keyframes": int(len(final_df)),
            "ratio_v2_btc": round(float(len(final_df) / max(1, len(btc_map))), 3),
        },
        "performance": timings,
        "representativeness_backend": repr_backend,
        "clip_info": clip_info,
        "warnings": warnings,
    }
