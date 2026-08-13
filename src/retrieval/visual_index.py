from __future__ import annotations

import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

from .clip_text_encoder import ClipTextEncoder
from .logging_utils import setup_logger, stage_summary
from .mapping_loader import load_keyframe_mapping
from .visual_feature_loader import load_visual_feature_file, sanitize_feature_batch


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def build_visual_index(dataset_root: str | Path, output_dir: str | Path, id_map_output: str | Path, batch_size: int = 4096, logger=None):
    dataset_root = Path(dataset_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    id_map_output = Path(id_map_output)
    id_map_output.parent.mkdir(parents=True, exist_ok=True)
    logger = logger or setup_logger("visual_index")

    started = time.time()
    feature_root = dataset_root / "clip-features-32-aic25-b1" / "clip-features-32"
    mapping_root = dataset_root / "map-keyframes-aic25-b1" / "map-keyframes"
    keyframe_root = dataset_root / "Keyframes_L21" / "keyframes"

    clip_model = ClipTextEncoder()
    feature_files = sorted(feature_root.glob("L21_V*.npy"))
    index = None
    rows = []
    excluded = []
    total_added = 0

    for feature_path in feature_files:
        video_id = feature_path.stem
        mapping_path = mapping_root / f"{video_id}.csv"
        if not mapping_path.exists():
            excluded.append({"video_id": video_id, "feature_path": str(feature_path), "reason": "missing_mapping"})
            continue
        try:
            feat = load_visual_feature_file(feature_path, mmap_mode="r")
            mapping = load_keyframe_mapping(mapping_path, keyframe_root)
            if len(mapping) != feat.shape[0]:
                excluded.append({"video_id": video_id, "feature_path": str(feature_path), "reason": f"row_mismatch feature={feat.shape[0]} mapping={len(mapping)}"})
                continue
            if feat.ndim != 2:
                excluded.append({"video_id": video_id, "feature_path": str(feature_path), "reason": f"bad_ndim {feat.ndim}"})
                continue
            if index is None:
                index = faiss.IndexFlatIP(feat.shape[1])
            if index.d != feat.shape[1]:
                excluded.append({"video_id": video_id, "feature_path": str(feature_path), "reason": f"dim_mismatch index={index.d} feat={feat.shape[1]}"})
                continue

            logger.info("Video: %s", video_id)
            logger.info("Feature path: %s", feature_path)
            logger.info("Shape: %s", feat.shape)
            logger.info("Mapping rows: %s", len(mapping))
            for start in range(0, feat.shape[0], batch_size):
                batch = sanitize_feature_batch(feat[start : start + batch_size])
                index.add(batch)
            added = feat.shape[0]
            total_added += added
            logger.info("Added vectors: %s", added)
            logger.info("Total indexed: %s", total_added)

            for _, row in mapping.iterrows():
                rows.append(
                    {
                        "global_id": len(rows),
                        "video_id": video_id,
                        "local_feature_index": int(row["feature_index"]),
                        "keyframe_name": row["keyframe_name"],
                        "keyframe_path": row["keyframe_path"],
                        "frame_idx": int(row["frame_idx"]),
                        "timestamp_seconds": float(row["timestamp_seconds"]),
                        "timestamp_text": f"{float(row['timestamp_seconds']):.3f}s",
                        "video_path": str(dataset_root / "Videos_L21_a" / "video" / f"{video_id}.mp4"),
                        "feature_source": str(feature_path),
                        "mapping_source": str(mapping_path),
                        "ocr_text": "",
                        "ocr_mean_confidence": np.nan,
                        "ocr_num_boxes": 0,
                        "has_ocr": False,
                    }
                )
        except Exception as exc:
            excluded.append({"video_id": video_id, "feature_path": str(feature_path), "reason": str(exc)})
            logger.exception("Failed video %s", video_id)

    if index is None:
        raise RuntimeError("No valid visual features found")
    if index.ntotal != len(rows):
        raise RuntimeError(f"Index/vector mismatch: index.ntotal={index.ntotal} map_rows={len(rows)}")

    index_path = output_dir / "l21_visual_flat_ip.faiss"
    meta_path = output_dir / "l21_visual_metadata.json"
    excluded_path = output_dir / "excluded_visual_sources.json"
    faiss.write_index(index, str(index_path))

    global_map = pd.DataFrame(rows)
    global_map.to_csv(id_map_output.with_suffix(".csv"), index=False, encoding="utf-8-sig")
    try:
        global_map.to_parquet(id_map_output, index=False)
    except Exception:
        pass

    clip_info = clip_model.info()
    meta = {
        "index_type": "IndexFlatIP",
        "metric": "IP",
        "normalized": True,
        "embedding_dim": int(index.d),
        "num_vectors": int(index.ntotal),
        "num_videos": int(global_map["video_id"].nunique()),
        "clip_model": clip_info.model_name,
        "clip_pretrained": clip_info.pretrained,
        "model_assumption": clip_info.model_assumption,
        "assumption_reason": clip_info.assumption_reason,
        "dataset_root": str(dataset_root),
        "feature_sources": [str(p) for p in feature_files],
        "excluded_sources": excluded,
        "faiss_version": faiss.__version__,
        "numpy_version": np.__version__,
        "build_started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(started)),
        "build_finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": round(time.time() - started, 2),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    excluded_path.write_text(json.dumps(excluded, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(stage_summary("visual_index", "ok", input_path=str(feature_root), processed=index.ntotal, skipped=len(excluded), output=str(index_path), elapsed=time.time() - started))
    return index, global_map, meta

