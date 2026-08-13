from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import faiss
import numpy as np
import pandas as pd

from _bootstrap import PROJECT_ROOT
from src.preprocessing.keyframe_v2.clip_scorer import ImageEmbeddingScorer
from src.preprocessing.keyframe_v2.pipeline import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Visual FAISS index for Keyframe V2 final keyframes.")
    parser.add_argument("--global-map", default=str(PROJECT_ROOT / "outputs" / "keyframe_v2_full" / "indexes" / "keyframe_v2_global_map.parquet"))
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "keyframe_v2.yaml"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "keyframe_v2_full" / "indexes" / "visual"))
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    started = time.time()
    cfg = load_config(args.config)
    global_map = pd.read_parquet(args.global_map)
    if global_map.empty:
        raise RuntimeError(f"Global V2 map is empty: {args.global_map}")

    scorer = ImageEmbeddingScorer(PROJECT_ROOT, cfg["clip"])
    if scorer.backend != "clip":
        raise RuntimeError(f"Real CLIP required for Visual V2, got {scorer.backend}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    index = None
    encoded = 0

    for start in range(0, len(global_map), args.batch_size):
        batch = global_map.iloc[start : start + args.batch_size]
        images = []
        valid_indices = []
        for row_idx, row in batch.iterrows():
            img = cv2.imread(str(row["image_path"]))
            if img is None:
                raise RuntimeError(f"Cannot read V2 keyframe image: {row['image_path']}")
            images.append(img)
            valid_indices.append(row_idx)
        feats = scorer.embed_images(images).astype(np.float32)
        if index is None:
            index = faiss.IndexFlatIP(feats.shape[1])
        index.add(feats)
        encoded += len(valid_indices)
        print(f"Encoded Visual V2: {encoded}/{len(global_map)}", end="\r", flush=True)
    print()

    if index is None:
        raise RuntimeError("No V2 keyframes encoded.")
    faiss.write_index(index, str(output_dir / "l21_visual_v2_flat_ip.faiss"))
    meta = {
        "index_type": "IndexFlatIP",
        "metric": "IP",
        "normalized": True,
        "embedding_dim": int(index.d),
        "num_vectors": int(index.ntotal),
        "global_map": str(Path(args.global_map).resolve()),
        "clip_info": scorer.info,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    (output_dir / "l21_visual_v2_metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
