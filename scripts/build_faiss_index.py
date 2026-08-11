"""
Build FAISS Visual Index from pre-extracted CLIP-32 .npy files.

Pipeline:
1. Scan all .npy files from Kaggle dataset
2. Load + normalize vectors
3. Build FAISS HNSW index with aligned faiss_ids
4. Parse all map-keyframes CSV → keyframe_master.parquet
5. Export faiss_ids_map.json (int → keyframe_id)

Usage (local):
    python scripts/build_faiss_index.py \\
        --npy-dir datasets/clip-features-32 \\
        --map-keyframes-dir datasets/map-keyframes \\
        --keyframes-img-dir datasets/keyframes \\
        --output-dir indexes/

Usage (Kaggle Notebook):
    !python AIC_System/scripts/build_faiss_index.py \\
        --npy-dir /kaggle/input/aic-hcmc-data/clip-features-32-aic25-b/clip-features-32 \\
        --map-keyframes-dir /kaggle/input/aic-hcmc-data/map-keyframes-aic25-b1/map-keyframes \\
        --keyframes-img-dir /kaggle/input/aic-hcmc-data/keyframes/keyframes \\
        --output-dir /kaggle/working/indexes
"""

import argparse
import sys
import time
from pathlib import Path

# Allow running from repo root or scripts/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.database.faiss_db import FaissDB
from src.storage.metadata_store import MetadataStore
from src.utils.logger import get_logger

logger = get_logger("build_faiss_index")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build FAISS index from CLIP-32 .npy files"
    )
    parser.add_argument(
        "--npy-dir",
        required=True,
        help="Directory containing L{XX}_{V}.npy CLIP feature files",
    )
    parser.add_argument(
        "--map-keyframes-dir",
        required=True,
        help="Directory containing L{XX}_{V}.csv map-keyframes files",
    )
    parser.add_argument(
        "--keyframes-img-dir",
        required=True,
        help="Root directory of keyframe images (Keyframes_{L}/keyframes/...)",
    )
    parser.add_argument(
        "--media-info-dir",
        default=None,
        help="Optional directory containing media-info JSON files ({L}_{V}.json)",
    )
    parser.add_argument(
        "--output-dir",
        default="indexes",
        help="Output directory for index files (default: indexes/)",
    )
    parser.add_argument(
        "--clip-dim",
        type=int,
        default=512,
        help="CLIP feature dimension (default: 512 for CLIP-32)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()

    # --------------------------------------------------------
    # Step 1: Build FAISS index from .npy files
    # --------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Step 1/3: Building FAISS HNSW index from .npy files")
    logger.info(f"  Source: {args.npy_dir}")

    faiss_db = FaissDB(dim=args.clip_dim)
    faiss_db.build_from_npy_files(
        npy_dir=args.npy_dir,
        id_offset=0,
        normalize=True,
    )

    faiss_index_path = str(output_dir / "faiss_visual.index")
    faiss_db.save(faiss_index_path)

    # --------------------------------------------------------
    # Step 2: Build keyframe master parquet from CSV files
    # --------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Step 2/3: Building keyframe_master.parquet from CSV files")
    logger.info(f"  Source: {args.map_keyframes_dir}")

    store = MetadataStore(
        map_keyframes_root=args.map_keyframes_dir,
        keyframes_image_root=args.keyframes_img_dir,
        media_info_root=args.media_info_dir,
    )
    parquet_path = str(output_dir / "keyframe_master.parquet")
    store.build(save_path=parquet_path)

    logger.info(f"  Total keyframes indexed: {store.total_keyframes:,}")

    # --------------------------------------------------------
    # Step 3: Export faiss_ids_map.json
    # --------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Step 3/3: Exporting FAISS ID → keyframe_id map")

    map_path = str(output_dir / "faiss_ids_map.json")
    store.export_faiss_ids_map(map_path)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------
    elapsed = time.time() - t_start
    logger.info("=" * 60)
    logger.info(f"Done in {elapsed:.1f}s")
    logger.info(f"Outputs written to: {output_dir.resolve()}")
    logger.info(f"  faiss_visual.index     ({faiss_db.total_vectors:,} vectors)")
    logger.info(f"  keyframe_master.parquet ({store.total_keyframes:,} rows)")
    logger.info(f"  faiss_ids_map.json")
    logger.info("=" * 60)

    # Sanity check: verify counts match
    if faiss_db.total_vectors != store.total_keyframes:
        logger.warning(
            f"MISMATCH: FAISS has {faiss_db.total_vectors} vectors "
            f"but parquet has {store.total_keyframes} rows. "
            f"Ensure .npy files and CSV files cover the same videos."
        )
    else:
        logger.info("Sanity check PASSED: vector count matches keyframe count.")


if __name__ == "__main__":
    main()
