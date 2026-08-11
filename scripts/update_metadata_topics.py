"""
Update existing keyframe_master.parquet with topic categories from media-info.

Fast in-place metadata updater (takes ~2 seconds).
Does NOT require rebuilding FAISS index or re-processing keyframe CSVs.

Usage:
    python scripts/update_metadata_topics.py \
        --parquet-path indexes/keyframe_master.parquet \
        --media-info-dir datasets/media-info
"""

import argparse
import sys
import time
from pathlib import Path
import pandas as pd

# Allow running from repo root or scripts/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.storage.media_info_store import MediaInfoStore
from src.reasoning.topic_classifier import TopicClassifier
from src.utils.logger import get_logger

logger = get_logger("update_metadata_topics")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Update existing keyframe_master.parquet with topic categories"
    )
    parser.add_argument(
        "--parquet-path",
        default="indexes/keyframe_master.parquet",
        help="Path to keyframe_master.parquet (default: indexes/keyframe_master.parquet)",
    )
    parser.add_argument(
        "--media-info-dir",
        required=True,
        help="Directory containing media-info JSON files ({L}_{V}.json)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    parquet_path = Path(args.parquet_path)
    media_info_dir = Path(args.media_info_dir)

    if not parquet_path.exists():
        logger.error(f"Parquet file not found: {parquet_path}")
        sys.exit(1)

    if not media_info_dir.exists():
        logger.error(f"Media-info directory not found: {media_info_dir}")
        sys.exit(1)

    t_start = time.time()
    logger.info(f"Loading media-info from: {media_info_dir}")
    mi_store = MediaInfoStore(str(media_info_dir)).load()

    classifier = TopicClassifier()
    video_topic_map = {}

    for vid, info in mi_store.get_all_media_info().items():
        res = classifier.classify_media_info(info)
        video_topic_map[vid] = res.topic

    logger.info(f"Classified topics for {len(video_topic_map):,} videos")

    logger.info(f"Reading existing parquet: {parquet_path}")
    df = pd.read_parquet(parquet_path)

    # Map topic category to video_id column
    df["topic_category"] = df["video_id"].map(lambda vid: video_topic_map.get(vid, ""))

    df.to_parquet(parquet_path, index=False)
    elapsed = time.time() - t_start

    classified_count = (df["topic_category"] != "").sum()
    logger.info(f"SUCCESS: Updated {parquet_path} ({len(df):,} rows, {classified_count:,} tagged with topics) in {elapsed:.2f}s")


if __name__ == "__main__":
    main()
