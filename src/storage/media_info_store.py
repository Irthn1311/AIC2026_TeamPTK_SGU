"""
MediaInfoStore for AIC Video Retrieval System.

Responsibilities:
- Discover and load all {L}_{V}.json files in media_info_root
- Parse YouTube metadata (author, description, keywords, channel_id)
- Provide fast lookup by video_id
- Generate combined text representations for topic classification & text search
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from src.common.types import MediaInfo
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MediaInfoStore:
    """
    Registry for video-level metadata parsed from media-info JSON files.

    Usage:
        store = MediaInfoStore(media_info_root="datasets/media-info")
        store.load()
        info = store.get_by_video_id("L21_V001")
        print(info.author, info.keywords)
    """

    def __init__(self, media_info_root: str):
        self.media_info_root = Path(media_info_root)
        self._store: Dict[str, MediaInfo] = {}

    def load() -> "MediaInfoStore":
        """Load all JSON files from media_info_root."""
        if not self.media_info_root.exists():
            logger.warning(f"Media info directory does not exist: {self.media_info_root}")
            return self

        json_files = sorted(
            list(self.media_info_root.glob("*.json"))
            + list(self.media_info_root.glob("*/*.json"))
            + list(self.media_info_root.rglob("*.json"))
        )

        seen = set()
        loaded_count = 0

        for json_path in json_files:
            video_id = json_path.stem  # e.g. "L21_V001"
            if video_id in seen:
                continue
            seen.add(video_id)

            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Handle nested "root" key if present (as seen in Kaggle JSON exports)
                if "root" in data and isinstance(data["root"], dict):
                    content = data["root"]
                else:
                    content = data

                keywords = content.get("keywords", [])
                if isinstance(keywords, str):
                    keywords = [k.strip() for k in keywords.split(",") if k.strip()]

                info = MediaInfo(
                    video_id=video_id,
                    author=content.get("author", ""),
                    channel_id=content.get("channel_id", ""),
                    channel_url=content.get("channel_url", ""),
                    description=content.get("description", ""),
                    keywords=keywords,
                )
                self._store[video_id] = info
                loaded_count += 1
            except Exception as e:
                logger.warning(f"Failed to parse media-info file {json_path.name}: {e}")

        logger.info(f"Loaded media-info for {loaded_count:,} videos from {self.media_info_root}")
        return self

    def get_by_video_id(self, video_id: str) -> Optional[MediaInfo]:
        """Get MediaInfo record for a specific video_id."""
        return self._store.get(video_id)

    def get_combined_text(self, video_id: str) -> str:
        """Get concatenated text (author + keywords + description) for a video."""
        info = self.get_by_video_id(video_id)
        if info:
            return info.get_combined_text()
        return ""

    def get_all_media_info() -> Dict[str, MediaInfo]:
        """Return full mapping of video_id → MediaInfo."""
        return self._store

    @property
    def total_videos(self) -> int:
        return len(self._store)
