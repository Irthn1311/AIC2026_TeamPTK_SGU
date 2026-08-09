"""
Metadata Store for AIC Video Retrieval System.

Responsibilities:
- Parse map-keyframes CSV files into KeyframeMeta objects
- Build and persist keyframe_master.parquet (single source of truth)
- Provide fast lookup: faiss_id → KeyframeMeta, video_id → all keyframes
- Map frame_idx → pts_time for timestamp reporting
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.common.constants import (
    CSV_COL_N, CSV_COL_PTS_TIME, CSV_COL_FPS, CSV_COL_FRAME_IDX,
    MASTER_PARQUET_COLS, KEYFRAME_NAME_FORMAT,
)
from src.common.types import KeyframeMeta
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MetadataStore:
    """
    Central metadata registry for all keyframes in the AIC dataset.

    Usage:
        store = MetadataStore(map_keyframes_root, keyframes_root)
        store.build()                           # Parse all CSVs → parquet
        store.load("indexes/keyframe_master.parquet")   # Load existing

        meta = store.get_by_faiss_id(42)        # KeyframeMeta
        results = store.get_by_video("L21_V001") # List[KeyframeMeta]
        frame_idx = store.n_to_frame_idx("L21_V001", n=5)
    """

    def __init__(
        self,
        map_keyframes_root: str,
        keyframes_image_root: str,
        dataset_slug: str = "aic-hcmc-data",
    ):
        self.map_keyframes_root = Path(map_keyframes_root)
        self.keyframes_image_root = Path(keyframes_image_root)
        self.dataset_slug = dataset_slug

        self._df: Optional[pd.DataFrame] = None

        # In-memory lookup indexes (built after load/build)
        self._faiss_id_to_meta: Dict[int, KeyframeMeta] = {}
        self._video_to_metas: Dict[str, List[KeyframeMeta]] = {}
        self._keyframe_id_to_meta: Dict[str, KeyframeMeta] = {}

    # ----------------------------------------------------------
    # Build
    # ----------------------------------------------------------

    def build(self, save_path: Optional[str] = None) -> pd.DataFrame:
        """
        Parse all map-keyframes CSV files and build the master DataFrame.

        CSV structure:
            n (1-based) | pts_time | fps | frame_idx
            1           | 0.0      | 30  | 0
            2           | 3.0      | 30  | 90
            ...

        Image path pattern:
            {keyframes_image_root}/Keyframes_{L}/keyframes/{L}_{V}/{n}.jpg
        """
        logger.info(f"Building keyframe master from: {self.map_keyframes_root}")
        records = []
        faiss_id = 0

        # Iterate all batch/video CSV files (recursively search subfolders)
        csv_files = sorted(
            list(self.map_keyframes_root.glob("*.csv")) + list(self.map_keyframes_root.glob("*/*.csv")) + list(self.map_keyframes_root.rglob("*.csv")),
            key=lambda p: p.stem
        )
        # Deduplicate paths preserving order
        seen_stems = set()
        unique_csv_files = []
        for p in csv_files:
            if p.stem not in seen_stems:
                seen_stems.add(p.stem)
                unique_csv_files.append(p)
        csv_files = unique_csv_files

        logger.info(f"Found {len(csv_files)} unique CSV files to process")

        for csv_path in csv_files:
            video_id = csv_path.stem  # e.g. "L21_V001"
            batch_id = video_id.split("_")[0]  # e.g. "L21"

            try:
                df_kf = pd.read_csv(csv_path)
                self._validate_csv(df_kf, csv_path)
            except Exception as e:
                logger.warning(f"Skipping {csv_path.name}: {e}")
                continue

            for _, row in df_kf.iterrows():
                n = int(row[CSV_COL_N])
                # Determine image filename (default 3-digit zero-padded: 001.jpg, 090.jpg)
                base_dir = (
                    self.keyframes_image_root
                    / f"Keyframes_{batch_id}"
                    / "keyframes"
                    / video_id
                )
                img_name = f"{n:03d}.jpg"
                for cand_name in [f"{n:03d}.jpg", f"{n}.jpg", f"{n:04d}.jpg"]:
                    if (base_dir / cand_name).exists():
                        img_name = cand_name
                        break

                image_path = str(base_dir / img_name)

                records.append({
                    "faiss_id":    faiss_id,
                    "keyframe_id": f"{video_id}_n{n}",
                    "video_id":    video_id,
                    "batch_id":    batch_id,
                    "n":           n,
                    "frame_idx":   int(row[CSV_COL_FRAME_IDX]),
                    "pts_time":    float(row[CSV_COL_PTS_TIME]),
                    "fps":         float(row[CSV_COL_FPS]),
                    "image_path":  image_path,
                })
                faiss_id += 1

        self._df = pd.DataFrame(records, columns=MASTER_PARQUET_COLS)
        logger.info(f"Built master index: {len(self._df):,} keyframes from {len(csv_files)} videos")

        if save_path:
            self.save(save_path)

        self._build_lookups()
        return self._df

    def _validate_csv(self, df: pd.DataFrame, path: Path) -> None:
        required = {CSV_COL_N, CSV_COL_PTS_TIME, CSV_COL_FPS, CSV_COL_FRAME_IDX}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns {missing} in {path.name}")

    # ----------------------------------------------------------
    # Save / Load
    # ----------------------------------------------------------

    def save(self, path: str) -> None:
        """Save master DataFrame to parquet."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        self._df.to_parquet(out, index=False)
        logger.info(f"Saved keyframe master → {out} ({out.stat().st_size / 1024:.1f} KB)")

    def load(self, path: str) -> "MetadataStore":
        """Load existing parquet and build in-memory indexes."""
        self._df = pd.read_parquet(path)
        logger.info(f"Loaded keyframe master: {len(self._df):,} keyframes from {path}")
        self._build_lookups()
        return self

    # ----------------------------------------------------------
    # Internal Index Build
    # ----------------------------------------------------------

    def _build_lookups(self) -> None:
        """Build fast in-memory lookup dicts from DataFrame."""
        self._faiss_id_to_meta.clear()
        self._video_to_metas.clear()
        self._keyframe_id_to_meta.clear()

        for row in self._df.itertuples(index=False):
            meta = KeyframeMeta(
                video_id=row.video_id,
                batch_id=row.batch_id,
                n=row.n,
                frame_idx=row.frame_idx,
                pts_time=row.pts_time,
                fps=row.fps,
                image_path=row.image_path,
            )
            self._faiss_id_to_meta[row.faiss_id] = meta
            self._keyframe_id_to_meta[row.keyframe_id] = meta
            self._video_to_metas.setdefault(row.video_id, []).append(meta)

        logger.debug(f"Lookup indexes ready: {len(self._faiss_id_to_meta):,} entries, "
                     f"{len(self._video_to_metas):,} videos")

    # ----------------------------------------------------------
    # Lookups
    # ----------------------------------------------------------

    def get_by_faiss_id(self, faiss_id: int) -> Optional[KeyframeMeta]:
        """Retrieve KeyframeMeta by its FAISS integer ID."""
        return self._faiss_id_to_meta.get(faiss_id)

    def get_by_keyframe_id(self, keyframe_id: str) -> Optional[KeyframeMeta]:
        """Retrieve by composite ID e.g. 'L21_V001_n5'."""
        return self._keyframe_id_to_meta.get(keyframe_id)

    def get_by_video(self, video_id: str) -> List[KeyframeMeta]:
        """Get all keyframes belonging to a video, sorted by n."""
        return sorted(self._video_to_metas.get(video_id, []), key=lambda m: m.n)

    def n_to_frame_idx(self, video_id: str, n: int) -> Optional[int]:
        """Convert keyframe number n → frame_idx (BTC submission value)."""
        kf_id = f"{video_id}_n{n}"
        meta = self._keyframe_id_to_meta.get(kf_id)
        return meta.frame_idx if meta else None

    def frame_idx_to_pts_time(self, video_id: str, frame_idx: int) -> Optional[float]:
        """Convert frame_idx → pts_time seconds."""
        metas = self._video_to_metas.get(video_id, [])
        for m in metas:
            if m.frame_idx == frame_idx:
                return m.pts_time
        return None

    def faiss_ids_for_video(self, video_id: str) -> List[int]:
        """Get all FAISS integer IDs belonging to a video (for TRAKE Phase 2)."""
        subset = self._df[self._df["video_id"] == video_id]
        return subset["faiss_id"].tolist()

    # ----------------------------------------------------------
    # Properties
    # ----------------------------------------------------------

    @property
    def total_keyframes(self) -> int:
        return len(self._df) if self._df is not None else 0

    @property
    def video_ids(self) -> List[str]:
        return list(self._video_to_metas.keys())

    @property
    def dataframe(self) -> pd.DataFrame:
        if self._df is None:
            raise RuntimeError("MetadataStore not loaded. Call build() or load() first.")
        return self._df

    def export_faiss_ids_map(self, path: str) -> None:
        """
        Export faiss_id → keyframe_id mapping as JSON.
        Used alongside FAISS index to reconstruct metadata from search results.
        """
        mapping = {str(row.faiss_id): row.keyframe_id
                   for row in self._df.itertuples(index=False)}
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(mapping, f, ensure_ascii=False, indent=2)
        logger.info(f"Exported FAISS ID map → {out} ({len(mapping):,} entries)")
