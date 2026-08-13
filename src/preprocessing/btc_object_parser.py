"""
BTC Object Detection Parser & Normalizer (Branch 3)
====================================================
Parses BTC provided object detection JSON files (objects-aic25-b1/objects/),
filters high-confidence objects, maps them to keyframes and timestamps,
and exports object metadata for indexing.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.retrieval.logging_utils import setup_logger, stage_summary
from src.retrieval.mapping_loader import load_keyframe_mapping

logger = logging.getLogger(__name__)

# Dictionary for common English-to-Vietnamese object translation
OBJECT_TRANSLATIONS: dict[str, str] = {
    "person": "con người người",
    "man": "người đàn ông đàn ông",
    "woman": "người phụ nữ phụ nữ",
    "girl": "cô gái bé gái",
    "boy": "cậu bé bé trai",
    "child": "trẻ em con nít",
    "car": "xe hơi ô tô xe con",
    "vehicle": "phương tiện xe",
    "land vehicle": "xe cộ phương tiện giao thông",
    "boat": "con thuyền cái thuyền tàu thủy",
    "watercraft": "thuyền tàu",
    "ship": "tàu thủy tàu lớn",
    "building": "tòa nhà nhà cửa công trình",
    "skyscraper": "tòa nhà cao tầng nhà cao tầng",
    "tree": "cây cối cây xanh",
    "house": "ngôi nhà nhà dân",
    "bicycle": "xe đạp",
    "motorcycle": "xe máy xe mô tô",
    "bus": "xe buýt xe bus",
    "truck": "xe tải",
    "airplane": "máy bay",
    "aircraft": "máy bay phi cơ",
    "dog": "con chó chó",
    "cat": "con mèo mèo",
    "bird": "con chim chim",
    "flower": "bông hoa hoa",
    "chair": "cái ghế ghế",
    "table": "cái bàn bàn",
    "television": "tivi truyền hình màn hình",
}

def parse_btc_objects_for_video(
    video_id: str,
    objects_dir: Path,
    mapping_dir: Path,
    keyframe_dir: Path,
    min_score: float = 0.25,
) -> list[dict[str, Any]]:
    vid_objects_dir = objects_dir / video_id
    mapping_path = mapping_dir / f"{video_id}.csv"

    if not vid_objects_dir.is_dir() or not mapping_path.exists():
        return []

    mapping_df = load_keyframe_mapping(mapping_path, keyframe_dir)
    records = []

    for idx, row in mapping_df.iterrows():
        n_idx = int(row["keyframe_name"].split(".")[0])
        json_path = vid_objects_dir / f"{n_idx:03d}.json"

        if not json_path.exists():
            # Try alternate padded name
            json_path = vid_objects_dir / f"{n_idx}.json"

        detected_entities: list[str] = []
        entity_scores: dict[str, float] = {}
        entity_counts: dict[str, int] = {}
        boxes: list[list[float]] = []

        if json_path.exists():
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
                raw_scores = [float(s) for s in data.get("detection_scores", [])]
                raw_entities = data.get("detection_class_entities", [])
                raw_boxes = data.get("detection_boxes", [])

                for score, entity, box in zip(raw_scores, raw_entities, raw_boxes):
                    if score >= min_score and entity:
                        entity_clean = entity.strip()
                        detected_entities.append(entity_clean)

                        # Max score per entity
                        if entity_clean not in entity_scores or score > entity_scores[entity_clean]:
                            entity_scores[entity_clean] = round(score, 4)

                        # Count per entity
                        entity_counts[entity_clean] = entity_counts.get(entity_clean, 0) + 1

                        # Store box
                        if isinstance(box, list) and len(box) >= 4:
                            boxes.append([round(float(b), 4) for b in box[:4]])
            except Exception as exc:
                logger.warning("Error parsing object JSON %s: %s", json_path, exc)

        unique_entities = sorted(list(set(detected_entities)))
        
        # Build search text combining English entity names and Vietnamese translations
        vi_terms = []
        for ent in unique_entities:
            ent_lower = ent.lower()
            if ent_lower in OBJECT_TRANSLATIONS:
                vi_terms.append(OBJECT_TRANSLATIONS[ent_lower])
        
        search_text = " ".join(unique_entities + vi_terms).strip()

        records.append({
            "video_id": video_id,
            "n_idx": n_idx,
            "keyframe_name": row["keyframe_name"],
            "keyframe_path": row["keyframe_path"],
            "frame_idx": int(row["frame_idx"]),
            "timestamp_seconds": float(row["timestamp_seconds"]),
            "num_detected_objects": len(detected_entities),
            "unique_object_classes": unique_entities,
            "object_scores": entity_scores,
            "object_counts": entity_counts,
            "search_text": search_text,
            "boxes": boxes[:10],  # Top 10 boxes for memory efficiency
        })

    return records


def build_btc_objects_corpus(
    dataset_root: str | Path,
    output_dir: str | Path,
    min_score: float = 0.25,
    logger_inst=None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    dataset_root = Path(dataset_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_inst = logger_inst or setup_logger("btc_objects")
    started = time.time()

    objects_root = dataset_root / "objects-aic25-b1" / "objects"
    mapping_root = dataset_root / "map-keyframes-aic25-b1" / "map-keyframes"
    keyframe_root = dataset_root / "Keyframes_L21" / "keyframes"

    video_dirs = sorted([p for p in objects_root.iterdir() if p.is_dir() and p.name.startswith("L21_V")])
    log_inst.info("Found %d video object directories in %s", len(video_dirs), objects_root)

    all_records = []
    for vid_dir in video_dirs:
        video_id = vid_dir.name
        recs = parse_btc_objects_for_video(video_id, objects_root, mapping_root, keyframe_root, min_score=min_score)
        all_records.extend(recs)
        log_inst.info("Parsed %d keyframe object records for %s", len(recs), video_id)

    df = pd.DataFrame(all_records)

    # Save outputs
    parquet_path = output_dir / "l21_objects.parquet"
    csv_path = output_dir / "l21_objects.csv"
    meta_path = output_dir / "l21_objects_metadata.json"

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    try:
        df.to_parquet(parquet_path, index=False)
    except Exception:
        pass

    meta = {
        "dataset_root": str(dataset_root),
        "total_records": len(df),
        "num_videos": int(df["video_id"].nunique()) if not df.empty else 0,
        "min_score_threshold": min_score,
        "elapsed_seconds": round(time.time() - started, 2),
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    log_inst.info("Saved BTC objects metadata: %s (%d records)", parquet_path, len(df))

    return df, meta
