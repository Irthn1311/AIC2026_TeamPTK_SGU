from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import queue
import sys
import threading
import time
import types
import unicodedata
from pathlib import Path
from typing import Any, Generator

import cv2
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def setup_e_drive_cache() -> None:
    cache_root = PROJECT_ROOT / ".ocr_cache_video_v4" / "paddle_stage"
    temp_dir = cache_root / "temp"
    home_dir = cache_root / "home"
    for p in (cache_root, temp_dir, home_dir):
        p.mkdir(parents=True, exist_ok=True)

    os.environ["PADDLE_PDX_CACHE_HOME"] = str(PROJECT_ROOT / ".ocr_cache" / "paddlex")
    os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
    os.environ["MODELSCOPE_CACHE"] = str(cache_root / "modelscope")
    os.environ["MODELSCOPE_CREDENTIALS_PATH"] = str(home_dir / ".modelscope" / "credentials")
    os.environ["HF_HOME"] = str(cache_root / "huggingface")
    os.environ["TRANSFORMERS_CACHE"] = str(cache_root / "huggingface" / "transformers")
    os.environ["HOME"] = str(home_dir)
    os.environ["USERPROFILE"] = str(home_dir)
    os.environ["HOMEDRIVE"] = str(home_dir.drive)
    os.environ["HOMEPATH"] = str(home_dir)
    os.environ["TMP"] = str(temp_dir)
    os.environ["TEMP"] = str(temp_dir)
    os.environ["TMPDIR"] = str(temp_dir)
    os.environ["MPLCONFIGDIR"] = str(temp_dir / "matplotlib")
    os.environ["FLAGS_use_mkldnn"] = "0"
    os.environ["FLAGS_enable_pir_api"] = "0"

    nvidia_root = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    nvidia_bins = list(
        nvidia_root / name / subdir
        for name, subdir in (
            ("cuda_runtime", "bin"),
            ("cublas", "bin"),
            ("cudnn", "bin"),
            ("cufft", "bin"),
            ("curand", "bin"),
            ("cusolver", "bin"),
            ("cusparse", "bin"),
            ("nvjitlink", "bin"),
            ("nvjitlink", "lib"),
        )
    )
    nvidia_bins.append(Path(sys.prefix) / "Lib" / "site-packages" / "torch" / "lib")
    existing_path = os.environ.get("PATH", "")
    prepend = [str(p) for p in nvidia_bins if p.exists()]
    if prepend:
        os.environ["PATH"] = os.pathsep.join(prepend + [existing_path])
        for p in prepend:
            if hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(p)
                except OSError:
                    pass


def normalize_light(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    return " ".join(text.strip().split())


def bbox_to_xyxy(bbox: Any, width: int, height: int) -> list[int]:
    if isinstance(bbox, np.ndarray):
        bbox = bbox.tolist()
    if isinstance(bbox, list) and len(bbox) == 4 and all(isinstance(v, (int, float)) for v in bbox):
        x1, y1, x2, y2 = bbox
    elif isinstance(bbox, list) and bbox and isinstance(bbox[0], (list, tuple)):
        xs = [float(p[0]) for p in bbox if len(p) >= 2]
        ys = [float(p[1]) for p in bbox if len(p) >= 2]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    else:
        x1, y1, x2, y2 = 0, 0, 0, 0
    return [
        int(max(0, min(width - 1, round(x1)))),
        int(max(0, min(height - 1, round(y1)))),
        int(max(0, min(width, round(x2)))),
        int(max(0, min(height, round(y2)))),
    ]


def classify_roi(bbox_xyxy: list[int], img_w: int, img_h: int) -> str:
    x1, y1, x2, y2 = bbox_xyxy
    x1_n = x1 / max(1, img_w)
    y1_n = y1 / max(1, img_h)
    x2_n = x2 / max(1, img_w)
    w_n = (x2 - x1) / max(1, img_w)
    h_n = (y2 - y1) / max(1, img_h)

    if y1_n < 0.25 and (x1_n < 0.30 or x2_n > 0.70) and h_n < 0.15 and w_n < 0.35:
        return "logo_channel"
    if y1_n < 0.30 and h_n < 0.10 and w_n < 0.25:
        return "clock_time"
    if y1_n >= 0.82 and w_n >= 0.30:
        return "ticker"
    if 0.45 <= y1_n <= 0.85 and w_n >= 0.20:
        return "headline"
    return "scene_text"


def video_metadata(video_path: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    if fps <= 0:
        raise RuntimeError(f"Invalid FPS for video: {video_path}")
    return {
        "fps": fps,
        "total_frames": total_frames,
        "duration_sec": total_frames / fps,
        "width": width,
        "height": height,
    }


def load_shots(video_id: str, cfg: dict[str, Any]) -> pd.DataFrame:
    v2_root = resolve_path(cfg["paths"].get("keyframe_v2_root", "outputs/keyframe_v2_full"))
    shot_path = v2_root / video_id / "shots.csv"
    if shot_path.exists():
        return pd.read_csv(shot_path)
    return pd.DataFrame(columns=["video_id", "shot_id", "start_frame", "end_frame"])


def shot_id_for_frame(shots: pd.DataFrame, frame_idx: int) -> int:
    if shots.empty:
        return -1
    mask = (shots["start_frame"].astype(int) <= frame_idx) & (shots["end_frame"].astype(int) >= frame_idx)
    if mask.any():
        return int(shots[mask].iloc[0]["shot_id"])
    return -1


def build_sparse_samples(video_id: str, meta: dict[str, Any], shots: pd.DataFrame, sparse_fps: float) -> pd.DataFrame:
    fps = float(meta["fps"])
    total_frames = int(meta["total_frames"])
    step = max(1, int(round(fps / max(0.001, sparse_fps))))
    frames = set(range(0, total_frames, step))
    for _, row in shots.iterrows():
        start = int(row["start_frame"])
        end = int(row["end_frame"])
        mid = int((start + end) // 2)
        for fid in (start, mid, end):
            if 0 <= fid < total_frames:
                frames.add(fid)
    return pd.DataFrame(
        {
            "video_id": video_id,
            "frame_idx": [int(fid) for fid in sorted(frames)],
            "timestamp_sec": [float(fid / fps) for fid in sorted(frames)],
            "shot_id": [shot_id_for_frame(shots, int(fid)) for fid in sorted(frames)],
            "sample_stage": "sparse",
        }
    )


def dense_samples_from_hits(
    hit_frames: pd.DataFrame,
    meta: dict[str, Any],
    shots: pd.DataFrame,
    dense_fps: float,
    dense_window_sec: float,
) -> pd.DataFrame:
    if hit_frames.empty or dense_fps <= 0.0 or dense_window_sec <= 0.0:
        return pd.DataFrame(columns=["video_id", "frame_idx", "timestamp_sec", "shot_id", "sample_stage"])
    fps = float(meta["fps"])
    total_frames = int(meta["total_frames"])
    step_sec = 1.0 / max(0.001, dense_fps)
    frames: set[int] = set()
    for _, row in hit_frames.iterrows():
        center = float(row["timestamp_sec"])
        t = center - dense_window_sec
        while t <= center + dense_window_sec + 1e-9:
            fid = int(round(t * fps))
            if 0 <= fid < total_frames:
                frames.add(fid)
            t += step_sec
    video_id = str(hit_frames.iloc[0]["video_id"])
    return pd.DataFrame(
        {
            "video_id": video_id,
            "frame_idx": [int(fid) for fid in sorted(frames)],
            "timestamp_sec": [float(fid / fps) for fid in sorted(frames)],
            "shot_id": [shot_id_for_frame(shots, int(fid)) for fid in sorted(frames)],
            "sample_stage": "dense",
        }
    )


class FastVideoPrefetcher:
    """
    Background worker thread to read video frames using smart grab-skipping.
    Eliminates OpenCV seeking lag and decouples CPU frame decode from GPU inference.
    """

    def __init__(self, video_path: Path, frame_indices: list[int], max_queue_size: int = 64):
        self.video_path = video_path
        self.frame_indices = sorted(set(int(f) for f in frame_indices))
        self.queue: queue.Queue[tuple[int, np.ndarray | None] | None] = queue.Queue(maxsize=max_queue_size)
        self.stopped = False
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self) -> None:
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            self.queue.put(None)
            return

        cur_pos = 0
        for fid in self.frame_indices:
            if self.stopped:
                break

            gap = fid - cur_pos
            if gap == 0:
                ok, frame = cap.read()
                cur_pos += 1
            elif 0 < gap <= 30:
                for _ in range(gap - 1):
                    cap.grab()
                ok, frame = cap.read()
                cur_pos = fid + 1
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
                ok, frame = cap.read()
                cur_pos = fid + 1

            if ok and frame is not None:
                self.queue.put((fid, frame))
            else:
                self.queue.put((fid, None))

        cap.release()
        self.queue.put(None)

    def iter_batches(self, batch_size: int = 16) -> Generator[tuple[list[int], list[np.ndarray]], None, None]:
        batch_fids = []
        batch_frames = []
        while True:
            item = self.queue.get()
            if item is None:
                if batch_frames:
                    yield batch_fids, batch_frames
                break
            fid, frame = item
            if frame is not None:
                batch_fids.append(fid)
                batch_frames.append(frame)
                if len(batch_frames) >= batch_size:
                    yield batch_fids, batch_frames
                    batch_fids = []
                    batch_frames = []
        self.stop()

    def stop(self) -> None:
        self.stopped = True


def init_paddle(cfg: dict[str, Any], device: str):
    import paddle

    requested_device = str(device).lower()
    if requested_device.startswith("gpu") or requested_device.startswith("cuda"):
        if not paddle.device.is_compiled_with_cuda():
            device = "cpu"
            requested_device = "cpu"

    if requested_device != "cpu":
        paddle.device.set_device(device)

    if "modelscope" not in sys.modules:
        modelscope_stub = types.ModuleType("modelscope")

        def _snapshot_download(*_args, **_kwargs):
            raise RuntimeError("ModelScope download is disabled in OCR V4 Paddle stage; local model dirs are required.")

        modelscope_stub.snapshot_download = _snapshot_download
        modelscope_stub.__path__ = []
        hub_stub = types.ModuleType("modelscope.hub")
        errors_stub = types.ModuleType("modelscope.hub.errors")

        class ModelScopeNotExistError(Exception):
            pass

        class ModelScopeHTTPError(Exception):
            pass

        errors_stub.NotExistError = ModelScopeNotExistError
        errors_stub.HTTPError = ModelScopeHTTPError
        hub_stub.errors = errors_stub
        modelscope_stub.hub = hub_stub
        sys.modules["modelscope"] = modelscope_stub
        sys.modules["modelscope.hub"] = hub_stub
        sys.modules["modelscope.hub.errors"] = errors_stub

    from paddleocr import PaddleOCR

    det_dir = resolve_path(cfg.get("ocr", {}).get("paddle_detection_model_dir", ".ocr_cache/paddlex/official_models/PP-OCRv6_medium_det"))
    rec_dir = resolve_path(cfg.get("ocr", {}).get("paddle_recognition_model_dir", ".ocr_cache/paddlex/official_models/PP-OCRv6_medium_rec"))
    kwargs = {
        "text_detection_model_dir": str(det_dir),
        "text_recognition_model_dir": str(rec_dir),
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": False,
        "device": device,
    }
    if not det_dir.is_dir():
        kwargs.pop("text_detection_model_dir", None)
    if not rec_dir.is_dir():
        kwargs.pop("text_recognition_model_dir", None)
    if str(device).lower() == "cpu":
        kwargs["enable_mkldnn"] = False
    try:
        return PaddleOCR(**kwargs)
    except TypeError:
        kwargs.pop("enable_mkldnn", None)
        return PaddleOCR(**kwargs)


def _async_save_worker(crop_img: np.ndarray, target_path: str) -> None:
    try:
        p = Path(target_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(target_path, crop_img)
    except Exception:
        pass


def paddle_ocr_batch(
    reader: Any,
    frames_bgr: list[np.ndarray],
    frame_indices: list[int],
    crop_dir: Path,
    min_confidence: float,
    crop_pool: concurrent.futures.ThreadPoolExecutor | None = None,
    save_crops: bool = False,
) -> list[list[dict[str, Any]]]:
    if not frames_bgr:
        return []

    try:
        raw_list = reader.predict(frames_bgr)
    except Exception:
        raw_list = []
        for f in frames_bgr:
            r = reader.predict(f)
            raw_list.append(r[0] if r else {})

    all_detections: list[list[dict[str, Any]]] = []
    for i, item in enumerate(raw_list):
        frame_bgr = frames_bgr[i]
        frame_idx = frame_indices[i]
        height, width = frame_bgr.shape[:2]
        item_dict = item if isinstance(item, dict) else (item[0] if isinstance(item, list) and item else {})
        boxes = item_dict.get("rec_polys", item_dict.get("dt_polys", []))
        paddle_texts = item_dict.get("rec_texts", [])
        paddle_scores = item_dict.get("rec_scores", [])
        detections = []
        for det_idx, bbox in enumerate(boxes):
            bbox_xyxy = bbox_to_xyxy(bbox, width, height)
            text_raw = normalize_light(str(paddle_texts[det_idx] if det_idx < len(paddle_texts) else ""))
            paddle_conf = float(paddle_scores[det_idx]) if det_idx < len(paddle_scores) else 0.0
            if not text_raw or paddle_conf < min_confidence:
                continue
            
            x1, y1, x2, y2 = bbox_xyxy
            crop_path_text = ""
            if save_crops and x2 - x1 > 4 and y2 - y1 > 4:
                crop = frame_bgr[y1:y2, x1:x2]
                if crop.size > 0:
                    crop_path = (crop_dir / f"{frame_idx:08d}_{det_idx:03d}.jpg").resolve()
                    crop_pool.submit(_async_save_worker, crop.copy(), str(crop_path))
                    try:
                        crop_path_text = str(crop_path.relative_to(PROJECT_ROOT))
                    except ValueError:
                        crop_path_text = str(crop_path)

            detections.append(
                {
                    "det_idx": det_idx,
                    "bbox": bbox_xyxy,
                    "box": [[int(x), int(y)] for x, y in bbox] if isinstance(bbox, (list, np.ndarray)) else [],
                    "region_type": classify_roi(bbox_xyxy, width, height),
                    "text": text_raw,
                    "text_raw": text_raw,
                    "text_normalized": text_raw,
                    "det_conf": paddle_conf,
                    "rec_conf": paddle_conf,
                    "confidence": paddle_conf,
                    "paddle_text": text_raw,
                    "crop_path": crop_path_text,
                    "ocr_backend": "paddleocr_gpu",
                }
            )
        all_detections.append(detections)
    return all_detections


def combine_text(detections: list[dict[str, Any]], min_confidence: float) -> tuple[str, float, int]:
    kept = []
    for d in detections:
        txt = str(d.get("text_normalized", "")).strip()
        region = str(d.get("region_type", "scene_text"))
        conf = float(d.get("confidence", 0.0))
        if not txt or conf < min_confidence:
            continue
        if region in ("logo_channel", "clock_time") and len(txt) <= 8:
            continue
        kept.append(txt)
    if not kept:
        kept = [
            str(d.get("text_normalized", "")).strip()
            for d in detections
            if str(d.get("text_normalized", "")).strip() and float(d.get("confidence", 0.0)) >= min_confidence
        ]
    mean_conf = float(np.mean([float(d.get("confidence", 0.0)) for d in detections])) if detections else 0.0
    return " ".join(kept), mean_conf, len(kept)


def process_samples(
    video_path: Path,
    samples: pd.DataFrame,
    reader: Any,
    crop_dir: Path,
    min_confidence: float,
    stage_label: str,
    batch_size: int = 16,
    crop_pool: concurrent.futures.ThreadPoolExecutor | None = None,
    save_crops: bool = False,
    checkpoint_path: Path | None = None,
    existing_frame_ids: set[int] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if samples.empty:
        return [], []

    existing_frame_ids = existing_frame_ids or set()
    samples = samples[~samples["frame_idx"].astype(int).isin(existing_frame_ids)].copy()
    if samples.empty:
        print(f"[PADDLE_STAGE] {stage_label}: all frames already checkpointed ({len(existing_frame_ids)} frames).")
        return [], []

    if save_crops and crop_pool is None:
        crop_pool = concurrent.futures.ThreadPoolExecutor(max_workers=8)

    samples_sorted = samples.sort_values("frame_idx").reset_index(drop=True)
    frame_indices = samples_sorted["frame_idx"].astype(int).tolist()
    sample_map = {int(row["frame_idx"]): row for _, row in samples_sorted.iterrows()}

    prefetcher = FastVideoPrefetcher(video_path, frame_indices, max_queue_size=64)
    frame_records = []
    observation_rows = []

    pbar = tqdm(total=len(frame_indices), desc=f"PaddleOCR {stage_label} (Batch {batch_size})", file=sys.stdout)
    checkpoint_fh = checkpoint_path.open("a", encoding="utf-8") if checkpoint_path is not None else None

    try:
        for b_fids, b_frames in prefetcher.iter_batches(batch_size=batch_size):
            b_detections = paddle_ocr_batch(reader, b_frames, b_fids, crop_dir, min_confidence, crop_pool, save_crops=save_crops)
            for fid, detections in zip(b_fids, b_detections):
                sample = sample_map[fid]
                text, mean_conf, num_boxes = combine_text(detections, min_confidence)
                record = {
                    "video_id": str(sample["video_id"]),
                    "frame_idx": fid,
                    "timestamp_seconds": float(sample["timestamp_sec"]),
                    "timestamp_sec": float(sample["timestamp_sec"]),
                    "shot_id": int(sample["shot_id"]),
                    "sample_stage": str(sample["sample_stage"]),
                    "detections": detections,
                    "combined_text": text,
                    "mean_confidence": mean_conf,
                    "num_text_boxes": num_boxes,
                    "ocr_status": "ok",
                }
                frame_records.append(record)
                if checkpoint_fh is not None:
                    checkpoint_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                    checkpoint_fh.flush()
                for det in detections:
                    observation_rows.append(
                        {
                            "video_id": record["video_id"],
                            "frame_idx": fid,
                            "timestamp": record["timestamp_seconds"],
                            "timestamp_sec": record["timestamp_seconds"],
                            "shot_id": record["shot_id"],
                            "sample_stage": record["sample_stage"],
                            "det_idx": int(det["det_idx"]),
                            "text_raw": det["text_raw"],
                            "text_normalized": det["text_normalized"],
                            "text": det["text"],
                            "det_conf": float(det["det_conf"]),
                            "rec_conf": float(det["rec_conf"]),
                            "confidence": float(det["confidence"]),
                            "region_type": det["region_type"],
                            "bbox": json.dumps(det["bbox"], ensure_ascii=False),
                            "crop_path": det.get("crop_path", ""),
                        }
                    )
            pbar.update(len(b_fids))
            sys.stdout.flush()
    finally:
        if checkpoint_fh is not None:
            checkpoint_fh.close()

    pbar.close()
    return frame_records, observation_rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def dedupe_records_by_frame(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_frame: dict[int, dict[str, Any]] = {}
    for row in rows:
        try:
            by_frame[int(row["frame_idx"])] = row
        except Exception:
            continue
    return [by_frame[k] for k in sorted(by_frame)]


def observation_rows_from_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        for det in record.get("detections", []):
            rows.append(
                {
                    "video_id": record["video_id"],
                    "frame_idx": int(record["frame_idx"]),
                    "timestamp": float(record["timestamp_seconds"]),
                    "timestamp_sec": float(record["timestamp_seconds"]),
                    "shot_id": int(record.get("shot_id", -1)),
                    "sample_stage": record.get("sample_stage", ""),
                    "det_idx": int(det.get("det_idx", 0)),
                    "text_raw": det.get("text_raw", ""),
                    "text_normalized": det.get("text_normalized", ""),
                    "text": det.get("text", ""),
                    "det_conf": float(det.get("det_conf", 0.0)),
                    "rec_conf": float(det.get("rec_conf", 0.0)),
                    "confidence": float(det.get("confidence", 0.0)),
                    "region_type": det.get("region_type", "scene_text"),
                    "bbox": json.dumps(det.get("bbox", [0, 0, 0, 0]), ensure_ascii=False),
                    "crop_path": det.get("crop_path", ""),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="PaddleOCR high-speed subprocess stage for OCR Video V4.")
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sparse-fps", type=float, required=True)
    parser.add_argument("--dense-fps", type=float, required=True)
    parser.add_argument("--dense-window", type=float, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="gpu:0")
    parser.add_argument("--save-crops", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    setup_e_drive_cache()
    cfg = load_config(args.config)
    video_id = args.video_id.strip()
    output_base = resolve_path(args.output_dir)
    output_dir = output_base / video_id
    output_dir.mkdir(parents=True, exist_ok=True)
    crop_dir = output_dir / "debug" / "paddle_crops"
    if args.save_crops:
        crop_dir.mkdir(parents=True, exist_ok=True)
    raw_stage_path = output_dir / "paddle_frame_records.jsonl"
    partial_stage_path = output_dir / "paddle_frame_records.partial.jsonl"
    if raw_stage_path.exists() and not args.force:
        print(f"[PADDLE_STAGE] Reusing {raw_stage_path}")
        return
    if args.force:
        for stale_path in (raw_stage_path, partial_stage_path, output_dir / "paddle_stage_summary.json", output_dir / "raw_paddle_observations.csv"):
            if stale_path.exists():
                stale_path.unlink()

    video_path = resolve_path(cfg["paths"].get("video_root", "datasets_L21/Videos_L21_a/video")) / f"{video_id}.mp4"
    started = time.time()
    meta = video_metadata(video_path)
    shots = load_shots(video_id, cfg)
    min_confidence = float(cfg["ocr"].get("min_confidence", 0.35))
    detect_threshold = float(cfg["sampling"].get("detection_confidence_threshold", 0.45))
    dense_trigger_regions = set(cfg["sampling"].get("dense_trigger_regions", ["headline", "ticker", "scene_text"]))
    batch_size = max(1, int(args.batch_size or cfg.get("ocr", {}).get("batch_size", 16)))

    crop_pool = concurrent.futures.ThreadPoolExecutor(max_workers=8) if args.save_crops else None

    sparse_samples = build_sparse_samples(video_id, meta, shots, sparse_fps=float(args.sparse_fps))
    existing_records = dedupe_records_by_frame(read_jsonl(partial_stage_path))
    existing_frame_ids = {int(r["frame_idx"]) for r in existing_records}
    if existing_records:
        print(f"[PADDLE_STAGE] Resuming from {partial_stage_path}: {len(existing_records)} frames already done.")

    reader = init_paddle(cfg, args.device)
    sparse_records, sparse_obs = process_samples(
        video_path,
        sparse_samples,
        reader,
        crop_dir,
        min_confidence,
        "sparse",
        batch_size=batch_size,
        crop_pool=crop_pool,
        save_crops=args.save_crops,
        checkpoint_path=partial_stage_path,
        existing_frame_ids=existing_frame_ids,
    )
    existing_records = dedupe_records_by_frame(read_jsonl(partial_stage_path))
    existing_frame_ids = {int(r["frame_idx"]) for r in existing_records}
    all_sparse_records = [r for r in existing_records if str(r.get("sample_stage", "")) == "sparse"] + sparse_records

    sparse_obs_all = observation_rows_from_records(all_sparse_records)
    if sparse_obs_all:
        sparse_obs_df = pd.DataFrame(sparse_obs_all)
        hit_frames = sparse_obs_df[
            (sparse_obs_df["det_conf"].astype(float) >= detect_threshold)
            & (sparse_obs_df["region_type"].astype(str).isin(dense_trigger_regions))
        ][["video_id", "frame_idx", "timestamp_sec"]].drop_duplicates()
    else:
        hit_frames = pd.DataFrame(columns=["video_id", "frame_idx", "timestamp_sec"])

    dense_samples = dense_samples_from_hits(
        hit_frames,
        meta,
        shots,
        dense_fps=float(args.dense_fps),
        dense_window_sec=float(args.dense_window),
    )
    if not dense_samples.empty:
        dense_samples = dense_samples[~dense_samples["frame_idx"].isin(set(sparse_samples["frame_idx"].astype(int)))].copy()
        dense_samples["video_id"] = video_id
    dense_records, dense_obs = process_samples(
        video_path,
        dense_samples,
        reader,
        crop_dir,
        min_confidence,
        "dense",
        batch_size=batch_size,
        crop_pool=crop_pool,
        save_crops=args.save_crops,
        checkpoint_path=partial_stage_path,
        existing_frame_ids=existing_frame_ids,
    ) if not dense_samples.empty else ([], [])

    if crop_pool is not None:
        crop_pool.shutdown(wait=True)

    frame_records = dedupe_records_by_frame(read_jsonl(partial_stage_path))
    observation_rows = observation_rows_from_records(frame_records)
    sampled_df = pd.concat([sparse_samples, dense_samples], ignore_index=True)
    sampled_df = sampled_df.drop_duplicates(["video_id", "frame_idx"]).sort_values("frame_idx")

    write_jsonl(raw_stage_path, frame_records)
    write_jsonl(partial_stage_path, frame_records)
    sampled_df.to_csv(output_dir / "sampled_frames.csv", index=False, encoding="utf-8-sig")
    write_csv(output_dir / "raw_paddle_observations.csv", observation_rows)
    stage_summary = {
        "video_id": video_id,
        "paddle_device": args.device,
        "sparse_frames": int(len(sparse_samples)),
        "dense_trigger_frames": int(len(hit_frames)),
        "dense_extra_frames": int(len(dense_samples)),
        "total_ocr_frames": int(len(sampled_df)),
        "raw_paddle_observations": int(len(observation_rows)),
        "dense_trigger_regions": sorted(dense_trigger_regions),
        "runtime_sec": round(time.time() - started, 2),
        "fps_throughput": round(len(sampled_df) / max(0.001, (time.time() - started)), 2),
        "outputs": {
            "paddle_frame_records": str(raw_stage_path),
            "sampled_frames": str(output_dir / "sampled_frames.csv"),
            "raw_paddle_observations": str(output_dir / "raw_paddle_observations.csv"),
            "crops": str(crop_dir),
        },
    }
    (output_dir / "paddle_stage_summary.json").write_text(json.dumps(stage_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(stage_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
