"""
Pipeline: BTC Keyframe Processing & Validation
==============================================
Loads BTC pre-extracted keyframes and CSV frame mapping.
Constructs shots & keyframe records, performs quality checks,
exports parquets, and generates contact sheet.

Usage:
    python scripts/test_btc_keyframe_pipeline.py --video-id L21_V005 [--dataset-dir <path>] [--output <dir>]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class VideoMeta:
    video_id: str
    fps: float
    total_frames: int
    duration_s: float
    num_btc_keyframes: int

@dataclass
class ShotInfo:
    video_id: str
    shot_id: str
    start_frame: int
    end_frame: int
    start_time_s: float
    end_time_s: float
    duration_s: float
    detector_name: str = "btc_keyframe"

@dataclass
class KeyframeInfo:
    video_id: str
    shot_id: str
    frame_id: int
    timestamp_s: float
    image_path: str
    n_idx: int

@dataclass
class QualityFlag:
    frame_id: int
    n_idx: int
    issue: str  # "blur", "black", "duplicate", "missing_file"
    detail: str

@dataclass
class PipelineResult:
    status: str  # PASS / FAIL
    video_meta: VideoMeta | None = None
    shots: list[ShotInfo] = field(default_factory=list)
    keyframes: list[KeyframeInfo] = field(default_factory=list)
    quality_flags: list[QualityFlag] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    output_dir: str = ""
    elapsed_s: float = 0.0

# ---------------------------------------------------------------------------
# 1. Load BTC Map CSV
# ---------------------------------------------------------------------------

def load_btc_map_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"BTC map CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    required_cols = {"n", "pts_time", "fps", "frame_idx"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV {csv_path} missing columns. Required: {required_cols}")
    return df

# ---------------------------------------------------------------------------
# 2. Quality Checks
# ---------------------------------------------------------------------------

def _laplacian_var(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def _is_black(gray: np.ndarray, threshold: int = 15) -> bool:
    return float(np.mean(gray)) < threshold

def _frame_hash(gray: np.ndarray, size: int = 16) -> str:
    resized = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    avg = resized.mean()
    bits = (resized > avg).flatten()
    return hashlib.md5(bits.tobytes()).hexdigest()

# ---------------------------------------------------------------------------
# 3. Process BTC Keyframes
# ---------------------------------------------------------------------------

def process_btc_keyframes(video_id: str, dataset_dir: Path, output_dir: Path,
                          blur_threshold: float = 50.0) -> tuple[VideoMeta, list[ShotInfo], list[KeyframeInfo], list[QualityFlag]]:
    
    # Paths
    csv_path = dataset_dir / "map-keyframes-aic25-b1" / "map-keyframes" / f"{video_id}.csv"
    kf_dir = dataset_dir / "Keyframes_L21" / "keyframes" / video_id
    
    df = load_btc_map_csv(csv_path)
    
    fps = float(df["fps"].iloc[0])
    num_kf = len(df)
    last_frame = int(df["frame_idx"].iloc[-1])
    duration_s = float(df["pts_time"].iloc[-1])
    
    meta = VideoMeta(
        video_id=video_id,
        fps=fps,
        total_frames=last_frame + 1,
        duration_s=duration_s,
        num_btc_keyframes=num_kf,
    )
    
    shots: list[ShotInfo] = []
    keyframes: list[KeyframeInfo] = []
    quality_flags: list[QualityFlag] = []
    seen_hashes: dict[str, int] = {}
    
    out_kf_dir = output_dir / "keyframes" / video_id
    out_kf_dir.mkdir(parents=True, exist_ok=True)
    
    for idx, row in df.iterrows():
        n_idx = int(row["n"])
        pts_time = float(row["pts_time"])
        frame_idx = int(row["frame_idx"])
        
        shot_id = f"{video_id}:btc_shot:{n_idx:06d}"
        
        # Shot boundary: from current frame_idx to next frame_idx - 1
        start_frame = frame_idx
        if idx < len(df) - 1:
            end_frame = int(df["frame_idx"].iloc[idx + 1]) - 1
        else:
            end_frame = frame_idx  # last keyframe
        
        start_time = pts_time
        end_time = round(end_frame / fps, 4) if fps > 0 else pts_time
        
        shots.append(ShotInfo(
            video_id=video_id,
            shot_id=shot_id,
            start_frame=start_frame,
            end_frame=max(start_frame, end_frame),
            start_time_s=start_time,
            end_time_s=max(start_time, end_time),
            duration_s=round(max(0.0, end_time - start_time), 4),
        ))
        
        # Locate BTC keyframe image
        img_name = f"{n_idx:03d}.jpg"
        src_img_path = kf_dir / img_name
        
        if not src_img_path.exists():
            # Try alternate naming e.g. 00001.jpg or 1.jpg
            alt1 = kf_dir / f"{n_idx:05d}.jpg"
            alt2 = kf_dir / f"{n_idx}.jpg"
            if alt1.exists():
                src_img_path = alt1
            elif alt2.exists():
                src_img_path = alt2
            else:
                quality_flags.append(QualityFlag(frame_idx, n_idx, "missing_file", f"Missing {src_img_path}"))
                continue
        
        # Read image for quality checks
        img = cv2.imread(str(src_img_path))
        if img is None:
            quality_flags.append(QualityFlag(frame_idx, n_idx, "read_error", f"Cannot decode {src_img_path}"))
            continue
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Blur check
        lap = _laplacian_var(gray)
        if lap < blur_threshold:
            quality_flags.append(QualityFlag(frame_idx, n_idx, "blur", f"laplacian_var={lap:.1f}"))
            
        # Black check
        if _is_black(gray):
            quality_flags.append(QualityFlag(frame_idx, n_idx, "black", f"mean={np.mean(gray):.1f}"))
            
        # Duplicate check
        fhash = _frame_hash(gray)
        if fhash in seen_hashes:
            quality_flags.append(QualityFlag(frame_idx, n_idx, "duplicate", f"same_hash_as_n_{seen_hashes[fhash]}"))
        seen_hashes[fhash] = n_idx
        
        # Copy / save to output directory
        dst_img_name = f"{frame_idx:09d}.jpg"
        dst_img_path = out_kf_dir / dst_img_name
        cv2.imwrite(str(dst_img_path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        keyframes.append(KeyframeInfo(
            video_id=video_id,
            shot_id=shot_id,
            frame_id=frame_idx,
            timestamp_s=pts_time,
            image_path=str(dst_img_path.relative_to(output_dir)),
            n_idx=n_idx,
        ))
        
    return meta, shots, keyframes, quality_flags

# ---------------------------------------------------------------------------
# 4. Parquet Export
# ---------------------------------------------------------------------------

def export_parquets(shots: list[ShotInfo], keyframes: list[KeyframeInfo], output_dir: Path) -> tuple[Path, Path]:
    shots_path = output_dir / "shots.parquet"
    kf_path = output_dir / "keyframes.parquet"
    
    shots_df = pd.DataFrame([asdict(s) for s in shots])
    kf_df = pd.DataFrame([asdict(k) for k in keyframes])
    
    shots_df.to_parquet(shots_path, index=False)
    kf_df.to_parquet(kf_path, index=False)
    log.info("Saved %s (%d rows), %s (%d rows)", shots_path.name, len(shots), kf_path.name, len(keyframes))
    return shots_path, kf_path

# ---------------------------------------------------------------------------
# 5. Contact Sheet
# ---------------------------------------------------------------------------

def make_contact_sheet(keyframes: list[KeyframeInfo], output_dir: Path,
                       cols: int = 6, thumb_w: int = 320, thumb_h: int = 180) -> Path:
    if not keyframes:
        log.warning("No keyframes for contact sheet.")
        return output_dir / "contact_sheet.jpg"

    rows = (len(keyframes) + cols - 1) // cols
    pad = 4
    label_h = 22
    cell_w = thumb_w + pad
    cell_h = thumb_h + label_h + pad
    sheet_w = cell_w * cols + pad
    sheet_h = cell_h * rows + pad

    sheet = Image.new("RGB", (sheet_w, sheet_h), (30, 30, 30))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except (OSError, IOError):
        font = ImageFont.load_default()

    for idx, kf in enumerate(keyframes):
        img_abs = output_dir / kf.image_path
        if not img_abs.exists():
            continue
        thumb = Image.open(img_abs).resize((thumb_w, thumb_h), Image.LANCZOS)
        r, c = divmod(idx, cols)
        x = pad + c * cell_w
        y = pad + r * cell_h
        sheet.paste(thumb, (x, y))
        label = f"n={kf.n_idx} | f{kf.frame_id} | {kf.timestamp_s:.2f}s"
        draw.text((x + 2, y + thumb_h + 2), label, fill=(200, 200, 200), font=font)

    out_path = output_dir / "contact_sheet.jpg"
    sheet.save(str(out_path), quality=92)
    log.info("Contact sheet saved: %s (%d×%d)", out_path, sheet_w, sheet_h)
    return out_path

# ---------------------------------------------------------------------------
# 6. Timestamp Validation
# ---------------------------------------------------------------------------

def validate_timestamp_mapping(keyframes: list[KeyframeInfo], fps: float) -> list[str]:
    errors = []
    for kf in keyframes:
        expected = round(kf.frame_id / fps, 4) if fps > 0 else 0
        if abs(kf.timestamp_s - expected) > 0.05:  # Allow 50ms tolerance for roundings
            errors.append(f"n={kf.n_idx} Frame {kf.frame_id}: ts={kf.timestamp_s}s expected={expected}s")
    return errors

# ---------------------------------------------------------------------------
# 7. Main Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(video_id: str, dataset_dir: str, output_dir: str) -> PipelineResult:
    t0 = time.time()
    result = PipelineResult(status="FAIL")
    ds_dir = Path(dataset_dir).resolve()
    odir = Path(output_dir).resolve()
    odir.mkdir(parents=True, exist_ok=True)
    result.output_dir = str(odir)

    try:
        log.info("=== Processing BTC Keyframes for Video: %s ===", video_id)
        
        # Step 1: process keyframes
        meta, shots, keyframes, qflags = process_btc_keyframes(video_id, ds_dir, odir)
        result.video_meta = meta
        result.shots = shots
        result.keyframes = keyframes
        result.quality_flags = qflags
        
        log.info("Video %s: %d BTC keyframes, %.1f fps, %.1fs duration",
                 meta.video_id, meta.num_btc_keyframes, meta.fps, meta.duration_s)

        # Step 2: export parquet
        log.info("=== Exporting Parquet ===")
        export_parquets(shots, keyframes, odir)

        # Step 3: contact sheet
        log.info("=== Generating Contact Sheet ===")
        make_contact_sheet(keyframes, odir)

        # Step 4: timestamp validation
        log.info("=== Validating Timestamp Mapping ===")
        ts_errors = validate_timestamp_mapping(keyframes, meta.fps)
        if ts_errors:
            for e in ts_errors:
                result.errors.append(f"TIMESTAMP_MISMATCH: {e}")

        # Check fatal errors
        fatal = [e for e in result.errors if "FATAL" in e or "missing_file" in e]
        result.status = "FAIL" if fatal or len(keyframes) == 0 else "PASS"

    except Exception as exc:
        log.exception("BTC Keyframe pipeline failed")
        result.errors.append(f"FATAL: {exc}")
        result.status = "FAIL"

    result.elapsed_s = round(time.time() - t0, 2)
    return result

# ---------------------------------------------------------------------------
# 8. Report
# ---------------------------------------------------------------------------

def print_report(r: PipelineResult) -> None:
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  BTC KEYFRAME PIPELINE RESULT: {r.status}")
    print(sep)
    if r.video_meta:
        m = r.video_meta
        print(f"  Video ID        : {m.video_id}")
        print(f"  FPS             : {m.fps:.2f}")
        print(f"  Duration        : {m.duration_s:.2f}s ({m.total_frames} frames)")
        print(f"  BTC Keyframes   : {m.num_btc_keyframes}")
    print(f"  Shots           : {len(r.shots)}")
    print(f"  Keyframes Saved : {len(r.keyframes)}")
    print(f"  Quality Flags   : {len(r.quality_flags)}")
    if r.quality_flags:
        from collections import Counter
        cnt = Counter(q.issue for q in r.quality_flags)
        for issue, n in cnt.most_common():
            print(f"    - {issue}: {n}")
    if r.errors:
        print(f"  Errors ({len(r.errors)}):")
        for e in r.errors[:20]:
            print(f"    ✗ {e}")
    print(f"  Elapsed Time    : {r.elapsed_s}s")
    print(f"  Output Dir      : {r.output_dir}")
    print(sep)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="BTC Keyframe Processing Pipeline")
    parser.add_argument("--video-id", default="L21_V005", help="Video ID (e.g. L21_V005)")
    from _bootstrap import PROJECT_ROOT
    parser.add_argument("--dataset-dir", default=str(PROJECT_ROOT / "datasets_L21"), help="Dataset root directory")
    parser.add_argument("--output", default=str(PROJECT_ROOT / "outputs" / "pipeline_btc_output"), help="Output directory")
    args = parser.parse_args()

    result = run_pipeline(args.video_id, args.dataset_dir, args.output)
    print_report(result)

    # Save summary JSON
    summary = {
        "status": result.status,
        "source_policy": "BTC_KEYFRAME",
        "video_id": result.video_meta.video_id if result.video_meta else None,
        "num_shots": len(result.shots),
        "num_keyframes": len(result.keyframes),
        "num_quality_flags": len(result.quality_flags),
        "quality_breakdown": {},
        "errors": result.errors,
        "elapsed_s": result.elapsed_s,
        "output_dir": result.output_dir,
    }
    if result.quality_flags:
        from collections import Counter
        summary["quality_breakdown"] = dict(Counter(q.issue for q in result.quality_flags))
    
    summary_path = Path(args.output) / "pipeline_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Summary saved: %s", summary_path)

    sys.exit(0 if result.status == "PASS" else 1)

if __name__ == "__main__":
    main()
