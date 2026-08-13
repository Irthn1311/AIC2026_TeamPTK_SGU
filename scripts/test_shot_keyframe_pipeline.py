"""
Pipeline: Video → Shot Detection (TransNetV2) → Adaptive Keyframe Extraction
==============================================================================
Single-video test script. Extensible for batch/Kaggle processing.

Usage:
    python scripts/test_shot_keyframe_pipeline.py --video <path> [--output <dir>] [--threshold 0.5]
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

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
    path: str
    fps: float
    total_frames: int
    width: int
    height: int
    duration_s: float

@dataclass
class ShotInfo:
    video_id: str
    shot_id: str
    start_frame: int
    end_frame: int
    start_time_s: float
    end_time_s: float
    duration_s: float
    confidence: float | None

@dataclass
class KeyframeInfo:
    video_id: str
    shot_id: str
    frame_id: int
    timestamp_s: float
    image_path: str

@dataclass
class QualityFlag:
    frame_id: int
    issue: str  # "blur", "black", "duplicate"
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
# 1. Video Probe
# ---------------------------------------------------------------------------

def probe_video(video_path: Path) -> VideoMeta:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    video_id = video_path.stem
    return VideoMeta(video_id=video_id, path=str(video_path), fps=fps,
                     total_frames=total, width=w, height=h,
                     duration_s=total / fps if fps > 0 else 0)

# ---------------------------------------------------------------------------
# 2. Shot Detection (TransNetV2)
# ---------------------------------------------------------------------------

def detect_shots_transnetv2(video_path: Path, video_meta: VideoMeta,
                            threshold: float = 0.5) -> list[ShotInfo]:
    from transnetv2_pytorch import TransNetV2
    log.info("Loading TransNetV2 model...")
    model = TransNetV2()
    log.info("Running shot detection (threshold=%.2f)...", threshold)
    scenes = model.detect_scenes(str(video_path), threshold=threshold)
    log.info("TransNetV2 detected %d shots.", len(scenes))

    shots: list[ShotInfo] = []
    for i, sc in enumerate(scenes):
        sf = int(sc["start_frame"])
        ef = int(sc["end_frame"])
        st = sf / video_meta.fps
        et = ef / video_meta.fps
        shots.append(ShotInfo(
            video_id=video_meta.video_id,
            shot_id=f"{video_meta.video_id}:shot:{i:06d}",
            start_frame=sf, end_frame=ef,
            start_time_s=round(st, 4), end_time_s=round(et, 4),
            duration_s=round(et - st, 4),
            confidence=sc.get("probability"),
        ))
    return shots

# ---------------------------------------------------------------------------
# 3. Adaptive Keyframe Sampling
# ---------------------------------------------------------------------------

def adaptive_keyframe_ids(shot: ShotInfo, fps: float,
                          short_max_s: float = 5.0,
                          medium_max_s: float = 15.0,
                          long_interval_s: float = 5.0) -> list[int]:
    """Return frame IDs to extract from one shot using adaptive sampling."""
    dur = shot.duration_s
    sf, ef = shot.start_frame, shot.end_frame
    n_frames = ef - sf + 1

    if n_frames <= 1:
        return [sf]

    # Short shot: center only
    if dur <= short_max_s:
        return [(sf + ef) // 2]

    # Medium shot: start-25%, center, end-75%
    if dur <= medium_max_s:
        q1 = sf + n_frames // 4
        mid = (sf + ef) // 2
        q3 = sf + 3 * n_frames // 4
        return sorted(set([q1, mid, q3]))

    # Long shot: uniform interval
    interval_frames = max(1, int(long_interval_s * fps))
    ids = list(range(sf, ef + 1, interval_frames))
    if ids[-1] != ef and (ef - ids[-1]) > interval_frames // 3:
        ids.append(ef)
    return ids

# ---------------------------------------------------------------------------
# 4. Frame Extraction & Quality Checks
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

def extract_keyframes(video_path: Path, video_meta: VideoMeta,
                      shots: list[ShotInfo], output_dir: Path,
                      blur_threshold: float = 50.0) -> tuple[list[KeyframeInfo], list[QualityFlag]]:
    frames_dir = output_dir / "keyframes" / video_meta.video_id
    frames_dir.mkdir(parents=True, exist_ok=True)

    all_frame_ids: list[tuple[ShotInfo, int]] = []
    for shot in shots:
        fids = adaptive_keyframe_ids(shot, video_meta.fps)
        for fid in fids:
            all_frame_ids.append((shot, fid))

    # Sort by frame_id for sequential read
    all_frame_ids.sort(key=lambda x: x[1])

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot reopen video: {video_path}")

    keyframes: list[KeyframeInfo] = []
    quality_flags: list[QualityFlag] = []
    seen_hashes: dict[str, int] = {}

    for shot, fid in all_frame_ids:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = cap.read()
        if not ret:
            quality_flags.append(QualityFlag(fid, "read_error", f"Cannot read frame {fid}"))
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Quality checks
        if _is_black(gray):
            quality_flags.append(QualityFlag(fid, "black", f"mean={np.mean(gray):.1f}"))

        lap = _laplacian_var(gray)
        if lap < blur_threshold:
            quality_flags.append(QualityFlag(fid, "blur", f"laplacian_var={lap:.1f}"))

        fhash = _frame_hash(gray)
        if fhash in seen_hashes:
            quality_flags.append(QualityFlag(fid, "duplicate",
                                             f"same_hash_as_frame_{seen_hashes[fhash]}"))
        seen_hashes[fhash] = fid

        # Save
        img_name = f"{fid:09d}.jpg"
        img_path = frames_dir / img_name
        cv2.imwrite(str(img_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

        ts = fid / video_meta.fps if video_meta.fps > 0 else 0
        keyframes.append(KeyframeInfo(
            video_id=video_meta.video_id, shot_id=shot.shot_id,
            frame_id=fid, timestamp_s=round(ts, 4),
            image_path=str(img_path.relative_to(output_dir)),
        ))

    cap.release()
    log.info("Extracted %d keyframes, %d quality flags.", len(keyframes), len(quality_flags))
    return keyframes, quality_flags

# ---------------------------------------------------------------------------
# 5. Parquet Export
# ---------------------------------------------------------------------------

def export_parquets(shots: list[ShotInfo], keyframes: list[KeyframeInfo],
                    output_dir: Path) -> tuple[Path, Path]:
    shots_path = output_dir / "shots.parquet"
    kf_path = output_dir / "keyframes.parquet"
    pd.DataFrame([asdict(s) for s in shots]).to_parquet(shots_path, index=False)
    pd.DataFrame([asdict(k) for k in keyframes]).to_parquet(kf_path, index=False)
    log.info("Saved %s (%d rows), %s (%d rows)", shots_path.name, len(shots),
             kf_path.name, len(keyframes))
    return shots_path, kf_path

# ---------------------------------------------------------------------------
# 6. Contact Sheet
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
        label = f"f{kf.frame_id} | {kf.timestamp_s:.2f}s"
        draw.text((x + 2, y + thumb_h + 2), label, fill=(200, 200, 200), font=font)

    out_path = output_dir / "contact_sheet.jpg"
    sheet.save(str(out_path), quality=92)
    log.info("Contact sheet saved: %s (%d×%d)", out_path, sheet_w, sheet_h)
    return out_path

# ---------------------------------------------------------------------------
# 7. Timestamp Mapping Validation
# ---------------------------------------------------------------------------

def validate_timestamp_mapping(keyframes: list[KeyframeInfo], fps: float) -> list[str]:
    errors = []
    for kf in keyframes:
        expected = round(kf.frame_id / fps, 4) if fps > 0 else 0
        if abs(kf.timestamp_s - expected) > 0.01:
            errors.append(f"Frame {kf.frame_id}: ts={kf.timestamp_s} expected={expected}")
    return errors

# ---------------------------------------------------------------------------
# 8. Main Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(video_path: str, output_dir: str, threshold: float = 0.5) -> PipelineResult:
    t0 = time.time()
    result = PipelineResult(status="FAIL")
    vpath = Path(video_path).resolve()
    odir = Path(output_dir).resolve()
    odir.mkdir(parents=True, exist_ok=True)
    result.output_dir = str(odir)

    try:
        # Step 1: probe
        log.info("=== Step 1: Probe video ===")
        meta = probe_video(vpath)
        result.video_meta = meta
        log.info("Video: %s | %.1f fps | %d frames | %dx%d | %.1fs",
                 meta.video_id, meta.fps, meta.total_frames, meta.width, meta.height, meta.duration_s)

        # Step 2: shot detection
        log.info("=== Step 2: TransNetV2 Shot Detection ===")
        shots = detect_shots_transnetv2(vpath, meta, threshold)
        result.shots = shots

        # Step 3: keyframe extraction
        log.info("=== Step 3: Adaptive Keyframe Extraction ===")
        keyframes, qflags = extract_keyframes(vpath, meta, shots, odir)
        result.keyframes = keyframes
        result.quality_flags = qflags

        # Step 4: export parquet
        log.info("=== Step 4: Export Parquet ===")
        export_parquets(shots, keyframes, odir)

        # Step 5: contact sheet
        log.info("=== Step 5: Contact Sheet ===")
        make_contact_sheet(keyframes, odir)

        # Step 6: timestamp validation
        log.info("=== Step 6: Timestamp Validation ===")
        ts_errors = validate_timestamp_mapping(keyframes, meta.fps)
        if ts_errors:
            for e in ts_errors:
                result.errors.append(f"TIMESTAMP_MISMATCH: {e}")

        # Determine PASS/FAIL
        fatal = [e for e in result.errors if "FATAL" in e]
        result.status = "FAIL" if fatal or len(shots) == 0 else "PASS"

    except Exception as exc:
        log.exception("Pipeline failed")
        result.errors.append(f"FATAL: {exc}")
        result.status = "FAIL"

    result.elapsed_s = round(time.time() - t0, 2)
    return result

# ---------------------------------------------------------------------------
# 9. Report
# ---------------------------------------------------------------------------

def print_report(r: PipelineResult) -> None:
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  PIPELINE RESULT: {r.status}")
    print(sep)
    if r.video_meta:
        m = r.video_meta
        print(f"  Video       : {m.video_id} ({m.path})")
        print(f"  Resolution  : {m.width}x{m.height} @ {m.fps:.2f} fps")
        print(f"  Duration    : {m.duration_s:.2f}s ({m.total_frames} frames)")
    print(f"  Shots       : {len(r.shots)}")
    print(f"  Keyframes   : {len(r.keyframes)}")
    print(f"  Quality flags: {len(r.quality_flags)}")
    if r.quality_flags:
        from collections import Counter
        cnt = Counter(q.issue for q in r.quality_flags)
        for issue, n in cnt.most_common():
            print(f"    - {issue}: {n}")
    if r.errors:
        print(f"  Errors ({len(r.errors)}):")
        for e in r.errors[:20]:
            print(f"    ✗ {e}")
    print(f"  Elapsed     : {r.elapsed_s}s")
    print(f"  Output dir  : {r.output_dir}")
    print(sep)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Shot Detection & Keyframe Extraction Pipeline")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--output", default="pipeline_output", help="Output directory")
    parser.add_argument("--threshold", type=float, default=0.5, help="TransNetV2 threshold")
    args = parser.parse_args()

    result = run_pipeline(args.video, args.output, args.threshold)
    print_report(result)

    # Also save JSON summary for programmatic consumption
    import json
    summary = {
        "status": result.status,
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
