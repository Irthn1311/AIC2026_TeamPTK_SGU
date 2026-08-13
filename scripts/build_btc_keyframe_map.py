from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from _bootstrap import PROJECT_ROOT


def natural_video_key(video_id: str) -> tuple[str, int]:
    try:
        prefix, number = video_id.rsplit("_V", 1)
        return prefix, int(number)
    except Exception:
        return video_id, 0


def resolve(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def find_mapping_path(data_root: Path, video_id: str) -> Path:
    candidates = [
        data_root / "map-keyframes-aic25-b1" / "map-keyframes" / f"{video_id}.csv",
        data_root / "map-keyframes" / f"{video_id}.csv",
    ]
    for path in candidates:
        if path.is_file():
            return path
    matches = sorted(data_root.glob(f"**/map-keyframes/{video_id}.csv"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"BTC mapping CSV not found for {video_id}")


def find_keyframe_dir(data_root: Path, video_id: str) -> Path:
    level = video_id.split("_", 1)[0]
    candidates = [
        data_root / "keyframes" / video_id,
        data_root / f"Keyframes_{level}" / "keyframes" / video_id,
    ]
    for path in candidates:
        if path.is_dir():
            return path
    matches = sorted(data_root.glob(f"Keyframes_{level}*/keyframes/{video_id}"))
    if matches:
        return matches[0]
    matches = sorted(data_root.glob(f"**/keyframes/{video_id}"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"BTC keyframe directory not found for {video_id}")


def discover_video_ids(data_root: Path) -> list[str]:
    ids = set()
    for path in sorted(data_root.glob("**/map-keyframes/L*_V*.csv")):
        ids.add(path.stem)
    if ids:
        return sorted(ids, key=natural_video_key)
    for path in sorted(data_root.glob("**/keyframes/L*_V*")):
        if path.is_dir():
            ids.add(path.name)
    return sorted(ids, key=natural_video_key)


def image_path_for_row(keyframe_dir: Path, n: int) -> Path:
    for name in (f"{n:03d}.jpg", f"{n:04d}.jpg", f"{n:05d}.jpg", f"{n:06d}.jpg"):
        path = keyframe_dir / name
        if path.is_file():
            return path
    matches = sorted(keyframe_dir.glob(f"*{n}*.jpg"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"BTC keyframe image not found for n={n} in {keyframe_dir}")


def build_video_rows(data_root: Path, output_root: Path, video_id: str, start_global_id: int) -> tuple[list[dict], dict]:
    mapping_path = find_mapping_path(data_root, video_id)
    keyframe_dir = find_keyframe_dir(data_root, video_id)
    df = pd.read_csv(mapping_path)
    if "n" not in df.columns or "frame_idx" not in df.columns:
        raise ValueError(f"BTC mapping must include n and frame_idx: {mapping_path}")

    out_video = output_root / video_id
    out_video.mkdir(parents=True, exist_ok=True)
    (out_video / "keyframes").mkdir(parents=True, exist_ok=True)

    rows = []
    for local_idx, row in enumerate(df.sort_values("n").to_dict("records")):
        n = int(row["n"])
        image_path = image_path_for_row(keyframe_dir, n).resolve(strict=False)
        timestamp_sec = float(row.get("pts_time", row.get("timestamp_seconds", 0.0)) or 0.0)
        actual_frame_id = int(row.get("frame_idx", 0) or 0)
        global_id = start_global_id + len(rows)
        rows.append(
            {
                "global_v2_id": global_id,
                "video_id": video_id,
                "keyframe_v2_idx": local_idx,
                "actual_frame_id": actual_frame_id,
                "timestamp_sec": timestamp_sec,
                "image_path": str(image_path),
                "shot_id": local_idx,
                "quality_score": 1.0,
                "representative_score": 1.0,
                "temporal_score": 1.0,
                "final_score": 1.0,
                "source": "btc_keyframe",
                "btc_n": n,
                "keyframe_name": image_path.name,
            }
        )

    video_df = pd.DataFrame(rows)
    map_df = pd.DataFrame(
        {
            "global_id": list(range(len(rows))),
            "video_id": video_df["video_id"] if not video_df.empty else [],
            "keyframe_v2_idx": video_df["keyframe_v2_idx"] if not video_df.empty else [],
            "actual_frame_id": video_df["actual_frame_id"] if not video_df.empty else [],
            "timestamp_ms": (video_df["timestamp_sec"] * 1000).round().astype(int) if not video_df.empty else [],
            "shot_id": video_df["shot_id"] if not video_df.empty else [],
            "image_path": video_df["image_path"] if not video_df.empty else [],
        }
    )
    map_df.to_csv(out_video / "keyframe_v2_map.csv", index=False, encoding="utf-8-sig")
    map_df.to_parquet(out_video / "keyframe_v2_map.parquet", index=False)
    video_df.to_csv(out_video / "final_keyframes.csv", index=False, encoding="utf-8-sig")
    summary = {
        "video_id": video_id,
        "source": "btc_keyframe",
        "mapping_path": str(mapping_path),
        "keyframe_dir": str(keyframe_dir),
        "final_keyframes": int(len(rows)),
    }
    (out_video / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return rows, summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Build keyframe_v2-compatible maps from provided BTC keyframes.")
    parser.add_argument("--data-root", default="/kaggle/input/datasets/nadkli/dataset-aic")
    parser.add_argument("--output", default="/kaggle/working/artifacts/keyframe_v2_full")
    parser.add_argument("--video-id", action="append", default=[])
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    started = time.time()
    data_root = resolve(args.data_root)
    output_root = resolve(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    index_dir = output_root / "indexes"
    index_dir.mkdir(parents=True, exist_ok=True)

    video_ids = [v.strip() for v in args.video_id if v.strip()] or discover_video_ids(data_root)
    video_ids = sorted(dict.fromkeys(video_ids), key=natural_video_key)
    if args.limit:
        video_ids = video_ids[: max(0, args.limit)]
    if not video_ids:
        raise FileNotFoundError(f"No BTC keyframe video ids found under {data_root}")

    all_rows = []
    summaries = []
    errors = []
    for idx, video_id in enumerate(video_ids, start=1):
        try:
            rows, summary = build_video_rows(data_root, output_root, video_id, len(all_rows))
            all_rows.extend(rows)
            summaries.append(summary)
            print(f"[{idx}/{len(video_ids)}] BTC keyframes {video_id}: {len(rows)}")
        except Exception as exc:
            errors.append({"video_id": video_id, "exception": repr(exc)})
            print(f"[{idx}/{len(video_ids)}] FAIL {video_id}: {exc}")

    global_df = pd.DataFrame(all_rows)
    if global_df.empty:
        raise RuntimeError(f"BTC keyframe map produced 0 rows. errors={errors[:5]}")
    global_df.to_csv(index_dir / "keyframe_v2_global_map.csv", index=False, encoding="utf-8-sig")
    global_df.to_parquet(index_dir / "keyframe_v2_global_map.parquet", index=False)
    report = {
        "source": "btc_keyframe",
        "data_root": str(data_root),
        "output_root": str(output_root),
        "videos": len(summaries),
        "keyframes": int(len(global_df)),
        "errors": errors,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    (output_root / "summary" / "btc_keyframe_summary.json").parent.mkdir(parents=True, exist_ok=True)
    (output_root / "summary" / "btc_keyframe_summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
