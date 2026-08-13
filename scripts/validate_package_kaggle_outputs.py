from __future__ import annotations

import argparse
import json
import os
import tarfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path


def resolve(path: str | Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def file_info(path: Path, root: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": path.as_posix(),
        "relative_path": path.relative_to(root).as_posix() if path.is_relative_to(root) else path.name,
        "size_bytes": stat.st_size,
        "modified_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(timespec="seconds"),
    }


def required_index_files(output_root: Path) -> list[Path]:
    keyframe_root = resolve(os.environ.get("AIC_KEYFRAME_OUTPUT_ROOT", output_root / "keyframe_v2_full"))
    btc_map = keyframe_root / "indexes" / "keyframe_btc_global_map.parquet"
    v2_map = keyframe_root / "indexes" / "keyframe_v2_global_map.parquet"
    keyframe_map = btc_map if btc_map.is_file() else v2_map
    visual_btc = keyframe_root / "indexes" / "visual" / "l21_visual_btc_flat_ip.faiss"
    visual_v2 = keyframe_root / "indexes" / "visual" / "l21_visual_v2_flat_ip.faiss"
    objects_btc = keyframe_root / "indexes" / "object" / "l21_objects_btc.parquet"
    objects_v2 = keyframe_root / "indexes" / "object" / "l21_objects_v2.parquet"
    return [
        keyframe_map,
        visual_btc if visual_btc.is_file() else visual_v2,
        objects_btc if objects_btc.is_file() else objects_v2,
        output_root / "indexes" / "ocr_temporal_v3_full_tracking" / "l21_ocr_temporal_v3_flat_ip.faiss",
        output_root / "indexes" / "ocr_temporal_v3_full_tracking" / "l21_ocr_temporal_v3_corpus.parquet",
        output_root / "indexes" / "asr_v3" / "l21_asr_v3_flat_ip.faiss",
        output_root / "indexes" / "asr_v3" / "l21_asr_v3_corpus.parquet",
    ]


def validate_files(paths: list[Path]) -> tuple[list[Path], list[dict[str, str]]]:
    ok: list[Path] = []
    missing: list[dict[str, str]] = []
    for path in paths:
        if not path.is_file():
            missing.append({"path": path.as_posix(), "reason": "missing"})
        elif path.stat().st_size <= 0:
            missing.append({"path": path.as_posix(), "reason": "empty"})
        else:
            ok.append(path)
    return ok, missing


def zip_files(paths: list[Path], output_path: Path, root: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in paths:
            zf.write(path, path.relative_to(root).as_posix())


def tar_keyframes(keyframe_root: Path, output_path: Path) -> int:
    image_paths = []
    btc_map = keyframe_root / "indexes" / "keyframe_btc_global_map.parquet"
    v2_map = keyframe_root / "indexes" / "keyframe_v2_global_map.parquet"
    map_path = btc_map if btc_map.is_file() else v2_map
    if map_path.is_file():
        import pandas as pd

        df = pd.read_parquet(map_path)
        if "source" in df.columns and (df["source"].astype(str) == "btc_keyframe").any():
            image_paths = [Path(str(path)) for path in df.get("image_path", []) if Path(str(path)).is_file()]
    if not image_paths:
        image_paths = sorted((keyframe_root).glob("L*_V*/keyframes/*.jpg"))
    if not image_paths and map_path.is_file():
        import pandas as pd

        df = pd.read_parquet(map_path)
        image_paths = [Path(str(path)) for path in df.get("image_path", []) if Path(str(path)).is_file()]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output_path, "w:gz") as tf:
        for path in image_paths:
            if path.is_relative_to(keyframe_root.parent):
                arcname = path.relative_to(keyframe_root.parent).as_posix()
            else:
                video_id = path.parent.name
                arcname = f"{keyframe_root.name}/{video_id}/keyframes/{path.name}"
            tf.add(path, arcname)
    return len(image_paths)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate and package Kaggle offline pipeline outputs.")
    parser.add_argument("--output-root", default="/kaggle/working/artifacts")
    parser.add_argument("--indices-zip", default=None)
    parser.add_argument("--keyframes-tar", default=None)
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()

    output_root = resolve(args.output_root)
    indices_zip = resolve(args.indices_zip or (output_root / "kaggle_outputs_indices.zip"))
    keyframes_tar = resolve(args.keyframes_tar or (output_root / "kaggle_outputs_keyframes.tar.gz"))
    required = required_index_files(output_root)
    ok_files, missing = validate_files(required)

    if missing and not args.allow_missing:
        report = {"output_root": str(output_root), "status": "failed", "missing_or_empty": missing}
        report_path = output_root / "kaggle_package_validation.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 2

    zip_files(ok_files, indices_zip, output_root)
    keyframe_root = resolve(os.environ.get("AIC_KEYFRAME_OUTPUT_ROOT", output_root / "keyframe_v2_full"))
    keyframe_count = tar_keyframes(keyframe_root, keyframes_tar)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "output_root": str(output_root),
        "status": "ok" if not missing else "partial",
        "required_count": len(required),
        "packaged_index_count": len(ok_files),
        "missing_or_empty": missing,
        "indices_zip": file_info(indices_zip, output_root),
        "keyframes_tar": file_info(keyframes_tar, output_root),
        "keyframe_images": keyframe_count,
        "index_files": [file_info(path, output_root) for path in ok_files],
    }
    report_path = output_root / "kaggle_package_validation.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
