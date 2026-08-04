"""Stage 1 report and optional index ZIP packaging."""

from __future__ import annotations

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

REPORT_MEMBERS = (
    "run_manifest.json",
    "stage1_summary.json",
    "stage1_report.md",
    "contract_notes.json",
    "index/index_manifest.json",
    "index/catalog_manifest.json",
    "encoder/encoder_contract.json",
    "encoder/compatibility_report.json",
    "benchmark/self_retrieval_report.json",
    "benchmark/benchmark_report.json",
    "benchmark/benchmark_report.md",
)
INDEX_MEMBERS = (
    "index/index_manifest.json",
    "index/catalog_manifest.json",
    "index/video_table.json",
    "index/clip_vectors.f16.npy",
    "index/vector_norms.f32.npy",
    "index/frame_video_index.npy",
    "index/frame_n.npy",
    "index/frame_original_idx.npy",
    "index/frame_pts_time.npy",
    "index/frame_mapping_fps.npy",
    "index/duplicate_group_size.npy",
)


def create_report_bundle(stage1_root: str | Path, zip_path: str | Path) -> Path:
    root = Path(stage1_root).resolve(strict=True)
    target = Path(zip_path).resolve(strict=False)
    if root in target.parents:
        raise ValueError("Report ZIP must be outside Stage 1 root")
    members = list(REPORT_MEMBERS)
    queries = root / "queries"
    if queries.is_dir():
        members.extend(
            path.relative_to(root).as_posix()
            for path in sorted(queries.glob("*/*"))
            if path.is_file() and path.suffix.lower() in {".json", ".jsonl", ".csv", ".md"}
        )
    missing = [name for name in REPORT_MEMBERS if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing report artifacts: {', '.join(missing)}")
    with ZipFile(target, "w", compression=ZIP_DEFLATED) as archive:
        for name in members:
            archive.write(root / name, arcname=name)
    return target


def create_index_bundle(stage1_root: str | Path, zip_path: str | Path) -> Path:
    root = Path(stage1_root).resolve(strict=True)
    target = Path(zip_path).resolve(strict=False)
    if root in target.parents:
        raise ValueError("Index ZIP must be outside Stage 1 root")
    missing = [name for name in INDEX_MEMBERS if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing index artifacts: {', '.join(missing)}")
    with ZipFile(target, "w", compression=ZIP_DEFLATED) as archive:
        for name in INDEX_MEMBERS:
            archive.write(root / name, arcname=name)
    return target
