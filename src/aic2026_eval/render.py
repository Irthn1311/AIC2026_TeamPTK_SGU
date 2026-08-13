"""Neutral BTC overview atlas and exact raw-frame dense evidence rendering."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .mapping import read_mapping


def evenly_spaced(values: Sequence[Any], count: int) -> list[Any]:
    if count <= 0 or not values:
        return []
    if len(values) <= count:
        return list(values)
    positions = np.linspace(0, len(values) - 1, count, dtype=int)
    return [values[int(index)] for index in positions]


def _font(size: int):
    from PIL import ImageFont

    for candidate in ("DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _contact_sheet(
    path: Path,
    *,
    title: str,
    images: Sequence[Any],
    labels: Sequence[str],
    columns: int = 6,
    tile_size: tuple[int, int] = (256, 144),
    quality: int = 82,
) -> None:
    from PIL import Image, ImageDraw, ImageOps

    if len(images) != len(labels):
        raise ValueError("contact-sheet images and labels must align")
    rows = max(1, math.ceil(len(images) / columns))
    width, image_height = tile_size
    label_height, header_height = 48, 52
    sheet = Image.new(
        "RGB",
        (columns * width, header_height + rows * (image_height + label_height)),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    draw.text((10, 12), title, fill="black", font=_font(20))
    for index, (image, label) in enumerate(zip(images, labels, strict=True)):
        x, y = (
            (index % columns) * width,
            header_height + (index // columns) * (image_height + label_height),
        )
        with (
            Image.open(image)
            if isinstance(image, str | Path)
            else Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB") as source
        ):
            fitted = ImageOps.fit(
                source.convert("RGB"),
                (width, image_height),
                method=Image.Resampling.LANCZOS,
            )
            sheet.paste(fitted, (x, y))
        draw.multiline_text(
            (x + 5, y + image_height + 3),
            label,
            fill="black",
            font=_font(13),
            spacing=1,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, format="JPEG", quality=quality, optimize=True)


def _keyframes_by_n(directory: Path) -> dict[int, Path]:
    result = {}
    if not directory.is_dir():
        return result
    for path in directory.iterdir():
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            try:
                result[int(path.stem)] = path
            except ValueError:
                continue
    return result


def render_overview_atlas(
    candidates: list[dict[str, Any]],
    output_root: str | Path,
    *,
    frames_per_video: int = 24,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    root = Path(output_root)
    atlas_root = root / "atlas"
    manifests, issues = [], []
    for candidate in candidates:
        mapping = read_mapping(candidate["mapping_path"])
        keyframes = _keyframes_by_n(Path(candidate["keyframe_directory"]))
        selected_rows = evenly_spaced(mapping, frames_per_video)
        tiles = []
        for row in selected_rows:
            image_path = keyframes.get(row["n"])
            if image_path is None:
                issues.append(
                    {
                        "severity": "WARNING",
                        "code": "ATLAS_KEYFRAME_MISSING",
                        "video_id": candidate["video_id"],
                        "btc_n": row["n"],
                    }
                )
                continue
            tiles.append(
                {
                    "btc_n": row["n"],
                    "original_frame_idx": row["frame_idx"],
                    "timestamp_sec": row["pts_time"],
                    "image_path": str(image_path),
                }
            )
        sheet = atlas_root / f"{candidate['video_id']}.jpg"
        _contact_sheet(
            sheet,
            title=f"{candidate['video_id']} | {candidate['role']} | BTC timeline overview",
            images=[row["image_path"] for row in tiles],
            labels=[
                (
                    f"{candidate['video_id']}  BTC n={row['btc_n']}\n"
                    f"frame={row['original_frame_idx']}  t={row['timestamp_sec']:.3f}s"
                )
                for row in tiles
            ],
        )
        manifests.append(
            {
                "video_id": candidate["video_id"],
                "role": candidate["role"],
                "sheet_path": sheet.relative_to(root).as_posix(),
                "tile_count": len(tiles),
                "target_tile_count": len(selected_rows),
                "selection_policy": "UNIFORM_OVER_FULL_BTC_MAPPING_ORDER",
                "tiles": [
                    {key: value for key, value in row.items() if key != "image_path"}
                    for row in tiles
                ],
            }
        )
    montage_paths = []
    page_count = min(6, max(4, math.ceil(len(manifests) / 8)))
    chunks = (
        [
            manifests[start : start + math.ceil(len(manifests) / page_count)]
            for start in range(0, len(manifests), math.ceil(len(manifests) / page_count))
        ]
        if manifests
        else []
    )
    for page_number, chunk in enumerate(chunks, 1):
        montage = atlas_root / f"_index_{page_number:02d}.jpg"
        _contact_sheet(
            montage,
            title=f"Held-out candidate atlas index {page_number}/{len(chunks)}",
            images=[root / row["sheet_path"] for row in chunk],
            labels=[f"{row['video_id']} | {row['role']}" for row in chunk],
            columns=2,
            tile_size=(480, 270),
            quality=80,
        )
        montage_paths.append(montage.relative_to(root).as_posix())
    index = {
        "status": (
            "READY"
            if len(manifests) == len(candidates)
            and all(
                row["tile_count"] > 0 and row["tile_count"] == row["target_tile_count"]
                for row in manifests
            )
            else "FAIL"
        ),
        "video_count": len(manifests),
        "frames_per_video_target": frames_per_video,
        "atlas_sheet_count": len(manifests),
        "montage_page_count": len(montage_paths),
        "montage_paths": montage_paths,
        "semantic_selection_used": False,
        "retrieval_scores_used": False,
        "labels_used": False,
    }
    return manifests, index, issues


RawDecoder = Callable[[Path, list[int]], list[tuple[int, np.ndarray]]]


def decode_raw_frames(video_path: Path, frame_ids: list[int]) -> list[tuple[int, np.ndarray]]:
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"VIDEO_OPEN_FAILED: {video_path}")
    frames = []
    try:
        for frame_id in frame_ids:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ok, bgr = capture.read()
            position = int(round(capture.get(cv2.CAP_PROP_POS_FRAMES))) - 1
            if not ok or position != frame_id:
                raise RuntimeError(
                    f"RAW_FRAME_IDENTITY_FAILED requested={frame_id} actual={position}"
                )
            frames.append((position, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    finally:
        capture.release()
    return frames


def render_actual_frame_sheet(
    path: str | Path,
    *,
    title: str,
    decoded_frames: list[tuple[int, np.ndarray]],
) -> None:
    """Render decoded raw frames without deriving or changing their frame identities."""
    _contact_sheet(
        Path(path),
        title=title,
        images=[image for _, image in decoded_frames],
        labels=[f"actual_frame_idx={frame_id}" for frame_id, _ in decoded_frames],
        columns=6,
        tile_size=(256, 144),
        quality=86,
    )


def requested_frame_ids(request: dict[str, Any], video: dict[str, Any]) -> list[int]:
    center = request.get("approx_original_frame_idx")
    if not isinstance(center, int) or isinstance(center, bool):
        raise ValueError("approx_original_frame_idx must be an integer")
    if not 0 <= center < video["total_frames"]:
        raise ValueError("approx_original_frame_idx is outside the raw video")
    mode = str(request.get("mode", "")).upper()
    fps = float(video["fps"])
    if mode == "INTERVAL":
        radius = float(request.get("radius_seconds", 3.0))
        start = max(0, center - round(radius * fps))
        end = min(video["total_frames"] - 1, center + round(radius * fps))
        return evenly_spaced(list(range(start, end + 1)), 28)
    if mode == "MOMENT_DENSE":
        radius = float(request.get("radius_seconds", 0.75))
        if not 0 < radius <= 1.0:
            raise ValueError("MOMENT_DENSE radius_seconds must be in (0, 1]")
        start = max(0, center - math.ceil(radius * fps))
        end = min(video["total_frames"] - 1, center + math.ceil(radius * fps))
        return list(range(start, end + 1))
    raise ValueError("mode must be INTERVAL or MOMENT_DENSE")


def render_dense_requests(
    requests: list[dict[str, Any]],
    inventory: list[dict[str, Any]],
    output_root: str | Path,
    *,
    decoder: RawDecoder = decode_raw_frames,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    root = Path(output_root)
    by_id = {row["video_id"]: row for row in inventory}
    manifests, issues, seen = [], [], set()
    for request in requests:
        anchor_id = request.get("anchor_id")
        if not isinstance(anchor_id, str) or not anchor_id or anchor_id in seen:
            raise ValueError("anchor_id must be a unique non-empty string")
        seen.add(anchor_id)
        video = by_id.get(request.get("video_id"))
        if video is None:
            raise ValueError(f"unknown video_id for anchor {anchor_id}")
        frame_ids = requested_frame_ids(request, video)
        decoded = decoder(Path(video["video_path"]), frame_ids)
        actual = [frame_id for frame_id, _ in decoded]
        if actual != frame_ids:
            raise RuntimeError(f"DENSE_FRAME_IDENTITY_MISMATCH: {anchor_id}")
        sheet = root / "dense" / f"{anchor_id}.jpg"
        _contact_sheet(
            sheet,
            title=f"{anchor_id} | {video['video_id']} | {request['mode']} | raw source frames",
            images=[image for _, image in decoded],
            labels=[f"actual_frame_idx={frame_id}" for frame_id in actual],
            columns=6,
            tile_size=(256, 144),
            quality=86,
        )
        manifests.append(
            {
                "anchor_id": anchor_id,
                "video_id": video["video_id"],
                "mode": str(request["mode"]).upper(),
                "approx_original_frame_idx": request["approx_original_frame_idx"],
                "requested_frame_ids": frame_ids,
                "actual_frame_ids": actual,
                "frame_identity_exact": True,
                "complete_dense_range": str(request["mode"]).upper() == "MOMENT_DENSE",
                "sheet_path": sheet.relative_to(root).as_posix(),
            }
        )
    return manifests, issues


__all__ = [
    "decode_raw_frames",
    "evenly_spaced",
    "render_dense_requests",
    "render_actual_frame_sheet",
    "render_overview_atlas",
    "requested_frame_ids",
]
