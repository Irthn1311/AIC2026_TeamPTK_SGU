from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import av
import cv2
import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import structural_similarity

from src.preprocessing.keyframe_v2.clip_scorer import ImageEmbeddingScorer
from src.preprocessing.keyframe_v2.exact_decoder import compare_images


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CLASSIFICATION_THRESHOLDS = {
    "real_plus_one_min_delta_ssim": 0.010,
    "real_plus_one_min_pixel_error_delta": 0.002,
    "ambiguous_abs_delta_ssim": 0.005,
    "ambiguous_high_ssim": 0.985,
    "ambiguous_pixel_error_delta": 0.003,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Deep validation for BTC frame-id fid+1 cases.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--validation-csv", required=True)
    parser.add_argument("--config", default="configs/keyframe_v2.yaml")
    parser.add_argument("--output-dir", default="outputs/keyframe_v2_test_real/L21_V001")
    args = parser.parse_args()

    cfg = load_config(args.config)
    force_e_local_env(cfg)
    out_dir = (PROJECT_ROOT / args.output_dir).resolve()
    debug_dir = out_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    validation = pd.read_csv(PROJECT_ROOT / args.validation_csv)
    mismatches = validation[validation["detected_convention"].eq("plus_1_anomaly")].copy()
    if mismatches.empty:
        raise RuntimeError("No plus_1_anomaly rows found.")

    video_path = (PROJECT_ROOT / args.video).resolve()
    needed = sorted(
        {
            int(row["btc_mapped_frame_id"]) + offset
            for _, row in mismatches.iterrows()
            for offset in [-2, -1, 0, 1, 2]
            if int(row["btc_mapped_frame_id"]) + offset >= 0
        }
    )
    decoded, gop = decode_frames_sequential_pyav(video_path, set(needed))

    embedder = ImageEmbeddingScorer(PROJECT_ROOT, cfg["clip"])
    if embedder.backend != "clip":
        raise RuntimeError(f"Expected real CLIP backend, got {embedder.backend}")

    btc_root = PROJECT_ROOT / "datasets_L21" / "Keyframes_L21" / "keyframes" / video_path.stem
    records: list[dict] = []
    sheet_rows: list[dict] = []
    for _, row in mismatches.iterrows():
        btc_name = str(row["btc_keyframe_name"])
        mapped_fid = int(row["btc_mapped_frame_id"])
        btc_img = cv2.imread(str(btc_root / btc_name))
        if btc_img is None:
            raise RuntimeError(f"Cannot read BTC keyframe: {btc_root / btc_name}")

        candidate_frames = [mapped_fid + offset for offset in [-2, -1, 0, 1, 2]]
        images = [decoded[fid] for fid in candidate_frames]
        clip_sims = clip_similarities(embedder, btc_img, images)

        metrics = {}
        for fid, label, img, clip_sim in zip(candidate_frames, ["fm2", "fm1", "f", "fp1", "fp2"], images, clip_sims):
            ssim = image_ssim(btc_img, img)
            pixel_err = normalized_pixel_error(btc_img, img)
            phash_dist = phash_distance(btc_img, img)
            metrics[label] = {
                "frame": fid,
                "ssim": ssim,
                "pixel_error": pixel_err,
                "phash": phash_dist,
                "clip_similarity": clip_sim,
                "key_frame": bool(gop["key_flags"].get(fid, False)),
                "pict_type": str(gop["pict_types"].get(fid, "")),
            }

        best_label = choose_best_lowlevel(metrics)
        best_frame = metrics[best_label]["frame"]
        classification = classify(metrics, best_label)
        rec = {
            "btc_keyframe": btc_name,
            "mapped_fid": mapped_fid,
            "ssim_fm2": metrics["fm2"]["ssim"],
            "ssim_fm1": metrics["fm1"]["ssim"],
            "ssim_f": metrics["f"]["ssim"],
            "ssim_fp1": metrics["fp1"]["ssim"],
            "ssim_fp2": metrics["fp2"]["ssim"],
            "pixel_error_f": metrics["f"]["pixel_error"],
            "pixel_error_fp1": metrics["fp1"]["pixel_error"],
            "phash_f": metrics["f"]["phash"],
            "phash_fp1": metrics["fp1"]["phash"],
            "clip_similarity_f": metrics["f"]["clip_similarity"],
            "clip_similarity_fp1": metrics["fp1"]["clip_similarity"],
            "best_lowlevel_frame": best_frame,
            "best_lowlevel_label": best_label,
            "delta_ssim_f_vs_fp1": metrics["fp1"]["ssim"] - metrics["f"]["ssim"],
            "classification": classification,
            "relative_position": mapped_fid / max(1, gop["decoded_frames"] - 1),
            "prev_gop_keyframe": gop["prev_key"].get(mapped_fid),
            "distance_to_prev_gop_keyframe": mapped_fid - gop["prev_key"].get(mapped_fid, mapped_fid),
            "pict_type_f": metrics["f"]["pict_type"],
            "pict_type_fp1": metrics["fp1"]["pict_type"],
            "thresholds_json": json.dumps(CLASSIFICATION_THRESHOLDS, sort_keys=True),
        }
        records.append(rec)
        sheet_rows.append(
            {
                "btc_name": btc_name,
                "mapped_fid": mapped_fid,
                "metrics": metrics,
                "images": dict(zip(["fm2", "fm1", "f", "fp1", "fp2"], images)),
                "best_label": best_label,
                "classification": classification,
                "btc_img": btc_img,
            }
        )

    deep = pd.DataFrame(records)
    shots_path = out_dir / "shots.csv"
    if shots_path.exists():
        deep = add_shot_boundary_distances(deep, pd.read_csv(shots_path))
    deep.to_csv(out_dir / "frame_id_validation_deep.csv", index=False, encoding="utf-8-sig")
    make_deep_contact_sheet(sheet_rows, debug_dir / "frame_id_deep_validation_contact_sheet.jpg")

    summary = {
        "input_cases": int(len(mismatches)),
        "classification_counts": deep["classification"].value_counts().to_dict(),
        "thresholds": CLASSIFICATION_THRESHOLDS,
        "clip_info": embedder.info,
        "decoded_backend": "pyav_sequential_full_video",
        "decoded_frames": int(gop["decoded_frames"]),
        "outputs": {
            "csv": str(out_dir / "frame_id_validation_deep.csv"),
            "contact_sheet": str(debug_dir / "frame_id_deep_validation_contact_sheet.jpg"),
        },
    }
    (out_dir / "frame_id_validation_deep_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def load_config(path: str | Path) -> dict:
    with open(PROJECT_ROOT / path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def force_e_local_env(cfg: dict) -> None:
    cache = PROJECT_ROOT / ".cache"
    model_cache = PROJECT_ROOT / cfg["paths"].get("model_cache", ".model_cache")
    for key, path in {
        "HF_HOME": cache / "huggingface",
        "HUGGINGFACE_HUB_CACHE": cache / "huggingface" / "hub",
        "TRANSFORMERS_CACHE": cache / "huggingface",
        "TORCH_HOME": model_cache / "torch",
        "XDG_CACHE_HOME": cache,
        "TMP": PROJECT_ROOT / "outputs" / "tmp",
        "TEMP": PROJECT_ROOT / "outputs" / "tmp",
    }.items():
        os.environ[key] = str(path)
        Path(path).mkdir(parents=True, exist_ok=True)


def decode_frames_sequential_pyav(video_path: Path, targets: set[int]) -> tuple[dict[int, np.ndarray], dict]:
    decoded: dict[int, np.ndarray] = {}
    prev_key_for_frame: dict[int, int] = {}
    key_flags: dict[int, bool] = {}
    pict_types: dict[int, str] = {}
    last_key = 0
    idx = -1
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for idx, frame in enumerate(container.decode(stream)):
            if frame.key_frame:
                last_key = idx
            if idx in targets:
                arr = frame.to_ndarray(format="bgr24")
                decoded[idx] = arr
                prev_key_for_frame[idx] = last_key
                key_flags[idx] = bool(frame.key_frame)
                pict_types[idx] = str(frame.pict_type)
            if len(decoded) == len(targets) and idx >= max(targets):
                break
    missing = sorted(targets.difference(decoded))
    if missing:
        raise RuntimeError(f"PyAV sequential decode missed target frames: {missing[:10]}")
    return decoded, {"decoded_frames": idx + 1, "prev_key": prev_key_for_frame, "key_flags": key_flags, "pict_types": pict_types}


def image_ssim(a_bgr: np.ndarray, b_bgr: np.ndarray) -> float:
    a, b = align_gray(a_bgr, b_bgr)
    return float(structural_similarity(a, b, data_range=255))


def normalized_pixel_error(a_bgr: np.ndarray, b_bgr: np.ndarray) -> float:
    a, b = align_rgb(a_bgr, b_bgr)
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))) / 255.0)


def align_gray(a_bgr: np.ndarray, b_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a, b = align_rgb(a_bgr, b_bgr)
    return cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)


def align_rgb(a_bgr: np.ndarray, b_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if a_bgr.shape[:2] != b_bgr.shape[:2]:
        b_bgr = cv2.resize(b_bgr, (a_bgr.shape[1], a_bgr.shape[0]), interpolation=cv2.INTER_AREA)
    return a_bgr, b_bgr


def phash_distance(a_bgr: np.ndarray, b_bgr: np.ndarray) -> int:
    return int(np.count_nonzero(phash(a_bgr) != phash(b_bgr)))


def phash(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(small)
    low = dct[:8, :8]
    med = np.median(low[1:, 1:])
    return low > med


def clip_similarities(embedder: ImageEmbeddingScorer, btc_img: np.ndarray, images: list[np.ndarray]) -> list[float]:
    embs = embedder.embed_images([btc_img] + images)
    base = embs[0]
    return [float(np.dot(base, emb)) for emb in embs[1:]]


def choose_best_lowlevel(metrics: dict) -> str:
    labels = ["fm2", "fm1", "f", "fp1", "fp2"]
    return max(labels, key=lambda label: (metrics[label]["ssim"], -metrics[label]["pixel_error"], -metrics[label]["phash"]))


def classify(metrics: dict, best_label: str) -> str:
    f = metrics["f"]
    fp1 = metrics["fp1"]
    delta_ssim = fp1["ssim"] - f["ssim"]
    pixel_delta = f["pixel_error"] - fp1["pixel_error"]
    if best_label == "f":
        return "EXACT_FID"
    if best_label == "fp1":
        ambiguous = (
            abs(delta_ssim) < CLASSIFICATION_THRESHOLDS["ambiguous_abs_delta_ssim"]
            or (
                f["ssim"] >= CLASSIFICATION_THRESHOLDS["ambiguous_high_ssim"]
                and fp1["ssim"] >= CLASSIFICATION_THRESHOLDS["ambiguous_high_ssim"]
                and abs(pixel_delta) < CLASSIFICATION_THRESHOLDS["ambiguous_pixel_error_delta"]
            )
        )
        if ambiguous:
            return "AMBIGUOUS_ADJACENT"
        if (
            delta_ssim >= CLASSIFICATION_THRESHOLDS["real_plus_one_min_delta_ssim"]
            and pixel_delta >= CLASSIFICATION_THRESHOLDS["real_plus_one_min_pixel_error_delta"]
            and fp1["phash"] <= f["phash"]
        ):
            return "REAL_PLUS_ONE"
        return "AMBIGUOUS_ADJACENT"
    if best_label in {"fm1"}:
        return "AMBIGUOUS_ADJACENT" if abs(metrics[best_label]["ssim"] - f["ssim"]) < CLASSIFICATION_THRESHOLDS["ambiguous_abs_delta_ssim"] else "OTHER"
    return "OTHER"


def add_shot_boundary_distances(deep: pd.DataFrame, shots: pd.DataFrame) -> pd.DataFrame:
    starts = shots["start_frame"].astype(int).to_numpy()
    ends = shots["end_frame"].astype(int).to_numpy()
    shot_ids = shots["shot_id"].astype(int).to_numpy()
    nearest = []
    shot_for_frame = []
    for fid in deep["mapped_fid"].astype(int):
        distances = np.minimum(np.abs(starts - fid), np.abs(ends - fid))
        hit = np.where((starts <= fid) & (fid <= ends))[0]
        shot_for_frame.append(int(shot_ids[hit[0]]) if len(hit) else -1)
        nearest.append(int(distances.min()) if len(distances) else -1)
    deep["transnetv2_shot_id_for_mapped_f"] = shot_for_frame
    deep["distance_to_nearest_transnetv2_boundary"] = nearest
    return deep


def make_deep_contact_sheet(rows: list[dict], output_path: Path) -> None:
    thumb_w, thumb_h = 220, 124
    label_h = 92
    pad = 8
    cols = 6
    header_h = 30
    width = cols * (thumb_w + pad) + pad
    height = header_h + len(rows) * (thumb_h + label_h + pad) + pad
    sheet = Image.new("RGB", (width, height), (22, 24, 30))
    draw = ImageDraw.Draw(sheet)
    font = load_font(12)
    small = load_font(10)
    draw.text((pad, 8), "Deep BTC frame-id validation: BTC | F-2 | F-1 | F | F+1 | F+2", fill=(245, 245, 245), font=font)

    for r, row in enumerate(rows):
        y = header_h + pad + r * (thumb_h + label_h + pad)
        cells = [("BTC", row["btc_img"], None)] + [(label, row["metrics"][label], row["metrics"][label]["frame"]) for label in ["fm2", "fm1", "f", "fp1", "fp2"]]
        for c, cell in enumerate(cells):
            x = pad + c * (thumb_w + pad)
            if c == 0:
                label, img_bgr, _ = cell
                label_lines = [row["btc_name"], f"mapped F={row['mapped_fid']}", row["classification"]]
                border = (180, 180, 180)
            else:
                label, metric, fid = cell
                img_bgr = row["images"][label]
                label_lines = [
                    f"{label} frame {fid}",
                    f"SSIM {metric['ssim']:.5f}",
                    f"PixErr {metric['pixel_error']:.5f}",
                    f"pHash {metric['phash']} CLIP {metric['clip_similarity']:.4f}",
                    f"{'MAPPED F ' if label == 'f' else ''}{'BEST' if label == row['best_label'] else ''}".strip(),
                ]
                border = (230, 190, 50) if label == "f" else (100, 100, 100)
                if label == row["best_label"]:
                    border = (60, 210, 110)
                if label == "f" and label == row["best_label"]:
                    border = (80, 180, 255)
            thumb = bgr_to_pil_thumb(img_bgr, thumb_w, thumb_h)
            sheet.paste(thumb, (x, y))
            draw.rectangle([x, y, x + thumb_w - 1, y + thumb_h - 1], outline=border, width=3)
            for i, text in enumerate(label_lines[:6]):
                draw.text((x + 2, y + thumb_h + 4 + i * 14), text, fill=(225, 225, 225), font=small)
    sheet.save(output_path, quality=92)

def bgr_to_pil_thumb(img_bgr: np.ndarray, thumb_w: int, thumb_h: int) -> Image.Image:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    image.thumbnail((thumb_w, thumb_h), Image.LANCZOS)
    canvas = Image.new("RGB", (thumb_w, thumb_h), (8, 8, 8))
    canvas.paste(image, ((thumb_w - image.width) // 2, (thumb_h - image.height) // 2))
    return canvas


def load_font(size: int):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


if __name__ == "__main__":
    main()
