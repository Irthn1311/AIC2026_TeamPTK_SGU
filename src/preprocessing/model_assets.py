from __future__ import annotations

import os
import time
import urllib.request
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_ROOT = PROJECT_ROOT / ".model_cache"
DEFAULT_OCR_CACHE_ROOT = PROJECT_ROOT / ".ocr_cache"
YOLOE_RELEASE_BASE_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0"
VIETOCR_WEIGHTS_URLS = [
    "https://github.com/pjh2512/vietocr/releases/download/v0.1/vgg_transformer.pth",
    "https://vocr.vn/data/vietocr/vgg_transformer.pth",
]
OPEN_CLIP_REPO_ID = "timm/vit_base_patch32_clip_224.openai"
OPEN_CLIP_FILENAME = "open_clip_model.safetensors"
DEFAULT_FASTER_WHISPER_MODEL = "large-v3-turbo"


def _resolve(path: str | Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def _download_with_progress(url: str, destination: Path) -> Path:
    destination = destination.expanduser().resolve(strict=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(destination.suffix + ".tmp")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as response, tmp_path.open("wb") as out_file:
        total = int(response.headers.get("Content-Length") or 0)
        downloaded = 0
        started = time.time()
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded * 100 / total
                elapsed = max(time.time() - started, 1e-6)
                speed = downloaded / 1024 / 1024 / elapsed
                print(
                    f"\rDownloading {destination.name}: {pct:5.1f}% "
                    f"({downloaded / 1024 / 1024:.1f}/{total / 1024 / 1024:.1f} MB, {speed:.1f} MB/s)",
                    end="",
                    flush=True,
                )
        if total:
            print()
    tmp_path.replace(destination)
    return destination


def ensure_vietocr_weights(destination: str | Path | None = None) -> Path:
    target = _resolve(destination or (DEFAULT_OCR_CACHE_ROOT / "temp" / "vgg_transformer.pth"))
    if target.is_file() and target.stat().st_size > 0:
        return target

    # 1. Look for pre-downloaded weights in /kaggle/input if available
    for p in Path("/kaggle/input").rglob("vgg_transformer.pth") if Path("/kaggle").exists() else []:
        if p.is_file() and p.stat().st_size > 10_000_000:
            print(f"✅ Found VietOCR weights in Kaggle input: {p}")
            target.parent.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy2(p, target)
            return target

    # 2. Try candidate URLs with timeout and progress
    for url in VIETOCR_WEIGHTS_URLS:
        try:
            print(f"Downloading VietOCR weights from: {url}")
            return _download_with_progress(url, target)
        except Exception as exc:
            print(f"⚠️ Failed download from {url}: {exc}")

    raise RuntimeError("Failed to download VietOCR weights from all candidate URLs.")


def ensure_yoloe_weights(model_name: str, cache_root: str | Path | None = None) -> Path:
    root = _resolve(cache_root or DEFAULT_CACHE_ROOT)
    target = root / "yoloe" / Path(model_name).name
    if target.is_file() and target.stat().st_size > 0:
        return target
    url = f"{YOLOE_RELEASE_BASE_URL}/{Path(model_name).name}"
    return _download_with_progress(url, target)


def ensure_open_clip_weights(
    repo_id: str = OPEN_CLIP_REPO_ID,
    filename: str = OPEN_CLIP_FILENAME,
    cache_root: str | Path | None = None,
) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError("huggingface_hub is required to download OpenCLIP weights") from exc

    cache_dir = _resolve(cache_root or (DEFAULT_CACHE_ROOT / "huggingface"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=str(cache_dir))
    return Path(path).expanduser().resolve(strict=False)


def ensure_faster_whisper_model(
    model_size: str = DEFAULT_FASTER_WHISPER_MODEL,
    download_root: str | Path | None = None,
    device: str = "cpu",
    compute_type: str = "int8",
) -> Path:
    from faster_whisper import WhisperModel

    root = _resolve(download_root or (DEFAULT_CACHE_ROOT / "faster_whisper"))
    root.mkdir(parents=True, exist_ok=True)
    WhisperModel(
        model_size,
        device=device,
        compute_type=compute_type,
        download_root=str(root),
    )
    return root


def ensure_paddleocr_models(
    device: str = "cpu",
    use_doc_orientation_classify: bool = False,
    use_doc_unwarping: bool = False,
    use_textline_orientation: bool = False,
) -> bool:
    from paddleocr import PaddleOCR

    kwargs: dict[str, object] = {
        "use_doc_orientation_classify": use_doc_orientation_classify,
        "use_doc_unwarping": use_doc_unwarping,
        "use_textline_orientation": use_textline_orientation,
        "device": device,
        "show_log": False,
    }
    if str(device).lower() == "cpu":
        kwargs["enable_mkldnn"] = False
    while True:
        try:
            PaddleOCR(**kwargs)
            return True
        except (TypeError, ValueError) as exc:
            message = str(exc)
            removed = False
            for key in ("show_log", "enable_mkldnn"):
                if key in kwargs and (key in message or "Unknown argument" in message):
                    kwargs.pop(key, None)
                    removed = True
                    break
            if not removed:
                raise


def ensure_cache_dirs() -> dict[str, Path]:
    root = _resolve(DEFAULT_CACHE_ROOT)
    ocr_root = _resolve(DEFAULT_OCR_CACHE_ROOT)
    dirs = {
        "cache_root": root,
        "huggingface_cache": root / "huggingface",
        "torch_cache": root / "torch",
        "ocr_cache_root": ocr_root,
        "ocr_temp": ocr_root / "temp",
        "model_cache_yoloe": root / "yoloe",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(dirs["huggingface_cache"]))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(dirs["huggingface_cache"] / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(dirs["huggingface_cache"] / "transformers"))
    os.environ.setdefault("TORCH_HOME", str(dirs["torch_cache"]))
    os.environ.setdefault("PIP_CACHE_DIR", str(root / "pip"))
    os.environ.setdefault("TMP", str(root / "tmp"))
    os.environ.setdefault("TEMP", str(root / "tmp"))
    os.environ.setdefault("TMPDIR", str(root / "tmp"))
    return dirs
