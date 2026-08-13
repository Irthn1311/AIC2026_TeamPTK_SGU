from __future__ import annotations

import hashlib
import os
import time
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_ROOT = PROJECT_ROOT / ".model_cache"
DEFAULT_MODEL_NAME = "yoloe-26s-seg.pt"
DEFAULT_PROMPT_FREE_MODEL_NAME = "yoloe-26s-seg-pf.pt"
YOLOE_RELEASE_BASE_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0"


def configure_model_cache(cache_root: str | Path = DEFAULT_CACHE_ROOT) -> Path:
    cache_root = Path(cache_root).resolve()
    cache_root.mkdir(parents=True, exist_ok=True)

    cache_env = {
        "HF_HOME": cache_root / "huggingface",
        "HUGGINGFACE_HUB_CACHE": cache_root / "huggingface" / "hub",
        "TRANSFORMERS_CACHE": cache_root / "huggingface" / "transformers",
        "TORCH_HOME": cache_root / "torch",
        "XDG_CACHE_HOME": cache_root / "xdg",
        "YOLO_CONFIG_DIR": cache_root / "ultralytics_config",
        "MPLCONFIGDIR": cache_root / "matplotlib",
        "PIP_CACHE_DIR": cache_root / "pip",
        "TMP": cache_root / "tmp",
        "TEMP": cache_root / "tmp",
    }
    for key, value in cache_env.items():
        os.environ[key] = str(value)
        value.mkdir(parents=True, exist_ok=True)
    return cache_root


configure_model_cache()

import cv2  # noqa: E402
import torch  # noqa: E402
from ultralytics import YOLOE  # noqa: E402


def _download_with_progress(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(destination.suffix + ".tmp")

    with urllib.request.urlopen(url) as response, tmp_path.open("wb") as out_file:
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
                mbps = downloaded / 1024 / 1024 / elapsed
                print(
                    f"\rDownloading {destination.name}: {pct:5.1f}% "
                    f"({downloaded / 1024 / 1024:.1f}/{total / 1024 / 1024:.1f} MB, {mbps:.1f} MB/s)",
                    end="",
                    flush=True,
                )
        if total:
            print()
    tmp_path.replace(destination)


def _resolve_weights(model_name_or_path: str | Path, cache_root: Path) -> Path:
    raw = Path(model_name_or_path)
    if raw.exists():
        return raw.resolve()

    model_name = raw.name
    weights_path = cache_root / "yoloe" / model_name
    if weights_path.exists():
        return weights_path.resolve()

    if model_name != str(model_name_or_path) and raw.parent != Path("."):
        weights_path = raw

    url = f"{YOLOE_RELEASE_BASE_URL}/{model_name}"
    print(f"YOLOE weights not found locally, downloading official Ultralytics weights to {weights_path}")
    _download_with_progress(url, weights_path)
    return weights_path.resolve()


@contextmanager
def _working_directory(path: Path):
    previous = Path.cwd()
    path.mkdir(parents=True, exist_ok=True)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _normalize_classes(classes: Iterable[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for item in classes:
        label = str(item).strip()
        if not label:
            continue
        key = label.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(label)
    if not normalized:
        raise ValueError("YOLOE classes list is empty.")
    return normalized


def _names_lookup(names: Any, cls_idx: int, classes: list[str]) -> str:
    if isinstance(names, dict) and cls_idx in names:
        return str(names[cls_idx])
    if isinstance(names, (list, tuple)) and 0 <= cls_idx < len(names):
        return str(names[cls_idx])
    if 0 <= cls_idx < len(classes):
        return classes[cls_idx]
    return str(cls_idx)


def _label_color(label: str) -> tuple[int, int, int]:
    digest = hashlib.md5(label.encode("utf-8")).digest()
    return (64 + digest[0] % 160, 64 + digest[1] % 160, 64 + digest[2] % 160)


class YOLOEDetector:
    def __init__(
        self,
        model_name: str | Path = DEFAULT_MODEL_NAME,
        prompt_free_model_name: str | Path = DEFAULT_PROMPT_FREE_MODEL_NAME,
        cache_dir: str | Path = DEFAULT_CACHE_ROOT,
        conf: float = 0.25,
        iou: float = 0.70,
        imgsz: int = 640,
        device: str | None = None,
        verbose: bool = False,
    ) -> None:
        self.cache_root = configure_model_cache(cache_dir)
        self.model_name = model_name
        self.prompt_free_model_name = prompt_free_model_name
        self.weights_path: Path | None = None
        self.prompt_free_weights_path: Path | None = None
        self.conf = float(conf)
        self.iou = float(iou)
        self.imgsz = int(imgsz)
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.ultralytics_device = "0" if self.device.startswith("cuda") else "cpu"
        self.verbose = verbose
        self.assets_dir = self.cache_root / "yoloe"
        self.model: YOLOE | None = None
        self.prompt_free_model: YOLOE | None = None
        self._classes: list[str] | None = None

    def _load_text_model(self) -> YOLOE:
        if self.model is None:
            self.weights_path = _resolve_weights(self.model_name, self.cache_root)
            self.model = YOLOE(str(self.weights_path))
        return self.model

    def _load_prompt_free_model(self) -> YOLOE:
        if self.prompt_free_model is None:
            self.prompt_free_weights_path = _resolve_weights(self.prompt_free_model_name, self.cache_root)
            self.prompt_free_model = YOLOE(str(self.prompt_free_weights_path))
            head = getattr(self.prompt_free_model.model, "model", [None])[-1]
            if not hasattr(head, "lrpc"):
                raise RuntimeError(
                    "Loaded YOLOE model does not expose the prompt-free LRPC head. "
                    f"Use an official *-pf.pt checkpoint, got: {self.prompt_free_weights_path}"
                )
        return self.prompt_free_model

    def set_classes(self, classes: Iterable[str]) -> list[str]:
        normalized = _normalize_classes(classes)
        if self._classes != normalized:
            model = self._load_text_model()
            with _working_directory(self.assets_dir):
                model.set_classes(normalized)
            self._classes = normalized
        return normalized

    @property
    def runtime_device(self) -> str:
        return self.device

    def detect(
        self,
        image_path: str | Path,
        classes: Iterable[str] | None = None,
        mode: str = "text",
    ) -> list[dict[str, Any]]:
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        if mode not in {"text", "prompt_free"}:
            raise ValueError(f"Unsupported YOLOE detection mode: {mode}")

        if mode == "text":
            if classes is None:
                raise ValueError("Text prompt YOLOE mode requires classes.")
            active_classes = self.set_classes(classes)
            model = self._load_text_model()
        else:
            active_classes = None
            model = self._load_prompt_free_model()

        with torch.inference_mode():
            results = model.predict(
                source=str(image_path),
                conf=self.conf,
                iou=self.iou,
                imgsz=self.imgsz,
                device=self.ultralytics_device,
                verbose=self.verbose,
            )

        if not results:
            return []

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        detections: list[dict[str, Any]] = []
        names = getattr(result, "names", None) or getattr(model, "names", None)
        xyxy = boxes.xyxy.detach().cpu().tolist()
        confs = boxes.conf.detach().cpu().tolist()
        clss = boxes.cls.detach().cpu().tolist()

        for coords, conf, cls_idx_float in zip(xyxy, confs, clss):
            cls_idx = int(cls_idx_float)
            detections.append(
                {
                    "label": _names_lookup(names, cls_idx, active_classes or []),
                    "confidence": round(float(conf), 4),
                    "bbox": [int(round(float(v))) for v in coords[:4]],
                }
            )
        return detections


def save_detection_visualization(
    image_path: str | Path,
    detections: list[dict[str, Any]],
    output_path: str | Path,
) -> Path:
    image_path = Path(image_path)
    output_path = Path(output_path)
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Cannot read image for visualization: {image_path}")

    height, width = image.shape[:2]
    thickness = max(2, int(round(min(width, height) / 360)))
    font_scale = max(0.45, min(width, height) / 900)

    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        x1 = max(0, min(width - 1, x1))
        x2 = max(0, min(width - 1, x2))
        y1 = max(0, min(height - 1, y1))
        y2 = max(0, min(height - 1, y2))
        label = str(det["label"])
        confidence = float(det["confidence"])
        color = _label_color(label)

        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        caption = f"{label} {confidence:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        y_text = max(text_h + baseline + 4, y1)
        cv2.rectangle(
            image,
            (x1, y_text - text_h - baseline - 6),
            (min(width - 1, x1 + text_w + 8), y_text + baseline),
            color,
            -1,
        )
        cv2.putText(
            image,
            caption,
            (x1 + 4, y_text - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(output_path), image)
    if not ok:
        raise IOError(f"Failed to write visualization: {output_path}")
    return output_path
