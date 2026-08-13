from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.preprocessing.model_assets import (
    DEFAULT_CACHE_ROOT,
    DEFAULT_FASTER_WHISPER_MODEL,
    ensure_cache_dirs,
    ensure_faster_whisper_model,
    ensure_open_clip_weights,
    ensure_paddleocr_models,
    ensure_vietocr_weights,
    ensure_yoloe_weights,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Warm up Kaggle model/checkpoint caches.")
    parser.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT))
    parser.add_argument("--yoloe", action="store_true", default=True)
    parser.add_argument("--vietocr", action="store_true", default=True)
    parser.add_argument("--open-clip", action="store_true", default=True)
    parser.add_argument("--paddle", action="store_true", default=True)
    parser.add_argument("--whisper", action="store_true", default=True)
    parser.add_argument("--whisper-model", default=DEFAULT_FASTER_WHISPER_MODEL)
    args = parser.parse_args()

    ensure_cache_dirs()
    report: dict[str, str] = {}
    failures: dict[str, str] = {}

    if args.yoloe:
        try:
            report["yoloe_main"] = str(ensure_yoloe_weights("yoloe-26s-seg.pt", args.cache_root))
            report["yoloe_prompt_free"] = str(ensure_yoloe_weights("yoloe-26s-seg-pf.pt", args.cache_root))
        except Exception as exc:
            failures["yoloe"] = repr(exc)

    if args.vietocr:
        try:
            report["vietocr_vgg_transformer"] = str(ensure_vietocr_weights())
        except Exception as exc:
            failures["vietocr"] = repr(exc)

    if args.open_clip:
        try:
            report["open_clip_weights"] = str(ensure_open_clip_weights(cache_root=args.cache_root))
        except Exception as exc:
            failures["open_clip"] = repr(exc)

    if args.paddle:
        try:
            ensure_paddleocr_models(device="cpu")
            report["paddleocr"] = "ready"
        except Exception as exc:
            failures["paddleocr"] = repr(exc)

    if args.whisper:
        try:
            report["faster_whisper_root"] = str(
                ensure_faster_whisper_model(args.whisper_model, download_root=Path(args.cache_root) / "faster_whisper")
            )
        except Exception as exc:
            failures["faster_whisper"] = repr(exc)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if failures:
        print(json.dumps({"failures": failures}, indent=2, ensure_ascii=False))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
