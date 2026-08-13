from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.correct_ocr_v2_with_qwen_context import load_qwen, parse_qwen_json
from src.retrieval.query_translation import looks_like_usable_english


SYSTEM_PROMPT = (
    "You are a precise Vietnamese to English translator for visual video retrieval. "
    "Translate the Vietnamese query into concise natural English. "
    "Preserve named entities, visible text, object names, colors, scene details, and OCR text. "
    "Do not add any video id, frame id, timestamp, or answer. Return JSON only."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a clean query_text -> English translation cache.")
    parser.add_argument("--gt", type=Path, default=PROJECT_ROOT / "JsonTest" / "gt_kis.json")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "cache" / "query_translation_vi_en.json")
    parser.add_argument("--metadata-output", type=Path, default=PROJECT_ROOT / "outputs" / "cache" / "query_translation_vi_en_metadata.json")
    parser.add_argument("--model-name", default="Qwen/Qwen3-4B")
    parser.add_argument("--model-cache-dir", type=Path, default=PROJECT_ROOT / ".model_cache" / "qwen3_4b")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--use-4bit", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def load_queries(path: Path, limit: int | None) -> list[dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for idx, item in enumerate(payload.get("queries", []), start=1):
        query = str(item.get("query", "")).strip()
        if not query:
            continue
        rows.append(
            {
                "query_id": str(item.get("query_id", f"query_{idx:03d}")),
                "query": query,
            }
        )
    return rows[:limit] if limit is not None else rows


def clean_translation(raw: str) -> str:
    text = str(raw or "").strip()
    text = re.sub(r"^translation\s*:\s*", "", text, flags=re.I).strip()
    text = text.strip("\"'` ")
    return re.sub(r"\s+", " ", text).strip()


def translate_one(tokenizer: Any, model: Any, query: str, max_new_tokens: int) -> tuple[str, str]:
    payload = {
        "task": "translate_vi_to_en_for_visual_retrieval",
        "query_vi": query,
        "json_schema": {"translation_en": "string"},
        "constraints": [
            "Do not summarize away visual details.",
            "Do not include explanations.",
            "Do not include video/frame/timestamp guesses.",
        ],
    }
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
    try:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            repetition_penalty=1.03,
            pad_token_id=tokenizer.eos_token_id,
        )
    raw = tokenizer.decode(outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True).strip()
    parsed = parse_qwen_json(raw)
    translation = clean_translation(parsed.get("translation_en", ""))
    return translation, raw


def main() -> None:
    args = parse_args()
    gt_path = args.gt if args.gt.is_absolute() else PROJECT_ROOT / args.gt
    output_path = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    metadata_path = args.metadata_output if args.metadata_output.is_absolute() else PROJECT_ROOT / args.metadata_output
    rows = load_queries(gt_path, args.limit)

    cache: dict[str, str] = {}
    if args.resume and output_path.exists():
        cache = json.loads(output_path.read_text(encoding="utf-8"))

    print(f"[translate-cache] queries={len(rows)} output={output_path}")
    tokenizer, model, model_meta = load_qwen(
        model_name=args.model_name,
        cache_dir=args.model_cache_dir if args.model_cache_dir.is_absolute() else PROJECT_ROOT / args.model_cache_dir,
        device=args.device,
        use_4bit=args.use_4bit,
    )

    debug_rows = []
    for idx, row in enumerate(rows, start=1):
        query = row["query"]
        if query in cache and looks_like_usable_english(query, cache[query]):
            print(f"[translate-cache] {idx:03d}/{len(rows):03d} cached {row['query_id']}")
            continue
        translation, raw = translate_one(tokenizer, model, query, args.max_new_tokens)
        usable = looks_like_usable_english(query, translation)
        if usable:
            cache[query] = translation
        print(f"[translate-cache] {idx:03d}/{len(rows):03d} {row['query_id']} usable={usable}: {translation}")
        debug_rows.append(
            {
                "query_id": row["query_id"],
                "query": query,
                "translation": translation,
                "usable": usable,
                "raw": raw,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    metadata_path.write_text(
        json.dumps(
            {
                "source_gt": str(gt_path),
                "output": str(output_path),
                "num_queries": len(rows),
                "num_cached": len(cache),
                "model_meta": model_meta,
                "debug_rows": debug_rows,
                "leakage_guard": "cache maps query text to English translation only; no GT video/frame/rank data is stored",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[translate-cache] wrote {output_path}")


if __name__ == "__main__":
    main()
