from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.ocr_temporal_merger import remove_vietnamese_accents


SYSTEM_PROMPT = """Bạn là bộ sửa lỗi OCR tiếng Việt cho video thời sự.

Bạn chỉ được sửa dựa trên bằng chứng có trong OCR hiện tại và OCR các keyframe lân cận.

Luật bắt buộc:
- Không thêm thông tin mới không có trong bằng chứng.
- Không diễn giải, không tóm tắt, không viết lại văn phong.
- Ưu tiên sửa lỗi dấu, lỗi ký tự OCR, tên riêng, địa danh, thuật ngữ, từ địa phương khi ngữ cảnh lân cận ủng hộ.
- Giữ nguyên số, ngày tháng, giờ, tỉ số, số điện thoại nếu không có bằng chứng rõ ràng.
- Nếu text là logo/giờ/số rác/không có giá trị tìm kiếm thì action = "drop_noise".
- Nếu không chắc thì action = "needs_review" và giữ corrected_text gần với current_text.
- Chỉ trả về JSON hợp lệ, không giải thích ngoài JSON."""


def setup_local_model_cache(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = PROJECT_ROOT / ".model_cache" / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["TRANSFORMERS_CACHE"] = str(cache_dir)
    os.environ["TORCH_HOME"] = str(PROJECT_ROOT / ".model_cache" / "torch")
    os.environ["TMP"] = str(tmp)
    os.environ["TEMP"] = str(tmp)
    os.environ["TMPDIR"] = str(tmp)


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_text(text: str) -> str:
    return " ".join(str(text or "").strip().split())


def normalize_match(text: str) -> str:
    return remove_vietnamese_accents(normalize_text(text).lower())


def tokens(text: str) -> list[str]:
    return re.findall(r"[\w%:/.-]+", normalize_match(text), flags=re.UNICODE)


def number_tokens(text: str) -> list[str]:
    return re.findall(r"\d+(?:[:./-]\d+)*", normalize_text(text))


def token_overlap(a: str, b: str) -> float:
    ta = {t for t in tokens(a) if len(t) > 1}
    tb = {t for t in tokens(b) if len(t) > 1}
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(1, min(len(ta), len(tb)))


COMMON_VIETNAMESE_FIXES: list[tuple[str, str]] = [
    (r"\bNGHỈ\s+LÊ\s+QUỐC\s+KHÁNH\b", "NGHỈ LỄ QUỐC KHÁNH"),
    (r"\bQUẢ\s+BƯỜI\b", "QUẢ BƯỞI"),
    (r"\bLẦN\s+ĐẦU\s+GIÁM\s+LẠI\s+SUẤT\b", "LẦN ĐẦU GIẢM LÃI SUẤT"),
    (r"\bGIÁM\s+LẠI\s+SUẤT\b", "GIẢM LÃI SUẤT"),
    (r"\bLẨN\s+ĐẨU\b", "LẦN ĐẦU"),
    (r"\bGIẢM\s+LÃI\b", "GIẢM LÃI"),
    (r"\bSÚI\s+CẢO\b", "SỦI CẢO"),
]


def apply_common_vietnamese_fixes(text: str) -> tuple[str, list[str]]:
    corrected = normalize_text(text)
    reasons = []
    for pattern, replacement in COMMON_VIETNAMESE_FIXES:
        new_text = re.sub(pattern, replacement, corrected, flags=re.IGNORECASE)
        if new_text != corrected:
            reasons.append(pattern)
            corrected = new_text
    return normalize_text(corrected), reasons


def is_noise_text(text: str) -> bool:
    text = normalize_text(text)
    if not text:
        return True
    if re.fullmatch(r"\d{1,2}:\d{2}(?::\d{2})?", text):
        return True
    if text.upper() in {"HD", "H", "HTV", "HTV9", "VTV", "VTV1", "VTV3"}:
        return True
    if text.lower() in {"giây", "gia"}:
        return True
    nums = re.findall(r"\d+", text)
    words = re.findall(r"[A-Za-zÀ-ỹĐđ]+", text)
    if len(nums) >= 8 and len(words) <= 2:
        return True
    if len(nums) >= 16 and len(nums) >= len(words) * 1.8:
        return True
    return False


def semantic_text_from_record(record: dict[str, Any]) -> str:
    by_region: dict[str, list[str]] = {"headline": [], "ticker": [], "scene_text": []}
    for det in record.get("detections", []) or []:
        region = str(det.get("region_type", ""))
        text = normalize_text(det.get("text", ""))
        conf = float(det.get("confidence", 0.0) or 0.0)
        if not text or conf < 0.35:
            continue
        if region in {"logo_channel", "clock_time"}:
            continue
        if is_noise_text(text):
            continue
        if region in by_region:
            by_region[region].append(text)
    for region in ["headline", "ticker", "scene_text"]:
        text = normalize_text(" ".join(by_region[region]))
        if text:
            return text[:700]
    return ""


def context_text(record: dict[str, Any]) -> str:
    semantic = semantic_text_from_record(record)
    return semantic or normalize_text(record.get("combined_text", ""))


def should_call_qwen(record: dict[str, Any]) -> bool:
    text = context_text(record)
    if is_noise_text(text):
        return False
    if len(text) > 420:
        return False
    if len(text) < 8:
        return False
    letters = re.findall(r"[A-Za-zÀ-ỹĐđ]", text)
    if len(letters) < 4:
        return False
    fixed, reasons = apply_common_vietnamese_fixes(text)
    if reasons and fixed != text:
        return False
    return True


def build_prompt_payload(records: list[dict[str, Any]], idx: int, context_keyframes: int, time_window: float) -> dict[str, Any]:
    current = records[idx]
    current_ts = float(current.get("timestamp_seconds", 0.0) or 0.0)
    left = max(0, idx - context_keyframes)
    right = min(len(records), idx + context_keyframes + 1)
    nearby = []
    for j in range(left, right):
        if j == idx:
            continue
        item = records[j]
        ts = float(item.get("timestamp_seconds", 0.0) or 0.0)
        if abs(ts - current_ts) <= time_window or context_keyframes > 0:
            txt = context_text(item)
            if txt and not is_noise_text(txt):
                nearby.append({
                    "offset": j - idx,
                    "timestamp_seconds": round(ts, 3),
                    "text": txt[:260],
                })

    dets = []
    for det in current.get("detections", []) or []:
        text = normalize_text(det.get("text", ""))
        if not text:
            continue
        region = str(det.get("region_type", ""))
        if region in {"logo_channel", "clock_time"} or is_noise_text(text):
            continue
        dets.append({
            "text": text,
            "region_type": region,
            "confidence": det.get("confidence", 0.0),
            "easyocr_text": det.get("easyocr_text", ""),
            "vietocr_text": det.get("vietocr_text", ""),
        })

    return {
        "video_id": current.get("video_id", ""),
        "keyframe_name": current.get("keyframe_name", ""),
        "timestamp_seconds": round(current_ts, 3),
        "current_text": context_text(current)[:420],
        "combined_text_raw": normalize_text(current.get("combined_text", "")),
        "nearby_keyframes": nearby,
        "detections_current_frame": dets[:12],
        "task": "Sửa OCR tiếng Việt bằng ngữ cảnh keyframe trước/sau. Trả JSON duy nhất.",
        "json_schema": {
            "corrected_text": "string",
            "action": "keep|correct|drop_noise|needs_review",
            "confidence": "number 0..1",
            "reason": "short Vietnamese explanation",
            "evidence": ["current_text", "nearby_keyframes", "detections_current_frame"],
            "needs_human_review": "boolean",
        },
    }


def load_qwen(model_name: str, cache_dir: Path, device: str, use_4bit: bool) -> tuple[Any, Any, dict[str, Any]]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    setup_local_model_cache(cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=str(cache_dir), local_files_only=True)
    has_cuda = torch.cuda.is_available() and device != "cpu"
    meta = {"model_name": model_name, "cache_dir": str(cache_dir), "device_requested": device}

    if has_cuda and use_4bit:
        try:
            from transformers import BitsAndBytesConfig

            qcfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=str(cache_dir),
                local_files_only=True,
                quantization_config=qcfg,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
            meta.update({"device_used": "cuda", "quantization": "4bit_nf4"})
            model.eval()
            return tokenizer, model, meta
        except Exception as exc:
            meta["four_bit_error"] = repr(exc)

    if has_cuda:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=str(cache_dir),
            local_files_only=True,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        meta.update({"device_used": "cuda", "quantization": "float16"})
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=str(cache_dir),
            local_files_only=True,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        meta.update({"device_used": "cpu", "quantization": "float32"})
    model.eval()
    return tokenizer, model, meta


def parse_qwen_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()

    def iter_json_candidates(raw_text: str) -> list[str]:
        candidates = []
        candidates.extend(re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, flags=re.S | re.I))
        starts = [idx for idx, char in enumerate(raw_text) if char == "{"]
        for start in starts:
            depth = 0
            in_string = False
            escape = False
            for pos in range(start, len(raw_text)):
                char = raw_text[pos]
                if escape:
                    escape = False
                    continue
                if char == "\\":
                    escape = True
                    continue
                if char == '"':
                    in_string = not in_string
                if in_string:
                    continue
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        candidates.append(raw_text[start : pos + 1])
                        break
        return candidates

    for candidate in iter_json_candidates(text):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and "action" in parsed and "corrected_text" in parsed:
            return parsed

    match = re.search(r"\{.*\}", text, flags=re.S)
    payload = match.group(0) if match else text
    try:
        parsed = json.loads(payload)
        if isinstance(parsed, dict):
            return parsed
        return {}
    except json.JSONDecodeError:
        def grab_string(name: str) -> str:
            matches = re.findall(rf'"{name}"\s*:\s*"([^"\n\r]*)', payload)
            value = matches[-1].strip() if matches else ""
            return "" if value.lower() in {"string", "keep|correct|drop_noise|needs_review", "keep|correct|needs_review"} else value

        def grab_number(name: str) -> float:
            m = re.search(rf'"{name}"\s*:\s*([0-9]*\.?[0-9]+)', payload)
            return float(m.group(1)) if m else 0.0

        corrected = grab_string("corrected_text")
        action = grab_string("action") or "needs_review"
        reason = grab_string("reason") or "recovered_from_malformed_json"
        if not corrected:
            lines = [normalize_text(line) for line in payload.splitlines() if normalize_text(line)]
            for line in lines:
                if not line.startswith("{") and not line.startswith('"') and ":" not in line[:20]:
                    corrected = line.strip('", ')
                    break
        return {
            "corrected_text": corrected,
            "action": action,
            "confidence": grab_number("confidence"),
            "reason": reason,
            "evidence": ["malformed_json_recovered"],
            "needs_human_review": True,
            "_malformed_json": True,
        }


def validate_correction(source: str, proposed: str, action: str) -> tuple[bool, str]:
    source = normalize_text(source)
    proposed = normalize_text(proposed)
    if action == "drop_noise":
        return True, "drop_noise"
    if not proposed:
        return False, "empty_corrected_text"
    src_words = source.split()
    prop_words = proposed.split()
    if len(prop_words) < max(1, int(len(src_words) * 0.35)) or len(prop_words) > int(len(src_words) * 2.2) + 6:
        return False, "length_mismatch"
    src_nums = number_tokens(source)
    prop_nums = number_tokens(proposed)
    missing_nums = [n for n in src_nums if n not in prop_nums]
    if missing_nums and len(src_nums) <= 8:
        return False, f"missing_numbers:{missing_nums[:4]}"
    if len(src_words) >= 4 and token_overlap(source, proposed) < 0.30:
        return False, "low_token_overlap"
    return True, "ok"


def correct_one(tokenizer: Any, model: Any, payload: dict[str, Any], max_new_tokens: int) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
    try:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
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
    parsed["_raw_response"] = raw
    return parsed


def process_video(
    video_path: Path,
    output_dir: Path,
    tokenizer: Any,
    model: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    records = read_jsonl(video_path)
    records.sort(key=lambda r: (float(r.get("timestamp_seconds", 0.0) or 0.0), int(r.get("keyframe_v2_idx", 0) or 0)))
    video_id = video_path.stem
    out_video_path = output_dir / "per_video" / f"{video_id}.jsonl"
    out_debug_path = output_dir / "per_video" / f"{video_id}_temporal_qwen_debug.csv"
    done_by_key = {}
    if args.resume and out_video_path.exists():
        for row in read_jsonl(out_video_path):
            done_by_key[str(row.get("keyframe_name", ""))] = row

    out_rows: list[dict[str, Any]] = []
    debug_rows: list[dict[str, Any]] = []
    qwen_invoked = qwen_changed = qwen_dropped = qwen_failed = skipped = rejected = 0
    local_rule_changed = 0

    for idx, record in enumerate(tqdm(records, desc=f"Qwen OCR context {video_id}")):
        key = str(record.get("keyframe_name", ""))
        if key in done_by_key:
            corrected = done_by_key[key]
            out_rows.append(corrected)
        else:
            corrected = dict(record)
            source_text = context_text(record)
            corrected["semantic_text"] = source_text
            corrected["qwen_context_prev"] = context_text(records[idx - 1]) if idx > 0 else ""
            corrected["qwen_context_next"] = context_text(records[idx + 1]) if idx + 1 < len(records) else ""
            corrected["qwen_status"] = "skipped"
            corrected["qwen_action"] = "keep"
            corrected["qwen_corrected_text"] = source_text
            corrected["search_text"] = source_text
            corrected["qwen_confidence"] = 0.0
            corrected["qwen_reason"] = "not_selected"
            corrected["needs_human_review"] = False

            locally_fixed, local_reasons = apply_common_vietnamese_fixes(source_text)
            if local_reasons and locally_fixed != source_text:
                local_rule_changed += 1
                corrected["qwen_status"] = "local_rule"
                corrected["qwen_action"] = "correct"
                corrected["qwen_corrected_text"] = locally_fixed
                corrected["search_text"] = locally_fixed
                corrected["qwen_confidence"] = 0.88
                corrected["qwen_reason"] = "local_context_rule:" + ";".join(local_reasons)
                corrected["needs_human_review"] = False
            elif not should_call_qwen(record):
                skipped += 1
                if is_noise_text(source_text):
                    corrected["qwen_action"] = "drop_noise"
                    corrected["search_text"] = ""
                    corrected["qwen_reason"] = "local_noise_filter"
                    qwen_dropped += 1
            else:
                payload = build_prompt_payload(records, idx, args.context_keyframes, args.time_window)
                try:
                    qwen_invoked += 1
                    result = correct_one(tokenizer, model, payload, args.max_new_tokens)
                    action = str(result.get("action", "needs_review")).strip()
                    proposed = normalize_text(result.get("corrected_text", source_text))
                    ok, validation_reason = validate_correction(source_text, proposed, action)
                    corrected["qwen_raw_response"] = result.get("_raw_response", "")
                    corrected["qwen_action"] = action if action in {"keep", "correct", "drop_noise", "needs_review"} else "needs_review"
                    corrected["qwen_confidence"] = float(result.get("confidence", 0.0) or 0.0)
                    corrected["qwen_reason"] = normalize_text(result.get("reason", ""))
                    corrected["qwen_evidence"] = result.get("evidence", [])
                    corrected["needs_human_review"] = bool(result.get("needs_human_review", False))
                    if ok:
                        corrected["qwen_status"] = "ok"
                        corrected["qwen_corrected_text"] = proposed
                        corrected["search_text"] = "" if corrected["qwen_action"] == "drop_noise" else proposed
                        qwen_changed += int(proposed != source_text and corrected["qwen_action"] != "drop_noise")
                        qwen_dropped += int(corrected["qwen_action"] == "drop_noise")
                    else:
                        rejected += 1
                        corrected["qwen_status"] = "rejected"
                        corrected["qwen_validation_reason"] = validation_reason
                        corrected["qwen_corrected_text"] = source_text
                        corrected["search_text"] = source_text
                        corrected["needs_human_review"] = True
                except Exception as exc:
                    qwen_failed += 1
                    corrected["qwen_status"] = "failed"
                    corrected["qwen_error"] = repr(exc)
                    corrected["qwen_corrected_text"] = source_text
                    corrected["search_text"] = source_text
                    corrected["needs_human_review"] = True
            out_rows.append(corrected)

        debug_rows.append({
            "video_id": corrected.get("video_id", video_id),
            "keyframe_name": corrected.get("keyframe_name", ""),
            "timestamp_seconds": corrected.get("timestamp_seconds", 0.0),
            "frame_idx": corrected.get("frame_idx", 0),
            "combined_text": corrected.get("combined_text", ""),
            "semantic_text": corrected.get("semantic_text", ""),
            "prev_context": corrected.get("qwen_context_prev", ""),
            "next_context": corrected.get("qwen_context_next", ""),
            "qwen_corrected_text": corrected.get("qwen_corrected_text", ""),
            "search_text": corrected.get("search_text", ""),
            "qwen_action": corrected.get("qwen_action", ""),
            "qwen_status": corrected.get("qwen_status", ""),
            "qwen_confidence": corrected.get("qwen_confidence", ""),
            "needs_human_review": corrected.get("needs_human_review", ""),
            "qwen_reason": corrected.get("qwen_reason", ""),
        })

        if args.save_every > 0 and len(out_rows) % args.save_every == 0:
            write_jsonl(out_video_path, out_rows)
            pd.DataFrame(debug_rows).to_csv(out_debug_path, index=False, encoding="utf-8-sig")

    write_jsonl(out_video_path, out_rows)
    pd.DataFrame(debug_rows).to_csv(out_debug_path, index=False, encoding="utf-8-sig")
    return {
        "video_id": video_id,
        "rows": len(out_rows),
        "qwen_invoked": qwen_invoked,
        "qwen_changed": qwen_changed,
        "qwen_dropped": qwen_dropped,
        "qwen_failed": qwen_failed,
        "qwen_rejected": rejected,
        "local_rule_changed": local_rule_changed,
        "skipped": skipped,
        "jsonl": str(out_video_path),
        "temporal_debug_csv": str(out_debug_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Contextual Qwen correction for OCR V2 selected-keyframe outputs.")
    parser.add_argument("--input-dir", default="outputs/ocr_v2_selected_keyframes")
    parser.add_argument("--output-dir", default="outputs/ocr_v2_selected_keyframes_qwen")
    parser.add_argument("--video-id", action="append", default=[])
    parser.add_argument("--model-name", default="Qwen/Qwen3-4B")
    parser.add_argument("--model-cache-dir", default=".model_cache/qwen3_4b")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--context-keyframes", type=int, default=2)
    parser.add_argument("--time-window", type=float, default=6.0)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--max-videos", type=int, default=0)
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-every", type=int, default=5)
    args = parser.parse_args()

    started = time.time()
    input_dir = resolve_path(args.input_dir)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "per_video").mkdir(parents=True, exist_ok=True)

    video_paths = sorted((input_dir / "per_video").glob("L21_V*.jsonl"))
    requested = {v.strip() for v in args.video_id if v.strip()}
    if requested:
        video_paths = [p for p in video_paths if p.stem in requested]
    if args.max_videos and args.max_videos > 0:
        video_paths = video_paths[: args.max_videos]
    if not video_paths:
        raise FileNotFoundError(f"No input per-video OCR JSONL found in {input_dir / 'per_video'}")

    tokenizer, model, model_meta = load_qwen(
        model_name=args.model_name,
        cache_dir=resolve_path(args.model_cache_dir),
        device=args.device,
        use_4bit=not args.no_4bit,
    )

    summaries = []
    for path in video_paths:
        if args.max_items and args.max_items > 0:
            original = read_jsonl(path)
            tmp_dir = output_dir / "_tmp_limited"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            tmp_path = tmp_dir / path.name
            write_jsonl(tmp_path, original[: args.max_items])
            path_to_process = tmp_path
        else:
            path_to_process = path
        summaries.append(process_video(path_to_process, output_dir, tokenizer, model, args))

    all_rows = []
    for summary in summaries:
        all_rows.extend(read_jsonl(Path(summary["jsonl"])))
    df = pd.DataFrame(all_rows)
    df.to_json(output_dir / "l21_keyframe_ocr_qwen.jsonl", orient="records", lines=True, force_ascii=False)
    export_cols = [
        "video_id",
        "keyframe_name",
        "keyframe_path",
        "keyframe_v2_idx",
        "global_id",
        "frame_idx",
        "timestamp_seconds",
        "shot_id",
        "combined_text",
        "semantic_text",
        "qwen_corrected_text",
        "search_text",
        "qwen_action",
        "qwen_status",
        "qwen_confidence",
        "qwen_reason",
        "needs_human_review",
        "num_text_boxes",
        "mean_confidence",
    ]
    existing_cols = [c for c in export_cols if c in df.columns]
    df[existing_cols].to_csv(output_dir / "l21_keyframe_ocr_qwen.csv", index=False, encoding="utf-8-sig")

    debug_frames = []
    for summary in summaries:
        debug_path = Path(summary["temporal_debug_csv"])
        if debug_path.exists():
            debug_frames.append(pd.read_csv(debug_path))
    if debug_frames:
        pd.concat(debug_frames, ignore_index=True).to_csv(output_dir / "temporal_qwen_debug.csv", index=False, encoding="utf-8-sig")

    meta = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "elapsed_seconds": round(time.time() - started, 2),
        "model": model_meta,
        "context_keyframes": args.context_keyframes,
        "time_window": args.time_window,
        "videos": summaries,
        "totals": {
            "rows": int(sum(s["rows"] for s in summaries)),
            "qwen_invoked": int(sum(s["qwen_invoked"] for s in summaries)),
            "qwen_changed": int(sum(s["qwen_changed"] for s in summaries)),
            "qwen_dropped": int(sum(s["qwen_dropped"] for s in summaries)),
            "qwen_failed": int(sum(s["qwen_failed"] for s in summaries)),
            "qwen_rejected": int(sum(s["qwen_rejected"] for s in summaries)),
            "local_rule_changed": int(sum(s["local_rule_changed"] for s in summaries)),
            "skipped": int(sum(s["skipped"] for s in summaries)),
        },
        "outputs": {
            "csv": str(output_dir / "l21_keyframe_ocr_qwen.csv"),
            "jsonl": str(output_dir / "l21_keyframe_ocr_qwen.jsonl"),
            "temporal_debug_csv": str(output_dir / "temporal_qwen_debug.csv"),
            "per_video": str(output_dir / "per_video"),
        },
    }
    (output_dir / "qwen_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
