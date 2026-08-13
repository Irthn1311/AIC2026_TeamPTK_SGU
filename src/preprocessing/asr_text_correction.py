"""
ASR Text Correction using Qwen3-4B Local with OCR and Temporal Context.
========================================================================
Performs contextual spelling and phonetic correction on raw Vietnamese ASR transcripts
using local Qwen3-4B model (4-bit quantized on GPU, auto fallback to CPU).
Incorporates neighboring speech segments and localized OCR keyframe text.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from _bootstrap import PROJECT_ROOT

DEFAULT_MODEL_DIR = PROJECT_ROOT / ".model_cache" / "qwen3_4b"

logger = logging.getLogger("asr-correction")


SYSTEM_PROMPT = """Bạn là chuyên gia ngôn ngữ học chuẩn hóa văn bản và sửa lỗi nhận dạng giọng nói (ASR) tiếng Việt trong các chương trình thời sự, phóng sự truyền hình.

Nhiệm vụ: Sửa lại các từ, cụm từ bị nhận dạng sai âm thanh (homophone, phương ngữ, nói ngọng, nghe nhầm âm thanh) trong câu ASR, kết hợp thông tin ngữ cảnh câu trước, câu sau và chữ OCR trên màn hình.

NGUYÊN TẮC XỬ LÝ:
1. Nhận diện và sửa lỗi ngữ âm / nghe nhầm phổ biến trong tiếng Việt:
   - Địa danh, danh từ riêng, đơn vị hành chính và thuật ngữ tự nhiên: Tự động khôi phục đúng tên chuẩn theo ngữ cảnh thời sự Việt Nam (ví dụ: các vùng miền, sông ngòi, tỉnh thành, hiện tượng thời tiết - thiên tai, thuật ngữ kinh tế - xã hội).
   - Lỗi âm cuối (c/t, n/ng), thanh điệu (hỏi/ngã), phụ âm đầu (d/gi/r, s/x, tr/ch, l/n) do phát âm vùng miền hoặc do bộ nhận dạng nghe nhầm.
2. Khai thác OCR: Nếu văn bản OCR trên khung hình có chứa từ khóa, tên riêng, địa danh hoặc thuật ngữ liên quan, ưu tiên chuẩn hóa câu ASR cho đồng nhất với OCR.
3. Bảo toàn nguyên bản:
   - Tuyệt đối KHÔNG diễn đạt lại (paraphrase), KHÔNG tóm tắt, KHÔNG viết lại văn phong.
   - KHÔNG thêm thông tin mới, KHÔNG xóa bỏ câu chữ vốn có nếu từ đó nhận dạng đúng.
   - Giữ nguyên toàn bộ cấu trúc và nhịp điệu tự nhiên của câu nói.
4. Định dạng phản hồi: Chỉ xuất ra duy nhất câu văn sau khi đã sửa lỗi, không thêm bất kỳ tiền tố, giải thích hay định dạng nào khác."""


def load_ocr_keyframes(ocr_jsonl_path: Path | str) -> list[dict[str, Any]]:
    """Load OCR keyframe detections from JSONL file."""
    ocr_path = Path(ocr_jsonl_path)
    if not ocr_path.exists():
        logger.warning("OCR file not found at %s. Proceeding without OCR context.", ocr_path)
        return []

    records = []
    try:
        with open(ocr_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    except Exception as e:
        logger.error("Error loading OCR JSONL %s: %s", ocr_path, e)
    return records


def extract_ocr_context(
    ocr_records: list[dict[str, Any]],
    start_s: float,
    end_s: float,
    time_window: float = 3.5,
) -> str:
    """Find relevant OCR scene texts within the timestamp window."""
    if not ocr_records:
        return ""

    matched_texts = []
    seen = set()

    for rec in ocr_records:
        ts = float(rec.get("timestamp_seconds", -999.0))
        if (start_s - time_window) <= ts <= (end_s + time_window):
            # Extract high-quality detections
            detections = rec.get("detections", [])
            for det in detections:
                t = str(det.get("text", "")).strip()
                region = det.get("region_type", "")
                conf = float(det.get("confidence", 0.0))
                # Skip TV channel logo artifacts / time overlays (e.g. 06:30:11, HD, 179, H)
                if len(t) < 2 or re.match(r"^\d{1,2}:\d{2}(:\d{2})?$", t) or t in {"HD", "179", "H", "V", "11%", "12"}:
                    continue
                if conf >= 0.70 and t.lower() not in seen:
                    seen.add(t.lower())
                    matched_texts.append(t)

            if not matched_texts and rec.get("combined_text"):
                c_text = str(rec["combined_text"]).strip()
                if c_text and c_text.lower() not in seen:
                    seen.add(c_text.lower())
                    matched_texts.append(c_text)

    return " | ".join(matched_texts[:5])


class Qwen3ASRCorrector:
    """Local Qwen3-4B ASR Text Corrector with 4-bit quantization and CPU fallback."""

    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen3-4B",
        cache_dir: Path | str | None = None,
        use_4bit: bool = True,
        device: str = "cuda",
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        if cache_dir is None:
            cache_dir = DEFAULT_MODEL_DIR
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        os.environ["HF_HOME"] = str(self.cache_dir)
        os.environ["TRANSFORMERS_CACHE"] = str(self.cache_dir)

        logger.info("Initializing Qwen3ASRCorrector with model '%s' in %s...", model_name_or_path, self.cache_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=str(self.cache_dir))

        self.device_used = "cpu"
        self.quantization_used = "None (CPU float32)"

        has_cuda = torch.cuda.is_available() and device.startswith("cuda")
        if has_cuda and use_4bit:
            try:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name_or_path,
                    cache_dir=str(self.cache_dir),
                    quantization_config=bnb_config,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                )
                self.device_used = "cuda"
                self.quantization_used = "4-bit NF4 (BitsAndBytes on CUDA)"
                logger.info("Loaded model on GPU with 4-bit NF4 quantization.")
            except Exception as e:
                logger.warning("Failed to load in 4-bit on CUDA (%s). Falling back to float16 or CPU...", e)
                try:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name_or_path,
                        cache_dir=str(self.cache_dir),
                        torch_dtype=torch.float16,
                        device_map="auto",
                        low_cpu_mem_usage=True,
                    )
                    self.device_used = "cuda"
                    self.quantization_used = "float16 (CUDA)"
                except Exception as e2:
                    logger.warning("CUDA float16 failed (%s). Falling back to CPU...", e2)
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name_or_path,
                        cache_dir=str(self.cache_dir),
                        torch_dtype=torch.float32,
                        low_cpu_mem_usage=True,
                    )
                    self.device_used = "cpu"
                    self.quantization_used = "None (CPU float32)"
        elif has_cuda:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                cache_dir=str(self.cache_dir),
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
            self.device_used = "cuda"
            self.quantization_used = "float16 (CUDA)"
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                cache_dir=str(self.cache_dir),
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
            )
            self.device_used = "cpu"
            self.quantization_used = "None (CPU float32)"

        self.model.eval()

    def correct_segment(
        self,
        curr_text: str,
        prev_text: str = "",
        next_text: str = "",
        ocr_text: str = "",
    ) -> str:
        """Correct a single ASR text string using contextual prompt."""
        curr_text_clean = curr_text.strip()
        if not curr_text_clean:
            return curr_text

        # Prepare user prompt
        user_lines = ["Ngữ cảnh:"]
        if prev_text.strip():
            user_lines.append(f"- Câu trước: {prev_text.strip()}")
        else:
            user_lines.append("- Câu trước: (Bắt đầu đoạn nói)")

        if next_text.strip():
            user_lines.append(f"- Câu sau: {next_text.strip()}")
        else:
            user_lines.append("- Câu sau: (Kết thúc đoạn nói)")

        if ocr_text.strip():
            user_lines.append(f"- Text OCR nhận diện trên màn hình: {ocr_text.strip()}")

        user_lines.append("")
        user_lines.append("Câu ASR cần sửa:")
        user_lines.append(curr_text_clean)
        user_lines.append("")
        user_lines.append("Câu đã sửa:")

        user_content = "\n".join(user_lines)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        # Apply chat template with non-thinking mode
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except Exception:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        max_tokens = min(128, max(32, int(len(curr_text_clean.split()) * 2.5) + 20))

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                repetition_penalty=1.05,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        gen_tokens = outputs[0][inputs.input_ids.shape[1]:]
        raw_response = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        # Clean response: remove any residual thinking / boilerplate
        cleaned = raw_response
        if "</think>" in cleaned:
            cleaned = cleaned.split("</think>")[-1].strip()

        # Strip lead prefixes like "Câu đã sửa:"
        cleaned = re.sub(r"^câu đã sửa\s*:\s*", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = cleaned.strip("\"' \n\t")

        # Sanity validation: If empty or drastically mismatched length, keep raw
        raw_words = curr_text_clean.split()
        corr_words = cleaned.split()
        if not cleaned or len(corr_words) == 0:
            return curr_text_clean
        if len(corr_words) < max(1, int(len(raw_words) * 0.4)) or len(corr_words) > int(len(raw_words) * 2.2) + 5:
            return curr_text_clean

        return cleaned

    def correct_all_segments(
        self,
        segments: list[dict[str, Any]],
        ocr_records: list[dict[str, Any]],
        video_id: str,
    ) -> list[dict[str, Any]]:
        """Correct all ASR segments in the list while preserving metadata & timestamps."""
        corrected_segments: list[dict[str, Any]] = []
        n = len(segments)

        logger.info("Starting text correction for %d segments of %s...", n, video_id)
        t0 = time.time()

        for i, seg in enumerate(segments):
            prev_t = segments[i - 1]["text_raw"] if i > 0 else ""
            curr_t = seg["text_raw"]
            next_t = segments[i + 1]["text_raw"] if i < (n - 1) else ""
            ocr_t = extract_ocr_context(ocr_records, float(seg["start"]), float(seg["end"]))

            normalized_t = self.correct_segment(
                curr_text=curr_t,
                prev_text=prev_t,
                next_text=next_t,
                ocr_text=ocr_t,
            )

            seg_copy = dict(seg)
            seg_copy["video_id"] = video_id
            seg_copy["text_raw"] = curr_t
            seg_copy["text_normalized"] = normalized_t
            corrected_segments.append(seg_copy)

            if (i + 1) % 50 == 0 or (i + 1) == n:
                logger.info("Processed %d/%d segments (%.1fs elapsed)...", i + 1, n, time.time() - t0)

        logger.info("Finished correcting %d segments in %.2fs.", n, time.time() - t0)
        return corrected_segments


def build_corrected_retrieval_chunks(
    corrected_segments: list[dict[str, Any]],
    raw_chunks: list[dict[str, Any]],
    video_id: str,
) -> list[dict[str, Any]]:
    """Build corrected retrieval chunks preserving chunk boundaries and timestamps."""
    # Map segment_id to segment object
    seg_map = {s["asr_id"]: s for s in corrected_segments}

    corrected_chunks = []
    for chunk in raw_chunks:
        seg_ids = chunk.get("segment_ids", [])
        norm_parts = []
        raw_parts = []

        for sid in seg_ids:
            if sid in seg_map:
                s_obj = seg_map[sid]
                norm_parts.append(s_obj.get("text_normalized", s_obj.get("text_raw", "")))
                raw_parts.append(s_obj.get("text_raw", ""))

        text_norm = " ".join(norm_parts).strip()
        text_raw = chunk.get("text", " ".join(raw_parts).strip())

        new_chunk = {
            "chunk_id": chunk["chunk_id"],
            "video_id": video_id,
            "start": chunk["start"],
            "end": chunk["end"],
            "duration": chunk.get("duration", round(chunk["end"] - chunk["start"], 3)),
            "text_raw": text_raw,
            "text_normalized": text_norm,
            "segment_ids": seg_ids,
        }
        corrected_chunks.append(new_chunk)

    return corrected_chunks
