"""
Caption Generator — Generate natural language descriptions for keyframes
using Qwen2.5-VL (Vision-Language Model), running offline on local GPU.

Input:  Keyframe images (.jpg) from the Kaggle dataset
Output: JSON files per video with captions per keyframe

Output format (datasets/captions/L21_V001.json):
    {
      "video_id": "L21_V001",
      "extractor": "qwen25_vl",
      "keyframes": [
        {
          "n": 1,
          "frame_idx": 0,
          "pts_time": 0.0,
          "caption_en": "A news anchor in a red blazer speaking at a podium...",
          "caption_vi": "Người dẫn chương trình mặc áo đỏ đứng phát biểu tại bục..."
        },
        ...
      ]
    }

Model: Qwen/Qwen2.5-VL-7B-Instruct (or smaller 3B variant)
Memory: ~14GB VRAM for 7B in float16; ~8GB with 4-bit quantisation (bitsandbytes)
Kaggle: T4 GPU has 16GB VRAM → 7B float16 fits; P100 also compatible.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.feature_extractors.base import BaseExtractor
from src.utils.logger import get_logger

logger = get_logger(__name__)

try:
    import torch
    from transformers import AutoTokenizer, AutoProcessor
    from qwen_vl_utils import process_vision_info
    _QWEN_AVAILABLE = True
    _QWEN_IMPORT_ERR = None
except ImportError as e:
    _QWEN_AVAILABLE = False
    _QWEN_IMPORT_ERR = e


def _get_qwen_model_class():
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration
        return Qwen2_5_VLForConditionalGeneration
    except ImportError:
        pass
    try:
        from transformers import Qwen2VLForConditionalGeneration
        return Qwen2VLForConditionalGeneration
    except ImportError:
        pass
    try:
        from transformers import AutoModelForVision2Seq
        return AutoModelForVision2Seq
    except ImportError:
        pass
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM

# ----------------------------------------------------------
# Prompt templates
# ----------------------------------------------------------
_PROMPT_EN = (
    "Describe this video frame in detail. Include: "
    "the main subject (people, objects, scene), "
    "clothing colors, actions being performed, "
    "location/setting, and any visible text or signs. "
    "Be specific and concise (2-3 sentences)."
)

_PROMPT_VI = (
    "Mô tả chi tiết khung hình video này bằng tiếng Việt. "
    "Bao gồm: chủ thể chính (người, vật, cảnh), "
    "màu sắc quần áo, hành động đang diễn ra, "
    "địa điểm/bối cảnh, và bất kỳ văn bản hay biển hiệu nào có thể nhìn thấy. "
    "Ngắn gọn và cụ thể (2-3 câu)."
)


class CaptionGenerator(BaseExtractor):
    """
    Generates image captions using Qwen2.5-VL vision-language model.

    Generates both English and Vietnamese captions for maximum retrieval coverage:
    - English: better alignment with CLIP text embedding space
    - Vietnamese: better for Vietnamese OCR/ASR text retrieval

    Args:
        model_name:   HuggingFace model ID
        device:       "cuda" or "cpu"
        load_in_4bit: Use 4-bit quantisation to reduce VRAM usage
        max_new_tokens: Max tokens per generated caption

    Usage:
        captioner = CaptionGenerator(load_in_4bit=True)
        captioner.load()
        result = captioner.extract_one("keyframes/L21/V001/5.jpg",
                                       n=5, frame_idx=450, pts_time=18.24)
        # result["caption_en"] → "A news anchor in a red blazer..."
        # result["caption_vi"] → "Người dẫn chương trình mặc áo đỏ..."
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: str = "cuda",
        load_in_4bit: bool = False,   # Set True to use ~8GB VRAM instead of 14GB
        max_new_tokens: int = 150,
        generate_vi: bool = True,     # Also generate Vietnamese caption
    ):
        self.model_name     = model_name
        self.device         = device
        self.load_in_4bit   = load_in_4bit
        self.max_new_tokens = max_new_tokens
        self.generate_vi    = generate_vi

        self._model     = None
        self._processor = None

    @property
    def name(self) -> str:
        return "qwen25_vl"

    def is_available(self) -> bool:
        return _QWEN_AVAILABLE

    def load(self) -> "CaptionGenerator":
        """Load Qwen2.5-VL model and processor."""
        global _QWEN_AVAILABLE, _QWEN_IMPORT_ERR
        if not _QWEN_AVAILABLE:
            try:
                import torch
                from transformers import AutoTokenizer, AutoProcessor
                from qwen_vl_utils import process_vision_info
                _QWEN_AVAILABLE = True
                _QWEN_IMPORT_ERR = None
            except ImportError as e:
                _QWEN_AVAILABLE = False
                _QWEN_IMPORT_ERR = e
                raise ImportError(
                    f"Qwen dependencies import failed ({e}). "
                    "Run: pip install transformers qwen-vl-utils accelerate bitsandbytes"
                )
        import torch
        from transformers import AutoProcessor

        logger.info(f"Loading {self.model_name} (4bit={self.load_in_4bit})")

        model_kwargs: Dict[str, Any] = {"torch_dtype": torch.float16}
        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["device_map"] = self.device

        model_cls = _get_qwen_model_class()
        self._model = model_cls.from_pretrained(
            self.model_name,
            **model_kwargs,
        )
        self._model.eval()

        self._processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        logger.info(f"{self.model_name} loaded.")
        return self

    def extract_one(
        self,
        input_path: str,
        n: int = 0,
        frame_idx: int = 0,
        pts_time: float = 0.0,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate English (and optionally Vietnamese) caption for one keyframe.

        Returns:
            {
              "n": 5, "frame_idx": 450, "pts_time": 18.24,
              "caption_en": "...", "caption_vi": "..."
            }
        """
        if self._model is None:
            raise RuntimeError("CaptionGenerator not loaded. Call load() first.")

        caption_en = self._generate_caption(input_path, _PROMPT_EN)
        caption_vi = self._generate_caption(input_path, _PROMPT_VI) if self.generate_vi else ""

        return {
            "n":          n,
            "frame_idx":  frame_idx,
            "pts_time":   pts_time,
            "caption_en": caption_en,
            "caption_vi": caption_vi,
        }

    def _generate_caption(self, image_path: str, prompt: str) -> str:
        """Run a single VLM inference on image + prompt."""
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": f"file://{image_path}"},
                        {"type": "text",  "text": prompt},
                    ],
                }
            ]
            text = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)

            target_device = self._model.device if hasattr(self._model, "device") else (
                "cuda" if torch.cuda.is_available() else self.device
            )
            inputs = self._processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(target_device)

            with torch.no_grad():
                output_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    temperature=None,
                )
            # Decode only the generated portion
            generated = output_ids[:, inputs.input_ids.shape[1]:]
            caption = self._processor.batch_decode(
                generated, skip_special_tokens=True
            )[0].strip()
            return caption
        except Exception as e:
            logger.warning(f"[Captioner] Failed for {image_path}: {e}")
            return ""

    # ----------------------------------------------------------
    # Video-level batch processing
    # ----------------------------------------------------------

    def extract_video(
        self,
        video_id: str,
        keyframes_dir: str,
        map_keyframes_csv: str,
        output_dir: str,
        batch_size: int = 4,
        overwrite: bool = False,
    ) -> Path:
        """
        Generate captions for all keyframes of one video and save to JSON.

        Args:
            batch_size: Number of keyframes processed before saving a checkpoint.
                        Use smaller batch_size on machines with limited VRAM.
        """
        import pandas as pd

        out_path = Path(output_dir) / f"{video_id}.json"
        if out_path.exists() and not overwrite:
            logger.debug(f"[Captioner] Skipping {video_id} (already exists)")
            return out_path

        df = pd.read_csv(map_keyframes_csv)
        batch_id = video_id.split("_")[0]

        # Robust multi-level keyframe folder resolution (AVOID recursive glob)
        root = Path(keyframes_dir)
        video_dir = None

        search_roots = []
        curr = root
        for _ in range(4):
            if curr.exists() and curr not in search_roots:
                search_roots.append(curr)
            if curr.parent == curr:
                break
            curr = curr.parent

        # 1. Fast subfolder matching for split batches (e.g., Keyframes_L26_a, Keyframes_L26_b, etc.)
        for r in search_roots:
            try:
                for sub in r.iterdir():
                    if sub.is_dir():
                        if sub.name.startswith(f"Keyframes_{batch_id}") or sub.name.startswith(batch_id) or sub.name == "keyframes":
                            for rel in [video_id, f"keyframes/{video_id}"]:
                                cand = sub / rel
                                if cand.exists() and cand.is_dir():
                                    video_dir = cand
                                    break
                        if video_dir:
                            break
            except Exception:
                pass
            if video_dir:
                break

        # 2. Direct candidate paths fallback
        if video_dir is None:
            for r in search_roots:
                candidates = [
                    r / f"Keyframes_{batch_id}" / "keyframes" / video_id,
                    r / f"Keyframes_{batch_id}" / video_id,
                    r / "keyframes" / f"Keyframes_{batch_id}" / "keyframes" / video_id,
                    r / "keyframes" / f"Keyframes_{batch_id}" / video_id,
                    r / "keyframes" / video_id,
                    r / video_id,
                    r / batch_id / "keyframes" / video_id,
                    r / batch_id / video_id,
                ]
                for cand in candidates:
                    if cand.exists() and cand.is_dir():
                        video_dir = cand
                        break
                if video_dir:
                    break

        keyframe_results = []
        for i, (_, row) in enumerate(df.iterrows()):
            n = int(row["n"])
            names = [f"{n:03d}.jpg", f"{n}.jpg", f"{n:04d}.jpg", f"{n:02d}.jpg"]

            img_path = None
            if video_dir:
                for name in names:
                    cand = video_dir / name
                    if cand.exists():
                        img_path = str(cand)
                        break

            if not img_path or not Path(img_path).exists():
                logger.warning(f"Image not found: {video_id} n={n}")
                keyframe_results.append({
                    "n": n, "frame_idx": int(row["frame_idx"]),
                    "pts_time": float(row["pts_time"]),
                    "caption_en": "", "caption_vi": "",
                })
                continue

            kf_result = self.extract_one(
                img_path,
                n=n,
                frame_idx=int(row["frame_idx"]),
                pts_time=float(row["pts_time"]),
            )
            keyframe_results.append(kf_result)

            if (i + 1) % batch_size == 0:
                logger.debug(f"[Captioner] {video_id}: {i+1}/{len(df)} done")

        # Save JSON
        out_path.parent.mkdir(parents=True, exist_ok=True)
        output = {
            "video_id":  video_id,
            "extractor": self.name,
            "model":     self.model_name,
            "total":     len(keyframe_results),
            "keyframes": keyframe_results,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        logger.info(f"[Captioner] {video_id}: {len(keyframe_results)} captions → {out_path}")
        return out_path
