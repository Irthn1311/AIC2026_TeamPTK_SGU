"""
Qwen2.5-VL Client — Shared VLM inference engine for Q&A and TRAKE alignment (v2).

Changes from v1:
- answer_question(): Now passes answer_type to build_qa_combined_prompt for
  type-specific instructions (count / color / name / yes_no / description).
- answer_question(): Returns QAAnswer (not just str) — richer result with found/confidence.
- score_alignment(): Passes activity_name for better TRAKE context.
- _infer(): Unchanged core logic.

Model VRAM requirements:
  - Qwen2.5-VL-7B-Instruct (fp16):  ~14 GB VRAM
  - Qwen2.5-VL-7B-Instruct (4bit):  ~8  GB VRAM (Kaggle T4 safe)
  - Qwen2.5-VL-3B-Instruct (fp16):  ~7  GB VRAM (faster, slightly lower quality)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from src.llm.prompt_templates import (
    build_qa_combined_prompt,
    build_qa_answer_prompt,
    build_qa_relevance_prompt,
    build_trake_align_prompt,
    SYSTEM_QA, SYSTEM_VERIFY, SYSTEM_TRAKE,
)
from src.llm.response_parser import ResponseParser, QAAnswer, RelevanceScore, AlignmentScore
from src.utils.logger import get_logger

logger = get_logger(__name__)

try:
    import torch
    from transformers import AutoProcessor
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


class QwenVLClient:
    """
    Qwen2.5-VL inference client for Visual Question Answering and TRAKE alignment.

    Provides high-level methods used by the Q&A and TRAKE pipelines:
      - answer_question()    → QAAnswer  (combined 1-call approach)
      - score_relevance()    → RelevanceScore  (legacy 2-step, still available)
      - score_alignment()    → AlignmentScore  (for TRAKE, chain-of-thought)

    Usage:
        client = QwenVLClient(load_in_4bit=True)
        client.load()

        result = client.answer_question(
            image_path="keyframes/L21/V001/5.jpg",
            event_description="Lễ trao giải âm nhạc...",
            question="Có bao nhiêu người lên sân khấu?",
            answer_type="count",
        )
        # result.answer      → "5"
        # result.confidence  → 0.9
        # result.found       → True
        # result.observation → "Tôi thấy 5 người đứng trên sân khấu..."
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: str = "cuda",
        load_in_4bit: bool = False,
        max_new_tokens: int = 256,
    ):
        self.model_name     = model_name
        self.device         = device
        self.load_in_4bit   = load_in_4bit
        self.max_new_tokens = max_new_tokens

        self._model     = None
        self._processor = None
        self._parser    = ResponseParser()

    def load(self) -> "QwenVLClient":
        """Load Qwen2.5-VL model and processor."""
        global _QWEN_AVAILABLE, _QWEN_IMPORT_ERR

        # Always import torch at the top so it is in scope for the entire method.
        # This avoids Python 3.12 UnboundLocalError caused by 'import torch'
        # appearing inside the inner try/except below.
        try:
            import torch
        except ImportError:
            raise ImportError("torch is required. Run: pip install torch")

        if not _QWEN_AVAILABLE:
            try:
                from transformers import AutoProcessor
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

        from transformers import AutoProcessor
        logger.info(f"Loading {self.model_name} (4bit={self.load_in_4bit}, device={self.device})")

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
            self.model_name, **model_kwargs
        )
        self._model.eval()

        self._processor = AutoProcessor.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        logger.info(f"Qwen2.5-VL ready ({self.model_name})")
        return self

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    def answer_question(
        self,
        image_path: str,
        event_description: str,
        question: str,
        answer_language: str = "vi",
        answer_type: str = "description",
    ) -> QAAnswer:
        """
        Answer a question about the event shown in the keyframe.
        Uses combined 1-call approach (relevance + answer in one prompt).

        Args:
            image_path:        Path to keyframe .jpg
            event_description: Context about the event
            question:          The question to answer
            answer_language:   "vi" | "en" | "auto"
            answer_type:       "count" | "color" | "name" | "yes_no" | "description"

        Returns:
            QAAnswer with answer text, confidence, found flag, and observation
        """
        messages = build_qa_combined_prompt(
            image_path=image_path,
            event_description=event_description,
            question=question,
            answer_type=answer_type,
            answer_language=answer_language if answer_language != "auto" else "vi",
        )
        raw = self._infer(messages, system=SYSTEM_QA)
        return self._parser.parse_qa_answer(raw)

    def score_relevance(
        self,
        image_path: str,
        event_description: str,
        question: str,
    ) -> RelevanceScore:
        """
        Score how relevant a keyframe is for answering the question.
        NOTE: In the new pipeline this is rarely called — answer_question()
        handles both relevance and answer in one call. Kept for backward compat.

        Returns:
            RelevanceScore with relevant bool and 0-1 confidence
        """
        messages = build_qa_relevance_prompt(image_path, event_description, question)
        raw = self._infer(messages, system=SYSTEM_VERIFY)
        return self._parser.parse_relevance(raw)

    def score_alignment(
        self,
        image_path: str,
        event_name: str,
        semantic_keyframe_hint: str,
        activity_name: str = "",
    ) -> AlignmentScore:
        """
        Verify if a keyframe matches a specific TRAKE event moment.
        Uses chain-of-thought: describe → then judge.

        Args:
            image_path:             Path to keyframe .jpg
            event_name:             e.g. "Giậm nhảy"
            semantic_keyframe_hint: Detailed description of the target moment
            activity_name:          e.g. "Nhảy cao" (optional context)

        Returns:
            AlignmentScore with match bool, confidence, observation, and reason
        """
        messages = build_trake_align_prompt(
            image_path=image_path,
            event_name=event_name,
            semantic_keyframe_hint=semantic_keyframe_hint,
            activity_name=activity_name,
        )
        raw = self._infer(messages, system=SYSTEM_TRAKE)
        return self._parser.parse_alignment(raw)

    # ----------------------------------------------------------
    # Internal inference
    # ----------------------------------------------------------

    def _infer(
        self,
        messages: List[Dict[str, Any]],
        system: Optional[str] = None,
    ) -> str:
        """
        Run one VLM forward pass and return the generated text.

        Args:
            messages: Chat messages list (user turn with image + text)
            system:   Optional system prompt prepended to the conversation
        """
        self._check_loaded()

        full_messages = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)

        try:
            text = self._processor.apply_chat_template(
                full_messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(full_messages)
            inputs = self._processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self._model.device)

            with torch.no_grad():
                output_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    temperature=None,  # greedy decoding for determinism
                )
            generated = output_ids[:, inputs.input_ids.shape[1]:]
            result = self._processor.batch_decode(
                generated, skip_special_tokens=True
            )[0].strip()
            return result

        except Exception as e:
            logger.warning(f"[QwenVLClient] Inference failed ({type(e).__name__}): {e}")
            return ""

    def _check_loaded(self) -> None:
        if self._model is None:
            raise RuntimeError("QwenVLClient not loaded. Call client.load() first.")

    @property
    def is_loaded(self) -> bool:
        return self._model is not None
