"""Lazy local-only Qwen2.5-VL adapter for bounded QA grounding."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .contracts import QWEN_REVISION
from .qa import GroundingCandidate, parse_qwen_output


class QwenEvidenceAdapter:
    def __init__(self, asset_root: Path, *, device: str = "cuda") -> None:
        self.asset_root = Path(asset_root).resolve(strict=True)
        self.device = device
        self.model: Any = None
        self.processor: Any = None

    def load(self) -> None:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(
            self.asset_root,
            local_files_only=True,
            min_pixels=200704,
            max_pixels=401408,
        )
        self.model = (
            Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.asset_root,
                local_files_only=True,
                torch_dtype=torch.float16,
                attn_implementation="sdpa",
                low_cpu_mem_usage=True,
            )
            .to(self.device)
            .eval()
        )

    def unload(self) -> None:
        self.model = self.processor = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def answer(
        self,
        candidate: GroundingCandidate,
        image: Any,
        *,
        description: str,
        question: str,
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        if self.model is None or self.processor is None:
            raise RuntimeError("QWEN_ADAPTER_NOT_LOADED")
        prompt = (
            "Answer only from the supplied frame. Return JSON with keys answer "
            "(concise string) and evidence_sufficient (boolean). "
            f"Event: {description}\nQuestion: {question}"
        )
        messages = [
            {
                "role": "user",
                "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=[text], images=[image], padding=True, return_tensors="pt").to(
            self.device
        )
        generated = self.model.generate(**inputs, max_new_tokens=64, do_sample=False)
        trimmed = generated[:, inputs.input_ids.shape[1] :]
        raw = self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]
        parsed = parse_qwen_output(raw, candidate)
        audit = {
            "model_revision": QWEN_REVISION,
            "candidate": candidate.__dict__,
            "prompt": prompt,
            "raw_output": raw,
            "parsed": parsed,
        }
        return parsed, audit
