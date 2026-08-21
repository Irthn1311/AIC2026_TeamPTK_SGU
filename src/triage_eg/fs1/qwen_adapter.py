"""Lazy local-only Qwen2.5-VL adapter for bounded QA grounding."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from triage_eg.e2e1.qa import garbage_reason, normalize_answer_text

from .contracts import QWEN_REVISION
from .qa import GroundingCandidate, parse_qwen_output


def _first_complete_json_object(raw: str) -> dict[str, Any]:
    """Return the first balanced JSON object; never repair truncated output."""

    text = str(raw).replace("```json", "").replace("```JSON", "").replace("```", "")
    start = text.find("{")
    if start < 0:
        raise ValueError("QWEN_JSON_OBJECT_MISSING")
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                value = json.loads(text[start : index + 1])
                if not isinstance(value, dict):
                    raise ValueError("QWEN_JSON_ROOT_NOT_OBJECT")
                return value
    raise ValueError("QWEN_JSON_OBJECT_TRUNCATED")


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
            use_fast=False,
        )
        self.model = (
            Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.asset_root,
                local_files_only=True,
                dtype=torch.float16,
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
        evidence_context: str = "",
        answer_type: str = "OTHER",
        answer_policy: str = "SHORT_SEMANTIC",
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        if self.model is None or self.processor is None:
            raise RuntimeError("QWEN_ADAPTER_NOT_LOADED")
        prompt = (
            "Answer only from the supplied frame and bounded evidence context. "
            "Return JSON with keys answer "
            "(concise string) and evidence_sufficient (boolean). "
            "Never return a video ID, frame ID, metadata MID, explanation, "
            "or unsupported OCR fragment. "
            f"Compiled answer type: {answer_type}. Answer policy: {answer_policy}. "
            f"Event: {description}\nQuestion: {question}\n"
            f"Evidence context: {evidence_context[:2000]}"
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
        parsed = parse_qwen_output(raw, candidate, answer_type=answer_type)
        if parsed is not None:
            parsed = {
                **parsed,
                "answer_policy": answer_policy,
                "qwen_verifier_result": {
                    "parsed": True,
                    "evidence_sufficient": bool(parsed["evidence_sufficient"]),
                    "model_revision": QWEN_REVISION,
                },
            }
        audit = {
            "model_revision": QWEN_REVISION,
            "candidate": candidate.__dict__,
            "prompt": prompt,
            "raw_output": raw,
            "parsed": parsed,
            "answer_type": answer_type,
            "answer_policy": answer_policy,
        }
        return parsed, audit

    def answer_extraction(
        self,
        candidate: GroundingCandidate,
        image: Any,
        *,
        description: str,
        question: str,
        evidence_rows: list[dict[str, Any]],
        answer_type: str,
        answer_policy: str,
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        """Bounded extraction only; deterministic verification is deliberately external.

        The compiler owns ``answer_type``.  Keeping that field out of the generated
        schema makes the model response smaller and prevents a model-emitted label
        from being mistaken for a deterministic contract decision.
        """

        if self.model is None or self.processor is None:
            raise RuntimeError("QWEN_ADAPTER_NOT_LOADED")
        catalog = [
            {
                "source_id": str(row["source_id"]),
                "modality": str(row.get("modality", "unknown")),
                "text": str(row.get("text", ""))[:500],
                "frame_distance": row.get("distance_frames"),
                "time_distance_seconds": row.get("time_distance_seconds"),
                "confidence": row.get("confidence"),
            }
            for row in evidence_rows
            if row.get("source_id")
        ]
        prompt = (
            "Extract an answer only from the supplied frame and bounded evidence catalog. "
            "Return one JSON object with exactly four fields: answer (string <=100 chars), "
            "supporting_source_ids (list of catalog source_id strings), "
            "supporting_spans (list of copied or near-verbatim catalog spans), and "
            "evidence_sufficient (boolean). Do not explain, invent IDs, return metadata MIDs, "
            "or cite evidence outside this catalog. Use at most 3 supporting_source_ids and "
            "at most 3 supporting_spans; every span must be <=160 characters. If context is "
            "empty or unrelated, set "
            "evidence_sufficient=false. "
            f"Compiled answer_type={answer_type}; answer_policy={answer_policy}. "
            f"Description={description}; Question={question}; Catalog="
            + json.dumps(catalog, ensure_ascii=False, sort_keys=True)
        )
        messages = [
            {
                "role": "user",
                "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}],
            }
        ]
        rendered = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[rendered], images=[image], padding=True, return_tensors="pt"
        ).to(self.device)
        generated = self.model.generate(**inputs, max_new_tokens=256, do_sample=False)
        trimmed = generated[:, inputs.input_ids.shape[1] :]
        raw = self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]
        parsed = None
        parse_reason = None
        try:
            value = _first_complete_json_object(raw)
            required = {
                "answer",
                "supporting_source_ids",
                "supporting_spans",
                "evidence_sufficient",
            }
            if not required.issubset(value):
                raise ValueError("QWEN_R4_EXTRACTION_REQUIRED_FIELD_MISSING")
            unknown_nested = {
                key
                for key, item in value.items()
                if key not in required and isinstance(item, dict | list)
            }
            if unknown_nested:
                raise ValueError("QWEN_R4_EXTRACTION_UNKNOWN_NESTED_FIELD")
            if not isinstance(value["answer"], str):
                raise ValueError("QWEN_R4_EXTRACTION_ANSWER_NOT_STRING")
            answer = normalize_answer_text(value["answer"])
            sources = value["supporting_source_ids"]
            spans = value["supporting_spans"]
            sufficient = value["evidence_sufficient"]
            if (
                len(answer) > 100
                or not isinstance(sources, list)
                or not all(isinstance(item, str) for item in sources)
                or len(sources) > 3
                or not isinstance(spans, list)
                or not all(isinstance(item, str) for item in spans)
                or len(spans) > 3
                or any(len(item) > 160 for item in spans)
                or not isinstance(sufficient, bool)
                or (
                    sufficient
                    and (
                        not answer
                        or garbage_reason(answer, answer_type)
                        or answer in {candidate.video_id, str(candidate.frame_id)}
                    )
                )
            ):
                raise ValueError("QWEN_R2_EXTRACTION_CONTRACT_INVALID")
            parsed = {
                "video_id": candidate.video_id,
                "frame_id": candidate.frame_id,
                "grounding_rank": candidate.evidence_rank,
                "answer": answer,
                "answer_type": str(answer_type).upper(),
                "answer_policy": answer_policy,
                "supporting_source_ids": sources,
                "supporting_spans": spans,
                "evidence_sufficient": sufficient,
                "qwen_model_revision": QWEN_REVISION,
            }
        except (KeyError, TypeError, ValueError, json.JSONDecodeError, AttributeError) as error:
            parse_reason = f"{type(error).__name__}:{error}"
        audit = {
            "model_revision": QWEN_REVISION,
            "candidate": candidate.__dict__,
            "answer_type": answer_type,
            "answer_policy": answer_policy,
            "evidence_catalog": catalog,
            "prompt": prompt,
            "raw_output": raw,
            "extraction": parsed,
            "parse_reason": parse_reason,
            "qwen_parse_pass": parsed is not None,
            "generated_schema_fields": [
                "answer",
                "supporting_source_ids",
                "supporting_spans",
                "evidence_sufficient",
            ],
            "compiled_answer_type_attached_after_parse": True,
            "final_evidence_sufficient_assigned_here": False,
        }
        return parsed, audit
