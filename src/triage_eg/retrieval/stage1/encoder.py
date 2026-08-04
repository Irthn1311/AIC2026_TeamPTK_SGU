"""Explicit encoder compatibility gate and lazy text-encoder adapters."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from triage_eg.retrieval.stage1.contracts import EncoderContract, TextEncoder


def load_encoder_contract(path: str | Path | None) -> EncoderContract:
    if path is None:
        return EncoderContract()
    candidate = Path(path)
    if not candidate.is_file():
        raise FileNotFoundError(f"ENCODER_ASSET_NOT_FOUND: {candidate}")
    value = yaml.safe_load(candidate.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Encoder config must contain a mapping")
    value = value.get("encoder", value)
    if not isinstance(value, dict):
        raise ValueError("Encoder contract must contain a mapping")
    return EncoderContract.from_dict(value)


def compatibility_gate(contract: EncoderContract, *, allow_unverified: bool = False) -> str:
    if contract.output_dimension != 512:
        raise ValueError("Encoder output dimension must equal 512")
    if contract.evidence_source not in {
        "NONE",
        "AUTHORITATIVE",
        "USER_ASSERTED",
        "EMPIRICAL_PROBE",
    }:
        raise ValueError("Unknown encoder evidence_source")
    if contract.compatibility_status not in {
        "VERIFIED",
        "USER_ASSERTED",
        "UNVERIFIED",
        "BLOCKED",
    }:
        raise ValueError("Unknown encoder compatibility_status")
    if contract.compatibility_status == "VERIFIED":
        if contract.evidence_source not in {"AUTHORITATIVE", "EMPIRICAL_PROBE"}:
            raise PermissionError("VERIFIED encoder requires authoritative or empirical evidence")
        return "VERIFIED"
    if contract.compatibility_status in {"USER_ASSERTED", "UNVERIFIED"} and allow_unverified:
        return "UNVERIFIED_OVERRIDE"
    if contract.compatibility_status == "USER_ASSERTED":
        raise PermissionError("USER_ASSERTED encoder requires --allow-unverified-encoder")
    raise PermissionError("Text retrieval is BLOCKED: encoder compatibility is unverified")


def validate_encoder_output(values: Any, batch_size: int) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.shape != (batch_size, 512):
        raise ValueError(f"Encoder output must have shape ({batch_size}, 512)")
    if not np.isfinite(matrix).all():
        raise ValueError("Encoder output contains non-finite values")
    if np.any(np.linalg.norm(matrix, axis=1) == 0):
        raise ValueError("Encoder output contains zero-norm rows")
    return matrix


class OpenClipTextEncoder:
    """Lazy OpenCLIP adapter; never downloads assets automatically."""

    def __init__(self, contract: EncoderContract) -> None:
        if not contract.model_name or not contract.checkpoint_path:
            raise FileNotFoundError(
                "ENCODER_ASSET_NOT_FOUND: model_name and local checkpoint_path required"
            )
        checkpoint = Path(contract.checkpoint_path).expanduser().resolve(strict=False)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"ENCODER_ASSET_NOT_FOUND: {checkpoint}")
        if contract.model_name.startswith("hf-hub:"):
            raise ValueError("hf-hub model names are blocked to prevent automatic downloads")
        if contract.tokenizer != "open_clip_simple":
            raise ValueError(
                "OpenCLIP requires tokenizer='open_clip_simple' for offline Stage 1 use"
            )
        try:
            import open_clip
        except ImportError as error:
            raise ImportError("open_clip is not installed") from error
        self._torch = __import__("torch")
        self._model, _, _ = open_clip.create_model_and_transforms(
            contract.model_name, pretrained=None, device="cpu"
        )
        open_clip.load_checkpoint(self._model, str(checkpoint))
        self._tokenizer = open_clip.tokenizer.SimpleTokenizer()

    def encode_text(self, texts: list[str]) -> np.ndarray:
        tokens = self._tokenizer(texts)
        with self._torch.no_grad():
            values = self._model.encode_text(tokens).cpu().numpy()
        return validate_encoder_output(values, len(texts))


def load_text_encoder(contract: EncoderContract) -> TextEncoder:
    if contract.implementation == "open_clip":
        return OpenClipTextEncoder(contract)
    raise FileNotFoundError(
        "ENCODER_ASSET_NOT_FOUND: unsupported or missing implementation "
        f"{contract.implementation!r}"
    )


def write_compatibility_report(
    root: Path, contract: EncoderContract, status: str, reason: str
) -> None:
    encoder_root = root / "encoder"
    encoder_root.mkdir(parents=True, exist_ok=True)
    (encoder_root / "encoder_contract.json").write_text(
        json.dumps(contract.__dict__, indent=2) + "\n", encoding="utf-8"
    )
    report = {
        "compatibility_status": status,
        "reason": reason,
        "text_search_available": status in {"VERIFIED", "UNVERIFIED_OVERRIDE"},
        "results_trusted": status == "VERIFIED",
    }
    if status == "UNVERIFIED_OVERRIDE":
        report["warning"] = (
            "Encoder compatibility is unverified; retrieval results are not trustworthy evidence"
        )
    (encoder_root / "compatibility_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    summary_path = root / "stage1_summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["encoder_compatibility_status"] = contract.compatibility_status
        summary["text_search_available"] = report["text_search_available"]
        summary.setdefault("next_stage_readiness", {})["text_retrieval"] = (
            "READY"
            if status == "VERIFIED"
            else "AVAILABLE_WITH_UNVERIFIED_OVERRIDE"
        )
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report_path = root / "stage1_report.md"
    if report_path.is_file():
        lines = report_path.read_text(encoding="utf-8").splitlines()
        text_status = (
            "- Text search: available with VERIFIED encoder"
            if status == "VERIFIED"
            else "- Text search: UNVERIFIED_OVERRIDE active; results are untrusted"
        )
        lines = [text_status if line.startswith("- Text search:") else line for line in lines]
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
