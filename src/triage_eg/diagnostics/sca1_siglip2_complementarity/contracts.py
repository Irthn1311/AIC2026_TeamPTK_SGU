"""Frozen contracts for SCA-1 SigLIP2 complementarity diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

MODEL_ID = "google/siglip2-base-patch16-224"
MODEL_REVISION = "0ad8c6e0ff16615356a08a1ad8c8bbc8930c434e"
MODEL_SAFETENSORS_SHA256 = "612923381c76ec5a9bed335d1c48827e3f2e506ac31b044b63b2031fadee6a0b"
FROZEN_PREPARATION_ZIP_SHA256 = "249a913c560e2ec90ede868cb507e9baa6b9e6fb9a72266747cdbb53d391c3ad"
TCA1_ANCHOR_COMMIT = "cd43d1d2b33d4a5fa808f8f3aad23b199325e119"
EXPECTED_A0_PREDICTION_SHA256 = "8a774e25aae0d4e23eafa905e468b25baeabc0b2ed74ba16491a1138b099ef9e"
EXPECTED_STAGE1_FINGERPRINT = "39ab968d2d957ce111cf8233d10ee08a281868c03b0b7d41ecf39ce5bb2c95b8"
EXPECTED_OPENAI_CLIP_ID = "openai_clip_vit_b32_openai_official"
EXPECTED_OPENAI_CLIP_SHA256 = "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"
EXPECTED_TRANSLATOR_ID = "Helsinki-NLP/opus-mt-vi-en"
EXPECTED_TRANSLATOR_REVISION = "c8d2853e77f5fae31124d993e0b35176b1c8914e"

EXPECTED_ROWS = 177_321
EMBEDDING_DIMENSION = 768
IMAGE_SIZE = 224
PATCH_SIZE = 16
TEXT_MAX_LENGTH = 64
TEXT_PADDING = "max_length"
TEXT_TRUNCATION = True
PROCESSOR_USE_FAST = False
PRIMARY_BENCHMARK = "DEV_CROSS_60"
PRIMARY_VARIANT = "G1_COVERAGE_COARSE"
SEMANTIC_UNIT_COUNT = 100

FUSION_RESCUE_THRESHOLD = 5
FUSION_TRAKE_EVENT_DELTA_THRESHOLD = 3
FUSION_TRAKE_CHAIN_DELTA_THRESHOLD = 2

RUNTIME_MODEL_FILES = (
    "config.json",
    "model.safetensors",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
)

PREPARATION_MEMBER_HASHES = {
    "sca1_preparation/decision_context.json": (
        "077a01697d68cca4496ccad8110e448a1407f79bc927035e6335d44419d42682"
    ),
    "sca1_preparation/model_selection.json": (
        "96be984e2f68aeb74de8d996d3dce88920bc4f03f81b61f442cbfbcb5b653464"
    ),
    "sca1_preparation/README.md": (
        "2848e2920692c391e2b513ea1912651fee4c99e0a2b805274d8a92a9d72ec433"
    ),
    "sca1_preparation/representation_ceiling_audit.json": (
        "3d0e3cd30ba78f9f9270b473017c362a3cffc2756ff3a67e6993c464d85a9bf3"
    ),
    "sca1_preparation/SCA1_PROTOCOL.md": (
        "4c89a24f1e4be6bd79c94b7d1eb46ae02d5b472e2659a313b92c8253a8b6a45e"
    ),
    "sca1_preparation/tca1_paired_score_delta.json": (
        "047c153b73df8f22d1f5a943f8ccb035d360b873d420badb7d4455c9955224a5"
    ),
    "sca1_preparation/tca1_summary.json": (
        "44fbf62b886647349f24e65e06ad1e5dce65c9164f5f314513a1d390f8cc67d4"
    ),
}


@dataclass(frozen=True)
class SCA1Settings:
    """Predeclared, non-tunable SCA-1 protocol."""

    benchmark: str = PRIMARY_BENCHMARK
    variant: str = PRIMARY_VARIANT
    expected_rows: int = EXPECTED_ROWS
    embedding_dimension: int = EMBEDDING_DIMENSION
    text_max_length: int = TEXT_MAX_LENGTH
    text_padding: str = TEXT_PADDING
    text_truncation: bool = TEXT_TRUNCATION
    rescue_threshold: int = FUSION_RESCUE_THRESHOLD
    trake_event_delta_threshold: int = FUSION_TRAKE_EVENT_DELTA_THRESHOLD
    trake_chain_delta_threshold: int = FUSION_TRAKE_CHAIN_DELTA_THRESHOLD
    direct_vietnamese: bool = False
    fusion: bool = False
    query_rewrite: bool = False
    run_m1: bool = False
    use_m2: bool = False
    use_m3: bool = False
    use_event_graph: bool = False
    use_vlm: bool = False
    use_agent: bool = False
    extra_frame_sampling: bool = False
    parameter_sweep: bool = False
    production_policy_changed: bool = False

    def __post_init__(self) -> None:
        frozen = (
            self.benchmark == PRIMARY_BENCHMARK
            and self.variant == PRIMARY_VARIANT
            and self.expected_rows == EXPECTED_ROWS
            and self.embedding_dimension == EMBEDDING_DIMENSION
            and self.text_max_length == TEXT_MAX_LENGTH
            and self.text_padding == TEXT_PADDING
            and self.text_truncation is TEXT_TRUNCATION
            and self.rescue_threshold == FUSION_RESCUE_THRESHOLD
            and self.trake_event_delta_threshold == FUSION_TRAKE_EVENT_DELTA_THRESHOLD
            and self.trake_chain_delta_threshold == FUSION_TRAKE_CHAIN_DELTA_THRESHOLD
        )
        disabled = not any(
            (
                self.direct_vietnamese,
                self.fusion,
                self.query_rewrite,
                self.run_m1,
                self.use_m2,
                self.use_m3,
                self.use_event_graph,
                self.use_vlm,
                self.use_agent,
                self.extra_frame_sampling,
                self.parameter_sweep,
                self.production_policy_changed,
            )
        )
        if not frozen or not disabled:
            raise ValueError("SCA-1 settings are frozen and diagnostic-only")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


__all__ = [
    "EMBEDDING_DIMENSION",
    "EXPECTED_A0_PREDICTION_SHA256",
    "EXPECTED_OPENAI_CLIP_ID",
    "EXPECTED_OPENAI_CLIP_SHA256",
    "EXPECTED_ROWS",
    "EXPECTED_STAGE1_FINGERPRINT",
    "EXPECTED_TRANSLATOR_ID",
    "EXPECTED_TRANSLATOR_REVISION",
    "FROZEN_PREPARATION_ZIP_SHA256",
    "FUSION_RESCUE_THRESHOLD",
    "FUSION_TRAKE_CHAIN_DELTA_THRESHOLD",
    "FUSION_TRAKE_EVENT_DELTA_THRESHOLD",
    "IMAGE_SIZE",
    "MODEL_ID",
    "MODEL_REVISION",
    "MODEL_SAFETENSORS_SHA256",
    "PATCH_SIZE",
    "PROCESSOR_USE_FAST",
    "PREPARATION_MEMBER_HASHES",
    "PRIMARY_BENCHMARK",
    "PRIMARY_VARIANT",
    "RUNTIME_MODEL_FILES",
    "SCA1Settings",
    "SEMANTIC_UNIT_COUNT",
    "TCA1_ANCHOR_COMMIT",
    "TEXT_MAX_LENGTH",
    "TEXT_PADDING",
    "TEXT_TRUNCATION",
]
