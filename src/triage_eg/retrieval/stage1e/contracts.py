"""Locked contracts for the Stage 1E AI evaluation and language-path freeze."""

from __future__ import annotations

STAGE1E_VERSION = "0.1.0"
EVALUATION_MODE = "AI_JUDGED_BLINDED_VISUAL_REVIEW"
AI_REVIEW_STATUS = "COMPLETE"
HUMAN_REVIEW_STATUS = "NOT_PERFORMED"
LANGUAGE_BRIDGE_INTERNAL_GATE = "ACCEPT"
LANGUAGE_BRIDGE_QUALITY_STATUS = "AI_EVALUATED_ACCEPTED"
LANGUAGE_PATH_STATUS = "FROZEN_FOR_INTERNAL_BASELINE"
STAGE2_READINESS = "READY"
JUDGE_PROVIDER = "OpenAI"
JUDGE_MODEL = "GPT-5.6 Sol"

TRANSLATOR_MODEL_ID = "Helsinki-NLP/opus-mt-vi-en"
TRANSLATOR_REVISION = "c8d2853e77f5fae31124d993e0b35176b1c8914e"
CLIP_CANDIDATE = "openai_clip_vit_b32_openai_official"

EXPECTED_JUDGMENTS = 210
EXPECTED_PAIRS = 14
CONDITIONS_PER_PAIR = 3
TOP_K = 5

PAIR_METRIC_FIELDS = (
    "pair_id",
    "category",
    "difficulty",
    "EN_DIRECT_relevance_top5",
    "EN_DIRECT_graded_top5",
    "VI_DIRECT_relevance_top5",
    "VI_DIRECT_graded_top5",
    "VI_TRANSLATED_EN_relevance_top5",
    "VI_TRANSLATED_EN_graded_top5",
    "translated_minus_vi_graded_top5",
    "translated_minus_en_graded_top5",
)
