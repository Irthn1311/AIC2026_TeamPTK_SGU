"""Markdown reporting for Stage 1D."""

from __future__ import annotations

import json
from typing import Any


def build_stage1d_report(
    summary: dict[str, Any],
    comparisons: list[dict[str, Any]],
    translations: list[dict[str, Any]],
) -> str:
    translated = {item["pair_id"]: item for item in translations}
    lines = [
        "# Stage 1D Vietnamese Translation Bridge Ablation",
        "",
        "LANGUAGE_BRIDGE_QUALITY_STATUS = NOT_REVIEWED",
        "",
        "## Provenance",
        "",
        f"- Build commit: {summary['build_git_commit']}",
        f"- Stage 1D version: {summary['stage1d_version']}",
        "",
        "## Frozen Stage 1C Baseline",
        "",
        f"- Status: {summary['stage1c_frozen_baseline']['status']}",
        f"- Query-suite fingerprint: "
        f"{summary['stage1c_frozen_baseline']['query_suite_fingerprint']}",
        "- EN_DIRECT and VI_DIRECT regenerated: false",
        "",
        "## Translator Asset",
        "",
        f"- Model: {summary['translator']['model_id']}",
        f"- Exact revision: {summary['translator']['exact_revision']}",
        f"- Asset status: {summary['translator']['asset_status']}",
        "",
        "## Translator Runtime",
        "",
        f"- Device: {summary['translator']['device']}",
        "- Local-only: true",
        "",
        "## Translation Outputs",
        "",
        f"- Completed: {summary['translation']['queries_completed']}/"
        f"{summary['translation']['queries_requested']}",
        "",
        "## Verified CLIP Runtime",
        "",
        f"- Candidate: {summary['stage1b_encoder']['candidate_id']}",
        f"- Compatibility: {summary['stage1b_encoder']['compatibility_status']}",
        f"- Model space: {summary['stage1b_encoder']['model_space_status']}",
        "",
        "## VI-Translated Retrieval",
        "",
        f"- Completed: {summary['retrieval']['translated_queries_completed']}",
        "- Ranking: raw Stage 1A exact cosine; no reranking or diversification",
        "",
        "## Three-Arm Comparison",
        "",
        json.dumps(summary["comparison"], ensure_ascii=False),
        "",
        "## CLIP Text-Space Diagnostics",
        "",
        json.dumps(summary["comparison"]["text_space"], ensure_ascii=False),
        "",
        "## Frame Ranking Alignment",
        "",
        json.dumps(summary["comparison"]["frame_overlap"], ensure_ascii=False),
        "",
        "## Video Ranking Alignment",
        "",
        json.dumps(summary["comparison"]["video_overlap"], ensure_ascii=False),
        "",
        "## Structural Diagnostics",
        "",
        json.dumps(summary["structural_diagnostics"], ensure_ascii=False),
        "",
        "## Human Review Status",
        "",
        f"- Status: {summary['human_review']['status']}",
        f"- Expected judgments: {summary['human_review']['judgments_expected']}",
        "- Blinded: true",
        "",
        "## Pair-by-Pair Summary",
        "",
        "| Pair | Category | EN | VI | Translation | EN-VI cosine | "
        "EN-MT cosine | EN-VI frame J@20 | EN-MT frame J@20 | "
        "VI-MT frame J@20 | Review |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for item in comparisons:
        ranking = item["ranking_alignment"]
        lines.append(
            f"| {item['pair_id']} | {item['category']} | {item['en_text']} | "
            f"{item['vi_text']} | {translated[item['pair_id']]['translated_text_for_clip']} | "
            f"{item['text_space']['clip_text_cosine_en_vi']:.4f} | "
            f"{item['text_space']['clip_text_cosine_en_translated']:.4f} | "
            f"{ranking['en_vs_vi_top20_frame_jaccard']:.4f} | "
            f"{ranking['en_vs_translated_top20_frame_jaccard']:.4f} | "
            f"{ranking['vi_direct_vs_vi_translated_en']['frame']['top20']['jaccard']:.4f} | "
            "NOT_REVIEWED |"
        )
    lines.extend(
        [
            "",
            "## Non-Claims",
            "",
            *[f"- {item}" for item in summary["non_claims"]],
            "",
            "## Next Decision Gate",
            "",
            "Complete the blinded human review before deciding whether the language "
            "bridge should become an operational query path.",
        ]
    )
    return "\n".join(lines) + "\n"

