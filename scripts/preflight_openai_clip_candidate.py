#!/usr/bin/env python3
"""Preflight local official OpenAI CLIP assets without downloading anything."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from triage_eg.retrieval.stage1b.adapters.openai_clip_official import (
    NetworkDownloadAttempted,
    OfficialOpenAIClipAdapter,
    preflight_official_openai_clip,
    resolve_official_asset_paths,
)
from triage_eg.retrieval.stage1b.contracts import CandidateContract


def _contract(paths, device: str) -> CandidateContract:
    return CandidateContract(
        candidate_id="openai_clip_vit_b32_openai_official",
        enabled=True,
        implementation="openai_clip",
        architecture="ViT-B/32",
        pretrained="openai",
        source_root=str(paths.source_root),
        checkpoint_path=str(paths.checkpoint_path),
        asset_manifest_path=str(paths.asset_manifest_path),
        tokenizer="official clip.tokenize",
        context_length=77,
        image_preprocessing={
            "source": "official_clip_load_return_value",
            "manual_preprocess_override": False,
        },
        text_preprocessing={
            "strip": False,
            "lowercase": False,
            "unicode_normalization": None,
        },
        device=device,
        evidence_source="HYPOTHESIS",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--load-model", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    paths = resolve_official_asset_paths(args.asset_root, args.source_root, args.checkpoint)
    provenance, issues, module = preflight_official_openai_clip(paths, requested_device=args.device)
    model_load_status = "NOT_REQUESTED"
    blockers = [item for item in issues if item["severity"] == "ERROR"]
    if args.load_model and blockers:
        model_load_status = (
            "BLOCKED_DEPENDENCY"
            if any(item["code"] == "ENCODER_DEPENDENCY_NOT_AVAILABLE" for item in blockers)
            else "BLOCKED_PREFLIGHT"
        )
    elif args.load_model:
        try:
            adapter = OfficialOpenAIClipAdapter(
                _contract(paths, args.device), paths, module, provenance
            )
            adapter.close()
            model_load_status = "PASS"
        except NetworkDownloadAttempted as error:
            model_load_status = "BLOCKED"
            issues.append(
                {
                    "severity": "ERROR",
                    "code": "NETWORK_DOWNLOAD_ATTEMPTED",
                    "candidate_id": "openai_clip_vit_b32_openai_official",
                    "global_row": None,
                    "video_id": None,
                    "path": str(paths.checkpoint_path),
                    "message": str(error),
                    "evidence": {},
                }
            )
        except Exception as error:
            model_load_status = "FAIL"
            issues.append(
                {
                    "severity": "ERROR",
                    "code": "ENCODER_CHECKPOINT_LOAD_FAILED",
                    "candidate_id": "openai_clip_vit_b32_openai_official",
                    "global_row": None,
                    "video_id": None,
                    "path": str(paths.checkpoint_path),
                    "message": str(error),
                    "evidence": {},
                }
            )
    ready = not any(item["severity"] == "ERROR" for item in issues)
    print(
        json.dumps(
            {
                "ready": ready,
                "model_load_status": model_load_status,
                "provenance": provenance,
                "issues": issues,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 2 if args.strict and not ready else 0


if __name__ == "__main__":
    raise SystemExit(main())
