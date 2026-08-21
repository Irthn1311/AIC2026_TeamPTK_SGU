#!/usr/bin/env python3
"""KIS BTC Submission Merger & Official CSV Exporter.

Executes and merges 3-Arm retrieval (Marian P0, VinAI B1, VinAI B3) for all BTC KIS queries:
  - Preserves primary arm Top ranks before inserting hedge candidates.
  - Per-query primary prioritization:
      • p1-21: Arm C (VinAI B3) primary.
      • p1-9: Arm B (VinAI B1) / Arm A (Marian P0) primary (Arm C hedge).
      • p1-2: Arm C (VinAI B3) primary (keeps 3-6 baby tigers + rare breed).
      • p1-13, p1-17, p1-24, p1-25, p1-6: VinAI B1 / B3 primary, Marian hedge.
      • Remaining queries: VinAI B1 primary, VinAI B3 secondary, Marian hedge.
  - Deduplicates exact (video_id, frame_id) tuples.
  - Strictly caps at <= 100 rows per query.
  - Exports headerless, UTF-8 CSVs matching BTC specification: video_id,frame_id
  - Performs 100% strict schema and integrity validation.
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

print("=" * 150, flush=True)
print("KIS BTC SUBMISSION MERGER & OFFICIAL CSV EXPORTER (MARIAN P0 + VINAI B1 + VINAI B3)", flush=True)
print("=" * 150, flush=True)

REPO_ROOT = Path(__file__).resolve().parents[1]
SYSTEM_TAI_SRC = REPO_ROOT / "systems" / "system_tai" / "src"
if str(SYSTEM_TAI_SRC) not in sys.path:
    sys.path.insert(0, str(SYSTEM_TAI_SRC))

try:
    import clip
except ImportError:
    print("Installing official openai-clip dependency ...", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "git+https://github.com/openai/CLIP.git", "ftfy", "regex"], check=True)
    import clip

try:
    import cv2
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "opencv-python-headless"], check=False)
    import cv2

import torch
import transformers
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, CLIPTokenizerFast

from system_tai.kis.session_engine import OperationalKISRuntime
from system_tai.kis.session_schema import SessionConfig

VINAI_MODEL_ID = "vinai/vinai-translate-vi2en-v2"
MARIAN_MODEL_ID = "Helsinki-NLP/opus-mt-vi-en"

THUNGHIEM_DIR = REPO_ROOT / "systems" / "system_tai" / "THUNGHIEM_20-8"

# BTC KIS Query Registry with Primary Arm policy
KIS_QUERY_POLICIES = {
    "query-p1-21-kis": {"primary": "Arm C (VinAI B3)", "secondary": "Arm B (VinAI B1)", "hedge": "Arm A (Marian P0)", "desc": "Cơ chế bay bọ làm robot ở ĐH Lausanne"},
    "query-p1-9-kis": {"primary": "Arm B (VinAI B1)", "secondary": "Arm A (Marian P0)", "hedge": "Arm C (VinAI B3)", "desc": "Thu hoạch dứa ở miền Tây"},
    "query-p1-2-kis": {"primary": "Arm C (VinAI B3)", "secondary": "Arm B (VinAI B1)", "hedge": "Arm A (Marian P0)", "desc": "Đàn hổ 3-6 con hổ con"},
    "query-p1-13-kis": {"primary": "Arm B (VinAI B1)", "secondary": "Arm C (VinAI B3)", "hedge": "Arm A (Marian P0)", "desc": "Vệ sinh máy ảnh, khăn tím hồng, tăm bông"},
    "query-p1-17-kis": {"primary": "Arm B (VinAI B1)", "secondary": "Arm C (VinAI B3)", "hedge": "Arm A (Marian P0)", "desc": "Trao quà từ thiện bệnh viện Xuân 2024 COVID-19"},
    "query-p1-24-kis": {"primary": "Arm B (VinAI B1)", "secondary": "Arm C (VinAI B3)", "hedge": "Arm A (Marian P0)", "desc": "Đua xe đạp góc quay trực diện từ trên cao"},
    "query-p1-25-kis": {"primary": "Arm B (VinAI B1)", "secondary": "Arm C (VinAI B3)", "hedge": "Arm A (Marian P0)", "desc": "Đua xe đạp flycam trên cao áo xanh vượt 3"},
    "query-p1-6-kis": {"primary": "Arm B (VinAI B1)", "secondary": "Arm C (VinAI B3)", "hedge": "Arm A (Marian P0)", "desc": "Gỏi cuốn chay bánh tráng tím vàng"},
    "query-p1-1-kis": {"primary": "Arm B (VinAI B1)", "secondary": "Arm C (VinAI B3)", "hedge": "Arm A (Marian P0)", "desc": "Phóng tàu vũ trụ / 4 phi hành gia áo đen"},
    "query-p1-5-kis": {"primary": "Arm B (VinAI B1)", "secondary": "Arm C (VinAI B3)", "hedge": "Arm A (Marian P0)", "desc": "Hai người phụ nữ cho dê ăn"},
    "query-p1-7-kis": {"primary": "Arm B (VinAI B1)", "secondary": "Arm C (VinAI B3)", "hedge": "Arm A (Marian P0)", "desc": "Chú chim lông đen ánh xanh cổ"},
    "query-p1-8-kis": {"primary": "Arm B (VinAI B1)", "secondary": "Arm C (VinAI B3)", "hedge": "Arm A (Marian P0)", "desc": "Hai mẹ con tập đi bộ trong phòng"},
    "query-p1-10-kis": {"primary": "Arm B (VinAI B1)", "secondary": "Arm C (VinAI B3)", "hedge": "Arm A (Marian P0)", "desc": "Chơi nhạc cụ kim loại tròn (Handpan)"},
    "query-p1-11-kis": {"primary": "Arm B (VinAI B1)", "secondary": "Arm C (VinAI B3)", "hedge": "Arm A (Marian P0)", "desc": "Đổ bóng tạo chân dung mặc vest"},
    "query-p1-12-kis": {"primary": "Arm B (VinAI B1)", "secondary": "Arm C (VinAI B3)", "hedge": "Arm A (Marian P0)", "desc": "Chế biến nấm xào ngô cải thảo"},
    "query-p1-14-kis": {"primary": "Arm B (VinAI B1)", "secondary": "Arm C (VinAI B3)", "hedge": "Arm A (Marian P0)", "desc": "Xe cứu thương trong đêm"},
    "query-p1-20-kis": {"primary": "Arm B (VinAI B1)", "secondary": "Arm C (VinAI B3)", "hedge": "Arm A (Marian P0)", "desc": "Thêm 2 ly panna cotta, hoa ăn được"},
    "query-p1-23-kis": {"primary": "Arm B (VinAI B1)", "secondary": "Arm C (VinAI B3)", "hedge": "Arm A (Marian P0)", "desc": "Người phụ nữ may vá máy khâu"},
}


class VinAIConfigurableTranslator:
    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self.model_id = VINAI_MODEL_ID
        print(f"\n[Loading VinAI Model '{self.model_id}' on {device}...]", flush=True)
        t0 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, src_lang="vi_VN", use_fast=False)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_id,
            low_cpu_mem_usage=False,
            dtype=torch.float32,
        ).to(device)
        self.model.eval()
        self.load_time = time.time() - t0
        self.forced_bos_id = self.tokenizer.lang_code_to_id.get("en_XX") if hasattr(self.tokenizer, "lang_code_to_id") else None
        print(f"      • Loaded VinAI in {self.load_time:.2f}s ✅", flush=True)

    def translate(self, text_vi: str, gen_params: dict[str, Any]) -> str:
        inputs = self.tokenizer(text_vi, return_tensors="pt", padding=True).to(self.device)
        params = dict(gen_params)
        if self.forced_bos_id is not None:
            params["forced_bos_token_id"] = self.forced_bos_id
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **params)
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


VINAI_ARM_CONFIGS = {
    "Arm B (VinAI B1)": {
        "num_beams": 3,
        "no_repeat_ngram_size": 3,
        "repetition_penalty": 1.15,
        "max_new_tokens": 256,
        "early_stopping": True,
    },
    "Arm C (VinAI B3)": {
        "num_beams": 4,
        "no_repeat_ngram_size": 3,
        "repetition_penalty": 1.05,
        "max_new_tokens": 256,
        "early_stopping": True,
    },
}


def get_reuse_manifest() -> Path | None:
    for p in [
        Path("/kaggle/working/manifest_cache.json"),
        Path("/kaggle/input/system-tai-manifest/feature_manifest.json"),
        Path("/kaggle/input/datasets/manifest_cache.json"),
        Path("/kaggle/input/manifest_cache.json"),
        REPO_ROOT / "systems" / "system_tai" / "data" / "feature_manifest.json",
    ]:
        if p.exists() and p.stat().st_size > 1000:
            return p
    return None


def merge_and_deduplicate_candidates(
    primary_list: list[Any],
    secondary_list: list[Any],
    hedge_list: list[Any],
    primary_top_lock: int = 15,
    max_total: int = 100,
) -> tuple[list[tuple[str, int, str]], int]:
    """Merges 3 ranked lists preserving primary arm's top ranks, then interleaving hedges."""
    merged: list[tuple[str, int, str]] = []
    seen_keys: set[tuple[str, int]] = set()
    duplicates_removed = 0

    # 1. Lock Primary Arm Top Ranks
    for c in primary_list[:primary_top_lock]:
        vid = str(c.video_id).removesuffix(".mp4")
        fid = int(c.frame_id)
        key = (vid, fid)
        if key not in seen_keys:
            seen_keys.add(key)
            merged.append((vid, fid, "Primary_TopLock"))
        else:
            duplicates_removed += 1

    # 2. Interleave remaining candidates from Primary, Secondary, Hedge
    max_len = max(len(primary_list), len(secondary_list), len(hedge_list))
    for i in range(max_len):
        if len(merged) >= max_total:
            break
        # Pick from Primary (beyond lock)
        if i >= primary_top_lock and i < len(primary_list):
            c = primary_list[i]
            vid = str(c.video_id).removesuffix(".mp4")
            fid = int(c.frame_id)
            key = (vid, fid)
            if key not in seen_keys:
                seen_keys.add(key)
                merged.append((vid, fid, "Primary_Interleaved"))
                if len(merged) >= max_total:
                    break
            else:
                duplicates_removed += 1

        # Pick from Secondary
        if i < len(secondary_list):
            c = secondary_list[i]
            vid = str(c.video_id).removesuffix(".mp4")
            fid = int(c.frame_id)
            key = (vid, fid)
            if key not in seen_keys:
                seen_keys.add(key)
                merged.append((vid, fid, "Secondary_Interleaved"))
                if len(merged) >= max_total:
                    break
            else:
                duplicates_removed += 1

        # Pick from Hedge
        if i < len(hedge_list):
            c = hedge_list[i]
            vid = str(c.video_id).removesuffix(".mp4")
            fid = int(c.frame_id)
            key = (vid, fid)
            if key not in seen_keys:
                seen_keys.add(key)
                merged.append((vid, fid, "Hedge_Interleaved"))
                if len(merged) >= max_total:
                    break
            else:
                duplicates_removed += 1

    return merged[:max_total], duplicates_removed


def validate_csv_file(csv_path: Path, expected_query_id: str) -> None:
    """Strictly validates submission CSV compliance."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing submission CSV: {csv_path}")
    
    content = csv_path.read_text(encoding="utf-8")
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    
    if len(lines) == 0:
        raise ValueError(f"Empty submission CSV: {csv_path}")
    if len(lines) > 100:
        raise ValueError(f"Submission CSV exceeds 100 rows ({len(lines)} rows): {csv_path}")
    
    seen_keys: set[tuple[str, int]] = set()
    for row_idx, line in enumerate(lines, start=1):
        parts = line.split(",")
        if len(parts) != 2:
            raise ValueError(f"Invalid column count ({len(parts)}) at line {row_idx} in {csv_path}: '{line}'")
        vid, fid_str = parts[0].strip(), parts[1].strip()
        if vid.endswith(".mp4"):
            raise ValueError(f"Video ID contains forbidden '.mp4' extension at line {row_idx} in {csv_path}: {vid}")
        if not vid:
            raise ValueError(f"Empty video ID at line {row_idx} in {csv_path}")
        try:
            fid = int(fid_str)
            if fid < 0:
                raise ValueError(f"Negative frame ID at line {row_idx}: {fid}")
        except ValueError:
            raise ValueError(f"Invalid integer frame ID at line {row_idx} in {csv_path}: '{fid_str}'")
        
        key = (vid, fid)
        if key in seen_keys:
            raise ValueError(f"Duplicate record at line {row_idx} in {csv_path}: {key}")
        seen_keys.add(key)


def run_kis_submission_merger() -> None:
    yaml_path = REPO_ROOT / "systems" / "system_tai" / "configs" / "production.yaml"
    input_root = Path("/kaggle/input/datasets") if Path("/kaggle/input/datasets").exists() else Path("/kaggle/input")
    reuse_manifest = get_reuse_manifest()
    out_dir = Path("/kaggle/working/output/kis_submission_session") if Path("/kaggle/working").exists() else REPO_ROOT / "scratch" / "kis_submission_session"

    cfg = SessionConfig.from_yaml(
        yaml_path,
        input_root=input_root,
        output_root=out_dir,
        reuse_manifest=reuse_manifest,
    )

    # 1. Bootstrap Runtime
    print("\n[1/4] Bootstrapping OperationalKISRuntime...", flush=True)
    t0_rt = time.time()
    runtime = OperationalKISRuntime.bootstrap(cfg)
    device = runtime.shared_encoder.identifiers.get("device", "cpu")
    if torch.cuda.is_available():
        device = "cuda"
    print(f"      • Runtime Bootstrapped in {time.time() - t0_rt:.2f}s (device={device}) ✅", flush=True)

    # 2. Initialize Translators
    print("\n[2/4] Initializing Multi-Arm Translators...", flush=True)
    translator_marian = runtime.translation_provider
    translator_vinai = VinAIConfigurableTranslator(device=device)

    # 3. Output Directory for Submissions
    submission_dir = Path("/kaggle/working/submission") if Path("/kaggle/working").exists() else REPO_ROOT / "scratch" / "submission"
    submission_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 150, flush=True)
    print("EXECUTING MULTI-ARM RETRIEVAL & MERGING BTC KIS SUBMISSION CSVs", flush=True)
    print("=" * 150, flush=True)

    summary_records: list[dict[str, Any]] = []

    for qid, policy in KIS_QUERY_POLICIES.items():
        q_file = THUNGHIEM_DIR / f"{qid}.txt"
        if not q_file.exists():
            print(f"Warning: {q_file} not found, skipping.", flush=True)
            continue

        q_vi = q_file.read_text(encoding="utf-8").strip()
        primary_arm = policy["primary"]
        secondary_arm = policy["secondary"]
        hedge_arm = policy["hedge"]
        desc = policy["desc"]

        # Run Arm A (Marian)
        raw_en_a = translator_marian.translate(q_vi).strip()
        eff_en_a, _, _ = runtime.token_budget_guard.guard_and_compact(raw_en_a)
        vec_a = runtime.shared_encoder.encode(eff_en_a)
        coarse_a = runtime.exact_retriever.search_vector(query_id=f"a-{qid}", query_vector=vec_a, top_k=100)
        final_a = runtime.video_conditioner.condition(
            global_result=coarse_a,
            query_vector=vec_a,
            config=runtime.config.video_conditioned_keyframe_config,
            protected_prefix_rank=1,
        ).result.ranked_candidates

        # Run Arm B (VinAI B1)
        raw_en_b = translator_vinai.translate(q_vi, VINAI_ARM_CONFIGS["Arm B (VinAI B1)"])
        eff_en_b, _, _ = runtime.token_budget_guard.guard_and_compact(raw_en_b)
        vec_b = runtime.shared_encoder.encode(eff_en_b)
        coarse_b = runtime.exact_retriever.search_vector(query_id=f"b-{qid}", query_vector=vec_b, top_k=100)
        final_b = runtime.video_conditioner.condition(
            global_result=coarse_b,
            query_vector=vec_b,
            config=runtime.config.video_conditioned_keyframe_config,
            protected_prefix_rank=1,
        ).result.ranked_candidates

        # Run Arm C (VinAI B3)
        raw_en_c = translator_vinai.translate(q_vi, VINAI_ARM_CONFIGS["Arm C (VinAI B3)"])
        eff_en_c, _, _ = runtime.token_budget_guard.guard_and_compact(raw_en_c)
        vec_c = runtime.shared_encoder.encode(eff_en_c)
        coarse_c = runtime.exact_retriever.search_vector(query_id=f"c-{qid}", query_vector=vec_c, top_k=100)
        final_c = runtime.video_conditioner.condition(
            global_result=coarse_c,
            query_vector=vec_c,
            config=runtime.config.video_conditioned_keyframe_config,
            protected_prefix_rank=1,
        ).result.ranked_candidates

        arm_candidates = {
            "Arm A (Marian P0)": list(final_a),
            "Arm B (VinAI B1)": list(final_b),
            "Arm C (VinAI B3)": list(final_c),
        }

        # Merge with Policy
        primary_cands = arm_candidates[primary_arm]
        secondary_cands = arm_candidates[secondary_arm]
        hedge_cands = arm_candidates[hedge_arm]

        merged_candidates, dups_removed = merge_and_deduplicate_candidates(
            primary_list=primary_cands,
            secondary_list=secondary_cands,
            hedge_list=hedge_cands,
            primary_top_lock=15,
            max_total=100,
        )

        # Write Official CSV
        csv_path = submission_dir / f"{qid}.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            for vid, fid, _ in merged_candidates:
                writer.writerow([vid, fid])

        # Validate CSV immediately
        validate_csv_file(csv_path, expected_query_id=qid)

        top3_preview = ", ".join([f"@{r}:{vid}(f={fid})" for r, (vid, fid, _) in enumerate(merged_candidates[:3], start=1)])

        summary_records.append({
            "qid": qid,
            "desc": desc,
            "primary": primary_arm,
            "rows": len(merged_candidates),
            "dups_removed": dups_removed,
            "top3": top3_preview,
            "csv_path": csv_path,
        })

        print(f"\n[{qid}] ({desc})")
        print(f"  • Primary Arm      : {primary_arm}")
        print(f"  • Rows Exported    : {len(merged_candidates)}/100 (Duplicates Removed: {dups_removed})")
        print(f"  • Merged Top 3     : [{top3_preview}]")
        print(f"  • File Written     : {csv_path} (Validated ✅)")

    # 4. Summary Audit Table
    print("\n" + "=" * 150, flush=True)
    print("BTC KIS SUBMISSION EXPORT SUMMARY AUDIT TABLE", flush=True)
    print("=" * 150, flush=True)
    print(f"{'Query ID':<18} | {'Primary Arm':<22} | {'Rows':<5} | {'Dups Removed':<13} | {'Merged Top 3 Preview':<55} | {'Validation':<10}")
    print("-" * 140)
    for s in summary_records:
        print(f"{s['qid']:<18} | {s['primary']:<22} | {s['rows']:<5} | {s['dups_removed']:<13} | {s['top3']:<55} | {'VALID ✅':<10}")
    print("=" * 150, flush=True)
    print("\n>>> DECLARATION: KIS_BTC_SUBMISSION_READY <<<\n", flush=True)


if __name__ == "__main__":
    run_kis_submission_merger()
