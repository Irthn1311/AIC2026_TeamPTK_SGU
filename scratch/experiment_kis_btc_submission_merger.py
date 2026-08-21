#!/usr/bin/env python3
"""KIS BTC Submission Merger & Official CSV Exporter (Patched).

Executes and merges 3-Arm retrieval (Marian P0, VinAI B1, VinAI B3) for ALL BTC KIS queries:
  - Dynamically discovers all KIS query files (*kis*.txt) in THUNGHIEM_20-8.
  - Generates full 100-candidate ranked list for every arm (Translation -> TokenBudgetGuard -> CLIP -> Phase4).
  - Primary arm Top-3 lock (primary_top_lock=3), followed by round-robin interleaving of Secondary & Hedge arms.
  - Per-query primary policies:
      • Inspected P1G1 probes:
          - p1-21: Arm C (VinAI B3) primary.
          - p1-9: Arm B (VinAI B1) primary, Arm A (Marian) secondary, Arm C (VinAI B3) last/hedge.
          - p1-2: Arm C (VinAI B3) primary.
          - p1-13, p1-17, p1-24, p1-25: Arm B (VinAI B1) primary, Arm C secondary, Arm A hedge.
          - p1-1, p1-5, p1-7, p1-10, p1-11, p1-20: Arm B (VinAI B1) primary, Arm C secondary, Arm A hedge.
      • Unseen P1G1 queries (p1-6, p1-8, p1-12, p1-14, p1-23, etc.):
          - Arm A (Marian P0) as conservative primary Top-3, Arm B secondary, Arm C hedge.
  - Strict deduplication of (video_id, frame_id) tuples.
  - Strictly caps at <= 100 rows per query.
  - Exports headerless, UTF-8 CSVs: video_id,frame_id (frame_id >= 0, video_id without .mp4).
  - Validates exact query set coverage: Expected == Generated.
  - Emits KIS_BTC_SUBMISSION_READY upon 100% verification.
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


def resolve_query_policy(qid: str) -> dict[str, str]:
    """Resolves primary, secondary, and hedge arm for any discovered KIS query."""
    # 1. P1G1-Audited Special Primary Queries
    if "p1-21" in qid:
        return {"primary": "Arm C (VinAI B3)", "secondary": "Arm B (VinAI B1)", "hedge": "Arm A (Marian P0)", "tag": "AUDITED_B3_PRIMARY"}
    if "p1-9" in qid:
        return {"primary": "Arm B (VinAI B1)", "secondary": "Arm A (Marian P0)", "hedge": "Arm C (VinAI B3)", "tag": "AUDITED_B1_PRIMARY_MARIAN_SEC"}
    if "p1-2" in qid:
        return {"primary": "Arm C (VinAI B3)", "secondary": "Arm B (VinAI B1)", "hedge": "Arm A (Marian P0)", "tag": "AUDITED_B3_PRIMARY"}
    if any(k in qid for k in ["p1-13", "p1-17", "p1-24", "p1-25"]):
        return {"primary": "Arm B (VinAI B1)", "secondary": "Arm C (VinAI B3)", "hedge": "Arm A (Marian P0)", "tag": "AUDITED_VINAI_PRIMARY"}
    if any(k in qid for k in ["p1-1", "p1-5", "p1-7", "p1-10", "p1-11", "p1-20"]):
        return {"primary": "Arm B (VinAI B1)", "secondary": "Arm C (VinAI B3)", "hedge": "Arm A (Marian P0)", "tag": "AUDITED_GUARD_B1_PRIMARY"}

    # 2. P1G1-Unseen Queries (p1-6, p1-8, p1-12, p1-14, p1-23, etc.) -> Conservative Marian P0 Primary Top-3
    return {"primary": "Arm A (Marian P0)", "secondary": "Arm B (VinAI B1)", "hedge": "Arm C (VinAI B3)", "tag": "UNSEEN_MARIAN_PRIMARY_TOP3"}


def merge_and_deduplicate_candidates(
    primary_list: list[Any],
    secondary_list: list[Any],
    hedge_list: list[Any],
    primary_arm_name: str,
    secondary_arm_name: str,
    hedge_arm_name: str,
    primary_top_lock: int = 3,
    max_total: int = 100,
) -> tuple[list[tuple[str, int, str]], int]:
    """Merges 3 full ranked lists preserving primary arm's Top-3 ranks, then interleaving hedges."""
    merged: list[tuple[str, int, str]] = []
    seen_keys: set[tuple[str, int]] = set()
    duplicates_removed = 0

    def short_tag(name: str) -> str:
        if "Marian" in name:
            return "MARIAN"
        if "B1" in name:
            return "B1"
        if "B3" in name:
            return "B3"
        return name

    # 1. Lock Primary Arm Top-3 Ranks
    for c in primary_list[:primary_top_lock]:
        vid = str(c.video_id).removesuffix(".mp4")
        fid = int(c.frame_id)
        key = (vid, fid)
        if key not in seen_keys:
            seen_keys.add(key)
            merged.append((vid, fid, f"{short_tag(primary_arm_name)}@TopLock"))
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
                merged.append((vid, fid, f"{short_tag(primary_arm_name)}@R{c.rank}"))
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
                merged.append((vid, fid, f"{short_tag(secondary_arm_name)}@R{c.rank}"))
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
                merged.append((vid, fid, f"{short_tag(hedge_arm_name)}@R{c.rank}"))
                if len(merged) >= max_total:
                    break
            else:
                duplicates_removed += 1

    return merged[:max_total], duplicates_removed


def validate_csv_file(csv_path: Path, expected_query_id: str) -> None:
    """Strictly validates submission CSV compliance against official BTC specs."""
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

    # 1. Discover all KIS Query Files from THUNGHIEM_20-8
    discovered_kis_files = sorted(list(THUNGHIEM_DIR.glob("*kis*.txt")))
    expected_kis_qids = [f.stem for f in discovered_kis_files]
    print(f"\n[Dynamic Discovery] Found {len(expected_kis_qids)} KIS Query files in {THUNGHIEM_DIR}:")
    for q in expected_kis_qids:
        print(f"  • {q}")

    # 2. Bootstrap Runtime
    print("\n[1/4] Bootstrapping OperationalKISRuntime...", flush=True)
    t0_rt = time.time()
    runtime = OperationalKISRuntime.bootstrap(cfg)
    device = runtime.shared_encoder.identifiers.get("device", "cpu")
    if torch.cuda.is_available():
        device = "cuda"
    print(f"      • Runtime Bootstrapped in {time.time() - t0_rt:.2f}s (device={device}) ✅", flush=True)

    # 3. Initialize Translators
    print("\n[2/4] Initializing Multi-Arm Translators...", flush=True)
    translator_marian = runtime.translation_provider
    translator_vinai = VinAIConfigurableTranslator(device=device)

    # 4. Output Directory for Submissions
    submission_dir = Path("/kaggle/working/submission") if Path("/kaggle/working").exists() else REPO_ROOT / "scratch" / "submission"
    submission_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 150, flush=True)
    print("EXECUTING FULL MULTI-ARM RETRIEVAL & MERGING BTC KIS SUBMISSION CSVs (Top-3 Primary Lock)", flush=True)
    print("=" * 150, flush=True)

    summary_records: list[dict[str, Any]] = []
    generated_qids: list[str] = []

    for q_file in discovered_kis_files:
        qid = q_file.stem
        q_vi = q_file.read_text(encoding="utf-8").strip()
        policy = resolve_query_policy(qid)
        primary_arm = policy["primary"]
        secondary_arm = policy["secondary"]
        hedge_arm = policy["hedge"]
        tag = policy["tag"]

        # Run Arm A (Marian P0) Full Retrieval
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

        # Run Arm B (VinAI B1) Full Retrieval
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

        # Run Arm C (VinAI B3) Full Retrieval
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

        # Merge with Policy (Top-3 Primary Lock + Interleaving + Deduplication)
        primary_cands = arm_candidates[primary_arm]
        secondary_cands = arm_candidates[secondary_arm]
        hedge_cands = arm_candidates[hedge_arm]

        merged_candidates, dups_removed = merge_and_deduplicate_candidates(
            primary_list=primary_cands,
            secondary_list=secondary_cands,
            hedge_list=hedge_cands,
            primary_arm_name=primary_arm,
            secondary_arm_name=secondary_arm,
            hedge_arm_name=hedge_arm,
            primary_top_lock=3,
            max_total=100,
        )

        # Write Official CSV (Only video_id,frame_id)
        csv_path = submission_dir / f"{qid}.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            for vid, fid, _ in merged_candidates:
                writer.writerow([vid, fid])

        # Validate CSV immediately
        validate_csv_file(csv_path, expected_query_id=qid)
        generated_qids.append(qid)

        top20_audit = " | ".join([f"@{r}:{vid}(f={fid})[{prov}]" for r, (vid, fid, prov) in enumerate(merged_candidates[:20], start=1)])

        summary_records.append({
            "qid": qid,
            "tag": tag,
            "primary": primary_arm,
            "rows": len(merged_candidates),
            "dups_removed": dups_removed,
            "top20_audit": top20_audit,
            "csv_path": csv_path,
        })

        print(f"\n[{qid}] [{tag}]")
        print(f"  • Primary Arm      : {primary_arm} (Top-3 Lock)")
        print(f"  • Rows Exported    : {len(merged_candidates)}/100 (Duplicates Removed: {dups_removed})")
        print(f"  • Top 20 Audit     : {top20_audit}")
        print(f"  • File Written     : {csv_path} (Validated ✅)")

    # 5. Exact Coverage & Integrity Verification
    missing_ids = set(expected_kis_qids) - set(generated_qids)
    extra_ids = set(generated_qids) - set(expected_kis_qids)
    total_dups_across_all = sum(s["dups_removed"] for s in summary_records)

    print("\n" + "=" * 150, flush=True)
    print("BTC KIS SUBMISSION EXPORT SUMMARY AUDIT TABLE", flush=True)
    print("=" * 150, flush=True)
    print(f"{'Query ID':<18} | {'Policy Tag':<28} | {'Primary Arm':<22} | {'Rows':<5} | {'Dups Removed':<13} | {'Validation':<10}")
    print("-" * 115)
    for s in summary_records:
        print(f"{s['qid']:<18} | {s['tag']:<28} | {s['primary']:<22} | {s['rows']:<5} | {s['dups_removed']:<13} | {'VALID ✅':<10}")
    print("=" * 150, flush=True)

    print(f"Expected KIS: {len(expected_kis_qids)}", flush=True)
    print(f"Generated KIS: {len(generated_qids)}", flush=True)
    print(f"Missing: {sorted(list(missing_ids))}", flush=True)
    print(f"Extra: {sorted(list(extra_ids))}", flush=True)
    print(f"Invalid CSV: []", flush=True)
    print(f"Total Duplicate Rows Filtered: {total_dups_across_all}", flush=True)

    assert len(missing_ids) == 0, f"FATAL: Missing KIS queries: {missing_ids}"
    assert len(extra_ids) == 0, f"FATAL: Extra KIS queries: {extra_ids}"
    assert len(expected_kis_qids) == len(generated_qids), "FATAL: Query count mismatch!"

    print("\n>>> DECLARATION: KIS_BTC_SUBMISSION_READY <<<\n", flush=True)


if __name__ == "__main__":
    run_kis_submission_merger()
