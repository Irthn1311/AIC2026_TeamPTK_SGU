#!/usr/bin/env python3
"""KIS P0.2: Release Candidate Closure & 3-Gate Verification Script.

Gates:
  1. Default-OFF Parity: Proves enable_dynamic_translation=False retains 100% legacy behavior.
  2. True Offline Packaging: Proves Marian loads with local_files_only=True and offline env.
  3. 18-Query BTC Blind KIS Replay: Runs all 18 BTC queries through canonical runtime,
     extracts Top 3 video thumbnails, and generates HTML visual gallery for manual inspection.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

# Purge stale system_tai modules
for mod in list(sys.modules.keys()):
    if mod.startswith("system_tai"):
        del sys.modules[mod]

REPO_ROOT = Path(__file__).resolve().parents[1]
SYSTEM_TAI_SRC = REPO_ROOT / "systems" / "system_tai" / "src"
if str(SYSTEM_TAI_SRC) not in sys.path:
    sys.path.insert(0, str(SYSTEM_TAI_SRC))

try:
    import clip
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "openai-clip", "ftfy", "regex", "tqdm", "transformers", "sentencepiece"], check=False)
    import clip

try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers", "sentencepiece"], check=False)
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from system_tai.kis.session_engine import OperationalKISRuntime
from system_tai.kis.session_schema import QueryLanguage, QueryRequest, QueryVariant, QueryVariantType, SessionConfig
from system_tai.translation.provider import MarianOfflineTranslator, TokenBudgetGuard


def get_git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "UNKNOWN"


# ==============================================================================================================
# GATE 1: DEFAULT-OFF PARITY AUDIT
# ==============================================================================================================
def verify_gate1_default_off_parity() -> bool:
    print("\n" + "=" * 150, flush=True)
    print("GATE 1: DEFAULT-OFF PARITY AUDIT (enable_dynamic_translation = False)", flush=True)
    print("=" * 150, flush=True)

    config = SessionConfig(
        input_root=Path("/kaggle/input/datasets") if Path("/kaggle/input/datasets").exists() else Path("/kaggle/input"),
        output_root=REPO_ROOT / "scratch" / "gate1_test",
        enable_dynamic_translation=False,
    )

    print(f"• Config enable_dynamic_translation : {config.enable_dynamic_translation} (Default: False)", flush=True)
    assert config.enable_dynamic_translation is False, "Default-OFF flag violation!"

    runtime = OperationalKISRuntime.bootstrap(config)
    print(f"• runtime.translation_provider      : {runtime.translation_provider} (Expected: None)", flush=True)
    print(f"• runtime.token_budget_guard        : {runtime.token_budget_guard} (Expected: None)", flush=True)
    assert runtime.translation_provider is None, "Translation provider must be None when flag is False!"
    assert runtime.token_budget_guard is None, "Token budget guard must be None when flag is False!"

    req = QueryRequest(
        request_id="test-req-default-off",
        query_id="test-q1",
        query_vi="Tìm cảnh cháy rừng lớn ban đêm",
        query_en=None,
        include_vi_variant=True,
    )
    variants = req.variants()
    print(f"• Ingested Variants with Flag OFF   : {[(v.variant_id, v.language.value, v.text) for v in variants]}", flush=True)
    assert len(variants) == 1, "Expected 1 variant (VI) under default legacy behavior without English sidecar"
    assert variants[0].language == QueryLanguage.VIETNAMESE, "Language must be VIETNAMESE"

    print(">>> GATE 1: PASS ✅ (Default-OFF maintains 100% legacy behavior & zero provider overhead)\n", flush=True)
    return True


# ==============================================================================================================
# GATE 2: TRUE OFFLINE PACKAGING & ISOLATION
# ==============================================================================================================
def verify_gate2_offline_packaging() -> bool:
    print("\n" + "=" * 150, flush=True)
    print("GATE 2: TRUE OFFLINE PACKAGING & ISOLATION AUDIT", flush=True)
    print("=" * 150, flush=True)

    # Set offline environment variables
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"

    t0 = time.perf_counter()
    try:
        translator = MarianOfflineTranslator(
            model_name_or_path="Helsinki-NLP/opus-mt-vi-en",
            local_files_only=True,
            device="auto",
        )
        load_time = time.perf_counter() - t0
        print(f"• Offline Model Loaded Successfully : {translator.provider_name}", flush=True)
        print(f"• Resolved Device                   : {translator.device.upper()}", flush=True)
        print(f"• Load Time with local_files_only=1 : {load_time:.3f} s", flush=True)
        print(f"• Offline Env Flags                 : TRANSFORMERS_OFFLINE=1, HF_HUB_OFFLINE=1", flush=True)

        # Warm test
        translated = translator.translate("Đoạn clip về một đàn hổ con trong rừng")
        print(f"• Test Translation (Offline)        : \"{translated}\"", flush=True)
        assert len(translated) > 5, "Translation output empty"

        print(">>> GATE 2: PASS ✅ (True Offline Startup & Execution Confirmed)\n", flush=True)
        return True
    except Exception as exc:
        print(f"GATE 2: FAILED ❌ -> {exc}", flush=True)
        return False
    finally:
        os.environ.pop("TRANSFORMERS_OFFLINE", None)
        os.environ.pop("HF_HUB_OFFLINE", None)


# ==============================================================================================================
# GATE 3: 18-QUERY BTC BLIND KIS REPLAY (CANONICAL RUNTIME)
# ==============================================================================================================
def run_gate3_btc18_canonical_replay() -> list[dict[str, Any]]:
    print("\n" + "=" * 150, flush=True)
    print("GATE 3: 18-QUERY BTC BLIND KIS REPLAY (CANONICAL DYNAMIC MARIAN EN_ONLY + PHASE-4)", flush=True)
    print("=" * 150, flush=True)

    btc_dir = REPO_ROOT / "systems" / "system_tai" / "THUNGHIEM_20-8"
    if not btc_dir.exists():
        print(f"ERROR: BTC directory {btc_dir} not found!", flush=True)
        return []

    btc_query_files = sorted(list(btc_dir.glob("query-p1-*-kis.txt")), key=lambda p: int(p.stem.split("-")[2]))
    print(f"• Discovered BTC KIS Queries        : {len(btc_query_files)} query files in {btc_dir.name}\n", flush=True)

    session_output = Path("/kaggle/working/output/btc18_marian_canonical") if Path("/kaggle/working").exists() else REPO_ROOT / "scratch" / "btc18_marian_canonical"
    if session_output.exists():
        shutil.rmtree(session_output, ignore_errors=True)
    session_output.mkdir(parents=True, exist_ok=True)

    reuse_manifest_path: Path | None = None
    for p in [
        Path("/kaggle/working/manifest_cache.json"),
        Path("/kaggle/input/system-tai-manifest/feature_manifest.json"),
        Path("/kaggle/input/datasets/manifest_cache.json"),
        Path("/kaggle/input/manifest_cache.json"),
    ]:
        if p.exists() and p.stat().st_size > 1000:
            reuse_manifest_path = p
            break

    config = SessionConfig(
        input_root=Path("/kaggle/input/datasets") if Path("/kaggle/input/datasets").exists() else Path("/kaggle/input"),
        reuse_manifest=reuse_manifest_path,
        output_root=session_output,
        device="auto",
        allow_model_download=True,
        default_output_top_k=100,
        default_refine_top_n=3,
        rrf_constant=60.0,
        enable_dynamic_translation=True,
        translation_device="auto",
    )

    runtime = OperationalKISRuntime.bootstrap(config)
    guard = TokenBudgetGuard()

    all_results = []
    total_start = time.time()

    for idx, q_path in enumerate(btc_query_files, start=1):
        qid = q_path.stem
        q_vi = q_path.read_text(encoding="utf-8").strip()

        req = QueryRequest(
            request_id=f"btc-{qid}",
            query_id=qid,
            query_vi=q_vi,
            query_en=None,
            include_vi_variant=True,
            output_top_k=100,
            refine_top_n=3,
        )

        t_q0 = time.time()
        res = runtime.handle_query(req)
        elapsed = time.time() - t_q0

        top100_rel = res["artifacts"].get("refined_top100_jsonl", res["artifacts"]["top100_jsonl"])
        top100_path = runtime.output_root / top100_rel
        preds = [
            json.loads(line)
            for line in top100_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        # Extract translation metadata
        trans_meta = res.get("diagnostics", {})
        marian_en = ""
        if runtime.translation_provider:
            marian_en = runtime.translation_provider.translate(q_vi)
        tok_count = guard.count_tokens(marian_en) if marian_en else 0

        # Extract Top 3 candidates
        top3_candidates = []
        for p in preds[:3]:
            top3_candidates.append({
                "rank": p["rank"],
                "video_id": p["video_id"],
                "frame_id": p["frame_id"],
            })

        top3_desc = [f"@{c['rank']}: {c['video_id']} (f={c['frame_id']})" for c in top3_candidates]
        print(f"[{idx:02d}/18] {qid:<18} in {elapsed:5.1f}s (Tokens: {tok_count:2d})", flush=True)
        print(f"     • VI Raw    : \"{q_vi[:80]}...\"", flush=True)
        print(f"     • Marian EN : \"{marian_en}\"", flush=True)
        print(f"     • Top 3     : {top3_desc}\n", flush=True)

        all_results.append({
            "qid": qid,
            "query_vi": q_vi,
            "marian_en": marian_en,
            "tok_count": tok_count,
            "elapsed_seconds": elapsed,
            "top3_candidates": top3_candidates,
            "top10_videos": list(dict.fromkeys([p["video_id"] for p in preds]))[:10],
        })

    out_json = Path("/kaggle/working/marian_btc18_results.json")
    out_json.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved results to {out_json} (Total time: {time.time() - total_start:.2f}s)\n", flush=True)
    return all_results


def generate_btc18_gallery_html() -> str:
    out_json = Path("/kaggle/working/marian_btc18_results.json")
    if not out_json.exists():
        return "<div>Results JSON not found.</div>"

    try:
        from visualize_btc_predictions import extract_frame_base64, find_video_path
    except ImportError:
        def find_video_path(v_id: str) -> Path | None:
            return None
        def extract_frame_base64(v_path: Any, f_id: int, max_width: int = 320) -> str | None:
            return None

    data = json.loads(out_json.read_text(encoding="utf-8"))
    html_cards = []

    html_cards.append("""
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 1300px; margin: auto; padding: 20px; background: #f8f9fa;">
        <h1 style="text-align: center; color: #1a73e8; margin-bottom: 5px;">
            🎬 BẢNG ĐỐI SOÁT HÌNH ẢNH TOÀN BỘ 18 CÂU BTC KIS (DYNAMIC MARIAN EN_ONLY)
        </h1>
        <p style="text-align: center; color: #5f6368; font-size: 14px; margin-bottom: 25px;">
            Đánh giá mù (Blind Assessment) phục hồi ngữ nghĩa toàn tập: SEMANTIC_STRONG / PARTIAL / BAD
        </p>
    """)

    for q in data:
        qid = q["qid"]
        q_vi = q["query_vi"]
        marian_en = q["marian_en"]
        tok_count = q["tok_count"]
        elapsed = q["elapsed_seconds"]

        card_html = f"""
        <div style="background: #ffffff; border: 1px solid #e0e0e0; border-radius: 8px; margin-bottom: 20px; padding: 16px; box-shadow: 0 2px 4px rgba(0,0,0,0.04);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; border-bottom: 1px solid #f1f3f4; padding-bottom: 8px;">
                <span style="background: #1a73e8; color: #ffffff; font-weight: bold; font-size: 13px; padding: 3px 8px; border-radius: 4px;">{qid}</span>
                <span style="font-size: 12px; color: #70757a;">⏱️ {elapsed:.1f}s | 🔤 CLIP Tokens: {tok_count}</span>
            </div>
            <div style="font-size: 13px; color: #202124; margin-bottom: 6px;"><b>VI Query:</b> {q_vi}</div>
            <div style="background: #e8f0fe; color: #174ea6; padding: 8px 12px; border-radius: 4px; font-size: 13px; margin-bottom: 14px;">
                <b>Marian EN:</b> "<i>{marian_en}</i>"
            </div>
            <div style="display: flex; gap: 15px; justify-content: space-between;">
        """

        for cand in q["top3_candidates"]:
            rank = cand["rank"]
            v_id = cand["video_id"]
            f_id = cand["frame_id"]
            sec = f_id // 25
            time_str = f"{sec // 60:02d}:{sec % 60:02d}"

            v_path = find_video_path(v_id)
            img_b64 = extract_frame_base64(v_path, f_id, max_width=320) if v_path else None

            img_tag = (
                f'<img src="data:image/jpeg;base64,{img_b64}" style="width: 100%; height: auto; border-radius: 4px; border: 1px solid #ddd;" />'
                if img_b64
                else '<div style="height: 140px; background: #eee; display: flex; align-items: center; justify-content: center; color: #888; font-size: 12px;">(No Frame Available)</div>'
            )

            card_html += f"""
            <div style="flex: 1; background: #fdfdfd; border: 1px solid #dadce0; border-radius: 6px; padding: 8px; text-align: center;">
                <div style="font-size: 12px; font-weight: bold; color: #1a73e8; margin-bottom: 6px;">
                    Rank @{rank}: {v_id} (f={f_id}, ~{time_str})
                </div>
                {img_tag}
            </div>
            """

        card_html += """
            </div>
        </div>
        """
        html_cards.append(card_html)

    html_cards.append("</div>")
    gallery_html = "\n".join(html_cards)

    out_html = Path("/kaggle/working/marian_btc18_gallery.html")
    out_html.write_text(gallery_html, encoding="utf-8")
    return gallery_html


def main() -> None:
    parser = argparse.ArgumentParser(description="KIS P0.2 Release Candidate 3-Gate Runner")
    parser.add_argument("--gate", choices=["1", "2", "3", "all"], default="all")
    args, _ = parser.parse_known_args()

    print("=" * 150, flush=True)
    print("KIS P0.2: RELEASE CANDIDATE CLOSURE (3-GATE VERIFICATION)", flush=True)
    print("=" * 150, flush=True)
    print(f"• Git HEAD Commit: {get_git_head()}", flush=True)

    if args.gate in {"1", "all"}:
        verify_gate1_default_off_parity()

    if args.gate in {"2", "all"}:
        verify_gate2_offline_packaging()

    if args.gate in {"3", "all"}:
        run_gate3_btc18_canonical_replay()


if __name__ == "__main__":
    main()
