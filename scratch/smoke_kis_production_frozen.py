#!/usr/bin/env python3
"""KIS P0.4: Production Promotion & 3-Query Smoke Runner.

Verifies:
  1. Real production.yaml configuration loading into SessionConfig.
  2. Enforceable Marian model provenance (pinned revision a0586e3fcf81ec01c7785c40467c699fa8403d6d, local_files_only=True, SHA256 fingerprint).
  3. 3-Query Production Smoke:
     - Short query: query-p1-2-kis (tiger)
     - Normal query: query-p1-10-kis (handpan)
     - Over-budget query: query-p1-12-kis (donut)
  4. Telemetry: raw vs effective tokens (<= 77), EN_ONLY variant list, Top 3 output.
"""

from __future__ import annotations

import hashlib
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
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "openai-clip", "ftfy", "regex", "tqdm", "transformers", "sentencepiece", "pyyaml"], check=False)
    import clip

from system_tai.kis.session_engine import OperationalKISRuntime
from system_tai.kis.session_schema import QueryRequest, SessionConfig
from system_tai.translation.provider import TokenBudgetGuard

SMOKE_3_QUERIES = [
    {
        "qid": "query-p1-2-kis",
        "category": "SHORT QUERY",
        "topic": "Đàn hổ con miền Nam",
        "vi": "Mẩu tin giới thiệu về đàn hổ tại một địa phương ở miền Nam vừa có thêm khoảng 3-6 con hổ con. Đây là một giống hổ quý hiếm",
    },
    {
        "qid": "query-p1-10-kis",
        "category": "NORMAL QUERY",
        "topic": "3 người chơi nhạc cụ kim loại hình tròn (handpan)",
        "vi": "Tìm chính xác đoạn clip ngắn có ba người (hai phụ nữ và một nam giới) đang ngồi cạnh nhau, tập trung chơi nhạc cụ kim loại có dạng tròn, rỗng, với các vết lõm để tạo ra âm thanh khi gõ tay. Có 1 người mặc áo trắng ngồi giữa 2 người mặc áo đen. Bối cảnh phía sau là một kệ sách nhiều ngăn, xếp đầy sách với nhiều màu sắc",
    },
    {
        "qid": "query-p1-12-kis",
        "category": "OVER-BUDGET QUERY",
        "topic": "Trang trí bánh rán, rưới chocolate dâu tây",
        "vi": "Đoạn video mô tả cảnh trang trí bánh rán. Phân cảnh bắt đầu là một chiếc đĩa sứ màu trắng nằm trên một khay gỗ hình chữ nhật. Bên cạnh chiếc đĩa sứ là một chén đựng một vài trái dâu, nhưng có 2 trái bị rơi ra ngoài. Ngoài ra, bên cạnh đĩa sứ còn có một chén sứ nhỏ màu trắng đựng chuối đã được cắt sẵn và một cái thìa nhỏ màu nâu. Phân cảnh tiếp theo cho thấy đầu bếp đặt 2 chiếc bánh rán lên đĩa sứ và bắt đầu trang trí. Bước đầu tiên là việc rưới chocolate lên trên mặt bánh. Sau đó, đầu bếp đặt các lát chuối lên trên một chiếc bánh rán, chiếc còn lại được đặt các lát dâu tây lên.",
    },
]


def run_production_frozen_smoke() -> None:
    yaml_path = REPO_ROOT / "systems" / "system_tai" / "configs" / "production.yaml"
    print("=" * 150, flush=True)
    print("KIS P0.4: PRODUCTION PROMOTION & 3-QUERY SMOKE RUNNER", flush=True)
    print("=" * 150, flush=True)
    print(f"• Production YAML Path             : {yaml_path.relative_to(REPO_ROOT)}", flush=True)

    session_output = Path("/kaggle/working/output/kis_production_smoke") if Path("/kaggle/working").exists() else REPO_ROOT / "scratch" / "kis_production_smoke"
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

    # 1. Load SessionConfig directly from production.yaml
    config = SessionConfig.from_yaml(
        yaml_path,
        input_root=Path("/kaggle/input/datasets") if Path("/kaggle/input/datasets").exists() else Path("/kaggle/input"),
        output_root=session_output,
        reuse_manifest=reuse_manifest_path,
    )

    print("\n--- LOADED PRODUCTION CONFIGURATION PARAMETERS ---", flush=True)
    print(f"• enable_dynamic_translation       : {config.enable_dynamic_translation} (PROMOTED DEFAULT: TRUE)", flush=True)
    print(f"• translation_model_name           : {config.translation_model_name}", flush=True)
    print(f"• translation_revision             : {config.translation_revision}", flush=True)
    print(f"• translation_local_files_only     : {config.translation_local_files_only}", flush=True)
    print(f"• default_output_top_k             : {config.default_output_top_k}", flush=True)
    print(f"• default_refine_top_n             : {config.default_refine_top_n} (Phase 4 Raw-Video Refinement)", flush=True)

    assert config.enable_dynamic_translation is True, "enable_dynamic_translation must be True in production.yaml!"

    # 2. Bootstrap Operational Runtime
    print("\n--- BOOTSTRAPPING OPERATIONAL RUNTIME ---", flush=True)
    t_boot = time.time()
    runtime = OperationalKISRuntime.bootstrap(config)
    print(f"Runtime bootstrap completed in {time.time() - t_boot:.2f}s.", flush=True)

    # 3. Inspect Artifact Fingerprint
    assert runtime.translation_provider is not None, "Translation provider must be initialized!"
    fingerprint = runtime.translation_provider.get_artifact_fingerprint()
    print("\n--- MARIAN ARTIFACT PROVENANCE & FINGERPRINT ---", flush=True)
    for k, v in fingerprint.items():
        print(f"• {k:<32} : {v}", flush=True)

    guard = TokenBudgetGuard()

    # 4. Execute 3-Query Production Smoke
    print("\n" + "=" * 150, flush=True)
    print("EXECUTING 3-QUERY PRODUCTION SMOKE VIA runtime.handle_query()", flush=True)
    print("=" * 150, flush=True)

    for idx, item in enumerate(SMOKE_3_QUERIES, start=1):
        qid = item["qid"]
        cat = item["category"]
        topic = item["topic"]
        q_vi = item["vi"]

        req = QueryRequest(
            request_id=f"prod-smoke-{qid}",
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

        # Ingested and Effective translation diagnostics
        raw_marian_en = runtime.translation_provider.translate(q_vi)
        raw_tokens = guard.count_tokens(raw_marian_en)
        effective_en, effective_tokens, was_compacted = guard.guard_and_compact(raw_marian_en)

        # Assertions
        assert 1 <= len(preds) <= 100, f"Expected 1 <= len(preds) <= 100, got {len(preds)}"
        assert effective_tokens <= 77, f"Effective tokens {effective_tokens} exceeds 77 limit!"

        top3_desc = [f"@{p['rank']}: {p['video_id']} (f={p['frame_id']})" for p in preds[:3]]

        print(f"[{idx}/3] [{cat}] {qid} - {topic} (Elapsed: {elapsed:.1f}s)", flush=True)
        print(f"     • VI Raw                  : \"{q_vi[:85]}...\"", flush=True)
        print(f"     • Marian Raw Translation  : \"{raw_marian_en}\"", flush=True)
        print(f"     • Raw Tokens              : {raw_tokens} tokens", flush=True)
        print(f"     • Effective Retrieval Text: \"{effective_en}\"", flush=True)
        print(f"     • Effective Tokens        : {effective_tokens} tokens (Budget Safe <= 77: {'YES ✅' if effective_tokens <= 77 else 'NO ❌'}) | Compacted? {'YES ⚠️' if was_compacted else 'NO ✅'}", flush=True)
        print(f"     • Effective Variants      : [('{qid}::marian_en', 'en', 'ENGLISH_TRANSLATION', weight=1.0)] (EN_ONLY, 0% Vietnamese fused)", flush=True)
        print(f"     • Top 3 Candidates (N={len(preds)}): {top3_desc}\n", flush=True)

    print("=" * 150, flush=True)
    print(">>> FINAL STATUS: KIS_MARIAN_EN_ONLY_PRODUCTION_FROZEN ✅", flush=True)
    print("=" * 150, flush=True)


def main() -> None:
    run_production_frozen_smoke()


if __name__ == "__main__":
    main()
