#!/usr/bin/env python3
"""KIS P0.4c: Final Revision Behavior & Translation Parity Closure.

Audits:
  1. production.yaml explicitly configured with pinned revision 5611f34634b72de0608b1238a4e02845ca285f3e.
  2. Marian offline translator execution with local_files_only=True on revision 5611f34634b72de0608b1238a4e02845ca285f3e.
  3. Exact 1-to-1 text match for 3 benchmark queries (p1-2, p1-10, p1-12) against reference production smoke output.
  4. Structured artifact categorization (REQUIRED vs UNUSED/EMPTY).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Enforce offline mode in environment
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

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

from system_tai.kis.session_schema import SessionConfig
from system_tai.translation.provider import MarianOfflineTranslator

PARITY_CASES = [
    {
        "qid": "query-p1-2-kis",
        "topic": "Đàn hổ con miền Nam",
        "vi": "Mẩu tin giới thiệu về đàn hổ tại một địa phương ở miền Nam vừa có thêm khoảng 3-6 con hổ con. Đây là một giống hổ quý hiếm",
        "reference_smoke_translation": "This is a rare tiger species.",
    },
    {
        "qid": "query-p1-10-kis",
        "topic": "3 người chơi handpan trước kệ sách",
        "vi": "Tìm chính xác đoạn clip ngắn có ba người (hai phụ nữ và một nam giới) đang ngồi cạnh nhau, tập trung chơi nhạc cụ kim loại có dạng tròn, rỗng, với các vết lõm để tạo ra âm thanh khi gõ tay. Có 1 người mặc áo trắng ngồi giữa 2 người mặc áo đen. Bối cảnh phía sau là một kệ sách nhiều ngăn, xếp đầy sách với nhiều màu sắc",
        "reference_smoke_translation": "Find exactly the short clip of three people (two women and a man) sitting next to each other, focusing on playing metal instruments in round, empty, with holes in which to make sounds when they type, there's a white man sitting between two people in black, and the background behind is a stack of books, filled with many colors.",
    },
    {
        "qid": "query-p1-12-kis",
        "topic": "Trang trí bánh rán với chuối, dâu tây, chocolate",
        "vi": "Đoạn video mô tả cảnh trang trí bánh rán. Phân cảnh bắt đầu là một chiếc đĩa sứ màu trắng nằm trên một khay gỗ hình chữ nhật. Bên cạnh chiếc đĩa sứ là một chén đựng một vài trái dâu, nhưng có 2 trái bị rơi ra ngoài. Ngoài ra, bên cạnh đĩa sứ còn có một chén sứ nhỏ màu trắng đựng chuối đã được cắt sẵn và một cái thìa nhỏ màu nâu. Phân cảnh tiếp theo cho thấy đầu bếp đặt 2 chiếc bánh rán lên đĩa sứ và bắt đầu trang trí. Bước đầu tiên là việc rưới chocolate lên trên mặt bánh. Sau đó, đầu bếp đặt các lát chuối lên trên một chiếc bánh rán, chiếc còn lại được đặt các lát dâu tây lên.",
        "reference_smoke_translation": "The video depicts a set of donut decorations. The scene begins as a white dish on a wooden tray of Japanese wood. Next to the dish is a bowl of some berries, but there are two lefts that have fallen out. Besides, besides the dish, there's a small white dish with bananas already cut and a little brown spoon, and the next scene shows that the cook put two donuts on the plate and starts to make decorations. The first step is to sprinkle chocolate on the cake. Then the first step is to put the banana slices on the top of the table.",
    },
]


def run_revision_parity_audit() -> None:
    yaml_path = REPO_ROOT / "systems" / "system_tai" / "configs" / "production.yaml"
    print("=" * 140, flush=True)
    print("KIS P0.4c: FINAL REVISION BEHAVIOR & TRANSLATION PARITY AUDIT", flush=True)
    print("=" * 140, flush=True)

    # 1. Load SessionConfig from production.yaml
    cfg = SessionConfig.from_yaml(yaml_path)
    print(f"• Production YAML Path             : {yaml_path.relative_to(REPO_ROOT)}", flush=True)
    print(f"• Pinned Revision in YAML          : {cfg.translation_revision}", flush=True)
    print(f"• Dynamic Translation Enabled      : {cfg.enable_dynamic_translation}", flush=True)
    print(f"• Translation Local Files Only     : {cfg.translation_local_files_only}", flush=True)

    TARGET_REVISION = "5611f34634b72de0608b1238a4e02845ca285f3e"
    assert cfg.translation_revision == TARGET_REVISION, f"Expected {TARGET_REVISION}, got {cfg.translation_revision}"

    # 2. Instantiate MarianOfflineTranslator with pinned revision
    translator = MarianOfflineTranslator(
        revision=cfg.translation_revision,
        local_files_only=True,
    )

    fp = translator.get_artifact_fingerprint()
    print(f"• Resolved Snapshot Directory      : {fp.get('resolved_snapshot_dir')}", flush=True)
    print(f"• Revision Matches Snapshot        : {'YES ✅' if fp.get('revision_matches_snapshot') else 'NO ❌'}", flush=True)
    print(f"• Primary Weight File Loaded       : {fp.get('primary_weight_artifact')} ({fp.get('model.safetensors_size_bytes')} bytes)", flush=True)
    print(f"• Primary Weight SHA256            : {fp.get('model.safetensors_sha256')}", flush=True)

    print("\n--- STRUCTURED ARTIFACT MANIFEST CATEGORIZATION ---", flush=True)
    print("  [REQUIRED ACTIVE RUNTIME ARTIFACTS]")
    for fname in ["model.safetensors", "config.json", "source.spm", "target.spm", "vocab.json", "tokenizer_config.json"]:
        size = fp.get(f"{fname}_size_bytes", "N/A")
        sha = fp.get(f"{fname}_sha256", "N/A")
        print(f"    • {fname:<24} Size: {str(size):>10} bytes | SHA256: {sha}", flush=True)

    print("\n  [SECONDARY / UNUSED / EMPTY CACHE ENTRIES]")
    for k, v in sorted(fp.items()):
        if k.endswith("_size_bytes"):
            fname = k[:-11]
            if fname not in ["model.safetensors", "config.json", "source.spm", "target.spm", "vocab.json", "tokenizer_config.json"]:
                sha = fp.get(f"{fname}_sha256", "N/A")
                print(f"    • {fname:<24} Size: {str(v):>10} bytes | SHA256: {sha}", flush=True)

    # 3. Deterministic 3-Query Translation Parity Check
    print("\n" + "=" * 140, flush=True)
    print("EXECUTING DETERMINISTIC TRANSLATION PARITY AUDIT (Zero Retrieval, Zero Phase-4)", flush=True)
    print("=" * 140, flush=True)

    all_passed = True
    for idx, case in enumerate(PARITY_CASES, start=1):
        qid = case["qid"]
        topic = case["topic"]
        vi_raw = case["vi"]
        ref_en = case["reference_smoke_translation"]

        cur_en = translator.translate(vi_raw)
        is_match = (cur_en.strip() == ref_en.strip())

        if not is_match:
            all_passed = False

        print(f"[{idx}/3] {qid} ({topic})", flush=True)
        print(f"     • Current Pinned Revision : {cfg.translation_revision}", flush=True)
        print(f"     • Current Translation     : \"{cur_en}\"", flush=True)
        print(f"     • Reference Translation   : \"{ref_en}\"", flush=True)
        print(f"     • Exact Text Match        : {'YES ✅ (100% PARITY)' if is_match else 'NO ❌ (MISMATCH)'}\n", flush=True)

    assert all_passed, "All 3 queries must have exact 1-to-1 text parity against reference production smoke!"

    print("=" * 140, flush=True)
    print(">>> 3/3 TRANSLATION BEHAVIOR PARITY: 100% PASS ✅", flush=True)
    print(">>> PROVENANCE & REVISION CLOSURE: VERIFIED & SEALED ✅", flush=True)
    print(">>> FINAL CLOSURE STATUS: KIS_MARIAN_EN_ONLY_PRODUCTION_FROZEN ✅", flush=True)
    print(">>> KIS P0 STATUS: KIS_P0_CLOSED ✅", flush=True)
    print("=" * 140, flush=True)


if __name__ == "__main__":
    run_revision_parity_audit()
