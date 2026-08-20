#!/usr/bin/env python3
"""KIS P0.1: Real Offline Translator (Helsinki-NLP/opus-mt-vi-en) Validation Script.

Performs:
  1. Cold load benchmark of Helsinki-NLP/opus-mt-vi-en on actual environment.
  2. Warm per-query translation benchmark on 5 BTC diagnostic queries:
     - Untouched Marian English translation output.
     - Exact CLIP token count & compaction status.
     - Visual HTML gallery of Top 3 retrieved frames for visual semantic recovery verification.
  3. Full-38 KIS DEV Benchmark with dynamic Marian translation:
     - Evaluates R@1/5/20/50/100, numerator/190, macro score, strict hits.
     - Compares directly against reference b49f628 (12/190 = 0.063158).
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

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

from system_tai.features.query_encoder import SharedOpenAIClipEncoder
from system_tai.kis.session_engine import OperationalKISRuntime
from system_tai.kis.session_schema import QueryRequest, SessionConfig
from system_tai.translation.provider import MarianOfflineTranslator, TokenBudgetGuard

BTC_5_QUERIES = [
    {
        "qid": "query-p1-1-kis",
        "topic": "Phóng tàu vũ trụ 4 phi hành gia",
        "vi": "Đây là phần giới thiệu việc phóng tàu vũ trụ tư nhân. Đoạn clip bắt đầu với hình ảnh 4 phi hành gia mặc áo đen. Một trong những nhiệm vụ dự kiến của tàu vũ trụ là nghiên cứu ánh sáng cực quang ở vùng cực",
    },
    {
        "qid": "query-p1-10-kis",
        "topic": "3 người chơi nhạc cụ kim loại hình tròn (handpan)",
        "vi": "Tìm chính xác đoạn clip ngắn có ba người (hai phụ nữ và một nam giới) đang ngồi cạnh nhau, tập trung chơi nhạc cụ kim loại có dạng tròn, rỗng, với các vết lõm để tạo ra âm thanh khi gõ tay. Có 1 người mặc áo trắng ngồi giữa 2 người mặc áo đen. Bối cảnh phía sau là một kệ sách nhiều ngăn, xếp đầy sách với nhiều màu sắc",
    },
    {
        "qid": "query-p1-12-kis",
        "topic": "Trang trí bánh rán, rưới chocolate dâu tây",
        "vi": "Đoạn video mô tả cảnh trang trí bánh rán. Phân cảnh bắt đầu là một chiếc đĩa sứ màu trắng nằm trên một khay gỗ hình chữ nhật. Bên cạnh chiếc đĩa sứ là một chén sứ nhỏ màu trắng đựng chuối đã được cắt sẵn và một cái thìa nhỏ màu nâu. Phân cảnh tiếp theo cho thấy đầu bếp đặt 2 chiếc bánh rán lên đĩa sứ và bắt đầu trang trí. Bước đầu tiên là việc rưới chocolate lên trên mặt bánh. Sau đó, đầu bếp đặt các lát chuối lên trên một chiếc bánh rán, chiếc còn lại được đặt các lát dâu tây lên.",
    },
    {
        "qid": "query-p1-2-kis",
        "topic": "Đàn hổ con miền Nam",
        "vi": "Mẩu tin giới thiệu về đàn hổ tại một địa phương ở miền Nam vừa có thêm khoảng 3-6 con hổ con. Đây là một giống hổ quý hiếm",
    },
    {
        "qid": "query-p1-24-kis",
        "topic": "Đua xe đạp quay từ flycam góc trên cao",
        "vi": "Đoạn video về tường thuật một cuộc đua xe đạp. Tìm phân cảnh với góc quay trực diện từ trên cao xuống dõi theo các tay đua. Trong khung hình gồm có 3 tay đua đang đạp thành một đường thẳng. Cả 3 tay đua đều đến từ cùng một đội, với đồng phục áo trắng quần vàng xanh. Tay đua đầu tiên đội nón trắng, tay đua thứ hai đội nón đỏ và tay đua cuối cùng đội nón đen.",
    },
]


def run_benchmark_5_btc(runtime: OperationalKISRuntime, translator: MarianOfflineTranslator, guard: TokenBudgetGuard) -> list[dict[str, Any]]:
    print("\n" + "=" * 150, flush=True)
    print("SECTION 1: UNTOUCHED MARIAN TRANSLATION & RETRIEVAL ON 5 BTC DIAGNOSTIC QUERIES", flush=True)
    print("=" * 150, flush=True)

    results = []

    for item in BTC_5_QUERIES:
        qid = item["qid"]
        topic = item["topic"]
        vi_text = item["vi"]

        # 1. Warm translation
        t0 = time.perf_counter()
        raw_en = translator.translate(vi_text)
        trans_lat_ms = (time.perf_counter() - t0) * 1000.0

        # 2. Token guard & compaction
        final_en, tok_count, was_compacted = guard.guard_and_compact(raw_en)

        # 3. EN_ONLY CLIP Retrieval
        emb = runtime.shared_encoder.encode_texts([final_en])[0]
        res = runtime.exact_retriever.search_vector(query_id=f"{qid}-marian", query_vector=emb, top_k=50)

        top10_vids = []
        seen_vids = set()
        top3_candidates = []

        for r in res.ranked_candidates:
            vid = r.video_id
            if vid not in seen_vids:
                seen_vids.add(vid)
                top10_vids.append(vid)
                if len(top3_candidates) < 3:
                    top3_candidates.append({
                        "rank": len(top3_candidates) + 1,
                        "video_id": vid,
                        "frame_id": r.frame_id,
                        "score": float(r.score) if hasattr(r, "score") else 0.0,
                    })
            if len(top10_vids) >= 10:
                break

        print(f"\n[{qid}] - {topic}:", flush=True)
        print(f"  • VI Input             : \"{vi_text[:90]}...\"", flush=True)
        print(f"  • Marian Translation   : \"{final_en}\"", flush=True)
        print(f"  • Latency              : {trans_lat_ms:.2f} ms | CLIP Tokens: {tok_count} | Compaction Invoked? {'YES ⚠️' if was_compacted else 'NO ✅'}", flush=True)
        print(f"  • Top 5 Videos         : {top10_vids[:5]}", flush=True)

        results.append({
            "qid": qid,
            "topic": topic,
            "vi_text": vi_text,
            "marian_en": final_en,
            "latency_ms": trans_lat_ms,
            "tok_count": tok_count,
            "was_compacted": was_compacted,
            "top10_vids": top10_vids,
            "top3_candidates": top3_candidates,
        })

    out_json = Path("/kaggle/working/marian_btc_results.json")
    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return results


def generate_marian_gallery() -> str:
    out_json = Path("/kaggle/working/marian_btc_results.json")
    if not out_json.exists():
        return "<div>Results not found.</div>"

    from visualize_btc_predictions import extract_frame_base64, find_video_path

    data = json.loads(out_json.read_text(encoding="utf-8"))
    html_cards = []

    html_cards.append("""
    <div style="font-family: Arial, sans-serif; max-width: 1200px; margin: auto;">
        <h1 style="text-align: center; color: #1a73e8; border-bottom: 2px solid #1a73e8; padding-bottom: 10px;">
            🤖 BẢNG ĐỐI SOÁT HÌNH ẢNH MARIAN TRANSLATOR (Helsinki-NLP/opus-mt-vi-en) TRÊN 5 CÂU BTC
        </h1>
        <p style="text-align: center; color: #5f6368; font-size: 14px;">
            Kiểm tra năng lực phục hồi ngữ nghĩa (Semantic Recovery) của mô hình dịch máy tự động
        </p>
    """)

    for q in data:
        qid = q["qid"]
        topic = q["topic"]
        marian_en = q["marian_en"]
        tok_count = q["tok_count"]
        lat = q["latency_ms"]

        card_html = f"""
        <div style="background: #ffffff; border: 1px solid #ddd; border-radius: 8px; margin-bottom: 25px; padding: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.08);">
            <div style="font-size: 15px; font-weight: bold; color: #202124; margin-bottom: 8px;">
                <span style="background: #e8f0fe; color: #1967d2; padding: 4px 10px; border-radius: 4px; margin-right: 8px;">{qid}</span>
                <span>{topic}</span>
            </div>
            <div style="background: #f1f3f4; padding: 8px 12px; border-radius: 4px; font-size: 13px; color: #202124; margin-bottom: 12px;">
                <b>Marian EN</b>: "<i>{marian_en}</i>" <br>
                <span style="font-size: 12px; color: #5f6368;">(Latency: {lat:.1f}ms | CLIP Tokens: {tok_count})</span>
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
                f'<img src="data:image/jpeg;base64,{img_b64}" style="width: 100%; height: auto; border-radius: 4px; border: 1px solid #ccc;" />'
                if img_b64
                else '<div style="height: 120px; background: #eee; display: flex; align-items: center; justify-content: center; color: #888;">(No Video)</div>'
            )

            card_html += f"""
            <div style="flex: 1; background: #fafafa; border: 1px solid #e0e0e0; border-radius: 6px; padding: 8px; text-align: center;">
                <div style="font-size: 12px; font-weight: bold; color: #1a73e8; margin-bottom: 4px;">
                    Top @{rank}: {v_id} (f={f_id}, ~{time_str})
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
    return "\n".join(html_cards)


def run_full38_dev_benchmark(runtime: OperationalKISRuntime, translator: MarianOfflineTranslator, guard: TokenBudgetGuard) -> None:
    print("\n" + "=" * 150, flush=True)
    print("SECTION 2: FULL-38 KIS DEV BENCHMARK WITH DYNAMIC MARIAN TRANSLATION (EN_ONLY)", flush=True)
    print("=" * 150, flush=True)

    kis_gt_path = REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "l21_150_diagnostic" / "kis_dev_gt.json"
    sidecar_path = REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "l21_150_diagnostic" / "q2_kis_dev_en_translation.json"

    if not kis_gt_path.exists():
        print(f"ERROR: Groundtruth not found at {kis_gt_path}", flush=True)
        return
    if not sidecar_path.exists():
        print(f"ERROR: Sidecar not found at {sidecar_path}", flush=True)
        return

    from system_tai.evaluation.groundtruth_loader import load_kis_groundtruth
    from system_tai.evaluation.kis_evaluator import evaluate_kis_predictions
    from system_tai.preliminary.schemas import KISPrediction

    gt_records = load_kis_groundtruth(kis_gt_path)
    sidecar_data = json.loads(sidecar_path.read_text(encoding="utf-8"))
    vi_queries = {rec["query_id"]: rec["source_vi"] for rec in sidecar_data.get("records", [])}

    print(f"Loaded {len(gt_records)} Groundtruth queries and {len(vi_queries)} Vietnamese source queries for Full-38 evaluation.\n", flush=True)

    all_predictions: list[KISPrediction] = []
    trans_latencies: list[float] = []
    total_time_start = time.time()

    query_details = []

    for idx, (qid, gt) in enumerate(gt_records.items(), start=1):
        vi_text = vi_queries.get(qid, "").strip()
        if not vi_text:
            print(f"WARNING: Vietnamese query not found for {qid}, skipping", flush=True)
            continue

        # 1. Translate
        t0 = time.perf_counter()
        raw_en = translator.translate(vi_text)
        trans_lat_ms = (time.perf_counter() - t0) * 1000.0
        trans_latencies.append(trans_lat_ms)

        # 2. Token guard
        final_en, tok_count, was_compacted = guard.guard_and_compact(raw_en)

        # 3. EN_ONLY Search
        emb = runtime.shared_encoder.encode_texts([final_en])[0]
        res = runtime.exact_retriever.search_vector(query_id=f"{qid}-dev", query_vector=emb, top_k=100)

        preds = []
        for r_idx, r in enumerate(res.ranked_candidates, start=1):
            preds.append(KISPrediction(
                query_id=qid,
                rank=r_idx,
                video_id=r.video_id,
                frame_id=r.frame_id,
            ))
        all_predictions.extend(preds)

        # Quick hit check
        hit_rank = None
        for p in preds:
            if p.video_id == gt.video_id and gt.start_frame_id <= p.frame_id <= gt.end_frame_id:
                hit_rank = p.rank
                break

        status_str = f"HIT @{hit_rank} ✅" if hit_rank is not None else "MISS ❌"
        print(f"[{idx:02d}/38] {qid:<10} in {trans_lat_ms:5.1f}ms (Tokens: {tok_count:2d}) -> {status_str} | Target: {gt.video_id} [{gt.start_frame_id}..{gt.end_frame_id}]", flush=True)

    eval_result = evaluate_kis_predictions(all_predictions, gt_records)

    total_elapsed = time.time() - total_time_start
    mean_trans_lat = sum(trans_latencies) / max(len(trans_latencies), 1)

    print("\n" + "=" * 150, flush=True)
    print("FULL-38 KIS DEV OFFICIAL METRICS COMPARISON", flush=True)
    print("=" * 150, flush=True)
    print(f"• Completed Queries    : {eval_result.completed_queries} / {eval_result.total_queries} (100.0%)")
    print(f"• Mean Translation Lat : {mean_trans_lat:.2f} ms / query")
    print(f"• Total Evaluation Time: {total_elapsed:.2f} s\n")

    print(f"{'Metric':<25} | {'Reference b49f628 (Arm-B)':<30} | {'Dynamic Marian EN_ONLY Candidate':<30}")
    print("-" * 95)
    print(f"{'Recall @1':<25} | {0:<30} | {eval_result.recall_at_1:<30}")
    print(f"{'Recall @5':<25} | {0:<30} | {eval_result.recall_at_5:<30}")
    print(f"{'Recall @20':<25} | {2:<30} | {eval_result.recall_at_20:<30}")
    print(f"{'Recall @50':<25} | {5:<30} | {eval_result.recall_at_50:<30}")
    print(f"{'Recall @100':<25} | {5:<30} | {eval_result.recall_at_100:<30}")
    print("-" * 95)
    print(f"{'Numerator / 190':<25} | {'12 / 190':<30} | {f'{eval_result.official_numerator} / 190':<30}")
    print(f"{'Macro Score':<25} | {'0.063158':<30} | {f'{eval_result.official_macro_score:.6f}':<30}")
    print(f"{'Strict Hit @100':<25} | {'5 / 38 (13.16%)':<30} | {f'{eval_result.strict_hit_count} / 38 ({eval_result.strict_hit_count/38*100:.2f}%)':<30}")
    print(f"{'Video Hit @100':<25} | {'34 / 38 (89.47%)':<30} | {f'{eval_result.video_hit_count} / 38 ({eval_result.video_hit_count/38*100:.2f}%)':<30}")
    print("=" * 150, flush=True)


def main(raw_args: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="KIS P0.1 Marian Dynamic Translation Validator")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--skip-dev", action="store_true", help="Skip Full-38 DEV benchmark")
    args, _ = parser.parse_known_args(raw_args)

    print("=" * 150, flush=True)
    print("KIS P0.1: OFFLINE TRANSLATOR (Helsinki-NLP/opus-mt-vi-en) BENCHMARK & VALIDATION", flush=True)
    print("=" * 150, flush=True)

    # 1. Cold Load Benchmark
    t_cold = time.perf_counter()
    translator = MarianOfflineTranslator(device=args.device)
    cold_load_sec = time.perf_counter() - t_cold
    guard = TokenBudgetGuard()

    print(f"• Model Loaded Successfully    : {translator.provider_name}", flush=True)
    print(f"• Actual Device Resolved       : {translator.device.upper()}", flush=True)
    print(f"• Cold Model Load Time         : {cold_load_sec:.3f} s", flush=True)

    # 2. Bootstrap KIS Runtime
    reuse_manifest_path = None
    for p in [
        Path("/kaggle/working/manifest_cache.json"),
        Path("/kaggle/input/system-tai-manifest/feature_manifest.json"),
        Path("/kaggle/input/datasets/manifest_cache.json"),
    ]:
        if p.exists() and p.stat().st_size > 1000:
            reuse_manifest_path = p
            break

    config = SessionConfig(
        input_root=Path("/kaggle/input/datasets") if Path("/kaggle/input/datasets").exists() else Path("/kaggle/input"),
        reuse_manifest=reuse_manifest_path,
        output_root=Path("/kaggle/working/output/marian_validation"),
        device="auto",
        allow_model_download=True,
    )
    runtime = OperationalKISRuntime.bootstrap(config)

    # 3. Benchmark 5 BTC Queries
    run_benchmark_5_btc(runtime, translator, guard)

    # 4. Full-38 DEV Benchmark
    if not args.skip_dev:
        run_full38_dev_benchmark(runtime, translator, guard)


if __name__ == "__main__":
    main()
