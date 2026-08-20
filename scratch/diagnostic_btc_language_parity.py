#!/usr/bin/env python3
"""P0 Diagnostic: Disentangling Language Mismatch from Token Truncation.

Compares:
  - Arm A: VI Raw (long verbose, truncated)
  - Arm E: VI Concise (short, <=75 CLIP tokens, NO truncation)
  - Arm B: EN Literal (literal translation, <=75 CLIP tokens)
  - Arm C: EN Concise (concise English, <=75 CLIP tokens)

This decisive experiment directly isolates the causal impact of:
  1. Truncation removal in Vietnamese (Arm A -> Arm E)
  2. Language translation to English (Arm E -> Arm B/C)
"""

from __future__ import annotations

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
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "openai-clip", "ftfy", "regex", "tqdm"], check=False)
    import clip

from system_tai.features.query_encoder import SharedOpenAIClipEncoder
from system_tai.kis.session_engine import OperationalKISRuntime
from system_tai.kis.session_schema import SessionConfig

BENCHMARK_QUERIES = [
    {
        "qid": "query-p1-1-kis",
        "topic": "Phóng tàu vũ trụ 4 phi hành gia",
        "vi_raw": "Đây là phần giới thiệu việc phóng tàu vũ trụ tư nhân. Đoạn clip bắt đầu với hình ảnh 4 phi hành gia mặc áo đen. Một trong những nhiệm vụ dự kiến của tàu vũ trụ là nghiên cứu ánh sáng cực quang ở vùng cực",
        "vi_concise": "phóng tàu vũ trụ tên lửa 4 phi hành gia áo đen",
        "en_literal": "This is the introduction of a private spacecraft launch. The clip begins with an image of 4 astronauts wearing black suits. One of the planned missions of the spacecraft is to study the aurora borealis at the poles",
        "en_concise": "a rocket launch with astronauts in black spacesuits and spacecraft polar aurora",
    },
    {
        "qid": "query-p1-10-kis",
        "topic": "3 người chơi nhạc cụ kim loại hình tròn (handpan)",
        "vi_raw": "Tìm chính xác đoạn clip ngắn có ba người (hai phụ nữ và một nam giới) đang ngồi cạnh nhau, tập trung chơi nhạc cụ kim loại có dạng tròn, rỗng, với các vết lõm để tạo ra âm thanh khi gõ tay. Có 1 người mặc áo trắng ngồi giữa 2 người mặc áo đen. Bối cảnh phía sau là một kệ sách nhiều ngăn, xếp đầy sách với nhiều màu sắc",
        "vi_concise": "ba người chơi nhạc cụ kim loại tròn kệ sách",
        "en_literal": "Find the exact short clip of three people (two women and one man) sitting next to each other, focused on playing a circular, hollow metallic musical instrument with indentations to create sound when tapped by hand. One person in white sits between two people in black. Behind is a colorful bookshelf.",
        "en_concise": "three people playing handpan steel tongue drum in front of colorful bookshelf",
    },
    {
        "qid": "query-p1-12-kis",
        "topic": "Trang trí bánh rán, rưới chocolate dâu tây",
        "vi_raw": "Đoạn video mô tả cảnh trang trí bánh rán. Phân cảnh bắt đầu là một chiếc đĩa sứ màu trắng nằm trên một khay gỗ hình chữ nhật. Bên cạnh chiếc đĩa sứ là một chén sứ nhỏ màu trắng đựng chuối đã được cắt sẵn và một cái thìa nhỏ màu nâu. Phân cảnh tiếp theo cho thấy đầu bếp đặt 2 chiếc bánh rán lên đĩa sứ và bắt đầu trang trí. Bước đầu tiên là việc rưới chocolate lên trên mặt bánh. Sau đó, đầu bếp đặt các lát chuối lên trên một chiếc bánh rán, chiếc còn lại được đặt các lát dâu tây lên.",
        "vi_concise": "trang trí bánh rán đĩa sứ rưới chocolate chuối dâu tây",
        "en_literal": "The video describes donut decoration. The scene starts with a white porcelain plate on a rectangular wooden tray. Next to it is a small bowl with sliced banana. The chef places 2 donuts on the plate, drizzles chocolate sauce on top, places banana slices on one donut and strawberry slices on the other.",
        "en_concise": "decorating donuts on white plate with chocolate drizzle sliced bananas and strawberries",
    },
    {
        "qid": "query-p1-2-kis",
        "topic": "Đàn hổ con miền Nam",
        "vi_raw": "Mẩu tin giới thiệu về đàn hổ tại một địa phương ở miền Nam vừa có thêm khoảng 3-6 con hổ con. Đây là một giống hổ quý hiếm",
        "vi_concise": "đàn hổ hổ mẹ sinh hổ con trong chuồng",
        "en_literal": "A news report introducing a tiger pack in a Southern locality having around 3-6 new tiger cubs. This is a rare tiger breed",
        "en_concise": "newborn baby tiger cubs in zoo enclosure",
    },
    {
        "qid": "query-p1-24-kis",
        "topic": "Đua xe đạp quay từ flycam góc trên cao",
        "vi_raw": "Đoạn video về tường thuật một cuộc đua xe đạp. Tìm phân cảnh với góc quay trực diện từ trên cao xuống dõi theo các tay đua. Trong khung hình gồm có 3 tay đua đang đạp thành một đường thẳng. Cả 3 tay đua đều đến từ cùng một đội, với đồng phục áo trắng quần vàng xanh. Tay đua đầu tiên đội nón trắng, tay đua thứ hai đội nón đỏ và tay đua cuối cùng đội nón đen.",
        "vi_concise": "đua xe đạp góc quay từ trên cao flycam 3 tay đua",
        "en_literal": "A video reporting on a bicycle race. Find the scene with an overhead top-down view following the cyclists. In the frame are 3 cyclists riding in a straight line from the same team with white jerseys and yellow-green shorts, wearing white, red, and black helmets.",
        "en_concise": "aerial top-down view of three cyclists in peloton riding in line on road",
    },
]


def inspect_token_budget() -> None:
    print("=" * 150, flush=True)
    print("SECTION 1: TOKEN BUDGET & TRUNCATION COMPARISON (Arm A vs Arm E vs Arm B vs Arm C)", flush=True)
    print("=" * 150, flush=True)
    tokenizer = clip.simple_tokenizer.SimpleTokenizer()

    for item in BENCHMARK_QUERIES:
        qid = item["qid"]
        topic = item["topic"]

        tok_a = len(tokenizer.encode(item["vi_raw"])) + 2
        tok_e = len(tokenizer.encode(item["vi_concise"])) + 2
        tok_b = len(tokenizer.encode(item["en_literal"])) + 2
        tok_c = len(tokenizer.encode(item["en_concise"])) + 2

        print(f"\n[{qid}] - {topic}:")
        print(f"  • Arm A (VI Raw)     : {tok_a:3d} tokens | Truncated? {'YES ❌ (lost ' + str(tok_a-77) + ' tokens)' if tok_a > 77 else 'NO ✅'}")
        print(f"  • Arm E (VI Concise) : {tok_e:3d} tokens | Truncated? {'YES ❌' if tok_e > 77 else 'NO ✅ (Clean 100% in budget)'} -> \"{item['vi_concise']}\"")
        print(f"  • Arm B (EN Literal) : {tok_b:3d} tokens | Truncated? {'YES ❌' if tok_b > 77 else 'NO ✅ (Clean 100% in budget)'}")
        print(f"  • Arm C (EN Concise) : {tok_c:3d} tokens | Truncated? {'YES ❌' if tok_c > 77 else 'NO ✅ (Clean 100% in budget)'}")


def run_isolation_ablation() -> None:
    print("\n" + "=" * 150, flush=True)
    print("SECTION 2: ISOLATION RETRIEVAL EXPERIMENT (Arm A vs Arm E vs Arm B vs Arm C)", flush=True)
    print("=" * 150, flush=True)

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
        output_root=Path("/kaggle/working/output/diagnostic_ablation"),
        device="auto",
        allow_model_download=True,
    )

    t0 = time.time()
    runtime = OperationalKISRuntime.bootstrap(config)
    print(f"Runtime bootstrap completed in {time.time() - t0:.2f}s.\n", flush=True)

    ablation_results = []

    for item in BENCHMARK_QUERIES:
        qid = item["qid"]
        topic = item["topic"]
        print("-" * 150, flush=True)
        print(f"Testing Query: {qid} ({topic})", flush=True)

        arms = {
            "Arm A (VI Raw)": item["vi_raw"],
            "Arm E (VI Concise)": item["vi_concise"],
            "Arm B (EN Literal)": item["en_literal"],
            "Arm C (EN Concise)": item["en_concise"],
        }

        query_arm_outputs = {}

        for arm_name, text in arms.items():
            emb = runtime.shared_encoder.encode_texts([text])[0]
            res = runtime.exact_retriever.search_vector(query_id=f"{qid}-{arm_name}", query_vector=emb, top_k=50)

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

            print(f"  • {arm_name:<24} -> Top 5 Videos: {top10_vids[:5]}", flush=True)
            query_arm_outputs[arm_name] = {
                "top10_vids": top10_vids,
                "top3_candidates": top3_candidates,
            }

        ablation_results.append({
            "qid": qid,
            "topic": topic,
            "arms": query_arm_outputs,
        })

    # Save to JSON
    out_json = Path("/kaggle/working/ablation_results.json")
    out_json.write_text(json.dumps(ablation_results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nAblation results saved to: {out_json}", flush=True)


def generate_ablation_html_gallery() -> str:
    out_json = Path("/kaggle/working/ablation_results.json")
    if not out_json.exists():
        return "<div>Ablation results JSON not found.</div>"

    from visualize_btc_predictions import extract_frame_base64, find_video_path

    data = json.loads(out_json.read_text(encoding="utf-8"))
    html_cards = []

    html_cards.append("""
    <div style="font-family: Arial, sans-serif; max-width: 1450px; margin: auto;">
        <h1 style="text-align: center; color: #1a73e8; border-bottom: 2px solid #1a73e8; padding-bottom: 10px;">
            🔬 BẢNG ĐỐI SOÁT TÁCH BẠCH NGUYÊN NHÂN: ARM A (VI Raw) vs ARM E (VI Concise) vs ARM B (EN Literal) vs ARM C (EN Concise)
        </h1>
        <p style="text-align: center; color: #5f6368; font-size: 14px;">
            <b>Arm A</b>: Tiếng Việt Dài (Truncated) | <b>Arm E</b>: Tiếng Việt Ngắn (Không Truncated) | <b>Arm B</b>: Tiếng Anh Literal | <b>Arm C</b>: Tiếng Anh Concise
        </p>
    """)

    for q in data:
        qid = q["qid"]
        topic = q["topic"]

        card_html = f"""
        <div style="background: #ffffff; border: 1px solid #ddd; border-radius: 8px; margin-bottom: 30px; padding: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.08);">
            <div style="font-size: 16px; font-weight: bold; color: #202124; margin-bottom: 12px;">
                <span style="background: #e8f0fe; color: #1967d2; padding: 4px 10px; border-radius: 4px; margin-right: 8px;">{qid}</span>
                <span>{topic}</span>
            </div>
            <div style="display: flex; gap: 12px; justify-content: space-between;">
        """

        for arm_name, arm_data in q["arms"].items():
            top3 = arm_data["top3_candidates"]
            color = "#d93025" if "Arm A" in arm_name else ("#9c27b0" if "Arm E" in arm_name else ("#188038" if "Arm C" in arm_name else "#1a73e8"))

            arm_html = f"""
            <div style="flex: 1; min-width: 220px; background: #f8f9fa; border: 1px solid #e0e0e0; border-radius: 6px; padding: 8px;">
                <div style="font-weight: bold; font-size: 13px; color: {color}; border-bottom: 1px solid #ddd; padding-bottom: 6px; margin-bottom: 8px; text-align: center;">
                    {arm_name}
                </div>
                <div style="display: flex; flex-direction: column; gap: 8px;">
            """

            for cand in top3:
                rank = cand["rank"]
                v_id = cand["video_id"]
                f_id = cand["frame_id"]
                sec = f_id // 25
                time_str = f"{sec // 60:02d}:{sec % 60:02d}"

                v_path = find_video_path(v_id)
                img_b64 = extract_frame_base64(v_path, f_id, max_width=260) if v_path else None

                img_tag = (
                    f'<img src="data:image/jpeg;base64,{img_b64}" style="width: 100%; height: auto; border-radius: 4px; border: 1px solid #ccc;" />'
                    if img_b64
                    else '<div style="height: 100px; background: #eee; display: flex; align-items: center; justify-content: center; color: #888; font-size: 11px;">(No Video)</div>'
                )

                arm_html += f"""
                <div style="background: #ffffff; border: 1px solid #eee; border-radius: 4px; padding: 5px; text-align: center;">
                    <div style="font-size: 11px; font-weight: bold; color: #333; margin-bottom: 3px;">
                        Top @{rank}: {v_id} (f={f_id}, ~{time_str})
                    </div>
                    {img_tag}
                </div>
                """

            arm_html += """
                </div>
            </div>
            """
            card_html += arm_html

        card_html += """
            </div>
        </div>
        """
        html_cards.append(card_html)

    html_cards.append("</div>")
    return "\n".join(html_cards)


if __name__ == "__main__":
    inspect_token_budget()
    run_isolation_ablation()
