"""Visualize Top-K Retrieved Keyframes from KIS V2-A.1 on Kaggle Notebook."""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from PIL import Image

from system_tai.data.corpus_discovery import discover_corpus, load_or_build_manifest_cache
from system_tai.kis.session_engine import OperationalKISRuntime
from system_tai.kis.session_schema import (
    KISVideoFirstConfig,
    QueryRequest,
    SessionConfig,
)


def run_visual_inspector(top_n_display: int = 5) -> None:
    print("=" * 100)
    print("🎨 KIS V2-A.1 — VISUAL KEYFRAME INSPECTOR (SƠ TUYỂN ĐỢT 1)")
    print("=" * 100)

    # 1. Locate Datasets
    input_root = Path("/kaggle/input")
    dataset_root = None
    manifest_cache = None

    if (input_root / "datasets").exists():
        dataset_root = input_root / "datasets"
    else:
        for p in input_root.glob("**/datasets"):
            if p.is_dir():
                dataset_root = p
                break
    if dataset_root is None:
        for p in input_root.iterdir():
            if p.is_dir() and any(p.glob("**/keyframes")):
                dataset_root = p
                break

    if dataset_root is None:
        dataset_root = input_root

    for p in input_root.glob("**/feature_manifest.json"):
        manifest_cache = p
        break

    print(f"• Dataset Root: {dataset_root}")
    print(f"• Manifest Cache: {manifest_cache}")

    output_root = Path("/kaggle/working/kis_visual_output")
    output_root.mkdir(parents=True, exist_ok=True)

    config = SessionConfig(
        input_root=dataset_root,
        output_root=output_root,
        enable_dynamic_translation=True,
        translation_model_name="vinai/vinai-translate-vi2en-v2",
        translation_allow_model_download=True,
        translation_device="auto",
        kis_video_first_config=KISVideoFirstConfig(
            enabled=True,
            v2_adaptive_enabled=True,
            selected_video_cap=32,
            top_m_evidence_cap=3,
            top_m_min_frame_gap=60,
            top_m_weights=(0.6, 0.3, 0.1),
            adaptive_budget_base=32,
            adaptive_budget_medium=48,
            adaptive_budget_high=64,
            coverage_threshold=0.75,
        ),
        manifest_cache=manifest_cache if manifest_cache and manifest_cache.exists() else None,
        reuse_manifest=None,
    )

    t0 = time.perf_counter()
    runtime = OperationalKISRuntime.bootstrap(config)
    print(f"• Bootstrap hoàn tất trong {time.perf_counter() - t0:.2f}s")
    print(f"• Tổng số video trong chỉ mục: {len(runtime.manifest.videos)}")

    # Index video artifacts by video_id
    videos_by_id = {v.video_id: v for v in runtime.manifest.videos}

    # 5 Official Preliminary Queries
    p1_queries = [
        {
            "query_id": "query-p1-1-kis",
            "source_vi": "Cảnh quay một nhóm hơn 5 người xếp thành hàng tập thể dục, cùng thực hiện động tác hai tay chạm mũi chân. Trong nhóm chỉ có một người đeo kính và ba người đội nón có màu đỏ.",
            "target_vid": "L30_V046",
            "target_frame": 2425,  # Keyframe 097
        },
        {
            "query_id": "query-p1-2-kis",
            "source_vi": "Đoạn phim bắt đầu bằng một bản đồ, trên đó một loại công trình thủy lợi lần lượt xuất hiện bốn lần. Sau đó chuyển sang cảnh một công trình thủy lợi lớn đang mở cửa xả nước dưới trời mưa.",
            "target_vid": "L29_V018",
            "target_frame": 6050,  # Keyframe 242
        },
        {
            "query_id": "query-p1-4-kis",
            "source_vi": "Đoạn phim bắt đầu với cảnh những chú sư tử đang nằm trên bục gỗ trong khu chuồng nuôi tại London Zoo. Sau đó là các cảnh quay nhân viên vườn thú đang tiến hành cân các loài động vật khác nhau.",
            "target_vid": "L28_V012",
            "target_frame": 1375,  # Keyframe 055
        },
        {
            "query_id": "query-p1-5-kis",
            "source_vi": "Cảnh một người cho đậu hà lan vào chảo xào cùng mực, sau đó là cảnh quay chậm cảnh người này xóc chảo để đảo đều thức ăn.",
            "target_vid": "L30_V021",
            "target_frame": 3325,  # Keyframe 133
        },
        {
            "query_id": "query-p1-6-kis",
            "source_vi": "Cảnh người đàn ông đang xem xét một khối đá quý, sau đó là cảnh một mỏ khai thác đá quý lộ thiên dạng bậc thang nhìn từ trên cao.",
            "target_vid": "L27_V005",
            "target_frame": 1150,  # Keyframe 046
        },
    ]

    image_extensions = {".jpg", ".jpeg", ".png", ".webp"}

    for idx, q in enumerate(p1_queries, start=1):
        qid = q["query_id"]
        target_vid = q["target_vid"]
        target_frame = q["target_frame"]
        q_vi = q["source_vi"]

        print("\n" + "=" * 100)
        print(f"🔍 [{idx}/{len(p1_queries)}] QUERY: {qid}")
        print(f"📝 Tiếng Việt: {q_vi}")
        print(f"🎯 Target Groundtruth: Video {target_vid} | Frame {target_frame}")
        print("=" * 100)

        req = QueryRequest(
            request_id=f"vis-{qid}",
            query_id=qid,
            query_vi=q_vi,
            query_en=None,
            include_vi_variant=True,
            output_top_k=100,
            refine_top_n=0,
        )

        out = runtime.handle_query(req)
        cand_path = runtime.output_root / out["artifacts"]["candidates_json"]
        cand_data = json.loads(cand_path.read_text(encoding="utf-8"))

        trans_meta = cand_data.get("translation", {})
        raw_en = trans_meta.get("raw_english") or trans_meta.get("primary_scene_en") or "N/A"
        print(f"🌐 VinAI Translation: {raw_en}")

        records = cand_data.get("records", [])
        top_candidates = records[:top_n_display]

        # Target rank check
        target_records = [r for r in records if r["video_id"] == target_vid]
        target_rank = target_records[0]["rank"] if target_records else 999
        print(f"🏆 Target Video Rank in Top 100: #{target_rank}")

        # Render Top Candidates with Matplotlib
        fig, axes = plt.subplots(1, top_n_display, figsize=(4 * top_n_display, 4.5))
        if top_n_display == 1:
            axes = [axes]

        fig.suptitle(
            f"[{qid}] Target: {target_vid} (Rank #{target_rank})\nVI: {q_vi[:80]}...\nEN: {raw_en[:80]}...",
            fontsize=11,
            fontweight="bold",
            y=1.08,
        )

        for col_idx, cand in enumerate(top_candidates):
            ax = axes[col_idx]
            vid = cand["video_id"]
            fid = cand["frame_id"]
            rank = cand["rank"]
            score = cand["fusion_score"]
            order = cand.get("keyframe_order_diagnostic") or 1

            is_target = (vid == target_vid)
            is_frame_match = is_target and (abs(fid - target_frame) <= 150)

            # Resolve Image Path
            img_path = None
            if vid in videos_by_id:
                v_record = videos_by_id[vid]
                kf_dir = v_record.keyframe_directory
                if kf_dir and kf_dir.exists():
                    # Match by order
                    matches = [
                        p for p in kf_dir.iterdir()
                        if p.is_file()
                        and p.suffix.lower() in image_extensions
                        and p.stem.isdigit()
                        and (int(p.stem) == order or int(p.stem) == fid)
                    ]
                    if matches:
                        img_path = matches[0]
                    else:
                        # try formatted names
                        for pattern in (f"{order:03d}.jpg", f"{order:04d}.jpg", f"{order:05d}.jpg", f"{order}.jpg"):
                            if (kf_dir / pattern).exists():
                                img_path = kf_dir / pattern
                                break

            # Load and show image
            if img_path and img_path.exists():
                try:
                    img = Image.open(img_path).convert("RGB")
                    ax.imshow(img)
                except Exception as e:
                    ax.text(0.5, 0.5, f"Err loading image:\n{e}", ha="center", va="center")
            else:
                ax.text(0.5, 0.5, f"Keyframe Image\nNot Found\nOrder: {order}", ha="center", va="center", color="gray")

            # Style title & borders
            ax.set_xticks([])
            ax.set_yticks([])

            if is_frame_match:
                title_color = "darkgreen"
                border_color = "lime"
                status_txt = "🎯 EXACT TARGET HIT!"
            elif is_target:
                title_color = "green"
                border_color = "green"
                status_txt = "🎬 TARGET VIDEO HIT"
            else:
                title_color = "navy"
                border_color = "gray"
                status_txt = f"Rank #{rank}"

            ax.set_title(
                f"Rank #{rank} | {vid}\nFrame: {fid} (Order: {order})\nScore: {score:.4f}\n{status_txt}",
                fontsize=9,
                color=title_color,
                fontweight="bold" if is_target else "normal",
            )
            for spine in ax.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(3 if is_target else 1)

        plt.tight_layout()
        plt.show()

    print("\n" + "=" * 100)
    print("✅ ĐÃ HOÀN TẤT HIỂN THỊ HÌNH ẢNH TOÀN BỘ 5 CÂU SƠ TUYỂN ĐỢT 1")
    print("=" * 100)


if __name__ == "__main__":
    top_k = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    run_visual_inspector(top_k)
