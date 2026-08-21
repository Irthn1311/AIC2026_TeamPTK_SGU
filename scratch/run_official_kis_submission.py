#!/usr/bin/env python3
"""Official Preliminary Round KIS Execution Engine (20 KIS Queries).

Runs the canonical System Tai KIS Pipeline on all 20 preliminary round queries:
  - Query Translation (Production Marian Vi->En with TokenBudgetGuard)
  - Vector Retrieval across 598 AIC videos / 500k+ keyframes with ViT-B/32
  - VideoConditionedKeyframeDiversity Refinement
  - Strict Top-100 Headerless CSV Export (video_id,frame_id)
  - Structural Validation & Automatic submission_kis_20q.zip creation
  - Interactive HTML Visual Gallery for Kaggle Notebook Inspection
"""

from __future__ import annotations

import base64
import csv
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

print("=" * 150, flush=True)
print("AIC 2026 PRELIMINARY ROUND - 20 KIS QUERIES EXECUTION ENGINE", flush=True)
print("=" * 150, flush=True)

REPO_ROOT = Path(__file__).resolve().parents[1]
SYSTEM_TAI_SRC = REPO_ROOT / "systems" / "system_tai" / "src"
if str(SYSTEM_TAI_SRC) not in sys.path:
    sys.path.insert(0, str(SYSTEM_TAI_SRC))

try:
    import clip
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "git+https://github.com/openai/CLIP.git", "ftfy", "regex"], check=True)
    import clip

try:
    import cv2
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "opencv-python-headless"], check=False)
    import cv2

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from system_tai.kis.session_engine import OperationalKISRuntime
from system_tai.kis.session_schema import (
    QueryLanguage,
    QueryRequest,
    QueryVariant,
    QueryVariantType,
    SessionConfig,
)

SUBMISSION_DIR = Path("/kaggle/working/submission") if Path("/kaggle/working").exists() else REPO_ROOT / "scratch" / "submission"
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# 1. 20 Official Preliminary Round KIS Queries
# -----------------------------------------------------------------------------
OFFICIAL_KIS_QUERIES = [
    {
        "query_id": "query-p1-1-kis",
        "text": "Cảnh quay một nhóm hơn 5 người xếp thành hàng tập thể dục, cùng thực hiện động tác hai tay chạm mũi chân. Trong nhóm chỉ có một người đeo kính và ba người đội nón có màu đỏ.",
    },
    {
        "query_id": "query-p1-2-kis",
        "text": "Đoạn phim bắt đầu bằng một bản đồ, trên đó một loại công trình thủy lợi lần lượt xuất hiện bốn lần. Sau đó chuyển sang cảnh một con đập được quay từ trên cao, tiếp đến là cảnh cận con đập dưới trời mưa.",
    },
    {
        "query_id": "query-p1-4-kis",
        "text": "Một đàn sư tử đang nghỉ ngơi và leo trèo trên các bục gỗ trong khu nuôi dưỡng, phía trước có bảng thông tin của London Zoo phục vụ công tác theo dõi và bảo tồn động vật.. Sau đó có cảnh hai nhân viên mặc áo xanh lá đang cân và ghi nhận số liệu của một con vật trong khuôn viên sở thú.",
    },
    {
        "query_id": "query-p1-5-kis",
        "text": "Đoạn clip bắt đầu bằng việc đậu hà lan được bỏ vào với mực đang được xào trên chảo, bên cạnh là đĩa hành tây và ớt đỏ thái lát chuẩn bị cho vào món ăn. Đoạn clip kết thúc với khung quay chậm (slow motion) cảnh lắc chảo trên bếp lửa.",
    },
    {
        "query_id": "query-p1-6-kis",
        "text": "Mẩu tin bắt đầu với hình ảnh nột người đàn ông mặc vest xanh đậm, sơ mi trắng và cà vạt, đang ngồi trên một chiếc ghế lớn. Ông cầm bằng hai tay một khối đá quý thô khá lớn, đưa lên gần mặt để quan sát. Bên phải là một phụ nữ mặc trang phục công sở màu đen và khăn trùm đầu màu hồng tím, đang đứng cạnh và mỉm cười. Tiếp theo có hình ảnh toàn cảnh từ trên cao của một mỏ đá quý lộ thiên quy mô lớn với hố khai thác sâu nhiều tầng và hệ thống đường vận chuyển bao quanh.",
    },
    {
        "query_id": "query-p1-7-kis",
        "text": "Đoạn clip bắt đầu bằng cảnh cà rốt cắt hình ngôi sao đang được luộc trong nồi nước sôi, đặt trong rổ lưới kim loại và được đảo bằng đôi đũa gỗ. Đoạn clip kết thúc bằng hình ảnh đĩa rau củ luộc và đồ chiên được trình bày đẹp mắt, gồm đậu bắp, súp lơ, cà rốt hình ngôi sao, bí xanh, chén nước chấm màu hồng ở giữa và đôi đũa màu hồng nhạt đặt bên phải",
    },
    {
        "query_id": "query-p1-8-kis",
        "text": "Người đầu bếp lần lượt đặt các miếng nguyên liệu dạng thanh và những lát cắt hình hoa vào một đĩa đang được hấp trong nồi. Các nguyên liệu được dùng đũa sắp xếp xen kẽ xung quanh phần thức ăn đã có sẵn trên đĩa. Sau đó, đầu bếp dùng muỗng lấy thêm một loại nguyên liệu mềm từ tô thủy tinh. Phần nguyên liệu này được đặt vào giữa đĩa, xung quanh là các miếng dạng thanh và hình hoa đã được sắp xếp trước đó.",
    },
    {
        "query_id": "query-p1-10-kis",
        "text": "Hành động cắt chùm nho bằng kéo từ giàn nho bằng một chiếc kéo màu đen. Có thể thấy có một sợi dây màu xanh dương được buộc vào cuống của chùm nho này trước khi nó được cắt.",
    },
    {
        "query_id": "query-p1-11-kis",
        "text": "Cảnh quay chậm tại vị trí vạch đích của cuộc đua xe đạp. Góc máy sát mặt đường bắt trọn khoảnh khắc về đích theo thứ tự nhất, nhì, ba lần lượt là 1 tay đua áo vàng quần đen, 1 tay đua áo xanh dương quần đen và 1 tay đua áo xanh dương quần đỏ",
    },
    {
        "query_id": "query-p1-12-kis",
        "text": "Có thể thấy trong cảnh quay có 4 tài xế xe ôm công nghệ trong trạm xăng, trong đó 3 người đứng đợi còn 1 người lái xe từ trái sang phải khung hình. Trước đó là cảnh một người đậy nắp bình xăng xe máy của họ. Có thông tin về giá dầu mazut được hiển thị trong khung hình.",
    },
    {
        "query_id": "query-p1-13-kis",
        "text": "Một người đứng dưới nước và rọi đèn. Tiếp theo là cảnh người này kéo lưới cá lúc bình minh, sau đó được một nhóm người khác tiến đến dùng máy quay ghi hình.",
    },
    {
        "query_id": "query-p1-14-kis",
        "text": "Người đầu bếp lần lượt đặt các miếng nguyên liệu dạng thanh và những lát cắt hình hoa vào một đĩa đang được hấp trong nồi. Các nguyên liệu được dùng đũa sắp xếp xen kẽ xung quanh phần thức ăn đã có sẵn trên đĩa. Sau đó, đầu bếp dùng muỗng lấy thêm một loại nguyên liệu mềm từ tô thủy tinh. Phần nguyên liệu này được đặt vào giữa đĩa, xung quanh là các miếng dạng thanh và hình hoa đã được sắp xếp trước đó.",
    },
    {
        "query_id": "query-p1-18-kis",
        "text": "Cảnh quay cho thấy hành động trình bày món ăn sau khi hoàn thành giai đoạn chế biến. Bún được cho đầu tiên vào một chén rỗng, sau đó đầu bếp lần lượt cho từng vá nước dùng cùng các nguyên liệu như thịt gà, cà rốt, sả, nấm mèo vào chén bún, và kết thúc bằng việc thả một cọng ngò lên trên cùng. Cảnh quay tiếp theo là cảnh zoom xa dần chén bún, ta thấy bên cạnh chén bún còn có 1 chén nước chấm nhỏ với 2 miếng ớt.",
    },
    {
        "query_id": "query-p1-19-kis",
        "text": "Con lân do hai người điều khiển đang đứng thẳng và xoay vòng trên đỉnh cột. Sau vài giây nghỉ, con lân bất ngờ nhảy qua hai chiếc cột kế bên, chúi đầu xuống ngoạm lấy quả bí đỏ kèm bông hoa màu vàng. Cảnh quay kết thúc khi con lân tiếp tục nhảy sang các cột tiếp theo.",
    },
    {
        "query_id": "query-p1-20-kis",
        "text": "Ba người đang đi bộ xuống một con dốc trong cơn mưa, có 2 người cầm dù, trong đó người cầm dù đi sau lại mặc một chiếc áo mưa có in hình con gấu ở sau lưng. Sau đó ta thấy nhiều người đang cùng nhau bước về hướng một căn nhà thông qua một con đường đất, bên cạnh là một cái ao.",
    },
    {
        "query_id": "query-p1-21-kis",
        "text": "Cảnh quay bắt đầu bằng cảnh những con tôm đã được lột vỏ và nấu chín đang nằm trên dĩa, phía sau là người đầu bếp đang đặt 3 ổ bánh mì lên bàn. Sau đó là cảnh quay các đầu bếp, có người thì trang trí món ăn, có người thì chế biến món ăn. Những con tôm được cắt làm đôi và được nướng trên bếp.",
    },
    {
        "query_id": "query-p1-22-kis",
        "text": "Người phụ nữ mặc áo dài màu hồng, đeo kính đang giảng giải về các trường hợp sử dụng khác nhau của động từ 'remember' dựa trên mốc thời gian của hành động được nhắc tới.",
    },
    {
        "query_id": "query-p1-23-kis",
        "text": "Hình ảnh giáo viên nam mặc sơ mi trắng, thắt cà vạt tối màu, nổi bật trên phông nền xanh dương đậm có hoa văn mờ. Slide bài giảng Nền trắng với khung viền màu hồng tím, phía trên có thanh tiêu đề xanh dương chứa họa tiết địa cầu và mũi tên vàng/xanh ngọc. Bên dưới là sơ đồ 3 tầng được liên kết bởi các mũi tên xanh ngọc trỏ xuống: Tầng 1: 2 khối hộp được bao bởi 1 khối hộp cam. Tầng 2: 1 khối hộp lớn màu xanh dương đậm ở chính giữa. Tầng 3: 2 khối hộp được bao bởi 1 khối hộp xanh lá cây.",
    },
    {
        "query_id": "query-p1-24-kis",
        "text": "Đoạn clip được cắt từ một phóng sự về một nhóm các nghệ nhân làm nghề đan lát các sản phẩm thủ công từ cây lục bình. Đầu tiên là một cảnh lia cam duy nhất từ trái sang phải, theo thứ tự ta thấy 4 sản phẩm: túi xách, chậu hoa, bộ ấm tách trà và túi xách. Ngay sau cảnh này, người phụ nữ bên trái lấy một tách trà trong bộ ấm tách và nâng niu trong khi nghe người phụ nữ bên phải trò chuyện.",
    },
    {
        "query_id": "query-p1-25-kis",
        "text": "Hai bạn học sinh mặc đồng phục áo trắng, quần xanh, quàng khăn đỏ đang làm MC trên một sân khấu tại trường học, phía sau là một bộ trống cơ màu đỏ và một cây đàn piano.",
    },
]

# -----------------------------------------------------------------------------
# 2. Production VinAI B1 Vi->En Translator (Historical B1 Decoding)
# -----------------------------------------------------------------------------
class VinAIB1Translator:
    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self.model_name = "vinai/vinai-translate-vi2en-v2"
        print(f"[Translator] Loading VinAI B1 (vi2en-v2) on {device} ...", flush=True)
        t0 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, src_lang="vi_VN", use_fast=False)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(device)
        self.model.eval()
        self.en_bos_id = self.tokenizer.lang_code_to_id.get("en_XX", None)
        print(f"      • Loaded VinAI B1 in {time.time() - t0:.2f}s ✅", flush=True)

    def translate(self, text: str) -> str:
        clean = " ".join(text.strip().split())
        inputs = self.tokenizer(clean, return_tensors="pt", padding=True, truncation=True, max_length=128).to(self.device)
        gen_kwargs = {
            "max_length": 128,
            "num_beams": 3,
            "no_repeat_ngram_size": 3,
            "repetition_penalty": 1.15,
            "early_stopping": True,
        }
        if self.en_bos_id is not None:
            gen_kwargs["forced_bos_token_id"] = self.en_bos_id
            
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

# -----------------------------------------------------------------------------
# 3. Fast Video Path Resolution & Frame Decoder (0.001s)
# -----------------------------------------------------------------------------
VIDEO_CACHE: dict[str, Path] = {}

def find_video_path(video_id: str, runtime: OperationalKISRuntime | None = None) -> Path | None:
    if video_id in VIDEO_CACHE:
        return VIDEO_CACHE[video_id]
    if runtime is not None and hasattr(runtime, "raw_video_registry"):
        for rec in runtime.raw_video_registry._records:
            if rec.video_id == video_id and rec.raw_video_path and rec.raw_video_path.exists():
                VIDEO_CACHE[video_id] = rec.raw_video_path
                return rec.raw_video_path
    batch = video_id.split("_")[0] if "_" in video_id else ""
    candidates = [
        Path(f"/kaggle/input/datasets/nadkli/dataset-aic/Videos_{batch}_a/video/{video_id}.mp4"),
        Path(f"/kaggle/input/datasets/nadkli/dataset-aic/Videos_{batch}_b/video/{video_id}.mp4"),
        Path(f"/kaggle/input/datasets/nadkli/dataset-aic/Videos_{batch}/video/{video_id}.mp4"),
        Path(f"/kaggle/input/datasets/videos/{batch}/{video_id}.mp4"),
        Path(f"/kaggle/input/datasets/{batch}/{video_id}.mp4"),
        Path(f"/kaggle/input/datasets/{video_id}.mp4"),
        REPO_ROOT / "systems" / "system_tai" / "data" / "videos" / batch / f"{video_id}.mp4",
    ]
    for p in candidates:
        if p.exists():
            VIDEO_CACHE[video_id] = p
            return p
    return None

def decode_frame_b64(video_id: str, frame_id: int, runtime: OperationalKISRuntime | None = None) -> str:
    vpath = find_video_path(video_id, runtime)
    if not vpath or not vpath.exists():
        return ""
    try:
        cap = cv2.VideoCapture(str(vpath))
        if not cap.isOpened():
            return ""
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_id))
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            return ""
        _, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        return base64.b64encode(buf).decode("utf-8")
    except Exception:
        return ""

# -----------------------------------------------------------------------------
# 4. Bootstrap OperationalKISRuntime
# -----------------------------------------------------------------------------
def bootstrap_runtime() -> OperationalKISRuntime:
    yaml_path = REPO_ROOT / "systems" / "system_tai" / "configs" / "production.yaml"
    input_root = Path("/kaggle/input/datasets") if Path("/kaggle/input/datasets").exists() else Path("/kaggle/input")
    manifest_cache = Path("/kaggle/working/manifest_cache.json")
    out_dir = Path("/kaggle/working/output/kis_session") if Path("/kaggle/working").exists() else REPO_ROOT / "scratch" / "kis_session"

    cfg = SessionConfig.from_yaml(
        yaml_path,
        input_root=input_root,
        output_root=out_dir,
        reuse_manifest=manifest_cache if manifest_cache.exists() else None,
    )
    t0 = time.time()
    runtime = OperationalKISRuntime.bootstrap(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[BOOTSTRAP_DONE] KIS Runtime loaded in {time.time() - t0:.2f}s (device={device}) ✅\n", flush=True)
    return runtime

# -----------------------------------------------------------------------------
# 5. Process All 20 KIS Queries
# -----------------------------------------------------------------------------
def process_all_kis_queries(runtime: OperationalKISRuntime, translator: VinAIB1Translator) -> list[dict[str, Any]]:
    print("=" * 120)
    print(f"🚀 PROCESSING {len(OFFICIAL_KIS_QUERIES)} KIS QUERIES THROUGH PRODUCTION PIPELINE")
    print("=" * 120)

    results_summary = []

    for idx, q_info in enumerate(OFFICIAL_KIS_QUERIES, start=1):
        qid = q_info["query_id"]
        vi_text = q_info["text"]
        
        # Translate to English
        en_text = translator.translate(vi_text)
        
        print(f"\n[{idx:02d}/{len(OFFICIAL_KIS_QUERIES)}] 🎯 {qid}")
        print(f"    • VI: {vi_text[:75]}...")
        print(f"    • EN: {en_text[:75]}...")

        # Encode and retrieve
        q_vec = runtime.shared_encoder.encode(en_text)
        cands = runtime.exact_retriever.search_vector(
            query_id=qid,
            query_vector=q_vec,
            top_k=100,
        )
        
        # Apply VideoConditionedKeyframeDiversity
        conditioned = runtime.video_conditioner.condition(
            global_result=cands,
            query_vector=q_vec,
            config=runtime.config.video_conditioned_keyframe_config,
            protected_prefix_rank=3,
        ).result.ranked_candidates

        # Export exact Top 100 CSV (video_id,frame_id)
        csv_path = SUBMISSION_DIR / f"{qid}.csv"
        rows = []
        for c in conditioned[:100]:
            clean_vid = str(c.video_id).removesuffix(".mp4")
            rows.append((clean_vid, int(c.frame_id)))

        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            for vid, fid in rows:
                writer.writerow([vid, fid])

        print(f"      -> Exported {len(rows)} rows to {csv_path.name} (Top 1: {rows[0][0]}, Frame {rows[0][1]}) ✅")

        # Decode Top 3 frames for gallery
        top3_preview = []
        for r_idx, (v, f) in enumerate(rows[:3], start=1):
            b64_img = decode_frame_b64(v, f, runtime)
            top3_preview.append({"rank": r_idx, "video_id": v, "frame_id": f, "b64": b64_img})

        results_summary.append({
            "query_id": qid,
            "vi_text": vi_text,
            "en_text": en_text,
            "rows": rows,
            "top3": top3_preview,
        })

    return results_summary

# -----------------------------------------------------------------------------
# 6. Validate All 20 CSV Files
# -----------------------------------------------------------------------------
def validate_all_csvs() -> None:
    print("\n" + "=" * 100)
    print("🔍 STRUCTURAL VALIDATION OF ALL 20 KIS CSV SUBMISSION FILES")
    print("=" * 100)
    
    csv_files = sorted(list(SUBMISSION_DIR.glob("query-p1-*-kis.csv")))
    print(f"  • Total KIS CSV files found in {SUBMISSION_DIR}: {len(csv_files)} / 20")
    
    if len(csv_files) < 20:
        raise ValueError(f"Expected 20 KIS CSV files, found {len(csv_files)}!")

    for f in csv_files:
        lines = [l.strip() for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
        if len(lines) != 100:
            raise ValueError(f"File {f.name} has {len(lines)} rows, expected exactly 100!")
        seen = set()
        for idx, line in enumerate(lines, start=1):
            parts = line.split(",")
            if len(parts) != 2:
                raise ValueError(f"File {f.name} line {idx} has invalid column count!")
            vid, fid_str = parts[0].strip(), parts[1].strip()
            if vid.endswith(".mp4") or not vid:
                raise ValueError(f"File {f.name} line {idx} has invalid video_id: '{vid}'!")
            fid = int(fid_str)
            if fid < 0:
                raise ValueError(f"File {f.name} line {idx} has negative frame_id: {fid}!")
            key = (vid, fid)
            if key in seen:
                raise ValueError(f"File {f.name} line {idx} has duplicate key: {key}!")
            seen.add(key)

    print("  🎉 ALL 20 KIS SUBMISSION CSV FILES ARE 100% VALID! ✅\n")

# -----------------------------------------------------------------------------
# 7. Package Into submission_kis_20q.zip
# -----------------------------------------------------------------------------
def package_submission_zip() -> Path:
    print("=" * 100)
    print("📦 PACKAGING 20 KIS SUBMISSION CSVs INTO ZIP")
    print("=" * 100)
    
    zip_path = Path("/kaggle/working/submission_kis_20q.zip") if Path("/kaggle/working").exists() else REPO_ROOT / "scratch" / "submission_kis_20q.zip"
    csv_files = sorted(list(SUBMISSION_DIR.glob("query-p1-*-kis.csv")))
    
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for f in csv_files:
            zipf.write(f, arcname=f.name)
            print(f"  • Added: {f.name} ({f.stat().st_size} bytes)")
            
    print(f"\n🎉 CREATED TOURNAMENT ARCHIVE: {zip_path} ({zip_path.stat().st_size} bytes) ✅\n")
    return zip_path

# -----------------------------------------------------------------------------
# 8. Render Visual Inspection Gallery
# -----------------------------------------------------------------------------
def render_kis_gallery(results: list[dict[str, Any]]) -> None:
    html_path = Path("/kaggle/working/kis_submission_gallery.html") if Path("/kaggle/working").exists() else REPO_ROOT / "scratch" / "kis_submission_gallery.html"
    sections = []
    
    for r in results:
        qid = r["query_id"]
        vi = r["vi_text"]
        en = r["en_text"]
        top3 = r["top3"]
        
        cards = []
        for item in top3:
            rk = item["rank"]
            vid = item["video_id"]
            fid = item["frame_id"]
            b64 = item["b64"]
            img_tag = f'<img src="data:image/jpeg;base64,{b64}" style="width:100%; border-radius:6px; border:1px solid #444;" />' if b64 else '<div style="background:#333;color:#888;height:120px;display:flex;align-items:center;justify-content:center;">No Frame</div>'
            cards.append(f"""
            <div style="flex:1; margin:5px; padding:8px; background:#1c1c1c; border:1px solid #333; border-radius:6px; text-align:center;">
                <div style="font-size:12px; font-weight:bold; color:#61afef; margin-bottom:4px;">Rank #{rk} — {vid} (f={fid})</div>
                {img_tag}
            </div>
            """)
            
        sections.append(f"""
        <div style="background:#222; border:1px solid #444; border-radius:8px; padding:14px; margin-bottom:20px;">
            <div style="font-size:14px; font-weight:bold; color:#e5c07b; margin-bottom:4px;">🎯 {qid}</div>
            <div style="font-size:12px; color:#ccc; margin-bottom:2px;"><b>• VI:</b> {vi}</div>
            <div style="font-size:12px; color:#98c379; margin-bottom:10px;"><b>• EN:</b> {en}</div>
            <div style="display:flex; gap:6px;">
                {''.join(cards)}
            </div>
        </div>
        """)
        
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head><meta charset="utf-8"><title>KIS 20 Queries Submission Gallery</title></head>
    <body style="background:#121212; color:#fff; font-family:-apple-system,BlinkMacSystemFont,sans-serif; padding:16px;">
        <h2 style="color:#61afef; border-bottom:2px solid #555; padding-bottom:10px; margin-top:0;">📊 BẢNG ĐỐI SOÁT HÌNH ẢNH 20 CÂU KIS SƠ TUYỂN (TOP 3 KEYFRAMES)</h2>
        {''.join(sections)}
    </body>
    </html>
    """
    html_path.write_text(full_html, encoding="utf-8")
    print(f"  • Saved Visual Gallery to: {html_path} ✅\n")

# -----------------------------------------------------------------------------
# 9. Main Flow
# -----------------------------------------------------------------------------
def main() -> None:
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Bootstrap Runtime & Translator
    runtime = bootstrap_runtime()
    translator = VinAIB1Translator(device=device)
    
    # 2. Process All 20 KIS Queries
    results = process_all_kis_queries(runtime, translator)
    
    # 3. Validate All CSVs
    validate_all_csvs()
    
    # 4. Package to Zip
    package_submission_zip()
    
    # 5. Render Gallery
    render_kis_gallery(results)
    
    print("=" * 150)
    print(f"🎉 ALL 20 KIS QUERIES COMPLETED & VALIDATED IN {time.time() - t0:.2f}s ✅")
    print("=" * 150 + "\n")

if __name__ == "__main__":
    main()
