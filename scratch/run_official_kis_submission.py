#!/usr/bin/env python3
"""Official Preliminary Round KIS Execution Engine (20 KIS Queries - Multi-Variant RRF Fusion + Q3.1 Diversity).

Architectural Highlights:
  1. Multi-Clause Scene Decomposition: Deconstructs each query into full distilled visual prompt + scene sub-clauses (<40 tokens).
  2. Domain-Accurate Visual Terminology: Fixes all translation traps (camera pan, lion dance, grape bunch with blue string, etc.).
  3. Weighted RRF Rank Fusion: Fuses multiple semantic visual perspectives per query before conditioning.
  4. VideoConditionedKeyframeDiversity (Q3.1): Bounded keyframe diversity across candidate videos.
  5. Exact 20-Query Target Verification: Strict validation, packaging into submission_kis_20q.zip, and visual inspection gallery.
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
print("AIC 2026 PRELIMINARY ROUND - 20 KIS MULTI-VARIANT RRF FUSION & Q3.1 DIVERSITY ENGINE", flush=True)
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
from system_tai.kis.session_engine import OperationalKISRuntime
from system_tai.kis.session_schema import (
    QueryLanguage,
    QueryVariant,
    QueryVariantType,
    SessionConfig,
)
from system_tai.retrieval.video_restricted import VideoConditionedKeyframeConfig

SUBMISSION_DIR = Path("/kaggle/working/submission") if Path("/kaggle/working").exists() else REPO_ROOT / "scratch" / "submission"
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

# Explicitly enable VideoConditionedKeyframeDiversity (Q3.1)
DIVERSITY_CONFIG = VideoConditionedKeyframeConfig(
    enabled=True,
    selected_video_global_rank_cap=50,
    max_selected_videos=50,
    max_anchors_per_video=3,
    minimum_anchor_gap_seconds=5.0,
    preserve_first_video_occurrence=True,
)

# -----------------------------------------------------------------------------
# 1. 20 Official Preliminary Round KIS Queries with Grounded Scene Variants
# -----------------------------------------------------------------------------
OFFICIAL_KIS_BENCHMARK = [
    {
        "query_id": "query-p1-1-kis",
        "vi_text": "Cảnh quay một nhóm hơn 5 người xếp thành hàng tập thể dục, cùng thực hiện động tác hai tay chạm mũi chân. Trong nhóm chỉ có một người đeo kính và ba người đội nón có màu đỏ.",
        "variants": [
            ("full", "group of over 5 people in a line doing exercise touching toes with both hands, one person wearing glasses, three people wearing red hats", 1.0),
            ("scene_a", "group of people exercising in a line touching their toes with both hands", 0.7),
            ("scene_b", "one person wearing glasses and three people wearing red hats exercising", 0.6),
        ],
    },
    {
        "query_id": "query-p1-2-kis",
        "vi_text": "Đoạn phim bắt đầu bằng một bản đồ, trên đó một loại công trình thủy lợi lần lượt xuất hiện bốn lần. Sau đó chuyển sang cảnh một con đập được quay từ trên cao, tiếp đến là cảnh cận con đập dưới trời mưa.",
        "variants": [
            ("full", "irrigation project map appearing four times, aerial view of a dam, close-up of the dam in the rain", 1.0),
            ("scene_a", "map with water irrigation construction project appearing four times", 0.7),
            ("scene_b", "aerial high angle view of water dam, close-up of dam under rain", 0.8),
        ],
    },
    {
        "query_id": "query-p1-4-kis",
        "vi_text": "Một đàn sư tử đang nghỉ ngơi và leo trèo trên các bục gỗ trong khu nuôi dưỡng, phía trước có bảng thông tin của London Zoo phục vụ công tác theo dõi và bảo tồn động vật.. Sau đó có cảnh hai nhân viên mặc áo xanh lá đang cân và ghi nhận số liệu của một con vật trong khuôn viên sở thú.",
        "variants": [
            ("full", "pride of lions resting and climbing on wooden platforms in enclosure with London Zoo sign, two zookeepers in green weighing an animal", 1.0),
            ("scene_a", "lions on wooden platforms in enclosure, London Zoo information sign board", 0.8),
            ("scene_b", "two zoo keepers in green shirts weighing and recording data of an animal in zoo", 0.7),
        ],
    },
    {
        "query_id": "query-p1-5-kis",
        "vi_text": "Đoạn clip bắt đầu bằng việc đậu hà lan được bỏ vào với mực đang được xào trên chảo, bên cạnh là đĩa hành tây và ớt đỏ thái lát chuẩn bị cho vào món ăn. Đoạn clip kết thúc với khung quay chậm (slow motion) cảnh lắc chảo trên bếp lửa.",
        "variants": [
            ("full", "peas added to squid stir-frying in a pan, plate of sliced onions and red chili, slow motion tossing pan over flame stove", 1.0),
            ("scene_a", "green peas added to squid stir-frying in cooking pan with sliced onion and red chili", 0.8),
            ("scene_b", "slow motion chef tossing stir-fry pan over stove fire flame", 0.7),
        ],
    },
    {
        "query_id": "query-p1-6-kis",
        "vi_text": "Mẩu tin bắt đầu với hình ảnh nột người đàn ông mặc vest xanh đậm, sơ mi trắng và cà vạt, đang ngồi trên một chiếc ghế lớn. Ông cầm bằng hai tay một khối đá quý thô khá lớn, đưa lên gần mặt để quan sát. Bên phải là một phụ nữ mặc trang phục công sở màu đen và khăn trùm đầu màu hồng tím, đang đứng cạnh và mỉm cười. Tiếp theo có hình ảnh toàn cảnh từ trên cao của một mỏ đá quý lộ thiên quy mô lớn với hố khai thác sâu nhiều tầng và hệ thống đường vận chuyển bao quanh.",
        "variants": [
            ("full", "man in dark blue suit white shirt holding large rough gemstone inspecting up close, woman in black business outfit purple-pink hijab smiling, aerial view of open-pit gemstone mine", 1.0),
            ("scene_a", "man in dark blue suit holding large rough gemstone close to face, woman in black suit and pink-purple hijab smiling", 0.8),
            ("scene_b", "aerial high angle view of large open-pit gemstone mining quarry with terraced pit and winding roads", 0.8),
        ],
    },
    {
        "query_id": "query-p1-7-kis",
        "vi_text": "Đoạn clip bắt đầu bằng cảnh cà rốt cắt hình ngôi sao đang được luộc trong nồi nước sôi, đặt trong rổ lưới kim loại và được đảo bằng đôi đũa gỗ. Đoạn clip kết thúc bằng hình ảnh đĩa rau củ luộc và đồ chiên được trình bày đẹp mắt, gồm đậu bắp, súp lơ, cà rốt hình ngôi sao, bí xanh, chén nước chấm màu hồng ở giữa và đôi đũa màu hồng nhạt đặt bên phải",
        "variants": [
            ("full", "star-shaped cut carrots boiling in water pot with metal mesh basket stirred with wooden chopsticks, plate of boiled vegetables with star carrots broccoli okra pink dip and pink chopsticks", 1.0),
            ("scene_a", "star cut carrots boiling in pot with wire mesh basket stirred with wooden chopsticks", 0.8),
            ("scene_b", "plate of boiled vegetables with star-shaped carrots, broccoli, okra, pink dipping sauce bowl in middle and pink chopsticks on right", 0.8),
        ],
    },
    {
        "query_id": "query-p1-8-kis",
        "vi_text": "Người đầu bếp lần lượt đặt các miếng nguyên liệu dạng thanh và những lát cắt hình hoa vào một đĩa đang được hấp trong nồi. Các nguyên liệu được dùng đũa sắp xếp xen kẽ xung quanh phần thức ăn đã có sẵn trên đĩa. Sau đó, đầu bếp dùng muỗng lấy thêm một loại nguyên liệu mềm từ tô thủy tinh. Phần nguyên liệu này được đặt vào giữa đĩa, xung quanh là các miếng dạng thanh và hình hoa đã được sắp xếp trước đó.",
        "variants": [
            ("full", "chef placing bar-shaped ingredients and flower-shaped slices on steamed plate in pot with chopsticks, chef using spoon to place soft filling in center of plate surrounded by bar and flower slices", 1.0),
            ("scene_a", "chef using chopsticks arranging bar-shaped ingredients and flower slices on plate steaming in pot", 0.8),
            ("scene_b", "chef using spoon taking soft ingredient from glass bowl placing into center of plate", 0.7),
        ],
    },
    {
        "query_id": "query-p1-10-kis",
        "vi_text": "Hành động cắt chùm nho bằng kéo từ giàn nho bằng một chiếc kéo màu đen. Có thể thấy có một sợi dây màu xanh dương được buộc vào cuống của chùm nho này trước khi nó được cắt.",
        "variants": [
            ("full", "cutting grape bunch with black scissors from vineyard trellis, blue string tied to stem of grape bunch before cutting", 1.0),
            ("scene_a", "cutting bunch of grapes with black scissors from grapevine trellis", 0.8),
            ("scene_b", "blue string tied to the stem of grape bunch before being cut with scissors", 0.7),
        ],
    },
    {
        "query_id": "query-p1-11-kis",
        "vi_text": "Cảnh quay chậm tại vị trí vạch đích của cuộc đua xe đạp. Góc máy sát mặt đường bắt trọn khoảnh khắc về đích theo thứ tự nhất, nhì, ba lần lượt là 1 tay đua áo vàng quần đen, 1 tay đua áo xanh dương quần đen và 1 tay đua áo xanh dương quần đỏ",
        "variants": [
            ("full", "slow motion finish line of cycling bicycle race ground level camera angle, first yellow jersey black shorts, second blue jersey black shorts, third blue jersey red shorts", 1.0),
            ("scene_a", "slow motion ground level camera angle at finish line of bicycle road race", 0.8),
            ("scene_b", "cyclists finishing race in order: yellow jersey black shorts, blue jersey black shorts, blue jersey red shorts", 0.8),
        ],
    },
    {
        "query_id": "query-p1-12-kis",
        "vi_text": "Có thể thấy trong cảnh quay có 4 tài xế xe ôm công nghệ trong trạm xăng, trong đó 3 người đứng đợi còn 1 người lái xe từ trái sang phải khung hình. Trước đó là cảnh một người đậy nắp bình xăng xe máy của họ. Có thông tin về giá dầu mazut được hiển thị trong khung hình.",
        "variants": [
            ("full", "petrol gas station with 4 motorbike ride-hailing drivers, three waiting and one driving left to right, closing motorcycle fuel cap, mazut oil fuel price display", 1.0),
            ("scene_a", "person closing motorcycle petrol gas fuel tank cap, mazut fuel oil price info on display", 0.8),
            ("scene_b", "four ride-hailing motorbike delivery drivers at gas station, three standing waiting, one driving left to right", 0.8),
        ],
    },
    {
        "query_id": "query-p1-13-kis",
        "vi_text": "Một người đứng dưới nước và rọi đèn. Tiếp theo là cảnh người này kéo lưới cá lúc bình minh, sau đó được một nhóm người khác tiến đến dùng máy quay ghi hình.",
        "variants": [
            ("full", "person standing in water shining flashlight, pulling fishing net at sunrise dawn, group of people approaching with video camera filming", 1.0),
            ("scene_a", "man standing in water shining flashlight lamp, pulling fish net at sunrise", 0.8),
            ("scene_b", "group of people approaching with video camera recording filming the fisherman", 0.7),
        ],
    },
    {
        "query_id": "query-p1-14-kis",
        "vi_text": "Người đầu bếp lần lượt đặt các miếng nguyên liệu dạng thanh và những lát cắt hình hoa vào một đĩa đang được hấp trong nồi. Các nguyên liệu được dùng đũa sắp xếp xen kẽ xung quanh phần thức ăn đã có sẵn trên đĩa. Sau đó, đầu bếp dùng muỗng lấy thêm một loại nguyên liệu mềm từ tô thủy tinh. Phần nguyên liệu này được đặt vào giữa đĩa, xung quanh là các miếng dạng thanh và hình hoa đã được sắp xếp trước đó.",
        "variants": [
            ("full", "chef placing bar-shaped ingredients and flower-shaped slices on steamed plate in pot with chopsticks, chef using spoon to place soft filling in center of plate surrounded by bar and flower slices", 1.0),
            ("scene_a", "chef using chopsticks arranging bar-shaped ingredients and flower slices on plate steaming in pot", 0.8),
            ("scene_b", "chef using spoon taking soft ingredient from glass bowl placing into center of plate", 0.7),
        ],
    },
    {
        "query_id": "query-p1-18-kis",
        "vi_text": "Cảnh quay cho thấy hành động trình bày món ăn sau khi hoàn thành giai đoạn chế biến. Bún được cho đầu tiên vào một chén rỗng, sau đó đầu bếp lần lượt cho từng vá nước dùng cùng các nguyên liệu như thịt gà, cà rốt, sả, nấm mèo vào chén bún, và kết thúc bằng việc thả một cọng ngò lên trên cùng. Cảnh quay tiếp theo là cảnh zoom xa dần chén bún, ta thấy bên cạnh chén bún còn có 1 chén nước chấm nhỏ với 2 miếng ớt.",
        "variants": [
            ("full", "plating food in bowl: rice vermicelli noodles first, ladle chicken broth with chicken meat, carrots, lemongrass, wood ear fungus, cilantro on top, zoom out showing bowl with small chili dipping sauce", 1.0),
            ("scene_a", "chef assembling noodle soup bowl: vermicelli noodles, ladle broth with chicken, carrots, lemongrass, black fungus, cilantro on top", 0.8),
            ("scene_b", "camera zooming out from noodle bowl showing small dipping sauce bowl with two chili slices next to it", 0.8),
        ],
    },
    {
        "query_id": "query-p1-19-kis",
        "vi_text": "Con lân do hai người điều khiển đang đứng thẳng và xoay vòng trên đỉnh cột. Sau vài giây nghỉ, con lân bất ngờ nhảy qua hai chiếc cột kế bên, chúi đầu xuống ngoạm lấy quả bí đỏ kèm bông hoa màu vàng. Cảnh quay kết thúc khi con lân tiếp tục nhảy sang các cột tiếp theo.",
        "variants": [
            ("full", "lion dance on high poles standing upright and spinning, lion jumping across two poles diving head down biting yellow pumpkin and yellow flower, jumping to next poles", 1.0),
            ("scene_a", "lion dance performance two performers standing upright and spinning on top of high poles", 0.8),
            ("scene_b", "lion jumping across poles diving head down biting pumpkin with yellow flower, leaping to next poles", 0.8),
        ],
    },
    {
        "query_id": "query-p1-20-kis",
        "vi_text": "Ba người đang đi bộ xuống một con dốc trong cơn mưa, có 2 người cầm dù, trong đó người cầm dù đi sau lại mặc một chiếc áo mưa có in hình con gấu ở sau lưng. Sau đó ta thấy nhiều người đang cùng nhau bước về hướng một căn nhà thông qua một con đường đất, bên cạnh là một cái ao.",
        "variants": [
            ("full", "three people walking down a slope in rain with two umbrellas, rear person wearing raincoat with bear print on back, people walking towards house on dirt path beside pond", 1.0),
            ("scene_a", "three people walking down a slope in the rain with umbrellas, rear person wearing raincoat with bear picture on back", 0.8),
            ("scene_b", "people walking together towards a house along a dirt path next to a pond", 0.7),
        ],
    },
    {
        "query_id": "query-p1-21-kis",
        "vi_text": "Cảnh quay bắt đầu bằng cảnh những con tôm đã được lột vỏ và nấu chín đang nằm trên dĩa, phía sau là người đầu bếp đang đặt 3 ổ bánh mì lên bàn. Sau đó là cảnh quay các đầu bếp, có người thì trang trí món ăn, có người thì chế biến món ăn. Những con tôm được cắt làm đôi và được nướng trên bếp.",
        "variants": [
            ("full", "peeled cooked shrimp on plate, chef placing 3 baguettes on table, chefs decorating and cooking food, halved shrimp grilling on stove", 1.0),
            ("scene_a", "peeled cooked shrimp on plate, chef placing three baguette breads on table", 0.8),
            ("scene_b", "chefs preparing and decorating dishes, halved shrimp grilling on barbecue stove", 0.8),
        ],
    },
    {
        "query_id": "query-p1-22-kis",
        "vi_text": "Người phụ nữ mặc áo dài màu hồng, đeo kính đang giảng giải về các trường hợp sử dụng khác nhau của động từ 'remember' dựa trên mốc thời gian của hành động được nhắc tới.",
        "variants": [
            ("full", "woman in pink ao dai wearing glasses teaching English grammar explaining usage of verb 'remember' based on timeline", 1.0),
            ("scene_a", "woman wearing pink traditional ao dai dress and glasses teaching lecture on stage or whiteboard", 0.8),
            ("scene_b", "English grammar lesson explaining different uses of verb 'remember' with timeline", 0.7),
        ],
    },
    {
        "query_id": "query-p1-23-kis",
        "vi_text": "Hình ảnh giáo viên nam mặc sơ mi trắng, thắt cà vạt tối màu, nổi bật trên phông nền xanh dương đậm có hoa văn mờ. Slide bài giảng Nền trắng với khung viền màu hồng tím, phía trên có thanh tiêu đề xanh dương chứa họa tiết địa cầu và mũi tên vàng/xanh ngọc. Bên dưới là sơ đồ 3 tầng được liên kết bởi các mũi tên xanh ngọc trỏ xuống: Tầng 1: 2 khối hộp được bao bởi 1 khối hộp cam. Tầng 2: 1 khối hộp lớn màu xanh dương đậm ở chính giữa. Tầng 3: 2 khối hộp được bao bởi 1 khối hộp xanh lá cây.",
        "variants": [
            ("full", "male teacher in white shirt and dark tie in front of dark blue patterned background, slide with pink-purple border, blue title bar with globe and yellow/cyan arrows, 3-tier diagram with cyan downward arrows: top orange box with 2 boxes, middle large blue box, bottom green box with 2 boxes", 1.0),
            ("scene_a", "male teacher wearing white shirt and dark tie standing in front of dark blue patterned background", 0.8),
            ("scene_b", "slide presentation 3-tier diagram with downward cyan arrows: orange box on top, blue box in middle, green box at bottom with pink border", 0.8),
        ],
    },
    {
        "query_id": "query-p1-24-kis",
        "vi_text": "Đoạn clip được cắt từ một phóng sự về một nhóm các nghệ nhân làm nghề đan lát các sản phẩm thủ công từ cây lục bình. Đầu tiên là một cảnh lia cam duy nhất từ trái sang phải, theo thứ tự ta thấy 4 sản phẩm: túi xách, chậu hoa, bộ ấm tách trà và túi xách. Ngay sau cảnh này, người phụ nữ bên trái lấy một tách trà trong bộ ấm tách và nâng niu trong khi nghe người phụ nữ bên phải trò chuyện.",
        "variants": [
            ("full", "documentary about artisans weaving water hyacinth craft products, camera pan from left to right showing 4 items: handbag, flowerpot, tea set, handbag, woman on left holding tea cup talking to woman on right", 1.0),
            ("scene_a", "documentary about artisans weaving water hyacinth crafts, camera pan left to right showing handbag, flower pot, teapot set, handbag", 0.8),
            ("scene_b", "woman on left holding woven tea cup admiring it while listening to woman on right talking", 0.8),
        ],
    },
    {
        "query_id": "query-p1-25-kis",
        "vi_text": "Hai bạn học sinh mặc đồng phục áo trắng, quần xanh, quàng khăn đỏ đang làm MC trên một sân khấu tại trường học, phía sau là một bộ trống cơ màu đỏ và một cây đàn piano.",
        "variants": [
            ("full", "two students in white uniform shirt blue pants red scarf acting as MC on school stage, red acoustic drum set and piano in background", 1.0),
            ("scene_a", "two young students wearing white shirts, blue pants and red scarves speaking into microphones as MCs on school stage", 0.8),
            ("scene_b", "school stage background showing a red acoustic drum set and a piano keyboard behind the presenters", 0.8),
        ],
    },
]

TARGET_QIDS = [q["query_id"] for q in OFFICIAL_KIS_BENCHMARK]

# -----------------------------------------------------------------------------
# 2. Fast Video Path Resolution & Frame Decoder (0.001s)
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
# 3. Bootstrap OperationalKISRuntime
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
# 4. Multi-Variant RRF Fusion & Retrieval Engine
# -----------------------------------------------------------------------------
def process_multi_variant_kis_queries(runtime: OperationalKISRuntime) -> list[dict[str, Any]]:
    print("=" * 120)
    print(f"🚀 PROCESSING {len(OFFICIAL_KIS_BENCHMARK)} KIS QUERIES VIA MULTI-VARIANT RRF FUSION + Q3.1 DIVERSITY")
    print("=" * 120)

    results_summary = []

    for idx, q_info in enumerate(OFFICIAL_KIS_BENCHMARK, start=1):
        qid = q_info["query_id"]
        vi_text = q_info["vi_text"]
        variants_def = q_info["variants"]

        print(f"\n[{idx:02d}/{len(OFFICIAL_KIS_BENCHMARK)}] 🎯 {qid}")
        print(f"    • VI: {vi_text[:80]}...")

        # 1. Encode variants & search
        variant_objs = []
        rankings = {}
        primary_vec = None

        for var_id, var_text, var_weight in variants_def:
            v_obj = QueryVariant(
                variant_id=var_id,
                text=var_text,
                language=QueryLanguage.ENGLISH,
                variant_type=QueryVariantType.ENGLISH_TRANSLATION,
                weight=var_weight,
            )
            variant_objs.append(v_obj)

            v_vec = runtime.shared_encoder.encode(var_text)
            if primary_vec is None:
                primary_vec = v_vec

            cands = runtime.exact_retriever.search_vector(
                query_id=f"{qid}::{var_id}",
                query_vector=v_vec,
                top_k=100,
            )
            rankings[var_id] = cands
            print(f"      - Variant [{var_id:<7}] (weight={var_weight}): '{var_text[:65]}...'")

        # 2. Weighted RRF Fusion
        fused_result = runtime.multi_query_fuser.fuse_rankings(
            query_id=qid,
            variants=tuple(variant_objs),
            rankings=rankings,
            output_top_k=100,
            rrf_constant=60.0,
        )

        # 3. VideoConditionedKeyframeDiversity (Q3.1)
        conditioned = runtime.video_conditioner.condition(
            global_result=fused_result,
            query_vector=primary_vec,
            config=DIVERSITY_CONFIG,
            protected_prefix_rank=3,
        ).result.ranked_candidates

        # 4. Export exact Top 100 CSV (video_id,frame_id)
        csv_path = SUBMISSION_DIR / f"{qid}.csv"
        rows = []
        for c in conditioned[:100]:
            clean_vid = str(c.video_id).removesuffix(".mp4")
            rows.append((clean_vid, int(c.frame_id)))

        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            for vid, fid in rows:
                writer.writerow([vid, fid])

        print(f"      🏆 Fused Top 1: Video={rows[0][0]}, Frame={rows[0][1]} -> Saved {len(rows)} rows to {csv_path.name} ✅")

        # 5. Decode Top 3 frames for gallery
        top3_preview = []
        for r_idx, (v, f) in enumerate(rows[:3], start=1):
            b64_img = decode_frame_b64(v, f, runtime)
            top3_preview.append({"rank": r_idx, "video_id": v, "frame_id": f, "b64": b64_img})

        results_summary.append({
            "query_id": qid,
            "vi_text": vi_text,
            "variants": variants_def,
            "rows": rows,
            "top3": top3_preview,
        })

    return results_summary

# -----------------------------------------------------------------------------
# 5. Validate All 20 CSV Files
# -----------------------------------------------------------------------------
def validate_all_csvs() -> None:
    print("\n" + "=" * 100)
    print("🔍 STRUCTURAL VALIDATION OF ALL 20 KIS CSV SUBMISSION FILES")
    print("=" * 100)
    
    csv_files = [SUBMISSION_DIR / f"{qid}.csv" for qid in TARGET_QIDS]
    print(f"  • Verifying exactly {len(csv_files)} target KIS CSV files in {SUBMISSION_DIR} ...")

    for f in csv_files:
        if not f.exists():
            raise FileNotFoundError(f"Missing target KIS file: {f.name}!")
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
# 6. Package Into submission_kis_20q.zip
# -----------------------------------------------------------------------------
def package_submission_zip() -> Path:
    print("=" * 100)
    print("📦 PACKAGING EXACT 20 KIS SUBMISSION CSVs INTO ZIP")
    print("=" * 100)
    
    zip_path = Path("/kaggle/working/submission_kis_20q.zip") if Path("/kaggle/working").exists() else REPO_ROOT / "scratch" / "submission_kis_20q.zip"
    csv_files = [SUBMISSION_DIR / f"{qid}.csv" for qid in TARGET_QIDS]
    
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for f in csv_files:
            zipf.write(f, arcname=f.name)
            print(f"  • Added: {f.name} ({f.stat().st_size} bytes)")
            
    print(f"\n🎉 CREATED TOURNAMENT ARCHIVE (EXACT 20 FILES): {zip_path} ({zip_path.stat().st_size} bytes) ✅\n")
    return zip_path

# -----------------------------------------------------------------------------
# 7. Render Visual Inspection Gallery
# -----------------------------------------------------------------------------
def render_kis_gallery(results: list[dict[str, Any]]) -> None:
    html_path = Path("/kaggle/working/kis_submission_gallery.html") if Path("/kaggle/working").exists() else REPO_ROOT / "scratch" / "kis_submission_gallery.html"
    sections = []
    
    for r in results:
        qid = r["query_id"]
        vi = r["vi_text"]
        variants = r["variants"]
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
            
        variants_html = "".join([f"<div style='font-size:11px; color:#98c379; margin-bottom:2px;'><b>• {v[0]} ({v[2]}):</b> {v[1]}</div>" for v in variants])
            
        sections.append(f"""
        <div style="background:#222; border:1px solid #444; border-radius:8px; padding:14px; margin-bottom:20px;">
            <div style="font-size:14px; font-weight:bold; color:#e5c07b; margin-bottom:4px;">🎯 {qid}</div>
            <div style="font-size:12px; color:#ccc; margin-bottom:6px;"><b>• VI:</b> {vi}</div>
            <div style="background:#181818; padding:8px; border-radius:4px; margin-bottom:10px;">
                {variants_html}
            </div>
            <div style="display:flex; gap:6px;">
                {''.join(cards)}
            </div>
        </div>
        """)
        
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head><meta charset="utf-8"><title>KIS 20 Queries Multi-Variant Submission Gallery</title></head>
    <body style="background:#121212; color:#fff; font-family:-apple-system,BlinkMacSystemFont,sans-serif; padding:16px;">
        <h2 style="color:#61afef; border-bottom:2px solid #555; padding-bottom:10px; margin-top:0;">📊 BẢNG ĐỐI SOÁT HÌNH ẢNH 20 CÂU KIS (MULTI-VARIANT RRF FUSION + Q3.1 DIVERSITY)</h2>
        {''.join(sections)}
    </body>
    </html>
    """
    html_path.write_text(full_html, encoding="utf-8")
    print(f"  • Saved Visual Gallery to: {html_path} ✅\n")

# -----------------------------------------------------------------------------
# 8. Main Flow
# -----------------------------------------------------------------------------
def main() -> None:
    t0 = time.time()
    
    # 0. Clean submission directory before running
    for old_f in SUBMISSION_DIR.glob("*.csv"):
        try:
            old_f.unlink()
        except Exception:
            pass
            
    # 1. Bootstrap Runtime
    runtime = bootstrap_runtime()
    
    # 2. Process All 20 KIS Queries via Multi-Variant RRF Fusion
    results = process_multi_variant_kis_queries(runtime)
    
    # 3. Validate All CSVs
    validate_all_csvs()
    
    # 4. Package to Zip
    package_submission_zip()
    
    # 5. Render Gallery
    render_kis_gallery(results)
    
    print("=" * 150)
    print(f"🎉 ALL 20 KIS MULTI-VARIANT QUERIES COMPLETED & VALIDATED IN {time.time() - t0:.2f}s ✅")
    print("=" * 150 + "\n")

if __name__ == "__main__":
    main()
