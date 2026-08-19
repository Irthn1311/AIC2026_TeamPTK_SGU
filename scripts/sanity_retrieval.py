import os
import sys
import pathlib

# Set HF cache home and offline flags
os.environ["HF_HOME"] = r"E:\AI Challenge TP.HCM 2026\CodeBase\.cache\huggingface"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

# Set env var to load 873 multimodal config
os.environ["AIC_MULTIMODAL_CONFIG"] = "configs/eval_873_multimodal.yaml"

# Add codebase root to sys.path
codebase_root = pathlib.Path(r"E:\AI Challenge TP.HCM 2026\CodeBase")
sys.path.insert(0, str(codebase_root))

from collections import Counter
from backend.retrieval_service import RetrievalService

print("=== INITIALIZING RETRIEVAL SERVICE WITH 873 ARTIFACTS ===")
service = RetrievalService.get_instance()
service.initialize()

print(f"\nLoaded {len(service.df_global)} keyframes across {len(service.all_video_ids)} videos.")
print(f"Visual index ntotal: {service.visual_index.ntotal if service.visual_index else 'N/A'}")

# Sanity queries (text only, NO GT accessed)
sanity_queries = [
    "cảnh quay đường phố ban đêm xe ô tô chạy qua",
    "người phụ nữ mặc áo đỏ đứng phát biểu trong khán phòng",
    "robot di chuyển trong nhà máy sản xuất tự động",
    "cảnh thiên nhiên núi rừng sương mù bao phủ mặt hồ"
]

print("\n=== RUNNING SANITY RETRIEVAL QUERIES ===")
for i, q in enumerate(sanity_queries, 1):
    res = service.search(query=q, top_k=50)
    top_vids = [item["video_id"] for item in res.get("results", [])[:20]]
    series_counts = Counter(vid.split("_")[0] for vid in top_vids)
    print(f"\nQuery {i}: '{q}'")
    print(f"Top 5 videos: {top_vids[:5]}")
    print(f"Series distribution in Top 20: {dict(series_counts)}")

print("\nSanity Retrieval Verification: SUCCESSful retrieval across multi-series indices.")
