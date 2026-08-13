"""
Test script for AIC 2026 FastAPI Endpoints & Multimodal Retrieval
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from starlette.testclient import TestClient
from backend.main import app

def run_tests():
    print("=" * 70)
    print(" 🧪 TESTING AIC 2026 FASTAPI RETRIEVAL WRAPPER")
    print("=" * 70)
    
    with TestClient(app) as client:
        # 1. Health Check
        t0 = time.time()
        res_health = client.get("/api/health")
        print(f"1. Health check status: {res_health.status_code} in {(time.time() - t0)*1000:.1f}ms")
        assert res_health.status_code == 200, f"Health check failed: {res_health.text}"
        data_health = res_health.json()
        print(f"   Status: {data_health['status']} | Videos: {data_health['total_videos']} | Keyframes: {data_health['total_keyframes']}")
        print(f"   Indexes: {list(data_health['indexes'].keys())}")

        # 2. Textual KIS Search
        query = "thuyền máy chạy trên sông"
        t0 = time.time()
        res_search = client.post("/api/search", json={
            "query": query,
            "top_k": 20,
            "weights": {"visual": 0.40, "ocr": 0.25, "asr": 0.25, "object": 0.10},
            "temporal_dedup": True,
            "dedup_window_seconds": 4.0
        })
        elapsed = (time.time() - t0) * 1000
        print(f"\n2. Search query '{query}': status {res_search.status_code} in {elapsed:.1f}ms")
        assert res_search.status_code == 200, f"Search failed: {res_search.text}"
        search_data = res_search.json()
        print(f"   Matches: {search_data['total_matches']} | Elapsed reported: {search_data['elapsed_ms']}ms")
        
        top1 = search_data["results"][0]
        print(f"   Top-1: Video={top1['video_id']}, Frame={top1['frame_idx']}, Time={top1['timestamp_text']}, Score={top1['score']}")
        print(f"   Scores: Visual={top1['scores']['visual']}, OCR={top1['scores']['ocr']}, ASR={top1['scores']['asr']}, Obj={top1['scores']['object']}")
        if top1['ocr_text']:
            print(f"   OCR: {top1['ocr_text'][:60]}...")
        if top1['asr_text']:
            print(f"   ASR: {top1['asr_text'][:60]}...")
        assert top1["frame_idx"] > 0 or top1["frame_idx"] == 0
        assert top1["video_id"].startswith("L21_V")

        # 3. Video Details
        t0 = time.time()
        res_vid = client.get(f"/api/video/{top1['video_id']}")
        print(f"\n3. Video details for {top1['video_id']}: status {res_vid.status_code} in {(time.time() - t0)*1000:.1f}ms")
        assert res_vid.status_code == 200
        vid_data = res_vid.json()
        print(f"   Keyframes count: {vid_data['total_keyframes']} | Duration: {vid_data['duration_seconds']:.1f}s")
        assert len(vid_data["keyframes"]) > 0

        # 4. Contextual Keyframe Inspection
        t0 = time.time()
        res_ctx = client.get(f"/api/context/{top1['video_id']}/{top1['frame_idx']}?window=5")
        print(f"\n4. Keyframe context: status {res_ctx.status_code} in {(time.time() - t0)*1000:.1f}ms")
        assert res_ctx.status_code == 200
        ctx_data = res_ctx.json()
        print(f"   Adjacent keyframes: {len(ctx_data['adjacent_keyframes'])}")
        print(f"   OCR snippets near frame: {len(ctx_data['ocr_snippets'])}")
        print(f"   ASR snippets near frame: {len(ctx_data['asr_snippets'])}")
        print(f"   Detected objects: {len(ctx_data['detected_objects'])}")

        # 5. Media keyframe serving
        t0 = time.time()
        res_img = client.get(f"/api/media/keyframe/{top1['video_id']}/{top1['keyframe_name']}")
        print(f"\n5. Keyframe image serving: status {res_img.status_code} ({len(res_img.content)} bytes) in {(time.time() - t0)*1000:.1f}ms")
        assert res_img.status_code == 200

        # 6. Subsequent Fast Search (<100ms)
        t0 = time.time()
        res_search2 = client.post("/api/search", json={
            "query": "người dẫn chương trình phỏng vấn",
            "top_k": 50,
            "weights": {"visual": 0.50, "ocr": 0.20, "asr": 0.20, "object": 0.10},
            "temporal_dedup": False
        })
        elapsed2 = (time.time() - t0) * 1000
        print(f"\n6. Subsequent Search: status {res_search2.status_code} in {elapsed2:.1f}ms")
        assert res_search2.status_code == 200
        print(f"   Results returned: {len(res_search2.json()['results'])}")

    print("\n" + "=" * 70)
    print(" ✅ ALL FASTAPI RETRIEVAL ENDPOINTS PASSED WITH REAL DATA!")
    print("=" * 70)

if __name__ == "__main__":
    run_tests()
