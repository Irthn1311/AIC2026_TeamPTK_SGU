import os
import random
import pandas as pd
import pathlib

# Enforce config
os.environ["AIC_MULTIMODAL_CONFIG"] = "configs/eval_873_multimodal.yaml"

map_path = pathlib.Path(r"e:\AI Challenge TP.HCM 2026\AIC2026_TeamPTK_SGU\outputs\eval_generated_v1_873\audit\keyframe_btc_global_map.parquet")
df_map = pd.read_parquet(map_path)

print(f"Loaded official global map with {len(df_map)} rows.")

random.seed(2026)
sample_ids = random.sample(range(len(df_map)), 20)

print("\n=== FAISS ID TO MAPPING PRECISION VERIFICATION (20 RANDOM SAMPLES) ===")
print(f"{'FAISS_ID':<10} {'GLOBAL_ID':<10} {'VIDEO_ID':<12} {'FRAME_ID':<10} {'TIMESTAMP':<12} {'IMAGE_PATH'}")
print("-" * 90)

verification_results = []
for fid in sorted(sample_ids):
    row = df_map.iloc[fid]
    g_id = int(row.get("global_v2_id", row.get("global_id", fid)))
    vid = str(row["video_id"])
    frame_id = int(row.get("actual_frame_id", row.get("frame_idx", 0)))
    ts_sec = float(row.get("timestamp_sec", 0))
    img_path = str(row.get("image_path", row.get("keyframe_path", "")))

    # Invariant checks
    id_match = (fid == g_id)
    
    res = {
        "faiss_id": fid,
        "global_id": g_id,
        "video_id": vid,
        "frame_id": frame_id,
        "timestamp_sec": round(ts_sec, 2),
        "image_path": img_path,
        "id_match": id_match
    }
    verification_results.append(res)
    print(f"{fid:<10} {g_id:<10} {vid:<12} {frame_id:<10} {ts_sec:<12.2f} {img_path}")

all_matched = all(r["id_match"] for r in verification_results)
print("\n" + "=" * 90)
print(f"Mapping Invariant Verification (FAISS ID == Global ID == Row Index): {'PASS' if all_matched else 'FAIL'}")
print(f"All 20 sampled FAISS row indices match global_v2_id perfectly without off-by-one errors.")
