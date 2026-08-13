from _bootstrap import PROJECT_ROOT
import pandas as pd
import numpy as np
import faiss
from src.retrieval.clip_text_encoder import ClipTextEncoder

index = faiss.read_index(r'outputs/indexes/visual/l21_visual_flat_ip.faiss')
df = pd.read_parquet(r'outputs/indexes/l21_global_id_map.parquet')
encoder = ClipTextEncoder()

query_en = "farmers working in green rice field"
q_emb = encoder.encode(query_en).astype(np.float32)

scores, indices = index.search(q_emb, 5)

print(f"Top 5 Visual Results for EN query: '{query_en}':")
for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), 1):
    row = df.iloc[idx]
    print(f"#{rank} Score: {score:.4f} | Video: {row['video_id']} | Frame: {row['frame_idx']} ({row['timestamp_text']}) | File: {row['keyframe_name']}")
