import tarfile
import pandas as pd
import io
import json
import pathlib
from collections import defaultdict
import numpy as np

tar_p = pathlib.Path(r'e:\AI Challenge TP.HCM 2026\CodeBase\full_873_multimodal_artifacts.tar.gz')
targets = ['L22_V023', 'L30_V028', 'L30_V029', 'L30_V038', 'L30_V045']

target_info = {}
series_counts = defaultdict(lambda: {'videos': set(), 'vectors': 0})
video_vector_counts = {}

with tarfile.open(tar_p, 'r:gz') as tar:
    btc_map_files = [m for m in tar.getmembers() if m.name.endswith('keyframe_btc_map.parquet')]
    dfs = []
    for m in btc_map_files:
        df = pd.read_parquet(io.BytesIO(tar.extractfile(m).read()))
        dfs.append(df)
    
    full_map_df = pd.concat(dfs, ignore_index=True)
    full_map_df.sort_values(by='global_id', inplace=True)
    full_map_df.reset_index(drop=True, inplace=True)

print('Full map concatenated shape:', full_map_df.shape)
print('Global ID min/max:', full_map_df['global_id'].min(), full_map_df['global_id'].max())

for vid, group in full_map_df.groupby('video_id'):
    n_vecs = len(group)
    video_vector_counts[vid] = n_vecs
    ser = vid.split('_')[0] if '_' in vid else 'UNKNOWN'
    series_counts[ser]['videos'].add(vid)
    series_counts[ser]['vectors'] += n_vecs

    if vid in targets:
        frame_col = 'actual_frame_id' if 'actual_frame_id' in group.columns else 'frame_idx'
        min_f = int(group[frame_col].min())
        max_f = int(group[frame_col].max())
        
        if 'timestamp_ms' in group.columns:
            min_ts = group['timestamp_ms'].min() / 1000.0
            max_ts = group['timestamp_ms'].max() / 1000.0
            ts_cov = f"{min_ts:.1f}s - {max_ts:.1f}s"
        else:
            ts_cov = f"frame {min_f}-{max_f}"

        target_info[vid] = {
            'status': 'FOUND',
            'vectors': n_vecs,
            'min_frame': min_f,
            'max_frame': max_f,
            'timestamp_cov': ts_cov
        }

for t in targets:
    if t not in target_info:
        target_info[t] = {'status': 'MISSING', 'vectors': 0, 'min_frame': 'N/A', 'max_frame': 'N/A', 'timestamp_cov': 'N/A'}

print('\n=== TARGET VIDEO COVERAGE ===')
for vid, info in target_info.items():
    print(f"{vid}: {info}")

print('\n=== SERIES BREAKDOWN ===')
for ser in sorted(series_counts.keys()):
    v_cnt = len(series_counts[ser]['videos'])
    vec_cnt = series_counts[ser]['vectors']
    print(f"{ser}: {v_cnt} videos, {vec_cnt} vectors")

vec_list = list(video_vector_counts.values())
print('\n=== VECTOR DISTRIBUTION PER VIDEO ===')
print('Min vecs/video:', np.min(vec_list))
print('Median vecs/video:', np.median(vec_list))
print('Max vecs/video:', np.max(vec_list))

# Save aggregated global_map to audit dir for fast loading later!
out_parquet = pathlib.Path(r'e:\AI Challenge TP.HCM 2026\AIC2026_TeamPTK_SGU\outputs\eval_generated_v1_873\audit\keyframe_btc_global_map.parquet')
full_map_df.to_parquet(out_parquet, index=False)
print('Saved aggregated global_map to:', out_parquet)
