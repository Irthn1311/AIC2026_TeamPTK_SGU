from __future__ import annotations

import numpy as np
import pandas as pd


def cross_shot_deduplicate(selected: pd.DataFrame, embeddings: dict[int, np.ndarray], cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    if selected.empty:
        return selected.copy(), pd.DataFrame()
    threshold = float(cfg.get("cross_shot_duplicate_threshold", 0.975))
    neighbor = int(cfg.get("cross_shot_max_neighbor_distance", 2))
    keep = set(int(x) for x in selected["candidate_frame_internal"].tolist())
    records = []
    rows = selected.sort_values(["shot_id", "candidate_frame_internal"]).to_dict("records")
    for i, a in enumerate(rows):
        for b in rows[i + 1 :]:
            if int(b["shot_id"]) - int(a["shot_id"]) > neighbor:
                break
            if int(a["candidate_frame_internal"]) not in keep or int(b["candidate_frame_internal"]) not in keep:
                continue
            ea = embeddings.get(int(a["candidate_frame_internal"]))
            eb = embeddings.get(int(b["candidate_frame_internal"]))
            if ea is None or eb is None:
                continue
            sim = float(np.dot(ea, eb))
            if sim >= threshold:
                if float(a["final_score"]) >= float(b["final_score"]):
                    kept, removed = a, b
                else:
                    kept, removed = b, a
                keep.discard(int(removed["candidate_frame_internal"]))
                records.append(
                    {
                        "frame_a": int(a["candidate_actual_frame_id"]),
                        "frame_b": int(b["candidate_actual_frame_id"]),
                        "shot_a": int(a["shot_id"]),
                        "shot_b": int(b["shot_id"]),
                        "timestamp_a": float(a["timestamp"]),
                        "timestamp_b": float(b["timestamp"]),
                        "clip_similarity": sim,
                        "score_a": float(a["final_score"]),
                        "score_b": float(b["final_score"]),
                        "kept_frame": int(kept["candidate_actual_frame_id"]),
                        "removed_frame": int(removed["candidate_actual_frame_id"]),
                        "reason": "cross_shot_duplicate",
                    }
                )
    final = selected[selected["candidate_frame_internal"].isin(keep)].copy()
    return final.sort_values("candidate_frame_internal").reset_index(drop=True), pd.DataFrame(records)
