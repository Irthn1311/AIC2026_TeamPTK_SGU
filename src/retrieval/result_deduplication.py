from __future__ import annotations

import pandas as pd


def temporal_deduplicate(df: pd.DataFrame, window_seconds: float = 5.0) -> pd.DataFrame:
    if df.empty or "video_id" not in df.columns or "timestamp_seconds" not in df.columns:
        return df.copy()
    rows = []
    for _, group in df.sort_values(["fused_score"], ascending=False).groupby("video_id", sort=False):
        kept = []
        for _, row in group.sort_values(["fused_score"], ascending=False).iterrows():
            ts = float(row["timestamp_seconds"])
            if all(abs(ts - float(prev["timestamp_seconds"])) > window_seconds for prev in kept):
                kept.append(row)
        rows.extend(kept)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("fused_score", ascending=False).reset_index(drop=True)
        out["dedup_rank"] = range(1, len(out) + 1)
    return out

