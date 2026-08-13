from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import faiss
import numpy as np
import pandas as pd
import torch

from _bootstrap import PROJECT_ROOT
from src.preprocessing.keyframe_v2.pipeline import load_config
from src.retrieval.object_index import ObjectIndex


DEFAULT_QUERIES = [
    "người dẫn chương trình trong trường quay",
    "người cầm micro",
    "xe ô tô trên đường",
    "thuyền trên sông",
    "đám đông",
    "màn hình TV",
]


class OpenClipTextEncoder:
    def __init__(self, cfg: dict):
        extra = cfg.get("open_clip_extra_site_packages")
        if extra:
            extra_path = str((PROJECT_ROOT / str(extra)).resolve())
            if extra_path not in sys.path:
                sys.path.append(extra_path)
        import open_clip

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        weights = sorted(PROJECT_ROOT.glob(str(cfg["open_clip_weights"])))
        if not weights:
            raise FileNotFoundError(cfg["open_clip_weights"])
        self.weights_path = weights[0]
        self.model, _, _ = open_clip.create_model_and_transforms(
            str(cfg.get("model_name", "ViT-B-32")),
            pretrained=str(self.weights_path),
            device=self.device,
        )
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(str(cfg.get("model_name", "ViT-B-32")))

    def encode(self, text: str) -> np.ndarray:
        tokens = self.tokenizer([text])
        if hasattr(tokens, "to"):
            tokens = tokens.to(self.device)
        with torch.no_grad():
            feat = self.model.encode_text(tokens).float()
            feat = feat / feat.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return feat.cpu().numpy().astype(np.float32)[0]


def minmax(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    lo, hi = float(series.min()), float(series.max())
    if abs(hi - lo) < 1e-9:
        return pd.Series([1.0] * len(series), index=series.index)
    return (series - lo) / (hi - lo)


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test Visual V2 + Object V2 retrieval.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "keyframe_v2.yaml"))
    parser.add_argument("--global-map", default=str(PROJECT_ROOT / "outputs" / "keyframe_v2_full" / "indexes" / "keyframe_v2_global_map.parquet"))
    parser.add_argument("--visual-index", default=str(PROJECT_ROOT / "outputs" / "keyframe_v2_full" / "indexes" / "visual" / "l21_visual_v2_flat_ip.faiss"))
    parser.add_argument("--object-corpus", default=str(PROJECT_ROOT / "outputs" / "keyframe_v2_full" / "indexes" / "object" / "l21_objects_v2.parquet"))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "outputs" / "keyframe_v2_full" / "summary" / "retrieval_v2_smoke_results.json"))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--candidate-pool", type=int, default=50)
    args = parser.parse_args()

    cfg = load_config(args.config)
    encoder = OpenClipTextEncoder(cfg["clip"])
    visual_index = faiss.read_index(args.visual_index)
    global_map = pd.read_parquet(args.global_map)
    object_index = ObjectIndex(args.object_corpus)

    all_results: dict[str, list[dict]] = {}
    for query in DEFAULT_QUERIES:
        q = encoder.encode(query)
        scores, ids = visual_index.search(q.reshape(1, -1), args.candidate_pool)
        visual_rows = []
        for rank, (idx, score) in enumerate(zip(ids[0], scores[0]), start=1):
            if idx < 0:
                continue
            row = global_map.iloc[int(idx)].to_dict()
            row["global_v2_id"] = int(row["global_v2_id"])
            row["visual_score"] = float(score)
            row["visual_rank"] = rank
            visual_rows.append(row)
        visual_df = pd.DataFrame(visual_rows)
        obj_df = object_index.search(query, top_k=args.candidate_pool)
        if not obj_df.empty:
            obj_df["global_v2_id"] = obj_df["global_v2_id"].astype(int)
        visual_keep = visual_df[["global_v2_id", "visual_score"]] if not visual_df.empty else pd.DataFrame(columns=["global_v2_id", "visual_score"])
        object_keep = obj_df[["global_v2_id", "object_match_score"]] if not obj_df.empty else pd.DataFrame(columns=["global_v2_id", "object_match_score"])
        fused = pd.merge(visual_keep, object_keep, on="global_v2_id", how="outer").fillna(0)
        if fused.empty:
            all_results[query] = []
            continue
        fused["visual_norm"] = minmax(fused["visual_score"].astype(float))
        fused["object_norm"] = minmax(fused["object_match_score"].astype(float))
        fused["final_score"] = 0.65 * fused["visual_norm"] + 0.35 * fused["object_norm"]
        fused = fused.sort_values("final_score", ascending=False).head(args.top_k)
        rows = []
        for _, row in fused.iterrows():
            g = global_map.iloc[int(row["global_v2_id"])].to_dict()
            rows.append(
                {
                    "video_id": str(g["video_id"]),
                    "actual_frame_id": int(g["actual_frame_id"]),
                    "timestamp_sec": float(g["timestamp_sec"]),
                    "image_path": str(g["image_path"]),
                    "visual_score": float(row["visual_score"]),
                    "object_score": float(row["object_match_score"]),
                    "final_score": float(row["final_score"]),
                    "global_v2_id": int(row["global_v2_id"]),
                }
            )
        all_results[query] = rows

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
    flat = [{"query": q, **row} for q, rows in all_results.items() for row in rows]
    pd.DataFrame(flat).to_csv(output.with_suffix(".csv"), index=False, encoding="utf-8-sig")
    print(json.dumps(all_results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
