"""
KIS Multimodal Reranker V1
=========================
Two-stage pipeline:
1. Top-50 Raw Candidate Temporal Grouping (TemporalCandidateGrouper)
2. Multimodal Evidence Aggregation (SegmentEvidenceAggregator)
3. Text Cross-Encoder Reranking (BAAI/bge-reranker-base)
4. Score Fusion & Candidate Ordering
"""

from __future__ import annotations

import os
import sys
import math
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Enforce offline mode and Drive E cache
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HOME"] = str(PROJECT_ROOT / ".cache" / "huggingface")
os.environ["TORCH_HOME"] = str(PROJECT_ROOT / ".cache" / "torch")
os.environ["TRANSFORMERS_CACHE"] = str(PROJECT_ROOT / ".cache" / "huggingface" / "hub")

from src.retrieval.temporal_grouper import TemporalCandidateGrouper, CandidateSegment
from src.retrieval.evidence_aggregator import SegmentEvidenceAggregator

logger = logging.getLogger("aic.kis_reranker")


class KISRerankerV1:
    """
    Reranks KIS candidate segments using a Text Cross-Encoder (BAAI/bge-reranker-base)
    for text queries and SigLIP2 Feature Rescoring for visual queries.
    """

    _instance: Optional[KISRerankerV1] = None

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        fusion_alpha: float = 0.55,
        window_seconds: float = 4.0,
        max_duration_seconds: float = 10.0,
        device: str = "auto",
    ):
        self.model_name = model_name
        self.fusion_alpha = fusion_alpha
        self.grouper = TemporalCandidateGrouper(
            window_seconds=window_seconds,
            max_duration_seconds=max_duration_seconds,
            prefer_shot_id=True,
        )
        self.aggregator = SegmentEvidenceAggregator()
        self.device = device
        self.cross_encoder = None
        self._initialized = False

        # SigLIP2 Visual Rescorer attributes (Disabled in Production V1.2; Stage-1 OpenCLIP ViT-B/32 Fusion is frozen)
        self.siglip2_beta = 0.0
        self.enable_siglip2 = False
        self.siglip2_processor = None
        self.siglip2_model = None
        self.siglip2_embeddings_tensor = None
        self.siglip2_gid_to_row = None
        self._siglip2_initialized = False
        self.translator = None

    @classmethod
    def get_instance(cls) -> KISRerankerV1:
        if cls._instance is None:
            cls._instance = KISRerankerV1()
        return cls._instance

    def initialize(self) -> None:
        """Load Cross-Encoder model into memory offline from Drive E cache."""
        if self._initialized:
            return

        try:
            from sentence_transformers import CrossEncoder
            cache_folder = PROJECT_ROOT / ".cache" / "huggingface" / "hub"
            
            import torch
            if self.device == "auto":
                dev = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                dev = self.device

            logger.info("Loading Cross-Encoder %s on %s...", self.model_name, dev)
            self.cross_encoder = CrossEncoder(
                self.model_name,
                device=dev,
                cache_folder=str(cache_folder),
                local_files_only=True,
            )
            self._initialized = True
            logger.info("Cross-Encoder %s ready.", self.model_name)
        except Exception as e:
            logger.error("Failed to load Cross-Encoder %s: %s", self.model_name, e, exc_info=True)
            self._initialized = False

    def initialize_siglip2(self) -> None:
        """Lazy load precomputed SigLIP2 embeddings and text encoder from Drive E (Experimental)."""
        if self._siglip2_initialized or not self.enable_siglip2:
            return

        try:
            import torch
            from transformers import AutoProcessor, AutoModel
            from src.retrieval.query_translation import CachedQueryTranslator

            emb_path = PROJECT_ROOT / "outputs" / "indexes" / "siglip2_keyframe_v2_embeddings.pt"
            if emb_path.exists():
                cached_data = torch.load(str(emb_path), map_location="cpu")
                self.siglip2_embeddings_tensor = cached_data["embeddings"].float()
                gids = cached_data["global_v2_ids"].numpy()
                self.siglip2_gid_to_row = {int(gid): idx for idx, gid in enumerate(gids)}
                logger.info("Loaded precomputed SigLIP2 embeddings (%d items) from Drive E.", len(self.siglip2_gid_to_row))

            cache_folder = str(PROJECT_ROOT / ".cache" / "huggingface" / "hub")
            model_id = "google/siglip2-base-patch16-224"

            import torch
            if self.device == "auto":
                dev = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                dev = self.device

            logger.info("Loading SigLIP2 Text Encoder %s on %s...", model_id, dev)
            self.siglip2_processor = AutoProcessor.from_pretrained(
                model_id, cache_dir=cache_folder, local_files_only=True
            )
            self.siglip2_model = AutoModel.from_pretrained(
                model_id, cache_dir=cache_folder, local_files_only=True
            ).to(dev)
            self.siglip2_model.eval()

            cache_path = PROJECT_ROOT / ".cache" / "translations.json"
            if not cache_path.exists():
                cache_path = PROJECT_ROOT / "outputs" / "query_translations.json"
            self.translator = CachedQueryTranslator(cache_path=cache_path)
            self._siglip2_initialized = True
            logger.info("SigLIP2 Visual Rescorer ready.")
        except Exception as e:
            logger.error("Failed to initialize SigLIP2 Visual Rescorer: %s", e, exc_info=True)
            self._siglip2_initialized = False

    def encode_siglip2_text(self, text: str) -> Optional[np.ndarray]:
        """Encode query text into normalized SigLIP2 embedding vector (768D)."""
        if not self._siglip2_initialized or self.siglip2_model is None or self.siglip2_processor is None:
            return None
        try:
            import torch
            dev = next(self.siglip2_model.parameters()).device
            inp = self.siglip2_processor(text=[text], padding=True, return_tensors="pt").to(dev)
            with torch.no_grad():
                out = self.siglip2_model.get_text_features(**inp)
                if hasattr(out, "pooler_output") and out.pooler_output is not None:
                    t_feat = out.pooler_output
                elif hasattr(out, "last_hidden_state"):
                    t_feat = out.last_hidden_state[:, 0, :]
                elif isinstance(out, (tuple, list)):
                    t_feat = out[0]
                else:
                    t_feat = out
                t_feat = t_feat / t_feat.norm(dim=-1, keepdim=True)
            return t_feat.cpu().numpy()[0]
        except Exception as e:
            logger.error("SigLIP2 text encoding error: %s", e)
            return None

    def _normalize_scores(self, scores: List[float], mode: str = "min_max") -> List[float]:
        """
        Normalizes a list of query scores to [0, 1] using the specified normalization strategy.
        Supported modes: 'min_max', 'z_score', 'rank', 'raw'
        """
        n = len(scores)
        if n == 0:
            return []
        if n == 1:
            return [1.0]

        arr = np.array(scores, dtype=np.float64)

        if mode == "min_max":
            s_min = float(arr.min())
            s_max = float(arr.max())
            denom = s_max - s_min
            if denom < 1e-9:
                return [0.5] * n
            return [float((v - s_min) / denom) for v in arr]

        elif mode == "z_score":
            mu = float(arr.mean())
            std = float(arr.std())
            if std < 1e-9:
                return [0.5] * n
            z = (arr - mu) / std
            return [float(1.0 / (1.0 + math.exp(-zv))) for zv in z]

        elif mode == "rank":
            order = np.argsort(-arr)
            ranks = np.empty_like(order)
            ranks[order] = np.arange(n)
            return [float(1.0 - r / (n - 1)) for r in ranks]

        elif mode == "raw":
            return [float(max(0.0, min(1.0, v))) for v in arr]

        else:
            s_min = float(arr.min())
            s_max = float(arr.max())
            denom = s_max - s_min
            if denom < 1e-9:
                return [0.5] * n
            return [float((v - s_min) / denom) for v in arr]

    DEFAULT_ALPHA_MAP = {
        "visual": 0.0,
        "ocr": 0.3,
        "asr": 0.2,
        "mixed": 0.2,
        "object": 0.1,
    }

    def group_only(
        self,
        raw_candidates: List[Dict[str, Any]],
    ) -> List[CandidateSegment]:
        """
        Stage 1: Temporal Candidate Grouping without Cross-Encoder Reranking.
        """
        segments = self.grouper.group_candidates(raw_candidates)
        for seg in segments:
            self.aggregator.aggregate(seg)
            seg.final_score = seg.fusion_score

        segments.sort(key=lambda s: s.fusion_score, reverse=True)
        for idx, seg in enumerate(segments, start=1):
            seg.rank = idx
        return segments

    def rerank(
        self,
        query: str,
        raw_candidates: List[Dict[str, Any]],
        alpha: Optional[float] = None,
        modality: str = "visual",
        norm_mode: str = "rank",
        top_n_rerank: int = 10,
    ) -> List[CandidateSegment]:
        """
        Stage 2: KIS Reranker V1.2 Multimodal Two-Stage Pipeline.
        1. Temporal Candidate Grouping
        2. Filter Top-10 Best Segments by fusion score for fast inference
        3. Query Modality Routing:
           - VISUAL: SigLIP2 Feature Vector Rescoring (top2_mean aggregation, min_max norm, beta=0.4)
           - TEXT (ocr, asr, mixed, object): BGE Text Cross-Encoder Reranking
        4. Score Blending & Final Segment Re-ordering
        """
        if not self._initialized:
            self.initialize()

        # 1. Temporal Grouping
        all_segments = self.grouper.group_candidates(raw_candidates)
        if not all_segments:
            return []

        # Sort all segments by initial fusion score
        all_segments.sort(key=lambda s: s.fusion_score, reverse=True)

        # 2. Select Top-N Segments for Evidence Aggregation & Reranking
        top_segments = all_segments[:top_n_rerank]
        tail_segments = all_segments[top_n_rerank:]

        # 3. Evidence Aggregation on Top Segments
        for seg in top_segments:
            self.aggregator.aggregate(seg)

        mod_lower = modality.lower()

        # --- ROUTE 1: VISUAL ROUTE (SigLIP2 Feature Rescoring) ---
        if mod_lower == "visual":
            self.initialize_siglip2()
            if self._siglip2_initialized and self.siglip2_embeddings_tensor is not None:
                # Query Translation to English for SigLIP2
                query_en = query
                if self.translator:
                    res = self.translator.translate(query)
                    if res.usable:
                        query_en = res.text

                q_emb = self.encode_siglip2_text(query_en)
                if q_emb is not None:
                    seg_siglip2_scores = []
                    for s in top_segments:
                        member_sims = []
                        for m in s.member_frames:
                            gid = int(m.get("global_id", m.get("global_v2_id", 0)))
                            if self.siglip2_gid_to_row and gid in self.siglip2_gid_to_row:
                                row_idx = self.siglip2_gid_to_row[gid]
                                img_emb = self.siglip2_embeddings_tensor[row_idx].numpy()
                                sim = float(np.dot(q_emb, img_emb))
                            else:
                                sim = 0.0
                            m["siglip2_score"] = sim
                            member_sims.append(sim)

                        # Aggregate segment visual score via top2_mean
                        if not member_sims:
                            seg_v = 0.0
                        else:
                            s_sorted = sorted(member_sims, reverse=True)
                            if len(s_sorted) >= 2:
                                seg_v = float((s_sorted[0] + s_sorted[1]) / 2.0)
                            else:
                                seg_v = float(s_sorted[0])
                        seg_siglip2_scores.append(seg_v)
                        s.rerank_score = seg_v

                    # Normalize & Blend using min_max
                    beta = self.siglip2_beta if alpha is None or alpha == 0.0 else alpha
                    fusion_raw = [float(s.fusion_score) for s in top_segments]
                    fusion_norm = self._normalize_scores(fusion_raw, mode="min_max")
                    siglip2_norm = self._normalize_scores(seg_siglip2_scores, mode="min_max")

                    for idx, s in enumerate(top_segments):
                        s.final_score = float((1.0 - beta) * fusion_norm[idx] + beta * siglip2_norm[idx])

                    for s in tail_segments:
                        s.final_score = float(s.fusion_score)

                    top_segments.sort(key=lambda s: s.final_score, reverse=True)
                    final_ordered = top_segments + tail_segments
                    for idx, s in enumerate(final_ordered, start=1):
                        s.rank = idx
                    return final_ordered

            # Fallback if SigLIP2 unavailable
            for seg in all_segments:
                seg.final_score = seg.fusion_score
            for idx, seg in enumerate(all_segments, start=1):
                seg.rank = idx
            return all_segments

        # --- ROUTE 2: TEXT ROUTE (BGE Cross-Encoder Reranking) ---
        if alpha is None:
            alpha = self.DEFAULT_ALPHA_MAP.get(mod_lower, 0.0)

        if alpha == 0.0 or self.cross_encoder is None or not self._initialized or mod_lower not in ["ocr", "asr", "mixed", "object"]:
            for seg in all_segments:
                seg.final_score = seg.fusion_score
            for idx, seg in enumerate(all_segments, start=1):
                seg.rank = idx
            return all_segments

        pairs = []
        for seg in top_segments:
            context = seg.evidence.get("full_text_context", "").strip()
            if not context:
                context = f"Video: {seg.video_id} thời gian {seg.start_sec:.1f}s đến {seg.end_sec:.1f}s"
            pairs.append((query, context))

        raw_logits = []
        try:
            logits = self.cross_encoder.predict(pairs, batch_size=32, show_progress_bar=False)
            if isinstance(logits, float) or isinstance(logits, np.float32) or isinstance(logits, np.float64):
                raw_logits = [float(logits)]
            else:
                raw_logits = [float(l) for l in logits]
        except Exception as e:
            logger.error("Cross-encoder inference error: %s", e)
            raw_logits = [0.0] * len(top_segments)

        for seg, logit in zip(top_segments, raw_logits):
            seg.rerank_score = logit

        fusion_raw = [float(s.fusion_score) for s in top_segments]
        fusion_norm = self._normalize_scores(fusion_raw, mode=norm_mode)
        rerank_norm = self._normalize_scores(raw_logits, mode=norm_mode)

        for idx, seg in enumerate(top_segments):
            f_n = fusion_norm[idx]
            r_n = rerank_norm[idx]
            seg.final_score = float((1.0 - alpha) * f_n + alpha * r_n)

        for seg in tail_segments:
            seg.final_score = float(seg.fusion_score)

        top_segments.sort(key=lambda s: s.final_score, reverse=True)
        final_ordered = top_segments + tail_segments
        for idx, seg in enumerate(final_ordered, start=1):
            seg.rank = idx

        return final_ordered

    def segments_to_predictions(
        self,
        segments: List[CandidateSegment],
        modality: str = "visual",
        top_k: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        V1.2 Decoupled Two-Stage Frame Localization:
        Stage A (Precision): Best-member frame selection per segment (1 rank slot per segment)
        Stage B (Recall): Tail-fill with secondary member frames up to top_k.
        """
        primary_frames = []
        secondary_frames = []

        def _get_branch_score(m_dict: Dict[str, Any], key: str) -> float:
            if f"{key}_score" in m_dict and m_dict[f"{key}_score"] is not None:
                return float(m_dict[f"{key}_score"])
            if key in m_dict and m_dict[key] is not None:
                return float(m_dict[key])
            sc = m_dict.get("scores", {})
            if isinstance(sc, dict) and key in sc and sc[key] is not None:
                return float(sc[key])
            return 0.0

        for seg in segments:
            members = seg.member_frames
            if not members:
                best_m = dict(seg.representative_frame)
            elif modality == "visual" and any("siglip2_score" in m for m in members):
                best_m = dict(max(members, key=lambda m: (m.get("siglip2_score", 0.0), float(m.get("score", m.get("fused_score", 0.0))))))
            elif modality == "ocr" and any(_get_branch_score(m, "ocr") > 0 for m in members):
                best_m = dict(max(members, key=lambda m: (_get_branch_score(m, "ocr"), float(m.get("score", m.get("fused_score", 0.0))))))
            elif modality == "asr" and any(_get_branch_score(m, "asr") > 0 for m in members):
                best_m = dict(max(members, key=lambda m: (_get_branch_score(m, "asr"), float(m.get("score", m.get("fused_score", 0.0))))))
            elif modality == "object" and any(_get_branch_score(m, "object") > 0 for m in members):
                best_m = dict(max(members, key=lambda m: (_get_branch_score(m, "object"), float(m.get("score", m.get("fused_score", 0.0))))))
            else:
                best_m = dict(max(members, key=lambda m: float(m.get("score", m.get("fused_score", 0.0)))))

            best_m["segment_id"] = seg.segment_id
            best_m["segment_rank"] = seg.rank
            best_m["score"] = round(float(seg.final_score), 4)
            best_m["fusion_score"] = round(float(seg.fusion_score), 4)
            if seg.rerank_score is not None:
                best_m["rerank_score"] = round(float(seg.rerank_score), 4)
            best_m["segment_start_sec"] = round(seg.start_sec, 3)
            best_m["segment_end_sec"] = round(seg.end_sec, 3)
            best_m["segment_duration_sec"] = round(seg.duration_sec, 3)
            best_m["cluster_size"] = len(seg.member_frames)
            best_m["evidence"] = seg.evidence
            primary_frames.append(best_m)

            best_id = int(best_m.get("frame_idx", best_m.get("frame_id", -1)))
            other_m = [dict(m) for m in members if int(m.get("frame_idx", m.get("frame_id", -1))) != best_id]
            for om in other_m:
                om["segment_id"] = seg.segment_id
                om["segment_rank"] = seg.rank
                om["score"] = round(float(seg.final_score) * 0.9999, 4)
                om["fusion_score"] = round(float(om.get("score", om.get("fused_score", 0.0))), 4)
                om["evidence"] = seg.evidence
            other_m.sort(key=lambda m: float(m.get("fusion_score", 0.0)), reverse=True)
            secondary_frames.extend(other_m)

        final_cands = (primary_frames + secondary_frames)[:top_k]
        out = []
        for r_idx, f in enumerate(final_cands, start=1):
            fc = dict(f)
            fc["rank"] = r_idx
            out.append(fc)

        return out
