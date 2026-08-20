"""
Retrieval Pipeline — end-to-end query processing for all 3 AIC task types (v2).

Flow:
  KIS   → visual + text retrieval → RRF fusion → best frame
  Q&A   → enriched retrieval (event_desc + question) → VLM 1-call per frame → voting
  TRAKE → compact Phase1 query → per-event alignment with temporal window → VLM verify
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from src.common.enums import QueryType
from src.common.types import EvidenceResult, SearchResult, QASubmission, TRAKESubmission, TRAKEEventResult
from src.reasoning.query_classifier import QueryClassifier
from src.reasoning.query_parser import QueryParser
from src.reasoning.batch_router import BatchRouter
from src.retrieval.visual_retriever import VisualRetriever
from src.retrieval.text_retriever import TextRetriever
from src.fusion.reciprocal_rank import ReciprocalRankFusion
from src.evidence.frame_selector import FrameSelector
from src.database.faiss_db import FaissDB
from src.storage.metadata_store import MetadataStore
from src.embeddings.visual.clip import CLIPEncoder
from src.preprocessing.text_cleaner import clean_query
from src.reranking.ocr_reranker import OCRRelevanceReranker
from src.reranking.temporal_reranker import TemporalReranker
from src.reranking.clip_reranker import CLIPReranker
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RetrievalPipeline:
    """
    End-to-end retrieval pipeline routing queries to the appropriate sub-pipeline.

    KIS  → _run_kis()     (Sprint 2)
    Q&A  → _run_qa()      (Sprint 4, requires vlm_client + keyframe_image_root)
    TRAKE → _run_trake()  (Sprint 5 stub)

    Usage (minimal — KIS only):
        pipeline = RetrievalPipeline.from_index_dir(index_dir="indexes")
        result = pipeline.run({"type": "textual_kis", "text": "..."})

    Usage (full — KIS + Q&A):
        pipeline = RetrievalPipeline.from_index_dir(
            index_dir="indexes",
            keyframe_image_root="datasets/keyframes/keyframes",
            enable_vlm=True,
            vlm_load_in_4bit=True,
        )
    """

    def __init__(
        self,
        faiss_db: FaissDB,
        meta_store: MetadataStore,
        encoder: CLIPEncoder,
        text_retrievers: Optional[List[TextRetriever]] = None,
        vlm_client=None,                # QwenVLClient (optional)
        keyframe_image_root: str = "",  # Required for QA / TRAKE
        rrf_k: int = 60,
        visual_weight: float = 1.0,
        text_weight: float = 0.8,
        top_k_retrieval: int = 100,
        top_k_fusion: int = 100,
        media_info_dir: Optional[str] = None,
        **kwargs,
    ):
        self._faiss_db   = faiss_db
        self._meta_store = meta_store
        self._encoder    = encoder
        self._vlm        = vlm_client
        self._kf_root    = keyframe_image_root

        self._classifier  = QueryClassifier()
        self._parser      = QueryParser()
        self._vis_ret     = VisualRetriever(faiss_db, meta_store, encoder)
        self._text_rets   = text_retrievers or []
        self._rrf         = ReciprocalRankFusion(k=rrf_k)
        self._selector    = FrameSelector()

        # Advanced Rerankers
        self._ocr_reranker      = OCRRelevanceReranker()
        self._temporal_reranker = TemporalReranker()
        self._clip_reranker     = CLIPReranker(visual_weight=0.6, fusion_weight=0.4)

        # BatchRouter: predicts which batch(es) to search when target_prefix is absent
        self._batch_router = BatchRouter(
            known_batches=self._meta_store.known_batches if hasattr(meta_store, 'known_batches') else None,
            media_info_dir=media_info_dir,
        )

        self._visual_weight = visual_weight
        self._text_weight   = text_weight
        self._top_k_ret     = top_k_retrieval
        self._top_k_fus     = top_k_fusion

        # Lazy-init sub-pipelines (init once on first use)
        self._qa_pipeline    = None
        self._trake_pipeline = None

    # ----------------------------------------------------------
    # Factory
    # ----------------------------------------------------------

    @classmethod
    def from_index_dir(
        cls,
        index_dir: str = "indexes",
        clip_model: str = "ViT-B-32",
        clip_pretrained: str = "openai",
        device: Optional[str] = None,
        keyframe_image_root: str = "",
        enable_vlm: bool = False,
        vlm_model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        vlm_load_in_4bit: bool = True,
        qdrant_url: Optional[str] = None,
        **kwargs,
    ) -> "RetrievalPipeline":
        """
        Load all components from a pre-built index directory.

        Args:
            index_dir:             Dir with faiss_visual.index + keyframe_master.parquet
            keyframe_image_root:   Root of keyframe images (for QA/TRAKE VLM inference)
            enable_vlm:            Load Qwen2.5-VL for Q&A support
            vlm_load_in_4bit:      Use 4-bit quantization for VLM
            qdrant_url:            If set, connect Qdrant and add text retrievers
        """
        idx = Path(index_dir)

        # FAISS
        faiss_db = FaissDB()
        faiss_db.load(str(idx / "faiss_visual.index"))

        # Metadata
        meta_store = MetadataStore(
            map_keyframes_root="",
            keyframes_image_root="",
        ).load(str(idx / "keyframe_master.parquet"))

        # CLIP encoder
        encoder = CLIPEncoder(
            model_name=clip_model,
            pretrained=clip_pretrained,
            device=device,
        ).load()

        # Optional: Text retrievers (Qdrant or In-Memory OCR)
        text_retrievers = []

        ocr_dir = kwargs.pop("ocr_dir", None)
        if ocr_dir and Path(ocr_dir).exists():
            try:
                from src.retrieval.ocr_retriever import InMemoryOCRRetriever
                ocr_ret = InMemoryOCRRetriever(ocr_dir=ocr_dir, meta_store=meta_store)
                ocr_ret.load()
                if ocr_ret.is_configured:
                    text_retrievers.append(ocr_ret)
                    logger.info(f"InMemoryOCRRetriever loaded from: {ocr_dir}")
            except Exception as e:
                logger.warning(f"Failed to load InMemoryOCRRetriever: {e}")

        if qdrant_url:
            try:
                from src.database.qdrant_db import QdrantDB
                from src.embeddings.text.bge import BGEEncoder

                bge = BGEEncoder()
                bge.load()
                qdrant_db = QdrantDB(url=qdrant_url)
                qdrant_db.connect()

                for modality in ["caption", "ocr", "asr"]:
                    text_retrievers.append(TextRetriever(
                        modality=modality,
                        qdrant_db=qdrant_db,
                        bge_encoder=bge,
                    ))
                logger.info(f"Qdrant text retrievers ready (3 modalities)")
            except Exception as e:
                logger.warning(f"Qdrant setup failed: {e} — visual-only mode")

        # Optional: VLM client
        vlm_client = None
        if enable_vlm:
            try:
                from src.llm.qwen_client import QwenVLClient
                vlm_client = QwenVLClient(
                    model_name=vlm_model,
                    device=device or "cuda",
                    load_in_4bit=vlm_load_in_4bit,
                )
                vlm_client.load()
                if vlm_client.is_loaded:
                    logger.info("Qwen2.5-VL client ready")
                else:
                    logger.warning("VLM load() completed but model is None — disabling VLM")
                    vlm_client = None
            except Exception as e:
                logger.warning(
                    f"VLM loading failed: {e} — disabling VLM, QA will use fallback heuristic"
                )
                vlm_client = None   # ← critical: ensure downstream code sees None, not broken object

        logger.info(
            f"RetrievalPipeline ready — "
            f"FAISS: {faiss_db.total_vectors:,} vectors | "
            f"text_rets: {len(text_retrievers)} | "
            f"VLM: {'yes' if vlm_client else 'no'}"
        )
        media_info_dir = kwargs.pop("media_info_dir", None)
        return cls(
            faiss_db, meta_store, encoder,
            text_retrievers=text_retrievers,
            vlm_client=vlm_client,
            keyframe_image_root=keyframe_image_root,
            media_info_dir=media_info_dir,
            **kwargs,
        )

    # ----------------------------------------------------------
    # Main Entry Point
    # ----------------------------------------------------------

    def run(
        self,
        query_dict: Dict[str, Any],
        query_id: str = "",
    ) -> Optional[EvidenceResult]:
        """Process one query dict → EvidenceResult (KIS / QA / TRAKE)."""
        qtype = self._classifier.classify(query_dict)
        logger.info(f"[Pipeline] query_id='{query_id}' type={qtype.value}")

        if qtype == QueryType.TEXTUAL_KIS:
            return self._run_kis(query_dict, query_id)

        elif qtype == QueryType.QA:
            return self._run_qa(query_dict, query_id)

        elif qtype == QueryType.TRAKE:
            return self._run_trake(query_dict, query_id)

        return None

    def run_batch(
        self,
        query_dicts: List[Dict[str, Any]],
    ) -> List[Optional[EvidenceResult]]:
        """Process a list of queries in order."""
        results = []
        for i, qdict in enumerate(query_dicts):
            qid = qdict.get("query_id", str(i))
            results.append(self.run(qdict, query_id=qid))
        return results

    # ----------------------------------------------------------
    # KIS Sub-Pipeline
    # ----------------------------------------------------------

    def _run_kis(
        self,
        query_dict: Dict[str, Any],
        query_id: str,
    ) -> Optional[EvidenceResult]:
        """Text → CLIP → FAISS → (Qdrant / InMemory OCR) → Topic-SoftScored RRF → best frame.

        Pillar 2+3: When target_prefix is absent (real competition), BatchRouter predicts
        likely batch(es), then balanced retrieval prevents L25/L26 from dominating.
        """
        raw_text = query_dict.get("text") or query_dict.get("description", "")
        # Apply preprocessing text cleaning
        raw_text = clean_query(raw_text)

        logger.info(f"\n{'='*75}")
        logger.info(f"📌 XỬ LÝ QUERY [id='{query_id}']")
        logger.info(f"   ► Câu gốc (Vietnamese) : '{raw_text}'")
        logger.info(f"   ► Chế độ tìm kiếm      : Global (Toàn bộ CSDL — 177K keyframes)")
        logger.info(f"{'-'*75}")

        # Step 1: Deep Query Analysis & Vi -> En translation
        kis_query = self._parser.parse_kis(raw_text, top_k=self._top_k_ret)
        retrieval_text = self._parser.build_retrieval_text(kis_query)
        topic_res = self._parser.extract_topic(raw_text)
        query_topic = topic_res.topic if topic_res.confidence >= 0.3 else None

        # Dynamic weights from IntentScorer
        dyn_visual_w = kis_query.retrieval_weights.get("visual", self._visual_weight)
        dyn_text_w   = kis_query.retrieval_weights.get("ocr",    self._text_weight)

        logger.info(f"[BƯỚC 1/5 - PHÂN TÍCH QUERY & CHUYỂN ĐỔI NGÔN NGỮ (Vi ➔ En)]")
        logger.info(f"  • Tiếng Việt (Gốc)     : '{raw_text}'")
        logger.info(f"  • Tiếng Anh (CLIP Prompt): '{retrieval_text}'")
        logger.info(f"  • Bối cảnh (Scene)    : '{kis_query.parsed_scene}' | Vật thể: {kis_query.parsed_objects} | Màu sắc: {kis_query.parsed_colors}")
        logger.info(f"  • Con người / Vai trò : {kis_query.persons} | Số lượng: {kis_query.quantities}")
        logger.info(f"  • Trang phục / Chi tiết: {kis_query.clothing_details} | Ánh sáng: '{kis_query.lighting}'")
        logger.info(f"  • OCR / Từ khóa văn bản: {kis_query.ocr_keywords}")
        logger.info(f"  • Phủ định (Negated)   : {kis_query.negated_attributes} | Bắt buộc (Must): {kis_query.must_have}")
        logger.info(f"  • Trọng số Retriever   : Visual={dyn_visual_w:.2f}, OCR/Text={dyn_text_w:.2f}")
        logger.info(f"  • Phân loại Chủ đề     : '{topic_res.topic}' (Độ tin cậy: {topic_res.confidence:.2f})")

        # Step 2: Multi-Query Expansion Prompts
        extra_prompts = self._build_expansion_prompts(kis_query, raw_text)
        logger.info(f"\n[BƯỚC 2/5 - MỞ RỘNG QUERY (MULTI-QUERY EXPANSION)]")
        logger.info(f"  • Prompt Chính (Primary) : '{retrieval_text}'")
        for i, ep in enumerate(extra_prompts):
            logger.info(f"  • Prompt Mở Rộng #{i+1}     : '{ep}'")

        # Step 3: Multimodal Candidate Retrieval — balanced global search
        logger.info(f"\n[BƯỚC 3/5 - TRUY XUẤT VECTOR BAN ĐẦU (FAISS HNSW efSearch=256)]")
        vis_results = self._vis_ret.retrieve_balanced(
            retrieval_text, top_k=self._top_k_ret, max_per_video=5
        )

        top1_score = vis_results[0].score if vis_results else 0.0
        logger.info(f"  • Truy xuất Thị giác (CLIP Primary) : {len(vis_results)} ứng viên (Top-1 Cosine: {top1_score:.4f})")

        all_lists   = []
        all_weights = []

        if vis_results:
            all_lists.append(vis_results)
            all_weights.append(dyn_visual_w)

        # Execute expansion queries as SEPARATE fusion streams (not concatenated into vis_results)
        for i, ep in enumerate(extra_prompts):
            try:
                ep_results = self._vis_ret.retrieve_balanced(
                    ep, top_k=max(30, self._top_k_ret // 3), max_per_video=3
                )
                if ep_results:
                    all_lists.append(ep_results)
                    all_weights.append(dyn_visual_w * 0.75)  # Slightly lower weight for secondary perspective
                    logger.info(f"  • Truy xuất bổ sung #{i+1}           : {len(ep_results)} ứng viên (Top-1 Cosine: {ep_results[0].score:.4f})")
            except Exception as e:
                logger.debug(f"  • Query mở rộng #{i+1} không thành công: {e}")

        # Only invoke OCR text retrievers if OCR keywords are explicitly present
        if kis_query.ocr_keywords:
            for text_ret in self._text_rets:
                if getattr(text_ret, "name", "") == "ocr_inmemory":
                    txt = text_ret.retrieve(kis_query, top_k=self._top_k_ret)
                else:
                    txt = text_ret.retrieve(raw_text, top_k=self._top_k_ret)
                if txt:
                    all_lists.append(txt)
                    all_weights.append(dyn_text_w)
                    logger.info(f"  • Truy xuất Văn bản ({getattr(text_ret, 'name', 'text')}): {len(txt)} ứng viên")
        else:
            logger.info("  • Bỏ qua Truy xuất OCR (Không phát hiện từ khóa chữ viết trong query)")

        if not all_lists:
            logger.warning(f"[KIS] Không tìm thấy ứng viên nào cho query_id='{query_id}'")
            return None

        # Step 4: RRF Fusion & Multi-Stage Reranking
        logger.info(f"\n[BƯỚC 4/5 - HÒA TRỘN & XẾP HẠNG ĐA TẦNG (FUSION & RERANKING)]")
        fused = self._rrf.fuse(
            all_lists, all_weights,
            top_k=self._top_k_fus,
            max_per_video=5,
            query_topic=query_topic,
            topic_boost_weight=0.15,
        )

        if not fused or (fused[0].score < 0.005 and query_topic is not None):
            logger.info(f"  • Fallback: Tắt Topic Boost do độ tin cậy thấp")
            fused = self._rrf.fuse(all_lists, all_weights, top_k=self._top_k_fus, query_topic=None)
        elif query_topic:
            logger.info(f"  • Đã áp dụng Topic Soft-Scoring: '{query_topic}' (+15%)")

        if not fused:
            logger.warning(f"[KIS] Không có kết quả sau Fusion cho query_id='{query_id}'")
            return None

        # Multi-Stage Reranking
        fused = self._clip_reranker.rerank(kis_query, fused, top_k=self._top_k_fus)
        logger.info(f"  • Tầng 1 - CLIP Visual Reranker     : Hoàn tất (Top-1 score: {fused[0].score:.4f})")

        fused = self._ocr_reranker.rerank(kis_query, fused, top_k=self._top_k_fus)
        logger.info(f"  • Tầng 2 - OCR Keyword Reranker      : Hoàn tất")

        fused = self._temporal_reranker.rerank(kis_query, fused, top_k=self._top_k_fus)
        logger.info(f"  • Tầng 3 - Temporal Continuity Rerank: Hoàn tất (Đã tối ưu cụm thời gian)")

        # Step 5: Final Selection & Top Output
        logger.info(f"\n[BƯỚC 5/5 - LỰA CHỌN KEYFRAME & XUẤT TOP 100 ĐÁP ÁN]")
        best_evidence = self._selector.select_best(fused, query_id=query_id)
        if best_evidence:
            clip_conf = round(float(top1_score), 4)

            # Log top 20 candidates table (full 100 saved in top_results)
            top20 = fused[:20]
            logger.info(f"\n  📋 DANH SÁCH TOP 20 / 100 ĐÁP ÁN ĐIỂM CAO NHẤT (QUERY '{query_id}'):")
            logger.info(f"  {'Hạng':<5} | {'Video ID':<12} | {'Frame Index':<12} | {'Keyframe (n)':<14} | {'PTS Time':<10} | {'Score':<8}")
            logger.info(f"  {'-'*75}")
            for rank, r in enumerate(top20, 1):
                logger.info(
                    f"  #{rank:<4} | {r.video_id:<12} | {r.frame_idx:<12} | n={r.n:<12} | {r.pts_time:>6.2f}s    | {r.score:.4f}"
                )
            logger.info(f"  {'-'*75}\n")

            # Health warnings
            if clip_conf < 0.20:
                logger.warning(f"  ⚠ CẢNH BÁO: Độ tin cậy CLIP thấp ({clip_conf:.4f}) — Kết quả có thể chưa chính xác!")
            if len(fused) >= 2:
                score_gap = fused[0].score - fused[1].score
                if score_gap < 0.001:
                    logger.warning(f"  ⚠ CẢNH BÁO: Khoảng cách điểm giữa Top 1 & Top 2 rất nhỏ ({score_gap:.6f})")
            top3_vids = list(dict.fromkeys(r.video_id for r in fused[:3]))
            if len(top3_vids) >= 3:
                logger.info(f"  ℹ GHI CHÚ: Top 3 ứng viên thuộc 3 video khác nhau ({top3_vids})")

            logger.info(
                f"  🏆 KẾT QUẢ TOP 1 ĐƯỢC CHỌN: Video='{best_evidence.video_id}' | "
                f"Frame={best_evidence.frame_idx} (PTS={best_evidence.pts_time:.2f}s) | "
                f"Độ tin cậy CLIP={clip_conf:.4f}"
            )
            best_evidence.confidence = clip_conf
        logger.info(f"{'='*75}\n")
        return best_evidence

    # ----------------------------------------------------------
    # Multi-Query Expansion
    # ----------------------------------------------------------

    def _build_expansion_prompts(
        self,
        kis_query,
        raw_text: str,
    ) -> list:
        """
        Generate 2-3 alternative CLIP prompts for the same query to improve recall.

        Different prompt perspectives capture different visual aspects:
          - Object-focused: emphasizes what things are visible
          - Scene-focused: emphasizes the setting/environment
          - Action-focused: emphasizes what's happening
        """
        prompts = []

        # Variant 1: Object-focused (if we have objects)
        objs = kis_query.parsed_objects
        colors = kis_query.parsed_colors
        if objs:
            obj_str = ", ".join(objs[:4])
            color_part = ""
            if colors:
                from src.preprocessing.entity_extractor import _COLOR_SIMPLE_VI
                en_colors = [_COLOR_SIMPLE_VI.get(c, c) for c in colors[:2]]
                color_part = f", {' and '.join(en_colors)}"
            prompts.append(f"A photo showing {obj_str}{color_part}")

        # Variant 2: Scene + subject description (if we have scene info)
        scene = kis_query.parsed_scene
        if scene:
            scene_map = {
                "news studio": "TV news broadcast studio",
                "press conference": "press conference with speakers",
                "stadium": "sports stadium with players",
                "outdoor": "outdoor scene",
                "indoor": "indoor scene",
                "street": "urban street scene",
                "kitchen": "kitchen cooking scene",
                "ceremony": "formal ceremony on stage",
            }
            scene_en = scene_map.get(scene, scene)
            persons = kis_query.persons
            if persons:
                role = persons[0].get("role_en", "person")
                prompts.append(f"A {role} in a {scene_en}")
            else:
                prompts.append(f"A {scene_en}")

        # Variant 3: Action-focused (if we have actions)
        actions = kis_query.actions
        if actions:
            act_str = " and ".join(a.get("en", "") for a in actions[:2] if a.get("en"))
            if act_str:
                prompts.append(f"Someone {act_str}")

        # Limit to 2 expansion prompts to control latency
        return prompts[:2]

    # ----------------------------------------------------------
    # Q&A Sub-Pipeline
    # ----------------------------------------------------------

    def _run_qa(
        self,
        query_dict: Dict[str, Any],
        query_id: str,
    ) -> Optional[EvidenceResult]:
        """
        Q&A: retrieve candidates, then run Qwen2.5-VL to extract answer.
        Generates a heuristic fallback answer if VLM is not available.
        """
        event_desc    = query_dict.get("description", "")
        question      = query_dict.get("question", "")
        answer_lang   = query_dict.get("answer_language", "auto")

        # Fallback if description is empty or missing
        if not event_desc:
            full_txt = query_dict.get("text", question)
            from src.common.query_loader import split_qa_text
            event_desc, question = split_qa_text(full_txt)

        # target_prefix intentionally disabled — full database search
        qa_query      = self._parser.parse_qa(
            event_desc, question, answer_language=answer_lang, target_prefix=""
        )

        logger.debug(
            f"[QA] id='{query_id}' answer_type='{qa_query.answer_type}' "
            f"subtype='{qa_query.answer_subtype}' format='{qa_query.expected_answer_format}' | "
            f"negated={qa_query.negated_attributes} | weights={qa_query.retrieval_weights}"
        )

        if self._vlm is None:
            logger.warning(
                f"[QA] id='{query_id}' — VLM not loaded. Falling back to KIS + OCR heuristic."
            )
            # Step A: Run KIS retrieval using combined description + question
            evidence = self._run_kis({
                "text": f"{event_desc} {question}",
            }, query_id)
            if evidence is None:
                logger.warning(f"[QA] id='{query_id}' — KIS fallback returned no results.")
                return None

            # Step B: Try to find OCR data for the best keyframe
            ocr_text_for_answer = ""
            best_keyframe_id = f"{evidence.video_id}_n{evidence.n}"

            # B1. Search OCR retrievers for the specific keyframe
            for tr in self._text_rets:
                if getattr(tr, "name", "") == "ocr_inmemory" and hasattr(tr, "_records"):
                    rec = tr._records.get(best_keyframe_id)
                    if rec and rec.get("text"):
                        ocr_text_for_answer = rec["text"]
                        logger.debug(f"[QA] Found OCR for {best_keyframe_id}: '{ocr_text_for_answer[:60]}'")
                        break

            # B2. If no OCR for this specific keyframe, search OCR for the question text
            if not ocr_text_for_answer:
                for tr in self._text_rets:
                    if getattr(tr, "name", "") == "ocr_inmemory" and getattr(tr, "is_configured", False):
                        ocr_results = tr.retrieve(question, top_k=5)
                        if ocr_results:
                            ocr_text_for_answer = ocr_results[0].metadata.get("ocr_text", "")
                            # Update evidence to use the OCR-matched keyframe instead
                            evidence.video_id = ocr_results[0].video_id
                            evidence.frame_idx = ocr_results[0].frame_idx
                            evidence.n = ocr_results[0].n
                            evidence.pts_time = ocr_results[0].pts_time
                            logger.debug(
                                f"[QA] OCR search found better match: {ocr_results[0].keyframe_id} "
                                f"OCR='{ocr_text_for_answer[:60]}'"
                            )
                        break

            # Step C: Generate heuristic answer using OCR data
            from src.pipeline.qa_pipeline import QAPipeline
            tmp_qa = QAPipeline(
                visual_retriever=self._vis_ret,
                vlm_client=None,
                keyframe_image_root=str(self._kf_root),
                text_retrievers=self._text_rets,
                rrf=self._rrf,
            )
            dummy_cand = SearchResult(
                keyframe_id=best_keyframe_id,
                video_id=evidence.video_id,
                n=evidence.n,
                frame_idx=evidence.frame_idx,
                pts_time=evidence.pts_time,
                score=evidence.confidence,
                retriever_source="fallback",
                metadata={"ocr_text": ocr_text_for_answer} if ocr_text_for_answer else {},
            )
            heuristic_answer = tmp_qa._generate_fallback_answer(dummy_cand, qa_query)
            evidence.metadata["answer"] = heuristic_answer
            evidence.metadata["query_type"] = "qa"
            logger.info(
                f"[QA] id='{query_id}' Fallback: "
                f"video={evidence.video_id} frame={evidence.frame_idx} "
                f"answer='{heuristic_answer}'"
            )
            return evidence

        # Lazy-init QA pipeline
        if self._qa_pipeline is None:
            from src.pipeline.qa_pipeline import QAPipeline
            self._qa_pipeline = QAPipeline(
                visual_retriever=self._vis_ret,
                vlm_client=self._vlm,
                keyframe_image_root=self._kf_root,
                text_retrievers=self._text_rets,
                rrf=self._rrf,
                top_k_retrieval=self._top_k_ret,
                top_k_fusion=self._top_k_fus,
                max_frames=20,
                min_confidence=0.30,
                high_conf_threshold=0.80,
            )

        logger.debug(
            f"[QA] answer_type='{qa_query.answer_type}' | "
            f"event='{event_desc[:60]}' | Q='{question[:60]}'"
        )

        # Run QA pipeline
        qa_result: Optional[QASubmission] = self._qa_pipeline.run(qa_query, query_id=query_id)
        if qa_result is None:
            return None

        # Wrap in EvidenceResult so the run_queries.py runner can handle uniformly
        # Pillar 1: QA confidence = 0.85 (VLM-verified answer, higher than pure CLIP retrieval)
        top_cands = getattr(self._qa_pipeline, "last_candidates", [])
        return EvidenceResult(
            video_id=qa_result.video_id,
            frame_idx=qa_result.frame_idx,
            n=0,
            pts_time=0.0,
            confidence=0.85,   # VLM-verified answer, not hardcoded 1.0
            explanation=f"QA answer: {qa_result.answer}",
            top_results=top_cands[:self._top_k_ret],
            metadata={"answer": qa_result.answer, "query_type": "qa"},
        )

    # ----------------------------------------------------------
    # TRAKE Sub-Pipeline
    # ----------------------------------------------------------

    def _run_trake(
        self,
        query_dict: Dict[str, Any],
        query_id: str,
    ) -> Optional[EvidenceResult]:
        """
        TRAKE: 3-phase pipeline (video retrieval → per-event alignment → best video).
        Falls back to KIS if required components are missing.
        """
        # Lazy-init TRAKE pipeline
        if self._trake_pipeline is None:
            from src.pipeline.trake_pipeline import TRAKEPipeline
            self._trake_pipeline = TRAKEPipeline(
                visual_retriever=self._vis_ret,
                clip_encoder=self._encoder,
                text_retrievers=self._text_rets,
                rrf=self._rrf,
                vlm_client=self._vlm,
                enable_vlm_verify=(self._vlm is not None),
                top_k_videos=self._top_k_ret,
                top_k_frames_per_event=self._top_k_ret,
            )

        # Parse TRAKE query from dict
        from src.common.types import TRAKEQuery, EventStep
        events_raw = query_dict.get("events", query_dict.get("event_sequence", []))
        event_seq = []
        for i, e in enumerate(events_raw):
            ev_id = e.get("event_id", e.get("id", i + 1))
            ev_name = e.get("name", e.get("event_name", e.get("description", f"Event {i+1}")))
            ev_desc = e.get("description", e.get("name", ev_name))
            ev_hint = e.get("hint", e.get("semantic_keyframe_hint", ""))
            event_seq.append(EventStep(
                event_id=ev_id,
                event_name=ev_name,
                description=ev_desc,
                semantic_keyframe_hint=ev_hint,
            ))

        trake_query = TRAKEQuery(
            activity_name=query_dict.get("activity", query_dict.get("activity_name", "")),
            event_sequence=event_seq,
            sport_category=query_dict.get("sport_category", ""),
            top_k_videos=query_dict.get("top_k_videos", self._top_k_ret),
            video_id=query_dict.get("video_id", ""),
            target_prefix="",  # disabled — full database search
        )

        trake_result: Optional[TRAKESubmission] = self._trake_pipeline.run(
            trake_query, query_id=query_id
        )
        if trake_result is None:
            return None

        # Return first event's frame as the EvidenceResult (for uniform handling)
        first_event = trake_result.events[0] if trake_result.events else None

        # Pillar 1: TRAKE confidence reflects how many events were successfully aligned
        n_events = len(trake_result.events)
        trake_conf = round(min(0.65 + n_events * 0.10, 0.95), 2)  # 0.75 (1ev), 0.85 (2ev), 0.95 (3+ev)

        top_cands = getattr(self._trake_pipeline, "last_phase1_results", [])
        return EvidenceResult(
            video_id=trake_result.video_id,
            frame_idx=first_event.frame_idx if first_event else 0,
            n=0,
            pts_time=first_event.pts_time if first_event else 0.0,
            confidence=trake_conf,
            explanation=f"TRAKE: {n_events} events aligned (conf={trake_conf:.2f})",
            top_results=top_cands[:self._top_k_ret],
            metadata={
                "query_type": "trake",
                "trake_submission": trake_result,
            },
        )

