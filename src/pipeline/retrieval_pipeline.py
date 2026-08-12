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
from src.common.types import EvidenceResult, QASubmission, TRAKESubmission, TRAKEEventResult
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
        top_k_fusion: int = 50,
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

        # BatchRouter: predicts which batch(es) to search when target_prefix is absent
        self._batch_router = BatchRouter(
            known_batches=self._meta_store.known_batches if hasattr(meta_store, 'known_batches') else None,
            media_info_dir=kwargs.pop("media_info_dir", None) if kwargs else None,
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
        return cls(
            faiss_db, meta_store, encoder,
            text_retrievers=text_retrievers,
            vlm_client=vlm_client,
            keyframe_image_root=keyframe_image_root,
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
        target_prefix = query_dict.get("target_prefix") or ""

        logger.info(f"\n{'='*70}")
        logger.info(f"PROCESSING QUERY [id='{query_id}']")
        logger.info(f"Raw Input: '{raw_text}'")
        logger.info(f"Target Prefix: '{target_prefix}' ('' = global balanced search)")

        # Step 1: Query Parsing & Intent Extraction
        kis_query = self._parser.parse_kis(raw_text, top_k=self._top_k_ret)
        retrieval_text = self._parser.build_retrieval_text(kis_query)
        topic_res = self._parser.extract_topic(raw_text)
        query_topic = topic_res.topic if topic_res.confidence >= 0.3 else None

        logger.info(f"[Step 1/4 - Query Analysis]")
        logger.info(f"  • Scene: '{kis_query.parsed_scene}' | Objects: {kis_query.parsed_objects} | Colors: {kis_query.parsed_colors}")
        logger.info(f"  • OCR Hints: {kis_query.ocr_keywords} | Spatial: {kis_query.spatial_hints}")
        logger.info(f"  • Topic Intent: '{topic_res.topic}' (Conf: {topic_res.confidence:.2f}, Kw: {topic_res.matched_keywords})")
        logger.info(f"  • Translated CLIP Prompt: '{retrieval_text}'")

        # Step 2: Multimodal Candidate Retrieval
        logger.info(f"[Step 2/4 - Candidate Retrieval]")

        if target_prefix:
            # Known batch: restrict to that prefix (highest precision)
            vis_results = self._vis_ret.retrieve(
                retrieval_text, top_k=self._top_k_ret, target_prefix=target_prefix
            )
            logger.info(f"  • Mode: Prefix-Scoped ('{target_prefix}')")
        else:
            # Pillar 3: BatchRouter predicts likely batches WITHOUT knowing prefix
            predicted_batches = self._batch_router.predict(raw_text, top_n=3)
            n_all = len(self._batch_router._batches)
            logger.info(f"  • Mode: Global — BatchRouter predicted: {predicted_batches} ({len(predicted_batches)}/{n_all} batches)")

            if len(predicted_batches) < n_all:
                # High-confidence routing: search within predicted batches only
                all_vis: List = []
                slots = max(self._top_k_ret // max(len(predicted_batches), 1), 20)
                for batch in predicted_batches:
                    batch_vis = self._vis_ret.retrieve(
                        retrieval_text, top_k=slots, target_prefix=batch
                    )
                    all_vis.extend(batch_vis)
                all_vis.sort(key=lambda r: r.score, reverse=True)
                vis_results = all_vis[:self._top_k_ret]
            else:
                # Low-confidence: use Pillar 2 balanced retrieval (inverse-sqrt slots)
                vis_results = self._vis_ret.retrieve_balanced(
                    retrieval_text, top_k=self._top_k_ret, max_per_video=2
                )

        top1_score = vis_results[0].score if vis_results else 0.0
        logger.info(f"  • Visual Retrieval (CLIP): {len(vis_results)} candidates (Top 1 cosine: {top1_score:.4f})")

        all_lists   = []
        all_weights = []
        if vis_results:
            all_lists.append(vis_results)
            all_weights.append(self._visual_weight)

        for text_ret in self._text_rets:
            q_input = kis_query if getattr(text_ret, "name", "") == "ocr_inmemory" else raw_text
            txt = text_ret.retrieve(q_input, top_k=self._top_k_ret, target_prefix=target_prefix)
            if txt:
                all_lists.append(txt)
                all_weights.append(self._text_weight)
                logger.info(f"  • Text Retrieval ({getattr(text_ret, 'name', 'text')}): {len(txt)} candidates")

        # Guard: if no candidates at all, bail early
        if not all_lists:
            logger.warning(f"[KIS] No retrieval candidates for query_id='{query_id}' — check index coverage.")
            return None

        # Step 3: RRF Fusion & Topic Soft-Scoring
        logger.info(f"[Step 3/4 - Fusion & Topic Soft-Scoring]")
        fused = self._rrf.fuse(
            all_lists, all_weights,
            top_k=self._top_k_fus,
            query_topic=query_topic,
            topic_boost_weight=0.20,
        )

        if not fused or (fused[0].score < 0.005 and query_topic is not None):
            logger.info(f"  [FallbackTrigger] Low conf (score={fused[0].score if fused else 0:.4f}) → Unboosted Search")
            fused = self._rrf.fuse(all_lists, all_weights, top_k=self._top_k_fus, query_topic=None)
        else:
            logger.info(f"  • Topic Soft-Scoring Applied: '{query_topic}' (+20%)")

        if not fused:
            logger.warning(f"[KIS] No fused results for query_id='{query_id}'")
            return None

        # Step 4: Final Selection + Pillar 1 Score Normalization
        logger.info(f"[Step 4/4 - Frame Selection]")
        best_evidence = self._selector.select_best(fused, query_id=query_id)
        if best_evidence:
            # Pillar 1: Use real CLIP cosine similarity [0.0–1.0] as confidence
            # instead of the raw RRF score (which is always ~0.01–0.03)
            clip_conf = round(float(top1_score), 4)
            logger.info(
                f"  FINAL RESULT: Video='{best_evidence.video_id}' | "
                f"Frame={best_evidence.frame_idx} (PTS={best_evidence.pts_time:.2f}s) | "
                f"CLIP Confidence={clip_conf:.4f}"
            )
            best_evidence.confidence = clip_conf
        logger.info(f"{'='*70}\n")
        return best_evidence

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
        target_prefix = query_dict.get("target_prefix", "")
        qa_query      = self._parser.parse_qa(
            event_desc, question, answer_language=answer_lang, target_prefix=target_prefix
        )

        if self._vlm is None:
            logger.warning(
                f"[QA] id='{query_id}' — VLM not loaded. Falling back to KIS + heuristic answer."
            )
            evidence = self._run_kis({
                "text": f"{event_desc} {question}",
                "target_prefix": target_prefix,
            }, query_id)
            if evidence is None:
                logger.warning(f"[QA] id='{query_id}' — KIS fallback returned no results.")
                return None
            # Generate heuristic answer (no VLM call needed)
            from src.pipeline.qa_pipeline import QAPipeline
            tmp_qa = QAPipeline(
                visual_retriever=self._vis_ret,
                vlm_client=None,
                keyframe_image_root=str(self._kf_root),
                text_retrievers=self._text_rets,
                rrf=self._rrf,
            )
            dummy_cand = SearchResult(
                keyframe_id=f"{evidence.video_id}_n{evidence.n}",
                video_id=evidence.video_id,
                n=evidence.n,
                frame_idx=evidence.frame_idx,
                pts_time=evidence.pts_time,
                score=evidence.confidence,
                retriever_source="fallback",
                metadata=evidence.metadata,
            )
            heuristic_answer = tmp_qa._generate_fallback_answer(dummy_cand, qa_query)
            evidence.metadata["answer"] = heuristic_answer
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
        return EvidenceResult(
            video_id=qa_result.video_id,
            frame_idx=qa_result.frame_idx,
            n=0,
            pts_time=0.0,
            confidence=0.85,   # VLM-verified answer, not hardcoded 1.0
            explanation=f"QA answer: {qa_result.answer}",
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
                top_k_videos=10,           # Increased from 5 for better sport coverage
                top_k_frames_per_event=self._top_k_fus,
            )

        # Parse TRAKE query from dict
        from src.common.types import TRAKEQuery, EventStep
        events_raw = query_dict.get("events", query_dict.get("event_sequence", []))
        event_seq = [
            EventStep(
                event_id=e.get("id", i + 1),
                event_name=e.get("name", f"Event {i+1}"),
                description=e.get("description", ""),
                semantic_keyframe_hint=e.get("hint", e.get("semantic_keyframe_hint", "")),
            )
            for i, e in enumerate(events_raw)
        ]
        trake_query = TRAKEQuery(
            activity_name=query_dict.get("activity", query_dict.get("activity_name", "")),
            event_sequence=event_seq,
            sport_category=query_dict.get("sport_category", ""),
            top_k_videos=query_dict.get("top_k_videos", 5),
            video_id=query_dict.get("video_id", ""),
            target_prefix=query_dict.get("target_prefix", ""),
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

        return EvidenceResult(
            video_id=trake_result.video_id,
            frame_idx=first_event.frame_idx if first_event else 0,
            n=0,
            pts_time=first_event.pts_time if first_event else 0.0,
            confidence=trake_conf,
            explanation=f"TRAKE: {n_events} events aligned (conf={trake_conf:.2f})",
            metadata={
                "query_type": "trake",
                "trake_submission": trake_result,
            },
        )

