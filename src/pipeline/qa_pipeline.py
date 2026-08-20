"""
Q&A Pipeline — End-to-end handler for Query Dạng 2 (Visual Question Answering) v2.

Changes from v1:
- Single VLM call per frame (combined relevance + answer), was 2 calls.
- max_frames increased 10 → 20 for better coverage.
- Multi-frame answer voting: if ≥ 2 frames return the same answer → boost confidence.
- Retrieval uses build_qa_retrieval_text() which combines event_description + question
  keywords → better candidate recall.
- answer_type is passed to VLM for type-specific answer formatting.
- Early-stop only on high-confidence + found frames.
- QAAnswer.found=False frames are skipped instead of used as fallback.

Flow:
  1. Build rich retrieval text (event_description + question keywords)
  2. Retrieve top-K candidate keyframes (CLIP + optional Qdrant)
  3. For each candidate (up to max_frames):
     a. Run combined VLM call → QAAnswer (found, answer, confidence, observation)
     b. If found=True and confidence > threshold → add to answer pool
     c. Early-stop if best confidence >= high_conf_threshold
  4. Multi-frame voting: find the most common answer among found frames
  5. Return best (frame, answer) pair; fallback to top retrieval if nothing found
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.common.types import EvidenceResult, SearchResult, QAQuery, QASubmission
from src.reasoning.query_parser import QueryParser
from src.retrieval.visual_retriever import VisualRetriever
from src.retrieval.text_retriever import TextRetriever
from src.fusion.reciprocal_rank import ReciprocalRankFusion
from src.evidence.frame_selector import FrameSelector
from src.llm.qwen_client import QwenVLClient
from src.llm.response_parser import QAAnswer
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Thresholds
_MIN_CONFIDENCE      = 0.30   # Minimum to consider a found answer
_HIGH_CONF_THRESHOLD = 0.80   # Early-stop if this confident
_MAX_FRAMES          = 20     # Max candidates to run VLM on
_VOTE_THRESHOLD      = 2      # Min frames agreeing on answer → boost confidence


class QAPipeline:
    """
    Visual Question Answering pipeline (v2).

    Args:
        visual_retriever:     Loaded VisualRetriever (FAISS)
        text_retrievers:      List of TextRetriever (Qdrant caption/ocr/asr)
        rrf:                  ReciprocalRankFusion instance
        vlm_client:           Loaded QwenVLClient
        keyframe_image_root:  Root dir of keyframe images for VLM inference
        max_frames:           Max candidates to pass through VLM (default: 20)
        min_confidence:       Minimum confidence to count an answer (default: 0.30)
        high_conf_threshold:  Early-stop if answer confidence >= this (default: 0.80)
        top_k_retrieval:      Candidates from FAISS/Qdrant
        top_k_fusion:         Candidates kept after RRF fusion

    Usage:
        pipeline = QAPipeline(
            visual_retriever=vis_ret,
            vlm_client=qwen_client,
            keyframe_image_root="datasets/keyframes/keyframes",
        )
        result = pipeline.run(qa_query)
        # result.answer → "5"
        # result.frame_idx → 1500
    """

    def __init__(
        self,
        visual_retriever: VisualRetriever,
        vlm_client: QwenVLClient,
        keyframe_image_root: str,
        text_retrievers: Optional[List[TextRetriever]] = None,
        rrf: Optional[ReciprocalRankFusion] = None,
        max_frames: int = _MAX_FRAMES,
        min_confidence: float = _MIN_CONFIDENCE,
        high_conf_threshold: float = _HIGH_CONF_THRESHOLD,
        top_k_retrieval: int = 100,
        top_k_fusion: int = 30,
    ):
        self._vis_ret    = visual_retriever
        self._vlm        = vlm_client
        self._kf_root    = Path(keyframe_image_root)
        self._text_rets  = text_retrievers or []
        self._rrf        = rrf or ReciprocalRankFusion(k=60)
        self._selector   = FrameSelector()
        self._parser     = QueryParser()

        self.max_frames         = max_frames
        self.min_confidence     = min_confidence
        self.high_conf_threshold = high_conf_threshold
        self._top_k_ret         = top_k_retrieval
        self._top_k_fus         = top_k_fusion

    # ----------------------------------------------------------
    # Main Entry
    # ----------------------------------------------------------

    def run(self, qa_query: QAQuery, query_id: str = "") -> Optional[QASubmission]:
        """
        Execute the full Q&A pipeline.

        Args:
            qa_query:  Parsed QAQuery (from QueryParser.parse_qa)
            query_id:  ID string for logging

        Returns:
            QASubmission with video_id, frame_idx, and answer string
            OR None if no candidates found
        """
        logger.info(
            f"[QA] query_id='{query_id}' | type={qa_query.answer_type} "
            f"| Q: {qa_query.question[:80]}"
        )

        # Step 1: Retrieve candidate keyframes using enriched QA text
        candidates = self._retrieve_candidates(qa_query)
        self.last_candidates = candidates
        if not candidates:
            logger.warning(f"[QA] No candidates for query_id='{query_id}'")
            return None

        logger.info(f"[QA] {len(candidates)} candidates after fusion")

        # Step 2: Run VLM on top candidates (single combined call per frame)
        found_answers: List[Tuple[SearchResult, QAAnswer]] = []
        best_conf = 0.0
        best_frame: Optional[SearchResult] = None
        best_qa: Optional[QAAnswer] = None

        for rank, candidate in enumerate(candidates[:self.max_frames]):
            img_path = self._get_image_path(candidate)
            if not img_path or not img_path.exists():
                logger.debug(f"[QA] Image not found for {candidate.keyframe_id}, skipping")
                continue

            # Combined 1-call: relevance + answer together
            # Use pre-built VLM verification prompt if available (contains negations + constraints)
            vlm_prompt = getattr(qa_query, "vlm_verification_prompt", None)
            answer_subtype = getattr(qa_query, "answer_subtype", qa_query.answer_type)
            qa_ans = self._vlm.answer_question(
                image_path=str(img_path),
                event_description=vlm_prompt or qa_query.event_description,
                question=qa_query.question if not vlm_prompt else "",
                answer_language=qa_query.answer_language if qa_query.answer_language != "auto" else "vi",
                answer_type=answer_subtype,
            )

            logger.debug(
                f"[QA] [{rank+1}/{self.max_frames}] {candidate.keyframe_id} "
                f"found={qa_ans.found} conf={qa_ans.confidence:.2f} "
                f"answer='{qa_ans.answer[:50]}'"
            )

            if not qa_ans.found or qa_ans.confidence < self.min_confidence:
                continue

            found_answers.append((candidate, qa_ans))

            if qa_ans.confidence > best_conf:
                best_conf  = qa_ans.confidence
                best_frame = candidate
                best_qa    = qa_ans

            # Early stop if very confident
            if best_conf >= self.high_conf_threshold:
                logger.info(f"[QA] Early stop at rank {rank+1} (conf={best_conf:.2f})")
                break

        # Step 3: Multi-frame answer voting — boost confidence if multiple frames agree
        if len(found_answers) >= _VOTE_THRESHOLD:
            best_frame, best_qa = self._vote_best_answer(found_answers)
            logger.info(
                f"[QA] Vote result: answer='{best_qa.answer}' "
                f"conf={best_qa.confidence:.2f} from {len(found_answers)} frames"
            )

        # Step 4: Fallback to top retrieval result if VLM found nothing
        if best_frame is None or best_qa is None:
            best_frame  = candidates[0]
            answer_text = self._generate_fallback_answer(best_frame, qa_query)
            logger.warning(f"[QA] VLM found no answer — using fallback answer: '{answer_text}'")
        else:
            answer_text = best_qa.answer
            if not answer_text or answer_text.strip() == "":
                answer_text = self._generate_fallback_answer(best_frame, qa_query)

        logger.info(
            f"[QA] Result: {best_frame.video_id} frame_idx={best_frame.frame_idx} "
            f"answer='{answer_text[:60]}' conf={best_conf:.2f}"
        )

        return QASubmission(
            query_id=query_id,
            video_id=best_frame.video_id,
            frame_idx=best_frame.frame_idx,
            answer=answer_text,
        )

    # ----------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------

    def _retrieve_candidates(self, qa_query: QAQuery) -> List[SearchResult]:
        """
        Retrieve and fuse candidates using enriched QA retrieval text
        (event_description + question visual keywords).
        target_prefix is disabled — always search full database.
        """
        # Use enriched QA retrieval text instead of just event_description
        search_text = self._parser.build_qa_retrieval_text(qa_query)
        logger.debug(f"[QA] Retrieval text: {search_text[:120]}")

        vis_results = self._vis_ret.retrieve(search_text, top_k=self._top_k_ret)

        all_lists   = [vis_results]
        all_weights = [1.0]
        for text_ret in self._text_rets:
            # Also search with just event_description for text retrievers
            txt = text_ret.retrieve(qa_query.event_description, top_k=self._top_k_ret)
            if txt:
                all_lists.append(txt)
                all_weights.append(0.8)

        fused = self._rrf.fuse(all_lists, all_weights, top_k=self._top_k_fus)
        return fused

    def _get_image_path(self, result: SearchResult) -> Optional[Path]:
        """
        Reconstruct keyframe image path from SearchResult with robust path resolution.

        Checks:
          1. Direct image_path in result.metadata
          2. MetadataStore lookup via keyframe_id
          3. Multiple filename patterns (001.jpg, 1.jpg, 0001.jpg, 01.jpg)
          4. Multiple subfolder locations under keyframe_image_root
        """
        try:
            # 1. Metadata in SearchResult
            if result.metadata and result.metadata.get("image_path"):
                p = Path(result.metadata["image_path"])
                if p.exists():
                    return p

            # 2. MetadataStore lookup
            if hasattr(self._vis_ret, "_meta_store") and self._vis_ret._meta_store:
                meta = self._vis_ret._meta_store.get_by_keyframe_id(result.keyframe_id)
                if meta and meta.image_path:
                    p = Path(meta.image_path)
                    if p.exists():
                        return p

            # 3. Flexible filename & subfolder resolution
            batch_id = result.video_id.split("_")[0]   # e.g. "L21"
            n = result.n
            filenames = [f"{n:03d}.jpg", f"{n}.jpg", f"{n:04d}.jpg", f"{n:02d}.jpg"]

            folders = [
                self._kf_root / f"Keyframes_{batch_id}" / "keyframes" / result.video_id,
                self._kf_root / f"Keyframes_{batch_id}" / result.video_id,
                self._kf_root / batch_id / "keyframes" / result.video_id,
                self._kf_root / batch_id / result.video_id,
                self._kf_root / "keyframes" / result.video_id,
                self._kf_root / result.video_id,
            ]

            for folder in folders:
                for fname in filenames:
                    p = folder / fname
                    if p.exists():
                        return p

            # Fallback direct under root
            for fname in [f"{result.keyframe_id}.jpg", f"{result.video_id}_{n}.jpg"]:
                p = self._kf_root / fname
                if p.exists():
                    return p

            return folders[0] / filenames[0]
        except Exception as e:
            logger.debug(f"[QA] _get_image_path error for {result.keyframe_id}: {e}")
            return None

    def _generate_fallback_answer(self, candidate: SearchResult, qa_query: QAQuery) -> str:
        """
        Generate a heuristic fallback answer when VLM is unavailable or returns nothing.
        
        CRITICAL: Never return event_description or question text as the answer.
        Priority: OCR text from keyframe > extracted entities > type-appropriate placeholder.
        """
        # ── Step 1: Try to get OCR text from the matched keyframe ──
        ocr_text = ""
        
        # 1a. Check candidate metadata
        if candidate.metadata:
            ocr_text = candidate.metadata.get("ocr_text", "") or candidate.metadata.get("text_snippet", "")
        
        # 1b. Look up in InMemoryOCRRetriever if loaded
        if not ocr_text and hasattr(self, "_text_rets") and self._text_rets:
            for tr in self._text_rets:
                if getattr(tr, "name", "") == "ocr_inmemory" and hasattr(tr, "_records"):
                    rec = tr._records.get(candidate.keyframe_id)
                    if rec and rec.get("text"):
                        ocr_text = rec["text"]
                        break
        
        ocr_text = ocr_text.strip() if ocr_text else ""
        
        # ── Step 2: Determine answer type ──
        q_type = getattr(qa_query, "answer_subtype", qa_query.answer_type)
        question_lower = qa_query.question.lower()
        
        # ── Step 3: Generate type-appropriate answer ──
        
        if q_type in ("count", "count_people", "count_objects", "count_events"):
            # Try extracting digits from OCR first
            if ocr_text:
                digits = re.findall(r"\b\d+\b", ocr_text)
                valid_digits = [d for d in digits if 0 < int(d) < 100]
                if valid_digits:
                    return valid_digits[0]
            return "1"

        elif q_type in ("color", "color_clothing", "color_object", "color_background"):
            colors_map = {
                "đỏ": "đỏ", "xanh": "xanh", "vàng": "vàng", "trắng": "trắng",
                "đen": "đen", "tím": "tím", "hồng": "hồng", "cam": "cam",
                "nâu": "nâu", "xám": "xám",
            }
            # Search OCR text for colors first
            search_text = (ocr_text or "").lower()
            for vi_color in colors_map:
                if vi_color in search_text:
                    return vi_color
            return "trắng"

        elif q_type in ("yes_no", "yes_no_presence", "yes_no_action", "yes_no_attribute"):
            return "có"

        elif q_type in ("number_score", "number_time"):
            if ocr_text:
                nums = re.findall(r"\b\d+[:\-./]?\d*\b", ocr_text)
                if nums:
                    return nums[0]
            return "0"

        elif q_type in ("name", "name_person", "name_place", "name_thing"):
            # Priority: OCR text (likely contains the name shown on screen)
            if ocr_text:
                return self._extract_best_ocr_answer(ocr_text, qa_query.question)
            # Try to extract proper nouns from question as clues
            # (e.g. "xã này có tên là gì" → cannot answer without OCR/VLM)
            return "Không xác định"

        else:
            # description_general and any other type
            if ocr_text:
                return self._extract_best_ocr_answer(ocr_text, qa_query.question)
            # NEVER return event_description words — that's the question, not the answer
            return "Không xác định"

    def _extract_best_ocr_answer(self, ocr_text: str, question: str) -> str:
        """
        Extract the most relevant portion of OCR text as an answer to the question.
        
        Filters out common noise (timestamps, channel logos) and returns
        the most informative OCR segment.
        """
        if not ocr_text or not ocr_text.strip():
            return "Không xác định"
        
        # Split OCR text into segments (by common delimiters)
        segments = re.split(r'[|\n\r;]+', ocr_text)
        segments = [s.strip() for s in segments if s.strip() and len(s.strip()) > 1]
        
        if not segments:
            return ocr_text[:80].strip()
        
        # Filter out common noise patterns
        noise_patterns = [
            r'^(vtv|htv|thvl|antv|vov|vnews|sctv)\d*$',  # Channel logos
            r'^\d{1,2}:\d{2}(:\d{2})?$',                  # Timestamps like 14:30
            r'^\d{1,2}/\d{1,2}/\d{2,4}$',                 # Dates
            r'^(http|www)\.',                               # URLs
        ]
        clean_segments = []
        for seg in segments:
            is_noise = any(re.match(pat, seg.strip(), re.IGNORECASE) for pat in noise_patterns)
            if not is_noise and len(seg) > 2:
                clean_segments.append(seg)
        
        if not clean_segments:
            return segments[0][:80] if segments else "Không xác định"
        
        # For "tên là gì" / "là gì" questions → prefer the longest non-noise segment
        # (likely the title, name, or text shown on screen)
        question_lower = question.lower()
        if any(kw in question_lower for kw in ["tên là gì", "là gì", "tiêu đề", "câu thơ", "nội dung"]):
            # Return the longest clean segment (most likely to contain the answer)
            best = max(clean_segments, key=len)
            return best[:100].strip()
        
        # Default: return first clean segment
        return clean_segments[0][:80].strip()


    def _vote_best_answer(
        self,
        found_answers: List[Tuple[SearchResult, QAAnswer]],
    ) -> Tuple[SearchResult, QAAnswer]:
        """
        Multi-frame voting: find the most common answer and boost confidence.

        Algorithm:
          - Normalize answers (lowercase, strip)
          - Count occurrences of each unique answer
          - Pick the answer with the highest (count × avg_confidence) score
          - Use the frame with the highest individual confidence for that answer
        """
        # Normalize answers for comparison
        def normalize(s: str) -> str:
            return re.sub(r"\s+", " ", s.strip().lower())

        # Group by normalized answer
        answer_groups: Dict[str, List[Tuple[SearchResult, QAAnswer]]] = {}
        for frame, qa in found_answers:
            key = normalize(qa.answer)
            if key not in answer_groups:
                answer_groups[key] = []
            answer_groups[key].append((frame, qa))

        # Score each group: count × mean_confidence
        best_key = ""
        best_score = -1.0
        for norm_answer, group in answer_groups.items():
            count = len(group)
            avg_conf = sum(qa.confidence for _, qa in group) / count
            group_score = count * avg_conf
            if group_score > best_score:
                best_score = group_score
                best_key = norm_answer

        best_group = answer_groups[best_key]
        # Pick the frame with highest individual confidence
        best_frame, best_qa = max(best_group, key=lambda x: x[1].confidence)

        # Boost confidence if multiple frames agree
        vote_count = len(best_group)
        if vote_count >= 3:
            boosted_conf = min(1.0, best_qa.confidence + 0.10)
        elif vote_count == 2:
            boosted_conf = min(1.0, best_qa.confidence + 0.05)
        else:
            boosted_conf = best_qa.confidence

        # Create boosted QAAnswer
        boosted_qa = QAAnswer(
            answer=best_qa.answer,
            confidence=boosted_conf,
            found=True,
            observation=best_qa.observation,
            raw_output=best_qa.raw_output,
        )
        return best_frame, boosted_qa


