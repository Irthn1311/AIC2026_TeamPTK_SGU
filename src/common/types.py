"""
Core Data Contracts for AIC Video Retrieval System.
Defines standardized Dataclasses passed across Pipeline layers.
All types align with the official AIC-HCMC Kaggle dataset structure.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


# ============================================================
# Dataset-Level Types
# ============================================================

@dataclass
class MediaInfo:
    """
    Video-level YouTube metadata parsed from media-info JSON file.
    Source: media-info-aic25-b1/media-info/{L}_{V}.json
    """
    video_id: str                   # e.g. "L21_V001"
    author: str = ""                # Channel / Creator name
    channel_id: str = ""
    channel_url: str = ""
    description: str = ""           # Full video description
    keywords: List[str] = field(default_factory=list) # Video tags / keywords
    topic_category: str = ""        # Classified topic (e.g. "tin_tuc", "the_thao")
    topic_confidence: float = 0.0

    def get_combined_text(self) -> str:
        """Combine author, keywords, and description for topic classification & text search."""
        kw_str = ", ".join(self.keywords) if isinstance(self.keywords, list) else str(self.keywords)
        return f"{self.author} | Keywords: {kw_str} | {self.description}".strip()


@dataclass
class KeyframeMeta:
    """
    Single keyframe record parsed from map-keyframes CSV.
    Source: map-keyframes-aic25-b1/map-keyframes/L{XX}_{V}.csv
    Columns: n, pts_time, fps, frame_idx
    """
    video_id: str           # e.g. "L21_V001"
    batch_id: str           # e.g. "L21"
    n: int                  # 1-based keyframe sequence number (also filename: {n}.jpg)
    frame_idx: int          # Actual frame index in video → value submitted to BTC
    pts_time: float         # Timestamp in seconds
    fps: float              # Video FPS (typically 30.0)
    image_path: str         # Absolute path to keyframe .jpg
    topic_category: str = "" # Associated video topic category

    @property
    def keyframe_id(self) -> str:
        """Unique identifier: L21_V001_n5"""
        return f"{self.video_id}_n{self.n}"

    @property
    def faiss_id(self) -> int:
        """Monotonic integer for FAISS mapping (set externally)."""
        return -1  # Overridden during index build


@dataclass
class KeyframeItem:
    """Keyframe with all extracted features (used in pipeline)."""
    keyframe_id: str             # e.g. "L21_V001_n5"
    video_id: str                # e.g. "L21_V001"
    n: int                       # 1-based keyframe number
    frame_idx: int               # Frame index → BTC submission value
    pts_time: float              # Timestamp in seconds
    image_path: str              # Path to .jpg
    ocr_text: Optional[str] = None
    asr_text: Optional[str] = None
    caption: Optional[str] = None
    detected_objects: List[Dict[str, Any]] = field(default_factory=list)


# ============================================================
# Query Types
# ============================================================

@dataclass
class TextualKISQuery:
    """Dạng 1: Textual Known-Item Search query."""
    raw_text: str
    # ── Legacy fields (backward-compatible) ──
    parsed_objects: List[str] = field(default_factory=list)
    parsed_scene: str = ""
    parsed_colors: List[str] = field(default_factory=list)
    ocr_keywords: List[str] = field(default_factory=list)
    spatial_hints: List[str] = field(default_factory=list)
    top_k: int = 10
    # ── Deep analysis fields (new) ──
    persons: List[Dict[str, Any]] = field(default_factory=list)
    """[{role_en, gender, count}] — extracted person roles & gender"""
    quantities: List[Dict[str, Any]] = field(default_factory=list)
    """[{value:int, entity:str}] — numeric quantities in query"""
    actions: List[Dict[str, str]] = field(default_factory=list)
    """[{vi:str, en:str}] — actions/verbs extracted"""
    temporal_cues: List[Dict[str, Any]] = field(default_factory=list)
    """[{marker_vi, marker_en, order}] — temporal ordering cues"""
    negated_attributes: List[str] = field(default_factory=list)
    """Attributes explicitly negated, e.g. ['áo đỏ', 'phòng học']"""
    must_have: List[str] = field(default_factory=list)
    """Positive constraints from non-negated parts of the query"""
    emotions: List[str] = field(default_factory=list)
    """Detected emotions/expressions in English, e.g. ['happy', 'tense']"""
    lighting: str = ""
    """Lighting/weather context, e.g. 'daytime', 'sunny'"""
    clothing_details: List[Dict[str, str]] = field(default_factory=list)
    """[{vi, en, color}] — clothing items with colors"""
    retrieval_weights: Dict[str, float] = field(
        default_factory=lambda: {"visual": 0.60, "ocr": 0.25, "caption": 0.15}
    )
    """Engine weights {visual, ocr, caption} computed by IntentScorer"""
    language_mix: Dict[str, float] = field(
        default_factory=lambda: {"vi": 1.0, "en": 0.0}
    )
    """Language ratio {vi, en} detected in raw_text"""
    # ── Pre-built prompts (set by QueryParser) ──
    clip_prompt: str = ""
    """Final English sentence for CLIP text encoder (≤77 tokens)"""
    ocr_query: str = ""
    """Lexical keyword string for Qdrant/BM25 OCR search"""
    vlm_verification_prompt: str = ""
    """Structured prompt for VLM (includes negations + must_have constraints)"""


@dataclass
class QAQuery:
    """Dạng 2: Question-Answering query."""
    event_description: str
    question: str
    # ── Legacy fields ──
    answer_type: str = "description"
    """Coarse type: 'count' | 'color' | 'name' | 'yes_no' | 'description'"""
    answer_language: str = "auto"
    top_k: int = 20
    target_prefix: str = ""
    # ── Deep analysis fields (new) ──
    answer_subtype: str = "description_general"
    """
    Fine-grained answer subtype (15 types):
    count_people | count_objects | count_events |
    color_clothing | color_object | color_background |
    name_person | name_place | name_thing |
    yes_no_presence | yes_no_action | yes_no_attribute |
    number_score | number_time | description_general
    """
    expected_answer_format: str = "text"
    """e.g. 'integer', 'color_name', 'person_name', 'yes_no', 'text'"""
    question_entities: List[Dict[str, Any]] = field(default_factory=list)
    """Entities extracted from the question itself"""
    scene_entities: List[Dict[str, Any]] = field(default_factory=list)
    """Entities extracted from event_description"""
    negated_attributes: List[str] = field(default_factory=list)
    """Negated constraints from either question or description"""
    retrieval_weights: Dict[str, float] = field(
        default_factory=lambda: {"visual": 0.60, "ocr": 0.25, "caption": 0.15}
    )
    vlm_verification_prompt: str = ""
    """Full VLM prompt embedding question + constraints for Qwen-VL"""


@dataclass
class EventStep:
    """Single event step within a TRAKE query sequence."""
    event_id: int           # 1-based
    event_name: str         # e.g. "Approach"
    description: str        # Natural language description
    semantic_keyframe_hint: str  # Exact moment hint for VLM alignment


@dataclass
class TRAKEQuery:
    """Dạng 3: Temporal Retrieval & Alignment of Key Events."""
    activity_name: str
    event_sequence: List[EventStep]
    sport_category: str = ""
    top_k_videos: int = 10    # Number of candidate videos to check in Phase 1
    top_k_frames: int = 20    # Candidates per event in Phase 2
    video_id: str = ""        # If set, skip Phase 1 and go directly to alignment
    target_prefix: str = ""   # e.g. "L23", "L24"


# ============================================================
# Retrieval & Result Types
# ============================================================

@dataclass
class QueryIntent:
    """Parsed structured intent extracted from a raw user query."""
    raw_query: str
    translated_query: str
    visual_descriptors: List[str] = field(default_factory=list)
    ocr_keywords: List[str] = field(default_factory=list)
    objects: List[str] = field(default_factory=list)
    spatial_constraints: List[str] = field(default_factory=list)
    temporal_order: List[str] = field(default_factory=list)
    strategy_weights: Dict[str, float] = field(
        default_factory=lambda: {"visual": 0.5, "ocr": 0.3, "caption": 0.2}
    )


@dataclass
class SearchResult:
    """Individual item returned by a Retriever or Reranker."""
    keyframe_id: str        # e.g. "L21_V001_n5"
    video_id: str
    n: int                  # 1-based keyframe number
    frame_idx: int          # BTC submission value
    pts_time: float
    score: float
    retriever_source: str   # e.g. "visual_clip32", "ocr_bm25", "fusion"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvidenceResult:
    """Final verified answer returned by the pipeline."""
    video_id: str
    frame_idx: int          # BTC submission value
    n: int                  # Keyframe number
    pts_time: float
    confidence: float
    explanation: str = ""
    top_results: List[SearchResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# Submission Types (BTC Format)
# ============================================================

@dataclass
class KISSubmission:
    """Submission record for Dạng 1 (KIS)."""
    query_id: str
    video_id: str
    frame_idx: int


@dataclass
class QASubmission:
    """Submission record for Dạng 2 (Q&A)."""
    query_id: str
    video_id: str
    frame_idx: int
    answer: str


@dataclass
class TRAKEEventResult:
    """Single event result within TRAKE submission."""
    event_id: int
    frame_idx: int
    pts_time: float


@dataclass
class TRAKESubmission:
    """Submission record for Dạng 3 (TRAKE)."""
    query_id: str
    video_id: str
    events: List[TRAKEEventResult]
