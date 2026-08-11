"""
System-wide constants for AIC Video Retrieval System.
Aligned with the official AIC-HCMC Kaggle dataset structure.
"""

# ============================================================
# Dataset Structure Constants
# ============================================================

# Kaggle dataset slug (user uploads their own dataset)
KAGGLE_DATASET_SLUG = "aic-hcmc-data"
KAGGLE_INDEXES_SLUG = "aic-hcmc-indexes"

# Batch ID format: L{XX} where XX is zero-padded batch number
BATCH_ID_PREFIX = "L"

# Video ID format within a batch: {L}_{V} e.g. L21_V001
VIDEO_ID_FORMAT = "{L}_{V}"

# ============================================================
# CSV Column Names (map-keyframes CSV)
# ============================================================
CSV_COL_N          = "n"           # 1-based keyframe sequence number
CSV_COL_PTS_TIME   = "pts_time"    # Timestamp in seconds
CSV_COL_FPS        = "fps"         # Video FPS
CSV_COL_FRAME_IDX  = "frame_idx"   # Frame index in video (BTC submission)

# ============================================================
# CLIP Feature Constants
# ============================================================
CLIP32_FEATURE_DIM  = 512   # CLIP-32 embedding dimension (from pre-extracted .npy)
CLIP_L14_DIM        = 768   # CLIP ViT-L/14 embedding dimension
SIGLIP_DIM          = 1152  # SigLIP embedding dimension

# ============================================================
# Default Retrieval Parameters
# ============================================================
DEFAULT_TOP_K          = 100   # Initial retrieval candidates
DEFAULT_RERANK_TOP_K   = 20    # After reranking
DEFAULT_FINAL_TOP_K    = 10    # Final results returned

# ============================================================
# Keyframe Image Format
# ============================================================
KEYFRAME_EXT        = ".jpg"
KEYFRAME_NAME_FORMAT = "{n:03d}.jpg"   # Standard 3-digit zero-padded filename (e.g. 001.jpg, 090.jpg)

# ============================================================
# Parquet Master Index Schema
# ============================================================
MASTER_PARQUET_COLS = [
    "faiss_id",        # int: monotonic index for FAISS
    "keyframe_id",     # str: "L21_V001_n5"
    "video_id",        # str: "L21_V001"
    "batch_id",        # str: "L21"
    "n",               # int: 1-based keyframe number
    "frame_idx",       # int: BTC submission value
    "pts_time",        # float: timestamp seconds
    "fps",             # float: video fps
    "image_path",      # str: absolute path to .jpg
    "topic_category",  # str: classified topic category (e.g. "nau_an", "tin_tuc")
]

# ============================================================
# FAISS Index Configuration
# ============================================================
FAISS_INDEX_TYPE    = "HNSW"    # Flat | IVFFlat | HNSW | IVF-PQ
FAISS_HNSW_M        = 32        # HNSW construction parameter
FAISS_HNSW_EF_SEARCH = 64       # HNSW search parameter
FAISS_METRIC        = "cosine"  # cosine | l2 | ip

# ============================================================
# Qdrant Collection Names
# ============================================================
QDRANT_COLLECTION_CAPTIONS = "captions"
QDRANT_COLLECTION_OCR      = "ocr"
QDRANT_COLLECTION_ASR      = "asr"
QDRANT_VECTOR_DIM_BGE      = 1024  # BGE-M3 output dimension
