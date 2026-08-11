"""
Demonstrate and verify step-by-step query processing logs in RetrievalPipeline.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

# Mock faiss and open_clip if not installed locally
for mod_name in ['faiss', 'open_clip']:
    try:
        __import__(mod_name)
    except ImportError:
        mock = MagicMock()
        mock.__spec__ = None
        sys.modules[mod_name] = mock

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.types import SearchResult
from src.pipeline.retrieval_pipeline import RetrievalPipeline
from src.fusion.reciprocal_rank import ReciprocalRankFusion
from src.evidence.frame_selector import FrameSelector
from src.reasoning.query_parser import QueryParser
from src.utils.logger import get_logger

logger = get_logger("test_logging")

class DummyVisualRetriever:
    def retrieve(self, query: str, top_k: int = 50, target_prefix=None):
        return [
            SearchResult("L21_V001_n1", "L21_V001", 1, 0, 0.0, 0.85, "visual_clip32", {"topic_category": "nau_an"}),
            SearchResult("L21_V002_n1", "L21_V002", 1, 0, 0.0, 0.80, "visual_clip32", {"topic_category": "tin_tuc"}),
        ]

def main():
    logger.info("Initializing Retrieval Pipeline with step-by-step logger...")

    pipeline = RetrievalPipeline(
        faiss_db=MagicMock(),
        meta_store=MagicMock(),
        encoder=MagicMock(),
    )
    pipeline._vis_ret = DummyVisualRetriever()

    sample_query = {
        "text": "Tìm đoạn clip đầu bếp đang hướng dẫn nấu món phở bò trong bếp",
        "query_id": "TEST_Q01",
    }

    res = pipeline.run(sample_query, query_id="TEST_Q01")
    print("\nTest completed successfully!")

if __name__ == "__main__":
    main()
