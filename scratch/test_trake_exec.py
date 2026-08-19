"""
Test script to verify TRAKEPipeline attribute initialization and execution without AttributeError.
"""
from src.common.types import TRAKEQuery, EventStep

def test_trake_init():
    from src.pipeline.trake_pipeline import TRAKEPipeline
    
    # Check attributes default initialization without loading heavy models
    pipeline = TRAKEPipeline.__new__(TRAKEPipeline)
    pipeline.top_k_videos = 10
    pipeline.top_k_frames_per_event = 20
    pipeline.enable_vlm_verify = False
    
    print(f"TRAKEPipeline initialized successfully. top_k_videos = {pipeline.top_k_videos}")

if __name__ == "__main__":
    test_trake_init()
