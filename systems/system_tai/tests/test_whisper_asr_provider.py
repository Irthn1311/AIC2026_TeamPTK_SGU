"""Unit tests for Whisper ASR Audio Extraction and Transcript Provider."""

from __future__ import annotations

from pathlib import Path

import pytest

from system_tai.evidence.asr_provider import ASRSegment, WhisperASRExtractor
from system_tai.retrieval.bm25_search import BM25Index


def test_asr_segment_dataclass_and_frame_conversion() -> None:
    seg = ASRSegment(
        video_id="L21_V001",
        start_sec=10.5,
        end_sec=14.0,
        text="Xin chào các bạn đang theo dõi chương trình",
        language="vi",
        confidence=0.98,
    )
    assert seg.video_id == "L21_V001"
    assert seg.start_sec == 10.5
    assert seg.end_sec == 14.0
    assert seg.text == "Xin chào các bạn đang theo dõi chương trình"

    # FPS = 25.0 -> frame = round(10.5 * 25) = 262, interval = (262, 350)
    assert seg.to_actual_frame_id(fps=25.0) == 262
    assert seg.to_frame_interval(fps=25.0) == (262, 350)

    # FPS = 30.0 -> frame = round(10.5 * 30) = 315
    assert seg.to_actual_frame_id(fps=30.0) == 315

    # Validation errors
    with pytest.raises(ValueError, match="video_id must not be empty"):
        ASRSegment("", 1.0, 2.0, "hello")
    with pytest.raises(ValueError, match="timestamps must be non-negative"):
        ASRSegment("V1", -1.0, 2.0, "hello")
    with pytest.raises(ValueError, match="start_sec must not exceed end_sec"):
        ASRSegment("V1", 5.0, 2.0, "hello")


def test_whisper_asr_cache_save_and_load(tmp_path: Path) -> None:
    segments = [
        ASRSegment("L21_V001", 0.0, 3.5, "Phóng viên đài truyền hình đang có mặt tại hiện trường", "vi", 0.95),
        ASRSegment("L21_V001", 3.8, 8.2, "Đám cháy đã được khống chế hoàn toàn", "vi", 0.92),
    ]
    cache_file = tmp_path / "L21_V001_asr.json"

    WhisperASRExtractor.save_transcript(segments, cache_file)
    assert cache_file.exists()

    loaded = WhisperASRExtractor.load_cached_transcript(cache_file)
    assert len(loaded) == 2
    assert loaded[0].video_id == "L21_V001"
    assert loaded[0].text == "Phóng viên đài truyền hình đang có mặt tại hiện trường"
    assert loaded[1].text == "Đám cháy đã được khống chế hoàn toàn"
    assert loaded[1].start_sec == 3.8


def test_asr_to_bm25_documents_and_search() -> None:
    segments = [
        ASRSegment("L21_V001", 12.0, 16.0, "Xe cứu hỏa đã đến dập tắt ngọn lửa lớn", "vi"),
        ASRSegment("L21_V002", 50.0, 55.0, "Người dân đang tập trung xem biểu diễn ca nhạc", "vi"),
    ]
    docs = WhisperASRExtractor.to_bm25_documents(segments, fps=25.0)
    assert len(docs) == 2
    assert docs[0].video_id == "L21_V001"
    assert docs[0].frame_id == 300  # 12.0 * 25.0
    assert docs[0].metadata["source"] == "asr_whisper"

    # Index into BM25 and search
    index = BM25Index()
    index.add_documents(docs)

    results = index.search("dập tắt lửa cứu hỏa", top_k=1)
    assert len(results) == 1
    matched_doc, score = results[0]
    assert matched_doc.video_id == "L21_V001"
    assert matched_doc.frame_id == 300
    assert score > 0.0


def test_whisper_extractor_missing_video_and_error_handling(tmp_path: Path) -> None:
    extractor = WhisperASRExtractor(cache_dir=tmp_path)
    # Non-existent video path
    fake_video = tmp_path / "non_existent_video.mp4"
    audio_out = tmp_path / "out.wav"

    success = extractor.extract_audio(fake_video, audio_out)
    assert not success
    assert not audio_out.exists()

    res = extractor.transcribe_video("L21_V999", fake_video)
    assert res == []
