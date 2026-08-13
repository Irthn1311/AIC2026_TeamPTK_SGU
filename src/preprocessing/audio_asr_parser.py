"""
Audio Extraction & Faster-Whisper ASR Transcriber (Branch 2)
============================================================
Extracts 16kHz mono audio from video files (FFmpeg -> .wav), runs Faster-Whisper ASR
with VAD filter and word-level timestamps, and produces structured ASR segments & retrieval chunks.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from src.retrieval.logging_utils import setup_logger

logger = setup_logger("audio-asr-parser")


def extract_audio_wav(
    video_path: Path | str,
    output_wav_path: Path | str,
    sample_rate: int = 16000,
    channels: int = 1,
) -> bool:
    """Extract 16kHz mono WAV audio from video using FFmpeg."""
    video_path = Path(video_path)
    output_wav_path = Path(output_wav_path)
    output_wav_path.parent.mkdir(parents=True, exist_ok=True)

    if output_wav_path.exists() and output_wav_path.stat().st_size > 0:
        logger.info("WAV file already exists: %s", output_wav_path)
        return True

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-ac", str(channels),
        str(output_wav_path),
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return output_wav_path.exists() and output_wav_path.stat().st_size > 0
    except Exception as exc:
        logger.error("FFmpeg audio extraction failed for %s: %s", video_path, exc)
        return False


def get_default_hf_cache() -> str:
    """Return a local cache dir without falling back to the C: drive."""
    env_cache = os.environ.get("AIC_FASTER_WHISPER_CACHE")
    if env_cache:
        cache = Path(env_cache)
        cache.mkdir(parents=True, exist_ok=True)
        return str(cache)

    project_cache = Path(__file__).resolve().parents[2] / ".cache" / "faster_whisper"
    if Path("/kaggle").exists():
        project_cache.mkdir(parents=True, exist_ok=True)
        return str(project_cache)

    e_cache = Path(r"E:\AI Challenge TP.HCM 2026\AIC2026_TeamPTK_SGU\.cache\faster_whisper")
    if e_cache.parent.exists():
        e_cache.mkdir(parents=True, exist_ok=True)
        return str(e_cache)
    project_cache.mkdir(parents=True, exist_ok=True)
    return str(project_cache)


def run_faster_whisper_asr(
    audio_path: Path | str,
    video_id: str = "video",
    model_size: str = "large-v3-turbo",
    device: str = "cuda",
    compute_type: str = "float16",
    language: str = "vi",
    vad_filter: bool = True,
    word_timestamps: bool = True,
    download_root: Optional[str] = None,
) -> list[dict[str, Any]]:
    """
    Transcribe audio with faster-whisper using large-v3-turbo, VAD, and word timestamps.
    Returns a list of segment dictionaries with:
      video_id, asr_id, start, end, text_raw, words, word_probability, avg_logprob, no_speech_prob.
    """
    from faster_whisper import WhisperModel

    if download_root is None:
        download_root = get_default_hf_cache()

    logger.info(
        "Loading faster-whisper model '%s' (device=%s, compute_type=%s, cache=%s)...",
        model_size, device, compute_type, download_root
    )
    t0 = time.time()
    model = WhisperModel(
        model_size,
        device=device,
        compute_type=compute_type,
        download_root=download_root,
    )
    load_time = time.time() - t0
    logger.info("Model loaded in %.2f seconds.", load_time)

    logger.info(
        "Transcribing %s (language=%s, vad_filter=%s, word_timestamps=%s)...",
        audio_path, language, vad_filter, word_timestamps
    )
    t1 = time.time()
    segments_generator, info = model.transcribe(
        str(audio_path),
        language=language,
        vad_filter=vad_filter,
        word_timestamps=word_timestamps,
        vad_parameters=dict(min_silence_duration_ms=500),
    )

    logger.info(
        "Detected language: '%s' with probability %.4f. Total audio duration: %.2fs",
        info.language, info.language_probability, info.duration
    )

    results: list[dict[str, Any]] = []
    for idx, seg in enumerate(segments_generator):
        text_raw = str(seg.text).strip()
        if not text_raw:
            continue

        words_list = []
        word_probs = []
        if seg.words:
            for w in seg.words:
                w_text = str(w.word).strip()
                w_start = round(float(w.start), 3)
                w_end = round(float(w.end), 3)
                w_prob = round(float(w.probability), 4)
                words_list.append({
                    "word": w_text,
                    "start": w_start,
                    "end": w_end,
                    "probability": w_prob,
                })
                word_probs.append(w.probability)

        if word_probs:
            avg_word_prob = round(float(np.mean(word_probs)), 4)
        elif seg.avg_logprob is not None:
            avg_word_prob = round(float(np.exp(seg.avg_logprob)), 4)
        else:
            avg_word_prob = 1.0

        segment_dict = {
            "video_id": video_id,
            "asr_id": f"{video_id}_asr_{idx:04d}",
            "start": round(float(seg.start), 3),
            "end": round(float(seg.end), 3),
            "text_raw": text_raw,
            "words": words_list,
            "word_probability": avg_word_prob,
            "avg_logprob": round(float(seg.avg_logprob), 4) if seg.avg_logprob is not None else None,
            "no_speech_prob": round(float(seg.no_speech_prob), 4) if seg.no_speech_prob is not None else None,
        }
        results.append(segment_dict)

    transcribe_time = time.time() - t1
    logger.info("Transcribed %d segments in %.2f seconds.", len(results), transcribe_time)
    return results


def build_asr_retrieval_chunks(
    segments: list[dict[str, Any]],
    video_id: str,
    min_chunk_duration: float = 15.0,
    target_chunk_duration: float = 25.0,
    max_chunk_duration: float = 30.0,
    overlap_duration: float = 5.0,
    max_gap_seconds: float = 8.0,
) -> list[dict[str, Any]]:
    """
    Generate retrieval chunks of ~15-30s with ~5s overlap from consecutive ASR segments.
    Each chunk contains:
      chunk_id, video_id, start, end, duration, text, segment_ids, words_count, avg_probability
    """
    if not segments:
        return []

    chunks: list[dict[str, Any]] = []
    num_segs = len(segments)
    chunk_idx = 0
    i = 0

    while i < num_segs:
        start_time = segments[i]["start"]
        current_text_parts = []
        current_segment_ids = []
        current_probs = []
        total_words = 0
        end_time = segments[i]["end"]

        j = i
        while j < num_segs:
            seg = segments[j]
            seg_start = seg["start"]
            seg_end = seg["end"]

            # If there is a large silence gap between segments, stop this chunk
            if j > i and (seg_start - end_time) > max_gap_seconds:
                break

            # If adding this segment exceeds max_chunk_duration and we already meet min_chunk_duration
            if j > i and (seg_end - start_time) > max_chunk_duration and (end_time - start_time) >= min_chunk_duration:
                break

            current_text_parts.append(seg["text_raw"])
            current_segment_ids.append(seg["asr_id"])
            current_probs.append(seg.get("word_probability", 1.0))
            total_words += len(seg.get("words", [])) or len(seg["text_raw"].split())
            end_time = seg_end

            # If reached target duration, advance j and stop chunk accumulation
            if (end_time - start_time) >= target_chunk_duration:
                j += 1
                break
            j += 1

        duration = round(end_time - start_time, 3)
        chunk_text = " ".join(current_text_parts).strip()
        avg_prob = round(float(np.mean(current_probs)), 4) if current_probs else 1.0

        chunks.append({
            "chunk_id": f"{video_id}_chunk_{chunk_idx:04d}",
            "video_id": video_id,
            "start": round(start_time, 3),
            "end": round(end_time, 3),
            "duration": duration,
            "text": chunk_text,
            "segment_ids": current_segment_ids,
            "words_count": total_words,
            "avg_probability": avg_prob,
        })
        chunk_idx += 1

        if j >= num_segs:
            break

        # Advance start index `i` with ~overlap_duration overlap
        target_next_start = end_time - overlap_duration
        next_i = i + 1
        while next_i < j and segments[next_i]["start"] < target_next_start:
            next_i += 1

        if next_i <= i:
            next_i = i + 1
        i = next_i

    logger.info("Constructed %d retrieval chunks for %s.", len(chunks), video_id)
    return chunks



def run_whisper_asr(wav_path: Path, model_size: str = "base") -> list[dict[str, Any]]:
    """Legacy OpenAI Whisper transcribe wrapper."""
    import whisper

    model = whisper.load_model(model_size)
    result = model.transcribe(str(wav_path), language="vi")
    segments = result.get("segments", [])

    clean_segments = []
    for seg in segments:
        text = str(seg.get("text", "")).strip()
        if text:
            clean_segments.append({
                "start_s": round(float(seg["start"]), 2),
                "end_s": round(float(seg["end"]), 2),
                "text": text,
            })
    return clean_segments


def map_asr_to_keyframes(
    video_id: str,
    asr_segments: list[dict[str, Any]],
    mapping_dir: Path,
    keyframe_dir: Path,
    time_window_s: float = 3.0,
) -> list[dict[str, Any]]:
    """Legacy keyframe alignment helper."""
    from src.retrieval.mapping_loader import load_keyframe_mapping

    mapping_path = mapping_dir / f"{video_id}.csv"
    if not mapping_path.exists():
        return []

    mapping_df = load_keyframe_mapping(mapping_path, keyframe_dir)
    records = []

    for _, row in mapping_df.iterrows():
        kf_time = float(row["timestamp_seconds"])
        matched_texts = []
        for seg in asr_segments:
            s_start = seg.get("start", seg.get("start_s", 0.0))
            s_end = seg.get("end", seg.get("end_s", 0.0))
            text = seg.get("text_raw", seg.get("text", ""))
            if (s_start <= kf_time <= s_end) or (abs(kf_time - s_start) <= time_window_s):
                matched_texts.append(text)

        asr_text = " ".join(matched_texts).strip()
        records.append({
            "video_id": video_id,
            "n_idx": int(row["keyframe_name"].split(".")[0]),
            "keyframe_name": row["keyframe_name"],
            "keyframe_path": row["keyframe_path"],
            "frame_idx": int(row["frame_idx"]),
            "timestamp_seconds": kf_time,
            "asr_text": asr_text,
            "has_asr": bool(asr_text),
        })

    return records
