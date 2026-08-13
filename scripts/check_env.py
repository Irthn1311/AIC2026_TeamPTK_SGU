"""
Environment Audit & Health Check Script
=======================================
Verifies Python version, GPU/CPU availability, PyTorch, FAISS,
Transformers, SentenceTransformers, Whisper, RapidFuzz, PaddleOCR, and Pandas.
"""

from __future__ import annotations

import sys

def audit_environment():
    print("=" * 70)
    print(" 🔍 AI CHALLENGE 2026 - ENVIRONMENT & HARDWARE AUDIT")
    print("=" * 70)

    print(f"Python Version : {sys.version.split()[0]}")

    # 1. PyTorch & CUDA
    try:
        import torch
        cuda_avail = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda_avail else "N/A (CPU Mode)"
        print(f"PyTorch        : {torch.__version__}")
        print(f"CUDA Available : {cuda_avail} (Device: {gpu_name})")
    except ImportError:
        print("PyTorch        : NOT INSTALLED ❌")

    # 2. FAISS
    try:
        import faiss
        print(f"FAISS          : Installed ✅ (Version: {getattr(faiss, '__version__', 'OK')})")
    except ImportError:
        print("FAISS          : NOT INSTALLED ❌")

    # 3. Transformers & OpenCLIP / SentenceTransformers
    try:
        import transformers
        print(f"Transformers   : {transformers.__version__} ✅")
    except ImportError:
        print("Transformers   : NOT INSTALLED ❌")

    try:
        import sentence_transformers
        print(f"SentenceTrans  : {sentence_transformers.__version__} ✅")
    except ImportError:
        print("SentenceTrans  : NOT INSTALLED ❌")

    # 4. OpenAI Whisper
    try:
        import whisper
        print(f"OpenAI Whisper : Installed ✅")
    except ImportError:
        print("OpenAI Whisper : NOT INSTALLED ❌")

    # 5. RapidFuzz & Pandas
    try:
        import rapidfuzz
        print(f"RapidFuzz      : {rapidfuzz.__version__} ✅")
    except ImportError:
        print("RapidFuzz      : NOT INSTALLED ❌")

    try:
        import pandas as pd
        print(f"Pandas         : {pd.__version__} ✅")
    except ImportError:
        print("Pandas         : NOT INSTALLED ❌")

    # 6. PaddleOCR
    try:
        import paddleocr
        print(f"PaddleOCR      : Installed ✅")
    except ImportError:
        print("PaddleOCR      : NOT INSTALLED ❌")

    print("=" * 70)
    print(" ✅ ALL ENVIRONMENT DEPENDENCIES AUDITED!")
    print("=" * 70)

if __name__ == "__main__":
    audit_environment()
