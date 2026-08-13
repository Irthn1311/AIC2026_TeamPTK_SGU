"""
Text Preprocessing & Normalization Module for AIC System.

Provides text cleaning, Vietnamese accent normalization, and query preprocessing
to improve recall and match precision in retrieval.
"""

from __future__ import annotations

import re
import unicodedata
from typing import List


def normalize_vietnamese_text(text: str) -> str:
    """
    Unicode NFC normalization for Vietnamese text.
    Removes duplicated spaces and unwanted control characters.
    """
    if not text:
        return ""
    # NFC normalization (ensures consistent diacritics)
    text = unicodedata.normalize("NFC", text)
    # Remove control characters
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C")
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_query(raw_text: str) -> str:
    """
    Clean raw query input before feeding to parser and retrievers.
    
    Steps:
      1. Unicode NFC normalization
      2. Strip leading/trailing quotes if wrapped
      3. Clean redundant punctuation while preserving quotes around OCR keywords
    """
    text = normalize_vietnamese_text(raw_text)
    if not text:
        return ""

    # Strip wrapping quotes around entire query
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        text = text[1:-1].strip()

    # Normalize repetitive punctuation (e.g., "???" -> "?", "..." -> "...")
    text = re.sub(r"\?{2,}", "?", text)
    text = re.sub(r"\!{2,}", "!", text)
    text = re.sub(r"\.{4,}", "...", text)

    return text
