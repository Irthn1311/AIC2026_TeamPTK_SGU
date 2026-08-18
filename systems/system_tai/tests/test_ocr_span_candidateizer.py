from __future__ import annotations

import pytest

from system_tai.qa.ocr_span_candidateizer import (
    OCRSpanCandidate,
    extract_and_rank_canonical_ocr_spans,
    is_junk_token,
    score_span_candidate,
)


def test_junk_token_detection() -> None:
    assert is_junk_token("~") is True
    assert is_junk_token("|") is True
    assert is_junk_token("---") is True
    assert is_junk_token("†.Il¿1f#!/ÿU/-PW/ĐMMAANWS") is True
    assert is_junk_token("DIOR") is False
    assert is_junk_token("CELINE") is False
    assert is_junk_token("Trưng") is False
    assert is_junk_token("50.000") is False


def test_span_candidate_ranking_is_strictly_deterministic() -> None:
    tsv_sample = (
        "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
        "5\t1\t1\t1\t1\t1\t89\t36\t85\t47\t92.0\tCHƯƠNG\n"
        '5\t1\t1\t1\t1\t2\t1247\t55\t5\t3\t27.0\t"\n'
        "5\t1\t1\t1\t14\t1\t144\t587\t47\t56\t85.0\tA\n"
        "5\t1\t1\t1\t14\t2\t199\t585\t93\t35\t21.0\tTV\n"
        "5\t1\t1\t1\t14\t3\t401\t593\t60\t19\t92.0\tDIOR\n"
        "5\t1\t1\t1\t14\t4\t470\t593\t89\t19\t91.0\tTRƯNG\n"
        "5\t1\t1\t1\t14\t5\t569\t587\t49\t24\t89.0\tBÀY\n"
        "5\t1\t1\t1\t14\t6\t626\t593\t87\t19\t96.0\tTRANG\n"
        "5\t1\t1\t1\t14\t7\t722\t589\t65\t33\t92.0\tPHỤC\n"
        "5\t1\t1\t1\t14\t8\t797\t586\t50\t25\t92.0\tCỦA\n"
        "5\t1\t1\t1\t14\t9\t857\t589\t87\t31\t90.0\tCELINE,\n"
    ).encode()

    ranked_1 = extract_and_rank_canonical_ocr_spans(tsv_sample, max_n=4)
    ranked_2 = extract_and_rank_canonical_ocr_spans(tsv_sample, max_n=4)

    # 1. Total spans count must be identical
    assert len(ranked_1) == len(ranked_2)

    # 2. Every candidate rank, score, and normalized span must be 100% bit-exact across repeated runs
    for c1, c2 in zip(ranked_1, ranked_2):
        assert c1.sort_key == c2.sort_key
        assert c1.normalized_span == c2.normalized_span
        assert c1.score == c2.score

    # 3. DIOR must be ranked in Top 5
    top_5_spans = [c.normalized_span for c in ranked_1[:5]]
    assert "dior" in top_5_spans
