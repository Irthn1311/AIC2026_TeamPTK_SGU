import json
from pathlib import Path
import pytest

from system_tai.quality.l21_150_schema import load_l21_150_benchmark


BENCHMARK_PATH = Path("systems/system_tai/benchmarks/l21_150_diagnostic/benchmark.json")


def test_qa_benchmark_anchors_contract():
    with open(BENCHMARK_PATH, encoding="utf-8") as f:
        bm = json.load(f)

    qa_dev = [q for q in bm["queries"] if q.get("task_type") == "qa" and q.get("split") == "DEV"]
    assert len(qa_dev) == 38

    q_map = {q["query_id"]: q for q in qa_dev}

    # QA-13: L21_V005, interval [23160, 23220]
    assert q_map["QA-13"]["video_id"] == "L21_V005"
    assert q_map["QA-13"]["proposed_interval"] == [23160, 23220]
    assert "Trâu" in q_map["QA-13"]["accepted_answers"]

    # QA-20: L21_V007, interval [14610, 14670]
    assert q_map["QA-20"]["video_id"] == "L21_V007"
    assert q_map["QA-20"]["proposed_interval"] == [14610, 14670]
    assert "2" in q_map["QA-20"]["accepted_answers"]

    # QA-46: L21_V016, interval [8190, 8250]
    assert q_map["QA-46"]["video_id"] == "L21_V016"
    assert q_map["QA-46"]["proposed_interval"] == [8190, 8250]
    assert "Dệt" in q_map["QA-46"]["accepted_answers"]
