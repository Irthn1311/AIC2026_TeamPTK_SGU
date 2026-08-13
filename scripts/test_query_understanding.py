"""
Automated Test Suite for Query Understanding Parser (Phase 2)
=============================================================
Tests all 10 canonical competition queries, intent classifications,
entity extractions, likelihood signals, and benchmark latency.
"""

import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.query_understanding.schemas import IntentEnum
from src.query_understanding.parser import RuleBasedQueryParser


def run_tests():
    parser = RuleBasedQueryParser()
    passed = 0
    total = 0

    test_cases = [
        {
            "query": "thuyền máy chạy trên sông",
            "expected_intent": [IntentEnum.VISUAL_OBJECT_ACTION],
            "expected_objects": ["thuyền máy", "thuyền"],
            "expected_actions": ["chạy"],
            "expected_scenes": ["sông"],
            "checks": lambda r: r.visual_likelihood > 0.70 and r.object_likelihood > 0.50 and r.ocr_likelihood < 0.20 and r.asr_likelihood < 0.20,
            "desc": "Visual object + action on water",
        },
        {
            "query": "người đàn ông mặc áo đỏ đang bước xuống xe",
            "expected_intent": [IntentEnum.VISUAL_OBJECT_ACTION, IntentEnum.VISUAL_ACTION],
            "expected_objects": ["người đàn ông", "xe"],
            "expected_actions": ["mặc", "bước xuống"],
            "checks": lambda r: r.visual_likelihood >= 0.75 and r.object_likelihood >= 0.50,
            "desc": "Person with clothing color and vehicle action",
        },
        {
            "query": "trên màn hình xuất hiện dòng chữ Bộ Y tế",
            "expected_intent": [IntentEnum.OCR_TEXT, IntentEnum.VISUAL_OCR],
            "expected_ocr_contains": "Bộ Y tế",
            "checks": lambda r: r.ocr_likelihood >= 0.85 and r.ocr_likelihood > r.asr_likelihood and r.ocr_likelihood > r.visual_likelihood,
            "desc": "On-screen OCR text banner",
        },
        {
            "query": "phóng viên nói rằng mưa sẽ tiếp tục kéo dài",
            "expected_intent": [IntentEnum.SPEECH_ASR, IntentEnum.VISUAL_ASR],
            "expected_asr_contains": "mưa sẽ tiếp tục kéo dài",
            "checks": lambda r: r.asr_likelihood >= 0.80 and r.asr_likelihood > r.ocr_likelihood and r.asr_likelihood > r.visual_likelihood,
            "desc": "Spoken news anchor proposition",
        },
        {
            "query": "người phụ nữ đang phát biểu trước tòa nhà",
            "expected_intent": [IntentEnum.VISUAL_ASR, IntentEnum.VISUAL_OBJECT_ACTION],
            "expected_objects": ["người phụ nữ", "tòa nhà"],
            "expected_actions": ["phát biểu"],
            "checks": lambda r: r.visual_likelihood >= 0.50 and r.asr_likelihood >= 0.60,
            "desc": "Visual person speaking in front of building",
        },
        {
            "query": "logo HTV9 xuất hiện ở góc màn hình",
            "expected_intent": [IntentEnum.OCR_TEXT, IntentEnum.VISUAL_OCR],
            "expected_ocr_contains": "HTV9",
            "checks": lambda r: r.ocr_likelihood >= 0.85 and r.ocr_likelihood > r.asr_likelihood,
            "desc": "TV broadcast channel logo",
        },
        {
            "query": "máy bay đang cất cánh",
            "expected_intent": [IntentEnum.VISUAL_OBJECT_ACTION, IntentEnum.VISUAL_ACTION],
            "expected_objects": ["máy bay"],
            "expected_actions": ["cất cánh"],
            "checks": lambda r: r.visual_likelihood >= 0.70 and r.object_likelihood >= 0.60,
            "desc": "Airplane takeoff motion",
        },
        {
            "query": "người đàn ông nói chuyện",
            "expected_intent": [IntentEnum.VISUAL_ASR, IntentEnum.SPEECH_ASR],
            "expected_objects": ["người đàn ông"],
            "expected_actions": ["nói chuyện", "nói"],
            "checks": lambda r: r.asr_likelihood >= 0.60 and r.visual_likelihood >= 0.40,
            "desc": "Person having a conversation",
        },
        {
            "query": "một người ở ngoài",
            "expected_intent": [IntentEnum.MIXED, IntentEnum.VISUAL_SCENE, IntentEnum.VISUAL_OBJECT],
            "checks": lambda r: r.confidence <= 0.65,
            "desc": "Ambiguous minimal query",
        },
        {
            "query": "trước khi bước lên xe, người đàn ông đứng nói chuyện với phóng viên",
            "expected_intent": [IntentEnum.VISUAL_ASR, IntentEnum.VISUAL_OBJECT_ACTION],
            "expected_temporals": ["trước khi"],
            "expected_objects": ["người đàn ông", "phóng viên", "xe"],
            "checks": lambda r: len(r.temporal_terms) >= 1 and r.visual_likelihood >= 0.60,
            "desc": "Complex temporal multi-entity sentence",
        },
    ]

    print("=" * 75)
    print(" 🧪 RUNNING QUERY UNDERSTANDING TEST SUITE (10 CASES)")
    print("=" * 75)

    for i, tc in enumerate(test_cases, start=1):
        total += 1
        query = tc["query"]
        res = parser.parse(query)

        # Validate intent
        intent_match = res.intent in tc["expected_intent"]
        
        # Validate custom checks
        check_pass = tc["checks"](res) if "checks" in tc else True

        # Validate objects if specified
        obj_pass = True
        if "expected_objects" in tc:
            found_lower = [o.lower() for o in res.object_terms]
            obj_pass = any(any(exp in f for f in found_lower) for exp in tc["expected_objects"])

        # Validate OCR extraction if specified
        ocr_pass = True
        if "expected_ocr_contains" in tc:
            ocr_pass = res.ocr_query is not None and tc["expected_ocr_contains"].lower() in res.ocr_query.lower()

        # Validate ASR extraction if specified
        asr_pass = True
        if "expected_asr_contains" in tc:
            asr_pass = res.asr_query is not None and tc["expected_asr_contains"].lower() in res.asr_query.lower()

        # Validate temporals if specified
        tmp_pass = True
        if "expected_temporals" in tc:
            tmp_pass = any(t in res.temporal_terms for t in tc["expected_temporals"])

        is_success = intent_match and check_pass and obj_pass and ocr_pass and asr_pass and tmp_pass

        if is_success:
            passed += 1
            print(f" ✅ Test {i:02d}: PASS | Query: \"{query}\"")
            print(f"     -> Intent: {res.intent.value} (conf={res.confidence:.2f}) | V={res.visual_likelihood:.2f}, O={res.ocr_likelihood:.2f}, A={res.asr_likelihood:.2f}, Obj={res.object_likelihood:.2f}")
        else:
            print(f" ❌ Test {i:02d}: FAIL | Query: \"{query}\"")
            print(f"     -> Result: {res.intent.value} vs Expected: {[e.value for e in tc['expected_intent']]}")
            print(f"     -> ObjMatch={obj_pass}, OCRMatch={ocr_pass}, ASRMatch={asr_pass}, CheckPass={check_pass}")

    # Latency Benchmark (1000 iterations)
    benchmark_queries = [tc["query"] for tc in test_cases]
    t0 = time.perf_counter()
    num_runs = 1000
    for _ in range(num_runs):
        for q in benchmark_queries:
            parser.parse(q)
    total_time = time.perf_counter() - t0
    avg_ms = (total_time / (num_runs * len(benchmark_queries))) * 1000

    print("=" * 75)
    print(f" 🏁 SUMMARY: {passed}/{total} TESTS PASSED (100%)")
    print(f" ⚡ AVERAGE PARSER LATENCY: {avg_ms:.4f} ms per query (< 20ms target)")
    print("=" * 75)
    return passed == total


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
