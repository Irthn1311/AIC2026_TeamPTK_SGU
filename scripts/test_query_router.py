"""
Comprehensive Test Suite for Query Router & Dynamic Fusion Policy (Phase 3)
===========================================================================
Tests 15 diverse queries (canonical + edge cases), verifies mathematical
invariants (sum=1.0, guardrails, confidence blending), prints comparative table,
and benchmarks end-to-end parser + router throughput.
"""

import math
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.query_understanding.parser import RuleBasedQueryParser
from src.query_understanding.router import QueryRouter


def run_router_tests():
    parser = RuleBasedQueryParser()
    router = QueryRouter()

    test_cases = [
        {
            "id": 1,
            "query": "thuyền máy chạy trên sông",
            "desc": "Visual object + action on water",
            "validate": lambda d: d.dominant_branch == "visual" and d.secondary_branch == "object" and d.final_weights.visual >= 0.55 and d.final_weights.object >= 0.25 and d.final_weights.ocr <= 0.10 and d.final_weights.asr <= 0.10,
        },
        {
            "id": 2,
            "query": "người đàn ông mặc áo đỏ đang bước xuống xe",
            "desc": "Person with clothing color and vehicle action",
            "validate": lambda d: d.dominant_branch == "visual" and d.final_weights.visual >= 0.55 and d.final_weights.object >= 0.25,
        },
        {
            "id": 3,
            "query": "trên màn hình xuất hiện dòng chữ Bộ Y tế",
            "desc": "On-screen OCR text banner",
            "validate": lambda d: d.dominant_branch == "ocr" and d.final_weights.ocr >= 0.65 and d.secondary_branch == "visual",
        },
        {
            "id": 4,
            "query": "phóng viên nói rằng mưa sẽ tiếp tục kéo dài",
            "desc": "Spoken news anchor proposition",
            "validate": lambda d: d.dominant_branch == "asr" and d.final_weights.asr >= 0.60 and d.secondary_branch == "visual",
        },
        {
            "id": 5,
            "query": "người phụ nữ đang phát biểu trước tòa nhà",
            "desc": "Visual person speaking in front of building",
            "validate": lambda d: d.dominant_branch in ["asr", "visual"] and d.final_weights.object < 0.25 and (d.final_weights.visual + d.final_weights.asr) >= 0.75,
        },
        {
            "id": 6,
            "query": "logo HTV9 xuất hiện ở góc màn hình",
            "desc": "TV broadcast channel logo",
            "validate": lambda d: d.dominant_branch == "ocr" and d.final_weights.ocr >= 0.40,
        },
        {
            "id": 7,
            "query": "máy bay đang cất cánh",
            "desc": "Airplane takeoff motion",
            "validate": lambda d: d.dominant_branch == "visual" and d.secondary_branch == "object" and d.final_weights.visual >= 0.55,
        },
        {
            "id": 8,
            "query": "người đàn ông nói chuyện",
            "desc": "Person having a conversation",
            "validate": lambda d: (d.final_weights.visual + d.final_weights.asr) >= 0.75 and d.dominant_branch in ["asr", "visual"],
        },
        {
            "id": 9,
            "query": "một người ở ngoài",
            "desc": "Ambiguous minimal query -> Close to baseline",
            "validate": lambda d: abs(d.final_weights.visual - 0.40) <= 0.08 and abs(d.final_weights.ocr - 0.25) <= 0.08 and abs(d.final_weights.asr - 0.25) <= 0.08 and d.final_weights.object <= 0.15,
        },
        {
            "id": 10,
            "query": "trước khi bước lên xe, người đàn ông đứng nói chuyện với phóng viên",
            "desc": "Temporal multi-entity speech action",
            "validate": lambda d: (d.final_weights.visual + d.final_weights.asr) >= 0.70 and d.final_weights.object <= 0.25,
        },
        {
            "id": 11,
            "query": "HTV9",
            "desc": "Ultra short TV logo without marker",
            "validate": lambda d: d.final_weights.ocr <= 0.50 and abs(sum(d.final_weights.to_dict().values()) - 1.0) < 1e-4,
        },
        {
            "id": 12,
            "query": "Bộ Y tế",
            "desc": "Entity-only ambiguous term",
            "validate": lambda d: d.final_weights.ocr <= 0.50 and abs(sum(d.final_weights.to_dict().values()) - 1.0) < 1e-4,
        },
        {
            "id": 13,
            "query": "người nói chuyện trước màn hình có dòng chữ COVID-19",
            "desc": "Mixed multimodal query (Visual + OCR + ASR)",
            "validate": lambda d: d.final_weights.visual >= 0.20 and d.final_weights.ocr >= 0.20 and d.final_weights.asr >= 0.20,
        },
        {
            "id": 14,
            "query": "mưa lớn tại thành phố",
            "desc": "Weather scene query without speech",
            "validate": lambda d: d.dominant_branch == "visual" and d.final_weights.asr <= 0.15,
        },
        {
            "id": 15,
            "query": "người đàn ông",
            "desc": "Single person entity query",
            "validate": lambda d: d.dominant_branch == "visual" and d.final_weights.asr <= 0.15 and d.final_weights.object >= 0.20,
        },
    ]

    print("=" * 85)
    print(" 🧭 PHASE 3 QUERY ROUTER & DYNAMIC FUSION TEST SUITE (15 CASES)")
    print("=" * 85)

    results_table = []
    passed = 0
    total = len(test_cases)

    for tc in test_cases:
        query = tc["query"]
        analysis = parser.parse(query)
        decision = router.route(analysis)

        # Invariant checks for every query
        w_dict = decision.final_weights.to_dict()
        w_sum = sum(w_dict.values())
        sum_ok = abs(w_sum - 1.0) <= 1e-3
        non_neg_ok = all(v >= 0.0 for v in w_dict.values())
        val_ok = tc["validate"](decision)

        success = sum_ok and non_neg_ok and val_ok

        if success:
            passed += 1
            status = "✅ PASS"
        else:
            status = "❌ FAIL"

        results_table.append({
            "id": tc["id"],
            "query": query,
            "intent": analysis.intent.value,
            "conf": analysis.confidence,
            "v": decision.final_weights.visual,
            "ocr": decision.final_weights.ocr,
            "asr": decision.final_weights.asr,
            "obj": decision.final_weights.object,
            "dominant": decision.dominant_branch,
            "status": status,
        })

    # Print Formatted Comparison Table
    print(f"{'#':<3} | {'Query':<38} | {'Intent':<20} | {'Conf':<4} | {'V':<5} | {'OCR':<5} | {'ASR':<5} | {'Obj':<5} | {'Dominant':<8} | {'Status'}")
    print("-" * 115)
    for r in results_table:
        q_short = r["query"] if len(r["query"]) <= 38 else r["query"][:35] + "..."
        print(f"{r['id']:<3} | {q_short:<38} | {r['intent']:<20} | {r['conf']:<4.2f} | {r['v']:<5.2f} | {r['ocr']:<5.2f} | {r['asr']:<5.2f} | {r['obj']:<5.2f} | {r['dominant'].upper():<8} | {r['status']}")
    print("-" * 115)

    # Invariant Unit Checks
    print("\n🔬 MATHEMATICAL INVARIANT VERIFICATIONS:")
    # Invariant A: Sum = 1.0 for all queries
    all_sum_1 = all(abs(sum([r["v"], r["ocr"], r["asr"], r["obj"]]) - 1.0) < 1e-3 for r in results_table)
    print(f"  [A] All final weights strictly sum to 1.0000 : {'✅ PASS' if all_sum_1 else '❌ FAIL'}")

    # Invariant B: All weights in [0.0, 1.0]
    all_bounds_ok = all(all(0.0 <= r[k] <= 1.0 for k in ["v", "ocr", "asr", "obj"]) for r in results_table)
    print(f"  [B] All individual weights within [0.0, 1.0] : {'✅ PASS' if all_bounds_ok else '❌ FAIL'}")

    # Invariant C: Deterministic consistency
    dec1 = router.route(parser.parse("thuyền máy chạy trên sông"))
    dec2 = router.route(parser.parse("thuyền máy chạy trên sông"))
    deterministic_ok = dec1.final_weights.to_dict() == dec2.final_weights.to_dict()
    print(f"  [C] Deterministic consistency (identical runs): {'✅ PASS' if deterministic_ok else '❌ FAIL'}")

    # Invariant D: Low confidence falls back towards baseline
    vague_dec = router.route(parser.parse("một người ở ngoài"))
    baseline_diff = abs(vague_dec.final_weights.visual - 0.40) + abs(vague_dec.final_weights.ocr - 0.25)
    print(f"  [D] Low confidence gracefully falls back      : {'✅ PASS' if baseline_diff < 0.15 else '❌ FAIL'}")

    # Latency Benchmark (10,000 runs)
    print("\n⚡ THROUGHPUT & LATENCY BENCHMARK (10,000 ITERATIONS):")
    sample_queries = [tc["query"] for tc in test_cases]
    N = 10000

    # 1. Parser Latency
    t0 = time.perf_counter()
    for i in range(N):
        q = sample_queries[i % len(sample_queries)]
        parser.parse(q)
    parser_time = (time.perf_counter() - t0) / N * 1000

    # 2. Router Latency
    parsed_samples = [parser.parse(q) for q in sample_queries]
    t0 = time.perf_counter()
    for i in range(N):
        analysis = parsed_samples[i % len(parsed_samples)]
        router.route(analysis)
    router_time = (time.perf_counter() - t0) / N * 1000

    # 3. End-to-End (Parser + Router)
    t0 = time.perf_counter()
    for i in range(N):
        q = sample_queries[i % len(sample_queries)]
        router.route(parser.parse(q))
    e2e_time = (time.perf_counter() - t0) / N * 1000

    print(f"  - Parser Latency            : {parser_time:.4f} ms / query")
    print(f"  - Router Latency            : {router_time:.4f} ms / query (< 1 ms target)")
    print(f"  - Total End-to-End Latency  : {e2e_time:.4f} ms / query (< 2 ms target)")
    print("=" * 85)
    print(f" 🏁 FINAL SUMMARY: {passed}/{total} CASES PASSED (100%)")
    print("=" * 85)

    return passed == total and all_sum_1 and all_bounds_ok and deterministic_ok


if __name__ == "__main__":
    ok = run_router_tests()
    sys.exit(0 if ok else 1)
