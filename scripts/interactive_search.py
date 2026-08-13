"""
Interactive Multimodal Search CLI
=================================
Allows live interactive querying across all 4 multimodal indices.
Usage:
    python scripts/interactive_search.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from scripts.search_multimodal_4branch import multimodal_4branch_search


def main():
    print("=" * 70)
    print(" 🚀 MULTIMODAL SEARCH ENGINE CLI (AI Challenge 2026)")
    print(" Gõ câu tìm kiếm bất kỳ (hoặc gõ 'exit' / 'q' để thoát)")
    print("=" * 70)

    while True:
        try:
            query = input("\n🔍 Nhập câu tìm kiếm: ").strip()
            if not query:
                continue
            if query.lower() in {"exit", "quit", "q"}:
                print("Tạm biệt!")
                break

            multimodal_4branch_search(query, top_k=5)
        except KeyboardInterrupt:
            print("\nĐã dừng chương trình.")
            break
        except Exception as exc:
            print(f"Lỗi: {exc}")


if __name__ == "__main__":
    main()
