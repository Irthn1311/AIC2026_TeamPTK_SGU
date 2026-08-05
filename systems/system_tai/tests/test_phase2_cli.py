from __future__ import annotations

from system_tai.kis.retrieve import build_parser


def test_cli_parser_accepts_exact_baseline_arguments() -> None:
    args = build_parser().parse_args(
        [
            "--manifest",
            "manifest.json",
            "--query-id",
            "q001",
            "--query",
            "a rainy street",
            "--output",
            "result.jsonl",
        ]
    )
    assert args.top_k == 100
    assert args.chunk_size == 4096
    assert args.device == "cpu"
    assert not args.temporal_suppression
    assert not args.allow_model_download
