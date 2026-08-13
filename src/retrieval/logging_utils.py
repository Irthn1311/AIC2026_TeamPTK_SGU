from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path


LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def setup_logger(name: str, log_file: str | Path | None = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    if logger.handlers:
        return logger

    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
    logger.addHandler(console)

    if log_file:
        path = Path(log_file)
        ensure_parent(path)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
        logger.addHandler(file_handler)

    return logger


def stage_summary(stage: str, status: str, input_path: str = "", processed: int = 0, skipped: int = 0, warnings: int = 0, errors: int = 0, output: str = "", elapsed: float | None = None) -> str:
    elapsed_txt = f"{elapsed:.2f}s" if elapsed is not None else ""
    return (
        "================ STAGE SUMMARY ================\n"
        f"Stage: {stage}\n"
        f"Status: {status}\n"
        f"Input: {input_path}\n"
        f"Processed: {processed}\n"
        f"Skipped: {skipped}\n"
        f"Warnings: {warnings}\n"
        f"Errors: {errors}\n"
        f"Output: {output}\n"
        f"Elapsed: {elapsed_txt}\n"
        "==============================================="
    )


def timestamp_token() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

