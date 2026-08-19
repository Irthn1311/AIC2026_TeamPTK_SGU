"""Isolated PP-OCR recovery planning and resumable JSONL checkpoints."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


def isolated_install_commands(python: Path) -> tuple[list[str], list[str]]:
    """Return the two permitted isolated repair attempts, never main-environment upgrades."""

    executable = str(python)
    paddle = [
        executable,
        "-m",
        "pip",
        "install",
        "--no-cache-dir",
        "paddlepaddle-gpu==3.2.0",
        "paddleocr==3.7.0",
    ]
    transformers_backend = [
        executable,
        "-m",
        "pip",
        "install",
        "--no-cache-dir",
        "transformers==4.57.6",
        "pillow",
    ]
    return paddle, transformers_backend


def run_install(command: list[str]) -> dict[str, Any]:
    process = subprocess.run(command, capture_output=True, text=True, check=False)
    return {
        "command": command,
        "returncode": process.returncode,
        "stdout_tail": process.stdout.splitlines()[-30:],
        "stderr_tail": process.stderr.splitlines()[-30:],
    }


class JsonlCheckpoint:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.processed = set()
        if self.path.is_file():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                self.processed.add((row["video_id"], int(row["actual_frame_id"])))

    def append(self, rows: list[dict[str, Any]]) -> None:
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                key = (str(row["video_id"]), int(row["actual_frame_id"]))
                if key not in self.processed:
                    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                    self.processed.add(key)
