#!/usr/bin/env python3
"""Run Stage 1B image compatibility probe with text smoke disabled."""

from __future__ import annotations

import sys

from run_stage1b_encoder_compatibility import main

if __name__ == "__main__":
    if "--skip-text-smoke" not in sys.argv:
        sys.argv.append("--skip-text-smoke")
    raise SystemExit(main())
