"""Build the packaging-only Trial P1 SAFE_R4/TRUE_BCF1 hybrid."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from triage_eg.trial_p1.submit2_hybrid import package_submit2_hybrid


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bcf1-bundle", type=Path, required=True)
    parser.add_argument("--r4-bundle", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    result = package_submit2_hybrid(args.bcf1_bundle, args.r4_bundle, args.output_root, head=head)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
