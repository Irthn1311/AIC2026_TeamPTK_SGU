"""Build Trial P1 Submission #3 without model inference or ground truth."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from triage_eg.trial_p1.submit3_m0_qa_bcf1 import package_submit3


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
    result = package_submit3(args.bcf1_bundle, args.r4_bundle, args.output_root, head=head)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
