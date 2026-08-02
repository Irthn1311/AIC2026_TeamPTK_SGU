"""Audit a CSV video manifest from YAML configuration."""

from __future__ import annotations

import argparse
import json
import sys

from triage_eg.common.config import load_yaml_config, validate_required_keys
from triage_eg.data.audit import audit_video_records
from triage_eg.data.manifest import read_video_manifest_csv


def main() -> int:
    """Run metadata-only video auditing."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    try:
        config = load_yaml_config(args.config)
        validate_required_keys(config, ["data_root", "manifest"])
        records = read_video_manifest_csv(config["manifest"])
        report = audit_video_records(records, config["data_root"])
    except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report.__dict__, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
