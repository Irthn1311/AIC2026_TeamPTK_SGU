import json
import subprocess
import sys
from pathlib import Path


def test_demo_pipeline_creates_complete_artifact(tmp_path):
    project_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            "scripts/demo_pipeline.py",
            "--config",
            "configs/experiments/exp001_template.yaml",
            "--output-root",
            str(tmp_path),
        ],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Dummy encoder is not a semantic model" in result.stdout
    run_dirs = list((tmp_path / "demo_pipeline").iterdir())
    assert len(run_dirs) == 1
    artifact_dir = run_dirs[0]
    manifest = json.loads((artifact_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "COMPLETED"
    assert (artifact_dir / "used_config.yaml").is_file()
    results = (artifact_dir / "retrieval_results.jsonl").read_text(encoding="utf-8")
    assert results.strip()
