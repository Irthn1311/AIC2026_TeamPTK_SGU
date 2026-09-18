# TRIAGE-EG — AI Challenge 2026

Team PTK / Saigon University workspace for **AI Challenge Ho Chi Minh City 2026**.

TRIAGE-EG is an experimental video-retrieval architecture designed around three task families:

- **Known-Item Search (KIS):** retrieve a target video/frame from a textual description
- **Question Answering (Q&A):** retrieve evidence and produce an answer
- **Temporal Retrieval / Event Alignment:** identify a video and align a sequence of relevant events

> **Status:** active competition project. The repository currently establishes reproducible contracts, data/frame mapping, baseline retrieval, evaluation, and experiment infrastructure. Deferred modules are clearly separated from implemented components.

## Target pipeline

```text
raw video
  -> data audit
  -> unified frame mapping
  -> frame bank
  -> feature extraction
  -> multimodal retrieval
  -> video ranking
  -> event graph
  -> temporal / semantic localization
  -> evidence verification
  -> ranked answers
```

Not every block above is implemented yet. The current repository intentionally distinguishes **working baselines** from **planned research modules**.

## Current implementation

The current baseline provides:

- common data / configuration contracts
- run manifests tied to exact Git commits and configs
- metadata audit and frame mapping
- baseline shot / center-frame selection
- deterministic dummy encoders for pipeline validation
- NumPy cosine retrieval for small-scale tests
- evaluation utilities
- unit / integration tests
- local → GitHub → Kaggle reproducibility workflow

## Repository structure

```text
configs/        experiment and module configuration
src/triage_eg/  Python package
scripts/        command-line and demo entry points
tests/          unit / integration tests and fixtures
docs/           architecture, contracts, ADRs, ownership
notebooks/      thin Kaggle bootstrap notebooks
kaggle/         Kaggle workflow documentation
```

Large datasets, videos, model checkpoints, extracted features, and indexes are intentionally excluded from Git.

## Quick start

Requires Python 3.11+.

```bash
python -m pip install -e ".[dev]"
ruff check .
pytest -q
python scripts/demo_pipeline.py --config configs/experiments/exp001_template.yaml
```

Example evaluation:

```bash
python scripts/evaluate.py \
  --task trake \
  --ground-truth tests/fixtures/sample_trake_ground_truth.json \
  --predictions tests/fixtures/sample_trake_predictions.json
```

Generated artifacts are written outside the tracked source tree and should remain linked to the exact commit/config that produced them.

## Reproducibility workflow

1. Develop and test on a feature branch.
2. Merge code/config/docs through GitHub pull requests.
3. Keep `main` runnable.
4. On Kaggle, clone an exact `COMMIT_SHA`.
5. Run experiments from repository scripts rather than notebook-only business logic.
6. Record the commit and configuration in each run manifest.

Secrets are read from Kaggle Secrets / environment variables and must never be committed or embedded in notebook URLs.

## Module status

| Module | Status |
| --- | --- |
| Common contracts / run manifest | Working template |
| Data audit / frame mapping | Baseline |
| Shot + center-frame selection | Baseline |
| Dummy feature encoder | Pipeline-validation template |
| NumPy cosine retrieval / evaluation | Baseline |
| Adaptive multi-frame | Experimental / incomplete |
| Event graph | Planned |
| Semantic localization | Planned |
| Agent layer | Planned |

A module is not labeled stable until it has been benchmarked and reviewed.

## Roadmap

Current progression:

```text
baseline retrieval
  -> team frame bank
  -> real feature extraction
  -> retrieval benchmark
  -> event graph
  -> semantic localization
  -> agent / evidence verification
```

## Notes

This repository is deliberately conservative about claims: dummy features validate software contracts, not AI quality. Competition results should only be reported from reproducible runs using the documented evaluation protocol.
