# Official OpenAI CLIP ViT-B/32 offline assets

Stage 1B needs two assets that must already exist on a trusted machine:

1. An official OpenAI CLIP source checkout containing the clip package,
   tokenizer vocabulary, and LICENSE.
2. The official OpenAI ViT-B-32.pt checkpoint.

The repository does not download either asset and the checkpoint must never be
committed to GitHub.

## Build a portable bundle

Run this on the machine that already has both assets:

    python scripts/build_openai_clip_asset_bundle.py \
      --source-root /path/to/OpenAI-CLIP \
      --checkpoint /path/to/ViT-B-32.pt \
      --output-root /path/to/openai_clip_vit_b32_asset_bundle \
      --source-commit YOUR_SOURCE_COMMIT \
      --dependency-wheel /path/to/ftfy-6.3.1-py3-none-any.whl \
      --dependency-wheel /path/to/wcwidth-0.2.13-py2.py3-none-any.whl \
      --zip

Use --dry-run first to inspect the selected files. The tool copies only runtime
source files, LICENSE, tokenizer asset, and checkpoint; it excludes .git,
caches, and tests. It writes the computed checkpoint hash, source commit, asset
manifest, and deterministic file inventory. Existing output is protected unless
--overwrite is supplied. Pure-Python dependency wheels are copied under
`source/dependencies/` and imported directly at runtime; Notebook 07 does not
run pip or access the Internet.

## Upload and attach to Kaggle

Upload the resulting directory or ZIP as a private Kaggle Dataset. Attach it to
Notebook 07 so the mounted layout is:

    /kaggle/input/datasets/irthn1311/aic2026-openai-clip-vit-b32/
    ├── checkpoint/ViT-B-32.pt
    ├── source/openai_clip/clip/
    ├── source/dependencies/
    └── manifests/

Kaggle can expose the tokenizer member without its outer gzip layer as
`bpe_simple_vocab_16e6.txt`. Notebook 07 detects that exact case, copies the
official source into `/kaggle/working`, recreates the required deterministic
`.txt.gz`, and points the runtime adapter at the working copy. Kaggle Input stays
read-only and unchanged.

Notebook 07 derives source and checkpoint paths from
AIC_OPENAI_CLIP_ASSET_ROOT. Override paths when the Kaggle mount name differs:

    export AIC_OPENAI_CLIP_ASSET_ROOT=/kaggle/input/datasets/irthn1311/aic2026-openai-clip-vit-b32
    export AIC_OPENAI_CLIP_SOURCE_ROOT=$AIC_OPENAI_CLIP_ASSET_ROOT/source/openai_clip
    export AIC_OPENAI_CLIP_CHECKPOINT=$AIC_OPENAI_CLIP_ASSET_ROOT/checkpoint/ViT-B-32.pt
    export AIC_STAGE1B_DEVICE=auto
    export AIC_STAGE1B_BATCH_SIZE=16

Notebook 07 writes resolved runtime YAML under /kaggle/working; it does not
modify the repository template or Kaggle Input.

## Preflight

    python scripts/preflight_openai_clip_candidate.py \
      --asset-root /kaggle/input/datasets/irthn1311/aic2026-openai-clip-vit-b32 \
      --load-model \
      --strict

Preflight verifies source/package origin, required load and tokenize APIs,
tokenizer asset, checkpoint size/hash, manifest paths, source commit,
dependencies, and selected device. Model load still uses only the absolute local
checkpoint and blocks the package download helper.

## Run Stage 1B

Stage 1B requires the complete saved Stage 1A index, including the vector matrix,
norms, and compact-catalog arrays. The small
`triage_eg_stage1_baseline_reports.zip` is report-only and is not sufficient.
Attach the saved Notebook 06 output that contains
`triage_eg_stage1_baseline/index/clip_vectors.f16.npy`, or attach a Dataset that
contains the single `/kaggle/working/triage_eg_stage1b_input_bundle.zip` created
by Notebook 06. Notebook 07 discovers the ZIP and safely materializes its strict
allowlist under `/kaggle/working/triage_eg_stage1b_saved_input`. The ZIP contains
`stage1_summary.json`, provenance, the vector matrix, norms, and every
compact-catalog array required by Notebook 07. This reuses the locked index; it
does not rebuild Stage 1A.

    python scripts/run_stage1b_encoder_compatibility.py \
      --repo-root /kaggle/working/AIC2026_TeamPTK_SGU \
      --dataset-root /kaggle/input/datasets/nadkli/dataset-aic \
      --stage0-root /kaggle/input/datasets/irthn1311/triage-eg-stage0-audit-bundle \
      --stage1-root /kaggle/input/datasets/irthn1311/triage-eg-stage1b-input-bundle \
      --output-root /kaggle/working/triage_eg_stage1b_encoder_compatibility \
      --candidate-config configs/retrieval/stage1b_openai_clip_official_kaggle.yaml \
      --queries configs/retrieval/stage1b_smoke_queries.jsonl \
      --sample-size 50 --seed 2026 --strict-root --overwrite \
      --zip-path /kaggle/working/triage_eg_stage1b_encoder_compatibility_reports.zip

Do not use an unverified-encoder override for acceptance.

## Interpret the result

- VERIFIED: local candidate passed the locked empirical model-space gate; text
  smoke may run.
- UNVERIFIED: evidence is close or incomplete and cannot support enablement.
- REJECTED: valid output ran but dimension, cosine, or exact stored-vector-equivalence
  alignment failed. Literal global-row alignment is diagnostic only.
- BLOCKED: source, dependency, checkpoint, integrity, import, or load failed.

MODEL_SPACE_VERIFIED does not prove the original BTC implementation, retrieval
quality, or Vietnamese-query quality. OpenCLIP fallback remains disabled and
must be evaluated in a separate user-authorized run.

The same asset directory can be copied to a laptop and referenced with the three
asset environment variables above. No Internet access is required at runtime.
