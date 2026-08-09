# Stage 1B — CLIP Encoder Compatibility Validation v0.1

Stage 1B identifies a reproducible text/image encoder contract that recreates the
stored BTC CLIP image-vector space. It reuses the completed Stage 0 manifests and
Stage 1A index; it does not rebuild either stage or scan the full corpus.

## Decision boundary

Candidates are disabled by default in
`configs/retrieval/stage1b_encoder_candidates.yaml`. A candidate runs only when it
is explicitly enabled, declares 512 output dimensions, has its dependency already
installed or supplied as an offline wheel in the asset bundle, and resolves to a
local checkpoint or model-asset directory. The code contains no model-download
path.

The configured thresholds are a `PROJECT_DEFINED_EMPIRICAL_GATE`, not a universal
CLIP standard. `VERIFIED` requires adequate samples, finite 512-dimensional output,
pairwise cosine and stored-row retrieval alignment above the configured thresholds,
and complete checkpoint, tokenizer, and preprocessing provenance. Dimension or
folder naming alone never verifies a candidate.

If equivalent implementations pass, candidate-to-candidate cosine is recorded and
the canonical adapter is chosen by the explicitly configured `runtime_priority`,
then stable candidate ID. This verifies a model space; it does not prove which
library originally produced BTC features.

## Bounded evidence and probe

Evidence collection examines a bounded set of small repository and dataset metadata
files. It skips keyframe, object, and video trees. The deterministic probe selects at
most 100 catalog rows across videos using early, middle, late, partition-spread, and
seeded-random positions. Each selected JPG is encoded once per enabled candidate and
compared with the corresponding Stage 1A vector by cosine and exact-search rank.

## Text-search gate

The existing Stage 1 encoder validation, exact search, compact catalog, ranking, and
query-output writer are reused. Text smoke queries run only for a `VERIFIED`
candidate. They validate execution, vector shape/finite/nonzero properties, mapping,
export, and latency; they are not a retrieval-quality benchmark. In particular,
model-space compatibility does not establish Vietnamese retrieval quality. No
translation, multilingual projection, or query expansion is added here.

## Official OpenAI CLIP candidate

The stage1b_openai_clip_official_kaggle.yaml config defines the first enabled
empirical hypothesis: official OpenAI CLIP ViT-B/32 with OpenAI weights. It is
not VERIFIED by default. Runtime must resolve an official source checkout and a
local ViT-B-32.pt checkpoint.

The adapter imports clip from the configured source root using an explicit module
spec and rejects a previously imported package from another origin. It passes the
absolute checkpoint path to clip.load, blocks the package download helper during
load, uses the preprocessing object returned by clip.load, and uses clip.tokenize
for text. Missing assets or integrity failures produce a BLOCKED report rather
than falling back to OpenCLIP.

Checkpoint SHA-256, optional declared hash, source commit, module origin, runtime
device, model dtype, preprocess provenance, and tokenizer behavior are recorded.
An empirical pass means MODEL_SPACE_VERIFIED; it does not prove that BTC used the
same original library implementation.

## Kaggle run

Provide Stage 0 and Stage 1A saved outputs plus any dependency/checkpoint as Kaggle
inputs, edit or generate an explicit candidate config containing the mounted local
checkpoint path, then run `notebooks/07_stage1b_encoder_compatibility.ipynb`.
Official OpenAI CLIP pure-Python wheels under `source/dependencies/` are imported
without pip. If Kaggle expands the tokenizer gzip into a `.txt` member, the notebook
recreates `.txt.gz` in `/kaggle/working` and leaves the mounted input unchanged.
The final cell creates
`/kaggle/working/triage_eg_stage1b_encoder_compatibility_reports.zip` containing
report artifacts only. If no enabled asset is available, a complete `BLOCKED`
report is the expected first-run result.

The equivalent CLI is:

```bash
python scripts/run_stage1b_encoder_compatibility.py \
  --repo-root /kaggle/working/AIC2026_TeamPTK_SGU \
  --dataset-root /kaggle/input/datasets/nadkli/dataset-aic \
  --stage0-root /kaggle/working/triage_eg_stage0_audit \
  --stage1-root /kaggle/working/triage_eg_stage1_baseline \
  --output-root /kaggle/working/triage_eg_stage1b_encoder_compatibility \
  --candidate-config configs/retrieval/stage1b_encoder_candidates.yaml \
  --queries configs/retrieval/stage1b_smoke_queries.jsonl \
  --sample-size 50 --seed 2026 --strict-root --overwrite \
  --zip-path /kaggle/working/triage_eg_stage1b_encoder_compatibility_reports.zip
```
