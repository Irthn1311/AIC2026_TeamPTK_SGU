# system_tai

Isolated foundation for Nguyễn Tài's independent AIC 2026 system.

Phase 1 implements trustworthy BTC input auditing. Phase 1.5C adds nested Kaggle input
discovery, numeric diagnostics, margin-aware raw-frame calibration, and multi-candidate
image-encoder identification. Phase 2 adds the minimal exact KIS baseline:

mapping CSV + BTC CLIP NPY
→ validated, memory-mapped feature registry
→ official OpenAI CLIP ViT-B/32 text encoder
→ exact chunked NumPy cosine retrieval
→ deterministic Top-100
→ proposed UTF-8 JSONL checkpoint
→ local validator.

It does not implement FAISS, advanced retrieval, Q&A, TRAKE, an API server, or a
frontend. Synthetic tests prove code mechanics only. Text-retrieval quality requires a
separate Kaggle smoke test and manual evidence review.

## Install

From the shared repository root:

```bash
python -m pip install -e "systems/system_tai[dev]"
```

## Verify

```bash
pytest systems/system_tai/tests -v
ruff check systems/system_tai
```

## Audit KIS inputs

```bash
python systems/system_tai/scripts/audit_kis_inputs.py \
  --video-catalog <catalog.csv> \
  --video-id <BTC_VIDEO_ID> \
  --mapping-csv <mapping.csv> \
  --clip-npy <features.npy> \
  --expected-dimension <dimension>
```

Add `--strict-video-path-check` when catalog paths must exist locally. The audit command
returns non-zero for invalid input.

UTF-8 JSONL remains a proposed team checkpoint format. Phase 2 implements it behind an
exporter adapter; team acceptance may still replace the schema. Official BTC submission
is a separate unresolved boundary.

## Kaggle Phase 1.5C

Attach the private Dataset_AIC2026 and clone the repository to
`/kaggle/working/AI_Challenge_HCM`. Do not hard-code or assume the runtime dataset slug.
Open and run `notebooks/phase_1_5_kaggle.ipynb` from top to bottom.

The discovery CLI scans bounded nested layouts beneath `/kaggle/input`, so it supports
`/kaggle/input/datasets/<owner-or-runtime-id>/<dataset-root>` without hard-coding a
dataset slug. It can also be run directly:

```bash
python systems/system_tai/scripts/discover_kaggle_inputs.py \
  --input-root /kaggle/input \
  --video-id L21_V001 \
  --output /kaggle/working/system_tai_outputs/calibration/discovery_L21_V001.json
```

Audit Decimal-exact, binary-float-truncation, and Decimal-nearest numeric models:

```bash
python systems/system_tai/scripts/audit_mapping_rounding.py \
  --mapping-csv <discovered-map-keyframes.csv> \
  --video-id L21_V001 \
  --output /kaggle/working/system_tai_outputs/calibration/mapping_rounding_L21_V001.json
```

The notebook repeats discovery, input audit, rounding audit, and visual calibration for
`L21_V001`, `L21_V002`, and `L22_V001`.

For multi-video calibration, create a JSON/YAML batch manifest from discovery reports:

```bash
python systems/system_tai/scripts/calibrate_frame_mapping.py \
  --batch-manifest /kaggle/working/system_tai_outputs/calibration/frame_calibration_batch_manifest.json \
  --output /kaggle/working/system_tai_outputs/calibration/frame_mapping_calibration.json
```

Frame calibration preserves every CSV `frame_idx` exactly. Numeric-generation-rule
identification is diagnostic and does not affect mapping validity. It separately reports
`keyframe_visual_frame_id = decimal_round_half_up(pts_time * fps)` and the visual
offset. An explained `+1` JPEG alignment never changes `actual_frame_id` or shared
`frame_id`.

Verified `L21_V001` evidence covers all 307 mapping rows: Decimal floor matches 303;
the other four have `frame_idx - Decimal floor = -1`; and Decimal-nearest offset is `0`
for 233 rows and `+1` for 74. Only 15 rows have been visually decoded. All 15 visual
best offsets match the Decimal-nearest prediction, and random/sequential decoding agrees
for all 15. Binary-float truncation as the mapping generation rule and nearest timestamp
alignment as the JPEG extraction rule remain inferred.

All three input/mapping/decoder gates pass for `L21_V001`, `L21_V002`, and `L22_V001`,
and every binary-float-truncation ratio is `1.0`. Of 45 visual samples, 42 are decisive
and match the Decimal-nearest prediction. The remaining three margins (`0.000007`,
`0.000014`, and `0.000017`) are below `superiority_margin = 0.0001`, so they are
ambiguous ties. There is no contradictory decisive sample.

Optional CLIP candidates can be compared without implementing text retrieval only after
all three videos pass mapping and feature-row gates:

```bash
python systems/system_tai/scripts/identify_btc_clip_pipeline.py \
  --batch-manifest /kaggle/working/system_tai_outputs/calibration/clip_identification_batch_manifest.json \
  --minimum-identification-videos 3 \
  --backend openai_clip \
  --backend open_clip_vit_b32_openai \
  --backend open_clip_vit_b32_quickgelu_openai \
  --backend huggingface_clip \
  --allow-model-download \
  --output /kaggle/working/system_tai_outputs/calibration/clip_pipeline_identification_phase_1_5c.json
```

The official OpenAI adapter validates `clip.load`, `clip.available_models`, and
`ViT-B/32` without private `_MODELS`. The Hugging Face adapter supports direct Tensor
and Transformers 5 pooled ModelOutput results. OpenCLIP `ViT-B-32` and
`ViT-B-32-quickgelu`, both with pretrained `openai`, remain separate candidates.

The completed full-corpus three-video run covers 867 rows: 307 for `L21_V001`, 262 for
`L21_V002`, and 298 for `L22_V001`. Official OpenAI CLIP `ViT-B/32` and OpenCLIP
`ViT-B-32-quickgelu` with `pretrained=openai` both achieve mean cosine approximately
`0.999162`, minimum p05 cosine `0.997200`, self-match Top-1 `1.0`, and mean self-match
rank `1.0`. They are numerically equivalent within approximately `1e-10` on this
three-video audit. Official OpenAI CLIP is canonical; standard OpenCLIP
`ViT-B-32/openai` without QuickGELU is not a compatibility backend.

This identifies compatible implementations for the audited corpus, not BTC-official
preprocessing and not dataset-wide compatibility.

Unavailable packages or weights are reported as `SKIPPED`. Add
`--allow-model-download` only when Kaggle internet policy permits it. All generated
reports must stay under `/kaggle/working/system_tai_outputs/calibration/`; source dataset
artifacts remain under `/kaggle/input`.

Verified three-video evidence and its limits are recorded in
`docs/KAGGLE_PHASE_1_5_REPORT.md`. It is compatibility calibration evidence, not an
official performance result or dataset-wide confirmation.

TRIAGE-EG is reference material only and must remain untouched.

## Phase 2 exact KIS baseline

The smoke notebook defaults to `DEVICE="auto"`: it uses CUDA when available and CPU
otherwise. `DEVICE="cuda"` fails clearly when CUDA is unavailable, while
`DEVICE="cpu"` is always supported. Only the canonical OpenAI CLIP text encoder uses
the selected device. Exact chunked NumPy cosine retrieval remains on CPU as the
correctness baseline. GPU selection changes latency, not retrieval semantics or ranking
quality, and the notebook does not attempt untested multi-GPU execution.

Create an explicit feature manifest whose paths point to attached Kaggle inputs:

```json
{"videos":[{"video_id":"L21_V001","mapping_csv_path":"/kaggle/input/.../L21_V001.csv","clip_npy_path":"/kaggle/input/.../L21_V001.npy"}]}
```

Run one query with cached official OpenAI CLIP weights:

```bash
python -m pip install git+https://github.com/openai/CLIP.git
```

Install/download only when Kaggle internet policy permits it. By default the CLI will
fail clearly rather than download missing weights; pass `--allow-model-download`
explicitly when intended.

```bash
python -m system_tai.kis.retrieve \
  --manifest /kaggle/working/system_tai_outputs/kis_smoke/feature_manifest.json \
  --query-id q001 \
  --query "a person riding a motorcycle in heavy rain" \
  --top-k 100 \
  --output /kaggle/working/system_tai_outputs/kis_smoke/q001.jsonl \
  --device cuda \
  --chunk-size 4096
```

The default exporter emits only `query_id`, `rank`, `video_id`, and `frame_id`.
`frame_id` is copied exactly from mapping CSV `frame_idx`. Temporal suppression is
optional and disabled by default. Use `notebooks/phase_2_kis_smoke.ipynb` for the clean
three-video diagnostic smoke path. Optional display diversity uses the existing
post-ranking suppression only for manual Top-10 visualization; canonical Top-100 JSONL
remains unsuppressed. Official BTC submission output remains a separate unresolved
boundary.

### Real semantic smoke findings

The executed Kaggle smoke passed the technical pipeline: exact retrieval completed,
500 canonical JSONL records were written, and validation passed with zero errors. This
is not a semantic-quality pass.

- Direct Vietnamese semantic retrieval: **FAIL**.
- English translations improved retrieval, but results were mixed.
- `vi_03_en` was semantically strong.
- `vi_01_en` and `vi_02_en` were only partially relevant.
- Limited three-video corpus coverage and ambiguity between an action/state description
  and a single still frame remain material limitations.

The notebook retains the Vietnamese inputs as negative/diagnostic evidence and pairs
them with explicit English translations. Translation is not automatically considered
successful and does not alter the canonical exact ranking policy.

## Phase 2.5 ground-truth KIS benchmark

Phase 2 technical retrieval is complete, but semantic quality has not passed. Phase 2.5
adds a reproducible evaluation harness for manually verified positive frame labels. It
does not create labels automatically, change the OpenAI CLIP ViT-B/32 baseline, or use
display-only temporal suppression in benchmark scores.

The human-editable template is `config/kis_benchmark.example.yaml`. Every checked-in
example is deliberately `draft` with no fabricated positives. A reviewer must inspect
the audited videos and add `(video_id, frame_id)` labels, where `frame_id` is copied
exactly from mapping CSV `frame_idx`, before changing a query to `verified`.

Validate a benchmark without loading CLIP or writing reports:

```bash
python -m system_tai.kis.benchmark \
  --manifest /kaggle/working/system_tai_outputs/kis_benchmark/feature_manifest.json \
  --benchmark /path/to/human_verified_benchmark.yaml \
  --validation-only \
  --include-draft-validation \
  --fail-on-invalid
```

Evaluate verified queries with the canonical unsuppressed exact baseline:

```bash
python -m system_tai.kis.benchmark \
  --manifest /kaggle/working/system_tai_outputs/kis_benchmark/feature_manifest.json \
  --benchmark /path/to/human_verified_benchmark.yaml \
  --output-directory /kaggle/working/system_tai_outputs/kis_benchmark \
  --device auto \
  --top-k 1 5 20 50 100 \
  --fail-on-invalid
```

The command writes `kis_benchmark_report.json`, `kis_benchmark_summary.csv`, and
`kis_benchmark_report.md`. The default destination is outside the cloned repository.
Kaggle working storage is ephemeral, so reports that must persist must be downloaded or
versioned externally. `notebooks/phase_2_5_kis_benchmark.ipynb` performs the same
bounded three-video workflow and exposes a draft-only annotation helper. The helper
retains official `frame_id`, score, diagnostic keyframe order, and an external image
path, but never marks a candidate relevant.

For one scored query, `Recall@K` is binary: `1.0` when at least one exact ground-truth
`(video_id, frame_id)` pair occurs in canonical unsuppressed Top-K, otherwise `0.0`.
Aggregate Recall@K is the arithmetic mean over valid verified queries only. Multi-label
coverage is reported separately as `ground_truth_coverage_at_k`; relevant-video
coverage is the fraction of declared relevant video IDs represented in Top-K. MRR is
the arithmetic mean of per-query reciprocal rank, with a miss contributing zero.

Drafts are excluded from scoring by default, invalid benchmarks are rejected, and a
draft-only benchmark returns the explicit `no_verified_queries` state without loading
the model or fabricating zero-valued quality metrics. Successful reports include
evaluated, excluded-draft, and invalid-query counts.

The draft-only example still exercises the explicit `no_verified_queries` state. A
separate pilot fixture, `config/kis_benchmark.pilot_three_groups.yaml`, contains three
semantic groups, nine human-verified comparable variants, and six drafts. It uses only
the three audited videos. Its positives were selected after retrieval inspection, so
retrieval-selection bias applies and no official or dataset-wide semantic-quality claim
is supported.

Observed canonical unsuppressed ranks in that pilot were:

- city pedestrians: Vietnamese missed Top-100, translation rank 1, expansion rank 3;
- conference attendees: Vietnamese missed Top-100, translation rank 14, expansion rank
  14;
- landslide warning sign: Vietnamese rank 1, translation rank 4, expansion rank 6.

These observations do not settle a multilingual policy. Gate B remains incomplete.

## Phase 2.6 opt-in Weighted RRF pilot

Phase 2.6 adds explicit immutable query variants and deterministic Weighted Reciprocal
Rank Fusion as a separate measurement path. Each variant is retrieved independently by
the unchanged canonical `ExactNumpyRetriever`, with no temporal suppression. For a
candidate present in one or more branches:

```text
fusion_score = sum(weight / (rrf_constant + one_based_rank))
```

Candidate identity is exactly `(video_id, frame_id)`. Raw cosine values, CLIP rows,
keyframe order, per-variant ranks, and fusion provenance remain internal diagnostics.
The default checkpoint exporter still emits only `query_id`, `rank`, `video_id`, and
`frame_id`. Weighted RRF is opt-in and does not change the Phase 2 single-query CLI,
Phase 2.5 evaluator, exact-retrieval tie rules, or canonical JSONL output.

Validate the pilot without loading model weights:

```bash
python -m system_tai.kis.benchmark_fusion \
  --manifest /kaggle/working/system_tai_outputs/kis_benchmark/feature_manifest.json \
  --benchmark systems/system_tai/config/kis_benchmark.pilot_three_groups.yaml \
  --validation-only
```

Run the real Kaggle fusion measurement:

```bash
python -m system_tai.kis.benchmark_fusion \
  --manifest /kaggle/working/system_tai_outputs/kis_benchmark/feature_manifest.json \
  --benchmark systems/system_tai/config/kis_benchmark.pilot_three_groups.yaml \
  --output-directory /kaggle/working/system_tai_outputs/kis_fusion_pilot \
  --device auto \
  --top-k 1 5 20 50 100 \
  --top-k-per-variant 100 \
  --rrf-constant 60 \
  --fail-on-invalid
```

Model downloads are disabled unless `--allow-model-download` is supplied. Reports are
written outside the repository as `kis_fusion_pilot_report.json`,
`kis_fusion_pilot_summary.csv`, and `kis_fusion_pilot_report.md`. The private Kaggle
dataset and CLIP weights are unavailable locally, so this implementation phase cannot
claim real fused metrics.

## Phase 3 contest-ready Textual KIS CLI MVP

`system_tai.kis.contest` is the end-to-end delivery entry point for Textual KIS. It
discovers complete mapping/CLIP/keyframe artifact sets with bounded family scans, builds
or reuses a fingerprinted feature manifest, loads the registry and OpenAI CLIP encoder
once, retrieves each explicit query variant with exact chunked NumPy cosine, applies
opt-in Weighted RRF, exports deterministic results, and validates the core checkpoint.

Single query:

```bash
python -m system_tai.kis.contest \
  --input-root /kaggle/input \
  --query-id Q001 \
  --query-vi "một người đi xe máy trong mưa lớn" \
  --query-en "a person riding a motorcycle in heavy rain" \
  --query-en-expansion "a motorcyclist on a wet road during heavy rainfall" \
  --output-directory /kaggle/working/system_tai_runs/Q001 \
  --device auto \
  --top-k-per-variant 100 \
  --output-top-k 100 \
  --rrf-constant 60
```

Batch input uses the safe UTF-8 YAML/JSON schema illustrated by
`config/contest_queries.example.yaml`:

```bash
python -m system_tai.kis.contest \
  --input-root /kaggle/input \
  --queries systems/system_tai/config/contest_queries.example.yaml \
  --output-directory /kaggle/working/system_tai_runs/batch_01 \
  --device auto \
  --top-k-per-variant 100 \
  --output-top-k 100 \
  --rrf-constant 60 \
  --continue-on-query-error
```

Reuse a previously validated manifest with `--reuse-manifest <feature_manifest.json>`.
No source video, keyframe, mapping, or NPY is copied. The CLI writes only derived run
artifacts to the selected output directory:

- `feature_manifest.json`;
- `top100.jsonl`, the proposed shared core containing only `query_id`, `rank`,
  `video_id`, and official `frame_id`;
- `top100.csv`, internal convenience output and **not** official BTC format;
- `candidates.json` and `candidate_inspection.md` with diagnostic provenance;
- optional `candidate_contact_sheet.jpg`, a derived low-resolution inspection image;
- `validation_report.json`, `run_manifest.json`, `timings.json`, and `run_summary.md`;
- isolated per-query JSONL/CSV/inspection files under `queries/`.

The registry and model load once per batch. `--fail-fast` is the default behavior;
`--continue-on-query-error` records failures while continuing other queries. Failed
queries receive no fabricated result or metric, and any failure or invalid combined
checkpoint returns a non-zero exit code. Model weights are never downloaded unless
`--allow-model-download` is explicitly provided.

Phase 3.1 adds explicit inspection modes without changing retrieval or checkpoint
semantics:

- `--inspection-mode none` writes lightweight candidate JSON/Markdown without scanning
  keyframe directories;
- `--inspection-mode top-n` is the backward-compatible default and resolves thumbnails
  only through `--inspection-top-n` (default 50);
- `--inspection-mode all` resolves thumbnails for every candidate and is opt-in.

`--contact-sheet` requires `top-n` or `all`. The fast automated-contest path is:

```bash
python -m system_tai.kis.contest \
  --reuse-manifest /kaggle/working/system_tai_runs/previous/feature_manifest.json \
  --queries systems/system_tai/config/contest_queries.example.yaml \
  --output-directory /kaggle/working/system_tai_runs/fast_retry \
  --device auto \
  --top-k-per-variant 100 \
  --output-top-k 100 \
  --rrf-constant 60 \
  --fast-contest-mode
```

Fast mode is exactly inspection `none` with contact-sheet generation disabled. It does
not change encoding, exact cosine retrieval, Weighted RRF, Top-100 ordering, core
JSONL/CSV, or validation. Candidate inspection uses a lazy Path-only thumbnail index
that scans each keyframe directory at most once per run; no decoded image is cached.

Exact NumPy remains the correctness backend. Phase 3 records discovery, manifest,
registry, model, per-variant encoding/retrieval, fusion, export, validation, per-query,
and total-batch timings. The real full-corpus technical run passed over 873 videos,
177,321 feature rows, five queries, 15 variants, and 500 validated records. Exact NumPy
took about 1.1–1.3 seconds per variant, while pre-Phase-3.1 export/inspection took about
185.8 seconds and was the clear bottleneck. FAISS is therefore not needed for this
milestone. These timings are operational evidence, not official BTC performance or
semantic-quality proof. Q&A, TRAKE, UI/API work, OCR, ASR, VLM, Agent, GNN, and
production frontend remain deferred. Official BTC export format is still unresolved,
and all generated run artifacts must stay outside Git.
## Phase 4 opt-in raw-video exact-frame refinement

Phase 4 is a separate command over completed Phase 3 artifacts. It does not rerun
retrieval and does not change Phase 3 `top100.jsonl`, Exact NumPy retrieval, Weighted
RRF, or `--fast-contest-mode`:

```bash
python -m system_tai.kis.refine \
  --run-directory /kaggle/working/system_tai_runs/phase3_1_fast_city_rebuild \
  --output-directory /kaggle/working/system_tai_runs/phase4_refined_city \
  --top-candidates-to-refine 20 \
  --window-before-seconds 5 \
  --window-after-seconds 5 \
  --coarse-stride-frames 15 \
  --coarse-top-n 3 \
  --fine-radius-frames 30 \
  --fine-stride-frames 1 \
  --image-batch-size 32 \
  --max-decoded-frames-per-candidate 500 \
  --output-top-k 100 \
  --missing-raw-video-policy keep-original \
  --candidate-failure-policy keep-original \
  --device auto \
  --allow-model-download
```

Raw video is the coordinate source of truth. Candidate, decoded, and refined frame IDs
are absolute original-video positions; a local decode-list index is never a shared
`frame_id`. Refinement uses bounded sequential decoding, one shared official OpenAI
CLIP ViT-B/32 model, independent per-variant cosine rankings, and local Weighted RRF.
It replaces frames while preserving Phase 3 rank order and never mixes retrieval and
local-refinement scores.

Missing-video and decode failures support explicit `keep-original`, `skip-candidate`,
and `fail-query` policies. The command writes a core JSONL, internal CSV, bounded
candidate/trace JSON, timings, validation, run manifest, summary, and per-query files.
The derived contact sheet is optional and off by default. Generated outputs stay
outside Git. Synthetic tests prove mechanics only; private Kaggle acceptance and
semantic review remain required. Official BTC export is unresolved. UI, Q&A, TRAKE,
FAISS, OCR/ASR, VLM, Agent, and GNN remain deferred.

## Phase 4.1 one-pass discovery and portable manifests

The accepted private-corpus preflight resolves 873/873 raw videos and 177,321 feature
rows with no copied source artifacts. Before Phase 4.1, a fresh manifest build took
about 579.27 seconds because mapping, CLIP, video, and keyframe families were traversed
separately and every discovered keyframe directory was scanned again for image counts.

Discovery now walks each bounded artifact-family root at most once. The keyframe walk
identifies video directories and counts `.jpg`, `.jpeg`, `.png`, and `.webp` files in
the same pass without reading or decoding them. Exact NumPy retrieval was already about
1.1–1.3 seconds per variant and is not this startup bottleneck; FAISS remains deferred.

Build a strict portable schema-v2 manifest:

```bash
python -m system_tai.kis.build_manifest \
  --input-root /kaggle/input \
  --output /kaggle/working/feature_manifest.json \
  --portable \
  --discovery-validation strict
```

`strict` validates mapping columns/counts, memory-mapped CLIP shape/dimension,
mapping/feature agreement, one-pass keyframe statistics, and raw-video ambiguity.
`fast` retains mapping columns/counts, unique artifact paths, row-count/shape/dimension
agreement, keyframe presence, and raw-video ambiguity while consuming the one-pass
family statistics for a trusted BTC layout. It is not a skip-validation mode.

Portable schema v2 stores POSIX artifact paths relative to the resolved dataset root
and a shallow SHA-256 identity over relative artifact metadata. On reuse, `--input-root`
resolves the current Kaggle mount, rebases paths, checks source existence/file sizes,
and rejects identity mismatch. Existing absolute schema-v1 manifests remain supported.

Use a portable manifest supplied as a persistent Kaggle input:

```bash
python -m system_tai.kis.contest \
  --input-root /kaggle/input \
  --reuse-manifest /kaggle/input/system-tai-manifest/feature_manifest.json \
  --queries queries.yaml \
  --output-directory /kaggle/working/run \
  --fast-contest-mode
```

Or use cache-or-build behavior:

```bash
python -m system_tai.kis.contest \
  --input-root /kaggle/input \
  --manifest-cache /kaggle/working/cache/feature_manifest.json \
  --queries queries.yaml \
  --output-directory /kaggle/working/run \
  --fast-contest-mode
```

A valid cache bypasses full discovery. A missing cache is built with strict validation.
An invalid cache fails unless `--rebuild-invalid-manifest-cache` is explicit. Kaggle
`/kaggle/working` is ephemeral: retain the portable manifest as notebook output, a
private lightweight Kaggle Dataset, or an uploaded input artifact. Generated manifests
and runs must not be committed. Phase 4 refinement and frame semantics are unchanged;
these startup mechanics do not prove semantic quality or official BTC performance.

## Phase 4.2 long-lived contest operational session

Phase 4.2 introduces a fail-closed, single-process JSON-line IPC session for unified contest operations.

- **Long-Lived Process:** Runs a continuous `OperationalKISRuntime` processing `health`, `query`, and `shutdown` JSON lines via `stdin` and `stdout`.
- **Resource Efficiency:** The feature registry and official OpenAI CLIP ViT-B/32 model are initialized exactly once per session. Text and image refinement encodings share the identical loaded model instance.
- **Workflow:** Handles retrieval-only requests and optional refine Top-N seamlessly. Malformed requests isolate into explicit JSON error responses without crashing the session (`--continue-on-request-error`).
- **Artifacts:** Artifacts are strictly isolated per-request in unique digest-based directories.

### Example Session Execution

```bash
python -m system_tai.kis.session \
  --input-root /kaggle/input \
  --reuse-manifest /kaggle/input/system-tai-manifest/feature_manifest.json \
  --output-root /kaggle/working/session_outputs \
  --device auto \
  --allow-model-download \
  --continue-on-request-error
```

Send JSON lines to `stdin`:
```json
{"type": "health", "request_id": "req-1"}
{"type": "query", "request_id": "req-2", "query_id": "Q1", "query_vi": "biển báo sạt lở đất", "refine_top_n": 3}
{"type": "shutdown", "request_id": "req-3"}
```

Phase 4.2 is locally validated. **Private Kaggle acceptance is pending**. Semantic quality remains unproven. Official BTC submission format is unresolved. Q&A, TRAKE, and UI remain deferred.
