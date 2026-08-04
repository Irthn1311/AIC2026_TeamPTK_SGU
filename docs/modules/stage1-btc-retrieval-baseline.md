# Stage 1 BTC Retrieval Baseline v0.1

Stage 1 consumes the completed Stage 0 manifests as its source of truth. It builds a
deterministic compact frame catalog and one contiguous float16 CLIP matrix, then runs
chunked exact cosine or dot-product search. It does not scan mapping/keyframe/Object
directories or extract frames.

Internal retrieval preserves every BTC ordinal, including duplicate `frame_idx`
rows. The KIS-compatible export alone stable-deduplicates `(video_id,
original_frame_idx)`, keeping the highest score and then the smallest global row.

Vector queries are always available after a successful index build. Text queries are
blocked until an explicit encoder contract is verified; dimension 512 alone is not
compatibility evidence. The implementation never auto-selects or downloads a CLIP
checkpoint. The optional `open_clip` adapter accepts only an existing local
`checkpoint_path`, rejects `hf-hub:` model names, constructs the model with
`pretrained=None`, and requires the packaged `open_clip_simple` tokenizer.

Index builds use a sibling staging tree. With `--overwrite`, the previous complete
output is swapped out only after vector copying, manifest generation, and
self-retrieval finish; a failed build leaves the previous output intact.

```bash
python scripts/build_stage1_index.py \
  --stage0-root /kaggle/working/triage_eg_stage0_audit \
  --dataset-root /kaggle/input/datasets/nadkli/dataset-aic \
  --output-root /kaggle/working/triage_eg_stage1_baseline \
  --metric cosine --search-chunk-rows 16384 --overwrite --strict-root
```

Vector search remains available without a text encoder:

```bash
python scripts/search_stage1.py \
  --stage1-root /kaggle/working/triage_eg_stage1_baseline \
  --query-vector /kaggle/working/query_vector.npy \
  --query-id vector_demo --top-k 100 --video-grouping max
```

Text search fails closed until its supplied contract passes the compatibility gate:

```bash
python scripts/search_stage1.py \
  --stage1-root /kaggle/working/triage_eg_stage1_baseline \
  --query-text "một người đang nấu ăn trong bếp" \
  --query-id text_demo --encoder-config /kaggle/working/encoder_contract.yaml

python scripts/benchmark_stage1.py \
  --stage1-root /kaggle/working/triage_eg_stage1_baseline \
  --random-queries 50 --self-queries 100 --top-k 100 --seed 2026
```

Notebook 06 creates `triage_eg_stage1_baseline_reports.zip` by default. This report
bundle excludes vector/catalog arrays; setting `AIC_ZIP_INDEX=1` creates the separate,
potentially large `triage_eg_stage1_index_bundle.zip`.

Self-retrieval verifies catalog/vector alignment, not semantic retrieval quality.
Latency benchmarks intentionally do not report Recall@K without ground truth.
