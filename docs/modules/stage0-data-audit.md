# Stage 0 BTC Data Audit v0.1

Stage 0 validates raw video and the locked BTC mapping, keyframe, CLIP, Object, and
metadata assets. It produces canonical BTC frame records using CSV `frame_idx` as the
authoritative original-frame coordinate. It does not extract frames or run retrieval.

Sample mode is the safe default. Full mode uses deterministic sequential processing
and one checkpoint per completed video. `--resume` only accepts checkpoints whose
audit version, config fingerprint, dataset root, and source-size fingerprint match.

```bash
python scripts/run_stage0_data_audit.py \
  --dataset-root /kaggle/input/datasets/nadkli/dataset-aic \
  --output-root /kaggle/working/triage_eg_stage0_audit \
  --mode sample --sample-size 10 \
  --clip-validation full --object-validation full \
  --seed 2026 --strict-root --overwrite
```

The BTC baseline and raw-video gates are reported separately. A missing `ffprobe`
fails only the raw-video gate. Bounding-box order and CLIP model compatibility remain
unknown contracts.
