# GPU G1 Kaggle audit

Run `notebooks/21_gpu_acceleration_parity.ipynb` with a T4 accelerator. The notebook
is bounded to four representative raw videos by default and writes
`/kaggle/working/triage_eg_gpu_g1_bundle.zip`.

## Required inputs

| Input | Default Kaggle mount | Resolved marker |
|---|---|---|
| Raw AIC dataset | `/kaggle/input/datasets/nadkli/dataset-aic` | bounded raw video files |
| Stage1 index | `/kaggle/input/datasets/irthn1311/triage-eg-stage1b-input-bundle` | `index/clip_vectors.f16.npy` |
| Stage1B verification | `/kaggle/input/datasets/irthn1311/triage-eg-stage1b-encoder-compatibility-reports` | `encoder/selected_encoder_contract.json` |
| Stage1E language freeze | `/kaggle/input/datasets/irthn1311/triage-eg-stage1e-language-path-freeze` | `language_path_contract.json` |
| Official OpenAI CLIP | `/kaggle/input/datasets/irthn1311/aic2026-openai-clip-vit-b32` | `checkpoint/ViT-B-32.pt` |
| Offline OPUS vi-en | `/kaggle/input/datasets/irthn1311/aic2026-opus-mt-vi-en` | `model/config.json` |

The prepared optional dataset defaults to
`/kaggle/input/datasets/irthn1311/aic2026-pynvvideocodec-wheel` and contains the
official NVIDIA `2.1.0` CPython 3.12 Linux x86-64 wheel. Nested roots are discovered
automatically; `AIC_PYNVVIDEOCODEC_WHEEL_ROOT` is only needed for a different slug.
If omitted or incompatible,
the notebook records `NVDEC_STATUS=UNAVAILABLE`, retains optimized OpenCV, and still
audits GPU CLIP and OPUS.

Internet is needed only if the notebook must clone the repository. For a fully offline
run, attach a repository snapshot and set `AIC_REPO_DIR`. Models are always loaded from
the attached offline assets; the notebook does not download models.

AUTO NVDEC promotion remains disabled in the checked-in GPU config. Only a real run
with 100% integer frame-identity parity and at least 1.5x useful workload speedup may
produce `DEFAULT_VIDEO_BACKEND=AUTO_NVDEC` in the audit decision. That evidence does
not mutate the checked-in policy automatically.
