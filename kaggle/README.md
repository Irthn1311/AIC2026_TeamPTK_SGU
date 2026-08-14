# Kaggle bootstrap

`bootstrap.sh` clone repository và checkout exact commit trước khi cài editable package. Notebook
`00_kaggle_bootstrap.ipynb` làm cùng nhiệm vụ rồi gọi demo/test; business logic luôn ở `src/` và
`scripts/`.

Các path Kaggle nên đi qua env:

- `AIC_DATA_ROOT`
- `AIC_OUTPUT_ROOT`
- `AIC_AUDIT_OUTPUT_ROOT`
- `AIC_STAGE0_ROOT`
- `AIC_STAGE1_OUTPUT_ROOT`

Install deps:

```bash
python -m pip install -e "$REPO_DIR"
python -m pip install -r "$REPO_DIR/kaggle/requirements-kaggle.txt"
```

Warm up model/checkpoint cache:

```bash
python scripts/prepare_kaggle_assets.py
```

Pipeline Kaggle chỉ chạy offline preprocessing, không chạy FastAPI/React UI:

1. Keyframe V2 + Visual CLIP FAISS
2. YOLOE Hybrid V2 object detection + optional bbox preview visualizations
3. OCR V2 selected keyframes + OCR Temporal V3 tracking/index
4. Faster-Whisper ASR V3 + E5 index
5. Validate/package outputs

Chạy smoke 1 video trước:

```bash
python scripts/run_kaggle_preprocessing.py --smoke-video-count 1
```

Khi smoke ổn, chạy full:

```bash
python scripts/run_kaggle_preprocessing.py --full
```

Full Kaggle runs disable object preview images by default (`--object-visualization-limit 0`)
so `/kaggle/working` is reserved for parquet/index/package artifacts. Use a small positive
limit only when you need a preview sample; `-1` writes every bbox image and can fill the disk.

Để kiểm 20 output mỗi nhóm sau khi chạy xong:

```bash
python scripts/kaggle_output_manifest.py --limit 20
```

File tải về chính sau Step 5:

- `/kaggle/working/artifacts/kaggle_outputs_indices.zip`
- `/kaggle/working/artifacts/kaggle_outputs_keyframes.tar.gz`
- `/kaggle/working/artifacts/kaggle_package_validation.json`

Với private repository, đọc token từ Kaggle Secrets trong runtime, không in token, không commit
token và không ghi token vào notebook URL. Sửa data path template theo Dataset đã attach.

