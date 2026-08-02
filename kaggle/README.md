# Kaggle bootstrap

`bootstrap.sh` clone repository và checkout exact commit trước khi cài editable package. Notebook
`00_kaggle_bootstrap.ipynb` làm cùng nhiệm vụ rồi gọi demo/test; business logic luôn ở `src/` và
`scripts/`.

Với private repository, đọc token từ Kaggle Secrets trong runtime, không in token, không commit
token và không ghi token vào notebook URL. Sửa data path template theo Dataset đã attach.

