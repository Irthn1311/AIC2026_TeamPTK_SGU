#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <repository-url> <commit-sha>" >&2
  exit 2
fi

repo_url="$1"
commit_sha="$2"
work_dir="${AIC_REPO_DIR:-/kaggle/working/AIC2026_TeamPTK_SGU}"

git clone "$repo_url" "$work_dir"
git -C "$work_dir" checkout --detach "$commit_sha"
python -m pip install -e "$work_dir"
if [[ -f "$work_dir/kaggle/requirements-kaggle.txt" ]]; then
  python -m pip install -r "$work_dir/kaggle/requirements-kaggle.txt"
fi
python "$work_dir/scripts/prepare_kaggle_assets.py"
printf 'Checked out %s at %s\n' "$work_dir" "$(git -C "$work_dir" rev-parse HEAD)"

