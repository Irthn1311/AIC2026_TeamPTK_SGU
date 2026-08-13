#!/usr/bin/env bash
set -euo pipefail

echo "================================================================="
echo " 🚀 KAGGLE OFFLINE PREPROCESSING PIPELINE (AI CHALLENGE 2026)"
echo "================================================================="

# Set Kaggle project directory
WORKDIR="${AIC_REPO_DIR:-/kaggle/working/AIC2026_TeamPTK_SGU}"

if [ ! -d "$WORKDIR" ]; then
    WORKDIR="$(pwd)"
fi

cd "$WORKDIR"

# Execute Python Master Orchestrator
python kaggle/run_kaggle_master.py "$@"
