#!/usr/bin/env bash
set -euo pipefail
P25="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$P25/external/deep-molecular-optimization"
MODEL_DIR="$P25/models/official_pretrained"
EXTRACT="$MODEL_DIR/extracted"
LOG="$P25/logs"
mkdir -p "$P25/external" "$MODEL_DIR" "$EXTRACT" "$LOG"

REPO_URL="https://github.com/MolecularAI/deep-molecular-optimization.git"
MODEL_URL="https://zenodo.org/records/5571965/files/train_transformer.zip?download=1"
ZIP="$MODEL_DIR/train_transformer.zip"

if [ ! -f "$REPO/README.md" ]; then
  git clone --depth 1 "$REPO_URL" "$REPO"
else
  echo "Official source already present; clone skipped."
fi

if [ -d "$REPO/.git" ]; then
  (cd "$REPO" && git rev-parse HEAD) \
    | tee "$LOG/official_repo_commit.txt"
else
  {
    echo "retrieval_method=official_GitHub_source_archive"
    echo "branch=main"
    if [ -f "$LOG/official_repo_archive_sha256.txt" ]; then
      cat "$LOG/official_repo_archive_sha256.txt"
    fi
  } | tee "$LOG/official_repo_commit.txt"
fi

if [ ! -s "$ZIP" ]; then
  if command -v wget >/dev/null 2>&1; then
    wget -c -O "$ZIP" "$MODEL_URL"
  else
    curl -L --continue-at - -o "$ZIP" "$MODEL_URL"
  fi
fi
if command -v sha256sum >/dev/null 2>&1; then
  sha256sum "$ZIP" | tee "$LOG/official_model_zip_sha256.txt"
else
  shasum -a 256 "$ZIP" | tee "$LOG/official_model_zip_sha256.txt"
fi

if ! find "$EXTRACT" -type f -name 'model_*.pt' -print -quit | grep -q .; then
  unzip -o "$ZIP" -d "$EXTRACT" | tee "$LOG/official_model_unzip.txt"
fi

echo "===== Official assets ready ====="
find "$EXTRACT" -type f \( -name 'model_*.pt' -o -name 'vocab.pkl' \) -exec ls -lh {} \;
