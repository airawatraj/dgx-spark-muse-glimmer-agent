#!/usr/bin/env bash
# setup/download_model.sh
# Downloads Inferact/Muse-Glimmer-30B-NVFP4-W4A4 weights via huggingface-cli.
# Safe to run multiple times — skips download if model is already cached.
# Run this before docker/start.sh
set -euo pipefail

MODEL_ID="Inferact/Muse-Glimmer-30B-NVFP4-W4A4"

# huggingface-cli places weights under $HF_HOME/hub by default.
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
MODEL_CACHE_DIR="$HF_HOME/hub/models--Inferact--Muse-Glimmer-30B-NVFP4-W4A4"

echo "=== Inferact/Muse-Glimmer-30B-NVFP4-W4A4 Weight Download ==="

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN environment variable is not set."
  echo "       The model may be gated — set it using: export HF_TOKEN='your_token'"
  exit 1
fi

# ── Check Cache ──────────────────────────────────────────────────────────────
if [[ -d "$MODEL_CACHE_DIR/snapshots" ]]; then
  SNAP_COUNT=$(find "$MODEL_CACHE_DIR/snapshots" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
  if [[ "$SNAP_COUNT" -gt 0 ]]; then
    echo "✓ Model already present at $MODEL_CACHE_DIR"
    echo ""
    echo "Next: bash docker/start.sh"
    exit 0
  fi
fi

echo "Downloading $MODEL_ID..."
echo "TIP: If on an SSH session, consider running inside tmux to survive disconnects."
echo ""

# Use huggingface-cli if available, fall back to huggingface_hub Python module
if command -v huggingface-cli &>/dev/null; then
  huggingface-cli download "$MODEL_ID" \
    --local-dir "$MODEL_CACHE_DIR" \
    --token "$HF_TOKEN"
else
  echo "huggingface-cli not found — using Python huggingface_hub..."
  python3 -c "
from huggingface_hub import snapshot_download
import os
snapshot_download(
    repo_id='$MODEL_ID',
    local_dir='$MODEL_CACHE_DIR',
    token=os.environ.get('HF_TOKEN'),
)
"
fi

echo ""
echo "✓ Download complete."
echo "Next: bash docker/start.sh"
