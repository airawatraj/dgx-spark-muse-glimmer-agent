#!/usr/bin/env bash
# setup/download_model.sh
# Downloads Inferact/Muse-Glimmer-30B-NVFP4-W4A4 weights and the
# DFlash speculative drafter (meta-models/Muse-Glimmer-30B-assistant) via huggingface-cli / uv.
# Safe to run multiple times (idempotent) — skips download if models are already cached.
# Run this before docker/start.sh
set -euo pipefail

MODEL_ID="${MODEL_ID:-Inferact/Muse-Glimmer-30B-NVFP4-W4A4}"
DFLASH_MODEL_ID="${DFLASH_MODEL_ID:-meta-models/Muse-Glimmer-30B-assistant}"

# huggingface-cli places weights under $HF_HOME/hub by default.
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
MODEL_CACHE_DIR="$HF_HOME/hub/models--${MODEL_ID//\//--}"
DFLASH_CACHE_DIR="$HF_HOME/hub/models--${DFLASH_MODEL_ID//\//--}"

echo "=== Inferact/Muse-Glimmer-30B + DFlash Drafter Download ==="

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN environment variable is not set."
  echo "       The model may be gated — set it using: export HF_TOKEN='your_token'"
  exit 1
fi

download_repo() {
  local repo_id="$1"
  local target_dir="$2"
  local label="$3"

  echo "── Checking $label ($repo_id) ──"
  if [[ -d "$target_dir" ]]; then
    local count
    count=$(find "$target_dir" -maxdepth 3 \
      \( -name "*.safetensors" -o -name "config.json" \) 2>/dev/null | wc -l)
    if [[ "$count" -gt 0 ]]; then
      echo "  ✓ Model already cached at $target_dir ($count weight/config files) — skipping download."
      return 0
    fi
  fi

  echo "  Downloading $repo_id to $target_dir..."
  if command -v huggingface-cli &>/dev/null; then
    huggingface-cli download "$repo_id" \
      --local-dir "$target_dir" \
      --token "$HF_TOKEN"
  elif command -v uv &>/dev/null; then
    echo "  Using isolated uv tool run for huggingface-cli..."
    uv tool run --from huggingface_hub huggingface-cli download "$repo_id" \
      --local-dir "$target_dir" \
      --token "$HF_TOKEN"
  else
    echo "  huggingface-cli not found — using Python huggingface_hub..."
    python3 -c "
from huggingface_hub import snapshot_download
import os
snapshot_download(
    repo_id='$repo_id',
    local_dir='$target_dir',
    token=os.environ.get('HF_TOKEN'),
)
"
  fi
  echo "  ✓ Finished downloading $label."
}

# 1. Download primary NVFP4 W4A4 model (~16 GB)
download_repo "$MODEL_ID" "$MODEL_CACHE_DIR" "Primary Target Model"

echo ""

# 2. Download DFlash Speculative Drafter model (~3 GB)
download_repo "$DFLASH_MODEL_ID" "$DFLASH_CACHE_DIR" "DFlash Speculative Drafter"

echo ""
echo "✓ All model weights verified in $HF_HOME."
echo "Next: bash setup/preflight.sh && bash docker/start.sh"


