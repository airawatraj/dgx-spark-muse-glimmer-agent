#!/usr/bin/env bash
# setup/install.sh
# Prerequisites for DGX Spark Mini PC · Muse-Glimmer-30B-NVFP4-W4A4 setup.
# Safe to run multiple times (idempotent).
set -euo pipefail

echo "=== DGX Spark · Muse-Glimmer Setup ==="

# ── 1. Disable swap permanently ───────────────────────────────────────────────
echo "[1/3] Disabling swap..."

# swapoff -a exits 0 when swap is already off, but guard with || true anyway
# so a second run on a system with no swap devices never aborts the script.
sudo swapoff -a || true

# Idempotent sed: only comments out lines not already commented.
# Running this a second time on already-commented lines is a no-op.
sudo sed -ri \
  '/^[[:space:]]*#/! s@^([[:space:]]*[^[:space:]#]+[[:space:]]+[^[:space:]]+[[:space:]]+swap[[:space:]].*)$@#\1@' \
  /etc/fstab

free -h
echo "    Swap disabled (or was already off)."

# ── 2. Verify Docker is running ───────────────────────────────────────────────
echo "[2/3] Checking Docker..."
docker version --format 'Docker {{.Server.Version}}' || {
  echo "ERROR: Docker not running. Start Docker first."
  exit 1
}

# ── 3. Install uv (for benchmark wrappers and scripts) ───────────────────────
echo "[3/3] Installing uv..."
if command -v uv &>/dev/null; then
  echo "    uv already installed: $(uv --version)"
else
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # Source the cargo env if the installer placed uv there
  source "$HOME/.cargo/env" 2>/dev/null || true
  export PATH="$HOME/.local/bin:$PATH"
  echo "    uv installed: $(uv --version)"
fi

echo ""
echo "=== Setup complete ==="
echo "Next: bash setup/download_model.sh"
