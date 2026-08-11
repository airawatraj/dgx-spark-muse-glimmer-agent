#!/usr/bin/env bash
# docker/start.sh
# Launches Inferact/Muse-Glimmer-30B-NVFP4-W4A4 (as Cogni-Brain)
# on a single DGX Spark Mini PC via plain docker run.
# Configured for 128K context window (131,072 tokens) and tool calling + reasoning.
# Safe to run multiple times — removes any existing spark-brain container first.
set -euo pipefail

CONTAINER="spark-brain"
MODEL_ID="Inferact/Muse-Glimmer-30B-NVFP4-W4A4"
SERVED_NAME="Cogni-Brain"
PORT=8000

# ── Preflight checks ──────────────────────────────────────────────────────────
echo "=== spark-brain preflight ==="

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN environment variable is not set."
  echo "       Set it using: export HF_TOKEN='your_token'"
  exit 1
fi

# ── Disable swap if still active ─────────────────────────────────────────────
SWAP=$(free 2>/dev/null | awk '/^Swap:/ {print $2}' || echo "0")
if [[ "${SWAP:-0}" -gt 0 ]]; then
  echo "WARNING: Swap is enabled ($SWAP kB). Disabling permanently now..."
  sudo swapoff -a || true
  sudo sed -ri \
    '/^[[:space:]]*#/! s@^([[:space:]]*[^[:space:]#]+[[:space:]]+[^[:space:]]+[[:space:]]+swap[[:space:]].*)$@#\1@' \
    /etc/fstab
  echo "✓ Swap disabled."
else
  echo "  Swap: off ✓"
fi

# Always force-remove any existing container to guarantee zero name conflicts.
if docker ps -a --format '{{.Names}}' 2>/dev/null | grep -q "^${CONTAINER}$"; then
  echo "Removing existing $CONTAINER container..."
  docker rm -f "$CONTAINER" 2>/dev/null || true
  echo "  Removed."
fi

echo ""
echo "    Model:    $MODEL_ID"
echo "    Alias:    $SERVED_NAME"
echo "    Hardware: NVIDIA DGX Spark (Grace-Blackwell sm_121a)"
echo "    Context:  131072 tokens (128K)"
echo "    Port:     $PORT"
echo ""

# ── Launch via plain docker run ───────────────────────────────────────────────
echo "Starting $CONTAINER (Muse-Glimmer-30B NVFP4 W4A4)..."

docker run -d \
  --name "$CONTAINER" \
  --gpus all \
  --shm-size=16gb \
  -e HF_TOKEN="$HF_TOKEN" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -p "${PORT}:8000" \
  vllm/vllm-openai:muse-glimmer \
  "$MODEL_ID" \
  --served-model-name "$SERVED_NAME" \
  --tensor-parallel-size 1 \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.85 \
  --kv-cache-dtype fp8 \
  --max-num-seqs 8 \
  --enable-auto-tool-choice \
  --tool-call-parser muse_glimmer \
  --reasoning-parser muse_glimmer \
  --generation-config auto \
  --disable-log-stats

echo ""
echo "Container started."
echo "Startup can take several minutes — the model is being loaded."
echo ""
echo "Monitor: docker logs -f $CONTAINER"
echo "Health:  curl -sf http://localhost:$PORT/health && echo OK"
echo "Stop:    bash docker/stop.sh"
