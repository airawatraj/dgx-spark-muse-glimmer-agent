#!/usr/bin/env bash
# docker/start.sh
# Launches Inferact/Muse-Glimmer-30B-NVFP4-W4A4 (as Cogni-Brain)
# on a single DGX Spark Mini PC with DFlash Speculative Decoding.
# Configured for 128K context window (131,072 tokens) and tool calling + reasoning.
# Safe to run multiple times — removes any existing spark-brain container first.
set -euo pipefail

CONTAINER="spark-brain"
MODEL_ID="${MODEL_ID:-Inferact/Muse-Glimmer-30B-NVFP4-W4A4}"
DFLASH_MODEL_ID="${DFLASH_MODEL_ID:-meta-models/Muse-Glimmer-30B-assistant}"
ENABLE_DFLASH="${ENABLE_DFLASH:-1}"
NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-15}"
SERVED_NAME="${SERVED_NAME:-Cogni-Brain}"
PORT="${PORT:-8000}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
KV_CACHE_BYTES="${KV_CACHE_BYTES:-4147483648}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-8196}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"

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
echo "    Model:           $MODEL_ID"
echo "    Alias:           $SERVED_NAME"
echo "    Hardware:        NVIDIA DGX Spark (Grace-Blackwell sm_121a)"
echo "    DFlash Drafter:  $([ "$ENABLE_DFLASH" = "1" ] && echo "$DFLASH_MODEL_ID (speculative tokens: $NUM_SPEC_TOKENS)" || echo "Disabled")"
echo "    Context Window:  $MAX_MODEL_LEN tokens (128K)"
echo "    Memory Util:     $GPU_MEM_UTIL"
echo "    Port:            $PORT"
echo ""

# ── Construct vLLM Engine Arguments ───────────────────────────────────────────
VLLM_ARGS=(
  "$MODEL_ID"
  --served-model-name "$SERVED_NAME"
  --tensor-parallel-size 1
  --max-model-len "$MAX_MODEL_LEN"
  --gpu-memory-utilization "$GPU_MEM_UTIL"
  --kv-cache-dtype fp8
  --kv-cache-memory-bytes "$KV_CACHE_BYTES"
  --max-num-batched-tokens "$MAX_BATCHED_TOKENS"
  --max-num-seqs 8
  --load-format fastsafetensors
  --attention-backend triton_attn
  --enable-chunked-prefill
  --enable-auto-tool-choice
  --tool-call-parser muse_glimmer
  --reasoning-parser muse_glimmer
  --generation-config auto
  --override-generation-config '{"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 0.0, "repetition_penalty": 1.0}'
  --disable-log-stats
)

if [[ "$ENABLE_DFLASH" == "1" ]]; then
  VLLM_ARGS+=(
    --speculative-config "{\"method\":\"dflash\",\"num_speculative_tokens\":$NUM_SPEC_TOKENS,\"model\":\"$DFLASH_MODEL_ID\"}"
  )
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Launch via plain docker run ───────────────────────────────────────────────
echo "Starting $CONTAINER (Muse-Glimmer-30B NVFP4 W4A4 + DFlash)..."

docker run -d \
  --name "$CONTAINER" \
  --gpus all \
  --shm-size=16gb \
  -e HF_TOKEN="$HF_TOKEN" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "${SCRIPT_DIR}/entrypoint.py:/entrypoint.py" \
  -p "${PORT}:8000" \
  --entrypoint python3 \
  vllm/vllm-openai:muse-glimmer \
  /entrypoint.py \
  "${VLLM_ARGS[@]}"

echo ""
echo "Container started."
echo "Startup takes ~30–60s with FastSafeTensors memory mapping."
echo ""
echo "Monitor: docker logs -f $CONTAINER"
echo "Health:  curl -sf http://localhost:$PORT/health && echo OK"
echo "Stop:    bash docker/stop.sh"

