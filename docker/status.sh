#!/usr/bin/env bash
# docker/status.sh
# Shows the health of the spark-brain container, system memory, and VmSwap.
# Read-only diagnostic — never modifies system state.
# Uses set -uo pipefail (NOT -e) so individual check failures print gracefully
# rather than aborting the whole status report.
set -uo pipefail

CONTAINER="spark-brain"
PORT=8000

echo "=== spark-brain status ==="
echo ""

# ── Container ─────────────────────────────────────────────────────────────────
CONTAINER_RUNNING=false
if DOCKER_PS=$(docker ps --format '{{.Names}}' 2>/dev/null); then
  if echo "$DOCKER_PS" | grep -q "^${CONTAINER}$"; then
    UPTIME=$(docker inspect "$CONTAINER" --format '{{.State.StartedAt}}' 2>/dev/null || echo "unknown")
    echo "  Container:  running (started $UPTIME)"
    CONTAINER_RUNNING=true
  else
    echo "  Container:  NOT RUNNING"
  fi
else
  echo "  Container:  Docker daemon unreachable"
fi

# ── vLLM health ───────────────────────────────────────────────────────────────
if curl -sf "http://localhost:${PORT}/health" &>/dev/null; then
  echo "  vLLM API:   healthy (http://localhost:${PORT})"
else
  echo "  vLLM API:   not reachable"
fi

# ── Memory ────────────────────────────────────────────────────────────────────
echo ""
echo "=== System memory ==="
free -h 2>/dev/null || echo "  (free not available)"

# ── Swap for vLLM process ─────────────────────────────────────────────────────
echo ""
echo "=== vLLM VmSwap ==="
if [[ "$CONTAINER_RUNNING" == "true" ]]; then
  PID=$(docker inspect --format '{{.State.Pid}}' "$CONTAINER" 2>/dev/null || echo "")
  if [[ -n "$PID" && "$PID" != "0" ]]; then
    VMSWAP=$(grep VmSwap "/proc/$PID/status" 2>/dev/null || echo "VmSwap: unavailable")
    echo "  $VMSWAP"
    if echo "$VMSWAP" | grep -q "0 kB"; then
      echo "  ✓ No swap in use"
    else
      echo "  ⚠ WARNING: vLLM is using swap — performance will be degraded"
    fi
  else
    echo "  PID unavailable (container may still be starting)"
  fi
else
  echo "  Container not running — skipping VmSwap check"
fi

# ── KV cache usage ────────────────────────────────────────────────────────────
echo ""
echo "=== KV cache ==="
METRICS=$(curl -sf "http://localhost:${PORT}/metrics" 2>/dev/null || echo "")
if [[ -n "$METRICS" ]]; then
  KV=$(printf '%s\n' "$METRICS" \
    | awk '!/^#/ && /gpu_cache_usage_perc/ {printf "%.1f%%", $2 * 100; exit}')
  RUNNING_REQ=$(printf '%s\n' "$METRICS" \
    | awk '!/^#/ && /(^|:)num_requests_running([[:space:]]|$)/ {print $2; exit}')
  echo "  KV cache used:     ${KV:-unknown}"
  echo "  Requests running:  ${RUNNING_REQ:-0}"
else
  echo "  Metrics endpoint not available"
fi

# ── Last 5 log lines ──────────────────────────────────────────────────────────
echo ""
echo "=== Recent logs ==="
if [[ "$CONTAINER_RUNNING" == "true" ]]; then
  docker logs "$CONTAINER" --tail 5 2>&1 | sed 's/^/  /' || echo "  (could not fetch logs)"
elif docker ps -a --format '{{.Names}}' 2>/dev/null | grep -q "^${CONTAINER}$"; then
  echo "  Container stopped — last 5 lines before exit:"
  docker logs "$CONTAINER" --tail 5 2>&1 | sed 's/^/  /' || echo "  (could not fetch logs)"
else
  echo "  No $CONTAINER container found"
fi
