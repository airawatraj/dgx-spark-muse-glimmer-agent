#!/usr/bin/env bash
# docker/stop.sh
# Stops and removes the spark-brain container.
# Safe to run multiple times — no-op if the container is already gone.
set -euo pipefail

CONTAINER="spark-brain"

# Check if container exists in any state (running, stopped, created)
if docker ps -a --format '{{.Names}}' 2>/dev/null | grep -q "^${CONTAINER}$"; then
  echo "Stopping and removing $CONTAINER..."
  docker rm -f "$CONTAINER" 2>/dev/null || true
  echo "✓ $CONTAINER removed."
else
  echo "$CONTAINER is not running — nothing to do."
fi
