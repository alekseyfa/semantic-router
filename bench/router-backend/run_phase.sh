#!/usr/bin/env bash
# Orchestrate one full bench phase: start router (selected backend) + Envoy,
# wait for them, run the bench, then stop.
#
# Usage:
#   AI_BINDING=openvino bench/router-backend/run_phase.sh
#   AI_BINDING=candle   bench/router-backend/run_phase.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

AI_BINDING=${AI_BINDING:-openvino}
LABEL=${LABEL:-$AI_BINDING}
CONCURRENCY=${CONCURRENCY:-16}
ENVOY_LOG=/tmp/envoy-bench.log
ROUTER_LOG=/tmp/router-bench-${AI_BINDING}.log

cleanup() {
  echo "[bench] Cleanup..."
  [[ -n "${ENVOY_PID:-}" ]] && kill "$ENVOY_PID" 2>/dev/null || true
  [[ -n "${ROUTER_PID:-}" ]] && kill "$ROUTER_PID" 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# ---------- launch router ----------
echo "[bench] Starting router (AI_BINDING=$AI_BINDING) -> $ROUTER_LOG"
> "$ROUTER_LOG"
AI_BINDING=$AI_BINDING CONFIG="config/bench-${AI_BINDING}.yaml" \
  bench/router-backend/run_router.sh >>"$ROUTER_LOG" 2>&1 &
ROUTER_PID=$!

# ---------- wait for router /health ----------
for i in {1..120}; do
  if curl -sf --noproxy localhost http://localhost:8080/health >/dev/null 2>&1; then
    echo "[bench] Router up after ${i}s"
    break
  fi
  if ! kill -0 "$ROUTER_PID" 2>/dev/null; then
    echo "[bench] Router died. Last 30 lines of log:"; tail -30 "$ROUTER_LOG"; exit 1
  fi
  sleep 1
done
curl -sf --noproxy localhost http://localhost:8080/health >/dev/null \
  || { echo "[bench] Router never came up"; tail -50 "$ROUTER_LOG"; exit 1; }

# ---------- launch Envoy ----------
echo "[bench] Starting Envoy -> $ENVOY_LOG"
> "$ENVOY_LOG"
scripts/start-envoy.sh &
ENVOY_PID=$!

for i in {1..30}; do
  if curl -sf --noproxy localhost http://localhost:19000/ready 2>/dev/null | grep -q LIVE; then
    echo "[bench] Envoy up after ${i}s"
    break
  fi
  sleep 1
done

# ---------- run bench ----------
echo "[bench] Running bench (label=$LABEL)"
cd bench/router-backend
"$ROOT/.venv/bin/python" run_bench.py --label "$LABEL" --concurrency "$CONCURRENCY"

echo "[bench] Done. Results: bench/router-backend/results/${LABEL}.csv"
