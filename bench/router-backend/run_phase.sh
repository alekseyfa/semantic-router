#!/usr/bin/env bash
# Orchestrate one full bench phase: start router (selected backend), wait for
# it, run the bench client (concurrency sweep), then stop.
#
# Router-only — no Envoy, no vLLM. We are measuring the cost of the router's
# classification pipeline (intent + PII + jailbreak), nothing downstream.
#
# Usage:
#   AI_BINDING=openvino MODE=throughput bench/router-backend/run_phase.sh
#   AI_BINDING=candle   MODE=throughput bench/router-backend/run_phase.sh
#
# Knobs:
#   AI_BINDING    openvino | candle
#   MODE          latency | throughput   (router-side: streams vs threads)
#   LABEL         CSV filename stem (default: ${AI_BINDING}-${MODE})
#   ROUTER_NODE   NUMA node for router    (default: 0)
#   CLIENT_NODE   NUMA node for bench py  (default: 5 — far end, distance 26)
#   CONCURRENCY   comma list, e.g. 1,2,4,8,16,32,64    (default: sweep)
#   N_REQUESTS    requests per concurrency point      (default: 5000)
#   WARMUP_REQ    requests dropped at start of each phase (default: 200)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

AI_BINDING=${AI_BINDING:-openvino}
MODE=${MODE:-throughput}
LABEL=${LABEL:-${AI_BINDING}-${MODE}}
ROUTER_NODE=${ROUTER_NODE:-0}
CLIENT_NODE=${CLIENT_NODE:-5}
CONCURRENCY=${CONCURRENCY:-1,2,4,8,16,32,64}
N_REQUESTS=${N_REQUESTS:-5000}
WARMUP_REQ=${WARMUP_REQ:-200}
ROUTER_LOG=/tmp/router-bench-${AI_BINDING}-${MODE}.log

cleanup() {
  echo "[bench] Cleanup..."
  [[ -n "${ROUTER_PID:-}" ]] && kill "$ROUTER_PID" 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# ---------- launch router on its NUMA node ----------
echo "[bench] Starting router (AI_BINDING=$AI_BINDING MODE=$MODE node=$ROUTER_NODE) -> $ROUTER_LOG"
> "$ROUTER_LOG"
AI_BINDING=$AI_BINDING MODE=$MODE ROUTER_NODE=$ROUTER_NODE \
  CONFIG="config/bench-${AI_BINDING}.yaml" \
  bench/router-backend/run_router.sh >>"$ROUTER_LOG" 2>&1 &
ROUTER_PID=$!

# ---------- wait for router /health ----------
for i in {1..180}; do
  if curl -sf --noproxy localhost http://localhost:8080/health >/dev/null 2>&1; then
    echo "[bench] Router up after ${i}s"
    break
  fi
  if ! kill -0 "$ROUTER_PID" 2>/dev/null; then
    echo "[bench] Router died. Last 40 lines of log:"; tail -40 "$ROUTER_LOG"; exit 1
  fi
  sleep 1
done
curl -sf --noproxy localhost http://localhost:8080/health >/dev/null \
  || { echo "[bench] Router never came up"; tail -60 "$ROUTER_LOG"; exit 1; }

# ---------- run bench (client pinned to a far NUMA node so it can't steal router cores) ----------
echo "[bench] Running bench  label=$LABEL  client_node=$CLIENT_NODE  concurrency=$CONCURRENCY  N=$N_REQUESTS"
cd bench/router-backend
numactl --cpunodebind="$CLIENT_NODE" --membind="$CLIENT_NODE" -- \
  "$ROOT/.venv/bin/python" run_bench.py \
    --label "$LABEL" \
    --concurrency "$CONCURRENCY" \
    --n-requests "$N_REQUESTS" \
    --warmup "$WARMUP_REQ"

echo "[bench] Done. Results: bench/router-backend/results/${LABEL}.csv"
