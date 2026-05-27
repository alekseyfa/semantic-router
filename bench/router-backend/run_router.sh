#!/usr/bin/env bash
# Launch the router for the bench. Selects binary + library paths by AI_BINDING.
#
# Methodology (router-only perf, OV vs Candle on Xeon)
# -----------------------------------------------------
# Two knobs control fairness across the OV / Candle runs:
#   1. CPU + memory affinity: pin the router to a single NUMA node so neither
#      backend pays remote-memory tax, and so the comparison generalises to
#      a real per-socket / per-NUMA deployment unit.
#   2. Thread pools: OV (OpenVINO) and Candle (Rayon) auto-derive parallelism
#      from /proc/cpuinfo at startup. Without explicit limits, OV may spin up
#      streams across all 384 logical CPUs and Candle may spawn 384 Rayon
#      workers — both ignore the taskset cgroup constraint on this kernel.
#      We export hard limits and rely on numactl for the actual CPU mask.
#
# Modes:
#   MODE=latency   -> 1 inference stream, threads = node cores. Optimises p50/p99
#                     at low concurrency. Best for "single-user" question.
#   MODE=throughput-> N streams (= node cores), 1 thread/stream. Optimises QPS
#                     under load. Best for "saturation" question.
#
# Usage:
#   AI_BINDING=openvino MODE=latency    bench/router-backend/run_router.sh
#   AI_BINDING=candle   MODE=throughput bench/router-backend/run_router.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

AI_BINDING=${AI_BINDING:-candle}
CONFIG=${CONFIG:-config/bench-${AI_BINDING}.yaml}
MODE=${MODE:-throughput}                 # latency | throughput
ROUTER_NODE=${ROUTER_NODE:-0}            # NUMA node the router pins to

case "$AI_BINDING" in
  openvino) BIN="$ROOT/bin/router-openvino" ;;
  candle)   BIN="$ROOT/bin/router-candle" ;;
  *) echo "Unknown AI_BINDING=$AI_BINDING"; exit 1 ;;
esac

[[ -x "$BIN" ]] || { echo "Binary not found: $BIN"; exit 1; }
[[ -f "$CONFIG" ]] || { echo "Config not found: $CONFIG"; exit 1; }

command -v numactl >/dev/null || { echo "numactl missing — install numactl"; exit 1; }

# Physical-core count for the chosen node (excluding SMT siblings).
NODE_CPULIST=$(numactl --hardware | awk -v n="$ROUTER_NODE" '$1=="node" && $2==n && $3=="cpus:" {for(i=4;i<=NF;i++) print $i}')
[[ -n "$NODE_CPULIST" ]] || { echo "Could not read CPU list for node $ROUTER_NODE"; exit 1; }
NODE_CORES=$(echo "$NODE_CPULIST" | wc -l)
# Strip SMT siblings: on this Xeon the second half of every node's cpu list is
# the HT pair (e.g. node0 = 0-31 + 192-223). We treat node_phys = node_cores/2.
NODE_PHYS=$(( NODE_CORES / 2 ))
[[ "$NODE_PHYS" -gt 0 ]] || NODE_PHYS=$NODE_CORES

# Both backends need candle/ml/nlp shared libs (router still links them all).
# OV backend additionally needs the openvino-binding lib + openvino runtime.
export LD_LIBRARY_PATH="$ROOT/candle-binding/target/release:$ROOT/ml-binding/target/release:$ROOT/nlp-binding/target/release:$ROOT/openvino-binding/build:$ROOT/.venv/lib/python3.13/site-packages/openvino/libs:$ROOT/.venv/lib/python3.13/site-packages/openvino_tokenizers/lib"
export OPENVINO_TOKENIZERS_LIB="$ROOT/.venv/lib/python3.13/site-packages/openvino_tokenizers/lib/libopenvino_tokenizers.so"

# ---------- thread pool topology ----------
# Three classifier slots run concurrently per request (domain, jailbreak, PII).
# We split the NUMA node's CPU budget across them so the slots don't
# oversubscribe each other; this matches what Candle/Rayon does naturally
# (one shared global thread pool of size NODE_PHYS).
CLASSIFIERS_PER_REQUEST=3
PER_CLASSIFIER_THREADS=$(( NODE_PHYS / CLASSIFIERS_PER_REQUEST ))
[[ "$PER_CLASSIFIER_THREADS" -gt 0 ]] || PER_CLASSIFIER_THREADS=1

case "$MODE" in
  latency)
    # Single-stream per classifier — minimises per-request latency at low
    # concurrency. Pin per-stream thread count so one request can fan out
    # across the node's CPU budget for that classifier.
    OV_STREAMS=1
    OV_THREADS=$PER_CLASSIFIER_THREADS
    RAYON_THREADS=$NODE_PHYS
    ;;
  throughput)
    # Many parallel streams. Let OV compute threads-per-stream itself; we
    # only tell it the total stream budget. Setting OV_INFERENCE_NUM_THREADS
    # on top of streams here used to cap CPU usage to streams×1 = ~10 cores
    # out of 32, which is exactly the c=16 throughput-collapse symptom.
    OV_STREAMS=$PER_CLASSIFIER_THREADS
    OV_THREADS=0                       # 0 = let OV decide
    RAYON_THREADS=$NODE_PHYS
    ;;
  *) echo "Unknown MODE=$MODE (expected: latency|throughput)"; exit 1 ;;
esac

# OpenVINO knobs read by ModelManager::buildEnvConfig in the C++ binding.
export OV_NUM_STREAMS=$OV_STREAMS
if [[ "$OV_THREADS" -gt 0 ]]; then
  export OV_INFERENCE_NUM_THREADS=$OV_THREADS
else
  unset OV_INFERENCE_NUM_THREADS
fi
# Candle / Rayon
export RAYON_NUM_THREADS=$RAYON_THREADS
# OMP / MKL — only meaningful for ops outside OV's CPU plugin.
export OMP_NUM_THREADS=$NODE_PHYS
export MKL_NUM_THREADS=$NODE_PHYS
# Avoid Go runtime stealing more cores than the NUMA node has.
export GOMAXPROCS=$NODE_PHYS

echo "[run_router] AI_BINDING=$AI_BINDING binary=$BIN config=$CONFIG"
echo "[run_router] NUMA node=$ROUTER_NODE  phys_cores=$NODE_PHYS  mode=$MODE"
echo "[run_router] OV_NUM_STREAMS=$OV_NUM_STREAMS OV_INFERENCE_NUM_THREADS=${OV_INFERENCE_NUM_THREADS:-auto}"
echo "[run_router] RAYON_NUM_THREADS=$RAYON_NUM_THREADS OMP_NUM_THREADS=$OMP_NUM_THREADS GOMAXPROCS=$GOMAXPROCS"

exec numactl --cpunodebind="$ROUTER_NODE" --membind="$ROUTER_NODE" -- \
  "$BIN" -config="$CONFIG" --enable-system-prompt-api=true
