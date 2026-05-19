#!/usr/bin/env bash
# Stop everything that scripts/start-all.sh started.
#
# Order of teardown matches the dependency direction:
#   1. start-all.sh wrapper (so its own EXIT trap doesn't race with us)
#   2. Envoy + Dashboard          (edge-facing — stop first to drain traffic)
#   3. Router                     (extproc gRPC server Envoy talks to)
#   4. vLLM container(s)          (upstream LLM)
#   5. Observability containers   (Prometheus / Grafana / Jaeger / vllm-sr)
#
# Each process layer is sent SIGTERM, given a few seconds to exit, and any
# survivors are SIGKILL'd. We avoid `set -e` so a single failed match doesn't
# leave the rest of the stack running.
#
# Env:
#   VLLM_CONTAINER   primary vLLM container name (default: vllm-xpu)
#                    Any other container starting with the same prefix
#                    (e.g. vllm-xpu-7b) is also stopped.
#   KEEP_VLLM=true   leave vLLM containers running (only kill router/envoy/dashboard).
#   KEEP_OBS=true    leave observability containers running.
#
# Usage:
#   scripts/stop-all.sh
#   VLLM_CONTAINER=vllm-server scripts/stop-all.sh
#   KEEP_VLLM=true KEEP_OBS=true scripts/stop-all.sh   # stop only local processes
set -uo pipefail

VLLM_CONTAINER=${VLLM_CONTAINER:-vllm-xpu}
KEEP_VLLM=${KEEP_VLLM:-false}
KEEP_OBS=${KEEP_OBS:-false}

log()  { echo "[stop-all] $*"; }
warn() { echo "[stop-all] WARN: $*" >&2; }

# Send SIGTERM to all PIDs matching $1 (regex for pgrep -f), wait up to $2 s,
# then SIGKILL anything still alive. Echoes a one-line status.
stop_pattern() {
  local label=$1 pattern=$2 timeout=${3:-5}
  local pids
  mapfile -t pids < <(pgrep -f -- "$pattern" 2>/dev/null || true)
  if (( ${#pids[@]} == 0 )); then
    log "  $label: not running"
    return 0
  fi
  kill -TERM "${pids[@]}" 2>/dev/null || true
  local i
  for ((i=0; i<timeout; i++)); do
    sleep 1
    mapfile -t pids < <(pgrep -f -- "$pattern" 2>/dev/null || true)
    (( ${#pids[@]} == 0 )) && break
  done
  if (( ${#pids[@]} > 0 )); then
    warn "  $label: ${#pids[@]} still alive after ${timeout}s, sending SIGKILL"
    kill -KILL "${pids[@]}" 2>/dev/null || true
    sleep 1
  fi
  log "  $label: stopped"
}

stop_container() {
  local name=$1
  if ! docker inspect "$name" &>/dev/null; then
    return 1  # doesn't exist
  fi
  local state
  state=$(docker inspect -f '{{.State.Status}}' "$name" 2>/dev/null || echo unknown)
  if [[ "$state" != "running" ]]; then
    log "  $name: not running ($state)"
    return 0
  fi
  if docker stop "$name" >/dev/null 2>&1; then
    log "  $name: stopped"
  else
    warn "  $name: docker stop failed"
  fi
}

log "Stopping semantic-router stack..."

# 1. Tear down the start-all.sh wrapper first so its EXIT trap doesn't race
#    with our shutdown of the same processes.
log "Layer 1: start-all.sh supervisor"
stop_pattern "start-all.sh"           "scripts/start-all.sh" 3

# 2. Edge-facing: stop these first so no new requests reach the router/upstream.
log "Layer 2: Envoy + Dashboard"
stop_pattern "Envoy (func-e)"         "bin/func-e run"        5
stop_pattern "Envoy"                  "envoy --config-path"   5
stop_pattern "Dashboard"              "dashboard-server"      3

# 3. Router: kill the bound binary AND its entrypoint.sh wrapper (start-all.sh
#    launches router via `setsid env … entrypoint.sh`, both should go).
log "Layer 3: Router"
stop_pattern "Router binary"          "bin/router-(openvino|candle|onnx)" 5
stop_pattern "Router entrypoint"      "scripts/entrypoint.sh"             3

# 4. vLLM containers — stop the primary plus any siblings sharing the prefix
#    (e.g. vllm-xpu-7b). Skipped if KEEP_VLLM=true.
if [[ "$KEEP_VLLM" == "true" ]]; then
  log "Layer 4: vLLM (skipped — KEEP_VLLM=true)"
else
  log "Layer 4: vLLM containers"
  matched=()
  while IFS= read -r name; do
    [[ -n "$name" ]] && matched+=("$name")
  done < <(docker ps --format '{{.Names}}' 2>/dev/null \
            | awk -v p="$VLLM_CONTAINER" '$0 == p || index($0, p"-") == 1')
  if (( ${#matched[@]} == 0 )); then
    log "  no running containers matching '$VLLM_CONTAINER*'"
  else
    for c in "${matched[@]}"; do stop_container "$c"; done
  fi
fi

# 5. Observability — Prometheus, Grafana, Jaeger, plus the vllm-sr router
#    container if it was bootstrapped.
if [[ "$KEEP_OBS" == "true" ]]; then
  log "Layer 5: Observability (skipped — KEEP_OBS=true)"
else
  log "Layer 5: Observability containers"
  for c in prometheus grafana jaeger vllm-sr-container; do
    docker inspect "$c" &>/dev/null || continue
    stop_container "$c"
  done
fi

# Final port sanity check — anything still bound on our ports is a problem.
log "Port check (anything listed below is still alive):"
ss -H -tln 2>/dev/null \
  | awk '$4 ~ /:(8080|8702|8801|9190|11434|11435|19000|50051)$/ {print "  still bound: " $4}' \
  || true

log "Done."
