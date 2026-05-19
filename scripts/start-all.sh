#!/usr/bin/env bash
# Start the full local stack: observability, vLLM backend, router, Envoy, dashboard.
#
# Env vars:
#   AI_BINDING           openvino (default) | candle | onnx
#   SKIP_OBSERVABILITY   true to skip Prometheus/Grafana/Jaeger (default: false)
#   STRICT               true to abort on any health check failure (default: false)
#   VLLM_CONTAINER       vLLM backend container name (default: vllm-xpu)
#   HF_TOKEN             Hugging Face token, passed through to router
#   http_proxy/https_proxy/no_proxy  passed through to router
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

AI_BINDING=${AI_BINDING:-openvino}
SKIP_OBSERVABILITY=${SKIP_OBSERVABILITY:-false}
STRICT=${STRICT:-false}
VLLM_CONTAINER=${VLLM_CONTAINER:-vllm-xpu}

log()  { echo "[start-all] $*"; }
warn() { echo "[start-all] WARN: $*" >&2; }
fail() { echo "[start-all] ERROR: $*" >&2; exit 1; }

check_port() { ss -H -tln "sport = :$1" 2>/dev/null | grep -q . ; }

# Wait for an HTTP endpoint to return 2xx; returns 0 on success, 1 on timeout.
wait_http() {
  local url=$1 timeout=${2:-60} elapsed=0
  while (( elapsed < timeout )); do
    if curl -sf --noproxy localhost -o /dev/null "$url" 2>/dev/null; then
      return 0
    fi
    sleep 2
    elapsed=$((elapsed + 2))
  done
  return 1
}

# Resolve the router binary path for a given AI_BINDING (mirrors entrypoint.sh).
router_binary_for() {
  case "$1" in
    onnx)     echo "$PROJECT_ROOT/bin/router-onnx" ;;
    openvino) echo "$PROJECT_ROOT/bin/router-openvino" ;;
    candle|"") echo "$PROJECT_ROOT/bin/router-candle" ;;
    *) return 1 ;;
  esac
}

# --- Preflight: fail fast with an actionable list of what's missing ---
preflight() {
  local missing=()
  local router_bin
  router_bin=$(router_binary_for "$AI_BINDING") || fail "Unknown AI_BINDING='$AI_BINDING'"

  [[ -x "$router_bin" ]] || missing+=("router binary: $router_bin (build with: cd src/semantic-router && go build -tags=$AI_BINDING -o ../../bin/router-$AI_BINDING ./cmd/main.go)")
  [[ -x "$PROJECT_ROOT/bin/func-e" ]] || missing+=("envoy launcher: bin/func-e (download from https://func-e.io and place in bin/)")
  [[ -x "$PROJECT_ROOT/dashboard/backend/dashboard-server" ]] || missing+=("dashboard backend: dashboard/backend/dashboard-server (build with: cd dashboard/backend && go build -o dashboard-server ./)")
  [[ -d "$PROJECT_ROOT/dashboard/frontend/dist" ]] || missing+=("dashboard frontend: dashboard/frontend/dist (build with: cd dashboard/frontend && npm install && npm run build)")
  [[ -f "$PROJECT_ROOT/config/config.yaml" ]] || missing+=("router config: config/config.yaml")
  [[ -f "$PROJECT_ROOT/config/envoy.yaml" ]] || missing+=("envoy config: config/envoy.yaml")

  if ! docker info &>/dev/null; then
    missing+=("docker daemon not reachable (run: sudo systemctl start docker, or check user groups)")
  else
    docker inspect "$VLLM_CONTAINER" &>/dev/null || missing+=("$VLLM_CONTAINER container not created (see CLAUDE.md → 'Running vLLM on Intel Arc' for the docker run command, or set VLLM_CONTAINER=<existing-name>)")
  fi

  if (( ${#missing[@]} > 0 )); then
    echo "[start-all] Preflight failed. Missing prerequisites:" >&2
    printf '  - %s\n' "${missing[@]}" >&2
    exit 1
  fi
  log "Preflight OK (AI_BINDING=$AI_BINDING)"
}

cleanup() {
  log "Shutting down..."
  # Order matters: take envoy down first so it stops fanning requests at upstreams.
  [[ -n "${ENVOY_PID:-}" ]]     && kill "$ENVOY_PID"     2>/dev/null || true
  [[ -n "${DASHBOARD_PID:-}" ]] && kill "$DASHBOARD_PID" 2>/dev/null || true
  # Router was started in its own process group via setsid — kill the whole group
  # so the entrypoint.sh wrapper AND the router binary both exit.
  if [[ -n "${ROUTER_PGID:-}" ]]; then
    kill -TERM -- "-$ROUTER_PGID" 2>/dev/null || true
  fi
  wait 2>/dev/null || true
  log "Done. ($VLLM_CONTAINER and observability containers were left running)"
}
trap cleanup EXIT INT TERM

preflight

# --- 1. Observability (Prometheus, Grafana, Jaeger) ---
if [[ "$SKIP_OBSERVABILITY" != "true" ]]; then
  obs_started=()
  obs_missing=()
  for c in prometheus grafana jaeger; do
    if docker ps --format '{{.Names}}' | grep -qx "$c"; then
      :  # already running
    elif docker inspect "$c" &>/dev/null; then
      docker start "$c" >/dev/null && obs_started+=("$c")
    else
      obs_missing+=("$c")
    fi
  done
  [[ ${#obs_started[@]} -gt 0 ]] && log "Started observability containers: ${obs_started[*]}"
  if [[ ${#obs_missing[@]} -gt 0 ]]; then
    warn "Observability containers not found: ${obs_missing[*]} (skipping; create them via 'vllm-sr serve' once or set SKIP_OBSERVABILITY=true)"
  fi
fi

# --- 2. vLLM backend ---
if check_port 11434; then
  log "vLLM already running on :11434"
else
  log "Starting $VLLM_CONTAINER container..."
  docker start "$VLLM_CONTAINER" >/dev/null || fail "Failed to start $VLLM_CONTAINER container"
  log "Waiting for vLLM to become ready (up to 120s)..."
  if wait_http "http://localhost:11434/v1/models" 120; then
    log "vLLM ready"
  else
    msg="vLLM not ready after 120s on :11434 (check: docker logs $VLLM_CONTAINER)"
    [[ "$STRICT" == "true" ]] && fail "$msg" || warn "$msg"
  fi
fi

# --- 3. Router ---
if check_port 8080; then
  if curl -sf --noproxy localhost http://localhost:8080/health &>/dev/null; then
    log "Router already healthy on :8080"
  else
    fail ":8080 is occupied by something that's not the router (curl /health failed). Free the port and retry."
  fi
else
  log "Starting router with AI_BINDING=$AI_BINDING..."
  : > /tmp/router.log
  # setsid puts router + entrypoint.sh in their own process group so cleanup()
  # can kill the whole tree on Ctrl-C.
  setsid env \
    AI_BINDING="$AI_BINDING" \
    HF_TOKEN="${HF_TOKEN:-}" \
    http_proxy="${http_proxy:-}" \
    https_proxy="${https_proxy:-}" \
    no_proxy="${no_proxy:-localhost,127.0.0.1}" \
    HTTP_PROXY="${HTTP_PROXY:-}" \
    HTTPS_PROXY="${HTTPS_PROXY:-}" \
    NO_PROXY="${NO_PROXY:-localhost,127.0.0.1}" \
    "$SCRIPT_DIR/entrypoint.sh" >> /tmp/router.log 2>&1 &
  ROUTER_PID=$!
  ROUTER_PGID=$ROUTER_PID  # setsid makes the child its own group leader; PGID == PID
  sleep 2
  if ! kill -0 "$ROUTER_PID" 2>/dev/null; then
    log "Router process exited early. Last lines of /tmp/router.log:"
    tail -20 /tmp/router.log >&2
    fail "Router failed to start"
  fi
  log "Waiting for router /health (up to 60s)..."
  if wait_http "http://localhost:8080/health" 60; then
    log "Router ready on :8080 (gRPC ExtProc on :50051)"
  else
    log "Router /health not responding. Last lines of /tmp/router.log:"
    tail -20 /tmp/router.log >&2
    msg="Router did not become healthy on :8080"
    [[ "$STRICT" == "true" ]] && fail "$msg" || warn "$msg"
  fi
fi

# --- 4. Envoy ---
if check_port 8801; then
  log "Envoy already running on :8801"
else
  log "Starting Envoy..."
  : > /tmp/envoy.log
  "$SCRIPT_DIR/start-envoy.sh" >> /tmp/envoy.log 2>&1 &
  ENVOY_PID=$!
  if wait_http "http://localhost:19000/ready" 30; then
    log "Envoy ready (admin :19000, proxy :8801)"
  else
    log "Envoy admin not responding. Last lines of /tmp/envoy.log:"
    tail -20 /tmp/envoy.log >&2
    msg="Envoy did not become ready"
    [[ "$STRICT" == "true" ]] && fail "$msg" || warn "$msg"
  fi
fi

# --- 5. Dashboard ---
if check_port 8702; then
  log "Dashboard already running on :8702"
else
  log "Starting Dashboard..."
  : > /tmp/dashboard.log
  "$SCRIPT_DIR/start-dashboard.sh" >> /tmp/dashboard.log 2>&1 &
  DASHBOARD_PID=$!
  if wait_http "http://localhost:8702/" 20; then
    log "Dashboard ready on :8702"
  else
    log "Dashboard not responding. Last lines of /tmp/dashboard.log:"
    tail -20 /tmp/dashboard.log >&2
    msg="Dashboard did not become ready"
    [[ "$STRICT" == "true" ]] && fail "$msg" || warn "$msg"
  fi
fi

# --- Summary ---
status_line() {
  local name=$1 port=$2 url=$3 required=${4:-true}
  if check_port "$port"; then
    log "  [OK]   $name  $url"
  elif [[ "$required" == "true" ]]; then
    log "  [FAIL] $name  (expected on :$port)"
  else
    log "  [skip] $name  (optional, :$port)"
  fi
}

echo ""
log "=== Stack Status ==="
status_line "Router           " 8080  "http://localhost:8080/health"
status_line "gRPC ExtProc     " 50051 ":50051"
status_line "Envoy proxy      " 8801  "http://localhost:8801/v1/chat/completions"
status_line "vLLM backend     " 11434 "http://localhost:11434/v1/models"
status_line "Dashboard        " 8702  "http://localhost:8702"
status_line "Metrics          " 9190  "http://localhost:9190/metrics"
status_line "Grafana          " 3000  "http://localhost:3000"           false
status_line "Prometheus       " 9090  "http://localhost:9090"           false
status_line "Jaeger           " 16686 "http://localhost:16686"          false
echo ""
log "Logs: /tmp/router.log, /tmp/envoy.log, /tmp/dashboard.log"
log "Stop: Ctrl-C ($VLLM_CONTAINER and observability containers will keep running)"
echo ""

wait
