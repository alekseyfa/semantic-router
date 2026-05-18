#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

AI_BINDING=${AI_BINDING:-openvino}
SKIP_OBSERVABILITY=${SKIP_OBSERVABILITY:-false}

log() { echo "[start-all] $*"; }
check_port() { ss -tlnp 2>/dev/null | grep -q ":$1 " && return 0 || return 1; }

cleanup() {
  log "Shutting down..."
  [[ -n "${ROUTER_PID:-}" ]] && kill "$ROUTER_PID" 2>/dev/null
  [[ -n "${ENVOY_PID:-}" ]] && kill "$ENVOY_PID" 2>/dev/null
  [[ -n "${DASHBOARD_PID:-}" ]] && kill "$DASHBOARD_PID" 2>/dev/null
  wait 2>/dev/null
  log "Done."
}
trap cleanup EXIT INT TERM

# --- 1. Observability (Prometheus, Grafana, Jaeger) ---
if [[ "$SKIP_OBSERVABILITY" != "true" ]]; then
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q "prometheus\|grafana\|jaeger"; then
    log "Observability containers already running"
  else
    log "Starting observability stack (Prometheus, Grafana, Jaeger)..."
    source "$PROJECT_ROOT/.venv/bin/activate" 2>/dev/null || true
    if command -v vllm-sr &>/dev/null; then
      HF_TOKEN="${HF_TOKEN:-}" vllm-sr serve --config config.yaml &
      VLLMSR_PID=$!
      sleep 10
      kill "$VLLMSR_PID" 2>/dev/null || true
      wait "$VLLMSR_PID" 2>/dev/null || true
      # Stop the Docker router — we use the local binary instead
      docker stop vllm-sr-container 2>/dev/null || true
      log "Observability stack started (Docker router stopped, using local binary)"
    else
      log "WARN: vllm-sr not found, skipping observability"
    fi
  fi
fi

# --- 2. vLLM backend ---
if check_port 11434; then
  log "vLLM already running on :11434"
else
  log "Starting vLLM-XPU container..."
  docker start vllm-xpu 2>/dev/null || {
    log "ERROR: vllm-xpu container not found. Create it first (see HOW_TO_RUN.md)"
    exit 1
  }
  log "Waiting for vLLM to become ready..."
  for i in $(seq 1 24); do
    if curl -sf --noproxy localhost http://localhost:11434/v1/models &>/dev/null; then
      log "vLLM ready"
      break
    fi
    [[ $i -eq 24 ]] && log "WARN: vLLM not ready after 120s, continuing anyway"
    sleep 5
  done
fi

# --- 3. Router ---
if check_port 8080; then
  log "Router already running on :8080"
else
  log "Starting router with AI_BINDING=$AI_BINDING..."
  AI_BINDING="$AI_BINDING" "$SCRIPT_DIR/entrypoint.sh" > /tmp/router.log 2>&1 &
  ROUTER_PID=$!
  sleep 2
  if ! kill -0 "$ROUTER_PID" 2>/dev/null; then
    log "ERROR: Router failed to start. Check /tmp/router.log"
    tail -5 /tmp/router.log
    exit 1
  fi
  # Wait for API to be ready (up to 60s for model download + Milvus timeout)
  for i in $(seq 1 12); do
    if curl -sf --noproxy localhost http://localhost:8080/health &>/dev/null; then
      log "Router ready on :8080 (gRPC on :50051)"
      break
    fi
    [[ $i -eq 12 ]] && log "WARN: Router API not responding yet, continuing"
    sleep 5
  done
fi

# --- 4. Envoy ---
if check_port 8801; then
  log "Envoy already running on :8801"
else
  log "Starting Envoy..."
  "$SCRIPT_DIR/start-envoy.sh" > /tmp/envoy.log 2>&1 &
  ENVOY_PID=$!
  sleep 3
  if curl -sf --noproxy localhost http://localhost:19000/ready &>/dev/null; then
    log "Envoy ready on :8801"
  else
    log "WARN: Envoy admin not responding on :19000"
  fi
fi

# --- 5. Dashboard ---
if check_port 8702; then
  log "Dashboard already running on :8702"
else
  log "Starting Dashboard..."
  "$SCRIPT_DIR/start-dashboard.sh" > /tmp/dashboard.log 2>&1 &
  DASHBOARD_PID=$!
  sleep 2
  if curl -sf --noproxy localhost -o /dev/null http://localhost:8702/ 2>/dev/null; then
    log "Dashboard ready on :8702"
  else
    log "WARN: Dashboard not responding on :8702"
  fi
fi

# --- Summary ---
echo ""
log "=== Stack Status ==="
check_port 8080  && log "  Router (OpenVINO):  http://localhost:8080/health" || log "  Router: NOT RUNNING"
check_port 50051 && log "  gRPC ExtProc:       :50051" || log "  gRPC ExtProc: NOT RUNNING"
check_port 8801  && log "  Envoy proxy:        http://localhost:8801/v1/chat/completions" || log "  Envoy: NOT RUNNING"
check_port 11434 && log "  vLLM (Qwen2.5-3B):  http://localhost:11434/v1/models" || log "  vLLM: NOT RUNNING"
check_port 8702  && log "  Dashboard:          http://localhost:8702" || log "  Dashboard: NOT RUNNING"
check_port 9190  && log "  Metrics:            http://localhost:9190/metrics" || log "  Metrics: NOT RUNNING"
check_port 3000  && log "  Grafana:            http://localhost:3000" || true
check_port 16686 && log "  Jaeger:             http://localhost:16686" || true
echo ""
log "Logs: /tmp/router.log, /tmp/envoy.log, /tmp/dashboard.log"
log "Stop: Ctrl-C or kill this process"
echo ""

# Keep running until interrupted
wait
