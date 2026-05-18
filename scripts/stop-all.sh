#!/usr/bin/env bash
set -euo pipefail

log() { echo "[stop-all] $*"; }

log "Stopping all semantic-router services..."

# Router
if pkill -f "router-openvino|router-candle|router-onnx" 2>/dev/null; then
  log "Router stopped"
else
  log "Router was not running"
fi

# Envoy
if pkill -f "func-e|envoy" 2>/dev/null; then
  log "Envoy stopped"
else
  log "Envoy was not running"
fi

# Dashboard
if pkill -f "dashboard-server" 2>/dev/null; then
  log "Dashboard stopped"
else
  log "Dashboard was not running"
fi

# vLLM
if docker stop vllm-xpu 2>/dev/null; then
  log "vLLM-XPU stopped"
else
  log "vLLM-XPU was not running"
fi

# Observability (Prometheus, Grafana, Jaeger)
STOPPED_OBS=false
for name in prometheus grafana jaeger; do
  if docker stop "$name" 2>/dev/null; then
    STOPPED_OBS=true
  fi
done
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q "vllm-sr"; then
  docker stop vllm-sr-container 2>/dev/null && STOPPED_OBS=true
fi
if [[ "$STOPPED_OBS" == "true" ]]; then
  log "Observability containers stopped"
else
  log "Observability was not running"
fi

# start-all.sh background process
pkill -f "start-all.sh" 2>/dev/null || true

log "Done. All services stopped."
