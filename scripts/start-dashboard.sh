#!/usr/bin/env bash
# Start the vLLM Semantic Router Dashboard
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export TARGET_ROUTER_API_URL="${TARGET_ROUTER_API_URL:-http://localhost:8080}"
export TARGET_ROUTER_METRICS_URL="${TARGET_ROUTER_METRICS_URL:-http://localhost:9190/metrics}"
export TARGET_ENVOY_URL="${TARGET_ENVOY_URL:-http://localhost:8801}"
export TARGET_GRAFANA_URL="${TARGET_GRAFANA_URL:-http://localhost:3000}"
export TARGET_PROMETHEUS_URL="${TARGET_PROMETHEUS_URL:-http://localhost:9090}"
export TARGET_JAEGER_URL="${TARGET_JAEGER_URL:-http://localhost:16686}"
export ROUTER_CONFIG_PATH="${ROUTER_CONFIG_PATH:-$PROJECT_ROOT/config/config.yaml}"
export DASHBOARD_PORT="${DASHBOARD_PORT:-8702}"

cd "$PROJECT_ROOT/dashboard/backend"
exec ./dashboard-server --static ../frontend/dist
