#!/bin/bash
# Start the vLLM Semantic Router Dashboard
export TARGET_ROUTER_API_URL=http://localhost:8080
export TARGET_ROUTER_METRICS_URL=http://localhost:9190/metrics
export TARGET_ENVOY_URL=http://localhost:8801
export TARGET_GRAFANA_URL=http://localhost:3000
export TARGET_PROMETHEUS_URL=http://localhost:9090
export TARGET_JAEGER_URL=http://localhost:16686
export ROUTER_CONFIG_PATH=/home/gta/semantic-router/config/config.yaml
export DASHBOARD_PORT=8702
cd /home/gta/semantic-router/dashboard/backend
exec ./dashboard-server --static ../frontend/dist
