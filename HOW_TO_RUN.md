# How to Run vLLM Semantic Router Locally

## What This Service Is

vLLM Semantic Router is an intelligent LLM proxy that classifies incoming requests and routes
them to the best backend model. It includes:

- **Router** (`vllm-sr-container`) — Go binary that classifies requests, applies caching,
  PII/jailbreak detection, and picks the right LLM backend
- **Envoy proxy** — local process forwarding LLM API traffic through the router's ExtProc
- **Dashboard** — React + Go web UI built from `dashboard/`
- **Observability** — Prometheus, Jaeger, Grafana (Docker containers)

---

## Port Reference

| Port | Service | URL |
|------|---------|-----|
| **8702** | Dashboard UI | `http://localhost:8702` |
| **8801** | Envoy proxy — LLM API | `http://localhost:8801/v1/chat/completions` |
| **8080** | Router REST API / health | `http://localhost:8080/health` |
| **9190** | Prometheus metrics | `http://localhost:9190/metrics` |
| **19000** | Envoy admin | `http://localhost:19000/ready` |
| **50051** | Router gRPC ExtProc (internal) | not accessed directly |
| **3000** | Grafana | `http://localhost:3000` (admin/admin) |
| **16686** | Jaeger tracing UI | `http://localhost:16686` |
| **9090** | Prometheus UI | `http://localhost:9090` |

> **Accessing from a remote machine or VM** — forward these ports via SSH:
> ```bash
> ssh -L 8702:localhost:8702 \
>     -L 8801:localhost:8801 \
>     -L 8080:localhost:8080 \
>     -L 3000:localhost:3000 \
>     -L 16686:localhost:16686 \
>     user@your-server
> ```
> Then open `http://localhost:8702` in your local browser.

---

## Prerequisites

```bash
# Already installed on this machine:
docker --version    # 29.1.3
go version          # 1.22.2
node --version      # v22.22.2
npm --version       # 10.9.7
python3 --version   # 3.12.x
```

You also need a **HuggingFace token** (`HF_TOKEN`) for first-run model downloads.

---

## Quickstart (Verified Working)

This is the exact sequence that was used to start the service.

### Step 1 — Install the CLI (one-time)

```bash
cd /home/gta/semantic-router
python3 -m venv .venv          # if not already present
source .venv/bin/activate
pip install -e src/vllm-sr
```

### Step 2 — config.yaml (already present at repo root)

The root-level `config.yaml` is what `vllm-sr serve` uses. It requires the **v0.3 format**
with `routing.modelCards` and `providers.models[].backend_refs`:

```yaml
version: v0.1

listeners:
  - name: "http-8899"
    address: "0.0.0.0"
    port: 8899
    timeout: "300s"

routing:
  modelCards:
    - name: "default-model"
  decisions:
    - name: "default-route"
      description: "Catch-all: route every request to the default model"
      priority: 100
      rules:
        operator: "AND"
        conditions: []
      modelRefs:
        - model: "default-model"

providers:
  models:
    - name: "default-model"
      backend_refs:
        - name: "primary"
          weight: 100
          endpoint: "host.docker.internal:11434/v1"   # ← your LLM endpoint
```

> Change `host.docker.internal:11434/v1` to your actual LLM endpoint.
> Inside Docker, `host.docker.internal` resolves to the host machine.
> A `config.yaml` with this content is already at the repo root.
>
> **Important**: `decisions` must be under `routing`, not at the top level.
> Without a decision, the router starts but never sets the upstream destination header,
> so Envoy returns 503 on every request.

### Step 3 — Start the router + observability stack

```bash
cd /home/gta/semantic-router
source .venv/bin/activate
HF_TOKEN=hf_your_token_here vllm-sr serve --config config.yaml
```

This will:
1. Pull `ghcr.io/vllm-project/semantic-router/vllm-sr:latest` (if not already present)
2. Start Jaeger, Prometheus, and Grafana containers
3. Start the `vllm-sr-container` (router on ports 8080, 8899, 9190, 50051)
4. Download HuggingFace classifier models into `./models/` on first run

Wait for `vLLM Semantic Router is running!`, then Ctrl-C (containers keep running).

**Verify:**
```bash
curl http://localhost:8080/health
# → {"status": "healthy", "service": "classification-api"}
```

### Step 4 — Start Envoy (one-time setup, then reuse)

Envoy is not in the Docker image — it runs locally via `func-e`.

**Install func-e (one-time):**
```bash
cd /home/gta/semantic-router
curl -sSL https://func-e.io/install.sh | bash -s -- -b bin/ v1.3.0
# Download Envoy 1.35.4 (needs proxy in corporate networks):
https_proxy=http://proxy-dmz.intel.com:912 bin/func-e use 1.35.4
```

**Start Envoy:**
```bash
cat > /tmp/start-envoy.sh << 'EOF'
#!/bin/bash
export https_proxy=http://proxy-dmz.intel.com:912
export http_proxy=http://proxy-dmz.intel.com:912
exec /home/gta/semantic-router/bin/func-e run \
  --config-path /home/gta/semantic-router/config/envoy.yaml \
  --component-log-level "ext_proc:info,router:info,http:warn" \
  >> /tmp/envoy.log 2>&1
EOF
chmod +x /tmp/start-envoy.sh
> /tmp/envoy.log
nohup /tmp/start-envoy.sh &
disown $!
```

**Verify:**
```bash
sleep 5 && curl -s http://localhost:19000/ready
# → LIVE
```

### Step 5 — Build and start the Dashboard (one-time build, then reuse binary)

The dashboard is not in the Docker image — it runs as a local Go+React process.

**Build (one-time, ~45s total):**
```bash
cd /home/gta/semantic-router/dashboard/frontend
npm install && npm run build

cd /home/gta/semantic-router/dashboard/backend
go build -o dashboard-server .
```

**Start:**
```bash
cat > /tmp/start-dashboard.sh << 'EOF'
#!/bin/bash
export TARGET_ROUTER_API_URL=http://localhost:8080
export TARGET_ROUTER_METRICS_URL=http://localhost:9190/metrics
export TARGET_ENVOY_URL=http://localhost:8801
export TARGET_GRAFANA_URL=http://localhost:3000
export TARGET_PROMETHEUS_URL=http://localhost:9090
export TARGET_JAEGER_URL=http://localhost:16686
export ROUTER_CONFIG_PATH=/home/gta/semantic-router/config.yaml
export DASHBOARD_PORT=8702
cd /home/gta/semantic-router/dashboard/backend
exec ./dashboard-server --static ../frontend/dist
EOF
chmod +x /tmp/start-dashboard.sh
> /tmp/dashboard.log
nohup /tmp/start-dashboard.sh >> /tmp/dashboard.log 2>&1 &
disown $!
```

**Verify:**
```bash
sleep 3 && curl -s -o /dev/null -w "%{http_code}" http://localhost:8702/
# → 200
```

Open **http://localhost:8702** in your browser.

---

## Restarting After a Reboot (full sequence)

```bash
cd /home/gta/semantic-router
source .venv/bin/activate

# 1. Router + observability (Ctrl-C after "vLLM Semantic Router is running!")
HF_TOKEN=hf_your_token_here vllm-sr serve --config config.yaml

# 2. Envoy
> /tmp/envoy.log && nohup /tmp/start-envoy.sh & disown $!

# 3. Dashboard
> /tmp/dashboard.log && nohup /tmp/start-dashboard.sh >> /tmp/dashboard.log 2>&1 & disown $!
```

---

## Sending a Test Request

```bash
# Through Envoy (full routing pipeline):
curl -s -w "\nHTTP:%{http_code}\n" http://localhost:8801/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"MoM","messages":[{"role":"user","content":"What is x^2 derivative?"}]}'

# Direct to router API (bypasses Envoy, useful for debugging):
curl http://localhost:8080/health
curl http://localhost:8080/version
```

> With no LLM backend running you get `503 no healthy upstream` — that is correct behavior.
> Configure your LLM endpoint in `config.yaml` under `providers.models[].backend_refs[].endpoint`.

---

## Health Checks

```bash
curl http://localhost:8080/health       # router
curl http://localhost:19000/ready       # envoy admin
curl http://localhost:8702/api/status   # all services (JSON)
curl http://localhost:9190/metrics      # prometheus metrics
```

Expected dashboard status (all green):
```json
"overall": "healthy",
"Router":    { "status": "running", "healthy": true }
"Envoy":     { "status": "running", "healthy": true }
"Dashboard": { "status": "running", "healthy": true }
```

---

## Stopping Everything

```bash
source .venv/bin/activate
vllm-sr stop                          # stops router + observability containers
pkill -9 -f 'func-e\|envoy'          # stops Envoy
pkill -f dashboard-server             # stops dashboard
```

---

## Architecture Notes (v0.3 container)

The `vllm-sr:latest` Docker image (as of May 2026) is **router-only** — no Envoy,
no dashboard. The architecture is:

```
[Browser] → localhost:8702 (dashboard, local Go process)
                ↓ proxies API
             localhost:8080 (router REST API, Docker)
             localhost:9190 (metrics, Docker)

[LLM client] → localhost:8801 (Envoy, local process)
                    ↓ ExtProc gRPC
               localhost:50051 (router gRPC, Docker)
                    ↓ routes to
               host.docker.internal:<llm-port>

Observability (Docker):
  localhost:3000  → Grafana
  localhost:9090  → Prometheus
  localhost:16686 → Jaeger
```

**Config format note**: `vllm-sr serve` requires `routing.modelCards` +
`providers.models[].backend_refs` (v0.3). The older `providers.models[].endpoints` +
`providers.default_model` format (used in `config/config.yaml`) is deprecated and causes
a fatal error in the `:latest` container.

**Port 8700**: Reserved by `vllm-sr-container` but unused. Dashboard runs on **8702**.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `deprecated config fields` fatal | Use `routing.modelCards` + `providers.models[].backend_refs` (see Step 2) |
| Router container crashes after startup | Missing proxy env vars — fixed in `src/vllm-sr/cli/commands/runtime_support.py` via `PASSTHROUGH_ENV_RULES` |
| Playground returns 403 | Authorino (port 50052) not running — fixed by `failure_mode_allow: true` in `config/envoy.yaml` ext_authz filter |
| Playground returns 502 | Envoy not started or wrong admin port checked — ensure `nohup /tmp/start-envoy.sh` is running |
| Playground returns 503 "no healthy upstream", `selected_model: null` | Missing `routing.decisions` block in `config.yaml` — router starts but never sets the upstream. Add the `decisions` block from Step 2 |
| Playground returns 503 "no healthy upstream", `selected_model: "default-model"` | Routing is correct but LLM backend at your configured endpoint isn't running — start Ollama or your vLLM server |
| Envoy `LIVE` but dashboard shows unknown | Dashboard rebuild needed after code changes; restart with `/tmp/start-dashboard.sh` |
| func-e download fails | Needs proxy: `https_proxy=http://proxy-dmz.intel.com:912 bin/func-e use 1.35.4` |
| Port 8700 bound but connection reset | Normal — `vllm-sr-container` reserves 8700 but nothing serves there. Use 8702 |
| Dashboard exits immediately | Use `nohup ... & disown $!` pattern (plain `&` gets killed by the shell in this env) |
| Router exits immediately | `docker logs vllm-sr-container` — usually proxy or HF_TOKEN issue |
| Models not downloading | Ensure `HF_TOKEN` is valid; proxy vars now pass automatically to container |

---

## Code Changes Made to the Repo

The following files were modified to make local dev work correctly:

| File | Change |
|------|--------|
| `src/vllm-sr/cli/commands/runtime_support.py` | Added `http_proxy`, `https_proxy`, `HTTP_PROXY`, `HTTPS_PROXY`, `no_proxy`, `NO_PROXY` to `PASSTHROUGH_ENV_RULES` so corporate proxy is forwarded into the Docker container |
| `config/envoy.yaml` | Changed ext_authz `failure_mode_allow: false` → `true` so requests aren't blocked when Authorino (port 50052) isn't running |
| `dashboard/backend/handlers/status.go` | Added Envoy HTTP health check fallback (admin port 19000) and hard-coded Dashboard=running when the handler is serving the response |
| `dashboard/backend/router/router.go` | Pass `cfg.EnvoyURL` to `StatusHandler` |
| `config.yaml` (repo root) | Added `routing.decisions` catch-all block — without it the router loads but never sets `x-vsr-destination-endpoint`, causing 503 on every request |
| `openvino-binding/CMakeLists.txt` | Fixed tokenizer discovery to run unconditionally (not only in the find_package fallback path) |

---

## Environment Variables

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `HF_TOKEN` | Yes (first run) | — | Download models from HuggingFace |
| `HF_ENDPOINT` | No | `https://huggingface.co` | Use a mirror: `https://hf-mirror.com` |
| `SR_LOG_LEVEL` | No | `info` | `debug`, `info`, `warn`, `error` |
| `DISABLE_DASHBOARD` | No | — | Set to `true` for headless mode |
| `DASHBOARD_PORT` | No | `8700` | Dashboard listen port (we use 8702 to avoid conflict) |
| `TARGET_ROUTER_API_URL` | No | `http://localhost:8080` | Router API for dashboard |
| `TARGET_ENVOY_URL` | No | — | Envoy proxy URL (playground sends requests here) |
| `TARGET_GRAFANA_URL` | No | — | Grafana URL for embedded view |
| `TARGET_JAEGER_URL` | No | — | Jaeger URL for embedded tracing |
| `TARGET_PROMETHEUS_URL` | No | — | Prometheus URL for embedded metrics |
| `VLLM_SR_NOFILE_LIMIT` | No | `65536` | File descriptor limit for Envoy |
