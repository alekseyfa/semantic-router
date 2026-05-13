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

You also need a **HuggingFace token** set in your environment for first-run model downloads:

```bash
export HF_TOKEN=your_token_here   # https://huggingface.co/settings/tokens
```

If you're behind a corporate proxy, export the standard variables before running any commands:

```bash
export https_proxy=http://your-proxy:port
export http_proxy=http://your-proxy:port
export no_proxy=localhost,127.0.0.1
```

All scripts and Docker commands in this guide inherit these from the environment — nothing is hardcoded.

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
HF_TOKEN="${HF_TOKEN}" vllm-sr serve --config config.yaml
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
# Download Envoy 1.35.4 (proxy is inherited from environment if set):
bin/func-e use 1.35.4
```

**Start Envoy:**
```bash
> /tmp/envoy.log
nohup /home/gta/semantic-router/scripts/start-envoy.sh &
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
> /tmp/dashboard.log
nohup /home/gta/semantic-router/scripts/start-dashboard.sh >> /tmp/dashboard.log 2>&1 &
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
HF_TOKEN="${HF_TOKEN}" vllm-sr serve --config config.yaml

# 2. vLLM on Arc B570 (if not already running)
docker start vllm-xpu   # starts existing container; see vLLM section below if missing

# 3. Envoy
> /tmp/envoy.log && nohup /home/gta/semantic-router/scripts/start-envoy.sh & disown $!

# 4. Dashboard
> /tmp/dashboard.log && nohup /home/gta/semantic-router/scripts/start-dashboard.sh >> /tmp/dashboard.log 2>&1 & disown $!
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

## Connecting a vLLM Endpoint (Intel Arc B570 dGPU)

### Model: `Qwen/Qwen2.5-3B-Instruct`

**Why this model:**
- **3B params × 2 bytes (BF16) = ~6 GB** — fits in 9.4 GB VRAM with 3 GB left for KV cache
- Strong Q&A quality in the 3B class, OpenAI-compatible chat format
- Well-tested with vLLM, fast on Intel XPU

**Hardware:** Intel Arc B570 (`renderD128`, 9.4 GB VRAM, driver 1.14.36300)

### Start vLLM on Arc B570

```bash
# One-time: start and keep as a persistent named container
RENDER_GID=$(stat -c '%g' /dev/dri/renderD128)
VIDEO_GID=$(stat -c '%g' /dev/dri/card1)

docker run -d \
  --name vllm-xpu \
  --restart unless-stopped \
  --device /dev/dri/renderD128 \
  --device /dev/dri/card1 \
  --group-add $RENDER_GID \
  --group-add $VIDEO_GID \
  -p 11434:8000 \
  -v /home/gta/.cache/huggingface:/root/.cache/huggingface \
  -e HF_TOKEN="${HF_TOKEN}" \
  -e https_proxy="${https_proxy}" \
  -e http_proxy="${http_proxy}" \
  -e no_proxy="${no_proxy:-localhost,127.0.0.1}" \
  -e ZE_AFFINITY_MASK=0 \
  intel/vllm:0.17.0-xpu \
  python3 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-3B-Instruct \
    --dtype bfloat16 \
    --port 8000 \
    --host 0.0.0.0 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85
```

First run downloads the model (~6 GB). Wait ~2 minutes for startup:

```bash
# Poll until ready
until curl -sf http://localhost:11434/health > /dev/null 2>&1; do
  echo "Waiting for vLLM..."; sleep 15
done && echo "vLLM ready"

# After reboot: just restart the existing container (no re-download)
docker start vllm-xpu
```

**Verify GPU inference directly:**
```bash
curl -s http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen2.5-3B-Instruct",
       "messages":[{"role":"user","content":"What is 2+2?"}],
       "max_tokens":20}' | python3 -c "
import json,sys; d=json.load(sys.stdin)
print(d['choices'][0]['message']['content'])"
```

### Connect to Semantic Router

The `config.yaml` at the repo root is already configured. The Envoy `config/envoy.yaml` uses
a **static cluster** pointing to `localhost:11434`.

**Port flow:**
```
[Playground / curl] → :8801 (Envoy) → ExtProc :50051 (Router) → :11434 (vLLM on Arc B570)
```

**Test end-to-end through the full pipeline:**
```bash
curl -s http://localhost:8801/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-3B-Instruct",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "max_tokens": 30
  }' | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['choices'][0]['message']['content'])"
# → Paris
```

**Or use the Dashboard Playground at http://localhost:8702** — set model to
`Qwen/Qwen2.5-3B-Instruct` and send any message.

### Switching to a different model or endpoint

Edit `config.yaml`:
```yaml
providers:
  models:
    - name: "your-model-name"         # must match the HF model ID served by vLLM
      backend_refs:
        - name: "primary"
          weight: 100
          endpoint: "localhost:11434"  # host:port only — no scheme, no /v1 path
```

Edit `config/envoy.yaml` static cluster if the port changes:
```yaml
  - name: vllm_dynamic_cluster
    ...
    load_assignment:
      cluster_name: vllm_dynamic_cluster
      endpoints:
      - lb_endpoints:
        - endpoint:
            address:
              socket_address:
                address: 127.0.0.1
                port_value: 11434   # ← change port here
```

Then restart the semantic router and Envoy:
```bash
source .venv/bin/activate
HF_TOKEN="${HF_TOKEN}" vllm-sr serve --config config.yaml
pkill -9 -f 'func-e|envoy'
> /tmp/envoy.log && nohup /home/gta/semantic-router/scripts/start-envoy.sh & disown $!
```

### Notes on the Envoy cluster design

The `vllm_dynamic_cluster` in `config/envoy.yaml` is configured as `STATIC` (pointing to
`127.0.0.1:11434`) for local development. The original `ORIGINAL_DST` design (which reads
the `x-vsr-destination-endpoint` header set by the router's ExtProc) **does not work as a
terminating HTTP proxy** in Envoy 1.35 — ORIGINAL_DST with `use_http_header` only works in
transparent proxy mode. For local dev, a STATIC cluster is simpler and reliable.

---

## Stopping Everything

```bash
source .venv/bin/activate
vllm-sr stop                          # stops router + observability containers
docker stop vllm-xpu                  # stops vLLM (keeps container for fast restart)
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

## How to connect vllm entry point

```bash
docker pull intel/vllm:0.17.0-xpu
docker run -d --privileged --net=host -p 8082:8082 intel/vllm:0.17.0-xpu --model
```

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
