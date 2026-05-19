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

Set `REPO_ROOT` once and reuse it throughout:

```bash
export REPO_ROOT=/home/intel/alexeyfa/semantic-router   # adjust to your clone
cd "$REPO_ROOT"
```

---

## Quickstart with `scripts/start-all.sh` (recommended)

`scripts/start-all.sh` brings up the full stack — observability, vLLM backend, router, Envoy,
dashboard — with a preflight check, health-gated startup, and clean shutdown on Ctrl-C.

### One-time setup (do this before the first `start-all.sh` run)

```bash
cd "$REPO_ROOT"

# 1. Python venv + vllm-sr CLI (used once to bootstrap observability containers)
python3 -m venv .venv
source .venv/bin/activate
pip install -e src/vllm-sr

# 2. Build the router with the OpenVINO backend
cd "$REPO_ROOT/src/semantic-router"
CGO_ENABLED=1 go build -tags=openvino -o "$REPO_ROOT/bin/router-openvino" ./cmd/main.go

# 3. Install func-e (Envoy launcher)
cd "$REPO_ROOT"
curl -sSL https://func-e.io/install.sh | bash -s -- -b bin/ v1.3.0
bin/func-e use 1.35.4

# 4. Build the dashboard (frontend + Go backend)
cd "$REPO_ROOT/dashboard/frontend" && npm install && npm run build
cd "$REPO_ROOT/dashboard/backend"  && go build -o dashboard-server ./

# 5. Create the vLLM container (see "Connecting a vLLM Endpoint" below for the full docker run)

# 6. Bootstrap observability containers once (prometheus, grafana, jaeger):
cd "$REPO_ROOT"
HF_TOKEN="${HF_TOKEN}" vllm-sr serve --config config.yaml
# Wait for "vLLM Semantic Router is running!", then Ctrl-C — containers persist for reuse.
```

### Daily startup with the OpenVINO backend

```bash
cd "$REPO_ROOT"
source .venv/bin/activate

AI_BINDING=openvino \
VLLM_CONTAINER=vllm-xpu \
HF_TOKEN="${HF_TOKEN}" \
  scripts/start-all.sh
```

What the script does, in order:

1. **Preflight** — verifies `bin/router-openvino`, `bin/func-e`, `dashboard/backend/dashboard-server`,
   `dashboard/frontend/dist`, both configs, Docker access, and the vLLM container. If anything is
   missing it prints a list with the exact build/install command and exits.
2. **Observability** — `docker start prometheus grafana jaeger` for existing containers
   (skipped if already running; WARN if never created).
3. **vLLM** — `docker start "$VLLM_CONTAINER"` and waits for `:11434/v1/models`.
4. **Router** — launched via `scripts/entrypoint.sh` in its own process group (so Ctrl-C kills the
   whole tree). `HF_TOKEN`, `http_proxy`, `https_proxy`, `no_proxy` and OpenVINO library paths are
   forwarded automatically.
5. **Envoy** — via `func-e`, waits for `:19000/ready`.
6. **Dashboard** — Go binary serving the React build.
7. Prints a status table marking each service `[OK]` / `[FAIL]` / `[skip]`, then waits.
   Ctrl-C tears down router/Envoy/dashboard cleanly; vLLM and observability containers stay up.

### Env vars accepted by `start-all.sh`

| Variable | Default | Purpose |
|----------|---------|---------|
| `AI_BINDING` | `openvino` | `openvino` \| `candle` \| `onnx` — selects `bin/router-<binding>` |
| `VLLM_CONTAINER` | `vllm-xpu` | Docker container name for the vLLM backend |
| `SKIP_OBSERVABILITY` | `false` | Skip prometheus/grafana/jaeger entirely |
| `STRICT` | `false` | Abort on the first failed health check instead of WARN-and-continue |
| `HF_TOKEN` | — | Forwarded to the router for HuggingFace downloads |
| `http_proxy` / `https_proxy` / `no_proxy` | — | Forwarded to the router |

Logs are written to `/tmp/router.log`, `/tmp/envoy.log`, `/tmp/dashboard.log` (truncated each run).
On a failed health check the script tails the last 20 lines of the relevant log so you don't have
to look it up manually.

> **Note for this machine:** the existing vLLM container is named `vllm-server`, not `vllm-xpu`,
> so use `VLLM_CONTAINER=vllm-server` (or recreate it with `--name vllm-xpu` to match the docs).

---

## Quickstart (Verified Working)

> The 5-step manual flow below is preserved for first-time setup and debugging. For day-to-day use,
> prefer `scripts/start-all.sh` (above).

This is the exact sequence that was used to start the service.

### Step 1 — Install the CLI (one-time)

```bash
cd "$REPO_ROOT"
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
cd "$REPO_ROOT"
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
cd "$REPO_ROOT"
curl -sSL https://func-e.io/install.sh | bash -s -- -b bin/ v1.3.0
# Download Envoy 1.35.4 (proxy is inherited from environment if set):
bin/func-e use 1.35.4
```

**Start Envoy:**
```bash
> /tmp/envoy.log
nohup "$REPO_ROOT/scripts/start-envoy.sh" &
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
cd "$REPO_ROOT/dashboard/frontend"
npm install && npm run build

cd "$REPO_ROOT/dashboard/backend"
go build -o dashboard-server .
```

**Start:**
```bash
> /tmp/dashboard.log
nohup "$REPO_ROOT/scripts/start-dashboard.sh" >> /tmp/dashboard.log 2>&1 &
disown $!
```

> `start-dashboard.sh` resolves `$REPO_ROOT` from its own location, so it works regardless of
> where the repo lives.

**Verify:**
```bash
sleep 3 && curl -s -o /dev/null -w "%{http_code}" http://localhost:8702/
# → 200
```

Open **http://localhost:8702** in your browser.

---

## Restarting After a Reboot (full sequence)

The fastest path is `scripts/start-all.sh` (see top of this file). It handles process-group
cleanup, health checks, and proxy/HF_TOKEN passthrough automatically:

```bash
cd "$REPO_ROOT"
source .venv/bin/activate
AI_BINDING=openvino VLLM_CONTAINER=vllm-xpu HF_TOKEN="${HF_TOKEN}" scripts/start-all.sh
```

If you need to bring services up by hand (debugging, partial restart):

```bash
cd "$REPO_ROOT"
source .venv/bin/activate

# 1. Router (local binary with OpenVINO backend)
#    entrypoint.sh resolves bin/router-openvino, sets LD_LIBRARY_PATH and OPENVINO_TOKENIZERS_LIB.
nohup env AI_BINDING=openvino HF_TOKEN="${HF_TOKEN}" \
  scripts/entrypoint.sh > /tmp/router.log 2>&1 &
disown $!

# 2. vLLM on Arc B570 (if not already running)
docker start vllm-xpu   # starts existing container; see vLLM section below if missing

# 3. Envoy
> /tmp/envoy.log && nohup scripts/start-envoy.sh & disown $!

# 4. Dashboard
> /tmp/dashboard.log && nohup scripts/start-dashboard.sh >> /tmp/dashboard.log 2>&1 & disown $!
```

### Alternative: Router via Docker (without OpenVINO)

If you prefer the Docker-based router (uses candle backend, v0.3 config format):
```bash
HF_TOKEN="${HF_TOKEN}" vllm-sr serve --config config.yaml
# Ctrl-C after "vLLM Semantic Router is running!" — containers keep running
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
# NOTE: Check which card device exists: ls /dev/dri/card*
#       On this machine it is card0 (no iGPU). Adjust if your setup differs.
RENDER_GID=$(stat -c '%g' /dev/dri/renderD128)
VIDEO_GID=$(stat -c '%g' /dev/dri/card0)

docker run -d \
  --name vllm-xpu \
  --restart unless-stopped \
  --init \
  --device /dev/dri/renderD128 \
  --device /dev/dri/card0 \
  --group-add $RENDER_GID \
  --group-add $VIDEO_GID \
  -p 11434:8000 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
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
    --gpu-memory-utilization 0.60 \
    --enforce-eager \
    --enable-auto-tool-choice \
    --tool-call-parser hermes
```

> **Important flags explained:**
> - `--init` — prevents zombie processes if vLLM crashes (driver abort leaves unkillable container without this)
> - `--gpu-memory-utilization 0.60` — avoids KV cache over-allocation that triggers a DRM abort in the NEO driver (0.85 tried to allocate 12.9 GiB KV cache on a 9.4 GB card)
> - `--enforce-eager` — skips torch.compile which emits `sycl_arch not recognized` warnings on Battlemage
> - `--enable-auto-tool-choice --tool-call-parser hermes` — required for the Dashboard Playground which sends `tool_choice: "auto"` in requests

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
> /tmp/envoy.log && nohup "$REPO_ROOT/scripts/start-envoy.sh" & disown $!
```

### Notes on the Envoy cluster design

The `vllm_dynamic_cluster` in `config/envoy.yaml` is configured as `STATIC` (pointing to
`127.0.0.1:11434`) for local development. The original `ORIGINAL_DST` design (which reads
the `x-vsr-destination-endpoint` header set by the router's ExtProc) **does not work as a
terminating HTTP proxy** in Envoy 1.35 — ORIGINAL_DST with `use_http_header` only works in
transparent proxy mode. For local dev, a STATIC cluster is simpler and reliable.

---

## OpenVINO Backend (Local Router Binary)

The router can use **OpenVINO** as the inference backend for classification models instead of
candle (Rust). This provides optimized CPU inference via Intel's OpenVINO toolkit.

### Prerequisites

- OpenVINO runtime installed (via pip: `pip install openvino openvino-tokenizers`)
- Models converted to OpenVINO IR format (`.xml` + `.bin`)
- The `openvino-binding` C++ library built (`openvino-binding/build/libopenvino_semantic_router.so`)

### Building the Router

The router binary is built per-binding so multiple variants can coexist in `bin/`. For OpenVINO:

```bash
cd "$REPO_ROOT/src/semantic-router"
CGO_ENABLED=1 go build -tags=openvino -o "$REPO_ROOT/bin/router-openvino" ./cmd/main.go
```

`scripts/entrypoint.sh` picks the binary based on `AI_BINDING` (`openvino` → `bin/router-openvino`,
`candle` → `bin/router-candle`, `onnx` → `bin/router-onnx`).

### Converting Models to OpenVINO IR

Models must be in OpenVINO IR format. Convert from HuggingFace using `optimum-cli`:

```bash
source .venv/bin/activate

# Domain/intent classifier
optimum-cli export openvino \
  --model LLM-Semantic-Router/lora_intent_classifier_bert-base-uncased_model \
  --task text-classification \
  models/mom-domain-classifier

# PII token classifier
optimum-cli export openvino \
  --model LLM-Semantic-Router/lora_pii_detector_bert-base-uncased_model \
  --task token-classification \
  models/mom-pii-classifier

# Jailbreak classifier
optimum-cli export openvino \
  --model LLM-Semantic-Router/jailbreak_classifier_modernbert-base_model \
  --task text-classification \
  models/mom-jailbreak-classifier
```

After conversion, create tokenizer symlinks (required by the C++ binding):
```bash
for d in models/mom-domain-classifier models/mom-pii-classifier models/mom-jailbreak-classifier; do
  ln -sf openvino_tokenizer.xml "$d/tokenizer.xml"
  ln -sf openvino_tokenizer.bin "$d/tokenizer.bin"
done
```

Also download the mapping files (not included in optimum export):
```bash
pip install huggingface_hub
python3 -c "
from huggingface_hub import hf_hub_download
import shutil, os
for repo, files, dest in [
    ('LLM-Semantic-Router/lora_intent_classifier_bert-base-uncased_model',
     ['category_mapping.json', 'label_mapping.json'], 'models/mom-domain-classifier'),
    ('LLM-Semantic-Router/lora_pii_detector_bert-base-uncased_model',
     ['label_mapping.json', 'pii_type_mapping.json'], 'models/mom-pii-classifier'),
    ('LLM-Semantic-Router/jailbreak_classifier_modernbert-base_model',
     ['jailbreak_type_mapping.json'], 'models/mom-jailbreak-classifier'),
]:
    for f in files:
        path = hf_hub_download(repo, f)
        shutil.copy2(path, os.path.join(dest, f))
"
# Symlink for jailbreak mapping (config references label_mapping.json)
ln -sf jailbreak_type_mapping.json models/mom-jailbreak-classifier/label_mapping.json
```

### Config (`config/config.yaml`)

Enable OpenVINO by adding `use_openvino: true` to the relevant sections:

```yaml
prompt_guard:
  use_openvino: true
  openvino_device: "CPU"   # or "GPU", "AUTO"

classifier:
  category_model:
    use_openvino: true
    openvino_device: "CPU"
  pii_model:
    use_openvino: true
    openvino_device: "CPU"

embedding_models:
  use_openvino: true
  openvino_device: "CPU"
```

### Running

The simplest path is via `scripts/entrypoint.sh`, which sets `LD_LIBRARY_PATH` and
`OPENVINO_TOKENIZERS_LIB` for you:

```bash
cd "$REPO_ROOT"
AI_BINDING=openvino HF_TOKEN="${HF_TOKEN}" scripts/entrypoint.sh
```

Or run the binary directly with the env vars set by hand:

```bash
cd "$REPO_ROOT"
export OPENVINO_TOKENIZERS_LIB="$PWD/.venv/lib/python3.12/site-packages/openvino_tokenizers/lib/libopenvino_tokenizers.so"
export LD_LIBRARY_PATH="$PWD/candle-binding/target/release:$PWD/openvino-binding/build:$PWD/nlp-binding/target/release:$PWD/ml-binding/target/release:${LD_LIBRARY_PATH:-}"
./bin/router-openvino --config config/config.yaml
```

### Verify the router is actually using OpenVINO

The backend is fixed at **build time** by Go build tags — `use_openvino: true` in `config.yaml`
is only documentation. If you launched `bin/router-openvino`, classifier/PII/jailbreak/embedding
inference goes through OpenVINO; if you launched `bin/router-candle`, it goes through Candle/Rust.
Use the layered checks below to confirm the OV path is live.

**1. Linker says it's against libopenvino:**
```bash
ldd bin/router-openvino | grep -E 'openvino|tokenizer'
# Expect three lines: libopenvino_semantic_router.so.0, libopenvino.so.<ver>, libopenvino_tokenizers.so
```

**2. Running process has the OV libs mapped in memory:**
```bash
PID=$(pgrep -f router-openvino)
grep -oE '/[^ ]*libopenvino[^ ]*' /proc/$PID/maps | sort -u
```

**3. Startup log shows the OV initializers ran:**
```bash
grep -E 'OpenVINO category classifier|OpenVINO jailbreak|OpenVINO PII|Loaded OpenVINO tokenizers' /tmp/router.log
# Expect "Initializing OpenVINO ... initialized successfully" for category, jailbreak, PII.
# A candle build prints "Initializing ModernBERT classifier model (optimized BERT)..." instead.
```

**4. Live inference returns results in single-digit ms (typical for OV CPU on a small classifier):**
```bash
curl -s --noproxy localhost http://localhost:8080/api/v1/classify/intent \
  -H 'Content-Type: application/json' \
  -d '{"text":"Solve the integral of x^2 dx using the power rule"}'
# → {"classification":{"category":"math_decision","confidence":1.0,"processing_time_ms":6}, ...}
```

**One-liner that combines all four:**
```bash
ldd bin/router-openvino | grep -q libopenvino && \
grep -q 'OpenVINO category classifier initialized' /tmp/router.log && \
curl -s --noproxy localhost http://localhost:8080/api/v1/classify/intent \
  -H 'Content-Type: application/json' \
  -d '{"text":"integrate sin(x) dx"}' | grep -q math_decision && \
echo "OpenVINO backend confirmed"
```

> The full domain-routing E2E suite (classifier + Envoy two-cluster routing + cache/PII/jailbreak)
> lives in [scripts/test-routing.sh](scripts/test-routing.sh). Run `scripts/test-routing.sh` to
> exercise everything.

#### Notes / current limitations

- Device is hardcoded to **CPU** in [classifier_backend_openvino.go](src/semantic-router/pkg/classification/classifier_backend_openvino.go) (`const openvinoDevice = "CPU"`). The
  `openvino_device:` field in `config.yaml` is not consumed by the code yet.
- Category classifier always goes through the ModernBERT FFI path. **BERT-base** OV exports
  fail at runtime (`Eltwise shape infer ... mismatch`); use the ModernBERT variants — e.g.
  `LLM-Semantic-Router/lora_intent_classifier_modernbert-base_model` and the matching PII
  ModernBERT model.
- The classifier returns the category label `"computer science"` (with a space); decisions in
  `config.yaml` use `computer_science` (with underscore), so CS-flagged prompts currently fall
  through to `general_decision`.

---

## Stopping Everything

If you started the stack with `scripts/start-all.sh`, just **Ctrl-C** the foreground process —
router, Envoy, and dashboard exit cleanly (vLLM and observability containers stay up by design).

For a full manual teardown:

```bash
source .venv/bin/activate
pkill -f 'router-openvino\|router-candle\|router-onnx'   # stops local router (any binding)
vllm-sr stop                          # stops Docker router + observability containers
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
| `OPENVINO_TOKENIZERS_LIB` | Yes (OpenVINO mode) | — | Path to `libopenvino_tokenizers.so` |
| `LD_LIBRARY_PATH` | Yes (local binary) | — | Must include paths to all `.so` binding libs |
| `SR_LOG_LEVEL` | No | `info` | `debug`, `info`, `warn`, `error` |
| `DISABLE_DASHBOARD` | No | — | Set to `true` for headless mode |
| `DASHBOARD_PORT` | No | `8700` | Dashboard listen port (we use 8702 to avoid conflict) |
| `TARGET_ROUTER_API_URL` | No | `http://localhost:8080` | Router API for dashboard |
| `TARGET_ENVOY_URL` | No | — | Envoy proxy URL (playground sends requests here) |
| `TARGET_GRAFANA_URL` | No | — | Grafana URL for embedded view |
| `TARGET_JAEGER_URL` | No | — | Jaeger URL for embedded tracing |
| `TARGET_PROMETHEUS_URL` | No | — | Prometheus URL for embedded metrics |
| `VLLM_SR_NOFILE_LIMIT` | No | `65536` | File descriptor limit for Envoy |
