# CLAUDE.md — Project Context for AI Agents

This file is read automatically by Claude Code at the start of every session.
It captures non-obvious decisions, known failure modes, and fixes made to this codebase
that are not derivable from reading the source code alone.

---

## Project in One Paragraph

vLLM Semantic Router is an intelligent request-routing proxy for LLM inference. Envoy
receives HTTP requests, forwards them through a gRPC ExtProc filter to the Go router
binary (which classifies the request and sets an upstream destination header), then Envoy
forwards to the selected LLM backend. A React+Go dashboard provides monitoring and a
playground UI. The router binary, Envoy, and the dashboard are **three separate
processes** — none of them run inside a single container together.

---

## Repository Layout

```
config/config.yaml          config for the local bin/router binary — v0.2 format
config/envoy.yaml           Envoy proxy config; vllm_dynamic_cluster → localhost:11434
config.yaml                 config for `vllm-sr serve` (Docker) — MUST be v0.3 format
dashboard/backend/          Go backend for the web UI (port 8702)
dashboard/frontend/         React frontend, built to static files served by the Go backend
scripts/start-envoy.sh      starts Envoy via func-e; proxy vars inherited from environment
scripts/start-dashboard.sh  starts the Go dashboard backend
src/vllm-sr/                Python CLI (`vllm-sr`) that manages Docker containers
openvino-binding/           Go + C++ OpenVINO binding for local CPU/GPU inference
```

---

## Config Format: v0.2 vs v0.3 (CRITICAL)

The `vllm-sr:latest` Docker image (≥ May 2026) enforces **v0.3 config format**.
The local `bin/router` binary (built from this repo's source) uses the older v0.2 format.
**They are not compatible — mixing them causes immediate fatal errors or silent misbehaviour.**

### v0.3 format (required by `vllm-sr serve` / Docker image):
```yaml
routing:
  modelCards:
    - name: "my-model"
  decisions:                          # MUST be under routing, NOT at top level
    - name: "default-route"
      priority: 100
      rules: {operator: "AND", conditions: []}
      modelRefs:
        - model: "my-model"
providers:
  models:
    - name: "my-model"
      backend_refs:                   # new field — replaces endpoints[]
        - name: "primary"
          weight: 100
          endpoint: "localhost:11434" # host:port ONLY — no scheme, no path suffix
```

### v0.3 mistakes that cause silent failures:

| Mistake | Symptom |
|---------|---------|
| `decisions` at config top level | `Unknown field` warning; router starts with zero decisions; every request returns 503 with `selected_model: null` in Envoy access log |
| `endpoint: "host:port/v1"` (with path) | ORIGINAL_DST cluster ignores it; `upstream_host: null` |
| `providers.models[].endpoints[]` (v0.2 field) | Fatal `deprecated config fields are no longer supported` error on startup |
| `providers.default_model` at top level | Same fatal error as above |

---

## Envoy Cluster: STATIC, not ORIGINAL_DST

`config/envoy.yaml` uses a **STATIC cluster** (`vllm_dynamic_cluster`) pointing to
`127.0.0.1:11434`. This is a deliberate change from the original `ORIGINAL_DST` design.

**Why ORIGINAL_DST does not work here:**
Envoy's `ORIGINAL_DST` cluster with `use_http_header: true` only works in L4 transparent
proxy mode. When Envoy acts as a terminating HTTP proxy (which is this setup), cluster
selection happens *before* the ext_proc filter returns its response — so the
`x-vsr-destination-endpoint` header set by the router's ExtProc is always read too late.

Symptom when using ORIGINAL_DST incorrectly:
```
warning: original_dst_load_balancer: No downstream connection or no original_dst.
```
Access log shows `upstream_host: null` even though `selected_model` is correctly set.

**To change the LLM backend port:** edit `port_value` in `vllm_dynamic_cluster`
`load_assignment` inside `config/envoy.yaml`, then restart Envoy.

---

## Port Map

| Port  | Service                         | Notes |
|-------|---------------------------------|-------|
| 8702  | Dashboard UI (local Go process) | Port 8700 is reserved by `vllm-sr-container` but unused; use 8702 |
| 8801  | Envoy proxy — LLM API endpoint  | Routes to `vllm_dynamic_cluster` |
| 8080  | Router REST API / health        | `GET /health` → `{"status":"healthy"}` |
| 9190  | Router Prometheus metrics       | `GET /metrics` |
| 50051 | Router gRPC ExtProc             | Envoy → router; not accessed directly |
| 19000 | Envoy admin                     | `GET /ready` → `LIVE` |
| 11434 | vLLM backend (example)         | Configurable; matches `vllm_dynamic_cluster` port |
| 3000  | Grafana                         | admin / admin |
| 9090  | Prometheus UI                   | |
| 16686 | Jaeger UI                       | |

---

## Intel GPU Setup (Intel Arc / Xe architecture)

- Kernel **6.17+** is required for the `xe` driver to recognise Arc Battlemage (device id
  `0xE20C` and related). Kernel 6.8 ships a `xe.ko` that does not list these device ids.
- After booting 6.17+, `/dev/dri/renderD128` (or similar) appears for the dGPU.
- OpenVINO reports the dGPU as `GPU.1`; the iGPU (if present) is `GPU.0`.
- Verify: `python3 -c "import openvino as ov; c=ov.Core(); [print(d, c.get_property(d,'FULL_DEVICE_NAME')) for d in c.available_devices]"`

### Running vLLM on Intel Arc with `intel/vllm:0.17.0-xpu`

Key flags and env vars for this image version:

```bash
RENDER_GID=$(stat -c '%g' /dev/dri/renderD128)
VIDEO_GID=$(stat -c '%g' /dev/dri/card0)   # card node for the dGPU (card0 on kernel 6.17+ single dGPU)

docker run -d --name vllm-xpu --restart unless-stopped --init \
  --device /dev/dri/renderD128 \
  --device /dev/dri/card0 \
  --group-add "$RENDER_GID" \
  --group-add "$VIDEO_GID" \
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
    --port 8000 --host 0.0.0.0 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.60 \
    --enforce-eager \
    --enable-auto-tool-choice \
    --tool-call-parser hermes
```

**Known quirks for `intel/vllm:0.17.0-xpu`:**
- `--device xpu` CLI flag does **not exist** in this version — XPU is auto-detected
- `ZE_AFFINITY_MASK=0` selects the first Level Zero device (the dGPU)
- `--init` is required — without it a DRM abort leaves a zombie container that cannot be killed
- `--gpu-memory-utilization 0.60` — 0.85 causes a DRM abort during KV cache allocation (12.9 GiB request exceeds 9.4 GB VRAM)
- `--enforce-eager` — skips torch.compile which emits `sycl_arch not recognized` on Battlemage
- `--enable-auto-tool-choice --tool-call-parser hermes` — needed for Dashboard Playground (`tool_choice: "auto"`)
- First startup takes ~2 min: model download + XPU JIT compile cache build
- Subsequent `docker start vllm-xpu` takes ~60 s (cache is reused, no torch.compile)
- `--device` Docker flag alone is insufficient — `--group-add` for both `render` and
  `video` GIDs is required or `torch.xpu.is_available()` returns `False`
- If vLLM crashes with a DRM abort, the GPU enters a bad state — reboot is required to recover

**Model fit guide for 9–10 GB VRAM at BF16 (2 bytes/param):**

| Model | Size | Fits? |
|-------|------|-------|
| Qwen/Qwen2.5-3B-Instruct | ~6 GB | Yes, ~3 GB headroom |
| Qwen/Qwen2.5-7B-Instruct | ~14 GB | No |
| microsoft/Phi-3-mini-4k-instruct | ~7 GB | Marginal |

---

## vllm-sr CLI Modifications

The Python CLI lives in `src/vllm-sr/`. Changes made to make it work in proxied
environments and with the lean container image:

### `src/vllm-sr/cli/commands/runtime_support.py` — proxy passthrough
`PASSTHROUGH_ENV_RULES` was extended with:
```python
("http_proxy", False),
("https_proxy", False),
("HTTP_PROXY", False),
("HTTPS_PROXY", False),
("no_proxy", False),
("NO_PROXY", False),
```
Without this, the router container cannot reach HuggingFace in proxied networks and
crashes during model download with `LocalEntryNotFoundError`.

### `config/envoy.yaml` — ext_authz bypass for local dev
`failure_mode_allow` on the `ext_authz` filter changed `false` → `true`.
Without this, all requests return 403 when Authorino (port 50052) is not running.

---

## Dashboard Backend Modifications

### `dashboard/backend/handlers/status.go`
- `StatusHandler` signature extended: `StatusHandler(routerAPIURL, configDir, envoyURL string)`
- When the Docker container log scan misses Envoy (because the lean image has no
  supervisord), falls back to `GET http://localhost:19000/ready` (Envoy admin port)
- Dashboard always self-reports as `running` when it is serving the response

### `dashboard/backend/router/router.go`
- Call site updated: `handlers.StatusHandler(cfg.RouterAPIURL, cfg.ConfigDir, cfg.EnvoyURL)`

---

## OpenVINO Binding

### `openvino-binding/CMakeLists.txt` — tokenizer discovery fix
Tokenizer discovery (`openvino_tokenizers`) ran only inside the `if(NOT OpenVINO_FOUND)`
fallback branch. When `find_package(OpenVINO)` succeeded via `OpenVINO_DIR`, the block
was skipped entirely and the library was never linked.

Fix: moved `find_package(Python3)` and the tokenizer probe to run unconditionally after
`OpenVINO_FOUND` is confirmed.

### Test model naming mismatch
`optimum-cli export openvino` writes `openvino_tokenizer.xml` but the C++ tokenizer
loader expects `tokenizer.xml`. Resolution: create symlinks in each test model directory:
```bash
ln -sf openvino_tokenizer.xml test_models/<model>/tokenizer.xml
ln -sf openvino_tokenizer.bin test_models/<model>/tokenizer.bin
```

### Running OpenVINO tests
```bash
source .venv/bin/activate
export OpenVINO_DIR=$(python3 -c \
  "import openvino, os; print(os.path.join(os.path.dirname(openvino.__file__), 'cmake'))")
cd openvino-binding
CGO_ENABLED=1 go test -v -timeout 10m -tags cgo
```

---

## Starting Everything Locally (quick reference)

```bash
cd /path/to/semantic-router
source .venv/bin/activate

# 1. Router + observability stack
#    Ctrl-C after "vLLM Semantic Router is running!" — containers keep running
HF_TOKEN="${HF_TOKEN}" vllm-sr serve --config config.yaml

# 2. vLLM inference backend (Intel Arc or other)
docker start vllm-xpu          # fast restart of existing container
# or create fresh — see GPU Setup section above

# 3. Envoy (inherits proxy vars from shell environment)
> /tmp/envoy.log
nohup scripts/start-envoy.sh &
disown $!

# 4. Dashboard
> /tmp/dashboard.log
nohup scripts/start-dashboard.sh >> /tmp/dashboard.log 2>&1 &
disown $!
```

All services healthy: `curl http://localhost:8702/api/status` → `"overall":"healthy"`.

Diagnose individual components:
```bash
curl http://localhost:8080/health   # router
curl http://localhost:19000/ready   # envoy admin → LIVE
curl http://localhost:11434/health  # vllm backend
```

---

## Known Limitations

| Issue | Symptom | Root Cause |
|-------|---------|------------|
| `vllm-sr:latest` with v0.2 config | Fatal startup error | `providers.default_model` and `endpoints[]` are deprecated |
| `decisions` at top level in v0.3 config | 503 on every request; `selected_model: null` in Envoy log | v0.3 router silently ignores unknown top-level field |
| Envoy ORIGINAL_DST cluster with ext_proc | `upstream_host: null`; `No downstream connection` warning | Cluster selection precedes ext_proc response in terminating proxy mode |
| `--device xpu` argument | `unrecognized arguments` error | Flag does not exist in `intel/vllm:0.17.0-xpu`; use `ZE_AFFINITY_MASK` |
| Dashboard on port 8700 | Connection reset | `vllm-sr-container` binds 8700 but nothing listens; use 8702 |
| OpenVINO tokenizers not linked | Build succeeds but runtime fails to load tokenizer | CMakeLists only searched for tokenizers in the `find_package` fallback path |
| `nohup cmd &` backgrounding fails in some shells | Process exits immediately | Use `nohup cmd & disown $!` — plain `&` can be killed by the shell in certain terminal environments |

---

## Karpathy Skills — Coding Principles

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

### 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

### 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
