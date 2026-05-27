# Router Classifier Bench: OpenVINO vs Candle on Xeon

Fair, reproducible head-to-head between the router's two CPU classifier
backends — **OpenVINO** and **Candle** — on the production Xeon platform.

The single varying knob across runs is `use_openvino` in the router config.
Everything else (corpus, concurrency, NUMA pin, thread budget, warmup) is
held identical so the delta is attributable to the backend.

> **End-to-end is intentionally out of scope.** vLLM dwarfs router cost; the
> e2e number tells you nothing about classifier perf. We measure the router's
> `/api/v1/classify/intent` endpoint only.

---

## What is measured

`POST /api/v1/classify/intent` on the router REST API (port 8080), bypassing
Envoy and any LLM backend. The path inside the router is:

```
HTTP -> JSON parse -> tokenizer -> intent classifier
                                -> PII classifier
                                -> jailbreak classifier (prompt_guard)
                                -> semantic-cache miss path -> JSON response
```

The `use_openvino` toggle flips **all three** classifiers — intent, PII,
prompt-guard. That's why both bench configs keep `prompt_guard` and the
classifier on; turning anything off would under-measure the OV-vs-Candle
delta.

`semantic_cache.enabled = true` in both configs but the corpus is unique-per-phase
so cache-hit rate ≈ 0.

---

## Methodology

### 1. Single-NUMA pin (router) on a 6-NUMA Xeon
The host is a 2-socket Xeon 6972P, 192 physical / 384 logical cores, 6 NUMA
nodes (SNC=3 per socket), node-to-node distance 10→26.

The router is launched under `numactl --cpunodebind=N --membind=N` so its
threads and its weight pages stay on one NUMA node (32 phys cores by default).
This:
- removes remote-memory tax (random tens of % noise),
- generalises results to per-NUMA scaling units,
- keeps the comparison about **algorithms**, not random Linux scheduler decisions.

### 2. Client on a far NUMA node
The bench client (this Python harness) is pinned to a **different** NUMA node
(default 5, distance 26 from node 0), so it **cannot** steal CPUs from the
router or share LLC. Without this, Python's asyncio loop would happily land on
cores 0–15 and silently distort results.

### 3. Explicit thread budgets (no auto-detection)
OpenVINO and Candle both auto-derive parallelism from `/proc/cpuinfo`. On this
host that's 384 logical CPUs — both libraries would happily oversubscribe.
We pin them to identical budgets:

| Mode         | OV streams | OV threads/stream | Rayon threads | Optimises |
|--------------|------------|-------------------|---------------|-----------|
| `latency`    | 1          | NODE_PHYS         | NODE_PHYS     | low-load p50/p99 |
| `throughput` | NODE_PHYS  | 1                 | NODE_PHYS     | saturation QPS  |

Both modes give each backend the *same* total CPU budget — `NODE_PHYS` cores
on the router's NUMA node. `GOMAXPROCS` is also set to `NODE_PHYS` so the Go
runtime doesn't oversubscribe.

### 4. Closed-loop concurrency sweep with steady-state RPS
One phase per concurrency point: a fixed pool of N async workers consumes
`warmup + n_requests` prompts back-to-back. Within a phase:
- The first `warmup` requests are **discarded** for every metric.
- Throughput is the **wall-clock** rate over the steady window, not
  `concurrency / mean_latency`.
- Per-request latencies are recorded; p50/p95/p99 come from the same window.

### 5. Bootstrap 95% CIs and significance flagging
Per-metric CIs come from 2000-iter resamples (`run_bench.py`). `compare.py`
also bootstraps the **difference** A−B for p50/p95/p99; if 0 ∉ CI, the result
is flagged significant at the 95% level. This kills the "is 3% real or noise"
debate.

### 6. Identical workload, both backends
Same `corpus.jsonl` (5000 unique prompts: 70% MMLU / 15% PII / 15% jailbreak),
same shuffle seed, same per-phase warmup, same concurrency list.

---

## How to run

```bash
# 1) One-time: build the corpus
python build_corpus.py --total 5000 --seed 42

# 2) Throughput sweep, OpenVINO
AI_BINDING=openvino MODE=throughput LABEL=ov-throughput \
  bench/router-backend/run_phase.sh

# 3) Throughput sweep, Candle (router restarts under the hood)
AI_BINDING=candle   MODE=throughput LABEL=candle-throughput \
  bench/router-backend/run_phase.sh

# 4) Compare with bootstrap CIs and significance flags
python compare.py results/ov-throughput.summary.json \
                  results/candle-throughput.summary.json

# 5) Plot
python plot.py    results/ov-throughput.summary.json \
                  results/candle-throughput.summary.json
```

Optional: also run `MODE=latency` to characterise the single-user case.
The two modes answer different questions and should be reported separately.

### Knobs (env vars)

Set on `run_phase.sh`:

| Var          | Default                | Meaning |
|--------------|------------------------|---------|
| `AI_BINDING` | `openvino`             | `openvino` \| `candle` |
| `MODE`       | `throughput`           | `latency` \| `throughput` (router thread topology) |
| `LABEL`      | `${AI_BINDING}-${MODE}`| CSV/JSON output stem |
| `ROUTER_NODE`| `0`                    | NUMA node the router pins to |
| `CLIENT_NODE`| `5`                    | NUMA node the bench client pins to |
| `CONCURRENCY`| `1,2,4,8,16,32,64`     | Comma-separated sweep |
| `N_REQUESTS` | `5000`                 | Steady-state requests per concurrency point |
| `WARMUP_REQ` | `200`                  | Per-phase warmup requests (discarded) |

Set on `run_bench.py` directly when invoking by hand:

| Flag              | Default | Meaning |
|-------------------|---------|---------|
| `--label`         | (req)   | Output stem |
| `--corpus`        | `corpus.jsonl` | |
| `--concurrency`   | `1,2,4,8,16,32,64` | comma list |
| `--n-requests`    | `5000`  | |
| `--warmup`        | `200`   | per-concurrency phase warmup |
| `--global-warmup` | `200`   | once before the sweep |
| `--bootstrap-iters` | `2000` | for CIs in summary JSON |

---

## Outputs

`results/<label>.csv` — one row per request:
```
concurrency,window,kind,expected_category,predicted_category,latency_ms,status
```
`window` ∈ {`warmup`, `steady`}. Only `steady` rows feed metrics.

`results/<label>.summary.json` — per-(concurrency) aggregates with bootstrap
CIs, wall-clock RPS, status counts.

`results/plots/`:
- `latency_vs_c.png` — p50/p95/p99 vs c, per backend (with CI bands).
- `rps_vs_c.png` — steady-state QPS vs c. Saturation knee shown with ◯.
- `latency_vs_rps.png` — p99 envelope. Lower-right is better.
- `latency_cdf.png` — full distributions per concurrency.
- `latency_bars.png` — kept for the deck.

---

## Decision criteria

OpenVINO is the better backend on this Xeon if **all three** hold:

1. **Lower p99 at low load** (`MODE=latency`, c=1). Tail-latency-dominated
   single-user experience.
2. **Higher peak RPS** (`MODE=throughput`). Saturation throughput.
3. **Lower p99 at the same RPS** somewhere along the latency-throughput
   envelope (the curve in `latency_vs_rps.png`). This is the metric a real
   capacity planner uses: at SLO p99 ≤ X ms, which backend serves more QPS?

If (1) and (2) point opposite ways, report both modes with their numbers and
let the deployment SLO decide.

---

## Why these defaults are sane (and what you can change)

| Default | Why |
|---------|-----|
| Router on node 0, client on node 5 | Maximum NUMA distance on this host (26). No shared CPU set, no shared L3. |
| `MODE=throughput` for the headline | Production routers see concurrent traffic; throughput-mode parallelism (streams) is what OV ships in production. |
| `c ∈ {1, 2, 4, 8, 16, 32, 64}` | Geometric sweep brackets the saturation knee for both backends within node 0's 32 phys cores. |
| `n_requests=5000` per phase | At c=64 that's ~78 req/worker; at c=1 it's 5000 samples — enough for stable p99. |
| `warmup=200` per phase | Empirically covers OV's per-phase JIT-cache warmup. |
| `bootstrap_iters=2000` | Stable CI to ~0.5 ms granularity at our sample sizes. |

If you need to sweep further (`c=128, 256`), one NUMA node will saturate;
either move to `--cpunodebind=0,1` (cross-NUMA, document the change) or run
N independent routers, one per node, behind a fan-out client.

---

## Repro-tier sanity checks

Print these before each run, save them with the artifacts:
- `numactl --hardware` (NUMA topology)
- `cat /proc/cmdline` (kernel boot flags — `mitigations=`, `isolcpus=`)
- `lscpu | grep MHz` (HWP / turbo state)
- `uname -r` and OpenVINO / Candle versions
- The router log header (echoes `OV_NUM_STREAMS`, `RAYON_NUM_THREADS`,
  `GOMAXPROCS`, NUMA node, mode).

The router's startup log line is the audit trail for thread topology.
