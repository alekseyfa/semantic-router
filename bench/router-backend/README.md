# Router-Backend Bench: OpenVINO vs Candle

Apples-to-apples latency / throughput comparison of the router's two
classifier backends. The single varying knob is `use_openvino` in the
config; everything else is held constant.

## What's measured

Phase 1 — **classifier-only** (router REST API on :8080)
  Hits `POST /api/v1/classify/intent`. Bypasses Envoy and vLLM entirely.
  Pure router-internal cost: tokenizer + classifier + PII + jailbreak.

Phase 2 — **end-to-end** (Envoy on :8801)
  Hits `POST /v1/chat/completions` with `max_tokens=16`. Real production path:
  Envoy -> ext_proc -> router classification -> upstream cluster -> vLLM.

Each phase runs at single-stream (concurrency=1) and concurrent (default 16).

## Workload

`build_corpus.py` produces `corpus.jsonl` of unique prompts:
  - 70% MMLU (balanced across 14 router categories)
  - 15% PII-bearing (`ai4privacy/pii-masking-200k`)
  - 15% jailbreak attempts (`JailbreakBench/JBB-Behaviors`)

Every prompt is sent exactly once -> semantic cache hit rate ~ 0.

## Why both bench configs keep cache + prompt_guard ON

The `use_openvino` toggle affects the intent classifier, the PII classifier,
*and* the prompt-guard jailbreak classifier. Disabling prompt_guard
under-measures the OV-vs-Candle delta. Cache stays on but the corpus is
unique so it doesn't fire.

## Hardware allocation (this box)

| Device          | Process                           |
|-----------------|-----------------------------------|
| CPU             | Router (both runs, same cores)    |
| GPU.0 (xpu0)    | vLLM #1: Llama-3.2-3B on :11434  |
| GPU.1 (xpu1)    | vLLM #2: Qwen2.5-7B on :11435   |

The classifier runs on CPU for both backends — Candle has no XPU path here,
so OV-CPU vs Candle-CPU is the only fair comparison.

## How to run

```bash
# 1) Build corpus once
python build_corpus.py --total 500 --seed 42

# 2) Start router with bench-openvino.yaml, then:
python run_bench.py --label ov --concurrency 16

# 3) Restart router with bench-candle.yaml, then:
python run_bench.py --label candle --concurrency 16

# 4) Compare
python compare.py results/ov.csv results/candle.csv
```

Pin the router to identical CPU cores in both runs (e.g.
`taskset -c 0-15 ./bin/router ...`) to remove core-contention noise.

## Decision criteria

OV is the better backend if it shows **lower p95 in classifier-only single
stream** AND **equal or higher RPS at the chosen concurrency**. The
end-to-end numbers should track but with smaller relative delta (LLM cost
dominates absolute time).
