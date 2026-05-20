"""
Router-backend benchmark harness: OpenVINO vs Candle.

Runs four phases against a single running router instance:
  1. classifier-only, single-stream (concurrency=1)
  2. classifier-only, concurrent (concurrency=N)
  3. end-to-end, single-stream
  4. end-to-end, concurrent

Each prompt from the corpus is sent exactly once per phase -> cache-hit rate ~ 0.

Outputs per-request rows to results/<label>.csv and prints summary tables.

Usage:
  # Run with the OV config first
  python run_bench.py --label ov --corpus corpus.jsonl --concurrency 16
  # Restart router with bench-candle.yaml, then:
  python run_bench.py --label candle --corpus corpus.jsonl --concurrency 16
  # Compare:
  python compare.py results/ov.csv results/candle.csv
"""
import argparse
import asyncio
import csv
import json
import statistics
import time
from pathlib import Path

import httpx

ROUTER_API = "http://localhost:8080"
ENVOY_API = "http://localhost:8801"


async def post(client: httpx.AsyncClient, url: str, payload: dict, timeout: float):
    t0 = time.perf_counter()
    try:
        r = await client.post(url, json=payload, timeout=timeout)
        latency_ms = (time.perf_counter() - t0) * 1000
        body = r.json() if r.status_code == 200 else None
        return r.status_code, latency_ms, body
    except Exception as e:
        return -1, (time.perf_counter() - t0) * 1000, {"error": str(e)}


async def run_phase(
    phase: str,
    rows: list[dict],
    concurrency: int,
    client: httpx.AsyncClient,
    out_writer: csv.writer,
) -> list[float]:
    sem = asyncio.Semaphore(concurrency)
    latencies: list[float] = []

    async def one(row: dict, idx: int):
        async with sem:
            if phase.startswith("classify"):
                url = f"{ROUTER_API}/api/v1/classify/intent"
                payload = {"text": row["prompt"]}
                timeout = 30.0
            else:
                url = f"{ENVOY_API}/v1/chat/completions"
                payload = {
                    "model": "auto",
                    "messages": [{"role": "user", "content": row["prompt"]}],
                    "max_tokens": 16,
                    "temperature": 0.0,
                }
                timeout = 120.0
            status, latency_ms, body = await post(client, url, payload, timeout)
            predicted = ""
            if body and isinstance(body, dict):
                # /classify/intent returns {"category": "...", ...}
                predicted = body.get("category", "") or body.get("predicted_category", "") or ""
                if not predicted and "choices" in body:
                    predicted = body.get("model", "")
            out_writer.writerow([
                phase, concurrency, row["kind"], row.get("category", ""),
                predicted, f"{latency_ms:.3f}", status,
            ])
            if status == 200:
                latencies.append(latency_ms)
            if idx % 50 == 0:
                print(f"  [{phase}] {idx}/{len(rows)}", end="\r", flush=True)

    t0 = time.perf_counter()
    await asyncio.gather(*(one(r, i) for i, r in enumerate(rows)))
    wall = time.perf_counter() - t0
    print(f"  [{phase}] done in {wall:.1f}s | "
          f"sent={len(rows)} ok={len(latencies)} rps={len(latencies)/wall:.1f}")
    return latencies


def pct(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    xs2 = sorted(xs)
    k = max(0, min(len(xs2) - 1, int(p / 100 * len(xs2))))
    return xs2[k]


def summarize(label: str, name: str, latencies: list[float]):
    if not latencies:
        print(f"  {name:<30} no successful requests")
        return
    print(f"  {name:<30} "
          f"n={len(latencies):4d} "
          f"p50={statistics.median(latencies):7.2f}ms "
          f"p95={pct(latencies, 95):7.2f}ms "
          f"p99={pct(latencies, 99):7.2f}ms "
          f"mean={statistics.mean(latencies):7.2f}ms")


async def warmup(client: httpx.AsyncClient, prompts: list[str]):
    print(f"Warmup: {len(prompts)} requests to /classify/intent")
    for p in prompts:
        await post(client, f"{ROUTER_API}/api/v1/classify/intent", {"text": p}, 30.0)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="ov or candle")
    ap.add_argument("--corpus", default="corpus.jsonl")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--skip-e2e", action="store_true",
                    help="Skip end-to-end phase (no vLLM running)")
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()

    rows = [json.loads(line) for line in Path(args.corpus).read_text().splitlines() if line.strip()]
    print(f"Loaded {len(rows)} prompts from {args.corpus}")

    Path(args.results_dir).mkdir(exist_ok=True)
    csv_path = Path(args.results_dir) / f"{args.label}.csv"
    print(f"Writing per-request results to {csv_path}")

    summary: dict[str, list[float]] = {}

    # trust_env=False stops httpx from reading https_proxy/http_proxy env vars.
    # Corporate proxy returns 403 for localhost otherwise.
    async with httpx.AsyncClient(http2=False, trust_env=False) as client:
        # Health check
        try:
            r = await client.get(f"{ROUTER_API}/health", timeout=5.0)
            print(f"Router /health: {r.status_code} {r.text}")
        except Exception as e:
            print(f"Router /health FAILED: {e}")
            return

        await warmup(client, [r["prompt"] for r in rows[:args.warmup]])

        with csv_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["phase", "concurrency", "kind", "expected_category",
                        "predicted_category", "latency_ms", "status"])

            print("\n=== Phase 1: classifier-only, single-stream ===")
            summary["classify_c1"] = await run_phase("classify", rows, 1, client, w)

            print(f"\n=== Phase 2: classifier-only, concurrency={args.concurrency} ===")
            summary[f"classify_c{args.concurrency}"] = await run_phase(
                "classify", rows, args.concurrency, client, w)

            if not args.skip_e2e:
                print("\n=== Phase 3: end-to-end, single-stream ===")
                summary["e2e_c1"] = await run_phase("e2e", rows, 1, client, w)

                print(f"\n=== Phase 4: end-to-end, concurrency={args.concurrency} ===")
                summary[f"e2e_c{args.concurrency}"] = await run_phase(
                    "e2e", rows, args.concurrency, client, w)

    print(f"\n========== Summary [{args.label}] ==========")
    for name, lats in summary.items():
        summarize(args.label, name, lats)


if __name__ == "__main__":
    asyncio.run(main())
