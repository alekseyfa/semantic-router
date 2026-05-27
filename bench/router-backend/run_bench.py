"""
Router-only classifier bench: OpenVINO vs Candle on Xeon.

What this measures
------------------
Hits POST /api/v1/classify/intent on the router REST API (port 8080) directly.
Bypasses Envoy and any LLM. The path exercised inside the router is:

    HTTP -> tokenizer -> intent classifier -> PII classifier -> jailbreak
            classifier (prompt_guard) -> (semantic-cache miss) -> JSON response

The single varying knob across runs is the classifier backend:
    bench-openvino.yaml -> use_openvino: true   (OV CPU runtime)
    bench-candle.yaml   -> use_openvino: false  (Candle/Rust CPU)

Methodology
-----------
- Concurrency sweep: one phase per concurrency point. Wall-clock RPS is
  measured *only* over the steady-state window (after WARMUP requests are
  discarded). RPS is the real wall-clock rate, not concurrency / mean_latency.
- Closed-loop: a fixed pool of N async workers each pull from a shared queue
  and fire requests back-to-back. This matches how a real load generator
  exercises a server and produces the latency-vs-QPS curve.
- Per-phase warmup: the first WARMUP requests are timed but excluded from
  every reported metric. This absorbs JIT, page-faults on weights, and
  TCP slow-start.
- Cache-miss workload: every prompt is sent at most once per phase, and the
  prompt list is shuffled per phase. Semantic cache hit rate ~ 0.

Outputs
-------
- results/<label>.csv: one row per request (lat_ms, status, phase, c, ...)
- results/<label>.summary.json: per-(phase, concurrency) p50/p95/p99/RPS
  including bootstrap 95% CIs.

Usage
-----
    # Sweep with sensible defaults
    python run_bench.py --label ov-throughput --concurrency 1,2,4,8,16,32,64

    # Single concurrency point, more iterations
    python run_bench.py --label ov-c16 --concurrency 16 --n-requests 20000
"""
import argparse
import asyncio
import csv
import json
import random
import statistics
import time
from pathlib import Path

import httpx

ROUTER_API = "http://localhost:8080"


# ---------- HTTP ----------

async def post_classify(client: httpx.AsyncClient, prompt: str, timeout: float):
    t0 = time.perf_counter()
    try:
        r = await client.post(
            f"{ROUTER_API}/api/v1/classify/intent",
            json={"text": prompt},
            timeout=timeout,
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        body = r.json() if r.status_code == 200 else None
        return r.status_code, latency_ms, body
    except Exception as e:
        return -1, (time.perf_counter() - t0) * 1000, {"error": str(e)}


# ---------- one closed-loop phase ----------

async def run_phase(
    concurrency: int,
    rows: list[dict],
    n_requests: int,
    warmup: int,
    client: httpx.AsyncClient,
    out_writer: csv.writer,
) -> dict:
    """
    Closed-loop driver. `concurrency` workers consume a shared queue of
    `n_requests + warmup` requests. Each worker times its own request.
    Wall-clock RPS is computed from the steady-state window only.
    """
    queue: asyncio.Queue = asyncio.Queue()
    total = warmup + n_requests
    # Cycle through the corpus to fill the request queue.
    for i in range(total):
        queue.put_nowait(rows[i % len(rows)])

    # All times are perf_counter() seconds.
    samples_lat: list[float] = []         # ms, steady-state only
    samples_t_start: list[float] = []     # s, steady-state only
    samples_t_end: list[float] = []       # s, steady-state only
    statuses: dict[int, int] = {}
    error_samples: list[str] = []

    completed = 0
    lock = asyncio.Lock()
    phase_t0 = None
    steady_t0 = None
    steady_t_end = None

    async def worker():
        nonlocal completed, phase_t0, steady_t0, steady_t_end
        while True:
            try:
                row = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            t_start = time.perf_counter()
            status, latency_ms, body = await post_classify(client, row["prompt"], 30.0)
            t_end = time.perf_counter()

            predicted = ""
            if body and isinstance(body, dict):
                cls = body.get("classification")
                if isinstance(cls, dict):
                    predicted = cls.get("category", "") or ""

            async with lock:
                if phase_t0 is None:
                    phase_t0 = t_start
                completed += 1
                idx = completed
                statuses[status] = statuses.get(status, 0) + 1
                if idx > warmup:
                    if steady_t0 is None:
                        steady_t0 = t_start
                    steady_t_end = t_end
                    if status == 200:
                        samples_lat.append(latency_ms)
                        samples_t_start.append(t_start)
                        samples_t_end.append(t_end)
                if status != 200 and len(error_samples) < 3 and body:
                    error_samples.append(f"status={status} body={str(body)[:200]}")
                out_writer.writerow([
                    concurrency,
                    "warmup" if idx <= warmup else "steady",
                    row["kind"], row.get("category", ""),
                    predicted, f"{latency_ms:.3f}", status,
                ])
                if idx % 200 == 0:
                    print(f"  [c={concurrency:>3}] {idx}/{total}", end="\r", flush=True)

    workers = [asyncio.create_task(worker()) for _ in range(concurrency)]
    await asyncio.gather(*workers)
    print(" " * 40, end="\r")

    nonok = sum(c for s, c in statuses.items() if s != 200)
    steady_wall = (steady_t_end - steady_t0) if steady_t0 and steady_t_end else 0.0
    rps = len(samples_lat) / steady_wall if steady_wall > 0 else 0.0

    summary = {
        "concurrency": concurrency,
        "n_total": total,
        "n_warmup": warmup,
        "n_steady_ok": len(samples_lat),
        "n_nonok": nonok,
        "statuses": statuses,
        "wall_steady_s": steady_wall,
        "rps": rps,
        "p50_ms": pct(samples_lat, 50),
        "p90_ms": pct(samples_lat, 90),
        "p95_ms": pct(samples_lat, 95),
        "p99_ms": pct(samples_lat, 99),
        "mean_ms": statistics.mean(samples_lat) if samples_lat else float("nan"),
        "stdev_ms": statistics.stdev(samples_lat) if len(samples_lat) > 1 else float("nan"),
        "_lat_ms": samples_lat,            # kept for bootstrap CI in summary file
    }

    print(f"  [c={concurrency:>3}] ok={len(samples_lat):>5}  nonok={nonok:>3}  "
          f"wall={steady_wall:6.2f}s  rps={rps:7.1f}  "
          f"p50={summary['p50_ms']:6.2f}  p95={summary['p95_ms']:6.2f}  p99={summary['p99_ms']:6.2f}")
    for s in error_samples:
        print(f"    error sample: {s}")
    return summary


# ---------- stats ----------

def pct(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    xs2 = sorted(xs)
    k = max(0, min(len(xs2) - 1, int(round(p / 100 * (len(xs2) - 1)))))
    return xs2[k]


def bootstrap_ci(xs: list[float], stat_fn, iters: int = 1000, alpha: float = 0.05,
                 rng: random.Random | None = None) -> tuple[float, float]:
    if not xs:
        return (float("nan"), float("nan"))
    rng = rng or random.Random(0xCAFE)
    n = len(xs)
    samples = []
    for _ in range(iters):
        resample = [xs[rng.randrange(n)] for _ in range(n)]
        samples.append(stat_fn(resample))
    samples.sort()
    lo = samples[int(alpha / 2 * iters)]
    hi = samples[min(iters - 1, int((1 - alpha / 2) * iters))]
    return (lo, hi)


# ---------- warmup ----------

async def warmup_router(client: httpx.AsyncClient, prompts: list[str], n: int):
    """Cheap warmup before the timed sweep: load weights, JIT, page-in."""
    print(f"Warmup: {n} classify calls (untimed)")
    for p in prompts[:n]:
        await post_classify(client, p, 30.0)


# ---------- main ----------

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="results/<label>.csv stem")
    ap.add_argument("--corpus", default="corpus.jsonl")
    ap.add_argument("--concurrency", default="1,2,4,8,16,32,64",
                    help="single int or comma-separated list")
    ap.add_argument("--n-requests", type=int, default=5000,
                    help="steady-state requests per concurrency point")
    ap.add_argument("--warmup", type=int, default=200,
                    help="requests dropped at start of every phase (per concurrency)")
    ap.add_argument("--global-warmup", type=int, default=200,
                    help="requests fired before the sweep begins (untimed)")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--bootstrap-iters", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = []
    with Path(args.corpus).open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"empty corpus: {args.corpus}")
    print(f"Loaded {len(rows)} prompts from {args.corpus}")

    concurrency_list = [int(x) for x in args.concurrency.split(",") if x.strip()]
    print(f"Concurrency sweep: {concurrency_list}")
    print(f"Per phase: warmup={args.warmup}  steady={args.n_requests}  total={(args.warmup + args.n_requests) * len(concurrency_list)}")

    Path(args.results_dir).mkdir(exist_ok=True)
    csv_path = Path(args.results_dir) / f"{args.label}.csv"
    summary_path = Path(args.results_dir) / f"{args.label}.summary.json"
    print(f"Per-request CSV  -> {csv_path}")
    print(f"Summary JSON     -> {summary_path}")

    summaries: list[dict] = []

    # trust_env=False stops httpx from reading https_proxy/http_proxy env vars.
    # Corporate proxy returns 403 for localhost otherwise.
    # Per-host connection pool sized for the largest concurrency point.
    max_conc = max(concurrency_list)
    limits = httpx.Limits(max_connections=max_conc * 2, max_keepalive_connections=max_conc * 2)
    async with httpx.AsyncClient(http2=False, trust_env=False, limits=limits) as client:
        try:
            r = await client.get(f"{ROUTER_API}/health", timeout=5.0)
            print(f"Router /health: {r.status_code} {r.text}")
        except Exception as e:
            print(f"Router /health FAILED: {e}")
            return

        await warmup_router(client, [r["prompt"] for r in rows], args.global_warmup)

        with csv_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["concurrency", "window", "kind", "expected_category",
                        "predicted_category", "latency_ms", "status"])

            for c in concurrency_list:
                print(f"\n=== concurrency = {c} ===")
                # Shuffle a copy so each phase sees a different order.
                phase_rows = rows[:]
                rng.shuffle(phase_rows)
                summary = await run_phase(c, phase_rows, args.n_requests,
                                          args.warmup, client, w)
                summaries.append(summary)

    # ---------- write summary JSON with bootstrap CIs ----------
    print("\n=== Bootstrap CIs (this takes a few seconds) ===")
    out_summaries = []
    for s in summaries:
        lat = s.pop("_lat_ms")
        rng_b = random.Random(args.seed)
        ci_p50 = bootstrap_ci(lat, lambda xs: pct(xs, 50), args.bootstrap_iters, 0.05, rng_b)
        ci_p95 = bootstrap_ci(lat, lambda xs: pct(xs, 95), args.bootstrap_iters, 0.05, rng_b)
        ci_p99 = bootstrap_ci(lat, lambda xs: pct(xs, 99), args.bootstrap_iters, 0.05, rng_b)
        s["p50_ci95"] = list(ci_p50)
        s["p95_ci95"] = list(ci_p95)
        s["p99_ci95"] = list(ci_p99)
        out_summaries.append(s)
        print(f"  c={s['concurrency']:>3}  "
              f"p50 [{ci_p50[0]:6.2f}, {ci_p50[1]:6.2f}]  "
              f"p95 [{ci_p95[0]:6.2f}, {ci_p95[1]:6.2f}]  "
              f"p99 [{ci_p99[0]:6.2f}, {ci_p99[1]:6.2f}]")

    summary_path.write_text(json.dumps({
        "label": args.label,
        "corpus": str(args.corpus),
        "n_prompts": len(rows),
        "n_requests_per_phase": args.n_requests,
        "warmup_per_phase": args.warmup,
        "global_warmup": args.global_warmup,
        "concurrency_sweep": concurrency_list,
        "phases": out_summaries,
    }, indent=2))
    print(f"\nWrote summary -> {summary_path}")

    # Classifier accuracy on the neutral/MMLU rows of the steady window.
    # The handler returns the decision name (e.g. "business_decision",
    # "computer_science_decision"); the corpus carries the bare category
    # ("business", "computer science"). Normalise both before compare.
    def norm_cat(s: str) -> str:
        s = s.strip().lower().replace(" ", "_")
        if s.endswith("_decision"):
            s = s[:-len("_decision")]
        return s

    print("\n=== Classifier accuracy (neutral/MMLU prompts, steady window only) ===")
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        per_c_ok = {}
        per_c_total = {}
        for row in reader:
            if row["window"] != "steady" or row["kind"] != "neutral" or row["status"] != "200":
                continue
            c = row["concurrency"]
            per_c_total[c] = per_c_total.get(c, 0) + 1
            if norm_cat(row["predicted_category"]) == norm_cat(row["expected_category"]):
                per_c_ok[c] = per_c_ok.get(c, 0) + 1
        for c, total in sorted(per_c_total.items(), key=lambda kv: int(kv[0])):
            ok = per_c_ok.get(c, 0)
            print(f"  c={c:>3}: {ok}/{total} = {100*ok/total:.1f}%")


if __name__ == "__main__":
    asyncio.run(main())
