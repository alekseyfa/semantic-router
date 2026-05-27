"""
Compare two run_bench.py result sets side-by-side with bootstrap-based stats.

Reads the *.summary.json produced by run_bench.py (preferred) and falls back to
the *.csv if the JSON is missing.

Usage:
  python compare.py results/ov-throughput.summary.json results/candle-throughput.summary.json
  python compare.py results/ov-throughput.csv          results/candle-throughput.csv
"""
import argparse
import csv
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path


# ---------- stats ----------

def pct(xs, p):
    if not xs:
        return float("nan")
    xs2 = sorted(xs)
    return xs2[max(0, min(len(xs2) - 1, int(round(p / 100 * (len(xs2) - 1)))))]


def bootstrap_diff(xs_a, xs_b, stat_fn, iters=2000, alpha=0.05, seed=0xCAFE):
    """
    Bootstrap CI for stat(a) - stat(b) using independent resamples.
    Returns (point_estimate, lo, hi). If 0 not in [lo, hi], the diff is significant
    at the (1 - alpha) level.
    """
    if not xs_a or not xs_b:
        return (float("nan"), float("nan"), float("nan"))
    rng = random.Random(seed)
    na, nb = len(xs_a), len(xs_b)
    diffs = []
    for _ in range(iters):
        ra = [xs_a[rng.randrange(na)] for _ in range(na)]
        rb = [xs_b[rng.randrange(nb)] for _ in range(nb)]
        diffs.append(stat_fn(ra) - stat_fn(rb))
    diffs.sort()
    lo = diffs[int(alpha / 2 * iters)]
    hi = diffs[min(iters - 1, int((1 - alpha / 2) * iters))]
    point = stat_fn(xs_a) - stat_fn(xs_b)
    return (point, lo, hi)


# ---------- loaders ----------

def load_summary(path: Path) -> dict:
    """
    Return:
      {
        "label": str,
        "phases": {concurrency:int -> {p50, p95, p99, rps, lat:[...]}}
      }
    The lat list is read from the CSV next to the summary if present.
    """
    if path.suffix == ".json":
        meta = json.loads(path.read_text())
        label = meta.get("label", path.stem.replace(".summary", ""))
        out = {"label": label, "phases": {}}
        for ph in meta["phases"]:
            out["phases"][int(ph["concurrency"])] = {
                "p50":   ph["p50_ms"],
                "p95":   ph["p95_ms"],
                "p99":   ph["p99_ms"],
                "rps":   ph["rps"],
                "n":     ph["n_steady_ok"],
                "wall":  ph["wall_steady_s"],
                "p50_ci": tuple(ph.get("p50_ci95", (float("nan"), float("nan")))),
                "p95_ci": tuple(ph.get("p95_ci95", (float("nan"), float("nan")))),
                "p99_ci": tuple(ph.get("p99_ci95", (float("nan"), float("nan")))),
                "lat": [],
            }
        # Also load raw latencies from the CSV for cross-run bootstrap.
        csv_path = path.with_suffix("").with_suffix(".csv")
        if not csv_path.exists():
            csv_path = path.parent / (label + ".csv")
        if csv_path.exists():
            for c, lats in _load_lat_from_csv(csv_path).items():
                if c in out["phases"]:
                    out["phases"][c]["lat"] = lats
        return out

    # Fallback: CSV only -- compute everything ourselves.
    label = path.stem
    out = {"label": label, "phases": {}}
    by_c = _load_lat_from_csv(path)
    for c, lats in by_c.items():
        out["phases"][c] = {
            "p50": pct(lats, 50), "p95": pct(lats, 95), "p99": pct(lats, 99),
            "rps": float("nan"),  # cannot recover wall time from CSV alone
            "n": len(lats), "wall": float("nan"),
            "p50_ci": (float("nan"), float("nan")),
            "p95_ci": (float("nan"), float("nan")),
            "p99_ci": (float("nan"), float("nan")),
            "lat": lats,
        }
    return out


def _load_lat_from_csv(path: Path) -> dict:
    by_c: dict[int, list[float]] = defaultdict(list)
    with path.open() as f:
        rdr = csv.DictReader(f)
        # Auto-detect new schema (window column) vs old schema (phase column).
        for row in rdr:
            if row.get("status") != "200":
                continue
            window = row.get("window")
            if window is not None:
                if window != "steady":
                    continue
            else:
                # old schema -- include classify rows only
                if row.get("phase") != "classify":
                    continue
            c = int(row["concurrency"])
            by_c[c].append(float(row["latency_ms"]))
    return dict(by_c)


# ---------- printing ----------

def fmt_ci(lo: float, hi: float) -> str:
    if any(map(lambda v: v != v, (lo, hi))):  # NaN
        return ""
    return f"[{lo:6.2f},{hi:6.2f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a", help="first results file (.summary.json or .csv)")
    ap.add_argument("b", help="second results file (.summary.json or .csv)")
    ap.add_argument("--bootstrap-iters", type=int, default=2000)
    args = ap.parse_args()

    A = load_summary(Path(args.a))
    B = load_summary(Path(args.b))

    label_a, label_b = A["label"], B["label"]
    print(f"\nA = {label_a}")
    print(f"B = {label_b}")
    print(f"Significance: bootstrap 95% CI of (A - B); * if 0 not in CI.\n")

    print(f"{'c':>4}  {'metric':<8}  "
          f"{label_a + ' [95% CI]':<28}  "
          f"{label_b + ' [95% CI]':<28}  "
          f"{'Δ (A-B)':<10}  {'95% CI of Δ':<20}  sig")
    print("-" * 120)

    cs = sorted(set(A["phases"]) | set(B["phases"]))
    for c in cs:
        a, b = A["phases"].get(c), B["phases"].get(c)
        if not a or not b:
            print(f"{c:>4}  (only one side has data; skipping)")
            continue

        for metric, key, percentile in [
            ("p50",  "p50",  50),
            ("p95",  "p95",  95),
            ("p99",  "p99",  99),
        ]:
            ci_a = a[f"{key}_ci"]
            ci_b = b[f"{key}_ci"]
            point, lo, hi = bootstrap_diff(
                a["lat"], b["lat"],
                lambda xs, p=percentile: pct(xs, p),
                iters=args.bootstrap_iters,
            )
            sig = "*" if (lo == lo and hi == hi and not (lo <= 0 <= hi)) else " "
            print(f"{c:>4}  {metric:<8}  "
                  f"{a[key]:7.2f} {fmt_ci(*ci_a):<20}  "
                  f"{b[key]:7.2f} {fmt_ci(*ci_b):<20}  "
                  f"{point:+7.2f}ms  {fmt_ci(lo, hi):<20}  {sig}")

        # Throughput row (no CI -- single wall-clock measurement).
        if a["rps"] == a["rps"] and b["rps"] == b["rps"]:
            d = a["rps"] - b["rps"]
            ratio = a["rps"] / b["rps"] if b["rps"] else float("nan")
            print(f"{c:>4}  {'rps':<8}  "
                  f"{a['rps']:7.1f} req/s  (wall {a['wall']:5.1f}s, n={a['n']})           "
                  f"{b['rps']:7.1f} req/s  (wall {b['wall']:5.1f}s, n={b['n']})           "
                  f"{d:+7.1f}     ratio A/B = {ratio:5.2f}x")
        print()


if __name__ == "__main__":
    main()
