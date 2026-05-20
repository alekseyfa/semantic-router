"""
Compare two run_bench.py result CSVs side-by-side.

Usage:
  python compare.py results/ov.csv results/candle.csv
"""
import argparse
import csv
import statistics
from collections import defaultdict


def pct(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    return xs[max(0, min(len(xs) - 1, int(p / 100 * len(xs))))]


def load(path):
    """Returns dict: (phase, concurrency, kind) -> list of latency_ms (status==200 only)."""
    buckets = defaultdict(list)
    with open(path) as f:
        for row in csv.DictReader(f):
            if row["status"] != "200":
                continue
            key = (row["phase"], row["concurrency"], row["kind"])
            buckets[key].append(float(row["latency_ms"]))
    return buckets


def fmt(xs):
    if not xs:
        return "          n=0"
    return (f"n={len(xs):4d} "
            f"p50={statistics.median(xs):7.2f} "
            f"p95={pct(xs, 95):7.2f} "
            f"p99={pct(xs, 99):7.2f} "
            f"mean={statistics.mean(xs):7.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a", help="first results csv (e.g. ov)")
    ap.add_argument("b", help="second results csv (e.g. candle)")
    args = ap.parse_args()

    a = load(args.a)
    b = load(args.b)

    keys = sorted(set(a) | set(b))
    a_label = args.a.split("/")[-1].replace(".csv", "")
    b_label = args.b.split("/")[-1].replace(".csv", "")

    print(f"\n{'phase':<12} {'conc':<5} {'kind':<10} | {a_label:<55} | {b_label:<55} | delta_p50")
    print("-" * 165)
    for phase, conc, kind in keys:
        la, lb = a.get((phase, conc, kind), []), b.get((phase, conc, kind), [])
        delta = ""
        if la and lb:
            d = statistics.median(la) - statistics.median(lb)
            sign = "+" if d > 0 else ""
            pct_change = 100 * d / statistics.median(lb) if statistics.median(lb) else 0
            delta = f"{sign}{d:7.2f}ms ({sign}{pct_change:5.1f}%)"
        print(f"{phase:<12} {conc:<5} {kind:<10} | {fmt(la):<55} | {fmt(lb):<55} | {delta}")

    # Aggregate by (phase, concurrency) ignoring kind
    print("\n=== Aggregate (all kinds) ===")
    agg_a = defaultdict(list)
    agg_b = defaultdict(list)
    for k, v in a.items():
        agg_a[(k[0], k[1])].extend(v)
    for k, v in b.items():
        agg_b[(k[0], k[1])].extend(v)
    for k in sorted(set(agg_a) | set(agg_b)):
        la, lb = agg_a.get(k, []), agg_b.get(k, [])
        delta = ""
        if la and lb:
            d = statistics.median(la) - statistics.median(lb)
            sign = "+" if d > 0 else ""
            pct_change = 100 * d / statistics.median(lb) if statistics.median(lb) else 0
            delta = f"{sign}{d:7.2f}ms ({sign}{pct_change:5.1f}%)"
        print(f"{k[0]:<12} {k[1]:<5} {'ALL':<10} | {fmt(la):<55} | {fmt(lb):<55} | {delta}")


if __name__ == "__main__":
    main()
