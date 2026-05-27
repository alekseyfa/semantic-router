"""
Plot router-backend bench results (classifier-only, concurrency sweep).

Reads *.summary.json (preferred) or *.csv files.

Charts written to --out dir:
  rps_vs_c.png      — grouped RPS bars per concurrency + OV/Candle speedup ratio
  latency_bars.png  — p50 / p99 grouped bars per concurrency + latency ratio

Usage:
  python plot.py openvino-throughput.summary.json candle-throughput.summary.json
  python plot.py results/ov.csv results/candle.csv
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ── colours ───────────────────────────────────────────────────────────────────
_COLORS = {
    "openvino": "#1565C0",   # blue
    "ov":       "#1565C0",
    "candle":   "#E65100",   # orange
    "ca":       "#E65100",
}
_FALLBACK = ["#1565C0", "#E65100", "#2E7D32", "#6A1B9A"]


def _color(label: str, idx: int) -> str:
    return _COLORS.get(label.lower().split("-")[0], _FALLBACK[idx % len(_FALLBACK)])


# ── shared style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.titleweight": "bold",
    "axes.titlepad":    14,
    "axes.labelsize":   10.5,
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "legend.frameon":   False,
    "legend.fontsize":  10,
    "savefig.dpi":      150,
    "savefig.bbox":     "tight",
})


def _style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#d8d8d8")
    ax.spines["bottom"].set_color("#d8d8d8")
    ax.tick_params(colors="#666666", length=3)
    ax.xaxis.label.set_color("#444444")
    ax.yaxis.label.set_color("#444444")
    ax.title.set_color("#1a1a1a")
    ax.yaxis.grid(True, color="#eeeeee", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def _save(fig, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"  ✓  {out}")


# ── loaders ───────────────────────────────────────────────────────────────────
def _pct(xs, p):
    if not xs:
        return float("nan")
    s = sorted(xs)
    return s[max(0, min(len(s) - 1, int(round(p / 100 * (len(s) - 1)))))]


def _csv_lats(path: Path) -> dict:
    by_c = defaultdict(list)
    with path.open() as f:
        for row in csv.DictReader(f):
            if row.get("status") != "200":
                continue
            win = row.get("window") or row.get("phase", "")
            if win not in ("steady", "classify"):
                continue
            by_c[int(row["concurrency"])].append(float(row["latency_ms"]))
    return dict(by_c)


def load_run(path: Path) -> dict:
    path = Path(path)
    if path.suffix == ".json":
        meta  = json.loads(path.read_text())
        label = meta.get("label", path.stem.replace(".summary", ""))
        by_c  = {}
        for ph in meta["phases"]:
            c = int(ph["concurrency"])
            by_c[c] = {
                "p50": ph["p50_ms"],
                "p95": ph["p95_ms"],
                "p99": ph["p99_ms"],
                "rps": ph["rps"],
                "n":   ph["n_steady_ok"],
                "lat": [],
            }
        for candidate in [path.with_suffix(".csv"), path.parent / (label + ".csv")]:
            if candidate.exists():
                for c, lats in _csv_lats(candidate).items():
                    if c in by_c:
                        by_c[c]["lat"] = lats
                break
        return {"label": label, "by_c": by_c}

    label = path.stem
    by_c  = {}
    for c, lats in _csv_lats(path).items():
        by_c[c] = {
            "p50": _pct(lats, 50), "p95": _pct(lats, 95), "p99": _pct(lats, 99),
            "rps": float("nan"), "n": len(lats), "lat": lats,
        }
    return {"label": label, "by_c": by_c}


# ── ratio helpers ─────────────────────────────────────────────────────────────

def _ratio_label(a, b):
    """Return e.g. '3.2×' or '' if either value is missing/zero."""
    if not a or not b or np.isnan(a) or np.isnan(b) or b == 0:
        return ""
    return f"{a / b:.1f}×"


def _annotate_ratio(ax, x_center, y_top, ratio_str, color="#333333"):
    if not ratio_str:
        return
    ax.annotate(
        ratio_str,
        (x_center, y_top),
        textcoords="offset points", xytext=(0, 18),
        ha="center", va="bottom",
        fontsize=10, fontweight="bold", color=color,
        bbox=dict(boxstyle="round,pad=0.25", fc="#f0f4ff", ec="#90a4c8", lw=0.7),
    )


# ── charts ────────────────────────────────────────────────────────────────────

def plot_rps_vs_c(runs, out: Path):
    """Grouped RPS bars per concurrency with OV-vs-Candle speedup labels."""
    cs = sorted({c for r in runs for c in r["by_c"]})
    if not cs:
        return

    n   = len(runs)
    w   = 0.60 / n
    x   = np.arange(len(cs))
    fig, ax = plt.subplots(figsize=(max(10, len(cs) * 1.6), 5.8))

    bar_tops = {}
    for i, run in enumerate(runs):
        rps = [run["by_c"].get(c, {}).get("rps") or 0.0 for c in cs]
        col = _color(run["label"], i)
        off = (i - (n - 1) / 2) * w
        bars = ax.bar(x + off, rps, w, color=col, label=run["label"],
                      zorder=3, edgecolor="white", linewidth=0.6)
        bar_tops[run["label"]] = list(zip(x + off + w / 2, rps))
        for b, v in zip(bars, rps):
            if v > 0:
                ax.annotate(
                    f"{v:.0f}",
                    (b.get_x() + b.get_width() / 2, v),
                    textcoords="offset points", xytext=(0, 3),
                    ha="center", va="bottom",
                    fontsize=8, color=col,
                )

    if n >= 2:
        l0, l1 = list(bar_tops.keys())[:2]
        for i, c in enumerate(cs):
            v0 = bar_tops[l0][i][1]
            v1 = bar_tops[l1][i][1]
            top = max(v0, v1)
            mid_x = (bar_tops[l0][i][0] + bar_tops[l1][i][0]) / 2
            _annotate_ratio(ax, mid_x, top, _ratio_label(v0, v1))

    ymax = max(v for tops in bar_tops.values() for _, v in tops) if bar_tops else 1
    ax.set_ylim(0, ymax * 1.35)
    ax.set_xticks(x)
    ax.set_xticklabels([f"c={c}" for c in cs])
    ax.set_xlabel("Concurrency")
    ax.set_ylabel("Requests / second")
    ax.set_title("Throughput vs Concurrency")
    _style(ax)
    ax.legend(loc="upper left")
    fig.tight_layout()
    _save(fig, out)


def plot_latency_bars(runs, out: Path):
    """Grouped p50 / p99 bars per concurrency with latency-improvement labels."""
    cs = sorted({c for r in runs for c in r["by_c"]})
    if not cs:
        return

    metrics = [("p50", "Median latency  p50 (ms)"), ("p99", "Tail latency  p99 (ms)")]
    fig, axes = plt.subplots(1, 2, figsize=(max(12, len(cs) * 2.2), 5.8), sharey=False)
    n = len(runs)
    w = 0.60 / n
    x = np.arange(len(cs))

    for ax, (key, title) in zip(axes, metrics):
        bar_tops = {}
        for i, run in enumerate(runs):
            ys  = [run["by_c"].get(c, {}).get(key) or 0.0 for c in cs]
            col = _color(run["label"], i)
            off = (i - (n - 1) / 2) * w
            bars = ax.bar(x + off, ys, w, color=col, label=run["label"],
                          zorder=3, edgecolor="white", linewidth=0.6)
            bar_tops[run["label"]] = list(zip(x + off + w / 2, ys))
            for b, v in zip(bars, ys):
                if v > 0:
                    ax.annotate(
                        f"{v:.0f}",
                        (b.get_x() + b.get_width() / 2, v),
                        textcoords="offset points", xytext=(0, 3),
                        ha="center", va="bottom",
                        fontsize=8, color=col,
                    )

        if n >= 2:
            l0, l1 = list(bar_tops.keys())[:2]
            for i in range(len(cs)):
                v0 = bar_tops[l0][i][1]
                v1 = bar_tops[l1][i][1]
                top = max(v0, v1)
                mid_x = (bar_tops[l0][i][0] + bar_tops[l1][i][0]) / 2
                _annotate_ratio(ax, mid_x, top, _ratio_label(v1, v0))

        ymax = max(v for tops in bar_tops.values() for _, v in tops) if bar_tops else 1
        ax.set_ylim(0, ymax * 1.35)
        ax.set_xticks(x)
        ax.set_xticklabels([f"c={c}" for c in cs])
        ax.set_ylabel("ms")
        ax.set_title(title)
        _style(ax)

    axes[0].legend(loc="upper left")
    fig.suptitle("Classifier Latency — OpenVINO vs Candle  ·  ratio = Candle / OV",
                 fontsize=13, fontweight="bold", color="#1a1a1a")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, out)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="*.summary.json or *.csv files")
    ap.add_argument("--out",  default="results/plots", help="output directory")
    args = ap.parse_args()

    runs = [load_run(p) for p in args.inputs]
    out  = Path(args.out)

    print(f"Writing charts to {out}/")
    plot_rps_vs_c(runs,      out / "rps_vs_c.png")
    plot_latency_bars(runs,  out / "latency_bars.png")

    print("\nSummary:")
    cs = sorted({c for r in runs for c in r["by_c"]})
    for c in cs:
        parts = [f"c={c:>3}"]
        for r in runs:
            d = r["by_c"].get(c)
            if d:
                parts.append(
                    f"{r['label']}: p50={d['p50']:6.1f}ms  "
                    f"p99={d['p99']:6.1f}ms  rps={d['rps']:6.1f}"
                )
        print("  " + "  |  ".join(parts))


if __name__ == "__main__":
    main()
