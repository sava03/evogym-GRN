#!/usr/bin/env python3
"""
Per-run robustness plots: one figure per gait metric, each showing the
three parent-child statistics across all 10 improved runs (200-209):

  1. Conservation ratio (PC |Δ| / random |Δ|)  -- reference line at 1
  2. Heritability r (child vs parent)           -- reference line at 0
  3. Similarity-fitness r (|Δ| vs child fitness) -- reference line at 0

Saves one PNG per metric so each can be added to the report separately.
"""
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Large fonts for thesis readability
plt.rcParams.update({
    "font.size": 20,
    "axes.titlesize": 24,
    "axes.labelsize": 22,
    "xtick.labelsize": 17,
    "ytick.labelsize": 17,
    "legend.fontsize": 18,
})

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from parent_child_gait import collect_pairs

BASE = ROOT.parent.parent / "tmp_out" / "evobots" / "evobots"
OUT = ROOT.parent.parent / "tmp_out" / "plots"
OUT.mkdir(parents=True, exist_ok=True)

RUNS = list(range(200, 210))
METRICS = ["displacement", "vel_x_std", "com_y_mean", "stride_regularity"]

# Human-readable display names for the metrics
LABELS = {
    "displacement":      "Displacement",
    "vel_x_std":         "Velocity variability",
    "com_y_mean":        "Body height",
    "stride_regularity": "Stride regularity",
}


def compute_per_run():
    data = {m: {"ratio": [], "herit": [], "simfit": []} for m in METRICS}
    for r in RUNS:
        db = BASE / f"run_{r}" / f"run_{r}"
        df = collect_pairs([str(db)])
        pc = df[df["type"] == "parent-child"]
        rand = df[df["type"] == "random"]
        fit = pc["child_displacement"].values.astype(float)
        for m in METRICS:
            a = pc[m].dropna().values
            b = rand[m].dropna().values
            data[m]["ratio"].append(a.mean() / b.mean())

            pv = pc[f"parent_{m}"].values.astype(float)
            cv = pc[f"child_{m}"].values.astype(float)
            mk = ~(np.isnan(pv) | np.isnan(cv))
            data[m]["herit"].append(stats.pearsonr(pv[mk], cv[mk])[0])

            diff = pc[m].values.astype(float)
            mk2 = ~(np.isnan(diff) | np.isnan(fit))
            data[m]["simfit"].append(stats.pearsonr(diff[mk2], fit[mk2])[0])
    return data


def plot_metric(metric, vals):
    runs = [str(r) for r in RUNS]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    fig.suptitle(LABELS[metric], fontsize=27, fontweight="bold")

    panels = [
        ("ratio",  "Conservation ratio (PC / random)", 1.0, "#3F7CAC"),
        ("herit",  "Heritability $r$",                 0.0, "#5B8C5A"),
        ("simfit", "Similarity--fitness $r$",          0.0, "#C1666B"),
    ]
    for ax, (key, title, ref, color) in zip(axes, panels):
        y = vals[key]
        ax.bar(runs, y, color=color, edgecolor="black", linewidth=0.4)
        ax.axhline(ref, color="black", linestyle="--", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("Run")
        ax.tick_params(axis="x", labelrotation=90)
        ax.grid(axis="y", alpha=0.3)
        mean = np.mean(y)
        ax.axhline(mean, color=color, linestyle=":", linewidth=1.5,
                   label=f"mean={mean:.3f}")
        ax.legend(loc="best")

    plt.tight_layout()
    out = OUT / f"perrun_{metric}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_heritability_scatter():
    """4-panel parent-vs-child scatter (all 10 runs pooled), with regression + r."""
    from parent_child_gait import collect_pairs
    dbs = [str(BASE / f"run_{r}" / f"run_{r}") for r in RUNS]
    df = collect_pairs(dbs)
    pc = df[df["type"] == "parent-child"]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5.5))
    for ax, m in zip(axes, METRICS):
        label = LABELS[m]
        pv = pc[f"parent_{m}"].values.astype(float)
        cv = pc[f"child_{m}"].values.astype(float)
        mk = ~(np.isnan(pv) | np.isnan(cv))
        pv, cv = pv[mk], cv[mk]
        r, _ = stats.pearsonr(pv, cv)
        slope, intercept, *_ = stats.linregress(pv, cv)

        ax.scatter(pv, cv, alpha=0.05, s=3, color="#3F7CAC", rasterized=True)
        xl = np.array([pv.min(), pv.max()])
        ax.plot(xl, slope * xl + intercept, color="#C1666B", linewidth=2,
                label=f"fit (slope={slope:.2f})")
        lo, hi = min(pv.min(), cv.min()), max(pv.max(), cv.max())
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, alpha=0.4, label="y=x")
        ax.set_title(f"{label}\n$r={r:+.3f}$")
        ax.set_xlabel(f"parent {label}")
        ax.set_ylabel(f"child {label}")
        ax.legend()

    plt.suptitle("Parent-child heritability (all 10 improved runs)",
                 fontsize=27, fontweight="bold")
    plt.tight_layout()
    out = OUT / "heritability_scatter.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_heritability_scatter_separate():
    """One PNG per metric: parent-vs-child scatter (all 10 runs) + regression + r."""
    from parent_child_gait import collect_pairs
    dbs = [str(BASE / f"run_{r}" / f"run_{r}") for r in RUNS]
    df = collect_pairs(dbs)
    pc = df[df["type"] == "parent-child"]

    for m in METRICS:
        label = LABELS[m]
        pv = pc[f"parent_{m}"].values.astype(float)
        cv = pc[f"child_{m}"].values.astype(float)
        mk = ~(np.isnan(pv) | np.isnan(cv))
        pv, cv = pv[mk], cv[mk]
        r, _ = stats.pearsonr(pv, cv)
        slope, intercept, *_ = stats.linregress(pv, cv)

        fig, ax = plt.subplots(figsize=(6, 5.6))
        ax.scatter(pv, cv, alpha=0.05, s=3, color="#3F7CAC", rasterized=True)
        xl = np.array([pv.min(), pv.max()])
        ax.plot(xl, slope * xl + intercept, color="#C1666B", linewidth=2,
                label=f"fit (slope={slope:.2f})")
        lo, hi = min(pv.min(), cv.min()), max(pv.max(), cv.max())
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, alpha=0.4, label="y=x")
        ax.set_title(f"{label}   $r={r:+.3f}$")
        ax.set_xlabel(f"parent {label}")
        ax.set_ylabel(f"child {label}")
        ax.legend()
        plt.tight_layout()
        out = OUT / f"heritability_{m}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out}")


if __name__ == "__main__":
    data = compute_per_run()
    for m in METRICS:
        plot_metric(m, data[m])
    plot_heritability_scatter()
    print("Done.")
