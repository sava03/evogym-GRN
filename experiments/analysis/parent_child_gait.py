#!/usr/bin/env python3
"""
Analyse how different gait traits are between parents and children.

Works with both old databases (displacement only) and new ones
(full gait metrics). Missing columns are treated as NaN.

Usage:
    python parent_child_gait.py <db_path> [<db_path> ...]

Outputs:
  - Console summary: |Δ| parent-child vs random-pair + heritability table
  - parent_child_gait.png:   histogram + per-metric |Δ| comparison
  - parent_child_herit.png:  parent-child scatter plots with regression lines
  - parent_child_gait.csv:   per-pair row data for further analysis
"""

import sys
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

GAIT_METRICS = [
    "displacement",
    "com_y_mean",
    "com_y_osc",
    "vel_x_std",
    "gait_period",
    "stride_regularity",
]


def load_robots(db_path: str) -> pd.DataFrame:
    con = sqlite3.connect(db_path)
    cols_in_db = {row[1] for row in con.execute("PRAGMA table_info(all_robots)")}

    base = ["robot_id", "born_generation", "parent1_id", "parent2_id", "valid"]
    available = [m for m in GAIT_METRICS if m in cols_in_db]
    missing = [m for m in GAIT_METRICS if m not in cols_in_db]

    if missing:
        print(f"  [warn] {Path(db_path).name}: columns not found (old DB): {missing}")

    select = ", ".join(base + available)
    df = pd.read_sql(f"SELECT {select} FROM all_robots WHERE valid = 1.0", con)
    con.close()

    for m in missing:
        df[m] = np.nan

    return df


def collect_pairs(db_paths):
    pc_rows = []
    rand_vecs = []

    for db_path in db_paths:
        print(f"Loading {db_path} ...")
        df = load_robots(db_path)
        by_id = df.set_index("robot_id")

        for _, child in df.iterrows():
            cv = child[GAIT_METRICS].values.astype(float)
            for pid_col, label in [("parent1_id", "parent1"), ("parent2_id", "parent2")]:
                pid = child[pid_col]
                if pd.isna(pid) or int(pid) not in by_id.index:
                    continue
                pv = by_id.loc[int(pid), GAIT_METRICS].values.astype(float)
                delta      = {m: float(abs(cv[i] - pv[i])) for i, m in enumerate(GAIT_METRICS)}
                parent_val = {f"parent_{m}": float(pv[i]) for i, m in enumerate(GAIT_METRICS)}
                child_val  = {f"child_{m}":  float(cv[i]) for i, m in enumerate(GAIT_METRICS)}
                pc_rows.append({
                    "type": "parent-child",
                    "child_id": int(child["robot_id"]),
                    "parent_id": int(pid),
                    "parent_role": label,
                    "child_gen": child["born_generation"],
                    **delta,
                    **parent_val,
                    **child_val,
                })

        rand_vecs.extend(df[GAIT_METRICS].values.tolist())

    # Random-pair baseline (same count as parent-child pairs)
    rand_rows = []
    if rand_vecs and pc_rows:
        pool = np.array(rand_vecs, dtype=float)
        rng = np.random.default_rng(0)
        target = len(pc_rows)
        attempts = 0
        while len(rand_rows) < target and attempts < target * 10:
            i, j = rng.integers(0, len(pool), size=2)
            attempts += 1
            if i == j:
                continue
            v1, v2 = pool[i], pool[j]
            per_metric = {m: float(abs(v1[k] - v2[k])) for k, m in enumerate(GAIT_METRICS)}
            rand_rows.append({"type": "random", **per_metric})

    return pd.DataFrame(pc_rows + rand_rows)


def _cohens_d(a, b):
    """Pooled-SD Cohen's d: positive means a > b."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    pooled_sd = np.sqrt(((na - 1) * a.std() ** 2 + (nb - 1) * b.std() ** 2) / (na + nb - 2))
    return (a.mean() - b.mean()) / pooled_sd if pooled_sd > 0 else float("nan")


def summarise(df):
    from scipy import stats

    available = [m for m in GAIT_METRICS if df[m].notna().any()]
    pc   = df[df["type"] == "parent-child"]
    rand = df[df["type"] == "random"]

    print(f"\n{'Metric':<22}  {'PC mean±std':>20}  {'Rand mean±std':>20}  "
          f"{'ratio':>6}  {'d':>6}  {'p (MW)':>9}  note")
    print("-" * 100)
    for m in available:
        a = pc[m].dropna().values
        b = rand[m].dropna().values
        if not len(a) or not len(b):
            continue
        ratio = a.mean() / b.mean() if b.mean() != 0 else float("nan")
        d     = _cohens_d(a, b)
        _, p  = stats.mannwhitneyu(a, b, alternative="two-sided")
        note  = "more similar" if ratio < 1 else "more different"
        print(f"  {m:<22}  {a.mean():>8.4f} ± {a.std():<8.4f}  "
              f"{b.mean():>8.4f} ± {b.std():<8.4f}  "
              f"{ratio:>6.3f}  {d:>+6.3f}  {p:>9.2e}  {note}")


def heritability(df, out_path="parent_child_herit.png"):
    from scipy import stats

    pc = df[df["type"] == "parent-child"]
    available = [m for m in GAIT_METRICS if f"parent_{m}" in df.columns and pc[f"parent_{m}"].notna().any()]

    print(f"\n{'Metric':<22}  {'r':>7}  {'h² (slope)':>10}  {'p':>9}  interpretation")
    print("-" * 75)
    results = {}
    for m in available:
        pv = pc[f"parent_{m}"].values.astype(float)
        cv = pc[f"child_{m}"].values.astype(float)
        mask = ~(np.isnan(pv) | np.isnan(cv))
        pv, cv = pv[mask], cv[mask]
        if len(pv) < 5:
            continue
        r, p = stats.pearsonr(pv, cv)
        slope, intercept, *_ = stats.linregress(pv, cv)
        results[m] = (r, slope, intercept, p)
        interp = "strongly inherited" if abs(r) > 0.5 else ("weakly inherited" if abs(r) > 0.2 else "not inherited")
        print(f"  {m:<22}  {r:>+7.3f}  {slope:>10.3f}  {p:>9.2e}  {interp}")

    # Scatter plots
    if not results:
        return
    ncols = min(len(results), 3)
    nrows = (len(results) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
    axes = np.array(axes).flatten()

    for i, m in enumerate(available):
        if m not in results:
            continue
        ax = axes[i]
        pv = pc[f"parent_{m}"].values.astype(float)
        cv = pc[f"child_{m}"].values.astype(float)
        mask = ~(np.isnan(pv) | np.isnan(cv))
        pv, cv = pv[mask], cv[mask]

        r, slope, intercept, p = results[m]
        ax.scatter(pv, cv, alpha=0.08, s=4, color="steelblue", rasterized=True)

        x_line = np.array([pv.min(), pv.max()])
        ax.plot(x_line, slope * x_line + intercept, color="red", linewidth=1.5, label=f"fit (h²={slope:.2f})")
        lo, hi = min(pv.min(), cv.min()), max(pv.max(), cv.max())
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, alpha=0.4, label="y=x")

        ax.set_xlabel(f"parent {m}", fontsize=8)
        ax.set_ylabel(f"child {m}", fontsize=8)
        ax.set_title(f"{m}\nr={r:+.3f}, p={p:.1e}", fontsize=9)
        ax.legend(fontsize=7)

    for ax in axes[len(available):]:
        ax.set_visible(False)

    plt.suptitle("Parent-child trait heritability", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")


def plot(df, out_path="parent_child_gait.png"):
    pc = df[df["type"] == "parent-child"]
    rand = df[df["type"] == "random"]
    available = [m for m in GAIT_METRICS if df[m].notna().any()]

    n_plots = 1 + len(available)
    ncols = 4
    nrows = (n_plots + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.5))
    axes = np.array(axes).flatten()

    # Box plot of all available metrics side by side
    ax = axes[0]
    labels, pc_data, rand_data = [], [], []
    for m in available:
        labels.append(m.replace("_", "\n"))
        pc_data.append(pc[m].dropna().values)
        rand_data.append(rand[m].dropna().values)

    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w/2, [np.mean(d) for d in pc_data],   w, yerr=[np.std(d) for d in pc_data],   label="parent-child", color="steelblue", capsize=3)
    ax.bar(x + w/2, [np.mean(d) for d in rand_data], w, yerr=[np.std(d) for d in rand_data], label="random",        color="salmon",    capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("|Δ metric|")
    ax.set_title("Mean difference per gait trait")
    ax.legend(fontsize=7)

    # Per-metric histograms
    for i, m in enumerate(available):
        ax = axes[i + 1]
        ax.hist(pc[m].dropna(),   bins=40, alpha=0.6, color="steelblue", label=f"PC (n={pc[m].notna().sum()})")
        ax.hist(rand[m].dropna(), bins=40, alpha=0.6, color="salmon",    label=f"rand (n={rand[m].notna().sum()})")
        ax.set_title(f"|Δ {m}|")
        ax.legend(fontsize=6)

    for ax in axes[n_plots:]:
        ax.set_visible(False)

    plt.suptitle("Parent-child vs random gait trait differences", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_path}")


def similarity_vs_fitness(df, out_path="parent_child_sim_fitness.png"):
    """
    For each gait metric: does lower |child - parent| predict higher child fitness?
    Plots scatter of |Δ metric| vs child_displacement, with regression line and r.
    Also prints a summary table.
    """
    from scipy import stats

    pc = df[df["type"] == "parent-child"].copy()
    available = [m for m in GAIT_METRICS if f"child_{m}" in pc.columns and pc[f"child_{m}"].notna().any()]

    if "child_displacement" not in pc.columns or pc["child_displacement"].isna().all():
        print("No child_displacement available — skipping similarity vs fitness.")
        return

    fitness = pc["child_displacement"].values.astype(float)

    ncols = 3
    nrows = (len(available) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 4))
    axes = np.array(axes).flatten()

    print(f"\n{'Metric':<22}  {'r':>7}  {'p':>9}  {'direction'}")
    print("-" * 65)

    for i, m in enumerate(available):
        ax = axes[i]
        diff = pc[m].values.astype(float)
        child_fit = fitness

        mask = ~(np.isnan(diff) | np.isnan(child_fit))
        d_m, f_m = diff[mask], child_fit[mask]

        # Remove top/bottom 0.5% outliers for readability
        lo_d, hi_d = np.percentile(d_m, 0.5), np.percentile(d_m, 99.5)
        lo_f, hi_f = np.percentile(f_m, 0.5), np.percentile(f_m, 99.5)
        plot_mask = (d_m >= lo_d) & (d_m <= hi_d) & (f_m >= lo_f) & (f_m <= hi_f)
        d_plot, f_plot = d_m[plot_mask], f_m[plot_mask]

        r, pval = stats.pearsonr(d_m, f_m)
        slope, intercept, *_ = stats.linregress(d_m, f_m)

        color = "#E53935" if r < 0 else "#2196F3"
        ax.scatter(d_plot, f_plot, alpha=0.06, s=3, color=color, rasterized=True)

        xline = np.linspace(d_m.min(), d_m.max(), 100)
        ax.plot(xline, slope * xline + intercept, color="black", linewidth=1.8,
                label=f"r={r:+.3f}\np={'<0.001' if pval < 0.001 else f'{pval:.3f}'}")

        ax.set_title(f"|Δ {m}| vs child fitness", fontsize=9)
        ax.set_xlabel(f"|child − parent {m}|", fontsize=8)
        ax.set_ylabel("Child displacement (fitness)", fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)

        sig = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else "ns"))
        direction = "similar → better" if r < 0 else "dissimilar → better"
        print(f"  {m:<22}  {r:>+7.4f}  {pval:>9.2e} {sig}  {direction}")

    for ax in axes[len(available):]:
        ax.set_visible(False)

    fig.suptitle("Does parent-child behavioral similarity predict child fitness?\n"
                 "(negative r = more similar → better fitness)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_path}")
    plt.show()


if __name__ == "__main__":
    db_paths = sys.argv[1:]
    if not db_paths:
        print(__doc__)
        sys.exit(1)

    out_dir = Path(db_paths[0]).parent
    df = collect_pairs(db_paths)

    if df.empty:
        print("No valid parent-child pairs found.")
        sys.exit(1)

    csv_path = out_dir / "parent_child_gait.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    summarise(df)
    print("\n--- Heritability (parent-child regression) ---")
    heritability(df, out_path=str(out_dir / "parent_child_herit.png"))
    plot(df, out_path=str(out_dir / "parent_child_gait.png"))
    print("\n--- Similarity vs Fitness ---")
    similarity_vs_fitness(df, out_path=str(out_dir / "parent_child_sim_fitness.png"))