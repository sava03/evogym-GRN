#!/usr/bin/env python3
"""
Thesis comparison plots:
  1. Fitness curve (best + mean per generation) — Improved vs Baseline
  2. Behavior traits per generation — Improved runs only (gait metrics)
"""
import sqlite3
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Configuration ────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent.parent / "tmp_out" / "evobots" / "evobots"
OUT  = Path(__file__).resolve().parent.parent.parent / "tmp_out" / "plots"
OUT.mkdir(parents=True, exist_ok=True)

IMPROVED_RUNS = list(range(200, 210))
BASELINE_RUNS = list(range(400, 410))

GAIT_METRICS = [
    ("displacement",      "Displacement (vox)",   None),
    ("com_y_mean",        "CoM Y mean (vox)",      None),
    ("com_y_osc",         "CoM Y oscillation",     None),
    ("vel_x_std",         "Velocity X std",        None),
    ("gait_period",       "Gait period (steps)",   None),
    ("stride_regularity", "Stride regularity",     None),
]


def _db(run: int) -> Path:
    return BASE / f"run_{run}" / f"run_{run}"


def _load_fitness_per_gen(runs):
    """Returns dict gen -> list of per-run best fitness values."""
    gen_best  = {}
    gen_mean  = {}
    for r in runs:
        p = _db(r)
        if not p.exists():
            print(f"[skip] run_{r} not found")
            continue
        conn = sqlite3.connect(str(p))
        rows = conn.execute(
            "SELECT generation, fitness FROM generation_survivors "
            "WHERE fitness IS NOT NULL AND fitness != -1e999"
        ).fetchall()
        conn.close()
        run_gens = {}
        for gen, fit in rows:
            run_gens.setdefault(gen, []).append(fit)
        for gen, vals in run_gens.items():
            gen_best.setdefault(gen, []).append(max(vals))
            gen_mean.setdefault(gen, []).append(np.mean(vals))
    return gen_best, gen_mean


def _load_metric_per_gen(runs, metric):
    """Returns dict gen -> list of per-run mean metric values (survivors only)."""
    gen_vals = {}
    for r in runs:
        p = _db(r)
        if not p.exists():
            continue
        conn = sqlite3.connect(str(p))
        cols = {row[1] for row in conn.execute("PRAGMA table_info(all_robots)").fetchall()}
        if metric not in cols:
            conn.close()
            continue
        rows = conn.execute(
            f"""
            SELECT gs.generation, AVG(r.{metric})
            FROM generation_survivors gs
            JOIN all_robots r ON r.robot_id = gs.robot_id
            WHERE r.{metric} IS NOT NULL
            GROUP BY gs.generation
            ORDER BY gs.generation
            """
        ).fetchall()
        conn.close()
        for gen, val in rows:
            gen_vals.setdefault(gen, []).append(val)
    return gen_vals


def _to_arrays(gen_dict):
    gens = sorted(gen_dict)
    means = np.array([np.mean(gen_dict[g]) for g in gens])
    stds  = np.array([np.std(gen_dict[g])  for g in gens])
    return np.array(gens), means, stds


def _shade(ax, gens, means, stds, color, label):
    ax.plot(gens, means, color=color, label=label, linewidth=1.8)
    ax.fill_between(gens, means - stds, means + stds, alpha=0.18, color=color)


# ── Plot 1: Fitness curves ────────────────────────────────────────────────────

def plot_fitness_curves():
    imp_best, imp_mean = _load_fitness_per_gen(IMPROVED_RUNS)
    bas_best, bas_mean = _load_fitness_per_gen(BASELINE_RUNS)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    fig.suptitle("Fitness Across Generations: Improved vs Baseline", fontsize=13, fontweight="bold")

    for ax, imp_d, bas_d, title in [
        (axes[0], imp_best, bas_best, "Best fitness per generation"),
        (axes[1], imp_mean, bas_mean, "Mean fitness per generation"),
    ]:
        ig, im, is_ = _to_arrays(imp_d)
        bg, bm, bs  = _to_arrays(bas_d)
        _shade(ax, ig, im, is_, "#2196F3", f"Improved (n={len(IMPROVED_RUNS)})")
        _shade(ax, bg, bm, bs, "#FF7043", f"Baseline (n={len(BASELINE_RUNS)})")
        ax.set_title(title)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Displacement (voxels)")
        ax.legend()
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = OUT / "fitness_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.show()


# ── Plot 2: Behavior traits ───────────────────────────────────────────────────

def plot_behavior_traits():
    n_metrics = len(GAIT_METRICS)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Behavior Traits Across Generations (Improved runs, mean ± std)", fontsize=13, fontweight="bold")
    axes_flat = axes.flatten()

    for ax, (col, label, _) in zip(axes_flat, GAIT_METRICS):
        gen_vals = _load_metric_per_gen(IMPROVED_RUNS, col)
        if not gen_vals:
            ax.set_visible(False)
            continue
        gens, means, stds = _to_arrays(gen_vals)
        _shade(ax, gens, means, stds, "#2196F3", "Improved")
        ax.set_title(label)
        ax.set_xlabel("Generation")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = OUT / "behavior_traits.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.show()


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    plot_fitness_curves()
    plot_behavior_traits()
