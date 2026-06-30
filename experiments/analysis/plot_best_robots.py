#!/usr/bin/env python3
"""
Render the best-performing robot (max displacement) from each improved run
(200-209) as a 2D voxel body plan, using the material colour scheme.

Outputs (to tmp_out/plots/):
  - best_run_<r>.png       one image per run
  - best_robots_grid.png   combined 2x5 grid, labelled with run + fitness
"""
import os, sys, json, sqlite3
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT))
from algorithms.GRN_2D import GRN
from algorithms.voxel_types import VOXEL_TYPES, VOXEL_TYPES_COLORS


def trim_phenotype_materials(phenotype):
    body = np.asarray(phenotype, dtype=int)
    body = body[np.any(body != 0, axis=1)]
    body = body[:, np.any(body != 0, axis=0)]
    return body

BASE = ROOT / "tmp_out" / "evobots" / "evobots"
OUT = ROOT / "tmp_out" / "plots"
OUT.mkdir(parents=True, exist_ok=True)
RUNS = list(range(200, 210))


def best_robot(run):
    db = BASE / f"run_{run}" / f"run_{run}"
    con = sqlite3.connect(str(db))
    rid, gtxt, disp = con.execute(
        "SELECT robot_id, genome, displacement FROM all_robots "
        "WHERE valid=1.0 AND displacement > -1e300 "
        "ORDER BY displacement DESC LIMIT 1"
    ).fetchone()
    con.close()
    genome = json.loads(gtxt) if isinstance(gtxt, str) else gtxt
    return rid, genome, disp


def develop(genome):
    ph = GRN(promoter_threshold=0.95, max_voxels=36, cube_face_size=6,
             genotype=list(genome), voxel_types="withbone",
             env_conditions="", plastic=0).develop()
    mats = np.zeros(ph.shape, dtype=int)
    for idx, v in np.ndenumerate(ph):
        mats[idx] = v.voxel_type if v != 0 else 0
    return trim_phenotype_materials(mats)


def to_rgb(grid):
    """grid[x,y] material id -> RGB image[y,x] (origin lower = y up)."""
    nx, ny = grid.shape
    img = np.ones((ny, nx, 3))  # white background
    inv = {vid: name for name, vid in VOXEL_TYPES.items()}
    for x in range(nx):
        for y in range(ny):
            v = grid[x, y]
            if v > 0:
                img[y, x] = np.array(VOXEL_TYPES_COLORS[inv[v]]) / 255.0
    return img


def draw_one(ax, grid, title):
    ax.imshow(to_rgb(grid), origin="lower", interpolation="nearest")
    ax.set_title(title, fontsize=22)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


if __name__ == "__main__":
    robots = [(r, *best_robot(r)) for r in RUNS]   # (run, rid, genome, disp)

    # individual images (labelled Run 1..10)
    for idx, (run, rid, genome, disp) in enumerate(robots, start=1):
        grid = develop(genome)
        fig, ax = plt.subplots(figsize=(2.8, 2.8))
        draw_one(ax, grid, f"Run {idx}  (fit={disp:.1f})")
        plt.tight_layout()
        out = OUT / f"best_run_{idx}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"Saved: {out}")

    # combined 2x5 grid
    legend_labels = {"bone": "Bone", "fat": "Fat",
                     "phase_muscle": "In-phase muscle",
                     "offphase_muscle": "Offphase muscle"}
    fig, axes = plt.subplots(2, 5, figsize=(17, 8))
    legend = [Patch(facecolor=np.array(VOXEL_TYPES_COLORS[n]) / 255.0,
                    edgecolor="k", label=legend_labels.get(n, n)) for n in VOXEL_TYPES]
    for idx, (ax, (run, rid, genome, disp)) in enumerate(zip(axes.flatten(), robots), start=1):
        draw_one(ax, develop(genome), f"Run {idx}  (fit={disp:.1f})")
    fig.legend(handles=legend, loc="lower center", ncol=4, fontsize=18)
    fig.suptitle("Best robot per run", fontsize=28, fontweight="bold")
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    out = OUT / "best_robots_grid.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {out}")
    print("Done.")
