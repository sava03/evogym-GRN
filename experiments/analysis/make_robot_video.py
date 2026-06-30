#!/usr/bin/env python3
"""
Record an mp4 of the best-performing robot from a given run, simulated in
EvoGym with its evolved controller.

Usage:  python make_robot_video.py [run]   (default run 202)
Output: tmp_out/plots/best_run_<run>.mp4
"""
import sys, json, sqlite3, math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT))
from algorithms.GRN_2D import GRN
from simulation.prepare_robot_files import trim_phenotype_materials, _build_evogym_robot_data

from evogym import EvoWorld, EvoSim
from evogym.viewer import EvoViewer
import cv2

BASE = ROOT / "tmp_out" / "evobots" / "evobots"
OUT = ROOT / "tmp_out" / "plots"
OUT.mkdir(parents=True, exist_ok=True)

run = int(sys.argv[1]) if len(sys.argv) > 1 else 202
SIM_STEPS = 500
INIT_X, INIT_Y = 3, 1
FPS = 50


def best_genome(run):
    db = BASE / f"run_{run}" / f"run_{run}"
    con = sqlite3.connect(str(db))
    rid, gtxt, disp = con.execute(
        "SELECT robot_id, genome, displacement FROM all_robots "
        "WHERE valid=1.0 AND displacement > -1e300 ORDER BY displacement DESC LIMIT 1"
    ).fetchone()
    con.close()
    return rid, (json.loads(gtxt) if isinstance(gtxt, str) else gtxt), disp


def main():
    rid, genome, disp = best_genome(run)
    print(f"run {run}: best robot {rid}, displacement {disp:.2f}")

    ph = GRN(promoter_threshold=0.95, max_voxels=36, cube_face_size=6,
             genotype=list(genome), voxel_types="withbone",
             env_conditions="", plastic=0).develop()
    mats = np.zeros(ph.shape, dtype=int)
    for idx, v in np.ndenumerate(ph):
        mats[idx] = v.voxel_type if v != 0 else 0
    body = trim_phenotype_materials(mats)
    structure, connections, phase_offsets, ctrl = _build_evogym_robot_data(body, "withbone", list(genome))
    bias, amp, period = ctrl["action_bias"], ctrl["action_amplitude"], ctrl["period_steps"]

    world = EvoWorld()
    world.add_from_array(name="robot", structure=structure, x=INIT_X, y=INIT_Y, connections=connections)
    sim = EvoSim(world)
    sim.reset()
    viewer = EvoViewer(sim)
    viewer.track_objects("robot")

    actuators = sim.get_actuator_indices("robot").astype(int).flatten()
    phases = phase_offsets.reshape(-1)[actuators] if actuators.size else np.array([])

    frames = []
    for t in range(SIM_STEPS):
        if actuators.size:
            action = np.clip(bias + amp * np.sin(2 * math.pi * (t / period) + phases), 0.6, 1.6).astype(np.float64)
            sim.set_action("robot", action)
        unstable = sim.step()
        frame = viewer.render("img")
        if frame is not None:
            frames.append(np.asarray(frame))
        if unstable:
            break
    viewer.close()

    if not frames:
        print("No frames captured (renderer returned None) — cannot write video.")
        return
    h, w = frames[0].shape[:2]
    out = OUT / f"best_run_{run}.mp4"
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
    for f in frames:
        writer.write(cv2.cvtColor(np.asarray(f, dtype=np.uint8), cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"Saved {len(frames)} frames -> {out}")


if __name__ == "__main__":
    main()
