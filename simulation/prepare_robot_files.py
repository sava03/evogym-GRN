import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from evogym import get_full_connectivity


def trim_phenotype_materials(phenotype):
    """
    Trim empty borders from a phenotype and return a 2D grid.
    """
    body = np.asarray(phenotype, dtype=int)

    if body.ndim != 2:
        raise ValueError(f"Expected 2D phenotype, got {body.shape}")

    x_mask = np.any(body != 0, axis=1)
    body = body[x_mask]
    y_mask = np.any(body != 0, axis=0)
    body = body[:, y_mask]

    return body




def _material_maps(voxel_types):
    """
    Map GRN material IDs -> EvoGym voxel IDs.
    We intentionally map both muscle classes to H_ACT so mechanics are equal;
    phase differences are carried by controller phase offsets.
    """
    EVOGYM = {
        "EMPTY": 0,
        "RIGID": 1,
        "SOFT": 2,
        "H_ACT": 3,
    }

    if voxel_types == "withbone":
        # bone, fat, phase_muscle, offphase_muscle
        material_to_evogym = {
            0: EVOGYM["EMPTY"],
            1: EVOGYM["RIGID"],
            2: EVOGYM["SOFT"],
            3: EVOGYM["H_ACT"],
            4: EVOGYM["H_ACT"],
        }
    elif voxel_types == "nobone":
        # fat, fat2, phase_muscle, offphase_muscle
        material_to_evogym = {
            0: EVOGYM["EMPTY"],
            1: EVOGYM["SOFT"],
            2: EVOGYM["SOFT"],
            3: EVOGYM["H_ACT"],
            4: EVOGYM["H_ACT"],
        }
    else:
        raise ValueError(f"Unsupported voxel_types: {voxel_types}")




    # Controller groups (phase in radians) based on original GRN labels.
    # Only actuator materials get non-zero phase offsets.

    return material_to_evogym


def _build_evogym_robot_data(body_materials, voxel_types, genome):
    material_to_evogym = _material_maps(voxel_types)

    structure = np.vectorize(lambda m: material_to_evogym.get(int(m), 0), otypes=[int])(body_materials)
    structure = structure.astype(np.int32)
    connections = get_full_connectivity(structure).astype(np.int32)

    # Controller params evolved via genome[2..4], each in [0, 1]
    controller = {
        "action_bias":      0.8 + float(genome[2]) * 0.4,   # [0.8, 1.2]
        "action_amplitude": 0.1 + float(genome[3]) * 0.5,   # [0.1, 0.6]
        "period_steps":     int(10 + float(genome[4]) * 30), # [10, 40]
    }

    # Material-aware phase offsets: phase_muscle (3) contracts in-phase,
    # offphase_muscle (4) contracts opposite (pi shift).
    phase_offsets = np.zeros_like(body_materials, dtype=np.float32)
    for x in range(body_materials.shape[0]):
        for y in range(body_materials.shape[1]):
            if body_materials[x, y] == 4:  # offphase_muscle
                phase_offsets[x, y] = np.pi

    return structure, connections, phase_offsets, controller


def prepare_robot_files(individual, args):
    """
    Prepare EvoGym robot artifacts from an evolved phenotype.
    Keeps the old function name so the EA loop can call it unchanged.
    """
    body = trim_phenotype_materials(individual.phenotype)
    structure, connections, phase_offsets, controller = _build_evogym_robot_data(
        body, args.voxel_types, individual.genome
    )

    # Keep data in-memory for upcoming EvoGym simulation adapter.
    individual.evogym_structure = structure
    individual.evogym_connections = connections
    individual.evogym_phase_offsets = phase_offsets
    individual.evogym_controller = controller
