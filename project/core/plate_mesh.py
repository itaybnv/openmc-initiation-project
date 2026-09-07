"""Config table and Cartesian fission mesh for the plate line.

Parallels ``rossi_mesh.py`` (which holds ``KEFF_CONFIGS`` and the SphericalMesh for
the homogeneous sphere). ``PLATE_CONFIGS`` records validated loosely-coupled
geometries by label; ``make_plate_fission_mesh`` provides the Cartesian
per-cell mesh the Green's-function phase (Phase 1) will tally on.
"""

import numpy as np
import openmc

from .plate_geometry import PITCH, ACTIVE_HEIGHT, coupled_fuel_x_hi

__all__ = ["PLATE_CONFIGS", "make_plate_fission_mesh"]

# Validated Phase-0 loosely-coupled configs (measured k_eff at 3k part / 50 batch).
# 'coupled095' is the gate config: subcritical, near the sphere study's 0.95 band,
# with a 6 cm central water gap (weak coupling) and a graphite-backed -X face.
PLATE_CONFIGS: dict[str, dict] = {
    "coupled095": {
        "n_a": 23, "n_b": 23, "coupling_gap": 6.0, "width": 40.0,
        "graphite_thickness": 10.0, "refl_water": 12.0, "keff": 0.942,
    },
}


def make_plate_fission_mesh(
    config: dict,
    nx_per_pack: int | None = None,
    n_y: int = 1,
    n_z: int = 8,
) -> openmc.RegularMesh:
    """Cartesian RegularMesh over the fuel envelope for the per-cell fission tally.

    X is the plate-stacking direction; by default it is resolved at one bin per
    plate pitch (``nx_per_pack = n_a + n_b`` bins across both packs plus the gap),
    Y is coarse (packs are wide/homogeneous), Z resolves the axial die-away.
    """
    x_hi = coupled_fuel_x_hi(config["n_a"], config["n_b"], config["coupling_gap"])
    width = config["width"]
    if nx_per_pack is None:
        n_x = int(round(x_hi / PITCH))
    else:
        n_x = nx_per_pack
    mesh = openmc.RegularMesh()
    mesh.dimension = (n_x, n_y, n_z)
    mesh.lower_left = (0.0, -width / 2, -ACTIVE_HEIGHT / 2)
    mesh.upper_right = (x_hi, width / 2, ACTIVE_HEIGHT / 2)
    return mesh
