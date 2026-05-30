"""Green's function simulation at k_eff ≈ 0.98 with FUNDAMENTAL-mode source.

Identical geometry / mesh / tally / detector / 1M-particle setup as
project/simulations/greens_function.py, but the source is the converged fission
bank from project.simulations.eigenvalue_for_source (a FileSource pointing at
the eigenvalue's converged_source.h5) instead of a uniform-volume Watt source.

Purpose: validate that the ~30% gap between measured fissions/source (14.0) and
the analytical k/[ν(1-k)] = 20.16 is due to the uniform-volume source mismatch
with the fundamental k-eigenmode. Expected with fundamental source: ratio ≥ 0.95.

Run with: python -m project.simulations.greens_function_k098_fundsrc
(Requires running eigenvalue_for_source first to produce converged_source.h5.)
"""
import numpy as np
import openmc

from project.core.geometry import core_with_detector
from project.core.materials import detector_material, mixture_material
from project.core.physics_config import detector_settings
from project.core.rossi_mesh import KEFF_CONFIGS, make_fission_mesh
from project.core.simulation import Simulation
from project.core.source_utils import map_source_to_mesh, pregenerate_source

CONFIG = KEFF_CONFIGS[0.98]
RADIUS = CONFIG["radius"]  # 28.7 cm
NAME = "greens-function-k098-fundsrc"

# Path to the converged eigenvalue source bank
EIG_SOURCE_PATH = (
    Simulation(  # construct just to resolve the output path
        materials=openmc.Materials([mixture_material]),
        geometry=openmc.Geometry(),
        settings={"run_mode": "eigenvalue", "particles": 1, "batches": 1, "inactive": 0},
        name="eigenvalue_source_k098",
    ).output_dir
    / "converged_source.h5"
)
if not EIG_SOURCE_PATH.exists():
    raise FileNotFoundError(
        f"Converged source not found at {EIG_SOURCE_PATH}. "
        f"Run `python -m project.simulations.eigenvalue_for_source` first."
    )

materials = openmc.Materials([mixture_material, detector_material])
geometry = core_with_detector(
    core_center=(0.0, 0.0, 0.0),
    core_radius=RADIUS,
    detector_center=(0.0, 0.0, RADIUS + 10.0 + 5.0),
    detector_radius=5.0,
)
settings = detector_settings

# THE difference vs greens_function.py: FileSource instead of IndependentSource/Watt
settings["source"] = openmc.FileSource(str(EIG_SOURCE_PATH))

mesh = make_fission_mesh(core_radius=RADIUS)
mesh_filter = openmc.MeshFilter(mesh)

fission_tally = openmc.Tally(name="fission_per_cell")
fission_tally.filters = [mesh_filter]
fission_tally.scores = ["fission"]
tallies = openmc.Tallies([fission_tally])

simulation = Simulation(
    materials=materials, geometry=geometry, settings=settings, name=NAME
)
simulation.build()
simulation.model.tallies = tallies

n_particles = settings["particles"]
source_path = pregenerate_source(
    model=simulation.model,
    n_particles=n_particles,
    output_dir=simulation.output_dir,
)
print(f"Pre-generated {n_particles:,} source particles → {source_path.name}")

mesh_ids = map_source_to_mesh(source_path, mesh)
np.save(simulation.output_dir / "source_mesh_ids.npy", mesh_ids)
print(f"Mesh cell mapping saved. {np.sum(mesh_ids >= 0):,} particles inside mesh.")

simulation.run()
