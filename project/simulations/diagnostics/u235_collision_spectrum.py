"""Energy spectrum of U-235 collisions in the uniform-pulse sphere.

Feeds `project/analysis/multiplicity_toy.py`: OpenMC banks floor(nu_bar sigma_f/sigma_t + xi)
neutrons at EVERY U-235 collision, so the pair moment it produces is an average of
that rule over the energies at which U-235 collisions actually happen. This run
tallies, in the core cell, per source neutron and per energy bin, the U-235
collision rate ('total'), fission, elastic and absorption rates. From these the
energy-resolved sigma_f/sigma_t and sigma_s/sigma_t follow directly.

Run:  python -m project.simulations.diagnostics.u235_collision_spectrum [label] [N]
Out:  project/results/u235-collision-spectrum-k<label>/statepoint.1.h5
"""
import sys

import numpy as np
import openmc

from project.core.geometry import CORE_CELL_ID, DETECTOR_CELL_ID, core_with_detector
from project.core.materials import detector_material, mixture_material
from project.core.rossi_mesh import KEFF_CONFIGS
from project.core.simulation import Simulation
from project.core.sources import uniform_core_source

label = sys.argv[1] if len(sys.argv) > 1 else "090"
n = int(sys.argv[2]) if len(sys.argv) > 2 else 200_000
k = [kk for kk, v in KEFF_CONFIGS.items() if v["label"] == f"k{label}"][0]
R = KEFF_CONFIGS[k]["radius"]
det_z = R + 10.0 + 5.0

geometry = core_with_detector(
    core_center=(0.0, 0.0, 0.0), core_radius=R,
    detector_center=(0.0, 0.0, det_z), detector_radius=5.0,
)
settings = {
    "run_mode": "fixed source", "batches": 10, "particles": n // 10,
    "source": uniform_core_source(R),
}
sim = Simulation(
    materials=openmc.Materials([mixture_material, detector_material]),
    geometry=geometry, settings=settings, name=f"u235-collision-spectrum-k{label}",
)
sim.build()
e_edges = np.logspace(-5, np.log10(2e7), 241)
t = openmc.Tally(name="u235_by_energy")
t.filters = [openmc.CellFilter([CORE_CELL_ID]), openmc.EnergyFilter(e_edges)]
t.nuclides = ["U235"]
t.scores = ["total", "fission", "elastic", "absorption", "nu-fission"]
sim.model.tallies = openmc.Tallies([t])
sim.run()
print("done ->", sim.output_dir)
