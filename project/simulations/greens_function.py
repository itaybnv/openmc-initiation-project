import numpy as np
import openmc

from project.core.geometry import core_with_detector
from project.core.materials import detector_material, mixture_material
from project.core.physics_config import detector_settings
from project.core.rossi_mesh import make_fission_mesh
from project.core.simulation import Simulation
from project.core.source_utils import map_source_to_mesh, pregenerate_source

materials = openmc.Materials([mixture_material, detector_material])
geometry = core_with_detector(
    core_center=(0.0, 0.0, 0.0),
    core_radius=3.0,
    detector_center=(0.0, 0.0, 3.0 + 5.0 + 10.0),
    detector_radius=5.0,
)
settings = detector_settings

settings["source"] = openmc.IndependentSource(
    space=openmc.stats.SphericalIndependent(
        r=openmc.stats.PowerLaw(0, 3.0, 2),  # p(r) ∝ r² → uniform volume
        cos_theta=openmc.stats.Uniform(-1, 1),
        phi=openmc.stats.Uniform(0, 2 * np.pi),
        origin=(0.0, 0.0, 0.0),
    ),
    energy=openmc.stats.Watt(a=0.988e6, b=2.249e-6),  # U235 fission Watt spectrum
    time=openmc.stats.Discrete([0.0], [1.0]),           # instantaneous pulse at t=0
    strength=1.0,
)

mesh = make_fission_mesh()
mesh_filter = openmc.MeshFilter(mesh)

fission_tally = openmc.Tally(name="fission_per_cell")
fission_tally.filters = [mesh_filter]
fission_tally.scores = ["fission"]

tallies = openmc.Tallies([fission_tally])

simulation = Simulation(
    materials=materials,
    geometry=geometry,
    settings=settings,
    name="greens-function-simulation",
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
