import openmc

from project.core.geometry import core_sphere
from project.core.materials import mixture_material
from project.core.physics_config import detector_settings
from project.core.rossi_mesh import CORE_RADIUS
from project.core.simulation import Simulation


# Strip collision_track: no detector cell to track captures in
settings = {k: v for k, v in detector_settings.items() if k != "collision_track"}

materials = openmc.Materials([mixture_material])
geometry = core_sphere(center=(0.0, 0.0, 0.0), radius=CORE_RADIUS)

cell_filter = openmc.CellFilter(geometry.get_all_cells()[1])  # core cell id=1
fission_tally = openmc.Tally(name="fission")
fission_tally.filters = [cell_filter]
fission_tally.scores = ["fission"]
nu_fission_tally = openmc.Tally(name="nu-fission")
nu_fission_tally.filters = [cell_filter]
nu_fission_tally.scores = ["nu-fission"]

simulation = Simulation(
    materials=materials,
    geometry=geometry,
    settings=settings,
    name="simulation_without_detector",
)
simulation.build()
simulation.model.tallies = openmc.Tallies([fission_tally, nu_fission_tally])
simulation.run()
