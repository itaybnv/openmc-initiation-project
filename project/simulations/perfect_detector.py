import openmc

from project.core.geometry import core_with_perfect_detector
from project.core.materials import mixture_material, detector_material
from project.core.physics_config import perfect_detector_settings
from project.core.rossi_mesh import CORE_RADIUS
from project.core.simulation import Simulation

materials = openmc.Materials([mixture_material, detector_material])
geometry = core_with_perfect_detector(
    core_radius=CORE_RADIUS,
    detector_inner_radius=CORE_RADIUS,
    detector_outer_radius=CORE_RADIUS + 3.0,
)
settings = perfect_detector_settings

cell_filter = openmc.CellFilter(
    geometry.get_all_cells()[1]
)  # Assuming the first cell is the detector cell
tally = openmc.Tally()
tally.filters = [cell_filter]
tally.scores = ["fission"]
tallies = openmc.Tallies([tally])

simulation = Simulation(
    materials=materials,
    geometry=geometry,
    settings=settings,
    name="simulation_with_perfect_detector",
)

simulation.build()
simulation.model.tallies = tallies
simulation.run()
