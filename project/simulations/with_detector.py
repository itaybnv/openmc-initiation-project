import openmc

from project.core.geometry import core_with_detector
from project.core.materials import mixture_material, detector_material
from project.core.physics_config import detector_settings
from project.core.simulation import Simulation


materials = openmc.Materials([mixture_material, detector_material])
geometry = core_with_detector(
    core_center=(0.0, 0.0, 0.0),
    core_radius=3.0,
    detector_center=(0.0, 3.0 + 5.0 + 10.0, 0.0),
    detector_radius=5.0,
)
settings = detector_settings
cell_filter = openmc.CellFilter(
    geometry.get_all_cells()[1]
)  # Assuming the first cell is the detector cell
fission_tally = openmc.Tally()
fission_tally.filters = [cell_filter]
fission_tally.scores = ["fission"]
nu_fission_tally = openmc.Tally()
nu_fission_tally.filters = [cell_filter]
nu_fission_tally.scores = ["nu-fission"]
tallies = openmc.Tallies([fission_tally, nu_fission_tally])

simulation = Simulation(
    materials=materials,
    geometry=geometry,
    settings=settings,
    name="simulation_with_detector",
)

simulation.build()
simulation.model.tallies = tallies
simulation.run()
