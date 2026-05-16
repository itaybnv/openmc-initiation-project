import openmc

from project.core.geometry import core_with_detector
from project.core.materials import mixture_material, detector_material
from project.core.physics_config import detector_settings
from project.core.rossi_mesh import CORE_RADIUS
from project.core.simulation import Simulation


materials = openmc.Materials([mixture_material, detector_material])
geometry = core_with_detector(
    core_center=(0.0, 0.0, 0.0),
    core_radius=CORE_RADIUS,
    detector_center=(0.0, CORE_RADIUS + 10.0 + 5.0, 0.0),
    detector_radius=5.0,
)
settings = detector_settings

# Core cell (id=1) filter for fission tallies
cell_filter = openmc.CellFilter(geometry.get_all_cells()[1])
fission_tally = openmc.Tally(name="fission")
fission_tally.filters = [cell_filter]
fission_tally.scores = ["fission"]
nu_fission_tally = openmc.Tally(name="nu-fission")
nu_fission_tally.filters = [cell_filter]
nu_fission_tally.scores = ["nu-fission"]

# Surface current tally on core sphere: separates inward J- and outward J+
# to quantify how many neutrons are reflected back into the core by the detector
core_surface = next(
    s for s in geometry.get_all_surfaces().values()
    if isinstance(s, openmc.Sphere)
    and abs(s.r - CORE_RADIUS) < 1e-9
    and abs(s.x0) < 1e-9
    and abs(s.y0) < 1e-9
    and abs(s.z0) < 1e-9
    and s.boundary_type == "transmission"
)
surface_current_tally = openmc.Tally(name="core_surface_current")
surface_current_tally.filters = [
    openmc.SurfaceFilter([core_surface]),
    openmc.MuSurfaceFilter([-1.0, 0.0, 1.0]),  # [-1,0] = inward J-, [0,1] = outward J+
]
surface_current_tally.scores = ["current"]

tallies = openmc.Tallies([fission_tally, nu_fission_tally, surface_current_tally])

simulation = Simulation(
    materials=materials,
    geometry=geometry,
    settings=settings,
    name="simulation_with_detector",
)

simulation.build()
simulation.model.tallies = tallies
simulation.run()
