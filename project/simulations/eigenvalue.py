import openmc
from project.core.geometry import core_sphere
from project.core.materials import mixture_material
from project.core.physics_config import eigenvalue_settings
from project.core.rossi_mesh import CORE_RADIUS
from project.core.simulation import Simulation

materials = openmc.Materials([mixture_material])
geometry = core_sphere(center=(0.0, 0.0, 0.0), radius=CORE_RADIUS)
settings = eigenvalue_settings
simulation = Simulation(
    materials=materials, geometry=geometry, settings=settings, name="eigenvalue_simulation"
)

simulation.build()
simulation.run()