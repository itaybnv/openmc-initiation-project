

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
simulation = Simulation(
    materials=materials,
    geometry=geometry,
    settings=settings,
    name="simulation_with_detector",
)

simulation.build()
simulation.run()