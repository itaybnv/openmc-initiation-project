# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import openmc
import openmc.stats as stats
import numpy as np

# -----------------------------------------------------------------------------
# Material definitions
# -----------------------------------------------------------------------------
u235_fraction = 0.00113

fuel_water = openmc.Material(name="U235-H2O mixture")
h_fraction = 2.0 * (1 - u235_fraction) / 3.0
o_fraction = 1.0 * (1 - u235_fraction) / 3.0

fuel_water.add_nuclide("U235", u235_fraction, "ao")
fuel_water.add_nuclide("H1", h_fraction, "ao")
fuel_water.add_nuclide("O16", o_fraction, "ao")
fuel_water.set_density("g/cm3", 1.0)
fuel_water.add_s_alpha_beta("c_H_in_H2O")

absorber = openmc.Material(name="Absorber")
absorber.add_nuclide("B10", 1.0, "ao")
absorber.set_density("g/cm3", 2.34 * 10)

materials = openmc.Materials([fuel_water, absorber])

# -----------------------------------------------------------------------------
# Geometry: surfaces and cells
# -----------------------------------------------------------------------------
radius = 20.0
outer_radius = radius + 100.0

detector_radius = 5.0
detector_distance = radius + 10.0

fuel_sphere = openmc.Sphere(r=radius)
outer_sphere = openmc.Sphere(r=outer_radius, boundary_type="vacuum")
detector1_sphere = openmc.Sphere(y0=detector_distance, r=detector_radius)
detector2_sphere = openmc.Sphere(y0=-detector_distance, r=detector_radius)

fuel_cell = openmc.Cell(name="fuel", fill=fuel_water, region=-fuel_sphere)
detector1_cell = openmc.Cell(name="detector1", fill=absorber, region=-detector1_sphere)
detector2_cell = openmc.Cell(name="detector2", fill=absorber, region=-detector2_sphere)

void_region = +fuel_sphere & +detector1_sphere & +detector2_sphere & -outer_sphere
void_cell = openmc.Cell(name="void", region=void_region)

geometry = openmc.Geometry([fuel_cell, detector1_cell, detector2_cell, void_cell])

# -----------------------------------------------------------------------------
# Simulation settings: fixed source and source definition
# -----------------------------------------------------------------------------
settings = openmc.Settings()
settings.run_mode = "fixed source"

# Internal point source inside the sphere
source_distance = 0.0
settings.source = openmc.IndependentSource(
    space=stats.Point((source_distance, 0.0, 0.0)),
    energy=stats.Discrete([2.5e6], [1.0]),
    strength=1.0,
    time=stats.Discrete([0.0], [1.0]),  # Emission at time zero - single pulse
)

settings.batches = (batches := 100)
settings.particles = (particles := 10000)

settings.collision_track = {
    "max_collisions": 50 * batches * particles,
    "reactions": ["absorption"],
    # "material_ids": [1,2],
    # "nuclides": ["U238", "O16"],
    "cell_ids": [detector1_cell.id, detector2_cell.id],
}

settings.create_delayed_neutrons = False
settings.output = {"path": "../data/simple_collision_track"}


# -----------------------------------------------------------------------------
# Build model and run
# -----------------------------------------------------------------------------
model = openmc.Model(geometry=geometry, materials=materials, settings=settings)

model.export_to_model_xml(path="./simulation")

openmc.run(cwd="./simulation")
