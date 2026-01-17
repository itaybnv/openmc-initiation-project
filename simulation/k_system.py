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

materials = openmc.Materials([fuel_water])

# -----------------------------------------------------------------------------
# Geometry: surfaces and cells
# -----------------------------------------------------------------------------
radius = 20.0
outer_radius = radius + 100.0

fuel_sphere = openmc.Sphere(r=radius)
outer_sphere = openmc.Sphere(r=outer_radius, boundary_type="vacuum")

fuel_cell = openmc.Cell(name="fuel", fill=fuel_water, region=-fuel_sphere)

void_region = +fuel_sphere & -outer_sphere
void_cell = openmc.Cell(name="void", region=void_region)

geometry = openmc.Geometry([fuel_cell, void_cell])

# -----------------------------------------------------------------------------
# Simulation settings: fixed source and source definition
# -----------------------------------------------------------------------------
settings = openmc.Settings()
settings.run_mode = "eigenvalue"

settings.batches = (batches := 100)
settings.inactive = (inactive := 30)
settings.particles = (particles := 100000)

settings.output = {"path": "../data/k_system"}

# -----------------------------------------------------------------------------
# Build model and run
# -----------------------------------------------------------------------------
model = openmc.Model(geometry=geometry, materials=materials, settings=settings)

model.export_to_model_xml(path="./simulation")

openmc.run(cwd="./simulation")
