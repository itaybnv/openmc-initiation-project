"""Materials for the heterogeneous MTR plate-fuel line.

Grounded in the IAEA 10 MW generic MTR benchmark (IAEA-TECDOC-233 / -643), HEU
case, which matches Soreq IRR-1 (93 wt% HEU plate fuel, UAlx-Al meat). The
homogeneous-sphere materials (``mixture_material``) are untouched; only
``detector_material`` is reused from ``project.core.materials``.

Meat loading is anchored to the benchmark HEU standard element: 280 g U-235 per
23-plate element over a 443.5 cm^3 meat volume => 0.631 g U-235/cm^3 (0.678 g
U/cm^3 total at 93 wt%). The meat is modelled as a homogenized U + Al dispersion;
its total density (3.4 g/cm^3, representative UAlx-Al) has only a weak neutronic
effect since the U loading is fixed exactly and Al is a mild scatterer. Refine
against IAEA-TECDOC-643 Vol.4 (Fuels) if a sourced meat density is needed.
"""

import openmc

from .materials import detector_material

__all__ = ["fuel_meat", "clad", "water", "graphite", "detector_material"]

# HEU 93 wt% UAlx-Al meat. Weight fractions give U-235 = 0.631 g/cm^3 (=> 280 g
# per standard element), U-235/(U-235+U-238) = 93.1 wt%.
fuel_meat = openmc.Material(name="HEU UAlx fuel meat")
fuel_meat.set_density("g/cm3", 3.4)
fuel_meat.add_nuclide("U235", 0.1856, "wo")
fuel_meat.add_nuclide("U238", 0.0138, "wo")
fuel_meat.add_element("Al", 0.8006, "wo")

# Aluminium cladding / structure. The benchmark models clad as generic Al at
# 2.70 g/cm^3 (AlMg2 is neutronically equivalent).
clad = openmc.Material(name="Al cladding")
clad.set_density("g/cm3", 2.70)
clad.add_element("Al", 1.0, "ao")

# Light-water coolant/moderator, cold clean (20 C), no soluble poison.
water = openmc.Material(name="Light water moderator")
water.set_density("g/cm3", 1.0)
water.add_element("H", 2.0, "ao")
water.add_element("O", 1.0, "ao")
water.add_s_alpha_beta("c_H_in_H2O")

# Graphite reflector (IRR-1 / IAEA benchmark reflect on two faces with graphite).
graphite = openmc.Material(name="Graphite reflector")
graphite.set_density("g/cm3", 1.70)
graphite.add_element("C", 1.0, "ao")
graphite.add_s_alpha_beta("c_Graphite")
