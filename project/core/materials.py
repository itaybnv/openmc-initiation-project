"""Material definitions for the simulation."""

import openmc

__all__ = ["mixture_material", "detector_material"]

mixture_material = openmc.Material(name="Sphere material")
mixture_material.set_density("g/cm3", 1.0)
mixture_material.add_element("H", 0.6634, "ao")
mixture_material.add_element("O", 0.3306, "ao")
mixture_material.add_nuclide("U235", 0.00055, "ao")
mixture_material.add_s_alpha_beta("c_H_in_H2O")

detector_material = openmc.Material(name="Detector material")
detector_material.add_nuclide("B10", 1.0, "ao")
detector_material.set_density("atom/b-cm", 100)
