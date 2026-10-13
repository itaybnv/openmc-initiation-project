"""Material definitions for the simulation."""

import openmc

__all__ = ["mixture_material", "detector_material", "detector_material_b10", "BLACK_B10_DENSITY"]

mixture_material = openmc.Material(name="Sphere material")
mixture_material.set_density("g/cm3", 1.0)
mixture_material.add_element("H", 0.6634, "ao")
mixture_material.add_element("O", 0.3306, "ao")
mixture_material.add_nuclide("U235", 0.00055, "ao")
mixture_material.add_s_alpha_beta("c_H_in_H2O")

detector_material = openmc.Material(name="Detector material")
detector_material.add_nuclide("B10", 1.0, "ao")
detector_material.set_density("atom/b-cm", 100)

BLACK_B10_DENSITY = 100.0   # atom/b-cm: every entering neutron is captured at its incoming energy


def detector_material_b10(density: float = BLACK_B10_DENSITY) -> openmc.Material:
    """Pure B-10 detector material at a given number density [atom/b-cm].

    100 (the default `detector_material`) is the BLACK detector of the benchmark.
    Realistic 1/v counters sit at ~1e-5 to 1e-3 (enriched BF3 at 1 atm is ~2.7e-5):
    the grey-detector study (analysis/grey_detector_study.py) dials this down to move
    the benchmark into the paper's low-efficiency regime at fixed geometry.
    """
    m = openmc.Material(name=f"Detector material (B10 {density:g} atom/b-cm)")
    m.add_nuclide("B10", 1.0, "ao")
    m.set_density("atom/b-cm", float(density))
    return m
