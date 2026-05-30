"""Parameterized eigenvalue sweep — pass a core radius, get k_eff back.

Usage:
    python -m project.simulations.eigenvalue_sweep <radius_cm>

Writes output to project/results/eigenvalue_r<radius>/ and prints
the final k_eff (value, std) for that radius.
"""
import sys
from pathlib import Path

import openmc

from project.core.geometry import core_sphere
from project.core.materials import mixture_material
from project.core.simulation import Simulation


def run_eigenvalue(radius: float) -> tuple[float, float]:
    materials = openmc.Materials([mixture_material])
    geometry = core_sphere(center=(0.0, 0.0, 0.0), radius=radius)
    settings = {
        "run_mode": "eigenvalue",
        "particles": 50_000,
        "batches": 60,
        "inactive": 20,
    }
    name = f"eigenvalue_r{radius:.3f}"
    sim = Simulation(materials=materials, geometry=geometry, settings=settings, name=name)
    sim.build()
    sim.run()

    sp_path = sim.output_dir / "statepoint.60.h5"
    with openmc.StatePoint(str(sp_path)) as sp:
        k = sp.keff
    return float(k.nominal_value), float(k.std_dev)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python -m project.simulations.eigenvalue_sweep <radius_cm>")
        sys.exit(1)
    R = float(sys.argv[1])
    k_val, k_std = run_eigenvalue(R)
    print(f"\n=== EIGENVALUE RESULT ===")
    print(f"R = {R:.3f} cm   k_eff = {k_val:.5f} ± {k_std:.5f}")
