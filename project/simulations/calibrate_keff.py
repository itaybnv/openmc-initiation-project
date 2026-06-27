"""Calibrate core radii for the KEFF_CONFIGS targets via criticality search.

Replaces manual radius hunting with ``openmc.search_for_keff`` (bisection on a
bracket around the current KEFF_CONFIGS radius). Prints a recommended
KEFF_CONFIGS block; it does NOT edit rossi_mesh.py -- apply the radii by hand if
they differ meaningfully from the current calibration.

Run with:  python -m project.simulations.calibrate_keff [0.40|0.70|0.90|0.98|all]
"""
import sys
from pathlib import Path

import openmc

from project.core.geometry import core_sphere
from project.core.materials import mixture_material
from project.core.rossi_mesh import KEFF_CONFIGS

RESULTS_ROOT = Path(__file__).parent.parent / "results"

# per-iteration transport statistics (matches the manual eigenvalue_sweep runs)
PARTICLES = 50_000
BATCHES = 60
INACTIVE = 20


def model_builder(radius: float) -> openmc.Model:
    geometry = core_sphere(center=(0.0, 0.0, 0.0), radius=float(radius))
    settings = openmc.Settings(
        run_mode="eigenvalue",
        particles=PARTICLES,
        batches=BATCHES,
        inactive=INACTIVE,
    )
    return openmc.Model(
        geometry=geometry,
        materials=openmc.Materials([mixture_material]),
        settings=settings,
    )


def calibrate(target_k: float) -> tuple[float, list, list]:
    r0 = KEFF_CONFIGS[target_k]["radius"]
    outdir = RESULTS_ROOT / f"calibrate-keff-{KEFF_CONFIGS[target_k]['label']}"
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"\nSearching k={target_k} in radius bracket [{r0 - 2.0:.2f}, {r0 + 2.0:.2f}] cm")
    crit_r, guesses, keffs = openmc.search_for_keff(
        model_builder,
        bracket=[r0 - 2.0, r0 + 2.0],
        target=target_k,
        tol=1e-3,  # relative tolerance on the radius
        print_iterations=True,
        run_args={"cwd": str(outdir), "output": False},
    )
    return crit_r, guesses, keffs


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    targets = sorted(KEFF_CONFIGS) if arg == "all" else [float(arg)]

    calibrated: dict[float, float] = {}
    for target_k in targets:
        crit_r, guesses, keffs = calibrate(target_k)
        calibrated[target_k] = crit_r
        print(f"  -> R(k={target_k}) = {crit_r:.3f} cm  "
              f"(was {KEFF_CONFIGS[target_k]['radius']:.3f} cm; "
              f"{len(guesses)} transport runs)")

    print("\nRecommended KEFF_CONFIGS block (apply to project/core/rossi_mesh.py if changed):")
    print("KEFF_CONFIGS: dict[float, dict] = {")
    for k in sorted(calibrated, reverse=True):
        label = KEFF_CONFIGS[k]["label"]
        print(f'    {k:.2f}: {{"radius": {calibrated[k]:.3f}, "label": "{label}"}},')
    print("}")
