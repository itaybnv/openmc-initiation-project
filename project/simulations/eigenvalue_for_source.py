"""Eigenvalue run that captures the converged fundamental-mode source bank.

Uses ``sourcepoint = {"separate": True, ...}`` so OpenMC writes a standalone
``source.<batch>.h5`` at the final batch -- directly consumable by
``openmc.FileSource`` (it carries the ``filetype`` attribute, unlike the
``initial_source.h5`` provenance dump). This replaces the old per-particle
ParticleList conversion loop over the statepoint source bank.

Run with: python -m project.simulations.eigenvalue_for_source [label]   (default 098)
Output:   project/results/eigenvalue_source_k<label>/source.<batch>.h5
Consumed by: python -m project.simulations.greens_function <label> --source fundamental
"""
import sys

import openmc

from project.core.geometry import core_sphere
from project.core.materials import mixture_material
from project.core.rossi_mesh import KEFF_CONFIGS
from project.core.simulation import Simulation

N_BATCHES = 60

if __name__ == "__main__":
    label = sys.argv[1] if len(sys.argv) > 1 else "098"
    target_k = [k for k, v in KEFF_CONFIGS.items() if v["label"] == f"k{label}"][0]
    radius = KEFF_CONFIGS[target_k]["radius"]

    settings = {
        "run_mode": "eigenvalue",
        "particles": 50_000,
        "batches": N_BATCHES,
        "inactive": 20,
        # separate=True -> standalone source.<batch>.h5 instead of embedding the
        # bank in the statepoint
        "sourcepoint": {"write": True, "separate": True, "batches": [N_BATCHES]},
    }

    sim = Simulation(
        materials=openmc.Materials([mixture_material]),
        geometry=core_sphere(center=(0.0, 0.0, 0.0), radius=radius),
        settings=settings,
        name=f"eigenvalue_source_k{label}",
    )
    sim.build()
    sim.run()

    with openmc.StatePoint(str(sim.output_dir / f"statepoint.{N_BATCHES}.h5")) as sp:
        k = sp.keff

    source_files = sorted(sim.output_dir.glob("source.*.h5"))
    if not source_files:
        raise FileNotFoundError(
            f"sourcepoint separate=True produced no source.*.h5 in {sim.output_dir}; "
            f"check the OpenMC version's sourcepoint file naming."
        )
    source_path = source_files[-1]
    # smoke-check the bank is FileSource-consumable
    n_bank = len(openmc.read_source_file(str(source_path)))

    print(f"\nR = {radius:.3f} cm   k_eff = {k.nominal_value:.5f} +/- {k.std_dev:.5f}")
    print(f"Converged source: {source_path} ({n_bank:,} particles)")
