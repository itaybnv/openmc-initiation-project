"""Global uniform-source run: source of the per-cell FISSION TALLY F_i (any k).

One parameterized runner replaces the former greens_function_k040/k070/k090 and
greens_function_k098_fundsrc copies. The fission tally F_i on the fine spherical
mesh is consumed by ``project.analysis.greens_loader.fission_coarse`` for the
Rossi R(tau) formula.

NOTE: this run no longer pre-generates a source file. The old
``pregenerate_source`` + ``mesh_ids[parent_id-1]`` attribution was INVALID
(openmc.FileSource samples random sites, so id != source-file row) and the
FileSource swap bought nothing for a tally-only run. Per-cell detector response
comes from ``greens_sweep`` (full chain) or ``greens_provenance`` (direct kernel).

Run with:
    python -m project.simulations.greens_function <label> [--source uniform|fundamental]
e.g.
    python -m project.simulations.greens_function 040
    python -m project.simulations.greens_function 098 --source fundamental

``--source fundamental`` emits from the converged eigenvalue source bank written by
``python -m project.simulations.eigenvalue_for_source <label>`` instead of the
uniform-volume Watt source.

Output: project/results/greens-function-k<label>[-fundsrc]/
"""
import argparse
from pathlib import Path

import openmc

from project.core.geometry import core_with_detector
from project.core.materials import BLACK_B10_DENSITY, detector_material_b10, mixture_material
from project.core.physics_config import detector_settings
from project.core.rossi_mesh import KEFF_CONFIGS, make_fission_mesh
from project.core.simulation import Simulation
from project.core.sources import uniform_core_source

RESULTS_ROOT = Path(__file__).parent.parent / "results"


def fundamental_source(label: str) -> openmc.FileSource:
    """FileSource over the converged eigenvalue bank for this k label."""
    src_dir = RESULTS_ROOT / f"eigenvalue_source_k{label}"
    candidates = sorted(src_dir.glob("source.*.h5")) + [src_dir / "converged_source.h5"]
    for path in candidates:
        if path.exists():
            return openmc.FileSource(str(path))
    raise FileNotFoundError(
        f"No converged source bank in {src_dir}. "
        f"Run `python -m project.simulations.eigenvalue_for_source {label}` first."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("label", nargs="?", default="098",
                        choices=sorted(v["label"].removeprefix("k") for v in KEFF_CONFIGS.values()))
    parser.add_argument("--source", choices=["uniform", "fundamental"], default="uniform")
    parser.add_argument("--particles", type=int, default=1_000_000)
    parser.add_argument("--gap", type=float, default=10.0,
                        help="core-surface-to-detector gap [cm] (detector center at R+gap+5)")
    parser.add_argument("--b10-density", type=float, default=BLACK_B10_DENSITY, dest="b10",
                        help="detector B-10 number density [atom/b-cm]; 100 = the black "
                             "detector of the benchmark; ~1e-5..1e-3 = realistic grey 1/v "
                             "counters (output name gets -b10d<density>)")
    parser.add_argument("--tag", default="",
                        help="extra output-name suffix (e.g. -bench10M to avoid clobbering "
                             "the default-name 1M run with a larger-budget run)")
    args = parser.parse_args()

    target_k = [k for k, v in KEFF_CONFIGS.items() if v["label"] == f"k{args.label}"][0]
    radius = KEFF_CONFIGS[target_k]["radius"]

    geometry = core_with_detector(
        core_center=(0.0, 0.0, 0.0),
        core_radius=radius,
        detector_center=(0.0, 0.0, radius + args.gap + 5.0),  # +Z, aligned with mesh polar axis
        detector_radius=5.0,
    )
    settings = detector_settings(particles=args.particles)
    settings["source"] = (
        uniform_core_source(radius) if args.source == "uniform"
        else fundamental_source(args.label)
    )

    fission_tally = openmc.Tally(name="fission_per_cell")
    fission_tally.filters = [openmc.MeshFilter(make_fission_mesh(core_radius=radius))]
    fission_tally.scores = ["fission"]

    name = (
        f"greens-function-k{args.label}"
        + ("-fundsrc" if args.source == "fundamental" else "")
        + (f"-gap{args.gap:g}" if args.gap != 10.0 else "")
        + (f"-b10d{args.b10:.0e}" if args.b10 != BLACK_B10_DENSITY else "")
        + args.tag
    )
    simulation = Simulation(
        materials=openmc.Materials([mixture_material, detector_material_b10(args.b10)]),
        geometry=geometry,
        settings=settings,
        name=name,
    )
    simulation.build()
    simulation.model.tallies = openmc.Tallies([fission_tally])

    print(f"k={target_k} (R={radius} cm), source={args.source}, N={args.particles:,} -> {name}")
    simulation.run()
