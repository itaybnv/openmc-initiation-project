"""Detector perturbation study WITH statistical error bars.

Runs the core with and without the external B-10 detector using multi-batch
fixed-source settings (tallies only -- no collision_track, which must stay
single-batch), so tally std_dev is defined and the with/without difference can
be judged against its uncertainty. The original comparison used the batches=1
collision-track runs, where OpenMC cannot compute variances at all.

Run with:  python -m project.simulations.detector_perturbation
Output:    project/results/detector-perturbation-{with,without}/statepoint.<batches>.h5
"""
import openmc

from project.core.geometry import CORE_CELL_ID, core_sphere, core_with_detector
from project.core.materials import detector_material, mixture_material
from project.core.physics_config import perturbation_settings
from project.core.rossi_mesh import CORE_RADIUS
from project.core.simulation import Simulation


def core_tallies(geometry: openmc.Geometry) -> openmc.Tallies:
    cell_filter = openmc.CellFilter(geometry.get_all_cells()[CORE_CELL_ID])
    fission = openmc.Tally(name="fission")
    fission.filters = [cell_filter]
    fission.scores = ["fission"]
    nu_fission = openmc.Tally(name="nu-fission")
    nu_fission.filters = [cell_filter]
    nu_fission.scores = ["nu-fission"]
    return openmc.Tallies([fission, nu_fission])


def run_leg(name: str, geometry: openmc.Geometry, materials: openmc.Materials) -> dict:
    settings = perturbation_settings()
    sim = Simulation(materials=materials, geometry=geometry, settings=settings, name=name)
    sim.build()
    sim.model.tallies = core_tallies(geometry)
    sim.run()

    results = {}
    with openmc.StatePoint(str(sim.output_dir / f"statepoint.{settings['batches']}.h5")) as sp:
        for tname in ("fission", "nu-fission"):
            t = sp.get_tally(name=tname)
            results[tname] = (float(t.mean.ravel()[0]), float(t.std_dev.ravel()[0]))
    return results


if __name__ == "__main__":
    with_geom = core_with_detector(
        core_center=(0.0, 0.0, 0.0),
        core_radius=CORE_RADIUS,
        detector_center=(0.0, 0.0, CORE_RADIUS + 15.0),
        detector_radius=5.0,
    )
    res_with = run_leg(
        "detector-perturbation-with",
        with_geom,
        openmc.Materials([mixture_material, detector_material]),
    )

    res_without = run_leg(
        "detector-perturbation-without",
        core_sphere(center=(0.0, 0.0, 0.0), radius=CORE_RADIUS),
        openmc.Materials([mixture_material]),
    )

    print("\nDetector perturbation (per source particle, mean +/- 1 sigma):")
    for tname in ("fission", "nu-fission"):
        mw, sw = res_with[tname]
        mo, so = res_without[tname]
        diff = mw - mo
        sigma = (sw**2 + so**2) ** 0.5
        nsig = diff / sigma if sigma > 0 else float("nan")
        print(
            f"  {tname:10s}: with={mw:.5f}+/-{sw:.5f}  without={mo:.5f}+/-{so:.5f}  "
            f"diff={diff:+.5f} ({100 * diff / mo:+.3f}%) = {nsig:+.2f} sigma"
        )
    print("\n|diff| within ~2 sigma => detector perturbation not statistically resolvable.")
