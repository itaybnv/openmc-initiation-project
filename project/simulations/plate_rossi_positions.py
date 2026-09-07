"""Phase-0 spatial-kinetics gate: detector-position-dependent prompt-alpha.

In the loosely-coupled plate config (``PLATE_CONFIGS``) an asymmetric source
(pack A only) excites the coupled two-region transient. The detector is placed at
two contrasting positions -- 'left' (near source pack A, backed by the graphite
reflector) and 'right' (the bare far face, fed through the water coupling gap).
The apparent prompt-alpha, fit over a fixed window, differs between the positions
iff higher spatial modes matter: the signature the single-alpha homogeneous sphere
cannot produce, and the time-domain extension of the static IRR-1 point-vs-exact
discrepancy of Gabrieli, Shokef & Neder (arXiv:2411.14388).

Reuses ``detector_settings`` + the ``collision_track`` readout + ``timing_utils``
unchanged; only the geometry/source are plate-specific. Results -> results/plate-*.

Run (from the project root):
  python -m project.simulations.plate_rossi_positions [particles] [label]
"""
import sys

import numpy as np
import openmc

from project.core.plate_materials import fuel_meat, clad, water, graphite, detector_material
from project.core import plate_geometry as pg
from project.core.plate_mesh import PLATE_CONFIGS
from project.core.plate_sources import uniform_fuel_source
from project.core.physics_config import detector_settings
from project.core.simulation import Simulation
from project.analysis.timing_utils import load_detector_times, log_hist, fit_exp_decay

MATERIALS = openmc.Materials([fuel_meat, clad, water, graphite, detector_material])

# Prompt die-away window (clean single-exponential region; before the delayed floor).
FIT_LO, FIT_HI = 3e-4, 8e-3


def run_position(cfg: dict, side: str, particles: int):
    """Run the coupled config with the detector at ``side`` and fit apparent alpha."""
    pos = pg.coupled_detector_positions(cfg["n_a"], cfg["n_b"],
                                        cfg["coupling_gap"], cfg["refl_water"])
    geom = pg.coupled_plate_packs(
        n_a=cfg["n_a"], n_b=cfg["n_b"], coupling_gap=cfg["coupling_gap"],
        width=cfg["width"], graphite_thickness=cfg["graphite_thickness"],
        refl_water=cfg["refl_water"], detector_center=pos[side])
    settings = detector_settings(particles=particles)
    # Asymmetric source confined to pack A (x in [0, n_a*PITCH]).
    settings["source"] = uniform_fuel_source(
        0.0, cfg["n_a"] * pg.PITCH, cfg["width"], pg.ACTIVE_HEIGHT)
    sim = Simulation(MATERIALS, geom, settings, name=f"plate-rossi-{side}")
    sim.run()

    t = load_detector_times(sim.output_dir / "collision_track.h5", prompt_only=True)
    h = log_hist(t, n_bins=80)
    m = (h["center"] > FIT_LO) & (h["center"] < FIT_HI)
    popt, perr = fit_exp_decay(h["center"][m].values, h["density"][m].values,
                               p0=[h["density"][m].max(), 900.0, 0.0])
    return {"n": int(t.size), "alpha": float(popt[1]), "alpha_err": float(perr[1])}


def main(particles: int = 500_000, label: str = "coupled095"):
    cfg = PLATE_CONFIGS[label]
    res = {side: run_position(cfg, side, particles) for side in ("left", "right")}
    for side in ("left", "right"):
        r = res[side]
        print(f"{side:5s}: n={r['n']:7d}  alpha = {r['alpha']:7.1f} +/- {r['alpha_err']:5.1f} /s")
    da = res["left"]["alpha"] - res["right"]["alpha"]
    sig = abs(da) / np.hypot(res["left"]["alpha_err"], res["right"]["alpha_err"])
    print(f"\nalpha_left - alpha_right = {da:+.1f} /s   ({sig:.1f} sigma)")
    print("GATE:", "PASS -- detector-position-dependent alpha"
          if sig > 3 else "inconclusive (raise statistics or coupling)")


if __name__ == "__main__":
    np_ = int(sys.argv[1]) if len(sys.argv) > 1 else 500_000
    lbl = sys.argv[2] if len(sys.argv) > 2 else "coupled095"
    main(np_, lbl)
