"""Criticality diagnostics for the heterogeneous plate line.

Reuses ``Simulation`` and ``eigenvalue_settings`` unchanged; only the geometry,
materials and source are plate-specific. Results go to ``results/plate-*`` so the
homogeneous-sphere artifacts are never touched.

Run (from the project root):
  python -m project.simulations.plate_criticality unitcell   # k_inf + disadvantage factor
  python -m project.simulations.plate_criticality scan        # water-gap moderation scan
  python -m project.simulations.plate_criticality stack 40    # finite N-plate k_eff
  python -m project.simulations.plate_criticality coupled     # loosely-coupled k_eff (PLATE_CONFIGS)
"""
import sys

import numpy as np
import openmc

from project.core.plate_materials import fuel_meat, clad, water, graphite, detector_material
from project.core import plate_geometry as pg
from project.core.plate_mesh import PLATE_CONFIGS
from project.core.physics_config import eigenvalue_settings
from project.core.simulation import Simulation

MATERIALS = openmc.Materials([fuel_meat, clad, water, graphite, detector_material])


def _box_source(lower, upper) -> openmc.IndependentSource:
    return openmc.IndependentSource(
        space=openmc.stats.Box(lower, upper), constraints={"fissionable": True})


def _keff(geometry, source, name, tallies=None,
          particles=5000, batches=80, inactive=25):
    settings = eigenvalue_settings(particles=particles, batches=batches, inactive=inactive)
    settings["source"] = source
    sim = Simulation(MATERIALS, geometry, settings, name=name)
    sim.build()
    if tallies is not None:
        sim.model.tallies = tallies
    sp = sim.run()
    with openmc.StatePoint(sp) as st:
        k = st.keff
        data = {t.name: st.get_tally(name=t.name).mean.ravel() for t in (tallies or [])}
    return k, data


def unitcell():
    """k_inf of the infinite plate lattice + thermal disadvantage factor."""
    geom = pg.unit_cell()
    bb = geom.bounding_box
    ef = openmc.EnergyFilter([0.0, 0.625, 20.0e6])
    cf = openmc.CellFilter([pg.MEAT_CELL_ID, pg.WATER_CELL_ID])
    t = openmc.Tally(name="fine_structure")
    t.filters = [cf, ef]
    t.scores = ["flux"]
    k, data = _keff(geom, _box_source(bb.lower_left, bb.upper_right),
                    "plate-unitcell", tallies=openmc.Tallies([t]))
    f = data["fine_structure"]  # [meat_fast, meat_th, water_fast, water_th]
    # volume-normalise flux by slab thickness (equal width/height)
    zeta = (f[3] / pg.WATER_GAP) / (f[1] / pg.T_MEAT)
    print(f"k_inf = {k.n:.4f} +/- {k.s:.4f}")
    print(f"thermal disadvantage factor zeta = {zeta:.3f}  (~1 expected for thin plates)")


def scan():
    """Water-gap moderation scan of the unit-cell k_inf."""
    print("water-gap moderation scan (unit cell k_inf):")
    for gap in (0.10, 0.15, 0.223, 0.30, 0.45, 0.60, 0.90, 1.30):
        geom = pg.unit_cell(water_gap=gap)
        bb = geom.bounding_box
        k, _ = _keff(geom, _box_source(bb.lower_left, bb.upper_right),
                     f"plate-unitcell-gap{gap:.3f}", particles=4000, batches=60, inactive=20)
        print(f"  gap={gap:.3f} cm: k_inf = {k.n:.4f} +/- {k.s:.4f}")


def stack(n_plates=40, width=40.0):
    """Finite N-plate slab k_eff (water + graphite reflected)."""
    geom = pg.plate_stack(n_plates=n_plates, width=width, graphite_thickness=10.0)
    x_hi = pg.stack_fuel_x_hi(n_plates)
    src = _box_source((0.0, -width / 2, -pg.ACTIVE_HEIGHT / 2),
                      (x_hi, width / 2, pg.ACTIVE_HEIGHT / 2))
    k, _ = _keff(geom, src, f"plate-stack-n{n_plates}")
    print(f"plate_stack n={n_plates} width={width}: k_eff = {k.n:.4f} +/- {k.s:.4f}")


def coupled(label="coupled095"):
    """Loosely-coupled two-pack k_eff for a PLATE_CONFIGS entry."""
    c = PLATE_CONFIGS[label]
    geom = pg.coupled_plate_packs(
        n_a=c["n_a"], n_b=c["n_b"], coupling_gap=c["coupling_gap"],
        width=c["width"], graphite_thickness=c["graphite_thickness"],
        refl_water=c["refl_water"])
    x_hi = pg.coupled_fuel_x_hi(c["n_a"], c["n_b"], c["coupling_gap"])
    src = _box_source((0.0, -c["width"] / 2, -pg.ACTIVE_HEIGHT / 2),
                      (x_hi, c["width"] / 2, pg.ACTIVE_HEIGHT / 2))
    k, _ = _keff(geom, src, f"plate-coupled-{label}")
    print(f"coupled '{label}' {c}: k_eff = {k.n:.4f} +/- {k.s:.4f}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "coupled"
    if cmd == "unitcell":
        unitcell()
    elif cmd == "scan":
        scan()
    elif cmd == "stack":
        stack(int(sys.argv[2]) if len(sys.argv) > 2 else 40)
    elif cmd == "coupled":
        coupled(sys.argv[2] if len(sys.argv) > 2 else "coupled095")
    else:
        raise SystemExit(f"unknown command {cmd!r} (unitcell|scan|stack|coupled)")
