"""Weight-window variance-reduction demonstration on the detector g(t) -- evaluated, not adopted.

Companion to `tally_vs_collision.py`. This is the experiment behind the project's
decision to STAY on collision_track (see notebooks/tally_vs_collision.ipynb): it
shows that weight windows are exactly unbiased on the singles detector response,
but that the variance reduction is modest, regime-dependent, and -- once the
one-time window-generation cost is charged -- did not pay for itself here.

Two-step CADIS-style recipe at matched histories:
  1. analog       -- multi-batch TimeFilter absorption tally, no variance reduction.
  2. generate     -- a MAGIC WeightWindowGenerator on a cylindrical mesh down the
                     +Z core->detector corridor writes weight_windows.h5.
  3. apply         -- the same tally re-run with those fixed weight windows.

Reports, for the total detector absorption/source:
  * UNBIASEDNESS -- the analog and weight-window means, and their difference in
    sigma (the physics check: |Δ| < ~2 sigma == statistically identical).
  * VARIANCE     -- per-history relative-error ratio (the "better variance" claim).
  * FOM          -- figure of merit 1/(rel_err^2 * time), charging the generation
    cost to the weight-window method (honest at small scale; the generation cost
    amortizes away over a large production run).

Run with:  python -m project.simulations.weight_window_demo [label] [--gap G]
                                                            [--histories N] [--batches B]
Measured (k=0.90, 240k histories/run, naive cylindrical map):
  gap 40 cm: means agree +0.30 sigma; 1.3x tighter/history; FOM 0.6x (WW lost)
  gap 90 cm: means agree -0.10 sigma; 1.5x tighter/history; FOM 0.7x (WW lost)
The variance win grows with starvation but stays modest: the detector sits across
a VOID, so the loss is geometric (solid angle), not attenuation -- the regime
where weight windows give orders of magnitude, not tens of percent.
"""
import argparse
import time

import numpy as np
import openmc

from project.core.geometry import DETECTOR_CELL_ID, core_with_detector
from project.core.materials import detector_material, mixture_material
from project.core.rossi_mesh import KEFF_CONFIGS
from project.core.sources import uniform_core_source

T_MAX = 1.0e-2
N_T = 64
DETECTION_SCORE = "absorption"


def _model(R, det_z, settings, time_edges):
    geom = core_with_detector(core_center=(0, 0, 0), core_radius=R,
                              detector_center=(0, 0, det_z), detector_radius=5.0)
    gt = openmc.Tally(name="g_of_t")
    gt.filters = [openmc.CellFilter([DETECTOR_CELL_ID]), openmc.TimeFilter(time_edges)]
    gt.scores = [DETECTION_SCORE]
    mats = openmc.Materials([mixture_material, detector_material])
    return openmc.Model(geometry=geom, materials=mats, settings=settings,
                        tallies=openmc.Tallies([gt]))


def _run(model, cwd):
    t0 = time.time()
    sp = model.run(cwd=cwd, output=False)
    wall = time.time() - t0
    with openmc.StatePoint(sp) as st:
        t = st.get_tally(name="g_of_t")
        return t.mean.ravel().astype(float), t.std_dev.ravel().astype(float), wall


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("label", nargs="?", default="090")
    p.add_argument("--gap", type=float, default=90.0)
    p.add_argument("--histories", type=int, default=240_000)
    p.add_argument("--batches", type=int, default=30)
    args = p.parse_args(argv)

    target_k = [k for k, v in KEFF_CONFIGS.items() if v["label"] == f"k{args.label}"][0]
    R = KEFF_CONFIGS[target_k]["radius"]
    det_z = R + args.gap + 5.0
    per = max(1, args.histories // args.batches)
    te = np.linspace(0.0, T_MAX, N_T + 1)
    base = dict(run_mode="fixed source", batches=args.batches, particles=per)

    print(f"k={target_k} (R={R} cm), gap={args.gap} cm, "
          f"{args.batches}x{per}={args.batches*per:,} histories/run")

    # 1. analog
    s = openmc.Settings(source=uniform_core_source(R), **base)
    s.output = {"summary": False}
    a_mean, a_std, a_wall = _run(_model(R, det_z, s, te), "/tmp/wwdemo_analog")

    # 2. generate weight windows (MAGIC on a +Z cylindrical corridor mesh)
    mesh = openmc.CylindricalMesh(
        r_grid=np.linspace(0, 30, 16),
        z_grid=np.linspace(-30, det_z + 10, 45),
        phi_grid=[0.0, 2 * np.pi],
    )
    s = openmc.Settings(source=uniform_core_source(R), **base)
    s.output = {"summary": False}
    s.weight_window_generators = [openmc.WeightWindowGenerator(
        mesh=mesh, particle_type="neutron", method="magic",
        max_realizations=args.batches, on_the_fly=True)]
    _run(_model(R, det_z, s, te), "/tmp/wwdemo_gen")
    ww = openmc.WeightWindowsList.from_hdf5("/tmp/wwdemo_gen/weight_windows.h5")

    # 3. apply fixed weight windows
    s = openmc.Settings(source=uniform_core_source(R), **base)
    s.output = {"summary": False}
    s.weight_windows = ww
    s.weight_windows_on = True
    w_mean, w_std, w_wall = _run(_model(R, det_z, s, te), "/tmp/wwdemo_apply")

    A, Aerr = a_mean.sum(), float(np.sqrt((a_std ** 2).sum()))
    W, Werr = w_mean.sum(), float(np.sqrt((w_std ** 2).sum()))
    relA, relW = Aerr / A, Werr / W
    diff_sigma = (W - A) / np.sqrt(Aerr ** 2 + Werr ** 2)
    fom_a = 1 / (relA ** 2 * a_wall)
    fom_w = 1 / (relW ** 2 * (w_wall + a_wall))   # charge generation (~analog cost) to WW

    print(f"\n==== detector absorption/source (k={target_k}, gap {args.gap} cm) ====")
    print(f"analog : {A:.4e} +/- {Aerr:.2e}  ({100*relA:.1f}%)")
    print(f"WW     : {W:.4e} +/- {Werr:.2e}  ({100*relW:.1f}%)")
    print(f"UNBIASED?  {diff_sigma:+.2f} sigma  (|<2| = identical)")
    print(f"VARIANCE?  {relA/relW:.1f}x tighter per history")
    print(f"FOM (gen charged): analog {fom_a:.1f}  WW {fom_w:.1f}  -> {fom_w/fom_a:.1f}x")


if __name__ == "__main__":
    main()
