"""Phase 3: corrected full spatial Green's function via a per-region source sweep.

Runs one confined-source fixed-source simulation per coarse mesh cell (r, theta). Because
the source is confined to a single cell, every detector capture originates from that cell
*by construction* -- no `parent_id` attribution, so this is immune to the FileSource
random-sampling bug that flattened the original method.

For each coarse cell i it records:
  * G_i      = (prompt detector captures) / (source neutrons)           [time-integrated g_i]
  * g_i(t)   = prompt detection-time histogram                          [for Rossi R(tau)]
  * geometry = distance to detector, r_mid, cos_mid

This is the "per emitted neutron including its full fission chain" definition required by
the Rossi/ACDC correlated-pair formula R(tau) = sum_i F_i * integral g_i(t) g_i(t+tau) dt.

Run one k_eff:   python -m project.simulations.greens_sweep 098 [n_r n_theta N]
Output:          project/results/greens-sweep-k<label>.npz

RECORD-LENGTH DESIGN RULE (--t-max):
The per-cell g_i(t) record must extend at least ~3 prompt decay times BEYOND the
upper edge of the intended Rossi fit window. The ACDC overlap integral at lag tau
only sees the part of the record where both g_i(t) and g_i(t+tau) exist, so a
finite record of length T_MAX suppresses R(tau) by the factor
    1 - exp(-2 |alpha| (T_MAX - tau)),
which droops the assembled R(tau) below truth as tau approaches T_MAX. The 10 ms
default is ample for k <= 0.95 (suppression < 1e-4 inside the window), but at
k = 0.98 (alpha ~ -370 /s, decay time ~2.7 ms) the window reaches ~9.5 ms and the
record must be ~40 ms -- see outputs/presentation/K098_EXTENSION_REPORT.md.
`--t-max` keeps the bin width fixed at DT = 1e-2/512 s and scales the bin count,
so a 40 ms record gives 2048 bins bit-compatible with the k=0.98 extension npz.

Optional cross-check (--tally-gt): in the SAME run, also measure g_i(t) with a
native CellFilter x TimeFilter absorption tally (stored as `g_t_tally` in the
npz, per source neutron). This does NOT replace the collision_track g_i(t) that
the ACDC method consumes -- it runs alongside it so the two estimators of the
same g_i(t) can be compared head-to-head. Over the prompt window (<=10 ms) the
tally and the prompt-only collision histogram measure the same thing: delayed
detections land at much later times (precursor decay ~0.1-10 s), so they fall
outside the window and don't contaminate the comparison.
"""
import sys
from pathlib import Path

import h5py
import numpy as np
import openmc

from project.core.geometry import DETECTOR_CELL_ID, core_with_detector
from project.core.materials import detector_material, mixture_material
from project.core.physics_config import detector_max_collisions
from project.core.rossi_mesh import KEFF_CONFIGS
from project.core.simulation import Simulation
from project.core.sources import confined_cell_source, fission_weighted_cell_source

# coarse sweep grid + statistics (overridable via argv)
N_R_COARSE = 8
N_TH_COARSE = 10
N_PARTICLES = 120_000

# time-histogram config for g_i(t) (prompt window). DT is the INVARIANT: --t-max
# scales the bin count so every record shares the same bin width (and so a longer
# record stays bin-compatible with the standard 10 ms one).
T_MAX = 1.0e-2          # 10 ms covers the prompt decay (~2.6 ms) for k <= 0.95
N_T = 512
DT = T_MAX / N_T        # 19.53125 us

# Tally analogue of the collision_track reactions=[101] detection: B-10 has no
# fission, so 'absorption' in the detector cell == the (n,a)+(n,g) capture rate.
DETECTION_SCORE = "absorption"


def coarse_edges(R: float, n_r: int, n_th: int):
    r_edges = R * np.linspace(0, 1, n_r + 1) ** (1 / 3)   # equal-volume shells
    cos_edges = np.linspace(1, -1, n_th + 1)              # equal solid-angle bands
    return r_edges, cos_edges


def radial_fission_density(sphere_dir: Path):
    """Fine radial fission density rho(r) from a uniform-pulse sphere run.

    Reads the `fission_per_cell` mesh tally (flat index i_theta*N_RADIAL + i_r),
    sums over theta and divides by the fine-shell volume (equal-volume shells, so
    this is a constant). Returns (r_fine_edges, rho_fine) with rho in
    fissions / source / cm^3; only its shape matters for the source weighting.
    """
    from project.core.rossi_mesh import N_RADIAL, N_THETA

    with openmc.StatePoint(str(Path(sphere_dir) / "statepoint.1.h5")) as sp:
        t = sp.get_tally(name="fission_per_cell")
        mesh = t.find_filter(openmc.MeshFilter).mesh
        r_fine_edges = np.asarray(mesh.r_grid, float)
        F_fine = t.mean.flatten().reshape(N_THETA, N_RADIAL)
    F_r = F_fine.sum(axis=0)
    vol = 4 * np.pi / 3 * np.diff(r_fine_edges**3)
    return r_fine_edges, F_r / vol


def run_one(R, label, ir, ith, r_lo, r_hi, cos_lo, cos_hi, n, time_edges, gap=10.0,
            tally_gt=False, fission_weight=None):
    det_z = R + gap + 5.0  # detector CENTER on +Z (gap to surface + detector radius)
    geometry = core_with_detector(
        core_center=(0.0, 0.0, 0.0), core_radius=R,
        detector_center=(0.0, 0.0, det_z), detector_radius=5.0,
    )
    settings = {
        "run_mode": "fixed source", "batches": 1, "particles": n,
        "collision_track": {
            "max_collisions": detector_max_collisions(n), "reactions": [101],
            "cell_ids": [DETECTOR_CELL_ID],
        },
        "source": (
            confined_cell_source(r_lo, r_hi, cos_lo, cos_hi)
            if fission_weight is None
            else fission_weighted_cell_source(r_lo, r_hi, cos_lo, cos_hi, *fission_weight)
        ),
    }
    sim = Simulation(
        materials=openmc.Materials([mixture_material, detector_material]),
        geometry=geometry, settings=settings,
        name=f"greens-sweep-k{label}-cell",   # reused dir; only the .h5 matters per-cell
    )
    sim.build()
    if tally_gt:
        # Native g_i(t): absorption rate in the detector cell, binned by detection
        # time on the SAME edges as the collision histogram. batches=1 here (forced
        # by collision_track), so this is a single-realization mean with no std_dev
        # -- enough for a shape/normalization cross-check against g_hist.
        gt_tally = openmc.Tally(name="g_of_t")
        gt_tally.filters = [
            openmc.CellFilter([DETECTOR_CELL_ID]),
            openmc.TimeFilter(time_edges),
        ]
        gt_tally.scores = [DETECTION_SCORE]
        sim.model.tallies = openmc.Tallies([gt_tally])
    sim.run()
    with h5py.File(sim.output_dir / "collision_track.h5", "r") as f:
        a = f["collision_track_bank"][()]
    prompt = a["delayed_group"] == 0
    t = a["time"][prompt]
    g_hist, _ = np.histogram(t, bins=time_edges)
    n_prompt = int(prompt.sum())
    r_c, cos_c = 0.5 * (r_lo + r_hi), 0.5 * (cos_lo + cos_hi)
    dist = float(np.sqrt(r_c**2 + det_z**2 - 2 * r_c * det_z * cos_c))
    # per-run wall time + optional tally g(t) (the per-cell dir is overwritten each
    # cell, so read NOW). g_tally is detections per SOURCE neutron per time bin,
    # directly comparable to g_hist / n.
    g_tally = None
    with openmc.StatePoint(str(sim.output_dir / "statepoint.1.h5")) as sp:
        runtime = dict(sp.runtime)
        if tally_gt:
            g_tally = sp.get_tally(name="g_of_t").mean.ravel().astype(float)
    return {
        "n_prompt": n_prompt, "g_hist": g_hist, "dist": dist,
        "r_mid": r_c, "cos_mid": cos_c, "n_total": int(a.shape[0]),
        "events": a, "transport_s": float(runtime["transport"]),
        "total_s": float(runtime["total"]), "g_tally": g_tally,
    }


def parse_args(argv=None):
    import argparse

    p = argparse.ArgumentParser(description="Confined-source per-cell Green's function sweep")
    p.add_argument("label", nargs="?", default="098")
    p.add_argument("n_r", nargs="?", type=int, default=N_R_COARSE)
    p.add_argument("n_th", nargs="?", type=int, default=N_TH_COARSE)
    p.add_argument("n", nargs="?", type=int, default=N_PARTICLES)
    p.add_argument("--gap", type=float, default=10.0,
                   help="core-surface-to-detector gap [cm] (detector center at R+gap+5)")
    p.add_argument("--t-max", type=float, default=T_MAX, dest="t_max",
                   help="per-cell g_i(t) record length [s] (default 1e-2). The bin "
                        "width is held at DT = 1e-2/512 s and the bin count scales, "
                        "so 4e-2 gives 2048 bins. Design rule: the record must reach "
                        "~3 prompt decay times past the top of the intended Rossi fit "
                        "window (40 ms at k=0.98) or the g*g overlap integral is "
                        "suppressed by 1 - exp(-2|alpha|(T_MAX - tau)).")
    p.add_argument("--save-events", action="store_true",
                   help="save raw per-cell detection events (time, parent_id, delayed_group) "
                        "to <out>-events.h5 for chain-level error analysis")
    p.add_argument("--tally-gt", action="store_true",
                   help="also measure g_i(t) with a native CellxTime absorption tally "
                        "(stored as g_t_tally) for a head-to-head check against the "
                        "collision_track g_i(t); does not change the collision method")
    p.add_argument("--cells", nargs="*", default=None, metavar="IR,ITH",
                   help="run only these coarse cells (e.g. --cells 7,0 7,9 0,0)")
    p.add_argument("--fission-weighted", default=None, metavar="SPHERE_DIR",
                   dest="fission_weighted",
                   help="sample the in-cell birth RADIUS from the fission map of this "
                        "uniform-pulse sphere run (results dir name, e.g. "
                        "greens-function-k098-bench100M) instead of uniformly in "
                        "volume; angle stays uniform. Tests the B8 within-cell "
                        "weighting diagnosis of the k=0.98 1.5%% residual")
    p.add_argument("--tag", default="",
                   help="extra output-name suffix (e.g. -gap10 to avoid clobbering "
                        "an existing default-name npz with a different grid/budget)")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    label, n_r, n_th, n = args.label, args.n_r, args.n_th, args.n
    suffix = ("" if args.gap == 10.0 else f"-gap{args.gap:g}") + args.tag

    target_k = [k for k, v in KEFF_CONFIGS.items() if v["label"] == f"k{label}"][0]
    R = KEFF_CONFIGS[target_k]["radius"]
    t_max = float(args.t_max)
    n_t = int(round(t_max / DT))          # bin width pinned to DT (see design rule)
    time_edges = np.linspace(0.0, n_t * DT, n_t + 1)
    r_edges, cos_edges = coarse_edges(R, n_r, n_th)

    fission_weight = None
    if args.fission_weighted:
        sphere_dir = Path(__file__).parent.parent / "results" / args.fission_weighted
        fission_weight = radial_fission_density(sphere_dir)
        r_fine, rho = fission_weight
        if not np.allclose(r_fine[::len(r_fine) // n_r][: n_r + 1], r_edges):
            raise SystemExit("fine fission mesh does not nest on the coarse sweep grid")
        print(f"  fission-weighted in-cell source from {sphere_dir.name}: "
              f"rho(0)/rho(R) = {rho[0] / rho[-1]:.2f}")

    cells = (
        [(ir, ith) for ir in range(n_r) for ith in range(n_th)]
        if args.cells is None
        else [tuple(int(x) for x in c.split(",")) for c in args.cells]
    )

    G = np.zeros((n_r, n_th))
    G_err = np.zeros((n_r, n_th))
    n_prompt = np.zeros((n_r, n_th), dtype=int)
    n_total = np.zeros((n_r, n_th), dtype=int)
    dist = np.zeros((n_r, n_th))
    r_mid = np.zeros((n_r, n_th))
    cos_mid = np.zeros((n_r, n_th))
    g_t = np.zeros((n_r, n_th, n_t))
    g_t_tally = np.zeros((n_r, n_th, n_t))   # native tally g_i(t), per source neutron
    transport_s = np.zeros((n_r, n_th))
    total_s = np.zeros((n_r, n_th))

    out_base = Path(__file__).parent.parent / "results" / f"greens-sweep-k{label}{suffix}"
    events_file = h5py.File(f"{out_base}-events.h5", "w") if args.save_events else None

    print(f"Sweep k={target_k} R={R} gap={args.gap} grid {n_r}x{n_th}, "
          f"{len(cells)} cell(s), N={n:,}/cell")
    print(f"  g_i(t) record: T_MAX={n_t * DT * 1e3:.3f} ms in {n_t} bins "
          f"(dt={DT * 1e6:.5f} us)")
    for ir, ith in cells:
        res = run_one(
            R, label, ir, ith,
            r_edges[ir], r_edges[ir + 1], cos_edges[ith + 1], cos_edges[ith],
            n, time_edges, gap=args.gap, tally_gt=args.tally_gt,
            fission_weight=fission_weight,
        )
        G[ir, ith] = res["n_prompt"] / n
        G_err[ir, ith] = G[ir, ith] / np.sqrt(max(res["n_prompt"], 1))
        n_prompt[ir, ith] = res["n_prompt"]
        n_total[ir, ith] = res["n_total"]
        dist[ir, ith] = res["dist"]
        r_mid[ir, ith] = res["r_mid"]
        cos_mid[ir, ith] = res["cos_mid"]
        g_t[ir, ith] = res["g_hist"]
        transport_s[ir, ith] = res["transport_s"]
        total_s[ir, ith] = res["total_s"]
        msg = (f"  cell ({ir:2d},{ith:2d}): G={G[ir, ith]:.3e}  "
               f"n_prompt={res['n_prompt']:6d}  transport={res['transport_s']:.2f}s")
        if res["g_tally"] is not None:
            g_t_tally[ir, ith] = res["g_tally"]
            # time-integrated tally detections/source vs collision G (should agree)
            msg += f"  G_tally={res['g_tally'].sum():.3e}"
        print(msg)
        if events_file is not None:
            grp = events_file.create_group(f"cell_{ir}_{ith}")
            a = res["events"]
            grp.create_dataset("time", data=a["time"])
            grp.create_dataset("parent_id", data=a["parent_id"])
            grp.create_dataset("delayed_group", data=a["delayed_group"])
            grp.attrs["n_source"] = n

    if events_file is not None:
        events_file.attrs.update({"k": target_k, "R": R, "gap": args.gap, "n_source": n})
        events_file.close()
        print(f"Events -> {out_base}-events.h5")

    if args.cells is None:  # only write the full npz for complete sweeps
        out = Path(f"{out_base}.npz")
        np.savez_compressed(
            out, k=target_k, R=R, n_source=n, gap=args.gap,
            r_edges=r_edges, cos_edges=cos_edges, time_edges=time_edges,
            G=G, G_err=G_err, n_prompt=n_prompt, n_total=n_total,
            dist=dist, r_mid=r_mid, cos_mid=cos_mid, g_t=g_t,
            g_t_tally=g_t_tally, transport_s=transport_s, total_s=total_s,
            fission_weighted=(args.fission_weighted or ""),
            rho_r_fine=(fission_weight[1] if fission_weight else np.zeros(0)),
        )
        print(f"Saved -> {out}  (near/far G ratio ~ {G[-1,0]/max(G[-1,-1],1e-12):.1f})")
        print(f"  total transport {transport_s.sum():.1f}s, total wall {total_s.sum():.1f}s")
