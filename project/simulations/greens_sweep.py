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
from project.core.sources import confined_cell_source

# coarse sweep grid + statistics (overridable via argv)
N_R_COARSE = 8
N_TH_COARSE = 10
N_PARTICLES = 120_000

# time-histogram config for g_i(t) (prompt window)
T_MAX = 1.0e-2          # 10 ms covers the prompt decay (~2.6 ms)
N_T = 512


def coarse_edges(R: float, n_r: int, n_th: int):
    r_edges = R * np.linspace(0, 1, n_r + 1) ** (1 / 3)   # equal-volume shells
    cos_edges = np.linspace(1, -1, n_th + 1)              # equal solid-angle bands
    return r_edges, cos_edges


def run_one(R, label, ir, ith, r_lo, r_hi, cos_lo, cos_hi, n, time_edges, gap=10.0):
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
        "source": confined_cell_source(r_lo, r_hi, cos_lo, cos_hi),
    }
    sim = Simulation(
        materials=openmc.Materials([mixture_material, detector_material]),
        geometry=geometry, settings=settings,
        name=f"greens-sweep-k{label}-cell",   # reused dir; only the .h5 matters per-cell
    )
    sim.build()
    sim.run()
    with h5py.File(sim.output_dir / "collision_track.h5", "r") as f:
        a = f["collision_track_bank"][()]
    prompt = a["delayed_group"] == 0
    t = a["time"][prompt]
    g_hist, _ = np.histogram(t, bins=time_edges)
    n_prompt = int(prompt.sum())
    r_c, cos_c = 0.5 * (r_lo + r_hi), 0.5 * (cos_lo + cos_hi)
    dist = float(np.sqrt(r_c**2 + det_z**2 - 2 * r_c * det_z * cos_c))
    # per-run wall time (the per-cell dir is overwritten each cell, so read NOW)
    with openmc.StatePoint(str(sim.output_dir / "statepoint.1.h5")) as sp:
        runtime = dict(sp.runtime)
    return {
        "n_prompt": n_prompt, "g_hist": g_hist, "dist": dist,
        "r_mid": r_c, "cos_mid": cos_c, "n_total": int(a.shape[0]),
        "events": a, "transport_s": float(runtime["transport"]),
        "total_s": float(runtime["total"]),
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
    p.add_argument("--save-events", action="store_true",
                   help="save raw per-cell detection events (time, parent_id, delayed_group) "
                        "to <out>-events.h5 for chain-level error analysis")
    p.add_argument("--cells", nargs="*", default=None, metavar="IR,ITH",
                   help="run only these coarse cells (e.g. --cells 7,0 7,9 0,0)")
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
    time_edges = np.linspace(0.0, T_MAX, N_T + 1)
    r_edges, cos_edges = coarse_edges(R, n_r, n_th)

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
    g_t = np.zeros((n_r, n_th, N_T))
    transport_s = np.zeros((n_r, n_th))
    total_s = np.zeros((n_r, n_th))

    out_base = Path(__file__).parent.parent / "results" / f"greens-sweep-k{label}{suffix}"
    events_file = h5py.File(f"{out_base}-events.h5", "w") if args.save_events else None

    print(f"Sweep k={target_k} R={R} gap={args.gap} grid {n_r}x{n_th}, "
          f"{len(cells)} cell(s), N={n:,}/cell")
    for ir, ith in cells:
        res = run_one(
            R, label, ir, ith,
            r_edges[ir], r_edges[ir + 1], cos_edges[ith + 1], cos_edges[ith],
            n, time_edges, gap=args.gap,
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
        print(f"  cell ({ir:2d},{ith:2d}): G={G[ir, ith]:.3e}  "
              f"n_prompt={res['n_prompt']:6d}  transport={res['transport_s']:.2f}s")
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
            transport_s=transport_s, total_s=total_s,
        )
        print(f"Saved -> {out}  (near/far G ratio ~ {G[-1,0]/max(G[-1,-1],1e-12):.1f})")
        print(f"  total transport {transport_s.sum():.1f}s, total wall {total_s.sum():.1f}s")
