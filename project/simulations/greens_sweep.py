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

from project.core.geometry import core_with_detector
from project.core.materials import detector_material, mixture_material
from project.core.rossi_mesh import KEFF_CONFIGS
from project.core.simulation import Simulation

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


def confined_source(r_lo, r_hi, cos_lo, cos_hi):
    return openmc.IndependentSource(
        space=openmc.stats.SphericalIndependent(
            r=openmc.stats.PowerLaw(r_lo, r_hi, 2),
            cos_theta=openmc.stats.Uniform(cos_lo, cos_hi),
            phi=openmc.stats.Uniform(0, 2 * np.pi),
            origin=(0.0, 0.0, 0.0),
        ),
        energy=openmc.stats.Watt(a=0.988e6, b=2.249e-6),
        time=openmc.stats.Discrete([0.0], [1.0]),
        strength=1.0,
    )


def run_one(R, label, ir, ith, r_lo, r_hi, cos_lo, cos_hi, n, time_edges):
    openmc.reset_auto_ids()
    geometry = core_with_detector(
        core_center=(0.0, 0.0, 0.0), core_radius=R,
        detector_center=(0.0, 0.0, R + 15.0), detector_radius=5.0,
    )
    settings = {
        "run_mode": "fixed source", "batches": 1, "particles": n,
        "collision_track": {"max_collisions": 50 * n, "reactions": [101], "cell_ids": [2]},
        "source": confined_source(r_lo, r_hi, cos_lo, cos_hi),
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
    det_z = R + 15.0
    dist = float(np.sqrt(r_c**2 + det_z**2 - 2 * r_c * det_z * cos_c))
    return n_prompt, g_hist, dist, r_c, cos_c, int(a.shape[0])


if __name__ == "__main__":
    label = sys.argv[1] if len(sys.argv) > 1 else "098"
    n_r = int(sys.argv[2]) if len(sys.argv) > 2 else N_R_COARSE
    n_th = int(sys.argv[3]) if len(sys.argv) > 3 else N_TH_COARSE
    n = int(sys.argv[4]) if len(sys.argv) > 4 else N_PARTICLES

    target_k = [k for k, v in KEFF_CONFIGS.items() if v["label"] == f"k{label}"][0]
    R = KEFF_CONFIGS[target_k]["radius"]
    time_edges = np.linspace(0.0, T_MAX, N_T + 1)
    r_edges, cos_edges = coarse_edges(R, n_r, n_th)

    G = np.zeros((n_r, n_th))
    G_err = np.zeros((n_r, n_th))
    n_prompt = np.zeros((n_r, n_th), dtype=int)
    n_total = np.zeros((n_r, n_th), dtype=int)
    dist = np.zeros((n_r, n_th))
    r_mid = np.zeros((n_r, n_th))
    cos_mid = np.zeros((n_r, n_th))
    g_t = np.zeros((n_r, n_th, N_T))

    print(f"Sweep k={target_k} R={R} grid {n_r}x{n_th}={n_r*n_th} cells, N={n:,}/cell")
    for ir in range(n_r):
        for ith in range(n_th):
            npr, gh, d, rc, cc, ntot = run_one(
                R, label, ir, ith,
                r_edges[ir], r_edges[ir + 1], cos_edges[ith + 1], cos_edges[ith], n, time_edges,
            )
            G[ir, ith] = npr / n
            G_err[ir, ith] = (npr / n) / np.sqrt(max(npr, 1))
            n_prompt[ir, ith] = npr
            n_total[ir, ith] = ntot
            dist[ir, ith] = d
            r_mid[ir, ith] = rc
            cos_mid[ir, ith] = cc
            g_t[ir, ith] = gh
        print(f"  r-shell {ir}: G(theta)= " + " ".join(f"{x:.2e}" for x in G[ir]))

    out = Path(__file__).parent.parent / "results" / f"greens-sweep-k{label}.npz"
    np.savez_compressed(
        out, k=target_k, R=R, n_source=n,
        r_edges=r_edges, cos_edges=cos_edges, time_edges=time_edges,
        G=G, G_err=G_err, n_prompt=n_prompt, n_total=n_total,
        dist=dist, r_mid=r_mid, cos_mid=cos_mid, g_t=g_t,
    )
    print(f"Saved -> {out}  (near/far G ratio ~ {G[-1,0]/max(G[-1,-1],1e-12):.1f})")
