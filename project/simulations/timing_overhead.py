"""Runtime equality experiment: one whole-sphere run vs 80 confined-cell runs.

Question: at equal total histories, does splitting transport into per-cell runs
cost the same as one global run? Expectation: equal transport time (same physics
work; per-cell rates differ -- center-born chains live longer -- but the SUM is
the meaningful comparison) plus a fixed per-run overhead (init, cross-section
load, statepoint IO) which at small per-cell N is no longer negligible.

The result feeds the wall-time axis of the G1 benchmark graph.

Run with:  python -m project.simulations.timing_overhead [label] [n_total]
Output:    project/results/timing-overhead-k<label>/summary.npz
"""
import sys
from pathlib import Path

import numpy as np
import openmc

from project.analysis.greens_loader import _fission_dir
from project.core.rossi_mesh import KEFF_CONFIGS
from project.simulations.greens_sweep import coarse_edges, run_one

N_R, N_TH = 8, 10
N_TOTAL = 1_000_000

if __name__ == "__main__":
    label = sys.argv[1] if len(sys.argv) > 1 else "090"
    n_total = int(sys.argv[2]) if len(sys.argv) > 2 else N_TOTAL
    n_cell = n_total // (N_R * N_TH)

    target_k = [k for k, v in KEFF_CONFIGS.items() if v["label"] == f"k{label}"][0]
    R = KEFF_CONFIGS[target_k]["radius"]
    r_edges, cos_edges = coarse_edges(R, N_R, N_TH)
    time_edges = np.linspace(0.0, 1.0e-2, 513)

    # reference: the existing 1M whole-sphere global run's recorded runtime
    with openmc.StatePoint(str(_fission_dir(label) / "statepoint.1.h5")) as sp:
        global_rt = dict(sp.runtime)
    print(f"global 1M run (existing): transport={global_rt['transport']:.2f}s "
          f"total={global_rt['total']:.2f}s")

    transport_s = np.zeros((N_R, N_TH))
    total_s = np.zeros((N_R, N_TH))
    print(f"per-cell: {N_R}x{N_TH} runs x {n_cell:,} histories = {N_R*N_TH*n_cell:,} total")
    for ir in range(N_R):
        for ith in range(N_TH):
            res = run_one(
                R, label, ir, ith,
                r_edges[ir], r_edges[ir + 1], cos_edges[ith + 1], cos_edges[ith],
                n_cell, time_edges,
            )
            transport_s[ir, ith] = res["transport_s"]
            total_s[ir, ith] = res["total_s"]
        print(f"  r-shell {ir}: transport "
              + " ".join(f"{x:.2f}" for x in transport_s[ir]))

    overhead = total_s - transport_s
    out_dir = Path(__file__).parent.parent / "results" / f"timing-overhead-k{label}"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / "summary.npz",
        k=target_k, n_total=n_total, n_cell=n_cell,
        transport_s=transport_s, total_s=total_s,
        global_transport=global_rt["transport"], global_total=global_rt["total"],
    )

    sum_tr, sum_tot = transport_s.sum(), total_s.sum()
    per_run_ovh = overhead.mean()
    hist_per_s = n_total / global_rt["transport"]
    print(f"\nSUM per-cell transport = {sum_tr:.2f}s vs global transport = "
          f"{global_rt['transport']:.2f}s  (ratio {sum_tr/global_rt['transport']:.3f})")
    print(f"SUM per-cell wall = {sum_tot:.2f}s vs global wall = {global_rt['total']:.2f}s")
    print(f"overhead/run = {per_run_ovh:.3f}s -> {N_R*N_TH} runs add "
          f"{overhead.sum():.1f}s ~= {overhead.sum()*hist_per_s/1e6:.2f}M histories "
          f"of transport at this k")
