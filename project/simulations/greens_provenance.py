"""Per-cell DIRECT detection kernel from a single run via eigenvalue source provenance.

Method (OpenMC-validated recipe): run eigenvalue mode with ``batches=1, inactive=0``
(one generation), a fully-specified IndependentSource, ``write_initial_source=True``
and ``collision_track`` on the detector cell. The realized initial source bank is
written in particle-id order, so every tracked capture joins to its true birth site
via ``source_bank[parent_id - 1]``. This is immune to the FileSource random-sampling
bug that scrambled the original ``mesh_ids[parent_id - 1]`` attribution.

VALIDATED LIMITATION (see diagnostics.chain_completeness_check, gate verdict
FIRST-GENERATION-ONLY): in eigenvalue mode fission progeny are banked for a next
generation that never runs, so this records only first-generation events. The
measured kernel is therefore the DIRECT (unmultiplied) per-cell response g0_i(t)
-- detector captures of the source neutron itself, no fission-chain contribution.
The full-chain Green's function g_i(t) ground truth remains
``project.simulations.greens_sweep`` (one confined-source run per cell). The direct
kernel is the single-run complement: it isolates the geometric first-flight channel
of the direct-vs-multiplied decomposition (cf. greens_loader.directional_vs_time)
and attributes ALL coarse cells from one transport run.

More statistics = more seeds (one run per seed), NEVER more batches/generations:
parent_id recycles per generation and the collision bank has no generation field.

Run with:  python -m project.simulations.greens_provenance <label> [n_r n_th N n_seeds]
Output:    project/results/greens-provenance-k<label>.npz   (greens-sweep-compatible schema)
"""
import sys
from pathlib import Path

import h5py
import numpy as np
import openmc

from project.core.geometry import DETECTOR_CELL_ID, core_with_detector
from project.core.materials import detector_material, mixture_material
from project.core.rossi_mesh import KEFF_CONFIGS
from project.core.simulation import Simulation
from project.core.source_utils import read_initial_source
from project.core.sources import uniform_core_source

# defaults match the greens_sweep coarse grid so outputs are directly comparable
N_R_COARSE = 8
N_TH_COARSE = 10
N_PARTICLES = 1_000_000
N_SEEDS = 1
BASE_SEED = 20260612

# time-histogram config for g0_i(t); same edges as greens_sweep for comparability.
# (The direct kernel itself dies out within ~1 ms; the shared window keeps the
# npz schema interchangeable.)
T_MAX = 1.0e-2
N_T = 512


def coarse_edges(R: float, n_r: int, n_th: int):
    r_edges = R * np.linspace(0, 1, n_r + 1) ** (1 / 3)   # equal-volume shells
    cos_edges = np.linspace(1, -1, n_th + 1)              # equal solid-angle bands
    return r_edges, cos_edges


def bin_coarse(xyz: np.ndarray, r_edges: np.ndarray, cos_edges: np.ndarray) -> np.ndarray:
    """Map (N, 3) birth positions to flat coarse-cell index ir * n_th + ith (-1 outside)."""
    n_r, n_th = len(r_edges) - 1, len(cos_edges) - 1
    r = np.sqrt((xyz**2).sum(axis=1))
    cos = xyz[:, 2] / np.maximum(r, 1e-30)
    i_r = np.searchsorted(r_edges, r, side="right") - 1
    # cos_edges descend from 1 to -1; negate for an ascending search
    i_th = np.searchsorted(-cos_edges, -cos, side="right") - 1
    # clamp the cos=+1 / r=0 boundary cases into the first/owning bin
    i_th = np.clip(i_th, 0, n_th - 1)
    valid = (i_r >= 0) & (i_r < n_r)
    return np.where(valid, i_r * n_th + i_th, -1)


def run_seed(label: str, R: float, n: int, seed: int) -> Path:
    geometry = core_with_detector(
        core_center=(0.0, 0.0, 0.0), core_radius=R,
        detector_center=(0.0, 0.0, R + 15.0), detector_radius=5.0,
    )
    settings = {
        "run_mode": "eigenvalue",
        "batches": 1,
        "inactive": 0,
        "particles": n,
        "seed": seed,
        "write_initial_source": True,  # realized source in particle-id order
        "collision_track": {
            "max_collisions": 50 * n, "reactions": [101], "cell_ids": [DETECTOR_CELL_ID],
        },
        "source": uniform_core_source(R),
    }
    sim = Simulation(
        materials=openmc.Materials([mixture_material, detector_material]),
        geometry=geometry, settings=settings,
        name=f"greens-provenance-k{label}-seed{seed}",
    )
    sim.build()
    sim.run()
    return sim.output_dir


if __name__ == "__main__":
    label = sys.argv[1] if len(sys.argv) > 1 else "098"
    n_r = int(sys.argv[2]) if len(sys.argv) > 2 else N_R_COARSE
    n_th = int(sys.argv[3]) if len(sys.argv) > 3 else N_TH_COARSE
    n = int(sys.argv[4]) if len(sys.argv) > 4 else N_PARTICLES
    n_seeds = int(sys.argv[5]) if len(sys.argv) > 5 else N_SEEDS

    target_k = [k for k, v in KEFF_CONFIGS.items() if v["label"] == f"k{label}"][0]
    R = KEFF_CONFIGS[target_k]["radius"]
    time_edges = np.linspace(0.0, T_MAX, N_T + 1)
    r_edges, cos_edges = coarse_edges(R, n_r, n_th)
    n_cells = n_r * n_th

    n_src_cell = np.zeros(n_cells, dtype=np.int64)
    n_prompt = np.zeros(n_cells, dtype=np.int64)
    n_delayed = np.zeros(n_cells, dtype=np.int64)
    n_total_events = 0
    g_t = np.zeros((n_cells, N_T))

    print(f"Provenance direct kernel k={target_k} R={R} grid {n_r}x{n_th}, "
          f"N={n:,}/seed x {n_seeds} seed(s)")
    for s in range(n_seeds):
        seed = BASE_SEED + s
        outdir = run_seed(label, R, n, seed)

        birth_xyz, bank = read_initial_source(outdir / "initial_source.h5")
        assert len(bank) == n, f"initial source bank {len(bank)} != particles {n}"
        birth_cell = bin_coarse(birth_xyz, r_edges, cos_edges)
        np.add.at(n_src_cell, birth_cell[birth_cell >= 0], 1)

        with h5py.File(outdir / "collision_track.h5", "r") as f:
            a = f["collision_track_bank"][()]
        pid = a["parent_id"].astype(np.int64) - 1  # 1-based -> row index
        ok = (pid >= 0) & (pid < n)
        if not ok.all():
            print(f"  !! seed {seed}: {int((~ok).sum())} events with out-of-range parent_id")
        cell = birth_cell[pid[ok]]
        t = np.asarray(a["time"], dtype=float)[ok]
        prompt = (a["delayed_group"][ok] == 0) & (cell >= 0)
        delayed = (a["delayed_group"][ok] != 0) & (cell >= 0)
        np.add.at(n_prompt, cell[prompt], 1)
        np.add.at(n_delayed, cell[delayed], 1)
        n_total_events += int(a.shape[0])

        it = np.searchsorted(time_edges, t[prompt], side="right") - 1
        in_window = (it >= 0) & (it < N_T)
        np.add.at(g_t, (cell[prompt][in_window], it[in_window]), 1.0)

        print(f"  seed {seed}: {a.shape[0]:,} detections, "
              f"{int(prompt.sum()):,} prompt attributed")

    # reshape to the (n_r, n_th) sweep layout
    shape = (n_r, n_th)
    n_src_cell = n_src_cell.reshape(shape)
    n_prompt = n_prompt.reshape(shape)
    n_delayed = n_delayed.reshape(shape)
    g_t = g_t.reshape(n_r, n_th, N_T)

    with np.errstate(divide="ignore", invalid="ignore"):
        G = np.where(n_src_cell > 0, n_prompt / np.maximum(n_src_cell, 1), 0.0)
    G_err = G / np.sqrt(np.maximum(n_prompt, 1))

    # cell centroids and detector distance (identical math to greens_sweep)
    r_c = 0.5 * (r_edges[:-1] + r_edges[1:])[:, None] * np.ones((1, n_th))
    cos_c = np.ones((n_r, 1)) * 0.5 * (cos_edges[:-1] + cos_edges[1:])[None, :]
    det_z = R + 15.0
    dist = np.sqrt(r_c**2 + det_z**2 - 2 * r_c * det_z * cos_c)

    out = Path(__file__).parent.parent / "results" / f"greens-provenance-k{label}.npz"
    np.savez_compressed(
        out, k=target_k, R=R,
        # scalar n_source kept for greens_loader schema compatibility: with a
        # uniform-volume source on equal-volume cells, births/cell ~ total/cells
        n_source=float(n * n_seeds) / n_cells,
        r_edges=r_edges, cos_edges=cos_edges, time_edges=time_edges,
        G=G, G_err=G_err, n_prompt=n_prompt, n_total=n_prompt + n_delayed,
        dist=dist, r_mid=r_c, cos_mid=cos_c, g_t=g_t,
        n_source_per_cell=n_src_cell, n_delayed=n_delayed,
        n_seeds=n_seeds, seeds=np.arange(BASE_SEED, BASE_SEED + n_seeds),
        method="provenance", kernel="direct",
    )
    print(f"Saved -> {out}")
    print(f"  total detections {n_total_events:,}; "
          f"G(min..max) = {G[G > 0].min():.2e} .. {G.max():.2e}")
    if n_prompt[-1, -1] > 0:
        print(f"  near/far outer-shell ratio ~ {G[-1, 0] / G[-1, -1]:.1f}")
    else:
        print(f"  far-pole outer cell has 0 prompt counts (near-pole has "
              f"{int(n_prompt[-1, 0])}) -- increase N or n_seeds for a ratio")
