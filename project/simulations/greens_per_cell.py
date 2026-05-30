"""Phase 2: ground-truth per-cell Green's function via confined-source simulations.

For a chosen set of mesh cells, run a fixed-source simulation with the source CONFINED
to that single cell. Every detector capture then originates from that cell *by
construction* -- no `parent_id` attribution, so this is immune to the FileSource
random-sampling bug. The measured

    G_cell = (detector captures) / (source neutrons)

is exactly the method's per-cell impulse response g_i integrated over time (response per
neutron emitted from cell i, including its full fission chain).

This is the bulletproof ground truth. The buggy combined method reports G ~ const for all
cells; if these confined-source runs show G varying between near-pole / far-pole / center,
the bug is proven and the true spatial structure is revealed.

Run with:  python -m project.simulations.greens_per_cell <keff-label>   e.g. 098 or 040
"""
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import openmc

from project.core.geometry import core_with_detector
from project.core.materials import detector_material, mixture_material
from project.core.rossi_mesh import KEFF_CONFIGS, N_RADIAL, N_THETA
from project.core.simulation import Simulation

N_PARTICLES = {"098": 200_000, "090": 300_000, "070": 400_000, "040": 500_000}


def mesh_edges(R: float):
    r_edges = R * np.linspace(0, 1, N_RADIAL + 1) ** (1 / 3)
    cos_edges = np.linspace(1, -1, N_THETA + 1)  # decreasing 1 -> -1
    return r_edges, cos_edges


def critical_cells(R: float) -> dict:
    """Return {label: (r_lo, r_hi, cos_lo, cos_hi, dist_to_detector_cm)} for key cells.

    near/far poles share the OUTERMOST radial shell (isolates the directional effect);
    center vs surface isolates the radial/importance effect.
    """
    r_edges, cos_edges = mesh_edges(R)
    det_z = R + 15.0

    def dist(r_lo, r_hi, cos_lo, cos_hi):
        r_c = 0.5 * (r_lo + r_hi)
        cos_c = 0.5 * (cos_lo + cos_hi)
        return float(np.sqrt(r_c**2 + det_z**2 - 2 * r_c * det_z * cos_c))

    outer = (r_edges[N_RADIAL - 1], r_edges[N_RADIAL])  # outermost shell
    cells = {
        # near pole: outer shell, cos in [0.9,1.0]  (closest to +Z detector)
        "near_pole": (*outer, cos_edges[1], cos_edges[0]),
        # far pole: outer shell, cos in [-1.0,-0.9] (farthest)
        "far_pole": (*outer, cos_edges[N_THETA], cos_edges[N_THETA - 1]),
        # center: innermost shell, full solid angle (the central ball)
        "center": (r_edges[0], r_edges[1], -1.0, 1.0),
        # mid equatorial: mid radius, cos ~ 0
        "mid_equator": (
            r_edges[N_RADIAL // 2],
            r_edges[N_RADIAL // 2 + 1],
            cos_edges[N_THETA // 2],
            cos_edges[N_THETA // 2 - 1],
        ),
    }
    return {k: (*v, dist(*v)) for k, v in cells.items()}


def confined_source(r_lo, r_hi, cos_lo, cos_hi) -> openmc.IndependentSource:
    return openmc.IndependentSource(
        space=openmc.stats.SphericalIndependent(
            r=openmc.stats.PowerLaw(r_lo, r_hi, 2),       # uniform-volume within shell
            cos_theta=openmc.stats.Uniform(cos_lo, cos_hi),
            phi=openmc.stats.Uniform(0, 2 * np.pi),
            origin=(0.0, 0.0, 0.0),
        ),
        energy=openmc.stats.Watt(a=0.988e6, b=2.249e-6),
        time=openmc.stats.Discrete([0.0], [1.0]),
        strength=1.0,
    )


def run_cell(label: str, R: float, cell_def: tuple, n: int) -> dict:
    r_lo, r_hi, cos_lo, cos_hi, dist = cell_def
    openmc.reset_auto_ids()  # detector cell -> id 2 deterministically
    geometry = core_with_detector(
        core_center=(0.0, 0.0, 0.0),
        core_radius=R,
        detector_center=(0.0, 0.0, R + 15.0),
        detector_radius=5.0,
    )
    settings = {
        "run_mode": "fixed source",
        "batches": 1,
        "particles": n,
        "collision_track": {"max_collisions": 50 * n, "reactions": [101], "cell_ids": [2]},
        "source": confined_source(r_lo, r_hi, cos_lo, cos_hi),
    }
    sim = Simulation(
        materials=openmc.Materials([mixture_material, detector_material]),
        geometry=geometry,
        settings=settings,
        name=f"greens-percell-k{KLABEL}-{label}",
    )
    sim.build()
    sim.run()

    with h5py.File(sim.output_dir / "collision_track.h5", "r") as f:
        a = f["collision_track_bank"][()]
    n_prompt = int(np.sum(a["delayed_group"] == 0))
    n_total = int(a.shape[0])
    g_prompt = n_prompt / n
    err = g_prompt / np.sqrt(max(n_prompt, 1))
    res = {
        "label": label, "R": R, "n_source": n, "dist_cm": dist,
        "r_mid": 0.5 * (r_lo + r_hi), "cos_mid": 0.5 * (cos_lo + cos_hi),
        "n_prompt": n_prompt, "n_total": n_total,
        "G_prompt": g_prompt, "G_prompt_err": err,
    }
    print(f"  {label:12s} dist={dist:5.1f}cm r={res['r_mid']:5.1f} cos={res['cos_mid']:+.2f}  "
          f"G_prompt={g_prompt:.4e} +/- {err:.1e}  (n_prompt={n_prompt})")
    return res


if __name__ == "__main__":
    KLABEL = sys.argv[1] if len(sys.argv) > 1 else "098"
    keff = {v["label"].replace("k", ""): k for k, v in KEFF_CONFIGS.items()}
    target_k = [k for k, v in KEFF_CONFIGS.items() if v["label"] == f"k{KLABEL}"][0]
    R = KEFF_CONFIGS[target_k]["radius"]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else N_PARTICLES.get(KLABEL, 300_000)

    print(f"Per-cell Green's function: k={target_k} (R={R} cm), N={n:,}/cell, detector +Z={R+15.0}")
    cells = critical_cells(R)
    results = [run_cell(lbl, R, cd, n) for lbl, cd in cells.items()]

    out = Path(__file__).parent.parent / "results" / f"greens-percell-k{KLABEL}" / "summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved -> {out}")
    near = next(r for r in results if r["label"] == "near_pole")
    far = next(r for r in results if r["label"] == "far_pole")
    ctr = next(r for r in results if r["label"] == "center")
    ratio_nf = near["G_prompt"] / far["G_prompt"]
    err_nf = ratio_nf * np.sqrt(1 / near["n_prompt"] + 1 / far["n_prompt"])
    ratio_cs = ctr["G_prompt"] / far["G_prompt"]
    print(f"\n  near_pole / far_pole = {ratio_nf:.3f} +/- {err_nf:.3f}   "
          f"(buggy combined method predicts ~1.00)")
    print(f"  center / far_pole    = {ratio_cs:.3f}")
