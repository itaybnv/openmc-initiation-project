"""Head-to-head: detector time spectrum g(t) via collision_track vs a TimeFilter tally.

Runs the SAME physics -- a uniform-core Watt pulse into the core_with_detector
geometry -- two ways, at matched total histories, and saves both g(t) estimates
plus the metadata a notebook needs to judge the trade:

  A) collision_track  (batches=1): per-event detection record. We histogram the
     prompt detection times. This is the method the project already trusts; it is
     also the ONLY one that carries parent_id (pairs/provenance). Its singles
     uncertainty is Poisson sqrt(counts) per bin; it CANNOT give a native std_dev
     because parent_id forces batches=1. The collision bank grows with detections.

  B) TimeFilter tally (batches>1): CellFilter(detector) x TimeFilter(edges) on an
     absorption score (B-10 has no fission, so absorption == the (n,a)+(n,g) capture
     the collision run records via reactions=[101]). Multi-batch -> native per-bin
     std_dev; fixed memory regardless of history count. No per-event identity, so
     no pairs -- singles only.

The point is NOT that one replaces the other: the ACDC/Rossi work stays on
collision_track. This quantifies, on identical physics, (1) that the tally g(t)
matches the collision g(t), (2) the tally's native error bars, and (3) the
collision bank's memory cost -- the singles-side case for adding tallies.

Run with:  python -m project.simulations.tally_vs_collision [label] [--gap G]
                                                            [--histories N] [--batches B]
Output:    project/results/tally-vs-collision-k<label>.npz
"""
import argparse
from pathlib import Path

import h5py
import numpy as np
import openmc

from project.core.geometry import DETECTOR_CELL_ID, core_with_detector
from project.core.materials import detector_material, mixture_material
from project.core.physics_config import detector_max_collisions
from project.core.rossi_mesh import KEFF_CONFIGS
from project.core.simulation import Simulation
from project.core.sources import uniform_core_source

RESULTS_ROOT = Path(__file__).parent.parent / "results"

# Same prompt window/binning convention as greens_sweep.
T_MAX = 1.0e-2
N_T = 256
DETECTION_SCORE = "absorption"   # B-10 capture; tally analogue of reactions=[101]


def _geometry(R, gap):
    return core_with_detector(
        core_center=(0.0, 0.0, 0.0), core_radius=R,
        detector_center=(0.0, 0.0, R + gap + 5.0), detector_radius=5.0,
    )


def run_collision(label, R, gap, histories, time_edges):
    """Method A: analog collision_track, batches=1. Returns prompt g(t) counts."""
    settings = {
        "run_mode": "fixed source", "batches": 1, "particles": histories,
        "collision_track": {
            "max_collisions": detector_max_collisions(histories),
            "reactions": [101], "cell_ids": [DETECTOR_CELL_ID],
        },
        "source": uniform_core_source(R),
    }
    sim = Simulation(
        materials=openmc.Materials([mixture_material, detector_material]),
        geometry=_geometry(R, gap), settings=settings,
        name=f"tally-vs-collision-k{label}-A",
    )
    sim.build()
    sim.run()
    track_path = sim.output_dir / "collision_track.h5"
    with h5py.File(track_path, "r") as f:
        a = f["collision_track_bank"][()]
    prompt = a["delayed_group"] == 0
    counts, _ = np.histogram(a["time"][prompt], bins=time_edges)
    return {
        "counts": counts.astype(float),
        "n_detections": int(prompt.sum()),
        "n_total_events": int(a.shape[0]),
        "bank_bytes": int(track_path.stat().st_size),
        "histories": histories,
    }


def run_tally(label, R, gap, histories, batches, time_edges):
    """Method B: multi-batch TimeFilter tally. Returns g(t) mean + native std_dev."""
    per_batch = max(1, histories // batches)
    settings = {
        "run_mode": "fixed source", "batches": batches, "particles": per_batch,
        "source": uniform_core_source(R),
    }
    sim = Simulation(
        materials=openmc.Materials([mixture_material, detector_material]),
        geometry=_geometry(R, gap), settings=settings,
        name=f"tally-vs-collision-k{label}-B",
    )
    sim.build()
    gt = openmc.Tally(name="g_of_t")
    gt.filters = [openmc.CellFilter([DETECTOR_CELL_ID]), openmc.TimeFilter(time_edges)]
    gt.scores = [DETECTION_SCORE]
    sim.model.tallies = openmc.Tallies([gt])
    sim.run()
    with openmc.StatePoint(str(sim.output_dir / f"statepoint.{batches}.h5")) as sp:
        t = sp.get_tally(name="g_of_t")
        mean = t.mean.ravel().astype(float)        # detections per source per bin
        std = t.std_dev.ravel().astype(float)      # native across-batch uncertainty
        sp_bytes = int((sim.output_dir / f"statepoint.{batches}.h5").stat().st_size)
    return {
        "mean": mean, "std": std, "histories": per_batch * batches,
        "batches": batches, "statepoint_bytes": sp_bytes,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("label", nargs="?", default="090",
                   choices=sorted(v["label"].removeprefix("k") for v in KEFF_CONFIGS.values()))
    p.add_argument("--gap", type=float, default=10.0,
                   help="core-surface-to-detector gap [cm] (detector center at R+gap+5)")
    p.add_argument("--histories", type=int, default=200_000,
                   help="total histories, matched across both methods")
    p.add_argument("--batches", type=int, default=40,
                   help="batches for the multi-batch tally (collision stays batches=1)")
    args = p.parse_args(argv)

    target_k = [k for k, v in KEFF_CONFIGS.items() if v["label"] == f"k{args.label}"][0]
    R = KEFF_CONFIGS[target_k]["radius"]
    time_edges = np.linspace(0.0, T_MAX, N_T + 1)

    print(f"k={target_k} (R={R} cm), gap={args.gap}, histories={args.histories:,}")
    print("Method A: collision_track (batches=1) ...")
    A = run_collision(args.label, R, args.gap, args.histories, time_edges)
    print(f"  detections={A['n_detections']:,}  collision_track.h5={A['bank_bytes']/1e6:.1f} MB")
    print(f"Method B: TimeFilter tally ({args.batches} batches) ...")
    B = run_tally(args.label, R, args.gap, args.histories, args.batches, time_edges)
    in_win = B["mean"].sum()
    print(f"  in-window detections/source (tally)={in_win:.3e}  "
          f"(collision={A['n_detections']/A['histories']:.3e})")

    out = RESULTS_ROOT / f"tally-vs-collision-k{args.label}.npz"
    np.savez_compressed(
        out, k=target_k, R=R, gap=args.gap, time_edges=time_edges,
        a_counts=A["counts"], a_n_detections=A["n_detections"],
        a_n_total_events=A["n_total_events"], a_bank_bytes=A["bank_bytes"],
        a_histories=A["histories"],
        b_mean=B["mean"], b_std=B["std"], b_histories=B["histories"],
        b_batches=B["batches"], b_statepoint_bytes=B["statepoint_bytes"],
    )
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
