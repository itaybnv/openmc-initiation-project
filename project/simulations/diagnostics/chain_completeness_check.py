"""Validation gate: does the eigenvalue provenance recipe carry the full fission chain?

The provenance recipe (OpenMC-validated, skill gotcha G-110) runs eigenvalue mode with
``batches=1, inactive=0`` and ``write_initial_source=True``: the realized initial source
bank is written in particle-id order, so ``source_bank[parent_id - 1]`` is a valid
collision->birth join. That solves the FileSource attribution bug in a single run.

Open physics question this script answers empirically: in eigenvalue mode fission
progeny are banked for the *next* generation rather than followed in-history, whereas
fixed-source mode follows the whole prompt+delayed chain in one history. If the
1-generation provenance run only records first-generation events, it yields a *direct
(unmultiplied)* kernel, not the full-chain g_i(t) that greens_sweep measures.

Runs two simulations with identical geometry and source (R=11.4 cm, k~0.40):
  A) fixed source, batches=1                       -- reference, full chain by construction
  B) eigenvalue, batches=1, inactive=0,
     write_initial_source=True                     -- the provenance recipe

Compares detections/source, multiplicity, time quantiles, and runs the early-time
birth-cos attribution test on B (fastest detections must come from cells facing the +Z
detector). Prints a GATE verdict:

  GATE: FULL-CHAIN            -> provenance run can replace the 80-run greens_sweep
  GATE: FIRST-GENERATION-ONLY -> provenance run gives the direct kernel only;
                                 greens_sweep stays the full-chain ground truth

Run with:  python -m project.simulations.diagnostics.chain_completeness_check [N]
"""
import sys
from collections import Counter
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

R = KEFF_CONFIGS[0.40]["radius"]  # 11.4 cm -- fast, low multiplication, strong direct signal
DET_Z = R + 15.0                  # detector center on +Z (gap 10 + radius 5)
N_DEFAULT = 200_000


def make_geometry() -> openmc.Geometry:
    return core_with_detector(
        core_center=(0.0, 0.0, 0.0),
        core_radius=R,
        detector_center=(0.0, 0.0, DET_Z),
        detector_radius=5.0,
    )


def collision_track(n: int) -> dict:
    return {"max_collisions": 50 * n, "reactions": [101], "cell_ids": [DETECTOR_CELL_ID]}


def run_sim(name: str, settings: dict) -> Path:
    sim = Simulation(
        materials=openmc.Materials([mixture_material, detector_material]),
        geometry=make_geometry(),
        settings=settings,
        name=name,
    )
    sim.build()
    sim.run()
    return sim.output_dir


def load_track(outdir: Path) -> np.ndarray:
    with h5py.File(outdir / "collision_track.h5", "r") as f:
        return f["collision_track_bank"][()]


def metrics(track: np.ndarray, n_source: int, label: str) -> dict:
    prompt = track["delayed_group"] == 0
    t = np.asarray(track["time"], dtype=float)
    pid = track["parent_id"].astype(np.int64)
    mult = Counter(Counter(pid).values())  # {multiplicity: #parents}
    n_parents = sum(mult.values())
    n_multi = sum(c for m, c in mult.items() if m >= 2)
    q = (
        np.quantile(t, [0.5, 0.9, 0.99]).tolist() + [t.max()]
        if len(t)
        else [np.nan] * 4
    )
    m = {
        "label": label,
        "n_det_total": int(track.shape[0]),
        "n_det_prompt": int(prompt.sum()),
        "det_per_source": track.shape[0] / n_source,
        "n_parents": n_parents,
        "mean_multiplicity": track.shape[0] / max(n_parents, 1),
        "p_multi": n_multi / max(n_parents, 1),
        "t_quantiles": q,
    }
    print(
        f"  {label}: detections={m['n_det_total']:,} ({m['n_det_prompt']:,} prompt)  "
        f"det/src={m['det_per_source']:.4f}  parents={n_parents:,}  "
        f"mean mult={m['mean_multiplicity']:.3f}  P(m>=2)={m['p_multi']:.4f}"
    )
    print(
        "    time quantiles 50/90/99/max [s]: "
        + "  ".join(f"{x:.3e}" for x in m["t_quantiles"])
    )
    return m


def early_time_birth_cos(track: np.ndarray, birth_xyz: np.ndarray) -> float:
    """Mean birth cos(theta) of the fastest prompt detections (join validity test).

    Correct attribution => fastest detections were born facing the +Z detector,
    so mean cos(theta) must be strongly positive.
    """
    prompt = track["delayed_group"] == 0
    pid = track["parent_id"][prompt].astype(np.int64) - 1
    t = track["time"][prompt]
    ok = (pid >= 0) & (pid < len(birth_xyz))
    n_bad = int((~ok).sum())
    if n_bad:
        print(f"    !! {n_bad} detections with out-of-range parent_id")
    pid, t = pid[ok], t[ok]
    b = birth_xyz[pid]
    br = np.sqrt((b**2).sum(axis=1))
    bcos = b[:, 2] / np.maximum(br, 1e-30)
    order = np.argsort(t)
    for K in (200, 500, 2000):
        sel = order[: min(K, len(order))]
        print(
            f"    fastest {len(sel):5d}: mean birth cos(theta)={bcos[sel].mean():+.3f}  "
            f"mean birth r={br[sel].mean():4.1f}/{R:.0f}cm  facing(cos>0)={np.mean(bcos[sel] > 0):.3f}"
        )
    print(f"    ALL {len(t):6d}    : mean birth cos(theta)={bcos.mean():+.3f}")
    sel = order[: min(200, len(order))]
    return float(bcos[sel].mean()) if len(sel) else float("nan")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else N_DEFAULT
    print(f"Chain-completeness gate: R={R} cm (k~0.40), N={n:,}, detector at +Z={DET_Z} cm\n")

    print("Run A: fixed source (reference, full chain by construction)")
    out_a = run_sim(
        "diag-chain-A-fixedsource",
        {
            "run_mode": "fixed source",
            "batches": 1,
            "particles": n,
            "collision_track": collision_track(n),
            "source": uniform_core_source(R),
        },
    )
    track_a = load_track(out_a)
    m_a = metrics(track_a, n, "A fixed-source")

    print("\nRun B: eigenvalue provenance recipe (batches=1, inactive=0, write_initial_source)")
    out_b = run_sim(
        "diag-chain-B-eigenvalue",
        {
            "run_mode": "eigenvalue",
            "batches": 1,
            "inactive": 0,
            "particles": n,
            "write_initial_source": True,  # eigenvalue-only; ignore the meaningless k-eff
            "collision_track": collision_track(n),
            "source": uniform_core_source(R),
        },
    )
    track_b = load_track(out_b)
    m_b = metrics(track_b, n, "B eigenvalue ")

    birth_xyz, bank = read_initial_source(out_b / "initial_source.h5")
    print(f"\n  initial_source.h5: {len(bank):,} particles (expected {n:,})")
    assert len(bank) == n, "initial source bank size != particles"

    print("\n  Early-time birth-cos attribution test (run B):")
    cos200 = early_time_birth_cos(track_b, birth_xyz)

    ratio = m_b["det_per_source"] / max(m_a["det_per_source"], 1e-30)
    print("\n================ GATE VERDICT ================")
    print(f"  B/A detections per source: {ratio:.3f}")
    print(f"  B P(m>=2): {m_b['p_multi']:.4f}   (A: {m_a['p_multi']:.4f})")
    print(f"  B fastest-200 mean birth cos(theta): {cos200:+.3f}")

    if not (cos200 > 0.5):
        print("  !! ATTRIBUTION TEST FAILED -- parent_id join is NOT valid; investigate before")
        print("     building anything on the provenance recipe.")
        print("GATE: JOIN-INVALID")
    elif ratio > 0.9 and m_b["p_multi"] > 0.5 * m_a["p_multi"]:
        print("  Chain events present: detections and multiplicity match fixed source.")
        print("GATE: FULL-CHAIN")
    else:
        print("  Detections/multiplicity collapsed vs fixed source: fission progeny are")
        print("  banked for the next generation, not transported. The provenance run")
        print("  yields the DIRECT (unmultiplied) kernel only.")
        print("GATE: FIRST-GENERATION-ONLY")
