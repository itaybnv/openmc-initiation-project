"""Phase 1 diagnostic: confirm the birth-cell attribution bug and test the fix.

The spatial Green's function attributes each detector capture to the mesh cell where
the *original* source neutron was born, via ``mesh_ids[parent_id - 1]``. This assumes
the simulated particle ``id`` equals the row index of the pre-generated source file.

`openmc.FileSource` samples a *random* site per history, so that assumption is false and
the attribution is scrambled. This script demonstrates it with a direct, physical test:

    A neutron detected at the +Z detector within a very short time must have been born in
    a cell *facing* the detector (cos(theta) -> +1), near the surface. So the *fastest*
    detections must, under correct attribution, map back to high-cos(theta) birth cells.

Two paths are run on identical physics (small R=11.4 cm core for speed):
  A) current method: pregenerate_source + FileSource, map the pre-generated file by id.
  B) proposed fix:   IndependentSource directly + settings.write_initial_source=True,
                     map the *realized* initial_source.h5 (written in id order) by id.

Expected: path A -> mean birth cos(theta) ~ 0 for fast detections (scrambled);
          path B -> mean birth cos(theta) strongly positive (correct).

Run with:  python -m project.simulations.diagnostics.attribution_check
"""
import numpy as np
import openmc
import h5py
from pathlib import Path

from project.core.geometry import core_with_detector
from project.core.materials import detector_material, mixture_material
from project.core.simulation import Simulation
from project.core.source_utils import pregenerate_source

R = 11.4                      # small core (k~0.40) -> fast, low multiplication, strong direct signal
N = 500_000                   # source particles
DET_Z = R + 10.0 + 5.0        # detector center on +Z, aligned with mesh polar axis


def make_source() -> openmc.IndependentSource:
    """Uniform-volume Watt source at t=0, identical to the real Green's-function runs."""
    return openmc.IndependentSource(
        space=openmc.stats.SphericalIndependent(
            r=openmc.stats.PowerLaw(0, R, 2),
            cos_theta=openmc.stats.Uniform(-1, 1),
            phi=openmc.stats.Uniform(0, 2 * np.pi),
            origin=(0.0, 0.0, 0.0),
        ),
        energy=openmc.stats.Watt(a=0.988e6, b=2.249e-6),
        time=openmc.stats.Discrete([0.0], [1.0]),
        strength=1.0,
    )


def make_geometry() -> openmc.Geometry:
    # reset so the detector cell is deterministically id=2 (matches collision_track cell_ids)
    openmc.reset_auto_ids()
    return core_with_detector(
        core_center=(0.0, 0.0, 0.0),
        core_radius=R,
        detector_center=(0.0, 0.0, DET_Z),
        detector_radius=5.0,
    )


def base_settings() -> dict:
    return {
        "run_mode": "fixed source",
        "batches": 1,
        "particles": N,
        "collision_track": {
            "max_collisions": 50 * N,
            "reactions": [101],
            "cell_ids": [2],
        },
    }


def early_time_test(results_dir: Path, birth_xyz: np.ndarray, label: str) -> None:
    """Report birth cos(theta)/r of the fastest detections under a given id->birth map."""
    with h5py.File(Path(results_dir) / "collision_track.h5", "r") as f:
        a = f["collision_track_bank"][()]
    prompt = a["delayed_group"] == 0
    pid = a["parent_id"][prompt].astype(np.int64) - 1
    t = a["time"][prompt]
    ok = (pid >= 0) & (pid < len(birth_xyz))
    pid, t = pid[ok], t[ok]
    bx, by, bz = birth_xyz[pid, 0], birth_xyz[pid, 1], birth_xyz[pid, 2]
    br = np.sqrt(bx**2 + by**2 + bz**2)
    bcos = bz / np.maximum(br, 1e-30)
    order = np.argsort(t)
    print(f"\n  === {label} ===  ({len(t)} prompt detections)")
    for K in (200, 500, 2000):
        sel = order[: min(K, len(order))]
        print(
            f"    fastest {len(sel):5d}: mean birth cos(theta)={bcos[sel].mean():+.3f}  "
            f"mean birth r={br[sel].mean():4.1f}/{R:.0f}cm  facing(cos>0)={np.mean(bcos[sel] > 0):.3f}"
        )
    print(f"    ALL {len(t):5d}     : mean birth cos(theta)={bcos.mean():+.3f}")


def run_path_a() -> None:
    """Current method: pregenerate + FileSource. Expected SCRAMBLED."""
    settings = base_settings() | {"source": make_source()}
    sim = Simulation(
        materials=openmc.Materials([mixture_material, detector_material]),
        geometry=make_geometry(),
        settings=settings,
        name="diag-attribution-A-pregen",
    )
    sim.build()
    source_path = pregenerate_source(sim.model, N, sim.output_dir)  # swaps to FileSource
    sim.run()
    birth = openmc.read_source_file(str(source_path)).to_dataframe()[["x", "y", "z"]].to_numpy()
    early_time_test(sim.output_dir, birth, "PATH A: pregenerate + FileSource (current)")


def run_path_b() -> None:
    """Proposed fix: IndependentSource direct + write_initial_source. Expected CORRECT."""
    settings = base_settings() | {"source": make_source()}
    sim = Simulation(
        materials=openmc.Materials([mixture_material, detector_material]),
        geometry=make_geometry(),
        settings=settings,
        name="diag-attribution-B-writeinit",
    )
    sim.build()
    sim.settings_obj.write_initial_source = True
    sim.run()
    realized = sim.output_dir / "initial_source.h5"
    if not realized.exists():
        print(f"\n  !! write_initial_source produced no file at {realized} -- fix B unavailable")
        return
    birth = openmc.read_source_file(str(realized)).to_dataframe()[["x", "y", "z"]].to_numpy()
    early_time_test(sim.output_dir, birth, "PATH B: IndependentSource + write_initial_source (fix)")


if __name__ == "__main__":
    print(f"Attribution diagnostic: R={R} cm, N={N:,}, detector at +Z={DET_Z} cm")
    print("Correct attribution => fastest detections come from facing cells (cos theta -> +1).")
    run_path_a()
    run_path_b()
