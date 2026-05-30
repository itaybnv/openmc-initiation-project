"""Eigenvalue run that captures the converged fundamental-mode source bank.

Writes a source file alongside the statepoint so a subsequent fixed-source
Green's function simulation can use FileSource to emit from the converged
fission distribution rather than a uniform-volume source.

Run with: python -m project.simulations.eigenvalue_for_source
"""
import openmc

from project.core.geometry import core_sphere
from project.core.materials import mixture_material
from project.core.rossi_mesh import KEFF_CONFIGS
from project.core.simulation import Simulation

RADIUS = KEFF_CONFIGS[0.98]["radius"]  # 28.7 cm
N_BATCHES = 60

materials = openmc.Materials([mixture_material])
geometry = core_sphere(center=(0.0, 0.0, 0.0), radius=RADIUS)
settings = {
    "run_mode": "eigenvalue",
    "particles": 50_000,
    "batches": N_BATCHES,
    "inactive": 20,
    "sourcepoint": {"write": True, "batches": [N_BATCHES]},
}

sim = Simulation(
    materials=materials,
    geometry=geometry,
    settings=settings,
    name="eigenvalue_source_k098",
)
sim.build()
sim.run()

sp_path = sim.output_dir / f"statepoint.{N_BATCHES}.h5"
with openmc.StatePoint(str(sp_path)) as sp:
    k = sp.keff
    src_array = sp.source  # numpy structured array

# OpenMC's sourcepoint XML doesn't emit a standalone source HDF5 here, so build
# one from the statepoint source bank for downstream use as openmc.FileSource.
particles = openmc.ParticleList([
    openmc.SourceParticle(
        r=tuple(p["r"]),
        u=tuple(p["u"]),
        E=float(p["E"]),
        time=float(p["time"]),
        wgt=float(p["wgt"]),
        delayed_group=int(p["delayed_group"]),
        surf_id=int(p["surf_id"]),
        particle=openmc.ParticleType(int(p["particle"])),
    )
    for p in src_array
])
source_path = sim.output_dir / "converged_source.h5"
particles.export_to_hdf5(source_path)

print(f"\nR = {RADIUS:.3f} cm   k_eff = {k.nominal_value:.5f} ± {k.std_dev:.5f}")
print(f"Converged source: {source_path} ({len(particles):,} particles)")
