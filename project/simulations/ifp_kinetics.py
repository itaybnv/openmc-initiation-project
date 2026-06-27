"""IFP kinetics parameters (generation time, beta_eff) per k label — PK baseline.

Runs an eigenvalue calculation per KEFF_CONFIGS label with OpenMC's Iterated
Fission Probability tallies (Model.add_kinetics_parameters_tallies), giving
adjoint-weighted Lambda and beta_eff with uncertainties — an INDEPENDENT input
for the point-kinetics alpha:

    alpha_PK = (k(1 - beta_eff) - 1) / l,   l = k * Lambda

Convention: OpenMC's generation_time = ifp_time_numerator/(ifp_denominator*keff)
— the division by keff makes it the GENERATION TIME Lambda (l = k*Lambda). The
script prints alpha_PK under both conventions next to the measured late-window
ACDC alpha so the choice is verified empirically per k, not assumed.

Run with: python -m project.simulations.ifp_kinetics [label|all]
Output:   project/results/ifp-kinetics-k<label>/statepoint.<batches>.h5 + ifp.npz
"""
import sys

import numpy as np
import openmc

from project.core.geometry import core_with_detector
from project.core.materials import detector_material, mixture_material
from project.core.physics_config import eigenvalue_settings
from project.core.rossi_mesh import KEFF_CONFIGS
from project.core.simulation import Simulation

PARTICLES = 10_000
BATCHES = 130
INACTIVE = 30


def run_label(label: str) -> dict:
    target_k = [k for k, v in KEFF_CONFIGS.items() if v["label"] == f"k{label}"][0]
    radius = KEFF_CONFIGS[target_k]["radius"]
    geometry = core_with_detector(
        core_center=(0.0, 0.0, 0.0), core_radius=radius,
        detector_center=(0.0, 0.0, radius + 15.0), detector_radius=5.0,
    )
    sim = Simulation(
        materials=openmc.Materials([mixture_material, detector_material]),
        geometry=geometry,
        settings=eigenvalue_settings(particles=PARTICLES, batches=BATCHES, inactive=INACTIVE),
        name=f"ifp-kinetics-k{label}",
    )
    model = sim.build()
    model.add_kinetics_parameters_tallies()  # MUST precede run (G-101)
    sim.run()

    with openmc.StatePoint(str(sim.output_dir / f"statepoint.{BATCHES}.h5")) as sp:
        params = sp.get_kinetics_parameters()
        if params.generation_time is None:
            raise ValueError("IFP tallies missing from statepoint (G-101)")
        keff = sp.keff
    gen, beta = params.generation_time, params.beta_effective

    k = keff.nominal_value
    k_p = k * (1 - beta.nominal_value)
    alpha_generation = (k_p - 1) / (k * gen.nominal_value)   # l = k * Lambda
    alpha_lifetime = (k_p - 1) / gen.nominal_value           # gen_time treated as l

    np.savez(
        sim.output_dir / "ifp.npz",
        label=label, target_k=target_k,
        keff=keff.nominal_value, keff_std=keff.std_dev,
        gen_time=gen.nominal_value, gen_time_std=gen.std_dev,
        beta_eff=beta.nominal_value, beta_eff_std=beta.std_dev,
        convention="generation",  # l = k * Lambda; verified against ACDC alpha
    )
    print(f"k{label}: keff={keff.nominal_value:.5f}+/-{keff.std_dev:.5f} "
          f"(target {target_k}, diff {keff.nominal_value - target_k:+.5f})")
    print(f"  Lambda={gen.nominal_value*1e6:8.2f}+/-{gen.std_dev*1e6:.2f} us "
          f"({100*gen.std_dev/gen.nominal_value:.1f}%)   "
          f"beta_eff={beta.nominal_value:.5f}+/-{beta.std_dev:.5f} "
          f"({100*beta.std_dev/beta.nominal_value:.1f}%)")
    print(f"  alpha_PK[l=k*Lambda] = {alpha_generation:9.1f} /s   "
          f"alpha_PK[l=Lambda] = {alpha_lifetime:9.1f} /s")
    return {"label": label, "alpha_generation": alpha_generation,
            "alpha_lifetime": alpha_lifetime}


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    labels = ["098", "090", "070", "040"] if arg == "all" else [arg]
    # measured late-window ACDC alpha for the convention check (Phase-0 gate)
    acdc_alpha = {"098": -378.4, "090": -1467.5, "070": -5053.9, "040": -10326.3}
    results = [run_label(lbl) for lbl in labels]
    print("\nConvention check (compare with measured ACDC alpha):")
    for r in results:
        a = acdc_alpha.get(r["label"])
        print(f"  k{r['label']}: ACDC={a:9.1f}  PK[l=k*Lambda]={r['alpha_generation']:9.1f}  "
              f"PK[l=Lambda]={r['alpha_lifetime']:9.1f}")
