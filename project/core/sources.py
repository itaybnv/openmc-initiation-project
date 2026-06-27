"""Shared source builders for Green's-function simulations.

Every builder returns a FULLY-specified IndependentSource (space, angle, energy
and time all set explicitly). Full specification is required by the eigenvalue
source-provenance recipe (write_initial_source=True), and harmless elsewhere.
"""

import numpy as np
import openmc

__all__ = [
    "WATT_U235",
    "watt_energy",
    "pulse_t0",
    "uniform_core_source",
    "confined_cell_source",
]

# U235 thermal-fission Watt spectrum parameters (ENDF/B-VII.0; also OpenMC's
# defaults for openmc.stats.Watt, kept explicit for traceability).
WATT_U235 = {"a": 0.988e6, "b": 2.249e-6}


def watt_energy() -> openmc.stats.Watt:
    return openmc.stats.Watt(**WATT_U235)


def pulse_t0() -> openmc.stats.Discrete:
    """Single emission pulse at t = 0."""
    return openmc.stats.Discrete([0.0], [1.0])


def uniform_core_source(
    radius: float, origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
) -> openmc.IndependentSource:
    """Uniform-volume Watt-spectrum pulse over the whole core sphere."""
    return openmc.IndependentSource(
        # spherical_uniform uses PowerLaw(0, R, n=2) internally: p(r) ~ r^2,
        # i.e. uniform per unit volume.
        space=openmc.stats.spherical_uniform(r_outer=radius, origin=origin),
        angle=openmc.stats.Isotropic(),
        energy=watt_energy(),
        time=pulse_t0(),
        strength=1.0,
    )


def confined_cell_source(
    r_lo: float,
    r_hi: float,
    cos_lo: float,
    cos_hi: float,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> openmc.IndependentSource:
    """Uniform-volume Watt-spectrum pulse confined to one (r, cos-theta) mesh cell.

    Kept as an explicit SphericalIndependent (rather than
    openmc.stats.spherical_uniform(thetas=...)) because the mesh cell bounds are
    native in cos(theta) and uniform-in-cos is what equal-solid-angle bands need.
    """
    return openmc.IndependentSource(
        space=openmc.stats.SphericalIndependent(
            r=openmc.stats.PowerLaw(r_lo, r_hi, 2),  # uniform-volume within shell
            cos_theta=openmc.stats.Uniform(cos_lo, cos_hi),
            phi=openmc.stats.Uniform(0, 2 * np.pi),
            origin=origin,
        ),
        angle=openmc.stats.Isotropic(),
        energy=watt_energy(),
        time=pulse_t0(),
        strength=1.0,
    )
