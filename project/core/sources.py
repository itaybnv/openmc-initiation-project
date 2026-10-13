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
    "fission_weighted_cell_source",
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


def fission_weighted_cell_source(
    r_lo: float,
    r_hi: float,
    cos_lo: float,
    cos_hi: float,
    r_fine_edges,
    rho_fine,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    n_sub: int = 20,
) -> openmc.IndependentSource:
    """Watt pulse confined to one (r, cos-theta) cell, radially FISSION-WEIGHTED.

    Same cell, same angle/energy/time as `confined_cell_source`, but the birth
    radius is drawn from p(r) ~ r^2 rho(r) instead of r^2, where rho(r) is the
    fission density per unit volume, piecewise-constant on the fine radial
    shells `r_fine_edges` (values `rho_fine`, one per fine shell). The coarse
    cell [r_lo, r_hi] must coincide with a block of fine shells (the sweep grid
    nests on the tally grid). Angular weighting is deliberately NOT applied: the
    fission map is angle-blind (the detector does not perturb the core).

    Purpose (GATE_DIAGNOSTICS / expert-deck B8): the within-cell pair source is
    F(x) G(x)^2, deeper than the uniform-volume average; sampling births from F
    moves the per-cell kernel g_i(t) toward that average at zero transport cost.
    """
    r_fine_edges = np.asarray(r_fine_edges, float)
    rho_fine = np.asarray(rho_fine, float)
    j = np.where((r_fine_edges >= r_lo - 1e-9) & (r_fine_edges <= r_hi + 1e-9))[0]
    if len(j) < 2:
        raise ValueError(f"coarse shell [{r_lo}, {r_hi}] does not nest on the fine edges")
    eps = 1e-7 * (r_hi - r_lo)
    xs, ps = [], []
    for a, b in zip(j[:-1], j[1:]):
        lo, hi = r_fine_edges[a], r_fine_edges[b]
        x = np.linspace(lo, hi, n_sub + 1)
        if xs:                       # open a new piece just after the shared edge
            x[0] = lo + eps
        xs.append(x)
        ps.append(rho_fine[a] * x**2)
    x = np.concatenate(xs)
    pdf = np.concatenate(ps)
    if not np.all(np.diff(x) > 0):
        raise ValueError("radial tabulation is not strictly increasing")
    if pdf.sum() <= 0:
        raise ValueError("fission density vanishes on this cell")
    return openmc.IndependentSource(
        space=openmc.stats.SphericalIndependent(
            r=openmc.stats.Tabular(x, pdf, interpolation="linear-linear"),
            cos_theta=openmc.stats.Uniform(cos_lo, cos_hi),
            phi=openmc.stats.Uniform(0, 2 * np.pi),
            origin=origin,
        ),
        angle=openmc.stats.Isotropic(),
        energy=watt_energy(),
        time=pulse_t0(),
        strength=1.0,
    )
