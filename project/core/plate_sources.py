"""Source builders for the plate-fuel line.

Reuses the sphere line's energy (U-235 Watt) and time (t=0 pulse) builders; only
the spatial distribution is plate-specific (a Cartesian box over the fuel instead
of a spherical-uniform distribution). Fully-specified sources, matching the
provenance-recipe convention in ``project.core.sources``.
"""

import openmc

from .sources import watt_energy, pulse_t0

__all__ = ["uniform_fuel_source", "confined_box_source"]


def uniform_fuel_source(
    x_lo: float,
    x_hi: float,
    width: float,
    height: float,
) -> openmc.IndependentSource:
    """Uniform-volume Watt-spectrum pulse over the fuel envelope.

    The ``fissionable`` constraint rejects box samples that miss the meat, so the
    emitted distribution is uniform over the fuel (not the water channels).
    """
    return openmc.IndependentSource(
        space=openmc.stats.Box((x_lo, -width / 2, -height / 2),
                               (x_hi, width / 2, height / 2)),
        constraints={"fissionable": True},
        angle=openmc.stats.Isotropic(),
        energy=watt_energy(),
        time=pulse_t0(),
        strength=1.0,
    )


def confined_box_source(
    lower_left: tuple[float, float, float],
    upper_right: tuple[float, float, float],
) -> openmc.IndependentSource:
    """Uniform-volume Watt-spectrum pulse confined to one Cartesian coarse cell.

    Cartesian analogue of ``sources.confined_cell_source``: uniform-in-xyz is
    already uniform-per-volume, so no PowerLaw radial weighting is needed. The
    fissionable constraint keeps emission in the meat.
    """
    return openmc.IndependentSource(
        space=openmc.stats.Box(lower_left, upper_right),
        constraints={"fissionable": True},
        angle=openmc.stats.Isotropic(),
        energy=watt_energy(),
        time=pulse_t0(),
        strength=1.0,
    )
