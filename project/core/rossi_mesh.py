"""Shared spherical mesh configuration for the Rossi-alpha meshing simulation."""
import numpy as np
import openmc

N_RADIAL: int = 100
CORE_RADIUS: float = 3.0  # cm


def make_fission_mesh(
    n_radial: int = N_RADIAL,
    core_radius: float = CORE_RADIUS,
) -> openmc.SphericalMesh:
    """Return a SphericalMesh with equal-volume radial shells covering the core.

    Each shell spans from r_i to r_{i+1} where r_i = core_radius * (i/n_radial)^(1/3),
    ensuring every shell has identical volume (4/3)π core_radius³ / n_radial.
    """
    r_edges = core_radius * np.linspace(0, 1, n_radial + 1) ** (1 / 3)
    return openmc.SphericalMesh(
        r_grid=r_edges,
        theta_grid=[0.0, np.pi],    # 1 cell: full polar range
        phi_grid=[0.0, 2 * np.pi],  # 1 cell: full azimuthal range
        origin=(0.0, 0.0, 0.0),
    )
