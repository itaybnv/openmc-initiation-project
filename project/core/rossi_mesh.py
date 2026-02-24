"""Shared spherical mesh configuration for the Rossi-alpha meshing simulation."""
import numpy as np
import openmc

N_RADIAL: int = 100
N_THETA: int = 50
N_PHI: int = 1
CORE_RADIUS: float = 3.0  # cm


def make_fission_mesh(
    n_radial: int = N_RADIAL,
    n_theta: int = N_THETA,
    n_phi: int = N_PHI,
    core_radius: float = CORE_RADIUS,
) -> openmc.SphericalMesh:
    """Return a SphericalMesh with equal-volume cells covering the core.

    Radial edges use r_i = core_radius * (i/n_radial)^(1/3) so each shell
    has identical volume.

    Theta edges are equally spaced in cos(theta) so each polar band
    subtends the same solid angle fraction, producing equal-volume cells
    when combined with the equal-volume radial shells.

    Phi edges are equally spaced (default 1 bin = full 2pi).

    Total cells: n_radial * n_theta * n_phi.
    """
    r_edges = core_radius * np.linspace(0, 1, n_radial + 1) ** (1 / 3)

    # Equal spacing in cos(theta) from cos=1 (theta=0) to cos=-1 (theta=pi)
    cos_theta_edges = np.linspace(1, -1, n_theta + 1)
    theta_edges = np.arccos(cos_theta_edges)

    phi_edges = np.linspace(0, 2 * np.pi, n_phi + 1)

    return openmc.SphericalMesh(
        r_grid=r_edges,
        theta_grid=theta_edges,
        phi_grid=phi_edges,
        origin=(0.0, 0.0, 0.0),
    )
