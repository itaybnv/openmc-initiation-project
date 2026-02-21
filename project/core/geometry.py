"""Geometry definitions for the simulation core and detector."""

import openmc
from .materials import mixture_material, detector_material

__all__ = ["core_sphere", "core_with_detector"]


DEFAULTS = {
    "core_center": (0.0, 0.0, 0.0),
    "core_radius": 3.0,
    "detector_center": (0.0, 3.0 + 5.0 + 10.0, 0.0),
    "detector_radius": 5.0,
}


def core_sphere(
    center: tuple[float, float, float] = DEFAULTS["core_center"],
    radius: float = DEFAULTS["core_radius"],
) -> openmc.Geometry:
    """Create a spherical core geometry filled with mixture_material.
    Parameters
    ----------
    center : tuple of float
        The (x, y, z) coordinates of the sphere center.
    radius : float
        The radius of the sphere.
    Returns
    -------
    openmc.Geometry
        The geometry object representing the spherical core.
    """
    core_surface = openmc.Sphere(r=radius, x0=center[0], y0=center[1], z0=center[2])
    outer_surface = openmc.Sphere(r=radius + 1.0, boundary_type="vacuum")

    core_cell = openmc.Cell(region=-core_surface, fill=mixture_material)
    void_cell = openmc.Cell(region=+core_surface & -outer_surface)

    return openmc.Geometry([core_cell, void_cell])


def _max_distance(center: tuple[float, float, float], radius: float) -> float:
    """Calculate the maximum distance from the origin to the surface of a sphere.
    Parameters
    ----------
    center : tuple of float
        The (x, y, z) coordinates of the sphere center.
    radius : float
        The radius of the sphere.
    Returns
    -------
    float
        The maximum distance from the origin to the surface of the sphere.
    """
    return (center[0] ** 2 + center[1] ** 2 + center[2] ** 2) ** 0.5 + radius


def core_with_detector(
    core_center: tuple[float, float, float] = DEFAULTS["core_center"],
    core_radius: float = DEFAULTS["core_radius"],
    detector_center: tuple[float, float, float] = DEFAULTS["detector_center"],
    detector_radius: float = DEFAULTS["detector_radius"],
) -> openmc.Geometry:
    """Create a geometry with a spherical core and a spherical detector.
    Parameters
    ----------
    core_center : tuple of float
        The (x, y, z) coordinates of the core sphere center.
    core_radius : float
        The radius of the core sphere.
    detector_center : tuple of float
        The (x, y, z) coordinates of the detector sphere center.
    detector_radius : float
        The radius of the detector sphere.
    Returns
    -------
    openmc.Geometry
        The geometry object representing the core and detector.
    """
    core_surface = openmc.Sphere(
        r=core_radius, x0=core_center[0], y0=core_center[1], z0=core_center[2]
    )
    detector_surface = openmc.Sphere(
        r=detector_radius,
        x0=detector_center[0],
        y0=detector_center[1],
        z0=detector_center[2],
    )

    max_d = max(
        _max_distance(core_center, core_radius),
        _max_distance(detector_center, detector_radius),
    )

    outer_surface = openmc.Sphere(r=max_d + 1.0, boundary_type="vacuum")

    core_cell = openmc.Cell(region=-core_surface, fill=mixture_material)
    detector_cell = openmc.Cell(region=-detector_surface, fill=detector_material)
    void_cell = openmc.Cell(region=+core_surface & +detector_surface & -outer_surface)

    return openmc.Geometry([core_cell, void_cell, detector_cell])
