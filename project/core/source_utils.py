"""Utilities for pre-generating source files and mapping source particles to mesh cells."""

import os
from pathlib import Path

import numpy as np
import openmc
import openmc.lib


def pregenerate_source(
    model: openmc.Model,
    n_particles: int,
    output_dir: Path,
    source_filename: str = "initial_source.h5",
) -> Path:
    """Sample source particles from the model's source distribution and write to HDF5.

    After writing, the model's source is swapped to an openmc.FileSource pointing
    at the generated file, so the subsequent simulation uses these exact particles.

    Parameters
    ----------
    model : openmc.Model
        A built OpenMC model (source must already be configured).
    n_particles : int
        Number of source particles to sample.
    output_dir : Path
        Directory for model XML and the source file.
    source_filename : str
        Name of the output HDF5 source file.

    Returns
    -------
    Path
        Absolute path to the generated source HDF5 file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_path = (output_dir / source_filename).resolve()

    # Export model XML so openmc.lib can load it
    model.export_to_model_xml(output_dir / "model.xml")

    # Sample particles from the defined source distribution
    orig_dir = os.getcwd()
    try:
        os.chdir(output_dir)
        openmc.lib.init(output=False)
        particles = openmc.lib.sample_external_source(n_samples=n_particles)
        openmc.lib.finalize()
    finally:
        os.chdir(orig_dir)

    particles.export_to_hdf5(source_path)

    # Swap source to FileSource so the simulation uses these exact particles
    model.settings.source = [openmc.FileSource(str(source_path))]

    # Re-export model XML with the FileSource
    model.export_to_model_xml(output_dir / "model.xml")

    return source_path


def map_source_to_mesh(
    source_path: Path | str,
    mesh: openmc.SphericalMesh,
) -> np.ndarray:
    """Map each source particle to its containing mesh cell index.

    Uses the mesh's r_grid and theta_grid (and phi_grid) to bin each
    particle's position. The flat index follows OpenMC's convention:
    r varies fastest, phi varies slowest.

    .. warning::
        Do NOT use the returned array to attribute detector captures back to a birth
        cell via ``mesh_ids[parent_id - 1]``. ``openmc.FileSource`` samples a *random*
        site per history, so the simulated particle ``id`` does not equal the source-file
        row index -- the attribution is scrambled and flattens the spatial Green's
        function to a constant. Use ``project.simulations.greens_sweep`` (one confined
        source per cell) for correct per-cell attribution. This function remains valid
        only for verifying source uniformity (counts per cell), not detector attribution.

    Parameters
    ----------
    source_path : Path or str
        Path to the source HDF5 file.
    mesh : openmc.SphericalMesh
        The spherical mesh used for tallies.

    Returns
    -------
    np.ndarray
        Array of shape (n_particles,) with the flat mesh cell index for
        each source particle. Particles outside all bins get index -1.
    """
    df = openmc.read_source_file(str(source_path)).to_dataframe()

    x = df["x"].to_numpy()
    y = df["y"].to_numpy()
    z = df["z"].to_numpy()

    # Spherical coordinates relative to mesh origin
    ox, oy, oz = mesh.origin
    dx, dy, dz = x - ox, y - oy, z - oz

    r = np.sqrt(dx**2 + dy**2 + dz**2)
    theta = np.arccos(np.clip(dz / np.maximum(r, 1e-30), -1, 1))

    r_edges = np.asarray(mesh.r_grid)
    theta_edges = np.asarray(mesh.theta_grid)
    phi_edges = np.asarray(mesh.phi_grid)

    n_r = len(r_edges) - 1
    n_theta = len(theta_edges) - 1
    n_phi = len(phi_edges) - 1

    # digitize returns 1-based bin indices; subtract 1 for 0-based
    i_r = np.digitize(r, r_edges) - 1
    i_theta = np.digitize(theta, theta_edges) - 1

    if n_phi > 1:
        phi = np.arctan2(dy, dx) % (2 * np.pi)
        i_phi = np.digitize(phi, phi_edges) - 1
    else:
        i_phi = np.zeros(len(r), dtype=int)

    # Mark out-of-bounds particles
    valid = (
        (i_r >= 0) & (i_r < n_r)
        & (i_theta >= 0) & (i_theta < n_theta)
        & (i_phi >= 0) & (i_phi < n_phi)
    )

    # OpenMC flat index: r fastest, phi slowest
    flat = np.where(
        valid,
        i_phi * n_theta * n_r + i_theta * n_r + i_r,
        -1,
    )

    return flat
