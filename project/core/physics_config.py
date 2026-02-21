"""Physics configuration settings for OpenMC simulations."""

import openmc
from openmc import stats

__all__ = ["eigenvalue_settings", "detector_settings"]

eigenvalue_settings = {
    "run_mode": "eigenvalue",
    "batches": 100,
    "inactive": 30,
    "particles": 10000,
}

detector_settings = {
    "run_mode": "fixed source",
    "batches": 1,
    "particles": (particles := 1_000_000),
    "collision_track": {
        "max_collisions": 50 * particles,
        "reactions": [101],  # Total capture reaction
        "cell_ids": [2],  # Assuming detector cell has ID 2
    },
    "source": openmc.IndependentSource(
        space=stats.Point((0.0, 0.0, 0.0)),
        energy=stats.Discrete([2.5e6], [1.0]),
        strength=1.0,
        time=stats.Discrete([0.0], [1.0]),  # Emission at time zero - single pulse
    ),
}

perfect_detector_settings = {
    "run_mode": "fixed source",
    "batches": 1,
    "particles": (particles := 1_000_000),
    "collision_track": {
        "max_collisions": 100 * particles,
        "reactions": [101],  # Total capture reaction
        "cell_ids": [2],  # Assuming perfect detector cell has ID 2
    },
    "source": openmc.IndependentSource(
        space=stats.Point((0.0, 0.0, 0.0)),
        energy=stats.Discrete([2.5e6], [1.0]),
        strength=1.0,
        time=stats.Discrete([0.0], [1.0]),  # Emission at time zero - single pulse
    ),
}
