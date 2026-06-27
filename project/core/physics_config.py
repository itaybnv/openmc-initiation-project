"""Physics configuration settings for OpenMC simulations.

Each public function returns a FRESH settings dict, so callers can mutate the
result (e.g. ``settings["source"] = ...``) without affecting other importers.
The module-level dicts these replaced were shared state: one script's mutation
leaked into every other script that imported them.
"""

import openmc
from openmc import stats

from .geometry import DETECTOR_CELL_ID

__all__ = [
    "eigenvalue_settings",
    "detector_settings",
    "perfect_detector_settings",
    "perturbation_settings",
]


def _point_pulse_source() -> openmc.IndependentSource:
    """2.5 MeV point source at the origin, single pulse at t=0."""
    return openmc.IndependentSource(
        space=stats.Point((0.0, 0.0, 0.0)),
        energy=stats.Discrete([2.5e6], [1.0]),
        strength=1.0,
        time=stats.Discrete([0.0], [1.0]),
    )


def detector_max_collisions(particles: int) -> int:
    """Upper bound on tracked detector captures for a point/small detector.

    The detector capture fraction is at most a few percent of histories (B-10
    point detector; ~2.6% even near critical at k=0.98), so the old
    ``50 * particles`` bound over-allocated the collision bank by ~1000x and made
    multi-million-history runs infeasible (10M -> 500M slots -> tens of GB).
    ``particles // 4`` with a 500k floor keeps >~10x headroom over the worst
    observed rate while staying allocatable for 10M+-history runs.
    """
    return max(500_000, particles // 4)


def eigenvalue_settings(particles: int = 10_000, batches: int = 100, inactive: int = 30) -> dict:
    """Settings for a k-effective criticality calculation."""
    return {
        "run_mode": "eigenvalue",
        "batches": batches,
        "inactive": inactive,
        "particles": particles,
    }


def detector_settings(particles: int = 1_000_000) -> dict:
    """Fixed-source pulse with collision tracking on the detector cell.

    batches must stay 1 for collision-track runs: parent_id recycles per batch,
    so multi-batch collision banks cannot be attributed unambiguously.
    """
    return {
        "run_mode": "fixed source",
        "batches": 1,
        "particles": particles,
        "collision_track": {
            "max_collisions": detector_max_collisions(particles),
            "reactions": [101],  # total capture
            "cell_ids": [DETECTOR_CELL_ID],
        },
        "source": _point_pulse_source(),
    }


def perfect_detector_settings(particles: int = 1_000_000) -> dict:
    """Like detector_settings but sized for the perfect-detector shell geometry."""
    settings = detector_settings(particles)
    settings["collision_track"]["max_collisions"] = 100 * particles
    return settings


def perturbation_settings(particles: int = 50_000, batches: int = 20) -> dict:
    """Fixed-source tally-only settings for detector-perturbation comparisons.

    Uses batches > 1 so tally std_dev is defined (batches=1 gives no variance).
    Deliberately has NO collision_track -- see detector_settings for why
    collision-track runs must stay single-batch.
    """
    return {
        "run_mode": "fixed source",
        "batches": batches,
        "particles": particles,
        "source": _point_pulse_source(),
    }
