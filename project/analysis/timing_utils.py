"""Shared timing-analysis utilities for the collision-track notebooks.

Extracted from the boilerplate previously duplicated across detector_timing,
perfect_detector_timing and delayed_neutrons: collision-track loading, log-spaced
histograms, exponential decay fits, sibling multiplicity, and the O(n)
searchsorted Rossi pair-count histogram.
"""
from collections import Counter
from pathlib import Path

import numpy as np
import openmc
import pandas as pd
from scipy.optimize import curve_fit

from project.core.geometry import DETECTOR_CELL_ID

__all__ = [
    "DETECTOR_CELL_ID",
    "load_collision_tracks",
    "load_detector_times",
    "log_hist",
    "decay_model",
    "fit_exp_decay",
    "rossi_alpha_model",
    "fit_rossi_alpha",
    "sibling_multiplicity",
    "rossi_pair_hist",
    "plot_histogram",
]


def load_collision_tracks(collision_h5: Path | str) -> pd.DataFrame:
    """Load a collision_track.h5 into a flat DataFrame (r/u compound fields expanded)."""
    data = openmc.read_collision_track_file(str(collision_h5))
    df = pd.DataFrame({name: data[name] for name in data.dtype.names if name not in ("r", "u")})
    for vec in ("r", "u"):
        for ax in ("x", "y", "z"):
            df[f"{vec}_{ax}"] = data[vec][ax]
    return df


def load_detector_times(
    collision_h5: Path | str,
    cell_id: int = DETECTOR_CELL_ID,
    prompt_only: bool = False,
) -> np.ndarray:
    """Detection times in the given cell, cleaned to finite positive values.

    prompt_only=False keeps delayed-neutron events (delayed_group != 0); they are
    ~26% of detections beyond 10 ms in near-critical configs, so dropping them is
    an analysis choice the caller must make explicitly.
    """
    data = openmc.read_collision_track_file(str(collision_h5))
    sel = data["cell_id"] == cell_id
    if prompt_only:
        sel &= data["delayed_group"] == 0
    t = np.asarray(data["time"][sel], dtype=float)
    return t[np.isfinite(t) & (t > 0)]


def log_hist(times: np.ndarray, n_bins: int = 256) -> pd.DataFrame:
    """Histogram on log-spaced bins spanning the data range.

    Returns a DataFrame with columns count, density, left_edge, right_edge,
    center, width (the layout the timing notebooks build by hand).
    """
    t = np.asarray(times, dtype=float)
    t = t[np.isfinite(t) & (t > 0)]
    if t.size == 0:
        raise ValueError("No valid positive timestamps found for histogram")
    edges = np.logspace(np.log10(t.min()), np.log10(t.max()), n_bins + 1)
    counts, edges = np.histogram(t, bins=edges)
    widths = np.diff(edges)
    return pd.DataFrame(
        {
            "count": counts,
            "density": counts / widths,
            "left_edge": edges[:-1],
            "right_edge": edges[1:],
            "center": 0.5 * (edges[:-1] + edges[1:]),
            "width": widths,
        }
    )


def decay_model(t, A, alpha, C):
    """Prompt-decay model A * exp(-alpha * t) + C."""
    return A * np.exp(-alpha * t) + C


def fit_exp_decay(t: np.ndarray, counts: np.ndarray, p0=None, maxfev: int = 20000):
    """Fit counts(t) = A exp(-alpha t) + C. Returns (popt, perr)."""
    if p0 is None:
        p0 = [counts.max(), 1000.0, counts.min()]
    popt, pcov = curve_fit(decay_model, t, counts, p0=p0, maxfev=maxfev)
    return popt, np.sqrt(np.diag(pcov))


def rossi_alpha_model(tau, A, B, alpha):
    """Rossi-alpha model A + B * exp(alpha * tau) (alpha < 0 below critical)."""
    return A + B * np.exp(alpha * tau)


def fit_rossi_alpha(tau: np.ndarray, counts: np.ndarray, p0=None, maxfev: int = 20000):
    """Fit the Rossi histogram. Returns (popt, perr)."""
    if p0 is None:
        tail = counts[-max(len(counts) // 25, 1):].mean()
        p0 = [tail, counts[0] - tail, -1e4]
    popt, pcov = curve_fit(rossi_alpha_model, tau, counts, p0=p0, maxfev=maxfev)
    return popt, np.sqrt(np.diag(pcov))


def sibling_multiplicity(parent_ids) -> tuple[np.ndarray, np.ndarray, dict]:
    """Distribution of detections per parent particle.

    Returns (multiplicities [1..max], counts per multiplicity, stats dict with
    total_parents, total_detections, mean/max multiplicity and the Fano factor
    of the multiplicity distribution).
    """
    detections_per_parent = Counter(parent_ids)
    multiplicity_dist = Counter(detections_per_parent.values())
    max_mult = max(multiplicity_dist)
    multiplicities = np.arange(1, max_mult + 1)
    counts = np.array([multiplicity_dist.get(m, 0) for m in multiplicities])

    total_parents = counts.sum()
    total_detections = (multiplicities * counts).sum()
    mean = total_detections / total_parents if total_parents else 0.0
    var = ((multiplicities - mean) ** 2 * counts).sum() / total_parents if total_parents else 0.0
    stats = {
        "total_parents": int(total_parents),
        "total_detections": int(total_detections),
        "mean_multiplicity": float(mean),
        "max_multiplicity": int(max_mult),
        "variance": float(var),
        "fano_factor": float(var / mean) if mean else float("nan"),
    }
    return multiplicities, counts, stats


def rossi_pair_hist(detection_times: np.ndarray, dt_max: float, n_bins: int):
    """Rossi pair-delay histogram with O(n) memory (searchsorted, no pair loop).

    detection_times must be sorted ascending. Returns (bin_centers, counts).
    """
    times = np.asarray(detection_times, dtype=np.float64)
    n = len(times)
    bin_edges = np.linspace(0, dt_max, n_bins + 1)
    counts = np.zeros(n_bins, dtype=np.int64)
    min_j = np.arange(1, n + 1)  # enforce j > i

    # Loop over bins, not over pairs: each bin edge's searchsorted result doubles
    # as the lower bound for the next bin.
    prev_bound = min_j  # bin_edges[0] = 0
    for b in range(n_bins):
        curr_bound = np.maximum(
            np.searchsorted(times, times + bin_edges[b + 1], side="left"),
            min_j,
        )
        counts[b] = (curr_bound - prev_bound).sum()
        prev_bound = curr_bound

    return 0.5 * (bin_edges[:-1] + bin_edges[1:]), counts


def plot_histogram(centers, values, widths, xscale, yscale, ylabel, title,
                   filename=None, figures_dir: Path | str | None = None):
    """Consistent bar-histogram plot; saves to figures_dir/filename when given."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6), dpi=80)
    ax.bar(centers, values, width=widths, align="center", edgecolor="k", alpha=0.8)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.tight_layout()
    if filename is not None:
        out = (Path(figures_dir) if figures_dir else Path(".")) / filename
        plt.savefig(out, dpi=300, bbox_inches="tight")
    return fig, ax
