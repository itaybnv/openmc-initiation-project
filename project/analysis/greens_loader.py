"""Loader + analysis helpers for the corrected per-cell Green's function sweep.

The spatial Green's function is produced by `project.simulations.greens_sweep`, which runs
one confined-source simulation per coarse mesh cell. Because each source is confined to a
single cell, every detector capture originates from that cell BY CONSTRUCTION -- no
`parent_id` attribution, so this data is immune to the FileSource random-sampling bug that
flattened the original `mesh_ids[parent_id-1]` method.

Each sweep file `results/greens-sweep-k<label>.npz` provides, on an (n_r, n_theta) grid:
    G        : prompt detector captures per source neutron  (time-integrated g_i)
    G_err    : Poisson 1-sigma on G
    g_t      : prompt detection-time histogram per cell      (for Rossi R(tau))
    dist     : distance from cell centroid to detector [cm]
    r_mid    : cell centroid radius [cm]
    cos_mid  : cell centroid cos(theta)  (+1 = toward +Z detector)
    r_edges, cos_edges, time_edges
"""
from pathlib import Path

import numpy as np
import openmc
from scipy.optimize import curve_fit
from scipy.signal import correlate as sp_correlate

from project.core.rossi_mesh import KEFF_CONFIGS, N_RADIAL, N_THETA

RESULTS_ROOT = Path(__file__).resolve().parent.parent / "results"
NU_NU_MINUS_1 = 4.63  # U235 second factorial moment of fission multiplicity

# label -> directory of the global (uniform-source) run holding the fission tally
_FISSION_DIR = {
    "098": "greens-function-simulation",
    "090": "greens-function-k090",
    "070": "greens-function-k070",
    "040": "greens-function-k040",
}
K_OF_LABEL = {v["label"].replace("k", ""): k for k, v in KEFF_CONFIGS.items()}


def load_sweep(label: str) -> dict:
    """Load a corrected per-cell sweep .npz into a plain dict of arrays."""
    f = np.load(RESULTS_ROOT / f"greens-sweep-k{label}.npz")
    d = {key: f[key] for key in f.files}
    d["label"] = label
    d["k"] = float(d["k"])
    return d


def fission_coarse(label: str, n_r: int, n_th: int) -> np.ndarray:
    """Aggregate the fine (N_RADIAL x N_THETA) fission tally onto the coarse sweep grid.

    Returns F_i = nu(nu-1) * fissions/source on an (n_r, n_th) grid. The r^(1/3) radial and
    equal-cos angular edges nest exactly (N_RADIAL % n_r == 0, N_THETA % n_th == 0), so the
    coarse cell is a clean block-sum of fine cells.
    """
    if N_RADIAL % n_r or N_THETA % n_th:
        raise ValueError(f"coarse grid {n_r}x{n_th} must divide fine {N_RADIAL}x{N_THETA}")
    res_dir = RESULTS_ROOT / _FISSION_DIR[label]
    with openmc.StatePoint(str(res_dir / "statepoint.1.h5")) as sp:
        F_tally = sp.get_tally(name="fission_per_cell").mean.flatten()  # flat: i_theta*N_RADIAL + i_r
    F_fine = F_tally.reshape(N_THETA, N_RADIAL)                  # (theta, r)
    kr, kth = N_RADIAL // n_r, N_THETA // n_th
    F_blk = F_fine.reshape(n_th, kth, n_r, kr).sum(axis=(1, 3))  # (n_th, n_r)
    return NU_NU_MINUS_1 * F_blk.T                               # (n_r, n_th)


def radial_profile(sweep: dict):
    """G marginalised over theta (detection-weighted): returns r_mid_1d, G_r, G_r_err."""
    det = sweep["n_prompt"].astype(float)          # (n_r, n_th)
    n = float(sweep["n_source"])
    det_r = det.sum(axis=1)
    G_r = det_r / (n * sweep["n_prompt"].shape[1])
    G_r_err = G_r / np.sqrt(np.maximum(det_r, 1))
    r_mid_1d = sweep["r_mid"].mean(axis=1)
    return r_mid_1d, G_r, G_r_err


def angular_profile(sweep: dict, r_frac_min: float = 0.0):
    """G vs cos(theta), restricted to outer shells with r/R >= r_frac_min.

    Returns cos_1d, G_theta, G_theta_err.
    """
    R = float(sweep["R"])
    sel = sweep["r_mid"] / R >= r_frac_min          # (n_r, n_th) bool
    det = np.where(sel, sweep["n_prompt"], 0).astype(float)
    n = float(sweep["n_source"])
    det_th = det.sum(axis=0)
    n_shells = sel.sum(axis=0).clip(min=1)
    G_th = det_th / (n * n_shells)
    G_th_err = G_th / np.sqrt(np.maximum(det_th, 1))
    cos_1d = sweep["cos_mid"].mean(axis=0)
    return cos_1d, G_th, G_th_err


def polar_meridian(ax, sweep: dict, norm=None, cmap: str = "plasma"):
    """Draw G_i as a full-disk meridional slice on a polar axis `ax`.

    The slice is azimuthally symmetric about the detector axis, so the disk is built by
    mirroring the (r, theta) data left/right. The detector direction (theta=0, +Z) is placed
    at the top and theta=pi (far side) at the bottom; disk centre = core centre, rim = surface.
    Returns the QuadMesh (for a shared colorbar).
    """
    r_edges = sweep["r_edges"]
    th = np.arccos(np.clip(sweep["cos_edges"], -1, 1))     # 0..pi  (len n_th+1)
    G = sweep["G"]                                          # (n_r, n_th)
    full_theta = np.concatenate([th, (2 * np.pi - th[::-1])[1:]])   # 0..pi..2pi
    full_C = np.concatenate([G, G[:, ::-1]], axis=1)               # (n_r, 2*n_th)
    ax.set_theta_zero_location("N")     # theta=0 (toward detector) at top
    ax.set_theta_direction(-1)
    im = ax.pcolormesh(full_theta, r_edges, full_C, shading="flat", cmap=cmap, norm=norm)
    ax.set_xticklabels([])              # angular tick labels redundant (theta=0 at top)
    ax.set_yticklabels([])              # radial cm scale lives in the (r, theta) heatmap above
    ax.grid(False)
    return im


def plot_geometry_mesh(ax, R: float, n_r: int = 20, n_th: int = 20,
                       detector_gap: float = 10.0, detector_r: float = 5.0):
    """Meridional (x-z) schematic proving the detector lies on the mesh polar axis.

    Draws the core circle, the actual tally mesh (equal-volume radial shells + equal-solid-angle
    theta bands), and the detector disk on +Z. theta is measured from +Z, so the detector sits on
    the theta=0 axis -- i.e. cos(theta)=+1 genuinely means "toward the detector". The near- and
    far-pole outer cells are highlighted with their cell->detector distances.
    """
    import matplotlib.patches as mpatches

    det_z = R + detector_gap + detector_r          # detector centre on +Z
    r_edges = R * np.linspace(0, 1, n_r + 1) ** (1 / 3)
    theta_edges = np.arccos(np.linspace(1, -1, n_th + 1))   # 0..pi from +Z

    # core
    ax.add_patch(mpatches.Circle((0, 0), R, fill=False, ec="k", lw=1.8, zorder=3))
    # radial shell circles (full disk)
    for r in r_edges[1:-1]:
        ax.add_patch(mpatches.Circle((0, 0), r, fill=False, ec="0.7", lw=0.5, zorder=1))
    # theta band lines (cones -> lines in the meridian, both x>0 and x<0 halves)
    for th in theta_edges:
        x, z = R * np.sin(th), R * np.cos(th)
        ax.plot([0, x], [0, z], color="0.7", lw=0.5, zorder=1)
        ax.plot([0, -x], [0, z], color="0.7", lw=0.5, zorder=1)
    # detector
    ax.add_patch(mpatches.Circle((0, det_z), detector_r, fc="#1f77b4", ec="k", lw=1.2, zorder=4))
    ax.annotate("detector\n(B-10)", (0, det_z), (0.6 * R, det_z), fontsize=9,
                ha="left", va="center", arrowprops=dict(arrowstyle="->", lw=1))
    # detector axis = theta = 0
    ax.plot([0, 0], [0, det_z], ls="--", color="#1f77b4", lw=1, zorder=2)
    ax.annotate(r"$\theta=0$ axis", (0, 0.6 * det_z), fontsize=8, color="#1f77b4",
                rotation=90, ha="right", va="center")

    # highlight near- & far-pole outer cells (outermost shell, polar caps)
    r_out = r_edges[-2]
    for th0, th1, lbl, col in [(theta_edges[0], theta_edges[1], "near pole", "#2ca02c"),
                               (theta_edges[-2], theta_edges[-1], "far pole", "#d62728")]:
        ths = np.linspace(th0, th1, 12)
        xs = np.concatenate([r_out * np.sin(ths), R * np.sin(ths[::-1])])
        zs = np.concatenate([r_out * np.cos(ths), R * np.cos(ths[::-1])])
        ax.fill(xs, zs, color=col, alpha=0.55, zorder=2)
        ax.fill(-xs, zs, color=col, alpha=0.55, zorder=2)
        r_c = 0.5 * (r_out + R)
        cell_z = r_c * np.cos(0.5 * (th0 + th1))
        dist = abs(det_z - cell_z)
        ax.annotate(f"{lbl}\n d={dist:.0f} cm", (0, cell_z), (-0.95 * R, cell_z),
                    fontsize=8, color=col, ha="right", va="center")

    ax.annotate("core\n(U235-H2O)", (0, -0.45 * R), fontsize=9, ha="center", va="center", color="k")
    lim = det_z + detector_r + 3
    ax.set_xlim(-1.15 * R, lim * 0.55 + 0.6 * R)
    ax.set_ylim(-1.15 * R, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("x (cm)"); ax.set_ylabel("z (cm)")


def fundamental_mode_shape(r, R: float):
    """Idealised bare-sphere fundamental mode sin(pi r/R)/(pi r/R), normalised to 1 at r=0.

    Zero-flux outer boundary (no extrapolation length) -> qualitative importance reference.
    """
    return np.sinc(np.asarray(r) / R)     # np.sinc(x) = sin(pi x)/(pi x)


def directional_vs_time(sweep: dict, r_frac_min: float = 0.5, windows=None):
    """Facing/away detection ratio vs neutron arrival time (direct vs multiplied decomposition).

    Early arrivals are direct first-flight (strongly directional); late arrivals are fission-chain
    (importance-like, contrast -> 1 near critical). Restricted to outer shells (r/R >= r_frac_min)
    where the directional contrast lives. Returns an array with columns
    [window_centre_s, facing/away ratio, ratio_err, n_facing, n_away].
    """
    g_t = sweep["g_t"].astype(np.float64)              # (n_r, n_th, N_T)
    te = sweep["time_edges"]; tc = 0.5 * (te[:-1] + te[1:])
    R = float(sweep["R"])
    outer = sweep["r_mid"] / R >= r_frac_min
    facing = outer & (sweep["cos_mid"] > 0)
    away = outer & (sweep["cos_mid"] < 0)
    gf = g_t[facing].sum(axis=0)                       # (N_T,)
    ga = g_t[away].sum(axis=0)
    if windows is None:
        # fine log-spaced windows so the monotonic decay to 1 is resolved (not just 5 points)
        windows = [(0, 2e-5), (2e-5, 4e-5), (4e-5, 8e-5), (8e-5, 1.6e-4), (1.6e-4, 3.2e-4),
                   (3.2e-4, 6.4e-4), (6.4e-4, 1.3e-3), (1.3e-3, 2.6e-3), (2.6e-3, 5e-3), (5e-3, 1e-2)]
    rows = []
    for lo, hi in windows:
        m = (tc >= lo) & (tc < hi)
        nf, na = gf[m].sum(), ga[m].sum()
        if nf > 0 and na > 0:
            ratio = nf / na
            err = ratio * np.sqrt(1 / nf + 1 / na)
        else:
            ratio = err = np.nan
        rows.append([0.5 * (lo + hi), ratio, err, nf, na])
    return np.array(rows)


def rossi(sweep: dict, F_coarse: np.ndarray, tau_max: float = 1e-2, n_tau: int = 256):
    """R(tau) = sum_i F_i * integral g_i(t) g_i(t+tau) dt from the corrected per-cell g_i(t).

    g_i(t) is the per-source detection rate in cell i (sweep g_t / n_source). Returns
    tau_centers, R_tau.
    """
    g_t = sweep["g_t"].astype(np.float64)           # (n_r, n_th, N_T) counts
    n = float(sweep["n_source"])
    time_edges = sweep["time_edges"]
    dt = time_edges[1] - time_edges[0]
    n_t = g_t.shape[-1]
    n_lag = min(n_tau, n_t - 1)
    tau_centers = (np.arange(1, n_lag + 1) - 0.5) * dt

    R_tau = np.zeros(n_lag)
    n_r, n_th = g_t.shape[:2]
    for ir in range(n_r):
        for ith in range(n_th):
            g = g_t[ir, ith] / n                    # per-source rate
            if g.sum() * n < 2:                     # need >=2 events for a pair
                continue
            corr = sp_correlate(g, g, mode="full", method="fft")
            R_tau += F_coarse[ir, ith] * corr[n_t: n_t + n_lag] * dt
    return tau_centers, R_tau


def rossi_model(tau, A, B, alpha):
    return A + B * np.exp(alpha * tau)


def fit_alpha(tau, R_tau):
    """Fit R(tau) = A + B exp(alpha tau). Returns (popt, perr) or (None, None)."""
    p0 = [R_tau[-1], R_tau[0] - R_tau[-1], -500.0]
    try:
        popt, pcov = curve_fit(rossi_model, tau, R_tau, p0=p0, maxfev=50000)
        return popt, np.sqrt(np.diag(pcov))
    except Exception:
        return None, None
