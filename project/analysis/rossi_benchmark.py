"""Rossi-alpha methods benchmark: point kinetics vs brute-force doubles vs ACDC.

Three estimators of the correlated Rossi-alpha distribution R(tau) and the prompt
decay constant alpha, compared on accuracy and statistical-error-vs-compute-budget:

* Brute force (BF): same-parent (= same fission chain) detection-pair time
  differences from a whole-sphere fixed-source pulse run. parent_id groups all
  in-history descendants of one source neutron, so grouping by it is exact in
  fixed-source mode. Budget emulation: histories are iid and parent_id in 1..N
  indexes them, so a parent-id block of size N' is an exact N'-history run.
* ACDC: R(tau) = sum_i F_i * integral g_i(t) g_i(t+tau) dt from the per-cell
  confined-source sweep (full chain), F_i = NU2_PROMPT * fissions/source.
* Point kinetics (PK): alpha = (k_p - 1)/l, k_p = k(1 - beta_eff); l and beta_eff
  from IFP eigenvalue runs (see project.simulations.ifp_kinetics).

Normalization convention (both estimators must produce the SAME absolute curve):
R(tau) = expected same-chain detection pairs per source neutron per second of lag,
with each unordered pair counted once at its positive separation. The un-halved
second factorial moment counts ordered pairs once each at +tau, matching this.

THE PAIR MOMENT — OpenMC UNDER-DISPERSES FISSION MULTIPLICITY (measured 2026-06):
OpenMC v0.15.3 does NOT sample the physical multiplicity distribution P(nu).
physics.cpp create_fission_sites banks  n = floor(wgt * nubar*Sigma_f/Sigma_t + xi)
neutrons per U235 collision — expectation-correct (k, fluxes, fission rates are all
right) but VARIANCE-MINIMAL. Direct measurement (particle-tracks join on the k=0.40
sphere, 978 fission events): daughters per site-creating collision {1: 42, 2: 901,
3: 35}, E[n] = 1.99, E[n(n-1)] = 2.06 — versus the physical U235 thermal
<nu(nu-1)> = 4.63 (Diven factor 0.80 vs simulated ~0.52).

Consequences:
* alpha (a time constant) is unaffected, but every PAIR AMPLITUDE in this
  codebase's simulations (Rossi B, doubles rates, sibling-pair counts) is lower
  than physical reality by the moment ratio ~1.6x.
* For the benchmark, BOTH estimators must live in the simulated world, so ACDC
  uses the empirically calibrated M_SIM below (per-fission-tally-unit pair moment;
  consistent across all four k: 2.92+-0.13, 2.57+-0.14, 2.75+-0.32, 2.92+-0.47).
* A real-world ACDC application replaces M_SIM with the physical prompt moment
  NU2_PHYSICAL — a pure scale factor on R(tau), irrelevant to alpha and to the
  sigma-vs-budget comparisons.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, minimize_scalar

from project.analysis import greens_loader as gl
from project.analysis.timing_utils import load_collision_tracks
from project.core.geometry import DETECTOR_CELL_ID

# Simulated-world pair moment per fission-tally unit (weighted mean over the four
# k labels, chain-bootstrap errors; see module docstring + calibrate_moment()).
M_SIM = 2.76

# ACDC over-dispersion correction on Poisson-bootstrap sigma (Phase-2 verdict,
# validate_error_model on 6 event-level cells, 2026-06-13): per-bin Poisson
# treatment of g_t under-estimates the chain-level sigma of the windowed R
# contribution by kappa_S = 1.14-1.33 at k=0.98 (within-chain multiplicity 1.1-1.8)
# and ~1.0 at k=0.40 (multiplicity 1.00). Interpolated per label; the k098 spread
# (+-0.1) is carried as a systematic on the ACDC error bar. Event-level chain
# bootstrap (no kappa needed) is used wherever events.h5 exists.
KAPPA_S = {"098": 1.3, "095": 1.25, "090": 1.2, "080": 1.15,
           "070": 1.1, "060": 1.05, "050": 1.03, "040": 1.0}
# Physical U235 prompt second factorial moment, for real-world application only:
# <nu_p(nu_p-1)> ~= <nu(nu-1)> - 2<nu_p><nu_d> = 4.63 - 2*2.42*0.0158 ~= 4.55
NU2_PHYSICAL = 4.55

T_MAX = 1.0e-2  # standard lag window [s], matches the sweep g_t histograms
N_T = 512
DT = T_MAX / N_T


# --------------------------------------------------------------------------- #
# Brute force: event loading, same-parent pairs, budget subsampling
# --------------------------------------------------------------------------- #

def load_chain_events(label: str, prompt_only: bool = True) -> pd.DataFrame:
    """Detector events (time, parent_id) from the 1M-history global run for k label.

    Times and parent_id chain grouping are valid regardless of the historical
    FileSource bug (which only scrambled birth-LOCATION attribution).
    """
    df = load_collision_tracks(gl._fission_dir(label) / "collision_track.h5")
    sel = df["cell_id"] == DETECTOR_CELL_ID
    if prompt_only:
        sel &= df["delayed_group"] == 0
    out = df.loc[sel, ["time", "parent_id"]].reset_index(drop=True)
    return out


def same_parent_pair_dts(times: np.ndarray, parents: np.ndarray, dt_max: float) -> np.ndarray:
    """All same-parent detection-pair separations Delta-t in (0, dt_max].

    Each unordered pair contributes once (its positive separation). Vectorized by
    in-group offset: after sorting by (parent, time), pair (i, i+d) is same-chain
    iff parents match d positions apart.
    """
    order = np.lexsort((times, parents))
    t = np.asarray(times, float)[order]
    p = np.asarray(parents)[order]
    dts = []
    d = 1
    while True:
        same = p[:-d] == p[d:]
        if not same.any():
            break
        dt = t[d:][same] - t[:-d][same]
        dts.append(dt[dt <= dt_max])
        d += 1
    return np.concatenate(dts) if dts else np.array([])


def same_parent_pair_hist(times, parents, lag_edges) -> np.ndarray:
    dts = same_parent_pair_dts(times, parents, lag_edges[-1])
    counts, _ = np.histogram(dts, bins=lag_edges)
    return counts


def bf_rossi(df: pd.DataFrame, n_source: int, tau_max: float = T_MAX, n_tau: int = 256):
    """Absolute BF R(tau): same-parent pairs / source neutron / second of lag.

    Lag bins [(L-1)dt, L*dt) labeled (L-0.5)dt — the same labeling greens_loader.rossi
    uses for discrete lag L, so the two curves overlay bin-for-bin. (Discrete
    autocorrelation smears pairs triangularly within +-dt of L*dt; second-order
    for alpha*dt << 1 — skip the first two bins in fits/overlays.)
    """
    dt = DT
    lag_edges = np.arange(0, n_tau + 1) * dt
    counts = same_parent_pair_hist(df["time"].to_numpy(), df["parent_id"].to_numpy(), lag_edges)
    tau_centers = (np.arange(1, n_tau + 1) - 0.5) * dt
    R = counts / n_source / dt
    return tau_centers, R, counts


def subsample_block(df: pd.DataFrame, n_prime: int, block: int = 0) -> pd.DataFrame:
    """Exact N'-history realization: events whose parent_id falls in block #block."""
    lo, hi = block * n_prime, (block + 1) * n_prime
    return df[(df["parent_id"] > lo) & (df["parent_id"] <= hi)]


def bootstrap_chains(df: pd.DataFrame, stat_fn, n_boot: int = 200, rng=None) -> np.ndarray:
    """Bootstrap over CHAINS (unique parent_id) — pairs within a chain are not
    independent, so resampling whole chains is the valid sigma estimator."""
    rng = np.random.default_rng(rng)
    order = np.argsort(df["parent_id"].to_numpy(), kind="stable")
    t = df["time"].to_numpy()[order]
    p = df["parent_id"].to_numpy()[order]
    uniq, starts = np.unique(p, return_index=True)
    bounds = np.append(starts, len(p))
    out = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        ts, ps = [], []
        for j, g in enumerate(pick):
            s, e = bounds[g], bounds[g + 1]
            ts.append(t[s:e])
            # relabel so two copies of the same chain are NOT merged into one
            ps.append(np.full(e - s, j))
        out.append(stat_fn(pd.DataFrame({
            "time": np.concatenate(ts), "parent_id": np.concatenate(ps),
        })))
    return np.array(out)


def stationarize(df: pd.DataFrame, t_window: float, rng=None) -> pd.DataFrame:
    """Shift each chain by iid U[0, t_window): within-chain Delta-t invariant,
    cross-chain (accidental) pairs become flat in tau for tau << t_window.

    Needed because in the pulsed run all chains start at t=0, so the accidental
    background decays with the same exp(alpha*tau) as the correlated term —
    A + B exp(alpha tau) is degenerate on the raw all-pairs histogram.
    """
    rng = np.random.default_rng(rng)
    uniq, inv = np.unique(df["parent_id"].to_numpy(), return_inverse=True)
    shifts = rng.uniform(0.0, t_window, len(uniq))
    out = df.copy()
    out["time"] = df["time"].to_numpy() + shifts[inv]
    return out


# --------------------------------------------------------------------------- #
# alpha fitting
# --------------------------------------------------------------------------- #

def fit_alpha_mle(dts: np.ndarray, tau_min: float, tau_max: float,
                  background: str = "flat"):
    """Unbinned MLE of alpha for pair separations on [tau_min, tau_max].

    background='none': pdf(t) = alpha exp(alpha t) / (exp(alpha b) - exp(alpha a)).
    background='flat': pdf(t) = w/(b-a) + (1-w) * trunc-exp — the BF same-parent
        histogram carries a small flat floor from between-delayed-branch pairs
        (separations set by precursor-decay time differences) that the prompt
        ACDC formula cannot contain; the ACDC fit absorbs its floor in the A
        term, so the BF estimator must model it too or alpha biases shallow.
    background='auto': 'flat' when n >= 200 (the 2-param fit needs the floor to
        be resolvable), else 'none'; falls back to 'none' if the flat fit
        degenerates (non-finite sigma or w pinned to a boundary).

    Valid at any pair count (a 512-bin LSQ fit is not, e.g. 39 pairs at k=0.40).
    Returns (alpha, sigma_asymptotic, n_pairs); sigma from the observed Fisher
    information (numeric Hessian). Chain bootstrap supersedes sigma when chains
    contribute multiple pairs.
    """
    t = np.asarray(dts, float)
    t = t[(t >= tau_min) & (t <= tau_max)]
    n = len(t)
    if n < 3:
        return np.nan, np.nan, n
    a, b = tau_min, tau_max

    if background == "auto":
        if n < 200:
            background = "none"
        else:
            alpha, sigma, _ = fit_alpha_mle(t, tau_min, tau_max, background="flat")
            if np.isfinite(sigma) and sigma > 0:
                return alpha, sigma, n
            background = "none"

    def log_truncexp(alpha):
        """log pdf values of the truncated exponential at t (stable, alpha<0)."""
        w = alpha * (b - a)
        log_int = alpha * a + np.log(np.expm1(w) / alpha) if w > -700 else alpha * a - np.log(-alpha)
        return alpha * t - log_int

    if background == "none":
        tbar = t.mean()

        def negll(alpha):
            w = alpha * (b - a)
            log_int = alpha * a + np.log(np.expm1(w) / alpha) if w > -700 else alpha * a - np.log(-alpha)
            return -(n * alpha * tbar - n * log_int)

        res = minimize_scalar(negll, bounds=(-1e6, -1e-2), method="bounded")
        alpha = res.x
        h = max(abs(alpha) * 1e-4, 1e-6)
        d2 = (negll(alpha + h) - 2 * negll(alpha) + negll(alpha - h)) / h**2
        return float(alpha), float(1.0 / np.sqrt(d2)) if d2 > 0 else np.nan, n

    if background != "flat":
        raise ValueError(f"unknown background {background!r}")

    from scipy.optimize import minimize

    def negll2(params):
        logit_w, log_neg_alpha = params
        w = 1.0 / (1.0 + np.exp(-logit_w))
        alpha = -np.exp(log_neg_alpha)
        lp = np.logaddexp(
            np.log(w) - np.log(b - a),
            np.log1p(-w) + log_truncexp(alpha),
        )
        return -lp.sum()

    # initialize from the pure-exponential fit, small background
    a0, _, _ = fit_alpha_mle(t, tau_min, tau_max, background="none")
    x0 = np.array([np.log(0.05 / 0.95), np.log(max(-a0, 1.0))])
    res = minimize(negll2, x0, method="Nelder-Mead",
                   options={"xatol": 1e-6, "fatol": 1e-9, "maxiter": 4000})
    logit_w, log_neg_alpha = res.x
    alpha = -np.exp(log_neg_alpha)
    # sigma on alpha via numeric Hessian in the transformed coords, projected
    h = 1e-4
    H = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            e_i = np.eye(2)[i] * h
            e_j = np.eye(2)[j] * h
            H[i, j] = (negll2(res.x + e_i + e_j) - negll2(res.x + e_i - e_j)
                       - negll2(res.x - e_i + e_j) + negll2(res.x - e_i - e_j)) / (4 * h * h)
    try:
        cov = np.linalg.inv(H)
        sigma = abs(alpha) * np.sqrt(max(cov[1, 1], 0))  # d(alpha)/d(log_neg_alpha) = alpha
    except np.linalg.LinAlgError:
        sigma = np.nan
    return float(alpha), float(sigma), n


def _binned_exp(tau, A, B, alpha, dt):
    """A + B exp(alpha tau) integrated over a bin of width dt (removes the
    discretization bias when |alpha| dt is not small, e.g. k=0.40)."""
    x = alpha * dt / 2
    corr = np.sinh(np.clip(x, -700, 700)) / x if abs(x) > 1e-12 else 1.0
    # clip the exponent: curve_fit transiently probes large positive alpha, which
    # overflows exp() and poisons the optimizer; the physical region is unaffected.
    return A + B * np.exp(np.clip(alpha * tau, -700, 700)) * corr


def fit_alpha_binned(tau, R, dt=DT, sigma=None, p0=None):
    """Fit R(tau) = A + B exp(alpha tau) with the bin-integrated model."""
    if p0 is None:
        p0 = [R[-1], R[0] - R[-1], -500.0]
    popt, pcov = curve_fit(
        lambda x, A, B, al: _binned_exp(x, A, B, al, dt),
        tau, R, p0=p0, sigma=sigma, maxfev=50000,
    )
    return popt, np.sqrt(np.diag(pcov))


# --------------------------------------------------------------------------- #
# ACDC
# --------------------------------------------------------------------------- #

def fission_coarse_sim(label: str, n_r: int, n_th: int, moment: float = M_SIM) -> np.ndarray:
    """F_i with the SIMULATED-world pair moment (gl.fission_coarse bakes in 4.63).

    Pass moment=NU2_PHYSICAL to predict what a physical-multiplicity system
    would show (real-world ACDC application); the benchmark default is M_SIM.
    """
    return gl.fission_coarse(label, n_r, n_th) * (moment / gl.NU_NU_MINUS_1)


def calibrate_moment(label: str, n_boot: int = 400, rng=None):
    """Empirical simulated pair moment for one k: ordered same-chain pairs per
    source divided by sum_i fissions_i * G_i^2. Chain-bootstrap sigma."""
    rng = np.random.default_rng(rng)
    sweep = gl.load_sweep(label)
    n_r, n_th = sweep["G"].shape
    F_raw = gl.fission_coarse(label, n_r, n_th) / gl.NU_NU_MINUS_1
    G = sweep["n_prompt"] / float(sweep["n_source"])
    denom = float((F_raw * G**2).sum())
    m = load_chain_events(label).groupby("parent_id").size().to_numpy()

    def stat(mm):
        return (mm * (mm - 1)).sum() / N_SOURCE_BF / denom

    M = stat(m)
    boots = [stat(m[rng.integers(0, len(m), len(m))]) for _ in range(n_boot)]
    return float(M), float(np.std(boots))


def acdc_rossi(sweep: dict, F_coarse: np.ndarray, tau_max: float = T_MAX, n_tau: int = 256):
    """Corrected-units ACDC R(tau) (delegates to the fixed greens_loader.rossi)."""
    return gl.rossi(sweep, F_coarse, tau_max=tau_max, n_tau=n_tau)


def acdc_rossi_events(events_by_cell: dict, F_flat: np.ndarray, n_source_cell: int,
                      time_edges=None, n_tau: int = 256):
    """Cross-chain (U-statistic) ACDC estimator from per-cell raw events.

    The plug-in g*g autocorrelation contains an O(1/N) positive bias from
    same-chain pair products; at sparse counts (far detector) it rivals the
    signal. Here same-chain ordered pairs are subtracted lag-by-lag, leaving the
    unbiased cross-chain estimate of N^2 p_a p_{a+L}.

    events_by_cell: {flat_cell_index: (times, parents)} prompt events.
    F_flat: F_i on the same flat cell indexing. n_source_cell: histories/cell.
    """
    if time_edges is None:
        time_edges = np.linspace(0.0, T_MAX, N_T + 1)
    dt = time_edges[1] - time_edges[0]
    n_t = len(time_edges) - 1
    n_lag = min(n_tau, n_t - 1)
    tau_centers = (np.arange(1, n_lag + 1) - 0.5) * dt
    R = np.zeros(n_lag)
    N = float(n_source_cell)
    for idx, (t, par) in events_by_cell.items():
        h, _ = np.histogram(t, bins=time_edges)
        if h.sum() < 2:
            continue
        # ordered cross-bin pair products at each lag (FFT-free; n_t is small)
        total = np.array([np.dot(h[:-L], h[L:]) for L in range(1, n_lag + 1)], float)
        # same-chain ordered pairs at each lag bin (each unordered pair once at +lag)
        ib = np.searchsorted(time_edges, t, side="right") - 1
        ok = (ib >= 0) & (ib < n_t)
        same = np.zeros(n_lag)
        if ok.sum() >= 2:
            tb, pb = ib[ok], np.asarray(par)[ok]
            order = np.lexsort((tb, pb))
            tb, pb = tb[order], pb[order]
            d = 1
            while True:
                m = pb[:-d] == pb[d:]
                if not m.any():
                    break
                lags = tb[d:][m] - tb[:-d][m]
                lags = lags[(lags >= 1) & (lags <= n_lag)]
                np.add.at(same, lags - 1, 1.0)
                d += 1
        # E[cross pairs] = N(N-1) p_a p_{a+L}; estimate p p via cross/(N(N-1))
        R += F_flat[idx] * (total - same) / (N * (N - 1)) / dt
    return tau_centers, R


# --------------------------------------------------------------------------- #
# ACDC error-model validation (Phase 2)
# --------------------------------------------------------------------------- #

def load_sweep_events(path: Path | str, prompt_only: bool = True) -> dict:
    """Read a greens-sweep events.h5 -> {(ir, ith): (times, parents)} + n_source."""
    import h5py

    out, n_source = {}, None
    with h5py.File(str(path), "r") as f:
        for name, grp in f.items():
            _, ir, ith = name.split("_")
            t = grp["time"][()]
            p = grp["parent_id"][()]
            if prompt_only:
                sel = grp["delayed_group"][()] == 0
                t, p = t[sel], p[sel]
            out[(int(ir), int(ith))] = (t, p)
            n_source = int(grp.attrs["n_source"])
    return out, n_source


def validate_error_model(events_path: Path | str, n_boot: int = 400, rng=None) -> pd.DataFrame:
    """Over-dispersion factors kappa = sigma_chain / sigma_poisson per cell.

    kappa_G: on the time-integrated G (per-bin Poisson treatment of the count
    total vs resampling whole chains incl. the undetected ones).
    kappa_S: on the cell's windowed plug-in R contribution sum (Poisson bootstrap
    of histogram bins vs chain bootstrap rebuilding the histogram).

    Decision rule (pre-registered): kappa in [1.0, 1.5] and stable across cells
    -> scale ACDC sigma_alpha(full) by the pooled kappa; otherwise switch the G1
    ACDC error bars to event-level chain machinery.
    """
    rng = np.random.default_rng(rng)
    events, n_src = load_sweep_events(events_path)
    time_edges = np.linspace(0.0, T_MAX, N_T + 1)
    rows = []
    for (ir, ith), (t, p) in events.items():
        uniq, inv = np.unique(p, return_inverse=True)
        m = np.bincount(inv)                       # detections per detected chain
        n_det, n_chains = int(m.sum()), len(uniq)
        # --- kappa_G ---
        boots = []
        for _ in range(n_boot):
            k_det = rng.binomial(n_src, n_chains / n_src)
            boots.append(m[rng.integers(0, n_chains, k_det)].sum())
        var_chain = np.var(boots)
        kappa_g = var_chain / n_det                # Poisson var of the total = N
        # --- kappa_S on the windowed plug-in R sum ---
        h, _ = np.histogram(t, bins=time_edges)
        chain_bins = np.searchsorted(time_edges, t, side="right") - 1
        in_win = (chain_bins >= 0) & (chain_bins < N_T)  # prompt tails past T_MAX exist

        def window_sum(hh):
            n_lag = 256
            return sum(np.dot(hh[:-L].astype(float), hh[L:]) for L in range(1, n_lag + 1))

        s_chain, s_pois = [], []
        for _ in range(n_boot // 2):
            k_det = rng.binomial(n_src, n_chains / n_src)
            pick = rng.integers(0, n_chains, k_det)
            hh = np.zeros(N_T, dtype=int)
            for g in pick:
                np.add.at(hh, chain_bins[(inv == g) & in_win], 1)
            s_chain.append(window_sum(hh))
            s_pois.append(window_sum(rng.poisson(h)))
        kappa_s = np.std(s_chain) / max(np.std(s_pois), 1e-30)
        rows.append({
            "cell": (ir, ith), "n_det": n_det, "n_chains": n_chains,
            "mean_mult": n_det / max(n_chains, 1),
            "kappa_G": float(kappa_g), "kappa_S": float(kappa_s),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Point kinetics
# --------------------------------------------------------------------------- #

def pk_alpha(k: float, beta_eff, gen_time, convention: str):
    """alpha_PK from IFP outputs. UFloat-transparent.

    convention='generation': gen_time is the generation time Lambda; the prompt
        lifetime is l = k * Lambda and alpha = (k(1-beta) - 1) / (k * Lambda).
    convention='lifetime': gen_time IS the prompt lifetime l;
        alpha = (k(1-beta) - 1) / l.
    The two differ by a factor k (2.5x at k=0.40) — resolve empirically against
    the previously fitted l ~= 52 us before trusting either (see ifp_kinetics).
    """
    k_p = k * (1.0 - beta_eff)
    if convention == "generation":
        return (k_p - 1.0) / (k * gen_time)
    if convention == "lifetime":
        return (k_p - 1.0) / gen_time
    raise ValueError(f"unknown convention {convention!r}")


# --------------------------------------------------------------------------- #
# G1 machinery: error vs compute budget
# --------------------------------------------------------------------------- #

MIN_PAIRS_FIT = 50  # below this the BF alpha fit is reported as censored


def bf_alpha_vs_budget(label: str, budgets=None, fit_window=None) -> pd.DataFrame:
    """BF sigma_alpha at each budget N' from DISJOINT parent-id blocks.

    Each block of N' consecutive parent ids is an exact independent realization
    of an N'-history experiment. Per block: unbinned MLE alpha in the frozen
    late-time window; sigma across blocks. Blocks (or whole budgets) with fewer
    than MIN_PAIRS_FIT window pairs are censored — that censoring IS a result
    (BF starves at deep subcritical / far detectors).
    """
    if budgets is None:
        budgets = [int(b) for b in np.logspace(4, 6, 9)]  # 10k .. 1M, 9 points
    lo, hi = fit_window or FIT_WINDOWS[label]
    df = load_chain_events(label)
    t_all, p_all = df["time"].to_numpy(), df["parent_id"].to_numpy()
    rows = []
    for n_prime in budgets:
        n_blocks = N_SOURCE_BF // n_prime
        alphas, pair_counts = [], []
        for b in range(n_blocks):
            sel = (p_all > b * n_prime) & (p_all <= (b + 1) * n_prime)
            dts = same_parent_pair_dts(t_all[sel], p_all[sel], hi)
            dts = dts[(dts >= lo) & (dts <= hi)]
            pair_counts.append(len(dts))
            if len(dts) >= MIN_PAIRS_FIT:
                a, _, _ = fit_alpha_mle(dts, lo, hi, background="auto")
                if np.isfinite(a):
                    alphas.append(a)
        n_conv = len(alphas)
        if n_prime == N_SOURCE_BF and n_conv == 1:
            # single block: sigma via chain bootstrap of the MLE
            boots = bootstrap_chains(
                df, lambda d: fit_alpha_mle(
                    same_parent_pair_dts(d["time"].to_numpy(), d["parent_id"].to_numpy(), hi),
                    lo, hi, background="auto")[0],
                n_boot=100,
            )
            boots = boots[np.isfinite(boots)]
            rows.append({"n_prime": n_prime, "alpha": alphas[0],
                         "sigma": float(np.std(boots)), "n_blocks": 1,
                         "n_convergent": 1, "mean_pairs": float(np.mean(pair_counts)),
                         "censored": False})
        elif n_conv >= 3:
            rows.append({"n_prime": n_prime, "alpha": float(np.mean(alphas)),
                         "sigma": float(np.std(alphas, ddof=1)), "n_blocks": n_blocks,
                         "n_convergent": n_conv, "mean_pairs": float(np.mean(pair_counts)),
                         "censored": False})
        else:
            rows.append({"n_prime": n_prime, "alpha": np.nan, "sigma": np.nan,
                         "n_blocks": n_blocks, "n_convergent": n_conv,
                         "mean_pairs": float(np.mean(pair_counts)), "censored": True})
    return pd.DataFrame(rows)


def bf_block_alphas(label: str, n_prime: int, fit_window=None) -> np.ndarray:
    """Per-block MLE alpha at budget n_prime (for scatter overlays). One value
    per disjoint parent-id block that reaches MIN_PAIRS_FIT window pairs."""
    lo, hi = fit_window or FIT_WINDOWS[label]
    df = load_chain_events(label)
    t, p = df["time"].to_numpy(), df["parent_id"].to_numpy()
    out = []
    for b in range(N_SOURCE_BF // n_prime):
        sel = (p > b * n_prime) & (p <= (b + 1) * n_prime)
        dts = same_parent_pair_dts(t[sel], p[sel], hi)
        if ((dts >= lo) & (dts <= hi)).sum() >= MIN_PAIRS_FIT:
            a, _, _ = fit_alpha_mle(dts, lo, hi, background="auto")
            if np.isfinite(a):
                out.append(a)
    return np.array(out)


def acdc_alpha_vs_budget(events_path, F_flat: np.ndarray, n_source_cell: int,
                         fit_window, budgets=None, n_cells: int = 80) -> pd.DataFrame:
    """MEASURED ACDC sigma_alpha vs total-history budget, by chain-subsampling
    the per-cell event sweep (no 1/sqrt(N) assumption, no Poisson approximation).

    Budget B -> B/n_cells histories per cell -> keep parent_id <= that per cell.
    Disjoint per-cell blocks give independent realizations; sigma across blocks.
    Returns columns n_prime (total histories), alpha, sigma, n_blocks, censored.
    """
    lo, hi = fit_window
    events, n_src = load_sweep_events(events_path)
    if budgets is None:
        budgets = [n_cells * n for n in (1250, 3125, 6250, 12500)]
    time_edges = np.linspace(0.0, T_MAX, N_T + 1)
    rows = []
    for B in budgets:
        n_per = B // n_cells
        n_blocks = max(n_src // n_per, 1)
        alphas = []
        for b in range(n_blocks):
            ev_b = {}
            for idx, (t, p) in events.items():
                m = (p > b * n_per) & (p <= (b + 1) * n_per)
                if m.any():
                    ev_b[idx] = (t[m], p[m])
            tau, R = acdc_rossi_events(ev_b, F_flat, n_per, time_edges)
            w = (tau >= lo) & (tau <= hi)
            if np.any(R[w] > 0):
                try:
                    popt, _ = fit_alpha_binned(tau[w], R[w])
                    if np.isfinite(popt[2]) and popt[2] < 0:
                        alphas.append(popt[2])
                except RuntimeError:
                    pass
        if len(alphas) >= 3:
            rows.append({"n_prime": B, "alpha": float(np.mean(alphas)),
                         "sigma": float(np.std(alphas, ddof=1)), "n_blocks": len(alphas),
                         "censored": False})
        elif alphas:
            rows.append({"n_prime": B, "alpha": float(np.mean(alphas)), "sigma": np.nan,
                         "n_blocks": len(alphas), "censored": False})
        else:
            rows.append({"n_prime": B, "alpha": np.nan, "sigma": np.nan,
                         "n_blocks": 0, "censored": True})
    return pd.DataFrame(rows)


def acdc_alpha_sigma(label: str, n_boot: int = 100, rng=None, fit_window=None):
    """ACDC alpha at full sweep statistics with chain-aware sigma.

    Parametric bootstrap: resample every g_t bin as Poisson, recompute R(tau),
    refit; multiply the resulting sigma by KAPPA_S[label] (Phase-2 validated
    over-dispersion of the chain structure that bin-level Poisson misses).
    Returns (alpha, sigma, sweep_total_histories).
    """
    rng = np.random.default_rng(rng)
    sweep = gl.load_sweep(label)
    n_r, n_th = sweep["G"].shape
    F = fission_coarse_sim(label, n_r, n_th)
    lo, hi = fit_window or FIT_WINDOWS[label]

    def fit_one(s):
        tau, R = acdc_rossi(s, F)
        w = (tau >= lo) & (tau <= hi)
        popt, _ = fit_alpha_binned(tau[w], R[w])
        return popt[2]

    alpha = fit_one(sweep)
    g_t = sweep["g_t"]
    boots = []
    for _ in range(n_boot):
        s2 = dict(sweep)
        s2["g_t"] = rng.poisson(g_t).astype(float)
        boots.append(fit_one(s2))
    sigma = float(np.std(boots)) * KAPPA_S[label]
    total_hist = int(n_r * n_th * float(sweep["n_source"]))
    return float(alpha), sigma, total_hist


# --------------------------------------------------------------------------- #
# Experiment-realistic BF: all pairs, no parent_id oracle
# --------------------------------------------------------------------------- #

def all_pairs_dts(sorted_times: np.ndarray, dt_max: float) -> np.ndarray:
    """All pair separations <= dt_max from a SORTED timestamp stream (no chain
    information used — this is what a real detector sees)."""
    t = np.asarray(sorted_times, float)
    dts = []
    d = 1
    while d < len(t):
        dt = t[d:] - t[:-d]
        m = dt <= dt_max
        if not m.any():
            break
        dts.append(dt[m])
        d += 1
    return np.concatenate(dts) if dts else np.array([])


def optimal_source_rate(df: pd.DataFrame, n_source: int, fit_window) -> float:
    """Source rate S* [n/s] at which accidental ~= correlated pairs in the window.

    For a steady-source experiment of N histories at rate S (duration T = N/S):
    correlated window pairs = N*c1 (c1 = window pairs per source), accidentals
    ~= N * eps^2 * w * S (w = window width). Equality gives S* = c1/(eps^2 w) —
    the classic Rossi-alpha operating point balancing signal against background.
    """
    lo, hi = fit_window
    dts = same_parent_pair_dts(df["time"].to_numpy(), df["parent_id"].to_numpy(), hi)
    c1 = ((dts >= lo) & (dts <= hi)).sum() / n_source
    eps = len(df) / n_source
    return float(c1 / (eps**2 * (hi - lo)))


def bf_realistic_alpha_sigma(df: pd.DataFrame, n_source: int, fit_window,
                             source_rate: float | None = None,
                             n_boot: int = 60, rng=None) -> dict:
    """BF WITHOUT the parent_id oracle: the experiment-realistic estimator.

    The pulsed dataset is converted to a steady-source experiment by shifting
    each chain by an iid U[0, T) start time (T = n_source / source_rate) —
    exactly the ensemble a real measurement at that rate produces; parent_id is
    used only for this dataset construction, never by the estimator. The
    estimator then histograms ALL pair separations (correlated + accidental)
    and fits flat background + truncated exponential by unbinned MLE.

    source_rate defaults to the per-config optimum (accidental ~= correlated).
    sigma via chain bootstrap (resample chains, re-shift, re-count, re-fit).
    """
    rng = np.random.default_rng(rng)
    lo, hi = fit_window
    if source_rate is None:
        source_rate = optimal_source_rate(df, n_source, fit_window)
    t_window = n_source / source_rate

    def estimate(d):
        sh = stationarize(d, t_window, rng)
        dts = all_pairs_dts(np.sort(sh["time"].to_numpy()), hi)
        a, s, n = fit_alpha_mle(dts, lo, hi, background="flat")
        return a, n

    alpha, n_pairs = estimate(df)
    if n_pairs < MIN_PAIRS_FIT or not np.isfinite(alpha):
        return {"alpha": np.nan, "sigma": np.nan, "n_pairs": int(n_pairs),
                "source_rate": source_rate, "censored": True}

    # chain bootstrap (independent of the estimator's oracle-freeness: it only
    # quantifies sampling variability of the underlying chain population)
    order = np.argsort(df["parent_id"].to_numpy(), kind="stable")
    t_all = df["time"].to_numpy()[order]
    p_all = df["parent_id"].to_numpy()[order]
    uniq, starts = np.unique(p_all, return_index=True)
    bounds = np.append(starts, len(p_all))
    boots = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        ts, ps = [], []
        for j, g in enumerate(pick):
            s0, e0 = bounds[g], bounds[g + 1]
            ts.append(t_all[s0:e0])
            ps.append(np.full(e0 - s0, j))
        a, _ = estimate(pd.DataFrame({
            "time": np.concatenate(ts), "parent_id": np.concatenate(ps)}))
        boots.append(a)
    boots = np.array(boots)
    boots = boots[np.isfinite(boots)]
    return {"alpha": float(alpha),
            "sigma": float(np.std(boots)) if len(boots) >= 10 else np.nan,
            "n_pairs": int(n_pairs), "source_rate": float(source_rate),
            "censored": False}


# --------------------------------------------------------------------------- #
# Pulsed die-away (singles) method
# --------------------------------------------------------------------------- #

def die_away_alpha_sigma(df: pd.DataFrame, fit_window, n_boot: int = 60, rng=None) -> dict:
    """Pulsed die-away: alpha from the SINGLES arrival-time envelope after a pulse.

    Fits the detection-time distribution in the late window with flat background
    (the delayed-neutron pedestal) + truncated exponential — no pairs involved,
    so it works at every k from modest statistics (e.g. 8.6k singles at k=0.40,
    where the pair methods see 5 usable pairs).

    *** NOT PRACTICAL IN MOST REAL EXPERIMENTS. *** It requires a PULSABLE
    source: the envelope exists only because all chains start together. With the
    usual steady source the singles rate is just a constant background — there
    is no die-away curve to fit, and alpha is only accessible through the
    correlation methods (Rossi doubles, noise analysis) or model-based ACDC.
    It is included here because (a) the simulation's t=0 pulse provides the
    envelope for free, and (b) the no-oracle all-pairs fit on the raw pulsed
    stream silently degenerates into exactly this measurement (signal fraction
    8e-5 — its "pairs" are the singles envelope autocorrelated with itself), so
    it deserves to be benchmarked under its honest name.

    sigma via chain bootstrap (detection times within a chain are correlated).
    """
    rng = np.random.default_rng(rng)
    lo, hi = fit_window
    t_all = df["time"].to_numpy()
    alpha, _, n_in = fit_alpha_mle(t_all, lo, hi, background="flat")

    order = np.argsort(df["parent_id"].to_numpy(), kind="stable")
    t_s = t_all[order]
    p_s = df["parent_id"].to_numpy()[order]
    uniq, starts = np.unique(p_s, return_index=True)
    bounds = np.append(starts, len(p_s))
    boots = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        tb = np.concatenate([t_s[bounds[g]:bounds[g + 1]] for g in pick])
        a, _, _ = fit_alpha_mle(tb, lo, hi, background="flat")
        boots.append(a)
    boots = np.array(boots)
    boots = boots[np.isfinite(boots)]
    return {"alpha": float(alpha), "sigma": float(np.std(boots)),
            "n_singles": int(n_in),
            "censored": n_in < MIN_PAIRS_FIT or not np.isfinite(alpha)}


# --------------------------------------------------------------------------- #
# G3 machinery: precision vs detector distance (event-level, k=0.90 scan)
# --------------------------------------------------------------------------- #

def fission_coarse_from_dir(res_dir: Path | str, n_r: int, n_th: int,
                            moment: float = M_SIM) -> np.ndarray:
    """fission_coarse for an explicit results dir (the gap-scan globals)."""
    import openmc

    from project.core.rossi_mesh import N_RADIAL, N_THETA
    with openmc.StatePoint(str(Path(res_dir) / "statepoint.1.h5")) as sp:
        F_tally = sp.get_tally(name="fission_per_cell").mean.flatten()
    F_fine = F_tally.reshape(N_THETA, N_RADIAL)
    kr, kth = N_RADIAL // n_r, N_THETA // n_th
    F_blk = F_fine.reshape(n_th, kth, n_r, kr).sum(axis=(1, 3))
    return moment * F_blk.T


def load_chain_events_from(track_path: Path | str, prompt_only: bool = True) -> pd.DataFrame:
    """Detector events (time, parent_id) from an explicit collision_track.h5."""
    df = load_collision_tracks(track_path)
    sel = df["cell_id"] == DETECTOR_CELL_ID
    if prompt_only:
        sel &= df["delayed_group"] == 0
    return df.loc[sel, ["time", "parent_id"]].reset_index(drop=True)


def bf_alpha_sigma_events(df: pd.DataFrame, fit_window, n_boot: int = 100, rng=None):
    """BF alpha +/- sigma (chain bootstrap) on one dataset; censored dict if the
    window pair count is below MIN_PAIRS_FIT."""
    lo, hi = fit_window
    dts = same_parent_pair_dts(df["time"].to_numpy(), df["parent_id"].to_numpy(), hi)
    n_pairs = int(((dts >= lo) & (dts <= hi)).sum())
    if n_pairs < MIN_PAIRS_FIT:
        return {"alpha": np.nan, "sigma": np.nan, "n_pairs": n_pairs, "censored": True}
    alpha, _, _ = fit_alpha_mle(dts, lo, hi, background="auto")
    boots = bootstrap_chains(
        df, lambda d: fit_alpha_mle(
            same_parent_pair_dts(d["time"].to_numpy(), d["parent_id"].to_numpy(), hi),
            lo, hi, background="auto")[0],
        n_boot=n_boot, rng=rng,
    )
    boots = boots[np.isfinite(boots)]
    return {"alpha": float(alpha), "sigma": float(np.std(boots)),
            "n_pairs": n_pairs, "censored": False}


def acdc_alpha_sigma_events(events: dict, n_source_cell: int, F_flat: np.ndarray,
                            fit_window, n_boot: int = 100, rng=None):
    """Event-level ACDC alpha +/- sigma: cross-chain (U-statistic) estimator for
    the mean, chain bootstrap (per cell, including undetected chains) for sigma.
    No kappa needed — the chain structure is resampled directly.
    """
    rng = np.random.default_rng(rng)
    lo, hi = fit_window
    time_edges = np.linspace(0.0, T_MAX, N_T + 1)

    def fit_from(ev):
        tau, R = acdc_rossi_events(ev, F_flat, n_source_cell, time_edges)
        w = (tau >= lo) & (tau <= hi)
        if not np.any(R[w] > 0):
            return np.nan
        try:
            popt, _ = fit_alpha_binned(tau[w], R[w])
            return popt[2]
        except RuntimeError:
            return np.nan

    # events keyed by (ir, ith); the 2-D F array indexes directly with that tuple
    flat = events
    alpha = fit_from(flat)

    # chain bootstrap per cell
    prepared = {}
    for idx, (t, p) in flat.items():
        uniq, inv = np.unique(p, return_inverse=True)
        prepared[idx] = (t, inv, len(uniq))
    boots = []
    for _ in range(n_boot):
        ev_b = {}
        for idx, (t, inv, n_chains) in prepared.items():
            if n_chains == 0:
                continue
            k_det = rng.binomial(n_source_cell, n_chains / n_source_cell)
            pick = rng.integers(0, n_chains, k_det)
            ts, ps = [], []
            for j, g in enumerate(pick):
                sel = inv == g
                ts.append(t[sel])
                ps.append(np.full(sel.sum(), j))
            if ts:
                ev_b[idx] = (np.concatenate(ts), np.concatenate(ps))
        boots.append(fit_from(ev_b))
    boots = np.array(boots)
    boots = boots[np.isfinite(boots)]
    sigma = float(np.std(boots)) if len(boots) >= 10 else np.nan
    return {"alpha": float(alpha) if np.isfinite(alpha) else np.nan,
            "sigma": sigma, "n_boot_ok": int(len(boots))}


# --------------------------------------------------------------------------- #
# Uniform per-k assembly for the methods-vs-k map (all methods at 1M)
# --------------------------------------------------------------------------- #

RESULTS_ROOT = gl.RESULTS_ROOT


def _ifp_pk_alpha(label: str):
    """(measured_k, alpha_PK UFloat) from the IFP run for this label."""
    import uncertainties as unc

    d = np.load(RESULTS_ROOT / f"ifp-kinetics-k{label}" / "ifp.npz")
    k = float(d["keff"])
    a = pk_alpha(k,
                 unc.ufloat(float(d["beta_eff"]), float(d["beta_eff_std"])),
                 unc.ufloat(float(d["gen_time"]), float(d["gen_time_std"])),
                 convention="generation")
    return k, a


def benchmark_row(label: str, n_boot: int = 60, rng=None) -> dict:
    """All five estimators' alpha +/- sigma at a uniform 1M-history budget for k=label.

    BF (oracle + realistic) and die-away from the 1M global collision track;
    ACDC from the 1M bench sweep (greens-sweep-k<label>-bench1M*, event-level
    chain bootstrap). PK from the IFP run (zero statistical error, fixed bias).
    Returns a flat dict ready for a DataFrame row.
    """
    rng = np.random.default_rng(rng)
    win = FIT_WINDOWS[label]
    meas_k, a_pk = _ifp_pk_alpha(label)

    df = load_chain_events(label)                       # 1M global, prompt detector events
    oracle = bf_alpha_sigma_events(df, win, n_boot=n_boot, rng=rng)
    realistic = bf_realistic_alpha_sigma(df, N_SOURCE_BF, win, n_boot=n_boot, rng=rng)
    die = die_away_alpha_sigma(df, win, n_boot=n_boot, rng=rng)

    # ACDC at 1M from the bench sweep events + this geometry's F_i
    ev_path = RESULTS_ROOT / f"greens-sweep-k{label}-bench1M-events.h5"
    events, n_cell = load_sweep_events(ev_path)
    F = fission_coarse_sim(label, 8, 10)
    acdc = acdc_alpha_sigma_events(events, n_cell, F, win, n_boot=n_boot, rng=rng)

    return {
        "label": label, "k": meas_k,
        "pk": a_pk.nominal_value, "pk_err": a_pk.std_dev,
        "acdc": acdc["alpha"], "acdc_err": acdc["sigma"],
        "bf_oracle": oracle["alpha"], "bf_oracle_err": oracle["sigma"],
        "bf_oracle_cens": oracle["censored"],
        "bf_real": realistic["alpha"], "bf_real_err": realistic["sigma"],
        "bf_real_cens": realistic["censored"],
        "die": die["alpha"], "die_err": die["sigma"],
    }


# --------------------------------------------------------------------------- #
# Phase-0 acceptance gate
# --------------------------------------------------------------------------- #

SWEEP_GRID = (20, 20)       # the existing greens-sweep-k*.npz grids
N_SOURCE_BF = 1_000_000     # histories in the existing global runs

# alpha is defined as the LATE-TIME (fundamental-mode) slope: R(tau) is
# multi-exponential (early lags carry higher spatial modes; measured early/late
# slopes at k=0.98: ~-550 vs ~-370), so the fit window must be identical for
# every method or the comparison is meaningless. Windows ~ [1, 4] late decay
# times, frozen here and reused in G1/G2/G3.
FIT_WINDOWS = {
    "098": (2.7e-3, 1.0e-2),
    "095": (1.7e-3, 6.7e-3),
    "090": (7.0e-4, 2.8e-3),
    "080": (4.6e-4, 1.8e-3),
    "070": (2.4e-4, 1.0e-3),
    "060": (2.2e-4, 8.6e-4),
    "050": (1.7e-4, 6.5e-4),
    "040": (1.1e-4, 4.4e-4),
}


def alignment_gate(label: str, verbose: bool = True) -> dict:
    """BF-vs-ACDC absolute overlay (with M_SIM), shape flatness, and same-window
    alpha agreement at full statistics."""
    sweep = gl.load_sweep(label)
    n_r, n_th = sweep["G"].shape
    F = fission_coarse_sim(label, n_r, n_th)
    tau, R_acdc = acdc_rossi(sweep, F)

    df = load_chain_events(label)
    tau_bf, R_bf, counts = bf_rossi(df, N_SOURCE_BF)
    assert np.allclose(tau, tau_bf)

    # shape test over the full usable range (skip lag bins 0-1), scale test = sum ratio
    full = (tau >= 2 * DT) & (R_acdc > 0)
    scale = R_bf[full].sum() / R_acdc[full].sum()
    M_emp, M_err = calibrate_moment(label, n_boot=200)

    lo, hi = FIT_WINDOWS[label]
    dts = same_parent_pair_dts(df["time"].to_numpy(), df["parent_id"].to_numpy(), T_MAX)
    a_bf, s_bf, n_pairs = fit_alpha_mle(dts, lo, hi, background="auto")
    fitw = (tau >= lo) & (tau <= hi)
    popt, perr = fit_alpha_binned(tau[fitw], R_acdc[fitw])
    a_ac, s_ac = popt[2], perr[2]

    agree = abs(a_bf - a_ac) / np.hypot(s_bf, s_ac) if np.isfinite(s_bf) else np.nan
    result = {
        "label": label, "n_pairs_window": int(n_pairs),
        "scale_bf_over_acdc": float(scale),
        "M_emp": M_emp, "M_err": M_err,
        "alpha_bf": a_bf, "alpha_bf_err": s_bf,
        "alpha_acdc": float(a_ac), "alpha_acdc_err": float(s_ac),
        "alpha_nsigma": float(agree),
        "tau": tau, "R_acdc": R_acdc, "R_bf": R_bf, "counts_bf": counts,
    }
    if verbose:
        print(f"k{label}: window pairs={n_pairs:6d}  scale={scale:5.3f}  "
              f"M_emp={M_emp:4.2f}+/-{M_err:4.2f}  "
              f"alpha BF={a_bf:9.1f}+/-{s_bf:6.1f}  ACDC={a_ac:9.1f}+/-{s_ac:6.1f}"
              + (f"  ({agree:.2f} sigma)" if np.isfinite(agree) else "  (no BF fit)"))
    return result


# --------------------------------------------------------------------------- #
# COST OF PRECISION: unified across-realization sigma(N), 1/sqrt(N) law, and the
# histories / wall-time each method needs to reach a target precision.
#
# One convergence metric for every method: the std of alpha across DISJOINT,
# equal-size realizations (sphere -> disjoint parent_id ranges; sweep -> disjoint
# parent_id ranges per cell, assembled across cells). No KAPPA_S fudge, no Poisson
# approximation -- chain bootstrap and the MLE Fisher error are kept only as
# cross-checks (sigma_crosscheck). sigma(N) vs N is the master curve: read it at
# sigma = target*|alpha_ref| for histories-to-target; the gap between two methods
# is the efficiency ratio.
# --------------------------------------------------------------------------- #

# Flagship deep-dive case (matched 10M-history sphere + per-cell sweep).
FLAGSHIP_LABEL = "090"
FLAGSHIP_DOUBLES_TRACK = RESULTS_ROOT / "greens-function-k090-bench10M" / "collision_track.h5"
FLAGSHIP_SWEEP_EVENTS = RESULTS_ROOT / "greens-sweep-k090-bench10M-events.h5"
FLAGSHIP_SWEEP_NPZ = RESULTS_ROOT / "greens-sweep-k090-bench10M.npz"

# Method display names (transparent "Doubles" terminology, per user).
METHOD_LABELS = {
    "oracle": "Doubles (parent-ID)",
    "realistic": "Doubles (realistic)",
    "die": "die-away*",
    "acdc": "ACDC",
    "pk": "point kinetics",
}


def budget_ladder(n_total: int, n_min: float = 3e4, n_points: int = 10,
                  min_blocks: int = 3) -> np.ndarray:
    """Log-spaced total-history budgets from n_min up to n_total // min_blocks
    (so every budget still affords >= min_blocks disjoint realizations)."""
    n_max = max(int(n_total // min_blocks), int(n_min))
    grid = np.logspace(np.log10(n_min), np.log10(n_max), n_points)
    return np.unique(np.round(grid).astype(int))


def _disjoint_blocks(parent_ids: np.ndarray, n_prime: int, n_total: int,
                     max_blocks: int | None = None):
    """Boolean masks for disjoint parent-id blocks [b*n', (b+1)*n'] of size n'.

    max_blocks caps how many realizations are drawn: a stable across-realization
    std needs only ~30 blocks, so capping bounds notebook runtime without biasing
    sigma (the drawn blocks are still iid). n_blocks (total available) is reported
    separately by sigma_vs_budget.
    """
    n_blocks = int(n_total // n_prime)
    if max_blocks is not None:
        n_blocks = min(n_blocks, max_blocks)
    for b in range(n_blocks):
        yield (parent_ids > b * n_prime) & (parent_ids <= (b + 1) * n_prime)


def oracle_block_alphas(df: pd.DataFrame, n_prime: int, n_total: int, fit_window,
                        max_blocks: int | None = None):
    """Per-realization Doubles (parent-ID) alpha at budget n_prime + window-pair
    counts. Blocks below MIN_PAIRS_FIT window pairs contribute no alpha (the
    starvation that IS the result)."""
    lo, hi = fit_window
    t, p = df["time"].to_numpy(), df["parent_id"].to_numpy()
    alphas, counts = [], []
    for sel in _disjoint_blocks(p, n_prime, n_total, max_blocks):
        dts = same_parent_pair_dts(t[sel], p[sel], hi)
        dts = dts[(dts >= lo) & (dts <= hi)]
        counts.append(len(dts))
        if len(dts) >= MIN_PAIRS_FIT:
            a, _, _ = fit_alpha_mle(dts, lo, hi, background="auto")
            if np.isfinite(a) and a < 0:
                alphas.append(a)
    return np.array(alphas), np.array(counts)


def realistic_block_alphas(df: pd.DataFrame, n_prime: int, n_total: int,
                           fit_window, rng=None, max_blocks: int | None = None):
    """Per-realization Doubles (realistic) alpha: each block stationarized to its
    own optimal steady-source rate, all-pairs + flat-background MLE (no oracle)."""
    rng = np.random.default_rng(rng)
    lo, hi = fit_window
    t, p = df["time"].to_numpy(), df["parent_id"].to_numpy()
    alphas, counts = [], []
    for sel in _disjoint_blocks(p, n_prime, n_total, max_blocks):
        sub = pd.DataFrame({"time": t[sel], "parent_id": p[sel]})
        if len(sub) < 2:
            counts.append(len(sub))
            continue
        sr = optimal_source_rate(sub, n_prime, fit_window)
        if not np.isfinite(sr) or sr <= 0:   # no window pairs -> no realistic measurement
            counts.append(0)
            continue
        sh = stationarize(sub, n_prime / sr, rng)
        dts = all_pairs_dts(np.sort(sh["time"].to_numpy()), hi)
        a, _, n = fit_alpha_mle(dts, lo, hi, background="flat")
        counts.append(int(n))
        if n >= MIN_PAIRS_FIT and np.isfinite(a) and a < 0:
            alphas.append(a)
    return np.array(alphas), np.array(counts)


def die_block_alphas(df: pd.DataFrame, n_prime: int, n_total: int, fit_window,
                     max_blocks: int | None = None):
    """Per-realization die-away (singles envelope) alpha + window-singles counts."""
    lo, hi = fit_window
    t, p = df["time"].to_numpy(), df["parent_id"].to_numpy()
    alphas, counts = [], []
    for sel in _disjoint_blocks(p, n_prime, n_total, max_blocks):
        ts = t[sel]
        a, _, n = fit_alpha_mle(ts, lo, hi, background="flat")
        counts.append(int(((ts >= lo) & (ts <= hi)).sum()))
        if np.isfinite(a) and a < 0:
            alphas.append(a)
    return np.array(alphas), np.array(counts)


def acdc_block_alphas(events: dict, F_flat: np.ndarray, n_prime: int,
                      n_source_cell: int, fit_window, n_cells: int = 80,
                      max_blocks: int | None = None):
    """Per-realization ACDC alpha at TOTAL budget n_prime: n_prime // n_cells
    histories per cell, disjoint per-cell parent-id blocks assembled across cells,
    cross-chain (U-statistic) R(tau) fit. Returns alphas + per-block event counts."""
    lo, hi = fit_window
    n_per = max(int(n_prime // n_cells), 1)
    time_edges = np.linspace(0.0, T_MAX, N_T + 1)
    alphas, counts = [], []
    n_blocks = int(n_source_cell // n_per)
    if max_blocks is not None:
        n_blocks = min(n_blocks, max_blocks)
    for b in range(n_blocks):
        ev_b, cnt = {}, 0
        for idx, (t, p) in events.items():
            m = (p > b * n_per) & (p <= (b + 1) * n_per)
            if m.any():
                ev_b[idx] = (t[m], p[m])
                cnt += int(m.sum())
        counts.append(cnt)
        tau, R = acdc_rossi_events(ev_b, F_flat, n_per, time_edges)
        w = (tau >= lo) & (tau <= hi)
        if np.any(R[w] > 0):
            try:
                popt, _ = fit_alpha_binned(tau[w], R[w])
                if np.isfinite(popt[2]) and popt[2] < 0:
                    alphas.append(popt[2])
            except RuntimeError:
                pass
    return np.array(alphas), np.array(counts)


def sigma_vs_budget(block_fn, budgets, n_total: int,
                    min_blocks_for_sigma: int = 3) -> pd.DataFrame:
    """Drive any *_block_alphas over a budget ladder -> across-realization sigma.

    block_fn(n_prime) -> (alphas, counts). sigma = std of the per-realization
    alphas (needs >= min_blocks_for_sigma convergent blocks, else censored).
    Columns: n_prime, alpha, sigma, n_blocks, n_convergent, mean_count, censored.
    """
    rows = []
    for n_prime in budgets:
        alphas, counts = block_fn(int(n_prime))
        n_conv = len(alphas)
        rows.append({
            "n_prime": int(n_prime),
            "alpha": float(np.mean(alphas)) if n_conv else np.nan,
            "sigma": float(np.std(alphas, ddof=1)) if n_conv >= min_blocks_for_sigma else np.nan,
            "n_blocks": int(n_total // n_prime),
            "n_convergent": n_conv,
            "mean_count": float(np.mean(counts)) if len(counts) else np.nan,
            "censored": n_conv < min_blocks_for_sigma,
        })
    return pd.DataFrame(rows)


def fit_sigma_scaling(budget_df: pd.DataFrame, tol: float = 0.15) -> dict:
    """Fit log(sigma) = intercept + slope*log(N) on the measured rows.

    slope_ok gates extrapolation: the 1/sqrt(N) law requires slope ~ -0.5
    (|slope + 0.5| < tol). Returns slope, intercept, slope_ok, n_points.
    """
    d = budget_df.dropna(subset=["sigma"])
    d = d[d["sigma"] > 0]
    if len(d) < 2:
        return {"slope": np.nan, "intercept": np.nan, "slope_ok": False,
                "n_points": int(len(d))}
    x = np.log(d["n_prime"].to_numpy(float))
    y = np.log(d["sigma"].to_numpy(float))
    slope, intercept = np.polyfit(x, y, 1)
    return {"slope": float(slope), "intercept": float(intercept),
            "slope_ok": bool(abs(slope + 0.5) < tol), "n_points": int(len(d))}


def histories_to_target(budget_df: pd.DataFrame, alpha_ref: float,
                        target: float = 0.05, scaling: dict | None = None,
                        min_anchor_blocks: int = 6) -> dict:
    """Histories to reach sigma_alpha = target*|alpha_ref|.

    Fits the LEVEL of the 1/sqrt(N) law with the slope FIXED at -1/2 through the
    reliably-measured budgets (>= min_anchor_blocks convergent realizations), then
    solves for N at the target. Slope -1/2 is the CLT for iid history blocks
    (disjoint parent_id ranges are independent by construction); it is validated
    empirically on the flagship deep-dive (fit_sigma_scaling there gives ~-0.5) --
    re-measuring it at every breadth/distance point is infeasible at 10M because
    the reliable-block window is too narrow, but it cannot break for iid samples.

    'no result' (ok=False) is reserved for genuine pair STARVATION: no budget
    reaches MIN_PAIRS_FIT pairs in >= 3 blocks, so there is no anchor at all.
    scaling (the free-slope fit) is returned as a diagnostic, not a gate.
    """
    if scaling is None:
        scaling = fit_sigma_scaling(budget_df)
    sig_target = target * abs(alpha_ref)
    d = budget_df.dropna(subset=["sigma"])
    d = d[(d["sigma"] > 0) & (~d["censored"])]
    if len(d) == 0:
        return {"n_target": np.nan, "extrapolated": True, "ok": False,
                "anchor_N": np.nan, "anchor_sigma": np.nan, "n_anchor_pts": 0,
                "slope": scaling["slope"]}
    reliable = d[d["n_convergent"] >= min_anchor_blocks]
    pool = reliable if len(reliable) else d
    # anchor on the most-converged HIGH-budget reliable point: it sits on the
    # asymptotic 1/sqrt(N) line, whereas low-budget sigma is biased high
    # (pre-asymptotic + cross-chain bias at sparse counts). Slide it down at -1/2.
    anchor = pool.sort_values("n_prime").iloc[-1]
    N0, s0 = float(anchor["n_prime"]), float(anchor["sigma"])
    n_target = N0 * (s0 / sig_target) ** 2
    return {"n_target": float(n_target),
            "extrapolated": bool(n_target > float(d["n_prime"].max())),
            "ok": True, "anchor_N": N0, "anchor_sigma": s0,
            "n_anchor_pts": int(len(pool)), "slope": scaling["slope"]}


def histories_for_sigma(sigma_at: float, n_at: float, alpha_ref: float,
                        target: float = 0.05) -> float:
    """Histories to reach sigma = target*|alpha_ref|, sliding a SINGLE measured
    (sigma_at, n_at) point down the 1/sqrt(N) law. Used for the breadth and
    distance scans, where a full-budget chain-bootstrap sigma is more stable than
    a subsampled across-realization curve; the law itself is validated on the
    flagship master curve. Returns nan for a non-finite/non-positive input sigma
    (i.e. a censored / pair-starved measurement -> 'no result')."""
    if not np.isfinite(sigma_at) or sigma_at <= 0:
        return np.nan
    return float(n_at * (sigma_at / (target * abs(alpha_ref))) ** 2)


def wall_time_to_target(n_target: float, seconds_per_history: float) -> float:
    """Wall-clock seconds to reach the target, given the method's measured
    per-history transport cost (the honest currency: per-cell chains differ in
    length by birth site, so histories alone understate ACDC's edge-cell cheapness)."""
    return n_target * seconds_per_history


def fom(sigma: float, wall_time: float) -> float:
    """Monte-Carlo figure of merit 1/(sigma^2 * T): higher = more efficient."""
    return 1.0 / (sigma ** 2 * wall_time)


def efficiency_ratio(budget_df_a: pd.DataFrame, budget_df_b: pd.DataFrame,
                     alpha_ref: float, target: float = 0.05) -> float:
    """N_a / N_b to reach equal precision ('a needs this many times more histories
    than b'). nan if either method's target is not reachable under the policy."""
    na = histories_to_target(budget_df_a, alpha_ref, target)
    nb = histories_to_target(budget_df_b, alpha_ref, target)
    if not (na["ok"] and nb["ok"]) or not np.isfinite(nb["n_target"]) or nb["n_target"] <= 0:
        return np.nan
    return na["n_target"] / nb["n_target"]


def seconds_per_history_sphere(track_dir: Path | str, n_total: int) -> float:
    """Transport seconds per source history for a whole-sphere collision-track run."""
    import openmc

    with openmc.StatePoint(str(Path(track_dir) / "statepoint.1.h5")) as sp:
        return float(sp.runtime["transport"]) / float(n_total)


def seconds_per_history_sweep(npz_path: Path | str) -> float:
    """Transport seconds per source history for a per-cell sweep (total transport
    over all cells / total histories = per-cell N x n_cells)."""
    d = np.load(npz_path)
    n_cells = int(d["transport_s"].size)
    return float(d["transport_s"].sum()) / (float(d["n_source"]) * n_cells)


def reference_alpha(label: str, events_path: Path | str | None = None) -> float:
    """Common high-statistics consensus |alpha| denominator: the full-statistics
    ACDC alpha for this label (most stable; methods agree where measurable)."""
    if events_path is None:
        events_path = RESULTS_ROOT / f"greens-sweep-k{label}-bench1M-events.h5"
    events, n_cell = load_sweep_events(events_path)
    F = fission_coarse_sim(label, 8, 10)
    res = acdc_alpha_sigma_events(events, n_cell, F, FIT_WINDOWS[label], n_boot=1)
    return float(res["alpha"])


def sigma_crosscheck(df: pd.DataFrame, n_prime: int, n_total: int, fit_window,
                     n_boot: int = 100, rng=None) -> dict:
    """Three sigma estimators for the Doubles (parent-ID) method at one budget, to
    validate that the cheap single-run estimators track the across-realization std:
      across : std of per-realization alpha over disjoint blocks (headline)
      boot   : chain bootstrap of one block
      fisher : mean MLE asymptotic (observed-Fisher) sigma over blocks.
    """
    lo, hi = fit_window
    t, p = df["time"].to_numpy(), df["parent_id"].to_numpy()
    block_alphas, fishers = [], []
    first_block = None
    for sel in _disjoint_blocks(p, n_prime, n_total):
        dts = same_parent_pair_dts(t[sel], p[sel], hi)
        dts = dts[(dts >= lo) & (dts <= hi)]
        if len(dts) >= MIN_PAIRS_FIT:
            a, s, _ = fit_alpha_mle(dts, lo, hi, background="auto")
            if np.isfinite(a):
                block_alphas.append(a)
                if np.isfinite(s):
                    fishers.append(s)
            if first_block is None:
                first_block = pd.DataFrame({"time": t[sel], "parent_id": p[sel]})
    across = float(np.std(block_alphas, ddof=1)) if len(block_alphas) >= 3 else np.nan
    boot = np.nan
    if first_block is not None:
        b = bootstrap_chains(
            first_block, lambda d: fit_alpha_mle(
                same_parent_pair_dts(d["time"].to_numpy(), d["parent_id"].to_numpy(), hi),
                lo, hi, background="auto")[0],
            n_boot=n_boot, rng=rng)
        b = b[np.isfinite(b)]
        boot = float(np.std(b)) if len(b) >= 10 else np.nan
    return {"n_prime": int(n_prime), "across": across, "boot": boot,
            "fisher": float(np.mean(fishers)) if fishers else np.nan,
            "n_blocks_used": len(block_alphas)}


def doubles_window_pairs(df: pd.DataFrame, fit_window, n_subset: int | None = None) -> int:
    """Same-chain detection pairs in the alpha fit window (the quantity that
    starves: a Doubles measurement is impossible below MIN_PAIRS_FIT)."""
    lo, hi = fit_window
    t, p = df["time"].to_numpy(), df["parent_id"].to_numpy()
    if n_subset is not None:
        sel = p <= n_subset
        t, p = t[sel], p[sel]
    dts = same_parent_pair_dts(t, p, hi)
    return int(((dts >= lo) & (dts <= hi)).sum())


def doubles_count_vs_k(labels, n_subset: int = N_SOURCE_BF) -> pd.DataFrame:
    """Window same-chain pair count per k (fixed budget n_subset) for the failure
    panel: where pairs < MIN_PAIRS_FIT the Doubles method has no result."""
    rows = []
    for L in labels:
        df = load_chain_events(L)
        rows.append({"label": L, "k": gl.K_OF_LABEL[L],
                     "pairs": doubles_window_pairs(df, FIT_WINDOWS[L], n_subset),
                     "n_subset": n_subset})
    return pd.DataFrame(rows)


def doubles_count_vs_gap(gaps, label: str = "090", n_subset: int = N_SOURCE_BF) -> pd.DataFrame:
    """Window same-chain pair count vs detector gap at fixed k (default 090)."""
    rows = []
    for g in gaps:
        suffix = "" if g == 10 else f"-gap{g:g}"
        track = RESULTS_ROOT / f"greens-function-k{label}{suffix}" / "collision_track.h5"
        if not track.exists():
            rows.append({"gap": g, "pairs": np.nan, "missing": True})
            continue
        df = load_chain_events_from(track)
        rows.append({"gap": g,
                     "pairs": doubles_window_pairs(df, FIT_WINDOWS[label], n_subset),
                     "missing": False})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys
    labels = sys.argv[1:] or ["098", "090", "070", "040"]
    print(f"Phase-0 alignment gate (M_SIM={M_SIM}): absolute overlay + late-window alpha")
    results = [alignment_gate(lbl) for lbl in labels]
    checks = []
    for r in results:
        # moment consistency with the pooled constant
        checks.append(abs(r["M_emp"] - M_SIM) < 3 * max(r["M_err"], 1e-9))
        if r["n_pairs_window"] >= 50:
            checks.append(0.85 <= r["scale_bf_over_acdc"] <= 1.15)
            checks.append(r["alpha_nsigma"] < 3)
    print("GATE:", "PASS" if all(checks) else "FAIL (diagnose: moment -> normalization -> F aggregation)")
