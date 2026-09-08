"""D1-D6 diagnostics for the ACDC-vs-Doubles alpha gap seen at k = 0.98 on 100M.

WHY THIS EXISTS.  On `bench100M` the two Green's-function-side estimators disagree:
alpha_ACDC (binned unweighted LSQ of A + B exp(alpha tau) on the ASSEMBLED curve)
= -360.9 +/- 1.7 against alpha_BF (unbinned MLE on same-parent separations with the
flat floor frozen at w0) = -369.2 +/- 1.2, i.e. 3.9 sigma, and the window-free scale
statistic sum(R_bf)/sum(R_acdc) divided by M_emp/M_SIM sits 1.55 % below 1 (3.9 sigma).
At k = 0.90 the same quantities agree.  Four hypotheses were on the table:

  (a) estimator asymmetry     -- LSQ-on-a-curve vs MLE-on-events are not the same
                                 functional, so they need not return the same alpha
  (b) small-lag / smearing    -- the scale statistic starts at L = 3, where the box
                                 (BF) and triangle (ACDC) kernels differ most
  (c) delayed-neutron floor   -- a floor that is flat for one estimator and sloped
                                 for the other biases the fitted alpha
  (d) genuine time structure  -- the assembly really does decay more slowly

Each test below is designed to give one hypothesis a yes/no answer, and every test
runs on BOTH 100M datasets (`bench100M` at k = 0.98, `k090_bench100M` at k = 0.90).

ADDITIVE.  Nothing here edits `rossi_benchmark`, `greens_loader` or `flagship_k098`;
every estimator is either imported from them verbatim or newly written here so that
the two curves can be pushed through the SAME functional.

RUN (from repo root, PYTHONPATH=.):
  python -m project.analysis.gate_diagnostics all
  python -m project.analysis.gate_diagnostics d1 --dataset k090_bench100M

Figures and JSON land in project/analysis/outputs/gate_diagnostics/ (git-ignored).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, minimize_scalar

import project.analysis.flagship_k098 as fk
import project.analysis.rossi_benchmark as rb

DT = rb.DT
OUT_DIR = Path(__file__).resolve().parent / "outputs" / "gate_diagnostics"
DATASETS = ("bench100M", "k090_bench100M")

# Fraction of the record where the prompt term is dead and only the delayed /
# accidental floor survives: [LATE_FRAC * T, T].  0.375 gives [15, 40] ms at
# k = 0.98 and [3.75, 10] ms at k = 0.90, both ~5.5 prompt e-foldings out.
LATE_FRAC = 0.375

_CACHE: dict = {}


# --------------------------------------------------------------------------- #
# shared inputs: the two absolute curves on ONE lag grid + the raw separations
# --------------------------------------------------------------------------- #

def curves(dataset: str = "bench100M", f_source: str = "legacy") -> dict:
    """Both Rossi curves on the record's own lag grid, plus the BF separations.

    tau      : lag centres L*dt, L = 1 .. n_t-1 (rb.bf_rossi / gl.rossi convention)
    R_bf     : same-parent pairs per source per second, box-averaged over
               [(L-0.5)dt, (L+0.5)dt]
    counts   : the raw pair counts behind R_bf (the Poisson weight of everything)
    R_acdc   : the plug-in assembly sum_i F_i (g_i * g_i)(tau)
    dts      : every same-parent separation in (0, T], unbinned (the BF MLE's data)
    window   : the dataset's official fit window (frozen at k = 0.90, plateau rule
               at k = 0.98)
    """
    key = ("curves", dataset, f_source)
    if key in _CACHE:
        return _CACHE[key]
    inp = fk.inputs(dataset)
    n_t = inp["n_t"]
    tau, R_bf, counts = rb.bf_rossi(inp["df"], inp["n_sphere"],
                                    tau_max=inp["t_record"], n_tau=n_t - 1)
    tau_a, R_acdc = rb.acdc_rossi(inp["sweep"], fk._F(inp, f_source),
                                  tau_max=inp["t_record"], n_tau=n_t - 1)
    assert np.allclose(tau, tau_a)
    dts = rb.same_parent_pair_dts(inp["df"]["time"].to_numpy(),
                                  inp["df"]["parent_id"].to_numpy(),
                                  inp["t_record"])
    out = {
        "dataset": dataset, "label": inp["label"], "f_source": f_source,
        "tau": tau, "R_bf": R_bf, "counts": counts, "R_acdc": R_acdc,
        "dts": dts, "dt": inp["dt"], "t_record": inp["t_record"],
        "n_source": inp["n_sphere"], "window": fk.fit_window(dataset),
    }
    _CACHE[key] = out
    return out


def _window_mask(tau, window):
    lo, hi = window
    return (tau >= lo) & (tau <= hi)


def _bin_edges_of(tau_w, dt):
    """The outer edges of the lag bins whose centres are `tau_w`."""
    return float(tau_w[0] - 0.5 * dt), float(tau_w[-1] + 0.5 * dt)


# --------------------------------------------------------------------------- #
# the estimator that rossi_benchmark does NOT have: binned Poisson MLE
# --------------------------------------------------------------------------- #

def _fit_alpha_binned_noA(tau, R, dt, p0=None):
    """rb.fit_alpha_binned with the constant term REMOVED: R = B exp(alpha tau),
    bin-integrated.  The A term is what the two curves disagree about most (the
    assembly's LSQ A even goes negative), and A is strongly anti-correlated with
    alpha, so the 2-parameter fit isolates the pure decay-constant difference."""
    from scipy.optimize import curve_fit

    def model(x, B, al):
        z = al * dt / 2
        corr = np.sinh(np.clip(z, -700, 700)) / z if abs(z) > 1e-12 else 1.0
        return B * np.exp(np.clip(al * x, -700, 700)) * corr

    if p0 is None:
        p0 = [float(R[0]), -500.0]
    popt, pcov = curve_fit(model, tau, R, p0=p0, maxfev=50000)
    return popt, np.sqrt(np.diag(pcov))


def _bin_probs(alpha: float, w: float, u, v, a: float, b: float):
    """Probability mass in bins [u, v) of pdf = w/(b-a) + (1-w) trunc-exp(alpha).

    Written with expm1 differences so it is exact for |alpha (b-a)| up to ~700.
    """
    denom = np.expm1(alpha * (b - a))
    p_exp = (np.expm1(alpha * (v - a)) - np.expm1(alpha * (u - a))) / denom
    return w * (v - u) / (b - a) + (1.0 - w) * p_exp


def fit_alpha_binned_mle(centres, n_counts, dt, a, b, w=None):
    """Binned-Poisson (= multinomial) MLE of alpha for a histogram of pair lags.

    This is `rb.fit_alpha_mle`'s model -- a flat floor plus a truncated exponential
    on [a, b] -- but read off a HISTOGRAM rather than unbinned samples, so that an
    assembled R(tau) curve can be fed to it as pseudo-counts n_L = R_L n_source dt.
    The likelihood is scale-free in n, so the pseudo-count normalisation moves only
    the (indicative) error bar, never the point estimate.

    w=None floats the background weight; a float freezes it (the 'fixed' analogue).
    Returns dict(alpha, sigma, w, n_eff, n_bins).
    """
    centres = np.asarray(centres, float)
    n = np.asarray(n_counts, float)
    u, v = centres - 0.5 * dt, centres + 0.5 * dt
    keep = n > 0
    u_k, v_k, n_k = u[keep], v[keep], n[keep]

    if w is not None:
        w0 = float(np.clip(w, 0.0, 1.0 - 1e-9))

        def negll(alpha):
            p = _bin_probs(alpha, w0, u_k, v_k, a, b)
            if np.any(p <= 0):
                return 1e30
            return -float((n_k * np.log(p)).sum())

        res = minimize_scalar(negll, bounds=(-1e6, -1e-2), method="bounded",
                              options={"xatol": 1e-8})
        alpha = float(res.x)
        h = max(abs(alpha) * 1e-4, 1e-6)
        d2 = (negll(alpha + h) - 2 * negll(alpha) + negll(alpha - h)) / h ** 2
        sigma = float(1.0 / np.sqrt(d2)) if d2 > 0 else np.nan
        return {"alpha": alpha, "sigma": sigma, "w": w0,
                "n_eff": float(n_k.sum()), "n_bins": int(keep.sum())}

    def negll2(par):
        logit_w, log_neg_alpha = par
        ww = float(np.clip(1.0 / (1.0 + np.exp(-logit_w)), 1e-12, 1.0 - 1e-12))
        alpha = -np.exp(log_neg_alpha)
        p = _bin_probs(alpha, ww, u_k, v_k, a, b)
        if np.any(p <= 0):
            return 1e30
        return -float((n_k * np.log(p)).sum())

    x0 = np.array([np.log(0.05 / 0.95), np.log(400.0)])
    res = minimize(negll2, x0, method="Nelder-Mead",
                   options={"xatol": 1e-8, "fatol": 1e-6, "maxiter": 6000})
    logit_w, log_neg_alpha = res.x
    ww = 1.0 / (1.0 + np.exp(-logit_w))
    alpha = -float(np.exp(log_neg_alpha))
    h = 1e-4
    H = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            e_i, e_j = np.eye(2)[i] * h, np.eye(2)[j] * h
            H[i, j] = (negll2(res.x + e_i + e_j) - negll2(res.x + e_i - e_j)
                       - negll2(res.x - e_i + e_j) + negll2(res.x - e_i - e_j)) / (4 * h * h)
    try:
        cov = np.linalg.inv(H)
        sigma = abs(alpha) * float(np.sqrt(max(cov[1, 1], 0.0)))
    except np.linalg.LinAlgError:
        sigma = np.nan
    return {"alpha": alpha, "sigma": sigma, "w": float(ww),
            "n_eff": float(n_k.sum()), "n_bins": int(keep.sum())}


# --------------------------------------------------------------------------- #
# D1: estimator symmetry -- both curves through the SAME functionals
# --------------------------------------------------------------------------- #

def d1(dataset: str = "bench100M", f_source: str = "legacy") -> dict:
    """Cross the two curves with the two fit functionals.

    Columns:
      lsq_flat  : rb.fit_alpha_binned, floating A, UNWEIGHTED (the ACDC recipe)
      lsq_pois  : the same LSQ with 1/sqrt(counts) Poisson weights
      mle_fix   : binned-Poisson MLE, background weight FROZEN at the BF w0
      mle_free  : the same with w floating
    Rows: ACDC (pseudo-counts R_L n_source dt) and BF (real counts), plus the
    published unbinned BF MLE rows for reference.
    """
    c = curves(dataset, f_source)
    tau, dt, win = c["tau"], c["dt"], c["window"]
    lo, hi = win
    m = _window_mask(tau, win)
    tau_w = tau[m]
    a_edge, b_edge = _bin_edges_of(tau_w, dt)
    n_src = c["n_source"]

    R_ac, R_bf, cnt = c["R_acdc"][m], c["R_bf"][m], c["counts"][m]
    n_ac = R_ac * n_src * dt          # ACDC pseudo-counts
    n_bf = cnt.astype(float)          # BF real counts

    # --- w0, on the published interval and on the bin-edge interval ---------- #
    dts = c["dts"]
    w0_win = rb.flat_background_weight(dts, lo, hi)
    w0_bin = rb.flat_background_weight(dts, a_edge, b_edge)

    out = {"dataset": dataset, "label": c["label"], "f_source": f_source,
           "window": (lo, hi), "bin_interval": (a_edge, b_edge),
           "n_bins": int(m.sum()), "w0_window": float(w0_win),
           "w0_bin_interval": float(w0_bin),
           "n_pairs_window": int(((dts >= lo) & (dts <= hi)).sum()),
           "n_pairs_bin_interval": int(((dts >= a_edge) & (dts <= b_edge)).sum())}

    # --- LSQ column (unweighted and Poisson-weighted) ------------------------ #
    for name, R, w_counts in (("acdc", R_ac, n_ac), ("bf", R_bf, n_bf)):
        popt, perr = rb.fit_alpha_binned(tau_w, R, dt=dt)
        out[f"lsq_flat_{name}"] = float(popt[2])
        out[f"lsq_flat_{name}_err"] = float(perr[2])
        out[f"lsq_flat_{name}_A"] = float(popt[0])
        out[f"lsq_flat_{name}_B"] = float(popt[1])
        # floor fraction implied by the LSQ A term, on the MLE's own definition
        A, B, al = float(popt[0]), float(popt[1]), float(popt[2])
        flat = A * (b_edge - a_edge)
        expo = B * (np.exp(al * b_edge) - np.exp(al * a_edge)) / al
        out[f"lsq_flat_{name}_wA"] = float(flat / (flat + expo))
        sig = np.where(w_counts > 0, R / np.sqrt(np.maximum(w_counts, 1.0)), np.inf)
        good = np.isfinite(sig) & (sig > 0)
        try:
            popt2, perr2 = rb.fit_alpha_binned(tau_w[good], R[good], dt=dt,
                                               sigma=sig[good], p0=list(popt))
            out[f"lsq_pois_{name}"] = float(popt2[2])
            out[f"lsq_pois_{name}_err"] = float(perr2[2])
        except Exception as exc:                                # pragma: no cover
            out[f"lsq_pois_{name}"] = np.nan
            out[f"lsq_pois_{name}_err"] = np.nan
            out[f"lsq_pois_{name}_error"] = repr(exc)

    # --- each curve's OWN measured floor, from the dead late tail ------------ #
    # w_meas = (late-lag level) x (b - a) / integral of R over the window: the
    # floor fraction MEASURED where the prompt term is gone, not fitted inside
    # the window where it is degenerate with alpha.
    c_full = curves(dataset, f_source)
    T = c_full["t_record"]
    ml = (c_full["tau"] >= LATE_FRAC * T) & (c_full["tau"] <= T)
    for name, R, Rfull in (("acdc", R_ac, c_full["R_acdc"]),
                           ("bf", R_bf, c_full["R_bf"])):
        lvl = float(Rfull[ml].mean())
        out[f"floor_level_{name}"] = lvl
        out[f"w_meas_{name}"] = float(lvl * (b_edge - a_edge) / (R.sum() * dt))

    # --- 2-parameter LSQ (no constant term) --------------------------------- #
    for name, R in (("acdc", R_ac), ("bf", R_bf)):
        popt, perr = _fit_alpha_binned_noA(tau_w, R, dt)
        out[f"lsq_noA_{name}"] = float(popt[1])
        out[f"lsq_noA_{name}_err"] = float(perr[1])

    # --- binned-Poisson MLE column ------------------------------------------ #
    for name, n_counts in (("acdc", n_ac), ("bf", n_bf)):
        for tag, wfix in (("mle_w00", 0.0), ("mle_fix", w0_bin),
                          ("mle_wmeas", out[f"w_meas_{name}"]),
                          ("mle_free", None)):
            r = fit_alpha_binned_mle(tau_w, n_counts, dt, a_edge, b_edge, w=wfix)
            out[f"{tag}_{name}"] = r["alpha"]
            out[f"{tag}_{name}_err"] = r["sigma"]
            out[f"{tag}_{name}_w"] = r["w"]
        # leverage of the assumed floor on alpha: d(alpha)/d(w) over [0, w0]
        out[f"dalpha_dw_{name}"] = ((out[f"mle_fix_{name}"] - out[f"mle_w00_{name}"])
                                    / max(w0_bin, 1e-12))

    # --- the published unbinned BF rows, for reference ----------------------- #
    a_fix, s_fix, n_fix = rb.fit_alpha_mle(dts, lo, hi, background="fixed",
                                           w_fixed=w0_win)
    out["unbinned_fix_bf"] = float(a_fix)
    out["unbinned_fix_bf_err"] = float(s_fix)
    a_fl, s_fl, _ = rb.fit_alpha_mle(dts, lo, hi, background="flat")
    out["unbinned_flat_bf"] = float(a_fl)
    out["unbinned_flat_bf_err"] = float(s_fl)
    a_be, s_be, _ = rb.fit_alpha_mle(dts, a_edge, b_edge, background="fixed",
                                     w_fixed=w0_bin)
    out["unbinned_fix_bf_bin_interval"] = float(a_be)
    out["unbinned_fix_bf_bin_interval_err"] = float(s_be)

    for tag in ("lsq_flat", "lsq_pois", "lsq_noA", "mle_w00", "mle_fix",
                "mle_wmeas", "mle_free"):
        out[f"gap_{tag}"] = out[f"{tag}_acdc"] - out[f"{tag}_bf"]
        out[f"gap_{tag}_rel_pct"] = 100.0 * out[f"gap_{tag}"] / abs(out[f"{tag}_bf"])
    out["gap_published"] = out["lsq_flat_acdc"] - out["unbinned_fix_bf"]
    out["gap_published_rel_pct"] = (100.0 * out["gap_published"]
                                    / abs(out["unbinned_fix_bf"]))
    return out


D1_TAGS = ("lsq_flat", "lsq_pois", "lsq_noA", "mle_w00", "mle_fix",
           "mle_wmeas", "mle_free")


def _d1_fits(tau_w, R, n_counts, dt, a_edge, b_edge, w0_bin, w_meas):
    """Every D1 estimator applied to ONE curve (used by the bootstraps)."""
    o = {}
    popt, _ = rb.fit_alpha_binned(tau_w, R, dt=dt)
    o["lsq_flat"] = float(popt[2])
    sig = np.where(n_counts > 0, R / np.sqrt(np.maximum(n_counts, 1.0)), np.inf)
    good = np.isfinite(sig) & (sig > 0)
    try:
        p2, _ = rb.fit_alpha_binned(tau_w[good], R[good], dt=dt,
                                    sigma=sig[good], p0=list(popt))
        o["lsq_pois"] = float(p2[2])
    except Exception:
        o["lsq_pois"] = np.nan
    try:
        p3, _ = _fit_alpha_binned_noA(tau_w, R, dt)
        o["lsq_noA"] = float(p3[1])
    except Exception:
        o["lsq_noA"] = np.nan
    for tag, wfix in (("mle_w00", 0.0), ("mle_fix", w0_bin),
                      ("mle_wmeas", w_meas), ("mle_free", None)):
        o[tag] = fit_alpha_binned_mle(tau_w, n_counts, dt, a_edge, b_edge,
                                      w=wfix)["alpha"]
    return o


def d1_errors(dataset: str = "bench100M", f_source: str = "legacy",
              base: dict | None = None, n_boot_bf: int = 200,
              n_boot_acdc: int = 60, seed: int = 7) -> dict:
    """Honest error bars on the D1 like-for-like gaps.

    Two independent noise sources are resampled separately and combined:
      BF   : Poisson on the pair counts of each lag bin,
      ACDC : Poisson on the sweep's per-cell g_t counts (kre's own bootstrap),
             inflated by rb.KAPPA_S[label] -- the audited correction for the
             within-chain correlation that a per-bin Poisson treatment misses.
    """
    base = base or d1(dataset, f_source)
    c = curves(dataset, f_source)
    inp = fk.inputs(dataset)
    dt, tau = c["dt"], c["tau"]
    m = _window_mask(tau, c["window"])
    tau_w = tau[m]
    a_edge, b_edge = _bin_edges_of(tau_w, dt)
    w0_bin = base["w0_bin_interval"]
    n_src, n_t = c["n_source"], inp["n_t"]
    rng = np.random.default_rng(seed)

    bf = {t: [] for t in D1_TAGS}
    for _ in range(int(n_boot_bf)):
        cb = rng.poisson(c["counts"][m]).astype(float)
        Rb = cb / n_src / dt
        o = _d1_fits(tau_w, Rb, cb, dt, a_edge, b_edge, w0_bin,
                     base["w_meas_bf"])
        for t in D1_TAGS:
            bf[t].append(o[t])

    F = fk._F(inp, f_source)
    g_t = inp["sweep"]["g_t"]
    ac = {t: [] for t in D1_TAGS}
    for _ in range(int(n_boot_acdc)):
        s2 = dict(inp["sweep"])
        s2["g_t"] = rng.poisson(g_t).astype(float)
        _, Ra_full = rb.acdc_rossi(s2, F, tau_max=c["t_record"], n_tau=n_t - 1)
        Ra = Ra_full[m]
        na = Ra * n_src * dt
        o = _d1_fits(tau_w, Ra, na, dt, a_edge, b_edge, w0_bin,
                     base["w_meas_acdc"])
        for t in D1_TAGS:
            ac[t].append(o[t])

    kap = rb.KAPPA_S[c["label"]]
    out = {"dataset": dataset, "n_boot_bf": int(n_boot_bf),
           "n_boot_acdc": int(n_boot_acdc), "kappa_S": kap}
    for t in D1_TAGS:
        s_bf = float(np.nanstd(bf[t]))
        s_ac = float(np.nanstd(ac[t])) * kap
        s_gap = float(np.hypot(s_bf, s_ac))
        out[f"sigma_bf_{t}"] = s_bf
        out[f"sigma_acdc_{t}"] = s_ac
        out[f"sigma_gap_{t}"] = s_gap
        out[f"nsigma_gap_{t}"] = abs(base[f"gap_{t}"]) / s_gap if s_gap > 0 else np.nan
    return out


# --------------------------------------------------------------------------- #
# D2: where the residual lives -- r(tau) = R_bf / R_acdc
# --------------------------------------------------------------------------- #

def d2(dataset: str = "bench100M", f_source: str = "legacy",
       n_coarse: int = 14, tau_lo: float = 4e-5, plot: bool = True) -> dict:
    """Coarse-binned ratio r(tau) = R_bf(tau) / R_acdc(tau) over the whole record.

    Poisson error from the BF counts inside each coarse bin.  A weighted straight
    line through log r over the official window measures alpha_BF - alpha_ACDC with
    no fit model on either curve (both A + B exp forms cancel to first order in the
    ratio, leaving exp((alpha_BF - alpha_ACDC) tau)).
    """
    c = curves(dataset, f_source)
    tau, R_bf, R_ac, cnt = c["tau"], c["R_bf"], c["R_acdc"], c["counts"]
    lo, hi = c["window"]
    T = c["t_record"]

    edges = np.geomspace(tau_lo, T, n_coarse + 1)
    idx = np.digitize(tau, edges) - 1
    rows = []
    for b in range(n_coarse):
        sel = (idx == b) & (R_ac > 0)
        if not sel.any():
            continue
        n_pairs = float(cnt[sel].sum())
        s_bf, s_ac = float(R_bf[sel].sum()), float(R_ac[sel].sum())
        if s_ac <= 0 or n_pairs < 1:
            continue
        r = s_bf / s_ac
        rows.append({"tau_lo": float(edges[b]), "tau_hi": float(edges[b + 1]),
                     "tau_mid": float(np.sqrt(edges[b] * edges[b + 1])),
                     "r": r, "r_err": r / np.sqrt(n_pairs),
                     "n_pairs": n_pairs, "n_lags": int(sel.sum())})

    # --- window slope of log r, on the FINE lag grid ------------------------- #
    m = _window_mask(tau, c["window"]) & (cnt > 0) & (R_ac > 0)
    x, y = tau[m], np.log(R_bf[m] / R_ac[m])
    sy = 1.0 / np.sqrt(cnt[m].astype(float))
    W = 1.0 / sy ** 2
    Sw, Sx, Sy = W.sum(), (W * x).sum(), (W * y).sum()
    Sxx, Sxy = (W * x * x).sum(), (W * x * y).sum()
    det = Sw * Sxx - Sx * Sx
    c1 = (Sw * Sxy - Sx * Sy) / det
    c0 = (Sxx * Sy - Sx * Sxy) / det
    var_c1 = Sw / det
    resid = (y - (c0 + c1 * x)) / sy
    chi2 = float((resid ** 2).sum())
    dof = int(m.sum() - 2)

    # --- the same slope by Poisson GLM (no log bias) ------------------------- #
    # N_L ~ Poisson(mu_L), mu_L = R_acdc_L n dt exp(c0 + c1 tau_L).  log(N) is a
    # biased estimator of log(mu) at small counts (-1/2mu), and the counts fall
    # 20x across the k=0.98 window, so the plain log-space WLS above acquires a
    # spurious negative slope; this one does not.
    off = np.log(R_ac[m] * c["n_source"] * c["dt"])
    nobs = cnt[m].astype(float)

    # fitted in a scaled coordinate z = (tau - mid)/half-width so the design
    # matrix is O(1): in seconds, tau^2 ~ 1e-5 and an IRLS/BFGS step never moves
    xm, xs = float(x.mean()), float(0.5 * (x[-1] - x[0])) or 1.0
    z = (x - xm) / xs

    def _glm_fit(deg):
        """Poisson log-link GLM of nobs on [1, z, z^2, ...] with `off` fixed,
        by Newton-Raphson (IRLS).  Returns (beta, cov)."""
        X = np.column_stack([z ** j for j in range(deg + 1)])
        beta = np.zeros(deg + 1)
        beta[0] = float(np.log(nobs.sum() / np.exp(off).sum()))
        for _ in range(100):
            mu = np.exp(off + X @ beta)
            g = X.T @ (nobs - mu)
            H = X.T @ (mu[:, None] * X)
            step = np.linalg.solve(H, g)
            beta = beta + step
            if np.max(np.abs(step)) < 1e-12:
                break
        mu = np.exp(off + X @ beta)
        return beta, np.linalg.inv(X.T @ (mu[:, None] * X)), X

    def _negll(beta, X):
        eta = off + X @ beta
        return float(np.exp(eta).sum() - (nobs * eta).sum())

    b1, cov1, X1 = _glm_fit(1)
    g1 = float(b1[1] / xs)                       # d log r / d tau
    g1_err = float(np.sqrt(cov1[1, 1]) / xs)
    g0 = float(b1[0] - b1[1] * xm / xs)          # intercept at tau = 0

    # --- is the ratio a PURE alpha shift, or does it curve? ------------------ #
    # a pure decay-constant difference makes log r exactly linear in tau; any
    # curvature is a shape difference the 3-parameter A + B exp fit will dump
    # into its A term, where the A-alpha degeneracy multiplies it up
    b2, cov2, X2 = _glm_fit(2)
    q2 = float(b2[2] / xs ** 2)
    q2_err = float(np.sqrt(cov2[2, 2]) / xs ** 2)
    d_chi2 = 2.0 * (_negll(b1, X1) - _negll(b2, X2))

    # --- internal consistency: is the fitted gap what the ratio implies? ----- #
    R_scaled = R_ac[m] * np.exp(g0 + g1 * x)
    p_sc, _ = rb.fit_alpha_binned(x, R_scaled, dt=c["dt"])
    p_ac, _ = rb.fit_alpha_binned(x, R_ac[m], dt=c["dt"])

    # --- zone aggregates ----------------------------------------------------- #
    zones = {}
    for zname, (z0, z1) in (("L <= 5", (0.0, 5.5 * c["dt"])),
                            ("early (<0.3 ms)", (0.0, 3e-4)),
                            ("0.3 ms .. window lo", (3e-4, lo)),
                            ("window", (lo, hi)),
                            ("post-window to 15 ms", (hi, min(1.5e-2, T))),
                            ("late (>15 ms)", (1.5e-2, T))):
        zm = (tau > z0) & (tau <= z1) & (R_ac > 0)
        if not zm.any() or cnt[zm].sum() < 1:
            continue
        rz = float(R_bf[zm].sum() / R_ac[zm].sum())
        zones[zname] = {"r": rz, "r_err": rz / np.sqrt(float(cnt[zm].sum())),
                        "n_pairs": int(cnt[zm].sum()), "n_lags": int(zm.sum())}

    mom = _moment(dataset)
    out = {"dataset": dataset, "label": c["label"], "f_source": f_source,
           "window": (lo, hi), "t_record": T, "rows": rows, "zones": zones,
           "c0": float(c0), "c1": float(c1), "c1_err": float(np.sqrt(var_c1)),
           "c1_glm": g1, "c1_glm_err": g1_err, "c0_glm": g0,
           "c2_glm": q2, "c2_glm_err": q2_err,
           "c2_nsigma": abs(q2) / q2_err if q2_err > 0 else np.nan,
           "curvature_delta_chi2": float(d_chi2),
           "c1_glm_rel_pct": 100.0 * g1 / abs(p_ac[2]),
           "lsq_alpha_acdc": float(p_ac[2]),
           "lsq_alpha_acdc_scaled": float(p_sc[2]),
           "lsq_gap_from_ratio": float(p_sc[2] - p_ac[2]),
           "chi2": chi2, "dof": dof, "chi2_per_dof": chi2 / max(dof, 1),
           "n_fine_bins": int(m.sum()),
           "M_ratio": mom["M_emp"] / rb.M_SIM,
           "M_ratio_err": mom["M_err"] / rb.M_SIM}

    if plot:
        out["figure"] = _plot_d2(out)
    return out


def _plot_d2(res: dict) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = res["rows"]
    x = np.array([r["tau_mid"] for r in rows]) * 1e3
    xlo = np.array([r["tau_lo"] for r in rows]) * 1e3
    xhi = np.array([r["tau_hi"] for r in rows]) * 1e3
    y = np.array([r["r"] for r in rows])
    ye = np.array([r["r_err"] for r in rows])
    lo, hi = res["window"][0] * 1e3, res["window"][1] * 1e3

    mr, mre = res["M_ratio"], res["M_ratio_err"]
    tt = np.geomspace(max(lo, 1e-3), hi, 50)
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 6.4), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    for ax, zoom in zip(axes, (True, False)):
        ax.axvspan(lo, hi, color="0.85", zorder=0,
                   label="fit window" if zoom else None)
        ax.axhline(1.0, color="0.4", lw=0.8, ls=":")
        ax.axhline(mr, color="tab:green", lw=1.2,
                   label=(f"$M_{{emp}}/M_{{SIM}}$ = {mr:.4f}") if zoom else None)
        ax.axhspan(mr - mre, mr + mre, color="tab:green", alpha=0.15, lw=0)
        ax.errorbar(x, y, yerr=ye, xerr=[x - xlo, xhi - x], fmt="o", ms=4,
                    color="tab:red", ecolor="tab:red", elinewidth=1, capsize=0,
                    label=r"$r(\tau)=R_{bf}/R_{acdc}$" if zoom else None)
        ax.plot(tt, np.exp(res["c0_glm"] + res["c1_glm"] * tt * 1e-3),
                color="tab:blue", lw=1.6,
                label=(f"window slope $c_1$ = {res['c1_glm']:.1f} "
                       f"$\\pm$ {res['c1_glm_err']:.1f} s$^{{-1}}$")
                if zoom else None)
        ax.set_xscale("log")
        ax.set_ylabel(r"$R_{bf}/R_{acdc}$")
        ax.grid(alpha=0.25)
        if zoom:
            inw = (x >= lo) | (x <= hi)
            span = np.concatenate([y[inw] - ye[inw], y[inw] + ye[inw]])
            ax.set_ylim(min(0.96, float(np.percentile(span, 2))),
                        max(1.04, float(np.percentile(span, 60))))
            ax.legend(fontsize=8, loc="lower left")
            ax.set_title(f"D2 residual shape, {res['dataset']} "
                         f"(k label {res['label']}) -- zoom")
        else:
            ax.set_title("full range", fontsize=9)
    axes[-1].set_xlabel(r"lag $\tau$ [ms]")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"d2_ratio_{res['dataset']}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


# --------------------------------------------------------------------------- #
# D3: mask sensitivity of the scale statistic
# --------------------------------------------------------------------------- #

def _moment(dataset: str, f_source: str = "legacy", n_boot: int = 200,
            seed: int = 7) -> dict:
    key = ("moment", dataset, f_source, n_boot, seed)
    if key not in _CACHE:
        _CACHE[key] = fk.moment(dataset, n_boot=n_boot, rng=seed, f_source=f_source)
    return _CACHE[key]


def d3(dataset: str = "bench100M", f_source: str = "legacy") -> dict:
    """scale / (M_emp / M_SIM) as the lag mask is moved.

    If the 1.5 % deficit is a small-lag artifact (box vs triangle kernel, or the
    L = 1, 2 leakage the lag-axis fix left behind), raising tau_min must walk the
    ratio back to 1.  If it survives every mask, it is not a small-lag effect.
    """
    c = curves(dataset, f_source)
    tau, R_bf, R_ac, cnt = c["tau"], c["R_bf"], c["R_acdc"], c["counts"]
    dt, T = c["dt"], c["t_record"]
    mom = _moment(dataset, f_source)
    m_ratio = mom["M_emp"] / rb.M_SIM
    rel_m = mom["M_err"] / mom["M_emp"]
    lo_win = c["window"][0]

    mins = [("3 dt", 3 * dt), ("5 dt", 5 * dt), ("10 dt", 10 * dt),
            ("20 dt", 20 * dt), ("0.5 ms", 5e-4), ("1 ms", 1e-3),
            ("window lo", lo_win)]
    maxs = [("10 ms", 1e-2), ("20 ms", 2e-2), ("40 ms", 4e-2)]
    rows = []
    for nlo, t_lo in mins:
        for nhi, t_hi in maxs:
            if t_hi > T + 0.5 * dt or t_lo >= t_hi:
                continue
            msk = (tau >= t_lo) & (tau <= t_hi) & (R_ac > 0)
            if not msk.any():
                continue
            n_pairs = int(cnt[msk].sum())
            scale = float(R_bf[msk].sum() / R_ac[msk].sum())
            shape = scale / m_ratio
            rel = float(np.hypot(1.0 / np.sqrt(max(n_pairs, 1)), rel_m))
            rows.append({"tau_min": nlo, "tau_max": nhi,
                         "tau_min_s": float(t_lo), "tau_max_s": float(t_hi),
                         "scale": scale, "shape_ratio": shape,
                         "shape_err": shape * rel,
                         "nsigma": abs(shape - 1.0) / (shape * rel),
                         "n_pairs": n_pairs, "n_lags": int(msk.sum())})
    return {"dataset": dataset, "label": c["label"], "f_source": f_source,
            "M_emp": mom["M_emp"], "M_err": mom["M_err"], "M_ratio": m_ratio,
            "rows": rows}


# --------------------------------------------------------------------------- #
# D4: the delayed / accidental floor
# --------------------------------------------------------------------------- #

def _wls_line(x, y, w):
    W = np.asarray(w, float)
    Sw, Sx, Sy = W.sum(), (W * x).sum(), (W * y).sum()
    Sxx, Sxy = (W * x * x).sum(), (W * x * y).sum()
    det = Sw * Sxx - Sx * Sx
    slope = (Sw * Sxy - Sx * Sy) / det
    inter = (Sxx * Sy - Sx * Sxy) / det
    return float(inter), float(slope), float(np.sqrt(Sw / det))


def d4(dataset: str = "bench100M", f_source: str = "legacy") -> dict:
    """Measure the late-lag floor of both curves and ask whether its SHAPE can
    move the LSQ alpha by the observed ~8 s^-1.

    A finite record truncates the floor x floor overlap of the assembly: the
    autocorrelation of a constant over [0, T] falls off as (T - tau), so the
    assembly's floor is a ramp while the BF MLE models a strictly flat one.  The
    synthetic step below fits A + B exp(alpha tau) to
        true exponential (alpha = alpha_BF) + the MEASURED floor shape
    and reports how far the recovered alpha moves.
    """
    c = curves(dataset, f_source)
    tau, R_bf, R_ac, cnt = c["tau"], c["R_bf"], c["R_acdc"], c["counts"]
    dt, T = c["dt"], c["t_record"]
    lo, hi = c["window"]
    late = (LATE_FRAC * T, T)
    ml = (tau >= late[0]) & (tau <= late[1])
    mw = _window_mask(tau, c["window"])

    out = {"dataset": dataset, "label": c["label"], "f_source": f_source,
           "window": (lo, hi), "late_window": late, "t_record": T,
           "n_late_lags": int(ml.sum()), "n_late_pairs": int(cnt[ml].sum())}

    # --- measured floor level and slope, both curves ------------------------- #
    w_bf = np.maximum(cnt[ml].astype(float), 1.0)
    i_bf, s_bf, se_bf = _wls_line(tau[ml], R_bf[ml], w_bf)
    lvl_bf = float(R_bf[ml].mean())
    w_ac = np.maximum(R_ac[ml] * c["n_source"] * dt, 1.0)
    i_ac, s_ac, se_ac = _wls_line(tau[ml], R_ac[ml], w_ac)
    lvl_ac = float(R_ac[ml].mean())
    out.update({"floor_bf_mean": lvl_bf, "floor_bf_intercept": i_bf,
                "floor_bf_slope": s_bf, "floor_bf_slope_err": se_bf,
                "floor_acdc_mean": lvl_ac, "floor_acdc_intercept": i_ac,
                "floor_acdc_slope": s_ac, "floor_acdc_slope_err": se_ac})
    # the pure-truncation prediction: a flat component correlated over [0, T]
    # gives a floor proportional to (T - tau), i.e. slope / level = -1/(T - tau_mid)
    tau_mid_late = float(tau[ml].mean())
    out["trunc_slope_over_level_pred"] = -1.0 / (T - tau_mid_late)
    out["slope_over_level_bf"] = s_bf / lvl_bf
    out["slope_over_level_acdc"] = s_ac / lvl_ac

    # --- prompt-term extrapolation into the late window ---------------------- #
    popt, _ = rb.fit_alpha_binned(tau[mw], R_ac[mw], dt=dt)
    A_ac, B_ac, al_ac = float(popt[0]), float(popt[1]), float(popt[2])
    popt_b, _ = rb.fit_alpha_binned(tau[mw], R_bf[mw], dt=dt)
    A_bf, B_bf, al_bf = float(popt_b[0]), float(popt_b[1]), float(popt_b[2])
    out.update({"window_fit_acdc": {"A": A_ac, "B": B_ac, "alpha": al_ac},
                "window_fit_bf": {"A": A_bf, "B": B_bf, "alpha": al_bf},
                "prompt_at_late_mid_acdc": float(B_ac * np.exp(al_ac * tau_mid_late)),
                "prompt_at_late_mid_bf": float(B_bf * np.exp(al_bf * tau_mid_late))})

    # --- how much of the window sum is floor? -------------------------------- #
    for name, R, lvl in (("acdc", R_ac, lvl_ac), ("bf", R_bf, lvl_bf)):
        tot = float(R[mw].sum())
        out[f"floor_frac_window_{name}"] = lvl * int(mw.sum()) / tot
        # and at the window's upper edge, where the floor matters most
        out[f"floor_frac_at_hi_{name}"] = lvl / float(R[mw][-1])

    # --- synthetic: does a sloped floor bend the LSQ alpha? ------------------ #
    alpha_true = out["alpha_true"] = float(
        rb.fit_alpha_mle(c["dts"], lo, hi, background="fixed",
                         w_fixed=rb.flat_background_weight(c["dts"], lo, hi))[0])
    tw = tau[mw]
    # amplitude fixed so the synthetic curve matches the measured ACDC curve at
    # the window's lower edge (the highest-leverage point of an unweighted LSQ)
    scen = {}
    for tag, floor in (
            ("flat_floor", np.full(tw.shape, lvl_ac)),
            ("measured_ramp", i_ac + s_ac * tw),
            ("truncation_ramp", lvl_ac * (T - tw) / (T - tau_mid_late)),
            ("no_floor", np.zeros_like(tw))):
        B = (R_ac[mw][0] - floor[0]) / np.exp(alpha_true * tw[0])
        R_syn = B * np.exp(alpha_true * tw) + floor
        p, _ = rb.fit_alpha_binned(tw, R_syn, dt=dt,
                                   p0=[float(floor.mean()), float(B), alpha_true])
        scen[tag] = {"alpha_fit": float(p[2]),
                     "d_alpha": float(p[2] - alpha_true),
                     "A_fit": float(p[0]), "B_fit": float(p[1])}
    out["synthetic"] = scen
    out["d_alpha_ramp_minus_flat"] = (scen["measured_ramp"]["alpha_fit"]
                                      - scen["flat_floor"]["alpha_fit"])
    out["d_alpha_trunc_minus_flat"] = (scen["truncation_ramp"]["alpha_fit"]
                                       - scen["flat_floor"]["alpha_fit"])
    return out


# --------------------------------------------------------------------------- #
# D6: within-cell discretisation of the assembly (first-order estimate)
# --------------------------------------------------------------------------- #

def _fine_shell_moments(r_edges, R_ext, n_sub: int = 64):
    """Volume-weighted <A> and <A^2> inside each fine radial shell, with
    A(r) = sinc(r/R_ext) the bare-sphere fundamental mode (extrapolated radius)."""
    mean_A = np.empty(len(r_edges) - 1)
    mean_A2 = np.empty(len(r_edges) - 1)
    for i in range(len(r_edges) - 1):
        r0, r1 = r_edges[i], r_edges[i + 1]
        # equal-volume sub-sampling inside the shell: r = (r0^3 + u (r1^3-r0^3))^(1/3)
        u = (np.arange(n_sub) + 0.5) / n_sub
        r = (r0 ** 3 + u * (r1 ** 3 - r0 ** 3)) ** (1.0 / 3.0)
        A = np.sinc(r / R_ext)
        mean_A[i] = A.mean()
        mean_A2[i] = (A ** 2).mean()
    return mean_A, mean_A2


def d6(dataset: str = "bench100M", r_extra: float = 2.0) -> dict:
    """First-order estimate of the assembly's within-cell discretisation error.

    The assembly evaluates sum_c F_c ghat_c(t) ghat_c(t+tau) with ghat_c the
    UNIFORM-source-averaged response of coarse cell c, where the exact object is
    sum over fine sub-regions of F(r) g(r,t) g(r,t+tau).  Using the bare-sphere
    fundamental A(r) = sinc(r/R_ext), R_ext = R + r_extra, as a stand-in for the
    within-cell importance, the per-cell exact/assembled amplitude ratio is

        ratio_c = <A^2>_{F, c} / (<A>_{uniform, c})^2

    (Jensen: >= 1 whenever A varies inside the cell), and the assembly-level error
    is its F G^2-weighted mean.  This is an AMPLITUDE statement; it bounds how much
    within-cell structure the assembly throws away, not directly how much alpha
    moves.
    """
    from project.core.rossi_mesh import N_RADIAL, N_THETA
    import openmc

    inp = fk.inputs(dataset)
    sweep = inp["sweep"]
    n_r, n_th = sweep["G"].shape
    R = float(sweep["R"])
    R_ext = R + r_extra

    with openmc.StatePoint(str(inp["paths"]["statepoint"])) as sp:
        F_tally = sp.get_tally(name="fission_per_cell").mean.flatten()
    F_fine = F_tally.reshape(N_THETA, N_RADIAL).T          # (r, theta)

    r_edges = R * np.linspace(0, 1, N_RADIAL + 1) ** (1 / 3)
    mA, mA2 = _fine_shell_moments(r_edges, R_ext)

    kr, kth = N_RADIAL // n_r, N_THETA // n_th
    F_blk = F_fine.reshape(n_r, kr, n_th, kth)             # (ir, sub_r, ith, sub_th)
    mA_blk = mA.reshape(n_r, kr)
    mA2_blk = mA2.reshape(n_r, kr)

    # <A^2>_F over the coarse cell (F-weighted over its fine sub-cells)
    num = (F_blk * mA2_blk[:, :, None, None]).sum(axis=(1, 3))
    den = F_blk.sum(axis=(1, 3))
    A2_F = num / np.maximum(den, 1e-300)
    # <A>_uniform over the coarse cell (equal-volume fine cells -> plain mean)
    A_unif = np.repeat(mA_blk.mean(axis=1)[:, None], n_th, axis=1)

    ratio = A2_F / A_unif ** 2
    # F-free variant (plain Jensen inside the cell), to show the fission weighting
    # is not what drives the sign
    A2_unif = np.repeat(mA2_blk.mean(axis=1)[:, None], n_th, axis=1)
    ratio_unif = A2_unif / A_unif ** 2

    G = sweep["n_prompt"] / float(sweep["n_source"])
    Wt = inp["F"] * G ** 2
    w_mean = float((Wt * ratio).sum() / Wt.sum())
    per_ring = [{"ir": int(i), "r_mid_cm": float(sweep["r_mid"][i].mean()),
                 "ratio": float((Wt[i] * ratio[i]).sum() / max(Wt[i].sum(), 1e-300)),
                 "weight_frac": float(Wt[i].sum() / Wt.sum())}
                for i in range(n_r)]
    return {"dataset": dataset, "label": inp["label"], "R_cm": R,
            "R_ext_cm": R_ext, "n_r": n_r, "n_th": n_th,
            "fine": (N_RADIAL, N_THETA),
            "ratio_min": float(ratio.min()), "ratio_max": float(ratio.max()),
            "ratio_FG2_mean": w_mean,
            "ratio_FG2_mean_unif": float((Wt * ratio_unif).sum() / Wt.sum()),
            "pct_amplitude_deficit": 100.0 * (w_mean - 1.0),
            "per_radial_ring": per_ring}


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #

def _p(*a):
    print(*a, flush=True)


def report_d1(r: dict):
    _p(f"\n=== D1 estimator symmetry: {r['dataset']} (k{r['label']}) ===")
    _p(f"window [{r['window'][0]*1e3:.4f}, {r['window'][1]*1e3:.4f}] ms, "
       f"{r['n_bins']} lag bins, bin interval "
       f"[{r['bin_interval'][0]*1e3:.4f}, {r['bin_interval'][1]*1e3:.4f}] ms")
    _p(f"w0 (window) = {r['w0_window']:.5f}   w0 (bin interval) = "
       f"{r['w0_bin_interval']:.5f}   window pairs = {r['n_pairs_window']:,}")
    _p(f"measured floor fraction on the window: ACDC {r['w_meas_acdc']:.5f}, "
       f"BF {r['w_meas_bf']:.5f}")
    hdr = (f"{'estimator':22s} {'ACDC':>12s} {'BF':>12s} {'gap':>9s} "
           f"{'gap %':>7s} {'sig_gap':>8s} {'nsig':>7s} {'w_ACDC':>8s} "
           f"{'w_BF':>8s}")
    _p(hdr)
    _p("-" * len(hdr))
    err = r.get("errors", {})
    for tag, name in (("lsq_flat", "LSQ floating A"),
                      ("lsq_pois", "LSQ Poisson-weighted"),
                      ("lsq_noA", "LSQ, NO A term"),
                      ("mle_w00", "binned MLE, w = 0"),
                      ("mle_fix", "binned MLE, w0 fix"),
                      ("mle_wmeas", "binned MLE, w measured"),
                      ("mle_free", "binned MLE, w free")):
        wa = r.get(f"{tag}_acdc_w", np.nan)
        wb = r.get(f"{tag}_bf_w", np.nan)
        sg = err.get(f"sigma_gap_{tag}", np.nan)
        ns = err.get(f"nsigma_gap_{tag}", np.nan)
        _p(f"{name:22s} {r[tag+'_acdc']:12.2f} {r[tag+'_bf']:12.2f} "
           f"{r['gap_'+tag]:+9.2f} {r['gap_'+tag+'_rel_pct']:+7.2f} "
           f"{sg:8.2f} {ns:7.2f} {wa:8.5f} {wb:8.5f}")
    _p(f"floor fraction implied by the LSQ A term: ACDC "
       f"{r['lsq_flat_acdc_wA']:.5f}, BF {r['lsq_flat_bf_wA']:.5f}")
    _p(f"floor leverage d(alpha)/d(w): ACDC {r['dalpha_dw_acdc']:.0f}, "
       f"BF {r['dalpha_dw_bf']:.0f} s^-1 per unit floor fraction "
       f"(so the {abs(r['lsq_flat_acdc_wA']-r['lsq_flat_bf_wA'])*100:.2f} % "
       f"baseline difference is worth "
       f"{abs(r['dalpha_dw_acdc'])*abs(r['lsq_flat_acdc_wA']-r['lsq_flat_bf_wA']):.1f}"
       f" s^-1)")
    _p(f"{'unbinned MLE w0 fix':22s} {'-':>12s} {r['unbinned_fix_bf']:12.2f} "
       f"{r['gap_published']:+9.2f}  <- published pairing")
    _p(f"{'unbinned MLE w free':22s} {'-':>12s} {r['unbinned_flat_bf']:12.2f}")
    _p(f"{'unbinned, bin interval':22s} {'-':>12s} "
       f"{r['unbinned_fix_bf_bin_interval']:12.2f}")


def report_d2(r: dict):
    _p(f"\n=== D2 residual shape: {r['dataset']} (k{r['label']}) ===")
    _p(f"M_emp/M_SIM = {r['M_ratio']:.5f} +/- {r['M_ratio_err']:.5f}")
    hdr = (f"{'tau [ms]':>18s} {'r':>9s} {'+/-':>8s} {'r/(M/Ms)':>9s} "
           f"{'pairs':>12s} {'lags':>6s}")
    _p(hdr)
    _p("-" * len(hdr))
    for row in r["rows"]:
        _p(f"{row['tau_lo']*1e3:8.3f}-{row['tau_hi']*1e3:8.3f} "
           f"{row['r']:9.4f} {row['r_err']:8.4f} "
           f"{row['r']/r['M_ratio']:9.4f} {int(row['n_pairs']):12,d} "
           f"{row['n_lags']:6d}")
    for zname, z in r["zones"].items():
        _p(f"  zone {zname:22s} r = {z['r']:.4f} +/- {z['r_err']:.4f} "
           f"({z['n_pairs']:,} pairs)")
    _p(f"window log-slope c1 (log-space WLS) = {r['c1']:.2f} +/- {r['c1_err']:.2f} "
       f"s^-1  (chi2/dof = {r['chi2_per_dof']:.2f}, {r['n_fine_bins']} fine bins)")
    _p(f"window log-slope c1 (Poisson GLM)   = {r['c1_glm']:.2f} +/- "
       f"{r['c1_glm_err']:.2f} s^-1  = {r['c1_glm_rel_pct']:+.2f} % of alpha; "
       f"intercept exp(c0) = {np.exp(r['c0_glm']):.4f}")
    _p(f"curvature of log r: c2 = {r['c2_glm']:.3g} +/- {r['c2_glm_err']:.3g} "
       f"s^-2 ({r['c2_nsigma']:.2f} sigma, delta chi2 = "
       f"{r['curvature_delta_chi2']:.1f})")
    _p(f"consistency: LSQ on R_acdc x exp(c0+c1 tau) gives "
       f"{r['lsq_alpha_acdc_scaled']:.2f} vs {r['lsq_alpha_acdc']:.2f} on R_acdc "
       f"-> the ratio alone moves the LSQ alpha by "
       f"{r['lsq_gap_from_ratio']:+.2f} s^-1")
    _p(f"figure: {r.get('figure')}")


def report_d3(r: dict):
    _p(f"\n=== D3 mask sensitivity: {r['dataset']} (k{r['label']}) ===")
    _p(f"M_emp = {r['M_emp']:.4f} +/- {r['M_err']:.4f}  -> M/M_SIM = {r['M_ratio']:.5f}")
    hdr = (f"{'tau_min':>10s} {'tau_max':>8s} {'scale':>9s} {'shape':>9s} "
           f"{'+/-':>8s} {'sigma':>7s} {'pairs':>12s}")
    _p(hdr)
    _p("-" * len(hdr))
    for row in r["rows"]:
        _p(f"{row['tau_min']:>10s} {row['tau_max']:>8s} {row['scale']:9.4f} "
           f"{row['shape_ratio']:9.4f} {row['shape_err']:8.4f} "
           f"{row['nsigma']:7.2f} {row['n_pairs']:12,d}")


def report_d4(r: dict):
    _p(f"\n=== D4 delayed floor: {r['dataset']} (k{r['label']}) ===")
    _p(f"late window [{r['late_window'][0]*1e3:.2f}, {r['late_window'][1]*1e3:.2f}] ms, "
       f"{r['n_late_lags']} lags, {r['n_late_pairs']:,} BF pairs")
    _p(f"BF   floor mean {r['floor_bf_mean']:.6g}  slope {r['floor_bf_slope']:.6g} "
       f"+/- {r['floor_bf_slope_err']:.3g}  slope/level {r['slope_over_level_bf']:.2f} /s")
    _p(f"ACDC floor mean {r['floor_acdc_mean']:.6g}  slope {r['floor_acdc_slope']:.6g} "
       f"+/- {r['floor_acdc_slope_err']:.3g}  slope/level "
       f"{r['slope_over_level_acdc']:.2f} /s")
    _p(f"pure (T-tau) truncation would give slope/level = "
       f"{r['trunc_slope_over_level_pred']:.2f} /s")
    _p(f"prompt term extrapolated to the late window: ACDC "
       f"{r['prompt_at_late_mid_acdc']:.3g}, BF {r['prompt_at_late_mid_bf']:.3g} "
       f"(vs floors above)")
    _p(f"floor as a fraction of the window sum: ACDC "
       f"{r['floor_frac_window_acdc']*100:.2f} %, BF "
       f"{r['floor_frac_window_bf']*100:.2f} %; at the window's upper edge "
       f"ACDC {r['floor_frac_at_hi_acdc']*100:.1f} %, BF "
       f"{r['floor_frac_at_hi_bf']*100:.1f} %")
    _p(f"synthetic (alpha_true = {r['alpha_true']:.2f}):")
    for tag, v in r["synthetic"].items():
        _p(f"   {tag:16s} alpha_fit {v['alpha_fit']:9.2f}  "
           f"d_alpha {v['d_alpha']:+7.2f}")
    _p(f"   ramp - flat      = {r['d_alpha_ramp_minus_flat']:+.2f} s^-1")
    _p(f"   trunc-ramp - flat= {r['d_alpha_trunc_minus_flat']:+.2f} s^-1")


def report_d6(r: dict):
    _p(f"\n=== D6 within-cell discretisation: {r['dataset']} (k{r['label']}) ===")
    _p(f"R = {r['R_cm']:.2f} cm, R_ext = {r['R_ext_cm']:.2f} cm, "
       f"coarse {r['n_r']}x{r['n_th']} from fine {r['fine'][0]}x{r['fine'][1]}")
    _p(f"per-cell <A^2>_F / <A>_unif^2 in [{r['ratio_min']:.5f}, "
       f"{r['ratio_max']:.5f}]; F G^2-weighted mean {r['ratio_FG2_mean']:.5f} "
       f"({r['pct_amplitude_deficit']:+.2f} % amplitude, first-order); "
       f"F-free variant {r['ratio_FG2_mean_unif']:.5f}")
    _p("sign: >1 means the assembly UNDER-states the amplitude, i.e. it predicts "
       "sum(R_bf)/sum(R_acdc) ABOVE 1")
    _p(f"{'ring':>5s} {'r_mid[cm]':>10s} {'ratio':>9s} {'weight':>8s}")
    for row in r["per_radial_ring"]:
        _p(f"{row['ir']:5d} {row['r_mid_cm']:10.2f} {row['ratio']:9.5f} "
           f"{row['weight_frac']:8.4f}")


def compare(res: dict | None = None) -> dict:
    """D5's actual question: is the k = 0.90 control statistically DIFFERENT from
    k = 0.98, or merely less precise?

    Every quantity is put in RELATIVE units (% of |alpha|, or the dimensionless
    shape ratio) and the two datasets are compared with their own error bars.
    The published alpha errors come from the flagship summary JSONs; everything
    else is recomputed here.
    """
    if res is None:
        res = {}
        for ds in DATASETS:
            b = d1(ds)
            b["errors"] = d1_errors(ds, base=b)
            res[ds] = {"d1": b, "d2": d2(ds, plot=False), "d3": d3(ds)}
    pub = {}
    for ds in DATASETS:
        p = fk.OUT_DIR / f"{ds}_summary.json"
        pub[ds] = json.loads(p.read_text()) if p.exists() else None

    rows = []
    # (1) published alpha gap, in % of |alpha_BF|
    vals = {}
    for ds in DATASETS:
        if pub[ds] is None:
            continue
        a = pub[ds]["alphas"]
        gap = a["acdc"] - a["bf"]
        err = float(np.hypot(a["acdc_sigma"], a["bf_sigma"]))
        vals[ds] = (100 * gap / abs(a["bf"]), 100 * err / abs(a["bf"]))
    if len(vals) == 2:
        rows.append(("alpha gap (ACDC - BF) [% of |alpha|]", vals))
    # (2) shape ratio at the headline mask
    vals = {ds: (res[ds]["d3"]["rows"][0]["shape_ratio"],
                 res[ds]["d3"]["rows"][0]["shape_err"]) for ds in DATASETS}
    rows.append(("shape ratio scale/(M_emp/M_SIM) [3dt .. record]", vals))
    # (3) directly measured ratio slope, in % of |alpha_ACDC|
    vals = {}
    for ds in DATASETS:
        d = res[ds]["d2"]
        a = abs(d["lsq_alpha_acdc"])
        vals[ds] = (100 * d["c1_glm"] / a, 100 * d["c1_glm_err"] / a)
    rows.append(("r(tau) log-slope c1 [% of |alpha|]", vals))
    # (4-6) like-for-like gaps, % of |alpha|, with the D1 bootstrap error
    for tag, nm in (("lsq_flat", "LSQ-vs-LSQ gap"),
                    ("lsq_noA", "no-A LSQ gap"),
                    ("mle_wmeas", "MLE, measured floor, gap")):
        vals = {}
        for ds in DATASETS:
            d = res[ds]["d1"]
            e = d.get("errors", {})
            a = abs(d[f"{tag}_bf"])
            vals[ds] = (d[f"gap_{tag}_rel_pct"],
                        100 * e.get(f"sigma_gap_{tag}", np.nan) / a)
        rows.append((f"{nm} [% of |alpha|]", vals))

    out = {"rows": []}
    _p("\n=== D5 cross-k compatibility (k = 0.98 vs k = 0.90) ===")
    hdr = (f"{'quantity':46s} {'k=0.98':>18s} {'k=0.90':>18s} {'n_sigma':>8s}")
    _p(hdr)
    _p("-" * len(hdr))
    for name, vals in rows:
        v98, e98 = vals["bench100M"]
        v90, e90 = vals["k090_bench100M"]
        ns = (abs(v98 - v90) / float(np.hypot(e98, e90))
              if np.isfinite(e98) and np.isfinite(e90) else np.nan)
        _p(f"{name:46s} {v98:9.4f}+/-{e98 if np.isfinite(e98) else 0:<8.4f} "
           f"{v90:9.4f}+/-{e90 if np.isfinite(e90) else 0:<8.4f} {ns:8.2f}")
        out["rows"].append({"quantity": name, "k098": v98, "k098_err": e98,
                            "k090": v90, "k090_err": e90, "nsigma": ns})
    return out


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, Path):
        return str(x)
    return x


def run_all(dataset: str = "bench100M", f_source: str = "legacy",
            write: bool = True) -> dict:
    t0 = time.time()
    out = {}

    def _d1_with_errors():
        b = d1(dataset, f_source)
        b["errors"] = d1_errors(dataset, f_source, base=b)
        return b

    for name, fn in (("d1", _d1_with_errors),
                     ("d2", lambda: d2(dataset, f_source)),
                     ("d3", lambda: d3(dataset, f_source)),
                     ("d4", lambda: d4(dataset, f_source)),
                     ("d6", lambda: d6(dataset))):
        t = time.time()
        r = fn()
        r["_seconds"] = time.time() - t
        out[name] = r
        globals()[f"report_{name}"](r)
        _p(f"[{name} took {r['_seconds']:.1f} s]")
    out["_total_seconds"] = time.time() - t0
    if write:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        path = OUT_DIR / f"{dataset}_diagnostics.json"
        path.write_text(json.dumps(_jsonable(out), indent=2, default=float))
        _p(f"\nwrote {path}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", nargs="?", default="all",
                    choices=["all", "d1", "d2", "d3", "d4", "d6", "both",
                             "compare"])
    ap.add_argument("--dataset", default="bench100M", choices=list(DATASETS))
    ap.add_argument("--f-source", default="legacy", choices=["legacy", "matched"])
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args(argv)

    if args.stage in ("both", "compare"):
        res = {}
        for ds in DATASETS:
            res[ds] = run_all(ds, args.f_source, write=not args.no_write)
        c = compare(res)
        if not args.no_write:
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            (OUT_DIR / "cross_k_compare.json").write_text(
                json.dumps(_jsonable(c), indent=2, default=float))
        return
    if args.stage == "all":
        run_all(args.dataset, args.f_source, write=not args.no_write)
        return
    fn = {"d1": d1, "d2": d2, "d3": d3, "d4": d4}.get(args.stage)
    r = fn(args.dataset, args.f_source) if fn else d6(args.dataset)
    globals()[f"report_{args.stage}"](r)


if __name__ == "__main__":
    main()
