"""k=0.98 extended-record (40 ms) analysis: repair of the 10 ms truncation bias.

WHY. The ACDC assembly R(tau) = sum_i F_i * integral g_i(t) g_i(t+tau) dt was built
from per-cell histograms g_i(t) recorded on a fixed 10 ms window (N_T=512, DT~19.53us).
At lag tau the overlap integral only runs over t in [0, T_MAX - tau], so the assembled
R(tau) is suppressed by the factor (1 - exp(-2|alpha|(T_MAX - tau))) relative to the
untruncated value. At k=0.98 (1/|alpha| ~ 2.9 ms) the record is only ~3.4 decay times:
the suppression is ~0.5% at tau = 2 ms but ~25% at 8 ms and ~50% at 9 ms, i.e. the late
part of the vetted fit window [2.014, 9.971] ms droops. This module rebuilds the k=0.98
per-cell histograms on a 40 ms record (2048 bins, SAME DT) from the raw event file
`greens-sweep-k098-bench10M-events.h5` -- no new simulations -- and re-runs the audited
analysis on the extended record. k=0.90 (and every other label) is untouched: at
1/|alpha| ~ 0.65 ms the truncation factor is < 1e-8 inside the fit window.

DESIGN. Everything here is ADDITIVE: the audited estimator code paths in
`rossi_benchmark` are imported and reused verbatim wherever they are record-length
agnostic (bf_rossi, acdc_rossi/gl.rossi, fit_alpha_binned, fit_alpha_mle,
bf_alpha_sigma_events, oracle/realistic/die block machinery, sigma_vs_budget,
histories_to_target ...). Only the few helpers that HARDCODE the 10 ms grid
(np.linspace(0, T_MAX, N_T+1) / the N_T-1 lag cap) get an *_ext mirror below that
takes the record length from the data instead. No existing results file is modified;
the extended histogram set is written to a NEW npz
(`greens-sweep-k098-bench10M-ext40ms.npz`, same schema, g_t/time_edges replaced).

RUN (from repo root, PYTHONPATH=.):
  python -m project.analysis.k098_record_extension build      # ext npz + gate G0
  python -m project.analysis.k098_record_extension window     # plateau-rule window on ext record
  python -m project.analysis.k098_record_extension alphas     # gate + half-window check + overlay fig
  python -m project.analysis.k098_record_extension cost-old   # reproduce vetted 10ms-window costs
  python -m project.analysis.k098_record_extension cost-new   # costs on the extended window
  python -m project.analysis.k098_record_extension k095       # numeric truncation-bias estimate

Stage outputs (json/npz/figures) land in project/analysis/outputs/k098_extension/
(git-ignored); the human-readable report lives in
project/analysis/outputs/presentation/K098_EXTENSION_REPORT.md.
"""
import json
import sys
from pathlib import Path

import numpy as np

import project.analysis.greens_loader as gl
import project.analysis.rossi_benchmark as rb

LABEL = "098"
DT = rb.DT                          # 1.953125e-05 s, unchanged
EXT_N_T = 2048
EXT_T_MAX = EXT_N_T * DT            # 0.04 s exactly (same DT, 4x the record)

RESULTS = gl.RESULTS_ROOT
BASE_NPZ = RESULTS / "greens-sweep-k098-bench10M.npz"
EVENTS_H5 = RESULTS / "greens-sweep-k098-bench10M-events.h5"
EXT_NPZ = RESULTS / "greens-sweep-k098-bench10M-ext40ms.npz"
SPHERE_DIR = RESULTS / "greens-function-k098-bench10M"
SPHERE_TRACK = SPHERE_DIR / "collision_track.h5"
N_SPHERE = 10_000_000               # statepoint n_particles of the bench10M sphere run

OUT_DIR = Path(__file__).resolve().parent / "outputs" / "k098_extension"
PRESENT_DIR = Path(__file__).resolve().parent / "outputs" / "presentation"
WINDOW_JSON = OUT_DIR / "ext_window.json"


def ext_time_edges() -> np.ndarray:
    """Extended-record bin edges: same DT, 2048 bins, 40 ms. The first 513 edges
    are bit-identical to the original 10 ms grid (verified in build), so the
    10 ms record is exactly the first 512 bins of this one."""
    return np.linspace(0.0, EXT_T_MAX, EXT_N_T + 1)


# --------------------------------------------------------------------------- #
# Stage: build -- extended npz from raw events + acceptance gate G0
# --------------------------------------------------------------------------- #

def build_extended_npz(verbose: bool = True) -> dict:
    """Rebuild g_i(t) on the 40 ms record from the per-cell event file and save a
    NEW npz with the base schema (g_t/time_edges replaced, all other fields
    copied verbatim -- G/n_prompt in the base file already count ALL prompt
    events regardless of time, so they are not record-limited and stay as-is).

    Gate G0: the rebuild restricted to the original 10 ms edges must match the
    base g_t bin-for-bin EXACTLY, and the per-cell event counts must equal the
    stored n_prompt exactly. Raises AssertionError on any mismatch.
    """
    base = dict(np.load(BASE_NPZ))
    events, n_src = rb.load_sweep_events(EVENTS_H5)          # prompt-only
    assert n_src == int(base["n_source"]), (n_src, base["n_source"])

    edges10 = base["time_edges"]
    edges40 = ext_time_edges()
    assert np.array_equal(edges40[: len(edges10)], edges10), \
        "extended edges do not extend the original grid bit-exactly"

    n_r, n_th = base["G"].shape
    g_t_ext = np.zeros((n_r, n_th, EXT_N_T))
    mismatches, n_gt10, n_gt40 = 0, 0, 0
    for (ir, ith), (t, _p) in events.items():
        h10, _ = np.histogram(t, bins=edges10)
        if not np.array_equal(h10.astype(np.float64), base["g_t"][ir, ith]):
            mismatches += 1
        assert len(t) == int(base["n_prompt"][ir, ith]), \
            f"cell {(ir, ith)}: events {len(t)} != stored n_prompt {base['n_prompt'][ir, ith]}"
        h40, _ = np.histogram(t, bins=edges40)
        g_t_ext[ir, ith] = h40
        n_gt10 += int((t > edges10[-1]).sum())
        n_gt40 += int((t > edges40[-1]).sum())
    assert mismatches == 0, f"gate G0 FAILED: {mismatches} cells mismatch at 10 ms"

    in10, in40 = float(base["g_t"].sum()), float(g_t_ext.sum())
    checks = {
        "gate_G0_pass": True,
        "cells": len(events),
        "n_prompt_total": int(base["n_prompt"].sum()),
        "in_record_10ms": int(in10),
        "in_record_40ms": int(in40),
        "gain_events_10_to_40ms": int(in40 - in10),
        "gain_pct_vs_10ms_record": 100.0 * (in40 - in10) / in10,
        "prompt_beyond_10ms": n_gt10,
        "prompt_beyond_40ms": n_gt40,
        "tail_beyond_40ms_pct_of_G": 100.0 * n_gt40 / float(base["n_prompt"].sum()),
    }

    out = dict(base)
    out["g_t"] = g_t_ext
    out["time_edges"] = edges40
    out["ext_note"] = np.str_(
        "40 ms extended record rebuilt from greens-sweep-k098-bench10M-events.h5 "
        "(prompt-only, same DT; first 512 bins bit-identical to the 10 ms base npz). "
        "G/n_prompt copied from the base file (they already integrate all times). "
        "Built by project/analysis/k098_record_extension.py."
    )
    np.savez_compressed(EXT_NPZ, **out)
    if verbose:
        print(f"gate G0 PASS: all {checks['cells']} cells match bin-for-bin at 10 ms; "
              f"per-cell event counts == stored n_prompt")
        print(f"in-record prompt events: 10 ms {in10:.0f} -> 40 ms {in40:.0f} "
              f"(+{checks['gain_events_10_to_40ms']} = "
              f"+{checks['gain_pct_vs_10ms_record']:.2f}% of the 10 ms record)")
        print(f"prompt beyond 10 ms: {n_gt10}; beyond 40 ms (delayed-branch floor, "
              f"stays un-recorded): {n_gt40} = "
              f"{checks['tail_beyond_40ms_pct_of_G']:.2f}% of G's integral")
        print(f"wrote {EXT_NPZ}")
    return checks


def load_ext_sweep() -> dict:
    """Extended sweep dict, same shape contract as gl.load_sweep (plus ext fields)."""
    f = np.load(EXT_NPZ, allow_pickle=False)
    d = {key: f[key] for key in f.files}
    d["label"] = LABEL
    d["k"] = float(d["k"])
    return d


# --------------------------------------------------------------------------- #
# Record-length-generic mirrors of the 10ms-hardcoded helpers
# --------------------------------------------------------------------------- #

def window_lag_count_ext(fit_window, n_t: int) -> int:
    """rb.window_lag_count with the lag cap taken from the record length n_t
    instead of the module constant N_T (the 10 ms cap is exactly what truncated
    the k098 window)."""
    return int(min(n_t - 1, int(np.ceil(fit_window[1] / DT)) + 2))


def plateau_window_ext(sweep: dict, F: np.ndarray, width_decay: float = 3.0,
                       ratio: float = 4.0, rtol: float = 0.10,
                       verbose: bool = False) -> dict:
    """rb.plateau_window on an EXPLICIT sweep dict over its FULL lag range.

    Identical rule (plateau scan -> band-median fundamental -> earliest sustained
    on-plateau start -> width_decay decay times), but the lag range is
    n_t - 1 from the sweep's own record instead of the hardcoded N_T - 1, so the
    window is no longer ceilinged by the 10 ms record.
    """
    n_t = sweep["g_t"].shape[-1]
    tau, R = rb.acdc_rossi(sweep, F, n_tau=n_t - 1)
    lo_grid, a_grid = rb.plateau_scan(tau, R, ratio=ratio, n_scan=40)

    a_fund = rb._alpha_fund(lo_grid, a_grid)
    if not np.isfinite(a_fund):
        a_fund = rb._alpha_on_window(tau, R, 2 * DT, float(tau[-1])) or -1000.0
    tau_d = 1.0 / abs(a_fund)

    rel = np.abs(rb._median3(a_grid) - a_fund) / abs(a_fund)
    eligible = lo_grid >= 0.5 * tau_d
    near = np.isfinite(rel) & (rel < rtol) & eligible
    sustained = near.copy()
    sustained[:-1] &= near[1:]
    i_lo = (int(np.argmax(sustained)) if sustained.any()
            else int(np.argmax(near)) if near.any()
            else int(np.argmin(np.abs(lo_grid - tau_d))))
    lo = float(lo_grid[i_lo])
    hi = float(min(lo + width_decay * tau_d, float(tau[-1])))
    alpha = rb._alpha_on_window(tau, R, lo, hi)
    if not (np.isfinite(alpha) and alpha < 0):
        alpha = a_fund
    if verbose:
        print(f"k{LABEL} ext: tau_d={tau_d:.2e}s  start={lo / tau_d:.2f} tau_d  "
              f"window=[{lo:.4e}, {hi:.4e}]  alpha={alpha:.0f}  (a_fund={a_fund:.0f})")
    return {"label": LABEL, "lo": lo, "hi": hi, "alpha": alpha,
            "tau_decay": tau_d, "lo_grid": lo_grid, "alpha_grid": a_grid,
            "i_lo": i_lo, "alpha_fund": a_fund}


def acdc_plugin_alpha_sigma(sweep: dict, F: np.ndarray, fit_window,
                            n_boot: int = 100, rng=None):
    """rb.acdc_alpha_sigma generalized to an explicit sweep dict / window: plug-in
    assembly, binned-exp fit, per-bin Poisson bootstrap x KAPPA_S['098'] sigma.
    Works on both the 10 ms and 40 ms records (lag range sized by window+record).
    Returns (alpha, sigma, curve_fit_perr_alpha)."""
    rng = np.random.default_rng(rng)
    lo, hi = fit_window
    n_t = sweep["g_t"].shape[-1]
    n_lag = window_lag_count_ext(fit_window, n_t)

    def fit_one(s, want_perr=False):
        tau, R = rb.acdc_rossi(s, F, n_tau=n_lag)
        w = (tau >= lo) & (tau <= hi)
        popt, perr = rb.fit_alpha_binned(tau[w], R[w])
        return (popt[2], perr[2]) if want_perr else popt[2]

    alpha, perr = fit_one(sweep, want_perr=True)
    g_t = sweep["g_t"]
    boots = []
    for _ in range(n_boot):
        s2 = dict(sweep)
        s2["g_t"] = rng.poisson(g_t).astype(float)
        boots.append(fit_one(s2))
    sigma = float(np.std(boots)) * rb.KAPPA_S[LABEL] if boots else np.nan
    return float(alpha), sigma, float(perr)


def acdc_block_alphas_ext(events: dict, F_flat: np.ndarray, n_prime: int,
                          n_source_cell: int, fit_window, n_cells: int = 80,
                          max_blocks: int | None = None, time_edges=None):
    """rb.acdc_block_alphas with the time grid taken from the extended record
    (rb hardcodes np.linspace(0, T_MAX, N_T+1)). Same block logic, same
    cross-chain estimator (rb.acdc_rossi_events already accepts time_edges)."""
    lo, hi = fit_window
    n_per = max(int(n_prime // n_cells), 1)
    if time_edges is None:
        time_edges = ext_time_edges()
    n_lag = window_lag_count_ext(fit_window, len(time_edges) - 1)
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
        tau, R = rb.acdc_rossi_events(ev_b, F_flat, n_per, time_edges, n_tau=n_lag)
        w = (tau >= lo) & (tau <= hi)
        if np.any(R[w] > 0):
            try:
                popt, _ = rb.fit_alpha_binned(tau[w], R[w])
                if np.isfinite(popt[2]) and popt[2] < 0:
                    alphas.append(popt[2])
            except RuntimeError:
                pass
    return np.array(alphas), np.array(counts)


# --------------------------------------------------------------------------- #
# Stage: window
# --------------------------------------------------------------------------- #

def derive_window(verbose: bool = True) -> dict:
    """Plateau-rule window on the extended record; cached to WINDOW_JSON."""
    sweep = load_ext_sweep()
    F = rb.fission_coarse_sim(LABEL, *sweep["G"].shape)
    r = plateau_window_ext(sweep, F, verbose=verbose)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache = {"lo": r["lo"], "hi": r["hi"], "alpha_at_derivation": r["alpha"],
             "alpha_fund": r["alpha_fund"], "tau_decay": r["tau_decay"],
             "rule": "rb.plateau_window (width_decay=3, ratio=4, rtol=0.10) on the "
                     "40 ms record, full lag range"}
    WINDOW_JSON.write_text(json.dumps(cache, indent=2))
    if verbose:
        print(f"cached -> {WINDOW_JSON}")
    return r


def ext_window() -> tuple:
    """The extended-record k=0.98 fit window (lo, hi) [s]; derives + caches on
    first use."""
    if not WINDOW_JSON.exists():
        derive_window(verbose=False)
    d = json.loads(WINDOW_JSON.read_text())
    return (d["lo"], d["hi"])


# --------------------------------------------------------------------------- #
# Notebook-facing mirrors (benchmark-notebook rebuild, 2026-07-12; see
# outputs/presentation/K098_EXTENSION_REPORT.md §9.1). Each mirrors ONE audited
# rb function with the identical estimator/output contract, swapping only the
# hardcoded 10 ms grid / FIT_WINDOWS["098"] for the extended record + window,
# so build_benchmark_notebook.py can branch per label without touching rb.
# --------------------------------------------------------------------------- #

def acdc_alpha_sigma_ext(n_boot: int = 100, rng=None):
    """rb.acdc_alpha_sigma('098') on the extended record: full-statistics plug-in
    ACDC alpha on the 40 ms bench10M sweep + extended window, Poisson-boot x
    KAPPA_S['098'] sigma. Same contract: (alpha, sigma, sweep_total_histories).

    This is the extended-record high-stats consensus (report headline
    -370.7 +/- 4.7); the old 10 ms consensus was -336.2 (truncation artifact).
    """
    sweep = load_ext_sweep()
    F = rb.fission_coarse_sim(LABEL, *sweep["G"].shape)
    a, s, _ = acdc_plugin_alpha_sigma(sweep, F, ext_window(), n_boot=n_boot, rng=rng)
    n_r, n_th = sweep["G"].shape
    total_hist = int(n_r * n_th * float(sweep["n_source"]))
    return float(a), float(s), total_hist


def acdc_alpha_sigma_events_ext(events: dict, n_source_cell: int, F_flat: np.ndarray,
                                fit_window, n_boot: int = 100, rng=None):
    """rb.acdc_alpha_sigma_events with the 40 ms time grid (the sweep event files
    store ALL prompt times, so only the histogramming grid changes). Same
    cross-chain (U-statistic) mean + per-cell chain bootstrap, same return dict."""
    rng = np.random.default_rng(rng)
    lo, hi = fit_window
    time_edges = ext_time_edges()
    n_lag = window_lag_count_ext(fit_window, EXT_N_T)

    def fit_from(ev):
        tau, R = rb.acdc_rossi_events(ev, F_flat, n_source_cell, time_edges, n_tau=n_lag)
        w = (tau >= lo) & (tau <= hi)
        if not np.any(R[w] > 0):
            return np.nan
        try:
            popt, _ = rb.fit_alpha_binned(tau[w], R[w])
            return popt[2]
        except RuntimeError:
            return np.nan

    alpha = fit_from(events)
    prepared = {}
    for idx, (t, p) in events.items():
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


def reference_alpha_ext(events_path=None) -> float:
    """rb.reference_alpha('098') on the extended record: event-level (cross-chain)
    full-statistics ACDC alpha, 40 ms grid + extended window. Default events file
    is the bench10M sweep (the record whose histograms were extended)."""
    events, n_cell = rb.load_sweep_events(events_path or EVENTS_H5)
    F = rb.fission_coarse_sim(LABEL, 8, 10)
    res = acdc_alpha_sigma_events_ext(events, n_cell, F, ext_window(), n_boot=1)
    return float(res["alpha"])


def benchmark_row_ext(n_boot: int = 60, rng=None) -> dict:
    """rb.benchmark_row('098') under the record-extension fix: same 1M-budget
    data and audited estimators, but every fit uses the extended-record window
    (the raw-time BF/die estimators carry no record ceiling -- only the window
    changes) and the ACDC-1M entry is assembled on the 40 ms grid so its late-lag
    overlap integral is not truncated. Same flat dict contract."""
    rng = np.random.default_rng(rng)
    win = ext_window()
    meas_k, a_pk = rb._ifp_pk_alpha(LABEL)

    df = rb.load_chain_events(LABEL)                    # 1M global, prompt events
    oracle = rb.bf_alpha_sigma_events(df, win, n_boot=n_boot, rng=rng)
    realistic = rb.bf_realistic_alpha_sigma(df, rb.N_SOURCE_BF, win,
                                            n_boot=n_boot, rng=rng)
    die = rb.die_away_alpha_sigma(df, win, n_boot=n_boot, rng=rng)

    ev_path = RESULTS / f"greens-sweep-k{LABEL}-bench1M-events.h5"
    events, n_cell = rb.load_sweep_events(ev_path)
    F = rb.fission_coarse_sim(LABEL, 8, 10)
    acdc = acdc_alpha_sigma_events_ext(events, n_cell, F, win, n_boot=n_boot, rng=rng)

    return {
        "label": LABEL, "k": meas_k,
        "pk": a_pk.nominal_value, "pk_err": a_pk.std_dev,
        "acdc": acdc["alpha"], "acdc_err": acdc["sigma"],
        "bf_oracle": oracle["alpha"], "bf_oracle_err": oracle["sigma"],
        "bf_oracle_cens": oracle["censored"],
        "bf_real": realistic["alpha"], "bf_real_err": realistic["sigma"],
        "bf_real_cens": realistic["censored"],
        "die": die["alpha"], "die_err": die["sigma"],
    }


def doubles_window_pairs_1M_ext() -> int:
    """Same-chain pairs in the EXTENDED k=0.98 fit window at the 1M global budget
    (the pair-starvation panel row; the old-window value 7350 is superseded
    because the window moved to the extended-record plateau window)."""
    df = rb.load_chain_events(LABEL)
    return rb.doubles_window_pairs(df, ext_window(), rb.N_SOURCE_BF)


def alignment_gate_ext(verbose: bool = True) -> dict:
    """rb.alignment_gate for k=0.98 on the EXTENDED 40 ms record: identical
    audited machinery (rb.acdc_rossi / rb.bf_rossi / integral-ratio scale /
    display rebin), but ACDC assembled from the extended bench10M sweep and the
    Doubles overlay from the matched 10M sphere pulse, with the extended-record
    fit window. Output dict mirrors rb.alignment_gate so the notebook's gate
    plotting cell stays uniform across labels. Expected scale ~1.014 (was ~1.04
    on the truncated 10 ms record)."""
    sweep = load_ext_sweep()
    n_r, n_th = sweep["G"].shape
    F = rb.fission_coarse_sim(LABEL, n_r, n_th)
    n_t = sweep["g_t"].shape[-1]
    tau, R_acdc = rb.acdc_rossi(sweep, F, n_tau=n_t - 1)

    df = rb.load_chain_events_from(SPHERE_TRACK)
    tau_bf, R_bf, counts = rb.bf_rossi(df, N_SPHERE, n_tau=n_t - 1)
    assert np.allclose(tau, tau_bf)

    # audited scale rule: full usable range, skip lag bins 0-1 (tau is labelled
    # L*dt, so L >= 3 is tau >= 3*DT)
    full = (tau >= 3 * DT) & (R_acdc > 0)
    scale = float(R_bf[full].sum() / R_acdc[full].sum())
    M_emp, M_err = rb.calibrate_moment(LABEL, n_boot=200)

    lo, hi = ext_window()
    dts = rb.same_parent_pair_dts(df["time"].to_numpy(), df["parent_id"].to_numpy(),
                                  EXT_T_MAX)
    a_bf, s_bf, n_pairs = rb.fit_alpha_mle(dts, lo, hi, background="auto")
    fitw = (tau >= lo) & (tau <= hi)
    popt, perr = rb.fit_alpha_binned(tau[fitw], R_acdc[fitw])
    a_ac, s_ac = popt[2], perr[2]

    # display rebin exactly as rb.alignment_gate (>= ~5 expected pairs/point)
    win_bins = max(int(round((hi - lo) / DT)), 1)
    per_bin = n_pairs / win_bins
    rebin = int(np.clip(np.ceil(5.0 / max(per_bin, 1e-9)), 1, 64))
    n_keep = (len(counts) // rebin) * rebin
    c_re = counts[:n_keep].reshape(-1, rebin).sum(axis=1).astype(float)
    tau_re = tau[:n_keep].reshape(-1, rebin).mean(axis=1)
    dt_re = rebin * DT
    R_bf_re = c_re / N_SPHERE / dt_re
    R_bf_re_err = np.sqrt(c_re) / N_SPHERE / dt_re

    agree = abs(a_bf - a_ac) / np.hypot(s_bf, s_ac) if np.isfinite(s_bf) else np.nan
    result = {
        "label": LABEL, "n_pairs_window": int(n_pairs),
        "scale_bf_over_acdc": scale,
        "M_emp": M_emp, "M_err": M_err,
        "alpha_bf": a_bf, "alpha_bf_err": s_bf,
        "alpha_acdc": float(a_ac), "alpha_acdc_err": float(s_ac),
        "alpha_nsigma": float(agree),
        "tau": tau, "R_acdc": R_acdc, "R_bf": R_bf, "counts_bf": counts,
        "rebin": rebin, "tau_bf_rebin": tau_re,
        "R_bf_rebin": R_bf_re, "R_bf_rebin_err": R_bf_re_err,
    }
    if verbose:
        print(f"k{LABEL} (ext 40 ms, bench10M pair): window pairs={n_pairs:6d}  "
              f"scale={scale:5.3f}  M_emp={M_emp:4.2f}+/-{M_err:4.2f}  "
              f"alpha BF={a_bf:9.1f}+/-{s_bf:6.1f}  ACDC={a_ac:9.1f}+/-{s_ac:6.1f}"
              + (f"  ({agree:.2f} sigma)" if np.isfinite(agree) else "  (no BF fit)"))
    return result


# --------------------------------------------------------------------------- #
# Stage: alphas -- headline fits, alignment gate, half-window check, overlay
# --------------------------------------------------------------------------- #

def truncation_factor(tau, alpha, t_record):
    """Analytic finite-record suppression of an exponential-mode autocorrelation:
    R_T(tau)/R_inf(tau) = 1 - exp(-2|alpha| (T - tau))."""
    return 1.0 - np.exp(-2.0 * abs(alpha) * np.clip(t_record - tau, 0.0, None))


def run_alphas(n_boot: int = 100, seed: int = 7) -> dict:
    """Everything that needs only full-statistics fits (fast except the BF chain
    bootstraps): old-vs-new alpha_ACDC / alpha_BF, alignment-gate scale ratio on
    the extended record, half-window truncation check, overlay figure data."""
    rng = np.random.default_rng(seed)
    win_old = rb.FIT_WINDOWS[LABEL]
    win_new = ext_window()

    base = dict(np.load(BASE_NPZ)); base["label"] = LABEL
    swee = load_ext_sweep()
    F_sim = rb.fission_coarse_sim(LABEL, *swee["G"].shape)      # legacy 1M tally (audited default)
    F_10m = rb.fission_coarse_from_dir(SPHERE_DIR, *swee["G"].shape)  # matched 10M tally

    # --- assembled curves, full lag range of each record
    tau_old, R_old = rb.acdc_rossi(base, F_sim, n_tau=base["g_t"].shape[-1] - 1)
    tau_ext, R_ext = rb.acdc_rossi(swee, F_sim, n_tau=EXT_N_T - 1)
    _, R_ext_F10m = rb.acdc_rossi(swee, F_10m, n_tau=EXT_N_T - 1)

    # --- BF from the 10M sphere pulse (raw times -> no record ceiling)
    df = rb.load_chain_events_from(SPHERE_TRACK)
    tau_bf, R_bf, counts_bf = rb.bf_rossi(df, N_SPHERE, n_tau=EXT_N_T - 1)
    assert np.allclose(tau_ext, tau_bf)

    # --- alignment-gate scale ratios (audited rule: skip lag bins 0-1, R_acdc > 0;
    # tau is labelled L*dt, so L >= 3 is tau >= 3*DT)
    def scale(tau, Rb, Ra, t_hi=None):
        m = (tau >= 3 * DT) & (Ra > 0)
        if t_hi is not None:
            m &= tau <= t_hi
        return float(Rb[m].sum() / Ra[m].sum())

    scales = {
        "ext_full_Fsim": scale(tau_ext, R_bf, R_ext),
        "ext_full_F10m": scale(tau_ext, R_bf, R_ext_F10m),
        "ext_to10ms_Fsim": scale(tau_ext, R_bf, R_ext, t_hi=1.0e-2),
        "old_to10ms_Fsim": scale(tau_old, R_bf[: len(tau_old)], R_old),
    }

    # --- ACDC alpha fits (plug-in + Poisson-boot x kappa), old and new
    a_ac_old, s_ac_old, perr_old = acdc_plugin_alpha_sigma(base, F_sim, win_old,
                                                           n_boot=n_boot, rng=rng)
    a_ac_new, s_ac_new, perr_new = acdc_plugin_alpha_sigma(swee, F_sim, win_new,
                                                           n_boot=n_boot, rng=rng)
    # cross-check: same fits with the matched 10M fission tally (alpha is F-insensitive)
    a_ac_new_F10m, _, _ = acdc_plugin_alpha_sigma(swee, F_10m, win_new, n_boot=0)

    # cross-chain (U-statistic) point estimate on the extended record: verifies the
    # plug-in same-chain bias is negligible at these counts
    events, n_cell = rb.load_sweep_events(EVENTS_H5)
    tau_u, R_u = rb.acdc_rossi_events(
        events, F_sim, n_cell, ext_time_edges(),
        n_tau=window_lag_count_ext(win_new, EXT_N_T))
    wu = (tau_u >= win_new[0]) & (tau_u <= win_new[1])
    a_ac_u = float(rb.fit_alpha_binned(tau_u[wu], R_u[wu])[0][2])

    # --- BF alpha fits (audited estimator: frozen-w0 MLE + chain bootstrap)
    bf_old = rb.bf_alpha_sigma_events(df, win_old, n_boot=n_boot, rng=rng)
    bf_new = rb.bf_alpha_sigma_events(df, win_new, n_boot=n_boot, rng=rng)

    # --- PK reference
    k_meas, a_pk = rb._ifp_pk_alpha(LABEL)

    # --- half-window truncation check (the money check)
    def halves(sweep_d, F, lo, hi, n_b=60):
        mid = 0.5 * (lo + hi)
        a1, s1, _ = acdc_plugin_alpha_sigma(sweep_d, F, (lo, mid), n_boot=n_b, rng=rng)
        a2, s2, _ = acdc_plugin_alpha_sigma(sweep_d, F, (mid, hi), n_boot=n_b, rng=rng)
        return {"lo": lo, "mid": mid, "hi": hi, "alpha_early": a1, "sigma_early": s1,
                "alpha_late": a2, "sigma_late": s2,
                "nsigma": abs(a1 - a2) / float(np.hypot(s1, s2))}

    half_old = halves(base, F_sim, *win_old)
    half_new = halves(swee, F_sim, *win_new)

    out = {
        "win_old": list(win_old), "win_new": list(win_new),
        "scales": scales,
        "acdc": {"old": [a_ac_old, s_ac_old, perr_old],
                 "new": [a_ac_new, s_ac_new, perr_new],
                 "new_F10m_point": a_ac_new_F10m,
                 "new_crosschain_point": a_ac_u},
        "bf": {"old": bf_old, "new": bf_new},
        "pk": {"k": k_meas, "alpha": float(a_pk.nominal_value),
               "err": float(a_pk.std_dev)},
        "half_old": half_old, "half_new": half_new,
        "nsigma_acdc_bf_new": abs(a_ac_new - bf_new["alpha"])
                              / float(np.hypot(s_ac_new, bf_new["sigma"])),
        "nsigma_acdc_bf_old": abs(a_ac_old - bf_old["alpha"])
                              / float(np.hypot(s_ac_old, bf_old["sigma"])),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "alphas.json").write_text(json.dumps(out, indent=2, default=float))
    np.savez_compressed(OUT_DIR / "curves.npz",
                        tau_old=tau_old, R_old=R_old, tau_ext=tau_ext, R_ext=R_ext,
                        tau_bf=tau_bf, R_bf=R_bf, counts_bf=counts_bf,
                        R_ext_F10m=R_ext_F10m)

    print(json.dumps({k: v for k, v in out.items()}, indent=2, default=float))
    _overlay_figure(tau_old, R_old, tau_ext, R_ext, win_old, win_new,
                    out["acdc"]["new"][0])
    return out


def _overlay_figure(tau_old, R_old, tau_ext, R_ext, win_old, win_new, alpha_new):
    """Old-record vs extended-record assembled R(tau) + the analytic truncation
    factor applied to the extended curve (should reproduce the old droop)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    ax.axvspan(win_new[0] * 1e3, win_new[1] * 1e3, color="0.55", alpha=0.10, lw=0,
               label="extended fit window")
    m_ext = (tau_ext <= 14e-3) & (R_ext > 0)
    ax.semilogy(tau_ext[m_ext] * 1e3, R_ext[m_ext], color="#1F77B4", lw=2.2,
                label="extended record (40 ms)")
    m_old = R_old > 0
    ax.semilogy(tau_old[m_old] * 1e3, R_old[m_old], color="#C0392B", lw=1.8, ls="--",
                label="old record (10 ms) — late-lag droop")
    pred = R_ext * truncation_factor(tau_ext, alpha_new, 1.0e-2)
    m_p = (tau_ext <= 9.99e-3) & (pred > 0)
    ax.semilogy(tau_ext[m_p] * 1e3, pred[m_p], color="0.3", lw=1.2, ls=":",
                label=r"extended $\times\,(1 - e^{-2|\alpha|(10\,\mathrm{ms}-\tau)})$")
    ax.set_xlabel(r"delay $\tau$  [ms]")
    ax.set_ylabel(r"$R(\tau)$  [pairs / source / s]")
    ax.set_title("k = 0.98 ACDC assembly: 10 ms record truncation vs 40 ms rebuild")
    ax.legend(frameon=True, framealpha=0.9)
    ax.set_xlim(0, 14)
    fig.tight_layout()
    PRESENT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PRESENT_DIR / "k098_ext_overlay.png", dpi=200)
    plt.close(fig)
    print(f"overlay figure -> {PRESENT_DIR / 'k098_ext_overlay.png'}")


# --------------------------------------------------------------------------- #
# Stage: cost -- master-curve N->5% numbers, old window (reproduction) vs new
# --------------------------------------------------------------------------- #

def cost_numbers(which: str, target: float = 0.05, maxb: int = 500) -> dict:
    """Reproduce (which='old') or recompute (which='new') the k=0.98
    histories-to-5% figures with the notebook's exact methodology (disjoint
    parent-id blocks -> sigma_vs_budget -> histories_to_target).

    'old': vetted 10 ms pipeline verbatim -- FIT_WINDOWS['098'], 10 ms time grid
    (rb.acdc_block_alphas), alpha_ref = rb.acdc_alpha_sigma('098') high-stats
    consensus. Expected: ACDC ~390k, Doubles oracle ~1.49M, realistic ~2.85M.
    'new': extended window + 40 ms grid (acdc_block_alphas_ext), alpha_ref = the
    extended-record full-statistics ACDC alpha.
    """
    assert which in ("old", "new")
    df = rb.load_chain_events_from(SPHERE_TRACK)
    n_sph = int(df["parent_id"].max())
    events, n_cell = rb.load_sweep_events(EVENTS_H5)
    n_cells = len(events)
    n_sw = n_cell * n_cells
    F = rb.fission_coarse_from_dir(SPHERE_DIR, 8, 10)      # notebook Section-2c F

    if which == "old":
        win = rb.FIT_WINDOWS[LABEL]
        aref = rb.acdc_alpha_sigma(LABEL, n_boot=1)[0]     # high-stats consensus (old)
        acdc_fn = lambda n: rb.acdc_block_alphas(events, F, n, n_cell, win,
                                                 n_cells=n_cells, max_blocks=maxb)
    else:
        win = ext_window()
        aref = json.loads((OUT_DIR / "alphas.json").read_text())["acdc"]["new"][0]
        acdc_fn = lambda n: acdc_block_alphas_ext(events, F, n, n_cell, win,
                                                  n_cells=n_cells, max_blocks=maxb)

    t, p = df["time"].to_numpy(), df["parent_id"].to_numpy()
    w0_o = rb.flat_background_weight(rb.same_parent_pair_dts(t, p, win[1]), *win)
    w0_d = rb.flat_background_weight(t, *win)
    print(f"[{which}] window={tuple(round(x, 7) for x in win)}  |alpha_ref|={abs(aref):.1f}  "
          f"w0_oracle={w0_o:.3f}  w0_die={w0_d:.3f}")

    curves = {}
    curves["oracle"] = rb.sigma_vs_budget(
        lambda n: rb.oracle_block_alphas(df, n, n_sph, win, max_blocks=maxb,
                                         w_fixed=w0_o),
        rb.budget_ladder(n_sph), n_sph)
    print("[stage] oracle done")
    curves["realistic"] = rb.sigma_vs_budget(
        lambda n: rb.realistic_block_alphas(df, n, n_sph, win, rng=1, max_blocks=maxb),
        rb.budget_ladder(n_sph), n_sph)
    print("[stage] realistic done")
    curves["die"] = rb.sigma_vs_budget(
        lambda n: rb.die_block_alphas(df, n, n_sph, win, max_blocks=maxb,
                                      w_fixed=w0_d),
        rb.budget_ladder(n_sph), n_sph)
    print("[stage] die done")
    curves["acdc"] = rb.sigma_vs_budget(acdc_fn, rb.budget_ladder(n_sw), n_sw)
    print("[stage] acdc done")

    sca = {m: rb.fit_sigma_scaling(c, aref) for m, c in curves.items()}
    h2t = {m: rb.histories_to_target(curves[m], aref, target, sca[m]) for m in curves}

    result = {"which": which, "window": list(win), "alpha_ref": float(aref),
              "w0_oracle": float(w0_o), "w0_die": float(w0_d),
              "n_to_5pct": {m: float(h2t[m]["n_target"]) if h2t[m]["ok"] else None
                            for m in h2t},
              "anchors": {m: {"N": float(h2t[m]["anchor_N"]),
                              "sigma": float(h2t[m]["anchor_sigma"]),
                              "slope_diag": float(h2t[m]["slope"])} for m in h2t}}
    if h2t["oracle"]["ok"] and h2t["acdc"]["ok"]:
        result["oracle_over_acdc"] = h2t["oracle"]["n_target"] / h2t["acdc"]["n_target"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"cost_{which}.json").write_text(json.dumps(result, indent=2))
    for m, c in curves.items():
        c.to_csv(OUT_DIR / f"cost_{which}_{m}.csv", index=False)
    print(json.dumps(result, indent=2))
    return result


# --------------------------------------------------------------------------- #
# Stage: k095 -- numeric truncation-bias estimate only (no processing)
# --------------------------------------------------------------------------- #

def k095_estimate() -> dict:
    """Truncation-bias suppression at the k=0.95 window end for the 10 ms record:
    1 - exp(-2|alpha|(T_MAX - tau_hi)). Numeric only, per the follow-up rule."""
    lo, hi = rb.FIT_WINDOWS["095"]
    k, a_pk = rb._ifp_pk_alpha("095")
    rows = {}
    for name, a in [("pk_ifp", abs(a_pk.nominal_value)), ("nominal_770", 770.0)]:
        supp = 1.0 - truncation_factor(hi, a, rb.T_MAX)
        rows[name] = {"alpha_abs": a, "suppression_at_window_end": supp}
    out = {"window_ms": [lo * 1e3, hi * 1e3], "k_meas": k, "estimates": rows,
           "material": bool(max(r["suppression_at_window_end"]
                                for r in rows.values()) > 0.02)}
    print(json.dumps(out, indent=2))
    return out


# --------------------------------------------------------------------------- #
# Stage: cost-band -- the BAND intercept estimator (rb.fit_intercept_band) and its
# split-sample error bar on the extended k=0.98 record. ADDITIVE: cost_numbers
# above (the single-anchor rule) is untouched and stays the comparison baseline.
# --------------------------------------------------------------------------- #

def cost_inputs_ext(maxb: int = 500) -> dict:
    """cost_numbers('new')'s data path packaged as an rb cost-inputs dict.

    Identical ingredients: bench10M sphere collision track, the raw per-cell sweep
    event file, F from the audited fission_coarse_from_dir(SPHERE_DIR, 8, 10) path,
    the extended-record window ext_window(), alpha_ref from the cached
    outputs/k098_extension/alphas.json (-370.65), and the two frozen full-dataset
    background weights. Consumed by rb.cost_curves / rb.split_sample_costs with
    acdc_block_fn=acdc_block_alphas_ext so the ACDC ladder uses the 40 ms grid.
    """
    win = ext_window()
    df = rb.load_chain_events_from(SPHERE_TRACK)
    n_sph = int(df["parent_id"].max())
    events, n_cell = rb.load_sweep_events(EVENTS_H5)
    n_cells = len(events)
    F = rb.fission_coarse_from_dir(SPHERE_DIR, 8, 10)
    aref = json.loads((OUT_DIR / "alphas.json").read_text())["acdc"]["new"][0]
    t, p = df["time"].to_numpy(), df["parent_id"].to_numpy()
    return {"label": LABEL, "fit_window": win, "df": df, "n_sphere": n_sph,
            "events": events, "n_cell": n_cell, "n_cells": n_cells,
            "n_sweep_total": n_cell * n_cells, "F": F,
            "alpha_ref": float(aref), "maxb": int(maxb),
            "w0_oracle": rb.flat_background_weight(
                rb.same_parent_pair_dts(t, p, win[1]), *win),
            "w0_die": rb.flat_background_weight(t, *win)}


def cost_summary_ext(n_splits_list=(5, 2), target: float = 0.05,
                     slope: float = -0.5, rel_cap: float = rb.BAND_REL_CAP,
                     min_blocks: int = rb.BAND_MIN_BLOCKS, maxb: int = 500,
                     methods=("oracle", "realistic", "die", "acdc"),
                     verbose: bool = True) -> dict:
    """rb.cost_summary_flagship for the k=0.98 extended record.

    Same contract and same estimator; only the inputs (cost_inputs_ext) and the
    ACDC block function (acdc_block_alphas_ext, 40 ms grid) differ. The finest
    split count in which EVERY split yields >= rb.BAND_MIN_POINTS band points for a
    method is the one whose SE is quoted for that method.
    """
    inputs = cost_inputs_ext(maxb=maxb)
    aref = inputs["alpha_ref"]
    if verbose:
        print(f"[k098-ext] alpha_ref = {aref:.2f}  window = "
              f"{tuple(round(x, 7) for x in inputs['fit_window'])}  "
              f"n_sphere = {inputs['n_sphere']:,}  n_sweep = {inputs['n_sweep_total']:,}")
    curves = rb.cost_curves(inputs, methods, acdc_block_fn=acdc_block_alphas_ext)
    band = {m: rb.fit_intercept_band(curves[m], aref, target, slope, rel_cap,
                                     min_blocks) for m in methods}
    anchor = {m: rb.histories_to_target(curves[m], aref, target) for m in methods}
    splits, summaries = {}, {}
    for ns in n_splits_list:
        if verbose:
            print(f"[k098-ext] split-sample, {ns} splits")
        tab = rb.split_sample_costs(inputs, ns, methods, target, slope, rel_cap,
                                    min_blocks,
                                    acdc_block_fn=acdc_block_alphas_ext,
                                    verbose=verbose)
        splits[ns] = tab
        full = {f"n_target_{m}": band[m]["n_target"] for m in methods}
        full.update({f"anchor_{m}": anchor[m]["n_target"] for m in methods})
        if "acdc" in methods and np.isfinite(band["acdc"]["n_target"]):
            for m in ("oracle", "realistic", "die"):
                if m in methods:
                    full[f"ratio_{m}_acdc"] = (band[m]["n_target"]
                                               / band["acdc"]["n_target"])
        summaries[ns] = rb.split_cost_summary(tab, ns, full)
    chosen = {}
    for m in methods:
        good = [ns for ns in n_splits_list
                if (splits[ns][f"n_points_{m}"] >= rb.BAND_MIN_POINTS).all()
                and splits[ns][f"ok_{m}"].all()]
        chosen[m] = max(good) if good else None
    return {"inputs": inputs, "alpha_ref": aref, "curves": curves, "band": band,
            "anchor": anchor, "splits": splits, "summaries": summaries,
            "chosen": chosen, "target": target}


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "build"
    if stage == "build":
        build_extended_npz()
    elif stage == "window":
        derive_window()
    elif stage == "alphas":
        nb = int(sys.argv[2]) if len(sys.argv) > 2 else 100
        run_alphas(n_boot=nb)
    elif stage == "cost-old":
        cost_numbers("old")
    elif stage == "cost-new":
        cost_numbers("new")
    elif stage == "k095":
        k095_estimate()
    else:
        raise SystemExit(f"unknown stage {stage!r}")
