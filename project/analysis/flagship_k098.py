"""Flagship k=0.98 analysis on a NATIVE 40 ms record, path-parameterised.

WHY THIS EXISTS. `k098_record_extension` was written to repair one specific dataset:
it REBUILDS a 40 ms per-cell record from the bench10M event file (because the
bench10M sweep was histogrammed on a 10 ms grid) and hardwires every path to that
pair. The k=0.98 flagship campaign writes a sweep whose g_t is already 40 ms
(2048 bins, same DT), so nothing needs rebuilding -- only the paths change. This
module is that generalisation: the same audited estimators, taking the record
length and the time grid FROM THE NPZ, over a named dataset.

ADDITIVE. Every function in `k098_record_extension` is untouched and still runs;
the record-generic mirrors there (`plateau_window_ext`, `acdc_plugin_alpha_sigma`,
`acdc_block_alphas_ext`) are imported and reused verbatim -- they already take an
explicit sweep dict / time grid, so they are dataset-agnostic once fed one.

DATASETS
  bench100M : project/results/greens-sweep-k098-bench100M{.npz,-events.h5}
              + greens-function-k098-bench100M (100M-history sphere pulse).
              Native 40 ms record, no rebuild.
  bench10M  : the existing 10M pair -- the ext40ms npz rebuilt by
              k098_record_extension.build_extended_npz + the bench10M events and
              sphere dir. Present today, so every stage below is testable now and
              reproduces the published k=0.98 numbers.

STAGES (all take a dataset name; all reuse rb/kre code paths)
  inputs   sweep npz + per-cell events + sphere events + F + the npz's time grid
  window   plateau-rule fit window and fundamental alpha ON THIS DATASET
  alphas   ACDC (binned plug-in fit) vs Doubles parent-ID (frozen-w0 MLE) alpha
  gate     absolute-scale overlay sum(R_bf)/sum(R_acdc) over L >= 3, with the
           section-8.4 error bar (Poisson on the masked BF counts (+) M_emp)
  costs    BAND intercept estimator (rb.fit_intercept_band) for all four methods
           with split-sample errors -- rb.cost_summary_flagship's recipe

RUN (from repo root, PYTHONPATH=.):
  python -m project.analysis.flagship_k098 --dataset bench10M all
  python -m project.analysis.flagship_k098 --dataset bench100M costs

Stage outputs land in project/analysis/outputs/flagship_k098/ (git-ignored).
"""
import argparse
import json
from pathlib import Path

import numpy as np

import project.analysis.greens_loader as gl
import project.analysis.k098_record_extension as kre
import project.analysis.rossi_benchmark as rb

LABEL = "098"
DT = rb.DT
RESULTS = gl.RESULTS_ROOT
OUT_DIR = Path(__file__).resolve().parent / "outputs" / "flagship_k098"

# The vetted extended-record window (k098_record_extension.ext_window(), cached in
# outputs/k098_extension/ext_window.json): [1.3655, 9.5096] ms, alpha_fund -368.4.
# Carried here as a fixed comparison for whatever the plateau rule returns on a new
# dataset -- a large move would mean the record, not the estimator, changed.
VETTED_EXT_WINDOW = (0.001365524906003438, 0.009509626505586757)
VETTED_EXT_ALPHA_FUND = -368.3647561756218

DATASETS = {
    "bench100M": {
        "sweep_npz": RESULTS / "greens-sweep-k098-bench100M.npz",
        "events": RESULTS / "greens-sweep-k098-bench100M-events.h5",
        "sphere_dir": RESULTS / "greens-function-k098-bench100M",
    },
    "bench10M": {
        "sweep_npz": kre.EXT_NPZ,          # 40 ms rebuild of the 10 ms bench10M sweep
        "events": kre.EVENTS_H5,
        "sphere_dir": kre.SPHERE_DIR,
    },
}
DEFAULT_DATASET = "bench100M"

# Finest split count that still leaves every split >= rb.BAND_MIN_POINTS band points
# (COST_ESTIMATOR_REPORT sections 2-3: 10 x 10M at 100M, 5 x 2M at 10M).
DEFAULT_N_SPLITS = {"bench100M": 10, "bench10M": 5}

# Which fission map weights the ACDC assembly in the alignment-gate statistic.
#   "legacy"  = rb.fission_coarse_sim(LABEL) -- the legacy label-keyed 1M tally that
#               rb.alignment_gate / kre.alignment_gate_ext / make_gate_deck use, and
#               the one the published scale 1.009 was measured with.
#   "matched" = rb.fission_coarse_from_dir(sphere_dir) -- the tally from THIS
#               dataset's own sphere run, which is what the cost path uses.
# The two tallies differ by ~0.8% in normalisation (outputs/k098_extension/alphas.json
# records both), so gate() reports the statistic under both and defaults to "legacy"
# for continuity with the published gate.
GATE_F_DEFAULT = "legacy"

_CACHE: dict = {}


# --------------------------------------------------------------------------- #
# (a) inputs
# --------------------------------------------------------------------------- #

def paths(dataset: str = DEFAULT_DATASET) -> dict:
    """Resolved file paths for a named dataset (no I/O)."""
    if dataset not in DATASETS:
        raise KeyError(f"unknown dataset {dataset!r}; choose from {list(DATASETS)}")
    p = dict(DATASETS[dataset])
    p["dataset"] = dataset
    p["sphere_track"] = p["sphere_dir"] / "collision_track.h5"
    p["statepoint"] = p["sphere_dir"] / "statepoint.1.h5"
    return p


def available(dataset: str = DEFAULT_DATASET) -> bool:
    """True when every file the dataset needs is on disk."""
    p = paths(dataset)
    return all(p[k].exists() for k in ("sweep_npz", "events", "sphere_track",
                                       "statepoint"))


def resolve_dataset(preferred: str = DEFAULT_DATASET,
                    fallback: str = "bench10M") -> str:
    """`preferred` if its files exist, else `fallback` (used by the deck/notebook
    so they can be built before the flagship campaign lands)."""
    return preferred if available(preferred) else fallback


def inputs(dataset: str = DEFAULT_DATASET, maxb: int = 500,
           use_cache: bool = True) -> dict:
    """Load everything the k=0.98 stages need, for an arbitrary dataset.

    sweep       : the npz as a dict (gl._load_npz contract: arrays + label + k)
    events      : {(ir, ith): (times, parents)} via rb.load_sweep_events
    n_cell      : source histories per swept cell
    df          : sphere detector events (time, parent_id) via load_chain_events_from
    n_sphere    : max parent_id (what make_master_deck uses as the sphere budget)
    F           : rb.fission_coarse_from_dir(sphere_dir, 8, 10) -- the matched-tally
                  F, M_SIM-scaled, the audited cost-path fission map
    F_raw       : the same tally with moment 1 (raw fissions/source), the denominator
                  of the calibrate_moment identity
    F_sim       : rb.fission_coarse_sim(LABEL, 8, 10) -- the legacy 1M tally the
                  published alignment gate used (see GATE_F_DEFAULT)
    time_edges  : the record's OWN bin edges, taken from the npz (not a constant)

    The rest of the keys are rb's cost-inputs contract (fit_window is filled in by
    cost_inputs(), which is what rb.cost_curves / rb.split_sample_costs consume).
    """
    key = (dataset, int(maxb))
    if use_cache and key in _CACHE:
        return _CACHE[key]
    p = paths(dataset)
    missing = [str(p[k]) for k in ("sweep_npz", "events", "sphere_track", "statepoint")
               if not p[k].exists()]
    if missing:
        raise FileNotFoundError(f"dataset {dataset!r} is not on disk yet: {missing}")

    f = np.load(p["sweep_npz"], allow_pickle=False)
    sweep = {k: f[k] for k in f.files}
    sweep["label"] = LABEL
    sweep["k"] = float(sweep["k"])

    time_edges = np.asarray(sweep["time_edges"], dtype=float)
    dt = float(time_edges[1] - time_edges[0])
    if not np.isclose(dt, DT, rtol=1e-9):
        raise ValueError(f"{p['sweep_npz'].name}: bin width {dt:.6e} != rb.DT {DT:.6e}")
    n_t = int(sweep["g_t"].shape[-1])
    if n_t != len(time_edges) - 1:
        raise ValueError(f"{p['sweep_npz'].name}: g_t has {n_t} bins but "
                         f"{len(time_edges) - 1} edges")

    events, n_cell = rb.load_sweep_events(p["events"])
    n_cells = len(events)
    df = rb.load_chain_events_from(p["sphere_track"])
    n_sphere = int(df["parent_id"].max())
    n_r, n_th = sweep["G"].shape
    F = rb.fission_coarse_from_dir(p["sphere_dir"], n_r, n_th)

    out = {
        "dataset": dataset, "label": LABEL, "paths": p,
        "sweep": sweep, "time_edges": time_edges, "n_t": n_t, "dt": dt,
        "t_record": float(time_edges[-1]),
        "events": events, "n_cell": int(n_cell), "n_cells": n_cells,
        "n_sweep_total": int(n_cell) * n_cells,
        "df": df, "n_sphere": n_sphere,
        "F": F, "F_raw": F / rb.M_SIM,
        "F_sim": rb.fission_coarse_sim(LABEL, n_r, n_th),
        "maxb": int(maxb),
    }
    if use_cache:
        _CACHE[key] = out
    return out


def _F(inp: dict, which: str) -> np.ndarray:
    if which not in ("matched", "legacy"):
        raise ValueError(f"F source must be 'matched' or 'legacy', got {which!r}")
    return inp["F"] if which == "matched" else inp["F_sim"]


# --------------------------------------------------------------------------- #
# (b) window
# --------------------------------------------------------------------------- #

def window(dataset: str = DEFAULT_DATASET, f_source: str = "legacy",
           verbose: bool = False) -> dict:
    """Plateau-rule fit window on THIS dataset's own ACDC curve.

    kre.plateau_window_ext verbatim (plateau scan -> band-median fundamental ->
    earliest sustained on-plateau start -> 3 decay times), which already sizes its
    lag range from the sweep's own record instead of a module constant. The vetted
    extended-record window is returned alongside for comparison.
    """
    inp = inputs(dataset)
    r = kre.plateau_window_ext(inp["sweep"], _F(inp, f_source), verbose=verbose)
    lo, hi = float(r["lo"]), float(r["hi"])
    return {
        "dataset": dataset, "f_source": f_source,
        "window": (lo, hi), "lo": lo, "hi": hi,
        "alpha_fund": float(r["alpha_fund"]),
        "alpha_at_derivation": float(r["alpha"]),
        "tau_decay": float(r["tau_decay"]),
        "vetted_ext_window": VETTED_EXT_WINDOW,
        "vetted_ext_alpha_fund": VETTED_EXT_ALPHA_FUND,
        "d_lo_rel": lo / VETTED_EXT_WINDOW[0] - 1.0,
        "d_hi_rel": hi / VETTED_EXT_WINDOW[1] - 1.0,
        "lo_grid": r["lo_grid"], "alpha_grid": r["alpha_grid"],
    }


def _window_of(dataset, win):
    if win is None:
        return window(dataset)["window"]
    if isinstance(win, dict):
        return tuple(win["window"])
    return (float(win[0]), float(win[1]))


# --------------------------------------------------------------------------- #
# (c) alphas
# --------------------------------------------------------------------------- #

def alphas(dataset: str = DEFAULT_DATASET, win=None, n_boot: int = 100,
           rng=None, f_source: str = "legacy") -> dict:
    """ACDC and Doubles(parent-ID) alpha on the same window, with their agreement.

    ACDC   : kre.acdc_plugin_alpha_sigma -- plug-in assembly on the record's own
             grid, binned A + B exp(alpha tau) fit, per-bin Poisson bootstrap x
             KAPPA_S['098'] for sigma. The curve_fit parameter error is returned too
             (`sigma_fit`); it is the +/-0.45-class number the gate reports quote,
             an order of magnitude tighter than the honest bootstrap sigma.
    Doubles: rb.bf_alpha_sigma_events -- same-chain pair-separation MLE with the
             delayed floor w0 frozen from the full dataset, chain-bootstrap sigma.
             The MLE Fisher sigma under background='auto' (the number
             kre.alignment_gate_ext prints) is returned as `sigma_fisher`.
    """
    inp = inputs(dataset)
    lo, hi = _window_of(dataset, win)
    a_ac, s_ac, perr = kre.acdc_plugin_alpha_sigma(
        inp["sweep"], _F(inp, f_source), (lo, hi), n_boot=n_boot, rng=rng)

    bf = rb.bf_alpha_sigma_events(inp["df"], (lo, hi), n_boot=n_boot, rng=rng)
    dts = rb.same_parent_pair_dts(inp["df"]["time"].to_numpy(),
                                  inp["df"]["parent_id"].to_numpy(),
                                  inp["t_record"])
    a_mle, s_fisher, n_pairs = rb.fit_alpha_mle(dts, lo, hi, background="auto")

    s_bf = bf["sigma"]
    nsig = (abs(a_ac - bf["alpha"]) / float(np.hypot(s_ac, s_bf))
            if np.isfinite(s_bf) else np.nan)
    nsig_fit = (abs(a_ac - bf["alpha"]) / float(np.hypot(perr, s_fisher))
                if np.isfinite(s_fisher) else np.nan)
    return {
        "dataset": dataset, "window": (lo, hi), "f_source": f_source,
        "acdc": float(a_ac), "acdc_sigma": float(s_ac), "acdc_sigma_fit": float(perr),
        "bf": float(bf["alpha"]), "bf_sigma": float(s_bf),
        "bf_sigma_fisher": float(s_fisher), "bf_alpha_mle_auto": float(a_mle),
        "bf_censored": bool(bf["censored"]), "n_pairs_window": int(n_pairs),
        "nsigma": float(nsig), "nsigma_fit_errors": float(nsig_fit),
    }


# --------------------------------------------------------------------------- #
# (d) alignment gate: absolute scale + its error bar
# --------------------------------------------------------------------------- #

def moment(dataset: str = DEFAULT_DATASET, n_boot: int = 200, rng=None,
           f_source: str = "matched") -> dict:
    """rb.calibrate_moment's identity on an arbitrary dataset.

        M_emp = (ordered same-parent detection pairs / n_source)
                / sum_i F_raw_i G_i^2

    with F_raw the RAW fissions-per-source map (moment 1) and G_i the sweep's
    time-integrated detections per source per cell. rb.calibrate_moment hardwires
    the legacy 1M global run and gl.fission_coarse(label); this takes both from the
    dataset. Chain bootstrap over the per-chain detection multiplicities for sigma.
    """
    inp = inputs(dataset)
    rng = np.random.default_rng(rng)
    F_raw = inp["F_raw"] if f_source == "matched" else inp["F_sim"] / rb.M_SIM
    G = inp["sweep"]["n_prompt"] / float(inp["sweep"]["n_source"])
    denom = float((F_raw * G ** 2).sum())
    m = inp["df"].groupby("parent_id").size().to_numpy()
    n_src = float(inp["n_sphere"])

    def stat(mm):
        return float((mm * (mm - 1)).sum() / n_src / denom)

    M = stat(m)
    boots = [stat(m[rng.integers(0, len(m), len(m))]) for _ in range(int(n_boot))]
    return {"M_emp": M, "M_err": float(np.std(boots)), "denom": denom,
            "n_chains_detected": int(len(m)), "f_source": f_source}


def gate(dataset: str = DEFAULT_DATASET, win=None,
         f_source: str = GATE_F_DEFAULT, n_boot: int = 200, rng=None) -> dict:
    """Absolute-scale overlay statistic sum(R_bf)/sum(R_acdc) with an error bar.

    Mask: tau >= 3*DT and R_acdc > 0 -- the audited lag set L >= 3 of
    rb.alignment_gate after the lag-axis fix (tau is labelled L*dt).

    Error bar, LAG_AXIS_FIX_REPORT section 8.4:
        rel = hypot(1/sqrt(BF pair counts inside the mask), M_err / M_emp)
    i.e. counting statistics on the numerator combined with the relative error of
    the empirical pair moment that sets the ACDC assembly's absolute normalisation.
    The Poisson term is a LOWER bound (one chain can contribute several pairs), so
    the true band is somewhat wider.

    The statistic is reported under BOTH fission maps (see GATE_F_DEFAULT): the
    legacy 1M tally the published gate used, and this dataset's matched tally.
    """
    inp = inputs(dataset)
    lo, hi = _window_of(dataset, win)
    n_t = inp["n_t"]
    df = inp["df"]
    tau_bf, R_bf, counts = rb.bf_rossi(df, inp["n_sphere"], tau_max=inp["t_record"],
                                       n_tau=n_t - 1)
    mom = moment(dataset, n_boot=n_boot, rng=rng)

    scales = {}
    for which in ("legacy", "matched"):
        tau, R_acdc = rb.acdc_rossi(inp["sweep"], _F(inp, which),
                                    tau_max=inp["t_record"], n_tau=n_t - 1)
        assert np.allclose(tau, tau_bf)
        mask = (tau >= 3 * DT) & (R_acdc > 0)
        scales[which] = {
            "scale": float(R_bf[mask].sum() / R_acdc[mask].sum()),
            "n_pairs_mask": int(counts[mask].sum()),
            "n_lags_mask": int(mask.sum()),
        }
        if which == f_source:
            tau_keep, R_keep, mask_keep = tau, R_acdc, mask

    s = scales[f_source]
    rel_poisson = 1.0 / np.sqrt(max(s["n_pairs_mask"], 1))
    rel_moment = mom["M_err"] / mom["M_emp"]
    rel = float(np.hypot(rel_poisson, rel_moment))
    return {
        "dataset": dataset, "window": (lo, hi), "f_source": f_source,
        "scale": s["scale"], "scale_err": s["scale"] * rel, "rel_err": rel,
        "rel_poisson": float(rel_poisson), "rel_moment": float(rel_moment),
        "band": (s["scale"] * (1 - rel), s["scale"] * (1 + rel)),
        "nsigma_vs_unity": float(abs(s["scale"] - 1.0) / (s["scale"] * rel)),
        "n_pairs_mask": s["n_pairs_mask"], "n_lags_mask": s["n_lags_mask"],
        "M_emp": mom["M_emp"], "M_err": mom["M_err"],
        "n_chains_detected": mom["n_chains_detected"],
        "scale_by_F": {k: v["scale"] for k, v in scales.items()},
        "tau": tau_keep, "R_acdc": R_keep, "R_bf": R_bf, "counts_bf": counts,
        "mask": mask_keep,
    }


# --------------------------------------------------------------------------- #
# (e) costs: band intercept estimator + split-sample errors
# --------------------------------------------------------------------------- #

def _acdc_block_fn(time_edges):
    """kre.acdc_block_alphas_ext bound to a record's own time grid, in the
    signature rb.cost_curves / rb.split_sample_costs call."""
    def fn(events, F_flat, n_prime, n_source_cell, fit_window, n_cells=80,
           max_blocks=None):
        return kre.acdc_block_alphas_ext(events, F_flat, n_prime, n_source_cell,
                                         fit_window, n_cells=n_cells,
                                         max_blocks=max_blocks,
                                         time_edges=time_edges)
    return fn


def cost_inputs(dataset: str = DEFAULT_DATASET, win=None, alpha_ref=None,
                maxb: int = 500) -> dict:
    """rb's cost-inputs contract for a k=0.98 dataset (what cost_inputs_ext builds
    for the bench10M pair, with the paths and the time grid taken from `dataset`).

    alpha_ref defaults to the plug-in ACDC alpha measured on this dataset's own
    record and window (n_boot=0: only the point estimate enters the target).
    """
    inp = inputs(dataset, maxb=maxb)
    lo, hi = _window_of(dataset, win)
    if alpha_ref is None:
        alpha_ref = kre.acdc_plugin_alpha_sigma(inp["sweep"], _F(inp, "legacy"),
                                                (lo, hi), n_boot=0)[0]
    t = inp["df"]["time"].to_numpy()
    p = inp["df"]["parent_id"].to_numpy()
    return {"label": LABEL, "fit_window": (lo, hi), "df": inp["df"],
            "n_sphere": inp["n_sphere"], "events": inp["events"],
            "n_cell": inp["n_cell"], "n_cells": inp["n_cells"],
            "n_sweep_total": inp["n_sweep_total"], "F": inp["F"],
            "alpha_ref": float(alpha_ref), "maxb": int(maxb),
            "w0_oracle": rb.flat_background_weight(
                rb.same_parent_pair_dts(t, p, hi), lo, hi),
            "w0_die": rb.flat_background_weight(t, lo, hi),
            "time_edges": inp["time_edges"]}


def costs(dataset: str = DEFAULT_DATASET, win=None, n_splits: int | None = None,
          alpha_ref=None, target: float = 0.05, slope: float = -0.5,
          rel_cap: float = rb.BAND_REL_CAP, min_blocks: int = rb.BAND_MIN_BLOCKS,
          maxb: int = 500, methods=("oracle", "realistic", "die", "acdc"),
          verbose: bool = True) -> dict:
    """rb.cost_summary_flagship's recipe on a k=0.98 dataset.

    Full-data BAND intercept (rb.fit_intercept_band: fixed -1/2 weighted fit through
    every asymptotic, populated ladder point) plus the incumbent single-anchor
    rb.histories_to_target for comparison, plus the honest split-sample error bar
    from n_splits DISJOINT sub-datasets (rb.split_sample_costs). The ACDC ladder
    uses this record's own time grid via _acdc_block_fn.
    """
    n_splits = int(n_splits or DEFAULT_N_SPLITS.get(dataset, 10))
    ci = cost_inputs(dataset, win, alpha_ref=alpha_ref, maxb=maxb)
    aref = ci["alpha_ref"]
    acdc_fn = _acdc_block_fn(ci["time_edges"])
    if verbose:
        print(f"[{dataset}] alpha_ref = {aref:.2f}  window = "
              f"({ci['fit_window'][0]:.7f}, {ci['fit_window'][1]:.7f})  "
              f"n_sphere = {ci['n_sphere']:,}  n_sweep = {ci['n_sweep_total']:,}")
    curves = rb.cost_curves(ci, methods, acdc_block_fn=acdc_fn)
    band = {m: rb.fit_intercept_band(curves[m], aref, target, slope, rel_cap,
                                     min_blocks) for m in methods}
    anchor = {m: rb.histories_to_target(curves[m], aref, target) for m in methods}
    if verbose:
        print(f"[{dataset}] split-sample, {n_splits} splits")
    tab = rb.split_sample_costs(ci, n_splits, methods, target, slope, rel_cap,
                               min_blocks, acdc_block_fn=acdc_fn, verbose=verbose)
    full = {f"n_target_{m}": band[m]["n_target"] for m in methods}
    full.update({f"anchor_{m}": anchor[m]["n_target"] for m in methods})
    if "acdc" in methods and np.isfinite(band["acdc"]["n_target"]):
        for m in ("oracle", "realistic", "die"):
            if m in methods:
                full[f"ratio_{m}_acdc"] = (band[m]["n_target"]
                                           / band["acdc"]["n_target"])
    summary = rb.split_cost_summary(tab, n_splits, full)
    usable = {m: bool((tab[f"n_points_{m}"] >= rb.BAND_MIN_POINTS).all()
                      and tab[f"ok_{m}"].all()) for m in methods}
    return {"dataset": dataset, "inputs": ci, "alpha_ref": aref, "curves": curves,
            "band": band, "anchor": anchor, "n_splits": n_splits, "splits": tab,
            "summary": summary, "splits_all_usable": usable, "target": target}


# --------------------------------------------------------------------------- #
# (f) CLI
# --------------------------------------------------------------------------- #

def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()
                if k not in ("rows", "curves", "inputs", "splits", "summary",
                             "tau", "R_acdc", "R_bf", "counts_bf", "mask",
                             "lo_grid", "alpha_grid", "paths")}
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


def _summary_row(res_costs, m):
    b, a = res_costs["band"][m], res_costs["anchor"][m]
    s = res_costs["summary"]
    row = s[s["quantity"] == f"n_target_{m}"]
    mean = float(row["mean"].iloc[0]) if len(row) else np.nan
    se = float(row["se_full"].iloc[0]) if len(row) else np.nan
    used = int(row["n_splits_used"].iloc[0]) if len(row) else 0
    return {"method": m, "band": b["n_target"], "band_points": b["n_points"],
            "band_n_min": b["n_min"], "band_n_max": b["n_max"],
            "slope_free": b["slope_free"], "slope_free_se": b["slope_free_se"],
            "split_mean": mean, "split_se_full": se, "splits_used": used,
            "all_splits_usable": res_costs["splits_all_usable"][m],
            "anchor": a["n_target"] if a["ok"] else np.nan}


def run_all(dataset: str = DEFAULT_DATASET, n_splits: int | None = None,
            n_boot: int = 100, seed: int = 7, write: bool = True) -> dict:
    """Stages (b)-(e) end to end; writes <dataset>_summary.json and prints a table."""
    inp = inputs(dataset)
    print(f"[{dataset}] sweep {inp['paths']['sweep_npz'].name}: "
          f"g_t {inp['sweep']['g_t'].shape}, record {inp['t_record'] * 1e3:.1f} ms, "
          f"{inp['n_cell']:,}/cell x {inp['n_cells']} = {inp['n_sweep_total']:,}; "
          f"sphere {inp['n_sphere']:,} histories")

    w = window(dataset)
    print(f"[{dataset}] window = [{w['lo'] * 1e3:.4f}, {w['hi'] * 1e3:.4f}] ms "
          f"(vetted ext [{VETTED_EXT_WINDOW[0] * 1e3:.4f}, "
          f"{VETTED_EXT_WINDOW[1] * 1e3:.4f}]; {w['d_lo_rel'] * 100:+.2f}% / "
          f"{w['d_hi_rel'] * 100:+.2f}%)   alpha_fund = {w['alpha_fund']:.1f} "
          f"(vetted {VETTED_EXT_ALPHA_FUND:.1f})")

    a = alphas(dataset, w["window"], n_boot=n_boot, rng=seed)
    print(f"[{dataset}] alpha ACDC = {a['acdc']:.2f} +/- {a['acdc_sigma']:.2f} "
          f"(fit err {a['acdc_sigma_fit']:.2f})   "
          f"Doubles = {a['bf']:.2f} +/- {a['bf_sigma']:.2f} "
          f"(Fisher {a['bf_sigma_fisher']:.2f})   "
          f"agree to {a['nsigma']:.2f} sigma")

    g = gate(dataset, w["window"], rng=seed)
    print(f"[{dataset}] gate scale Doubles/ACDC = {g['scale']:.4f} +/- "
          f"{g['scale_err']:.4f} ({g['rel_err'] * 100:.1f}%: Poisson "
          f"{g['rel_poisson'] * 100:.1f}% (+) M {g['rel_moment'] * 100:.1f}%)   "
          f"{g['nsigma_vs_unity']:.2f} sigma from unity   "
          f"M_emp = {g['M_emp']:.2f} +/- {g['M_err']:.2f}   "
          f"pairs in mask = {g['n_pairs_mask']:,}   "
          f"[F {g['f_source']}; matched-F scale "
          f"{g['scale_by_F']['matched']:.4f}]")

    c = costs(dataset, w["window"], n_splits=n_splits, alpha_ref=a["acdc"])
    rows = [_summary_row(c, m) for m in ("acdc", "oracle", "realistic", "die")]
    print(f"\n[{dataset}] histories to sigma_alpha = 5%|alpha|  "
          f"(band estimator, {c['n_splits']} splits)")
    head = (f"{'method':10s} {'band N':>10s} {'pts':>4s} {'split mean':>11s} "
            f"{'+/- SE':>9s} {'all ok':>7s} {'anchor N':>10s} {'band/ACDC':>10s}")
    print(head)
    print("-" * len(head))
    n_acdc = c["band"]["acdc"]["n_target"]
    for r in rows:
        ratio = (r["band"] / n_acdc if np.isfinite(n_acdc) and n_acdc else np.nan)
        print(f"{r['method']:10s} {r['band']:10.4g} {r['band_points']:4d} "
              f"{r['split_mean']:11.4g} {r['split_se_full']:9.3g} "
              f"{str(r['all_splits_usable']):>7s} {r['anchor']:10.4g} "
              f"{ratio:10.4g}")

    out = {"dataset": dataset, "n_splits": c["n_splits"],
           "window": _jsonable(w), "alphas": _jsonable(a), "gate": _jsonable(g),
           "alpha_ref": c["alpha_ref"],
           "costs": {r["method"]: _jsonable(r) for r in rows},
           "ratios_vs_acdc": {r["method"]: (r["band"] / n_acdc)
                              for r in rows if np.isfinite(n_acdc)},
           "band_rel_cap": rb.BAND_REL_CAP, "band_min_blocks": rb.BAND_MIN_BLOCKS}
    if write:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        path = OUT_DIR / f"{dataset}_summary.json"
        path.write_text(json.dumps(out, indent=2, default=float))
        print(f"\nwrote {path}")
    return {"summary": out, "window": w, "alphas": a, "gate": g, "costs": c}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", nargs="?", default="all",
                    choices=["all", "inputs", "window", "alphas", "gate", "costs"])
    ap.add_argument("--dataset", default=DEFAULT_DATASET, choices=list(DATASETS))
    ap.add_argument("--n-splits", type=int, default=None)
    ap.add_argument("--n-boot", type=int, default=100)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args(argv)
    ds = args.dataset

    if args.stage == "all":
        run_all(ds, n_splits=args.n_splits, n_boot=args.n_boot, seed=args.seed,
                write=not args.no_write)
    elif args.stage == "inputs":
        inp = inputs(ds)
        print(json.dumps({"dataset": ds, "sweep": str(inp["paths"]["sweep_npz"]),
                          "record_ms": inp["t_record"] * 1e3, "n_t": inp["n_t"],
                          "n_cell": inp["n_cell"], "n_cells": inp["n_cells"],
                          "n_sweep_total": inp["n_sweep_total"],
                          "n_sphere": inp["n_sphere"],
                          "n_detector_events": int(len(inp["df"]))}, indent=2))
    elif args.stage == "window":
        print(json.dumps(_jsonable(window(ds, verbose=True)), indent=2))
    elif args.stage == "alphas":
        print(json.dumps(_jsonable(alphas(ds, n_boot=args.n_boot, rng=args.seed)),
                         indent=2))
    elif args.stage == "gate":
        print(json.dumps(_jsonable(gate(ds, rng=args.seed)), indent=2))
    elif args.stage == "costs":
        c = costs(ds, n_splits=args.n_splits)
        for m in ("acdc", "oracle", "realistic", "die"):
            print(json.dumps(_jsonable(_summary_row(c, m))))


if __name__ == "__main__":
    main()
