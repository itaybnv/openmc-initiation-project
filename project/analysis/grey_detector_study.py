"""Grey-detector study: the benchmark at realistic detector efficiency, fixed geometry.

WHY. The black B-10 sphere (100 atom/b-cm) captures every entering neutron, so its
efficiency is pure geometry and the detector is the WORST case for the Green's-
function method: pairs are plentiful (mu = 3.3e-3 captures per fission at k = 0.90,
30 to 300x the paper's research-reactor example). Real counters are grey 1/v
absorbers that catch a fraction of the entering neutrons. This module dials the
B-10 density down at fixed geometry and asks what each method costs.

HOW, WITHOUT NEW TRANSPORT. Every black-detector capture row carries the neutron's
incoming energy E, its position r (within microns of the detector surface) and its
direction u. A grey detector of density N captures that same entering neutron with
probability

    w = 1 - exp(-N sigma_a(E) * chord(r, u)),   chord = remaining path to the
                                                 far surface along u,

independently of everything else (a dilute gas: scattering negligible; an
uncaptured neutron leaves into the void and never returns, exactly as the black
detector removed it, so the chain physics is identical). Slow neutrons, the only
ones a grey detector catches, are captured on first contact in the black detector,
so (r, u) are the true entry point and direction for them. Therefore:

  * ANALOG grey detector  = Bernoulli THINNING of the black event list with w.
    Pairs and the analog assembly run on the thinned events through the production
    cost pipeline (rossi_benchmark: oracle_block_alphas, acdc_block_alphas,
    fit_intercept_band, split-sample errors) unchanged.
  * EXPECTED-VALUE (tally) estimate of the grey g_i(t) = the black events WEIGHTED
    by w: this is what a track-length capture tally scores (every neutron that
    traverses the detector contributes Sigma_a * path), and it keeps parent_id so
    the same disjoint-block error model applies (acdc_block_alphas_w below).
  * Validation: two real grey transport runs (10M, N = 1e-4 and 3e-5, default seed =
    block 0 of the 100M black run) are compared with the thinning prediction on
    singles, g(t) and window pairs (stage `validate`).

THE CLAIM BEING TESTED. Pair counting costs ~ 1/eps^2 (pairs die), the analog
assembly ~ 1/eps (captures thin), the tally assembly is FLAT in eps (its variance is
set by the ENTERING population, unchanged). So the tallies-vs-events decision flips
exactly where real detectors live, while the assembly still lands on the (thinned)
pair truth with an energy-dependent response (stage `gate`).

STAGES (repo root, PYTHONPATH=.):
  python -m project.analysis.grey_detector_study sphere     # pairs cost vs eps (now)
  python -m project.analysis.grey_detector_study sweep      # assembly costs (needs ev2)
  python -m project.analysis.grey_detector_study gate       # overlay at two eps
  python -m project.analysis.grey_detector_study validate   # real grey runs vs thinning
  python -m project.analysis.grey_detector_study figure
Outputs: project/analysis/outputs/grey_detector_study/ (git-ignored).
"""
from __future__ import annotations

import argparse
import json
import time as _time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

import project.analysis.rossi_benchmark as rb
from project.analysis.grey_detector_preview import b10_sigma_a
from project.core.rossi_mesh import KEFF_CONFIGS

LABEL = "090"
R_CORE = KEFF_CONFIGS[0.90]["radius"]
DET_C = np.array([0.0, 0.0, R_CORE + 10.0 + 5.0])
DET_R = 5.0
DENSITIES = (1e-1, 3e-2, 1e-2, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6)   # atom/b-cm; "black" = 100 reference
RESULTS = rb.RESULTS_ROOT
SPHERE_TRACK = rb.FLAGSHIP_DOUBLES_TRACK          # k=0.90 100M black sphere run
SPHERE_DIR = rb.FLAGSHIP_DOUBLES_DIR
import os
SWEEP_EV2 = Path(os.environ.get("GDS_SWEEP_EV2", RESULTS / "greens-sweep-k090-bench100M-ev2-events.h5"))
GREY_RUNS = {1e-4: RESULTS / "greens-function-k090-b10d1e-04-bench10M",
             3e-5: RESULTS / "greens-function-k090-b10d3e-05-bench10M"}
OUT_DIR = Path(__file__).resolve().parent / "outputs" / "grey_detector_study"
MAXB = 60          # blocks per budget (SE on sigma 9 %): enough for a cost scan
N_SPLITS = 4
THIN_SEED = 20261014

_CACHE: dict = {}


# --------------------------------------------------------------------------- #
# capture probability of an entering neutron in a grey sphere
# --------------------------------------------------------------------------- #

def chord(r: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Remaining path from point r along u to the detector sphere's far surface."""
    d = r - DET_C
    du = (d * u).sum(1)
    disc = du**2 - ((d * d).sum(1) - DET_R**2)
    return -du + np.sqrt(np.maximum(disc, 0.0))


def capture_weight(E: np.ndarray, r: np.ndarray, u: np.ndarray, density: float,
                   sig: np.ndarray | None = None) -> np.ndarray:
    sig = b10_sigma_a(E) if sig is None else sig
    return 1.0 - np.exp(-float(density) * sig * chord(r, u))


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #

def load_sphere() -> dict:
    """Prompt detector captures of the 100M black sphere run with E, r, u."""
    if "sphere" in _CACHE:
        return _CACHE["sphere"]
    with h5py.File(SPHERE_TRACK, "r") as f:
        a = f["collision_track_bank"]
        sel = (a["cell_id"][()] == 2) & (a["delayed_group"][()] == 0)
        t, p, E = a["time"][()][sel], a["parent_id"][()][sel], a["E"][()][sel]
        rr, uu = a["r"][()][sel], a["u"][()][sel]
    r = np.stack([rr["x"], rr["y"], rr["z"]], 1)
    u = np.stack([uu["x"], uu["y"], uu["z"]], 1)
    out = {"t": t, "p": p, "E": E, "r": r, "u": u, "n_source": int(p.max()),
           "sig": b10_sigma_a(E)}
    _CACHE["sphere"] = out
    return out


def load_sweep_ev2() -> tuple[dict, int]:
    """{(ir, ith): (t, p, E, r, u)} prompt events of the re-run k=0.90 100M sweep."""
    if "sweep" in _CACHE:
        return _CACHE["sweep"]
    if not SWEEP_EV2.exists():
        raise FileNotFoundError(f"{SWEEP_EV2} not there yet (chain.sh writes it)")
    out, n_source = {}, None
    with h5py.File(SWEEP_EV2, "r") as f:
        for name, grp in f.items():
            _, ir, ith = name.split("_")
            sel = grp["delayed_group"][()] == 0
            t, p, E = grp["time"][()][sel], grp["parent_id"][()][sel], grp["E"][()][sel]
            r, u = grp["r"][()][sel], grp["u"][()][sel]
            out[(int(ir), int(ith))] = (t, p, E, r, u, b10_sigma_a(E))
            n_source = int(grp.attrs["n_source"])
    _CACHE["sweep"] = (out, n_source)
    return out, n_source


def sphere_weights(density: float) -> np.ndarray:
    s = load_sphere()
    if density >= 100:
        return np.ones_like(s["t"])
    return capture_weight(s["E"], s["r"], s["u"], density, s["sig"])


def thin_mask(w: np.ndarray, seed: int = THIN_SEED) -> np.ndarray:
    return np.random.default_rng(seed).random(len(w)) < w


# --------------------------------------------------------------------------- #
# weighted (expected-value) ACDC: rb.acdc_rossi_events / acdc_block_alphas with w
# --------------------------------------------------------------------------- #

def acdc_rossi_events_w(events_w: dict, F_flat: np.ndarray, n_source_cell: int,
                        time_edges, n_tau: int):
    """rb.acdc_rossi_events with per-event weights: histograms are weight sums and
    the same-chain subtraction removes w_a * w_b products."""
    dt = time_edges[1] - time_edges[0]
    n_t = len(time_edges) - 1
    n_lag = min(n_tau, n_t - 1)
    tau_centers = np.arange(1, n_lag + 1) * dt
    R = np.zeros(n_lag)
    N = float(n_source_cell)
    for idx, (t, par, w) in events_w.items():
        if len(t) < 2:
            continue
        h, _ = np.histogram(t, bins=time_edges, weights=w)
        total = np.array([np.dot(h[:-L], h[L:]) for L in range(1, n_lag + 1)], float)
        ib = np.searchsorted(time_edges, t, side="right") - 1
        ok = (ib >= 0) & (ib < n_t)
        same = np.zeros(n_lag)
        if ok.sum() >= 2:
            tb, pb, wb = ib[ok], np.asarray(par)[ok], np.asarray(w)[ok]
            order = np.lexsort((tb, pb))
            tb, pb, wb = tb[order], pb[order], wb[order]
            d = 1
            while True:
                m = pb[:-d] == pb[d:]
                if not m.any():
                    break
                lags = tb[d:][m] - tb[:-d][m]
                prod = wb[d:][m] * wb[:-d][m]
                keep = (lags >= 1) & (lags <= n_lag)
                np.add.at(same, lags[keep] - 1, prod[keep])
                d += 1
        R += F_flat[idx] * (total - same) / (N * (N - 1)) / dt
    return tau_centers, R


def acdc_block_alphas_w(events_w: dict, F_flat: np.ndarray, n_prime: int,
                        n_source_cell: int, fit_window, n_cells: int = 80,
                        max_blocks: int | None = None):
    """rb.acdc_block_alphas on weighted events {(ir, ith): (t, p, w)}."""
    lo, hi = fit_window
    n_per = max(int(n_prime // n_cells), 1)
    time_edges = np.linspace(0.0, rb.T_MAX, rb.N_T + 1)
    n_lag = rb.window_lag_count(fit_window)
    alphas, counts = [], []
    n_blocks = int(n_source_cell // n_per)
    if max_blocks is not None:
        n_blocks = min(n_blocks, max_blocks)
    for b in range(n_blocks):
        ev_b, cnt = {}, 0.0
        for idx, (t, p, w) in events_w.items():
            m = (p > b * n_per) & (p <= (b + 1) * n_per)
            if m.any():
                ev_b[idx] = (t[m], p[m], w[m])
                cnt += float(w[m].sum())
        counts.append(cnt)
        tau, R = acdc_rossi_events_w(ev_b, F_flat, n_per, time_edges, n_tau=n_lag)
        msk = (tau >= lo) & (tau <= hi)
        if np.any(R[msk] > 0):
            try:
                popt, _ = rb.fit_alpha_binned(tau[msk], R[msk])
                if np.isfinite(popt[2]) and popt[2] < 0:
                    alphas.append(popt[2])
            except RuntimeError:
                pass
    return np.array(alphas), np.array(counts)


def _split_events_w(events_w: dict, split: int, n_cell_split: int) -> dict:
    off = int(split) * int(n_cell_split)
    out = {}
    for idx, (t, p, w) in events_w.items():
        m = (p > off) & (p <= off + int(n_cell_split))
        out[idx] = (t[m], p[m] - off, w[m])
    return out


# --------------------------------------------------------------------------- #
# shared inputs (the flagship cost-inputs contract, thinned / weighted)
# --------------------------------------------------------------------------- #

def base_inputs() -> dict:
    if "inputs" in _CACHE:
        return _CACHE["inputs"]
    win = rb.FIT_WINDOWS[LABEL]
    F = rb.fission_coarse_from_dir(SPHERE_DIR, 8, 10)
    alpha_ref = float(rb.acdc_alpha_sigma(LABEL)[0])
    inp = {"label": LABEL, "fit_window": win, "F": F, "n_cells": 80,
           "alpha_ref": alpha_ref, "maxb": MAXB}
    _CACHE["inputs"] = inp
    return inp


def sphere_inputs(density: float) -> dict:
    """Thinned sphere record as a cost-inputs dict (oracle / realistic / die)."""
    s = load_sphere()
    w = sphere_weights(density)
    m = thin_mask(w)
    df = pd.DataFrame({"time": s["t"][m], "parent_id": s["p"][m]})
    inp = dict(base_inputs())
    win = inp["fit_window"]
    t, p = df["time"].to_numpy(), df["parent_id"].to_numpy()
    dts = rb.same_parent_pair_dts(t, p, win[1])
    inp.update({"df": df, "n_sphere": s["n_source"], "events": {}, "n_cell": 1,
                "w0_oracle": rb.flat_background_weight(dts, *win) if len(dts) >= rb.MIN_PAIRS_FIT else 0.0,
                "w0_die": rb.flat_background_weight(t, *win),
                "eps_rel": float(w.mean()), "n_det": int(m.sum()),
                "window_pairs": int(((dts >= win[0]) & (dts <= win[1])).sum()),
                "expected_det": float(w.sum())})
    return inp


def band_with_splits(curve_fn, n_total: int, aref: float, n_splits: int = N_SPLITS):
    """fit_intercept_band on the full curve + the std over disjoint splits.

    curve_fn(split or None) -> sigma_vs_budget DataFrame (full data for None)."""
    full = rb.fit_intercept_band(curve_fn(None), aref)
    vals = []
    for s in range(n_splits):
        b = rb.fit_intercept_band(curve_fn(s), aref)
        vals.append(b["n_target"])
    vals = np.array(vals, float)
    ok = np.isfinite(vals)
    split_err = (float(np.std(vals[ok], ddof=1) / np.sqrt(ok.sum()) / full["n_target"])
                 if ok.sum() >= 2 and np.isfinite(full["n_target"]) else np.nan)
    formal = float(full.get("n_target_rel_se_formal", np.nan))
    # split-sample error when the splits support a band; else the formal weighted-fit
    # SE (labelled: optimistic, nested-budget correlations ignored)
    return {"n_target": float(full["n_target"]), "ok": bool(full["ok"]),
            "n_points": int(full["n_points"]), "slope_free": float(full.get("slope_free", np.nan)),
            "splits": vals.tolist(), "n_target_rel_err_split": split_err,
            "n_target_rel_err_formal": formal,
            "n_target_rel_err": split_err if np.isfinite(split_err) else formal,
            "err_kind": "split" if np.isfinite(split_err) else "formal"}


# --------------------------------------------------------------------------- #
# stage: sphere (pairs vs eps)
# --------------------------------------------------------------------------- #

def stage_sphere(densities=("black",) + DENSITIES, methods=("oracle", "realistic")) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {}
    for dens in densities:
        d = 100.0 if dens == "black" else float(dens)
        t0 = _time.time()
        inp = sphere_inputs(d)
        aref, n_sp = inp["alpha_ref"], inp["n_sphere"]
        w = sphere_weights(d)
        res = {"density": d, "eps_rel": inp["eps_rel"], "n_det": inp["n_det"],
               "window_pairs": inp["window_pairs"],
               # single-neutron variance predictions relative to the black detector:
               # analog captures (Poisson on sum w)  -> cost x 1/<w>
               # expected-value weights (var sum w^2) -> cost x <w^2>/<w>^2 (saturates
               #   in the thin limit at the spread of the 1/v weights)
               "pred_analog_factor": float(1.0 / w.mean()),
               "pred_tally_factor": float((w**2).mean() / w.mean()**2)}
        for m in methods:
            def curve(split, m=m):
                if split is None:
                    return rb.cost_curves(inp, (m,))[m]
                n_s = n_sp // N_SPLITS
                return rb.cost_curves(inp, (m,), df=rb._split_sphere(inp["df"], split, n_s),
                                      n_sphere=n_s)[m]
            res[m] = band_with_splits(curve, n_sp, aref)
        res["seconds"] = _time.time() - t0
        out[str(dens)] = res
        print(f"[sphere] N={d:g}: eps_rel={res['eps_rel']:.4f} det={res['n_det']:,} "
              f"window pairs={res['window_pairs']:,} " +
              " ".join(f"{m}={res[m]['n_target']:.3g} (+/-{100*res[m]['n_target_rel_err']:.0f}%)"
                       for m in methods) + f"  [{res['seconds']:.0f}s]")
        (OUT_DIR / "sphere.json").write_text(json.dumps(out, indent=2))
    return out


# --------------------------------------------------------------------------- #
# stage: sweep (assembly costs vs eps: analog thinning and expected-value weights)
# --------------------------------------------------------------------------- #

def sweep_events_for(density: float):
    ev, n_cell = load_sweep_ev2()
    analog, weighted = {}, {}
    rng = np.random.default_rng(THIN_SEED + 1)
    for idx, (t, p, E, r, u, sig) in ev.items():
        w = np.ones_like(t) if density >= 100 else capture_weight(E, r, u, density, sig)
        m = rng.random(len(w)) < w
        analog[idx] = (t[m], p[m])
        weighted[idx] = (t, p, w)
    return analog, weighted, n_cell


def stage_sweep(densities=("black",) + DENSITIES) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    inp = base_inputs()
    aref, F, win = inp["alpha_ref"], inp["F"], inp["fit_window"]
    out = {}
    for dens in densities:
        d = 100.0 if dens == "black" else float(dens)
        t0 = _time.time()
        analog, weighted, n_cell = sweep_events_for(d)
        n_sweep = n_cell * 80
        full_inp = dict(inp, events=analog, n_cell=n_cell, df=None, n_sphere=0)

        def curve_analog(split):
            if split is None:
                return rb.cost_curves(full_inp, ("acdc",))["acdc"]
            n_s = n_cell // N_SPLITS
            return rb.cost_curves(full_inp, ("acdc",), events=rb._split_events(analog, split, n_s),
                                  n_cell=n_s)["acdc"]

        def curve_weighted(split):
            ev = weighted if split is None else _split_events_w(weighted, split, n_cell // N_SPLITS)
            n_c = n_cell if split is None else n_cell // N_SPLITS
            return rb.sigma_vs_budget(
                lambda n: acdc_block_alphas_w(ev, F, n, n_c, win, n_cells=80, max_blocks=MAXB),
                rb.budget_ladder(n_c * 80), n_c * 80)

        res = {"density": d,
               "eps_rel": float(np.mean([w.mean() for (_, _, w) in weighted.values()])),
               "acdc_analog": band_with_splits(curve_analog, n_sweep, aref),
               "acdc_tally": band_with_splits(curve_weighted, n_sweep, aref)}
        res["seconds"] = _time.time() - t0
        out[str(dens)] = res
        print(f"[sweep] N={d:g}: eps_rel~{res['eps_rel']:.4f} "
              f"analog={res['acdc_analog']['n_target']:.3g} (+/-{100*res['acdc_analog']['n_target_rel_err']:.0f}%) "
              f"tally={res['acdc_tally']['n_target']:.3g} (+/-{100*res['acdc_tally']['n_target_rel_err']:.0f}%)"
              f"  [{res['seconds']:.0f}s]")
        (OUT_DIR / "sweep.json").write_text(json.dumps(out, indent=2))
    return out


# --------------------------------------------------------------------------- #
# stage: gate (absolute overlay, thinned pairs vs weighted assembly)
# --------------------------------------------------------------------------- #

def stage_gate(densities=("black", 1e-3, 1e-4)) -> dict:
    inp = base_inputs()
    F, win = inp["F"], inp["fit_window"]
    s = load_sphere()
    time_edges = np.linspace(0.0, rb.T_MAX, rb.N_T + 1)
    n_lag = rb.window_lag_count(win)
    out = {}
    for dens in densities:
        d = 100.0 if dens == "black" else float(dens)
        w = sphere_weights(d)
        m = thin_mask(w)
        df = pd.DataFrame({"time": s["t"][m], "parent_id": s["p"][m]})
        tau_bf, R_bf, _ = rb.bf_rossi(df, s["n_source"], n_tau=n_lag)
        _, weighted, n_cell = sweep_events_for(d)
        tau_ac, R_ac = acdc_rossi_events_w(weighted, F, n_cell, time_edges, n_lag)
        msk = (tau_ac >= win[0]) & (tau_ac <= win[1])
        ratio = float(R_bf[msk].sum() / R_ac[msk].sum())
        n_pairs = int(rb.doubles_window_pairs(df, win))
        out[str(dens)] = {"density": d, "eps_rel": float(w.mean()), "scale_ratio": ratio,
                          "scale_ratio_err_poisson": ratio / np.sqrt(max(n_pairs, 1)),
                          "window_pairs": n_pairs,
                          "tau_ms": (tau_ac * 1e3).tolist(), "R_bf": R_bf.tolist(), "R_acdc": R_ac.tolist()}
        print(f"[gate] N={d:g}: eps_rel={w.mean():.4f} pairs={n_pairs:,} "
              f"sum(R_bf)/sum(R_acdc) over window = {ratio:.3f} +/- {ratio/np.sqrt(max(n_pairs,1)):.3f}")
    (OUT_DIR / "gate.json").write_text(json.dumps(out))
    return out


# --------------------------------------------------------------------------- #
# stage: validate (real grey transport vs the thinning prediction, block 0 = 10M)
# --------------------------------------------------------------------------- #

def expected_same_parent_pairs(t, p, w, lo, hi) -> float:
    """sum over same-parent pairs with lo <= |dt| <= hi of w_a w_b (expected window
    pairs of the thinned detector)."""
    order = np.lexsort((t, p))
    t, p, w = t[order], p[order], w[order]
    tot, d = 0.0, 1
    while True:
        m = p[:-d] == p[d:]
        if not m.any():
            break
        dt = t[d:][m] - t[:-d][m]
        keep = (dt >= lo) & (dt <= hi)
        tot += float((w[d:][m] * w[:-d][m])[keep].sum())
        d += 1
    return tot


def stage_validate() -> dict:
    inp = base_inputs()
    win = inp["fit_window"]
    s = load_sphere()
    edges = np.linspace(0, rb.T_MAX, 41)
    out = {}
    for d, run in GREY_RUNS.items():
        track = run / "collision_track.h5"
        if not track.exists():
            print(f"[validate] {run.name}: not run yet")
            continue
        with h5py.File(track, "r") as f:
            a = f["collision_track_bank"]
            sel = (a["cell_id"][()] == 2) & (a["delayed_group"][()] == 0)
            tg, pg = a["time"][()][sel], a["parent_id"][()][sel]
        n_run = int(pg.max())           # ~10M: block 0 of the 100M black run
        blk = s["p"] <= n_run
        w = sphere_weights(d)[blk]
        tb, pb = s["t"][blk], s["p"][blk]
        pred_det, pred_err = float(w.sum()), float(np.sqrt((w * (1 - w)).sum()))
        h_pred, _ = np.histogram(tb, bins=edges, weights=w)
        h_real, _ = np.histogram(tg, bins=edges)
        real_pairs = int(rb.doubles_window_pairs(pd.DataFrame({"time": tg, "parent_id": pg}), win))
        pred_pairs = expected_same_parent_pairs(tb, pb, w, *win)
        chi2 = float((((h_real - h_pred) ** 2) / np.maximum(h_pred, 1))[h_pred > 5].sum())
        ndf = int((h_pred > 5).sum())
        out[f"{d:.0e}"] = {"density": d, "n_run": n_run, "real_det": int(len(tg)),
                           "pred_det": pred_det, "pred_det_err": pred_err,
                           "det_pull": (len(tg) - pred_det) / max(pred_err, 1e-9),
                           "real_window_pairs": real_pairs, "pred_window_pairs": pred_pairs,
                           "pairs_pull": (real_pairs - pred_pairs) / np.sqrt(max(pred_pairs, 1)),
                           "gt_chi2": chi2, "gt_ndf": ndf,
                           "edges_ms": (edges * 1e3).tolist(), "h_real": h_real.tolist(), "h_pred": h_pred.tolist()}
        print(f"[validate] N={d:g}: detections real {len(tg):,} vs predicted {pred_det:,.0f} +/- {pred_err:.0f} "
              f"({out[f'{d:.0e}']['det_pull']:+.1f} sigma); window pairs real {real_pairs} vs {pred_pairs:.0f} "
              f"({out[f'{d:.0e}']['pairs_pull']:+.1f} sigma); g(t) chi2/ndf = {chi2:.1f}/{ndf}")
    (OUT_DIR / "validate.json").write_text(json.dumps(out))
    return out


# --------------------------------------------------------------------------- #
# figure
# --------------------------------------------------------------------------- #

def stage_figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INK, GREY, CRIMSON, BLUE, ORANGE = "#30363D", "#8A939B", "#C0392B", "#1F77B4", "#C4620E"
    sph = json.loads((OUT_DIR / "sphere.json").read_text())
    swp = json.loads((OUT_DIR / "sweep.json").read_text()) if (OUT_DIR / "sweep.json").exists() else {}
    gate = json.loads((OUT_DIR / "gate.json").read_text()) if (OUT_DIR / "gate.json").exists() else {}
    val = json.loads((OUT_DIR / "validate.json").read_text()) if (OUT_DIR / "validate.json").exists() else {}

    n_panels = 1 + bool(gate) + bool(val)
    fig, axes = plt.subplots(1, n_panels, figsize=(5.4 * n_panels, 4.4))
    axes = np.atleast_1d(axes)
    ax = axes[0]

    def series(table, key):
        xs, ys, es, cens = [], [], [], []
        for k, r in table.items():
            v = r[key]
            xs.append(r["eps_rel"])
            ok = v["ok"] and np.isfinite(v["n_target"])
            ys.append(v["n_target"] if ok else np.nan)
            es.append(v["n_target"] * v["n_target_rel_err"] if ok and np.isfinite(v["n_target_rel_err"]) else 0.0)
            cens.append(not ok)
        o = np.argsort(xs)
        return np.array(xs)[o], np.array(ys)[o], np.array(es)[o], np.array(cens)[o]

    censored = []
    for table, key, col, mk, lab in ((sph, "oracle", CRIMSON, "s", "pair counting (parent-ID doubles)"),
                                     (sph, "realistic", ORANGE, "d", "pair counting (blind doubles)"),
                                     (swp, "acdc_analog", BLUE, "o", "assembly, analog captures"),
                                     (swp, "acdc_tally", INK, "^", "assembly, expected-value tally")):
        if not table:
            continue
        x, y, e, c = series(table, key)
        ax.errorbar(x[~c], y[~c], yerr=e[~c], fmt=mk + "-", color=col, ms=6, lw=1.2, capsize=2, label=lab)
        censored.append((x[c], col))
        # predicted curves through the black point: pairs ~ eps^-2 (pair rate);
        # analog assembly ~ 1/<w>; tally assembly ~ <w^2>/<w>^2 (single-neutron
        # variance, from the sphere weights; saturates in the thin limit)
        xb, yb = x[-1], y[-1]
        if np.isfinite(yb):
            if key in ("oracle", "realistic"):
                xx = np.logspace(np.log10(x.min()), 0, 20)
                ax.plot(xx, yb * (xx / xb) ** -2, ":", color=col, lw=0.9, alpha=0.7)
            else:
                fac = "pred_analog_factor" if key == "acdc_analog" else "pred_tally_factor"
                px = np.array([r["eps_rel"] for r in sph.values()])
                py = np.array([r[fac] for r in sph.values()]) * yb
                o = np.argsort(px)
                ax.plot(px[o], py[o], ":", color=col, lw=0.9, alpha=0.7)
    ax.set_xscale("log"); ax.set_yscale("log")
    # censored points: no asymptotic band at any budget up to the full record
    ytop = ax.get_ylim()[1] * 1.6
    for i_c, (xc, col) in enumerate(censored):
        if len(xc):
            ax.plot(xc, np.full(len(xc), ytop * 1.25**i_c), "x", color=col, ms=6, mew=1.4,
                    label="no result at 100M" if i_c == 0 else None)
    ax.set_ylim(top=ytop * 1.25**max(len(censored) - 1, 0) * 1.8)
    ax.set_xlabel("detector efficiency relative to black")
    ax.set_ylabel("histories to σ_α = 5 %")
    ax.set_title("k = 0.90, fixed geometry: cost vs efficiency", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.text(0.02, 0.02, "dotted: predicted from the black point (pairs ε⁻²; analog 1/⟨w⟩; tally ⟨w²⟩/⟨w⟩²)",
            transform=ax.transAxes, fontsize=7, color=GREY)

    i = 1
    if gate:
        ax = axes[i]; i += 1
        for k, col in zip(gate, (INK, BLUE, ORANGE)):
            g = gate[k]
            tau = np.array(g["tau_ms"])
            ax.semilogy(tau, g["R_acdc"], "-", color=col, lw=1.2,
                        label=f"assembly, ε = {g['eps_rel']:.2f}")
            ax.semilogy(tau, g["R_bf"], ".", color=col, ms=3, alpha=0.6,
                        label=f"pairs, ε = {g['eps_rel']:.2f}: ratio {g['scale_ratio']:.3f} ± {g['scale_ratio_err_poisson']:.3f}")
        ax.set_xlabel("lag τ [ms]"); ax.set_ylabel("R(τ) [pairs / source / s]")
        ax.set_title("Assembly still lands on the pairs (absolute scale)", color=INK, fontsize=11)
        ax.legend(frameon=False, fontsize=7.5)
    if val:
        ax = axes[i]
        for k, col in zip(val, (BLUE, ORANGE)):
            v = val[k]
            c = 0.5 * (np.array(v["edges_ms"][1:]) + np.array(v["edges_ms"][:-1]))
            ax.semilogy(c, v["h_pred"], "-", color=col, lw=1.2,
                        label=f"N = {v['density']:.0e}: thinning prediction")
            ax.semilogy(c, v["h_real"], ".", color=col, ms=4,
                        label=f"real grey run: {v['real_det']:,} vs {v['pred_det']:,.0f} det ({v['det_pull']:+.1f}σ), "
                              f"pairs {v['real_window_pairs']} vs {v['pred_window_pairs']:.0f}")
        ax.set_xlabel("detection time [ms]"); ax.set_ylabel("prompt captures per bin (10M)")
        ax.set_title("Real grey transport vs reweighted black events", color=INK, fontsize=11)
        ax.legend(frameon=False, fontsize=7)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "grey_detector_study.png", dpi=170)
    print("figure ->", OUT_DIR / "grey_detector_study.png")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=["sphere", "sweep", "gate", "validate", "figure", "all"])
    ap.add_argument("--densities", nargs="*", default=None)
    a = ap.parse_args(argv)
    dens = tuple(("black" if d == "black" else float(d)) for d in a.densities) if a.densities else None
    if a.stage in ("sphere", "all"):
        stage_sphere(dens or ("black",) + DENSITIES)
    if a.stage in ("sweep", "all"):
        stage_sweep(dens or ("black",) + DENSITIES)
    if a.stage in ("gate", "all"):
        stage_gate(dens or ("black", 1e-3, 1e-4))
    if a.stage in ("validate", "all"):
        stage_validate()
    if a.stage in ("figure", "all"):
        stage_figure()


if __name__ == "__main__":
    main()
