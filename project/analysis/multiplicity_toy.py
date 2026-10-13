"""The multiplicity point, made exact: OpenMC's fission rule vs physical P(nu).

WHAT OPENMC DOES (physics.cpp v0.15.3, create_fission_sites, l. 170-230).
At every collision with a fissionable nuclide, BEFORE the reaction is sampled,
OpenMC computes the expected number of fission neutrons for that collision,
    m(E) = nu_bar(E) * sigma_f(E) / sigma_t(E)        (~2.03 at thermal),
and banks n = floor(m + xi) of them, xi ~ U(0,1): n is always one of two
adjacent integers. Then, separately, the parent scatters (sigma_s/sigma_t) or is
absorbed and killed. A discrete "fission event" never happens: a capture spawns
neutrons, a scatter spawns neutrons, every U-235 collision is 84 % of a fission.

WHY FIRST MOMENTS ARE EXACT.  E[n] = m exactly, so k, flux, F_i, g_i(t), alpha
are all exact: they are linear in the mean number of neutrons per collision.

WHY PAIRS ARE NOT.  Rossi-alpha counts sibling pairs: a fission making nu
neutrons makes nu(nu-1) ordered pairs. Physical thermal U-235: nu in 0..7,
<nu> = 2.41, <nu(nu-1)> = 4.63. OpenMC's "2 or 3" rule has the same mean and
almost no spread. Same mean, different spread; pairs count the spread.

THIS MODULE DOES THREE THINGS.
  1. `predict_M_sim(spectrum_statepoint)` -- integrates OpenMC's rule over the
     energies at which U-235 collisions actually occur in our sphere (the tally
     from diagnostics/u235_collision_spectrum.py) and returns the pair moment
     OpenMC must produce, per fission-tally unit:
         M_sim = [ sum_coll E[n(n-1)] + 2 sum_coll s * E[n] ] / sum_coll f
     (progeny-progeny ordered pairs + parent-progeny ordered pairs, divided by
     the expected-value fission count; f = sigma_f/sigma_t, s = sigma_s/sigma_t).
     For n = floor(m + xi): E[n(n-1)] = a (a - 1 + 2 phi), a = floor(m),
     phi = m - a. To be compared with the MEASURED 2.770 +/- 0.016 (k = 0.90) and
     2.768 +/- 0.011 (k = 0.98).
  2. `branching_rossi(rule, ...)` -- a point-model branching process in
     continuous time with two interchangeable dice (OpenMC's rule / physical
     P(nu)) and identical mean dynamics. Returns the same-chain pair histogram
     R(tau): same decay constant, amplitude ratio = M_phys / M_sim.
  3. `grid_dependence(label)` -- the measured pair moment is M_emp = (same-chain
     pairs per source) / sum_i F_i G_i^2 on the 8 x 10 assembly grid. The
     denominator is a Jensen-biased (low) estimate of the continuum integral
     F G^2 because G varies inside a cell, so M_emp on a coarse grid sits ABOVE
     the pair moment OpenMC's rule produces. Using the legacy 20 x 20 sweeps
     (block-summable to every coarser grid, Poisson noise in G^2 removed) this
     measures D(grid) = sum F G^2 and extrapolates it to the continuum, turning
     the published 2.77 (8 x 10) into the grid-free value to compare with (1).
  4. `make_figure()` -- the three-panel figure for the talk.

RUN (repo root, PYTHONPATH=.):
  python -m project.analysis.multiplicity_toy
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OUT_DIR = Path(__file__).resolve().parent / "outputs" / "multiplicity_toy"
SPECTRUM_SP = (Path(__file__).resolve().parents[1] / "results"
               / "u235-collision-spectrum-k090" / "statepoint.10.h5")

# Physical thermal U-235 P(nu), Holden & Zucker 1988 (expert-deck Numbers appendix).
P_NU_PHYS = np.array([0.0317, 0.1720, 0.3363, 0.3038, 0.1268, 0.0266, 0.0026, 0.0002])
P_NU_PHYS = P_NU_PHYS / P_NU_PHYS.sum()
NU = np.arange(len(P_NU_PHYS))
NU_BAR_PHYS = float((NU * P_NU_PHYS).sum())                 # 2.41
M_PHYS = float((NU * (NU - 1) * P_NU_PHYS).sum())           # 4.63

# Thermal single-group OpenMC rule inputs (ENDF/B-VII.1 via the run's own library,
# 0.0253 eV): sigma_f 584, sigma_g 99, sigma_el 15 b -> f = 0.837, s = 0.022.
THERMAL = {"nu_bar": float((np.arange(8) * (np.array([0.0317, 0.1720, 0.3363, 0.3038, 0.1268, 0.0266, 0.0026, 0.0002]) / 0.9999 if False else np.array([0.0317, 0.1720, 0.3363, 0.3038, 0.1268, 0.0266, 0.0026, 0.0002]) / np.array([0.0317, 0.1720, 0.3363, 0.3038, 0.1268, 0.0266, 0.0026, 0.0002]).sum())).sum()), "f": 0.837, "s": 0.022}  # nu_bar = mean of the physical P(nu), so both dice share the mean EXACTLY


# --------------------------------------------------------------------------- #
# 1. the pair moment OpenMC's rule must give, integrated over real collisions
# --------------------------------------------------------------------------- #

def floor_rule_moments(m: np.ndarray):
    """E[n] and E[n(n-1)] for n = floor(m + xi)."""
    a = np.floor(m)
    phi = m - a
    return m, a * (a - 1.0 + 2.0 * phi)


def predict_M_sim(statepoint=SPECTRUM_SP, g_parent_over_g_watt: float = 1.0) -> dict:
    """Integrate OpenMC's banking rule over the measured U-235 collision spectrum.

    `g_parent_over_g_watt` scales the parent-progeny channel: the surviving
    parent is a slow neutron, not a Watt neutron, so its chain-detection
    probability differs from a fresh fission neutron's; 1.0 is the naive
    estimate and the result is reported for a small range of this ratio.
    """
    import openmc

    with openmc.StatePoint(str(statepoint)) as sp:
        t = sp.get_tally(name="u235_by_energy")
        e_edges = t.find_filter(openmc.EnergyFilter).bins
        names = {sc.replace("(n,", "").rstrip(")"): sc for sc in t.scores}
        get = lambda s: t.get_values(scores=[names.get(s, s)]).ravel()
        coll, fis, el, nufis = get("total"), get("fission"), get("elastic"), get("nu-fission")
    ok = coll > 0
    f = np.where(ok, fis / np.maximum(coll, 1e-300), 0.0)        # sigma_f/sigma_t per bin
    s = np.where(ok, el / np.maximum(coll, 1e-300), 0.0)         # sigma_s/sigma_t per bin
    nu_bar = np.where(fis > 0, nufis / np.maximum(fis, 1e-300), 0.0)
    m = nu_bar * f
    En, Enn1 = floor_rule_moments(m)
    w = coll                                                      # collisions per bin
    pp = (w * Enn1).sum()                                         # progeny-progeny ordered pairs
    par = 2.0 * (w * s * En).sum()                                # parent-progeny ordered pairs
    F = (w * f).sum()                                             # expected-value fissions
    # per-collision daughter distribution, collision-weighted (compare with tracks)
    a = np.floor(m); phi = m - a
    dist = {}
    for k_ in range(0, 5):
        dist[k_] = float((w * (((a == k_) * (1 - phi)) + ((a + 1 == k_) * phi))).sum() / w.sum())
    e_mid = np.sqrt(e_edges[:, 0] * e_edges[:, 1])
    thermal_frac = float(w[e_mid < 0.625].sum() / w.sum())
    return {
        "M_sim_pred": float((pp + par * g_parent_over_g_watt) / F),
        "M_progeny_only": float(pp / F),
        "M_parent_channel": float(par / F),
        "mean_daughters_per_collision": float((w * En).sum() / w.sum()),
        "daughter_dist": dist,
        "fissions_per_collision": float(F / w.sum()),
        "parent_survival_per_collision": float((w * s).sum() / w.sum()),
        "thermal_collision_fraction_below_0.625eV": thermal_frac,
        "M_phys": M_PHYS, "nu_bar_phys": NU_BAR_PHYS,
    }


# --------------------------------------------------------------------------- #
# 2. the toy: one point-model branching process, two dice
# --------------------------------------------------------------------------- #

def branching_rossi(rule: str, n_source: int = 200_000, k: float = 0.90,
                    lifetime: float = 6.7e-5, eps: float = 0.05,
                    tau_max: float = 6e-3, n_bins: int = 120, seed: int = 1,
                    nu_bar: float = THERMAL["nu_bar"], f: float = THERMAL["f"],
                    s: float = THERMAL["s"]) -> dict:
    """Continuous-time point-model chains; same mean dynamics under both rules.

    Each neutron flies for an Exp(lifetime) time and then either leaks (and is
    detected with probability eps) or collides with U-235. The U-235 collision
    probability q is set so that the mean neutrons per neutron equals k:
        k = q * nu_bar * f / (1 - q * s)     (parent scatters s of the time)
    rule = "openmc": bank floor(m + xi) neutrons, m = nu_bar f; parent survives
                     with probability s (independently of the banking).
    rule = "physical": sample the reaction: fission (f) -> nu ~ P_NU_PHYS
                     (rescaled to the same nu_bar), parent dies; scatter (s) ->
                     parent continues, nothing banked; capture -> dies.
    Returns detection times, same-chain pair histogram and the moments.
    """
    rng = np.random.default_rng(seed)
    q = k / (nu_bar * f + k * s)
    m = nu_bar * f
    # physical P(nu) rescaled to nu_bar (keep the SHAPE of the spread): shift mean
    # by mixing with a one-step shifted copy is overkill; the Holden-Zucker mean
    # (2.41) is already within 0.5 % of nu_bar, so use it as is.
    p_phys = P_NU_PHYS
    det_t, det_chain = [], []
    n_fiss = 0
    for c in range(n_source):
        stack = [0.0]
        while stack:
            t = stack.pop() + rng.exponential(lifetime)
            if rng.random() > q:                     # leak
                if rng.random() < eps:
                    det_t.append(t); det_chain.append(c)
                continue
            if rule == "openmc":
                n_fiss_eq = f                        # expected-value fission count
                n = int(m) + (1 if rng.random() < (m - int(m)) else 0)
                stack.extend([t] * n)
                if rng.random() < s:                 # parent scatters on
                    stack.append(t)
            else:
                u = rng.random()
                if u < f:                            # fission
                    n = int(rng.choice(NU, p=p_phys))
                    stack.extend([t] * n)
                elif u < f + s:                      # scatter
                    stack.append(t)
                # else capture: dies
    det_t = np.asarray(det_t); det_chain = np.asarray(det_chain)
    order = np.argsort(det_chain, kind="stable")
    det_t, det_chain = det_t[order], det_chain[order]
    # same-chain pair separations (every unordered pair once, positive lag)
    seps = []
    starts = np.flatnonzero(np.r_[True, det_chain[1:] != det_chain[:-1], True])
    for a, b in zip(starts[:-1], starts[1:]):
        if b - a < 2:
            continue
        tt = np.sort(det_t[a:b])
        d = tt[None, :] - tt[:, None]
        seps.append(d[np.triu_indices(len(tt), 1)])
    seps = np.concatenate(seps) if seps else np.zeros(0)
    edges = np.linspace(0, tau_max, n_bins + 1)
    h, _ = np.histogram(seps, bins=edges)
    dt = edges[1] - edges[0]
    return {
        "rule": rule, "k": k, "q": q, "n_source": n_source,
        "tau": 0.5 * (edges[1:] + edges[:-1]), "R": h / n_source / dt,
        "n_det": int(len(det_t)), "n_pairs": int(len(seps)),
        "singles_per_source": len(det_t) / n_source,
    }


def fit_alpha(tau, R, lo, hi):
    msk = (tau >= lo) & (tau <= hi) & (R > 0)
    A = np.vstack([np.ones(msk.sum()), tau[msk]]).T
    coef, *_ = np.linalg.lstsq(A, np.log(R[msk]), rcond=None)
    return float(coef[1]), float(np.exp(coef[0]))


# --------------------------------------------------------------------------- #
# 3. grid dependence of the measured pair moment (existing 20 x 20 sweeps)
# --------------------------------------------------------------------------- #

GRIDS_R = (1, 2, 4, 5, 10, 20)
GRIDS_TH = (1, 2, 4, 5, 10, 20)
M_EMP_8x10 = {"090": (2.770, 0.016), "098": (2.768, 0.011)}   # 100M campaigns


def grid_dependence(label: str) -> dict:
    """D(grid) = sum_i F_i G_i^2 for every grid nested in the 20 x 20 sweep.

    Noise-corrected: E[Ghat^2] = G^2 + G/n, so G^2 -> Ghat^2 - Ghat/n. Returns the
    table normalised to D(20x20), a radial Richardson extrapolation to the
    continuum, and the implied grid-free pair moment from the published 8 x 10
    M_emp (8 does not nest in 20; D(8x10) is bracketed by D(5x10) and D(10x10)).
    """
    import project.analysis.greens_loader as gl

    sw = gl.load_sweep(label)
    n_r, n_th = sw["G"].shape
    n = float(sw["n_source"])
    det = sw["n_prompt"].astype(float)
    F = gl.fission_coarse(label, n_r, n_th) / gl.NU_NU_MINUS_1
    D = {}
    for kr in GRIDS_R:
        for kth in GRIDS_TH:
            br, bth = n_r // kr, n_th // kth
            Fb = F.reshape(kr, br, kth, bth).sum(axis=(1, 3))
            db = det.reshape(kr, br, kth, bth).sum(axis=(1, 3))
            nb = n * br * bth
            Gb = db / nb
            D[(kr, kth)] = float((Fb * (Gb**2 - Gb / nb)).sum())
    d5, d10, d20 = D[(5, 20)], D[(10, 20)], D[(20, 20)]
    p = np.log((d10 - d5) / (d20 - d10)) / np.log(2)
    D_inf = d20 + (d20 - d10) / (2**p - 1)
    D_8x10 = 0.5 * (D[(5, 10)] + D[(10, 10)])
    D_8x10_lo, D_8x10_hi = D[(5, 10)], D[(10, 10)]
    M, Merr = M_EMP_8x10[label]
    return {
        "label": label, "n_hist_per_cell": int(n), "D_over_D20": {f"{kr}x{kth}": v / d20 for (kr, kth), v in D.items()},
        "richardson_p": float(p), "D_inf_over_D20": float(D_inf / d20),
        "D_8x10_over_D20_bracket": (D_8x10_lo / d20, D_8x10_hi / d20),
        "M_emp_8x10": M, "M_emp_8x10_err": Merr,
        "M_continuum": float(M * D_8x10 / D_inf),
        "M_continuum_bracket": (float(M * D_8x10_lo / D_inf), float(M * D_8x10_hi / D_inf)),
        "M_vs_grid_radial_at_n_th_10": {kr: float(M * D_8x10 / D[(kr, 10)]) for kr in GRIDS_R},
    }


# --------------------------------------------------------------------------- #
# 4. figure
# --------------------------------------------------------------------------- #

def make_figure(n_source: int = 300_000, seed: int = 1):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    INK, GREY, CRIMSON, BLUE = "#30363D", "#8A939B", "#C0392B", "#1F77B4"

    pred = predict_M_sim() if SPECTRUM_SP.exists() else None
    grids = {lab: grid_dependence(lab) for lab in ("090", "098")}
    toy = {r: branching_rossi(r, n_source=n_source, seed=seed) for r in ("openmc", "physical")}
    lo, hi = 0.5e-3, 3.0e-3
    fits = {r: fit_alpha(toy[r]["tau"], toy[r]["R"], lo, hi) for r in toy}
    amp_ratio = toy["physical"]["R"][:20].sum() / toy["openmc"]["R"][:20].sum()
    # the thermal single-group analytic value for the toy's own parameters
    m = THERMAL["nu_bar"] * THERMAL["f"]
    En, Enn1 = floor_rule_moments(np.array([m]))
    M_toy_openmc = float((Enn1[0] + 2 * THERMAL["s"] * En[0]) / THERMAL["f"])
    summary = {
        "toy": {r: {"alpha_fit": fits[r][0], "n_det": toy[r]["n_det"], "n_pairs": toy[r]["n_pairs"],
                    "singles_per_source": toy[r]["singles_per_source"]} for r in toy},
        "toy_amplitude_ratio_phys_over_openmc": float(amp_ratio),
        "toy_pairs_ratio_phys_over_openmc": toy["physical"]["n_pairs"] / max(toy["openmc"]["n_pairs"], 1),
        "M_phys_over_M_openmc_thermal_analytic": M_PHYS / M_toy_openmc,
        "M_openmc_thermal_analytic": M_toy_openmc,
        "prediction_from_collision_spectrum": pred,
        "measured_M_sim": {"k090_100M": [2.770, 0.016], "k098_100M": [2.768, 0.011]},
        "grid_dependence": grids,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.2))
    ax = axes[0]
    a = np.floor(m); phi = m - a
    p_openmc = np.zeros_like(P_NU_PHYS); p_openmc[int(a)] = 1 - phi; p_openmc[int(a) + 1] = phi
    w = 0.38
    ax.bar(NU - w / 2, P_NU_PHYS, w, color=CRIMSON, label=f"physical U-235, <ν(ν−1)> = {M_PHYS:.2f}")
    ax.bar(NU + w / 2, p_openmc, w, color=GREY,
           label=f"OpenMC rule, floor(m + ξ), m = {m:.2f}")
    ax.set_xlabel("neutrons per fission (per U-235 collision for OpenMC)")
    ax.set_ylabel("probability")
    ax.set_title("Same mean, different spread", color=INK)
    ax.legend(frameon=False)
    ax = axes[1]
    for r, col, lab in (("physical", CRIMSON, "physical P(ν)"), ("openmc", GREY, "OpenMC rule")):
        ax.semilogy(toy[r]["tau"] * 1e3, toy[r]["R"], ".", color=col, ms=4,
                    label=f"{lab}: α = {fits[r][0]:.0f} /s")
        tt = np.linspace(lo, hi, 50)
        ax.semilogy(tt * 1e3, fits[r][1] * np.exp(fits[r][0] * tt), "-", color=col, lw=1.2, alpha=0.7)
    ax.set_xlabel("lag τ [ms]")
    ax.set_ylabel("same-chain pairs per source per s")
    ax.set_title(f"Point-model toy, k = {toy['openmc']['k']}: amplitude ×{amp_ratio:.2f}, same slope",
                 color=INK)
    ax.legend(frameon=False)
    ax = axes[2]
    for lab, col, mk in (("090", BLUE, "o"), ("098", CRIMSON, "s")):
        g = grids[lab]
        xs = np.array(GRIDS_R, float)
        ys = np.array([g["M_vs_grid_radial_at_n_th_10"][kr] for kr in GRIDS_R])
        ax.plot(xs, ys, mk + "-", color=col, ms=5, lw=1, label=f"k = 0.{lab[1:]}, measured M_emp (10 polar bands)")
        ax.axhline(g["M_continuum"], color=col, ls=":", lw=1)
    ax.axhspan(2.770 - 0.016, 2.770 + 0.016, color=GREY, alpha=0.25, lw=0)
    ax.axhline(2.770, color=GREY, lw=1, label="published M_sim, 8 × 10 grid: 2.77")
    if pred:
        ax.axhline(pred["M_sim_pred"], color=INK, lw=1.4, ls="--",
                   label=f"OpenMC rule over real collisions: {pred['M_sim_pred']:.2f}")
    ax.set_xscale("log"); ax.set_xticks(GRIDS_R); ax.set_xticklabels([str(k) for k in GRIDS_R])
    ax.set_xlabel("radial cells in the assembly grid")
    ax.set_ylabel("pair moment per fission")
    ax.set_title("Measured moment depends on the grid; its limit is the rule", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "multiplicity_toy.png", dpi=170)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    make_figure()
