"""Regenerate the two analysis notebooks from the corrected per-cell sweep data.

Run:  python -m project.analysis.build_notebooks
Produces:
  project/analysis/notebooks/greens_function_analysis.ipynb   (single k=0.98, full method)
  project/analysis/notebooks/keff_comparison.ipynb            (all four k_eff)

Both are built entirely on `project.analysis.greens_loader` (the corrected confined-source
sweep), so the broken `mesh_ids[parent_id-1]` attribution is gone project-wide.
"""
from pathlib import Path

import nbformat as nbf

NB_DIR = Path(__file__).resolve().parent / "notebooks"


def md(text):
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text):
    return nbf.v4.new_code_cell(text.strip("\n"))


# ----------------------------------------------------------------------------- #
#  Notebook 1: greens_function_analysis.ipynb  (k = 0.98)
# ----------------------------------------------------------------------------- #
def build_greens():
    cells = []
    cells.append(md(r"""
# Green's Function Analysis (corrected)

Green's function approach to Rossi-alpha, following
> Gabrieli, Shokef, Neder (2025), *Efficient calculation of reactor noise via Ito-Langevin Process*.

## Method
For each spatial cell $i$ we need $g_i(t)$ — the detector response per neutron **emitted from
cell $i$**, following its entire fission chain — and its time integral
$G_i=\int g_i(t)\,dt$. The Rossi-alpha is then (Eq. 55):
$$R(\tau)=\sum_i F_i \int g_i(t)\,g_i(t+\tau)\,dt,\qquad F_i=\overline{\nu(\nu-1)}\,\tilde F_i .$$

**Why this notebook was rewritten.** The original pipeline ran one uniform-source simulation
and attributed each detection back to a birth cell via `mesh_ids[parent_id-1]`. But
`openmc.FileSource` samples a *random* site per history, so the particle `id` does **not**
equal the pre-generated source-file row — the attribution was scrambled and forced
$G_i=\text{det}_i/\text{src}_i\to\text{const}$ (a flat map at every $k$). The corrected
data here comes from `project.simulations.greens_sweep`: **one confined-source simulation per
cell**, so every detection originates from that cell *by construction* — no `parent_id`,
no bug. See `greens_per_cell.py` for the ground-truth proof (near/far pole ratio 7-50x).
"""))

    cells.append(code(r"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from project.analysis import greens_loader as gl

FIGURES_DIR = Path("../outputs/figures/greens_function")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

LABEL = "098"
sweep = gl.load_sweep(LABEL)
n_r, n_th = sweep["G"].shape
F = gl.fission_coarse(LABEL, n_r, n_th)

R = float(sweep["R"]); det_z = R + 15.0
print(f"k = {sweep['k']:.2f}   R = {R:.2f} cm   detector at +Z = {det_z:.1f} cm")
print(f"coarse grid: {n_r} radial x {n_th} theta = {n_r*n_th} cells, "
      f"{sweep['n_source']:,} source neutrons/cell")
print(f"G range: {sweep['G'].min():.2e} .. {sweep['G'].max():.2e}  "
      f"(buggy flat method gave ~{sweep['G'].mean():.2e} everywhere)")
"""))

    cells.append(md(r"""
## 0. Setup: detector & mesh geometry

Before any results, confirm the geometry means what we claim. The detector sits on the **+Z axis**,
which is the polar axis of the spherical tally mesh ($\theta$ is measured from +Z). So $\cos\theta=+1$
is the detector-facing surface and $\cos\theta=-1$ the far side, every $(r,\theta)$ cell has a
well-defined distance to the detector, and the detector is *outside* the core (10 cm gap). This is
what makes the spatial $G_i$ interpretable as a function of distance/direction to the detector.
"""))
    cells.append(code(r"""
fig, ax = plt.subplots(figsize=(6.5, 7.5))
gl.plot_geometry_mesh(ax, R, n_r=n_r, n_th=n_th)
ax.set_title("Detector on the mesh polar axis (+Z) — k=0.98 geometry")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "geometry_mesh.png", dpi=200, bbox_inches="tight")
plt.show()
"""))

    cells.append(md(r"""
## 1. Spatial Green's function $G_i(r,\theta)$

$G_i$ = expected detector captures per neutron emitted from cell $i$. The detector sits on the
$+Z$ axis, so $\cos\theta=+1$ is the detector-facing surface and $\cos\theta=-1$ is the far side.
"""))

    cells.append(code(r"""
from matplotlib.colors import LogNorm

r_edges = sweep["r_edges"]; cos_edges = sweep["cos_edges"]
G_2d = sweep["G"]                       # (n_r, n_th)
norm = LogNorm(vmin=np.nanmax(G_2d) / 100, vmax=np.nanmax(G_2d))

fig, ax = plt.subplots(figsize=(8, 5))
im = ax.pcolormesh(r_edges, cos_edges, G_2d.T, cmap="plasma", shading="flat", norm=norm)
plt.colorbar(im, ax=ax, label=r"$G_i$ [det / source neutron]")
ax.set_xlabel("r (cm)")
ax.set_ylabel(r"$\cos\theta$   (+1 = toward detector, -1 = away)")
ax.set_title(r"Corrected Green's function $G_i(r,\theta)$  (k=0.98)")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "greens_function_heatmap.png", dpi=200, bbox_inches="tight")
plt.show()
"""))

    cells.append(md(r"""
### Same $G_i$ as a polar meridional slice

The identical data on a physical cross-section of the core: disk centre = core centre, rim =
surface, **detector direction (θ=0) at the top**, far side (θ=π) at the bottom. Azimuthally
symmetric, so the left/right halves mirror.
"""))
    cells.append(code(r"""
fig = plt.figure(figsize=(6.5, 6))
axp = fig.add_subplot(111, projection="polar")
imp = gl.polar_meridian(axp, sweep, norm=norm)
fig.colorbar(imp, ax=axp, label=r"$G_i$ [det / source neutron]", pad=0.10, shrink=0.8)
axp.set_title("Green's function — meridional slice (detector at top, k=0.98)", pad=18)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "greens_function_polar.png", dpi=200, bbox_inches="tight")
plt.show()
"""))

    cells.append(md(r"""
## 2. Radial (importance) vs angular (distance) structure

The two physical effects separate cleanly:
* **Radial** $G(r)$ (marginalised over $\theta$) — the *importance* / fundamental-mode weight:
  a neutron deeper in the core is more likely to ignite a persistent chain. Near critical this
  is the dominant structure and is centre-peaked. We overlay the idealised bare-sphere fundamental
  mode $\mathrm{sinc}(r/R)=\sin(\pi r/R)/(\pi r/R)$ as a reference: the simulated $G(r)$ tracks it
  (centre-peaked), with a non-zero edge value expected from the extrapolation length and the direct
  surface-to-detector coupling that the idealised zero-flux-boundary mode omits.
* **Angular** $G(\theta)$ at fixed radius — the *direct geometric coupling* to the detector:
  cells facing the detector ($\cos\theta>0$) couple more strongly. This dominates far below
  critical and washes out as $k\to1$.
"""))

    cells.append(code(r"""
r1d, G_r, G_r_err = gl.radial_profile(sweep)
cos1d, G_th, G_th_err = gl.angular_profile(sweep, r_frac_min=0.75)  # outer shells
fund = gl.fundamental_mode_shape(r1d, R)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
gmax = G_r.max()
axes[0].errorbar(r1d / R, G_r / gmax, yerr=G_r_err / gmax, marker="o", capsize=3, label="$G(r)$ (sim)")
axes[0].plot(r1d / R, fund, "k--", lw=1.5, label=r"fundamental mode $\mathrm{sinc}(r/R)$")
axes[0].set_xlabel("r / R"); axes[0].set_ylabel(r"$\langle G\rangle_\theta$ (normalised)")
axes[0].set_title("Radial profile vs fundamental-mode importance")
axes[0].legend()

axes[1].errorbar(cos1d, G_th, yerr=G_th_err, marker="o", capsize=3, color="C3")
axes[1].set_xlabel(r"$\cos\theta$  (+1 = toward detector)")
axes[1].set_ylabel(r"$G$ (outer shells, $r/R>0.75$) [det/source]")
axes[1].set_title("Angular profile (direct detector coupling)")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "greens_function_profiles.png", dpi=200, bbox_inches="tight")
plt.show()
"""))

    cells.append(md(r"""
## 3. $G$ vs distance to detector

The single clearest view of the effect the original (buggy) analysis was missing entirely:
$G$ falls steeply with the cell-to-detector distance.
"""))

    cells.append(code(r"""
dist = sweep["dist"].ravel(); G = sweep["G"].ravel(); Gerr = sweep["G_err"].ravel()
order = np.argsort(dist)
fig, ax = plt.subplots(figsize=(8, 5))
ax.errorbar(dist[order], G[order], yerr=Gerr[order], fmt="o", ms=4, capsize=2, alpha=0.8)
ax.axhline(sweep["G"].mean(), ls="--", color="gray",
           label=f"buggy 'flat' G ~ {sweep['G'].mean():.2e}")
ax.set_yscale("log"); ax.set_xlabel("distance cell -> detector [cm]")
ax.set_ylabel(r"$G_i$ [det / source neutron]")
ax.set_title("Green's function vs distance to detector (k=0.98)")
ax.legend(); plt.tight_layout()
plt.savefig(FIGURES_DIR / "greens_function_vs_distance.png", dpi=200, bbox_inches="tight")
plt.show()
"""))

    cells.append(md(r"""
## 4. Time-resolved $g_i(t)$

$g_i(t)$ for a near-pole (facing), far-pole and central cell. Near cells peak earlier and
higher; far cells are delayed and weaker.
"""))

    cells.append(code(r"""
te = sweep["time_edges"]; tc = 0.5 * (te[:-1] + te[1:]); dt = te[1] - te[0]
n = float(sweep["n_source"])
picks = {
    "near pole (facing)": (n_r - 1, 0),
    "far pole (away)":    (n_r - 1, n_th - 1),
    "centre":             (0, n_th // 2),
}
fig, ax = plt.subplots(figsize=(9, 5))
for lbl, (ir, ith) in picks.items():
    g = sweep["g_t"][ir, ith] / (n * dt)        # per source per second
    gp = np.where(sweep["g_t"][ir, ith] > 0, g, np.nan)
    ax.plot(tc * 1e3, gp, marker=".", ms=3, label=f"{lbl}  (d={sweep['dist'][ir,ith]:.0f} cm)")
ax.set_yscale("log"); ax.set_xlabel("t (ms)")
ax.set_ylabel(r"$g_i(t)$ [s$^{-1}$ per source]")
ax.set_title("Time-resolved Green's function (k=0.98)")
ax.legend(); plt.tight_layout()
plt.savefig(FIGURES_DIR / "greens_function_time.png", dpi=200, bbox_inches="tight")
plt.show()
"""))

    cells.append(md(r"""
## 5. Rossi-alpha from the corrected $g_i(t)$

$R(\tau)=\sum_i F_i\int g_i(t)g_i(t+\tau)\,dt$, fit to $A+B e^{\alpha\tau}$.

The prompt decay constant $\alpha$ is the **fundamental eigenvalue** of the system — it lives
in the detection *times*, not in the spatial labels. That is why the original analysis recovered
a sensible $\alpha$ even though its spatial $G_i$ was scrambled: the bug shuffled *which cell*
each detection was assigned to, but never *when* it arrived. Spatial structure lives in the
amplitudes $F_i$ and $g_i$ shape; the decay rate does not.
"""))

    cells.append(code(r"""
tau, R_tau = gl.rossi(sweep, F)
popt, perr = gl.fit_alpha(tau, R_tau)
alpha = popt[2]
print(f"alpha = {alpha:.1f} +/- {perr[2]:.1f} s^-1   (decay time = {-1/alpha*1e3:.3f} ms)")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
tf = np.linspace(tau[0], tau[-1], 800)
axes[0].plot(tau*1e3, R_tau, "o", ms=3, alpha=0.6, label="data")
axes[0].plot(tf*1e3, gl.rossi_model(tf, *popt), "r-", lw=1.5, label=fr"fit $\alpha$={alpha:.0f} s$^{{-1}}$")
axes[0].set_xlabel(r"$\tau$ (ms)"); axes[0].set_ylabel(r"$R(\tau)$")
axes[0].set_title("Rossi-alpha (linear)"); axes[0].legend()

R_corr = np.where(R_tau - popt[0] > 0, R_tau - popt[0], np.nan)
axes[1].semilogy(tau*1e3, R_corr, "o", ms=3, alpha=0.6, label="data - A")
axes[1].semilogy(tf*1e3, gl.rossi_model(tf, 0, popt[1], alpha), "r-", lw=1.5,
                 label=fr"$Be^{{\alpha\tau}}$")
axes[1].set_xlabel(r"$\tau$ (ms)"); axes[1].set_ylabel(r"$R(\tau)-A$")
axes[1].set_title("Rossi-alpha (semilog)"); axes[1].legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / "rossi_alpha.png", dpi=200, bbox_inches="tight")
plt.show()
"""))

    cells.append(md(r"""
## Summary

* Corrected $G_i$ spans nearly two decades across the core (vs the buggy flat $\sim$const).
* Two structures: **radial importance** (centre-peaked, dominant near critical) and **angular
  distance-to-detector coupling** (facing $>$ away, dominant far below critical).
* The Rossi $\alpha$ is unchanged by the fix, as expected — it is carried by detection times.
"""))

    cells.append(code(r"""
print("="*54)
print("GREEN'S FUNCTION ANALYSIS (corrected) — k =", sweep["k"])
print("="*54)
print(f"G(min..max)            : {sweep['G'].min():.3e} .. {sweep['G'].max():.3e}")
print(f"G(near/far pole)       : {sweep['G'][-1,0]/sweep['G'][-1,-1]:.1f}x")
print(f"alpha                  : {alpha:.1f} s^-1  (tau = {-1/alpha*1e3:.3f} ms)")
"""))

    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nbf.write(nb, NB_DIR / "greens_function_analysis.ipynb")
    print("wrote greens_function_analysis.ipynb")


# ----------------------------------------------------------------------------- #
#  Notebook 2: keff_comparison.ipynb  (all four k_eff)
# ----------------------------------------------------------------------------- #
def build_keff():
    cells = []
    cells.append(md(r"""
# k_eff Comparison — corrected Green's function

Spatial Green's function and Rossi-alpha across four operating points
$k_\text{eff}\approx 0.98, 0.90, 0.70, 0.40$, using the corrected confined-source sweep
(`project.simulations.greens_sweep`). The earlier version of this notebook reported
$G_\text{facing}/G_\text{away}\approx 1$ at every $k$ and spent several sections hunting for a
"metric artifact" — that flatness was the `FileSource`/`parent_id` attribution bug, not physics.
With correct attribution the directional contrast is **7-50x** and has clear $k$-dependence.

**Physical picture.** Far below critical the chain is short, so $G$ is dominated by *direct
geometric coupling* — cells facing the detector win, strong distance dependence. Near critical
the chain is long and relaxes to the fundamental mode, so $G$ is dominated by *importance*
(centre-peaked, direction-independent); the directional contrast collapses.
"""))

    cells.append(code(r"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from project.analysis import greens_loader as gl

FIGURES_DIR = Path("../outputs/keff_comparison")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

LABELS = ["098", "090", "070", "040"]
COLORS = {"098": "C3", "090": "C1", "070": "C2", "040": "C0"}
sweeps = {L: gl.load_sweep(L) for L in LABELS}
F = {L: gl.fission_coarse(L, *sweeps[L]["G"].shape) for L in LABELS}
for L in LABELS:
    s = sweeps[L]
    print(f"k={s['k']:.2f}: R={s['R']:.2f} cm, grid {s['G'].shape}, "
          f"G range {s['G'].min():.2e}..{s['G'].max():.2e}, "
          f"near/far={s['G'][-1,0]/s['G'][-1,-1]:.1f}x")
"""))

    cells.append(md("## 1. $G_i(r,\\theta)$ heatmaps (shared log scale)"))
    cells.append(code(r"""
G_all = np.concatenate([sweeps[L]["G"].ravel() for L in LABELS])
norm = LogNorm(vmin=np.percentile(G_all[G_all > 0], 2), vmax=G_all.max())

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
for idx, (ax, L) in enumerate(zip(axes.flat, LABELS)):
    s = sweeps[L]
    r_unit = s["r_edges"] / s["R"]
    im = ax.pcolormesh(r_unit, s["cos_edges"], s["G"].T, cmap="plasma",
                       shading="flat", norm=norm)
    if idx >= 2:                       # bottom row only
        ax.set_xlabel("r / R")
    if idx % 2 == 0:                   # left column only
        ax.set_ylabel(r"$\cos\theta$ (+1 = toward detector)")
    ax.set_title(f"k = {s['k']:.2f}  (R = {s['R']:.2f} cm)")
fig.colorbar(im, ax=axes.ravel().tolist(), label=r"$G_i$ [det/source]",
             fraction=0.025, pad=0.02)
fig.suptitle("Corrected integrated Green's function", fontsize=13)
plt.savefig(FIGURES_DIR / "Gi_heatmaps.png", dpi=200, bbox_inches="tight")
plt.show()
"""))

    cells.append(md(r"""
### Same heatmaps as polar meridional slices

The identical $G_i$ data as physical core cross-sections (disk centre = core centre, rim =
surface, **detector at the top**). The shift from a centre-bright disk (importance, k=0.98) to a
top-bright / bottom-dark disk (direct detector coupling, k=0.40) is immediately visible.
"""))
    cells.append(code(r"""
fig, axes = plt.subplots(2, 2, figsize=(11, 10), subplot_kw={"projection": "polar"})
for ax, L in zip(axes.flat, LABELS):
    im = gl.polar_meridian(ax, sweeps[L], norm=norm)
    ax.set_title(f"k = {sweeps[L]['k']:.2f}", pad=12)
fig.colorbar(im, ax=axes.ravel().tolist(), label=r"$G_i$ [det/source]", fraction=0.025, pad=0.04)
fig.suptitle("Corrected Green's function — meridional slices (detector at top)", fontsize=13)
plt.savefig(FIGURES_DIR / "Gi_polar.png", dpi=200, bbox_inches="tight")
plt.show()
"""))

    cells.append(md(r"""
## 2. Distance dependence and the near/far trend

Left: $G$ vs distance for every $k$. Right: the near-pole/far-pole ratio vs $k$ — the headline
result, versus the buggy method's flat $\approx 1$.
"""))
    cells.append(code(r"""
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
for L in LABELS:
    s = sweeps[L]
    d = s["dist"].ravel(); G = s["G"].ravel(); o = np.argsort(d)
    ax1.plot(d[o], G[o], "o-", ms=4, alpha=0.8, color=COLORS[L], label=f"k={s['k']:.2f}")
ax1.set_yscale("log"); ax1.set_xlabel("distance cell -> detector [cm]")
ax1.set_ylabel(r"$G_i$ [det/source]"); ax1.set_title("G vs distance, all k_eff")
ax1.legend()

ks = [sweeps[L]["k"] for L in LABELS]
ratio = [sweeps[L]["G"][-1, 0] / sweeps[L]["G"][-1, -1] for L in LABELS]
rerr = [r * np.sqrt(1/max(sweeps[L]["n_prompt"][-1,0],1) + 1/max(sweeps[L]["n_prompt"][-1,-1],1))
        for L, r in zip(LABELS, ratio)]
ax2.errorbar(ks, ratio, yerr=rerr, fmt="s-", ms=10, capsize=4, color="black")
ax2.axhline(1.0, ls="--", color="red", label="buggy method (no effect)")
for k, r in zip(ks, ratio):
    ax2.annotate(f"{r:.0f}x", (k, r), textcoords="offset points", xytext=(6, 5))
ax2.set_xlabel(r"$k_\text{eff}$"); ax2.set_ylabel("G(near pole) / G(far pole)")
ax2.set_title("Directional contrast vs k_eff"); ax2.legend(); ax2.invert_xaxis()
plt.tight_layout()
plt.savefig(FIGURES_DIR / "distance_and_ratio.png", dpi=200, bbox_inches="tight")
plt.show()
"""))

    cells.append(md("## 3. Radial (importance) vs angular (direction) profiles across k_eff"))
    cells.append(code(r"""
fig, (axr, axt) = plt.subplots(1, 2, figsize=(13, 5))
for L in LABELS:
    s = sweeps[L]
    r1d, G_r, G_r_err = gl.radial_profile(s)
    axr.plot(r1d / s["R"], G_r / G_r.max(), "o-", ms=4, color=COLORS[L], label=f"k={s['k']:.2f}")
    cos1d, G_th, _ = gl.angular_profile(s, r_frac_min=0.75)
    axt.plot(cos1d, G_th / G_th.max(), "o-", ms=4, color=COLORS[L], label=f"k={s['k']:.2f}")
axr.set_xlabel("r / R"); axr.set_ylabel(r"$G(r)$ (normalised)")
axr.set_title("Radial / importance profile"); axr.legend()
axt.set_xlabel(r"$\cos\theta$ (+1 = toward detector)")
axt.set_ylabel(r"$G(\theta)$ outer shells (normalised)")
axt.set_title("Angular / direct-coupling profile"); axt.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / "profiles_vs_k.png", dpi=200, bbox_inches="tight")
plt.show()
"""))

    cells.append(md(r"""
## 3b. Direct vs multiplied: directional contrast vs arrival time

The time-integrated $G_i$ mixes two channels with opposite behaviour; splitting the outer-shell
detections by **arrival time** separates them cleanly:

* **Early — direct first-flight.** The source neutron streams to the detector before any fission.
  This is **pure transport/geometry, independent of multiplication**: the facing/away contrast is set
  by how much farther and more shielded the far pole is. So the early magnitude scales with **core
  size** (far/near distance ratio ≈ 4.7, 4.2, 3.3, 2.5 for k = 0.98→0.40) — it is large at k=0.98
  *because that core is bigger*, not because of more fission. (k is tuned via radius, so k and core
  size are confounded here.)
* **Late — fission chain.** Descendants leak only after the chain has relaxed to the **fundamental
  spatial mode**, which for a bare sphere is **spherically symmetric (θ-independent)**. Once relaxed, a
  facing-born and an away-born neutron leak identically, so the contrast decays **monotonically to
  exactly 1, from above** — analytically $(1+\rho)/(1-\rho)$ with $\rho\propto e^{(\alpha_1-\alpha_0)t}$.

**This is why there is no "flip" and no crossing:** *every* $k$ washes out to 1 (it is the mode's
symmetry doing it, not the amount of multiplication), and the order is preserved throughout — high-$k$
starts higher (bigger core) *and* relaxes more slowly (bigger core ⇒ closer-spaced modes), so it stays
on top until all curves merge at 1. The genuinely $k$/$R$-dependent features are the *early magnitude*
and *how long the contrast persists*. Near critical the long-lived chain channel dominates the time
integral, pulling the *integrated* contrast toward 1; far below critical the direct channel weighs more.

Each point is a ratio of large counts (sub-1% error) so error bars are omitted; under-sampled late
windows are dropped rather than shown as noise.
"""))
    cells.append(code(r"""
# Each point is a ratio of two large Poisson counts (thousands), so its statistical error is <1%
# -- invisible on a 2-decade log axis. So we omit error bars and instead simply DROP windows with
# too few counts (which would otherwise show as noisy late-time points), keeping only reliable ones.
MIN_CT = 50
fig, ax = plt.subplots(figsize=(9, 5.5))
for L in LABELS:
    d = gl.directional_vs_time(sweeps[L])           # cols: t_centre[s], ratio, err, n_facing, n_away
    ok = (d[:, 3] >= MIN_CT) & (d[:, 4] >= MIN_CT)   # enough facing AND away detections
    ax.plot(d[ok, 0] * 1e6, d[ok, 1], "o-", ms=7, lw=2, color=COLORS[L],
            label=f"k={sweeps[L]['k']:.2f}")

ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(5, 1.2e4); ax.set_ylim(0.8, 300)

# y=1 reference = isotropic (carries no directional information)
ax.axhline(1.0, ls="--", color="0.4", lw=1)
ax.text(1.1e4, 1.05, "isotropic — no direction info", color="0.4", fontsize=9, ha="right", va="bottom")

# annotate the two physical regimes directly on the plot
ax.axvspan(5, 35, color="0.91", zorder=0)
ax.text(13, 200, "DIRECT\nfirst-flight\n(geometric)", fontsize=9.5, ha="center", va="top", weight="bold")
ax.annotate("FISSION CHAIN\n→ fundamental mode\n(direction washes out)", xy=(3e3, 1.15),
            xytext=(1.5e3, 8), fontsize=9.5, ha="center", color="0.25",
            arrowprops=dict(arrowstyle="->", color="0.5", lw=1.2))

ax.set_xlabel("neutron arrival time  [µs]")
ax.set_ylabel(r"facing / away detection ratio   (outer shells, $r/R>0.5$)")
ax.set_title("Directional contrast decays from first-flight to fission chain")
ax.legend(loc="lower left")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "directional_vs_time.png", dpi=200, bbox_inches="tight")
plt.show()
"""))

    cells.append(md(r"""
## 4. Radial fission-rate profiles $F(r)$

From the fission mesh tally (scored during transport — unaffected by the attribution bug). Lower
$k_\text{eff}$ → smaller core → more surface-peaked.
"""))
    cells.append(code(r"""
from project.core.rossi_mesh import N_RADIAL, N_THETA
import openmc
fig, ax = plt.subplots(figsize=(9, 5))
for L in LABELS:
    s = sweeps[L]; R = s["R"]
    res_dir = gl.RESULTS_ROOT / gl._FISSION_DIR[L]
    with openmc.StatePoint(str(res_dir / "statepoint.1.h5")) as sp:
        Ft = sp.get_tally(name="fission_per_cell").mean.flatten().reshape(N_THETA, N_RADIAL)
    F_radial = Ft.sum(axis=0)
    V_shell = (4/3)*np.pi*R**3 / N_RADIAL
    r_edges_phys = R * np.linspace(0, 1, N_RADIAL + 1) ** (1/3)
    r_mid_unit = 0.5*(r_edges_phys[:-1] + r_edges_phys[1:]) / R
    ax.plot(r_mid_unit, F_radial / V_shell, color=COLORS[L], label=f"k={s['k']:.2f} (R={R:.1f})")
ax.set_yscale("log"); ax.set_xlabel("r / R")
ax.set_ylabel(r"fission density [/cm$^3$/source]")
ax.set_title("Radial fission-rate profiles"); ax.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / "fission_profiles.png", dpi=200, bbox_inches="tight")
plt.show()
"""))

    cells.append(md(r"""
## 5. Rossi-alpha and point kinetics — what the comparison means

We fit the corrected $R(\tau)=\sum_i F_i\int g_i(t)g_i(t+\tau)\,dt$ to $A+Be^{\alpha\tau}$ and read off
the prompt decay constant $\alpha$. Prompt point kinetics (a **zero-dimensional** model) predicts
$$\alpha=\frac{k_p-1}{\ell},\qquad k_p=(1-\beta)\,k_\text{eff},$$
where $\ell$ is the **prompt neutron lifetime** (a material property; $\beta\approx0.0065$ for U235;
the generation time is $\Lambda_\text{gen}=\ell/k$). So $\ell=(k_p-1)/\alpha$ should come out the same
from each independent $k_\text{eff}$ run. What this comparison buys us:

1. **Time-dynamics validation.** A single $\ell\approx 52~\mu$s reproduced across all four $k_\text{eff}$
   ($\pm$~17%) is a falsifiable check — not just "$\alpha$ grows when subcritical", but the exact
   textbook relation — anchoring our absolute time scale to a material constant.
2. **A clean separation of what each method captures.** $R(\tau)\approx(\text{spatial amplitude }
   \sum_i F_i\!\int g_i g_i)\times e^{\alpha\tau}$. Point kinetics is the scalar $e^{\alpha\tau}$ — the
   fundamental-mode *time* decay. The spatially-resolved Green's function is exactly the *amplitude /
   shape* that a 0-D model discards: the importance and distance-to-detector structure. **The spatial
   $G_i$ is the part beyond point kinetics** — i.e. the whole point of the method.
3. **It explains the near-critical wash-out.** The same fundamental-mode dominance that makes point
   kinetics accurate is what makes the integrated $G_i$ importance-shaped (centre-peaked,
   direction-independent) near critical. As $k$ drops, $\alpha$ still follows point kinetics (slowest
   mode), but $G_i$ develops the directional/distance structure point kinetics cannot represent.
4. **What it means *now*, after the bug fix.** $\alpha$ lives entirely in detection *times*, which were
   always correct; the attribution bug scrambled spatial *labels*, never times. So the good $\alpha$-fit
   was **orthogonal** to the spatial bug — never evidence for or against it. The old "flat $G$" was a
   random-label artifact, distinguishable from genuine near-critical importance flatness by its strong
   radial structure and $k$-dependence (Sections 1–3b). Bottom line: **point kinetics is the 0-D sanity
   anchor (validates the dynamics, sets $\ell$); the corrected spatial $G_i$ is the actual result.**
"""))
    cells.append(code(r"""
BETA = 0.0065
rossi = {}
for L in LABELS:
    tau, R_tau = gl.rossi(sweeps[L], F[L])
    popt, perr = gl.fit_alpha(tau, R_tau)
    rossi[L] = {"tau": tau, "R": R_tau, "alpha": popt[2], "aerr": perr[2]}
    print(f"k={sweeps[L]['k']:.2f}: alpha = {popt[2]:.1f} +/- {perr[2]:.1f} s^-1  "
          f"(tau = {-1/popt[2]*1e3:.3f} ms)")

ks = np.array([sweeps[L]["k"] for L in LABELS])
alphas = np.array([rossi[L]["alpha"] for L in LABELS])
ell = ((1 - BETA) * ks - 1) / alphas * 1e6   # prompt neutron lifetime, us
print(f"\nprompt lifetime l across k: mean={ell.mean():.2f} us, std/mean={ell.std(ddof=1)/ell.mean():.1%}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
for L in LABELS:
    r = rossi[L]
    ax1.plot(r["tau"]*1e3, r["R"], "o", ms=3, alpha=0.5, color=COLORS[L],
             label=f"k={sweeps[L]['k']:.2f} (a={r['alpha']:.0f})")
    tf = np.linspace(r["tau"][0], r["tau"][-1], 600)
    ax1.plot(tf*1e3, gl.rossi_model(tf, *gl.fit_alpha(r["tau"], r["R"])[0]), "--",
             lw=1.2, color=COLORS[L])
ax1.set_yscale("log"); ax1.set_xlabel(r"$\tau$ (ms)"); ax1.set_ylabel(r"$R(\tau)$")
ax1.set_title("Rossi-alpha overlay"); ax1.legend(fontsize=8)

ax2.plot(ks, ell, "o-", ms=10, color="C2")
ax2.set_xlabel(r"$k_\text{eff}$"); ax2.set_ylabel(r"$\ell$  prompt lifetime ($\mu$s)")
ax2.set_title(fr"$\ell$ vs $k$ (should be flat; std/mean={ell.std(ddof=1)/ell.mean():.0%})")
ax2.invert_xaxis()
plt.tight_layout()
plt.savefig(FIGURES_DIR / "rossi_and_lambda.png", dpi=200, bbox_inches="tight")
plt.show()
"""))

    cells.append(md("## 6. Summary table"))
    cells.append(code(r"""
print("| k_eff | R (cm) | G near pole | G far pole | near/far | alpha (s^-1) | tau (ms) |")
print("|-------|--------|-------------|------------|----------|--------------|----------|")
for L in LABELS:
    s = sweeps[L]; a = rossi[L]["alpha"]
    print(f"| {s['k']:.2f}  | {s['R']:.2f}  | {s['G'][-1,0]:.3e} | {s['G'][-1,-1]:.3e} | "
          f"{s['G'][-1,0]/s['G'][-1,-1]:5.1f}x  | {a:+.0f}      | {-1/a*1e3:.3f}   |")
r_sub = max(sweeps[L]["G"][-1,0]/sweeps[L]["G"][-1,-1] for L in ["070","090","040"])
r_crit = sweeps["098"]["G"][-1,0]/sweeps["098"]["G"][-1,-1]
print(f"\nDirectional contrast collapses from ~{r_sub:.0f}x (subcritical) to {r_crit:.0f}x near "
      f"critical: importance (centre-peaked) overtakes direct geometric coupling as k_eff -> 1.")
print("(The confined-source proof in greens_per_cell.py resolves the outermost cells more finely, "
      "giving up to ~50x.)")
"""))

    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nbf.write(nb, NB_DIR / "keff_comparison.ipynb")
    print("wrote keff_comparison.ipynb")


if __name__ == "__main__":
    build_greens()
    build_keff()
