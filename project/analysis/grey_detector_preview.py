"""Grey-detector preview from the EXISTING black-detector event list (zero transport).

The black B-10 sphere (100 atom/b-cm) captures every neutron that enters it, at its
incoming energy, and `collision_track` records that energy. So the spectrum of
neutrons ENTERING the detector is already on disk, and for any realistic B-10
number density N the capture probability of an entering neutron is

    P_cap(E) ~ 1 - exp(-N sigma_a(E) * chord),   chord = 4R/3 (sphere, 5 cm)

(scattering in a dilute gas neglected). Weighting the recorded captures by P_cap
gives, at zero cost: the grey detector's efficiency relative to black, the energy
mix it actually detects, and its detection-time spectrum. This is the design input
for the grey-detector study (see outputs/presentation/2026-10-14_mentor_catchup).

RUN (repo root, PYTHONPATH=.):  python -m project.analysis.grey_detector_preview [label]
"""
from __future__ import annotations

import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import h5py
import numpy as np

OUT_DIR = Path(__file__).resolve().parent / "outputs" / "grey_detector_preview"
RESULTS = Path(__file__).resolve().parents[1] / "results"
DENSITIES = (1e2, 1e-2, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6)   # atom/b-cm of B-10
BF3_1ATM = 2.7e-5    # ~ B-10 density of enriched BF3 at 1 atm, 20 C (reference point)
CHORD = 4 * 5.0 / 3  # mean chord of the 5 cm detector sphere


def b10_sigma_a(E: np.ndarray) -> np.ndarray:
    import openmc
    from openmc.data import IncidentNeutron

    lib = openmc.config["cross_sections"]
    root = ET.parse(lib).getroot()
    path = [os.path.join(os.path.dirname(lib), e.get("path"))
            for e in root if e.get("materials") == "B10"][0]
    b10 = IncidentNeutron.from_hdf5(path)
    T = sorted(b10.temperatures, key=lambda s: abs(float(s[:-1]) - 294))[0]
    return b10[107].xs[T](E) + b10[102].xs[T](E)       # (n,alpha) + (n,gamma)


def preview(label: str = "090", sphere_dir: str | None = None) -> dict:
    sphere_dir = sphere_dir or f"greens-function-k{label}-bench100M"
    with h5py.File(RESULTS / sphere_dir / "collision_track.h5") as f:
        a = f["collision_track_bank"]
        E, t, dg = a["E"][()], a["time"][()], a["delayed_group"][()]
    prompt = dg == 0
    sig = b10_sigma_a(E)
    edges = np.linspace(0, 10e-3, 101)
    h_black, _ = np.histogram(t[prompt], bins=edges)
    rows, gt = [], {"edges": edges, "black": h_black}
    for N in DENSITIES:
        pc = 1 - np.exp(-N * sig * CHORD)
        h, _ = np.histogram(t[prompt], bins=edges, weights=pc[prompt])
        gt[f"{N:.0e}"] = h
        rows.append({
            "N_B10": N, "eps_rel": float(pc.mean()), "eps_rel_prompt": float(pc[prompt].mean()),
            "thermal_fraction_of_captures": float((pc * (E < 0.625)).sum() / pc.sum()),
            "early_1ms_fraction_vs_black": float((h[:10].sum() / h.sum()) / (h_black[:10].sum() / h_black.sum())),
        })
    out = {
        "label": label, "sphere_dir": sphere_dir, "n_events": int(len(E)),
        "entering_spectrum": {
            "frac_below_0.625eV": float((E < 0.625).mean()), "frac_below_1keV": float((E < 1e3).mean()),
            "frac_above_100keV": float((E > 1e5).mean()), "median_eV": float(np.median(E)),
        },
        "chord_cm": CHORD, "bf3_1atm_reference_density": BF3_1ATM, "scan": rows,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"preview_k{label}.json").write_text(json.dumps(out, indent=2))
    np.savez_compressed(OUT_DIR / f"gt_k{label}.npz", **gt)
    return out, gt


def make_figure(label: str = "090"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INK, GREY, CRIMSON, BLUE, ORANGE = "#30363D", "#8A939B", "#C0392B", "#1F77B4", "#C4620E"
    out, gt = preview(label)
    N = np.array([r["N_B10"] for r in out["scan"]])
    eps = np.array([r["eps_rel"] for r in out["scan"]])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ok = N < 1
    ax.loglog(N[ok], eps[ok], "o-", color=BLUE, label="efficiency relative to black (entering neutrons)")
    ax.loglog(N[ok], eps[ok] ** 2, "s--", color=CRIMSON, label="pair rate relative to black (∝ ε²)")
    ax.axvline(BF3_1ATM, color=GREY, lw=1, ls=":")
    ax.text(BF3_1ATM * 1.1, 2e-4, "BF₃, 1 atm", color=GREY, fontsize=8)
    ax.set_xlabel("B-10 number density [atom/b·cm]")
    ax.set_ylabel("relative to the black detector")
    ax.set_title("Dialling the detector from black to grey", color=INK)
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1]
    c = 0.5 * (gt["edges"][1:] + gt["edges"][:-1]) * 1e3
    ax.semilogy(c, gt["black"] / gt["black"].sum(), "-", color=INK, label="black (all entering neutrons)")
    for key, col in (("1e-04", BLUE), ("1e-05", ORANGE)):
        th = [r for r in out["scan"] if f"{r['N_B10']:.0e}" == key][0]["thermal_fraction_of_captures"]
        ax.semilogy(c, gt[key] / gt[key].sum(), "-", color=col,
                    label=f"grey, N = {key} (1/v: ~{th * 100:.0f} % thermal)")
    ax.set_xlabel("detection time [ms]")
    ax.set_ylabel("normalised g(t)")
    ax.set_title("Shape barely moves: timing is chain-driven, not slowing-down", color=INK, fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"grey_preview_k{label}.png", dpi=170)
    print(json.dumps(out["entering_spectrum"], indent=2))
    for r in out["scan"]:
        print(f"N={r['N_B10']:.0e}  eps_rel={r['eps_rel']:.4f}  thermal share={r['thermal_fraction_of_captures']:.2f}  "
              f"early fraction x{r['early_1ms_fraction_vs_black']:.2f}")


if __name__ == "__main__":
    make_figure(sys.argv[1] if len(sys.argv) > 1 else "090")
