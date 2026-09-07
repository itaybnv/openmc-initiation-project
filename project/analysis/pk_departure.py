"""Band-median fundamental |alpha| per k, with an honest bootstrap sigma.

PK-departure audit (2026-07-12; see outputs/presentation/PK_DEPARTURE_AUDIT.md):
the benchmark notebook's "Point-kinetics departure" panel (section 1) had been
computed from the WINDOW-FIT ACDC alpha (rb.acdc_alpha_sigma /
kre.acdc_alpha_sigma_ext). The frozen fit windows start at the plateau-criterion
edge, where the second-moment fit still carries a few-percent higher-mode tail
at mid k (k=0.90: window-fit 1535 vs band-median fundamental 1433 vs PK 1420),
so the plotted "departure" was mostly window systematic, not physics. The claim
"PK misses spatial modes" must be quantified against the FUNDAMENTAL-mode alpha
-- the self-consistent band median of the window-start scan that section 0b
itself draws as the dashed reference line -- not against the practical-window
fit.

This module provides exactly that quantity per label, reusing the audited
rossi_benchmark machinery verbatim (acdc_rossi -> plateau_scan -> _alpha_fund;
same scan parameters as rb.plateau_window, so the point value here is
IDENTICAL to the alpha_fund the section-0b figure shows). The sigma is a
per-bin Poisson bootstrap of the sweep g_t, re-running the full scan + band
median per replicate, scaled by KAPPA_S[label] -- the same chain-aware error
model the audited rb.acdc_alpha_sigma uses. k="098" runs on the extended 40 ms
record (k098_record_extension), matching kre.derive_window()['alpha_fund'].

rossi_benchmark.py itself is untouched (additive-module constraint).
"""
import numpy as np

from project.analysis import greens_loader as gl
from project.analysis import k098_record_extension as kre
from project.analysis import rossi_benchmark as rb


def _scan_fund(sweep, F, ratio=4.0, n_scan=40):
    """One full window-start scan + band median on an explicit sweep dict.

    Mirrors the scan inside rb.plateau_window / kre.plateau_window_ext: full
    available lag range from the sweep's own record (n_t - 1 lags), same
    plateau_scan parameters, same _alpha_fund band rule.
    """
    n_t = sweep["g_t"].shape[-1]
    tau, R = rb.acdc_rossi(sweep, F, n_tau=n_t - 1)
    lo_grid, a_grid = rb.plateau_scan(tau, R, ratio=ratio, n_scan=n_scan)
    return rb._alpha_fund(lo_grid, a_grid), lo_grid, a_grid


# Band-choice variants: the audited rb._alpha_fund band (0.8, 2.5) plus an
# earlier and a later reading of the quasi-plateau. The EARLY variant is a
# known-biased read (it re-includes the pre-plateau higher-mode descent the
# 0.5 tau_d eligibility floor exists to exclude; measured at k=0.90 it jumps
# 1433 -> 1653 while the plateau proper is flat to 0.4 s^-1), so it is reported
# as a diagnostic but NOT used for the error. The band-read systematic is the
# LATE shift |median(0.8,2.5) - median(1.0,3.0)|: at low k the scan keeps
# descending past the plateau (background pull), which biases the fundamental
# LOW -- i.e. this systematic is conservative for the "fundamental sits above
# PK" departure claim.
ALT_BANDS = ((0.6, 2.0), (0.8, 2.5), (1.0, 3.0))
_AUDITED_BAND, _LATE_BAND = (0.8, 2.5), (1.0, 3.0)


def alpha_fund_sigma(label: str, n_boot: int = 30, rng=None) -> dict:
    """Band-median fundamental alpha for one k label, with an honest sigma.

    Point value: identical to rb.plateau_window(label)['alpha_fund'] (and, for
    label='098', to kre.derive_window()['alpha_fund'] on the extended record).

    Error model (sigma_total = quadrature of the two):
    * statistical: std of the band median over n_boot per-bin Poisson resamples
      of g_t, x rb.KAPPA_S[label] (same chain-aware error model as the audited
      rb.acdc_alpha_sigma);
    * band-read systematic: the late-band shift (see ALT_BANDS note) -- at low
      k the scan descends across the band instead of flattening (measured:
      plateau clean at k>=0.90, quasi-plateau below), so where you read the
      median is a genuine systematic and is propagated, not hidden.

    Returns a dict with alpha_fund, sigma_total (use this), the components
    (sigma_stat, band_sys), the per-band medians, and plateau diagnostics
    (n_band points, in-band scatter).
    """
    rng = np.random.default_rng(rng)
    if label == "098":
        # extended 40 ms record (K098_EXTENSION_REPORT.md); the 10 ms record
        # truncates the assembly at this k and biases the whole scan.
        sweep = kre.load_ext_sweep()
    else:
        sweep = gl.load_sweep(label)
    F = rb.fission_coarse_sim(label, *sweep["G"].shape)

    af, lo_grid, a_grid = _scan_fund(sweep, F)
    prod = lo_grid * np.abs(a_grid)
    sel = np.isfinite(a_grid) & (prod >= 0.8) & (prod <= 2.5)   # _alpha_fund band
    n_band = int(sel.sum())
    band_scatter = float(np.std(a_grid[sel], ddof=1)) if n_band > 1 else np.nan

    # band-read systematic: median under earlier/audited/later plateau reads;
    # error uses the LATE shift only (early read is known-biased, see ALT_BANDS)
    band_medians = {}
    for b_lo, b_hi in ALT_BANDS:
        s = np.isfinite(a_grid) & (prod >= b_lo) & (prod <= b_hi)
        band_medians[(b_lo, b_hi)] = float(np.nanmedian(a_grid[s])) if s.any() else np.nan
    m_aud, m_late = band_medians[_AUDITED_BAND], band_medians[_LATE_BAND]
    band_sys = (float(abs(abs(m_aud) - abs(m_late)))
                if np.isfinite(m_aud) and np.isfinite(m_late) else np.nan)

    g_t = sweep["g_t"]
    boots = []
    for _ in range(n_boot):
        s2 = dict(sweep)
        s2["g_t"] = rng.poisson(g_t).astype(float)
        b, _, _ = _scan_fund(s2, F)
        if np.isfinite(b):
            boots.append(b)
    sigma_raw = float(np.std(boots, ddof=1)) if len(boots) > 1 else np.nan
    sigma_stat = sigma_raw * rb.KAPPA_S[label]
    sigma_total = float(np.hypot(sigma_stat, band_sys))

    return {
        "label": label, "alpha_fund": float(af),
        "sigma_total": sigma_total, "sigma_stat": float(sigma_stat),
        "band_sys": band_sys, "sigma_boot_raw": sigma_raw,
        "band_medians": band_medians,
        "kappa": rb.KAPPA_S[label], "n_boot_ok": len(boots),
        "n_band": n_band, "band_scatter": band_scatter,
        "lo_grid": lo_grid, "alpha_grid": a_grid,
    }
