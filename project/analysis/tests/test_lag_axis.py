"""Regression test: the ACDC lag axis must be labelled tau_L = L*dt.

The discrete lag-L autocorrelation sum_a p_a p_{a+L} / dt estimates
R(tau) = integral g(t) g(t+tau) dt at tau = L*dt -- overlapping two copies of the
bin-integrated histogram L bins apart is a TRIANGULAR smear of half-width dt
CENTRED on L*dt, whose first-order term cancels. Labelling that value at
(L - 0.5)*dt (as greens_loader.rossi and rossi_benchmark.acdc_rossi_events used
to) shifts the whole curve half a bin early, which leaves any fitted alpha alone
(the sequence stays geometric with the same ratio) but biases every ABSOLUTE
comparison against a correctly-labelled estimator by exp(|alpha| dt / 2) --
first order in alpha*dt, ~1.5 % at k=0.90 and ~5 % at k=0.70.

Run:  PYTHONPATH=. python -m pytest project/analysis/tests/test_lag_axis.py -v
"""
import numpy as np
import pytest

from project.analysis import greens_loader as gl

T_MAX = 1.0e-2          # standard sweep record
N_T = 512               # standard bin count
DT = T_MAX / N_T        # 19.53125 us
N_SOURCE = 10**12       # large enough that rossi()'s ">= 2 events" guard passes

# |alpha| [1/s] at k = 0.90 and k = 0.70 (the vetted benchmark anchors)
ALPHAS = [1535.0, 4850.0]

# lags used for the checks: skip L=1,2 (the fits/overlays skip them too) and stop
# at tau = 5 ms, well inside the 10 ms record
L_MIN = 3
L_MAX = int(5e-3 / DT)


def _fake_sweep(a: float) -> dict:
    """One-cell sweep whose g_t is the EXACT bin integral of g(t) = exp(-a t)."""
    edges = np.linspace(0.0, T_MAX, N_T + 1)
    p_bin = (np.exp(-a * edges[:-1]) - np.exp(-a * edges[1:])) / a
    return {
        "g_t": (p_bin * N_SOURCE).reshape(1, 1, N_T),
        "n_source": float(N_SOURCE),
        "time_edges": edges,
    }


def _analytic_R(a: float, tau):
    """R(tau) = int_0^inf e^{-a t} e^{-a(t+tau)} dt = e^{-a tau} / (2a)."""
    return np.exp(-a * np.asarray(tau)) / (2.0 * a)


def _bf_box_average(a: float, tau):
    """Exact box average of R over a bin of width dt CENTRED on tau -- what the
    brute-force pair histogram measures once its bin edges are (L -+ 0.5) dt."""
    tau = np.asarray(tau)
    lo, hi = tau - 0.5 * DT, tau + 0.5 * DT
    return (np.exp(-a * lo) - np.exp(-a * hi)) / (2.0 * a * a * DT)


def _acdc(a: float):
    sweep = _fake_sweep(a)
    tau, R = gl.rossi(sweep, np.array([[1.0]]), n_tau=L_MAX)
    return tau, R


def test_tau_grid_is_L_times_dt():
    """The returned lag axis itself: tau_L = L*dt, not (L - 0.5)*dt."""
    tau, _ = _acdc(ALPHAS[0])
    assert np.allclose(tau, np.arange(1, len(tau) + 1) * DT, rtol=0, atol=1e-15)


@pytest.mark.parametrize("a", ALPHAS)
def test_acdc_matches_analytic_R_on_returned_tau(a):
    """(a) ACDC value / R(tau_returned) == 1 to 2e-3 on 3 <= L <= 5 ms/dt.

    Fails on the old (L-0.5)*dt labelling by exp(|alpha| dt/2): ~1.5 % at
    a=1535, ~4.9 % at a=4850.
    """
    tau, R = _acdc(a)
    sel = slice(L_MIN - 1, L_MAX)
    ratio = R[sel] / _analytic_R(a, tau[sel])
    assert np.max(np.abs(ratio - 1.0)) < 2e-3, (
        f"a={a}: max |ratio-1| = {np.max(np.abs(ratio - 1.0)):.4g}"
    )
    # and the OLD labelling really would have failed by the predicted factor
    old_ratio = R[sel] / _analytic_R(a, tau[sel] - 0.5 * DT)
    assert np.allclose(old_ratio, np.exp(-a * DT / 2), rtol=3e-3)


@pytest.mark.parametrize("a", ALPHAS)
def test_acdc_agrees_with_bf_box_average(a):
    """(b) The two estimators now sit on the same tau grid: ACDC (triangular
    smear centred on L*dt) vs BF (box average over a bin centred on L*dt)."""
    tau, R = _acdc(a)
    sel = slice(L_MIN - 1, L_MAX)
    rel = R[sel] / _bf_box_average(a, tau[sel]) - 1.0
    assert np.max(np.abs(rel)) < 3e-3, f"a={a}: max |rel| = {np.max(np.abs(rel)):.4g}"


@pytest.mark.parametrize("a", ALPHAS)
def test_acdc_log_slope_recovers_alpha(a):
    """(c) The decay constant is unaffected by the labelling (sanity check)."""
    tau, R = _acdc(a)
    sel = slice(L_MIN - 1, L_MAX)
    slope = np.polyfit(tau[sel], np.log(R[sel]), 1)[0]
    assert abs(-slope / a - 1.0) < 1e-3, f"a={a}: fitted {-slope:.2f}"


if __name__ == "__main__":  # plain-script fallback (no pytest needed)
    test_tau_grid_is_L_times_dt()
    for _a in ALPHAS:
        test_acdc_matches_analytic_R_on_returned_tau(_a)
        test_acdc_agrees_with_bf_box_average(_a)
        test_acdc_log_slope_recovers_alpha(_a)
    print("all lag-axis checks passed")
