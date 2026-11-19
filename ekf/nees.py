"""Filter consistency checks: NEES and NIS over Monte Carlo runs.

A filter is CONSISTENT if the errors it actually makes match the uncertainty it reports.

NEES (normalised estimation error squared), at one time step of one run:
    e = (x_true - x_hat)^T P^{-1} (x_true - x_hat)
If the filter is consistent (errors are zero-mean Gaussian with covariance P), e follows a
chi-square distribution with d degrees of freedom (d = number of state components checked),
so its mean is d.

NIS (normalised innovation squared) is the same idea for the measurement residual:
    nis = y^T S^{-1} y,   y = z - h(x_hat),  S = innovation covariance,
chi-square with m = measurement dimension. NIS can be computed on real data (no truth needed).

ANEES: the average of e over N Monte Carlo runs at the same time step. N * ANEES is
chi-square with N*d degrees of freedom, which gives a two-sided 95% interval:
    [chi2.ppf(0.025, N d) / N,  chi2.ppf(0.975, N d) / N]
  ANEES above the interval -> errors bigger than P claims  -> OVERCONFIDENT (dangerous)
  ANEES below the interval -> errors smaller than P claims -> conservative (safe, wasteful)
"""
import numpy as np
from scipy.stats import chi2


def nees(err, P):
    """e = err^T P^{-1} err  (err = x_true - x_hat, any dimension)."""
    err = np.atleast_1d(err)
    P = np.atleast_2d(P)
    return float(err @ np.linalg.solve(P, err))


def nis(innov, S):
    """Same formula, applied to the innovation y and its covariance S."""
    return nees(innov, S)


def anees_bounds(dof, n_runs, prob=0.95):
    """Two-sided chi-square interval for the average of n_runs NEES values."""
    lo = chi2.ppf((1 - prob) / 2, n_runs * dof) / n_runs
    hi = chi2.ppf((1 + prob) / 2, n_runs * dof) / n_runs
    return lo, hi


def consistency_report(nees_runs, dof, prob=0.95, skip=0):
    """Summarise a (n_runs, n_steps) array of NEES (or NIS) values; NaN = no value.
    Returns the ANEES curve over time and the fraction of time steps inside the bounds.
    `skip` ignores the first steps (initial transient)."""
    nees_runs = np.asarray(nees_runs, float)[:, skip:]
    n_valid = np.sum(~np.isnan(nees_runs), axis=0)
    anees_t = np.nanmean(nees_runs, axis=0)
    n_runs = int(np.median(n_valid))
    lo, hi = anees_bounds(dof, n_runs, prob)
    inside = (anees_t >= lo) & (anees_t <= hi)
    return {
        "dof": dof, "n_runs": n_runs, "bounds": [float(lo), float(hi)],
        "anees_mean": float(np.nanmean(anees_t)),
        "anees_median": float(np.nanmedian(anees_t)),
        "frac_steps_inside": float(np.mean(inside)),
        "frac_steps_above": float(np.mean(anees_t > hi)),
        "frac_steps_below": float(np.mean(anees_t < lo)),
        "verdict": verdict(np.nanmean(anees_t), lo, hi),
        "anees_t": anees_t,
    }


def verdict(anees, lo, hi):
    if anees > hi:
        return "overconfident"
    if anees < lo:
        return "conservative"
    return "consistent"
