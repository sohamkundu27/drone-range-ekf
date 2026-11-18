"""Filter consistency metrics for evaluating the EKF in simulation (stub).

Planned contents:
- `nees(x_true, x_est, P) -> e`, the normalised estimation error squared per step,
  e_k = (x - x_hat)^T P^{-1} (x - x_hat).
- `nis(innov, S)`, the normalised innovation squared for each measurement update.
- `anees(...)`: average over N Monte-Carlo runs, with two-sided chi-square bounds
  (scipy.stats.chi2) at 95%, to tell whether the filter is consistent, overconfident
  or underconfident.
- Position, range and range-rate RMSE versus time and versus true range.
- Plot helpers that save to figures/: NEES vs time with bounds, and range error vs range.
"""
