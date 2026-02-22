"""CRC selector: choose lambda_hat s.t. U_w(lambda) <= r, maximize utility. Slack = max(0, U_w(lambda_hat)-r)."""

import numpy as np

from .bounds import Rhat_w, Rad, U_w
from ..weights.neff import neff


def crc_select(
    L: np.ndarray,
    w: np.ndarray,
    r_target: float,
    delta: float,
    tau: float,
    lambda_grid: np.ndarray,
    util_per_lambda: np.ndarray,
    C: float = 1.0,
    C_prime: float = 0.5,
):
    """Select lambda_hat: feasible set {lambda : U_w(lambda) <= r}, then argmax utility.
    If feasible set empty, choose conservative (max lambda).
    Returns: lambda_hat, slack, U_at_hat, n_eff.
    """
    L = np.asarray(L)
    w = np.asarray(w, dtype=np.float64).ravel()
    n_cal = L.shape[0]
    n_lambda = L.shape[1]
    if w.shape[0] != n_cal or len(lambda_grid) != n_lambda or len(util_per_lambda) != n_lambda:
        raise ValueError("L, w, lambda_grid, util_per_lambda shape mismatch")
    w_sum = np.sum(w)
    if w_sum <= 0:
        w_norm = np.ones(n_cal) / n_cal
    else:
        w_norm = w / w_sum
    # For each lambda index j: Rhat_w(L[:,j], w_norm) then U_w
    Rhat_vals = np.array([Rhat_w(L[:, j], w_norm) for j in range(n_lambda)])
    rad = Rad(w, n_cal, delta, tau, C=C, C_prime=C_prime)
    U_vals = Rhat_vals + rad
    feasible = U_vals <= r_target
    if np.any(feasible):
        # Among feasible, pick argmax utility
        util_feas = np.where(feasible, util_per_lambda, -np.inf)
        j_hat = int(np.argmax(util_feas))
    else:
        # Conservative: max lambda (largest index, typically higher threshold)
        j_hat = n_lambda - 1
    lambda_hat = float(lambda_grid[j_hat])
    U_at_hat = float(U_vals[j_hat])
    slack = max(0.0, U_at_hat - r_target)
    n_eff_val = neff(w)
    return lambda_hat, slack, U_at_hat, n_eff_val
