"""Non-X CRC bounds: Rhat_w, Rad, U_w.

Theorem 2: Rad(w,n,delta) <= C * sqrt(log(1/delta)/n_eff) + C' * tau * log(1/delta) / sum(w).
"""

import numpy as np

from ..weights.neff import neff


def Rhat_w(L_j: np.ndarray, w: np.ndarray) -> float:
    """Weighted empirical risk for one lambda (column L_j).
    w should be normalized (sum to 1) for calibration set.
    """
    w = np.asarray(w, dtype=np.float64).ravel()
    L_j = np.asarray(L_j, dtype=np.float64).ravel()
    if w.shape[0] != L_j.shape[0]:
        raise ValueError("w and L_j length mismatch")
    s = np.sum(w)
    if s <= 0:
        return 0.0
    w_norm = w / s
    return float(np.dot(w_norm, L_j))


def Rad(w: np.ndarray, n: int, delta: float, tau: float, C: float = 1.0, C_prime: float = 0.5) -> float:
    """Uncertainty term for U_w. Theorem 2 style.
    Rad <= C * sqrt(log(1/delta)/n_eff) + C' * tau * log(1/delta) / sum(w).
    """
    w = np.asarray(w, dtype=np.float64).ravel()
    n_eff = neff(w)
    s = np.sum(w)
    if n_eff <= 0 or s <= 0:
        return 1.0  # conservative
    logd = np.log(1.0 / max(delta, 1e-10))
    term1 = C * np.sqrt(logd / n_eff)
    term2 = C_prime * tau * logd / s
    return float(term1 + term2)


def U_w(
    Rhat_val: float,
    w: np.ndarray,
    n: int,
    delta: float,
    tau: float,
    C: float = 1.0,
    C_prime: float = 0.5,
) -> float:
    """Upper bound: U_w(lambda; delta) = Rhat_w(lambda) + Rad(w, n, delta, tau)."""
    return Rhat_val + Rad(w, n, delta, tau, C=C, C_prime=C_prime)
