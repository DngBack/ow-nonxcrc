"""Effective sample size: n_eff = (sum w)^2 / sum(w^2)."""

import numpy as np


def neff(w: np.ndarray) -> float:
    """Effective sample size for weighted sample."""
    w = np.asarray(w, dtype=np.float64).ravel()
    s = np.sum(w)
    s2 = np.sum(w * w)
    if s2 <= 0:
        return 0.0
    return float(s * s / s2)
