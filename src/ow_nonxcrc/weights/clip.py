"""Weight clipping: w_tau(x) = min(r_hat(x), tau)."""

import numpy as np


def clip_weights(r_hat_values: np.ndarray, tau: float) -> np.ndarray:
    """Elementwise clip: w_tau = min(r_hat, tau)."""
    r_hat_values = np.asarray(r_hat_values, dtype=np.float64)
    return np.minimum(r_hat_values, float(tau))
