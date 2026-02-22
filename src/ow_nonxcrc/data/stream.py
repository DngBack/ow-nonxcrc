"""Streaming drift: first half P1, second half P2 (Gaussian mean shift)."""

from typing import Dict

import numpy as np


def make_stream_data(
    seed: int,
    T: int,
    n_window: int,
    d: int,
    severity_second_half: float = 1.5,
) -> Dict[str, np.ndarray]:
    """Generate stream: t=1..T, first half N(0,I), second half N(mu,I).
    Returns: X_stream (T, d), Y_stream (T,), time_domain (T,) 0 or 1, mu for target.
    """
    rng = np.random.default_rng(seed)
    half = T // 2
    mu = rng.standard_normal(d)
    mu = mu / (np.linalg.norm(mu) + 1e-10) * severity_second_half
    theta = rng.standard_normal(d)
    theta = theta / (np.linalg.norm(theta) + 1e-10)
    X1 = rng.standard_normal((half, d))
    X2 = rng.standard_normal((T - half, d)) + mu
    X_stream = np.vstack([X1, X2])
    logit = X_stream @ theta
    Y_stream = (rng.uniform(0, 1, T) < 1 / (1 + np.exp(-logit))).astype(np.int64)
    time_domain = np.array([0] * half + [1] * (T - half), dtype=np.int64)
    return {
        "X_stream": X_stream,
        "Y_stream": Y_stream,
        "time_domain": time_domain,
        "mu": mu,
        "theta": theta,
        "changepoint": half,
    }
