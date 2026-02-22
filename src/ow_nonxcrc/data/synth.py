"""Synthetic covariate shift: Gaussian mean shift, oracle ratio r(x)=exp(mu'x - ||mu||^2/2)."""

from typing import Dict, Optional

import numpy as np


def _oracle_ratio_gaussian(X: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """r(x) = exp(mu'x - ||mu||^2/2) for target N(mu,I) vs source N(0,I)."""
    X = np.asarray(X)
    mu = np.asarray(mu).ravel()
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    dot = X @ mu
    return np.exp(dot - 0.5 * np.dot(mu, mu))


def make_synth_data(
    seed: int,
    n_cal: int,
    n_test: int,
    m_unlabeled: int,
    d: int,
    severity: float,
    n_train: int = 2000,
    split_frac: float = 0.5,
) -> Dict[str, np.ndarray]:
    """Generate synthetic data: source N(0,I), target N(mu,I), ||mu|| = severity.
    Labels: P(Y=1|X) = sigma(theta'X), theta fixed (covariate shift).
    Returns dict with: X_train, Y_train, X_fit, Y_fit, X_cal, Y_cal, X_test, Y_test,
    X_unlabeled, oracle_ratio_cal, oracle_ratio_test, theta (for consistency).
    """
    rng = np.random.default_rng(seed)
    # Theta for P(Y|X) fixed
    theta = rng.standard_normal(d)
    theta = theta / (np.linalg.norm(theta) + 1e-10)
    # Mean for target
    mu = rng.standard_normal(d)
    mu = mu / (np.linalg.norm(mu) + 1e-10) * severity

    # Train: source only
    X_train = rng.standard_normal((n_train, d))
    logit_train = X_train @ theta
    Y_train = (rng.uniform(0, 1, n_train) < 1 / (1 + np.exp(-logit_train))).astype(np.int64)

    # Calibration pool (source) -> split into fit and cal
    n_cal_total = int(np.ceil(n_cal / split_frac))  # so that after split we have ~n_cal
    X_cal_pool = rng.standard_normal((n_cal_total, d))
    logit_cal = X_cal_pool @ theta
    Y_cal_pool = (rng.uniform(0, 1, n_cal_total) < 1 / (1 + np.exp(-logit_cal))).astype(np.int64)
    # Oracle ratio on cal pool is source, so r = 1 for source; we need ratio at cal points as if they were "reference"
    # Actually: D_fit and D_cal are both SOURCE; oracle_ratio for weighting is r(X)= p_target(X)/p_source(X).
    # So on source points, r(x) = exp(mu'x - ||mu||^2/2).
    oracle_cal_pool = _oracle_ratio_gaussian(X_cal_pool, mu)
    idx = rng.permutation(n_cal_total)
    n_fit = int(n_cal_total * split_frac)
    fit_idx, cal_idx = idx[:n_fit], idx[n_fit:]
    X_fit = X_cal_pool[fit_idx]
    Y_fit = Y_cal_pool[fit_idx]
    X_cal = X_cal_pool[cal_idx]
    Y_cal = Y_cal_pool[cal_idx]
    oracle_ratio_fit = oracle_cal_pool[fit_idx]
    oracle_ratio_cal = oracle_cal_pool[cal_idx]

    # Test: target
    X_test = rng.standard_normal((n_test, d)) + mu
    logit_test = X_test @ theta
    Y_test = (rng.uniform(0, 1, n_test) < 1 / (1 + np.exp(-logit_test))).astype(np.int64)
    oracle_ratio_test = _oracle_ratio_gaussian(X_test, mu)

    # Unlabeled target
    X_unlabeled = rng.standard_normal((m_unlabeled, d)) + mu

    return {
        "X_train": X_train,
        "Y_train": Y_train,
        "X_fit": X_fit,
        "Y_fit": Y_fit,
        "X_cal": X_cal,
        "Y_cal": Y_cal,
        "X_test": X_test,
        "Y_test": Y_test,
        "X_unlabeled": X_unlabeled,
        "oracle_ratio_fit": oracle_ratio_fit,
        "oracle_ratio_cal": oracle_ratio_cal,
        "oracle_ratio_test": oracle_ratio_test,
        "theta": theta,
    }
