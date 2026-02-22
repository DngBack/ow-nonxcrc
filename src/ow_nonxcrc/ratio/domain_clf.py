"""Domain classifier for density ratio: r_hat(x) = p_t(x)/p_s(x) via d(x)=P(target|x).

r_hat(x) = d(x)/(1-d(x)) * (1-pi)/pi, pi = m/(m+n).
"""

import time
from typing import Callable, Tuple

import numpy as np

from ..models import LogisticClassifier, MLPClassifier


def fit_domain_ratio(
    X_fit: np.ndarray,
    X_target: np.ndarray,
    model: str = "logreg",
    **model_kw,
) -> Tuple[Callable[[np.ndarray], np.ndarray], np.ndarray, float]:
    """Train domain classifier: source (D_fit) vs target (U).
    Labels: 0 = source, 1 = target.
    Returns: (r_hat_callable, r_hat_on_fit, runtime_sec).
    """
    t0 = time.perf_counter()
    n, m = len(X_fit), len(X_target)
    X = np.vstack([np.asarray(X_fit), np.asarray(X_target)])
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    y = np.array([0] * n + [1] * m, dtype=np.int64)
    pi = m / (n + m)
    if model == "logreg":
        clf = LogisticClassifier(**model_kw)
    elif model == "mlp":
        clf = MLPClassifier(**model_kw)
    else:
        raise ValueError(f"Unknown model {model}")
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    # P(target|x) = proba[:, 1]; P(source|x) = proba[:, 0]
    d = proba[:, 1]
    # r(x) = p_t/p_s = (d/(1-d)) * ((1-pi)/pi)
    r_all = (d / (1 - d + 1e-10)) * ((1 - pi) / (pi + 1e-10))
    r_all = np.clip(r_all, 0, 1e10)
    r_fit = r_all[:n]
    runtime = time.perf_counter() - t0

    def r_hat(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        proba = clf.predict_proba(X)
        d = proba[:, 1]
        r = (d / (1 - d + 1e-10)) * ((1 - pi) / (pi + 1e-10))
        return np.clip(r, 0, 1e10)

    return r_hat, r_fit, runtime
