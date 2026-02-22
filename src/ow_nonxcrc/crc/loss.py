"""Monotone loss L_lambda and utility (accept rate).

Policy: predict if max p >= lambda else reject.
Loss: L_lambda(x,y) = 1[max p < lambda] + c * 1[yhat != y], c in (0,1).
"""

import numpy as np


def loss_matrix(
    probs: np.ndarray,
    y_true: np.ndarray,
    lambda_grid: np.ndarray,
    c: float = 0.5,
) -> np.ndarray:
    """Compute loss matrix L[i, j] = L_lambda_j(x_i, y_i).

    probs: (n,) or (n, n_classes) - max over classes is used as confidence.
    y_true: (n,) integer labels.
    lambda_grid: (n_lambda,) thresholds.
    c: weight for misclassification term in (0, 1).

    Returns: (n, n_lambda) array. Monotone in j (larger lambda -> larger loss).
    """
    probs = np.asarray(probs)
    if probs.ndim == 2:
        max_p = np.max(probs, axis=1)
        yhat = np.argmax(probs, axis=1)
    else:
        max_p = np.asarray(probs).ravel()
        yhat = np.zeros(len(max_p), dtype=int)
    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    n = len(y_true)
    n_lambda = len(lambda_grid)
    # L[i,j] = 1[max_p_i < lambda_j] + c * 1[yhat_i != y_i]
    reject = (max_p.reshape(-1, 1) < lambda_grid.reshape(1, -1)).astype(np.float64)
    wrong = (yhat != y_true).astype(np.float64).reshape(-1, 1)
    L = reject + c * wrong
    return L


def accept_rate_utility(
    probs: np.ndarray,
    lambda_grid: np.ndarray,
) -> np.ndarray:
    """Utility = accept rate = mean(1[max p >= lambda]) per lambda.
    probs: (n,) or (n, n_classes). lambda_grid: (n_lambda,).
    Returns: (n_lambda,) utility per lambda (higher = more accepts = better utility).
    """
    if np.asarray(probs).ndim == 2:
        max_p = np.max(probs, axis=1)
    else:
        max_p = np.asarray(probs).ravel()
    n = len(max_p)
    # (n, n_lambda): 1 if accept
    accept = (max_p.reshape(-1, 1) >= lambda_grid.reshape(1, -1)).astype(np.float64)
    return np.mean(accept, axis=0)
