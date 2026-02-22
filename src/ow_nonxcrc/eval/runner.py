"""Common runner: make_dataset_synth, fit_predictor_synth, get_weights, run_one_trial."""

import time
from typing import Any, Dict, List, Optional

import numpy as np

from ..data import make_synth_data
from ..models import LogisticClassifier, MLPClassifier
from ..ratio import fit_domain_ratio
from ..weights import clip_weights, neff
from ..crc import loss_matrix, accept_rate_utility, crc_select
from ..utils import set_seed, make_lambda_grid
from .metrics import violation as violation_fn


def make_dataset_synth(
    seed: int,
    n_cal: int,
    n_test: int,
    m_unlabeled: int,
    d: int,
    severity: float,
    n_train: int = 2000,
    split_frac: float = 0.5,
) -> Dict[str, np.ndarray]:
    """Wrapper over data.make_synth_data."""
    return make_synth_data(
        seed=seed,
        n_cal=n_cal,
        n_test=n_test,
        m_unlabeled=m_unlabeled,
        d=d,
        severity=severity,
        n_train=n_train,
        split_frac=split_frac,
    )


def fit_predictor_synth(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    model_type: str = "logreg",
    **model_kw,
):
    """Train classifier f on source train. Returns predictor with predict_proba."""
    if model_type == "logreg":
        clf = LogisticClassifier(**model_kw)
    elif model_type == "mlp":
        clf = MLPClassifier(**model_kw)
    else:
        raise ValueError(f"Unknown model_type {model_type}")
    clf.fit(X_train, Y_train)
    return clf


def get_weights(
    method: str,
    X_fit: np.ndarray,
    X_cal: np.ndarray,
    X_unlabeled: np.ndarray,
    oracle_ratio_cal: Optional[np.ndarray],
    tau: float,
    ratio_model: str = "logreg",
    C: float = 1.0,
    C_prime: float = 0.5,
) -> np.ndarray:
    """Return weight vector for calibration set (length = len(X_cal)).
    method: uniform | oracle | oracle+clip | learned | learned+clip | naive-learned-no-split.
    """
    n_cal = len(X_cal)
    if method == "uniform":
        return np.ones(n_cal, dtype=np.float64)
    if method == "oracle":
        if oracle_ratio_cal is None:
            return np.ones(n_cal, dtype=np.float64)
        return np.asarray(oracle_ratio_cal, dtype=np.float64).ravel()
    if method == "oracle+clip":
        if oracle_ratio_cal is None:
            return np.ones(n_cal, dtype=np.float64)
        return clip_weights(oracle_ratio_cal, tau)
    if method == "learned":
        r_hat_fn, _, _ = fit_domain_ratio(X_fit, X_unlabeled, model=ratio_model)
        return r_hat_fn(X_cal)
    if method == "learned+clip":
        r_hat_fn, _, _ = fit_domain_ratio(X_fit, X_unlabeled, model=ratio_model)
        return clip_weights(r_hat_fn(X_cal), tau)
    if method == "naive-learned-no-split":
        # Double-dip: fit ratio on X_cal vs X_unlabeled, then w = r_hat(X_cal)
        r_hat_fn, _, _ = fit_domain_ratio(X_cal, X_unlabeled, model=ratio_model)
        return r_hat_fn(X_cal)
    raise ValueError(f"Unknown method {method}")


def run_one_trial(
    seed: int,
    data: Dict[str, np.ndarray],
    predictor: Any,
    lambda_grid: np.ndarray,
    r_target: float,
    delta: float,
    method: str,
    tau: float,
    loss_c: float,
    ratio_model: str,
    C: float = 1.0,
    C_prime: float = 0.5,
) -> Dict[str, Any]:
    """Run one (method, tau) trial. Returns one row dict for CSV/JSON."""
    set_seed(seed)
    X_fit = data["X_fit"]
    X_cal = data["X_cal"]
    Y_cal = data["Y_cal"]
    X_test = data["X_test"]
    Y_test = data["Y_test"]
    X_unlabeled = data["X_unlabeled"]
    oracle_ratio_cal = data.get("oracle_ratio_cal")
    n_cal = len(X_cal)
    m = len(X_unlabeled)

    # Weights for cal set
    t0_ratio = time.perf_counter()
    w = get_weights(
        method=method,
        X_fit=X_fit,
        X_cal=X_cal,
        X_unlabeled=X_unlabeled,
        oracle_ratio_cal=oracle_ratio_cal,
        tau=tau,
        ratio_model=ratio_model,
    )
    runtime_fit_ratio = time.perf_counter() - t0_ratio

    # Loss matrix on cal and utility on cal
    probs_cal = predictor.predict_proba(X_cal)
    L_cal = loss_matrix(probs_cal, Y_cal, lambda_grid, c=loss_c)
    util_cal = accept_rate_utility(probs_cal, lambda_grid)

    t0_crc = time.perf_counter()
    lambda_hat, slack, U_at_hat, n_eff = crc_select(
        L_cal,
        w,
        r_target=r_target,
        delta=delta,
        tau=tau,
        lambda_grid=lambda_grid,
        util_per_lambda=util_cal,
        C=C,
        C_prime=C_prime,
    )
    runtime_crc = time.perf_counter() - t0_crc

    # Evaluate on test
    probs_test = predictor.predict_proba(X_test)
    L_test_one = loss_matrix(probs_test, Y_test, np.array([lambda_hat]), c=loss_c)
    risk_test = float(np.mean(L_test_one))
    util_test = accept_rate_utility(probs_test, np.array([lambda_hat]))[0]
    viol = violation_fn(risk_test, r_target)

    return {
        "seed": seed,
        "n": n_cal,
        "m": m,
        "target_r": r_target,
        "delta": delta,
        "method": method,
        "tau": tau,
        "neff": n_eff,
        "lambda_hat": lambda_hat,
        "achieved_risk_test": risk_test,
        "slack": slack,
        "utility": util_test,
        "violation": viol,
        "runtime_fit_ratio": runtime_fit_ratio,
        "runtime_crc": runtime_crc,
    }


def run_one_trial_tau_star(
    seed: int,
    data: Dict[str, np.ndarray],
    predictor: Any,
    lambda_grid: np.ndarray,
    r_target: float,
    delta: float,
    method: str,
    tau_list: List[float],
    loss_c: float,
    ratio_model: str,
    C: float = 1.0,
    C_prime: float = 0.5,
) -> Dict[str, Any]:
    """Run for method that uses tau; select tau_star = argmin Rad(w_tau) over tau_list, then report that row."""
    from ..crc.bounds import Rad
    X_fit = data["X_fit"]
    X_cal = data["X_cal"]
    X_unlabeled = data["X_unlabeled"]
    oracle_ratio_cal = data.get("oracle_ratio_cal")
    n_cal = len(X_cal)
    best_tau = tau_list[0]
    best_rad = float("inf")
    best_row = None
    for tau in tau_list:
        w = get_weights(
            method=method,
            X_fit=X_fit,
            X_cal=X_cal,
            X_unlabeled=X_unlabeled,
            oracle_ratio_cal=oracle_ratio_cal,
            tau=tau,
            ratio_model=ratio_model,
        )
        rad = Rad(w, n_cal, delta, tau, C=C, C_prime=C_prime)
        if rad < best_rad:
            best_rad = rad
            best_tau = tau
    row = run_one_trial(
        seed=seed,
        data=data,
        predictor=predictor,
        lambda_grid=lambda_grid,
        r_target=r_target,
        delta=delta,
        method=method,
        tau=best_tau,
        loss_c=loss_c,
        ratio_model=ratio_model,
        C=C,
        C_prime=C_prime,
    )
    row["tau_star"] = best_tau
    return row
