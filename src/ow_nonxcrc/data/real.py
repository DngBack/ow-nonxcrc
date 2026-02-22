"""Real domain shift: CIFAR-10 + noise as corruption proxy, or tabular Adult."""

from typing import Dict, Optional

import numpy as np


def load_cifar10c(
    severity: int = 1,
    n_cal: int = 1000,
    n_test: int = 1000,
    m_unlabeled: int = 1000,
    seed: int = 0,
    data_dir: Optional[str] = None,
    corruption: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """Load CIFAR-10 (source) and target = CIFAR-10 test + noise (severity). Uses torchvision.
    Returns: X_train, Y_train (source), X_cal, Y_cal (source test split),
    X_test, Y_test (target), X_unlabeled (target, no labels used for ratio).
    For speed we use features (e.g. from a pretrained model or simple CNN);
    here we return raw images and let caller extract features if needed.
    """
    try:
        import torch
        from torchvision import datasets, transforms
    except ImportError:
        raise ImportError("torch and torchvision required for CIFAR-10-C. pip install torch torchvision")
    rng = np.random.default_rng(seed)
    # Source: CIFAR-10 train and test (raw .data is numpy)
    train_ds = datasets.CIFAR10(root=data_dir or "./data", train=True, download=True)
    test_ds = datasets.CIFAR10(root=data_dir or "./data", train=False, download=True)
    X_train = np.array(train_ds.data)
    Y_train = np.array(train_ds.targets)
    X_train_flat = X_train.reshape(X_train.shape[0], -1).astype(np.float32) / 255.0
    X_test_src = np.array(test_ds.data)
    Y_test_src = np.array(test_ds.targets)
    X_test_src_flat = X_test_src.reshape(X_test_src.shape[0], -1).astype(np.float32) / 255.0
    # Subsample cal from source test, then split fit/cal
    n_src = len(X_test_src_flat)
    n_cal_use = min(n_cal, n_src)
    idx_cal = rng.choice(n_src, size=n_cal_use, replace=False)
    X_cal_all = X_test_src_flat[idx_cal]
    Y_cal_all = Y_test_src[idx_cal]
    n_fit = n_cal_use // 2
    X_fit = X_cal_all[:n_fit]
    Y_fit = Y_cal_all[:n_fit]
    X_cal = X_cal_all[n_fit:]
    Y_cal = Y_cal_all[n_fit:]
    # Target: CIFAR-10 test + noise as proxy for corruption (severity = noise scale)
    # Full CIFAR-10-C can be added via external dataset.
    noise_scale = 0.05 * severity
    X_tgt = X_test_src.astype(np.float32) + rng.standard_normal(X_test_src.shape).astype(np.float32) * noise_scale
    Y_tgt = Y_test_src
    X_tgt_flat = X_tgt.reshape(X_tgt.shape[0], -1).astype(np.float32) / 255.0
    idx_tst = rng.choice(len(X_tgt_flat), size=min(n_test, len(X_tgt_flat)), replace=False)
    idx_unl = rng.choice(len(X_tgt_flat), size=min(m_unlabeled, len(X_tgt_flat)), replace=False)
    X_test = X_tgt_flat[idx_tst]
    Y_test = Y_tgt[idx_tst]
    X_unlabeled = X_tgt_flat[idx_unl]
    return {
        "X_train": X_train_flat,
        "Y_train": Y_train,
        "X_fit": X_fit,
        "Y_fit": Y_fit,
        "X_cal": X_cal,
        "Y_cal": Y_cal,
        "X_test": X_test,
        "Y_test": Y_test,
        "X_unlabeled": X_unlabeled,
        "oracle_ratio_cal": None,
        "oracle_ratio_test": None,
    }


def load_adult_shift(
    seed: int,
    n_cal: int,
    n_test: int,
    m_unlabeled: int,
    shift_strength: float = 1.0,
) -> Dict[str, np.ndarray]:
    """Tabular: Adult (source) vs reweighted/sliced target. Uses sklearn fetch_openml."""
    from sklearn.datasets import fetch_openml
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(seed)
    try:
        data = fetch_openml("adult", version=2, as_frame=False, parser="auto")
    except TypeError:
        data = fetch_openml("adult", version=2, as_frame=False)
    X_all = data.data
    y_all = (data.target == ">50K").astype(np.int64)
    if hasattr(X_all, "to_numpy"):
        X_all = X_all.to_numpy()
    # Select numeric columns if mixed types (Adult has categoricals)
    if X_all.dtype == object or not np.issubdtype(X_all.dtype, np.number):
        try:
            import pandas as pd
            df = pd.DataFrame(X_all)
            X_all = df.select_dtypes(include=[np.number]).to_numpy()
        except Exception:
            X_all = np.asarray(X_all, dtype=np.float64, copy=False)
    X_all = np.asarray(X_all, dtype=np.float64)
    # Simple slice: e.g. first half of features scaled differently as "target"
    n_total = len(X_all)
    idx = rng.permutation(n_total)
    train_end = int(0.6 * n_total)
    cal_end = train_end + n_cal
    X_train = X_all[idx[:train_end]]
    Y_train = y_all[idx[:train_end]]
    X_cal = X_all[idx[train_end:cal_end]]
    Y_cal = y_all[idx[train_end:cal_end]]
    # "Target": shift by adding noise and scaling
    target_idx = idx[cal_end:cal_end + n_test + m_unlabeled]
    X_tgt = X_all[target_idx].astype(np.float64) + rng.standard_normal((len(target_idx), X_all.shape[1])) * shift_strength
    Y_tgt = y_all[target_idx]
    X_test = X_tgt[:n_test]
    Y_test = Y_tgt[:n_test]
    X_unlabeled = X_tgt[n_test : n_test + m_unlabeled]
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_cal = scaler.transform(X_cal)
    X_test = scaler.transform(X_test)
    X_unlabeled = scaler.transform(X_unlabeled)
    n_fit = len(X_cal) // 2
    return {
        "X_train": X_train,
        "Y_train": Y_train,
        "X_fit": X_cal[:n_fit],
        "Y_fit": Y_cal[:n_fit],
        "X_cal": X_cal[n_fit:],
        "Y_cal": Y_cal[n_fit:],
        "X_test": X_test,
        "Y_test": Y_test,
        "X_unlabeled": X_unlabeled,
        "oracle_ratio_cal": None,
        "oracle_ratio_test": None,
    }
