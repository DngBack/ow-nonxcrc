"""Logistic regression classifier (sklearn)."""

import numpy as np
from sklearn.linear_model import LogisticRegression

from .base import BaseClassifier


class LogisticClassifier(BaseClassifier):
    def __init__(self, max_iter: int = 1000, C: float = 1.0, random_state: int = 0):
        self.max_iter = max_iter
        self.C = C
        self.random_state = random_state
        self._clf: LogisticRegression = LogisticRegression(
            max_iter=max_iter,
            C=C,
            random_state=random_state,
            solver="lbfgs",
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticClassifier":
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self._clf.fit(X, np.asarray(y).ravel())
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return self._clf.predict_proba(X)
