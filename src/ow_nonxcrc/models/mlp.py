"""MLP classifier (sklearn)."""

import numpy as np
from sklearn.neural_network import MLPClassifier as SklearnMLP

from .base import BaseClassifier


class MLPClassifier(BaseClassifier):
    def __init__(
        self,
        hidden_layer_sizes: tuple = (64, 32),
        max_iter: int = 1000,
        random_state: int = 0,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.random_state = random_state
        self._clf = SklearnMLP(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            random_state=random_state,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLPClassifier":
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
