"""Base interface for classifier (predictor or domain classifier)."""

from abc import ABC, abstractmethod
from typing import Union

import numpy as np


class BaseClassifier(ABC):
    """Predict p(y|x) or p(domain|x)."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseClassifier":
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Shape (n_samples, n_classes)."""
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        P = self.predict_proba(X)
        return np.argmax(P, axis=1)
