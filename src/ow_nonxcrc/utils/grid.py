"""Lambda grid for CRC: monotone threshold search."""

import numpy as np


def make_lambda_grid(size: int = 200, low: float = 0.01, high: float = 0.99) -> np.ndarray:
    """Create a grid of lambda values in (0, 1) for policy threshold.
    Default: 200 points from 0.01 to 0.99 (linear or log spacing).
    """
    return np.linspace(low, high, size).astype(np.float64)
