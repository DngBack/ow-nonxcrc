"""Reproducibility: set global seeds for numpy, sklearn, torch."""

import os
import random
from typing import Optional

import numpy as np


def set_seed(seed: int) -> None:
    """Set seed for numpy, random, and optionally torch if available."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    try:
        import sklearn

        # sklearn uses numpy RNG; no separate global seed API in recent versions
        pass
    except ImportError:
        pass
