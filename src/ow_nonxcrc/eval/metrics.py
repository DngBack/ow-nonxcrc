"""Metrics: violation, achieved_risk, slack, neff, accept_rate; aggregation mean±std."""

from typing import Dict, List, Optional

import numpy as np


def violation(achieved_risk: float, r_target: float) -> float:
    """Excess risk over target: max(0, achieved_risk - r_target)."""
    return max(0.0, float(achieved_risk) - float(r_target))


def achieved_risk(losses: np.ndarray) -> float:
    """Mean loss on test set."""
    return float(np.mean(losses))


def aggregate_mean_std(
    values: List[float],
) -> tuple:
    """Return (mean, std)."""
    a = np.asarray(values, dtype=np.float64)
    return float(np.mean(a)), float(np.std(a)) if len(a) > 1 else 0.0


def violation_rate(achieved_risks: List[float], r_target: float) -> float:
    """Fraction of runs with achieved_risk > r_target."""
    if not achieved_risks:
        return 0.0
    return float(np.mean([r > r_target for r in achieved_risks]))
