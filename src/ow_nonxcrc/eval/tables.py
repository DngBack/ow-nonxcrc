"""Tables T1 (Exp1 worst-case / high severity), T2 (Exp3 over corruption levels)."""

from typing import Optional

import pandas as pd


def table1_exp1_high_severity(
    df: pd.DataFrame,
    severity_high: Optional[float] = None,
    n_cal: Optional[int] = None,
    r_target: float = 0.1,
) -> pd.DataFrame:
    """T1: Mean ± std violation and utility at high severity, n_cal=1000."""
    severity_high = severity_high if severity_high is not None else df["severity"].max()
    n_cal = n_cal if n_cal is not None else df["n_cal"].max()
    sub = df[(df["severity"] == severity_high) & (df["n_cal"] == n_cal)]
    if sub.empty:
        sub = df[(df["severity"] == df["severity"].max()) & (df["n_cal"] == df["n_cal"].max())]
    if sub.empty:
        return pd.DataFrame()
    t = sub.groupby("method").agg(
        violation_mean=("violation", "mean"),
        violation_std=("violation", "std"),
        utility_mean=("utility", "mean"),
        utility_std=("utility", "std"),
    ).reset_index()
    t["violation"] = t["violation_mean"].round(4).astype(str) + " ± " + t["violation_std"].fillna(0).round(4).astype(str)
    t["utility"] = t["utility_mean"].round(4).astype(str) + " ± " + t["utility_std"].fillna(0).round(4).astype(str)
    return t[["method", "violation", "utility"]]


def table2_exp1_worst_case(
    df: pd.DataFrame,
    r_target: float = 0.1,
) -> pd.DataFrame:
    """T2: Worst-case over severity: max violation, max slack per method."""
    t = df.groupby("method").agg(
        max_violation=("violation", "max"),
        max_slack=("slack", "max"),
        mean_violation=("violation", "mean"),
    ).reset_index()
    return t


def table2_exp3_corruption(
    df: pd.DataFrame,
    r_target: float = 0.1,
) -> pd.DataFrame:
    """T2 Exp3: Violation rate (% runs > r) and mean utility over corruption levels."""
    df = df.copy()
    df["violated"] = df["achieved_risk_test"] > r_target
    t = df.groupby("method").agg(
        violation_rate=("violated", "mean"),
        utility_mean=("utility", "mean"),
        utility_std=("utility", "std"),
    ).reset_index()
    t["violation_rate"] = (t["violation_rate"] * 100).round(2)
    t["utility"] = t["utility_mean"].round(4).astype(str) + " ± " + t["utility_std"].fillna(0).round(4).astype(str)
    return t[["method", "violation_rate", "utility"]]


def print_tables(
    exp1_csv: str = "outputs/exp1/results.csv",
    exp3_csv: str = "outputs/exp3/results.csv",
    r_target: float = 0.1,
) -> None:
    """Print T1 and T2 to stdout."""
    import os
    if os.path.isfile(exp1_csv):
        df1 = pd.read_csv(exp1_csv)
        print("--- Table 1 (Exp1 high severity, n_cal=1000) ---")
        print(table1_exp1_high_severity(df1, r_target=r_target).to_string(index=False))
        print("\n--- Table 2 (Exp1 worst-case over severity) ---")
        print(table2_exp1_worst_case(df1, r_target=r_target).to_string(index=False))
    if os.path.isfile(exp3_csv):
        df3 = pd.read_csv(exp3_csv)
        print("\n--- Table 2 Exp3 (violation rate, utility) ---")
        print(table2_exp3_corruption(df3, r_target=r_target).to_string(index=False))
