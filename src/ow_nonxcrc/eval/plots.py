"""Figures F1–F6: risk vs severity, slack vs neff, Pareto, tau tradeoff, risk(t), risk vs corruption."""

import os
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def plot_f1_risk_vs_severity(
    df: pd.DataFrame,
    r_target: float,
    output_path: str = "outputs/exp1/fig1_risk_vs_severity.pdf",
) -> None:
    """F1: Risk vs severity; one curve per method; horizontal line at r_target."""
    df_agg = df.groupby(["severity", "method"]).agg(
        risk_mean=("achieved_risk_test", "mean"),
        risk_std=("achieved_risk_test", "std"),
    ).reset_index()
    fig, ax = plt.subplots()
    for method in df_agg["method"].unique():
        sub = df_agg[df_agg["method"] == method]
        ax.errorbar(
            sub["severity"],
            sub["risk_mean"],
            yerr=sub["risk_std"],
            label=method,
            capsize=2,
        )
    ax.axhline(r_target, color="k", linestyle="--", label=f"r={r_target}")
    ax.set_xlabel("Severity")
    ax.set_ylabel("Achieved risk (test)")
    ax.legend()
    ax.set_title("Risk vs severity (Exp1)")
    _ensure_dir(output_path)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_f2_slack_vs_neff(
    df: pd.DataFrame,
    output_path: str = "outputs/exp1/fig2_slack_vs_neff.pdf",
) -> None:
    """F2: Slack vs n_eff scatter + trendline."""
    fig, ax = plt.subplots()
    ax.scatter(df["neff"], df["slack"], alpha=0.3, s=10)
    z = np.polyfit(df["neff"], df["slack"], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df["neff"].min(), df["neff"].max(), 100)
    ax.plot(x_line, p(x_line), "r-", label="trend")
    ax.set_xlabel("n_eff")
    ax.set_ylabel("Slack")
    ax.set_title("Slack vs n_eff (Exp1)")
    ax.legend()
    _ensure_dir(output_path)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_f3_utility_vs_risk(
    df: pd.DataFrame,
    output_path: str = "outputs/exp1/fig3_utility_vs_risk.pdf",
) -> None:
    """F3: Pareto – utility vs risk (scatter by method)."""
    fig, ax = plt.subplots()
    for method in df["method"].unique():
        sub = df[df["method"] == method]
        ax.scatter(sub["achieved_risk_test"], sub["utility"], label=method, alpha=0.5, s=15)
    ax.set_xlabel("Achieved risk (test)")
    ax.set_ylabel("Utility (accept rate)")
    ax.set_title("Utility vs risk (Exp1)")
    ax.legend()
    _ensure_dir(output_path)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_f4_tau_tradeoff(
    df: pd.DataFrame,
    r_target: float,
    output_path: str = "outputs/exp1/fig4_tau_tradeoff.pdf",
) -> None:
    """F4: tau vs (violation mean, slack mean, neff mean) for clip methods."""
    clip_df = df[df["method"].isin(["oracle+clip", "learned+clip"])]
    if clip_df.empty:
        return
    agg = clip_df.groupby(["tau", "method"]).agg(
        violation=("violation", "mean"),
        slack=("slack", "mean"),
        neff=("neff", "mean"),
    ).reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, col in zip(axes, ["violation", "slack", "neff"]):
        for method in agg["method"].unique():
            sub = agg[agg["method"] == method]
            ax.plot(sub["tau"], sub[col], "o-", label=method)
        ax.set_xlabel("tau")
        ax.set_ylabel(col)
        ax.legend()
    axes[0].axhline(r_target, color="k", linestyle="--", alpha=0.5)
    fig.suptitle("Tau tradeoff (Exp1)")
    _ensure_dir(output_path)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_f4_stream_risk_over_time(
    df: pd.DataFrame,
    changepoint: int,
    output_path: str = "outputs/exp2/fig4_risk_over_time.pdf",
) -> None:
    """F4 (Exp2): risk(t) over time; mark changepoint."""
    t_agg = df.groupby("time_slice").agg(
        risk=("achieved_risk_test", "mean"),
        violation=("violation", "mean"),
    ).reset_index()
    fig, ax = plt.subplots()
    ax.plot(t_agg["time_slice"], t_agg["risk"], label="risk")
    ax.axvline(changepoint, color="r", linestyle="--", label="changepoint")
    ax.set_xlabel("Time slice")
    ax.set_ylabel("Risk")
    ax.legend()
    ax.set_title("Risk over time (Exp2)")
    _ensure_dir(output_path)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_f5_risk_vs_corruption(
    df: pd.DataFrame,
    r_target: float,
    output_path: str = "outputs/exp3/fig5_risk_vs_corruption.pdf",
) -> None:
    """F5: Risk vs corruption level (Exp3)."""
    agg = df.groupby(["corruption_level", "method"]).agg(
        risk_mean=("achieved_risk_test", "mean"),
        risk_std=("achieved_risk_test", "std"),
    ).reset_index()
    fig, ax = plt.subplots()
    for method in agg["method"].unique():
        sub = agg[agg["method"] == method]
        ax.errorbar(
            sub["corruption_level"],
            sub["risk_mean"],
            yerr=sub["risk_std"],
            label=method,
            capsize=2,
        )
    ax.axhline(r_target, color="k", linestyle="--", label=f"r={r_target}")
    ax.set_xlabel("Corruption level")
    ax.set_ylabel("Achieved risk (test)")
    ax.legend()
    ax.set_title("Risk vs corruption level (Exp3)")
    _ensure_dir(output_path)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_f6_pareto_exp3(
    df: pd.DataFrame,
    output_path: str = "outputs/exp3/fig6_pareto_utility_risk.pdf",
) -> None:
    """F6: Pareto utility vs risk (Exp3)."""
    fig, ax = plt.subplots()
    for method in df["method"].unique():
        sub = df[df["method"] == method]
        ax.scatter(sub["achieved_risk_test"], sub["utility"], label=method, alpha=0.5, s=15)
    ax.set_xlabel("Achieved risk (test)")
    ax.set_ylabel("Utility (accept rate)")
    ax.set_title("Utility vs risk (Exp3)")
    ax.legend()
    _ensure_dir(output_path)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close()


def generate_all_plots(
    exp1_csv: str = "outputs/exp1/results.csv",
    exp2_csv: str = "outputs/exp2/results.csv",
    exp3_csv: str = "outputs/exp3/results.csv",
    r_target: float = 0.1,
) -> None:
    """Generate F1–F6 from existing CSVs."""
    if os.path.isfile(exp1_csv):
        df1 = pd.read_csv(exp1_csv)
        plot_f1_risk_vs_severity(df1, r_target)
        plot_f2_slack_vs_neff(df1)
        plot_f3_utility_vs_risk(df1)
        plot_f4_tau_tradeoff(df1, r_target)
    if os.path.isfile(exp2_csv):
        df2 = pd.read_csv(exp2_csv)
        cp = df2["changepoint"].iloc[0] if "changepoint" in df2.columns else 0
        plot_f4_stream_risk_over_time(df2, cp)
    if os.path.isfile(exp3_csv):
        df3 = pd.read_csv(exp3_csv)
        plot_f5_risk_vs_corruption(df3, r_target)
        plot_f6_pareto_exp3(df3)
