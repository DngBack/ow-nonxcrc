"""Experiment 2: Streaming drift. Rolling window; log risk(t), violations(t), neff(t)."""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ow_nonxcrc.data import make_stream_data
from ow_nonxcrc.eval.runner import fit_predictor_synth, get_weights, run_one_trial
from ow_nonxcrc.utils import set_seed, make_lambda_grid


def load_config(path: str):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "exp2_stream_drift.yaml"))
    parser.add_argument("--output_dir", type=str, default=str(ROOT / "outputs" / "exp2"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    os.makedirs(args.output_dir, exist_ok=True)
    T = cfg["T"]
    n_window = cfg["n_window"]
    d = cfg["d"]
    severity_second = cfg.get("severity_second_half", 1.5)
    seeds = cfg.get("seeds", 20)
    r_target = cfg["r_target"]
    delta = cfg["delta"]
    ratio_model = cfg.get("ratio_model", "logreg")
    lambda_grid_size = cfg.get("lambda_grid_size", 200)
    loss_c = cfg.get("loss_c", 0.5)
    methods = cfg.get("methods", ["uniform", "learned+clip"])
    tau = 10.0
    lambda_grid = make_lambda_grid(size=lambda_grid_size)
    all_rows = []
    for seed in range(seeds):
        set_seed(seed)
        stream = make_stream_data(
            seed=seed,
            T=T,
            n_window=n_window,
            d=d,
            severity_second_half=severity_second,
        )
        X_stream = stream["X_stream"]
        Y_stream = stream["Y_stream"]
        changepoint = stream["changepoint"]
        # Use first half as "train" for predictor
        n_train = T // 2
        predictor = fit_predictor_synth(
            X_stream[:n_train],
            Y_stream[:n_train],
            model_type="logreg",
        )
        for t_start in range(n_window, T - n_window):
            # Window: [t_start - n_window, t_start)
            X_win = X_stream[t_start - n_window : t_start]
            Y_win = Y_stream[t_start - n_window : t_start]
            n_fit = n_window // 2
            X_fit = X_win[:n_fit]
            X_cal = X_win[n_fit:]
            Y_cal = Y_win[n_fit:]
            # "Unlabeled" = next window (current regime)
            n_next = min(n_window, T - t_start)
            X_unlabeled = X_stream[t_start : t_start + n_next]
            if len(X_unlabeled) < 10:
                continue
            n_test_use = min(50, n_next)
            data = {
                "X_fit": X_fit,
                "X_cal": X_cal,
                "Y_cal": Y_cal,
                "X_test": X_stream[t_start : t_start + n_test_use],
                "Y_test": Y_stream[t_start : t_start + n_test_use],
                "X_unlabeled": X_unlabeled,
                "oracle_ratio_cal": None,
            }
            for method in methods:
                row = run_one_trial(
                    seed=seed,
                    data=data,
                    predictor=predictor,
                    lambda_grid=lambda_grid,
                    r_target=r_target,
                    delta=delta,
                    method=method,
                    tau=tau,
                    loss_c=loss_c,
                    ratio_model=ratio_model,
                )
                row["time_slice"] = t_start
                row["changepoint"] = changepoint
                row["severity"] = severity_second
                all_rows.append(row)
    df = pd.DataFrame(all_rows)
    out_path = os.path.join(args.output_dir, "results.csv")
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(all_rows)} rows to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
