"""Experiment 1: Synthetic covariate shift. Loop severity x n_cal x tau x seeds x methods; write CSV."""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Add project root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ow_nonxcrc.eval.runner import (
    make_dataset_synth,
    fit_predictor_synth,
    run_one_trial,
)
from ow_nonxcrc.utils import set_seed, make_lambda_grid


def load_config(path: str):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "exp1_synth_covshift.yaml"))
    parser.add_argument("--output_dir", type=str, default=str(ROOT / "outputs" / "exp1"))
    parser.add_argument("--quick", action="store_true", help="Small sweep for quick validation")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.quick:
        cfg["severity_list"] = [0, 1]
        cfg["n_cal_list"] = [200, 500]
        cfg["tau_list"] = [1, 5, 20]
        cfg["seeds"] = 3
        cfg["methods"] = ["uniform", "oracle", "learned+clip"]
    os.makedirs(args.output_dir, exist_ok=True)
    d = cfg["d"]
    n_train = cfg.get("n_train", 2000)
    severity_list = cfg["severity_list"]
    n_cal_list = cfg["n_cal_list"]
    tau_list = cfg["tau_list"]
    seeds = cfg["seeds"]
    r_target = cfg["r_target"]
    delta = cfg["delta"]
    ratio_model = cfg.get("ratio_model", "logreg")
    split_frac = cfg.get("split_frac", 0.5)
    lambda_grid_size = cfg.get("lambda_grid_size", 200)
    loss_c = cfg.get("loss_c", 0.5)
    methods = cfg["methods"]
    m_unlabeled = cfg.get("m_unlabeled", 500)
    n_test = 1000
    lambda_grid = make_lambda_grid(size=lambda_grid_size)
    rows = []
    for seed in range(seeds):
        for severity in severity_list:
            for n_cal in n_cal_list:
                set_seed(seed)
                data = make_dataset_synth(
                    seed=seed,
                    n_cal=n_cal,
                    n_test=n_test,
                    m_unlabeled=m_unlabeled,
                    d=d,
                    severity=severity,
                    n_train=n_train,
                    split_frac=split_frac,
                )
                predictor = fit_predictor_synth(
                    data["X_train"],
                    data["Y_train"],
                    model_type="logreg",
                )
                for method in methods:
                    taus = tau_list if method in ("oracle+clip", "learned+clip") else [1.0]
                    for tau in taus:
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
                        row["severity"] = severity
                        row["n_cal"] = n_cal
                        rows.append(row)
    df = pd.DataFrame(rows)
    out_path = os.path.join(args.output_dir, "results.csv")
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(rows)} rows to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
