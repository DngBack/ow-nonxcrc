"""Experiment 3: Real domain shift. CIFAR-10 + noise or Adult; methods uniform, learned, learned+clip, naive."""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ow_nonxcrc.data import load_cifar10c, load_adult_shift
from ow_nonxcrc.eval.runner import fit_predictor_synth, run_one_trial
from ow_nonxcrc.utils import set_seed, make_lambda_grid


def load_config(path: str):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "exp3_real_domainshift.yaml"))
    parser.add_argument("--output_dir", type=str, default=str(ROOT / "outputs" / "exp3"))
    parser.add_argument("--dataset", type=str, default="", help="Override: cifar10c or adult")
    args = parser.parse_args()
    cfg = load_config(args.config)
    os.makedirs(args.output_dir, exist_ok=True)
    dataset = args.dataset or cfg.get("dataset", "cifar10c")
    corruption_levels = cfg.get("corruption_levels", [1, 2, 3, 4, 5])
    seeds = cfg.get("seeds", 20)
    r_target = cfg["r_target"]
    delta = cfg["delta"]
    ratio_model = cfg.get("ratio_model", "mlp")
    lambda_grid_size = cfg.get("lambda_grid_size", 200)
    loss_c = cfg.get("loss_c", 0.5)
    tau_list = cfg.get("tau_list", [1, 2, 5, 10, 20])
    methods = cfg.get("methods", ["uniform", "learned", "learned+clip", "naive-learned-no-split"])
    n_cal = 1000
    n_test = 1000
    m_unlabeled = 1000
    lambda_grid = make_lambda_grid(size=lambda_grid_size)
    rows = []
    for seed in range(seeds):
        for severity in corruption_levels:
            set_seed(seed)
            if dataset == "adult":
                data = load_adult_shift(
                    seed=seed,
                    n_cal=n_cal,
                    n_test=n_test,
                    m_unlabeled=m_unlabeled,
                    shift_strength=0.1 * severity,
                )
            else:
                data = load_cifar10c(
                    severity=severity,
                    n_cal=n_cal,
                    n_test=n_test,
                    m_unlabeled=m_unlabeled,
                    seed=seed,
                )
            predictor = fit_predictor_synth(
                data["X_train"],
                data["Y_train"],
                model_type="logreg" if dataset == "adult" else "logreg",
            )
            for method in methods:
                taus = tau_list if method == "learned+clip" else [1.0]
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
                    row["corruption_level"] = severity
                    row["dataset"] = dataset
                    rows.append(row)
    df = pd.DataFrame(rows)
    out_path = os.path.join(args.output_dir, "results.csv")
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(rows)} rows to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
