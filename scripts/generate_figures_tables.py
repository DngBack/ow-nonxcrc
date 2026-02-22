#!/usr/bin/env python3
"""Generate all figures (F1–F6) and print tables from experiment CSVs."""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ow_nonxcrc.eval.plots import generate_all_plots
from ow_nonxcrc.eval.tables import print_tables


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp1", type=str, default=str(ROOT / "outputs" / "exp1" / "results.csv"))
    parser.add_argument("--exp2", type=str, default=str(ROOT / "outputs" / "exp2" / "results.csv"))
    parser.add_argument("--exp3", type=str, default=str(ROOT / "outputs" / "exp3" / "results.csv"))
    parser.add_argument("--r_target", type=float, default=0.1)
    args = parser.parse_args()
    generate_all_plots(
        exp1_csv=args.exp1,
        exp2_csv=args.exp2,
        exp3_csv=args.exp3,
        r_target=args.r_target,
    )
    print_tables(exp1_csv=args.exp1, exp3_csv=args.exp3, r_target=args.r_target)
    return 0


if __name__ == "__main__":
    sys.exit(main())
