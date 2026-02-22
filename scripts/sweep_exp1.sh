#!/usr/bin/env bash
# Sweep Exp1: synthetic covariate shift. Run from repo root.
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
python experiments/run_exp1_synth.py --config configs/exp1_synth_covshift.yaml --output_dir outputs/exp1
