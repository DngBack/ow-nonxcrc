#!/usr/bin/env bash
# Sweep Exp3: real domain shift. Run from repo root.
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
python experiments/run_exp3_real.py --config configs/exp3_real_domainshift.yaml --output_dir outputs/exp3
