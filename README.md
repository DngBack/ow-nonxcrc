# ow-nonxcrc

OW-NonXCRC: minimax weighted non-exchangeable conformal risk control (OW-NonXCRC).  
Relevance weighting under covariate shift with clipped density-ratio weights and Non-X CRC bounds.

## Setup

```bash
pip install -e .
```

Requirements: Python ≥3.9, numpy, scipy, scikit-learn, torch, torchvision, hydra-core, pandas, matplotlib, seaborn, tqdm.

## Reproduce experiments

### Exp1 — Synthetic covariate shift

```bash
# Full sweep (severity × n_cal × tau × seeds × methods)
./scripts/sweep_exp1.sh
# or
python experiments/run_exp1_synth.py --config configs/exp1_synth_covshift.yaml --output_dir outputs/exp1

# Quick validation (few seeds, severities, taus)
python experiments/run_exp1_synth.py --config configs/exp1_synth_covshift.yaml --output_dir outputs/exp1 --quick
```

Results: `outputs/exp1/results.csv`.

### Exp2 — Streaming drift

```bash
python experiments/run_exp2_stream.py --config configs/exp2_stream_drift.yaml --output_dir outputs/exp2
```

Results: `outputs/exp2/results.csv`.

### Exp3 — Real domain shift

```bash
./scripts/sweep_exp3.sh
# or
python experiments/run_exp3_real.py --config configs/exp3_real_domainshift.yaml --output_dir outputs/exp3
# Tabular (no torchvision): --dataset adult
python experiments/run_exp3_real.py --dataset adult --output_dir outputs/exp3
```

Results: `outputs/exp3/results.csv`.

## Figures and tables

From the repo root, after running experiments:

```python
from ow_nonxcrc.eval.plots import generate_all_plots
from ow_nonxcrc.eval.tables import print_tables

generate_all_plots(
    exp1_csv="outputs/exp1/results.csv",
    exp2_csv="outputs/exp2/results.csv",
    exp3_csv="outputs/exp3/results.csv",
    r_target=0.1,
)
print_tables(exp1_csv="outputs/exp1/results.csv", exp3_csv="outputs/exp3/results.csv", r_target=0.1)
```

Or run a small script:

```bash
python -c "
from ow_nonxcrc.eval.plots import generate_all_plots
from ow_nonxcrc.eval.tables import print_tables
generate_all_plots(r_target=0.1)
print_tables(r_target=0.1)
"
```

## Output format

Each run writes one row (CSV) with:

- `seed`, `n`, `m`, `severity`, `target_r`, `delta`, `method`, `tau`, `neff`, `lambda_hat`
- `achieved_risk_test`, `slack`, `utility`, `violation`
- `runtime_fit_ratio`, `runtime_crc`
- (Exp2: `time_slice`, `changepoint`; Exp3: `corruption_level`, `dataset`)

## Methods

- **uniform**: \(w_i = 1\)
- **oracle**: \(w_i = r(X_i)\) (synthetic only)
- **oracle+clip**: \(w_i = \min(r(X_i), \tau)\)
- **learned**: \(w_i = \hat{r}(X_i)\) from domain classifier on D_fit vs U (split)
- **learned+clip**: \(w_i = \min(\hat{r}(X_i), \tau)\) (main)
- **naive-learned-no-split**: \(\hat{r}\) learned on full cal vs U, then calibrate on same cal (double-dip baseline)

## Config

- `configs/exp1_synth_covshift.yaml`: severity_list, n_cal_list, tau_list, seeds, r_target, delta, ratio_model, split_frac, lambda_grid_size, loss_c, methods.
- `configs/exp2_stream_drift.yaml`: T, n_window, severity_second_half, seeds, methods.
- `configs/exp3_real_domainshift.yaml`: dataset (cifar10c / adult), corruption_levels, seeds, tau_list, methods.

## License

See LICENSE.
