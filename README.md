# SOC Estimation using a Physics-Informed TCN (Assessment 10)

## Overview
State of Charge (SOC) estimation for the BMW i3 electric vehicle using a physics-informed
Temporal Convolutional Network (TCN), based on the preprocessing pipeline from:

> M. H. Sulaiman and Z. Mustaffa, "State of charge estimation for electric vehicles
> using random forest", *Green Energy and Intelligent Transportation*, 3 (2024) 100177.

## Dataset
- **Source:** "Battery and Heating Data in Real Driving Cycles" (BMW i3, 60 Ah pack),
  IEEE DataPort, doi:10.21227/6jr9-5235
- **Files:** `BMW_i3_dataset/Final_60trips_Training_BMW_Data.csv` (trips 1-60) and
  `BMW_i3_dataset/Final_10trips_Testing_BMW_Data.csv` (trips 61-70)
- **Total instances:** 1,063,999 after cleaning (training: 945,025 / testing: 118,974), 10 Hz
- **Inputs:** Battery voltage (V), battery current (A), battery temperature (C), ambient temperature (C)
- **Target:** Manufacturer-estimated SOC (%)
- **Cleaning:** NaN / out-of-range rows removed, plus rows where SOC jumps by more than 5 % between
  consecutive samples (one logging glitch at the end of trip 46)

## Final Metrics (test trips #61-#70, 118,974 instances)

| Metric | Physics-informed TCN (proposed) | Physics fusion only (no TCN) | RF, 25 trees (re-impl.) | RF, 25 trees (paper) |
|---|---|---|---|---|
| RMSE (%) | **3.5815** | 3.4867 | 4.9021 | 5.9028 |
| MAE (%) | 2.6937 | 2.3973 | 3.3739 | 4.4321 |
| MAX (%) | 11.9606 | 11.2787 | 20.8800 | 24.2175 |
| STD. DEV. (%) | 3.5748 | 3.2555 | 4.8320 | 5.8999 |
| Total time (s) | 673.85 | 15.19 | 78.73 | 165.28 |

Times are from the executed notebook run on a laptop CPU (Intel i5-1235U, no GPU); they vary between runs.

Per-trip TCN RMSE is in `outputs/per_trip_rmse.csv`. The hardest trips are #61 (very low SOC,
31 % → 15 %) and #70 (battery at ~2 °C, colder than almost all training trips).

## Method
Every input to the model is computed causally (past and present samples only) from the four measured signals.

1. **Coulomb counting:** ΔSOC since the start of the trip = k · ∫ I dt. k is fitted on the training
   trips (effective capacity 61.9 Ah). Given the true starting SOC this tracks the test trips to ~0.05 %,
   so the remaining task is estimating each trip's starting SOC.
2. **Stage 1 – voltage model:** polynomial ridge regression of SOC on voltage, battery temperature and
   current history (10 / 60 / 300 s means): OCV curve + temperature shift + IR / polarisation drop.
3. **Stage 2 – fusion:** the offset (stage 1 − ΔSOC) is averaged within each trip with causal exponential
   averages (half-lives 30 s–50 min, optionally weighting low-current samples) and added back to ΔSOC;
   a second ridge model blends these estimates. On training trips all features are **out-of-fold**
   (6-fold trip-wise CV), so later stages see realistic errors.
4. **TCN:** a causal TCN looks at the last 120 samples of the four signals and the physics estimates, and
   adds a bounded correction (±3 % SOC) to the stage-2 estimate.

Model choices were made on cross-validation / validation trips, not on the test set.
The TCN variant with the best validation RMSE (2.63 %) was kept. On the test set its correction
is slightly worse than the fusion alone (3.58 % vs 3.49 %); both rows are reported above.

## TCN Configuration
- Architecture: 5 residual blocks, 32 channels, kernel size 3, dilations [1,2,4,8,16] (30,625 parameters)
- Input: 7 channels (V, I, T_batt, T_amb, fused estimate, voltage-model estimate, ΔSOC), window 120 samples
- Output: fused estimate + 0.03 · tanh(TCN), then 60-sample causal moving average
- Training: AdamW (lr=1e-3, weight decay 0.01), Huber loss, ReduceLROnPlateau, early stopping
  (patience 3, max 8 epochs) on the last 6 training trips, every 40th window per epoch
- SEED = 42

## How to Re-run

```bash
# From the project folder:
pip install torch pandas numpy matplotlib scikit-learn ipykernel nbconvert jupyter jupytext psutil py-cpuinfo
python soc_tcn.py
```

Or execute the notebook:
```bash
jupyter nbconvert --to notebook --execute --inplace soc_tcn.ipynb --ExecutePreprocessor.timeout=-1
```

## File List
```
SOC_Assessment10/
  BMW_i3_dataset/           # training and testing CSV files
  soc_tcn.py                # main script (VS Code "# %%" cells; notebook generated with jupytext)
  soc_tcn.ipynb             # executed notebook
  outputs/
    figures/                # all graphs as PNG (300 dpi)
    results_table.csv       # final metrics table
    per_trip_rmse.csv       # RMSE per test trip
    predictions_test.csv    # actual vs predicted SOC for all test instances
    tcn_best.pt             # trained TCN model weights
    run_log.txt             # console log of the final run
  SOC_TCN_Assessment10.pdf  # submission report (all readings, tables and figures)
  make_report.py            # builds the PDF report from outputs/ (pip install reportlab)
  README.md                 # this file
```
