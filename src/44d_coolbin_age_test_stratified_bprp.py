#!/usr/bin/env python3
"""
44d_coolbin_age_test_stratified_bprp.py

Within the 3200-4000K bin, test window vs non-window age differences
AFTER matching/stratifying by BP-RP quantiles.

Outputs:
- YAML: /mnt/g/STAR_HPV/results/ripeness/coolbin_bprp_stratified_age_test.yaml
- CSV:  /mnt/g/STAR_HPV/results/ripeness/coolbin_bprp_stratified_age_test.csv
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
import yaml
from scipy.stats import ks_2samp, mannwhitneyu

INP = "/mnt/g/STAR_HPV/processed/gyro/kepler_gmmgate_with_gyro_ages_mamajek.parquet"
OUT_YAML = "/mnt/g/STAR_HPV/results/ripeness/coolbin_bprp_stratified_age_test.yaml"
OUT_CSV  = "/mnt/g/STAR_HPV/results/ripeness/coolbin_bprp_stratified_age_test.csv"

BIN = "3200-4000"
AGE = "age_gyr_gyro"
WIN = "in_window_gmm"
BPRP = "bp_rp"

def ensure_dir(p: str) -> None:
    os.makedirs(os.path.dirname(p), exist_ok=True)

def safe_arr(s: pd.Series) -> np.ndarray:
    x = pd.to_numeric(s, errors="coerce").astype(float).to_numpy()
    return x[np.isfinite(x)]

def main() -> None:
    ensure_dir(OUT_YAML)
    ensure_dir(OUT_CSV)

    df = pd.read_parquet(INP)
    for c in ["teff_bin_s", AGE, WIN, BPRP]:
        if c not in df.columns:
            raise RuntimeError(f"Missing {c}. Have: {list(df.columns)[:50]}")

    cool = df[df["teff_bin_s"] == BIN].copy()
    cool[WIN] = cool[WIN].astype(bool)

    # drop missing age or bp-rp
    cool = cool[np.isfinite(pd.to_numeric(cool[AGE], errors="coerce"))].copy()
    cool = cool[np.isfinite(pd.to_numeric(cool[BPRP], errors="coerce"))].copy()

    # Build BP-RP strata (deciles by default)
    bprp = pd.to_numeric(cool[BPRP], errors="coerce").astype(float)
    qs = np.quantile(bprp, np.linspace(0, 1, 11))
    # make unique edges
    qs = np.unique(qs)

    rows = []
    pooled_win = []
    pooled_non = []

    for i in range(len(qs) - 1):
        lo, hi = qs[i], qs[i+1]
        mask = (bprp >= lo) & (bprp <= hi if i == len(qs)-2 else bprp < hi)
        sub = cool[mask].copy()
        if len(sub) < 30:
            continue

        w = safe_arr(sub.loc[sub[WIN] == True, AGE])
        n = safe_arr(sub.loc[sub[WIN] == False, AGE])
        if len(w) < 10 or len(n) < 10:
            continue

        pooled_win.append(w)
        pooled_non.append(n)

        ks = ks_2samp(w, n)
        mw = mannwhitneyu(w, n, alternative="greater")

        rows.append({
            "bprp_lo": float(lo),
            "bprp_hi": float(hi),
            "n_window": int(len(w)),
            "n_nonwindow": int(len(n)),
            "med_window": float(np.median(w)),
            "med_nonwindow": float(np.median(n)),
            "delta_med": float(np.median(w) - np.median(n)),
            "ks_p": float(ks.pvalue),
            "mw_p_window_older": float(mw.pvalue),
        })

    # pooled across strata (concatenate)
    if pooled_win and pooled_non:
        W = np.concatenate(pooled_win)
        N = np.concatenate(pooled_non)
        ks = ks_2samp(W, N)
        mw = mannwhitneyu(W, N, alternative="greater")
        pooled = {
            "n_window": int(len(W)),
            "n_nonwindow": int(len(N)),
            "med_window": float(np.median(W)),
            "med_nonwindow": float(np.median(N)),
            "delta_med": float(np.median(W) - np.median(N)),
            "ks_p": float(ks.pvalue),
            "mw_p_window_older": float(mw.pvalue),
        }
    else:
        pooled = {"note": "No strata had sufficient sample sizes."}

    out = {
        "input": INP,
        "bin": BIN,
        "age_col": AGE,
        "window_col": WIN,
        "bprp_col": BPRP,
        "n_rows_used": int(len(cool)),
        "n_strata_used": int(len(rows)),
        "pooled_stratified_test": pooled,
    }

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    with open(OUT_YAML, "w") as f:
        yaml.safe_dump(out, f, sort_keys=False)

    print("saved:", OUT_YAML)
    print("saved:", OUT_CSV)
    print("pooled:", pooled)

if __name__ == "__main__":
    main()
