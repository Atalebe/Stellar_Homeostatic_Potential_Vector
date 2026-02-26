#!/usr/bin/env python3
"""
43d3_h3_window_age_permutation_null.py

Purpose
-------
Permutation null for the core H3 window statement:
"Are window stars older than non-window stars?"

Approach
--------
- Work within Teff bins to preserve bin composition.
- Shuffle ages within each bin, keeping window membership fixed.
- Compute DeltaMedian = median(age|window) - median(age|non-window).
- Empirical p-value: P(null DeltaMedian >= observed DeltaMedian).

Outputs
-------
/mnt/g/STAR_HPV/results/ripeness/h3_gyro_window_age_permutation_null.yaml
"""

from __future__ import annotations

import os
import yaml
import numpy as np
import pandas as pd

INP = "/mnt/g/STAR_HPV/processed/gyro/kepler_gmmgate_with_gyro_ages_mamajek.parquet"
OUT = "/mnt/g/STAR_HPV/results/ripeness/h3_gyro_window_age_permutation_null.yaml"

AGE_COL = "age_gyr_gyro"
BIN_COL = "teff_bin_s"
WIN_COL = "in_window_gmm"

N_SHUFFLES = 1000
SEED = 123
MIN_PER_GROUP = 30  # min window and non-window per bin to include


def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def _pick_bin_col(df: pd.DataFrame) -> str:
    if BIN_COL in df.columns:
        return BIN_COL
    for c in ["teff_bin_s", "teff_bin", "teff_bin_s_kep", "teff_bin_s_tail"]:
        if c in df.columns:
            return c
    raise RuntimeError(f"Could not find a Teff bin column. Columns: {list(df.columns)[:60]}")


def _summ(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0}
    return {
        "n": int(x.size),
        "min": float(np.min(x)),
        "p50": float(np.median(x)),
        "p95": float(np.percentile(x, 95)),
        "max": float(np.max(x)),
    }


def _delta_median(age: np.ndarray, win: np.ndarray) -> float:
    a = age[win]
    b = age[~win]
    if a.size == 0 or b.size == 0:
        return np.nan
    return float(np.median(a) - np.median(b))


def main() -> None:
    _ensure_dir(OUT)
    rng = np.random.default_rng(SEED)

    df = pd.read_parquet(INP)
    bin_col = _pick_bin_col(df)

    required = [AGE_COL, WIN_COL, bin_col]
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise RuntimeError(f"Missing required columns: {miss}. Available: {list(df.columns)[:60]}")

    d = df[[AGE_COL, WIN_COL, bin_col]].copy()
    d[AGE_COL] = pd.to_numeric(d[AGE_COL], errors="coerce")
    d = d.dropna(subset=[AGE_COL, WIN_COL, bin_col]).reset_index(drop=True)

    # enforce boolean
    d[WIN_COL] = d[WIN_COL].astype(bool)
    d[bin_col] = d[bin_col].astype(str)

    # Keep only bins with enough members in BOTH groups
    keep_bins = []
    by = d.groupby(bin_col)
    for b, g in by:
        n_win = int(g[WIN_COL].sum())
        n_out = int((~g[WIN_COL]).sum())
        if n_win >= MIN_PER_GROUP and n_out >= MIN_PER_GROUP:
            keep_bins.append(b)

    d = d[d[bin_col].isin(keep_bins)].reset_index(drop=True)
    if len(d) < 100:
        raise RuntimeError(f"Too few rows after bin filtering. rows={len(d)} keep_bins={keep_bins}")

    age = d[AGE_COL].to_numpy(dtype=float)
    win = d[WIN_COL].to_numpy(dtype=bool)
    bins = d[bin_col].to_numpy(dtype=str)

    # Observed per-bin deltas and pooled delta
    obs_pooled = _delta_median(age, win)

    obs_by_bin = {}
    for b in keep_bins:
        mask = bins == b
        obs_by_bin[b] = _delta_median(age[mask], win[mask])

    # Null: permute ages within bin
    null_pooled = np.empty(N_SHUFFLES, dtype=float)
    null_by_bin = {b: np.empty(N_SHUFFLES, dtype=float) for b in keep_bins}

    idx_by_bin = {b: np.flatnonzero(bins == b) for b in keep_bins}

    for i in range(N_SHUFFLES):
        age_perm = age.copy()
        for b, idx in idx_by_bin.items():
            age_perm[idx] = rng.permutation(age_perm[idx])

        null_pooled[i] = _delta_median(age_perm, win)

        for b, idx in idx_by_bin.items():
            null_by_bin[b][i] = _delta_median(age_perm[idx], win[idx])

    # Empirical p-values (one-sided ge)
    p_emp_pooled = (np.sum(null_pooled >= obs_pooled) + 1.0) / (N_SHUFFLES + 1.0)

    p_emp_by_bin = {}
    for b in keep_bins:
        arr = null_by_bin[b]
        obs = obs_by_bin[b]
        p_emp_by_bin[b] = float((np.sum(arr >= obs) + 1.0) / (N_SHUFFLES + 1.0))

    out = {
        "input": INP,
        "age_col": AGE_COL,
        "window_col": WIN_COL,
        "bin_col": bin_col,
        "n_rows_used": int(len(d)),
        "min_per_group": int(MIN_PER_GROUP),
        "bins_used": {"n_bins": int(len(keep_bins)), "bins": keep_bins},
        "observed": {
            "delta_median_pooled": float(obs_pooled),
            "delta_median_by_bin": {k: float(v) for k, v in obs_by_bin.items()},
        },
        "null": {
            "n_shuffles": int(N_SHUFFLES),
            "seed": int(SEED),
            "delta_median_pooled": _summ(null_pooled),
            "p_empirical_one_sided_ge": {
                "pooled": float(p_emp_pooled),
                "by_bin": p_emp_by_bin,
            },
        },
        "notes": [
            "Ages are permuted within Teff bins; window membership is held fixed.",
            "The statistic is DeltaMedian = median(age|window) - median(age|non-window).",
            "Empirical p-values report P(null DeltaMedian >= observed DeltaMedian), one-sided, with +1 smoothing.",
        ],
    }

    with open(OUT, "w") as f:
        yaml.safe_dump(out, f, sort_keys=False)

    print(f"saved: {OUT}")
    print(f"observed pooled delta_median = {obs_pooled:.6f} ; emp p = {p_emp_pooled:.6g}")
    for b in keep_bins:
        print(f"{b}: delta={obs_by_bin[b]:.6f} ; emp p={p_emp_by_bin[b]:.6g}")


if __name__ == "__main__":
    main()
