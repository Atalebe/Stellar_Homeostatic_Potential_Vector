#!/usr/bin/env python3
"""
43d2_h3_permutation_null_residualized.py

Purpose
-------
A strict permutation null for H3 that removes Teff-bin structure ("bin geometry").

Old issue:
- Shuffling ages within Teff bins preserves bin-level trends.
- Global Kendall tau(age, Phi) can remain invariant if the signal is between bins.

Fix:
- Residualize age and Phi within bins (subtract bin medians).
- Shuffle age residuals within each bin.
- Compute Kendall tau on residuals and empirical p-values.

Outputs
-------
/mnt/g/STAR_HPV/results/ripeness/h3_gyro_age_permutation_null_residualized.yaml

Notes
-----
- Uses Kendall's tau on residualized variables:
  tau(age_resid, phi_resid) and tau(age_resid, ripeness_resid)
- Defines ripeness_resid = phi_resid * age_resid (a pure within-bin coupling measure).
"""

from __future__ import annotations

import os
import math
import yaml
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

INP = "/mnt/g/STAR_HPV/processed/gyro/kepler_gmmgate_with_gyro_ages_mamajek.parquet"
OUT = "/mnt/g/STAR_HPV/results/ripeness/h3_gyro_age_permutation_null_residualized.yaml"

AGE_COL = "age_gyr_gyro"
PHI_COL = "Phi_gmm"
BIN_COL = "teff_bin_s"

N_SHUFFLES = 500
SEED = 123
MIN_PER_BIN = 20  # bins smaller than this are excluded from permutation pool


def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def _pick_bin_col(df: pd.DataFrame) -> str:
    if BIN_COL in df.columns:
        return BIN_COL
    # fallback
    for c in ["teff_bin_s", "teff_bin", "teff_bin_s_kep", "teff_bin_s_tail"]:
        if c in df.columns:
            return c
    raise RuntimeError(f"Could not find a Teff bin column. Columns: {list(df.columns)[:60]}")


def _residualize_by_bin(df: pd.DataFrame, col: str, bin_col: str) -> pd.Series:
    med = df.groupby(bin_col)[col].transform("median")
    return df[col] - med


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


def main() -> None:
    _ensure_dir(OUT)
    rng = np.random.default_rng(SEED)

    df = pd.read_parquet(INP)

    bin_col = _pick_bin_col(df)
    required = [AGE_COL, PHI_COL, bin_col]
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise RuntimeError(f"Missing required columns: {miss}. Available: {list(df.columns)[:60]}")

    # Drop rows without ages or phi or bin
    d = df[[AGE_COL, PHI_COL, bin_col]].copy()
    d[AGE_COL] = pd.to_numeric(d[AGE_COL], errors="coerce")
    d[PHI_COL] = pd.to_numeric(d[PHI_COL], errors="coerce")
    d = d.dropna(subset=[AGE_COL, PHI_COL, bin_col]).reset_index(drop=True)

    # Exclude bins too small for meaningful within-bin permutation
    bin_sizes = d.groupby(bin_col).size()
    keep_bins = bin_sizes[bin_sizes >= MIN_PER_BIN].index.astype(str).tolist()
    d = d[d[bin_col].astype(str).isin(keep_bins)].reset_index(drop=True)

    if len(d) < 100:
        raise RuntimeError(f"Too few rows after filtering. rows={len(d)} keep_bins={keep_bins}")

    # Residualize
    d["age_resid"] = _residualize_by_bin(d, AGE_COL, bin_col)
    d["phi_resid"] = _residualize_by_bin(d, PHI_COL, bin_col)
    d["ripeness_resid"] = d["phi_resid"] * d["age_resid"]

    # Observed taus on residuals
    tau_age_phi, p_age_phi = kendalltau(d["age_resid"], d["phi_resid"])
    tau_age_rip, p_age_rip = kendalltau(d["age_resid"], d["ripeness_resid"])

    # Null distribution: shuffle age residuals within bin
    taus_phi = np.empty(N_SHUFFLES, dtype=float)
    taus_rip = np.empty(N_SHUFFLES, dtype=float)

    # Pre-split indices per bin for speed
    idx_by_bin = {b: np.flatnonzero(d[bin_col].astype(str).values == str(b)) for b in keep_bins}

    age_resid = d["age_resid"].to_numpy(dtype=float)
    phi_resid = d["phi_resid"].to_numpy(dtype=float)

    for i in range(N_SHUFFLES):
        age_perm = age_resid.copy()
        for b, idx in idx_by_bin.items():
            # shuffle residuals within bin
            age_perm[idx] = rng.permutation(age_perm[idx])

        rip_perm = phi_resid * age_perm

        taus_phi[i] = kendalltau(age_perm, phi_resid).statistic
        taus_rip[i] = kendalltau(age_perm, rip_perm).statistic

    # Empirical p-values (one-sided: null >= observed)
    # +1 smoothing
    p_emp_phi = (np.sum(taus_phi >= tau_age_phi) + 1.0) / (N_SHUFFLES + 1.0)
    p_emp_rip = (np.sum(taus_rip >= tau_age_rip) + 1.0) / (N_SHUFFLES + 1.0)

    out = {
        "input": INP,
        "age_col": AGE_COL,
        "phi_col": PHI_COL,
        "bin_col": bin_col,
        "n_rows_used": int(len(d)),
        "min_per_bin": int(MIN_PER_BIN),
        "bins_used": {"n_bins": int(len(keep_bins)), "bins": keep_bins},
        "observed": {
            "kendall_tau_age_phi_resid": float(tau_age_phi),
            "kendall_p_age_phi_resid": float(p_age_phi) if p_age_phi is not None else None,
            "kendall_tau_age_ripeness_resid": float(tau_age_rip),
            "kendall_p_age_ripeness_resid": float(p_age_rip) if p_age_rip is not None else None,
        },
        "null": {
            "n_shuffles": int(N_SHUFFLES),
            "seed": int(SEED),
            "tau_age_phi_resid": _summ(taus_phi),
            "tau_age_ripeness_resid": _summ(taus_rip),
            "p_empirical_one_sided_ge": {
                "tau_age_phi_resid": float(p_emp_phi),
                "tau_age_ripeness_resid": float(p_emp_rip),
            },
        },
        "notes": [
            "This null removes Teff-bin geometry by residualizing age and Phi within bins (subtract bin medians).",
            "Age residuals are permuted within bins; Phi residuals are held fixed; ripeness_resid = phi_resid * age_perm.",
            "Empirical p-values report P(null tau >= observed tau), one-sided, with +1 smoothing.",
        ],
    }

    with open(OUT, "w") as f:
        yaml.safe_dump(out, f, sort_keys=False)

    print(f"saved: {OUT}")
    print(f"observed tau(age_resid,phi_resid) = {tau_age_phi} ; emp p = {p_emp_phi}")
    print(f"observed tau(age_resid,ripeness_resid) = {tau_age_rip} ; emp p = {p_emp_rip}")


if __name__ == "__main__":
    main()
