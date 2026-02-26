#!/usr/bin/env python3
"""
H3 robustness check #2: permutation null.

Goal:
- Test whether the observed age->(Phi,ripeness) association could arise by chance.
- We permute ages within each Teff bin (or globally, configurable) and recompute
  ripeness_time_gyr = Phi_gmm * permuted_age.
- Then compare Kendall tau (age vs Phi, age vs ripeness) to the observed value.

Input:
  /mnt/g/STAR_HPV/processed/gyro/kepler_gmmgate_with_gyro_ages_mamajek.parquet

Output:
  /mnt/g/STAR_HPV/results/ripeness/h3_gyro_age_permutation_null.yaml
"""

from __future__ import annotations

import os
import yaml
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

INP = "/mnt/g/STAR_HPV/processed/gyro/kepler_gmmgate_with_gyro_ages_mamajek.parquet"
OUT_YAML = "/mnt/g/STAR_HPV/results/ripeness/h3_gyro_age_permutation_null.yaml"

AGE_COL = "age_gyr_gyro"
PHI_COL = "Phi_gmm"
WIN_COL = "in_window_gmm"
BIN_COL = "teff_bin_s"

RIP_COL_CANDIDATES = [
    "ripeness_time_gyr",
    "ripeness_time",
    "ripeness_gmm",
    "ripeness_rot",
    "ripeness",
]

N_SHUFFLES = 500
SEED = 123

# Shuffle ages within each teff bin (recommended).
# If False, shuffles globally (less strict).
SHUFFLE_WITHIN_BIN = True


def pick_ripeness_col(df: pd.DataFrame) -> str:
    for c in RIP_COL_CANDIDATES:
        if c in df.columns:
            return c
    raise RuntimeError(
        f"No ripeness column found. Tried {RIP_COL_CANDIDATES}. "
        f"Available columns sample: {list(df.columns)[:60]}"
    )


def main():
    os.makedirs(os.path.dirname(OUT_YAML), exist_ok=True)
    rng = np.random.default_rng(SEED)

    df = pd.read_parquet(INP)
    df["source_id"] = df["source_id"].astype(str)

    rip_col_in = pick_ripeness_col(df)

    required = [AGE_COL, PHI_COL, WIN_COL, BIN_COL, "source_id"]
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise RuntimeError(f"Missing required columns: {miss}. Available: {list(df.columns)[:40]}")

    # coerce
    df[AGE_COL] = pd.to_numeric(df[AGE_COL], errors="coerce")
    df[PHI_COL] = pd.to_numeric(df[PHI_COL], errors="coerce")
    df = df.dropna(subset=[AGE_COL, PHI_COL, BIN_COL]).copy()

    # observed statistics
    obs_tau_age_phi, obs_p_age_phi = kendalltau(df[AGE_COL], df[PHI_COL])
    # define canonical ripeness_time_gyr for this null test
    df["ripeness_time_gyr"] = df[PHI_COL] * df[AGE_COL]
    obs_tau_age_rip, obs_p_age_rip = kendalltau(df[AGE_COL], df["ripeness_time_gyr"])

    # run permutations
    taus_phi = []
    taus_rip = []

    ages = df[AGE_COL].to_numpy(dtype=float, copy=True)
    phi  = df[PHI_COL].to_numpy(dtype=float, copy=True)
    bins = df[BIN_COL].astype(str).to_numpy()

    # precompute indices per bin if needed
    if SHUFFLE_WITHIN_BIN:
        uniq_bins = sorted(pd.unique(bins).tolist())
        idx_by_bin = {b: np.where(bins == b)[0] for b in uniq_bins}
    else:
        idx_by_bin = {"ALL": np.arange(len(df))}

    for s in range(N_SHUFFLES):
        ages_perm = ages.copy()

        if SHUFFLE_WITHIN_BIN:
            for b, idx in idx_by_bin.items():
                rng.shuffle(ages_perm[idx])
        else:
            rng.shuffle(ages_perm)

        # recompute ripeness using permuted ages
        rip_perm = phi * ages_perm

        t1, _ = kendalltau(ages_perm, phi)
        t2, _ = kendalltau(ages_perm, rip_perm)

        taus_phi.append(float(t1) if t1 == t1 else np.nan)
        taus_rip.append(float(t2) if t2 == t2 else np.nan)

    taus_phi = np.array(taus_phi, dtype=float)
    taus_rip = np.array(taus_rip, dtype=float)
    taus_phi = taus_phi[np.isfinite(taus_phi)]
    taus_rip = taus_rip[np.isfinite(taus_rip)]

    # empirical p-values (one-sided: observed greater than null)
    p_emp_phi = float((np.sum(taus_phi >= obs_tau_age_phi) + 1) / (len(taus_phi) + 1))
    p_emp_rip = float((np.sum(taus_rip >= obs_tau_age_rip) + 1) / (len(taus_rip) + 1))

    out = {
        "input": INP,
        "n_rows_used": int(len(df)),
        "age_col": AGE_COL,
        "phi_col": PHI_COL,
        "window_col": WIN_COL,
        "bin_col": BIN_COL,
        "ripeness_input_col_detected": rip_col_in,
        "ripeness_null_col_used": "ripeness_time_gyr",
        "n_shuffles": int(N_SHUFFLES),
        "seed": int(SEED),
        "shuffle_within_bin": bool(SHUFFLE_WITHIN_BIN),
        "observed": {
            "kendall_tau_age_phi": float(obs_tau_age_phi),
            "kendall_p_age_phi": float(obs_p_age_phi) if obs_p_age_phi == obs_p_age_phi else None,
            "kendall_tau_age_ripeness_time": float(obs_tau_age_rip),
            "kendall_p_age_ripeness_time": float(obs_p_age_rip) if obs_p_age_rip == obs_p_age_rip else None,
        },
        "null": {
            "tau_age_phi": {
                "n": int(len(taus_phi)),
                "min": float(np.min(taus_phi)) if len(taus_phi) else None,
                "p50": float(np.quantile(taus_phi, 0.50)) if len(taus_phi) else None,
                "p95": float(np.quantile(taus_phi, 0.95)) if len(taus_phi) else None,
                "max": float(np.max(taus_phi)) if len(taus_phi) else None,
                "p_empirical_one_sided_ge": p_emp_phi,
            },
            "tau_age_ripeness_time": {
                "n": int(len(taus_rip)),
                "min": float(np.min(taus_rip)) if len(taus_rip) else None,
                "p50": float(np.quantile(taus_rip, 0.50)) if len(taus_rip) else None,
                "p95": float(np.quantile(taus_rip, 0.95)) if len(taus_rip) else None,
                "max": float(np.max(taus_rip)) if len(taus_rip) else None,
                "p_empirical_one_sided_ge": p_emp_rip,
            }
        },
        "notes": [
            "This is a permutation null. Ages are permuted (within Teff bins by default) and ripeness recomputed as Phi_gmm * age_perm.",
            "Empirical p-values report probability that null tau >= observed tau (one-sided, +1 smoothing).",
        ]
    }

    with open(OUT_YAML, "w") as f:
        yaml.safe_dump(out, f, sort_keys=False)

    print(f"saved: {OUT_YAML}")
    print("observed tau(age,phi) =", float(obs_tau_age_phi), "emp p =", p_emp_phi)
    print("observed tau(age,ripeness) =", float(obs_tau_age_rip), "emp p =", p_emp_rip)


if __name__ == "__main__":
    main()
