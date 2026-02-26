#!/usr/bin/env python3
"""
H3: Gyro-age ripeness validation.

Tests:
1) In-window stars older than out-of-window stars (global + per Teff bin).
2) Tail stars older than non-tail stars (within-window per-bin tails q90/q95).
3) Correlation tests: age vs Phi_gmm; age vs ripeness_time.

Inputs assume Mamajek ages already merged into:
  /mnt/g/STAR_HPV/processed/gyro/kepler_gmmgate_with_gyro_ages_mamajek.parquet

Outputs:
  /mnt/g/STAR_HPV/results/ripeness/h3_gyro_age_tests.yaml
  /mnt/g/STAR_HPV/results/ripeness/h3_gyro_age_tests_by_bin.csv
"""

from __future__ import annotations

import os
import yaml
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Any, Optional

from scipy.stats import ks_2samp, mannwhitneyu, kendalltau


INP = "/mnt/g/STAR_HPV/processed/gyro/kepler_gmmgate_with_gyro_ages_mamajek.parquet"

TAIL_PERBIN_Q90 = "/mnt/g/STAR_HPV/results/anchor_catalogs/kepler_gyro_ripe_tail_window_perbin_q90.parquet"
TAIL_PERBIN_Q95 = "/mnt/g/STAR_HPV/results/anchor_catalogs/kepler_gyro_ripe_tail_window_perbin_q95.parquet"

OUT_DIR = "/mnt/g/STAR_HPV/results/ripeness"
OUT_YAML = os.path.join(OUT_DIR, "h3_gyro_age_tests.yaml")
OUT_CSV  = os.path.join(OUT_DIR, "h3_gyro_age_tests_by_bin.csv")

TEFF_BINS = ["3200-4000", "4000-5200", "5200-6000", "6000-7500"]


def pick_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def series_clean(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan).dropna()
    return x


def dist_stats(x: pd.Series) -> Dict[str, float]:
    x = series_clean(x)
    if len(x) == 0:
        return {"n": 0}
    q = np.quantile(x, [0.0, 0.1, 0.5, 0.9, 1.0])
    return {
        "n": int(len(x)),
        "min": float(q[0]),
        "p10": float(q[1]),
        "p50": float(q[2]),
        "p90": float(q[3]),
        "max": float(q[4]),
    }


def two_sample_tests(a: pd.Series, b: pd.Series) -> Dict[str, Any]:
    a = series_clean(a); b = series_clean(b)
    out: Dict[str, Any] = {
        "n_a": int(len(a)),
        "n_b": int(len(b)),
        "median_a": float(a.median()) if len(a) else np.nan,
        "median_b": float(b.median()) if len(b) else np.nan,
        "delta_median": float(a.median() - b.median()) if (len(a) and len(b)) else np.nan,
    }
    if len(a) < 10 or len(b) < 10:
        out["note"] = "insufficient sample for tests (need >=10 per group)"
        return out

    # KS (two-sided)
    ks = ks_2samp(a, b, alternative="two-sided", mode="auto")
    out["ks_stat"] = float(ks.statistic)
    out["ks_p"] = float(ks.pvalue)

    # Mann-Whitney (one-sided: a > b means older)
    mw = mannwhitneyu(a, b, alternative="greater")
    out["mw_u"] = float(mw.statistic)
    out["mw_p_greater"] = float(mw.pvalue)

    return out


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_parquet(INP)

    age_col = pick_col(df, ["age_gyr_gyro", "age_gyr", "age_gyr_mamajek", "age_gyr_gyro_mamajek"])
    if age_col is None:
        raise RuntimeError(f"No gyro age column found. Columns sample: {list(df.columns)[:40]}")

    phi_col = pick_col(df, ["Phi_gmm", "Phi", "phi", "phi_gmm"])
    if phi_col is None:
        raise RuntimeError("No Phi column found (expected Phi_gmm).")

    win_col = pick_col(df, ["in_window_gmm", "in_window", "in_window_gmmgate"])
    if win_col is None:
        raise RuntimeError("No window membership column found (expected in_window_gmm).")

    bin_col = pick_col(df, ["teff_bin_s", "teff_bin"])
    if bin_col is None:
        raise RuntimeError("No Teff bin column found (expected teff_bin_s).")

    # Define ripeness_time explicitly (time-weighted)
    df["ripeness_time"] = pd.to_numeric(df[phi_col], errors="coerce") * pd.to_numeric(df[age_col], errors="coerce")

    # Load tails (per-bin within-window)
    t90 = pd.read_parquet(TAIL_PERBIN_Q90, columns=["source_id"])
    t95 = pd.read_parquet(TAIL_PERBIN_Q95, columns=["source_id"])
    s90 = set(t90["source_id"].astype("string"))
    s95 = set(t95["source_id"].astype("string"))

    df["source_id"] = df["source_id"].astype("string")
    df["in_tail_q90"] = df["source_id"].isin(s90)
    df["in_tail_q95"] = df["source_id"].isin(s95)

    # Global window comparison
    in_win = df.loc[df[win_col] == True, age_col]
    out_win = df.loc[df[win_col] == False, age_col]

    global_tests = {
        "age_col": age_col,
        "phi_col": phi_col,
        "window_col": win_col,
        "bin_col": bin_col,
        "rows_total": int(len(df)),
        "age_stats_overall": dist_stats(df[age_col]),
        "phi_stats_overall": dist_stats(df[phi_col]),
        "ripeness_time_stats_overall": dist_stats(df["ripeness_time"]),
        "window_age_test_global": two_sample_tests(in_win, out_win),
    }

    # Correlations (pooled)
    a = series_clean(df[age_col])
    p = series_clean(df[phi_col])
    r = series_clean(df["ripeness_time"])
    # Align indices by merge-on-index after dropping NaNs
    ap = pd.DataFrame({"age": pd.to_numeric(df[age_col], errors="coerce"),
                       "phi": pd.to_numeric(df[phi_col], errors="coerce"),
                       "ripe": pd.to_numeric(df["ripeness_time"], errors="coerce")}).dropna()

    if len(ap) >= 20:
        tau_age_phi, p_age_phi = kendalltau(ap["age"], ap["phi"])
        tau_age_ripe, p_age_ripe = kendalltau(ap["age"], ap["ripe"])
        global_tests["kendall_tau_age_phi"] = float(tau_age_phi)
        global_tests["kendall_p_age_phi"] = float(p_age_phi)
        global_tests["kendall_tau_age_ripeness"] = float(tau_age_ripe)
        global_tests["kendall_p_age_ripeness"] = float(p_age_ripe)

    # Per-bin tests
    rows_out = []
    by_bin: Dict[str, Any] = {}

    for b in TEFF_BINS:
        sub = df[df[bin_col].astype("string") == b].copy()
        if len(sub) == 0:
            continue

        sub_in = sub.loc[sub[win_col] == True, age_col]
        sub_out = sub.loc[sub[win_col] == False, age_col]

        # Tail tests: tail older than non-tail (one-sided)
        sub_t90 = sub.loc[sub["in_tail_q90"] == True, age_col]
        sub_nt90 = sub.loc[sub["in_tail_q90"] == False, age_col]
        sub_t95 = sub.loc[sub["in_tail_q95"] == True, age_col]
        sub_nt95 = sub.loc[sub["in_tail_q95"] == False, age_col]

        rec = {
            "teff_bin": b,
            "rows": int(len(sub)),
            "age_n": int(series_clean(sub[age_col]).shape[0]),
            "age_med": float(series_clean(sub[age_col]).median()) if series_clean(sub[age_col]).shape[0] else np.nan,
            "window_age_test": two_sample_tests(sub_in, sub_out),
            "tail_q90_age_test": two_sample_tests(sub_t90, sub_nt90),
            "tail_q95_age_test": two_sample_tests(sub_t95, sub_nt95),
            "phi_stats": dist_stats(sub[phi_col]),
            "ripeness_time_stats": dist_stats(sub["ripeness_time"]),
        }

        by_bin[b] = rec

        rows_out.append({
            "teff_bin": b,
            "rows": int(len(sub)),
            "age_n": rec["age_n"],
            "age_med": rec["age_med"],
            "win_delta_med": rec["window_age_test"].get("delta_median", np.nan),
            "win_ks_p": rec["window_age_test"].get("ks_p", np.nan),
            "win_mw_p_greater": rec["window_age_test"].get("mw_p_greater", np.nan),
            "tail90_delta_med": rec["tail_q90_age_test"].get("delta_median", np.nan),
            "tail90_mw_p_greater": rec["tail_q90_age_test"].get("mw_p_greater", np.nan),
            "tail95_delta_med": rec["tail_q95_age_test"].get("delta_median", np.nan),
            "tail95_mw_p_greater": rec["tail_q95_age_test"].get("mw_p_greater", np.nan),
        })

    out = {"global": global_tests, "by_bin": by_bin}

    with open(OUT_YAML, "w") as f:
        yaml.safe_dump(out, f, sort_keys=False)

    pd.DataFrame(rows_out).to_csv(OUT_CSV, index=False)

    print("saved:", OUT_YAML)
    print("saved:", OUT_CSV)
    print("age_col:", age_col, "phi_col:", phi_col, "window_col:", win_col, "bin_col:", bin_col)


if __name__ == "__main__":
    main()
