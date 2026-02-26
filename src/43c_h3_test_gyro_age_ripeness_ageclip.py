#!/usr/bin/env python3
"""
H3 robustness check #1: age clipping.

Goal:
- Re-run the H3 tests after clipping gyro ages to a physically plausible ceiling
  (default 13.8 Gyr) and recomputing ripeness_time = Phi_gmm * age_gyr_gyro.

Input:
  /mnt/g/STAR_HPV/processed/gyro/kepler_gmmgate_with_gyro_ages_mamajek.parquet

Output:
  /mnt/g/STAR_HPV/results/ripeness/h3_gyro_age_tests_ageclip.yaml
  /mnt/g/STAR_HPV/results/ripeness/h3_gyro_age_tests_ageclip_by_bin.csv
"""

from __future__ import annotations

import os
import yaml
import numpy as np
import pandas as pd

from scipy.stats import ks_2samp, mannwhitneyu, kendalltau

INP = "/mnt/g/STAR_HPV/processed/gyro/kepler_gmmgate_with_gyro_ages_mamajek.parquet"
OUT_YAML = "/mnt/g/STAR_HPV/results/ripeness/h3_gyro_age_tests_ageclip.yaml"
OUT_CSV  = "/mnt/g/STAR_HPV/results/ripeness/h3_gyro_age_tests_ageclip_by_bin.csv"

AGE_COL = "age_gyr_gyro"
PHI_COL = "Phi_gmm"
WIN_COL = "in_window_gmm"
BIN_COL = "teff_bin_s"

# Prefer this, because your file uses it
RIP_COL_CANDIDATES = [
    "ripeness_time_gyr",  # expected in your file
    "ripeness_time",
    "ripeness_gmm",
    "ripeness_rot",
    "ripeness",
]

AGE_MAX_GYR = 13.8  # clip ceiling
MIN_N_FOR_TEST = 10


def qstats(x: pd.Series) -> dict:
    x = pd.to_numeric(x, errors="coerce").dropna().astype(float)
    if len(x) == 0:
        return {"n": 0}
    return {
        "n": int(len(x)),
        "min": float(np.min(x)),
        "p10": float(np.quantile(x, 0.10)),
        "p50": float(np.quantile(x, 0.50)),
        "p90": float(np.quantile(x, 0.90)),
        "max": float(np.max(x)),
    }


def window_age_test(df: pd.DataFrame, age_col: str, win_col: str) -> dict | None:
    a = pd.to_numeric(df.loc[df[win_col] == True, age_col], errors="coerce").dropna().astype(float)
    b = pd.to_numeric(df.loc[df[win_col] == False, age_col], errors="coerce").dropna().astype(float)
    if len(a) < MIN_N_FOR_TEST or len(b) < MIN_N_FOR_TEST:
        return {
            "n_a": int(len(a)),
            "n_b": int(len(b)),
            "note": f"insufficient sample for tests (need >= {MIN_N_FOR_TEST} per group)"
        }

    # KS test (two-sided)
    ks = ks_2samp(a, b, alternative="two-sided", mode="auto")

    # Mann-Whitney U for "a > b" (window older)
    mw = mannwhitneyu(a, b, alternative="greater")

    return {
        "n_a": int(len(a)),
        "n_b": int(len(b)),
        "median_a": float(np.median(a)),
        "median_b": float(np.median(b)),
        "delta_median": float(np.median(a) - np.median(b)),
        "ks_stat": float(ks.statistic),
        "ks_p": float(ks.pvalue),
        "mw_u": float(mw.statistic),
        "mw_p_greater": float(mw.pvalue),
    }


def tail_age_test(df_bin: pd.DataFrame, age_col: str, tail_ids: set[str]) -> dict:
    in_tail = df_bin["source_id"].astype(str).isin(tail_ids)
    a = pd.to_numeric(df_bin.loc[in_tail, age_col], errors="coerce").dropna().astype(float)
    b = pd.to_numeric(df_bin.loc[~in_tail, age_col], errors="coerce").dropna().astype(float)

    if len(a) < MIN_N_FOR_TEST or len(b) < MIN_N_FOR_TEST:
        return {
            "n_a": int(len(a)),
            "n_b": int(len(b)),
            "median_a": float(np.median(a)) if len(a) else None,
            "median_b": float(np.median(b)) if len(b) else None,
            "delta_median": float(np.median(a) - np.median(b)) if (len(a) and len(b)) else None,
            "note": f"insufficient sample for tests (need >= {MIN_N_FOR_TEST} per group)"
        }

    ks = ks_2samp(a, b, alternative="two-sided", mode="auto")
    mw = mannwhitneyu(a, b, alternative="greater")

    return {
        "n_a": int(len(a)),
        "n_b": int(len(b)),
        "median_a": float(np.median(a)),
        "median_b": float(np.median(b)),
        "delta_median": float(np.median(a) - np.median(b)),
        "ks_stat": float(ks.statistic),
        "ks_p": float(ks.pvalue),
        "mw_u": float(mw.statistic),
        "mw_p_greater": float(mw.pvalue),
    }


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

    df = pd.read_parquet(INP)
    df["source_id"] = df["source_id"].astype(str)

    rip_col = pick_ripeness_col(df)

    required = [AGE_COL, PHI_COL, WIN_COL, BIN_COL, rip_col, "source_id"]
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise RuntimeError(f"Missing required columns: {miss}. Available: {list(df.columns)[:40]}")

    # coerce
    df[AGE_COL] = pd.to_numeric(df[AGE_COL], errors="coerce").astype(float)
    df[PHI_COL] = pd.to_numeric(df[PHI_COL], errors="coerce").astype(float)
    df[WIN_COL] = df[WIN_COL].astype(bool)

    # clip ages
    df["age_gyr_gyro_raw"] = df[AGE_COL]
    df[AGE_COL] = df[AGE_COL].clip(lower=0.0, upper=float(AGE_MAX_GYR))

    # recompute ripeness_time using clipped ages
    # we write into rip_col if it is a time-based ripeness col; otherwise create ripeness_time_gyr
    out_rip_col = "ripeness_time_gyr"
    df[out_rip_col] = df[PHI_COL] * df[AGE_COL]

    # also keep original ripeness column as-is
    # (this script evaluates ripeness_time_gyr computed from age clipping)
    use_rip_col = out_rip_col

    # load tails (per-bin within-window tails are the cleanest H3 target)
    tail_q90_path = "/mnt/g/STAR_HPV/results/anchor_catalogs/kepler_gyro_ripe_tail_window_perbin_q90.parquet"
    tail_q95_path = "/mnt/g/STAR_HPV/results/anchor_catalogs/kepler_gyro_ripe_tail_window_perbin_q95.parquet"
    tail_q90 = pd.read_parquet(tail_q90_path, columns=["source_id"]).astype({"source_id": "string"})
    tail_q95 = pd.read_parquet(tail_q95_path, columns=["source_id"]).astype({"source_id": "string"})
    tail_ids_q90 = set(tail_q90["source_id"].astype(str).tolist())
    tail_ids_q95 = set(tail_q95["source_id"].astype(str).tolist())

    # global stats
    out = {
        "global": {
            "input": INP,
            "age_clip_max_gyr": float(AGE_MAX_GYR),
            "age_col": AGE_COL,
            "phi_col": PHI_COL,
            "window_col": WIN_COL,
            "bin_col": BIN_COL,
            "ripeness_col_used": use_rip_col,
            "rows_total": int(len(df)),
            "age_stats_overall_clipped": qstats(df[AGE_COL]),
            "age_stats_overall_raw": qstats(df["age_gyr_gyro_raw"]),
            "phi_stats_overall": qstats(df[PHI_COL]),
            "ripeness_time_stats_overall": qstats(df[use_rip_col]),
            "window_age_test_global": window_age_test(df, AGE_COL, WIN_COL),
        },
        "by_bin": {}
    }

    # global correlations
    dsub = df[[AGE_COL, PHI_COL, use_rip_col]].dropna()
    if len(dsub) >= MIN_N_FOR_TEST:
        tau1, p1 = kendalltau(dsub[AGE_COL], dsub[PHI_COL])
        tau2, p2 = kendalltau(dsub[AGE_COL], dsub[use_rip_col])
        out["global"]["kendall_tau_age_phi"] = float(tau1) if tau1 == tau1 else None
        out["global"]["kendall_p_age_phi"] = float(p1) if p1 == p1 else None
        out["global"]["kendall_tau_age_ripeness"] = float(tau2) if tau2 == tau2 else None
        out["global"]["kendall_p_age_ripeness"] = float(p2) if p2 == p2 else None

    rows_csv = []

    for teff_bin, g in df.groupby(BIN_COL):
        g = g.copy()
        g_age = pd.to_numeric(g[AGE_COL], errors="coerce").dropna()
        out_bin = {
            "teff_bin": str(teff_bin),
            "rows": int(len(g)),
            "age_n": int(g_age.shape[0]),
            "age_med": float(np.median(g_age)) if len(g_age) else None,
            "window_age_test": window_age_test(g, AGE_COL, WIN_COL),
            "tail_q90_age_test": tail_age_test(g, AGE_COL, tail_ids_q90),
            "tail_q95_age_test": tail_age_test(g, AGE_COL, tail_ids_q95),
            "phi_stats": qstats(g[PHI_COL]),
            "ripeness_time_stats": qstats(g[use_rip_col]),
        }
        out["by_bin"][str(teff_bin)] = out_bin

        rows_csv.append({
            "teff_bin": str(teff_bin),
            "rows": int(len(g)),
            "age_n": out_bin["age_n"],
            "age_median": out_bin["age_med"],
            "window_median_age_in": out_bin["window_age_test"].get("median_a") if isinstance(out_bin["window_age_test"], dict) else None,
            "window_median_age_out": out_bin["window_age_test"].get("median_b") if isinstance(out_bin["window_age_test"], dict) else None,
            "window_delta_median": out_bin["window_age_test"].get("delta_median") if isinstance(out_bin["window_age_test"], dict) else None,
            "window_ks_p": out_bin["window_age_test"].get("ks_p") if isinstance(out_bin["window_age_test"], dict) else None,
            "window_mw_p_greater": out_bin["window_age_test"].get("mw_p_greater") if isinstance(out_bin["window_age_test"], dict) else None,
            "tail_q90_delta_median": out_bin["tail_q90_age_test"].get("delta_median") if isinstance(out_bin["tail_q90_age_test"], dict) else None,
            "tail_q90_mw_p_greater": out_bin["tail_q90_age_test"].get("mw_p_greater") if isinstance(out_bin["tail_q90_age_test"], dict) else None,
            "tail_q95_delta_median": out_bin["tail_q95_age_test"].get("delta_median") if isinstance(out_bin["tail_q95_age_test"], dict) else None,
            "tail_q95_mw_p_greater": out_bin["tail_q95_age_test"].get("mw_p_greater") if isinstance(out_bin["tail_q95_age_test"], dict) else None,
        })

    # write outputs
    with open(OUT_YAML, "w") as f:
        yaml.safe_dump(out, f, sort_keys=False)

    pd.DataFrame(rows_csv).to_csv(OUT_CSV, index=False)

    print(f"saved: {OUT_YAML}")
    print(f"saved: {OUT_CSV}")
    print("age_col:", AGE_COL, "phi_col:", PHI_COL, "ripeness_col_used:", use_rip_col, "window_col:", WIN_COL, "bin_col:", BIN_COL)


if __name__ == "__main__":
    main()
