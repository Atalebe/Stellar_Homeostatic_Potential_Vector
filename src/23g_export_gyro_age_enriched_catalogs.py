#!/usr/bin/env python3
"""
Export age-enriched gyro catalogs for H3.

Reads:
  /mnt/g/STAR_HPV/processed/gyro/kepler_gmmgate_with_gyro_ages_mamajek.parquet

Writes (suggested):
  /mnt/g/STAR_HPV/results/anchor_catalogs/kepler_gmmgate_with_gyroages.csv/parquet
  /mnt/g/STAR_HPV/results/anchor_catalogs/kepler_anchor_window_with_gyroages.csv/parquet
  /mnt/g/STAR_HPV/results/anchor_catalogs/kepler_gyro_time_ripe_tail_*.csv/parquet
  /mnt/g/STAR_HPV/results/anchor_catalogs/kepler_gyro_time_exports_summary.yaml

Notes:
- Uses ripeness_time = Phi_gmm * age_gyr_gyro
- Tail membership is computed (if not already exported) from in-window subsets.
"""

from __future__ import annotations

import os
import yaml
import numpy as np
import pandas as pd

INP = "/mnt/g/STAR_HPV/processed/gyro/kepler_gmmgate_with_gyro_ages_mamajek.parquet"
OUT_DIR = "/mnt/g/STAR_HPV/results/anchor_catalogs"
OUT_YAML = os.path.join(OUT_DIR, "kepler_gyro_time_exports_summary.yaml")

TEFF_BINS = ["3200-4000", "4000-5200", "5200-6000", "6000-7500"]


def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def qstats(x):
    x = pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) == 0:
        return {"n": 0}
    qs = np.quantile(x, [0.0, 0.1, 0.5, 0.9, 1.0])
    return {"n": int(len(x)), "min": float(qs[0]), "p10": float(qs[1]), "p50": float(qs[2]), "p90": float(qs[3]), "max": float(qs[4])}


def export(df, stem):
    csv = os.path.join(OUT_DIR, stem + ".csv")
    pq  = os.path.join(OUT_DIR, stem + ".parquet")
    df.to_csv(csv, index=False)
    df.to_parquet(pq, index=False)
    return {"csv": csv, "parquet": pq, "rows": int(len(df))}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_parquet(INP)

    age_col = pick_col(df, ["age_gyr_gyro", "age_gyr", "age_gyr_mamajek", "age_gyr_gyro_mamajek"])
    phi_col = pick_col(df, ["Phi_gmm", "Phi"])
    win_col = pick_col(df, ["in_window_gmm", "in_window"])
    bin_col = pick_col(df, ["teff_bin_s", "teff_bin"])

    if not all([age_col, phi_col, win_col, bin_col]):
        raise RuntimeError(f"Missing required columns. age={age_col} phi={phi_col} win={win_col} bin={bin_col}")

    df["source_id"] = df["source_id"].astype("string")
    df["ripeness_time"] = pd.to_numeric(df[phi_col], errors="coerce") * pd.to_numeric(df[age_col], errors="coerce")

    # Core exports
    out = {"input": INP, "age_col": age_col, "phi_col": phi_col, "window_col": win_col, "bin_col": bin_col}

    out["exports"] = {}
    out["exports"]["gmmgate_with_gyroages"] = export(df, "kepler_gmmgate_with_gyroages")

    win = df[df[win_col] == True].copy()
    out["exports"]["anchor_window_with_gyroages"] = export(win, "kepler_anchor_window_with_gyroages")

    # Global postgate tails on ripeness_time (entire gated sample)
    tails = {}
    for q in [0.9, 0.95]:
        thr = float(np.nanquantile(df["ripeness_time"], q))
        t = df[df["ripeness_time"] >= thr].copy()
        tails[f"global_postgate_q{int(q*100)}"] = {
            "q": q,
            "threshold": thr,
            "rows": int(len(t)),
            "by_bin": t[bin_col].astype("string").value_counts().to_dict(),
            "age_stats": qstats(t[age_col]),
            "outputs": export(t, f"kepler_gyro_time_ripe_tail_global_postgate_q{int(q*100)}"),
        }

    # Within-window tails (global threshold within window)
    for q in [0.9, 0.95]:
        thr = float(np.nanquantile(win["ripeness_time"], q))
        t = win[win["ripeness_time"] >= thr].copy()
        tails[f"window_q{int(q*100)}"] = {
            "q": q,
            "threshold": thr,
            "rows": int(len(t)),
            "by_bin": t[bin_col].astype("string").value_counts().to_dict(),
            "age_stats": qstats(t[age_col]),
            "outputs": export(t, f"kepler_gyro_time_ripe_tail_window_q{int(q*100)}"),
        }

    # Per-bin within-window tails (balanced)
    perbin = {}
    for q in [0.9, 0.95]:
        parts = []
        thresholds = {}
        bybin = {}
        for b in TEFF_BINS:
            w = win[win[bin_col].astype("string") == b].copy()
            if len(w) < 20:
                continue
            thr = float(np.nanquantile(w["ripeness_time"], q))
            thresholds[b] = thr
            tw = w[w["ripeness_time"] >= thr].copy()
            bybin[b] = int(len(tw))
            parts.append(tw)
        t = pd.concat(parts, ignore_index=True) if parts else win.iloc[0:0].copy()
        perbin[f"q{int(q*100)}"] = {
            "q": q,
            "thresholds_by_bin": thresholds,
            "rows": int(len(t)),
            "by_bin": bybin,
            "age_stats": qstats(t[age_col]),
            "outputs": export(t, f"kepler_gyro_time_ripe_tail_window_perbin_q{int(q*100)}"),
        }

    out["tails"] = tails
    out["perbin_window_tails"] = perbin
    out["window"] = {"rows_in_window": int(len(win)), "fraction": float(len(win)/len(df))}

    with open(OUT_YAML, "w") as f:
        yaml.safe_dump(out, f, sort_keys=False)

    print("saved:", OUT_YAML)


if __name__ == "__main__":
    main()
