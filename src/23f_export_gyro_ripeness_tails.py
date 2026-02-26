#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

INP = "/mnt/g/STAR_HPV/processed/gyro/kepler_gmmgate_with_gyro_ages_mamajek.parquet"
OUT_DIR = Path("/mnt/g/STAR_HPV/results/anchor_catalogs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

Q_LIST = [0.9, 0.95]

def qtile(x, q):
    x = pd.to_numeric(x, errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan).dropna()
    return float(x.quantile(q)) if len(x) else np.nan

def main():
    df = pd.read_parquet(INP)

    # Require window + gyro age
    need = ["source_id","kic","teff_bin_s","prot_days","Phi_gmm","in_window_gmm","age_gyr_gyro","ripeness_time_gyr"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns in {INP}: {missing}")

    # Global postgate tails (no window restriction)
    out = {"input": INP, "rows_total": int(len(df)), "tails": {}}

    for q in Q_LIST:
        thr = qtile(df["ripeness_time_gyr"], q)
        tail = df[df["ripeness_time_gyr"] >= thr].copy()

        p_csv = OUT_DIR / f"kepler_gyro_ripe_tail_global_postgate_q{int(q*100)}.csv"
        p_parq = OUT_DIR / f"kepler_gyro_ripe_tail_global_postgate_q{int(q*100)}.parquet"
        tail.to_csv(p_csv, index=False)
        tail.to_parquet(p_parq, index=False)

        out["tails"][f"global_postgate_q{int(q*100)}"] = {
            "q": q, "threshold": thr, "rows": int(len(tail)),
            "by_bin": tail["teff_bin_s"].value_counts().to_dict(),
            "outputs": {"csv": str(p_csv), "parquet": str(p_parq)}
        }

    # Window-restricted tails
    win = df[df["in_window_gmm"] == True].copy()
    out["window"] = {"rows_in_window": int(len(win)), "fraction": float(len(win)/len(df))}

    for q in Q_LIST:
        thr = qtile(win["ripeness_time_gyr"], q)
        tail = win[win["ripeness_time_gyr"] >= thr].copy()

        p_csv = OUT_DIR / f"kepler_gyro_ripe_tail_window_q{int(q*100)}.csv"
        p_parq = OUT_DIR / f"kepler_gyro_ripe_tail_window_q{int(q*100)}.parquet"
        tail.to_csv(p_csv, index=False)
        tail.to_parquet(p_parq, index=False)

        out["tails"][f"window_q{int(q*100)}"] = {
            "q": q, "threshold": thr, "rows": int(len(tail)),
            "by_bin": tail["teff_bin_s"].value_counts().to_dict(),
            "outputs": {"csv": str(p_csv), "parquet": str(p_parq)}
        }

    # Per-bin window tails (threshold computed within each teff bin)
    perbin_rows = {}
    for q in Q_LIST:
        pieces = []
        thr_by_bin = {}
        for b, g in win.groupby("teff_bin_s"):
            thr = qtile(g["ripeness_time_gyr"], q)
            thr_by_bin[b] = thr
            pieces.append(g[g["ripeness_time_gyr"] >= thr])
        tail = pd.concat(pieces, ignore_index=True) if pieces else win.iloc[0:0].copy()

        p_csv = OUT_DIR / f"kepler_gyro_ripe_tail_window_perbin_q{int(q*100)}.csv"
        p_parq = OUT_DIR / f"kepler_gyro_ripe_tail_window_perbin_q{int(q*100)}.parquet"
        tail.to_csv(p_csv, index=False)
        tail.to_parquet(p_parq, index=False)

        perbin_rows[f"q{int(q*100)}"] = {
            "q": q,
            "thresholds_by_bin": thr_by_bin,
            "rows": int(len(tail)),
            "by_bin": tail["teff_bin_s"].value_counts().to_dict(),
            "outputs": {"csv": str(p_csv), "parquet": str(p_parq)},
        }

    out["perbin_window_tails"] = perbin_rows

    p_yaml = OUT_DIR / "kepler_gyro_ripe_tail_summary.yaml"
    with open(p_yaml, "w") as f:
        yaml.safe_dump(out, f, sort_keys=False)

    print(f"saved: {p_yaml}")
    print(out)

if __name__ == "__main__":
    main()
