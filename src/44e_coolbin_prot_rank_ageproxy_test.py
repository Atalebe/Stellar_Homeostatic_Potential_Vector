#!/usr/bin/env python3
"""
44e_coolbin_prot_rank_ageproxy_test.py

In 3200-4000K, replace gyro age with a monotonic Prot-based proxy and re-test
window vs non-window. This isolates whether the anomaly is just "window rotates faster".

Outputs:
- YAML: /mnt/g/STAR_HPV/results/ripeness/coolbin_protproxy_window_test.yaml
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
import yaml
from scipy.stats import ks_2samp, mannwhitneyu

INP = "/mnt/g/STAR_HPV/processed/gyro/kepler_gmmgate_with_gyro_ages_mamajek.parquet"
OUT = "/mnt/g/STAR_HPV/results/ripeness/coolbin_protproxy_window_test.yaml"

BIN = "3200-4000"
WIN = "in_window_gmm"
PROT = "prot_days"

def ensure_dir(p: str) -> None:
    os.makedirs(os.path.dirname(p), exist_ok=True)

def arr(s: pd.Series) -> np.ndarray:
    x = pd.to_numeric(s, errors="coerce").astype(float).to_numpy()
    return x[np.isfinite(x)]

def main() -> None:
    ensure_dir(OUT)
    df = pd.read_parquet(INP)
    for c in ["teff_bin_s", WIN, PROT]:
        if c not in df.columns:
            raise RuntimeError(f"Missing {c}. Have: {list(df.columns)[:50]}")

    cool = df[df["teff_bin_s"] == BIN].copy()
    cool[WIN] = cool[WIN].astype(bool)

    # define prot-only proxy: z-score of log Prot within bin (monotonic)
    prot = pd.to_numeric(cool[PROT], errors="coerce").astype(float)
    logp = np.log10(prot)
    mu = np.nanmean(logp)
    sd = np.nanstd(logp)
    cool["ageproxy_prot_z"] = (logp - mu) / (sd + 1e-12)

    w = arr(cool.loc[cool[WIN] == True, "ageproxy_prot_z"])
    n = arr(cool.loc[cool[WIN] == False, "ageproxy_prot_z"])

    out = {
        "input": INP,
        "bin": BIN,
        "window_col": WIN,
        "prot_col": PROT,
        "proxy": "ageproxy_prot_z = zscore(log10(prot_days)) within bin",
        "n_window": int(len(w)),
        "n_nonwindow": int(len(n)),
        "median_window": float(np.median(w)),
        "median_nonwindow": float(np.median(n)),
        "delta_median": float(np.median(w) - np.median(n)),
    }

    if len(w) >= 10 and len(n) >= 10:
        ks = ks_2samp(w, n)
        mw = mannwhitneyu(w, n, alternative="greater")
        out["ks_p"] = float(ks.pvalue)
        out["mw_p_window_olderproxy"] = float(mw.pvalue)
    else:
        out["note"] = "insufficient sample for tests"

    with open(OUT, "w") as f:
        yaml.safe_dump(out, f, sort_keys=False)

    print("saved:", OUT)
    print(out)

if __name__ == "__main__":
    main()
