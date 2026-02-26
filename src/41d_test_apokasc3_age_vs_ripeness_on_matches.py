#!/usr/bin/env python3
"""
Run age correlation tests on the subset that actually matches APOKASC ages (by KIC).

This avoids "coverage=0.0000" confusion by operating only on matched rows.
"""

from __future__ import annotations
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

INP = "/mnt/g/STAR_HPV/processed/ages_merged/kepler_gmmgate_with_apokasc3_ages_by_kic.parquet"
OUT = Path("/mnt/g/STAR_HPV/results/ripeness/apokasc3_age_tests_on_matches.yaml")

def _kt(x, y):
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 10:
        return {"n": int(m.sum())}
    tau, p = kendalltau(x[m], y[m])
    return {"n": int(m.sum()), "kendall_tau": float(tau), "p_value": float(p)}

def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(INP)

    # pick the columns you already have in kepler_gmmgate:
    # age_gyr from APOKASC, prot_days, Phi_gmm, ripeness_gmm (if present)
    res = {"input": INP, "matched_rows": int(df["age_gyr"].notna().sum()), "tests": {}}

    sub = df[df["age_gyr"].notna()].copy()
    if len(sub) == 0:
        with open(OUT, "w") as f:
            yaml.safe_dump(res, f, sort_keys=False)
        print(f"saved: {OUT} (no matches)")
        return

    if "prot_days" in sub.columns:
        res["tests"]["age_vs_prot_days"] = _kt(sub["age_gyr"], sub["prot_days"])
    if "Phi_gmm" in sub.columns:
        res["tests"]["age_vs_phi_gmm"] = _kt(sub["age_gyr"], sub["Phi_gmm"])
    if "ripeness_gmm" in sub.columns:
        res["tests"]["age_vs_ripeness_gmm"] = _kt(sub["age_gyr"], sub["ripeness_gmm"])
    else:
        # compute if missing
        if ("Phi_gmm" in sub.columns) and ("prot_days" in sub.columns):
            sub["ripeness_gmm"] = pd.to_numeric(sub["Phi_gmm"], errors="coerce") * pd.to_numeric(sub["prot_days"], errors="coerce")
            res["tests"]["age_vs_ripeness_gmm"] = _kt(sub["age_gyr"], sub["ripeness_gmm"])

    with open(OUT, "w") as f:
        yaml.safe_dump(res, f, sort_keys=False)

    print(f"saved: {OUT}")
    print(res)

if __name__ == "__main__":
    main()
