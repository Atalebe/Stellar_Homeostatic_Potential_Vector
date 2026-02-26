#!/usr/bin/env python3
"""
Merge APOKASC3 ages into Kepler outputs and run basic H3-style tests.

Inputs:
- /mnt/g/STAR_HPV/raw/ages/apokasc3_ages_sourceid_agegyr.parquet
- Kepler products (post-gate, window, tails)

Outputs:
- /mnt/g/STAR_HPV/processed/ages_merged/apokasc3_*_with_ages.parquet
- /mnt/g/STAR_HPV/results/ripeness/apokasc3_age_tests.yaml
"""

from __future__ import annotations
import os
import yaml
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

AGE_PARQ = "/mnt/g/STAR_HPV/raw/ages/apokasc3_ages_sourceid_agegyr.parquet"

TARGETS = {
    "kepler_gmmgate": "/mnt/g/STAR_HPV/processed/state_vectors/kepler_led_state_vector_gmmgate.parquet",
    "anchor_window": "/mnt/g/STAR_HPV/results/anchor_catalogs/kepler_anchor_catalog.parquet",
    "tail_global_postgate_q90": "/mnt/g/STAR_HPV/results/anchor_catalogs/kepler_ripe_tail_global_postgate_q90.parquet",
    "tail_global_postgate_q95": "/mnt/g/STAR_HPV/results/anchor_catalogs/kepler_ripe_tail_global_postgate_q95.parquet",
    "tail_window_perbin_q90": "/mnt/g/STAR_HPV/results/anchor_catalogs/kepler_ripe_tail_window_perbin_q90.parquet",
    "tail_window_perbin_q95": "/mnt/g/STAR_HPV/results/anchor_catalogs/kepler_ripe_tail_window_perbin_q95.parquet",
}

OUT_DIR = "/mnt/g/STAR_HPV/processed/ages_merged"
OUT_YAML = "/mnt/g/STAR_HPV/results/ripeness/apokasc3_age_tests.yaml"

def finite_series(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").astype(float)
    return x[np.isfinite(x)]

def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(OUT_YAML), exist_ok=True)

    ages = pd.read_parquet(AGE_PARQ)
    ages["source_id"] = ages["source_id"].astype("string").str.strip()
    ages = ages.dropna(subset=["source_id"])
    ages = ages.drop_duplicates("source_id", keep="first")

    report = {
        "age_file": AGE_PARQ,
        "ages_rows": int(len(ages)),
        "ages_unique_source_id": int(ages["source_id"].nunique()),
        "targets": {},
        "tests": {},
        "note": "APOKASC overlap is expected to favor evolved stars; interpret with care for dwarf-only pipelines.",
    }

    merged_frames = {}

    # Merge
    for name, path in TARGETS.items():
        df = pd.read_parquet(path)
        if "source_id" not in df.columns:
            raise RuntimeError(f"{name} missing source_id: {path}")

        df["source_id"] = df["source_id"].astype("string").str.strip()
        m = df.merge(ages, on="source_id", how="left")

        rows = len(m)
        rows_age = int(pd.to_numeric(m["age_gyr"], errors="coerce").notna().sum())
        cov = float(rows_age / rows) if rows else 0.0

        outp = f"{OUT_DIR}/{name}_with_apokasc3_ages.parquet"
        m.to_parquet(outp, index=False)

        report["targets"][name] = {
            "input_rows": int(len(df)),
            "rows_after_merge": int(rows),
            "rows_with_age": int(rows_age),
            "coverage": cov,
            "out_parquet": outp,
        }
        merged_frames[name] = m

    # Tests on the main post-gate sample, if enough ages exist
    base = merged_frames["kepler_gmmgate"].copy()
    base["age_gyr"] = pd.to_numeric(base["age_gyr"], errors="coerce")
    base["prot_days"] = pd.to_numeric(base.get("prot_days"), errors="coerce")
    base["Phi_gmm"] = pd.to_numeric(base.get("Phi_gmm"), errors="coerce")

    # Pick the correct window flag (your files use in_window_gmm)
    win_col = "in_window_gmm" if "in_window_gmm" in base.columns else None

    base_valid = base.dropna(subset=["age_gyr"])
    if len(base_valid) >= 30:
        # age vs prot
        tau_p, p_p = kendalltau(base_valid["age_gyr"], base_valid["prot_days"], nan_policy="omit")
        # age vs Phi
        tau_phi, p_phi = kendalltau(base_valid["age_gyr"], base_valid["Phi_gmm"], nan_policy="omit")

        report["tests"]["kepler_gmmgate_pooled"] = {
            "rows_with_age": int(len(base_valid)),
            "kendall_tau_age_vs_prot": float(tau_p) if tau_p == tau_p else None,
            "p_value_age_vs_prot": float(p_p) if p_p == p_p else None,
            "kendall_tau_age_vs_phi": float(tau_phi) if tau_phi == tau_phi else None,
            "p_value_age_vs_phi": float(p_phi) if p_phi == p_phi else None,
        }

        # window vs non-window age shift (only if win flag exists)
        if win_col is not None:
            w = base_valid[base_valid[win_col] == True]["age_gyr"]
            n = base_valid[base_valid[win_col] == False]["age_gyr"]
            report["tests"]["kepler_gmmgate_window_age_shift"] = {
                "win_col": win_col,
                "n_window": int(w.shape[0]),
                "n_nonwindow": int(n.shape[0]),
                "median_age_window": float(np.nanmedian(w)) if w.shape[0] else None,
                "median_age_nonwindow": float(np.nanmedian(n)) if n.shape[0] else None,
                "delta_median_age_gyr": float(np.nanmedian(w) - np.nanmedian(n)) if (w.shape[0] and n.shape[0]) else None,
            }

    with open(OUT_YAML, "w") as f:
        yaml.safe_dump(report, f, sort_keys=False)

    print(f"saved: {OUT_YAML}")
    print("targets coverage:")
    for k,v in report["targets"].items():
        print(f"  {k}: rows_with_age={v['rows_with_age']} coverage={v['coverage']:.4f}")

if __name__ == "__main__":
    main()
