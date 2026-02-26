#!/usr/bin/env python3
"""
Merge APOKASC3 ages (source_id keyed) into Kepler/Gaia products and run quick age diagnostics.

- Reads /mnt/g/STAR_HPV/raw/ages/apokasc3_ages_sourceid_agegyr.parquet
- Merges into:
  * kepler_gmmgate
  * anchor_window
  * tail_global_postgate_q90/q95
  * tail_window_q90/q95
  * tail_window_perbin_q90/q95
- Produces:
  * merged parquet outputs (ages_merged/)
  * summary yaml with coverage and age stats
  * matched CSVs for quick inspection
"""

from __future__ import annotations
import os
import yaml
import numpy as np
import pandas as pd

AGES = "/mnt/g/STAR_HPV/raw/ages/apokasc3_ages_sourceid_agegyr.parquet"

TARGETS = {
    "kepler_gmmgate": "/mnt/g/STAR_HPV/processed/state_vectors/kepler_led_state_vector_gmmgate.parquet",
    "anchor_window": "/mnt/g/STAR_HPV/results/anchor_catalogs/kepler_anchor_catalog.parquet",

    "tail_global_postgate_q90": "/mnt/g/STAR_HPV/results/anchor_catalogs/kepler_ripe_tail_global_postgate_q90.parquet",
    "tail_global_postgate_q95": "/mnt/g/STAR_HPV/results/anchor_catalogs/kepler_ripe_tail_global_postgate_q95.parquet",

    "tail_window_q90": "/mnt/g/STAR_HPV/results/anchor_catalogs/kepler_ripe_tail_window_q90.parquet",
    "tail_window_q95": "/mnt/g/STAR_HPV/results/anchor_catalogs/kepler_ripe_tail_window_q95.parquet",

    "tail_window_perbin_q90": "/mnt/g/STAR_HPV/results/anchor_catalogs/kepler_ripe_tail_window_perbin_q90.parquet",
    "tail_window_perbin_q95": "/mnt/g/STAR_HPV/results/anchor_catalogs/kepler_ripe_tail_window_perbin_q95.parquet",
}

OUTDIR = "/mnt/g/STAR_HPV/processed/ages_merged"
YAML_OUT = "/mnt/g/STAR_HPV/results/ripeness/apokasc3_age_tests_v2.yaml"

def safe_float(x):
    try:
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return None
        return float(x)
    except Exception:
        return None

def age_quality_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add simple flags; do NOT delete anything.
    """
    df = df.copy()
    a = pd.to_numeric(df.get("age_gyr"), errors="coerce").astype(float)
    e = pd.to_numeric(df.get("age_gyr_err"), errors="coerce").astype(float)

    df["age_has"] = np.isfinite(a)
    df["age_err_has"] = np.isfinite(e)
    df["age_err_gt_age"] = df["age_has"] & df["age_err_has"] & (e > a)
    df["age_err_gt_5gyr"] = df["age_err_has"] & (e > 5.0)
    df["age_err_gt_10gyr"] = df["age_err_has"] & (e > 10.0)

    # compact scalar quality: 1=best, 0=usable, -1=very uncertain
    df["age_qflag_simple"] = np.where(df["age_has"], 0, -9)
    df.loc[df["age_has"] & (~df["age_err_gt_age"]) & (~df["age_err_gt_5gyr"]), "age_qflag_simple"] = 1
    df.loc[df["age_has"] & (df["age_err_gt_age"] | df["age_err_gt_10gyr"]), "age_qflag_simple"] = -1
    return df

def stats_block(df: pd.DataFrame, age_col="age_gyr") -> dict:
    a = pd.to_numeric(df.get(age_col), errors="coerce").astype(float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return {"n": 0}
    return {
        "n": int(len(a)),
        "min": safe_float(a.min()),
        "p10": safe_float(np.quantile(a, 0.10)),
        "p50": safe_float(np.quantile(a, 0.50)),
        "p90": safe_float(np.quantile(a, 0.90)),
        "max": safe_float(a.max()),
    }

def main() -> None:
    os.makedirs(OUTDIR, exist_ok=True)
    os.makedirs(os.path.dirname(YAML_OUT), exist_ok=True)

    ages = pd.read_parquet(AGES, columns=["source_id","age_gyr","age_gyr_err","age_source","age_quality_flag","kic"])
    ages["source_id"] = ages["source_id"].astype("string").str.strip()

    summary = {
        "age_file": AGES,
        "ages_rows": int(len(ages)),
        "ages_unique_source_id": int(ages["source_id"].nunique()),
        "ages_age_stats": stats_block(ages, "age_gyr"),
        "targets": {},
        "tests": {},
        "note": "APOKASC overlap may be limited; interpret in light of evolved-star selection and rotator-led filtering.",
    }

    # Load kepler_gmmgate once for reference comparisons
    base = pd.read_parquet(TARGETS["kepler_gmmgate"])
    base["source_id"] = base["source_id"].astype("string").str.strip()

    for name, path in TARGETS.items():
        df = pd.read_parquet(path)
        df["source_id"] = df["source_id"].astype("string").str.strip()

        m = df.merge(ages, on="source_id", how="left", suffixes=("", "_age"))
        m = age_quality_flags(m)

        rows_with_age = int(m["age_has"].sum())
        coverage = float(rows_with_age / len(m)) if len(m) else 0.0

        out_parq = os.path.join(OUTDIR, f"{name}_with_apokasc3_ages.parquet")
        m.to_parquet(out_parq, index=False)

        # Save matched-only quick view
        out_match_csv = None
        if rows_with_age > 0:
            keep_cols = [c for c in [
                "source_id","kic","teff_bin_s","teff_bin","prot_days",
                "Phi_gmm","Phi","ripeness_gmm","in_window_gmm","in_window",
                "age_gyr","age_gyr_err","age_qflag_simple"
            ] if c in m.columns]
            match = m.loc[m["age_has"], keep_cols].copy()
            out_match_csv = os.path.join(OUTDIR, f"{name}_apokasc3_matches.csv")
            match.to_csv(out_match_csv, index=False)

        # Optional: compare “window vs non-window” ages when possible
        # Uses in_window_gmm if present, else in_window.
        test = {}
        wcol = "in_window_gmm" if "in_window_gmm" in m.columns else ("in_window" if "in_window" in m.columns else None)
        if wcol and rows_with_age >= 30:
            aa = m.loc[m["age_has"], ["age_gyr", wcol]].copy()
            aa[wcol] = aa[wcol].astype(bool)
            test = {
                "age_median_window": safe_float(aa.loc[aa[wcol], "age_gyr"].median()) if aa[wcol].any() else None,
                "age_median_nonwindow": safe_float(aa.loc[~aa[wcol], "age_gyr"].median()) if (~aa[wcol]).any() else None,
                "rows_with_age": rows_with_age,
            }

        summary["targets"][name] = {
            "input_rows": int(len(df)),
            "rows_after_merge": int(len(m)),
            "rows_with_age": rows_with_age,
            "coverage": coverage,
            "age_stats": stats_block(m.loc[m["age_has"]], "age_gyr"),
            "out_parquet": out_parq,
            "out_matched_csv": out_match_csv,
            "window_age_test": test if test else None,
        }

    with open(YAML_OUT, "w") as f:
        yaml.safe_dump(summary, f, sort_keys=False)

    print(f"saved: {YAML_OUT}")
    for k, v in summary["targets"].items():
        print(f"{k}: rows_with_age={v['rows_with_age']} coverage={v['coverage']:.4f}")

if __name__ == "__main__":
    main()
