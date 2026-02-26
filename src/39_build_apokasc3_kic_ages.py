#!/usr/bin/env python3
"""
Build compact KIC->age table from APOKASC3 Table 5.

Inputs:
- /mnt/g/STAR_HPV/raw/ages/apokasc3_table5_clean.parquet

Outputs:
- /mnt/g/STAR_HPV/raw/ages/apokasc3_kic_ages.csv
- /mnt/g/STAR_HPV/raw/ages/apokasc3_kic_ages.parquet
- /mnt/g/STAR_HPV/raw/ages/apokasc3_kic_ages_summary.yaml
"""

from __future__ import annotations
import os
import yaml
import numpy as np
import pandas as pd

INP = "/mnt/g/STAR_HPV/raw/ages/apokasc3_table5_clean.parquet"
OUT_CSV = "/mnt/g/STAR_HPV/raw/ages/apokasc3_kic_ages.csv"
OUT_PARQ = "/mnt/g/STAR_HPV/raw/ages/apokasc3_kic_ages.parquet"
SUM = "/mnt/g/STAR_HPV/raw/ages/apokasc3_kic_ages_summary.yaml"

# Candidate columns (APOKASC3 naming varies; we include a few likely ones)
CANDIDATES = [
    # corrected / preferred first
    "FAgeCOR",        # appears in your preview
    # common age columns in some APOKASC-style tables
    "Age", "age", "Age_Gyr", "age_gyr",
    "RGBage", "RCage", "RGBAge", "RCAge",
    "FAgeMDRGB", "FAgeMDRC",
]

def pick_age_column(df: pd.DataFrame) -> str:
    for c in CANDIDATES:
        if c in df.columns:
            # require at least some finite values
            x = pd.to_numeric(df[c], errors="coerce")
            if np.isfinite(x.to_numpy(dtype=float, na_value=np.nan)).sum() > 100:
                return c
    # last resort: find any column with 'age' substring with many finite vals
    for c in df.columns:
        if "age" in c.lower():
            x = pd.to_numeric(df[c], errors="coerce")
            if np.isfinite(x.to_numpy(dtype=float, na_value=np.nan)).sum() > 100:
                return c
    raise RuntimeError("No suitable age-like column found in APOKASC table.")

def main() -> None:
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df = pd.read_parquet(INP)

    if "KIC" not in df.columns:
        raise RuntimeError(f"Missing KIC column in {INP}")

    # Some KIC may be repeated; keep best row per KIC by preferring non-null age
    age_col = pick_age_column(df)
    df["age_gyr_best"] = pd.to_numeric(df[age_col], errors="coerce").astype(float)

    # Optional: if there is an uncertainty column, try to use it
    err_col = None
    for c in df.columns:
        if c.lower() in {"e_age","age_err","age_error","e_fagecor","e_fage"}:
            err_col = c
            break

    if err_col is not None:
        df["age_gyr_err"] = pd.to_numeric(df[err_col], errors="coerce").astype(float)
    else:
        df["age_gyr_err"] = np.nan

    # Filter to valid KIC and ages
    x = df.loc[df["KIC"].notna()].copy()
    x = x.loc[np.isfinite(x["age_gyr_best"].to_numpy(dtype=float, na_value=np.nan))].copy()

    # For duplicates: keep row with smallest age_err if available else first
    x["__rank"] = np.where(np.isfinite(x["age_gyr_err"]), x["age_gyr_err"], np.inf)
    x = x.sort_values(["KIC", "__rank"], ascending=[True, True])
    x = x.drop_duplicates("KIC", keep="first")

    out = pd.DataFrame({
        "kic": x["KIC"].astype("Int64"),
        "age_gyr": x["age_gyr_best"].astype(float),
        "age_gyr_err": x["age_gyr_err"].astype(float),
        "age_source": f"APOKASC-3 (VizieR table5), column={age_col}",
        "age_quality_flag": 1,
    })

    out.to_csv(OUT_CSV, index=False)
    out.to_parquet(OUT_PARQ, index=False)

    summary = {
        "input_rows_clean": int(len(df)),
        "age_column_used": age_col,
        "rows_with_age": int(len(x)),
        "unique_kic_with_age": int(out["kic"].nunique()),
        "out_csv": OUT_CSV,
        "out_parquet": OUT_PARQ,
    }
    with open(SUM, "w") as f:
        yaml.safe_dump(summary, f, sort_keys=False)

    print(f"saved: {OUT_PARQ} rows={len(out)}")
    print(f"summary: {SUM}")

if __name__ == "__main__":
    main()
