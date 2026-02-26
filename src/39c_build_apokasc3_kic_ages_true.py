#!/usr/bin/env python3
"""
Build KIC->age table from APOKASC3 using a real age-like column (auto-selected).

Inputs:
- /mnt/g/STAR_HPV/raw/ages/apokasc3_table5_clean.parquet
- /mnt/g/STAR_HPV/raw/ages/apokasc3_age_column_diagnostics.yaml

Outputs:
- /mnt/g/STAR_HPV/raw/ages/apokasc3_kic_ages_true.csv
- /mnt/g/STAR_HPV/raw/ages/apokasc3_kic_ages_true.parquet
- /mnt/g/STAR_HPV/raw/ages/apokasc3_kic_ages_true_summary.yaml
"""

from __future__ import annotations
import os
import yaml
import numpy as np
import pandas as pd

INP = "/mnt/g/STAR_HPV/raw/ages/apokasc3_table5_clean.parquet"
DIAG = "/mnt/g/STAR_HPV/raw/ages/apokasc3_age_column_diagnostics.yaml"

OUT_CSV = "/mnt/g/STAR_HPV/raw/ages/apokasc3_kic_ages_true.csv"
OUT_PARQ = "/mnt/g/STAR_HPV/raw/ages/apokasc3_kic_ages_true.parquet"
SUM = "/mnt/g/STAR_HPV/raw/ages/apokasc3_kic_ages_true_summary.yaml"

def main() -> None:
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

    with open(DIAG, "r") as f:
        d = yaml.safe_load(f)

    top = d.get("top_plausible_age_columns", [])
    if not top:
        raise RuntimeError("No plausible age columns found. Check diagnostics YAML.")

    age_col = top[0]["col"]

    df = pd.read_parquet(INP)
    df["KIC"] = pd.to_numeric(df["KIC"], errors="coerce").astype("Int64")

    age = pd.to_numeric(df[age_col], errors="coerce").astype(float)

    # If it looks like log(age/yr) or log10(age/yr), try to detect and convert.
    # Heuristic: values around 9-10 => log10(age/yr). Convert to Gyr.
    if np.nanmedian(age) > 5 and np.nanmedian(age) < 12:
        # assume log10(age/yr)
        age_gyr = (10 ** age) / 1e9
        age_source = f"APOKASC-3 (VizieR table5), log10(age/yr) column={age_col}"
    else:
        age_gyr = age
        age_source = f"APOKASC-3 (VizieR table5), age_gyr column={age_col}"

    x = df.loc[df["KIC"].notna()].copy()
    x["age_gyr"] = pd.to_numeric(age_gyr, errors="coerce").astype(float)
    x = x.loc[np.isfinite(x["age_gyr"])].copy()

    # keep 0 < age < 20 Gyr sanity
    x = x.loc[(x["age_gyr"] > 0) & (x["age_gyr"] < 20)].copy()

    # duplicates: keep first (or you can add a quality ranking later)
    x = x.drop_duplicates("KIC", keep="first")

    out = pd.DataFrame({
        "kic": x["KIC"].astype("Int64"),
        "age_gyr": x["age_gyr"].astype(float),
        "age_gyr_err": np.nan,
        "age_source": age_source,
        "age_quality_flag": 1,
    })

    out.to_csv(OUT_CSV, index=False)
    out.to_parquet(OUT_PARQ, index=False)

    summary = {
        "age_column_used": age_col,
        "rows_out": int(len(out)),
        "unique_kic": int(out["kic"].nunique()),
        "age_gyr_range": [float(out["age_gyr"].min()), float(out["age_gyr"].max())] if len(out) else None,
        "out_csv": OUT_CSV,
        "out_parquet": OUT_PARQ,
    }
    with open(SUM, "w") as f:
        yaml.safe_dump(summary, f, sort_keys=False)

    print(f"saved: {OUT_PARQ} rows={len(out)}")
    print(f"age col: {age_col}")
    print(f"age range: {summary['age_gyr_range']}")
    print(f"summary: {SUM}")

if __name__ == "__main__":
    main()
